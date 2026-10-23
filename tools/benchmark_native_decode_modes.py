"""Local native SSE ablation: batching off, compatible, throughput.

Starts only temporary loopback servers, sharing one locally loaded checkpoint.
Measures actual usage and visible text timing, not per-token GPU latency. No
quality claim follows from matching tokens or small numerical differences.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import socket
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import torch
import uvicorn

from saddlellm.models.NativeBatchDecode import NativeBatchDecodeRunner
from saddlellm.runtime.InferenceServer import (
    InferenceServerSettings,
    create_inference_app,
    load_inference_model,
)
from tools.benchmark_serving import load_prompts, percentile, read_stream


def source_hashes():
    paths = (
        "saddlellm/models/NativeBatchDecode.py",
        "saddlellm/models/NativeGeneration.py",
        "saddlellm/models/SaddleModeling.py",
        "saddlellm/models/InferenceCache.py",
        "saddlellm/models/InferenceOptimizer.py",
        "saddlellm/models/OperatorBackends.py",
        "saddlellm/runtime/NativeStreamBatcher.py",
        "saddlellm/runtime/InferenceServer.py",
        "tools/benchmark_native_decode_modes.py",
        "tools/benchmark_serving.py",
    )
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths
    }


@torch.inference_mode()
def numerical_probe(model, *, mixed_lengths=False):
    """Same-history comparison: overwrite each next token with the B1 reference.

    EOS, repetition penalties and streaming are disabled only in this probe. KV
    stores prior tokens, so forcing the newly appended token preserves identical
    histories for the following forward. No hook changes production fast paths.
    """
    device = model.embed_tokens.weight.device
    rng = torch.Generator(device=device).manual_seed(731)
    prompts = torch.randint(
        4, model.config.vocab_size, (4, 32), device=device, generator=rng
    )
    result = {
        mode: {
            "max_absolute_logit_error": 0.0,
            "mean_absolute_logit_error": 0.0,
            "argmax_disagreements": 0,
            "request_steps": 0,
            "all_finite": True,
        }
        for mode in ("compatible", "throughput")
    }
    with ExitStack() as stack:
        sessions = {
            mode: [
                stack.enter_context(
                    model.create_generation_session(
                        row[None, : 8 * (index + 1)] if mixed_lengths else row[None],
                        max_new_tokens=48,
                        do_sample=False,
                        eos_token_id=[],
                        output_scores=True,
                    )
                )
                for index, row in enumerate(prompts)
            ]
            for mode in ("individual", "compatible", "throughput")
        }
        runners = {
            mode: stack.enter_context(NativeBatchDecodeRunner(model, decode_mode=mode))
            for mode in result
        }
        for _ in range(48):
            for session in sessions["individual"]:
                session.step()
            for mode, runner in runners.items():
                runner.step(sessions[mode])
                for candidate, reference in zip(sessions[mode], sessions["individual"]):
                    observed, expected = (
                        candidate.scores[-1].float(),
                        reference.scores[-1].float(),
                    )
                    error = (observed - expected).abs()
                    row = result[mode]
                    row["max_absolute_logit_error"] = max(
                        row["max_absolute_logit_error"], error.max().item()
                    )
                    row["mean_absolute_logit_error"] += error.mean().item()
                    row["argmax_disagreements"] += int(
                        (observed.argmax(-1) != expected.argmax(-1)).sum().item()
                    )
                    row["request_steps"] += 1
                    row["all_finite"] &= bool(torch.isfinite(observed).all())
                    candidate.buffers.tokens[:, candidate.buffers.length - 1].copy_(
                        reference.buffers.tokens[:, reference.buffers.length - 1]
                    )
        for mode, row in result.items():
            row["mean_absolute_logit_error"] /= row["request_steps"]
            row["runner"] = runners[mode].snapshot()
    return result


async def run(args):
    hashes_before = source_hashes()
    torch.set_num_threads(args.cpu_threads)
    options = dict(
        model_path=args.model,
        backend="native",
        model_name="native-mode-benchmark",
        device=args.device,
        dtype=args.dtype,
        local_files_only=True,
        web_ui=False,
        max_input_tokens=256,
        max_new_tokens=args.tokens,
        default_max_new_tokens=args.tokens,
        max_concurrency=max(2, args.concurrency),
        batch_size=max(2, args.concurrency),
        batch_token_budget=max(2, args.concurrency) * (256 + args.tokens),
        batch_wait_ms=args.wait_ms,
    )
    model, tokenizer = load_inference_model(InferenceServerSettings(**options))
    if args.fixed_length:
        model.config.eos_token_id = []
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.eos_token_id = []
    prompts = (
        load_prompts(args.prompts_file)
        if args.prompts_file
        else ["Once upon a time, there was a little girl who lived in a small village."]
    )
    if not prompts or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError("prompts must be nonempty strings")
    numerical = {
        "equal": numerical_probe(model),
        "mixed": numerical_probe(model, mixed_lengths=True),
    }
    print("Numerical probe completed; starting loopback SSE", flush=True)
    servers, tasks, sockets, clients, apps = [], [], [], {}, {}
    records = []
    try:
        for name in ("off", "compatible", "throughput"):
            config = InferenceServerSettings(
                **options,
                native_stream_batching=name != "off",
                native_decode_mode="compatible" if name == "off" else name,
            )
            app = create_inference_app(config, model=model, tokenizer=tokenizer)
            apps[name] = app
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            port = sock.getsockname()[1]
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=port,
                    log_level="error",
                    access_log=False,
                )
            )
            servers.append(server)
            task = asyncio.create_task(server.serve(sockets=[sock]))
            tasks.append(task)
            for _ in range(1000):
                if server.started:
                    break
                if task.done():
                    await task
                    raise RuntimeError("server exited before readiness")
                await asyncio.sleep(0.01)
            if not server.started:
                raise RuntimeError("server readiness timeout")
            clients[name] = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False
            )

        async def workload(name, pair):
            client = clients[name]

            async def request(index):
                start = time.perf_counter()
                try:
                    async with client.stream(
                        "POST",
                        "/v1/completions",
                        json={
                            "prompt": prompts[index % len(prompts)],
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_tokens": args.tokens,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        },
                    ) as response:
                        response.raise_for_status()
                        body, metrics = await read_stream(response, start, args.tokens)
                    return {
                        "ok": True,
                        "seconds": time.perf_counter() - start,
                        **metrics,
                        **body,
                    }
                except Exception as exc:
                    return {
                        "ok": False,
                        "seconds": time.perf_counter() - start,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            rows = await asyncio.gather(*(request(i) for i in range(args.concurrency)))
            elapsed = time.perf_counter() - start
            return {
                "variant": name,
                "pair": pair,
                "seconds": elapsed,
                "requests": rows,
                "output_tokens": sum(
                    r["usage"]["completion_tokens"] for r in rows if r["ok"]
                ),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if args.device == "cuda"
                else None,
            }

        for name in clients:
            for _ in range(2):
                warm = await workload(name, -1)
                if not all(r["ok"] for r in warm["requests"]):
                    raise RuntimeError(f"warmup failed: {warm}")
        rng = random.Random(20260929)
        for pair in range(args.repeats):
            order = list(clients)
            rng.shuffle(order)
            for name in order:
                records.append(await workload(name, pair))
            print(f"Completed SSE block {pair + 1}/{args.repeats}", flush=True)
        health = {
            name: (await client.get("/health")).json()
            for name, client in clients.items()
        }
        summary = {}
        for name in clients:
            runs = [r for r in records if r["variant"] == name]
            rows = [row for run in runs for row in run["requests"]]
            ttfts = [
                row["ttft_ms"]
                for row in rows
                if row["ok"] and row["ttft_ms"] is not None
            ]
            gaps = [
                gap for row in rows if row["ok"] for gap in row["inter_text_chunk_ms"]
            ]
            state = health[name].get("native_stream_batching", {})
            summary[name] = {
                "output_tokens_per_second": sum(r["output_tokens"] for r in runs)
                / sum(r["seconds"] for r in runs),
                "failed_requests": sum(not row["ok"] for row in rows),
                "p95_visible_ttft_ms": percentile(ttfts, 0.95) if ttfts else None,
                "p95_response_ms": percentile(
                    [row["seconds"] * 1000 for row in rows], 0.95
                ),
                "p95_inter_text_chunk_ms": percentile(gaps, 0.95) if gaps else None,
                "idle": health[name]["admission"]["active"] == 0
                and all(
                    state.get(k, 0) == 0
                    for k in ("registered", "pending", "inflight", "retained_cohorts")
                ),
                "peak_allocated_bytes": max(
                    (
                        r["peak_allocated_bytes"]
                        for r in runs
                        if r["peak_allocated_bytes"] is not None
                    ),
                    default=None,
                ),
            }
        comparisons = {}
        for before, after in (
            ("off", "compatible"),
            ("compatible", "throughput"),
            ("off", "throughput"),
        ):
            a = [r for r in records if r["variant"] == before]
            b = [r for r in records if r["variant"] == after]
            ratios = [
                (y["output_tokens"] / y["seconds"])
                / (x["output_tokens"] / x["seconds"])
                for x, y in zip(a, b)
                if x["output_tokens"] and y["output_tokens"]
            ]
            comparisons[f"{before}->{after}"] = {
                "median_token_rate_ratio": statistics.median(ratios)
                if ratios
                else None,
                "faster_pairs": sum(v > 1 for v in ratios),
                "pairs": len(ratios),
            }
        return {
            "scope": "Shared local checkpoint; real loopback SSE; same admission limits, two warmups, randomized order. Visible text-chunk gaps are NOT per-token ITL. Sampling output equality is not a performance gate. Fixed-length mode disables EOS in this process only. Non-isolated desktop hardware.",
            "settings": vars(args) | {"output": str(args.output)},
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "numerical_probe": numerical,
            "summary": summary,
            "comparisons": comparisons,
            "health_after": health,
            "records": records,
            "source_hashes": hashes_before,
            "sources_unchanged": hashes_before == source_hashes(),
        }
    finally:
        for client in clients.values():
            await client.aclose()
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)
        for sock in sockets:
            sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--wait-ms", type=float, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--fixed-length", action="store_true")
    parser.add_argument("--prompts-file")
    args = parser.parse_args()
    if (
        args.output.exists()
        or min(args.concurrency, args.tokens, args.repeats, args.cpu_threads) < 1
    ):
        parser.error("output must be new and counts positive")
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        parser.error("temperature must be in [0,2] and top-p in (0,1]")
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {key: report[key] for key in ("summary", "comparisons", "numerical_probe")},
            indent=2,
        )
    )
    return int(
        not report["sources_unchanged"]
        or any(
            not row["all_finite"]
            for probe in report["numerical_probe"].values()
            for row in probe.values()
        )
        or any(
            row["failed_requests"] or not row["idle"]
            for row in report["summary"].values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
