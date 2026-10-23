"""Compare saved/current native schedulers through real loopback SSE.

Normally only scheduler classes differ; a baseline adapter can declare a more
specific BASELINE_CONTRACT and BASELINE_SOURCES. Model, numerical mode, limits
and prompts are shared. Host trace durations are not CUDA kernel timings or ITL.
The baseline is an explicitly supplied, trusted local Python source snapshot.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import random
import socket
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import torch
import uvicorn

from saddlellm.runtime.InferenceServer import (
    InferenceServerSettings,
    create_inference_app,
    load_inference_model,
)
from saddlellm.runtime.NativeStreamBatcher import NativeStreamBatcher
from tools.benchmark_native_decode_modes import source_hashes
from tools.benchmark_serving import load_prompts, percentile, read_stream


def traced(base):
    class TracedBatcher(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dispatch_trace = []

        def _execute(self, jobs):
            started = time.monotonic()
            with self._condition:
                entry = {
                    "batch": len(jobs),
                    "registered": len(self._registered),
                    "oldest_queue_ms": max((started - job[3]) * 1000 for job in jobs),
                    "generated_tokens": [job[0].generated_tokens for job in jobs],
                    "prompt_tokens": [job[0].prompt_tokens for job in jobs],
                }
            try:
                return super()._execute(jobs)
            finally:
                entry["execute_wall_ms"] = (time.monotonic() - started) * 1000
                self.dispatch_trace.append(entry)

    return TracedBatcher


def hashes(baseline, extra_sources=()):
    return source_hashes() | {
        str(baseline): hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "tools/benchmark_native_dispatch.py": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    } | {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in extra_sources}


async def run(args):
    spec = importlib.util.spec_from_file_location(
        "saddlellm.runtime._dispatch_baseline", args.baseline
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    extra_sources = getattr(baseline, "BASELINE_SOURCES", ())
    before = hashes(args.baseline, extra_sources)
    variants = {
        "baseline": traced(baseline.NativeStreamBatcher),
        "candidate": traced(NativeStreamBatcher),
    }
    if args.baseline_only:
        variants.pop("candidate")
    torch.set_num_threads(args.cpu_threads)
    options = dict(
        model_path=args.model,
        backend="native",
        model_name="dispatch-benchmark",
        device=args.device,
        dtype=args.dtype,
        local_files_only=True,
        web_ui=False,
        max_input_tokens=1024,
        max_new_tokens=args.tokens,
        default_max_new_tokens=args.tokens,
        max_concurrency=max(2, args.concurrency),
        batch_size=max(2, args.concurrency),
        batch_token_budget=max(2, args.concurrency) * (1024 + args.tokens),
        batch_wait_ms=args.wait_ms,
        native_stream_batching=True,
        native_decode_mode=args.decode_mode,
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
        raise ValueError("nonempty text prompts required")
    servers, tasks, sockets, clients, apps, records = [], [], [], {}, {}, []
    report = None
    try:
        for name, batcher_class in variants.items():
            with patch(
                "saddlellm.runtime.NativeStreamBatcher.NativeStreamBatcher",
                batcher_class,
            ):
                app = create_inference_app(
                    InferenceServerSettings(**options), model=model, tokenizer=tokenizer
                )
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

        async def execute_workload(name, pair):
            async def request(index):
                start = time.perf_counter()
                try:
                    async with clients[name].stream(
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

            batcher = apps[name].state.native_stream_batcher
            batcher.dispatch_trace.clear()
            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            rows = await asyncio.gather(*(request(i) for i in range(args.concurrency)))
            elapsed = time.perf_counter() - start
            health = (await clients[name].get("/health")).json()
            state = health["native_stream_batching"]
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
                "trace": list(batcher.dispatch_trace),
                "idle": health["admission"]["active"] == 0
                and all(
                    state[k] == 0
                    for k in ("registered", "pending", "inflight", "retained_cohorts")
                ),
            }

        async def workload(name, pair):
            from saddlellm.models.NativeGeneration import NativeGenerationSession, _batch_delivery_methods

            factory = getattr(baseline, "baseline_context", None)
            with factory() if name == "baseline" and factory is not None else nullcontext():
                result = await execute_workload(name, pair)
                if not result["idle"]:
                    raise RuntimeError("workload must fully drain before changing global baseline methods")
            restored = all(getattr(NativeGenerationSession, method) is original
                           for method, original in _batch_delivery_methods)
            if not restored:
                raise RuntimeError("baseline instrumentation did not restore generation methods")
            result["runtime_methods_restored"] = True
            return result

        for name in clients:
            for _ in range(args.warmups):
                warm = await workload(name, -1)
                if not warm["idle"] or not all(row["ok"] for row in warm["requests"]):
                    raise RuntimeError(f"warmup failed: {warm}")
        rng = random.Random(20260930)
        for pair in range(args.repeats):
            order = list(clients)
            rng.shuffle(order)
            for name in order:
                records.append(await workload(name, pair))
            print(f"SSE block {pair + 1}/{args.repeats}", flush=True)
        summary = {}
        for name in clients:
            runs = [r for r in records if r["variant"] == name]
            rows = [row for r in runs for row in r["requests"]]
            events = [entry for r in runs for entry in r["trace"]]
            ttft = [r["ttft_ms"] for r in rows if r["ok"] and r["ttft_ms"] is not None]
            gaps = [
                value for r in rows if r["ok"] for value in r["inter_text_chunk_ms"]
            ]
            summary[name] = {
                "output_tokens_per_second": sum(r["output_tokens"] for r in runs)
                / sum(r["seconds"] for r in runs),
                "failed_requests": sum(not r["ok"] for r in rows),
                "idle": all(r["idle"] for r in runs),
                "p95_visible_ttft_ms": percentile(ttft, 0.95) if ttft else None,
                "p95_response_ms": percentile(
                    [r["seconds"] * 1000 for r in rows], 0.95
                ),
                "p95_inter_text_chunk_ms": percentile(gaps, 0.95) if gaps else None,
                "peak_allocated_bytes": max(
                    (
                        r["peak_allocated_bytes"]
                        for r in runs
                        if r["peak_allocated_bytes"] is not None
                    ),
                    default=None,
                ),
                "dispatch_batch_histogram": {
                    str(size): sum(e["batch"] == size for e in events)
                    for size in sorted({e["batch"] for e in events})
                },
                "expired_partial_dispatches": sum(
                    e["batch"] < min(args.concurrency, e["registered"])
                    and e["oldest_queue_ms"] >= args.wait_ms
                    for e in events
                ),
                "execute_wall_ms": sum(e["execute_wall_ms"] for e in events),
            }
        pairs = []
        if "candidate" in clients:
            for i in range(args.repeats):
                a, b = [
                    next(r for r in records if r["variant"] == name and r["pair"] == i)
                    for name in ("baseline", "candidate")
                ]
                if a["output_tokens"] and b["output_tokens"]:
                    pairs.append(
                        (b["output_tokens"] / b["seconds"])
                        / (a["output_tokens"] / a["seconds"])
                    )
        report = {
            "scope": getattr(baseline, "BASELINE_CONTRACT", "Saved vs current scheduler") + "; same local model and numerical mode; fixed length disables EOS only here. Host traces are not CUDA timings; visible chunk intervals are not ITL. Desktop GPU is not isolated.",
            "settings": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "summary": summary,
            "comparisons": {
                "median_token_rate_ratio": statistics.median(pairs) if pairs else None,
                "faster_pairs": sum(r > 1 for r in pairs),
                "pairs": len(pairs),
            },
            "health_after": {
                name: (await client.get("/health")).json()
                for name, client in clients.items()
            },
            "records": records,
            "source_hashes": before,
            "sources_unchanged": before == hashes(args.baseline, extra_sources),
        }
        return report
    finally:
        for client in clients.values():
            await client.aclose()
        for server in servers:
            server.should_exit = True
        stopped = await asyncio.gather(*tasks, return_exceptions=True)
        for sock in sockets:
            sock.close()
        errors = [repr(item) for item in stopped if isinstance(item, BaseException)]
        if report is not None:
            report["shutdown"] = {"clean": not errors, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--decode-mode", choices=["compatible", "throughput"], default="throughput"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--wait-ms", type=float, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--fixed-length", action="store_true")
    parser.add_argument("--prompts-file")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--require-delivery-batching", action="store_true",
                        help="Fail if the candidate never enters fused sampling/host delivery")
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.baseline.is_file()
        or min(args.concurrency, args.tokens, args.repeats, args.cpu_threads) < 1
        or args.warmups < 0
    ):
        parser.error(
            "baseline must exist; output must be new; counts must be positive and warmups nonnegative"
        )
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("summary", "comparisons")}, indent=2))
    return int(
        not report["sources_unchanged"]
        or not report["shutdown"]["clean"]
        or (args.require_delivery_batching and (
            "candidate" not in report["health_after"]
            or not report["health_after"]["candidate"]["native_stream_batching"]["batched_host_transfers"]
            or not report["health_after"]["candidate"]["native_stream_batching"]["batched_sampling_forwards"]
        ))
        or any(
            r["failed_requests"] or not r["idle"] for r in report["summary"].values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
