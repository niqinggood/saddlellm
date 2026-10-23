"""Compare uncached and saved/current prefix caches through real loopback SSE.

One loaded model, sequential randomized A/B/C blocks, complete outputs and usage.
This measures a repeated long prompt, not generic single-user decode throughput.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import torch
import uvicorn

from saddlellm.models.NativePrefixCache import NativePrefixCache
from saddlellm.runtime.InferenceServer import InferenceServerSettings, create_inference_app, load_inference_model
from tools.benchmark_prefix_cache_compare import load_baseline
from tools.benchmark_serving import read_stream


def source_hashes(baseline):
    paths = [baseline, Path(__file__), *[ROOT / "saddlellm" / path for path in (
        "models/NativePrefixCache.py", "models/NativeGeneration.py",
        "models/SaddleModeling.py", "runtime/InferenceServer.py",
    )]]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


async def benchmark(args):
    torch.set_num_threads(1)
    before = source_hashes(args.baseline)
    settings = InferenceServerSettings(
        model_path=str(args.model), backend="native", device=args.device,
        dtype=args.dtype, local_files_only=True, web_ui=False, chat_history=False,
        max_input_tokens=args.prompt_tokens + 128,
        max_new_tokens=args.tokens, default_max_new_tokens=args.tokens,
    )
    model, tokenizer = load_inference_model(settings)
    if args.fixed_length:
        model.config.eos_token_id = []
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.eos_token_id = []
    passage = "系统需要处理训练、评估和部署任务。保持模型精度与接口不变，测量首字延迟和实际输出吞吐。\n"
    ids = tokenizer.encode(passage, add_special_tokens=False)
    ids = (ids * (args.prompt_tokens // len(ids) + 1))[:args.prompt_tokens]
    prompt = tokenizer.decode(ids, skip_special_tokens=True)
    classes = {"uncached": None, "baseline_cache": load_baseline(args.baseline),
               "current_cache": NativePrefixCache}
    servers, tasks, sockets, clients, apps = [], [], [], {}, {}
    records = []
    report = None
    try:
        for name, cache_class in classes.items():
            app = create_inference_app(settings, model=model, tokenizer=tokenizer)
            if cache_class is not None:
                app.state.engine.prefix_cache = cache_class(
                    args.cache_mb * 1024**2, min_prefix_tokens=256,
                    reuse_partial_prefixes=True,
                    **({"min_store_tokens": args.min_store_tokens} if name == "current_cache" else {}),
                )
            apps[name] = app
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
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
            else:
                raise RuntimeError("server readiness timed out")
            clients[name] = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", timeout=120, trust_env=False,
            )

        async def request(name, pair):
            cache = apps[name].state.engine.prefix_cache
            prior = cache.snapshot() if cache is not None else None
            request_prompt = prompt + (f"\n请回答第 {pair + 3} 个问题：给出一句概括。" if args.vary_suffix else "")
            started = time.perf_counter()
            async with clients[name].stream("POST", "/v1/completions", json={
                "prompt": request_prompt, "temperature": 0, "max_tokens": args.tokens,
                "stream": True, "stream_options": {"include_usage": True},
            }) as response:
                response.raise_for_status()
                body, metrics = await read_stream(response, started, args.tokens)
            elapsed = time.perf_counter() - started
            health = (await clients[name].get("/health")).json()
            if health["admission"]["active"]:
                raise RuntimeError("request permits were not released")
            delta = None
            if prior is not None:
                current = health["prefix_cache"]
                delta = {key: current[key] - prior[key]
                         for key in ("hits", "reused_tokens", "misses", "invalidations")}
                if pair >= 0:
                    expected = body["usage"]["prompt_tokens"]
                    if delta["hits"] != 1 or delta["invalidations"]:
                        raise RuntimeError(f"expected a cache hit, got {delta}")
                    if (delta["reused_tokens"] != expected if not args.vary_suffix
                            else not 256 <= delta["reused_tokens"] < expected):
                        raise RuntimeError(f"unexpected prefix reuse: {delta}, prompt tokens={expected}")
            return {"variant": name, "pair": pair, "seconds": elapsed,
                    **body, **metrics, "prefix_cache": health["prefix_cache"], "cache_delta": delta}

        for name in clients:
            for warmup in (-2, -1):
                await request(name, warmup)
            cache = apps[name].state.engine.prefix_cache
            if cache is not None and not cache.snapshot()["entries"]:
                raise RuntimeError("cache budget or prompt did not permit a stored prefix")
        rng = random.Random(20261002)
        for pair in range(args.repeats):
            order = list(clients)
            rng.shuffle(order)
            for name in order:
                records.append(await request(name, pair))
            print(f"SSE prefix block {pair + 1}/{args.repeats}", flush=True)
        summary = {}
        for name in clients:
            rows = [row for row in records if row["variant"] == name]
            visible = [row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]
            summary[name] = {
                "median_seconds": statistics.median(row["seconds"] for row in rows),
                "median_visible_ttft_ms": statistics.median(visible) if visible else None,
                "visible_responses": len(visible),
                "output_tokens_per_second": sum(row["usage"]["completion_tokens"] for row in rows)
                / sum(row["seconds"] for row in rows),
            }
        comparisons = {}
        for baseline in ("uncached", "baseline_cache"):
            ratios, identical = [], []
            for pair in range(args.repeats):
                a, b = [next(row for row in records if row["variant"] == variant and row["pair"] == pair)
                        for variant in (baseline, "current_cache")]
                ratios.append(a["seconds"] / b["seconds"])
                identical.append(a["choices"] == b["choices"] and a["usage"] == b["usage"])
            comparisons[baseline] = {"median_speed_ratio": statistics.median(ratios),
                                     "faster_pairs": sum(ratio > 1 for ratio in ratios),
                                     "identical_text_finish_usage": all(identical), "pairs": len(ratios)}
        report = {"scope": "Warmed prefix reuse, real single-client HTTP SSE; optional changing suffix, with measured hit/token counters. Not cold requests or decode-only speed. Shared desktop GPU, not isolated.",
                "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                "summary": summary, "comparisons": comparisons, "records": records,
                "source_hashes": before, "sources_unchanged": before == source_hashes(args.baseline)}
        return report
    finally:
        for client in clients.values():
            await client.aclose()
        for server in servers:
            server.should_exit = True
        stopped = await asyncio.gather(*tasks, return_exceptions=True)
        for sock in sockets:
            sock.close()
        caches = {name: app.state.engine.prefix_cache.snapshot() for name, app in apps.items()
                  if app.state.engine.prefix_cache is not None}
        errors = [repr(item) for item in stopped if isinstance(item, BaseException)]
        if report is not None:
            report["shutdown"] = {"errors": errors, "caches": caches,
                                  "clean": not errors and all(not row["enabled"] and row["entries"] == 0
                                                              and row["retained_tensor_bytes"] == 0
                                                              for row in caches.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--prompt-tokens", type=int, default=768)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--cache-mb", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--fixed-length", action="store_true")
    parser.add_argument("--vary-suffix", action="store_true", help="Change a short question after the shared long prefix on every round")
    parser.add_argument("--min-store-tokens", type=int, default=0, help="Candidate only: minimum uncached suffix before publishing another snapshot")
    args = parser.parse_args()
    if (args.output.exists() or not args.baseline.is_file() or not args.model.is_dir()
            or args.prompt_tokens < 256 or args.min_store_tokens < 0
            or min(args.tokens, args.cache_mb, args.repeats) < 1):
        parser.error("local model/baseline required; output must be new; positive counts and prompt >= 256 required")
    report = asyncio.run(benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("summary", "comparisons")}, indent=2))
    return int(not report["sources_unchanged"] or not report["shutdown"]["clean"] or not all(
        row["identical_text_finish_usage"] for row in report["comparisons"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
