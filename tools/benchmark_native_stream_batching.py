"""Paired real loopback SSE test of native batching OFF/ON, one shared model.

No download, global backend changes, existing-server restart or fixed token
count assumption. Records failures as well as successes; not an 8B/vLLM claim.
"""
import argparse
import asyncio
import hashlib
import json
import platform
import socket
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import torch
import uvicorn
import saddlellm

from saddlellm.runtime.InferenceServer import InferenceServerSettings, create_inference_app, load_inference_model
from tools import benchmark_serving


async def run(args):
    torch.set_num_threads(args.cpu_threads)
    options = dict(model_path=args.model, model_name="native-stream-benchmark", backend="native",
                   device=args.device, dtype=args.dtype, local_files_only=True, web_ui=False,
                   max_input_tokens=256, max_new_tokens=args.tokens, default_max_new_tokens=args.tokens,
                   max_concurrency=max(2, args.concurrency), batch_size=max(2, args.concurrency),
                   batch_token_budget=max(2, args.concurrency) * (256 + args.tokens), batch_wait_ms=args.wait_ms)
    model, tokenizer = load_inference_model(InferenceServerSettings(**options))
    servers, tasks, sockets, targets, records = [], [], [], [], []
    try:
        for enabled in (False, True):
            current = dict(options)
            if not enabled and args.baseline_max_concurrency is not None:
                current["max_concurrency"] = args.baseline_max_concurrency
            app = create_inference_app(InferenceServerSettings(**current, native_stream_batching=enabled),
                                       model=model, tokenizer=tokenizer)
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            port = sock.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False))
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
            targets.append(f"{'batched' if enabled else 'baseline'}=http://127.0.0.1:{port}/v1")
        print("Shared checkpoint ready; measuring real SSE with batching OFF / ON", flush=True)
        configuration = SimpleNamespace(
            target=targets, model="native-stream-benchmark", prompts_file=args.prompts_file,
            prompt="Once upon a time, there was a little girl who lived in a small village.",
            requests=args.concurrency, concurrency=args.concurrency, max_tokens=args.tokens,
            stream=True, capture_text=True, warmup=2, repeats=args.repeats, timeout=120.0,
            api_key_env=None, checkpoint_sha256=None, cache_policy="disabled",
        )
        original = benchmark_serving.run_workload

        async def measured(client, *values, **options):
            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            rows, wall = await original(client, *values, **options)
            record = {"variant": "baseline" if str(client.base_url).rstrip("/") == targets[0].split("=", 1)[1] else "batched",
                      "wall_s": wall, "output_tokens": sum(row["completion_tokens"] for row in rows),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None}
            records.append(record)
            print(json.dumps(record), flush=True)
            return rows, wall

        with patch.object(benchmark_serving, "run_workload", measured):
            report = await benchmark_serving.compare(configuration)
        report["records_including_warmup"] = records
        report["health_after"] = {}
        async with httpx.AsyncClient(trust_env=False) as client:
            for target in targets:
                name, url = target.split("=", 1)
                report["health_after"][name] = (await client.get(url[:-3] + "/health")).json()
        baseline, batched = (report["targets"][key] for key in ("baseline", "batched"))
        rounds = {key: [row for row in records if row["variant"] == key][2:] for key in ("baseline", "batched")}
        ratios = [a["wall_s"] / b["wall_s"] for a, b in zip(rounds["baseline"], rounds["batched"])]
        status = report["health_after"]["batched"]["native_stream_batching"]
        checks = {
            "text_and_usage_equal": report["all_output_text_equal"] and report["all_token_counts_equal"],
            "actual_batch_forwards": status["batched_forwards"] > 0 if args.concurrency > 1 else True,
            "idle": all(h["admission"]["active"] == 0 for h in report["health_after"].values())
                    and all(status[key] == 0 for key in ("registered", "pending", "inflight", "retained_cohorts")),
            "seven_or_more_pairs": args.repeats >= 7,
            "median_speedup_at_least_3pct": statistics.median(ratios) >= 1.03,
            "at_least_85pct_pair_wins": sum(ratio > 1 for ratio in ratios) / len(ratios) >= 0.85,
            "p95_visible_ttft_within_5pct": (batched["p95_ttft_ms"] is not None and baseline["p95_ttft_ms"] is not None
                                             and batched["p95_ttft_ms"] <= baseline["p95_ttft_ms"] * 1.05),
            "peak_allocated_within_5pct": (args.device != "cuda" or
                max(row["peak_allocated_bytes"] for row in rounds["batched"]) <=
                max(row["peak_allocated_bytes"] for row in rounds["baseline"]) * 1.05),
        }
        report["development_gate"] = {"checks": checks, "passed": all(checks.values()),
                                      "paired_wall_speedups": ratios, "median_speedup": statistics.median(ratios)}
        return report
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)
        for sock in sockets:
            sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--baseline-max-concurrency", type=int,
                        help="Optional baseline server permit limit; 1 measures the default serial admission against concurrent clients")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--wait-ms", type=float, default=5)
    parser.add_argument("--prompts-file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-speedup", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or min(args.tokens, args.repeats, args.concurrency, args.cpu_threads) < 1:
        parser.error("output must be new and counts positive")
    if args.baseline_max_concurrency is not None and args.baseline_max_concurrency < 1:
        parser.error("baseline-max-concurrency must be positive")
    root = Path(__file__).resolve().parents[1]
    paths = ("saddlellm/runtime/InferenceServer.py", "saddlellm/runtime/NativeStreamBatcher.py",
             "saddlellm/models/NativeBatchDecode.py", "saddlellm/models/SaddleModeling.py",
             "saddlellm/models/NativeGeneration.py", "saddlellm/models/OperatorBackends.py",
             "tools/benchmark_native_stream_batching.py", "tools/benchmark_serving.py")
    # A remote wheel validation need not upload the source checkout. Resolve
    # runtime hashes from the package actually imported, not a nearby copy.
    package_root = Path(saddlellm.__file__).resolve().parent.parent
    sources = {name: (package_root if name.startswith("saddlellm/") else root) / name for name in paths}
    hashes_before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        report = {"error": f"{type(exc).__name__}: {exc}"}
    hashes_after = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    report.update(created_at_utc=datetime.now(timezone.utc).isoformat(), settings=vars(args) | {"output": str(args.output)},
                  torch_version=torch.__version__, platform=platform.platform(),
                  gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  source_sha256=hashes_before, source_unchanged_during_run=hashes_before == hashes_after,
                  imported_package=str(Path(saddlellm.__file__).resolve()),
                  external_load_isolation="not guaranteed; existing services untouched",
                  scope_note=("native HTTP SSE; same weights/precision; NOT vLLM or 8B; "
                              + ("same request permits, batching OFF/ON" if args.baseline_max_concurrency is None else
                                 f"baseline request permits={args.baseline_max_concurrency}, batched permits={max(2, args.concurrency)}")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if "error" in report or not report["source_unchanged_during_run"]:
        print(json.dumps({"error": report.get("error"), "source_unchanged_during_run": report["source_unchanged_during_run"]}))
        return 1
    print(json.dumps({"targets": {name: {k: v for k, v in row.items() if k != "samples"} for name, row in report["targets"].items()},
                      "gate": report["development_gate"]}, indent=2))
    if not report["development_gate"]["checks"]["text_and_usage_equal"] or not report["development_gate"]["checks"]["idle"]:
        return 1
    return 2 if args.require_speedup and not report["development_gate"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
