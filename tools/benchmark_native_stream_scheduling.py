"""Same-model HTTP comparison of serial, legacy threaded, bounded and fair SSE.

One GPU-resident model, four local HTTP endpoints measured sequentially in
shuffled order. Only admission/pull scheduling changes. No model download.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import torch
import uvicorn

from tools.benchmark_serving import compare


async def run(args):
    runtime = importlib.import_module("saddlellm.runtime.InferenceServer")
    settings = dict(model_path=args.model, model_name="native-benchmark", backend="native",
                    device="cuda", dtype="bfloat16", local_files_only=True, web_ui=False,
                    max_input_tokens=1024, max_new_tokens=args.tokens, default_max_new_tokens=args.tokens)
    model, tokenizer = runtime.load_inference_model(runtime.InferenceServerSettings(**settings))
    records, servers, tasks, sockets, targets = {}, [], [], [], []
    memory_records = []
    try:
        for name, slots in (("serial", 1), ("threaded", args.concurrency), ("bounded", args.concurrency), ("cooperative", args.concurrency)):
            current = runtime.InferenceServerSettings(**settings, max_concurrency=slots)
            if name in {"serial", "threaded"}:
                # Recreate the previous multi-thread pull policy, without
                # changing the model, tokenizer, generation or SSE machinery.
                with patch.object(runtime, "_cooperative_native_streaming", return_value=False), patch.object(runtime, "_native_stream_worker_count", return_value=None):
                    app = runtime.create_inference_app(current, model=model, tokenizer=tokenizer)
            elif name == "cooperative":
                with patch.object(runtime, "_cooperative_native_streaming", return_value=True):
                    app = runtime.create_inference_app(current, model=model, tokenizer=tokenizer)
            else:
                with patch.object(runtime, "_native_stream_worker_count", return_value=slots):
                    app = runtime.create_inference_app(current, model=model, tokenizer=tokenizer)
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
            targets.append(f"{name}=http://127.0.0.1:{port}/v1")
        async with httpx.AsyncClient(trust_env=False) as client:
            for target in targets:
                name, url = target.split("=", 1)
                records[name] = (await client.get(url[:-3] + "/health")).json()
        print("One shared model ready; measuring serial / threaded / bounded / cooperative HTTP streams", flush=True)
        config = SimpleNamespace(
            target=targets, model="native-benchmark", prompts_file=args.prompts_file,
            prompt="Write a numbered list of 20 practical tips for reading technical documentation. Keep each tip to one sentence.",
            requests=args.concurrency, concurrency=args.concurrency, max_tokens=args.tokens,
            stream=True, capture_text=True, warmup=1, repeats=args.repeats, timeout=180.0,
            api_key_env=None, checkpoint_sha256=None, cache_policy="disabled",
        )
        from tools import benchmark_serving

        workload = benchmark_serving.run_workload

        async def measured_workload(client, *values, **options):
            torch.cuda.reset_peak_memory_stats()
            free_before, total = torch.cuda.mem_get_info()
            result = await workload(client, *values, **options)
            free_after, _ = torch.cuda.mem_get_info()
            memory_records.append({
                "url": str(client.base_url), "free_before": free_before, "free_after": free_after,
                "device_total": total, "allocated_peak": torch.cuda.max_memory_allocated(),
                "reserved_peak": torch.cuda.max_memory_reserved(), "wall_s": result[1],
            })
            print(json.dumps(memory_records[-1]), flush=True)
            return result

        with patch.object(benchmark_serving, "run_workload", measured_workload):
            report = await compare(config)
        report["memory_records_including_warmup"] = memory_records
        report["health_before"] = records
        report["health_after"] = {}
        async with httpx.AsyncClient(trust_env=False) as client:
            for target in targets:
                name, url = target.split("=", 1)
                report["health_after"][name] = (await client.get(url[:-3] + "/health")).json()
        report["all_idle_after"] = all(item["admission"]["active"] == 0 for item in report["health_after"].values())
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
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompts-file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.tokens, args.repeats) < 1 or args.concurrency < 2:
        parser.error("output must be new, positive tokens/repeats, concurrency >= 2")
    paths = ("saddlellm/runtime/InferenceServer.py", "saddlellm/models/OperatorBackends.py",
             "saddlellm/models/NativeQwen3.py", "tools/benchmark_native_stream_scheduling.py")
    hashes = {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest() for name in paths}
    report = asyncio.run(run(args))
    report.update(created_at_utc=datetime.now(timezone.utc).isoformat(), model_path=args.model,
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(), source_sha256=hashes,
                  comparison="same weights and generation; legacy threaded policy isolated via factory patch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary = {name: {k: v for k, v in row.items() if k != "samples"} for name, row in report["targets"].items()}
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if report["all_output_text_equal"] and report["all_token_counts_equal"] and report["all_idle_after"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
