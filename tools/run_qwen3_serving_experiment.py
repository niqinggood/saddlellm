"""Measure ONE already-running loopback backend; no install/start/stop actions.

Run engines sequentially outside this script. All outputs are non-sensitive toy
prompts. Measurements are closed-loop client observations, not capacity claims.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import benchmark_serving as benchmark
import httpx


async def wait_ready(client, protocol, timeout=120):
    deadline = time.perf_counter() + timeout
    path = "/api/version" if protocol == "ollama" else "/health"
    while True:
        try:
            response = await client.get(path)
            if response.is_success:
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                if not isinstance(body, dict) or body.get("ready") is not False:
                    return
        except httpx.HTTPError:
            pass
        if time.perf_counter() >= deadline:
            raise RuntimeError("Backend did not become ready; no measurements recorded")
        await asyncio.sleep(0.5)


async def run(args):
    parsed = urlsplit(args.url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "Experiment endpoints must be loopback; use SSH forwarding for access"
        )
    target = f"{args.label}={args.url.rstrip('/')}/v1"
    benchmark.parse_target(target)
    prepared = Path(args.prepared_dir)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    meta = {
        "backend": args.label,
        "url": args.url,
        "model_alias": args.model,
        "precision_asserted_by_operator": args.precision,
        "protocol": args.protocol,
        "checkpoint_manifest": manifest,
        "checkpoint_manifest_scope": "HF prompt source; does not identify Ollama GGUF weights",
        "started_unix": time.time(),
        "warning": "Other retained services may share this GPU; do not infer isolated-device capacity.",
    }
    async with httpx.AsyncClient(
        base_url=args.url, timeout=15, trust_env=False
    ) as client:
        await wait_ready(client, args.protocol)
        output.mkdir(parents=True, exist_ok=False)
        if args.protocol == "ollama":
            for path in ["/api/version", "/api/ps"]:
                response = await client.get(path)
                response.raise_for_status()
                meta[path] = response.json()
            response = await client.post("/api/show", json={"model": args.model})
            response.raise_for_status()
            meta["ollama_details"] = response.json().get("details")
            meta["ollama_model_info"] = response.json().get("model_info")
        else:
            for path in ["/health", "/v1/models", "/version"]:
                response = await client.get(path)
                if response.is_success:
                    try:
                        meta[path] = response.json()
                    except ValueError:
                        meta[path] = {"http_status": response.status_code}
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        meta["gpu_before"] = gpu.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        meta["gpu_before"] = "unavailable"
    (output / "environment.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    phases = [
        (f"performance_c{c}", "performance.json", 16, c, 64, 1, 2, False)
        for c in [1, 4, 8]
    ]
    phases.append(("quality", "quality.json", 8, 1, 256, 0, 1, False))
    if args.stream:
        phases.extend(
            (f"stream_c{c}", "performance.json", 8, c, 32, 1, 1, True) for c in [1, 4]
        )
    for (
        name,
        filename,
        requests,
        concurrency,
        tokens,
        warmup,
        repeats,
        stream,
    ) in phases:
        options = SimpleNamespace(
            target=[target],
            model=args.model,
            prompts_file=str(prepared / filename),
            prompt=None,
            requests=requests,
            concurrency=concurrency,
            max_tokens=tokens,
            warmup=warmup,
            repeats=repeats,
            timeout=300.0,
            api_key_env=None,
            checkpoint_sha256=manifest["files_manifest_sha256"]
            if args.protocol == "openai"
            else None,
            cache_policy=args.cache_policy,
            protocol=args.protocol,
            ollama_context_limit=4096,
            capture_text=True,
            stream=stream,
        )
        report = await benchmark.compare(options)
        report["phase"] = name
        report["precision_asserted_by_operator"] = args.precision
        report["comparison_note"] = (
            "Single backend only; equality flags are self-comparisons, not cross-engine evidence."
        )
        (output / f"{name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        result = report["targets"][args.label]
        print(
            json.dumps(
                {
                    "phase": name,
                    "backend": args.label,
                    "tokens_per_second": result["generated_tokens_per_second"],
                    "p50_ms": result["p50_response_ms"],
                    "p95_ms": result["p95_response_ms"],
                    "p50_ttft_ms": result.get("p50_ttft_ms"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--precision", required=True)
    parser.add_argument("--cache-policy", required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
