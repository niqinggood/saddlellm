"""Real loopback A/B: an abandoned long request followed by a short request.

Uses a local native checkpoint on CPU, identical weights and greedy settings.
Measures wasted forward steps and following-request latency, NOT general token
throughput. Starts/stops only its own ephemeral loopback servers, no downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import socket
import statistics
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import torch

from saddlellm.models.ModelLoader import load_native_causal_lm
from saddlellm.models.TokenizerLoader import load_tokenizer_compatible
from saddlellm.runtime.InferenceServer import InferenceServerSettings, create_inference_app
from tools.benchmark_native_serving import loopback_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.model.is_dir() or args.output.exists() or args.tokens < 8 or args.repeats < 1:
        parser.error("require a local checkpoint, new output, >=8 tokens, positive repeats")
    torch.set_num_threads(1)
    model = load_native_causal_lm(str(args.model), device="cpu", dtype="float32", local_files_only=True)
    tokenizer = load_tokenizer_compatible(str(args.model), local_files_only=True)
    model.config.eos_token_id = []  # Explicit fixed-output workload for both sides.
    lifecycle = importlib.import_module("saddlellm.runtime.InferenceLifecycle")
    original_lifecycle = lifecycle.run_until_disconnect

    async def legacy(receive, operation, cancel_event):
        return await operation  # Former non-stream behavior: never observe disconnect.

    entered = threading.Event()
    forwards = [0]

    def observe(_model, _inputs):
        forwards[0] += 1
        entered.set()

    hook = model.register_forward_pre_hook(observe)
    report = {"scope": "loopback client-disconnect recovery, not ordinary throughput or model quality",
              "model": str(args.model), "device": "cpu", "dtype": "float32", "threads": 1,
              "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
              "eos_disabled": True, "abandoned_output_budget": args.tokens, "follower_output_budget": 2,
              "repeats": args.repeats, "accepted": False, "records": [],
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                  Path("saddlellm/runtime/InferenceServer.py"), Path(lifecycle.__file__), Path(__file__),
              )}}
    try:
        with ExitStack() as stack:
            endpoints = {}
            for kind in ("legacy", "cancel_aware"):
                lifecycle.run_until_disconnect = legacy if kind == "legacy" else original_lifecycle
                try:
                    app = create_inference_app(InferenceServerSettings(
                        web_ui=False, max_input_tokens=128, max_new_tokens=args.tokens,
                        default_max_new_tokens=args.tokens, max_concurrency=1,
                    ), model=model, tokenizer=tokenizer)
                finally:
                    lifecycle.run_until_disconnect = original_lifecycle
                endpoints[kind] = stack.enter_context(loopback_server(app))
            client = stack.enter_context(httpx.Client(trust_env=False, timeout=30))
            prompt = "请介绍人工智能的常见应用，并给出一些具体的例子。"

            def trial(kind):
                base = endpoints[kind]
                target = urlsplit(base)
                entered.clear()
                forwards[0] = 0
                body = json.dumps({"prompt": prompt, "temperature": 0, "max_tokens": args.tokens}).encode()
                with socket.create_connection((target.hostname, target.port), timeout=5) as connection:
                    headers = (f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                               f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode()
                    connection.sendall(headers + body)
                    if not entered.wait(10):
                        raise TimeoutError("model did not enter forward")
                    connection.shutdown(socket.SHUT_RDWR)
                started = time.perf_counter()
                response = client.post(base + "/completions", json={"prompt": prompt, "temperature": 0, "max_tokens": 2})
                elapsed_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                result = response.json()
                assert result["usage"]["completion_tokens"] == 2
                deadline = time.monotonic() + 10
                while True:
                    health = client.get(base.removesuffix("/v1") + "/health").json()
                    if health["admission"]["active"] == 0:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("admission did not return to zero")
                    time.sleep(0.01)
                signature = hashlib.sha256(json.dumps({"choices": result["choices"], "usage": result["usage"]},
                                                     sort_keys=True).encode()).hexdigest()
                return {"variant": kind, "follower_ms": elapsed_ms,
                        "abandoned_forward_steps": forwards[0] - 2, "follower_signature": signature,
                        "admission_final": health["admission"]["active"]}

            for kind in endpoints:
                trial(kind)
            for pair in range(args.repeats):
                for kind in (list(endpoints) if pair % 2 == 0 else list(reversed(endpoints))):
                    report["records"].append({"pair": pair, **trial(kind)})
            report["summary"] = {}
            for kind in endpoints:
                records = [row for row in report["records"] if row["variant"] == kind]
                report["summary"][kind] = {
                    "median_follower_ms": statistics.median(row["follower_ms"] for row in records),
                    "max_follower_ms": max(row["follower_ms"] for row in records),
                    "abandoned_forward_steps": [row["abandoned_forward_steps"] for row in records],
                }
            report["accepted"] = (
                len({row["follower_signature"] for row in report["records"]}) == 1
                and all(row["admission_final"] == 0 for row in report["records"])
                and all(row["abandoned_forward_steps"] < args.tokens for row in report["records"]
                        if row["variant"] == "cancel_aware")
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        lifecycle.run_until_disconnect = original_lifecycle
        hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"accepted": report["accepted"], "summary": report.get("summary"), "error": report.get("error")}, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
