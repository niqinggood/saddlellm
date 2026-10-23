"""Read-only loopback inference smoke; does not start, stop or modify a service.

Output contains test prompts and responses, never authentication credentials.
Latency samples are functional checks, NOT a comparative serving benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8112")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-seconds", type=int, default=300)
    args = parser.parse_args()
    parsed = urlparse(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("this smoke only accesses a local loopback HTTP service")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("credentials, query and fragment are not accepted")
    if not 1 <= args.wait_seconds <= 600:
        parser.error("wait-seconds must be in 1..600")
    destination = Path(args.output)
    if destination.exists():
        parser.error("output already exists; keep previous evidence")
    base = args.base_url.rstrip("/")
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Full-model loopback API functional smoke, NOT a serving speed comparison",
        "base_url": base,
        "checks": {},
        "accepted": False,
    }

    def request(path, body=None):
        headers = {"Content-Type": "application/json"} if body is not None else {}
        data = json.dumps(body).encode("utf-8") if body is not None else None
        return urlopen(Request(base + path, data=data, headers=headers), timeout=90)

    def get(path):
        with request(path) as response:
            return json.load(response)

    def post(path, body):
        start = time.perf_counter()
        with request(path, body) as response:
            data = json.load(response)
        return {"elapsed_ms": (time.perf_counter() - start) * 1000, "response": data}

    def stream(body, disconnect=False):
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        start = time.perf_counter()
        first_text_ms = None
        events = []
        done = False
        with request("/v1/chat/completions", body) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                events.append(event)
                if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                    if first_text_ms is None:
                        first_text_ms = (time.perf_counter() - start) * 1000
                    if disconnect:
                        break
        return {
            "elapsed_ms": (time.perf_counter() - start) * 1000,
            "first_text_ms": first_text_ms,
            "done": done,
            "client_disconnected": disconnect,
            "events": events,
        }

    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            try:
                health = get("/health")
                if health.get("ready"):
                    break
            except (HTTPError, URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("service readiness deadline exceeded")
            time.sleep(1)
        report["health_before"] = health
        assert health["backend"] == health["execution_backend"] == "native"
        assert health["quantization"] == "saddle-w8a16"
        quantization = health["native_int8"]
        assert quantization["status"] == "ready"
        assert quantization["cuda_linears"] == quantization["linear_count"] == 144
        assert quantization["kernel_backend"] == "triton_w8a16"
        report["models"] = get("/v1/models")
        model = report["models"]["data"][0]["id"]
        body = {
            "model": model,
            "messages": [{"role": "user", "content":
                "请用中文简洁解释大语言模型的 KV 缓存：它为什么能减少重复计算，"
                "以及上下文变长时会有什么显存成本？"}],
            "temperature": 0,
            "max_tokens": 64,
        }
        for name in ("chat_first", "chat_repeat"):
            result = post("/v1/chat/completions", body)
            assert result["response"]["choices"][0]["message"]["content"]
            assert 0 < result["response"]["usage"]["completion_tokens"] <= 64
            report["checks"][name] = result
        first = report["checks"]["chat_first"]["response"]
        repeated = report["checks"]["chat_repeat"]["response"]
        assert first["choices"][0]["message"]["content"] == repeated["choices"][0]["message"]["content"]
        assert first["usage"] == repeated["usage"]
        report["checks"]["completion"] = post("/v1/completions", {
            "model": model, "prompt": "The sum of 2 and 3 is", "temperature": 0, "max_tokens": 8,
        })
        assert report["checks"]["completion"]["response"]["choices"][0]["text"]
        streaming = stream(body)
        report["checks"]["stream"] = streaming
        assert streaming["done"] and streaming["first_text_ms"] is not None
        assert any((e.get("usage") or {}).get("completion_tokens", 0) > 0 for e in streaming["events"])
        report["checks"]["client_disconnect"] = stream({**body, "max_tokens": 256}, disconnect=True)
        report["checks"]["after_disconnect"] = post("/v1/chat/completions", {**body, "max_tokens": 8})
        assert report["checks"]["after_disconnect"]["response"]["choices"]
        budget = post("/v1/chat/completions", {**body, "max_tokens": 257})
        report["checks"]["output_budget_clamped"] = budget
        # The existing API clamps positive requests to the configured ceiling.
        assert 0 < budget["response"]["usage"]["completion_tokens"] <= 256
        try:
            post("/v1/chat/completions", {**body, "max_tokens": 0})
        except HTTPError as error:
            report["checks"]["invalid_budget_rejected"] = {
                "status": error.code, "response": error.read().decode("utf-8"),
            }
            assert error.code in {400, 422}
        else:
            raise AssertionError("invalid nonpositive token budget was not rejected")
        report["health_after"] = get("/health")
        assert report["health_after"]["native_int8"]["kernel_calls"] > quantization["kernel_calls"]
        assert report["health_after"]["prefix_cache"]["hits"] > health["prefix_cache"]["hits"]
        report["accepted"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"API smoke accepted: {report['accepted']}; report: {destination}", flush=True)


if __name__ == "__main__":
    main()
