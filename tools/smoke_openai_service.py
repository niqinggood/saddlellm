"""Read-only loopback service checks using the official OpenAI Python SDK.

Does not install packages, start/stop services, or call the OpenAI cloud.
The new report contains public test prompts/outputs, never the API key.
Functional smoke only: not a model-quality or comparative throughput test.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import openai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8112")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-seconds", type=int, default=300)
    parser.add_argument("--api-key-env")
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.path not in {"", "/"} or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        parser.error("use a loopback HTTP server root URL without credentials")
    if not 1 <= args.wait_seconds <= 600:
        parser.error("wait-seconds must be in 1..600")
    destination = Path(args.output)
    if destination.exists():
        parser.error("output already exists; preserve previous evidence")
    key = os.environ.get(args.api_key_env) if args.api_key_env else "unused-local-sdk-key"
    if not key:
        parser.error("api-key-env is unset or empty")
    base = args.base_url.rstrip("/")
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Official SDK loopback functional smoke, NOT quality/throughput comparison",
        "sdk_version": version("openai"), "base_url": base,
        "checks": {}, "accepted": False,
    }
    try:
        with httpx.Client(base_url=base, timeout=90, trust_env=False) as http:
            deadline = time.monotonic() + args.wait_seconds
            while True:
                try:
                    response = http.get("/health", timeout=5)
                    response.raise_for_status()
                    health = response.json()
                    if health.get("ready"):
                        break
                except (httpx.HTTPError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("readiness deadline exceeded")
                time.sleep(1)
            report["health_before"] = health
            with openai.OpenAI(api_key=key, base_url=base + "/v1", max_retries=0,
                               http_client=httpx.Client(timeout=90, trust_env=False)) as sdk:
                models = sdk.models.list()
                report["models"] = models.model_dump()
                assert models.data, "service did not advertise a model"
                model = models.data[0].id
                messages = [{"role": "developer", "content": "Answer briefly."},
                            {"role": "user", "content": [{"type": "text", "text":
                                "Explain what a KV cache does in an LLM."}]}]
                result = sdk.chat.completions.create(
                    model=model, messages=messages, temperature=0, max_completion_tokens=32,
                )
                assert result.choices[0].message.content
                assert 0 < result.usage.completion_tokens <= 32
                report["checks"]["chat_developer_text_parts_token_alias"] = result.model_dump()
                limited = sdk.chat.completions.create(
                    model=model, messages=messages, temperature=0, max_completion_tokens=2,
                )
                assert 0 < limited.usage.completion_tokens <= 2
                report["checks"]["max_completion_tokens_applied"] = limited.model_dump()
                stream = sdk.chat.completions.create(
                    model=model, messages=messages, temperature=0, max_completion_tokens=16,
                    stream=True, stream_options={"include_usage": True},
                )
                try:
                    chunks = list(stream)
                finally:
                    stream.close()
                assert chunks and chunks[-1].choices == [] and chunks[-1].usage
                assert 0 < chunks[-1].usage.completion_tokens <= 16
                assert "".join(c.delta.content or "" for chunk in chunks for c in chunk.choices)
                assert len({chunk.id for chunk in chunks}) == 1
                report["checks"]["sdk_stream_usage"] = [chunk.model_dump() for chunk in chunks]
                completion = sdk.completions.create(
                    model=model, prompt="The sum of two and three is", temperature=0, max_tokens=8,
                )
                assert completion.choices[0].text and 0 < completion.usage.completion_tokens <= 8
                report["checks"]["completion"] = completion.model_dump()
                try:
                    sdk.completions.create(model="invalid-smoke-model", prompt="hello", max_tokens=2)
                except openai.NotFoundError as error:
                    report["checks"]["wrong_model_404"] = {"status": error.status_code, "body": error.body}
                else:
                    raise AssertionError("wrong model alias was silently accepted")
                try:
                    sdk.chat.completions.create(model=model, messages=messages, max_tokens=2,
                                                tools=[{"type": "function", "function": {"name": "smoke_tool"}}])
                except openai.UnprocessableEntityError as error:
                    assert error.body["param"] == "tools"
                    report["checks"]["unsupported_tools_rejected"] = {"status": error.status_code, "body": error.body}
                else:
                    raise AssertionError("unsupported tools were silently ignored")
                # An early closed HTTP stream must not keep the model/admission slot.
                stream = sdk.chat.completions.create(
                    model=model, messages=messages, temperature=0, max_completion_tokens=64,
                    stream=True,
                )
                try:
                    for chunk in stream:
                        if any(c.delta.content for c in chunk.choices):
                            report["checks"]["client_disconnect"] = {"first_text_received": True}
                            break
                    else:
                        raise AssertionError("no text before stream close")
                finally:
                    stream.close()
                after = sdk.completions.create(model=model, prompt="Hello", max_tokens=2, temperature=0)
                assert 0 < after.usage.completion_tokens <= 2
                report["checks"]["after_disconnect"] = after.model_dump()
            report["health_after"] = http.get("/health").json()
            assert report["health_after"]["ready"]
            admission = report["health_after"].get("admission", {})
            if admission.get("owner") == "local_http":
                assert admission["active"] == 0
        report["accepted"] = True
    except Exception as error:
        # Exception strings can include Authorization headers; do not copy them.
        report["error_type"] = type(error).__name__
        raise
    finally:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"Official SDK smoke accepted: {report['accepted']}; report: {destination}", flush=True)


if __name__ == "__main__":
    main()
