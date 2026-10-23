"""Compare already-running OpenAI-compatible completion servers.

Works with SaddleLLM, SGLang and vLLM /v1 endpoints. Does not start servers,
load models, verify remote weights/hardware, or estimate missing token counts.
Also supports Ollama's native /api/generate with --protocol ollama (raw prompts,
thinking disabled). Measures total latency and actual output throughput. --stream also measures
time to the first non-empty text chunk (not headers or role events). Chunk
timing is not per-token timing. Repeated prompts may benefit prefix caches.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def parse_target(value: str) -> tuple[str, str]:
    name, separator, url = value.partition("=")
    parsed = urlsplit(url)
    if (
        not separator
        or not name.strip()
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ValueError(
            "target must be LABEL=http(s)://HOST:PORT/v1, without credentials"
        )
    return name, url.rstrip("/") + "/"


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def validate_usage(usage, max_tokens):
    if not isinstance(usage, dict):
        raise RuntimeError("missing token usage; no estimates used")
    for field in ("prompt_tokens", "completion_tokens"):
        count = usage.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError(f"missing/invalid usage.{field}; no estimates used")
    if usage["completion_tokens"] > max_tokens:
        raise RuntimeError("server returned more tokens than the requested budget")


def load_prompts(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        prompts = json.loads(text)
        if not isinstance(prompts, list):
            raise ValueError("JSON prompts file must contain a list of strings")
        return prompts
    return [line for line in text.splitlines() if line]


def ollama_completion(body, max_tokens):
    if not isinstance(body, dict) or "error" in body or body.get("done") is not True:
        raise RuntimeError("incomplete/failed Ollama response")
    if body.get("thinking"):
        raise RuntimeError("Ollama thinking must be disabled for this experiment")
    text = body.get("response")
    if not isinstance(text, str):
        raise RuntimeError("Ollama response must contain text")
    usage = {
        "prompt_tokens": body.get("prompt_eval_count"),
        "completion_tokens": body.get("eval_count"),
    }
    validate_usage(usage, max_tokens)
    metrics = {
        name: body[name]
        for name in (
            "prompt_eval_cached_count",
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        )
        if name in body
    }
    return {
        "model": body.get("model"),
        "choices": [{"text": text, "finish_reason": body.get("done_reason")}],
        "usage": usage,
        "backend_metrics": metrics,
    }


async def read_ollama_stream(response, started, max_tokens):
    parts, arrivals = [], []
    final = None
    async for line in response.aiter_lines():
        if not line:
            continue
        if len(line) > 1024 * 1024:
            raise RuntimeError("Ollama event exceeds benchmark size limit")
        try:
            body = json.loads(line)
        except ValueError:
            raise RuntimeError("invalid Ollama stream JSON") from None
        if not isinstance(body, dict) or "error" in body:
            raise RuntimeError("Ollama reported a streaming error")
        if body.get("thinking"):
            raise RuntimeError("Ollama thinking must be disabled for this experiment")
        value = body.get("response", "")
        if not isinstance(value, str):
            raise RuntimeError("invalid Ollama stream text")
        if value:
            parts.append(value)
            arrivals.append((time.perf_counter() - started) * 1000)
        if body.get("done") is True:
            final = {**body, "response": "".join(parts)}
            break
    return ollama_completion(final, max_tokens), {
        "ttft_ms": arrivals[0] if arrivals else None,
        "text_chunks": len(arrivals),
        "inter_text_chunk_ms": [
            right - left for left, right in zip(arrivals, arrivals[1:])
        ],
    }


async def sse_events(response):
    """Frame SSE across arbitrary byte/UTF-8 boundaries, including multiline data."""
    data = []
    size = 0
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data, size = [], 0
            continue
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            size += len(value)
            if size > 1024 * 1024:
                raise RuntimeError("SSE event exceeds benchmark size limit")
            data.append(value)
    if data:
        yield "\n".join(data)


async def read_stream(response, started, max_tokens):
    parts, arrivals = [], []
    usage, finish_reason, reported_model = None, None, None
    done = False
    async for event in sse_events(response):
        if event == "[DONE]":
            done = True
            break
        try:
            body = json.loads(event)
        except ValueError:
            raise RuntimeError("invalid SSE JSON") from None
        if not isinstance(body, dict) or "error" in body:
            raise RuntimeError("server reported a streaming error")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise RuntimeError("server must stream a single completion choice")
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get("text"), str):
                raise RuntimeError("server must stream text completion choices")
            if choice["text"]:
                arrivals.append((time.perf_counter() - started) * 1000)
                parts.append(choice["text"])
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
        if body.get("usage") is not None:
            validate_usage(body["usage"], max_tokens)
            if usage is not None and body["usage"] != usage:
                raise RuntimeError("stream returned conflicting token usage")
            usage = body["usage"]
        reported_model = body.get("model", reported_model)
    if not done or finish_reason is None:
        raise RuntimeError("incomplete SSE stream; missing finish reason or DONE")
    validate_usage(usage, max_tokens)
    return {
        "usage": usage,
        "choices": [{"text": "".join(parts), "finish_reason": finish_reason}],
        "model": reported_model,
    }, {
        "ttft_ms": arrivals[0] if arrivals else None,
        "text_chunks": len(arrivals),
        "inter_text_chunk_ms": [
            right - left for left, right in zip(arrivals, arrivals[1:])
        ],
    }


async def run_workload(
    client,
    model,
    prompts,
    max_tokens,
    concurrency,
    *,
    stream=False,
    protocol="openai",
    context_limit=4096,
    capture_text=False,
):
    """Bounded worker count; output order is independent of response order."""
    if protocol not in {"openai", "ollama"}:
        raise ValueError("protocol must be openai or ollama")
    results = [None] * len(prompts)
    pending = iter(enumerate(prompts))

    async def worker():
        for index, prompt in pending:
            started = time.perf_counter()
            try:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": stream,
                }
                timing = {}
                if protocol == "ollama":
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "raw": True,
                        "think": False,
                        "stream": stream,
                        "keep_alive": "5m",
                        "options": {
                            "num_predict": max_tokens,
                            "num_ctx": context_limit,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "repeat_penalty": 1.0,
                            "seed": 173,
                        },
                    }
                    url = client.base_url.copy_with(path="/api/generate")
                    if stream:
                        async with client.stream("POST", url, json=payload) as response:
                            response.raise_for_status()
                            body, timing = await read_ollama_stream(
                                response, started, max_tokens
                            )
                    else:
                        response = await client.post(url, json=payload)
                        response.raise_for_status()
                        body = ollama_completion(response.json(), max_tokens)
                elif stream:
                    payload["stream_options"] = {"include_usage": True}
                    async with client.stream(
                        "POST", "completions", json=payload
                    ) as response:
                        response.raise_for_status()
                        body, timing = await read_stream(response, started, max_tokens)
                else:
                    response = await client.post("completions", json=payload)
                    response.raise_for_status()
                    body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                reason = (
                    f"HTTP {exc.response.status_code}"
                    if isinstance(exc, httpx.HTTPStatusError)
                    else type(exc).__name__
                )
                raise RuntimeError(f"benchmark request failed ({reason})") from None
            elapsed = (time.perf_counter() - started) * 1000
            if (
                not isinstance(body, dict)
                or not isinstance(body.get("usage"), dict)
                or not isinstance(body.get("choices"), list)
                or len(body["choices"]) != 1
                or not isinstance(body["choices"][0], dict)
                or not isinstance(body["choices"][0].get("text"), str)
            ):
                raise RuntimeError("server must return one text choice and token usage")
            usage = body["usage"]
            validate_usage(usage, max_tokens)
            results[index] = {
                "latency_ms": elapsed,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "text_sha256": hashlib.sha256(
                    body["choices"][0]["text"].encode("utf-8")
                ).hexdigest(),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "reported_model": body.get("model"),
                **timing,
            }
            if "backend_metrics" in body:
                results[index]["backend_metrics"] = body["backend_metrics"]
            if capture_text:
                results[index]["text"] = body["choices"][0]["text"]

    started = time.perf_counter()
    tasks = [
        asyncio.create_task(worker()) for _ in range(min(concurrency, len(prompts)))
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return results, time.perf_counter() - started


async def compare(args, transport=None, *, prompts_override=None):
    streaming = getattr(args, "stream", False)
    protocol = getattr(args, "protocol", "openai")
    workload_options = {
        "stream": streaming,
        "protocol": protocol,
        "context_limit": getattr(args, "ollama_context_limit", 4096),
        "capture_text": getattr(args, "capture_text", False),
    }
    targets = dict(parse_target(value) for value in args.target)
    if len(targets) != len(args.target):
        raise ValueError("target labels must be unique")
    prompts = (
        list(prompts_override)
        if prompts_override is not None
        else (load_prompts(args.prompts_file) if args.prompts_file else [args.prompt])
    )
    if not prompts or any(
        not isinstance(prompt, str) or not prompt for prompt in prompts
    ):
        raise ValueError("provide at least one non-empty prompt")
    unique_prompts = getattr(args, "unique_prompts", False)
    if unique_prompts:
        required = (args.warmup + args.repeats) * args.requests
        if len(prompts) < required or len(set(prompts[:required])) != required:
            raise ValueError(f"unique-prompts requires at least {required} distinct prompts")
        prompts = prompts[:required]
    else:
        prompts = [prompts[index % len(prompts)] for index in range(args.requests)]

    def round_prompts(index):
        if unique_prompts:
            return prompts[index * args.requests:(index + 1) * args.requests]
        return prompts
    headers = {}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            raise ValueError("api_key_env is unset or empty")
        headers["Authorization"] = f"Bearer {key}"
    records = {name: [] for name in targets}
    walls = {name: [] for name in targets}
    rng = random.Random(173)
    # Targets are measured sequentially; simultaneous GPU engines can compete
    # for memory even when inactive. Operators must control deployment state.
    clients = {
        name: httpx.AsyncClient(
            base_url=url,
            headers=headers,
            timeout=args.timeout,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
            limits=httpx.Limits(max_connections=args.concurrency),
        )
        for name, url in targets.items()
    }
    try:
        for name, client in clients.items():
            for warmup_index in range(args.warmup):
                await run_workload(
                    client,
                    args.model,
                    round_prompts(warmup_index),
                    args.max_tokens,
                    args.concurrency,
                    **workload_options,
                )
        for round_index in range(args.repeats):
            order = list(targets)
            rng.shuffle(order)
            for name in order:
                rows, elapsed = await run_workload(
                    clients[name],
                    args.model,
                    round_prompts(args.warmup + round_index),
                    args.max_tokens,
                    args.concurrency,
                    **workload_options,
                )
                records[name].append(rows)
                walls[name].append(elapsed)
    finally:
        await asyncio.gather(*(client.aclose() for client in clients.values()))
    first = records[next(iter(targets))]
    report = {
        "scope": (
            f"{'streaming' if streaming else 'non-streaming'} HTTP completions; total latency, actual generated tokens"
        ),
        "ttft": "first non-empty streamed text; includes tokenizer buffering"
        if streaming
        else "not measured",
        "load_model": "closed-loop bounded client workers; not an open-loop arrival trace",
        "model_alias": args.model,
        "protocol": protocol,
        "ollama_context_limit": workload_options["context_limit"]
        if protocol == "ollama"
        else None,
        "checkpoint_sha256_asserted_by_operator": args.checkpoint_sha256,
        "remote_weights_and_hardware_verified": False,
        "cache_policy": args.cache_policy,
        "unique_prompts": unique_prompts,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "requests_per_round": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "warmup_rounds": args.warmup,
        "repeats": args.repeats,
        "targets": {},
    }
    for name, rounds in records.items():
        flattened = [row for rows in rounds for row in rows]
        latencies = [row["latency_ms"] for row in flattened]
        counts_equal = all(
            (row["prompt_tokens"], row["completion_tokens"])
            == (ref["prompt_tokens"], ref["completion_tokens"])
            for rows, refs in zip(rounds, first)
            for row, ref in zip(rows, refs)
        )
        outputs_equal = all(
            row["text_sha256"] == ref["text_sha256"]
            for rows, refs in zip(rounds, first)
            for row, ref in zip(rows, refs)
        )
        report["targets"][name] = {
            "p50_response_ms": statistics.median(latencies),
            "p95_response_ms": percentile(latencies, 0.95),
            "p99_response_ms": percentile(latencies, 0.99),
            "latency_samples": len(latencies),
            "requests_per_second": len(flattened) / sum(walls[name]),
            "generated_tokens_per_second": (
                sum(row["completion_tokens"] for row in flattened) / sum(walls[name])
            ),
            "fixed_budget_response_fraction": sum(
                row["completion_tokens"] == args.max_tokens for row in flattened
            )
            / len(flattened),
            "token_counts_equal_to_first_target": counts_equal,
            "output_text_equal_to_first_target": outputs_equal,
            "samples": flattened,
        }
        if streaming:
            ttfts = [row["ttft_ms"] for row in flattened if row["ttft_ms"] is not None]
            gaps = [gap for row in flattened for gap in row["inter_text_chunk_ms"]]
            report["targets"][name].update(
                {
                    "ttft_samples": len(ttfts),
                    "p50_ttft_ms": statistics.median(ttfts) if ttfts else None,
                    "p95_ttft_ms": percentile(ttfts, 0.95) if ttfts else None,
                    "p99_ttft_ms": percentile(ttfts, 0.99) if ttfts else None,
                    "p50_inter_text_chunk_ms": statistics.median(gaps)
                    if gaps
                    else None,
                    "p95_inter_text_chunk_ms": percentile(gaps, 0.95) if gaps else None,
                }
            )
    report["all_token_counts_equal"] = all(
        target["token_counts_equal_to_first_target"]
        for target in report["targets"].values()
    )
    report["all_output_text_equal"] = all(
        target["output_text_equal_to_first_target"]
        for target in report["targets"].values()
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", required=True, help="LABEL=URL/v1")
    parser.add_argument(
        "--model", required=True, help="Same served model alias on all targets"
    )
    parser.add_argument(
        "--prompt", default="Once upon a time, there was a small village"
    )
    parser.add_argument(
        "--prompts-file", help="UTF-8 lines, or .json list for multiline raw prompts"
    )
    parser.add_argument("--protocol", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--ollama-context-limit", type=int, default=4096)
    parser.add_argument(
        "--capture-text",
        action="store_true",
        help="Store outputs; use only non-sensitive test prompts",
    )
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--unique-prompts", action="store_true",
                        help="Use distinct prompts per round, including warmups; file must cover all requests")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Measure first visible SSE text; requires actual final usage",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key-env")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--cache-policy", default="server-controlled; repeated prompts")
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens, args.repeats) < 1:
        parser.error("request/concurrency/token/repeat counts must be positive")
    if args.requests > 65536 or args.warmup < 0:
        parser.error("requests must be <= 65536 and warmup non-negative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    if args.ollama_context_limit < 1:
        parser.error("ollama context limit must be positive")
    report = asyncio.run(compare(args))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["all_token_counts_equal"] or not report["all_output_text_equal"]:
        raise SystemExit(
            "Responses differ; inspect weights/tokenizers/numerics before comparison."
        )


if __name__ == "__main__":
    main()
