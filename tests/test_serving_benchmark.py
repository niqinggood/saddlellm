import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

script = Path(__file__).resolve().parents[1] / "tools/benchmark_serving.py"
spec = importlib.util.spec_from_file_location("serving_benchmark", script)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def _response(text="same", prompt_tokens=5):
    return {
        "model": "same-checkpoint",
        "choices": [{"text": text, "finish_reason": "length"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 3},
    }


def _args(**kwargs):
    values = dict(
        target=["saddle=http://one/v1", "other=http://two/v1"],
        model="same-checkpoint",
        prompt="input",
        prompts_file=None,
        requests=7,
        concurrency=2,
        max_tokens=3,
        warmup=1,
        repeats=2,
        timeout=1,
        api_key_env=None,
        checkpoint_sha256=None,
        cache_policy="server controlled",
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_benchmark_bounds_requests_preserves_order_and_requires_usage():
    active = maximum = 0
    calls = []

    async def handle(request):
        nonlocal active, maximum
        body = json.loads(request.content)
        assert request.url.path == "/v1/completions"
        assert body["temperature"] == 0
        assert body["stream"] is False
        assert body["max_tokens"] == 3
        calls.append(request.url.host)
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.001)
            return httpx.Response(200, json=_response())
        finally:
            active -= 1

    report = asyncio.run(benchmark.compare(_args(), httpx.MockTransport(handle)))
    assert maximum == 2
    assert len(calls) == 2 * 7 * 3
    assert report["all_token_counts_equal"]
    assert report["all_output_text_equal"]
    assert report["ttft"] == "not measured"
    assert report["remote_weights_and_hardware_verified"] is False
    assert report["targets"]["saddle"]["generated_tokens_per_second"] > 0
    assert report["targets"]["saddle"]["fixed_budget_response_fraction"] == 1


@pytest.mark.parametrize("mismatch", ["text", "counts"])
def test_benchmark_reports_non_equivalent_responses(mismatch):
    def handle(request):
        body = _response()
        if request.url.host == "two":
            body = _response(
                text="other" if mismatch == "text" else "same",
                prompt_tokens=8 if mismatch == "counts" else 5,
            )
        return httpx.Response(200, json=body)

    report = asyncio.run(benchmark.compare(_args(), httpx.MockTransport(handle)))
    assert report["all_output_text_equal"] is (mismatch != "text")
    assert report["all_token_counts_equal"] is (mismatch != "counts")


def test_unique_prompts_cover_warmups_and_rounds_identically_for_each_target():
    prompts = [f"unique-{index}" for index in range(6)]
    seen = {"one": [], "two": []}

    def handle(request):
        prompt = json.loads(request.content)["prompt"]
        seen[request.url.host].append(prompt)
        return httpx.Response(200, json=_response(prompt))

    report = asyncio.run(benchmark.compare(
        _args(requests=2, warmup=1, repeats=2, unique_prompts=True, capture_text=True),
        httpx.MockTransport(handle), prompts_override=prompts,
    ))
    assert seen == {"one": prompts, "two": prompts}
    assert report["unique_prompts"] and report["all_output_text_equal"]
    assert [row["text"] for row in report["targets"]["saddle"]["samples"]] == prompts[2:]


@pytest.mark.parametrize("prompts", [["a", "b"], ["a", "b", "a", "b", "c", "d"]])
def test_unique_prompts_reject_short_or_repeated_inputs_before_requests(prompts):
    def unexpected_request(_request):
        pytest.fail("invalid prompt schedule must not start requests")

    with pytest.raises(ValueError, match="6 distinct prompts"):
        asyncio.run(benchmark.compare(
            _args(requests=2, warmup=1, repeats=2, unique_prompts=True),
            httpx.MockTransport(unexpected_request), prompts_override=prompts,
        ))


def test_benchmark_does_not_estimate_missing_tokens_or_leak_server_errors():
    async def scenario():
        def handle(request):
            body = _response()
            body["usage"].pop("completion_tokens")
            return httpx.Response(200, json=body)

        async with httpx.AsyncClient(
            base_url="http://one/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            with pytest.raises(RuntimeError, match="no estimates"):
                await benchmark.run_workload(client, "model", ["secret input"], 3, 1)
        async with httpx.AsyncClient(
            base_url="http://one/v1/",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, text="secret input or credentials")
            ),
        ) as client:
            with pytest.raises(RuntimeError, match="HTTP 500") as error:
                await benchmark.run_workload(client, "model", ["secret input"], 3, 1)
            assert "secret" not in str(error.value)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "target",
    ["http://localhost/v1", "bad=http://secret:key@localhost/v1", "a=http://one/"],
)
def test_target_validation(target):
    with pytest.raises(ValueError):
        benchmark.parse_target(target)


@pytest.mark.parametrize("stream", [False, True])
def test_native_ollama_raw_nonthinking_usage_and_optional_text(stream):
    final = {
        "model": "qwen3:8b",
        "response": "答案",
        "done": True,
        "done_reason": "length",
        "prompt_eval_count": 12,
        "eval_count": 3,
        "prompt_eval_cached_count": 4,
        "eval_duration": 100000,
    }

    def handle(request):
        assert request.url.path == "/api/generate"
        payload = json.loads(request.content)
        assert payload["raw"] is True and payload["think"] is False
        assert payload["options"]["num_predict"] == 3
        assert payload["options"]["repeat_penalty"] == 1
        assert payload["options"]["num_ctx"] == 2048
        if stream:
            events = [
                {"response": "答", "done": False},
                {"response": "案", "done": False},
                {**final, "response": ""},
            ]
            return httpx.Response(
                200, content="\n".join(json.dumps(e) for e in events) + "\n"
            )
        return httpx.Response(200, json=final)

    report = asyncio.run(
        benchmark.compare(
            _args(
                target=["ollama=http://one/v1"],
                protocol="ollama",
                model="qwen3:8b",
                ollama_context_limit=2048,
                capture_text=True,
                stream=stream,
            ),
            httpx.MockTransport(handle),
        )
    )
    sample = report["targets"]["ollama"]["samples"][0]
    assert sample["text"] == "答案" and sample["prompt_tokens"] == 12
    assert sample["backend_metrics"]["prompt_eval_cached_count"] == 4
    assert report["protocol"] == "ollama"
    if stream:
        assert sample["ttft_ms"] is not None and sample["text_chunks"] == 2


@pytest.mark.parametrize(
    "body",
    [
        None,
        {"done": False},
        {"done": True, "thinking": "reason"},
        {"done": True, "response": "x", "eval_count": 1},
    ],
)
def test_ollama_rejects_incomplete_thinking_and_missing_usage(body):
    with pytest.raises(RuntimeError):
        benchmark.ollama_completion(body, 3)


def test_json_multiline_prompts_are_not_split(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(["line1\nline2", "second"]), encoding="utf-8")
    assert benchmark.load_prompts(path) == ["line1\nline2", "second"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        benchmark.load_prompts(path)


class Pieces(httpx.AsyncByteStream):
    def __init__(self, events, *, delay=0):
        self.events, self.delay = events, delay

    async def __aiter__(self):
        for event in self.events:
            if self.delay:
                await asyncio.sleep(self.delay)
            encoded = event.encode("utf-8")
            # Every UTF-8 and SSE boundary can be split by network framing.
            for start in range(0, len(encoded), 2):
                yield encoded[start : start + 2]


def stream_events(*, text="你好", usage=True, finished=True, done=True):
    events = [
        ": keepalive\r\n\r\n",
        "data: "
        + json.dumps({"choices": [{"text": "", "finish_reason": None}]})
        + "\n\n",
    ]
    events.append(
        "data: "
        + json.dumps(
            {
                "model": "same-checkpoint",
                "choices": [{"text": text, "finish_reason": None}],
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )
    if finished:
        events.append(
            "data: "
            + json.dumps({"choices": [{"text": "", "finish_reason": "length"}]})
            + "\n\n"
        )
    if usage:
        events.append(
            "data: "
            + json.dumps(
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
            )
            + "\n\n"
        )
    if done:
        events.append("data: [DONE]\n\n")
    return events


def test_stream_benchmark_frames_utf8_and_measures_nonempty_text_not_metadata(
    monkeypatch,
):
    async def run():
        elapsed = [0.0]

        class ClockPieces(Pieces):
            async def __aiter__(self):
                for event in self.events:
                    elapsed[0] += 0.01
                    yield event.encode()

        monkeypatch.setattr(benchmark.time, "perf_counter", lambda: elapsed[0])
        async with httpx.AsyncClient(
            base_url="http://one/v1/",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=ClockPieces(stream_events()))
            ),
        ) as client:
            rows, wall = await benchmark.run_workload(
                client, "model", ["input"], 3, 1, stream=True
            )
        assert rows[0]["ttft_ms"] == pytest.approx(30)
        assert rows[0]["latency_ms"] == pytest.approx(60)
        assert rows[0]["text_chunks"] == 1
        assert wall == pytest.approx(0.06)

    asyncio.run(run())


def test_stream_comparison_uses_real_counts_and_reports_tail_sample_counts():
    def handle(request):
        body = json.loads(request.content)
        assert body["stream"] and body["stream_options"] == {"include_usage": True}
        return httpx.Response(200, stream=Pieces(stream_events(), delay=0.001))

    report = asyncio.run(
        benchmark.compare(_args(stream=True), httpx.MockTransport(handle))
    )
    assert report["all_output_text_equal"] and report["all_token_counts_equal"]
    assert "first non-empty" in report["ttft"]
    for target in report["targets"].values():
        assert target["ttft_samples"] == target["latency_samples"] == 14
        assert (
            0
            < target["p50_ttft_ms"]
            <= target["p99_ttft_ms"]
            < target["p99_response_ms"]
        )
        assert target["p50_inter_text_chunk_ms"] is None


@pytest.mark.parametrize(
    "options, match",
    [
        ({"usage": False}, "no estimates"),
        ({"finished": False}, "incomplete"),
        ({"done": False}, "incomplete"),
    ],
)
def test_stream_benchmark_rejects_missing_usage_and_truncated_streams(options, match):
    async def run():
        async with httpx.AsyncClient(
            base_url="http://one/v1/",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, stream=Pieces(stream_events(**options))
                )
            ),
        ) as client:
            with pytest.raises(RuntimeError, match=match):
                await benchmark.run_workload(
                    client, "model", ["secret"], 3, 1, stream=True
                )

    asyncio.run(run())


def test_empty_text_stream_has_no_ttft_not_a_fabricated_zero():
    report = asyncio.run(
        benchmark.compare(
            _args(stream=True),
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, stream=Pieces(stream_events(text=""))
                )
            ),
        )
    )
    assert report["targets"]["saddle"]["ttft_samples"] == 0
    assert report["targets"]["saddle"]["p99_ttft_ms"] is None


def test_streaming_error_is_redacted_and_other_client_workers_are_cancelled():
    async def run():
        closed = asyncio.Event()

        class ErrorOrBlocked(httpx.AsyncByteStream):
            def __init__(self, fail):
                self.fail = fail

            async def __aiter__(self):
                if self.fail:
                    await asyncio.sleep(0.01)
                    yield b'data: {"error": {"message": "private prompt"}}\n\n'
                else:
                    await asyncio.Event().wait()

            async def aclose(self):
                if not self.fail:
                    closed.set()

        def handle(request):
            return httpx.Response(
                200,
                stream=ErrorOrBlocked(json.loads(request.content)["prompt"] == "fail"),
            )

        async with httpx.AsyncClient(
            base_url="http://one/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            with pytest.raises(RuntimeError, match="streaming error") as error:
                await benchmark.run_workload(
                    client, "model", ["fail", "wait"], 3, 2, stream=True
                )
        assert "private" not in str(error.value)
        assert closed.is_set()

    asyncio.run(run())


def test_sse_multiline_data_and_multiple_events_per_network_chunk():
    async def run():
        response = httpx.Response(
            200,
            stream=Pieces(
                [
                    'data: {"choices":\ndata: [{"text": "x", "finish_reason": "length"}],\ndata: "usage": {"prompt_tokens": 1, "completion_tokens": 1}}\n\ndata: [DONE]\n\n'
                ]
            ),
        )
        body, timing = await benchmark.read_stream(
            response, benchmark.time.perf_counter(), 1
        )
        assert body["choices"][0]["text"] == "x"
        assert timing["text_chunks"] == 1
        await response.aclose()

    asyncio.run(run())
