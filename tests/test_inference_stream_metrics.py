import asyncio
import json
import threading

import httpx
import pytest
import torch

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.InferenceBatcher import InferenceBatcher, _BatchJob
from saddlellm.runtime.InferenceServer import (
    CompletionRequest,
    InferenceServerSettings,
    create_inference_app,
)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_full_batch_wakes_without_waiting_for_coalescing_deadline():
    async def run():
        entered = threading.Event()

        def generate(rows, events):
            entered.set()
            return rows

        batcher = InferenceBatcher(
            generate, asyncio.Semaphore(1), max_batch_size=2, wait_ms=5000
        )
        first = asyncio.create_task(batcher.submit("a", "same"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not entered.is_set()
        second = asyncio.create_task(batcher.submit("b", "same"))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            assert await asyncio.wait_for(asyncio.gather(first, second), 1) == [
                "a",
                "b",
            ]
        finally:
            await batcher.close()

    asyncio.run(run())


def test_token_saturated_batch_does_not_wait_to_fill_request_count():
    async def run():
        batcher = InferenceBatcher(
            lambda rows, events: rows,
            asyncio.Semaphore(1),
            max_batch_size=4,
            max_batch_tokens=8,
            wait_ms=5000,
        )
        try:
            assert (
                await asyncio.wait_for(
                    batcher.submit("a", "same", prompt_tokens=6, max_new_tokens=2), 1
                )
                == "a"
            )
        finally:
            await batcher.close()

    asyncio.run(run())


def test_coalescing_uses_oldest_admission_deadline(monkeypatch):
    async def run():
        batcher = InferenceBatcher(
            lambda rows, events: rows,
            asyncio.Semaphore(1),
            max_batch_size=2,
            wait_ms=10,
            clock=lambda: 0.004,
        )
        batcher.pending.append(
            _BatchJob(
                "a", "a", asyncio.get_running_loop().create_future(), admitted_at=0
            )
        )
        timeouts = []

        async def timeout(awaitable, *, timeout):
            timeouts.append(timeout)
            awaitable.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr(asyncio, "wait_for", timeout)
        await batcher._coalesce()
        assert timeouts == pytest.approx([0.006])
        batcher.pending.clear()
        await batcher.close()

    asyncio.run(run())


def test_close_wakes_coalescing_and_fails_pending_request():
    async def run():
        batcher = InferenceBatcher(
            lambda rows, events: pytest.fail("closed queue generated"),
            asyncio.Semaphore(1),
            max_batch_size=2,
            wait_ms=5000,
        )
        request = asyncio.create_task(batcher.submit("a", "a"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.wait_for(batcher.close(), 1)
        with pytest.raises(RuntimeError, match="shutting down"):
            await request
        assert batcher.worker is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "aged, expected",
    [(False, ["short_job", "long_output"]), (True, ["long_output", "short_job"])],
)
def test_shortest_work_includes_output_budget_and_preserves_aging(aged, expected):
    async def run():
        now, order = [0.0], []
        semaphore = asyncio.Semaphore(0)

        def generate(rows, events):
            order.extend(rows)
            return rows

        batcher = InferenceBatcher(
            generate,
            semaphore,
            max_batch_size=1,
            wait_ms=0,
            scheduling_policy="shortest_work",
            clock=lambda: now[0],
        )
        first = asyncio.create_task(
            batcher.submit("long_output", "a", prompt_tokens=1, max_new_tokens=128)
        )
        await asyncio.sleep(0)
        second = asyncio.create_task(
            batcher.submit("short_job", "b", prompt_tokens=20, max_new_tokens=1)
        )
        await asyncio.sleep(0)
        now[0] = 2.0 if aged else 0.01
        semaphore.release()
        try:
            assert await asyncio.gather(first, second) == ["long_output", "short_job"]
            assert order == expected
        finally:
            await batcher.close()

    asyncio.run(run())


def test_timing_metrics_include_queue_wait_padding_and_remain_bounded():
    async def run():
        now = [0.0]
        semaphore = asyncio.Semaphore(0)

        def generate(rows, events):
            now[0] += 0.004
            return rows

        batcher = InferenceBatcher(
            generate, semaphore, max_batch_size=2, wait_ms=0, clock=lambda: now[0]
        )
        assert batcher.snapshot()["queue_wait"]["p99_ms"] is None
        tasks = [
            asyncio.create_task(
                batcher.submit(str(n), "same", prompt_tokens=n, max_new_tokens=2)
            )
            for n in (2, 4)
        ]
        await asyncio.sleep(0)
        now[0] = 0.010
        semaphore.release()
        await asyncio.gather(*tasks)
        snapshot = batcher.snapshot()
        assert snapshot["queue_wait"]["samples"] == 2
        assert snapshot["queue_wait"]["p99_ms"] == pytest.approx(10)
        assert snapshot["batch_service"]["p50_ms"] == pytest.approx(4)
        assert snapshot["recent_padding_tokens"] == 2
        batcher._queue_wait_ms.extend(range(2048))
        snapshot = batcher.snapshot()
        assert snapshot["queue_wait"]["samples"] == 1024
        assert snapshot["queue_wait"]["p50_ms"] == 1535.5
        await batcher.close()

    asyncio.run(run())


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(self, ids, **kwargs):
        return "".join(f"{int(item)} " for item in ids if item not in {0, 2})


def make_app():
    torch.manual_seed(173)
    model = SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=13,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            eos_token_id=[],
        )
    ).eval()
    settings = InferenceServerSettings(
        default_max_new_tokens=4, max_new_tokens=8, web_ui=False
    )
    return create_inference_app(settings, model=model, tokenizer=Tokenizer())


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
@pytest.mark.parametrize("include_usage", [False, True])
def test_streaming_endpoints_have_equal_text_finish_reason_and_real_optional_usage(
    endpoint, include_usage
):
    async def run():
        app = make_app()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                body = {
                    "temperature": 0,
                    "max_tokens": 4,
                    **(
                        {"messages": [{"role": "user", "content": "hello"}]}
                        if endpoint == "chat/completions"
                        else {"prompt": "hello"}
                    ),
                }
                reference = (await client.post(f"/v1/{endpoint}", json=body)).json()
                response = await client.post(
                    f"/v1/{endpoint}",
                    json={
                        **body,
                        "stream": True,
                        "stream_options": {"include_usage": include_usage},
                    },
                )
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                events = [
                    line[6:]
                    for line in response.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert events[-1] == "[DONE]"
                chunks = [json.loads(line) for line in events[:-1]]
                text = "".join(
                    choice.get("text", choice.get("delta", {}).get("content", ""))
                    for chunk in chunks
                    for choice in chunk["choices"]
                )
                expected = reference["choices"][0]
                assert text == expected.get(
                    "text", expected.get("message", {}).get("content")
                )
                finished = [
                    choice
                    for chunk in chunks
                    for choice in chunk["choices"]
                    if choice["finish_reason"]
                ]
                assert (
                    len(finished) == 1
                    and finished[0]["finish_reason"] == expected["finish_reason"]
                )
                assert len({chunk["id"] for chunk in chunks}) == 1
                if include_usage:
                    assert chunks[-1]["choices"] == []
                    assert chunks[-1]["usage"] == reference["usage"]
                    assert all(chunk["usage"] is None for chunk in chunks[:-1])
                else:
                    assert all("usage" not in chunk for chunk in chunks)

    asyncio.run(run())


def test_streaming_echo_is_not_counted_as_generated_tokens():
    async def run():
        app = make_app()
        engine = app.state.engine
        request = CompletionRequest(
            prompt="hello", temperature=0, max_tokens=4, echo=True
        )
        reference = engine.complete(request)
        state = {}
        assert (
            "".join(engine.stream_complete(request, state))
            == reference["choices"][0]["text"]
        )
        assert (
            state["completion_tokens"] == reference["usage"]["completion_tokens"] == 4
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions", json={"prompt": "", "stream": True}
            )
            assert response.status_code == 400

    asyncio.run(run())


def test_streaming_server_errors_do_not_echo_private_data(monkeypatch):
    async def run():
        app = make_app()

        def fail(**kwargs):
            raise RuntimeError("private prompt or secret token")

        monkeypatch.setattr(app.state.engine.model, "forward", fail)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions", json={"prompt": "secret", "stream": True}
            )
            assert "server_error" in response.text and "[DONE]" in response.text
            assert "private" not in response.text and "secret" not in response.text

    asyncio.run(run())


@pytest.mark.parametrize("policy", ["bad", True, None, []])
def test_invalid_work_scheduling_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        InferenceServerSettings(batch_scheduling_policy=policy)
    with pytest.raises(ValueError):
        InferenceBatcher(
            lambda rows, events: rows,
            asyncio.Semaphore(1),
            max_batch_size=2,
            scheduling_policy=policy,
        )
