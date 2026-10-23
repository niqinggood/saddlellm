import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.InferenceBatcher import InferenceBatcher, InferenceQueueFull
from saddlellm.runtime.InferenceServer import (
    ChatCompletionRequest,
    CompletionRequest,
    InferenceServerSettings,
    LocalGenerationEngine,
    create_inference_app,
)


class NumericTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        ids = [int(item) for item in text.split()][: kwargs["max_length"]]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(item) for item in ids if item not in {0, 2})

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(message["content"] for message in messages) + " 3"


def make_engine(**kwargs):
    torch.manual_seed(173)
    model = SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_backend="sdpa",
        )
    ).eval()
    settings = InferenceServerSettings(
        batch_size=4,
        default_max_new_tokens=5,
        max_new_tokens=8,
        **kwargs,
    )
    return LocalGenerationEngine(model, NumericTokenizer(), settings)


def content(response):
    return response["choices"], response["usage"]


def record_batches(monkeypatch, model):
    sizes = []
    generate = model.generate

    def recorded(**kwargs):
        sizes.append(kwargs["input_ids"].shape[0])
        return generate(**kwargs)

    monkeypatch.setattr(model, "generate", recorded)
    return sizes


def test_native_batch_matches_individual_greedy_results(monkeypatch):
    engine = make_engine()
    requests = [
        CompletionRequest(prompt=prompt, temperature=0, max_tokens=5)
        for prompt in ["4 5", "6 7", "8 9"]
    ]
    reference = [engine.complete(request) for request in requests]
    sizes = record_batches(monkeypatch, engine.model)
    actual = engine.complete_batch(requests)
    assert sizes == [3]
    assert [content(row) for row in actual] == [content(row) for row in reference]
    assert len({row["id"] for row in actual}) == 3


def test_batch_groups_parameters_and_lengths_and_preserves_order(monkeypatch):
    engine = make_engine()
    requests = [
        CompletionRequest(prompt=prompt, temperature=0, max_tokens=count)
        for prompt, count in [("4 5", 3), ("6", 3), ("8 9", 3), ("10 11", 4)]
    ]
    reference = [engine.complete(request) for request in requests]
    sizes = record_batches(monkeypatch, engine.model)
    actual = engine.complete_batch(requests)
    assert sizes == [2, 1, 1]
    assert [content(row) for row in actual] == [content(row) for row in reference]


def test_chat_batch_and_completion_echo(monkeypatch):
    engine = make_engine()
    chats = [
        ChatCompletionRequest(
            messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=3
        )
        for prompt in ["4 5", "6 7"]
    ]
    reference = [engine.chat(row) for row in chats]
    sizes = record_batches(monkeypatch, engine.model)
    actual = engine.chat_batch(chats)
    assert sizes == [2]
    assert [content(row) for row in actual] == [content(row) for row in reference]
    echoed = engine.complete_batch(
        [
            CompletionRequest(prompt="4 5", temperature=0, max_tokens=3, echo=True),
        ]
    )[0]
    assert echoed["choices"][0]["text"].startswith("4 5")
    assert engine.complete_batch([]) == []


def test_batch_limit_is_enforced_for_import_api(monkeypatch):
    engine = make_engine()
    sizes = record_batches(monkeypatch, engine.model)
    engine.complete_batch(
        [CompletionRequest(prompt="4 5", temperature=0) for _ in range(9)]
    )
    assert sizes == [4, 4, 1]


def test_sampling_and_non_native_models_fall_back(monkeypatch):
    engine = make_engine()
    sizes = record_batches(monkeypatch, engine.model)
    engine.complete_batch(
        [CompletionRequest(prompt="4 5", temperature=0.7) for _ in range(2)]
    )
    assert sizes == [1, 1]
    engine.model = SimpleNamespace()
    assert not engine.can_batch(CompletionRequest(prompt="4 5", temperature=0))
    with pytest.raises(ValueError, match="stream"):
        engine.complete_batch([CompletionRequest(prompt="4 5", stream=True)])


def test_row_eos_stop_and_cancel_do_not_inflate_usage(monkeypatch):
    engine = make_engine()
    requests = [
        CompletionRequest(prompt="4 5", temperature=0, stop=stop)
        for stop in [None, "7", None, None]
    ]
    prepared = [engine.prepare_batch_request(row) for row in requests]
    events = [threading.Event() for _ in requests]
    events[2].set()

    def scripted(input_ids, stopping_criteria, **kwargs):
        finished = torch.zeros(4, dtype=torch.bool)
        out = input_ids
        # Row 0 hits EOS, row 1 hits a stop string, row 2 is cancelled,
        # row 3 runs to its length limit. Already finished rows receive PAD.
        for step in range(5):
            token = torch.tensor([2, 7, 8, 9]) if step == 0 else torch.full((4,), 9)
            token = torch.where(finished, 0, token)
            out = torch.cat([out, token[:, None]], dim=1)
            finished |= stopping_criteria(out, None)
        return out

    monkeypatch.setattr(engine.model, "generate", scripted)
    actual = engine.generate_batch(prepared, events)
    assert [row["usage"]["completion_tokens"] for row in actual] == [1, 1, 1, 5]
    assert [row["choices"][0]["finish_reason"] for row in actual] == [
        "stop",
        "stop",
        "stop",
        "length",
    ]
    assert [row["choices"][0]["text"] for row in actual] == ["", "", "8", "9 9 9 9 9"]


def test_stop_string_in_prompt_is_not_a_generated_stop(monkeypatch):
    engine = make_engine()
    requests = [
        CompletionRequest(prompt="4 5", temperature=0, stop="4") for _ in range(2)
    ]
    prepared = [engine.prepare_batch_request(row) for row in requests]

    def scripted(input_ids, stopping_criteria, **kwargs):
        output = torch.cat([input_ids, torch.tensor([[8], [9]])], dim=1)
        assert not stopping_criteria(output, None).any()
        return output

    monkeypatch.setattr(engine.model, "generate", scripted)
    actual = engine.generate_batch(prepared)
    assert [row["choices"][0]["text"] for row in actual] == ["8", "9"]


def test_batch_rejects_incompatible_prepared_rows():
    engine = make_engine()
    rows = [
        engine.prepare_batch_request(CompletionRequest(prompt=text, temperature=0))
        for text in ["4", "4 5"]
    ]
    with pytest.raises(ValueError, match="matching"):
        engine.generate_batch(rows)
    with pytest.raises(ValueError, match="empty"):
        engine.prepare_batch_request(CompletionRequest(prompt="", temperature=0))


def test_native_generate_stops_rows_independently_and_keeps_exact_lengths(monkeypatch):
    engine = make_engine()

    def logits(input_ids, **kwargs):
        scores = torch.zeros((input_ids.shape[0], 1, 32))
        for row, token in enumerate(input_ids[:, 0].tolist()):
            scores[row, 0, {4: 2, 5: 7, 6: 9}[token]] = 10
        return SimpleNamespace(logits=scores, past_key_values=None)

    monkeypatch.setattr(engine.model, "forward", logits)
    responses = engine.complete_batch(
        [
            CompletionRequest(prompt="4 8", temperature=0),
            CompletionRequest(prompt="5 8", temperature=0, stop="7"),
            CompletionRequest(prompt="6 8", temperature=0),
        ]
    )
    assert [row["usage"]["completion_tokens"] for row in responses] == [1, 1, 5]
    assert [row["choices"][0]["text"] for row in responses] == ["", "", "9 9 9 9 9"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_wait_ms": -1},
        {"batch_wait_ms": 1001},
        {"max_pending_requests": 0},
        {"batch_size": 1.5},
        {"batch_size": True},
    ],
)
def test_batch_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        InferenceServerSettings(**kwargs)


def test_scheduler_groups_compatible_jobs_with_bounded_batches():
    async def run():
        seen = []

        def generate(payloads, events):
            seen.append(payloads)
            return [value * 10 for value in payloads]

        batcher = InferenceBatcher(generate, asyncio.Semaphore(1), max_batch_size=2)
        results = await asyncio.gather(
            *(
                batcher.submit(value, key)
                for value, key in [(1, "a"), (2, "b"), (3, "a"), (4, "a")]
            )
        )
        assert results == [10, 20, 30, 40]
        assert seen == [[1, 3], [2], [4]]
        assert batcher.snapshot()["largest_batch"] == 2
        assert batcher.snapshot()["completed_requests"] == 4
        await batcher.close()

    asyncio.run(run())


def test_cancelled_queued_request_is_not_executed():
    async def run():
        seen = []
        semaphore = asyncio.Semaphore(0)
        batcher = InferenceBatcher(
            lambda rows, events: seen.extend(rows) or rows,
            semaphore,
            max_batch_size=2,
            wait_ms=0,
        )
        first = asyncio.create_task(batcher.submit(1, "a"))
        second = asyncio.create_task(batcher.submit(2, "a"))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        semaphore.release()
        assert await second == 2
        assert seen == [2]
        await batcher.close()

    asyncio.run(run())


def test_cancelled_active_request_sets_only_its_event_and_holds_permit():
    async def run():
        entered, release = threading.Event(), threading.Event()
        captured = []

        def generate(rows, events):
            captured.extend(events)
            entered.set()
            assert release.wait(5)
            return rows

        semaphore = asyncio.Semaphore(1)
        batcher = InferenceBatcher(generate, semaphore, max_batch_size=2)
        first = asyncio.create_task(batcher.submit(1, "a"))
        second = asyncio.create_task(batcher.submit(2, "a"))
        assert await asyncio.to_thread(entered.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert captured[0].is_set() and not captured[1].is_set()
        assert semaphore.locked()
        release.set()
        assert await second == 2
        await batcher.close()
        assert not semaphore.locked()

    asyncio.run(run())


def test_queue_overflow_is_explicit_and_shutdown_drains_worker():
    async def run():
        semaphore = asyncio.Semaphore(0)
        batcher = InferenceBatcher(
            lambda rows, events: rows,
            semaphore,
            max_batch_size=2,
            max_pending=1,
            wait_ms=0,
        )
        first = asyncio.create_task(batcher.submit(1, "a"))
        await asyncio.sleep(0)
        with pytest.raises(InferenceQueueFull):
            await batcher.submit(2, "a")
        closing = asyncio.create_task(batcher.close())
        await asyncio.sleep(0)
        semaphore.release()
        await closing
        with pytest.raises(RuntimeError, match="shutting down"):
            await first
        with pytest.raises(RuntimeError, match="closed"):
            await batcher.submit(3, "a")

    asyncio.run(run())


def test_failed_batch_does_not_poison_following_requests():
    async def run():
        def generate(rows, events):
            if rows[0] == "bad":
                raise RuntimeError("model failed")
            return rows

        batcher = InferenceBatcher(generate, asyncio.Semaphore(1), max_batch_size=2)
        with pytest.raises(RuntimeError, match="model failed"):
            await batcher.submit("bad", "a")
        assert await batcher.submit("good", "a") == "good"
        await batcher.close()

    asyncio.run(run())


def test_worker_cancellation_does_not_release_running_model_permit():
    async def run():
        entered, release = threading.Event(), threading.Event()

        def generate(rows, events):
            entered.set()
            assert release.wait(5)
            assert events[0].is_set()
            return rows

        semaphore = asyncio.Semaphore(1)
        batcher = InferenceBatcher(generate, semaphore, max_batch_size=2, wait_ms=0)
        job = asyncio.create_task(batcher.submit(1, "a"))
        assert await asyncio.to_thread(entered.wait, 5)
        worker = batcher.worker
        worker.cancel()
        await asyncio.sleep(0)
        assert semaphore.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await worker
        with pytest.raises(asyncio.CancelledError):
            await job
        assert not semaphore.locked()
        await batcher.close()

    asyncio.run(run())


def test_service_shutdown_signals_active_rows_and_waits_for_model():
    async def run():
        entered = threading.Event()

        def generate(rows, events):
            entered.set()
            assert events[0].wait(5)
            return rows

        semaphore = asyncio.Semaphore(1)
        batcher = InferenceBatcher(generate, semaphore, max_batch_size=2, wait_ms=0)
        job = asyncio.create_task(batcher.submit(1, "a"))
        assert await asyncio.to_thread(entered.wait, 5)
        await batcher.close()
        with pytest.raises(RuntimeError, match="shutting down"):
            await job
        assert batcher.worker is None
        assert not semaphore.locked()

    asyncio.run(run())


def test_invalid_batch_result_count_is_reported_to_all_clients():
    async def run():
        batcher = InferenceBatcher(
            lambda rows, events: [], asyncio.Semaphore(1), max_batch_size=2
        )
        with pytest.raises(RuntimeError, match="invalid result count"):
            await batcher.submit(1, "a")
        await batcher.close()

    asyncio.run(run())


def test_http_concurrent_native_requests_are_batched_and_keep_response_order(
    monkeypatch,
):
    async def run():
        engine = make_engine()
        sizes = record_batches(monkeypatch, engine.model)
        app = create_inference_app(
            engine.settings, model=engine.model, tokenizer=engine.tokenizer
        )
        # Threaded tokenization can finish >5 ms apart on a busy CPU. Control
        # cohort arrival at submit, not incidental operating-system timing.
        original_submit = app.state.batcher.submit
        cohort_ready = asyncio.Event()
        cohort_arrivals = 0

        async def submit_together(*args, **kwargs):
            nonlocal cohort_arrivals
            cohort_arrivals += 1
            if cohort_arrivals == 3:
                cohort_ready.set()
            await asyncio.wait_for(cohort_ready.wait(), timeout=5)
            return await original_submit(*args, **kwargs)

        monkeypatch.setattr(app.state.batcher, "submit", submit_together)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                responses = await asyncio.gather(
                    *(
                        client.post(
                            "/v1/completions",
                            json={
                                "prompt": prompt,
                                "temperature": 0,
                                "max_tokens": 3,
                            },
                        )
                        for prompt in ["4 5", "6 7", "8 9"]
                    )
                )
                assert all(response.status_code == 200 for response in responses)
                assert sizes == [3]
                health = (await client.get("/health")).json()
                assert health["batching"]["enabled"]
                assert health["batching"]["completed_requests"] == 3
                for prompt, response in zip(["4 5", "6 7", "8 9"], responses):
                    reference = engine.complete(
                        CompletionRequest(
                            prompt=prompt,
                            temperature=0,
                            max_tokens=3,
                        )
                    )
                    assert content(response.json()) == content(reference)

    asyncio.run(run())


def test_http_queue_full_returns_retryable_503(monkeypatch):
    async def run():
        engine = make_engine(max_pending_requests=1, batch_wait_ms=100)
        app = create_inference_app(
            engine.settings, model=engine.model, tokenizer=engine.tokenizer
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                responses = await asyncio.gather(
                    *(
                        client.post(
                            "/v1/completions",
                            json={
                                "prompt": "4 5",
                                "temperature": 0,
                                "max_tokens": 3,
                            },
                        )
                        for _ in range(2)
                    )
                )
                assert sorted(response.status_code for response in responses) == [
                    200,
                    503,
                ]
                busy = next(
                    response for response in responses if response.status_code == 503
                )
                assert busy.headers["retry-after"] == "1"

    asyncio.run(run())


def test_streaming_and_sampling_bypass_batch_queue(monkeypatch):
    async def run():
        engine = make_engine()
        sizes = record_batches(monkeypatch, engine.model)
        app = create_inference_app(
            engine.settings, model=engine.model, tokenizer=engine.tokenizer
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                stream = await client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "4 5"}],
                        "temperature": 0,
                        "max_tokens": 3,
                        "stream": True,
                    },
                )
                sample = await client.post(
                    "/v1/completions",
                    json={
                        "prompt": "4 5",
                        "temperature": 0.7,
                        "max_tokens": 3,
                    },
                )
                assert stream.status_code == sample.status_code == 200
                assert "data: [DONE]" in stream.text
                assert sizes == [1, 1]
                assert app.state.batcher.snapshot()["completed_batches"] == 0

    asyncio.run(run())


def test_cancelled_single_http_request_keeps_model_permit(monkeypatch):
    async def run():
        engine = make_engine()
        engine.settings.batch_size = 1
        app = create_inference_app(
            engine.settings, model=engine.model, tokenizer=engine.tokenizer
        )
        entered, release, second_entered = (threading.Event() for _ in range(3))
        calls = []
        original = engine.model.generate

        def blocking(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                assert release.wait(5)
            else:
                second_entered.set()
            return original(**kwargs)

        monkeypatch.setattr(engine.model, "generate", blocking)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                body = {"prompt": "4 5", "temperature": 0, "max_tokens": 3}
                first = asyncio.create_task(client.post("/v1/completions", json=body))
                assert await asyncio.to_thread(entered.wait, 5)
                first.cancel()
                second = asyncio.create_task(client.post("/v1/completions", json=body))
                try:
                    await asyncio.sleep(0.03)
                    assert not second_entered.is_set()
                finally:
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert (await second).status_code == 200
                assert second_entered.is_set()

    asyncio.run(run())


def test_batch_switch_reaches_server_settings(monkeypatch):
    import uvicorn

    from saddlellm import cli
    from saddlellm.runtime import InferenceServer

    captured = []
    monkeypatch.setattr(
        InferenceServer,
        "create_inference_app",
        lambda settings: captured.append(settings) or object(),
    )
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    assert cli.main(["serve", "release", "--batch-size", "4"]) == 0
    assert captured[0].batch_size == 4
    assert captured[0].max_concurrency == 1


def test_serve_cli_exposes_one_batch_switch(monkeypatch):
    from saddlellm import cli

    captured = {}
    monkeypatch.setattr(
        cli, "_cmd_serve_model", lambda args: captured.update(vars(args)) or 0
    )
    assert cli.main(["serve", "release", "--batch-size", "4"]) == 0
    assert captured["batch_size"] == 4


def test_benchmark_can_load_a_real_local_native_checkpoint(tmp_path):
    engine = make_engine()
    engine.model.save_pretrained(str(tmp_path))
    script = (
        Path(__file__).resolve().parents[1] / "tools" / "benchmark_native_batching.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(tmp_path),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--prompt-tokens",
            "8",
            "--decode-tokens",
            "3",
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    report = json.loads(result.stdout)
    assert report["outputs_equal"]
    assert report["session_outputs_equal"]
    assert report["session_phases"]["prefill_ms"] > 0
    assert report["session_phases"]["decode_step_ms"] > 0
    assert report["model"] == str(tmp_path)
    assert report["batch_size"] == 2
    assert report["individual"]["aggregate_decode_tokens_per_second"] > 0
