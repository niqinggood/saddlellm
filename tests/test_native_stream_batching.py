import asyncio
import gc
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import torch

from saddlellm.runtime.InferenceServer import CompletionRequest, InferenceServerSettings, create_inference_app
from saddlellm.runtime.NativeStreamBatcher import NativeStreamBatcher
from test_native_streaming import Tokenizer, capture_sessions, engine, stream


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def settings(**kwargs):
    return InferenceServerSettings(**dict(
        backend="native", native_stream_batching=True, max_concurrency=4,
        batch_size=4, batch_token_budget=128, max_input_tokens=16,
        max_new_tokens=8, default_max_new_tokens=4, web_ui=False,
    ) | kwargs)


def coordinator(instance, **kwargs):
    return NativeStreamBatcher(instance.model, **dict(
        max_batch_size=4, max_resident=4, max_batch_tokens=128, wait_ms=5,
    ) | kwargs)


def new_session(instance, tokens=4, **kwargs):
    with torch.inference_mode():
        return instance.model.create_generation_session(
            input_ids=torch.tensor([[4, 5]]), max_new_tokens=tokens,
            **(dict(do_sample=False) | kwargs),
        )


@pytest.mark.parametrize("values", [
    {"native_stream_batching": 1}, {"backend": "auto"}, {"backend": "local"},
    {"batch_size": 1}, {"max_concurrency": 1}, {"quantization": "int8", "device": "cuda"},
    {"max_pending_tokens": 100}, {"batch_length_bucket_size": 16},
])
def test_explicit_stream_batch_settings(values):
    with pytest.raises(ValueError):
        settings(**values)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unsupported_injected_dtype_rejected(dtype):
    instance = engine()
    instance.model.to(dtype=dtype)
    with pytest.raises(ValueError, match="FP32 or CUDA BF16"):
        create_inference_app(settings(), model=instance.model, tokenizer=instance.tokenizer)


def test_lazy_close_is_idempotent_and_rejects_new_work():
    instance = engine()
    batcher = coordinator(instance)
    assert not batcher.snapshot()["worker_alive"]
    batcher.close()
    batcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        batcher.register(new_session(instance))


def test_single_request_bypasses_collection_and_worker_handoff(monkeypatch):
    instance = engine()
    batcher = coordinator(instance, wait_ms=1000)
    session = new_session(instance)
    original = batcher._runner.step
    owner = threading.get_ident()

    def same_thread(sessions):
        assert threading.get_ident() == owner
        return original(sessions)

    monkeypatch.setattr(batcher._runner, "step", same_thread)
    try:
        batcher.register(session)
        while not session.done:
            batcher.step(session, threading.Event())
        assert batcher.snapshot()["single_request_steps"] == 4
        assert batcher.snapshot()["inflight"] == 0
    finally:
        batcher.unregister(session)
        session.close()
        batcher.close()


@pytest.mark.parametrize("ahead", [1, 2])
def test_one_tick_ahead_request_joins_new_equal_length_arrivals(ahead):
    instance = engine()
    batcher = coordinator(instance, wait_ms=50)
    sessions = [new_session(instance, tokens=8) for _ in range(4)]
    reference = instance.model.generate(torch.tensor([[4, 5]]), max_new_tokens=8, do_sample=False)
    try:
        batcher.register(sessions[0])
        for _ in range(ahead):
            batcher.step(sessions[0], threading.Event())
        for session in sessions[1:]:
            batcher.register(session)

        def finish(session):
            while not session.done:
                batcher.step(session, threading.Event())

        with ThreadPoolExecutor(4) as pool:
            list(pool.map(finish, sessions))
        status = batcher.snapshot()
        assert status["alignment_deferrals"] > 0
        assert status["largest_batch"] == 4
        assert all(torch.equal(session.sequences(), reference) for session in sessions)
    finally:
        for session in sessions:
            batcher.unregister(session)
            session.close()
        batcher.close()


def test_idle_worker_releases_departed_session_references():
    instance = engine()
    batcher = coordinator(instance)
    sessions = [new_session(instance) for _ in range(2)]
    references = [weakref.ref(session) for session in sessions]
    try:
        for session in sessions:
            batcher.register(session)
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda item: batcher.step(item, threading.Event()), sessions))
        for session in sessions:
            batcher.unregister(session)
            session.close()
        del session
        sessions.clear()
        gc.collect()
        assert all(reference() is None for reference in references)
    finally:
        batcher.close()


def test_resident_limits_and_duplicate_registration():
    instance = engine()
    batcher = coordinator(instance, max_resident=1)
    first, second = new_session(instance), new_session(instance)
    try:
        batcher.register(first)
        with pytest.raises(ValueError, match="already"):
            batcher.register(first)
        with pytest.raises(RuntimeError, match="resident"):
            batcher.register(second)
        with pytest.raises(ValueError, match="not registered"):
            batcher.step(second, threading.Event())
    finally:
        batcher.unregister(first)
        first.close()
        second.close()
        batcher.close()


@pytest.mark.parametrize("endpoint", ["completions", "chat/completions"])
@pytest.mark.parametrize("cuda_bf16", [False, True])
@pytest.mark.parametrize("sampling", [False, True])
@pytest.mark.parametrize("decode_mode", ["compatible", "throughput"])
def test_http_actual_batch_text_usage_and_cleanup(monkeypatch, endpoint, cuda_bf16, sampling, decode_mode):
    if cuda_bf16 and not torch.cuda.is_available():
        pytest.skip("CUDA BF16 integration requires CUDA")
    instance = engine()
    if cuda_bf16:
        instance.model.to(device="cuda", dtype=torch.bfloat16)
    instance.tokenizer.apply_chat_template = lambda *a, **k: "prompt"
    reference = instance.complete(CompletionRequest(prompt="prompt", temperature=0, max_tokens=8))
    sessions = capture_sessions(monkeypatch, instance)
    app = create_inference_app(settings(native_decode_mode=decode_mode), model=instance.model, tokenizer=instance.tokenizer)
    batcher = app.state.native_stream_batcher
    barrier = threading.Barrier(4, timeout=10)
    register = batcher.register

    def aligned(session):
        register(session)
        barrier.wait()

    monkeypatch.setattr(batcher, "register", aligned)
    shapes = []
    hook = instance.model.register_forward_pre_hook(
        lambda module, args, kwargs: shapes.append(tuple(kwargs["input_ids"].shape)), with_kwargs=True,
    )

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                body = dict(stream=True, max_tokens=8, stream_options={"include_usage": True})
                if not sampling:
                    body["temperature"] = 0
                body.update({"prompt": "prompt"} if endpoint == "completions" else
                            {"messages": [{"role": "user", "content": "prompt"}]})
                responses = await asyncio.wait_for(asyncio.gather(*(
                    client.post(f"/v1/{endpoint}", json=body) for _ in range(4)
                )), 20)
                for response in responses:
                    assert response.status_code == 200
                    assert response.text.endswith("data: [DONE]\n\n")
                    parts = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
                    assert not any("error" in part for part in parts)
                    text = "".join(choice.get("text", choice.get("delta", {}).get("content", ""))
                                   for part in parts for choice in part.get("choices", []))
                    if not sampling and decode_mode == "compatible":
                        assert text == reference["choices"][0]["text"]
                    assert [part["usage"] for part in parts if part.get("usage")] == [reference["usage"]]
                    assert sum(choice["finish_reason"] == "length" for part in parts for choice in part.get("choices", [])) == 1
                health = (await client.get("/health")).json()
                assert health["stream_scheduling"] == "native_batch_decode"
                assert health["admission"]["active"] == 0
                status = health["native_stream_batching"]
                assert status["batched_forwards"] > 0 and status["largest_batch"] == 4
                assert status["decode_mode"] == decode_mode
                assert bool(status["sampling_batched_tokens"]) == sampling
                assert sum(int(size) * count for size, count in status["batch_size_histogram"].items()) == status["batched_tokens"]
                assert sum(status["individual_reasons"].values()) == status["individual_steps"]
                assert status["registered"] == status["pending"] == status["inflight"] == status["retained_cohorts"] == 0

    try:
        asyncio.run(run())
    finally:
        hook.remove()
        batcher.close()
    assert (4, 1) in shapes
    assert all(session.closed and session.past_key_values is None for session in sessions)
    assert not batcher.snapshot()["worker_alive"]


def test_nonstream_sampling_uses_same_batcher_and_cleans_up(monkeypatch):
    instance = engine()
    app = create_inference_app(settings(), model=instance.model, tokenizer=instance.tokenizer)
    batcher = app.state.native_stream_batcher
    register = batcher.register
    barrier = threading.Barrier(4, timeout=10)

    def aligned(session):
        register(session)
        barrier.wait()

    monkeypatch.setattr(batcher, "register", aligned)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                results = await asyncio.wait_for(asyncio.gather(*(
                    client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 4}) for _ in range(4)
                )), 20)
                assert all(result.status_code == 200 and result.json()["usage"]["completion_tokens"] == 4 for result in results)
                status = batcher.snapshot()
                assert status["sampling_batched_tokens"] > 0
                assert status["registered"] == status["inflight"] == status["pending"] == status["retained_cohorts"] == 0
    asyncio.run(run())


@pytest.mark.parametrize("cuda_bf16", [False, True])
@pytest.mark.parametrize("decode_mode", ["compatible", "throughput"])
@pytest.mark.parametrize("streaming", [False, True])
def test_http_mixed_lengths_share_projections_and_keep_private_usage(monkeypatch, cuda_bf16, decode_mode, streaming):
    if cuda_bf16 and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    class MixedTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            ids = torch.tensor([[4 + index % 5 for index in range(int(text))]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    instance = engine(MixedTokenizer())
    if cuda_bf16:
        instance.model.cuda().bfloat16()
    created = capture_sessions(monkeypatch, instance)
    app = create_inference_app(settings(native_decode_mode=decode_mode), model=instance.model, tokenizer=instance.tokenizer)
    batcher = app.state.native_stream_batcher
    register = batcher.register
    barrier = threading.Barrier(3, timeout=10)
    def aligned(session):
        register(session)
        barrier.wait()
    monkeypatch.setattr(batcher, "register", aligned)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                async def request(length):
                    body = {"prompt": str(length), "max_tokens": 8, "stream": streaming}
                    if streaming:
                        body["stream_options"] = {"include_usage": True}
                    response = await client.post("/v1/completions", json=body)
                    assert response.status_code == 200
                    if streaming:
                        parts = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
                        assert not any("error" in part for part in parts)
                        usage = [part["usage"] for part in parts if part.get("usage")][-1]
                        assert response.text.endswith("data: [DONE]\n\n")
                    else:
                        usage = response.json()["usage"]
                    assert usage == {"prompt_tokens": length, "completion_tokens": 8, "total_tokens": length + 8}
                await asyncio.wait_for(asyncio.gather(*(request(n) for n in (2,5,9))), 20)
                status = batcher.snapshot()
                assert status["independent_kv_forwards"] > 0 and status["sampling_batched_tokens"] > 0
                assert status["largest_batch"] == 3
                assert all(status[k] == 0 for k in ("registered","pending","inflight","retained_cohorts"))
    asyncio.run(run())
    assert all(s.closed and s.past_key_values is None for s in created)


@pytest.mark.parametrize("sampling", [False, True])
def test_running_and_queued_cancellation_wait_for_actual_work(monkeypatch, sampling):
    instance = engine()
    batcher = coordinator(instance, max_batch_size=1, wait_ms=0)
    first = new_session(instance, do_sample=sampling, seed=14)
    second = new_session(instance, do_sample=sampling, seed=15)
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    original = batcher._runner.step

    def blocked(sessions):
        entered.set()
        assert release.wait(10)
        return original(sessions)

    monkeypatch.setattr(batcher._runner, "step", blocked)
    batcher.register(first)
    batcher.register(second)
    try:
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(batcher.step, first, cancelled)
            assert entered.wait(5)
            queued = pool.submit(batcher.step, second, cancelled)
            cancelled.set()
            assert not active.done()
            with pytest.raises(RuntimeError, match="outstanding"):
                batcher.step(first, cancelled)
            with pytest.raises(RuntimeError, match="actual step"):
                batcher.unregister(first)
            release.set()
            assert active.result(10) is not None
            assert queued.result(10) is None
        assert first.generated_tokens == 1 and second.generated_tokens == 0
    finally:
        release.set()
        batcher.unregister(first)
        batcher.unregister(second)
        first.close()
        second.close()
        batcher.close()


def test_shutdown_drains_inflight_and_fails_pending(monkeypatch):
    instance = engine()
    batcher = coordinator(instance, max_batch_size=1, wait_ms=0)
    sessions = [new_session(instance) for _ in range(2)]
    entered, release = threading.Event(), threading.Event()
    original = batcher._runner.step

    def blocked(sessions):
        entered.set()
        assert release.wait(10)
        return original(sessions)

    monkeypatch.setattr(batcher._runner, "step", blocked)
    for session in sessions:
        batcher.register(session)
    with ThreadPoolExecutor(3) as pool:
        first = pool.submit(batcher.step, sessions[0], threading.Event())
        try:
            assert entered.wait(5)
            second = pool.submit(batcher.step, sessions[1], threading.Event())
            closing = pool.submit(batcher.close)
            assert not closing.done()
        finally:
            release.set()
        assert first.result(10) is not None
        with pytest.raises(RuntimeError, match="closed"):
            second.result(10)
        closing.result(10)
    for session in sessions:
        batcher.unregister(session)
        session.close()
    assert not batcher.snapshot()["worker_alive"]


def test_failure_notifies_all_and_new_request_can_recover(monkeypatch):
    instance = engine()
    batcher = coordinator(instance)
    original = batcher._runner.step
    monkeypatch.setattr(batcher._runner, "step", lambda sessions: (_ for _ in ()).throw(RuntimeError("planned")))
    session = new_session(instance)
    try:
        batcher.register(session)
        with pytest.raises(RuntimeError, match="planned"):
            batcher.step(session, threading.Event())
        batcher.unregister(session)
        session.close()
        monkeypatch.setattr(batcher._runner, "step", original)
        session = new_session(instance)
        batcher.register(session)
        assert batcher.step(session, threading.Event()) is not None
    finally:
        batcher.unregister(session)
        session.close()
        batcher.close()


def test_slow_consumer_does_not_get_unrequested_tokens(monkeypatch):
    instance = engine()
    created = capture_sessions(monkeypatch, instance)
    batcher = coordinator(instance)
    slow_state = {"native_step_batcher": batcher}
    slow = stream(instance, slow_state)
    try:
        assert next(slow)
        count = created[0].generated_tokens
        state = {"native_step_batcher": batcher}
        assert "".join(stream(instance, state))
        assert created[0].generated_tokens == count
        assert not created[0].closed
        assert created[1].closed
        assert batcher.snapshot()["registered"] == 1
    finally:
        slow.close()
        batcher.close()
    assert created[0].closed
    assert slow_state["worker_done"].is_set()


def test_stream_stop_and_usage_match_single_request():
    instance = engine()
    text = "".join(stream(instance))
    stop = text.split()[1]
    reference_state = {}
    reference = "".join(stream(instance, reference_state, stop=stop))
    batcher = coordinator(instance)
    state = {"native_step_batcher": batcher}
    try:
        assert "".join(stream(instance, state, stop=stop)) == reference
        assert state["completion_tokens"] == reference_state["completion_tokens"]
        assert state["finish_reason"] == "stop"
    finally:
        batcher.close()


def test_departure_housekeeping_cannot_race_the_next_forward(monkeypatch):
    instance = engine()
    batcher = coordinator(instance, wait_ms=0)
    first, second = new_session(instance), new_session(instance)
    batcher.register(first)
    batcher.register(second)
    entered, release = threading.Event(), threading.Event()

    def delayed_forget(session):
        with batcher._runner._lock:
            entered.set()
            assert release.wait(10)

    monkeypatch.setattr(batcher._runner, "forget", delayed_forget)
    try:
        with ThreadPoolExecutor(2) as pool:
            departing = pool.submit(batcher.unregister, first)
            assert entered.wait(5)
            stepping = pool.submit(batcher.step, second, threading.Event())
            # A step queued during cleanup must wait, not manufacture a model
            # failure from the runner's nonblocking ownership check.
            assert not stepping.done()
            release.set()
            departing.result(10)
            assert stepping.result(10) is not None
    finally:
        release.set()
        batcher.unregister(second)
        first.close()
        second.close()
        batcher.close()
