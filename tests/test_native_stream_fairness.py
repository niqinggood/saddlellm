import asyncio
import json
import time
import threading

import httpx
import pytest

from saddlellm.runtime.InferenceServer import InferenceServerSettings, create_inference_app
from test_native_streaming import Tokenizer, capture_sessions, engine, stream


@pytest.fixture(autouse=True)
def experimental_cooperative_policy(monkeypatch):
    monkeypatch.setattr("saddlellm.runtime.InferenceServer._cooperative_native_streaming", lambda engine, settings: True)


def test_cooperative_ticks_yield_even_without_visible_words(monkeypatch):
    class BufferedTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return "x" * len(ids)

    instance = engine(BufferedTokenizer())
    created = capture_sessions(monkeypatch, instance)
    state = {"cooperative_ticks": True}
    iterator = stream(instance, state)
    pieces = []
    previous = 0
    for piece in iterator:
        current = created[0].generated_tokens
        assert 0 <= current - previous <= 1
        previous = current
        pieces.append(piece)
    # The final token reaches the length limit and flushes visible text directly.
    assert pieces == ["", "", "", "xxxx"]
    assert state["completion_tokens"] == 4
    assert created[0].closed


@pytest.mark.parametrize("endpoint", ["completions", "chat/completions"])
@pytest.mark.parametrize("fail_first", [False, True])
def test_native_streams_interleave_without_overlapping_model_steps(monkeypatch, endpoint, fail_first):
    instance = engine()
    instance.tokenizer.apply_chat_template = lambda *a, **k: "prompt"
    settings = InferenceServerSettings(max_concurrency=2, max_new_tokens=8, default_max_new_tokens=8, web_ui=False)
    app = create_inference_app(settings, model=instance.model, tokenizer=instance.tokenizer)
    create = instance.model.create_generation_session
    sessions, order, worker_threads = [], [], set()
    active = peak = 0

    def create_session(*args, **kwargs):
        session = create(*args, **kwargs)
        number = len(sessions)
        sessions.append(session)
        step = session.step

        def counted():
            nonlocal active, peak
            worker_threads.add(threading.get_ident())
            active += 1
            peak = max(peak, active)
            try:
                order.append(number)
                time.sleep(0.003)
                if fail_first and number == 0 and session.generated_tokens == 2:
                    raise RuntimeError("planned failure")
                return step()
            finally:
                active -= 1

        session.step = counted
        return session

    monkeypatch.setattr(instance.model, "create_generation_session", create_session)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            body = {"stream": True, "temperature": 0, "max_tokens": 8, "stream_options": {"include_usage": True}}
            body.update({"prompt": "hello"} if endpoint == "completions" else {"messages": [{"role": "user", "content": "hello"}]})
            results = await asyncio.gather(*(client.post(f"/v1/{endpoint}", json=body) for _ in range(2)))
            health = (await client.get("/health")).json()
        assert health["stream_scheduling"] == "cooperative_token"
        assert health["admission"]["active"] == 0
        for result in results:
            assert result.status_code == 200
            assert result.text.endswith("data: [DONE]\n\n")
        parsed = [[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: {")] for r in results]
        assert sum(any("error" in part for part in row) for row in parsed) == int(fail_first)
        successful = [row for row in parsed if not any("error" in part for part in row)]
        assert all(next(part["usage"] for part in row if part.get("usage"))["completion_tokens"] == 8 for row in successful)
        if not fail_first:
            texts = ["".join(choice.get("text", choice.get("delta", {}).get("content", "")) for part in row for choice in part.get("choices", [])) for row in successful]
            assert texts[0] == texts[1]

    asyncio.run(run())
    assert peak == 1
    assert len(worker_threads) == 1
    assert order.index(1) < (3 if fail_first else 8)
    assert all(session.closed and session.buffers is None for session in sessions)


def test_visible_text_is_itself_a_scheduling_boundary():
    instance = engine()
    state = {"cooperative_ticks": True}
    pieces = list(stream(instance, state))
    # The first word may be buffered, but do not add another empty turn after
    # every visible text chunk. Four token steps plus at most one flush.
    assert len(pieces) <= state["completion_tokens"] + 1


@pytest.mark.parametrize("batch_decode", [False, True])
def test_cooperative_disconnect_keeps_turn_until_inflight_forward_exits(monkeypatch, batch_decode):
    instance = engine()
    settings = InferenceServerSettings(max_concurrency=2, max_new_tokens=8, default_max_new_tokens=4, web_ui=False)
    if batch_decode:
        settings = InferenceServerSettings(backend="native", native_stream_batching=True,
            batch_size=2, max_concurrency=2, max_new_tokens=8, default_max_new_tokens=4, web_ui=False)
    app = create_inference_app(settings, model=instance.model, tokenizer=instance.tokenizer)
    original = instance.model.forward
    entered, release, next_entered = (threading.Event() for _ in range(3))
    count = 0

    def blocked(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert release.wait(5)
        else:
            next_entered.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(instance.model, "forward", blocked)

    async def run():
        disconnected = asyncio.Event()
        body = json.dumps({"prompt": "hello", "stream": True, "temperature": 0, "max_tokens": 4}).encode()
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/v1/completions", "raw_path": b"/v1/completions", "root_path": "",
                 "query_string": b"", "headers": [(b"content-type", b"application/json")],
                 "client": ("127.0.0.1", 1234), "server": ("test", 80)}
        pending = asyncio.create_task(app(scope, receive, send))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            disconnected.set()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                second = asyncio.create_task(client.post("/v1/completions", json=json.loads(body)))
                await asyncio.sleep(0.04)
                assert not next_entered.is_set()
                assert not pending.done()
                release.set()
                await asyncio.wait_for(pending, 5)
                assert (await asyncio.wait_for(second, 5)).status_code == 200
                assert (await client.get("/health")).json()["admission"]["active"] == 0
        finally:
            release.set()
            await asyncio.wait_for(pending, 5)

    try:
        asyncio.run(run())
    finally:
        if app.state.native_stream_batcher is not None:
            app.state.native_stream_batcher.close()


@pytest.mark.parametrize("batch_decode", [False, True])
def test_slow_network_client_does_not_hold_model_turn(monkeypatch, batch_decode):
    instance = engine()
    settings = InferenceServerSettings(max_concurrency=2, max_new_tokens=8, default_max_new_tokens=4, web_ui=False)
    if batch_decode:
        settings = InferenceServerSettings(backend="native", native_stream_batching=True,
            batch_size=2, max_concurrency=2, max_new_tokens=8, default_max_new_tokens=4, web_ui=False)
    app = create_inference_app(settings, model=instance.model, tokenizer=instance.tokenizer)

    async def run():
        blocked_send, release_send = asyncio.Event(), asyncio.Event()
        request_sent = False
        body = json.dumps({"prompt": "hello", "stream": True, "temperature": 0, "max_tokens": 4}).encode()

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body") and not blocked_send.is_set():
                blocked_send.set()
                await release_send.wait()

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/v1/completions", "raw_path": b"/v1/completions", "root_path": "",
                 "query_string": b"", "headers": [(b"content-type", b"application/json")],
                 "client": ("127.0.0.1", 1234), "server": ("test", 80)}
        slow = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(blocked_send.wait(), 5)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                second = await asyncio.wait_for(client.post("/v1/completions", json=json.loads(body)), 5)
                assert second.status_code == 200
                assert not slow.done()
        finally:
            release_send.set()
            await asyncio.wait_for(slow, 5)

    try:
        asyncio.run(run())
    finally:
        if app.state.native_stream_batcher is not None:
            app.state.native_stream_batcher.close()
