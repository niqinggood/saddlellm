"""Real ASGI disconnects, not just cancellation of a test HTTP coroutine."""
import asyncio
import json
import socket
import threading
from concurrent.futures import CancelledError as GenerationCancelled

import httpx
import pytest
import torch

from saddlellm.runtime.InferenceServer import (
    ChatCompletionRequest, CompletionRequest, InferenceServerSettings,
    LocalGenerationEngine, create_inference_app,
)
from test_inference_batching import make_engine


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def disconnected_request(app, endpoint="completions", *, prompt="4 5"):
    disconnected, consumed = asyncio.Event(), asyncio.Event()
    body = {"temperature": 0, "max_tokens": 5}
    body.update({"prompt": prompt} if endpoint == "completions" else
                {"messages": [{"role": "user", "content": prompt}]})
    messages = []

    async def receive():
        if not consumed.is_set():
            consumed.set()
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    path = "/v1/" + endpoint
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
             "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 1234), "server": ("test", 80)}
    return asyncio.create_task(app(scope, receive, send)), disconnected, consumed, messages


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), 2)


@pytest.mark.parametrize("endpoint", ["completions", "chat/completions"])
@pytest.mark.parametrize("batched", [False, True])
def test_disconnected_queued_request_never_reaches_model(monkeypatch, endpoint, batched):
    async def run():
        engine = make_engine(web_ui=False, batch_wait_ms=0)
        engine.settings.batch_size = 2 if batched else 1
        app = create_inference_app(engine.settings, model=engine.model, tokenizer=engine.tokenizer)
        entered, release = threading.Event(), threading.Event()
        calls = []
        original = engine.model.generate

        def blocked(**kwargs):
            calls.append(1)
            entered.set()
            assert release.wait(5)
            return original(**kwargs)

        monkeypatch.setattr(engine.model, "generate", blocked)
        first = second = None
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                try:
                    first = asyncio.create_task(client.post("/v1/completions", json={"prompt": "6 7", "temperature": 0}))
                    assert await asyncio.to_thread(entered.wait, 2)
                    second, disconnect, consumed, _ = disconnected_request(app, endpoint)
                    await consumed.wait()
                    await eventually(lambda: app.state.admission.active == 2)
                    if batched:
                        await eventually(lambda: len(app.state.batcher.pending) == 1)
                    disconnect.set()
                    await eventually(second.done)
                    await second
                    assert app.state.admission.active == 1
                    if batched:
                        assert not app.state.batcher.pending
                    assert len(calls) == 1
                    release.set()
                    assert (await first).status_code == 200
                    assert app.state.admission.active == 0
                    assert len(calls) == 1
                finally:
                    release.set()
                    await asyncio.gather(*(task for task in (first, second) if task), return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize("backend", ["native", "hf"])
@pytest.mark.parametrize("chat", [False, True])
def test_cancellation_hook_preserves_output_and_stops_real_model(monkeypatch, backend, chat):
    engine = make_engine(web_ui=False)
    engine.model.config.eos_token_id = []
    if backend == "hf":
        from transformers import GPT2Config, GPT2LMHeadModel
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_embd=16, n_inner=32, n_layer=1, n_head=4,
            n_positions=64, eos_token_id=None, pad_token_id=0,
        )).eval()
        engine = LocalGenerationEngine(model, engine.tokenizer, InferenceServerSettings(web_ui=False))
    request = (ChatCompletionRequest(messages=[{"role": "user", "content": "4 5"}], max_tokens=5, temperature=0)
               if chat else CompletionRequest(prompt="4 5", max_tokens=5, temperature=0))
    method = engine.chat if chat else engine.complete
    reference = method(request)
    cancel = threading.Event()
    actual = method(request, cancel_event=cancel)
    assert actual["choices"] == reference["choices"]
    assert actual["usage"] == reference["usage"]
    assert actual["usage"]["completion_tokens"] == 5
    original = engine.model.forward
    calls = []

    def stop_after_forward(*args, **kwargs):
        calls.append(1)
        result = original(*args, **kwargs)
        cancel.set()
        return result

    monkeypatch.setattr(engine.model, "forward", stop_after_forward)
    stopped = method(request, cancel_event=cancel)
    assert stopped["usage"]["completion_tokens"] == 1
    assert len(calls) == 1
    with pytest.raises(GenerationCancelled):
        method(request, cancel_event=cancel)
    assert len(calls) == 1


def test_repeated_http_task_cancellation_keeps_live_worker_owned(monkeypatch):
    async def run():
        engine = make_engine(web_ui=False)
        engine.settings.batch_size = 1
        app = create_inference_app(engine.settings, model=engine.model, tokenizer=engine.tokenizer)
        entered, release = threading.Event(), threading.Event()
        calls = []
        original = engine.model.generate

        def blocked(**kwargs):
            calls.append(1)
            entered.set()
            assert release.wait(5)
            return original(**kwargs)

        monkeypatch.setattr(engine.model, "generate", blocked)
        first = second = None
        async with app.router.lifespan_context(app):
            try:
                first, _, _, _ = disconnected_request(app)
                assert await asyncio.to_thread(entered.wait, 2)
                first.cancel()
                await asyncio.sleep(0.02)
                first.cancel()
                second, disconnected, _, _ = disconnected_request(app)
                await eventually(lambda: app.state.admission.active == 2)
                assert len(calls) == 1 and not first.done()
                disconnected.set()
                await asyncio.wait_for(second, 2)
                assert app.state.admission.active == 1
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(first, 2)
                assert app.state.admission.active == 0
                assert len(calls) == 1
            finally:
                release.set()
                await asyncio.gather(*(task for task in (first, second) if task), return_exceptions=True)
    asyncio.run(run())


def test_active_batch_disconnect_does_not_cancel_other_rows(monkeypatch):
    async def run():
        engine = make_engine(web_ui=False, batch_wait_ms=1000)
        engine.settings.batch_size = 2
        app = create_inference_app(engine.settings, model=engine.model, tokenizer=engine.tokenizer)
        entered, release = threading.Event(), threading.Event()
        captured = []
        original = app.state.batcher.generate

        def blocked(rows, events):
            captured.extend(events)
            entered.set()
            assert release.wait(5)
            return original(rows, events)

        monkeypatch.setattr(app.state.batcher, "generate", blocked)
        first = second = None
        async with app.router.lifespan_context(app):
            try:
                first, disconnect, _, _ = disconnected_request(app)
                await eventually(lambda: len(app.state.batcher.pending) == 1)
                second, _, _, messages = disconnected_request(app, prompt="6 7")
                assert await asyncio.to_thread(entered.wait, 2)
                disconnect.set()
                await asyncio.wait_for(first, 2)
                assert captured[0].is_set() and not captured[1].is_set()
                assert len(app.state.batcher.active) == 2
                release.set()
                await asyncio.wait_for(second, 2)
                assert messages[0]["status"] == 200
                response = json.loads(b"".join(m.get("body", b"") for m in messages))
                reference = engine.complete(CompletionRequest(prompt="6 7", temperature=0, max_tokens=5))
                assert response["choices"] == reference["choices"]
                assert response["usage"] == reference["usage"]
                assert app.state.admission.active == 0
                assert not app.state.batcher.active and not app.state.batcher.pending
            finally:
                release.set()
                await asyncio.gather(*(task for task in (first, second) if task), return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize("endpoint", ["completions", "chat/completions"])
def test_active_disconnect_signals_stop_but_keeps_permit_until_worker_exits(monkeypatch, endpoint):
    async def run():
        engine = make_engine(web_ui=False)
        engine.settings.batch_size = 1
        app = create_inference_app(engine.settings, model=engine.model, tokenizer=engine.tokenizer)
        entered, cancelled, release = (threading.Event() for _ in range(3))
        calls = []
        original = engine.model.generate

        def blocked(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                control = kwargs.get("stopping_criteria")
                while not release.wait(0.005):
                    if control is not None and bool(control(kwargs["input_ids"], None).all()):
                        cancelled.set()
                assert cancelled.is_set()
            return original(**kwargs)

        monkeypatch.setattr(engine.model, "generate", blocked)
        request = second = None
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                try:
                    request, disconnect, _, _ = disconnected_request(app, endpoint)
                    assert await asyncio.to_thread(entered.wait, 2)
                    disconnect.set()
                    assert await asyncio.to_thread(cancelled.wait, 2)
                    second = asyncio.create_task(client.post("/v1/completions", json={"prompt": "6 7", "temperature": 0}))
                    await eventually(lambda: app.state.admission.active == 2)
                    assert len(calls) == 1 and not request.done()
                    release.set()
                    await asyncio.wait_for(request, 2)
                    assert (await asyncio.wait_for(second, 2)).status_code == 200
                    assert app.state.admission.active == 0
                finally:
                    release.set()
                    await asyncio.gather(*(task for task in (request, second) if task), return_exceptions=True)
    asyncio.run(run())


def test_real_tcp_disconnect_reclaims_work_and_next_request_succeeds(monkeypatch):
    from urllib.parse import urlsplit
    from tools.benchmark_native_serving import loopback_server

    engine = make_engine(web_ui=False)
    engine.settings.batch_size = 1
    app = create_inference_app(engine.settings, model=engine.model, tokenizer=engine.tokenizer)
    entered, cancelled, release = (threading.Event() for _ in range(3))
    original = engine.model.generate
    calls = []

    def blocked(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            while not release.wait(0.005):
                if bool(kwargs["stopping_criteria"](kwargs["input_ids"], None).all()):
                    cancelled.set()
        return original(**kwargs)

    monkeypatch.setattr(engine.model, "generate", blocked)
    with loopback_server(app) as base:
        url = urlsplit(base)
        try:
            with socket.create_connection((url.hostname, url.port), timeout=2) as connection:
                body = json.dumps({"prompt": "4 5", "temperature": 0, "max_tokens": 5}).encode()
                headers = (f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                           f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode()
                connection.sendall(headers + body)
                assert entered.wait(2)
                connection.shutdown(socket.SHUT_RDWR)
            assert cancelled.wait(2)
            release.set()
            with httpx.Client(base_url=base, trust_env=False, timeout=5) as client:
                response = client.post("/completions", json={"prompt": "6 7", "temperature": 0, "max_tokens": 2})
                assert response.status_code == 200
                assert response.json()["usage"]["completion_tokens"] == 2
                health = client.get(base.removesuffix("/v1") + "/health").json()
                assert health["admission"]["active"] == 0
        finally:
            release.set()


def test_disconnect_watcher_does_not_leak_after_repeated_success_or_error():
    from saddlellm.runtime.InferenceLifecycle import run_until_disconnect

    async def run():
        before = asyncio.all_tasks()

        async def receive():
            await asyncio.Event().wait()

        async def generate(fail):
            await asyncio.sleep(0)
            if fail:
                raise ValueError("controlled test failure")
            return {"ok": True}

        for index in range(50):
            event = threading.Event()
            if index % 2:
                with pytest.raises(ValueError, match="controlled test failure"):
                    await run_until_disconnect(receive, generate(True), event)
            else:
                assert await run_until_disconnect(receive, generate(False), event) == {"ok": True}
                assert not event.is_set()
        await asyncio.sleep(0)
        assert asyncio.all_tasks() == before
    asyncio.run(run())
