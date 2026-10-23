import importlib
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from test_native_streaming import engine


def test_native_stream_pool_is_bounded_and_closed_with_app(monkeypatch):
    runtime = importlib.import_module("saddlellm.runtime.InferenceServer")
    created = []

    class CapturedExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(runtime, "ThreadPoolExecutor", CapturedExecutor)
    monkeypatch.setattr(runtime, "_native_stream_worker_count", lambda engine, settings: settings.max_concurrency)
    instance = engine()
    marker = ContextVar("native_stream_test_marker", default=None)
    observed = []
    forward = instance.model.forward

    def checked_forward(*args, **kwargs):
        observed.append(marker.get())
        return forward(*args, **kwargs)

    monkeypatch.setattr(instance.model, "forward", checked_forward)
    app = runtime.create_inference_app(
        runtime.InferenceServerSettings(max_concurrency=2, max_new_tokens=8, default_max_new_tokens=4, web_ui=False),
        model=instance.model, tokenizer=instance.tokenizer,
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["stream_scheduling"] == "bounded_workers"
        assert health["stream_worker_limit"] == 2
        token = marker.set("request-context")
        try:
            result = client.post("/v1/completions", json={"prompt": "hello", "stream": True, "max_tokens": 4, "temperature": 0})
        finally:
            marker.reset(token)
        assert result.status_code == 200 and "[DONE]" in result.text
        assert observed and set(observed) == {"request-context"}
        assert created[0]._max_workers == 2
        assert 0 < len(created[0]._threads) <= 2
    assert all(not thread.is_alive() for thread in created[0]._threads)


def test_native_defaults_keep_validated_request_permit_policy():
    runtime = importlib.import_module("saddlellm.runtime.InferenceServer")
    instance = engine()
    app = runtime.create_inference_app(
        runtime.InferenceServerSettings(max_concurrency=4, web_ui=False),
        model=instance.model, tokenizer=instance.tokenizer,
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["stream_scheduling"] == "request_permit"
        assert health["stream_worker_limit"] is None
        assert health["stream_resident_limit"] == 4


def test_custom_generation_has_no_native_stream_executor():
    runtime = importlib.import_module("saddlellm.runtime.InferenceServer")
    instance = engine()
    instance.model.generate = lambda **kwargs: None
    app = runtime.create_inference_app(
        runtime.InferenceServerSettings(max_concurrency=2, web_ui=False),
        model=instance.model, tokenizer=instance.tokenizer,
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["stream_scheduling"] == "request_permit"
        assert health["stream_worker_limit"] is None
