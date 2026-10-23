"""CPU-only protocol/lifecycle tests. No real vLLM/GPU throughput is asserted."""

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from saddlellm.runtime.InferenceServer import (
    InferenceServerSettings,
    create_inference_app,
)
from saddlellm.runtime.VLLMConfig import VLLMServingConfig
from saddlellm.runtime.VLLMServing import (
    Admission,
    OwnedStreamingResponse,
    VLLMRuntime,
    configure_vllm_multiprocessing,
    create_vllm_inference_app,
)


class Tokenizer:
    def encode(self, text, add_special_tokens):
        return list(range(len(text)))

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.template_kwargs = messages, kwargs
        return "chat"


class Engine:
    def __init__(self):
        self.model_config = SimpleNamespace(
            max_model_len=40, dtype="bfloat16", quantization=None
        )
        self.aborted, self.requests = [], []
        self.closed = self.fail = self.unhealthy = self.incomplete = False
        self.entered = asyncio.Event()
        self.release = None
        self.abort_release = None
        self.abort_entered = asyncio.Event()

    async def check_health(self):
        if self.unhealthy:
            raise RuntimeError("PRIVATE engine health failure")

    def shutdown(self, timeout):
        self.closed = True

    async def abort(self, request_id):
        self.abort_entered.set()
        if self.abort_release is not None:
            await self.abort_release.wait()
        self.aborted.append(request_id)

    async def generate(self, prompt, params, request_id):
        self.requests.append((prompt, params, request_id))
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.fail:
            raise RuntimeError("PRIVATE model exception and prompt")
        yield SimpleNamespace(
            outputs=[SimpleNamespace(text="你", token_ids=[3], finish_reason=None)],
            finished=False,
        )
        if not self.incomplete:
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="你好", token_ids=[3, 4, 5], finish_reason="stop"
                    )
                ],
                finished=True,
            )


def setup_app(**options):
    config = dict(
        backend="vllm",
        model_path="MODEL",
        model_name="alias",
        web_ui=False,
        enable_thinking=False,
        max_input_tokens=20,
        default_max_new_tokens=4,
        max_new_tokens=8,
        vllm=VLLMServingConfig(max_num_seqs=2),
    )
    config.update(options)
    settings = InferenceServerSettings(**config)
    engine, tokenizer = Engine(), Tokenizer()
    runtime = VLLMRuntime(
        settings,
        engine=engine,
        tokenizer=tokenizer,
        sampling_factory=lambda **kwargs: kwargs,
    )
    app = create_vllm_inference_app(settings, runtime=runtime)
    return app, engine, tokenizer


def client_for(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.mark.parametrize(
    "config",
    [
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": True},
        {"gpu_memory_utilization": float("nan")},
        {"gpu_memory_utilization": 1.1},
        {"tensor_parallel_size": 0},
        {"max_num_seqs": False},
        {"max_model_len": 3.5},
        {"max_num_batched_tokens": -1},
        {"enable_prefix_caching": 1},
        {"enforce_eager": "yes"},
        {"max_num_seqs": 4, "max_num_batched_tokens": 2},
    ],
)
def test_invalid_engine_config(config):
    with pytest.raises(ValueError):
        VLLMServingConfig(**config)


@pytest.mark.parametrize(
    "config",
    [
        {"device": "cpu"},
        {"device": "mps"},
        {"quantization": "4bit"},
        {"batch_size": 4},
        {"max_concurrency": 4},
        {"batch_token_budget": 30},
        {"batch_length_bucket_size": 8},
        {"prefix_cache_mb": 1},
    ],
)
def test_native_knobs_cannot_silently_disable_vllm_scheduling(config):
    with pytest.raises(ValueError):
        InferenceServerSettings(backend="vllm", **config)


def test_kwargs_and_lazy_export_do_not_import_vllm_or_torch():
    command = "from saddlellm import VLLMServingConfig; import sys; assert 'vllm' not in sys.modules; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True, timeout=20)
    settings = InferenceServerSettings(
        backend="vllm",
        vllm={"gpu_memory_utilization": 0.47, "max_num_batched_tokens": 4096},
    )
    kwargs = settings.vllm.engine_kwargs(settings, "/model")
    assert kwargs["enable_log_requests"] is False
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["cpu_offload_gb"] == 0
    assert kwargs["generation_config"] == "vllm"
    assert kwargs["max_num_seqs"] == 16  # Not the local max_concurrency=1.
    assert kwargs["gpu_memory_utilization"] == 0.47
    assert "api_key" not in kwargs and "local_files_only" not in kwargs


def test_public_factory_dispatches_without_loading_hf(monkeypatch):
    from saddlellm.runtime import VLLMServing

    marker = object()
    monkeypatch.setattr(
        VLLMServing, "create_vllm_inference_app", lambda settings: marker
    )
    assert (
        create_inference_app(
            InferenceServerSettings(backend="vllm", model_path="MODEL")
        )
        is marker
    )
    with pytest.raises(ValueError, match="owns"):
        create_inference_app(
            InferenceServerSettings(backend="vllm"), model=object(), tokenizer=object()
        )


def test_cli_knobs_are_forwarded_and_explicit_backend_required(monkeypatch):
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
    assert (
        cli.main(
            [
                "serve",
                "MODEL",
                "--backend",
                "vllm",
                "--gpu-memory-utilization",
                ".47",
                "--max-num-seqs",
                "8",
                "--max-model-len",
                "4096",
                "--vllm-prefix-caching",
                "--max-pending-requests",
                "12",
            ]
        )
        == 0
    )
    assert captured[0].vllm.max_num_seqs == 8 and captured[0].max_pending_requests == 12
    assert captured[0].vllm.enable_prefix_caching
    assert captured[0].vllm.max_model_len == 4096
    assert cli.main(["serve", "MODEL", "--max-num-seqs", "8"]) != 0


def test_health_readiness_auth_and_lifecycle():
    async def run():
        app, engine, _ = setup_app(api_key="private-key", web_ui=True)
        async with client_for(app) as client:
            assert (await client.get("/livez")).status_code == 200
            assert (await client.get("/readyz")).status_code == 503
            assert (
                await client.post("/v1/completions", json={"prompt": "abc"})
            ).status_code == 401
            async with app.router.lifespan_context(app):
                assert (await client.get("/readyz")).status_code == 200
                health = (await client.get("/health")).json()
                assert health["batching"]["mode"] == "vllm_continuous"
                assert health["admission"]["capacity"] == 66
                assert health["streaming_backend"] == "vllm_async"
                assert "private-key" not in json.dumps(health)
                assert (await client.get("/")).status_code == 200
                assert (await client.get("/v1/models")).status_code == 401
                assert (
                    await client.get(
                        "/v1/models", headers={"Authorization": "Bearer private-key"}
                    )
                ).json()["data"][0]["id"] == "alias"
                engine.unhealthy = True
                assert (await client.get("/readyz")).status_code == 503
                assert (await client.get("/health")).json()["ready"] is False
            assert engine.closed
            assert (await client.get("/livez")).status_code == 200

    asyncio.run(run())


def test_vllm_webui_assets_and_history_share_the_common_routes(tmp_path, monkeypatch):
    import uuid

    monkeypatch.chdir(tmp_path)

    async def run():
        app, _, _ = setup_app(web_ui=True, api_key="private-key")
        async with client_for(app) as client:
            assert (await client.get("/assets/chat.js")).status_code == 200
            assert (await client.get("/assets/katex/katex.min.css")).status_code == 200
            assert (await client.get("/assets/katex/fonts/KaTeX_Main-Regular.woff2")).status_code == 200
            session_id = str(uuid.uuid4())
            path = f"/api/chat/sessions/{session_id}"
            payload = {"messages": [{"role": "user", "content": "hello"},
                                    {"role": "assistant", "content": "hi"}]}
            assert (await client.put(path, json=payload)).status_code == 401
            headers = {"Authorization": "Bearer private-key"}
            assert (await client.put(path, json=payload, headers=headers)).status_code == 200
            assert (await client.get(path, headers=headers)).json()["messages"] == payload["messages"]

    asyncio.run(run())


def test_spawn_is_explicit_without_changing_global_python_context(monkeypatch):
    import multiprocessing
    import os

    before = multiprocessing.get_start_method(allow_none=True)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    configure_vllm_multiprocessing()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert multiprocessing.get_start_method(allow_none=True) == before
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    with pytest.raises(ValueError, match="spawn"):
        configure_vllm_multiprocessing()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_text_usage_finish_and_echo(chat, stream):
    async def run():
        app, engine, tokenizer = setup_app()
        async with app.router.lifespan_context(app), client_for(app) as client:
            body = {
                "model": "alias",
                "temperature": 0,
                "max_tokens": 4,
                "stream": stream,
            }
            body.update(
                {"messages": [{"role": "user", "content": "abc"}]}
                if chat
                else {"prompt": "abc", "echo": True}
            )
            if stream:
                body["stream_options"] = {"include_usage": True}
            response = await client.post(
                "/v1/chat/completions" if chat else "/v1/completions", json=body
            )
            assert response.status_code == 200
            if stream:
                events = [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                text = "".join(
                    (
                        e["choices"][0]["delta"].get("content", "")
                        if chat
                        else e["choices"][0]["text"]
                    )
                    for e in events
                    if e["choices"]
                )
                assert text == ("你好" if chat else "abc你好")
                assert events[-2]["choices"][0]["finish_reason"] == "stop"
                usage = events[-1]["usage"]
                assert response.text.endswith("data: [DONE]\n\n")
            else:
                result = response.json()
                choice = result["choices"][0]
                assert (choice["message"]["content"] if chat else choice["text"]) == (
                    "你好" if chat else "abc你好"
                )
                usage = result["usage"]
                assert choice["finish_reason"] == "stop"
            assert usage == {
                "prompt_tokens": 4 if chat else 3,
                "completion_tokens": 3,
                "total_tokens": 7 if chat else 6,
            }
            assert engine.requests[0][1]["temperature"] == 0
            assert (
                len(engine.requests[0][0]["prompt_token_ids"]) == usage["prompt_tokens"]
            )
            if chat:
                assert tokenizer.template_kwargs["enable_thinking"] is False
            assert app.state.admission.snapshot()["completed"] == 1
            assert app.state.admission.tokens == 0 and not engine.aborted

    asyncio.run(run())


@pytest.mark.parametrize(
    "body,status",
    [
        ({"prompt": ""}, 400),
        ({"prompt": "a" * 21}, 400),
        ({"prompt": "abc", "stop": [""]}, 400),
        ({"prompt": "abc", "stop": ["a", "b", "c", "d", "e"]}, 400),
        ({"prompt": "abc", "model": "other"}, 404),
        ({"prompt": "abc", "n": 2}, 422),
    ],
)
def test_invalid_request_fails_before_engine_submission(body, status):
    async def run():
        app, engine, _ = setup_app()
        async with app.router.lifespan_context(app), client_for(app) as client:
            assert (
                await client.post("/v1/completions", json=body)
            ).status_code == status
            assert not engine.requests and not app.state.admission.leases

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("incomplete", [False, True])
def test_engine_failure_is_sanitized_and_aborts(stream, incomplete):
    async def run():
        app, engine, _ = setup_app()
        engine.incomplete, engine.fail = incomplete, not incomplete
        async with app.router.lifespan_context(app), client_for(app) as client:
            response = await client.post(
                "/v1/completions", json={"prompt": "abc", "stream": stream}
            )
            assert response.status_code == (200 if stream else 500)
            assert "PRIVATE" not in response.text
            assert "Model generation failed" in response.text
            assert len(engine.aborted) == 1
            assert not app.state.admission.leases
            assert app.state.admission.snapshot()["failed"] == 1

    asyncio.run(run())


def test_parallel_requests_are_not_serialized_by_local_default():
    async def run():
        app, engine, _ = setup_app(max_pending_requests=1)
        engine.release = asyncio.Event()
        async with app.router.lifespan_context(app), client_for(app) as client:
            tasks = [
                asyncio.create_task(
                    client.post("/v1/completions", json={"prompt": "abc"})
                )
                for _ in range(3)
            ]
            for _ in range(100):
                if len(engine.requests) == 3:
                    break
                await asyncio.sleep(0.005)
            assert len(engine.requests) == 3  # No semaphore(max_concurrency=1).
            rejected = await client.post("/v1/completions", json={"prompt": "abc"})
            assert (
                rejected.status_code == 503 and rejected.headers["Retry-After"] == "1"
            )
            engine.release.set()
            assert all(r.status_code == 200 for r in await asyncio.gather(*tasks))
            assert app.state.admission.snapshot()["rejected"] == 1

    asyncio.run(run())


def test_cancel_retains_admission_until_abort_acknowledged():
    async def run():
        app, engine, _ = setup_app(max_pending_requests=1)
        engine.release, engine.abort_release = asyncio.Event(), asyncio.Event()
        async with app.router.lifespan_context(app), client_for(app) as client:
            task = asyncio.create_task(
                client.post("/v1/completions", json={"prompt": "abc"})
            )
            await engine.entered.wait()
            task.cancel()
            await engine.abort_entered.wait()
            assert len(app.state.admission.leases) == 1
            engine.abort_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(engine.aborted) == 1 and not app.state.admission.leases
            assert app.state.admission.tokens == 0

    asyncio.run(run())


def test_token_reservation_budget_rejects_before_gpu_submission():
    async def run():
        app, engine, _ = setup_app(max_pending_tokens=6)
        async with app.router.lifespan_context(app), client_for(app) as client:
            response = await client.post("/v1/completions", json={"prompt": "abc"})
            assert response.status_code == 503  # 3 input + 4 output budget > 6.
            assert not engine.requests and app.state.admission.tokens == 0

    asyncio.run(run())


def test_streaming_response_releases_unstarted_iterator():
    async def run():
        app, _, _ = setup_app()
        admission = Admission(app.state.settings)
        lease = admission.acquire(app.state.engine)
        started = []

        async def events():
            started.append(True)
            yield "data"

        response = OwnedStreamingResponse(events(), lease)

        async def send(message):
            raise asyncio.CancelledError()

        async def receive():
            return {"type": "http.disconnect"}

        try:
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )
        except asyncio.CancelledError:
            pass  # Older Starlette suppresses cancellation in its task group.
        assert not started and not admission.leases

    asyncio.run(run())
