"""Direct asynchronous vLLM serving: one engine, bounded requests, real usage.

vLLM imports and GPU allocation happen at app startup, not package import.
This is a text-only, n=1 API adapter, not a claim to implement all OpenAI APIs.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import platform
import secrets
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from .InferenceServer import (
    ChatCompletionRequest,
    CompletionRequest,
    _load_release_metadata,
)

logger = logging.getLogger(__name__)


def configure_vllm_multiprocessing():
    """CUDA cannot safely fork. Do not change Python's global start method."""
    requested = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if requested not in {None, "spawn"}:
        raise ValueError(
            "This CUDA adapter requires VLLM_WORKER_MULTIPROC_METHOD=spawn; do not use fork. Python scripts need an if __name__ == '__main__' guard."
        )
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@dataclass
class PreparedRequest:
    token_ids: list[int]
    sampling_params: object
    max_tokens: int


class VLLMRuntime:
    """Small boundary around the verified vLLM 0.30 async engine API."""

    def __init__(self, settings, *, engine=None, tokenizer=None, sampling_factory=None):
        self.settings = settings
        self.engine = engine
        self.tokenizer = tokenizer
        self.sampling_factory = sampling_factory
        self.context_limit = settings.vllm.max_model_len or (
            settings.max_input_tokens + settings.max_new_tokens
        )
        self.model_name = settings.model_name or Path(settings.model_path).name
        self.engine_version = None

    async def start(self):
        try:
            if self.engine is None:
                if platform.system() != "Linux":
                    raise RuntimeError(
                        "backend=vllm currently requires Linux CUDA; use backend=local on Windows/macOS."
                    )
                configure_vllm_multiprocessing()
                try:
                    import torch
                    from packaging.specifiers import SpecifierSet

                    self.engine_version = version("vllm")
                    if self.engine_version not in SpecifierSet(">=0.30,<0.31"):
                        raise RuntimeError(
                            "This adapter supports vLLM >=0.30,<0.31. Install saddlellm[serve-vllm] in a separate environment."
                        )
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "backend=vllm requires available CUDA; no CPU fallback is performed."
                        )
                    if (
                        torch.cuda.device_count()
                        < self.settings.vllm.tensor_parallel_size
                    ):
                        raise RuntimeError(
                            "tensor_parallel_size exceeds the visible CUDA GPU count."
                        )
                    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
                except ImportError as exc:
                    raise ImportError(
                        "Install saddlellm[serve-vllm] in a separate Linux CUDA environment."
                    ) from exc
                from ..models.ModelLoader import resolve_model_reference

                reference = resolve_model_reference(
                    self.settings.model_path,
                    local_files_only=self.settings.local_files_only,
                )
                directory = Path(reference)
                if any(
                    (directory / name).is_file()
                    for name in (
                        "saddle_config.json",
                        "quantization_manifest.json",
                        "adapter_config.json",
                    )
                ):
                    raise ValueError(
                        "vLLM requires a supported full HF checkpoint, not a native Saddle checkpoint or a standalone adapter; merge/export first."
                    )
                kwargs = self.settings.vllm.engine_kwargs(self.settings, reference)
                self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
                self.sampling_factory = SamplingParams
                self.tokenizer = self.engine.get_tokenizer()
                if inspect.isawaitable(self.tokenizer):
                    self.tokenizer = await self.tokenizer
            if self.tokenizer is None or self.sampling_factory is None:
                raise ValueError("vLLM runtime needs a tokenizer and sampling factory.")
            model_config = getattr(self.engine, "model_config", None)
            self.context_limit = getattr(
                model_config, "max_model_len", self.context_limit
            )
            await self.check_health()
        except BaseException:
            await self.close()
            raise

    async def close(self):
        engine, self.engine = self.engine, None
        if engine is not None:
            # vLLM 0.30 shutdown cleans its owned worker/IPC resources.
            result = await asyncio.to_thread(engine.shutdown, timeout=20.0)
            if inspect.isawaitable(result):
                await result

    async def check_health(self):
        if self.engine is None:
            raise RuntimeError("vLLM engine is not running.")
        await self.engine.check_health()

    async def prepare(self, request):
        def tokenize():
            is_chat = isinstance(request, ChatCompletionRequest)
            if is_chat:
                messages = request.template_messages()
                template_options = {"tokenize": False, "add_generation_prompt": True}
                if self.settings.enable_thinking is not None:
                    template_options["enable_thinking"] = self.settings.enable_thinking
                try:
                    prompt = self.tokenizer.apply_chat_template(messages, **template_options)
                except TypeError:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            else:
                prompt = request.prompt
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    "prompt must not be empty and chat requires a supported template."
                )
            return self.tokenizer.encode(prompt, add_special_tokens=not is_chat)

        tokens = await asyncio.to_thread(tokenize)
        if not tokens:
            raise ValueError("prompt must contain at least one token.")
        budget = min(
            request.max_tokens or self.settings.default_max_new_tokens,
            self.settings.max_new_tokens,
        )
        if (
            len(tokens) > self.settings.max_input_tokens
            or len(tokens) + budget > self.context_limit
        ):
            raise ValueError(
                "Input plus output budget exceeds the configured token/context limit; shorten the request. No silent truncation is performed."
            )
        if not math.isfinite(request.repetition_penalty):
            raise ValueError("repetition_penalty must be finite.")
        stops = [request.stop] if isinstance(request.stop, str) else request.stop
        if stops is not None and (
            len(stops) > 4 or any(not isinstance(s, str) or not s for s in stops)
        ):
            raise ValueError("stop must contain at most four non-empty strings.")
        params = self.sampling_factory(
            n=1,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_tokens=budget,
            stop=stops,
            include_stop_str_in_output=False,
        )
        return PreparedRequest(list(tokens), params, budget)

    def generate(self, prepared, request_id):
        return self.engine.generate(
            {"prompt_token_ids": prepared.token_ids},
            prepared.sampling_params,
            request_id,
        )

    async def abort(self, request_id):
        if self.engine is not None:
            await self.engine.abort(request_id)


class Admission:
    """Event-loop-local admission, not a competing GPU scheduler."""

    def __init__(self, settings):
        self.capacity = settings.vllm.max_num_seqs + settings.max_pending_requests
        self.token_limit = settings.max_pending_tokens
        self.tokens = 0
        self.leases = {}
        self.closed = False
        self.counts = Counter()
        self.recent = deque(maxlen=256)

    def acquire(self, runtime):
        if self.closed or len(self.leases) >= self.capacity:
            self.reject()
        lease = Lease(self, runtime)
        self.leases[lease.id] = lease
        self.counts["accepted"] += 1
        return lease

    def reject(self):
        self.counts["rejected"] += 1
        raise HTTPException(
            503,
            "Inference admission is full or unavailable.",
            headers={"Retry-After": "1"},
        )

    def snapshot(self):
        return {
            "capacity": self.capacity,
            "in_flight": len(self.leases),
            "reserved_tokens": self.tokens,
            "token_limit": self.token_limit,
            "closed": self.closed,
            **dict(self.counts),
            "recent_requests": list(self.recent),
        }


class Lease:
    def __init__(self, admission, runtime):
        self.admission, self.runtime = admission, runtime
        self.id = uuid.uuid4().hex
        self.tokens = self.completion_tokens = self.prompt_tokens = 0
        self.started = self.finished = self.closed = False
        self.outcome = "cancelled"
        self.created = time.perf_counter()
        self.first_text = None
        self.cleanup_task = None

    def reserve(self, prepared):
        tokens = len(prepared.token_ids) + prepared.max_tokens
        if (
            self.admission.token_limit is not None
            and self.admission.tokens + tokens > self.admission.token_limit
        ):
            self.admission.reject()
        self.tokens = tokens
        self.prompt_tokens = len(prepared.token_ids)
        self.admission.tokens += tokens

    async def close(self):
        if self.cleanup_task is None:
            self.closed = True
            self.cleanup_task = asyncio.create_task(self._release())
        with anyio.CancelScope(shield=True):
            await asyncio.shield(self.cleanup_task)

    async def _release(self):
        with anyio.CancelScope(shield=True):
            try:
                if self.started and not self.finished:
                    await asyncio.wait_for(self.runtime.abort(self.id), timeout=10)
            except Exception:
                # Abort acknowledgement is unknown: stop new admission instead
                # of pretending that GPU/KV ownership has safely been released.
                self.admission.closed = True
                logger.exception("vLLM abort failed; new admission disabled")
            finally:
                self.admission.leases.pop(self.id, None)
                self.admission.tokens -= self.tokens
                self.admission.counts[self.outcome] += 1
                self.admission.counts["reported_completion_tokens"] += (
                    self.completion_tokens
                )
                self.admission.recent.append(
                    {
                        "outcome": self.outcome,
                        "response_ms": (time.perf_counter() - self.created) * 1000,
                        "ttft_ms": None
                        if self.first_text is None
                        else (self.first_text - self.created) * 1000,
                        "prompt_tokens": self.prompt_tokens,
                        "completion_tokens": self.completion_tokens,
                    }
                )


def _snapshot(output, prepared, lease):
    if not output.outputs and not output.finished:
        return None
    if len(output.outputs) != 1:
        raise RuntimeError("Expected one vLLM completion.")
    candidate = output.outputs[0]
    text, count = candidate.text, len(candidate.token_ids)
    if (
        not isinstance(text, str)
        or count > prepared.max_tokens
        or count < lease.completion_tokens
    ):
        raise RuntimeError("Invalid cumulative vLLM output.")
    lease.completion_tokens = count
    if text and lease.first_text is None:
        lease.first_text = time.perf_counter()
    if output.finished:
        if candidate.finish_reason not in {"stop", "length"}:
            raise RuntimeError("vLLM did not complete successfully.")
        lease.finished = True
        lease.outcome = "completed"
    return text, candidate.finish_reason if output.finished else None


class OwnedStreamingResponse(StreamingResponse):
    """Release admission even when disconnected before iteration starts."""

    def __init__(self, iterator, lease):
        super().__init__(
            iterator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        self.lease = lease

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.lease.close()


async def _until_disconnect(request, stopped):
    # Starlette's is_disconnected uses a cancelled scope. An explicit stop
    # event prevents that scope swallowing Task.cancel and leaking a poller.
    while not stopped.is_set():
        if await request.is_disconnected():
            return
        try:
            await asyncio.wait_for(stopped.wait(), timeout=0.05)
        except asyncio.TimeoutError:
            pass


def create_vllm_inference_app(settings, *, runtime=None):
    """Shared public factory dispatches here; runtime injection is for tests."""
    if not settings.model_path and runtime is None:
        raise ValueError("model_path is required for the vLLM backend.")
    runtime = runtime or VLLMRuntime(settings)
    admission = Admission(settings)
    release_metadata = _load_release_metadata(settings.model_path)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await runtime.start()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            admission.closed = True
            for lease in list(admission.leases.values()):
                await lease.close()
            await runtime.close()

    app = FastAPI(title="SaddleLLM OpenAI-Compatible API", lifespan=lifespan)
    from .OpenAIProtocol import API_CAPABILITIES, install_openai_error_handlers

    install_openai_error_handlers(app)
    app.state.settings, app.state.engine = settings, runtime
    app.state.admission, app.state.ready = admission, False

    async def authorize(authorization: str | None = Header(default=None)):
        if settings.api_key:
            scheme, _, credential = (authorization or "").partition(" ")
            if scheme.lower() != "bearer" or not secrets.compare_digest(
                credential, settings.api_key
            ):
                raise HTTPException(401, "Invalid or missing bearer token.")

    async def ready():
        if not app.state.ready or admission.closed:
            raise HTTPException(503, "Inference service is not ready.")
        try:
            await asyncio.wait_for(runtime.check_health(), timeout=3)
        except Exception as exc:
            app.state.ready = False
            raise HTTPException(503, "Inference engine is unavailable.") from exc

    @app.get("/livez")
    async def liveness():
        return {"status": "alive"}

    @app.get("/readyz")
    async def readiness():
        await ready()
        return {"status": "ready"}

    @app.get("/health")
    async def health():
        from ..evaluation.ReleaseGate import release_gate_accepted

        available = True
        try:
            await ready()
        except HTTPException:
            available = False
        model_config = getattr(runtime.engine, "model_config", None)
        gate = release_metadata.get("gate")
        return {
            "status": "ok" if available else "unavailable",
            "ready": available,
            "backend": "vllm",
            "backend_selection": settings.backend_selection,
            "api_capabilities": API_CAPABILITIES,
            "thinking_policy": ("model_default" if settings.enable_thinking is None
                                else "enabled" if settings.enable_thinking else "disabled"),
            "input_policy": "reject_overlong",
            "backend_version": runtime.engine_version,
            "device": "cuda",
            "model": runtime.model_name,
            "dtype": str(getattr(model_config, "dtype", "unresolved")),
            "quantization": getattr(model_config, "quantization", None) or "none",
            "streaming": True,
            "streaming_backend": "vllm_async",
            "requires_auth": bool(settings.api_key),
            "web_ui": settings.web_ui,
            "chat_history_enabled": settings.web_ui and settings.chat_history,
            "release_id": release_metadata.get("release_id"),
            "gate_accepted": release_gate_accepted(gate)
            if isinstance(gate, dict)
            else None,
            "context_limit": runtime.context_limit,
            "gpu_memory_utilization": settings.vllm.gpu_memory_utilization,
            "tensor_parallel_size": settings.vllm.tensor_parallel_size,
            "prefix_cache": {
                "enabled": settings.vllm.enable_prefix_caching,
                "owner": "vllm",
            },
            "batching": {
                "enabled": True,
                "mode": "vllm_continuous",
                "max_num_seqs": settings.vllm.max_num_seqs,
            },
            "admission": admission.snapshot(),
        }

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    if settings.web_ui:
        from .ChatWebRoutes import install_chat_web_routes

        install_chat_web_routes(app, settings, runtime.model_name, authorize)

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": runtime.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "saddlellm",
                }
            ],
        }

    def usage(lease):
        return {
            "prompt_tokens": lease.prompt_tokens,
            "completion_tokens": lease.completion_tokens,
            "total_tokens": lease.prompt_tokens + lease.completion_tokens,
        }

    async def respond(body, http_request):
        await ready()
        if body.model is not None and body.model != runtime.model_name:
            raise HTTPException(404, "Requested model is not served by this instance.")
        lease = admission.acquire(runtime)
        try:
            try:
                prepared = await runtime.prepare(body)
            except ValueError as exc:
                lease.outcome = "invalid"
                raise HTTPException(400, str(exc)) from exc
            lease.reserve(prepared)
            is_chat = isinstance(body, ChatCompletionRequest)
            response_id = ("chatcmpl-" if is_chat else "cmpl-") + lease.id
            base = {
                "id": response_id,
                "created": int(time.time()),
                "model": runtime.model_name,
            }

            async def collect():
                lease.started = True
                try:
                    final = None
                    async for output in runtime.generate(prepared, lease.id):
                        item = _snapshot(output, prepared, lease)
                        if item is not None:
                            final = item
                    if final is None or not lease.finished:
                        raise RuntimeError("vLLM stream ended without a final result.")
                    text, finish = final
                    choice = {"index": 0, "finish_reason": finish}
                    if is_chat:
                        choice["message"] = {"role": "assistant", "content": text}
                    else:
                        choice["text"] = (body.prompt if body.echo else "") + text
                    return {
                        **base,
                        "object": "chat.completion" if is_chat else "text_completion",
                        "choices": [choice],
                        "usage": usage(lease),
                    }
                except Exception:
                    lease.outcome = "failed"
                    raise

            if not body.stream:
                stopped = asyncio.Event()
                work = asyncio.create_task(collect())
                disconnect = asyncio.create_task(
                    _until_disconnect(http_request, stopped)
                )
                try:
                    done, _ = await asyncio.wait(
                        [work, disconnect], return_when=asyncio.FIRST_COMPLETED
                    )
                    if work not in done:
                        raise asyncio.CancelledError()
                    return await work
                finally:
                    stopped.set()
                    if not work.done():
                        work.cancel()
                    with anyio.CancelScope(shield=True):
                        await asyncio.gather(work, disconnect, return_exceptions=True)
                        await lease.close()

            def event(payload):
                return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

            def chunk(text=None, finish=None, *, role=False):
                choice = {"index": 0, "finish_reason": finish}
                if is_chat:
                    choice["delta"] = (
                        {"role": "assistant"}
                        if role
                        else ({"content": text} if text is not None else {})
                    )
                else:
                    choice["text"] = text or ""
                return {
                    **base,
                    "object": "chat.completion.chunk" if is_chat else "text_completion",
                    "choices": [choice],
                }

            async def events():
                sent, final = "", None
                try:
                    if is_chat:
                        yield event(chunk(role=True))
                    elif body.echo:
                        yield event(chunk(body.prompt))
                    lease.started = True
                    async for output in runtime.generate(prepared, lease.id):
                        item = _snapshot(output, prepared, lease)
                        if item is None:
                            continue
                        text, finish = final = item
                        if not text.startswith(sent):
                            raise RuntimeError(
                                "vLLM cumulative text changed a previously emitted prefix."
                            )
                        delta, sent = text[len(sent) :], text
                        if delta:
                            yield event(chunk(delta))
                    if final is None or not lease.finished:
                        raise RuntimeError("vLLM stream ended without a final result.")
                    yield event(chunk(finish=final[1]))
                    if body.stream_options and body.stream_options.include_usage:
                        yield event(
                            {
                                **base,
                                "object": "chat.completion.chunk"
                                if is_chat
                                else "text_completion",
                                "choices": [],
                                "usage": usage(lease),
                            }
                        )
                    yield "data: [DONE]\n\n"
                except Exception:
                    lease.outcome = "failed"
                    logger.exception("vLLM streaming generation failed")
                    yield event(
                        {
                            "error": {
                                "message": "Model generation failed.",
                                "type": "server_error",
                            }
                        }
                    )
                    yield "data: [DONE]\n\n"
                finally:
                    await lease.close()

            return OwnedStreamingResponse(events(), lease)
        except BaseException as exc:
            if not isinstance(exc, (HTTPException, asyncio.CancelledError)):
                lease.outcome = "failed"
                logger.exception("vLLM model generation failed")
            await lease.close()
            if isinstance(exc, (HTTPException, asyncio.CancelledError)):
                raise
            raise HTTPException(500, "Model generation failed.") from exc

    @app.post("/v1/completions", dependencies=[Depends(authorize)])
    async def completions(body: CompletionRequest, request: Request):
        return await respond(body, request)

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        return await respond(body, request)

    return app
