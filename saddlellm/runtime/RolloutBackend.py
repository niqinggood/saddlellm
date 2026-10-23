"""Config-selected native or SGLang sampling; no heavyweight eager imports.

SGLang HTTP uses its native /generate protocol, not OpenAI chat completions.
This module samples batches. It does not implement an RL training algorithm.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import uuid
from array import array
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

from .BackendConfig import RuntimeBackendConfig, SGLangBackendConfig


@dataclass(frozen=True)
class RolloutSample:
    request_id: str
    text: str
    backend: str
    policy_version: str | None = None
    output_ids: list[int] | None = None
    output_logprobs: list[float] | None = None
    finish_reason: Any = None
    logprob_semantics: str | None = None


def _sampling_params(params):
    params = dict(params)
    allowed = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "stop",
        "min_new_tokens",
    }
    if set(params) - allowed:
        raise ValueError(
            f"Unsupported rollout parameters: {sorted(set(params) - allowed)}"
        )
    tokens = params.get("max_new_tokens", 128)
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    for name, default in (
        ("temperature", 1.0),
        ("top_p", 1.0),
        ("repetition_penalty", 1.0),
    ):
        value = params.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite")
        if (name == "temperature" and value < 0) or (
            name != "temperature" and value <= 0
        ):
            raise ValueError(f"invalid {name}")
        if name == "top_p" and value > 1:
            raise ValueError("top_p must be <= 1")
    for name, default in (("top_k", 0), ("min_new_tokens", 0)):
        value = params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        if name == "min_new_tokens" and value > tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")
    stops = params.get("stop")
    if stops is not None and not (
        isinstance(stops, str)
        or isinstance(stops, list)
        and all(isinstance(item, str) for item in stops)
    ):
        raise ValueError("stop must be a string or list of strings")
    return {"max_new_tokens": tokens, **params}


class _OwnedBackend:
    def __init__(self):
        self._lock = Lock()
        self._closed = False

    @contextmanager
    def _operation(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "rollout backend is busy; wait before sampling/updating/closing"
            )
        try:
            if self._closed:
                raise RuntimeError("rollout backend is closed")
            yield
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


async def _await_owned_sync(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Repeated cancellation must not abandon an active GPU/model operation.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise


class NativeRolloutBackend(_OwnedBackend):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer

    def generate(self, prompts, params):
        import torch

        params = _sampling_params(params)
        if isinstance(prompts, str):
            raise TypeError("provide a list of rollout prompts, not a single string")
        with self._operation():
            if self.model is None or self.tokenizer is None:
                raise ValueError("native rollout requires model and tokenizer")
            was_training = self.model.training
            samples = []
            self.model.eval()
            try:
                device = next(self.model.parameters()).device
                for prompt in prompts:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                    limit = getattr(
                        getattr(self.model, "config", None),
                        "max_position_embeddings",
                        None,
                    )
                    if (
                        isinstance(limit, int)
                        and inputs["input_ids"].shape[1] + params["max_new_tokens"]
                        > limit
                    ):
                        raise ValueError(
                            "native rollout prompt and budget exceed model context"
                        )
                    kwargs = dict(params)
                    stop = kwargs.pop("stop", None)
                    if stop:
                        from transformers import StoppingCriteria, StoppingCriteriaList

                        from .GenerationStop import StopStringRows

                        stops = [stop] if isinstance(stop, str) else stop
                        checker = StopStringRows(
                            self.tokenizer, inputs["input_ids"].shape[1], [stops]
                        )

                        class TextStop(StoppingCriteria):
                            def __call__(self, input_ids, scores, **kwargs):
                                return checker(input_ids)

                        kwargs["stopping_criteria"] = StoppingCriteriaList([TextStop()])
                    kwargs["do_sample"] = kwargs.get("temperature", 1.0) > 0
                    if not kwargs["do_sample"]:
                        kwargs.pop("temperature", None)
                    pad = self.tokenizer.pad_token_id
                    kwargs["pad_token_id"] = (
                        pad if pad is not None else self.tokenizer.eos_token_id
                    )
                    with torch.inference_mode():
                        output = self.model.generate(**inputs, **kwargs)
                    sequence = getattr(output, "sequences", output)
                    ids = sequence[0, inputs["input_ids"].shape[1] :].tolist()
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    if stop:
                        positions = [
                            text.find(item) for item in stops if item and item in text
                        ]
                        if positions:
                            text = text[: min(positions)]
                    samples.append(
                        RolloutSample(
                            uuid.uuid4().hex,
                            text,
                            "native",
                            output_ids=ids,
                        )
                    )
            finally:
                self.model.train(was_training)
            return samples

    async def agenerate(self, prompts, params):
        return await _await_owned_sync(self.generate, prompts, params)

    def close(self):
        if self._closed:
            return
        with self._operation():
            self._closed = True
            self.model = self.tokenizer = None


class SGLangRolloutBackend(_OwnedBackend):
    def __init__(
        self, config: SGLangBackendConfig, *, http_transport=None, engine_factory=None
    ):
        super().__init__()
        config.validate()
        self.config = replace(config, engine_kwargs=dict(config.engine_kwargs))
        self._transport = http_transport
        self._engine_factory = engine_factory
        self._engine = None
        self._failed_update = False

    def _http_client(self):
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "SGLang HTTP requires pip install 'saddlellm[rollout]' (httpx)"
            ) from exc
        headers = {}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if not key:
                raise ValueError(
                    f"SGLang credential env {self.config.api_key_env} is empty"
                )
            headers["Authorization"] = f"Bearer {key}"
        return httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=self.config.timeout_seconds,
            transport=self._transport,
            limits=httpx.Limits(max_connections=self.config.concurrency),
            follow_redirects=False,
            trust_env=False,
        )

    async def _post(self, client, route, payload):
        import httpx

        try:
            response = await client.post(route, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Do not expose server bodies, URLs with credentials, or API keys.
            reason = (
                f"HTTP {exc.response.status_code}"
                if isinstance(exc, httpx.HTTPStatusError)
                else type(exc).__name__
            )
            raise RuntimeError(f"SGLang {route} failed ({reason})") from None

    def _parse(self, body, request_id):
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            raise RuntimeError("SGLang returned an invalid generation response")
        meta = body.get("meta_info", {})
        if not isinstance(meta, dict):
            raise RuntimeError("SGLang returned invalid meta_info")
        version = meta.get("weight_version")
        if (
            self.config.policy_version is not None
            and version != self.config.policy_version
        ):
            raise RuntimeError(
                "SGLang policy version mismatch or missing weight_version"
            )
        reason = meta.get("finish_reason")
        if isinstance(reason, dict) and reason.get("type") == "abort":
            raise RuntimeError("SGLang aborted generation; no completed sample")
        ids = body.get("output_ids")
        if isinstance(ids, (tuple, array)):
            ids = list(ids)
        if ids is not None and (
            not isinstance(ids, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in ids
            )
        ):
            raise RuntimeError("SGLang returned invalid output_ids")
        logprobs = None
        if self.config.return_logprob:
            entries = meta.get("output_token_logprobs")
            if not isinstance(entries, list):
                raise RuntimeError("SGLang omitted requested output_token_logprobs")
            if any(
                not isinstance(entry, (tuple, list))
                or len(entry) < 2
                or isinstance(entry[1], bool)
                or not isinstance(entry[1], int)
                or entry[1] < 0
                for entry in entries
            ):
                raise RuntimeError("SGLang returned invalid token logprobs")
            try:
                logprobs = [float(entry[0]) for entry in entries]
                token_ids = [entry[1] for entry in entries]
            except (TypeError, ValueError, IndexError):
                raise RuntimeError("SGLang returned invalid token logprobs") from None
            if not all(math.isfinite(value) for value in logprobs):
                raise RuntimeError("SGLang returned non-finite token logprobs")
            if ids is None:
                ids = token_ids
            if ids != token_ids:
                raise RuntimeError("SGLang output_ids/logprobs are not aligned")
        return RolloutSample(
            request_id,
            body["text"],
            "sglang",
            version,
            ids,
            logprobs,
            meta.get("finish_reason"),
            "backend_reported" if logprobs is not None else None,
        )

    def _prompts(self, prompts):
        if isinstance(prompts, str):
            raise TypeError("provide a list of rollout prompts, not a single string")
        prompts = list(prompts)
        if len(prompts) > self.config.max_pending_requests:
            raise ValueError(
                "rollout batch exceeds max_pending_requests; split the batch"
            )
        if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError("rollout prompts must be non-empty strings")
        if self._failed_update:
            raise RuntimeError("weight update failed; restore policy before sampling")
        return prompts

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        if not self.config.model_path:
            raise ValueError("SGLang engine requires model_path")
        factory = self._engine_factory
        if factory is None:
            if sys.platform != "linux":
                raise RuntimeError(
                    "SGLang local Engine requires a supported Linux GPU environment; "
                    "use mode=http"
                )
            try:
                from sglang import Engine
            except ImportError as exc:
                raise ImportError(
                    "Install SGLang in the optional Linux GPU runtime"
                ) from exc
            factory = Engine
        kwargs = {"model_path": self.config.model_path, **self.config.engine_kwargs}
        if self.config.policy_version is not None:
            kwargs["weight_version"] = self.config.policy_version
        self._engine = factory(**kwargs)
        return self._engine

    def generate(self, prompts, params):
        if self.config.mode == "http":
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.agenerate(prompts, params))
            raise RuntimeError("use await backend.agenerate() inside an event loop")
        with self._operation():
            prompts = self._prompts(prompts)
            sampling = _sampling_params(params)
            if sampling.get("top_k") == 0:
                sampling["top_k"] = -1  # SGLang's unbounded top-k sentinel
            if not prompts:
                return []
            bodies = self._ensure_engine().generate(
                prompt=prompts,
                sampling_params=sampling,
                return_logprob=self.config.return_logprob,
            )
            if not isinstance(bodies, list) or len(bodies) != len(prompts):
                raise RuntimeError("SGLang Engine returned the wrong batch size")
            return [self._parse(body, uuid.uuid4().hex) for body in bodies]

    async def agenerate(self, prompts, params):
        if self.config.mode == "engine":
            # Engine's synchronous loop has one owner and runs off the event loop.
            return await _await_owned_sync(self.generate, prompts, params)
        with self._operation():
            prompts = self._prompts(prompts)
            sampling = _sampling_params(params)
            if sampling.get("top_k") == 0:
                sampling["top_k"] = -1
            if not prompts:
                return []
            results = [None] * len(prompts)
            jobs = iter(enumerate(prompts))
            active = set()
            async with self._http_client() as client:

                async def worker():
                    for index, prompt in jobs:
                        rid = uuid.uuid4().hex
                        active.add(rid)
                        body = await self._post(
                            client,
                            "generate",
                            {
                                "text": prompt,
                                "sampling_params": sampling,
                                "rid": rid,
                                "stream": False,
                                "return_logprob": self.config.return_logprob,
                            },
                        )
                        results[index] = self._parse(body, rid)
                        active.discard(rid)

                tasks = [
                    asyncio.create_task(worker())
                    for _ in range(min(len(prompts), self.config.concurrency))
                ]
                try:
                    await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

                    # Best effort, request-scoped abort; never abort other users.
                    async def abort(rid):
                        try:
                            await client.post(
                                "abort_request", json={"rid": rid}, timeout=5
                            )
                        except Exception:
                            pass

                    await asyncio.gather(*(abort(rid) for rid in active))
                    raise
            return results

    def prepare_policy(self, model_path, policy_version):
        """A batch barrier: publish weights before sampling, never during it."""
        if not model_path or not isinstance(policy_version, str) or not policy_version:
            raise ValueError(
                "policy preparation requires model_path and policy_version"
            )
        if self.config.mode == "http":
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.aprepare_policy(model_path, policy_version))
            raise RuntimeError(
                "use await backend.aprepare_policy() inside an event loop"
            )
        with self._operation():
            self._failed_update = True
            if self._engine is not None:
                self._engine.shutdown()
                self._engine = None
            # Recreate on the next batch: no old-model KV can survive.
            self.config.model_path = str(model_path)
            self.config.policy_version = policy_version
            self._failed_update = False

    async def aprepare_policy(self, model_path, policy_version):
        if self.config.mode == "engine":
            return await _await_owned_sync(
                self.prepare_policy, model_path, policy_version
            )
        if not model_path or not isinstance(policy_version, str) or not policy_version:
            raise ValueError(
                "policy preparation requires model_path and policy_version"
            )
        with self._operation():
            self._failed_update = True
            async with self._http_client() as client:
                body = await self._post(
                    client,
                    "update_weights_from_disk",
                    {
                        "model_path": str(model_path),
                        "weight_version": policy_version,
                        "abort_all_requests": False,
                        "is_async": False,
                    },
                )
                if not isinstance(body, dict) or body.get("success") is not True:
                    raise RuntimeError("SGLang rejected the policy weight update")
            self.config.model_path = str(model_path)
            self.config.policy_version = policy_version
            self._failed_update = False

    def close(self):
        if self._closed:
            return
        with self._operation():
            if self._engine is not None:
                self._engine.shutdown()
                self._engine = None
            self._closed = True


def create_rollout_backend(
    config=None, *, model=None, tokenizer=None, **adapter_kwargs
):
    """Create a lazy, owned backend from a full config or its backends block."""
    if not isinstance(config, RuntimeBackendConfig):
        raw = config or {}
        config = RuntimeBackendConfig.from_dict(raw.get("backends", raw))
    config.validate()
    if config.rollout == "sglang":
        return SGLangRolloutBackend(config.sglang, **adapter_kwargs)
    if adapter_kwargs:
        raise ValueError("SGLang adapter arguments cannot be used for native rollout")
    return NativeRolloutBackend(model, tokenizer)
