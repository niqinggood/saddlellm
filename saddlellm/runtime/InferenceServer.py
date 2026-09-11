"""OpenAI-compatible, non-streaming inference service for SaddleLLM releases."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


@dataclass
class InferenceServerSettings:
    model_path: str = ""
    model_name: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    max_concurrency: int = 1
    max_input_tokens: int = 4096
    default_max_new_tokens: int = 256
    max_new_tokens: int = 1024
    api_key: Optional[str] = None
    merge_adapter: bool = True

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1.")
        if self.max_input_tokens < 1 or self.default_max_new_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("Token limits must be positive integers.")
        if self.default_max_new_tokens > self.max_new_tokens:
            raise ValueError("default_max_new_tokens cannot exceed max_new_tokens.")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)
    repetition_penalty: float = Field(default=1.0, gt=0.0)


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)
    echo: bool = False
    repetition_penalty: float = Field(default=1.0, gt=0.0)


class LocalGenerationEngine:
    """Small synchronous generation core used behind the async HTTP layer."""

    def __init__(self, model: Any, tokenizer: Any, settings: InferenceServerSettings):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.model_name = settings.model_name or self._infer_model_name(settings.model_path)

    def chat(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        messages = [message.model_dump(exclude_none=True) for message in request.messages]
        if not messages:
            raise ValueError("messages must contain at least one item.")
        prompt = self._chat_prompt(messages)
        text, prompt_tokens, completion_tokens, finish_reason = self._generate(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def complete(self, request: CompletionRequest) -> Dict[str, Any]:
        if not request.prompt:
            raise ValueError("prompt must not be empty.")
        text, prompt_tokens, completion_tokens, finish_reason = self._generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop,
        )
        if request.echo:
            text = request.prompt + text
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _chat_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str:
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                return apply_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (ValueError, TypeError, AttributeError):
                pass
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop: Optional[Union[str, List[str]]],
    ) -> Tuple[str, int, int, str]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for local inference.") from exc

        requested_tokens = max_tokens or self.settings.default_max_new_tokens
        requested_tokens = min(requested_tokens, self.settings.max_new_tokens)
        context_limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        input_limit = self.settings.max_input_tokens
        if isinstance(context_limit, int) and context_limit > 1:
            requested_tokens = min(requested_tokens, context_limit - 1)
            input_limit = min(input_limit, max(1, context_limit - requested_tokens))
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(1, input_limit),
        )
        input_ids = encoded["input_ids"]
        prompt_tokens = int(input_ids.shape[-1])
        target_device = self._input_device()
        encoded = {
            key: value.to(target_device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        do_sample = temperature > 0.0
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        generate_kwargs: Dict[str, Any] = {
            **encoded,
            "max_new_tokens": requested_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            generate_kwargs["top_p"] = top_p
        with torch.inference_mode():
            output = self.model.generate(**generate_kwargs)
        sequences = getattr(output, "sequences", output)
        generated_ids = sequences[0, prompt_tokens:]
        completion_tokens = int(generated_ids.shape[-1])
        token_values = generated_ids.tolist() if hasattr(generated_ids, "tolist") else generated_ids
        text = self.tokenizer.decode(token_values, skip_special_tokens=True)
        text, stopped = self._apply_stop(text, stop)
        finish_reason = "stop" if stopped or completion_tokens < requested_tokens else "length"
        return text, prompt_tokens, completion_tokens, finish_reason

    def _input_device(self):
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return "cpu"

    @staticmethod
    def _apply_stop(
        text: str,
        stop: Optional[Union[str, List[str]]],
    ) -> Tuple[str, bool]:
        if stop is None:
            return text, False
        stops = [stop] if isinstance(stop, str) else stop
        positions = [text.find(item) for item in stops if item and text.find(item) >= 0]
        if not positions:
            return text, False
        return text[: min(positions)], True

    @staticmethod
    def _infer_model_name(model_path: str) -> str:
        if not model_path:
            return "saddlellm-model"
        normalized = model_path.rstrip("/\\")
        return os.path.basename(normalized) or normalized


def create_inference_app(
    settings: Optional[InferenceServerSettings] = None,
    *,
    model: Any = None,
    tokenizer: Any = None,
):
    """Create a FastAPI app, optionally with an already loaded model for tests."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as exc:
        raise ImportError("FastAPI is required to create the inference service.") from exc

    settings = settings or InferenceServerSettings()
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must be provided together.")
    if model is None:
        if not settings.model_path:
            raise ValueError("settings.model_path is required when no model is supplied.")
        model, tokenizer = load_inference_model(settings)
    engine = LocalGenerationEngine(model, tokenizer, settings)
    semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))
    release_metadata = _load_release_metadata(settings.model_path)

    app = FastAPI(title="SaddleLLM OpenAI-Compatible API", version="1.0.0")
    app.state.settings = settings
    app.state.engine = engine

    async def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        if not settings.api_key:
            return
        scheme, _, credential = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(credential, settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        gate = release_metadata.get("gate") if release_metadata else None
        return {
            "status": "ok",
            "model": engine.model_name,
            "release_id": release_metadata.get("release_id") if release_metadata else None,
            "gate_accepted": gate.get("accepted") if isinstance(gate, dict) else None,
        }

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "saddlellm",
                }
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat_completions(request: ChatCompletionRequest) -> Dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported in this release.")
        async with semaphore:
            try:
                return await asyncio.to_thread(engine.chat, request)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/completions", dependencies=[Depends(authorize)])
    async def completions(request: CompletionRequest) -> Dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported in this release.")
        async with semaphore:
            try:
                return await asyncio.to_thread(engine.complete, request)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def load_inference_model(settings: InferenceServerSettings) -> Tuple[Any, Any]:
    """Load a full checkpoint or PEFT adapter for local generation."""
    if settings.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if settings.dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be one of: auto, float32, float16, bfloat16.")
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Local inference requires torch and transformers.") from exc

    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but CUDA is unavailable.")
    dtype_name = settings.dtype
    if dtype_name == "auto":
        dtype_name = "float16" if device == "cuda" else "float32"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    load_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": settings.trust_remote_code,
        "local_files_only": settings.local_files_only,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    model_path = settings.model_path
    local_path = Path(model_path).expanduser()
    adapter_config_path = local_path / "adapter_config.json"
    if adapter_config_path.is_file():
        from ..training.PostTrainingCompatibility import stabilize_peft_optional_backends

        for warning in stabilize_peft_optional_backends():
            logger.warning(warning)
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Loading an adapter release requires peft.") from exc
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("adapter_config.json does not declare base_model_name_or_path.")
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        model = PeftModel.from_pretrained(
            model,
            model_path,
            local_files_only=settings.local_files_only,
        )
        if settings.merge_adapter:
            model = model.merge_and_unload()
        tokenizer_source = model_path if _has_tokenizer_files(local_path) else base_model
    else:
        from ..models.ModelLoader import load_causal_lm

        model = load_causal_lm(
            model_path,
            device=device,
            dtype=dtype,
            trust_remote_code=settings.trust_remote_code,
            local_files_only=settings.local_files_only,
        )
        tokenizer_source = model_path
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    from ..models.TokenizerLoader import load_tokenizer_compatible

    tokenizer = load_tokenizer_compatible(
        tokenizer_source,
        trust_remote_code=settings.trust_remote_code,
        local_files_only=settings.local_files_only,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _has_tokenizer_files(path: Path) -> bool:
    return any(
        (path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "vocab.json",
            "vocab.txt",
        )
    )


def _load_release_metadata(model_path: str) -> Dict[str, Any]:
    if not model_path:
        return {}
    manifest = Path(model_path).expanduser() / "release_manifest.json"
    if not manifest.is_file():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
