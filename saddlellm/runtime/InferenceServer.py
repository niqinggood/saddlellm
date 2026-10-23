"""OpenAI-compatible inference service and built-in chat page."""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import os
import queue
import secrets
import threading
import time
import uuid
from collections import deque
from concurrent.futures import CancelledError as GenerationCancelled, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from starlette.requests import Request

from .VLLMConfig import VLLMServingConfig

logger = logging.getLogger(__name__)


def _cooperative_native_streaming(engine, settings) -> bool:
    """Experimental policy: benchmarks opt in; tail-latency gate not passed."""
    return False


def _native_stream_worker_count(engine, settings):
    """Experimental pool: opt in only in benchmarks until performance passes."""
    return None


def _can_batch_tokenizer_delivery(tokenizer) -> bool:
    """Only standard fast detokenizers may run after another row is sampled.

    A user tokenizer can consume RNG or execute arbitrary callbacks in decode.
    Keep its original sample/delivery ordering even inside our TextStreamer.
    """
    from tokenizers import Tokenizer
    from tokenizers.decoders import Decoder
    from transformers import tokenization_utils_base, tokenization_utils_fast

    base = tokenization_utils_base.PreTrainedTokenizerBase
    fast = tokenization_utils_fast.PreTrainedTokenizerFast
    if not isinstance(tokenizer, fast) or not type(tokenizer).__module__.startswith("transformers."):
        return False

    methods = (base.decode, fast._decode, base.clean_up_tokenization)
    originals = getattr(_can_batch_tokenizer_delivery, "_original_methods", None)
    if originals is None:
        # Capture lazily to keep importing the service cheap. Check provenance
        # first: a class may already have been patched before our first request.
        for method, owner, module, name in zip(
            methods, (base, fast, base),
            (tokenization_utils_base, tokenization_utils_fast, tokenization_utils_base),
            ("decode", "_decode", "clean_up_tokenization"),
        ):
            code = getattr(method, "__code__", None)
            if (code is None or method.__module__ != module.__name__
                    or method.__qualname__ != f"{owner.__name__}.{name}"
                    or os.path.normcase(os.path.abspath(code.co_filename))
                    != os.path.normcase(os.path.abspath(module.__file__))):
                return False
        originals = tuple((method, method.__code__) for method in methods)
        _can_batch_tokenizer_delivery._original_methods = originals
    if any(method is not original or getattr(method, "__code__", None) is not code
           for method, (original, code) in zip(methods, originals)):
        return False
    if (getattr(tokenizer.decode, "__func__", None) is not methods[0]
            or getattr(tokenizer._decode, "__func__", None) is not methods[1]
            or tokenizer.clean_up_tokenization is not methods[2]):
        return False

    backend = tokenizer.backend_tokenizer
    if type(backend) is not Tokenizer:
        return False
    decoder = backend.decoder
    if decoder is None:
        return True
    try:
        # Rust decoders serialize; Python custom decoders, including those
        # nested in Sequence, fail. Serialize only this tiny configuration,
        # never the vocabulary, and recheck mutations on each stream request.
        Decoder.__getstate__(decoder)
    except Exception:
        return False
    return True


def _generation_control(tokenizer, prompt_tokens, stop, cancel_event=None,
                        *, allow_incremental=False):
    """Request-local stop conditions shared by HF and native generation."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    stops = [stop] if isinstance(stop, str) else list(stop or [])
    stops = [item for item in stops if item]

    class RequestControl(StoppingCriteria):
        checker = None

        def __call__(self, input_ids, scores, **kwargs):
            import torch

            if cancel_event is not None and cancel_event.is_set():
                return torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            if not stops:
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            from .GenerationStop import StopStringRows

            if self.checker is None or len(self.checker.rows) != input_ids.shape[0]:
                self.checker = StopStringRows(
                    tokenizer, prompt_tokens, [stops] * input_ids.shape[0],
                    allow_incremental=allow_incremental,
                )
            return self.checker(input_ids)

    return StoppingCriteriaList([RequestControl()])


@dataclass
class InferenceServerSettings:
    model_path: str = ""
    model_name: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = "none"
    resolved_quantization: Optional[str] = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    max_concurrency: int = 1
    max_input_tokens: int = 4096
    default_max_new_tokens: int = 256
    max_new_tokens: int = 512
    api_key: Optional[str] = None
    merge_adapter: bool = True
    enable_thinking: Optional[bool] = None
    web_ui: bool = True
    chat_history: bool = True
    ui_title: str = "SaddleLLM 对话"
    batch_size: int = 1
    batch_wait_ms: float = 5.0
    max_pending_requests: int = 64
    batch_token_budget: Optional[int] = None
    max_pending_tokens: Optional[int] = None
    batch_length_bucket_size: int = 0
    batch_starvation_ms: float = 1000.0
    batch_scheduling_policy: str = "shortest_work"
    fuse_projections: bool = True
    prefix_cache_mb: int = 0
    prefix_cache_partial: bool = False
    prefix_cache_min_tokens: int = 0
    prefix_cache_min_store_tokens: int = 0
    backend: str = "local"
    require_high_throughput: bool = False
    backend_selection: Optional[Dict[str, Any]] = field(default=None, repr=False)
    truncate_input: bool = False
    vllm: VLLMServingConfig = field(default_factory=VLLMServingConfig)
    native_attention_backend: Optional[str] = None
    cpu_prepack_linear: bool = False
    native_stream_batching: bool = False
    native_decode_mode: str = "compatible"
    native_fused_kernels: bool = False
    native_prefill_chunk_size: int = 0
    native_max_step_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if (isinstance(self.native_prefill_chunk_size, bool)
                or not isinstance(self.native_prefill_chunk_size, int) or self.native_prefill_chunk_size < 0):
            raise ValueError("native_prefill_chunk_size must be a non-negative integer.")
        if self.native_max_step_tokens is not None and (
            isinstance(self.native_max_step_tokens, bool) or not isinstance(self.native_max_step_tokens, int)
            or self.native_max_step_tokens < 1
        ):
            raise ValueError("native_max_step_tokens must be a positive integer or None.")
        if self.native_max_step_tokens is not None and not self.native_prefill_chunk_size:
            raise ValueError("native_max_step_tokens requires native_prefill_chunk_size.")
        if self.native_prefill_chunk_size:
            if not self.native_stream_batching or self.backend != "native" or self.prefix_cache_mb:
                raise ValueError("Chunked prefill requires native_stream_batching, backend=native and prefix cache disabled.")
            if self.native_max_step_tokens is not None and self.native_max_step_tokens < self.batch_size:
                raise ValueError("native_max_step_tokens must cover batch_size decode tokens.")
        if not isinstance(self.native_fused_kernels, bool):
            raise ValueError("native_fused_kernels must be boolean.")
        if self.native_fused_kernels and (
            self.backend not in {"native", "auto"} or self.device not in {"cuda", "auto"}
            or self.quantization != "none"
        ):
            raise ValueError("native_fused_kernels requires the native CUDA backend with float weights.")
        if not isinstance(self.chat_history, bool):
            raise ValueError("chat_history must be a boolean")
        if self.quantization == "int8" and (self.backend not in {"native", "auto"} or self.device not in {"auto", "cuda"}):
            raise ValueError("Saddle W8A16 int8 requires backend=native and CUDA.")
        if self.quantization == "int8" and self.dtype == "float32":
            raise ValueError("Saddle W8A16 int8 compute requires float16 or bfloat16.")
        if not isinstance(self.cpu_prepack_linear, bool):
            raise ValueError("cpu_prepack_linear must be boolean.")
        if self.cpu_prepack_linear and (self.backend not in {"native", "auto"} or self.device != "cpu"):
            raise ValueError("CPU prepacking requires backend=native and device=cpu.")
        if self.backend not in {"auto", "local", "native", "vllm"}:
            raise ValueError("backend must be auto, local, native or vllm.")
        for name in ("require_high_throughput", "truncate_input"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool):
            raise ValueError("enable_thinking must be boolean or None (model template default).")
        if self.native_attention_backend not in (None, "sdpa", "sdpa_reference", "eager"):
            raise ValueError("native_attention_backend must be sdpa, sdpa_reference, eager or None.")
        if self.native_attention_backend is not None and self.backend not in {"native", "auto"}:
            raise ValueError("native_attention_backend requires backend=native.")
        if self.backend == "native":
            if self.quantization not in {"none", "int8"}:
                raise ValueError("The native backend accepts float weights or optional Saddle int8, not bitsandbytes.")
            if self.trust_remote_code:
                raise ValueError("The native backend does not execute remote modeling code.")
        if isinstance(self.vllm, dict):
            self.vllm = VLLMServingConfig(**self.vllm)
        if not isinstance(self.vllm, VLLMServingConfig):
            raise TypeError("vllm must be VLLMServingConfig or a settings mapping.")
        self.vllm.validate()
        if self.backend == "vllm":
            if self.device not in {"auto", "cuda"}:
                raise ValueError("The vLLM backend currently requires Linux CUDA; use backend=local for CPU/MPS.")
            if self.quantization != "none":
                raise ValueError("4bit/8bit bitsandbytes loading belongs to backend=local; vLLM loads its supported checkpoint formats.")
            if self.max_concurrency != 1:
                raise ValueError("For vLLM use vllm.max_num_seqs / --max-num-seqs, not --max-concurrency.")
            if self.batch_size != 1 or self.batch_token_budget is not None or self.batch_length_bucket_size:
                raise ValueError("Native microbatch options do not apply to vLLM; use --max-num-seqs and --max-num-batched-tokens.")
            if self.prefix_cache_mb or self.prefix_cache_partial:
                raise ValueError("Native prefix snapshots do not apply to vLLM; use --vllm-prefix-caching.")
            if self.truncate_input:
                raise ValueError("vLLM rejects overlong input; truncate_input belongs to the local/native route.")
        if not isinstance(self.fuse_projections, bool):
            raise ValueError("fuse_projections must be a boolean.")
        if (isinstance(self.prefix_cache_mb, bool)
                or not isinstance(self.prefix_cache_mb, int) or self.prefix_cache_mb < 0):
            raise ValueError("prefix_cache_mb must be a non-negative integer.")
        if not isinstance(self.prefix_cache_partial, bool):
            raise ValueError("prefix_cache_partial must be a boolean.")
        for name in ("prefix_cache_min_tokens", "prefix_cache_min_store_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
            if value and not self.prefix_cache_mb:
                raise ValueError(f"{name} requires a positive prefix_cache_mb.")
        if self.prefix_cache_partial and not self.prefix_cache_mb:
            raise ValueError("prefix_cache_partial requires a positive prefix_cache_mb.")
        if self.quantization not in {"none", "4bit", "8bit", "int8"}:
            raise ValueError("quantization must be one of: none, 4bit, 8bit, int8.")
        for name in ("batch_size", "max_pending_requests", "max_concurrency", "max_input_tokens", "default_max_new_tokens", "max_new_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("batch_token_budget", "max_pending_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None.")
        if (isinstance(self.batch_length_bucket_size, bool)
                or not isinstance(self.batch_length_bucket_size, int) or self.batch_length_bucket_size < 0):
            raise ValueError("batch_length_bucket_size must be a non-negative integer.")
        if (isinstance(self.batch_starvation_ms, bool)
                or not isinstance(self.batch_starvation_ms, (int, float))
                or not math.isfinite(self.batch_starvation_ms) or self.batch_starvation_ms <= 0):
            raise ValueError("batch_starvation_ms must be positive and finite.")
        if not isinstance(self.batch_scheduling_policy, str) or self.batch_scheduling_policy not in {"fifo", "shortest_prompt", "shortest_work"}:
            raise ValueError("batch_scheduling_policy must be fifo, shortest_prompt or shortest_work.")
        if (isinstance(self.batch_wait_ms, bool) or not isinstance(self.batch_wait_ms, (int, float))
                or not math.isfinite(self.batch_wait_ms) or not 0 <= self.batch_wait_ms <= 1000):
            raise ValueError("batch_wait_ms must be between 0 and 1000.")
        if self.max_input_tokens < 1 or self.default_max_new_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("Token limits must be positive integers.")
        if self.default_max_new_tokens > self.max_new_tokens:
            raise ValueError("default_max_new_tokens cannot exceed max_new_tokens.")
        if not isinstance(self.native_stream_batching, bool):
            raise ValueError("native_stream_batching must be boolean.")
        if self.native_decode_mode not in ("compatible", "throughput"):
            raise ValueError("native_decode_mode must be compatible or throughput.")
        if self.native_decode_mode != "compatible" and not self.native_stream_batching:
            raise ValueError("native_decode_mode=throughput requires native_stream_batching.")
        if self.native_stream_batching:
            if self.backend != "native" or self.quantization != "none":
                raise ValueError("native_stream_batching requires explicit backend=native and unquantized weights.")
            if self.batch_size < 2 or self.max_concurrency < 2:
                raise ValueError("native_stream_batching requires batch_size >= 2 and max_concurrency >= 2.")
            if self.max_pending_tokens is not None or self.batch_length_bucket_size:
                raise ValueError("native_stream_batching does not support fixed-microbatch max_pending_tokens or length buckets; use max_pending_requests and per-request token limits.")

    @property
    def resolved_batch_token_budget(self) -> int:
        # A conservative default: one maximum-sized request worth of token slots.
        # Small requests can still batch; explicit budgets permit larger batches.
        return self.batch_token_budget or self.max_input_tokens + self.max_new_tokens


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None

    @field_validator("content", mode="before")
    @classmethod
    def text_parts(cls, value):
        if isinstance(value, list):
            if not value or any(not isinstance(part, dict) or set(part) != {"type", "text"}
                                or part["type"] != "text" or not isinstance(part["text"], str)
                                for part in value):
                raise ValueError("Only nonempty arrays of text content parts are supported.")
            return "".join(part["text"] for part in value)
        return value


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, strict=True,
                                      validation_alias=AliasChoices("max_tokens", "max_completion_tokens"))
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    n: int = Field(default=1, ge=1, le=1)
    repetition_penalty: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def compatible_token_limit(cls, values):
        if isinstance(values, dict) and "max_tokens" in values and "max_completion_tokens" in values:
            if values["max_tokens"] != values["max_completion_tokens"]:
                raise ValueError("Pass one consistent completion token limit.")
            values = {key: value for key, value in values.items() if key != "max_completion_tokens"}
        return values

    def template_messages(self):
        messages = [message.model_dump(exclude_none=True) for message in self.messages]
        # Text model templates commonly recognize system, not OpenAI's developer role.
        return [{**message, "role": "system"} if message["role"] == "developer" else message
                for message in messages]


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: Optional[str] = None
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    n: int = Field(default=1, ge=1, le=1)
    echo: bool = False
    repetition_penalty: float = Field(default=1.0, gt=0.0)


@dataclass
class _PreparedGeneration:
    encoded: Dict[str, Any]
    requested_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    stop: Optional[Union[str, List[str]]]

    @property
    def prompt_tokens(self) -> int:
        return int(self.encoded["input_ids"].shape[-1])

    @property
    def batch_key(self) -> tuple:
        # Equal lengths avoid extra padding, position changes and wasteful
        # long/short prompt combinations. Stops remain request-local.
        return (
            self.prompt_tokens, self.requested_tokens, self.temperature,
            self.top_p, self.repetition_penalty,
            tuple((key, tuple(value.shape[1:]), value.dtype)
                  for key, value in sorted(self.encoded.items())),
        )


@dataclass
class _PreparedRequest:
    request: Union[ChatCompletionRequest, CompletionRequest]
    generation: _PreparedGeneration


class _BatchGenerationControl:
    """Record exact row lengths before batch padding, including EOS tokens."""

    def __init__(self, tokenizer, prepared, eos_token_id, cancel_events, device,
                 *, allow_incremental=False, prompt_width=None):
        import torch

        self.tokenizer = tokenizer
        self.prompt_tokens = prepared[0].prompt_tokens if prompt_width is None else prompt_width
        self.stops = [
            [item for item in ([row.stop] if isinstance(row.stop, str) else row.stop or []) if item]
            for row in prepared
        ]
        from .GenerationStop import StopStringRows

        self.stop_checker = (StopStringRows(
            tokenizer, self.prompt_tokens, self.stops,
            allow_incremental=allow_incremental,
        ) if any(self.stops) else None)
        if isinstance(eos_token_id, torch.Tensor):
            eos_token_id = eos_token_id.flatten().tolist()
        self.eos_ids = (list(eos_token_id) if isinstance(eos_token_id, (list, tuple, set))
                        else [] if eos_token_id is None else [int(eos_token_id)])
        self.cancel_events = cancel_events
        self.finished = torch.zeros(len(prepared), dtype=torch.bool, device=device)
        self.lengths = torch.zeros(len(prepared), dtype=torch.long, device=device)

    def __call__(self, input_ids, scores):
        import torch

        stop = torch.tensor([event.is_set() for event in self.cancel_events],
                            dtype=torch.bool, device=input_ids.device)
        for token_id in self.eos_ids:
            stop |= input_ids[:, -1].eq(token_id)
        if self.stop_checker is not None:
            stop |= self.stop_checker(input_ids)
        newly_finished = stop & ~self.finished
        self.lengths = torch.where(newly_finished,
                                   input_ids.shape[1] - self.prompt_tokens, self.lengths)
        self.finished |= stop
        return self.finished


class LocalGenerationEngine:
    """Small synchronous generation core used behind the async HTTP layer."""

    def __init__(self, model: Any, tokenizer: Any, settings: InferenceServerSettings):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.model_name = settings.model_name or self._infer_model_name(settings.model_path)
        self.prefix_cache = None
        if settings.prefix_cache_mb and self.supports_step_generation:
            from ..models.NativePrefixCache import NativePrefixCache

            self.prefix_cache = NativePrefixCache(
                settings.prefix_cache_mb * 1024 * 1024,
                reuse_partial_prefixes=settings.prefix_cache_partial,
                min_prefix_tokens=max(16, settings.prefix_cache_min_tokens),
                min_store_tokens=settings.prefix_cache_min_store_tokens,
            )
        elif settings.prefix_cache_mb:
            logger.warning(
                "Prefix cache is only supported by native step generation; "
                "this model retains its original generation path."
            )

    @property
    def supports_batching(self) -> bool:
        from ..models.SaddleModeling import SaddleForCausalLM

        return isinstance(self.model, SaddleForCausalLM)

    @property
    def supports_step_generation(self) -> bool:
        from ..models.SaddleModeling import SaddleForCausalLM

        # Custom generate overrides retain their own generation semantics.
        return (self.supports_batching and
                getattr(self.model.generate, "__func__", None) is SaddleForCausalLM.generate)

    def can_batch(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> bool:
        # Sampling has request-specific RNG semantics; arbitrary HF/custom
        # models may have different stopping/output contracts. Keep them on
        # the established individual generation path in this first version.
        return (self.supports_batching
                and not request.stream and request.temperature == 0)

    def prepare_batch_request(self, request) -> _PreparedRequest:
        if not self.can_batch(request):
            raise ValueError("microbatching requires non-streaming native greedy generation")
        if isinstance(request, ChatCompletionRequest):
            prompt = self._chat_prompt(request.template_messages())
        else:
            prompt = request.prompt
        if not prompt:
            raise ValueError("prompt must not be empty.")
        prepared = self._prepare_generation(
            prompt, max_tokens=request.max_tokens, temperature=request.temperature,
            top_p=request.top_p, repetition_penalty=request.repetition_penalty, stop=request.stop,
        )
        if prepared.prompt_tokens < 1:
            raise ValueError("prompt must produce at least one token")
        return _PreparedRequest(request, prepared)

    def prepare_stream_request(self, request) -> _PreparedGeneration:
        """Validate/tokenize before SSE headers, then reuse the encoded prompt."""
        prompt = (self._chat_prompt(request.template_messages())
                  if isinstance(request, ChatCompletionRequest) else request.prompt)
        if not prompt:
            raise ValueError("prompt must not be empty.")
        return self._prepare_generation(
            prompt, max_tokens=request.max_tokens, temperature=request.temperature,
            top_p=request.top_p, repetition_penalty=request.repetition_penalty, stop=request.stop,
        )

    def batch_key(self, item: _PreparedRequest) -> tuple:
        """Opt-in neighboring lengths; unusual/custom generation stays exact-length."""
        from ..models.SaddleModeling import SaddleForCausalLM

        row = item.generation
        width = self.settings.batch_length_bucket_size
        if (not width or not self.supports_step_generation
                or getattr(self.model.forward, "__func__", None) is not SaddleForCausalLM.forward
                or row.repetition_penalty != 1.0
                or set(row.encoded) - {"input_ids", "attention_mask"}
                or any(value.ndim != 2 or value.shape != (1, row.prompt_tokens)
                       for value in row.encoded.values())):
            return row.batch_key
        mask = row.encoded.get("attention_mask")
        if mask is not None and not bool((mask == 1).all()):
            return row.batch_key
        return ("length_bucket", (row.prompt_tokens - 1) // width, row.requested_tokens,
                row.temperature, row.top_p, row.repetition_penalty,
                row.encoded["input_ids"].dtype, None if mask is None else mask.dtype)

    @staticmethod
    def _batch_token_slots(rows) -> int:
        return len(rows) * (max(row.prompt_tokens for row in rows)
                            + max(row.requested_tokens for row in rows))

    def generate_batch(self, requests, cancel_events=None) -> List[Dict[str, Any]]:
        """Execute one compatible prepared batch; the scheduler owns grouping."""
        import torch

        if not requests:
            return []
        rows = [item.generation for item in requests]
        first = rows[0]
        key = self.batch_key(requests[0])
        if any(self.batch_key(item) != key for item in requests):
            raise ValueError("batch requests must have matching input lengths and generation settings")
        prompt_width = max(row.prompt_tokens for row in rows)
        if self._batch_token_slots(rows) > self.settings.resolved_batch_token_budget:
            raise ValueError("Batch exceeds the inference batch token budget")
        cancel_events = cancel_events or [threading.Event() for _ in rows]
        if len(cancel_events) != len(rows):
            raise ValueError("cancel events must match the batch size")
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if key[0] == "length_bucket":
            from torch.nn import functional as F

            pad_token_id = self.model.config.pad_token_id if pad_token_id is None else pad_token_id
            encoded = {
                "input_ids": torch.cat([F.pad(row.encoded["input_ids"], (prompt_width - row.prompt_tokens, 0),
                                               value=pad_token_id) for row in rows], dim=0),
                "attention_mask": torch.cat([
                    F.pad(row.encoded.get("attention_mask", torch.ones_like(row.encoded["input_ids"])),
                          (prompt_width - row.prompt_tokens, 0), value=0) for row in rows
                ], dim=0),
            }
            encoded = {name: value.to(self._input_device()) for name, value in encoded.items()}
        else:
            encoded = {name: torch.cat([row.encoded[name] for row in rows], dim=0).to(self._input_device())
                       for name in first.encoded}
        control = _BatchGenerationControl(
            self.tokenizer, rows, getattr(self.model.config, "eos_token_id", None),
            cancel_events, encoded["input_ids"].device,
            allow_incremental=self.supports_step_generation,
            prompt_width=prompt_width,
        )
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, max_new_tokens=first.requested_tokens, do_sample=False,
                repetition_penalty=first.repetition_penalty, pad_token_id=pad_token_id,
                stopping_criteria=control,
            )
        sequences = getattr(output, "sequences", output)
        generated = sequences[:, prompt_width:].tolist()
        lengths = control.lengths.tolist()
        responses = []
        for index, (item, token_ids) in enumerate(zip(requests, generated)):
            count = lengths[index] or len(token_ids)
            text = self.tokenizer.decode(token_ids[:count], skip_special_tokens=True)
            text, stopped = self._apply_stop(text, item.generation.stop)
            reason = ("stop" if lengths[index] or stopped or count < first.requested_tokens
                      else "length")
            result = (text, item.generation.prompt_tokens, count, reason)
            if isinstance(item.request, ChatCompletionRequest):
                responses.append(self._chat_response(result))
            else:
                responses.append(self._completion_response(item.request, result))
        return responses

    def complete_batch(self, requests: Sequence[CompletionRequest]) -> List[Dict[str, Any]]:
        """Synchronous import API; incompatible requests use the individual path."""
        return self._request_batch(requests)

    def chat_batch(self, requests: Sequence[ChatCompletionRequest]) -> List[Dict[str, Any]]:
        return self._request_batch(requests)

    def _request_batch(self, requests) -> List[Dict[str, Any]]:
        responses = [None] * len(requests)
        groups = {}
        for index, request in enumerate(requests):
            if request.stream:
                raise ValueError("the synchronous batch API does not support stream=true")
            if not self.can_batch(request):
                responses[index] = (self.chat(request) if isinstance(request, ChatCompletionRequest)
                                    else self.complete(request))
                continue
            item = self.prepare_batch_request(request)
            if item.generation.prompt_tokens + item.generation.requested_tokens > self.settings.resolved_batch_token_budget:
                raise ValueError("Request exceeds the inference batch token budget")
            groups.setdefault(self.batch_key(item), []).append((index, item))
        for group in groups.values():
            remaining = list(group)
            while remaining:
                chunk = [remaining.pop(0)]
                while remaining and len(chunk) < self.settings.batch_size:
                    rows = [item.generation for _, item in chunk + [remaining[0]]]
                    if self._batch_token_slots(rows) > self.settings.resolved_batch_token_budget:
                        break
                    chunk.append(remaining.pop(0))
                for (index, _), response in zip(chunk, self.generate_batch([item for _, item in chunk])):
                    responses[index] = response
        return responses

    def chat(self, request: ChatCompletionRequest, *, cancel_event=None, step_batcher=None) -> Dict[str, Any]:
        messages = request.template_messages()
        if not messages:
            raise ValueError("messages must contain at least one item.")
        prompt = self._chat_prompt(messages)
        result = self._generate(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop,
            cancel_event=cancel_event,
            step_batcher=step_batcher,
        )
        return self._chat_response(result)

    def _chat_response(self, result) -> Dict[str, Any]:
        text, prompt_tokens, completion_tokens, finish_reason = result
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

    def stream_chat(
        self,
        request: ChatCompletionRequest,
        stream_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Yield decoded chat text as it is generated."""
        messages = request.template_messages()
        if not messages:
            raise ValueError("messages must contain at least one item.")
        prompt = self._chat_prompt(messages)
        yield from self._stream_generate(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop,
            stream_state=stream_state,
        )

    def complete(self, request: CompletionRequest, *, cancel_event=None, step_batcher=None) -> Dict[str, Any]:
        if not request.prompt:
            raise ValueError("prompt must not be empty.")
        result = self._generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop,
            cancel_event=cancel_event,
            step_batcher=step_batcher,
        )
        return self._completion_response(request, result)

    def stream_complete(self, request: CompletionRequest,
                        stream_state: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        """Stream completion text; echo is prompt text, not generated token usage."""
        if not request.prompt:
            raise ValueError("prompt must not be empty.")
        if request.echo:
            yield request.prompt
        yield from self._stream_generate(
            request.prompt, max_tokens=request.max_tokens, temperature=request.temperature,
            top_p=request.top_p, repetition_penalty=request.repetition_penalty,
            stop=request.stop, stream_state=stream_state,
        )

    def _completion_response(self, request: CompletionRequest, result) -> Dict[str, Any]:
        text, prompt_tokens, completion_tokens, finish_reason = result
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
            template_options = {"tokenize": False, "add_generation_prompt": True}
            if self.settings.enable_thinking is not None:
                template_options["enable_thinking"] = self.settings.enable_thinking
            try:
                return apply_template(list(messages), **template_options)
            except TypeError:
                # Older/custom templates may not accept Qwen's optional switch.
                try:
                    return apply_template(
                        list(messages),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except (ValueError, TypeError, AttributeError) as exc:
                    raise ValueError("The model has no usable chat template; use a matching chat tokenizer or /v1/completions.") from exc
            except (ValueError, AttributeError) as exc:
                raise ValueError("The model has no usable chat template; use a matching chat tokenizer or /v1/completions.") from exc
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _prepare_generation(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop: Optional[Union[str, List[str]]],
    ) -> _PreparedGeneration:
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
            truncation=self.settings.truncate_input,
            max_length=max(1, input_limit),
        )
        if encoded["input_ids"].shape[-1] > input_limit:
            raise ValueError("Input plus output budget exceeds the configured token/context limit; shorten the request. No silent truncation is performed.")
        if encoded["input_ids"].shape[-1] < 1:
            raise ValueError("prompt must produce at least one token.")
        return _PreparedGeneration(dict(encoded), requested_tokens, temperature,
                                   top_p, repetition_penalty, stop)

    def _generate(self, prompt: str, *, cancel_event=None, step_batcher=None, **kwargs) -> Tuple[str, int, int, str]:
        import torch

        if step_batcher is not None:
            if not self.supports_step_generation:
                raise ValueError("step batching requires native generation sessions")
            cancel_event = cancel_event or threading.Event()
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("Generation was cancelled before execution")
        prepared = self._prepare_generation(prompt, **kwargs)
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("Generation was cancelled during preparation")
        requested_tokens = prepared.requested_tokens
        prompt_tokens = prepared.prompt_tokens
        temperature, top_p = prepared.temperature, prepared.top_p
        repetition_penalty, stop = prepared.repetition_penalty, prepared.stop
        target_device = self._input_device()
        encoded = {
            key: value.to(target_device) if hasattr(value, "to") else value
            for key, value in prepared.encoded.items()
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
        native_cancel = cancel_event is not None and self.supports_step_generation
        if stop or (cancel_event is not None and not native_cancel):
            generate_kwargs["stopping_criteria"] = _generation_control(
                self.tokenizer, prompt_tokens, stop,
                cancel_event=None if native_cancel else cancel_event,
                allow_incremental=self.supports_step_generation,
            )
        if self.prefix_cache is not None:
            generate_kwargs["prefix_cache"] = self.prefix_cache
        with torch.inference_mode():
            if native_cancel:
                # Same native generation loop, with a host-side cancellation
                # check: no additional device tensor or GPU sync per token.
                with self.model.create_generation_session(**generate_kwargs) as session:
                    registered = False
                    try:
                        if step_batcher is not None:
                            step_batcher.register(session)
                            registered = True
                        while not session.done and not cancel_event.is_set():
                            if step_batcher is None:
                                session.step()
                            else:
                                step_batcher.step(session, cancel_event)
                        output = session.sequences()
                    finally:
                        if registered:
                            step_batcher.unregister(session)
            else:
                output = self.model.generate(**generate_kwargs)
        sequences = getattr(output, "sequences", output)
        generated_ids = sequences[0, prompt_tokens:]
        completion_tokens = int(generated_ids.shape[-1])
        token_values = generated_ids.tolist() if hasattr(generated_ids, "tolist") else generated_ids
        text = self.tokenizer.decode(token_values, skip_special_tokens=True)
        text, stopped = self._apply_stop(text, stop)
        finish_reason = (
            "stop" if stopped or completion_tokens < requested_tokens
            or self._ends_with_eos(token_values) else "length"
        )
        return text, prompt_tokens, completion_tokens, finish_reason

    def _stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop: Optional[Union[str, List[str]]],
        stream_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        try:
            import torch
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            raise RuntimeError("Streaming inference requires torch and transformers.") from exc

        prepared = (stream_state or {}).get("prepared_generation")
        if prepared is None:
            prepared = self._prepare_generation(
                prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty, stop=stop,
            )
        requested_tokens, prompt_tokens = prepared.requested_tokens, prepared.prompt_tokens
        target_device = self._input_device()
        encoded = {
            key: value.to(target_device) if hasattr(value, "to") else value
            for key, value in prepared.encoded.items()
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
        if self.supports_step_generation:
            if self.prefix_cache is not None:
                generate_kwargs["prefix_cache"] = self.prefix_cache
            yield from self._stream_native(
                generate_kwargs, prompt_tokens, requested_tokens, stop, stream_state,
            )
            return
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.2,
        )
        generate_kwargs["streamer"] = streamer
        errors: List[BaseException] = []
        generation_result: Dict[str, Any] = {}
        cancel_event = (stream_state or {}).get("cancel_event") or threading.Event()
        worker_done = threading.Event()
        if stream_state is not None:
            stream_state["worker_done"] = worker_done
        generate_kwargs["stopping_criteria"] = _generation_control(
            self.tokenizer, prompt_tokens, stop, cancel_event
        )

        def generate() -> None:
            try:
                with torch.inference_mode():
                    output = self.model.generate(**generate_kwargs)
                sequences = getattr(output, "sequences", output)
                generated_ids = sequences[0, prompt_tokens:]
                generation_result["completion_tokens"] = int(generated_ids.shape[-1])
                generation_result["ended_with_eos"] = self._ends_with_eos(
                    generated_ids[-1:].tolist()
                )
            except BaseException as exc:  # surfaced to the request thread below
                errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)
            finally:
                worker_done.set()

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        from .StreamingText import StreamingTextBuffer

        text_buffer = StreamingTextBuffer(stop)
        iterator = iter(streamer)
        try:
            while True:
                try:
                    piece = next(iterator)
                except queue.Empty:
                    if not worker.is_alive():
                        break
                    continue
                except StopIteration:
                    break
                visible = text_buffer.push(piece)
                if text_buffer.stopped:
                    cancel_event.set()
                if visible:
                    yield visible
        finally:
            cancel_event.set()
            worker.join()
        if errors:
            raise RuntimeError(f"Model generation failed: {errors[0]}") from errors[0]
        tail = text_buffer.finish()
        if tail:
            yield tail
        completion_tokens = int(generation_result.get("completion_tokens", 0))
        if stream_state is not None:
            stream_state.update(
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": (
                        "stop"
                        if text_buffer.stopped or completion_tokens < requested_tokens
                        or generation_result.get("ended_with_eos", False)
                        else "length"
                    ),
                }
            )

    def _stream_native(self, generate_kwargs, prompt_tokens, requested_tokens, stop,
                       stream_state) -> Iterator[str]:
        """Pull token steps directly; no per-request background model thread."""
        import torch
        from transformers import TextStreamer

        from .StreamingText import StreamingTextBuffer

        chunks = deque()

        class TextChunks(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                chunks.append(text)

        from ..models.NativeGeneration import _register_batch_streamer_type

        if _can_batch_tokenizer_delivery(self.tokenizer):
            _register_batch_streamer_type(TextChunks)
        streamer = TextChunks(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs["streamer"] = streamer
        if stop:
            generate_kwargs["stopping_criteria"] = _generation_control(
                self.tokenizer, prompt_tokens, stop,
                allow_incremental=True,
            )
        cancel_event = (stream_state or {}).get("cancel_event") or threading.Event()
        worker_done = threading.Event()
        if stream_state is not None:
            stream_state["worker_done"] = worker_done
        text_buffer = StreamingTextBuffer(stop)
        step_batcher = (stream_state or {}).get("native_step_batcher")
        registered = False
        session = None
        try:
            if cancel_event.is_set():
                return
            with torch.inference_mode():
                session = self.model.create_generation_session(**generate_kwargs)
            if step_batcher is not None:
                step_batcher.register(session)
                registered = True
            while not session.done and not cancel_event.is_set():
                if step_batcher is None:
                    session.step()
                else:
                    step_batcher.step(session, cancel_event)
                emitted_text = False
                while chunks:
                    visible = text_buffer.push(chunks.popleft())
                    if text_buffer.stopped:
                        cancel_event.set()
                    if visible:
                        emitted_text = True
                        yield visible
                    if cancel_event.is_set():
                        break
                # Internal scheduling boundary, never an SSE text event. Yield
                # even when tokenization has not produced a whole word yet so
                # one stream cannot monopolize the model until its next chunk.
                if not emitted_text and (stream_state or {}).get("cooperative_ticks"):
                    yield ""
            if not cancel_event.is_set():
                tail = text_buffer.finish()
                if tail:
                    yield tail
        except Exception as exc:
            raise RuntimeError(f"Model generation failed: {exc}") from exc
        finally:
            completion_tokens = session.generated_tokens if session is not None else 0
            stopped = (text_buffer.stopped or cancel_event.is_set() or
                       (session is not None and session.finished_count > 0))
            cancel_event.set()
            try:
                if session is not None:
                    try:
                        if registered:
                            step_batcher.unregister(session)
                    finally:
                        session.close()
            finally:
                if stream_state is not None:
                    stream_state.update({
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "finish_reason": (
                            "stop" if stopped or completion_tokens < requested_tokens else "length"
                        ),
                    })
                worker_done.set()

    def _ends_with_eos(self, token_values) -> bool:
        if not token_values:
            return False
        config = getattr(self.model, "generation_config", None)
        if config is None:
            config = getattr(self.model, "config", None)
        eos_ids = getattr(config, "eos_token_id", None)
        if hasattr(eos_ids, "flatten"):
            eos_ids = eos_ids.flatten().tolist()
        if not isinstance(eos_ids, (list, tuple, set)):
            eos_ids = [] if eos_ids is None else [eos_ids]
        return token_values[-1] in eos_ids

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
        import anyio
        from fastapi import Depends, FastAPI, Header, HTTPException, Response
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise ImportError("FastAPI is required to create the inference service.") from exc

    settings = settings or InferenceServerSettings()
    from .ServingPlan import resolve_serving_settings

    if settings.backend_selection is None or settings.backend == "auto":
        settings = resolve_serving_settings(settings, injected_model=model is not None)
    elif settings.require_high_throughput and settings.backend != "vllm":
        raise ValueError("High-throughput continuous batching requires the vLLM route.")
    if settings.backend == "vllm":
        if model is not None or tokenizer is not None:
            raise ValueError("The vLLM backend owns model/tokenizer loading; do not supply a local model.")
        from .VLLMServing import create_vllm_inference_app

        return create_vllm_inference_app(settings)
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must be provided together.")
    if model is not None and settings.cpu_prepack_linear:
        raise ValueError("CPU prepacking is a loading option; optimize an injected model explicitly.")
    if model is not None and settings.native_fused_kernels:
        raise ValueError("Fused kernels are a loading option; optimize an injected model explicitly.")
    if model is not None and settings.quantization == "int8":
        raise ValueError("Native INT8 is a loading option; configure an injected model directly.")
    if model is not None and settings.native_attention_backend is not None:
        raise ValueError("Configure attention on an injected model directly; native_attention_backend is a loading option.")
    release_metadata = _load_release_metadata(settings.model_path)
    if model is None:
        if not settings.model_path:
            raise ValueError("settings.model_path is required when no model is supplied.")
        model, tokenizer = load_inference_model(settings)
    from .ServingPerformance import finalize_serving_performance

    settings = finalize_serving_performance(settings, model)
    performance_plan = (settings.backend_selection or {}).get("performance", {})
    if performance_plan.get("state") == "fallback":
        logger.warning("Serving performance preset %s not applied: %s",
                       performance_plan["requested"], performance_plan["reason"])
    engine = LocalGenerationEngine(model, tokenizer, settings)
    int8_manifest = getattr(model, "_saddle_cuda_int8", None)
    if int8_manifest is not None and int8_manifest.get("status") != "ready":
        raise ValueError("Native INT8 preparation failed or is incomplete; discard this model.")
    if settings.backend == "native" and not engine.supports_step_generation:
        raise ValueError("backend=native requires a Saddle model with native step generation; no HF fallback.")
    stream_batcher = None
    if settings.native_stream_batching:
        import torch

        from .NativeStreamBatcher import NativeStreamBatcher

        weight = model.embed_tokens.weight
        if (int8_manifest is not None or weight.dtype not in {torch.float32, torch.bfloat16}
                or (weight.dtype == torch.bfloat16 and weight.device.type != "cuda")):
            raise ValueError("native_stream_batching supports FP32 or CUDA BF16 float weights only.")
        stream_batcher = NativeStreamBatcher(
            model, max_batch_size=settings.batch_size, max_resident=settings.max_concurrency,
            max_batch_tokens=settings.resolved_batch_token_budget, wait_ms=settings.batch_wait_ms,
            prefill_chunk_size=settings.native_prefill_chunk_size, max_step_tokens=settings.native_max_step_tokens,
            decode_mode=settings.native_decode_mode,
        )
    semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))
    # Preserve the validated request-permit policy by default. Private benchmark
    # hooks can select a bounded executor or fair token pulls; neither has passed
    # the performance gates. Neither is fused batching or paged KV management.
    cooperative_streams = stream_batcher is None and _cooperative_native_streaming(engine, settings)
    stream_step_lock = asyncio.Lock() if cooperative_streams else None
    stream_workers = (None if stream_batcher is not None else
                      1 if cooperative_streams else _native_stream_worker_count(engine, settings))
    stream_executor = (ThreadPoolExecutor(max_workers=stream_workers, thread_name_prefix="saddle-native-stream")
                       if stream_workers is not None else None)
    from .InferenceBatcher import InferenceBatcher, InferenceQueueFull

    batcher = (InferenceBatcher(
        engine.generate_batch, semaphore, max_batch_size=settings.batch_size,
        wait_ms=settings.batch_wait_ms, max_pending=settings.max_pending_requests,
        max_batch_tokens=settings.resolved_batch_token_budget,
        max_pending_tokens=settings.max_pending_tokens,
        scheduling_policy=settings.batch_scheduling_policy, starvation_ms=settings.batch_starvation_ms,
    ) if (stream_batcher is None and (
          settings.batch_size > 1 or settings.batch_token_budget is not None
          or settings.max_pending_tokens is not None or settings.batch_length_bucket_size > 0)) else None)

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            try:
                if batcher is not None:
                    await batcher.close()
            finally:
                try:
                    if stream_executor is not None:
                        await asyncio.to_thread(stream_executor.shutdown, wait=True)
                finally:
                    try:
                        if stream_batcher is not None:
                            await asyncio.to_thread(stream_batcher.close)
                    finally:
                        if engine.prefix_cache is not None:
                            engine.prefix_cache.close()

    app = FastAPI(title="SaddleLLM OpenAI-Compatible API", version="1.0.0", lifespan=lifespan)
    from .OpenAIProtocol import API_CAPABILITIES, install_openai_error_handlers

    install_openai_error_handlers(app)
    from .InferenceAdmission import LocalAdmissionMiddleware, LocalHTTPAdmission

    admission = LocalHTTPAdmission(settings.max_concurrency + settings.max_pending_requests)
    app.add_middleware(LocalAdmissionMiddleware, admission=admission)
    app.state.admission = admission
    app.state.settings = settings
    app.state.engine = engine
    app.state.batcher = batcher
    app.state.native_stream_batcher = stream_batcher
    app.state.ready = False

    from .InferenceLifecycle import drain_task, run_until_disconnect

    async def nonstream_work(request, cancel_event):
        try:
            if batcher is not None and engine.can_batch(request):
                preparation = asyncio.create_task(asyncio.to_thread(engine.prepare_batch_request, request))
                try:
                    prepared = await asyncio.shield(preparation)
                except asyncio.CancelledError:
                    await drain_task(preparation)
                    raise
                return await batcher.submit(prepared, engine.batch_key(prepared),
                                            prompt_tokens=prepared.generation.prompt_tokens,
                                            max_new_tokens=prepared.generation.requested_tokens)
            async with semaphore:
                method = engine.chat if isinstance(request, ChatCompletionRequest) else engine.complete
                options = {"step_batcher": stream_batcher} if stream_batcher is not None else {}
                thread = asyncio.create_task(asyncio.to_thread(method, request, cancel_event=cancel_event, **options))
                try:
                    return await asyncio.shield(thread)
                except asyncio.CancelledError:
                    # to_thread keeps running after an HTTP task is cancelled.
                    # Keep its model permit until the actual invocation ends.
                    cancel_event.set()
                    await drain_task(thread)
                    raise
        except InferenceQueueFull as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def nonstream_response(request, http_request):
        cancel_event = threading.Event()
        response = await run_until_disconnect(
            http_request.receive, nonstream_work(request, cancel_event), cancel_event,
        )
        # The peer has already left; do not manufacture a successful model
        # completion or send internal cancellation details on that transport.
        return Response(status_code=204) if response is None else response

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    async def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        if not settings.api_key:
            return
        scheme, _, credential = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(credential, settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        from ..evaluation.ReleaseGate import release_gate_accepted

        gate = release_metadata.get("gate") if release_metadata else None
        cpu_prepack = None
        native_int8 = None
        if engine.supports_step_generation:
            from ..models.CPULinear import cpu_prepack_status

            cpu_prepack = cpu_prepack_status(engine.model)
            if getattr(engine.model, "_saddle_cuda_int8", None) is not None:
                from ..models.CUDAInt8 import cuda_int8_status

                native_int8 = cuda_int8_status(engine.model)
        return {
            "status": "ok",
            "ready": app.state.ready,
            "backend": settings.backend,
            "backend_selection": settings.backend_selection,
            "api_capabilities": API_CAPABILITIES,
            "thinking_policy": ("model_default" if settings.enable_thinking is None
                                else "enabled" if settings.enable_thinking else "disabled"),
            "input_policy": "truncate" if settings.truncate_input else "reject_overlong",
            "admission": admission.snapshot(),
            "execution_backend": "native" if engine.supports_step_generation else "hf_or_custom",
            "native_source": getattr(engine.model, "_saddle_native_source", None),
            "native_attention_backend": (getattr(engine.model.config, "attention_backend", None)
                                         if engine.supports_step_generation else None),
            "device": str(engine._input_device()),
            "model": engine.model_name,
            "release_id": release_metadata.get("release_id") if release_metadata else None,
            "gate_accepted": (
                release_gate_accepted(gate) if isinstance(gate, dict) else None
            ),
            "streaming": True,
            "streaming_backend": (
                "native_step" if engine.supports_step_generation else "generate_worker"
            ),
            "stream_scheduling": (
                "native_token_budget" if settings.native_prefill_chunk_size else
                "native_batch_decode" if stream_batcher is not None else
                "cooperative_token" if cooperative_streams else
                "bounded_workers" if stream_executor is not None else "request_permit"
            ),
            "stream_resident_limit": settings.max_concurrency,
            "stream_worker_limit": 1 if stream_batcher is not None else stream_workers,
            "native_stream_batching": (stream_batcher.snapshot() if stream_batcher is not None
                                       else {"enabled": False}),
            "web_ui": settings.web_ui,
            "chat_history_enabled": settings.web_ui and settings.chat_history,
            "requires_auth": bool(settings.api_key),
            "quantization": (native_int8["method"] if native_int8 else
                             settings.resolved_quantization or settings.quantization),
            "native_projection_packing": getattr(
                engine.model, "_saddle_inference_optimization", None
            ),
            "cpu_prepack": cpu_prepack,
            "native_int8": native_int8,
            "prefix_cache": (engine.prefix_cache.snapshot() if engine.prefix_cache is not None
                             else {"enabled": False, "configured_mb": settings.prefix_cache_mb}),
            "batching": {
                "enabled": batcher is not None and engine.supports_batching,
                "mode": "native_greedy_nonstream",
                "length_bucket_size": settings.batch_length_bucket_size,
                **(batcher.snapshot() if batcher is not None else {"max_batch_size": 1}),
            },
        }

    @app.get("/livez")
    async def liveness():
        # No GPU forward, dependency probe or queue-empty requirement.
        return {"status": "alive"}

    @app.get("/readyz")
    async def readiness():
        if (not app.state.ready or (batcher is not None and batcher.closed)
                or (stream_batcher is not None and stream_batcher.closed)):
            raise HTTPException(status_code=503, detail="Inference service is not ready.")
        return {"status": "ready"}

    if settings.web_ui:
        from .ChatWebRoutes import install_chat_web_routes

        install_chat_web_routes(app, settings, engine.model_name, authorize)

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

    def stream_response(request, prepared_generation):
        is_chat = isinstance(request, ChatCompletionRequest)
        include_usage = bool(request.stream_options and request.stream_options.include_usage)
        if not is_chat and not request.prompt:
            raise HTTPException(status_code=400, detail="prompt must not be empty.")
        # Both endpoints use the same cancellation/permit ownership path.
        if request.stream:
            completion_id = f"{'chatcmpl' if is_chat else 'cmpl'}-{uuid.uuid4().hex}"
            created = int(time.time())

            def event(payload: Dict[str, Any]) -> str:
                return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

            def chunk(piece=None, *, finish_reason=None, role=False):
                choice = {"index": 0, "finish_reason": finish_reason}
                if is_chat:
                    choice["delta"] = ({"role": "assistant"} if role else
                                       {} if piece is None else {"content": piece})
                else:
                    choice["text"] = piece or ""
                payload = {
                    "id": completion_id, "object": "chat.completion.chunk" if is_chat else "text_completion",
                    "created": created, "model": engine.model_name, "choices": [choice],
                }
                if include_usage:
                    payload["usage"] = None
                return payload

            async def generate_events():
                async with semaphore:
                    if is_chat:
                        yield event(chunk(role=True))
                    cancel_event = threading.Event()
                    stream_state: Dict[str, Any] = {"cancel_event": cancel_event,
                                                  "prepared_generation": prepared_generation,
                                                  "cooperative_ticks": cooperative_streams}
                    if stream_batcher is not None:
                        stream_state["native_step_batcher"] = stream_batcher
                    method = engine.stream_chat if is_chat else engine.stream_complete
                    iterator = iter(method(request, stream_state))
                    piece_task = None

                    def next_piece():
                        try:
                            return False, next(iterator)
                        except StopIteration:
                            return True, None

                    async def pull_piece():
                        context = contextvars.copy_context()
                        if stream_step_lock is None:
                            if stream_executor is None:
                                return await asyncio.to_thread(next_piece)
                            return await asyncio.get_running_loop().run_in_executor(stream_executor, context.run, next_piece)
                        async with stream_step_lock:
                            # Keep CUDA/BLAS thread-local state on one worker;
                            # arbitrary executor hopping per token is costly.
                            return await asyncio.get_running_loop().run_in_executor(stream_executor, context.run, next_piece)

                    try:
                        while True:
                            # Shield the whole lock+forward task, not just its
                            # thread: disconnect must not release the turn while
                            # the actual model invocation is still executing.
                            piece_task = asyncio.create_task(pull_piece())
                            done, piece = await asyncio.shield(piece_task)
                            if done:
                                break
                            if not piece:
                                continue
                            yield event(chunk(piece))
                    except Exception:
                        logger.exception("Streaming model generation failed")
                        yield event(
                            {
                                "error": {
                                    "message": "Model generation failed.",
                                    "type": "server_error",
                                }
                            }
                        )
                        yield "data: [DONE]\n\n"
                        return
                    finally:
                        cancel_event.set()
                        # The thread executing next(iterator) survives HTTP
                        # cancellation. Wait for that pull, then close the
                        # paused generator before returning the model permit.
                        with anyio.CancelScope(shield=True):
                            if piece_task is not None:
                                try:
                                    await asyncio.shield(piece_task)
                                except Exception:
                                    pass
                            await asyncio.to_thread(iterator.close)
                            worker_done = stream_state.get("worker_done")
                            if worker_done is not None:
                                await asyncio.to_thread(worker_done.wait)
                    yield event(chunk(finish_reason=stream_state.get("finish_reason", "stop")))
                    if include_usage:
                        prompt_tokens = stream_state["prompt_tokens"]
                        completion_tokens = stream_state["completion_tokens"]
                        yield event({
                            "id": completion_id, "object": "chat.completion.chunk" if is_chat else "text_completion",
                            "created": created, "model": engine.model_name, "choices": [],
                            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                      "total_tokens": prompt_tokens + completion_tokens},
                        })
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        if request.model is not None and request.model != engine.model_name:
            raise HTTPException(404, "Requested model is not served by this instance.")
        if request.stream:
            try:
                prepared = await asyncio.to_thread(engine.prepare_stream_request, request)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return stream_response(request, prepared)
        return await nonstream_response(request, http_request)

    @app.post("/v1/completions", dependencies=[Depends(authorize)])
    async def completions(request: CompletionRequest, http_request: Request):
        if request.model is not None and request.model != engine.model_name:
            raise HTTPException(404, "Requested model is not served by this instance.")
        if request.stream:
            try:
                prepared = await asyncio.to_thread(engine.prepare_stream_request, request)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return stream_response(request, prepared)
        return await nonstream_response(request, http_request)

    return app


def load_inference_model(settings: InferenceServerSettings) -> Tuple[Any, Any]:
    """Load a full checkpoint or PEFT adapter for local generation."""
    if settings.backend == "auto":
        from .ServingPlan import resolve_serving_settings

        settings = resolve_serving_settings(settings)
    if settings.backend == "vllm":
        raise ValueError("vLLM owns model loading; use create_inference_app() or serve(), not load_inference_model().")
    if settings.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")
    if settings.dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be one of: auto, float32, float16, bfloat16.")
    if settings.quantization not in {"none", "4bit", "8bit", "int8"}:
        raise ValueError("quantization must be one of: none, 4bit, 8bit, int8.")
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Local inference requires torch and transformers.") from exc

    from .InferenceDevice import resolve_inference_device, resolve_inference_dtype

    device = resolve_inference_device(settings.device, torch)
    dtype_name = resolve_inference_dtype(settings.dtype, device)
    if (device == "cuda" and settings.dtype == "auto" and settings.backend_selection
            and settings.backend_selection["requested"] == "auto"
            and settings.quantization not in {"4bit", "8bit"}):
        # Preserve checkpoint dtype on the new auto deployment route. Do not
        # cast a BF16 Qwen model to FP16 simply because a CUDA device exists.
        dtype_name = "auto"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }[dtype_name]
    load_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": settings.trust_remote_code,
        "local_files_only": settings.local_files_only,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    if settings.quantization in {"4bit", "8bit"}:
        if device != "cuda":
            raise ValueError("4bit/8bit inference quantization requires device='cuda'.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Quantized inference requires bitsandbytes; install SaddleLLM with "
                "the serve and qlora extras."
            ) from exc
        if settings.quantization == "4bit":
            compute_dtype = dtype if dtype_name in {"float16", "bfloat16"} else torch.float16
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model_path = settings.model_path
    local_path = Path(model_path).expanduser()
    adapter_config_path = local_path / "adapter_config.json"
    quantization_manifest_path = local_path / "quantization_manifest.json"
    if settings.backend == "native":
        from ..models.ModelLoader import load_native_causal_lm

        native_kwargs = {}
        if settings.quantization == "int8":
            native_kwargs = {"quantization": "int8",
                             "fuse_projections": settings.fuse_projections}
        model = load_native_causal_lm(
            model_path, device=device, dtype=dtype,
            local_files_only=settings.local_files_only,
            attention_backend=settings.native_attention_backend, **native_kwargs,
        )
        if settings.quantization == "int8":
            settings.resolved_quantization = model._saddle_cuda_int8["method"]
        tokenizer_source = getattr(model, "_saddle_tokenizer_source", model_path)
    elif quantization_manifest_path.is_file():
        if settings.quantization != "none":
            raise ValueError(
                "The model directory is already quantized; do not also pass --quantization."
            )
        from ..models.ModelLoader import load_causal_lm, packaged_quantization_kind

        kind = packaged_quantization_kind(local_path)
        if kind not in {"dynamic-int8", "portable-int8-weight-only", "compressed-tensors"}:
            raise ValueError(
                "This packaged quantization method is not supported by saddlellm serve."
            )
        if kind == "dynamic-int8" and device != "cpu":
            raise ValueError("Packaged dynamic INT8 models require --device cpu.")
        if kind == "portable-int8-weight-only" and settings.dtype not in {"auto", "float32"}:
            raise ValueError("Portable INT8 weight-only QAT serving requires dtype='float32'.")
        model = load_causal_lm(
            str(local_path), device=device,
            dtype=torch.float32 if kind in {"dynamic-int8", "portable-int8-weight-only"} else dtype,
            trust_remote_code=settings.trust_remote_code,
            local_files_only=settings.local_files_only,
        )
        settings.resolved_quantization = kind
        tokenizer_source = str(local_path)
    elif adapter_config_path.is_file():
        from ..training.PostTrainingCompatibility import (
            stabilize_peft_optional_backends,
        )

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
        if settings.merge_adapter and settings.quantization == "none":
            model = model.merge_and_unload()
        elif settings.merge_adapter:
            logger.info(
                "Keeping the PEFT adapter attached because %s base weights cannot "
                "be safely merged in-place.",
                settings.quantization,
            )
        tokenizer_source = model_path if _has_tokenizer_files(local_path) else base_model
    else:
        from ..models.ModelLoader import is_saddle_checkpoint, load_causal_lm

        if settings.quantization != "none" and is_saddle_checkpoint(model_path):
            raise ValueError(
                "Runtime bitsandbytes quantization supports Hugging Face models and "
                "PEFT adapters, not native SaddleLLM checkpoints."
            )
        quantized_kwargs = {}
        model_device = device
        if "quantization_config" in load_kwargs:
            quantized_kwargs = {
                "quantization_config": load_kwargs["quantization_config"],
                "device_map": "auto",
            }
            model_device = None

        model = load_causal_lm(
            model_path,
            device=model_device,
            dtype=dtype,
            trust_remote_code=settings.trust_remote_code,
            local_files_only=settings.local_files_only,
            **quantized_kwargs,
        )
        tokenizer_source = model_path
    if device != "cuda":
        model = model.to(device)
    model.eval()
    if settings.fuse_projections or settings.cpu_prepack_linear:
        from ..models.SaddleModeling import SaddleForCausalLM

        if isinstance(model, SaddleForCausalLM) and not getattr(model, "_saddle_cuda_int8", None):
            from ..models.InferenceOptimizer import optimize_native_inference

            if settings.fuse_projections:
                native_source = getattr(model, "_saddle_native_source", None) or {}
                # Real Qwen3 BF16 weights showed different greedy answers when
                # Q/K/V were packed into one GEMM. Keep gate/up packing, which
                # preserved those answers, without changing other model paths.
                pack_qkv = not (
                    native_source.get("model_type") == "qwen3"
                    and next(model.parameters()).dtype == torch.bfloat16
                )
                report = optimize_native_inference(
                    model,
                    pack_qkv=pack_qkv,
                    cpu_prepack=settings.cpu_prepack_linear,
                )
            else:
                from ..models.CPULinear import prepare_cpu_linears
                from ..models.InferenceOptimizer import (
                    NativeInferenceOptimizationReport,
                )

                report = NativeInferenceOptimizationReport()
                report.cpu_prepack = prepare_cpu_linears(model)
            model._saddle_inference_optimization = report.to_dict()
            logger.info("Native inference projection packing: %s", report.to_dict())
            if settings.cpu_prepack_linear and not report.cpu_prepack.get("prepared"):
                logger.warning("CPU prepacking was requested but no cache was prepared: %s", report.cpu_prepack)
    from ..models.TokenizerLoader import load_tokenizer_compatible

    if settings.native_fused_kernels:
        from ..models.NativeFusedKernels import configure_decode_kernels, warmup_decode_kernels

        counts = configure_decode_kernels(model)
        warmup_decode_kernels(model, max_batch_size=settings.batch_size)
        model._saddle_inference_optimization = {
            **getattr(model, "_saddle_inference_optimization", {}), **counts,
        }

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
