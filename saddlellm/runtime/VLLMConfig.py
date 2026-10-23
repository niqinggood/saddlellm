"""Dependency-light, explicit configuration for the optional vLLM server."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VLLMServingConfig:
    """vLLM owns GPU scheduling; these are not native Saddle microbatch knobs."""

    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    max_num_seqs: int = 16
    max_num_batched_tokens: int | None = None
    enable_prefix_caching: bool = False
    enforce_eager: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        value = self.gpu_memory_utilization
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError("vllm.gpu_memory_utilization must be in (0, 1].")
        for name in ("tensor_parallel_size", "max_num_seqs"):
            self._positive_int(getattr(self, name), name)
        for name in ("max_model_len", "max_num_batched_tokens"):
            if getattr(self, name) is not None:
                self._positive_int(getattr(self, name), name)
        for name in ("enable_prefix_caching", "enforce_eager"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"vllm.{name} must be a boolean.")
        if (
            self.max_num_batched_tokens is not None
            and self.max_num_batched_tokens < self.max_num_seqs
        ):
            raise ValueError("vllm.max_num_batched_tokens must be >= max_num_seqs.")

    @staticmethod
    def _positive_int(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"vllm.{name} must be a positive integer.")

    def engine_kwargs(self, settings, model_reference):
        """Only known arguments; no unrestricted overrides or secret forwarding."""
        self.validate()
        kwargs = {
            "model": model_reference,
            "tokenizer": model_reference,
            "dtype": settings.dtype,
            "trust_remote_code": settings.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len
            or settings.max_input_tokens + settings.max_new_tokens,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enforce_eager": self.enforce_eager,
            "generation_config": "vllm",
            "enable_log_requests": False,
            "cpu_offload_gb": 0,
        }
        if self.max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
        return kwargs
