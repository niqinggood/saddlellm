"""Operator backend registry for modular SaddleLLM models.

This layer keeps optional acceleration libraries behind a small registry so
model code can stay focused on architecture.  Optional backends fall back to
PyTorch SDPA unless explicitly implemented and available in the environment.
"""
import importlib.util
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool], torch.Tensor]


@dataclass
class AttentionBackendSpec:
    name: str
    family: str
    available: bool
    supports_causal: bool = True
    supports_mask: bool = True
    fallback: Optional[str] = None
    required_packages: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class AttentionBackendRegistry:
    """Runtime registry for attention kernels used by SaddleAttention."""

    _registry: Dict[str, AttentionBackendSpec] = {}
    _handlers: Dict[str, AttentionFn] = {}

    @classmethod
    def register(cls, spec: AttentionBackendSpec, handler: AttentionFn) -> None:
        cls._registry[spec.name] = spec
        cls._handlers[spec.name] = handler

    @classmethod
    def list_backends(cls) -> Dict[str, Dict]:
        cls.ensure_defaults()
        return {name: spec.to_dict() for name, spec in cls._registry.items()}

    @classmethod
    def available_backends(cls) -> List[str]:
        cls.ensure_defaults()
        return [name for name, spec in cls._registry.items() if spec.available]

    @classmethod
    def resolve(cls, name: str) -> AttentionBackendSpec:
        cls.ensure_defaults()
        backend = (name or "auto").lower()
        if backend == "auto":
            for candidate in ("flash", "sdpa", "eager"):
                spec = cls._registry.get(candidate)
                if spec and spec.available:
                    return spec
        if backend in {"te", "transformer-engine"}:
            backend = "transformer_engine"
        if backend not in cls._registry:
            raise ValueError(f"Unknown attention backend: {name}")
        spec = cls._registry[backend]
        if spec.available:
            return spec
        if spec.fallback:
            return cls.resolve(spec.fallback)
        return spec

    @classmethod
    def run(
        cls,
        name: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        spec = cls.resolve(name)
        handler = cls._handlers.get(spec.name)
        if handler is None:
            raise ValueError(f"Attention backend has no handler: {spec.name}")
        return handler(q, k, v, attn_mask, is_causal)

    @classmethod
    def ensure_defaults(cls) -> None:
        if cls._registry:
            return
        sdpa_available = hasattr(F, "scaled_dot_product_attention")
        cls.register(
            AttentionBackendSpec(
                name="sdpa",
                family="torch",
                available=sdpa_available,
                required_packages=["torch>=2.0"],
                notes=["Uses torch.nn.functional.scaled_dot_product_attention."],
            ),
            _sdpa_attention,
        )
        cls.register(
            AttentionBackendSpec(
                name="flash",
                family="torch",
                available=sdpa_available,
                fallback="sdpa",
                required_packages=["torch>=2.0"],
                notes=["Uses PyTorch SDPA dispatch; CUDA may select flash kernels when supported."],
            ),
            _sdpa_attention,
        )
        cls.register(
            AttentionBackendSpec(
                name="eager",
                family="torch",
                available=True,
                required_packages=["torch"],
                notes=["Explicit matmul + softmax path for debugging and numerical checks."],
            ),
            _eager_attention,
        )
        cls.register(
            AttentionBackendSpec(
                name="xformers",
                family="xformers",
                available=importlib.util.find_spec("xformers") is not None,
                supports_mask=False,
                fallback="sdpa",
                required_packages=["xformers"],
                notes=[
                    "Uses xformers memory_efficient_attention for simple causal/no-mask calls.",
                    "Falls back to SDPA for padded or cached masks.",
                ],
            ),
            _xformers_attention,
        )
        cls.register(
            AttentionBackendSpec(
                name="transformer_engine",
                family="nvidia-transformer-engine",
                available=importlib.util.find_spec("transformer_engine") is not None,
                fallback="sdpa",
                required_packages=["transformer_engine"],
                notes=[
                    "Capability is detected for planning.",
                    "Current SaddleLLM module dispatch falls back to SDPA until TE module wiring is enabled.",
                ],
            ),
            _sdpa_attention,
        )


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


def _eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if is_causal:
        q_len, kv_len = q.shape[-2], k.shape[-2]
        causal = torch.triu(
            torch.full((q_len, kv_len), torch.finfo(scores.dtype).min, device=scores.device, dtype=scores.dtype),
            diagonal=kv_len - q_len + 1,
        )
        scores = scores + causal[None, None, :, :]
    if attn_mask is not None:
        scores = scores + attn_mask
    probs = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v)


def _xformers_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
) -> torch.Tensor:
    if attn_mask is not None or not q.is_cuda:
        return _sdpa_attention(q, k, v, attn_mask, is_causal)
    try:
        import xformers.ops as xops

        bias = xops.LowerTriangularMask() if is_causal else None
        out = xops.memory_efficient_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_bias=bias,
        )
        return out.transpose(1, 2)
    except Exception:
        return _sdpa_attention(q, k, v, attn_mask, is_causal)


def list_attention_backends() -> Dict[str, Dict]:
    """Return registered attention backend capabilities."""
    return AttentionBackendRegistry.list_backends()
