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
    supports_compact_gqa: bool = False

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
        *,
        enable_gqa: Optional[bool] = None,
    ) -> torch.Tensor:
        spec = cls.resolve(name)
        handler = cls._handlers.get(spec.name)
        if handler is None:
            raise ValueError(f"Attention backend has no handler: {spec.name}")
        if handler is _sdpa_reference_attention:
            return handler(q, k, v, attn_mask, is_causal)
        compact_gqa = (
            q.ndim >= 3
            and k.ndim >= 3
            and v.ndim >= 3
            and (q.shape[-3] != k.shape[-3] or k.shape[-3] != v.shape[-3])
        )
        # ``None`` is the safe automatic policy used by the model: preserve
        # compact K/V only when CUDA can actually dispatch a Flash kernel.
        # PyTorch 2.6 otherwise accepts enable_gqa but can silently fall back
        # to its math kernel, which is a severe performance/memory regression.
        use_compact_gqa = (
            spec.supports_compact_gqa
            and (
                enable_gqa is True
                or (
                    enable_gqa is None
                    and _can_use_flash_gqa(q, k, v, attn_mask, is_causal)
                )
            )
        )
        if compact_gqa and not use_compact_gqa:
            if (
                enable_gqa is None
                and handler is _sdpa_attention
                and _can_fold_gqa_decode(q, k, v, attn_mask, is_causal)
            ):
                return _folded_gqa_decode(q, k, v)
            k, v = _expand_grouped_kv(q, k, v)
        return handler(q, k, v, attn_mask, is_causal)

    @classmethod
    def ensure_defaults(cls) -> None:
        if cls._registry:
            return
        sdpa_available = hasattr(F, "scaled_dot_product_attention")
        flash_available = sdpa_available and _flash_attention_available()
        cls.register(
            AttentionBackendSpec(
                name="sdpa",
                family="torch",
                available=sdpa_available,
                required_packages=["torch>=2.0"],
                notes=["Uses torch.nn.functional.scaled_dot_product_attention."],
                supports_compact_gqa=_sdpa_supports_gqa(),
            ),
            _sdpa_attention,
        )
        cls.register(
            AttentionBackendSpec(
                name="sdpa_reference", family="torch", available=sdpa_available,
                required_packages=["torch>=2.5"],
                notes=["Reference compact-GQA SDPA dispatch; may use slower math kernels. No decode folding."],
                supports_compact_gqa=_sdpa_supports_gqa(),
            ),
            _sdpa_reference_attention,
        )
        cls.register(
            AttentionBackendSpec(
                name="flash",
                family="torch",
                available=flash_available,
                fallback="sdpa",
                required_packages=["torch>=2.0"],
                notes=[
                    "Uses PyTorch SDPA dispatch only when this Torch/CUDA build "
                    "reports FlashAttention support."
                ],
                supports_compact_gqa=_sdpa_supports_gqa(),
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
                supports_compact_gqa=_sdpa_supports_gqa(),
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
    enable_gqa = q.shape[-3] != k.shape[-3]
    kwargs = {"enable_gqa": True} if enable_gqa else {}
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        **kwargs,
    )


def _sdpa_reference_attention(q, k, v, attn_mask, is_causal):
    """Match the unmasked compact-GQA SDPA contract, including math fallback.

    The ordinary speed-oriented SDPA route deliberately avoids that fallback
    and may change low-precision rounding. This separate route is for numerical
    compatibility; it neither imports an HF model nor forces a global backend.
    """
    compact = attn_mask is None and _sdpa_supports_gqa()
    if not compact:
        k, v = _expand_grouped_kv(q, k, v)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal,
        scale=q.shape[-1] ** -0.5, **({"enable_gqa": True} if compact else {}),
    )


def _sdpa_supports_gqa() -> bool:
    """Return whether the installed SDPA exposes the ``enable_gqa`` keyword."""

    if not hasattr(F, "scaled_dot_product_attention"):
        return False
    doc = getattr(F.scaled_dot_product_attention, "__doc__", "") or ""
    return "enable_gqa" in doc


def _flash_attention_available() -> bool:
    """Report actual local FlashAttention capability, not just the SDPA API."""

    if not torch.cuda.is_available():
        return False
    probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except (RuntimeError, TypeError):
        return False


def _can_use_flash_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
) -> bool:
    """Check compact GQA kernel support without executing an attention op."""

    if not q.is_cuda or not hasattr(torch.backends.cuda, "can_use_flash_attention"):
        return False
    try:
        params = torch.backends.cuda.SDPAParams(
            q,
            k,
            v,
            attn_mask,
            0.0,
            is_causal,
            True,
        )
        return bool(torch.backends.cuda.can_use_flash_attention(params))
    except (AttributeError, RuntimeError, TypeError):
        return False


def _can_fold_gqa_decode(q, k, v, attn_mask, is_causal) -> bool:
    """CUDA no-grad single-query SDPA can group query heads as query rows.

    Every query attends to the same complete KV history, so those rows need
    neither a causal mask nor repeated K/V heads. Do not use this for prefill,
    multi-query decoding, explicit masks, gradients or custom handlers.
    Native Flash GQA (when supported) retains precedence in the registry.
    """
    return (
        q.is_cuda
        and not torch.is_grad_enabled()
        and q.ndim == k.ndim == v.ndim == 4
        and q.shape[-2] == 1
        and attn_mask is None
        and not is_causal
        and q.shape[0] == k.shape[0] == v.shape[0]
        and k.shape[1] == v.shape[1]
        and k.shape[1] > 0
        and q.shape[1] > k.shape[1]
        and q.shape[1] % k.shape[1] == 0
    )


def _folded_gqa_decode(q, k, v):
    batch, query_heads, _, head_dim = q.shape
    kv_heads = k.shape[1]
    grouped_query = q.reshape(batch, kv_heads, query_heads // kv_heads, head_dim)
    output = _sdpa_attention(grouped_query, k, v, None, False)
    return output.reshape(batch, query_heads, 1, v.shape[-1])


def _expand_grouped_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand compact grouped K/V heads for legacy five-argument handlers."""

    if q.ndim < 3 or k.ndim < 3 or v.ndim < 3:
        raise ValueError("GQA query, key, and value tensors must have at least 3 dimensions")
    query_heads = q.shape[-3]
    key_heads = k.shape[-3]
    value_heads = v.shape[-3]
    if key_heads != value_heads:
        raise ValueError("GQA key and value tensors must have the same number of heads")
    if query_heads == key_heads:
        return k, v
    if key_heads <= 0 or query_heads % key_heads:
        raise ValueError(
            "GQA query heads must be divisible by key/value heads; got "
            f"{query_heads} and {key_heads}"
        )
    repeat = query_heads // key_heads
    return (
        k.repeat_interleave(repeat, dim=-3),
        v.repeat_interleave(repeat, dim=-3),
    )


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
