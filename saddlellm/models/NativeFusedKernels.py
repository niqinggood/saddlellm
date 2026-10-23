"""Optional CUDA decode kernels that preserve the eager rounding boundaries.

Triton is imported only when explicitly enabling this optimization. RMSNorm
keeps PyTorch's variance reduction: a different parallel reduction can change
BF16 rounding and greedy output. No weights, KV representation or GEMMs change.
"""
from __future__ import annotations

import torch


def load_decode_kernels():
    """Load the optional implementation before serving, or fail explicitly."""
    if not torch.cuda.is_available():
        raise RuntimeError("native fused kernels require CUDA and Triton")
    try:
        from ._NativeTritonKernels import rms_norm_decode, rotary_decode
    except ImportError as exc:
        raise RuntimeError("native fused kernels require a compatible Triton installation") from exc
    return rms_norm_decode, rotary_decode


def configure_decode_kernels(model, enabled=True):
    """Configure an exclusively owned native eval model before serving."""
    from .SaddleModeling import SaddleAttention, SaddleForCausalLM, SaddleRMSNorm

    if not isinstance(enabled, bool):
        raise ValueError("fused_decode_kernels must be boolean")
    if not isinstance(model, SaddleForCausalLM):
        raise TypeError("fused decode kernels require a native SaddleForCausalLM")
    if any(module.training for module in model.modules()):
        raise ValueError("call model.eval() before configuring fused decode kernels")
    norm = rotary = None
    if enabled:
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise ValueError("fused_decode_kernels requires a CUDA-resident model")
        norm, rotary = load_decode_kernels()
    counts = {"fused_decode_norms": 0, "fused_decode_rotary": 0}
    for module in model.modules():
        if type(module) is SaddleRMSNorm:
            module._fused_decode = norm
            counts["fused_decode_norms"] += int(enabled)
        elif type(module) is SaddleAttention:
            module._fused_rotary_decode = rotary
            counts["fused_decode_rotary"] += int(enabled)
    return counts


@torch.inference_mode()
def warmup_decode_kernels(model, max_batch_size=1):
    """Compile common decode shapes before accepting traffic; no KV or weights change.

    Bound startup work to batches 1..8. Other shapes can compile on first use.
    No request buffers are retained by the launcher cache.
    """
    from .SaddleModeling import SaddleAttention, SaddleRMSNorm

    for module in model.modules():
        if type(module) is SaddleRMSNorm and getattr(module, "_fused_decode", None):
            weight = module.weight
            x = torch.zeros((1, 1, weight.numel()), device=weight.device, dtype=weight.dtype)
            if can_fuse_norm(x, weight):
                module._fused_decode(x, weight, module.eps, module.weight_in_fp32)
        elif type(module) is SaddleAttention and getattr(module, "_fused_rotary_decode", None):
            weight = next(module.parameters())
            for batch in range(1, min(max_batch_size, 8) + 1):
                q = torch.zeros((batch, module.num_heads, 1, module.head_dim), device=weight.device, dtype=weight.dtype)
                k = torch.zeros((batch, module.num_kv_heads, 1, module.head_dim), device=weight.device, dtype=weight.dtype)
                cos, sin = torch.ones_like(q[0,0]), torch.zeros_like(q[0,0])
                if can_fuse_rotary(q, k, cos, sin):
                    module._fused_rotary_decode(q, k, cos, sin)
    torch.cuda.synchronize()


def can_fuse_norm(x, weight):
    return (
        x.is_cuda and x.device.index == torch.cuda.current_device()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim in (3, 4) and x.shape[-2] == 1 and x.shape[-1] > 0
        and x.shape[-1] <= 8192 and x.is_contiguous()
        and weight.dtype == x.dtype and weight.device == x.device
        and weight.ndim == 1 and weight.is_contiguous() and weight.numel() == x.shape[-1]
        and x.numel() > 0
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
    )


def can_fuse_rotary(q, k, cos, sin):
    return (
        q.is_cuda and q.device.index == torch.cuda.current_device()
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and q.ndim == k.ndim == 4 and q.shape[-2] == k.shape[-2] == 1
        and q.shape[0] == k.shape[0] and q.shape[-1] == k.shape[-1]
        and q.shape[0] > 0 and q.shape[1] > 0 and k.shape[1] > 0
        and 0 < q.shape[-1] <= 512 and q.shape[-1] % 2 == 0
        and cos.ndim == sin.ndim == 2 and cos.shape == sin.shape == (1, q.shape[-1])
        and q.dtype == k.dtype == cos.dtype == sin.dtype
        and q.device == k.device == cos.device == sin.device
        and q.stride(-1) == k.stride(-1) == 1 and cos.is_contiguous() and sin.is_contiguous()
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
    )
