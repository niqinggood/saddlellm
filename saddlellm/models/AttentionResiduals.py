"""Depth-wise attention residuals, independent of token attention and KV state."""

from __future__ import annotations

import torch
from torch import nn


class SaddleAttentionResidual(nn.Module):
    """Mix raw depth sources using RMS-normalized keys and a learned query.

    Sources have shape [batch, tokens, hidden]. Softmax is over sources, never
    tokens. A zero-initialized pseudo-query produces uniform averaging, NOT
    the standard additive residual. All state is local to one forward call.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        if hidden_size <= 0 or eps <= 0:
            raise ValueError("hidden_size and eps must be positive")
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.key_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, *sources: torch.Tensor, return_weights: bool = False):
        if not sources:
            raise ValueError("attention residuals require at least one source")
        first = sources[0]
        if (
            not first.is_floating_point()
            or first.ndim != 3
            or first.shape[-1] != self.query.numel()
        ):
            raise ValueError("sources must have shape [batch, tokens, hidden]")
        if any(
            source.shape != first.shape
            or source.device != first.device
            or not source.is_floating_point()
            for source in sources
        ):
            raise ValueError(
                "depth sources must be floating and share shape and device"
            )
        # AMP normally leaves embeddings in FP32 but computes branch outputs
        # in FP16/BF16. Those are valid sources in one residual stream; require
        # matching shapes/devices, not dtypes. Keep the first (embedding) dtype
        # as the stream/output dtype instead of downcasting the embedding.
        compute_dtype = (
            torch.float64
            if any(source.dtype == torch.float64 for source in sources)
            else torch.float32
        )
        values = torch.stack([source.to(compute_dtype) for source in sources])
        keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        keys = keys * self.key_norm_weight.to(compute_dtype)
        logits = (keys * self.query.to(compute_dtype)).sum(-1)
        weights = logits.softmax(dim=0)
        output = (weights.unsqueeze(-1) * values).sum(dim=0).to(first.dtype)
        return (output, weights) if return_weights else output
