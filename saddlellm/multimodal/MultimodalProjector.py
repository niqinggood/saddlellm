"""Projectors that connect vision features to SaddleLLM decoder embeddings."""
from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultimodalProjectorConfig:
    kind: str = "mlp"  # linear | mlp | gated_mlp | resampler
    vision_hidden_size: int = 768
    llm_hidden_size: int = 1024
    intermediate_size: int = 0
    num_query_tokens: int = 64
    dropout: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class LinearProjector(nn.Module):
    def __init__(self, config: MultimodalProjectorConfig):
        super().__init__()
        self.proj = nn.Linear(config.vision_hidden_size, config.llm_hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class MLPProjector(nn.Module):
    def __init__(self, config: MultimodalProjectorConfig):
        super().__init__()
        mid = config.intermediate_size or max(config.vision_hidden_size, config.llm_hidden_size)
        self.net = nn.Sequential(
            nn.Linear(config.vision_hidden_size, mid),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mid, config.llm_hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class GatedMLPProjector(nn.Module):
    def __init__(self, config: MultimodalProjectorConfig):
        super().__init__()
        mid = config.intermediate_size or max(config.vision_hidden_size, config.llm_hidden_size)
        self.up = nn.Linear(config.vision_hidden_size, mid)
        self.gate = nn.Linear(config.vision_hidden_size, mid)
        self.down = nn.Linear(mid, config.llm_hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(F.silu(self.gate(features)) * self.up(features)))


class ResamplerProjector(nn.Module):
    """Small Perceiver-style resampler for reducing image tokens."""

    def __init__(self, config: MultimodalProjectorConfig):
        super().__init__()
        self.query = nn.Parameter(torch.randn(config.num_query_tokens, config.llm_hidden_size) * 0.02)
        self.k_proj = nn.Linear(config.vision_hidden_size, config.llm_hidden_size)
        self.v_proj = nn.Linear(config.vision_hidden_size, config.llm_hidden_size)
        self.out = nn.Linear(config.llm_hidden_size, config.llm_hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        q = self.query[None, :, :].expand(batch, -1, -1)
        k = self.k_proj(features)
        v = self.v_proj(features)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        probs = F.softmax(scores.float(), dim=-1).to(dtype=features.dtype)
        return self.out(torch.matmul(probs, v))


class MultimodalProjectorFactory:
    @staticmethod
    def build(config: MultimodalProjectorConfig) -> nn.Module:
        kind = (config.kind or "mlp").lower()
        if kind == "linear":
            return LinearProjector(config)
        if kind == "mlp":
            return MLPProjector(config)
        if kind == "gated_mlp":
            return GatedMLPProjector(config)
        if kind == "resampler":
            return ResamplerProjector(config)
        raise ValueError(f"Unknown multimodal projector kind: {config.kind}")
