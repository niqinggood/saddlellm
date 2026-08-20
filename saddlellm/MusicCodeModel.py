"""Text-conditioned autoregressive model for residual audio-codec streams."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MusicCodeConfig:
    num_codebooks: int = 4
    codebook_size: int = 1024
    max_sequence_length: int = 1024
    condition_dim: int = 768
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bos_token_id: int = 0

    def __post_init__(self) -> None:
        for name in (
            "num_codebooks",
            "codebook_size",
            "max_sequence_length",
            "condition_dim",
            "hidden_size",
            "num_layers",
            "num_heads",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0 <= self.bos_token_id < self.codebook_size:
            raise ValueError("bos_token_id must be inside the codec vocabulary")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConditionalMusicCodeTransformer(nn.Module):
    """Predict all codec codebooks at the next audio frame."""

    def __init__(self, config: MusicCodeConfig) -> None:
        super().__init__()
        self.config = config
        self.code_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.codebook_size, config.hidden_size)
                for _ in range(config.num_codebooks)
            ]
        )
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.hidden_size
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.hidden_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_size * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_heads = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.codebook_size)
                for _ in range(config.num_codebooks)
            ]
        )

    def forward(
        self,
        audio_codes: torch.Tensor,
        condition: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(
            audio_codes, condition, attention_mask=attention_mask
        )
        batch, _, length = audio_codes.shape
        tokens = sum(
            embedding(audio_codes[:, index])
            for index, embedding in enumerate(self.code_embeddings)
        ) / self.config.num_codebooks**0.5
        positions = torch.arange(length, device=audio_codes.device)
        tokens = (
            tokens
            + self.position_embedding(positions)[None]
            + self.condition_projection(condition)[:, None, :].to(tokens.dtype)
        )
        causal_mask = torch.triu(
            torch.ones(
                (length, length),
                device=audio_codes.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        padding_mask = attention_mask.eq(0) if attention_mask is not None else None
        hidden = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        hidden = self.output_norm(hidden)
        return torch.stack([head(hidden) for head in self.output_heads], dim=1)

    def compute_loss(
        self,
        audio_codes: torch.Tensor,
        condition: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._validate_inputs(
            audio_codes, condition, attention_mask=attention_mask
        )
        if audio_codes.shape[-1] < 1:
            raise ValueError("music training requires at least one codec frame")
        bos = torch.full(
            (audio_codes.shape[0], self.config.num_codebooks, 1),
            self.config.bos_token_id,
            device=audio_codes.device,
            dtype=audio_codes.dtype,
        )
        inputs = torch.cat([bos, audio_codes[:, :, :-1]], dim=-1)
        targets = audio_codes
        if attention_mask is not None:
            input_mask = torch.cat(
                [
                    torch.ones_like(attention_mask[:, :1]),
                    attention_mask[:, :-1],
                ],
                dim=-1,
            )
            target_mask = attention_mask
        else:
            input_mask = None
            target_mask = None
        logits = self(inputs, condition, attention_mask=input_mask)
        losses = []
        for index in range(self.config.num_codebooks):
            per_token = F.cross_entropy(
                logits[:, index].reshape(-1, self.config.codebook_size),
                targets[:, index].reshape(-1),
                reduction="none",
            ).reshape(targets.shape[0], targets.shape[-1])
            if target_mask is not None:
                denominator = target_mask.sum().clamp_min(1)
                loss = (per_token * target_mask).sum() / denominator
            else:
                loss = per_token.mean()
            losses.append(loss)
        total = torch.stack(losses).mean()
        result = {"loss": total, "codebook_loss": total.detach()}
        for index, loss in enumerate(losses):
            result[f"codebook_{index}_loss"] = loss.detach()
        return result

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        *,
        max_frames: int,
        prompt_codes: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        if max_frames <= 0 or max_frames > self.config.max_sequence_length:
            raise ValueError("max_frames must be within the configured sequence length")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        batch = condition.shape[0]
        bos = torch.full(
            (batch, self.config.num_codebooks, 1),
            self.config.bos_token_id,
            device=condition.device,
            dtype=torch.long,
        )
        if prompt_codes is None:
            codes = bos
        else:
            self._validate_inputs(prompt_codes, condition)
            if prompt_codes.shape[-1] > max_frames:
                raise ValueError("prompt_codes cannot be longer than max_frames")
            codes = torch.cat([bos, prompt_codes.clone()], dim=-1)
        was_training = self.training
        self.eval()
        try:
            while codes.shape[-1] - 1 < max_frames:
                logits = self(codes, condition)[:, :, -1]
                if temperature == 0:
                    next_codes = logits.argmax(dim=-1)
                else:
                    logits = logits / temperature
                    if top_k is not None and 0 < top_k < self.config.codebook_size:
                        values, indices = torch.topk(logits, top_k, dim=-1)
                        samples = torch.multinomial(
                            values.softmax(dim=-1).reshape(-1, top_k), 1
                        ).reshape(batch, self.config.num_codebooks, 1)
                        next_codes = indices.gather(-1, samples).squeeze(-1)
                    else:
                        next_codes = torch.multinomial(
                            logits.softmax(dim=-1).reshape(
                                -1, self.config.codebook_size
                            ),
                            1,
                        ).reshape(batch, self.config.num_codebooks)
                codes = torch.cat([codes, next_codes.unsqueeze(-1)], dim=-1)
        finally:
            self.train(was_training)
        return codes[:, :, 1:]

    def _validate_inputs(
        self,
        audio_codes: torch.Tensor,
        condition: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if audio_codes.ndim != 3:
            raise ValueError("audio_codes must have shape [batch, codebooks, frames]")
        if audio_codes.shape[1] != self.config.num_codebooks:
            raise ValueError(
                f"audio_codes must contain {self.config.num_codebooks} codebooks"
            )
        if audio_codes.shape[-1] > self.config.max_sequence_length:
            raise ValueError("audio code sequence exceeds max_sequence_length")
        if audio_codes.dtype not in {torch.int32, torch.int64}:
            raise ValueError("audio_codes must be integer tensors")
        if audio_codes.numel() and (
            audio_codes.min() < 0 or audio_codes.max() >= self.config.codebook_size
        ):
            raise ValueError("audio_codes contain values outside the codec vocabulary")
        if condition.shape != (audio_codes.shape[0], self.config.condition_dim):
            raise ValueError(
                f"condition must have shape [batch, {self.config.condition_dim}]"
            )
        if attention_mask is not None:
            if attention_mask.shape != (audio_codes.shape[0], audio_codes.shape[-1]):
                raise ValueError("attention_mask must have shape [batch, frames]")
            if attention_mask.dtype not in {
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise ValueError("attention_mask must be a boolean or integer tensor")


__all__ = ["MusicCodeConfig", "ConditionalMusicCodeTransformer"]
