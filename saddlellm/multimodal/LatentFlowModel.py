"""Conditioned flow-matching Transformer for image latent generation.

The model operates on codec latents instead of raw pixels.  Text encoding and
image decoding are intentionally supplied by separate components so this
objective can evolve without coupling SaddleLLM to a specific VAE provider.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentFlowConfig:
    latent_channels: int = 4
    latent_height: int = 32
    latent_width: int = 32
    patch_size: int = 2
    condition_dim: int = 768
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "latent_channels": self.latent_channels,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "patch_size": self.patch_size,
            "condition_dim": self.condition_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.latent_height % self.patch_size or self.latent_width % self.patch_size:
            raise ValueError("latent height/width must be divisible by patch_size")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    @property
    def num_patches(self) -> int:
        return (self.latent_height // self.patch_size) * (
            self.latent_width // self.patch_size
        )

    @property
    def patch_dim(self) -> int:
        return self.latent_channels * self.patch_size * self.patch_size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    if half == 0:
        return timesteps[:, None]
    scale = -math.log(10_000.0) / max(1, half - 1)
    frequencies = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32) * scale
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if embedding.shape[-1] < width:
        embedding = F.pad(embedding, (0, width - embedding.shape[-1]))
    return embedding


class ConditionalLatentFlowTransformer(nn.Module):
    """Predict velocity in a continuous image-latent flow."""

    def __init__(self, config: LatentFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.latent_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches, config.hidden_size)
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.hidden_size),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
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
        self.output_projection = nn.Linear(config.hidden_size, config.patch_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        # A zero output projection starts with zero velocity, a stable initial
        # state for flow-matching optimization.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(noisy_latents, timesteps, condition)
        tokens = self.patch_embed(noisy_latents).flatten(2).transpose(1, 2)
        time_tokens = self.time_projection(
            _timestep_embedding(timesteps, self.config.hidden_size)
        ).to(dtype=tokens.dtype)
        condition_tokens = self.condition_projection(condition).to(dtype=tokens.dtype)
        tokens = (
            tokens
            + self.position_embedding.to(dtype=tokens.dtype)
            + time_tokens[:, None, :]
            + condition_tokens[:, None, :]
        )
        tokens = self.transformer(tokens)
        patches = self.output_projection(self.output_norm(tokens))
        return self._unpatchify(patches)

    def compute_flow_loss(
        self,
        target_latents: torch.Tensor,
        condition: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Linear interpolation flow matching from Gaussian noise to data."""

        batch = target_latents.shape[0]
        noise = torch.randn_like(target_latents) if noise is None else noise
        if noise.shape != target_latents.shape:
            raise ValueError("noise must have the same shape as target_latents")
        if timesteps is None:
            timesteps = torch.rand(
                batch, device=target_latents.device, dtype=target_latents.dtype
            )
        if timesteps.shape != (batch,):
            raise ValueError("timesteps must have shape [batch]")
        interpolation = timesteps.reshape(batch, 1, 1, 1)
        noisy_latents = (1.0 - interpolation) * noise + interpolation * target_latents
        target_velocity = target_latents - noise
        predicted_velocity = self(noisy_latents, timesteps, condition)
        loss = F.mse_loss(predicted_velocity.float(), target_velocity.float())
        return {
            "loss": loss,
            "flow_loss": loss.detach(),
            "velocity_rmse": torch.sqrt(loss.detach().clamp_min(0.0)),
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        num_steps: int = 30,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler-integrate the learned velocity field from noise to data."""

        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        batch = condition.shape[0]
        parameter = next(self.parameters())
        if initial_noise is None:
            latents = torch.randn(
                batch,
                self.config.latent_channels,
                self.config.latent_height,
                self.config.latent_width,
                device=condition.device,
                dtype=parameter.dtype,
            )
        else:
            latents = initial_noise.clone()
        step_size = 1.0 / num_steps
        was_training = self.training
        self.eval()
        try:
            for step in range(num_steps):
                timesteps = torch.full(
                    (batch,),
                    step / num_steps,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                latents = latents + step_size * self(latents, timesteps, condition)
        finally:
            self.train(was_training)
        return latents

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        height = self.config.latent_height // self.config.patch_size
        width = self.config.latent_width // self.config.patch_size
        patches = patches.reshape(
            batch,
            height,
            width,
            self.config.patch_size,
            self.config.patch_size,
            self.config.latent_channels,
        )
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
        )

    def _validate_inputs(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        expected = (
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
        )
        if latents.ndim != 4 or tuple(latents.shape[1:]) != expected:
            raise ValueError(
                f"latents must have shape [batch, {expected[0]}, {expected[1]}, {expected[2]}]"
            )
        if timesteps.shape != (latents.shape[0],):
            raise ValueError("timesteps must have shape [batch]")
        if condition.shape != (latents.shape[0], self.config.condition_dim):
            raise ValueError(
                f"condition must have shape [batch, {self.config.condition_dim}]"
            )


__all__ = ["LatentFlowConfig", "ConditionalLatentFlowTransformer"]
