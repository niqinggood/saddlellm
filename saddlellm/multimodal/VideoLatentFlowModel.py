"""Text-conditioned flow matching over causal-video-codec latents."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .LatentFlowModel import _timestep_embedding


@dataclass
class VideoLatentFlowConfig:
    latent_channels: int = 4
    latent_frames: int = 16
    latent_height: int = 32
    latent_width: int = 32
    temporal_patch_size: int = 1
    patch_size: int = 2
    condition_dim: int = 768
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "latent_channels",
            "latent_frames",
            "latent_height",
            "latent_width",
            "temporal_patch_size",
            "patch_size",
            "condition_dim",
            "hidden_size",
            "num_layers",
            "num_heads",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.latent_frames % self.temporal_patch_size:
            raise ValueError("latent_frames must be divisible by temporal_patch_size")
        if self.latent_height % self.patch_size or self.latent_width % self.patch_size:
            raise ValueError("latent height/width must be divisible by patch_size")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    @property
    def temporal_tokens(self) -> int:
        return self.latent_frames // self.temporal_patch_size

    @property
    def spatial_tokens(self) -> int:
        return (self.latent_height // self.patch_size) * (
            self.latent_width // self.patch_size
        )

    @property
    def patch_dim(self) -> int:
        return (
            self.latent_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConditionalVideoLatentFlowTransformer(nn.Module):
    """Predict a velocity field over a complete spatiotemporal latent clip."""

    def __init__(self, config: VideoLatentFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv3d(
            config.latent_channels,
            config.hidden_size,
            kernel_size=(
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ),
            stride=(
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ),
        )
        self.temporal_position = nn.Parameter(
            torch.zeros(1, config.temporal_tokens, 1, config.hidden_size)
        )
        self.spatial_position = nn.Parameter(
            torch.zeros(1, 1, config.spatial_tokens, config.hidden_size)
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
        nn.init.normal_(self.temporal_position, std=0.02)
        nn.init.normal_(self.spatial_position, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(noisy_latents, timesteps, condition)
        embedded = self.patch_embed(noisy_latents)
        batch, hidden, frames, height, width = embedded.shape
        tokens = embedded.flatten(3).permute(0, 2, 3, 1)
        tokens = tokens + self.temporal_position.to(tokens.dtype)
        tokens = tokens + self.spatial_position.to(tokens.dtype)
        tokens = tokens.reshape(batch, frames * height * width, hidden)
        time_token = self.time_projection(
            _timestep_embedding(timesteps, self.config.hidden_size)
        ).to(tokens.dtype)
        condition_token = self.condition_projection(condition).to(tokens.dtype)
        tokens = tokens + time_token[:, None] + condition_token[:, None]
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
        batch = target_latents.shape[0]
        noise = torch.randn_like(target_latents) if noise is None else noise
        if noise.shape != target_latents.shape:
            raise ValueError("noise must have the same shape as target_latents")
        if timesteps is None:
            timesteps = torch.rand(
                batch, device=target_latents.device, dtype=target_latents.dtype
            )
        interpolation = timesteps.reshape(batch, 1, 1, 1, 1)
        noisy = (1.0 - interpolation) * noise + interpolation * target_latents
        target_velocity = target_latents - noise
        prediction = self(noisy, timesteps, condition)
        loss = F.mse_loss(prediction.float(), target_velocity.float())
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
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        batch = condition.shape[0]
        parameter = next(self.parameters())
        shape = (
            batch,
            self.config.latent_channels,
            self.config.latent_frames,
            self.config.latent_height,
            self.config.latent_width,
        )
        if initial_noise is None:
            latents = torch.randn(
                *shape, device=condition.device, dtype=parameter.dtype
            )
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(f"initial_noise must have shape {shape}")
            latents = initial_noise.clone()
        was_training = self.training
        self.eval()
        try:
            for step in range(num_steps):
                times = torch.full(
                    (batch,),
                    step / num_steps,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                latents = latents + self(latents, times, condition) / num_steps
        finally:
            self.train(was_training)
        return latents

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch = patches.shape[0]
        temporal = cfg.temporal_tokens
        height = cfg.latent_height // cfg.patch_size
        width = cfg.latent_width // cfg.patch_size
        patches = patches.reshape(
            batch,
            temporal,
            height,
            width,
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
            cfg.latent_channels,
        )
        return patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            batch,
            cfg.latent_channels,
            cfg.latent_frames,
            cfg.latent_height,
            cfg.latent_width,
        )

    def _validate_inputs(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        expected = (
            self.config.latent_channels,
            self.config.latent_frames,
            self.config.latent_height,
            self.config.latent_width,
        )
        if latents.ndim != 5 or tuple(latents.shape[1:]) != expected:
            raise ValueError(f"video latents must have shape [batch, C, T, H, W]={expected}")
        if timesteps.shape != (latents.shape[0],):
            raise ValueError("timesteps must have shape [batch]")
        if condition.shape != (latents.shape[0], self.config.condition_dim):
            raise ValueError(
                f"condition must have shape [batch, {self.config.condition_dim}]"
            )


__all__ = ["VideoLatentFlowConfig", "ConditionalVideoLatentFlowTransformer"]
