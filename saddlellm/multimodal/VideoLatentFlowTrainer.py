"""Training and checkpoint lifecycle for cached video-codec latent clips."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .VideoLatentFlowModel import (
    ConditionalVideoLatentFlowTransformer,
    VideoLatentFlowConfig,
)


@dataclass
class VideoLatentFlowTrainingConfig:
    data_path: str
    output_dir: str
    batch_size: int = 2
    epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    checkpoint_steps: int = 500
    seed: int = 42
    device: str = "auto"
    mixed_precision: str = "no"
    codec_name: str = "external-causal-video-codec"
    condition_encoder: str = "external-text-encoder"

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs", "gradient_accumulation_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if self.max_steps > 0 and self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be no, fp16, or bf16")


class CachedVideoLatentDataset(Dataset):
    """NPZ contract: video_latents[N,C,T,H,W] and conditions[N,D]."""

    def __init__(self, path: str) -> None:
        source = Path(path)
        self._sharded = None
        if source.is_dir() or source.suffix.lower() == ".json":
            from .MediaCache import ShardedNpzStore

            self._sharded = ShardedNpzStore(
                path, ("video_latents", "conditions")
            )
            self.video_latents = None
            self.conditions = None
            self._latent_shape = self._sharded.array_shape("video_latents")
            self._condition_dim = self._sharded.array_shape("conditions")[0]
            return
        if not source.is_file():
            raise FileNotFoundError(f"Cached video-latent dataset not found: {path}")
        if source.suffix.lower() != ".npz":
            raise ValueError("Cached video-latent data must use .npz format")
        archive = np.load(source, allow_pickle=False)
        missing = {"video_latents", "conditions"} - set(archive.files)
        if missing:
            raise ValueError(
                "Cached video-latent dataset is missing array(s): "
                + ", ".join(sorted(missing))
            )
        self.video_latents = archive["video_latents"]
        self.conditions = archive["conditions"]
        if self.video_latents.ndim != 5:
            raise ValueError("video_latents must have shape [samples, C, T, H, W]")
        if self.conditions.ndim != 2:
            raise ValueError("conditions must have shape [samples, condition_dim]")
        if len(self.video_latents) != len(self.conditions) or not len(self.video_latents):
            raise ValueError("video_latents and conditions require equal, non-zero samples")
        if not np.isfinite(self.video_latents).all():
            raise ValueError("video_latents cannot contain NaN or infinite values")
        if not np.isfinite(self.conditions).all():
            raise ValueError("conditions cannot contain NaN or infinite values")
        self._latent_shape = tuple(self.video_latents.shape[1:])
        self._condition_dim = int(self.conditions.shape[1])

    def __len__(self) -> int:
        return len(self._sharded) if self._sharded is not None else len(self.video_latents)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._sharded is not None:
            item = self._sharded.get(index)
            return {
                "video_latents": torch.from_numpy(item["video_latents"]).float(),
                "conditions": torch.from_numpy(item["conditions"]).float(),
            }
        return {
            "video_latents": torch.from_numpy(
                np.array(self.video_latents[index], copy=True)
            ).float(),
            "conditions": torch.from_numpy(
                np.array(self.conditions[index], copy=True)
            ).float(),
        }

    def infer_model_config(self, **overrides: Any) -> VideoLatentFlowConfig:
        channels, frames, height, width = self._latent_shape
        return VideoLatentFlowConfig(
            latent_channels=int(channels),
            latent_frames=int(frames),
            latent_height=int(height),
            latent_width=int(width),
            condition_dim=self._condition_dim,
            **overrides,
        )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "no":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def save_video_latent_flow_checkpoint(
    model: ConditionalVideoLatentFlowTransformer,
    path: str,
    *,
    training: Optional[VideoLatentFlowTrainingConfig] = None,
    global_step: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
) -> str:
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "video_latent_flow_model.bin"))
    manifest = {
        "schema_version": 1,
        "model_type": "conditional_video_latent_flow",
        "model_config": model.config.to_dict(),
        "global_step": int(global_step),
        "training_config": asdict(training) if training is not None else None,
    }
    with open(
        os.path.join(path, "video_latent_flow_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    if optimizer is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "global_step": int(global_step),
            },
            os.path.join(path, "trainer_state.pt"),
        )
    return path


def load_video_latent_flow_checkpoint(
    path: str, *, map_location: str = "cpu"
) -> ConditionalVideoLatentFlowTransformer:
    directory = Path(path)
    config_path = directory / "video_latent_flow_config.json"
    weights_path = directory / "video_latent_flow_model.bin"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "video flow checkpoint requires video_latent_flow_config.json and "
            "video_latent_flow_model.bin"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model = ConditionalVideoLatentFlowTransformer(
        VideoLatentFlowConfig(**payload["model_config"])
    )
    model.load_state_dict(
        torch.load(weights_path, map_location=map_location, weights_only=True)
    )
    return model


def train_video_latent_flow(
    model_config: VideoLatentFlowConfig,
    training_config: VideoLatentFlowTrainingConfig,
) -> Dict[str, Any]:
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    dataset = CachedVideoLatentDataset(training_config.data_path)
    expected = (
        model_config.latent_channels,
        model_config.latent_frames,
        model_config.latent_height,
        model_config.latent_width,
    )
    if dataset._latent_shape != expected:
        raise ValueError("dataset video latent shape does not match model config")
    if dataset._condition_dim != model_config.condition_dim:
        raise ValueError("dataset condition_dim does not match model config")

    device = _resolve_device(training_config.device)
    model = ConditionalVideoLatentFlowTransformer(model_config).to(device)
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    updates_per_epoch = math.ceil(
        len(loader) / training_config.gradient_accumulation_steps
    )
    total_steps = (
        training_config.max_steps
        if training_config.max_steps > 0
        else training_config.epochs * updates_per_epoch
    )
    epoch_count = (
        max(training_config.epochs, math.ceil(total_steps / max(1, updates_per_epoch)))
        if training_config.max_steps > 0
        else training_config.epochs
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_multiplier(step, total_steps, training_config.warmup_steps),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and training_config.mixed_precision == "fp16",
    )
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    final_loss = float("nan")
    stop = False
    for _epoch in range(epoch_count):
        for batch_index, batch in enumerate(loader):
            latents = batch["video_latents"].to(device)
            conditions = batch["conditions"].to(device)
            with _autocast(device, training_config.mixed_precision):
                loss = model.compute_flow_loss(latents, conditions)["loss"]
                scaled_loss = loss / training_config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % training_config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            final_loss = float(loss.detach().cpu())
            if (
                training_config.checkpoint_steps > 0
                and global_step % training_config.checkpoint_steps == 0
            ):
                save_video_latent_flow_checkpoint(
                    model,
                    os.path.join(training_config.output_dir, f"checkpoint-{global_step}"),
                    training=training_config,
                    global_step=global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
            if global_step >= total_steps:
                stop = True
                break
        if stop:
            break
    save_video_latent_flow_checkpoint(
        model,
        training_config.output_dir,
        training=training_config,
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return {
        "status": "completed",
        "output_path": os.path.abspath(training_config.output_dir),
        "global_step": global_step,
        "train_samples": len(dataset),
        "final_loss": final_loss,
        "codec_name": training_config.codec_name,
        "condition_encoder": training_config.condition_encoder,
        "model_config": model_config.to_dict(),
    }


__all__ = [
    "VideoLatentFlowTrainingConfig",
    "CachedVideoLatentDataset",
    "save_video_latent_flow_checkpoint",
    "load_video_latent_flow_checkpoint",
    "train_video_latent_flow",
]
