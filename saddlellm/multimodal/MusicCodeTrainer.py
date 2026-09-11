"""Training and persistence for text-conditioned audio-code generation."""
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

from .MusicCodeModel import ConditionalMusicCodeTransformer, MusicCodeConfig


@dataclass
class MusicCodeTrainingConfig:
    data_path: str
    output_dir: str
    batch_size: int = 4
    epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    checkpoint_steps: int = 500
    seed: int = 42
    device: str = "auto"
    mixed_precision: str = "no"
    codec_name: str = "external-audio-codec"
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


class CachedMusicCodeDataset(Dataset):
    """NPZ contract: codes[N,Q,T], conditions[N,D], optional attention_mask[N,T]."""

    def __init__(self, path: str) -> None:
        source = Path(path)
        self._sharded = None
        if source.is_dir() or source.suffix.lower() == ".json":
            from .MediaCache import ShardedNpzStore

            self._sharded = ShardedNpzStore(path, ("codes", "conditions"))
            if "attention_mask" in self._sharded.manifest.get("arrays", {}):
                self._sharded.required_arrays = (
                    *self._sharded.required_arrays,
                    "attention_mask",
                )
            self.codes = None
            self.conditions = None
            self.attention_mask = None
            self._code_shape = self._sharded.array_shape("codes")
            self._condition_dim = self._sharded.array_shape("conditions")[0]
            self._code_min = int(self._sharded.array_stat("codes", "min"))
            self._code_max = int(self._sharded.array_stat("codes", "max"))
            if self._code_min < 0:
                raise ValueError("codes cannot contain negative codec token IDs")
            return
        if not source.is_file():
            raise FileNotFoundError(f"Cached music-code dataset not found: {path}")
        if source.suffix.lower() != ".npz":
            raise ValueError("Cached music-code data must use .npz format")
        archive = np.load(source, allow_pickle=False)
        missing = {"codes", "conditions"} - set(archive.files)
        if missing:
            raise ValueError(
                "Cached music-code dataset is missing array(s): "
                + ", ".join(sorted(missing))
            )
        self.codes = archive["codes"]
        self.conditions = archive["conditions"]
        self.attention_mask = archive["attention_mask"] if "attention_mask" in archive else None
        if self.codes.ndim != 3:
            raise ValueError("codes must have shape [samples, codebooks, frames]")
        if self.conditions.ndim != 2:
            raise ValueError("conditions must have shape [samples, condition_dim]")
        if len(self.codes) != len(self.conditions) or not len(self.codes):
            raise ValueError("codes and conditions require equal, non-zero sample counts")
        if not np.issubdtype(self.codes.dtype, np.integer):
            raise ValueError("codes must use an integer dtype")
        if self.codes.shape[1] == 0 or self.codes.shape[2] == 0:
            raise ValueError("codes require at least one codebook and one frame")
        if int(self.codes.min()) < 0:
            raise ValueError("codes cannot contain negative codec token IDs")
        if not np.isfinite(self.conditions).all():
            raise ValueError("conditions cannot contain NaN or infinite values")
        if self.attention_mask is not None:
            if self.attention_mask.shape != (len(self.codes), self.codes.shape[-1]):
                raise ValueError("attention_mask must have shape [samples, frames]")
            if not np.isin(self.attention_mask, (0, 1)).all():
                raise ValueError("attention_mask values must be 0 or 1")
        self._code_shape = tuple(self.codes.shape[1:])
        self._condition_dim = int(self.conditions.shape[1])
        self._code_min = int(self.codes.min())
        self._code_max = int(self.codes.max())

    def __len__(self) -> int:
        return len(self._sharded) if self._sharded is not None else len(self.codes)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._sharded is not None:
            item = self._sharded.get(index)
            result = {
                "codes": torch.from_numpy(item["codes"]).long(),
                "conditions": torch.from_numpy(item["conditions"]).float(),
            }
            if "attention_mask" in item:
                result["attention_mask"] = torch.from_numpy(
                    item["attention_mask"]
                ).long()
            return result
        item = {
            "codes": torch.from_numpy(np.array(self.codes[index], copy=True)).long(),
            "conditions": torch.from_numpy(
                np.array(self.conditions[index], copy=True)
            ).float(),
        }
        if self.attention_mask is not None:
            item["attention_mask"] = torch.from_numpy(
                np.array(self.attention_mask[index], copy=True)
            ).long()
        return item

    def infer_model_config(self, **overrides: Any) -> MusicCodeConfig:
        codebooks, frames = self._code_shape
        return MusicCodeConfig(
            num_codebooks=int(codebooks),
            codebook_size=self._code_max + 1,
            max_sequence_length=int(frames),
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


def save_music_code_checkpoint(
    model: ConditionalMusicCodeTransformer,
    path: str,
    *,
    training: Optional[MusicCodeTrainingConfig] = None,
    global_step: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
) -> str:
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "music_code_model.bin"))
    manifest = {
        "schema_version": 1,
        "model_type": "conditional_music_code_transformer",
        "model_config": model.config.to_dict(),
        "global_step": int(global_step),
        "training_config": asdict(training) if training is not None else None,
    }
    with open(os.path.join(path, "music_code_config.json"), "w", encoding="utf-8") as file:
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


def load_music_code_checkpoint(
    path: str, *, map_location: str = "cpu"
) -> ConditionalMusicCodeTransformer:
    directory = Path(path)
    config_path = directory / "music_code_config.json"
    weights_path = directory / "music_code_model.bin"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "music checkpoint requires music_code_config.json and music_code_model.bin"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model = ConditionalMusicCodeTransformer(
        MusicCodeConfig(**payload["model_config"])
    )
    model.load_state_dict(
        torch.load(weights_path, map_location=map_location, weights_only=True)
    )
    return model


def train_music_code(
    model_config: MusicCodeConfig,
    training_config: MusicCodeTrainingConfig,
) -> Dict[str, Any]:
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    dataset = CachedMusicCodeDataset(training_config.data_path)
    if dataset._code_shape[0] != model_config.num_codebooks:
        raise ValueError("dataset codebook count does not match model config")
    if dataset._code_shape[-1] > model_config.max_sequence_length:
        raise ValueError("dataset frame length exceeds model config")
    if dataset._code_max >= model_config.codebook_size:
        raise ValueError("dataset code exceeds configured codebook_size")
    if dataset._condition_dim != model_config.condition_dim:
        raise ValueError("dataset condition_dim does not match model config")

    device = _resolve_device(training_config.device)
    model = ConditionalMusicCodeTransformer(model_config).to(device)
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
            codes = batch["codes"].to(device)
            conditions = batch["conditions"].to(device)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device)
            with _autocast(device, training_config.mixed_precision):
                loss = model.compute_loss(
                    codes, conditions, attention_mask=mask
                )["loss"]
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
                save_music_code_checkpoint(
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
    save_music_code_checkpoint(
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
    "MusicCodeTrainingConfig",
    "CachedMusicCodeDataset",
    "save_music_code_checkpoint",
    "load_music_code_checkpoint",
    "train_music_code",
]
