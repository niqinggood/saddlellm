"""Standalone trainer and configuration entry point for native world models."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from ._WorldModel import WorldModel
from .WorldModelBackends import (
    NATIVE_WORLD_MODEL_BACKENDS,
    create_world_model,
    normalize_world_model_backend,
)
from .WorldModelData import (
    WorldModelTrajectoryDataset,
    infer_world_model_dimensions,
    load_world_model_trajectories,
    split_world_model_trajectories,
)

logger = logging.getLogger(__name__)


@dataclass
class WorldModelDataConfig:
    """Offline trajectory input and sequence-window configuration."""

    train_path: str
    validation_path: Optional[str] = None
    sequence_length: int = 32
    stride: Optional[int] = None
    validation_split: float = 0.1
    validation_group_key: Optional[str] = None
    normalize_images: bool = True
    pad_short_trajectories: bool = True

    def __post_init__(self) -> None:
        if not self.train_path:
            raise ValueError("data.train_path is required")
        if self.sequence_length <= 0:
            raise ValueError("data.sequence_length must be positive")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("data.stride must be positive")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("data.validation_split must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldModelDataConfig":
        values = dict(data or {})
        if "train_path" not in values and "dataset_path" in values:
            values["train_path"] = values.pop("dataset_path")
        _reject_unknown_fields(cls, values, "world-model data")
        return cls(**values)


@dataclass
class WorldModelTrainingConfig:
    """Optimization, runtime, checkpoint, and logging settings."""

    output_dir: str = "./outputs/world_model"
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 100.0
    warmup_steps: int = 0
    max_steps: int = -1
    num_workers: int = 0
    mixed_precision: str = "auto"  # auto | no | fp16 | bf16
    device: str = "auto"  # auto | cpu | cuda | cuda:N | mps
    log_steps: int = 10
    checkpoint_steps: int = 500
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("training.output_dir is required")
        if self.num_epochs <= 0:
            raise ValueError("training.num_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay cannot be negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if self.max_grad_norm < 0:
            raise ValueError("training.max_grad_norm cannot be negative")
        if self.warmup_steps < 0:
            raise ValueError("training.warmup_steps cannot be negative")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("training.max_steps must be -1 or positive")
        if self.num_workers < 0:
            raise ValueError("training.num_workers cannot be negative")
        if self.log_steps < 0 or self.checkpoint_steps < 0:
            raise ValueError("logging/checkpoint steps cannot be negative")
        self.mixed_precision = str(self.mixed_precision).lower()
        if self.mixed_precision not in {"auto", "no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be auto, no, fp16, or bf16")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldModelTrainingConfig":
        values = dict(data or {})
        if "epochs" in values and "num_epochs" not in values:
            values["num_epochs"] = values.pop("epochs")
        _reject_unknown_fields(cls, values, "world-model training")
        return cls(**values)


@dataclass
class PreparedWorldModelRun:
    model: torch.nn.Module
    train_dataset: WorldModelTrajectoryDataset
    validation_dataset: Optional[WorldModelTrajectoryDataset]
    data_config: WorldModelDataConfig
    training_config: WorldModelTrainingConfig
    source_config: Dict[str, Any]
    summary: Dict[str, Any]


class WorldModelTrainer:
    """Native PyTorch trainer for registered SaddleLLM world models.

    The trainer performs episode-window batching, AMP on CUDA, gradient
    accumulation/clipping, cosine decay, validation, and resumable checkpoints.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: WorldModelTrajectoryDataset,
        config: Optional[WorldModelTrainingConfig] = None,
        validation_dataset: Optional[WorldModelTrajectoryDataset] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config or WorldModelTrainingConfig()
        self.device = _resolve_device(self.config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.amp_dtype, scaler_enabled = _resolve_amp(self.config.mixed_precision, self.device)
        self.scaler = _make_grad_scaler(scaler_enabled)
        self.global_step = 0
        self.start_epoch = 0
        self.history: List[Dict[str, Any]] = []
        self.best_eval_loss = float("inf")
        self._seed_everything(self.config.seed)

    def train(self) -> Dict[str, Any]:
        """Run optimization and save the final model plus JSON metrics."""

        os.makedirs(self.config.output_dir, exist_ok=True)
        train_loader = self._data_loader(self.train_dataset, shuffle=True)
        validation_loader = (
            self._data_loader(self.validation_dataset, shuffle=False)
            if self.validation_dataset is not None
            else None
        )
        updates_per_epoch = math.ceil(
            len(train_loader) / self.config.gradient_accumulation_steps
        )
        planned_steps = updates_per_epoch * self.config.num_epochs
        total_steps = (
            min(planned_steps, self.config.max_steps)
            if self.config.max_steps > 0
            else planned_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._learning_rate_lambda(max(total_steps, 1)),
        )
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)

        self.optimizer.zero_grad(set_to_none=True)
        stopped_early = self.global_step >= total_steps
        epochs_completed = self.start_epoch
        final_train_metrics: Dict[str, float] = {}
        final_eval_metrics: Dict[str, float] = {}
        for epoch in range(self.start_epoch, self.config.num_epochs):
            if stopped_early:
                break
            self.model.train()
            update_sums: Dict[str, float] = {}
            update_batches = 0
            for batch_index, batch in enumerate(train_loader):
                batch = self._move_batch(batch)
                remainder = len(train_loader) % self.config.gradient_accumulation_steps
                in_final_partial_group = bool(remainder) and batch_index >= len(train_loader) - remainder
                accumulation_divisor = (
                    remainder
                    if in_final_partial_group
                    else self.config.gradient_accumulation_steps
                )
                with self._autocast_context():
                    losses = self.model.compute_loss(batch, sample_state=True)
                    loss = losses["loss"] / accumulation_divisor
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite world-model loss at optimizer step {self.global_step}"
                    )
                self.scaler.scale(loss).backward()
                update_batches += 1
                for name, value in losses.items():
                    if name == "valid_transitions":
                        continue
                    update_sums[name] = update_sums.get(name, 0.0) + float(
                        value.detach().float().cpu()
                    )

                last_batch = batch_index + 1 == len(train_loader)
                should_update = (
                    (batch_index + 1) % self.config.gradient_accumulation_steps == 0
                    or last_batch
                )
                if not should_update:
                    continue
                self.scaler.unscale_(self.optimizer)
                if self.config.max_grad_norm:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    grad_norm_value = float(grad_norm.detach().float().cpu())
                else:
                    grad_norm_value = _gradient_norm(self.model)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

                final_train_metrics = {
                    name: value / max(update_batches, 1)
                    for name, value in update_sums.items()
                }
                final_train_metrics.update(
                    {
                        "grad_norm": grad_norm_value,
                        "learning_rate": float(self.scheduler.get_last_lr()[0]),
                    }
                )
                self._maybe_log(epoch, final_train_metrics)
                update_sums = {}
                update_batches = 0

                if (
                    self.config.checkpoint_steps
                    and self.global_step % self.config.checkpoint_steps == 0
                ):
                    self._save_checkpoint(epoch)
                if self.global_step >= total_steps:
                    stopped_early = True
                    break

            epochs_completed = epoch + 1
            if validation_loader is not None:
                final_eval_metrics = self.evaluate(validation_loader)
                eval_record = {
                    "type": "validation",
                    "epoch": epoch + 1,
                    "step": self.global_step,
                    **final_eval_metrics,
                }
                self.history.append(eval_record)
                logger.info(
                    "world-model validation epoch=%d step=%d loss=%.6f",
                    epoch + 1,
                    self.global_step,
                    final_eval_metrics["eval_loss"],
                )
                if final_eval_metrics["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = final_eval_metrics["eval_loss"]
                    self.model.save_pretrained(
                        os.path.join(self.config.output_dir, "best_model")
                    )

        self.model.save_pretrained(self.config.output_dir)
        result: Dict[str, Any] = {
            "status": "completed",
            "backend": getattr(self.model, "backend_name", "rssm"),
            "output_dir": os.path.abspath(self.config.output_dir),
            "global_step": self.global_step,
            "epochs_completed": epochs_completed,
            "device": str(self.device),
            "mixed_precision": _dtype_name(self.amp_dtype),
            "parameters": self.model.parameter_count(),
            "train_metrics": final_train_metrics,
            "eval_metrics": final_eval_metrics,
            "best_eval_loss": (
                self.best_eval_loss if math.isfinite(self.best_eval_loss) else None
            ),
        }
        self._save_json("training_history.json", self.history)
        self._save_json("training_metrics.json", result)
        self._save_json(
            "world_model_run_config.json",
            {
                "backend": getattr(self.model, "backend_name", "rssm"),
                "model": self.model.config.to_dict(),
                "training": self.config.to_dict(),
                "train_data": self.train_dataset.summary(),
                "validation_data": (
                    self.validation_dataset.summary()
                    if self.validation_dataset is not None
                    else None
                ),
            },
        )
        return result

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate with posterior means for stable, reproducible metrics."""

        if loader is None:
            if self.validation_dataset is None:
                raise ValueError("No validation dataset was provided")
            loader = self._data_loader(self.validation_dataset, shuffle=False)
        self.model.eval()
        totals: Dict[str, float] = {}
        total_weight = 0.0
        for batch in loader:
            batch = self._move_batch(batch)
            with self._autocast_context():
                losses = self.model.compute_loss(batch, sample_state=False)
            weight = float(losses["valid_transitions"].detach().float().cpu())
            total_weight += weight
            for name, value in losses.items():
                if name == "valid_transitions":
                    continue
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach().float().cpu()
                ) * weight
        denominator = max(total_weight, 1.0)
        metrics = {
            f"eval_{name}": value / denominator for name, value in totals.items()
        }
        self.model.train()
        return metrics

    def _data_loader(
        self,
        dataset: Optional[WorldModelTrajectoryDataset],
        shuffle: bool,
    ) -> DataLoader:
        if dataset is None:
            raise ValueError("dataset cannot be None")
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
            generator=generator,
        )

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=self.device.type == "cuda")
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

    def _autocast_context(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _learning_rate_lambda(self, total_steps: int):
        warmup_steps = min(self.config.warmup_steps, total_steps)

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return schedule

    def _maybe_log(self, epoch: int, metrics: Dict[str, float]) -> None:
        record = {
            "type": "train",
            "epoch": epoch + 1,
            "step": self.global_step,
            **metrics,
        }
        if not self.config.log_steps or self.global_step % self.config.log_steps != 0:
            return
        self.history.append(record)
        logger.info(
            "world-model epoch=%d step=%d loss=%.6f recon=%.6f reward=%.6f kl=%.6f",
            epoch + 1,
            self.global_step,
            metrics.get("loss", float("nan")),
            metrics.get("reconstruction_loss", float("nan")),
            metrics.get("reward_loss", float("nan")),
            metrics.get("kl_loss", float("nan")),
        )

    def _save_checkpoint(self, epoch: int) -> str:
        if self.scheduler is None:
            raise RuntimeError("Cannot checkpoint before the scheduler is initialized")
        path = os.path.join(
            self.config.output_dir,
            "checkpoints",
            f"step-{self.global_step:08d}",
        )
        self.model.save_pretrained(path)
        state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": epoch,
            "best_eval_loss": self.best_eval_loss,
            "history": self.history,
        }
        torch.save(state, os.path.join(path, "trainer_state.pt"))
        return path

    def _load_checkpoint(self, path: str) -> None:
        if self.scheduler is None:
            raise RuntimeError("Scheduler must exist before loading a checkpoint")
        weights_name = getattr(self.model, "WEIGHTS_NAME", WorldModel.WEIGHTS_NAME)
        weights_path = os.path.join(path, weights_name)
        state_path = os.path.join(path, "trainer_state.pt")
        if not os.path.isfile(weights_path) or not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"Checkpoint must contain {weights_name} and trainer_state.pt: {path}"
            )
        weights = _torch_load(weights_path, self.device, weights_only=True)
        self.model.load_state_dict(weights)
        state = _torch_load(state_path, self.device, weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.global_step = int(state.get("global_step", 0))
        self.start_epoch = int(state.get("epoch", 0))
        self.best_eval_loss = float(state.get("best_eval_loss", float("inf")))
        self.history = list(state.get("history", []))
        logger.info("Resumed world-model training from %s at step %d", path, self.global_step)

    def _save_json(self, filename: str, payload: Any) -> None:
        path = os.path.join(self.config.output_dir, filename)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def prepare_world_model_training(
    config: Union[str, os.PathLike, Dict[str, Any]],
) -> PreparedWorldModelRun:
    """Load/validate a run config, infer dimensions, and construct datasets/model."""

    source_config, config_directory = _load_run_config(config)
    data_config = WorldModelDataConfig.from_dict(source_config.get("data", {}))
    data_config.train_path = _resolve_input_path(data_config.train_path, config_directory)
    if data_config.validation_path:
        data_config.validation_path = _resolve_input_path(
            data_config.validation_path, config_directory
        )
    training_config = WorldModelTrainingConfig.from_dict(
        source_config.get("training", {})
    )

    train_trajectories = load_world_model_trajectories(data_config.train_path)
    if data_config.validation_path:
        validation_trajectories = load_world_model_trajectories(
            data_config.validation_path
        )
    else:
        train_trajectories, validation_trajectories = split_world_model_trajectories(
            train_trajectories,
            data_config.validation_split,
            training_config.seed,
            group_key=data_config.validation_group_key,
        )

    model_values = dict(source_config.get("model", {}) or {})
    model_backend = model_values.pop("backend", None)
    backend = normalize_world_model_backend(
        source_config.get("backend", model_backend or "rssm")
    )
    if backend not in NATIVE_WORLD_MODEL_BACKENDS:
        raise ValueError(
            f"Unknown SaddleLLM world-model backend {backend!r}; choose from "
            + ", ".join(sorted(NATIVE_WORLD_MODEL_BACKENDS))
        )
    action_type = str(model_values.get("action_type", "continuous")).lower()
    dimensions = infer_world_model_dimensions(train_trajectories, action_type)
    configured_observation_shape = model_values.get("observation_shape")
    if configured_observation_shape is None:
        model_values["observation_shape"] = dimensions["observation_shape"]
    elif tuple(configured_observation_shape) != tuple(dimensions["observation_shape"]):
        raise ValueError(
            "Configured observation_shape does not match data: "
            f"{tuple(configured_observation_shape)} vs {dimensions['observation_shape']}"
        )
    configured_action_dim = model_values.get("action_dim")
    if configured_action_dim is None:
        model_values["action_dim"] = dimensions["action_dim"]
    elif action_type == "discrete" and int(configured_action_dim) < int(
        dimensions["action_dim"]
    ):
        raise ValueError(
            "Configured discrete action_dim cannot represent the largest observed id: "
            f"{configured_action_dim} vs required {dimensions['action_dim']}"
        )
    elif action_type == "continuous" and int(configured_action_dim) != int(
        dimensions["action_dim"]
    ):
        raise ValueError(
            "Configured action_dim does not match data: "
            f"{configured_action_dim} vs {dimensions['action_dim']}"
        )
    model_values["action_type"] = action_type
    WorldModelTrainer._seed_everything(training_config.seed)
    model = create_world_model(backend, model_values)
    model_config = model.config

    dataset_kwargs = {
        "sequence_length": data_config.sequence_length,
        "stride": data_config.stride,
        "action_type": model_config.action_type,
        "action_dim": model_config.action_dim,
        "normalize_images": data_config.normalize_images,
        "pad_short_trajectories": data_config.pad_short_trajectories,
    }
    train_dataset = WorldModelTrajectoryDataset(train_trajectories, **dataset_kwargs)
    validation_dataset = (
        WorldModelTrajectoryDataset(validation_trajectories, **dataset_kwargs)
        if validation_trajectories
        else None
    )
    summary = {
        "status": "ready",
        "backend": backend,
        "train_path": os.path.abspath(data_config.train_path),
        "validation_path": (
            os.path.abspath(data_config.validation_path)
            if data_config.validation_path
            else None
        ),
        "model": model_config.to_dict(),
        "parameters": model.parameter_count(),
        "train_data": train_dataset.summary(),
        "validation_data": (
            validation_dataset.summary() if validation_dataset is not None else None
        ),
        "training": training_config.to_dict(),
    }
    return PreparedWorldModelRun(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        data_config=data_config,
        training_config=training_config,
        source_config=source_config,
        summary=summary,
    )


def train_world_model_from_config(
    config: Union[str, os.PathLike, Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """High-level YAML/JSON/dict world-model training entry point."""

    prepared = prepare_world_model_training(config)
    if dry_run:
        return {**prepared.summary, "status": "dry_run"}
    trainer = WorldModelTrainer(
        prepared.model,
        prepared.train_dataset,
        prepared.training_config,
        validation_dataset=prepared.validation_dataset,
    )
    result = trainer.train()
    result["data"] = {
        "train": prepared.train_dataset.summary(),
        "validation": (
            prepared.validation_dataset.summary()
            if prepared.validation_dataset is not None
            else None
        ),
    }
    return result


def train_world_model(
    config: Union[str, os.PathLike, Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Alias for :func:`train_world_model_from_config`."""

    return train_world_model_from_config(config, dry_run=dry_run)


def _load_run_config(
    config: Union[str, os.PathLike, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str]]:
    if isinstance(config, dict):
        return dict(config), None
    path = os.fspath(config)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"World-model config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        if path.lower().endswith((".yaml", ".yml")):
            import yaml

            payload = yaml.safe_load(file) or {}
        else:
            payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("World-model config root must be a mapping")
    return payload, os.path.dirname(os.path.abspath(path))


def _resolve_input_path(path: str, config_directory: Optional[str]) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path))
    if os.path.isfile(expanded) or os.path.isabs(expanded) or config_directory is None:
        return expanded
    relative_to_config = os.path.join(config_directory, expanded)
    return relative_to_config if os.path.isfile(relative_to_config) else expanded


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested or "auto").lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def _resolve_amp(
    requested: str,
    device: torch.device,
) -> Tuple[Optional[torch.dtype], bool]:
    if device.type != "cuda" or requested == "no":
        return None, False
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 mixed precision was requested but is not supported")
        return torch.bfloat16, False
    if requested == "fp16":
        return torch.float16, True
    return None, False


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().norm(2).cpu()) ** 2
    return math.sqrt(squared)


def _dtype_name(dtype: Optional[torch.dtype]) -> str:
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    return "no"


def _torch_load(path: str, map_location: Any, weights_only: bool) -> Any:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=weights_only,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _reject_unknown_fields(cls: Any, values: Dict[str, Any], label: str) -> None:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: " + ", ".join(unknown))
