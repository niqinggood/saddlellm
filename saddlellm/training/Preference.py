"""Unified preference post-training for DPO/ORPO/KTO.

The API normalizes common preference schemas and selects the available TRL
trainer at runtime.  It is intended to sit after SFT or rollout filtering.
"""
import copy
import importlib
import inspect
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

from .PostTrainingCompatibility import (
    format_post_training_runtime_error,
    instantiate_supported,
    latest_eval_metrics,
    latest_train_metrics,
    metric_snapshot,
    resolve_post_training_precision,
    resolve_trainer_resume_checkpoint,
    supported_kwargs,
    validate_resume_training_args,
)

try:
    from peft import prepare_model_for_kbit_training
except Exception:  # pragma: no cover - old PEFT versions
    prepare_model_for_kbit_training = None

logger = logging.getLogger(__name__)


@dataclass
class PreferenceTrainConfig:
    model_path: str
    dataset_path: str
    output_path: str
    method: Literal["dpo", "orpo", "kto"] = "dpo"
    beta: float = 0.1
    learning_rate: float = 5e-6
    batch_size: int = 1
    num_epochs: int = 1
    max_steps: int = -1
    max_length: int = 2048
    max_prompt_length: int = 1024
    use_lora: bool = True
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[Union[str, List[str]]] = None
    trust_remote_code: bool = True
    local_files_only: bool = False
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10
    validation_split: float = 0.0
    validation_dataset_path: Optional[str] = None
    load_best_model_at_end: bool = True
    save_total_limit: int = 3
    eval_steps: int = 0
    report_to: str = "none"
    seed: int = 42
    gradient_checkpointing: bool = True
    resume_from_checkpoint: Optional[Union[bool, str]] = None
    precision: str = "auto"
    ddp_backend: Optional[str] = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _local_process_index() -> int:
    raw = os.environ.get("LOCAL_RANK", "0")
    try:
        rank = int(raw)
    except ValueError as exc:
        raise ValueError(f"LOCAL_RANK must be an integer, got {raw!r}") from exc
    if rank < 0:
        raise ValueError("LOCAL_RANK must be >= 0")
    if torch.cuda.is_available() and rank >= torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK={rank} cannot address only "
            f"{torch.cuda.device_count()} visible CUDA device(s)"
        )
    return rank


def _local_model_device(device: str) -> str:
    return f"cuda:{_local_process_index()}" if device == "cuda" else device


def _maybe_quantization_config(config: PreferenceTrainConfig, device: str):
    if not (config.use_lora and config.use_qlora):
        return None
    if device != "cuda":
        raise RuntimeError(
            "QLoRA was requested, but CUDA is unavailable. Set `qlora: false` "
            "and `lora: true` for regular LoRA, or run on a CUDA device."
        )
    _, compute_dtype, _, _ = resolve_post_training_precision(
        config.precision, device
    )
    try:
        import bitsandbytes  # noqa: F401 - validates the native runtime

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    except Exception as exc:
        raise RuntimeError(
            "QLoRA requires a working bitsandbytes installation compatible with "
            f"the current PyTorch/CUDA runtime. Original error: {exc}"
        ) from exc


def _load_dataset(path: str):
    if os.path.isdir(path):
        return load_dataset(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".json", ".jsonl"}:
        return load_dataset("json", data_files=path)
    if ext == ".csv":
        return load_dataset("csv", data_files=path)
    return load_dataset(path)


def _adapter_base_model(path: str) -> Optional[str]:
    adapter_config = Path(path).expanduser() / "adapter_config.json"
    if not adapter_config.is_file():
        return None
    try:
        payload = json.loads(adapter_config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid PEFT adapter config: {adapter_config}: {exc}") from exc
    base_model = payload.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"PEFT adapter config does not declare base_model_name_or_path: {adapter_config}"
        )
    return str(base_model)


def _normalize_preference_batch(batch, tokenizer, method: str):
    from ..data.PostTrainingData import PostTrainingDataAdapter

    n = len(next(iter(batch.values()))) if batch else 0
    rows = {"prompt": [], "chosen": [], "rejected": []}
    kto_rows = {"prompt": [], "completion": [], "label": []}

    for i in range(n):
        item = {k: v[i] for k, v in batch.items()}
        if method == "kto":
            normalized = PostTrainingDataAdapter.normalize_record(item, task="kto")
            if normalized is None:
                continue
            kto_rows["prompt"].append(str(normalized["prompt"]))
            kto_rows["completion"].append(str(normalized["completion"]))
            kto_rows["label"].append(bool(normalized["label"]))
        else:
            normalized = PostTrainingDataAdapter.normalize_record(item, task=method)
            if normalized is None:
                continue
            rows["prompt"].append(str(normalized["prompt"]))
            rows["chosen"].append(str(normalized["chosen"]))
            rows["rejected"].append(str(normalized["rejected"]))

    return kto_rows if method == "kto" else rows


def _trainer_class(method: str):
    mapping = {"dpo": "DPOTrainer", "orpo": "ORPOTrainer", "kto": "KTOTrainer"}
    name = mapping[method]
    try:
        # TRL 0.26 moved ORPO/KTO under experimental while retaining wrappers
        # at the top level. Import the concrete class to preserve signatures.
        if method in {"orpo", "kto"}:
            try:
                module = importlib.import_module(f"trl.experimental.{method}")
                trainer_cls = getattr(module, name)
                return trainer_cls
            except (ImportError, AttributeError):
                pass
        import trl

        trainer_cls = getattr(trl, name)
        return trainer_cls
    except Exception as exc:
        raise RuntimeError(format_post_training_runtime_error(name, exc)) from exc


def _training_args_class(method: str):
    mapping = {"dpo": "DPOConfig", "orpo": "ORPOConfig", "kto": "KTOConfig"}
    name = mapping[method]
    try:
        if method in {"orpo", "kto"}:
            try:
                module = importlib.import_module(f"trl.experimental.{method}")
                return getattr(module, name)
            except (ImportError, AttributeError):
                pass
        import trl

        return getattr(trl, name, TrainingArguments)
    except Exception as exc:
        raise RuntimeError(format_post_training_runtime_error(name, exc)) from exc


def _make_training_args(config: PreferenceTrainConfig):
    args_cls = _training_args_class(config.method)
    _, _, bf16, fp16 = resolve_post_training_precision(
        config.precision, _device()
    )
    has_eval = bool(
        config.validation_dataset_path
        or (config.validation_split and 0 < config.validation_split < 1)
    )
    eval_strategy = "no"
    if has_eval:
        eval_strategy = "steps" if config.eval_steps > 0 else "epoch"
    load_best = bool(has_eval and config.load_best_model_at_end)
    if (
        load_best
        and eval_strategy == "steps"
        and config.save_steps % config.eval_steps != 0
    ):
        raise ValueError(
            "save_steps must be a multiple of eval_steps when selecting the "
            "best preference checkpoint"
        )
    kwargs = {
        "output_dir": config.output_path,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "save_steps": config.save_steps,
        "save_strategy": eval_strategy if load_best else "steps",
        "logging_steps": config.logging_steps,
        "report_to": config.report_to,
        "bf16": bf16,
        "fp16": fp16,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": _device() == "cuda",
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_strategy": eval_strategy,
        "evaluation_strategy": eval_strategy,
        "eval_steps": config.eval_steps if config.eval_steps > 0 else None,
        "load_best_model_at_end": load_best,
        "metric_for_best_model": "eval_loss" if load_best else None,
        "greater_is_better": False if load_best else None,
        "save_total_limit": config.save_total_limit,
        "use_cpu": _device() == "cpu",
        "seed": config.seed,
        "beta": config.beta,
        "max_length": config.max_length,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": max(1, config.max_length - config.max_prompt_length),
        "ddp_backend": config.ddp_backend,
    }
    return instantiate_supported(args_cls, kwargs)


def _trainer_parameter_names(trainer_cls) -> set:
    """Find the first concrete constructor signature through wrapper classes."""
    for cls in trainer_cls.__mro__:
        parameters = inspect.signature(cls.__init__).parameters
        concrete = {
            name
            for name, parameter in parameters.items()
            if name != "self" and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        }
        if concrete:
            return set(parameters)
    return set(inspect.signature(trainer_cls.__init__).parameters)


def _build_trainer(trainer_cls, **kwargs):
    accepted = _trainer_parameter_names(trainer_cls)
    if "processing_class" in accepted and "tokenizer" in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    return trainer_cls(**supported_kwargs(trainer_cls.__init__, kwargs))


def _initial_adapter_reference(model, *, method: str, continuing_adapter: bool):
    """DPO after SFT must compare against that SFT policy, not bare weights."""
    if method != "dpo" or not continuing_adapter:
        return None
    reference = copy.deepcopy(model)
    reference.requires_grad_(False)
    reference.eval()
    return reference


def train_preference(config: PreferenceTrainConfig):
    device = _device()
    if device == "cuda":
        torch.cuda.set_device(_local_process_index())
    actual_precision, model_dtype, _, _ = resolve_post_training_precision(
        config.precision, device
    )
    if config.use_lora:
        from .PostTrainingCompatibility import stabilize_peft_optional_backends

        for warning in stabilize_peft_optional_backends():
            logger.warning(warning)

    from ..models.TokenizerLoader import load_tokenizer_compatible

    tokenizer = load_tokenizer_compatible(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.validation_dataset_path and config.validation_split:
        raise ValueError(
            "Use either validation_dataset_path or validation_split for preference "
            "training, not both."
        )
    if (
        config.validation_dataset_path
        and os.path.isfile(config.dataset_path)
        and os.path.isfile(config.validation_dataset_path)
        and os.path.abspath(config.dataset_path)
        == os.path.abspath(config.validation_dataset_path)
    ):
        raise ValueError("Preference training and validation files must be different.")

    def select_dataset(path: str, preferred_splits: tuple[str, ...]) -> Dataset:
        loaded = _load_dataset(path)
        if isinstance(loaded, Dataset):
            return loaded
        for split_name in preferred_splits:
            if split_name in loaded:
                selected = loaded[split_name]
                if isinstance(selected, Dataset):
                    return selected
        available = list(loaded.keys()) if hasattr(loaded, "keys") else []
        if len(available) == 1 and isinstance(loaded[available[0]], Dataset):
            return loaded[available[0]]
        raise ValueError(
            f"Preference dataset {path!r} has no usable split; available splits: "
            f"{available}"
        )

    def normalize_dataset(source: Dataset) -> Dataset:
        return source.map(
            lambda batch: _normalize_preference_batch(batch, tokenizer, config.method),
            batched=True,
            remove_columns=source.column_names,
        )

    source_dataset = select_dataset(config.dataset_path, ("train",))
    if not isinstance(source_dataset, Dataset):
        raise ValueError("Preference dataset must resolve to a HuggingFace Dataset")

    from ..data.GroupSplit import source_record_groups, split_group_indices
    from ..data.PostTrainingData import PostTrainingDataAdapter
    from ..data.TrainingSplitAudit import audit_record_splits

    if (
        isinstance(config.validation_split, bool)
        or not isinstance(config.validation_split, (int, float))
        or not math.isfinite(config.validation_split)
        or not 0 <= config.validation_split < 1
    ):
        raise ValueError("validation_split must be a finite number in [0, 1)")

    # Hold out source groups before normalization can discard metadata.
    training_source = source_dataset
    validation_source = None
    eval_dataset = None
    if config.validation_dataset_path:
        validation_source = select_dataset(
            config.validation_dataset_path, ("validation", "eval", "test", "train")
        )
    elif config.validation_split:
        groups = source_record_groups(
            source_dataset,
            prompts=[
                PostTrainingDataAdapter._prompt_from_record(row)
                for row in source_dataset
            ],
        )
        train_indices, validation_indices = split_group_indices(
            groups, config.validation_split, config.seed
        )
        training_source = source_dataset.select(train_indices)
        validation_source = source_dataset.select(validation_indices)

    split_audit = None
    if validation_source is not None:
        split_audit = audit_record_splits(
            training_source,
            validation_source,
            task=config.method,
            report_path=Path(config.output_path) / "split_audit.json",
            protected_paths=(config.dataset_path, config.validation_dataset_path),
        )
        eval_dataset = normalize_dataset(validation_source)
        if len(eval_dataset) == 0:
            raise ValueError(
                "Preference validation dataset produced zero usable samples after normalization"
            )

    dataset = normalize_dataset(training_source)
    if len(dataset) == 0:
        raise ValueError(
            f"Preference dataset produced zero usable samples after normalization: {config.dataset_path}"
        )

    quant_config = _maybe_quantization_config(config, device)

    model_kwargs = {
        "dtype": model_dtype,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    if device == "cuda" and quant_config is not None:
        # Keep each quantized model inside its owning worker. Automatic device
        # maps can otherwise make every DDP process span all visible GPUs.
        model_kwargs["device_map"] = {"": _local_process_index()}

    adapter_base_model = _adapter_base_model(config.model_path)
    if adapter_base_model and not config.use_lora:
        raise ValueError(
            "The upstream checkpoint is a PEFT adapter; preference training must "
            "keep use_lora=true or consume a previously merged full model."
        )
    from ..models.ModelLoader import resolve_model_reference

    model_source = resolve_model_reference(
        adapter_base_model or config.model_path,
        local_files_only=config.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if not (device == "cuda" and quant_config is not None):
        model = model.to(_local_model_device(device))
    if (
        quant_config is not None
        and config.use_lora
        and prepare_model_for_kbit_training is not None
    ):
        prepare_kwargs = supported_kwargs(
            prepare_model_for_kbit_training,
            {"use_gradient_checkpointing": config.gradient_checkpointing},
        )
        model = prepare_model_for_kbit_training(model, **prepare_kwargs)
    if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config") and config.gradient_checkpointing:
        model.config.use_cache = False

    continuing_adapter = adapter_base_model is not None
    if continuing_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            config.model_path,
            is_trainable=True,
        )

    ref_model = _initial_adapter_reference(
        model, method=config.method, continuing_adapter=continuing_adapter
    )
    if config.method in {"dpo", "kto"} and not config.use_lora:
        ref_model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        if not (device == "cuda" and quant_config is not None):
            ref_model = ref_model.to(_local_model_device(device))

    peft_config = None
    if config.use_lora and not continuing_adapter:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules
            or ["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )

    args = _make_training_args(config)

    trainer_cls = _trainer_class(config.method)
    trainer = _build_trainer(
        trainer_cls,
        model=model,
        ref_model=ref_model,
        args=args,
        beta=config.beta,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )
    prepared_split_audit = None
    if eval_dataset is not None:
        prepared_split_audit = audit_record_splits(
            [],
            [],
            task=config.method,
            train_features=trainer.train_dataset,
            validation_features=trainer.eval_dataset,
            report_path=Path(config.output_path) / "prepared_split_audit.json",
            protected_paths=(config.dataset_path, config.validation_dataset_path),
        )
    from .TrainingProgress import LocalProgressCallback

    trainer.add_callback(LocalProgressCallback(config.output_path))
    resumed_from_checkpoint = resolve_trainer_resume_checkpoint(
        config.resume_from_checkpoint,
        config.output_path,
    )
    validate_resume_training_args(resumed_from_checkpoint, args)
    train_output = trainer.train(resume_from_checkpoint=resumed_from_checkpoint)
    global_step = int(getattr(trainer.state, "global_step", 0))
    eval_metrics = latest_eval_metrics(getattr(trainer.state, "log_history", []))
    if eval_dataset is not None and eval_metrics.get("step") != global_step:
        eval_metrics = metric_snapshot(trainer.evaluate())
        eval_metrics.setdefault("step", global_step)
    trainer.save_model(config.output_path)
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_path)
    from .DistributedRuntime import distributed_barrier

    distributed_barrier()
    return {
        "output_path": config.output_path,
        "train_samples": len(dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "method": config.method,
        "use_lora": config.use_lora,
        "use_qlora": config.use_qlora,
        "continued_adapter": continuing_adapter,
        "reference_policy": "initial_adapter_snapshot"
        if config.method == "dpo" and continuing_adapter else "base_model",
        "max_length": config.max_length,
        "precision": actual_precision,
        "global_step": global_step,
        "train_metrics": metric_snapshot(getattr(train_output, "metrics", {})),
        "latest_train_metrics": latest_train_metrics(
            getattr(trainer.state, "log_history", [])
        ),
        "eval_metrics": eval_metrics,
        "validation_dataset_path": config.validation_dataset_path,
        "split_audit": split_audit,
        "prepared_split_audit": prepared_split_audit,
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "lora_target_modules": config.lora_target_modules,
        "resume_from_checkpoint": config.resume_from_checkpoint,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "trainer_state_path": os.path.join(config.output_path, "trainer_state.json"),
    }


class PreferenceTrainer:
    def __init__(self, config: PreferenceTrainConfig):
        self.config = config

    def train(self):
        return train_preference(self.config)
