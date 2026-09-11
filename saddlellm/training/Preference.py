"""Unified preference post-training for DPO/ORPO/KTO.

The API normalizes common preference schemas and selects the available TRL
trainer at runtime.  It is intended to sit after SFT or rollout filtering.
"""
import inspect
import importlib
import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

from .PostTrainingCompatibility import (
    format_post_training_runtime_error,
    instantiate_supported,
    supported_kwargs,
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
    trust_remote_code: bool = True
    local_files_only: bool = False
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10
    validation_split: float = 0.0
    eval_steps: int = 0
    report_to: str = "none"
    seed: int = 42
    gradient_checkpointing: bool = True
    resume_from_checkpoint: Optional[Union[bool, str]] = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _maybe_quantization_config(config: PreferenceTrainConfig, device: str):
    if not (config.use_lora and config.use_qlora):
        return None
    if device != "cuda":
        raise RuntimeError(
            "QLoRA was requested, but CUDA is unavailable. Set `qlora: false` "
            "and `lora: true` for regular LoRA, or run on a CUDA device."
        )
    try:
        import bitsandbytes  # noqa: F401 - validates the native runtime

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if _bf16() else torch.float16,
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
    eval_strategy = "no"
    if config.validation_split and 0 < config.validation_split < 1:
        eval_strategy = "steps" if config.eval_steps > 0 else "epoch"
    kwargs = {
        "output_dir": config.output_path,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "save_steps": config.save_steps,
        "save_strategy": "steps",
        "logging_steps": config.logging_steps,
        "report_to": config.report_to,
        "bf16": _bf16(),
        "fp16": _device() == "cuda" and not _bf16(),
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": _device() == "cuda",
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_strategy": eval_strategy,
        "evaluation_strategy": eval_strategy,
        "eval_steps": config.eval_steps if config.eval_steps > 0 else None,
        "use_cpu": _device() == "cpu",
        "seed": config.seed,
        "beta": config.beta,
        "max_length": config.max_length,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": max(1, config.max_length - config.max_prompt_length),
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


def train_preference(config: PreferenceTrainConfig):
    device = _device()
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

    dataset_dict = _load_dataset(config.dataset_path)
    dataset = dataset_dict["train"] if "train" in dataset_dict else dataset_dict
    if not isinstance(dataset, Dataset):
        raise ValueError("Preference dataset must resolve to a HuggingFace Dataset")
    dataset = dataset.map(
        lambda batch: _normalize_preference_batch(batch, tokenizer, config.method),
        batched=True,
        remove_columns=dataset.column_names,
    )
    if len(dataset) == 0:
        raise ValueError(f"Preference dataset produced zero usable samples after normalization: {config.dataset_path}")

    eval_dataset = None
    if config.validation_split and 0 < config.validation_split < 1:
        split = dataset.train_test_split(test_size=config.validation_split, seed=config.seed)
        dataset = split["train"]
        eval_dataset = split["test"]

    quant_config = _maybe_quantization_config(config, device)

    model_kwargs = {
        "dtype": torch.bfloat16 if _bf16() else (torch.float16 if device == "cuda" else torch.float32),
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    if device == "cuda" and quant_config is not None:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
    if not (device == "cuda" and quant_config is not None):
        model = model.to(device)
    if quant_config is not None and config.use_lora and prepare_model_for_kbit_training is not None:
        prepare_kwargs = supported_kwargs(
            prepare_model_for_kbit_training,
            {"use_gradient_checkpointing": config.gradient_checkpointing},
        )
        model = prepare_model_for_kbit_training(model, **prepare_kwargs)
    if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config") and config.gradient_checkpointing:
        model.config.use_cache = False

    ref_model = None
    if config.method in {"dpo", "kto"} and not config.use_lora:
        ref_model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
        if not (device == "cuda" and quant_config is not None):
            ref_model = ref_model.to(device)

    peft_config = None
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_path)
    tokenizer.save_pretrained(config.output_path)
    return {
        "output_path": config.output_path,
        "train_samples": len(dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "method": config.method,
        "use_lora": config.use_lora,
        "use_qlora": config.use_qlora,
        "max_length": config.max_length,
    }


class PreferenceTrainer:
    def __init__(self, config: PreferenceTrainConfig):
        self.config = config

    def train(self):
        return train_preference(self.config)
