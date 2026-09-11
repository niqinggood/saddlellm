"""VLA behavior-cloning trainer.

This trainer optimizes the language model to emit discretized robot action
tokens from normalized vision-language-action chat samples.  Image and proprio
metadata are kept in the batch, but this entry point intentionally starts with
the text/action-token objective so it can run on ordinary causal-LM backbones.
"""
import json
import logging
import os
import random
import inspect
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
from torch.utils.data import Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .VLA import (
    VLAActionSpace,
    VLAActionTokenizer,
    VLADataAdapter,
    VLACollatorConfig,
    VLADataCollator,
    VLADataset,
)

logger = logging.getLogger(__name__)


@dataclass
class VLASFTConfig:
    model_path: str
    dataset_path: str
    output_path: str
    image_root: Optional[str] = None
    action_space: VLAActionSpace = field(default_factory=VLAActionSpace)
    use_lora: bool = True
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    learning_rate: float = 2e-4
    batch_size: int = 1
    num_epochs: int = 1
    max_steps: int = -1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 50
    save_steps: int = 200
    logging_steps: int = 10
    eval_steps: int = 0
    validation_split: float = 0.0
    max_seq_length: int = 2048
    train_on_prompt: bool = False
    local_files_only: bool = False
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"
    seed: int = 42

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["action_space"] = self.action_space.to_dict()
        return data


def train_vla_sft(config: VLASFTConfig) -> Dict:
    os.makedirs(config.output_path, exist_ok=True)
    dataset_path = _ensure_normalized_dataset(config)
    dataset = VLADataset.from_file(dataset_path)
    if len(dataset) == 0:
        raise ValueError(f"VLA dataset is empty: {dataset_path}")

    train_dataset, eval_dataset = _split_dataset(dataset, config.validation_split, config.seed)
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": _dtype_for_device(device),
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    quantization_config = _maybe_quantization_config(config, device)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if device == "cuda" and quantization_config is not None:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
    action_tokenizer = VLAActionTokenizer(config.action_space)
    token_report = action_tokenizer.register_with_tokenizer(tokenizer, resize_model=model)
    if not (device == "cuda" and quantization_config is not None):
        model = model.to(device)
    if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False

    if config.use_lora:
        model = _maybe_wrap_lora(model, config)

    collator = VLADataCollator(
        tokenizer=tokenizer,
        config=VLACollatorConfig(
            max_seq_length=config.max_seq_length,
            train_on_prompt=config.train_on_prompt,
        ),
        action_tokenizer=action_tokenizer,
    )
    train_collator = _model_input_collator(collator)
    args = _make_training_args(config, device)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": train_collator,
    }
    trainer_parameters = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(
        **trainer_kwargs,
    )
    trainer.train()
    trainer.save_model(config.output_path)
    tokenizer.save_pretrained(config.output_path)

    result = {
        "output_path": config.output_path,
        "dataset_path": dataset_path,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "token_report": token_report,
    }
    with open(os.path.join(config.output_path, "vla_sft_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("VLA SFT model saved to: %s", config.output_path)
    return result


def _ensure_normalized_dataset(config: VLASFTConfig) -> str:
    if _looks_normalized(config.dataset_path):
        return config.dataset_path
    normalized_path = os.path.join(config.output_path, "vla_normalized.jsonl")
    VLADataAdapter.normalize_file(
        config.dataset_path,
        normalized_path,
        image_root=config.image_root,
        action_space=config.action_space,
    )
    return normalized_path


def _looks_normalized(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    first = json.loads(line)
                    return _record_is_normalized(first)
                return False
            if path.lower().endswith(".json"):
                data = json.load(f)
                first = data[0] if isinstance(data, list) and data else data
                return isinstance(first, dict) and _record_is_normalized(first)
    except Exception:
        return False
    return False


def _record_is_normalized(record: Dict) -> bool:
    return bool(record.get("messages")) and bool(record.get("images")) and record.get("task") == "vla_sft"


def _split_dataset(dataset: VLADataset, validation_split: float, seed: int):
    if not validation_split or validation_split <= 0 or validation_split >= 1 or len(dataset) < 2:
        return dataset, None
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, int(round(len(indices) * validation_split)))
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:] or indices[eval_size - 1:]
    return Subset(dataset, train_indices), Subset(dataset, eval_indices)


def _model_input_collator(collator: VLADataCollator):
    def wrapped(features):
        batch = collator(features)
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    return wrapped


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _dtype_for_device(device: str):
    if device == "cuda":
        return torch.bfloat16 if _supports_bf16() else torch.float16
    return torch.float32


def _maybe_quantization_config(config: VLASFTConfig, device: str):
    if not (config.use_lora and config.use_qlora and device == "cuda"):
        return None
    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if _supports_bf16() else torch.float16,
        )
    except Exception as exc:
        logger.warning("QLoRA quantization unavailable, falling back to regular LoRA: %s", exc)
        return None


def _maybe_wrap_lora(model, config: VLASFTConfig):
    try:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )
        return get_peft_model(model, peft_config)
    except Exception as exc:
        logger.warning("LoRA unavailable, falling back to full fine-tuning: %s", exc)
        return model


def _make_training_args(config: VLASFTConfig, device: str) -> TrainingArguments:
    bf16 = device == "cuda" and _supports_bf16()
    fp16 = device == "cuda" and not bf16
    kwargs = {
        "output_dir": config.output_path,
        "per_device_train_batch_size": config.batch_size,
        "num_train_epochs": config.num_epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "fp16": fp16,
        "bf16": bf16,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_strategy": "steps",
        "eval_steps": config.eval_steps if config.eval_steps else None,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_steps": config.warmup_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": config.report_to,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": config.seed,
    }
    eval_mode = "steps" if config.eval_steps and config.validation_split else "no"
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in accepted:
        kwargs["eval_strategy"] = eval_mode
    else:
        kwargs["evaluation_strategy"] = eval_mode
    kwargs = {key: value for key, value in kwargs.items() if key in accepted and value is not None}
    return TrainingArguments(**kwargs)
