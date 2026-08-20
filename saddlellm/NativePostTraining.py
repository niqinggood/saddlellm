"""Full-parameter post-training for native SaddleLLM checkpoints.

The first native path intentionally supports the two objectives required for
an end-to-end model lifecycle: supervised fine-tuning and DPO.  Adapter/QLoRA
training and additional preference objectives stay behind explicit capability
checks until their checkpoint semantics are implemented.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import TrainingArguments

from .ModelAdapter import SaddleModelAdapter
from .NativeTrainer import SaddleTrainer
from .PostTrainingData import PostTrainingDataAdapter


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _format_messages(tokenizer: Any, messages: Sequence[Dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except (ValueError, TypeError, AttributeError):
            pass
    lines = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
    if add_generation_prompt:
        lines.append("assistant:")
    return "\n".join(lines)


def _encode(tokenizer: Any, text: str, max_length: int) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    return list(encoded["input_ids"])


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def _sft_features(record: Dict[str, Any], tokenizer: Any, max_length: int) -> Optional[Dict[str, List[int]]]:
    normalized = PostTrainingDataAdapter.normalize_record(record, task="sft")
    if normalized is None:
        return None
    if normalized.get("text"):
        input_ids = _encode(tokenizer, str(normalized["text"]), max_length)
        return {"input_ids": input_ids, "labels": list(input_ids)} if len(input_ids) > 1 else None

    messages = list(normalized.get("messages") or [])
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        return None
    # Train the final assistant response.  This keeps prompt masking correct for
    # both two-turn instruction data and multi-turn chat records.
    assistant_index = assistant_indices[-1]
    context = messages[:assistant_index]
    supervised = messages[: assistant_index + 1]
    prompt_text = _format_messages(tokenizer, context, add_generation_prompt=True)
    full_text = _format_messages(tokenizer, supervised, add_generation_prompt=False)
    input_ids = _encode(tokenizer, full_text, max_length)
    prompt_ids = _encode(tokenizer, prompt_text, max_length)
    prompt_length = _common_prefix_length(prompt_ids, input_ids)
    if prompt_length >= len(input_ids):
        # Some generic tokenizers do not have a compatible chat template.  In
        # that case retain at least the final token as a supervised target.
        prompt_length = max(0, len(input_ids) - 1)
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    return {"input_ids": input_ids, "labels": labels} if len(input_ids) > 1 else None


def _preference_features(
    record: Dict[str, Any], tokenizer: Any, max_length: int, max_prompt_length: int
) -> Optional[Dict[str, List[int]]]:
    normalized = PostTrainingDataAdapter.normalize_record(record, task="dpo")
    if normalized is None:
        return None
    prompt_text = str(normalized["prompt"])

    def completion(value: Any) -> Dict[str, List[int]]:
        completion_text = str(value)
        separator = "" if prompt_text.endswith((" ", "\n", "\t")) else " "
        full_text = f"{prompt_text}{separator}{completion_text}"
        ids = _encode(tokenizer, full_text, max_length)
        prompt_ids = _encode(tokenizer, prompt_text, min(max_prompt_length, max_length))
        prompt_length = min(_common_prefix_length(prompt_ids, ids), max_prompt_length)
        if prompt_length >= len(ids):
            prompt_length = max(0, len(ids) - 1)
        labels = [-100] * prompt_length + ids[prompt_length:]
        return {"input_ids": ids, "labels": labels}

    chosen = completion(normalized["chosen"])
    rejected = completion(normalized["rejected"])
    if not any(label != -100 for label in chosen["labels"]):
        return None
    if not any(label != -100 for label in rejected["labels"]):
        return None
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_labels": chosen["labels"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_labels": rejected["labels"],
    }


class _FeatureDataset(Dataset):
    def __init__(self, features: Sequence[Dict[str, List[int]]]) -> None:
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.features[index]


class NativeSFTCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Native SFT requires a tokenizer with pad_token_id or eos_token_id")

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class NativeDPOCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Native DPO requires a tokenizer with pad_token_id or eos_token_id")

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for prefix in ("chosen", "rejected"):
            key = f"{prefix}_input_ids"
            width = max(len(item[key]) for item in features)
            ids, masks, labels = [], [], []
            for item in features:
                padding = width - len(item[key])
                ids.append(item[key] + [self.pad_token_id] * padding)
                masks.append([1] * len(item[key]) + [0] * padding)
                labels.append(item[f"{prefix}_labels"] + [-100] * padding)
            result[key] = torch.tensor(ids, dtype=torch.long)
            result[f"{prefix}_attention_mask"] = torch.tensor(masks, dtype=torch.long)
            result[f"{prefix}_labels"] = torch.tensor(labels, dtype=torch.long)
        return result


def _sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_log_probs = shifted_logits.log_softmax(dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


class NativeDPOTrainer(SaddleTrainer):
    def __init__(self, *args: Any, reference_model: torch.nn.Module, beta: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model.eval()
        self.reference_model.requires_grad_(False)
        self.beta = float(beta)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        chosen = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        rejected = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        policy_chosen = _sequence_log_probs(chosen.logits, inputs["chosen_labels"])
        policy_rejected = _sequence_log_probs(rejected.logits, inputs["rejected_labels"])

        device = inputs["chosen_input_ids"].device
        if next(self.reference_model.parameters()).device != device:
            self.reference_model.to(device)
        with torch.no_grad():
            reference_chosen = self.reference_model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
            )
            reference_rejected = self.reference_model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
            )
            ref_chosen = _sequence_log_probs(reference_chosen.logits, inputs["chosen_labels"])
            ref_rejected = _sequence_log_probs(reference_rejected.logits, inputs["rejected_labels"])

        logits = self.beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
        loss = -F.logsigmoid(logits).mean()
        if return_outputs:
            return loss, {
                "logits": chosen.logits,
                "chosen_reward": self.beta * (policy_chosen - ref_chosen).detach(),
                "rejected_reward": self.beta * (policy_rejected - ref_rejected).detach(),
            }
        return loss


@dataclass
class NativeSFTConfig:
    model_path: str
    dataset_path: str
    output_path: str
    learning_rate: float = 2e-4
    batch_size: int = 1
    num_epochs: int = 1
    max_steps: int = -1
    max_length: int = 2048
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    save_steps: int = 200
    logging_steps: int = 10
    validation_split: float = 0.0
    eval_steps: int = 0
    gradient_checkpointing: bool = True
    report_to: str = "none"
    seed: int = 42


@dataclass
class NativeDPOConfig:
    model_path: str
    dataset_path: str
    output_path: str
    beta: float = 0.1
    learning_rate: float = 5e-6
    batch_size: int = 1
    num_epochs: int = 1
    max_steps: int = -1
    max_length: int = 2048
    max_prompt_length: int = 1024
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    save_steps: int = 200
    logging_steps: int = 10
    validation_split: float = 0.0
    eval_steps: int = 0
    gradient_checkpointing: bool = True
    report_to: str = "none"
    seed: int = 42


def _training_arguments(config: Any, *, remove_unused_columns: bool) -> TrainingArguments:
    device = _device()
    if config.max_steps > 0 and config.warmup_steps >= config.max_steps:
        raise ValueError(
            "Native post-training requires warmup_steps < max_steps; otherwise "
            "the entire short run can execute at zero learning rate"
        )
    has_eval = bool(config.validation_split and 0 < config.validation_split < 1)
    eval_strategy = "steps" if has_eval and config.eval_steps > 0 else ("epoch" if has_eval else "no")
    return TrainingArguments(
        output_dir=config.output_path,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        save_strategy="steps" if config.save_steps > 0 else "no",
        logging_steps=max(1, config.logging_steps),
        eval_strategy=eval_strategy,
        eval_steps=config.eval_steps if config.eval_steps > 0 else None,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=config.report_to,
        remove_unused_columns=remove_unused_columns,
        dataloader_num_workers=0,
        dataloader_pin_memory=device == "cuda",
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=device == "cuda" and not torch.cuda.is_bf16_supported(),
        use_cpu=device == "cpu",
        seed=config.seed,
    )


def _split_features(features: List[Dict[str, List[int]]], fraction: float, seed: int):
    if not fraction or not 0 < fraction < 1:
        return _FeatureDataset(features), None
    if len(features) < 2:
        raise ValueError("validation_split requires at least two usable samples")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(features), generator=generator).tolist()
    eval_size = max(1, int(math.ceil(len(features) * fraction)))
    eval_indices = set(order[:eval_size])
    train = [item for index, item in enumerate(features) if index not in eval_indices]
    evaluation = [item for index, item in enumerate(features) if index in eval_indices]
    return _FeatureDataset(train), _FeatureDataset(evaluation)


def train_native_sft(config: NativeSFTConfig) -> Dict[str, Any]:
    adapter = SaddleModelAdapter()
    adapter.validate_stage("sft")
    tokenizer = adapter.load_tokenizer(config.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    records = PostTrainingDataAdapter.load_records(config.dataset_path)
    features = [
        feature
        for record in records
        if (feature := _sft_features(record, tokenizer, config.max_length)) is not None
    ]
    if not features:
        raise ValueError("Native SFT dataset produced zero usable tokenized samples")
    train_dataset, eval_dataset = _split_features(features, config.validation_split, config.seed)
    model = adapter.load_model(config.model_path, map_location="cpu")
    trainer = SaddleTrainer(
        model=model,
        args=_training_arguments(config, remove_unused_columns=True),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=NativeSFTCollator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(config.output_path)
    tokenizer.save_pretrained(config.output_path)
    trainer.save_state()
    return {
        "output_path": os.path.abspath(config.output_path),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "global_step": int(trainer.state.global_step),
        "training_loss": float(result.training_loss),
        "objective": "sft",
        "backend": "saddle",
        "use_lora": False,
        "use_qlora": False,
    }


def train_native_dpo(config: NativeDPOConfig) -> Dict[str, Any]:
    adapter = SaddleModelAdapter()
    adapter.validate_stage("preference", method="dpo")
    tokenizer = adapter.load_tokenizer(config.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    records = PostTrainingDataAdapter.load_records(config.dataset_path)
    features = [
        feature
        for record in records
        if (
            feature := _preference_features(
                record, tokenizer, config.max_length, config.max_prompt_length
            )
        )
        is not None
    ]
    if not features:
        raise ValueError("Native DPO dataset produced zero usable tokenized samples")
    train_dataset, eval_dataset = _split_features(features, config.validation_split, config.seed)
    model = adapter.load_model(config.model_path, map_location="cpu")
    reference_model = copy.deepcopy(model)
    trainer = NativeDPOTrainer(
        model=model,
        reference_model=reference_model,
        beta=config.beta,
        args=_training_arguments(config, remove_unused_columns=False),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=NativeDPOCollator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(config.output_path)
    tokenizer.save_pretrained(config.output_path)
    trainer.save_state()
    return {
        "output_path": os.path.abspath(config.output_path),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "global_step": int(trainer.state.global_step),
        "training_loss": float(result.training_loss),
        "objective": "dpo",
        "backend": "saddle",
        "beta": config.beta,
        "use_lora": False,
        "use_qlora": False,
    }


__all__ = [
    "NativeSFTConfig",
    "NativeDPOConfig",
    "NativeSFTCollator",
    "NativeDPOCollator",
    "NativeDPOTrainer",
    "train_native_sft",
    "train_native_dpo",
]
