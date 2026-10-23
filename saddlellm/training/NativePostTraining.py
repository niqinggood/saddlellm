"""Full-parameter post-training for native SaddleLLM checkpoints.

The first native path intentionally supports the two objectives required for
an end-to-end model lifecycle: supervised fine-tuning and DPO.  Adapter/QLoRA
training and additional preference objectives stay behind explicit capability
checks until their checkpoint semantics are implemented.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import TrainingArguments

from ..data.PostTrainingData import PostTrainingDataAdapter
from ..models.ModelAdapter import SaddleModelAdapter
from .NativeTrainer import SaddleTrainer
from .Retention import ReferenceKLMixin


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _activate_local_device() -> None:
    if not torch.cuda.is_available():
        return
    raw = os.environ.get("LOCAL_RANK", "0")
    try:
        local_rank = int(raw)
    except ValueError as exc:
        raise ValueError(f"LOCAL_RANK must be an integer, got {raw!r}") from exc
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK={local_rank} cannot address only "
            f"{torch.cuda.device_count()} visible CUDA device(s)"
        )
    torch.cuda.set_device(local_rank)


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


def _sft_features(
    record: Dict[str, Any], tokenizer: Any, max_length: int
) -> List[Dict[str, List[int]]]:
    normalized = PostTrainingDataAdapter.normalize_record(record, task="sft")
    if normalized is None:
        return []
    if normalized.get("text"):
        input_ids = _encode(tokenizer, str(normalized["text"]), max_length)
        return (
            [{"input_ids": input_ids, "labels": list(input_ids)}]
            if len(input_ids) > 1
            else []
        )

    messages = list(normalized.get("messages") or [])
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        return []

    features: List[Dict[str, List[int]]] = []
    for assistant_index in assistant_indices:
        context = messages[:assistant_index]
        supervised = messages[: assistant_index + 1]
        prompt_text = _format_messages(
            tokenizer, context, add_generation_prompt=True
        )
        full_text = _format_messages(
            tokenizer, supervised, add_generation_prompt=False
        )
        input_ids = _encode(tokenizer, full_text, max_length)
        prompt_ids = _encode(tokenizer, prompt_text, max_length)
        prompt_length = _common_prefix_length(prompt_ids, input_ids)
        if prompt_length >= len(input_ids):
            # The answer was fully truncated or the template could not provide
            # a trustworthy response boundary.  Dropping it is safer than
            # silently supervising a prompt token.
            continue
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if len(input_ids) > 1 and any(label != -100 for label in labels):
            features.append({"input_ids": input_ids, "labels": labels})
    return features


def _preference_features(
    record: Dict[str, Any], tokenizer: Any, max_length: int, max_prompt_length: int
) -> Optional[Dict[str, List[int]]]:
    if (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2
            or isinstance(max_prompt_length, bool) or not isinstance(max_prompt_length, int)
            or max_prompt_length < 1):
        raise ValueError("DPO max_length must be >= 2 and max_prompt_length must be a positive integer")
    normalized = PostTrainingDataAdapter.normalize_record(record, task="dpo")
    if normalized is None:
        return None
    prompt_text = str(normalized["prompt"])
    answers = [str(normalized[name]) for name in ("chosen", "rejected")]
    if any(not answer.strip() for answer in answers):
        return None
    separator = "" if prompt_text.endswith((" ", "\n", "\t")) else " "
    context_text = prompt_text + separator

    def encode_full(text: str) -> List[int]:
        return list(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    # Find the response boundary BEFORE applying either length budget. Capping
    # only the mask length leaves long prompt tokens in the supervised answer.
    # Joint tokenization preserves tokens merged across the text boundary; the
    # two alternatives share exactly the same retained causal context.
    prompt_ids = encode_full(context_text)
    alternatives = [encode_full(context_text + answer) for answer in answers]
    boundary = min(_common_prefix_length(prompt_ids, ids) for ids in alternatives)
    if boundary == 0 or any(boundary >= len(ids) for ids in alternatives):
        return None
    # Keep the recent prompt context, reserving at least one response token.
    # Never rescue an empty/truncated answer by supervising a prompt token.
    retained_prompt = prompt_ids[:boundary][-min(max_prompt_length, max_length - 1):]
    response_budget = max_length - len(retained_prompt)

    def completion(ids: List[int]) -> Dict[str, List[int]]:
        response = ids[boundary:boundary + response_budget]
        return {"input_ids": retained_prompt + response,
                "labels": [-100] * len(retained_prompt) + response}

    chosen, rejected = [completion(ids) for ids in alternatives]
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


class NativeRetentionSFTTrainer(ReferenceKLMixin, SaddleTrainer):
    """Native SFT with explicit full-logit reference KL during training."""


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
    validation_dataset_path: Optional[str] = None
    load_best_model_at_end: bool = True
    save_total_limit: int = 3
    eval_steps: int = 0
    gradient_checkpointing: bool = True
    report_to: str = "none"
    seed: int = 42
    ddp_backend: Optional[str] = None
    replay_data_path: Optional[str] = None
    replay_ratio: float = 0.0
    replay_count: Optional[int] = None
    reference_model_path: Optional[str] = None
    retention_kl_weight: float = 0.0
    retention_temperature: float = 1.0


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
    validation_dataset_path: Optional[str] = None
    load_best_model_at_end: bool = True
    save_total_limit: int = 3
    eval_steps: int = 0
    gradient_checkpointing: bool = True
    report_to: str = "none"
    seed: int = 42
    ddp_backend: Optional[str] = None


def _training_arguments(config: Any, *, remove_unused_columns: bool) -> TrainingArguments:
    device = _device()
    if config.max_steps > 0 and config.warmup_steps >= config.max_steps:
        raise ValueError(
            "Native post-training requires warmup_steps < max_steps; otherwise "
            "the entire short run can execute at zero learning rate"
        )
    has_eval = bool(
        config.validation_dataset_path
        or (config.validation_split and 0 < config.validation_split < 1)
    )
    eval_strategy = "steps" if has_eval and config.eval_steps > 0 else ("epoch" if has_eval else "no")
    load_best = bool(has_eval and config.load_best_model_at_end)
    if (
        load_best
        and eval_strategy == "steps"
        and config.save_steps % config.eval_steps != 0
    ):
        raise ValueError(
            "save_steps must be a multiple of eval_steps when selecting the "
            "best native post-training checkpoint"
        )
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
        save_strategy=(
            eval_strategy
            if load_best
            else ("steps" if config.save_steps > 0 else "no")
        ),
        logging_steps=max(1, config.logging_steps),
        eval_strategy=eval_strategy,
        eval_steps=config.eval_steps if config.eval_steps > 0 else None,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=config.report_to,
        remove_unused_columns=remove_unused_columns,
        prediction_loss_only=not remove_unused_columns,
        label_names=(
            None
            if remove_unused_columns
            else ["chosen_labels", "rejected_labels"]
        ),
        dataloader_num_workers=0,
        dataloader_pin_memory=device == "cuda",
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=device == "cuda" and not torch.cuda.is_bf16_supported(),
        use_cpu=device == "cpu",
        ddp_backend=config.ddp_backend,
        seed=config.seed,
    )


def _split_features(features: List[Dict[str, List[int]]], fraction: float, seed: int, *, groups=None):
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 <= fraction < 1):
        raise ValueError("validation_split must be a finite fraction in [0, 1)")
    if not fraction:
        return _FeatureDataset(features), None
    groups = list(range(len(features))) if groups is None else list(groups)
    if len(groups) != len(features):
        raise ValueError("validation_split groups must match tokenized samples")
    unique_groups = list(dict.fromkeys(groups))
    if len(unique_groups) < 2:
        raise ValueError("validation_split requires at least two usable source records")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(unique_groups), generator=generator).tolist()
    eval_size = min(len(unique_groups) - 1, max(1, int(math.ceil(len(unique_groups) * fraction))))
    eval_groups = {unique_groups[index] for index in order[:eval_size]}
    train = [item for item, group in zip(features, groups) if group not in eval_groups]
    evaluation = [item for item, group in zip(features, groups) if group in eval_groups]
    return _FeatureDataset(train), _FeatureDataset(evaluation)


def train_native_sft(config: NativeSFTConfig) -> Dict[str, Any]:
    from ..data.ReplayData import (
        sft_source_groups,
        validate_replay_config,
    )
    from .Retention import validate_reference_tokenizers, validate_retention_config
    from .TrainingProgress import LocalProgressCallback

    replay_enabled = validate_replay_config(
        config.replay_data_path, config.replay_ratio, config.replay_count
    )
    retention_enabled = validate_retention_config(
        config.reference_model_path, config.retention_kl_weight, config.retention_temperature
    )
    _activate_local_device()
    adapter = SaddleModelAdapter()
    adapter.validate_stage("sft")
    tokenizer = adapter.load_tokenizer(config.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    records = PostTrainingDataAdapter.load_records(config.dataset_path)
    source_groups = sft_source_groups(records)
    features, feature_groups, feature_sources = [], [], []
    for source_index, record in enumerate(records):
        turns = _sft_features(record, tokenizer, config.max_length)
        features.extend(turns)
        feature_groups.extend([source_groups[source_index]] * len(turns))
        feature_sources.extend([source_index] * len(turns))
    if not features:
        raise ValueError("Native SFT dataset produced zero usable tokenized samples")
    if config.validation_dataset_path and config.validation_split:
        raise ValueError(
            "Use either validation_dataset_path or validation_split for native "
            "SFT, not both."
        )
    validation_records = []
    source_train_records = records
    if config.validation_dataset_path:
        validation_records = PostTrainingDataAdapter.load_records(
            config.validation_dataset_path
        )
        validation_features = [
            feature
            for record in validation_records
            for feature in _sft_features(record, tokenizer, config.max_length)
        ]
        if not validation_features:
            raise ValueError(
                "Native SFT validation dataset produced zero usable tokenized samples"
            )
        train_dataset = _FeatureDataset(features)
        eval_dataset = _FeatureDataset(validation_features)
    else:
        train_dataset, eval_dataset = _split_features(
            features, config.validation_split, config.seed, groups=feature_groups
        )
        if eval_dataset is not None:
            held_out_features = {id(item) for item in eval_dataset.features}
            held_out_sources = {
                source_index for feature, source_index in zip(features, feature_sources)
                if id(feature) in held_out_features
            }
            validation_records = [records[index] for index in sorted(held_out_sources)]
            source_train_records = [
                record for index, record in enumerate(records)
                if index not in held_out_sources
            ]
    replay_info = {"enabled": False}
    if replay_enabled:
        from ..data.ReplayData import (
            filter_replay_records,
            mix_replay_samples,
            replay_source_fingerprint,
        )

        replay_records = PostTrainingDataAdapter.load_records(config.replay_data_path)
        eligible, filtering = filter_replay_records(replay_records, validation_records)
        replay_features = [
            feature for record in eligible
            for feature in _sft_features(record, tokenizer, config.max_length)
        ]
        mixed, replay_info = mix_replay_samples(
            train_dataset.features, replay_features,
            replay_ratio=config.replay_ratio, replay_count=config.replay_count,
            seed=config.seed,
            validation_samples=eval_dataset.features if eval_dataset is not None else [],
        )
        replay_info.update(filtering)
        replay_info["data_path"] = os.path.abspath(config.replay_data_path)
        replay_info["source_fingerprint"] = replay_source_fingerprint(replay_records)
        train_dataset = _FeatureDataset(mixed)
    split_audit = None
    if eval_dataset is not None:
        from ..data.TrainingSplitAudit import audit_record_splits

        split_audit = audit_record_splits(
            source_train_records,
            validation_records,
            train_features=train_dataset.features,
            validation_features=eval_dataset.features,
            report_path=os.path.join(config.output_path, "split_audit.json"),
            protected_paths=(config.dataset_path, config.validation_dataset_path, config.replay_data_path),
        )
    model = adapter.load_model(config.model_path, map_location="cpu")
    retention_info = {"enabled": False}
    retention_kwargs = {}
    if retention_enabled:
        reference_path = config.reference_model_path or config.model_path
        reference_tokenizer = adapter.load_tokenizer(reference_path, local_files_only=True)
        if reference_tokenizer.pad_token_id is None and reference_tokenizer.eos_token_id is not None:
            reference_tokenizer.pad_token = reference_tokenizer.eos_token
        validate_reference_tokenizers(tokenizer, reference_tokenizer)
        reference_model = adapter.load_model(reference_path, map_location="cpu")
        if model.config.vocab_size != reference_model.config.vocab_size:
            raise ValueError("Retention student and reference models must have matching output vocabulary sizes")
        retention_kwargs = {
            "reference_model": reference_model,
            "retention_kl_weight": config.retention_kl_weight,
            "retention_temperature": config.retention_temperature,
        }
        retention_info = {
            "enabled": True, "reference_model_path": reference_path,
            "kl_weight": config.retention_kl_weight, "temperature": config.retention_temperature,
            "direction": "reference_to_student", "applies_to": "training_supervised_tokens",
            "requires_full_logits": True,
        }
    trainer_class = NativeRetentionSFTTrainer if retention_enabled else SaddleTrainer
    trainer = trainer_class(
        model=model,
        args=_training_arguments(config, remove_unused_columns=True),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=NativeSFTCollator(tokenizer),
        processing_class=tokenizer,
        **retention_kwargs,
    )
    trainer.add_callback(LocalProgressCallback(config.output_path))
    result = trainer.train()
    if retention_enabled:
        retention_info.update(trainer.retention_metrics())
    trainer.save_model(config.output_path)
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_path)
        if replay_enabled:
            with open(os.path.join(config.output_path, "replay_manifest.json"), "w", encoding="utf-8") as handle:
                json.dump(replay_info, handle, ensure_ascii=False, indent=2)
        if retention_enabled:
            with open(os.path.join(config.output_path, "retention_manifest.json"), "w", encoding="utf-8") as handle:
                json.dump(retention_info, handle, ensure_ascii=False, indent=2)
    from .DistributedRuntime import distributed_barrier

    distributed_barrier()
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
        "validation_dataset_path": config.validation_dataset_path,
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "replay": replay_info,
        "retention": retention_info,
        "split_audit": split_audit,
    }


def train_native_dpo(config: NativeDPOConfig) -> Dict[str, Any]:
    from .TrainingProgress import LocalProgressCallback

    _activate_local_device()
    adapter = SaddleModelAdapter()
    adapter.validate_stage("preference", method="dpo")
    tokenizer = adapter.load_tokenizer(config.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    records = PostTrainingDataAdapter.load_records(config.dataset_path)
    features, usable_records = [], []
    for record in records:
        feature = _preference_features(
            record, tokenizer, config.max_length, config.max_prompt_length
        )
        if feature is not None:
            features.append(feature)
            usable_records.append(record)
    if not features:
        raise ValueError("Native DPO dataset produced zero usable tokenized samples")
    if config.validation_dataset_path and config.validation_split:
        raise ValueError(
            "Use either validation_dataset_path or validation_split for native "
            "DPO, not both."
        )
    if config.validation_dataset_path:
        validation_records = PostTrainingDataAdapter.load_records(
            config.validation_dataset_path
        )
        validation_features = [
            feature
            for record in validation_records
            if (
                feature := _preference_features(
                    record, tokenizer, config.max_length, config.max_prompt_length
                )
            )
            is not None
        ]
        if not validation_features:
            raise ValueError(
                "Native DPO validation dataset produced zero usable tokenized samples"
            )
        train_dataset = _FeatureDataset(features)
        eval_dataset = _FeatureDataset(validation_features)
    else:
        from ..data.GroupSplit import source_record_groups

        feature_groups = source_record_groups(
            usable_records,
            prompts=[PostTrainingDataAdapter._prompt_from_record(record) for record in usable_records],
        )
        train_dataset, eval_dataset = _split_features(
            features, config.validation_split, config.seed, groups=feature_groups
        )
    split_audit = None
    if eval_dataset is not None:
        from ..data.TrainingSplitAudit import audit_record_splits

        # Split datasets retain the original feature objects. Recover their
        # source records so the audit checks identities as well as tokens.
        training_feature_ids = {id(feature) for feature in train_dataset.features}
        source_train_records = [
            record for feature, record in zip(features, usable_records)
            if id(feature) in training_feature_ids
        ]
        if not config.validation_dataset_path:
            validation_records = [
                record for feature, record in zip(features, usable_records)
                if id(feature) not in training_feature_ids
            ]
        split_audit = audit_record_splits(
            source_train_records,
            validation_records,
            task="dpo",
            train_features=train_dataset.features,
            validation_features=eval_dataset.features,
            report_path=os.path.join(config.output_path, "split_audit.json"),
            protected_paths=(config.dataset_path, config.validation_dataset_path),
        )
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
    trainer.add_callback(LocalProgressCallback(config.output_path))
    result = trainer.train()
    trainer.save_model(config.output_path)
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_path)
    from .DistributedRuntime import distributed_barrier

    distributed_barrier()
    return {
        "output_path": os.path.abspath(config.output_path),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "global_step": int(trainer.state.global_step),
        "training_loss": float(result.training_loss),
        "objective": "dpo",
        "backend": "saddle",
        "beta": config.beta,
        "split_audit": split_audit,
        "use_lora": False,
        "use_qlora": False,
        "validation_dataset_path": config.validation_dataset_path,
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_metric": getattr(trainer.state, "best_metric", None),
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
