"""Stable SFT/LoRA training entry points.

This module keeps the historical ``train_model`` function, but makes it usable
as a library API across CPU/CUDA, LoRA/QLoRA/full fine-tuning, and common SFT
dataset schemas.
"""
import argparse
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer

from .PostTrainingCompatibility import (
    instantiate_supported,
    resolve_post_training_precision,
    resolve_trainer_resume_checkpoint,
    supported_kwargs,
    validate_resume_training_args,
)
from .PostTrainingCompatibility import (
    latest_eval_metrics as _latest_eval_metrics,
)
from .PostTrainingCompatibility import (
    latest_train_metrics as _latest_train_metrics,
)
from .PostTrainingCompatibility import (
    metric_snapshot as _metric_snapshot,
)
from .TrainingProgress import LocalProgressCallback

try:
    from peft import prepare_model_for_kbit_training
except Exception:  # pragma: no cover - old PEFT versions
    prepare_model_for_kbit_training = None

try:
    from trl import SFTConfig
except Exception:  # pragma: no cover - old TRL versions
    SFTConfig = None

try:
    from trl import DataCollatorForCompletionOnlyLM
except Exception:  # pragma: no cover - depends on TRL version
    DataCollatorForCompletionOnlyLM = None

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainConfig:
    model_path: str
    dataset_path: str
    output_path: str
    use_lora: bool = True
    use_qlora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Union[str, List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = 5e-4
    batch_size: int = 1
    num_epochs: int = 3
    max_steps: int = -1
    save_steps: int = 100
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    eval_steps: int = 0
    logging_steps: int = 10
    validation_split: float = 0.0
    validation_dataset_path: Optional[str] = None
    load_best_model_at_end: bool = True
    save_total_limit: int = 3
    response_template: Optional[str] = "### Answer:"
    chat_template: Optional[str] = None
    trust_remote_code: bool = True
    local_files_only: bool = False
    gradient_checkpointing: bool = True
    optim: Optional[str] = None
    report_to: str = "none"
    seed: int = 42
    test_prompts: Optional[List[str]] = None
    resume_from_checkpoint: Optional[Union[bool, str]] = None
    precision: str = "auto"
    ddp_backend: Optional[str] = None
    replay_data_path: Optional[str] = None
    replay_ratio: float = 0.0
    replay_count: Optional[int] = None
    reference_model_path: Optional[str] = None
    retention_kl_weight: float = 0.0
    retention_temperature: float = 1.0


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _load_sft_dataset(dataset_path: str):
    if os.path.isdir(dataset_path):
        return load_dataset(dataset_path)
    ext = os.path.splitext(dataset_path)[1].lower()
    if ext in {".json", ".jsonl"}:
        return load_dataset("json", data_files=dataset_path)
    if ext == ".csv":
        return load_dataset("csv", data_files=dataset_path)
    if ext in {".txt", ".md"}:
        return load_dataset("text", data_files=dataset_path)
    return load_dataset(dataset_path)


def _format_messages(tokenizer, messages, add_generation_prompt: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("assistant:")
    return "\n".join(lines)


def _prompt_completion_pairs_from_messages(
    tokenizer, messages
) -> List[tuple[str, str]]:
    """Create one completion-only sample for every assistant turn.

    TRL >=0.26 natively masks prompt tokens when datasets expose separate
    ``prompt`` and ``completion`` columns.  Splitting the rendered chat text
    preserves model-specific role tokens and assistant prefixes.  Expanding a
    multi-turn conversation prevents all earlier assistant answers from being
    silently discarded.
    """
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant" and str(message.get("content", "")).strip()
    ]
    pairs: List[tuple[str, str]] = []
    for assistant_index in assistant_indexes:
        prompt_messages = messages[:assistant_index]
        target_messages = messages[: assistant_index + 1]
        prompt = _format_messages(
            tokenizer, prompt_messages, add_generation_prompt=True
        )
        full_text = _format_messages(
            tokenizer, target_messages, add_generation_prompt=False
        )
        if full_text.startswith(prompt):
            completion = full_text[len(prompt) :]
        else:
            completion = str(messages[assistant_index].get("content", "")).strip()
        if prompt and completion:
            pairs.append((prompt, completion))
    return pairs


def _prompt_completion_from_messages(tokenizer, messages) -> Optional[tuple[str, str]]:
    """Return the final assistant pair for callers using the legacy helper."""
    pairs = _prompt_completion_pairs_from_messages(tokenizer, messages)
    return pairs[-1] if pairs else None


def _format_sft_batch(
    examples: Dict,
    tokenizer,
    completion_only: bool = False,
    response_template: Optional[str] = "### Answer:",
) -> Dict[str, List[str]]:
    from ..data.PostTrainingData import PostTrainingDataAdapter

    size = len(next(iter(examples.values()))) if examples else 0
    texts = []
    prompts = []
    completions = []
    for i in range(size):
        item = {key: values[i] for key, values in examples.items()}
        normalized = PostTrainingDataAdapter.normalize_record(item, task="sft")
        if normalized is None:
            continue
        if completion_only:
            pairs: Sequence[tuple[str, str]] = []
            if normalized.get("messages"):
                pairs = _prompt_completion_pairs_from_messages(
                    tokenizer, normalized["messages"]
                )
            elif normalized.get("text") and response_template and response_template in str(normalized["text"]):
                prefix, suffix = str(normalized["text"]).rsplit(response_template, 1)
                pairs = [(f"{prefix}{response_template}", suffix)]
            if not pairs:
                continue
            prompts.extend(pair[0] for pair in pairs)
            completions.extend(pair[1] for pair in pairs)
            continue
        if normalized.get("text"):
            texts.append(str(normalized["text"]))
        elif normalized.get("messages"):
            # Legacy TRL uses a marker-based collator.  Render a stable marker
            # instead of a model chat template that may not contain it.
            if response_template and DataCollatorForCompletionOnlyLM is not None:
                pairs = _prompt_completion_pairs_from_messages(
                    tokenizer, normalized["messages"]
                )
                if not pairs:
                    continue
                texts.extend(
                    f"{prompt}{response_template} {completion}"
                    for prompt, completion in pairs
                )
            else:
                texts.append(_format_messages(tokenizer, normalized["messages"]))
        else:
            instruction = normalized.get("instruction", "")
            extra_input = normalized.get("input", "")
            output = normalized.get("output", "")
            marker = response_template or "### Answer:"
            prompt = f"### Question: {instruction}\n\n{extra_input}\n{marker}" if extra_input else f"### Question: {instruction}\n{marker}"
            texts.append(f"{prompt} {output}".strip())
    if completion_only:
        return {"prompt": prompts, "completion": completions}
    return {"text": texts}


def _maybe_quantization_config(config: SFTTrainConfig, device: str):
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


def _make_training_args(config: SFTTrainConfig, device: str, native_completion_only: bool = False):
    _, _, bf16, fp16 = resolve_post_training_precision(config.precision, device)
    optim = config.optim or ("adamw_torch_fused" if device == "cuda" else "adamw_torch")
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
            "best SFT checkpoint"
        )
    values = {
        "output_dir": config.output_path,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "num_train_epochs": config.num_epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "fp16": fp16,
        "bf16": bf16,
        "optim": optim,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_strategy": eval_strategy if load_best else "steps",
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_steps": config.warmup_steps,
        "dataloader_num_workers": 0 if device == "cpu" else 1,
        "dataloader_pin_memory": device == "cuda",
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": config.report_to,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": config.seed,
        "remove_unused_columns": True,
        "eval_strategy": eval_strategy,
        "evaluation_strategy": eval_strategy,
        "eval_steps": config.eval_steps if config.eval_steps > 0 else None,
        "load_best_model_at_end": load_best,
        "metric_for_best_model": "eval_loss" if load_best else None,
        "greater_is_better": False if load_best else None,
        "save_total_limit": config.save_total_limit,
        "use_cpu": device == "cpu",
        "dataset_text_field": "text",
        # TRL <=0.15 used max_seq_length; newer releases use max_length.
        "max_seq_length": config.max_seq_length,
        "max_length": config.max_seq_length,
        "packing": False,
        # The replay boundary is tracked through TRL's exact preparation path;
        # sampling and training still shuffle the final mixture afterward.
        "shuffle_dataset": False if config.replay_data_path else None,
        "completion_only_loss": native_completion_only,
        "ddp_backend": config.ddp_backend,
    }
    # Passing plain TrainingArguments makes TRL 0.15 reconstruct SFTConfig and
    # access fields removed by Transformers 5. Always use SFTConfig when TRL
    # provides it, with runtime signature filtering for cross-version support.
    args_class = SFTConfig or TrainingArguments
    return instantiate_supported(args_class, values)


def _build_sft_trainer(**kwargs):
    reference_model = kwargs.pop("retention_reference_model", None)
    retention_weight = kwargs.pop("retention_kl_weight", 0.0)
    retention_temperature = kwargs.pop("retention_temperature", 1.0)
    signature = inspect.signature(SFTTrainer.__init__)
    accepted = set(signature.parameters)
    if "processing_class" in accepted and "tokenizer" in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    trainer_kwargs = supported_kwargs(SFTTrainer.__init__, kwargs)
    if reference_model is not None:
        from .Retention import ReferenceKLMixin

        class RetentionSFTTrainer(ReferenceKLMixin, SFTTrainer):
            pass

        return RetentionSFTTrainer(
            **trainer_kwargs, reference_model=reference_model,
            retention_kl_weight=retention_weight,
            retention_temperature=retention_temperature,
        )
    return SFTTrainer(**trainer_kwargs)


@torch.no_grad()
def generate_response(model, tokenizer, prompt, device: Optional[str] = None, max_new_tokens: int = 256) -> str:
    """Generate a short validation response without assuming CUDA."""
    device = device or str(next(model.parameters()).device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    amp_enabled = device.startswith("cuda")
    dtype = torch.bfloat16 if _supports_bf16() else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def train_model(
    model_path: str,
    dataset_path: str,
    output_path: str,
    use_lora: bool = True,
    use_qlora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Union[str, List[str]]] = None,
    learning_rate: float = 5e-4,
    batch_size: int = 1,
    num_epochs: int = 3,
    max_steps: int = -1,
    save_steps: int = 100,
    max_seq_length: int = 512,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    eval_steps: int = 0,
    logging_steps: int = 10,
    test_prompts: Optional[List[str]] = None,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    validation_split: float = 0.0,
    validation_dataset_path: Optional[str] = None,
    load_best_model_at_end: bool = True,
    save_total_limit: int = 3,
    response_template: Optional[str] = "### Answer:",
    gradient_checkpointing: bool = True,
    optim: Optional[str] = None,
    report_to: str = "none",
    resume_from_checkpoint: Optional[Union[bool, str]] = None,
    precision: str = "auto",
    ddp_backend: Optional[str] = None,
    replay_data_path: Optional[str] = None,
    replay_ratio: float = 0.0,
    replay_count: Optional[int] = None,
    seed: int = 42,
    reference_model_path: Optional[str] = None,
    retention_kl_weight: float = 0.0,
    retention_temperature: float = 1.0,
):
    config = SFTTrainConfig(
        model_path=model_path,
        dataset_path=dataset_path,
        output_path=output_path,
        use_lora=use_lora,
        use_qlora=use_qlora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=(
            lora_target_modules
            if lora_target_modules is not None
            else ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        max_seq_length=max_seq_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        test_prompts=test_prompts,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        validation_split=validation_split,
        validation_dataset_path=validation_dataset_path,
        load_best_model_at_end=load_best_model_at_end,
        save_total_limit=save_total_limit,
        response_template=response_template,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        report_to=report_to,
        resume_from_checkpoint=resume_from_checkpoint,
        precision=precision,
        ddp_backend=ddp_backend,
        replay_data_path=replay_data_path,
        replay_ratio=replay_ratio,
        replay_count=replay_count,
        seed=seed,
        reference_model_path=reference_model_path,
        retention_kl_weight=retention_kl_weight,
        retention_temperature=retention_temperature,
    )
    return train_sft(config)


def train_sft(config: SFTTrainConfig):
    from ..data.ReplayData import (
        sft_source_groups,
        validate_replay_config,
    )
    from .Retention import validate_retention_config

    replay_enabled = validate_replay_config(
        config.replay_data_path, config.replay_ratio, config.replay_count
    )
    retention_enabled = validate_retention_config(
        config.reference_model_path, config.retention_kl_weight, config.retention_temperature
    )
    device = _device()
    if device == "cuda":
        torch.cuda.set_device(_local_process_index())
    actual_precision, model_dtype, _, _ = resolve_post_training_precision(
        config.precision, device
    )
    logger.info("Using device: %s with precision: %s", device, actual_precision)

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
    if config.chat_template:
        tokenizer.chat_template = config.chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.validation_dataset_path and config.validation_split:
        raise ValueError(
            "Use either validation_dataset_path or validation_split for SFT, not both."
        )
    if (
        config.validation_dataset_path
        and os.path.isfile(config.dataset_path)
        and os.path.isfile(config.validation_dataset_path)
        and os.path.abspath(config.dataset_path)
        == os.path.abspath(config.validation_dataset_path)
    ):
        raise ValueError("SFT training and validation files must be different.")

    def select_dataset(path: str, preferred_splits: Sequence[str]) -> Dataset:
        loaded = _load_sft_dataset(path)
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
            f"SFT dataset {path!r} has no usable split; available splits: {available}"
        )

    dataset = select_dataset(config.dataset_path, ("train",))
    if not isinstance(dataset, Dataset):
        raise ValueError("SFT dataset must resolve to a HuggingFace Dataset")

    native_completion_only = bool(
        config.response_template
        and DataCollatorForCompletionOnlyLM is None
        and SFTConfig is not None
        and "completion_only_loss" in inspect.signature(SFTConfig.__init__).parameters
    )
    if config.response_template and DataCollatorForCompletionOnlyLM is None and not native_completion_only:
        raise RuntimeError(
            "Completion-only SFT was requested through response_template, but this TRL version "
            "provides neither DataCollatorForCompletionOnlyLM nor native completion_only_loss. "
            "Set response_template to null to explicitly train all tokens or use the tested runtime."
        )
    if native_completion_only:
        logger.info("Using TRL native prompt/completion masking for completion-only SFT")

    def format_dataset(source: Dataset) -> Dataset:
        return source.map(
            lambda batch: _format_sft_batch(
                batch,
                tokenizer,
                completion_only=native_completion_only,
                response_template=config.response_template,
            ),
            batched=True,
            remove_columns=source.column_names,
        )

    eval_dataset = None
    validation_source = None
    train_source = dataset
    if config.validation_dataset_path:
        formatted_dataset = format_dataset(dataset)
        validation_source = select_dataset(
            config.validation_dataset_path, ("validation", "eval", "test", "train")
        )
        eval_dataset = format_dataset(validation_source)
    elif config.validation_split:
        from ..data.GroupSplit import split_group_indices

        # Split source conversations/cases before expanding assistant turns;
        # row-level splitting after formatting leaks sibling turns and variants.
        groups = sft_source_groups(dataset)
        train_indices, validation_indices = split_group_indices(
            groups, config.validation_split, config.seed
        )
        train_source = dataset.select(train_indices)
        formatted_dataset = format_dataset(train_source)
        validation_source = dataset.select(validation_indices)
        eval_dataset = format_dataset(validation_source)
    else:
        formatted_dataset = format_dataset(dataset)
    if len(formatted_dataset) == 0:
        raise ValueError(f"SFT dataset produced zero usable samples after normalization: {config.dataset_path}")
    if eval_dataset is not None and len(eval_dataset) == 0:
        raise ValueError(
            "SFT validation dataset produced zero usable samples after normalization: "
            f"{config.validation_dataset_path or config.dataset_path}"
        )

    split_audit = None
    if validation_source is not None:
        from ..data.TrainingSplitAudit import audit_record_splits

        split_audit = audit_record_splits(
            train_source,
            validation_source,
            report_path=os.path.join(config.output_path, "split_audit.json"),
            protected_paths=(config.dataset_path, config.validation_dataset_path, config.replay_data_path),
        )

    replay_info = {"enabled": False}
    replay_primary_count = len(formatted_dataset)
    replay_prepared_count = replay_primary_count
    replay_filtering = {}
    if replay_enabled:
        from ..data.PostTrainingData import PostTrainingDataAdapter
        from ..data.ReplayData import (
            filter_replay_records,
            replay_source_fingerprint,
        )

        replay_source = select_dataset(config.replay_data_path, ("train",))
        replay_records = list(replay_source)
        eligible, filtering = filter_replay_records(
            replay_records, list(validation_source) if validation_source is not None else []
        )
        # Give Arrow a consistent schema even when replay mixes text, Alpaca,
        # prompt/completion, and conversation records in the same file.
        normalized_replay = [
            PostTrainingDataAdapter.normalize_record(item, task="sft", keep_metadata=False)
            for item in eligible
        ]
        replay_rows = [
            {"text": item.get("text"), "messages": item.get("messages")}
            for item in normalized_replay if item is not None
        ]
        formatted_replay = list(format_dataset(Dataset.from_list(replay_rows))) if replay_rows else []
        replay_filtering = {
            **filtering,
            "data_path": os.path.abspath(config.replay_data_path),
            "source_fingerprint": replay_source_fingerprint(replay_records),
        }
        # Let this TRL version perform its real EOS handling, tokenization,
        # completion masking and truncation before selecting any replay rows.
        # Distinct strings can become identical model inputs after truncation.
        formatted_dataset = Dataset.from_list(
            list(formatted_dataset) + formatted_replay, features=formatted_dataset.features
        )
        replay_prepared_count = len(formatted_dataset)
        formatted_dataset = formatted_dataset.add_column(
            "_saddle_replay_origin",
            [0] * replay_primary_count + [1] * (replay_prepared_count - replay_primary_count),
        )

    quantization_config = _maybe_quantization_config(config, device)
    model_kwargs = {
        "dtype": model_dtype,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if device == "cuda" and quantization_config is not None:
        # One process owns one GPU. device_map="auto" may shard every worker
        # across all visible GPUs and conflicts with DDP process ownership.
        model_kwargs["device_map"] = {"": _local_process_index()}

    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
    reference_model = None
    retention_info = {"enabled": False}
    if retention_enabled:
        from .Retention import validate_reference_tokenizers

        reference_path = config.reference_model_path or config.model_path
        reference_tokenizer = load_tokenizer_compatible(
            reference_path, trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        if reference_tokenizer.pad_token is None:
            reference_tokenizer.pad_token = reference_tokenizer.eos_token
        validate_reference_tokenizers(tokenizer, reference_tokenizer)
        reference_model = AutoModelForCausalLM.from_pretrained(
            reference_path, dtype=model_dtype,
            trust_remote_code=config.trust_remote_code, local_files_only=config.local_files_only,
        )
        if model.config.vocab_size != reference_model.config.vocab_size:
            raise ValueError("Retention student and reference models must have matching output vocabulary sizes")
        retention_info = {
            "enabled": True, "reference_model_path": reference_path,
            "kl_weight": config.retention_kl_weight, "temperature": config.retention_temperature,
            "direction": "reference_to_student", "applies_to": "training_supervised_tokens",
            "requires_full_logits": True,
        }
    if not (device == "cuda" and quantization_config is not None):
        model = model.to(_local_model_device(device))
    if quantization_config is not None and config.use_lora and prepare_model_for_kbit_training is not None:
        prepare_kwargs = supported_kwargs(
            prepare_model_for_kbit_training,
            {"use_gradient_checkpointing": config.gradient_checkpointing},
        )
        model = prepare_model_for_kbit_training(model, **prepare_kwargs)
    if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = None
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )

    collator = None
    if config.response_template and DataCollatorForCompletionOnlyLM is not None:
        collator = DataCollatorForCompletionOnlyLM(config.response_template, tokenizer=tokenizer)

    training_args = _make_training_args(config, device, native_completion_only=native_completion_only)
    trainer = _build_sft_trainer(
        model=model,
        train_dataset=formatted_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=peft_config,
        data_collator=collator,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        retention_reference_model=reference_model,
        retention_kl_weight=config.retention_kl_weight,
        retention_temperature=config.retention_temperature,
    )
    if replay_enabled:
        from ..data.ReplayData import mix_replay_samples

        prepared = trainer.train_dataset
        if not isinstance(prepared, Dataset) or len(prepared) != replay_prepared_count:
            raise ValueError(
                "SFT replay requires TRL preparation to preserve sample order/count; "
                "packing or row-expanding preparation is unsupported"
            )
        if "input_ids" not in prepared.column_names:
            raise ValueError("SFT replay requires tokenized input_ids from TRL before selecting samples")
        if "_saddle_replay_origin" not in prepared.column_names:
            raise ValueError(
                "This TRL version drops replay provenance during tokenization; "
                "use a runtime that preserves dataset columns"
            )
        origins = list(prepared["_saddle_replay_origin"])
        primary_indices = [index for index, origin in enumerate(origins) if origin == 0]
        replay_indices = [index for index, origin in enumerate(origins) if origin == 1]
        if len(primary_indices) != replay_primary_count or len(primary_indices) + len(replay_indices) != len(prepared):
            raise ValueError("TRL preparation changed replay provenance; refusing an unverified training mixture")
        prepared = prepared.remove_columns(["_saddle_replay_origin"])
        prepared_eval = trainer.eval_dataset
        if prepared_eval is not None and "input_ids" not in prepared_eval.column_names:
            raise ValueError("SFT replay requires tokenized validation input_ids from TRL")
        mixed, replay_info = mix_replay_samples(
            list(prepared.select(primary_indices)), list(prepared.select(replay_indices)),
            replay_ratio=config.replay_ratio, replay_count=config.replay_count,
            seed=config.seed,
            validation_samples=list(prepared_eval) if prepared_eval is not None else [],
        )
        replay_info.update(replay_filtering)
        replay_info["exclusion_stage"] = "after_trl_tokenization_and_truncation"
        trainer.train_dataset = Dataset.from_list(mixed, features=prepared.features)
        formatted_dataset = trainer.train_dataset
    trainer.add_callback(LocalProgressCallback(config.output_path))

    distributed_world_size = int(
        getattr(trainer.args, "world_size", None)
        or os.environ.get("WORLD_SIZE", "1")
        or "1"
    )
    if config.test_prompts and config.eval_steps > 0 and distributed_world_size > 1:
        logger.warning(
            "Skipping test-prompt generation during DDP training; rank-local generation "
            "would desynchronize workers. Run generation after the distributed stage."
        )
    if config.test_prompts and config.eval_steps > 0 and distributed_world_size == 1:
        class ValidationCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step and state.global_step % config.eval_steps == 0:
                    model.eval()
                    for prompt in config.test_prompts or []:
                        full_prompt = f"### Question: {prompt}\n### Answer:"
                        response = generate_response(model, tokenizer, full_prompt, max_new_tokens=128)
                        logger.info("Prompt: %s", prompt)
                        logger.info("Response: %s", response[len(full_prompt):])
                    model.train()
                return control

        trainer.add_callback(ValidationCallback())

    prepared_split_audit = None
    if eval_dataset is not None:
        from ..data.TrainingSplitAudit import audit_record_splits

        prepared_split_audit = audit_record_splits(
            [],
            [],
            train_features=trainer.train_dataset,
            validation_features=trainer.eval_dataset,
            report_path=os.path.join(config.output_path, "prepared_split_audit.json"),
            protected_paths=(config.dataset_path, config.validation_dataset_path, config.replay_data_path),
        )
    resumed_from_checkpoint = resolve_trainer_resume_checkpoint(
        config.resume_from_checkpoint,
        config.output_path,
    )
    validate_resume_training_args(resumed_from_checkpoint, training_args)
    train_output = trainer.train(resume_from_checkpoint=resumed_from_checkpoint)
    if retention_enabled:
        retention_info.update(trainer.retention_metrics())
    global_step = int(getattr(trainer.state, "global_step", 0))
    eval_metrics = _latest_eval_metrics(getattr(trainer.state, "log_history", []))
    if eval_dataset is not None and eval_metrics.get("step") != global_step:
        eval_metrics = _metric_snapshot(trainer.evaluate())
        eval_metrics.setdefault("step", global_step)
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
    logger.info("SFT model saved to: %s", config.output_path)
    train_metrics = _metric_snapshot(getattr(train_output, "metrics", {}))
    latest_train = _latest_train_metrics(getattr(trainer.state, "log_history", []))
    return {
        "output_path": config.output_path,
        "train_samples": len(formatted_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "use_lora": config.use_lora,
        "use_qlora": config.use_qlora,
        "max_seq_length": config.max_seq_length,
        "precision": actual_precision,
        "global_step": global_step,
        "train_metrics": train_metrics,
        "latest_train_metrics": latest_train,
        "eval_metrics": eval_metrics,
        "validation_dataset_path": config.validation_dataset_path,
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "lora_target_modules": config.lora_target_modules,
        "resume_from_checkpoint": config.resume_from_checkpoint,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "trainer_state_path": os.path.join(config.output_path, "trainer_state.json"),
        "replay": replay_info,
        "retention": retention_info,
        "split_audit": split_audit,
        "prepared_split_audit": prepared_split_audit,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="SFT/LoRA fine-tuning")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    parser.add_argument("--no_qlora", dest="use_qlora", action="store_false")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto"
    )
    parser.set_defaults(use_lora=True, use_qlora=True)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    train_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        local_files_only=args.local_files_only,
        precision=args.precision,
    )
