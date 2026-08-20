"""Stable SFT/LoRA training entry points.

This module keeps the historical ``train_model`` function, but makes it usable
as a library API across CPU/CUDA, LoRA/QLoRA/full fine-tuning, and common SFT
dataset schemas.
"""
import argparse
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer

from .PostTrainingCompatibility import instantiate_supported, supported_kwargs

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
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
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


def _format_messages(tokenizer, messages) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_sft_batch(examples: Dict, tokenizer) -> Dict[str, List[str]]:
    from .PostTrainingData import PostTrainingDataAdapter

    size = len(next(iter(examples.values()))) if examples else 0
    texts = []
    for i in range(size):
        item = {key: values[i] for key, values in examples.items()}
        normalized = PostTrainingDataAdapter.normalize_record(item, task="sft")
        if normalized is None:
            continue
        if normalized.get("text"):
            texts.append(str(normalized["text"]))
        elif normalized.get("messages"):
            texts.append(_format_messages(tokenizer, normalized["messages"]))
        else:
            instruction = normalized.get("instruction", "")
            extra_input = normalized.get("input", "")
            output = normalized.get("output", "")
            prompt = f"### Question: {instruction}\n\n{extra_input}\n### Answer:" if extra_input else f"### Question: {instruction}\n### Answer:"
            texts.append(f"{prompt} {output}".strip())
    return {"text": texts}


def _maybe_quantization_config(config: SFTTrainConfig, device: str):
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
            bnb_4bit_compute_dtype=torch.bfloat16 if _supports_bf16() else torch.float16,
        )
    except Exception as exc:
        raise RuntimeError(
            "QLoRA requires a working bitsandbytes installation compatible with "
            f"the current PyTorch/CUDA runtime. Original error: {exc}"
        ) from exc


def _make_training_args(config: SFTTrainConfig, device: str):
    bf16 = device == "cuda" and _supports_bf16()
    fp16 = device == "cuda" and not bf16
    optim = config.optim or ("adamw_torch_fused" if device == "cuda" else "adamw_torch")
    eval_strategy = "no"
    if config.validation_split and 0 < config.validation_split < 1:
        eval_strategy = "steps" if config.eval_steps > 0 else "epoch"
    values = {
        "output_dir": config.output_path,
        "per_device_train_batch_size": config.batch_size,
        "num_train_epochs": config.num_epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "fp16": fp16,
        "bf16": bf16,
        "optim": optim,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_strategy": "steps",
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
        "use_cpu": device == "cpu",
        "dataset_text_field": "text",
        # TRL <=0.15 used max_seq_length; newer releases use max_length.
        "max_seq_length": config.max_seq_length,
        "max_length": config.max_seq_length,
        "packing": False,
    }
    # Passing plain TrainingArguments makes TRL 0.15 reconstruct SFTConfig and
    # access fields removed by Transformers 5. Always use SFTConfig when TRL
    # provides it, with runtime signature filtering for cross-version support.
    args_class = SFTConfig or TrainingArguments
    return instantiate_supported(args_class, values)


def _build_sft_trainer(**kwargs):
    signature = inspect.signature(SFTTrainer.__init__)
    accepted = set(signature.parameters)
    if "processing_class" in accepted and "tokenizer" in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    return SFTTrainer(**supported_kwargs(SFTTrainer.__init__, kwargs))


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
    lora_target_modules: Optional[List[str]] = None,
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
    response_template: Optional[str] = "### Answer:",
    gradient_checkpointing: bool = True,
    optim: Optional[str] = None,
    report_to: str = "none",
    resume_from_checkpoint: Optional[Union[bool, str]] = None,
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
        lora_target_modules=lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
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
        response_template=response_template,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        report_to=report_to,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return train_sft(config)


def train_sft(config: SFTTrainConfig):
    device = _device()
    logger.info("Using device: %s", device)

    if config.use_lora:
        from .PostTrainingCompatibility import stabilize_peft_optional_backends

        for warning in stabilize_peft_optional_backends():
            logger.warning(warning)

    from .TokenizerLoader import load_tokenizer_compatible

    tokenizer = load_tokenizer_compatible(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    if config.chat_template:
        tokenizer.chat_template = config.chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_dict = _load_sft_dataset(config.dataset_path)
    dataset = dataset_dict["train"] if "train" in dataset_dict else dataset_dict
    if not isinstance(dataset, Dataset):
        raise ValueError("SFT dataset must resolve to a HuggingFace Dataset")

    formatted_dataset = dataset.map(
        lambda batch: _format_sft_batch(batch, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    if len(formatted_dataset) == 0:
        raise ValueError(f"SFT dataset produced zero usable samples after normalization: {config.dataset_path}")

    eval_dataset = None
    if config.validation_split and 0 < config.validation_split < 1:
        split = formatted_dataset.train_test_split(test_size=config.validation_split, seed=config.seed)
        formatted_dataset = split["train"]
        eval_dataset = split["test"]

    quantization_config = _maybe_quantization_config(config, device)
    model_kwargs = {
        "dtype": _dtype_for_device(device),
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if device == "cuda" and quantization_config is not None:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
    if not (device == "cuda" and quantization_config is not None):
        model = model.to(device)
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

    training_args = _make_training_args(config, device)
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
    )

    if config.test_prompts and config.eval_steps > 0:
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

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_path)
    tokenizer.save_pretrained(config.output_path)
    logger.info("SFT model saved to: %s", config.output_path)
    return {
        "output_path": config.output_path,
        "train_samples": len(formatted_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "use_lora": config.use_lora,
        "use_qlora": config.use_qlora,
        "max_seq_length": config.max_seq_length,
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
    )
