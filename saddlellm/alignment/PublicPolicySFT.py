"""SFT training for AASB public-policy datasets.

Produces LoRA adapters from Dataset A training examples that are compatible
with the RLVR pipeline in :mod:`PublicPolicyRLVR`.  Uses the same chat
template, prompt builder, and LoRA target modules so SFT checkpoints can be
loaded directly by ``load_qwen3_lora``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from .PublicPolicyRLVR import (
    AASBDataModule,
    PolicyExample,
    PublicPolicyPilotConfig,
    build_user_prompt,
    render_chat_prompt,
)

logger = logging.getLogger(__name__)


def build_sft_completion(example: PolicyExample) -> str:
    """Render the ground-truth answer as an SFT target completion."""
    if example.task == "A":
        return f"<answer>{example.primary_label}</answer>"
    labels = ",".join(example.answer)
    return f"<answer>{labels}</answer>"


def build_sft_message_pair(
    example: PolicyExample,
    tokenizer,
    config: PublicPolicyPilotConfig,
) -> Dict[str, str]:
    """Return ``{"prompt": ..., "completion": ...}`` for one example."""
    user_prompt = build_user_prompt(
        example,
        use_cot=config.use_cot,
        max_response_chars=config.max_response_chars,
    )
    chat_prompt = render_chat_prompt(
        tokenizer,
        user_prompt,
        enable_thinking=config.enable_qwen_thinking,
    )
    completion = build_sft_completion(example)
    return {"prompt": chat_prompt, "completion": completion}


def prepare_sft_dataset(
    examples: Sequence[PolicyExample],
    tokenizer,
    config: PublicPolicyPilotConfig,
) -> Dataset:
    """Build a HuggingFace Dataset of (prompt + completion) texts."""
    records: List[Dict[str, str]] = []
    for example in examples:
        records.append(build_sft_message_pair(example, tokenizer, config))

    def _format_text(record: Dict[str, str]) -> Dict[str, str]:
        return {"text": record["prompt"] + record["completion"]}

    dataset = Dataset.from_list(records)
    dataset = dataset.map(_format_text)
    return dataset


def _load_base_model(config: PublicPolicyPilotConfig):
    """Load Qwen3 base model and tokenizer WITHOUT LoRA."""
    from packaging.version import Version
    from transformers import __version__

    if Version(__version__) < Version("4.51.0"):
        raise RuntimeError(
            f"Qwen3 requires transformers>=4.51.0; found {__version__}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 training requires a CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = False

    try:
        from peft.tuners.lora import awq as peft_awq
        peft_awq.is_auto_awq_available = lambda: False
    except ImportError:
        pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model_metadata = {
        "total_parameters": total,
        "trainable_parameters_before_lora": trainable,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }
    return model, tokenizer, model_metadata


@dataclass
class SFTExperimentConfig:
    """Configuration scoped to one SFT ablation run."""

    model_name: str = "Qwen/Qwen3-0.6B"
    model_revision: str = "c1899de289a04d12100db370d81485cdf75e47ca"
    output_dir: str = "outputs/sft_qwen3_0_6b"

    use_cot: bool = False
    enable_qwen_thinking: bool = False
    max_response_chars: int = 3000

    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 50
    max_seq_length: int = 1024

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    train_samples: int = 1024
    seed: int = 42
    source_task: str = "A"

    def validate(self) -> None:
        if self.train_samples < 1:
            raise ValueError("train_samples must be >= 1")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.source_task not in ("A", "B"):
            raise ValueError("source_task must be 'A' or 'B'")


def train_public_policy_sft(
    pilot_config: PublicPolicyPilotConfig,
    sft_config: SFTExperimentConfig,
) -> Dict[str, Any]:
    """Run SFT on the configured source-task training split and return the LoRA adapter path."""

    from trl import SFTConfig, SFTTrainer

    # Seed before SFTTrainer.__init__: trl wraps the model in PEFT (LoRA init
    # draws from the global torch RNG) only inside its own __init__, before its
    # internal set_seed, so without this the adapter weights are not
    # reproducible even with a fixed seed in SFTConfig.
    seed = sft_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("Loading data for SFT (source task %s)", sft_config.source_task)
    data = AASBDataModule(pilot_config).load()
    train_examples = data.select(
        sft_config.source_task,
        "train",
        sft_config.train_samples,
        balanced=sft_config.source_task == "A",
    )

    logger.info("Loading base model for SFT")
    model, tokenizer, model_metadata = _load_base_model(pilot_config)

    logger.info("Building SFT dataset (%d examples)", len(train_examples))
    sft_dataset = prepare_sft_dataset(train_examples, tokenizer, pilot_config)

    output_dir = Path(sft_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=sft_config.batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        num_train_epochs=sft_config.num_epochs,
        learning_rate=sft_config.learning_rate,
        warmup_steps=sft_config.warmup_steps,
        logging_steps=10,
        save_strategy="epoch",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=True,
        seed=sft_config.seed,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_seq_length=sft_config.max_seq_length,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=sft_config.lora_rank,
        lora_alpha=sft_config.lora_alpha,
        lora_dropout=sft_config.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        args=sft_args,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting SFT training (%d examples, %d epochs)",
                len(train_examples), sft_config.num_epochs)
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("SFT adapter saved to %s", adapter_dir)

    result = {
        "status": "completed",
        "model_name": sft_config.model_name,
        "adapter_path": str(adapter_dir),
        "train_samples": len(train_examples),
        "num_epochs": sft_config.num_epochs,
        "use_cot": pilot_config.use_cot,
        "model_metadata": model_metadata,
    }
    return result
