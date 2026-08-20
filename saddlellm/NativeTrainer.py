"""Hugging Face Trainer integration for native SaddleLLM checkpoints."""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME


class SaddleTrainer(Trainer):
    """Trainer that makes every native ``checkpoint-*`` self-contained.

    Trainer's optimizer, scheduler, RNG and trainer-state machinery remains in
    charge; only model persistence is changed from a bare ``nn.Module``
    safetensors file to the SaddleLLM checkpoint contract.
    """

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict] = None) -> None:
        output_dir = output_dir or self.args.output_dir
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)

        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if not hasattr(model, "save_pretrained"):
            raise TypeError("SaddleTrainer requires a model with save_pretrained()")
        if state_dict is None:
            state_dict = model.state_dict()
        from .DistributedRuntime import current_distributed_runtime

        runtime = current_distributed_runtime("single")
        strategy = "single"
        if runtime.world_size > 1:
            if getattr(self, "is_fsdp_enabled", False):
                strategy = "fsdp"
            elif getattr(self, "is_deepspeed_enabled", False):
                strategy = "deepspeed"
            else:
                strategy = "ddp"
        model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            metadata={
                "checkpoint_type": "trainer",
                "global_step": int(getattr(self.state, "global_step", 0)),
            },
            parallelism={
                "strategy": strategy,
                "world_size": runtime.world_size,
                "rank": runtime.rank,
                "state_dict_type": "full",
            },
        )

        processing_class = getattr(self, "processing_class", None)
        if processing_class is not None:
            processing_class.save_pretrained(output_dir)
        else:
            tokenizer = getattr(self.data_collator, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


__all__ = ["SaddleTrainer"]
