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

    @staticmethod
    def _native_model_config(model):
        """Find Saddle's config through common distributed wrappers."""

        pending = [model]
        visited = set()
        while pending:
            current = pending.pop()
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))
            config = getattr(current, "config", None)
            if config is not None and hasattr(config, "loss_chunk_size"):
                return config
            pending.extend(
                getattr(current, attribute, None)
                for attribute in (
                    "module",
                    "_orig_mod",
                    "_fsdp_wrapped_module",
                    "base_model",
                    "model",
                )
            )
        return None

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """Use chunked linear/CE during loss-only train and evaluation calls.

        Direct model evaluation keeps returning logits for API compatibility.
        Trainer already knows when predictions are loss-only, so it can opt in
        explicitly and avoid materialising the full vocabulary tensor there as
        well as during training.
        """

        config = self._native_model_config(model)
        use_chunked_loss = bool(
            config is not None
            and int(config.loss_chunk_size) > 0
            and "labels" in inputs
            and (model.training or self.args.prediction_loss_only)
            # Trainer removes labels before calling the model when either hook
            # owns loss computation.  Keep logits available in that mode.
            and self.label_smoother is None
            and self.compute_loss_func is None
        )
        if use_chunked_loss and "return_logits" not in inputs:
            inputs = dict(inputs)
            inputs["return_logits"] = False
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

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
