"""Rank-local worker, launched only for configured Megatron jobs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from .MegatronBackend import prepare_sft_jsonl, write_json


def apply_recipe_overrides(config, overrides, prefix=""):
    """Set existing recipe fields; typos must never create unused attributes."""
    if not isinstance(overrides, dict):
        raise TypeError("Megatron overrides must be nested mappings")
    for key, value in overrides.items():
        name = f"{prefix}.{key}" if prefix else key
        if not isinstance(key, str) or not key.isidentifier() or key.startswith("_"):
            raise ValueError(f"invalid Megatron override field: {name}")
        if not hasattr(config, key):
            raise ValueError(f"unknown Megatron recipe field: {name}")
        current = getattr(config, key)
        if isinstance(value, dict):
            if isinstance(current, dict):
                setattr(config, key, {**current, **value})
            elif current is None:
                raise ValueError(
                    f"cannot construct {name} from a mapping; "
                    "select a recipe with that config"
                )
            else:
                apply_recipe_overrides(current, value, name)
        else:
            # Explicit supported dtype spellings, without eval or executable YAML.
            if isinstance(value, str) and value.startswith("torch."):
                import torch

                allowed = {
                    "torch.float32": torch.float32,
                    "torch.float16": torch.float16,
                    "torch.bfloat16": torch.bfloat16,
                }
                if value not in allowed:
                    raise ValueError(f"unsupported dtype override: {name}")
                value = allowed[value]
            setattr(config, key, value)


def run_job(job):
    try:
        from megatron.bridge.training.finetune import finetune
        from megatron.bridge.training.gpt_step import forward_step
        from megatron.bridge.training.pretrain import pretrain
    except ImportError as exc:
        raise ImportError(
            "Megatron Bridge training API/dependencies are unavailable"
        ) from exc
    module, function = job["recipe"].split(":", 1)
    recipe = getattr(importlib.import_module(module), function)
    config = recipe()
    apply_recipe_overrides(config, job["overrides"])
    config.checkpoint.save = job["checkpoint_path"]
    if "load" not in job["overrides"].get("checkpoint", {}):
        config.checkpoint.load = None
    checkpoint_source = job.get("model_path")
    if job["stage"] == "pretrain" and job["pretrain_mode"] == "scratch":
        if config.checkpoint.pretrained_checkpoint:
            raise ValueError(
                "scratch recipe unexpectedly initializes pretrained weights"
            )
        # A recipe's default load directory is not authorization to resume it.
        if not job["overrides"].get("checkpoint", {}).get("load"):
            config.checkpoint.load = None
    elif (
        checkpoint_source
        and not config.checkpoint.load
        and not job["overrides"].get("checkpoint", {}).get("pretrained_checkpoint")
    ):
        config.checkpoint.pretrained_checkpoint = checkpoint_source
    if job["stage"] == "pretrain" and job["pretrain_mode"] == "continue":
        if not config.checkpoint.load and not config.checkpoint.pretrained_checkpoint:
            raise ValueError(
                "continue pretraining requires a native/HF local checkpoint"
            )
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        import torch
        import torch.distributed as dist

        if getattr(getattr(config, "inprocess_restart", None), "enabled", False):
            raise ValueError(
                "managed multi-rank barriers do not support inprocess_restart"
            )
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    if job.get("sft_data_path"):
        from megatron.bridge.data.builders import (
            GPTSFTDatasetConfig,
            PromptCompletionSFTPreprocessingConfig,
        )

        dataset_root = Path(job["checkpoint_path"]).parent / "sft_dataset"
        if int(os.environ.get("RANK", "0")) == 0:
            dataset_root.mkdir(parents=True, exist_ok=True)
            prepare_sft_jsonl(job["sft_data_path"], dataset_root / "training.jsonl")
            if job.get("validation_data_path"):
                prepare_sft_jsonl(
                    job["validation_data_path"], dataset_root / "validation.jsonl"
                )
        # The managed world group coordinates shared dataset preparation.
        import torch.distributed as dist

        if world > 1:
            dist.barrier()
        config.dataset = GPTSFTDatasetConfig(
            dataset_root=str(dataset_root),
            seq_length=config.dataset.seq_length,
            preprocessing=PromptCompletionSFTPreprocessingConfig(
                prompt_column="input",
                completion_column="output",
                separator="\n",
            ),
        )
        dataset_overrides = dict(job["overrides"].get("dataset", {}))
        if any(
            key in dataset_overrides
            for key in ("dataset_root", "per_split_data_source_manifest_path")
        ):
            raise ValueError(
                "sft.data_path cannot be combined with a Bridge dataset source override"
            )
        apply_recipe_overrides(config.dataset, dataset_overrides)
    parallel = 1
    for key in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
    ):
        parallel *= int(getattr(config.model, key, 1))
    if world % parallel:
        raise ValueError("Megatron world size must be divisible by TP * PP * CP")
    dp = world // parallel
    if config.train.global_batch_size % (config.train.micro_batch_size * dp):
        raise ValueError("Megatron global batch must be divisible by micro batch * DP")
    entry = pretrain if job["stage"] == "pretrain" else finetune
    entry(config, forward_step_func=forward_step)
    if job.get("hf_path"):
        from megatron.bridge.models import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(job["hf_export_source"])
        bridge.export_ckpt(
            job["checkpoint_path"],
            job["hf_path"],
            strict=True,
            source_path=job["hf_export_source"],
        )
    result = {
        "run_id": job["run_id"],
        "status": "completed",
        "checkpoint_path": job["checkpoint_path"],
        "checkpoint_format": "megatron",
    }
    if job.get("hf_path"):
        result["model_path"] = job["hf_path"]
        result["model_format"] = "hf"
    if int(os.environ.get("RANK", "0")) == 0:
        write_json(job["result_path"], result)
    if world > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            # Publish rank 0's shared result before controllers return.
            dist.barrier()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    args = parser.parse_args()
    try:
        run_job(json.loads(Path(args.job).read_text(encoding="utf-8")))
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
