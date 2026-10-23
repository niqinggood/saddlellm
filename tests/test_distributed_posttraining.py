from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import pytest


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _posttraining_worker(
    rank: int,
    world_size: int,
    port: int,
    config_path: str,
    result_dir: str,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "USE_LIBUV": "0",
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "GROUP_WORLD_SIZE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    import torch
    import torch.distributed as dist

    from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator

    torch.set_num_threads(1)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        result = TrainingOrchestrator.from_dict(config).run()
        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "sft": result["sft"]["status"],
                    "preference": result["preference"]["status"],
                    "rank": dist.get_rank(),
                    "world_size": dist.get_world_size(),
                }
            ),
            encoding="utf-8",
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _save_tiny_native_checkpoint(path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    import torch
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from saddlellm.models.ModelBlueprint import (
        AttentionBlueprint,
        FFNBlueprint,
        ModelBlueprint,
    )

    torch.manual_seed(731)
    model = ModelBlueprint(
        name="tiny-ddp-posttraining",
        hidden_size=16,
        vocab_size=8,
        num_layers=1,
        max_position_embeddings=32,
        attention=AttentionBlueprint(
            kind="gqa",
            backend="eager",
            num_heads=4,
            num_kv_heads=2,
        ),
        ffn=FFNBlueprint(kind="swiglu", intermediate_size=32),
    ).build_model()
    model.save_pretrained(path)

    backend = tokenizers.Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "hello": 4,
                "world": 5,
                "tiny": 6,
                "model": 7,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    ).save_pretrained(path)


def test_two_process_native_sft_then_dpo_pipeline(tmp_path, monkeypatch) -> None:
    import torch
    from torch.multiprocessing import start_processes

    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("requires torch.distributed Gloo")

    # The acceptance test is intentionally CPU/Gloo even on GPU workstations.
    # Setting visibility before spawning makes both child runtimes deterministic.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    checkpoint = tmp_path / "model"
    _save_tiny_native_checkpoint(checkpoint)
    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "hello", "output": "world"}),
                json.dumps({"instruction": "tiny", "output": "model"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    preference_data = tmp_path / "dpo.jsonl"
    preference_data.write_text(
        "\n".join(
            [
                json.dumps(
                    {"prompt": "hello", "chosen": "world", "rejected": "tiny"}
                ),
                json.dumps(
                    {"prompt": "tiny", "chosen": "model", "rejected": "world"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "config": "qwen-tiny-160m",
                    "backend": "saddle",
                    "name_or_path": str(checkpoint),
                },
                "stages": ["sft", "preference"],
                "training": {
                    "max_steps": 1,
                    "dataloader_num_workers": 0,
                    "stability_monitor": False,
                },
                "distributed": {
                    "strategy": "ddp",
                    "num_gpus": 2,
                    "num_nodes": 1,
                    "ddp_backend": "gloo",
                    "bf16": False,
                    "fp16": False,
                },
                "logging": {"output_dir": str(output_dir), "backend": "local"},
                "sft": {
                    "enabled": True,
                    "data_path": str(sft_data),
                    "use_lora": False,
                    "use_qlora": False,
                    "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "gradient_checkpointing": False,
                    "max_seq_length": 16,
                    "warmup_steps": 0,
                    "save_steps": 0,
                    "logging_steps": 1,
                },
                "preference": {
                    "enabled": True,
                    "method": "dpo",
                    "data_path": str(preference_data),
                    "use_lora": False,
                    "use_qlora": False,
                    "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "gradient_checkpointing": False,
                    "max_length": 16,
                    "max_prompt_length": 8,
                    "warmup_steps": 0,
                    "save_steps": 0,
                    "logging_steps": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    context = start_processes(
        _posttraining_worker,
        args=(2, _free_tcp_port(), str(config_path), str(tmp_path)),
        nprocs=2,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if context.join(timeout=1):
            break
    else:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
        pytest.fail("two-process post-training did not finish within 90 seconds")

    for rank in (0, 1):
        result = json.loads(
            (tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")
        )
        assert result == {
            "sft": "completed",
            "preference": "completed",
            "rank": rank,
            "world_size": 2,
        }

    sft_output = output_dir / "sft_checkpoints"
    preference_output = output_dir / "preference_checkpoints"
    for checkpoint_dir in (sft_output, preference_output):
        manifest = json.loads(
            (checkpoint_dir / "saddle_checkpoint.json").read_text(encoding="utf-8")
        )
        assert manifest["parallelism"] == {
            "strategy": "ddp",
            "world_size": 2,
            "rank": 0,
            "state_dict_type": "full",
        }
        progress = json.loads(
            (checkpoint_dir / "training_progress.json").read_text(encoding="utf-8")
        )
        assert progress["status"] == "completed"

    preference_plan = json.loads(
        (output_dir / "plans" / "preference_plan.json").read_text(encoding="utf-8")
    )
    assert Path(preference_plan["model_path"]).resolve() == sft_output.resolve()
    assert sorted(path.name for path in (output_dir / "plans").glob("*_plan.json")) == [
        "preference_plan.json",
        "sft_plan.json",
    ]
