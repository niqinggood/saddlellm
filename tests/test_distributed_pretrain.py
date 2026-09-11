from __future__ import annotations

import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import start_processes

from saddlellm.factory.BackendAdapters import BackendAdapterRegistry
from saddlellm.training.DistributedConfig import DistributedConfig
from saddlellm.training.DistributedRuntime import DistributedRuntimeInfo
from saddlellm.factory.FactoryBackendPlanner import FactoryBackendPlanner
from saddlellm.models.ModelBlueprint import AttentionBlueprint, FFNBlueprint, ModelBlueprint
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


_SEED = 917
_LR = 0.025
_INPUTS = torch.tensor(
    [
        [1, 4, 5, 6, 7],
        [2, 3, 6, 5, 4],
        [7, 6, 5, 4, 3],
        [3, 5, 7, 2, 1],
    ],
    dtype=torch.long,
)


def _tiny_model():
    return ModelBlueprint(
        name="tiny-ddp-acceptance",
        hidden_size=8,
        vocab_size=11,
        num_layers=1,
        max_position_embeddings=16,
        attention=AttentionBlueprint(
            kind="gqa",
            backend="eager",
            num_heads=2,
            num_kv_heads=1,
        ),
        ffn=FFNBlueprint(kind="swiglu", intermediate_size=16),
    ).build_model()


def _one_step(model, input_ids):
    optimizer = torch.optim.SGD(model.parameters(), lr=_LR)
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()
    return float(loss.detach()), {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_worker(rank: int, world_size: int, port: int, result_dir: str) -> None:
    import torch.distributed as dist

    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            # Some Windows PyTorch wheels ship Gloo/TCPStore without libuv.
            "USE_LIBUV": "0",
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
        }
    )
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(_SEED)
        model = DistributedDataParallel(_tiny_model())
        optimizer = torch.optim.SGD(model.parameters(), lr=_LR)
        optimizer.zero_grad(set_to_none=True)

        local_inputs = _INPUTS.chunk(world_size, dim=0)[rank]
        loss = model(input_ids=local_inputs, labels=local_inputs).loss
        loss.backward()
        optimizer.step()

        state = {
            name: value.detach().cpu().clone()
            for name, value in model.module.state_dict().items()
        }
        path = Path(result_dir)
        torch.save(state, path / f"rank-{rank}.pt")
        (path / f"side-effect-rank-{rank}.txt").write_text(str(rank), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="requires torch.distributed Gloo",
)
def test_two_process_cpu_gloo_ddp_one_step_matches_single_process(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(_SEED)
    _, reference = _one_step(_tiny_model(), _INPUTS)

    start_processes(
        _ddp_worker,
        args=(2, _free_tcp_port(), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

    rank0 = torch.load(tmp_path / "rank-0.pt", map_location="cpu", weights_only=True)
    rank1 = torch.load(tmp_path / "rank-1.pt", map_location="cpu", weights_only=True)
    assert rank0.keys() == rank1.keys() == reference.keys()
    for name in reference:
        torch.testing.assert_close(rank0[name], rank1[name], atol=0, rtol=0)
        torch.testing.assert_close(rank0[name], reference[name], atol=2e-6, rtol=2e-5)


def test_rank_zero_checkpoint_manifest_contract(tmp_path, monkeypatch):
    model = _tiny_model()
    checkpoint = tmp_path / "checkpoint-3"
    rank0 = DistributedRuntimeInfo(
        strategy="ddp", rank=0, local_rank=0, world_size=2, local_world_size=2
    )
    rank1 = DistributedRuntimeInfo(
        strategy="ddp", rank=1, local_rank=1, world_size=2, local_world_size=2
    )

    assert rank0.is_main_process
    assert not rank1.is_main_process
    if rank0.is_main_process:
        model.save_pretrained(
            checkpoint,
            metadata={"checkpoint_type": "trainer", "global_step": 3},
            parallelism={
                "strategy": "ddp",
                "world_size": 2,
                "rank": 0,
                "state_dict_type": "full",
            },
        )
    if rank1.is_main_process:
        pytest.fail("a non-zero rank must not execute checkpoint side effects")

    manifest = json.loads((checkpoint / "saddle_checkpoint.json").read_text(encoding="utf-8"))
    assert manifest["metadata"] == {"checkpoint_type": "trainer", "global_step": 3}
    assert manifest["parallelism"] == {
        "strategy": "ddp",
        "world_size": 2,
        "rank": 0,
        "state_dict_type": "full",
    }

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(TrainingOrchestrator, "_validate_config", lambda self: None)
    monkeypatch.setattr(TrainingOrchestrator, "_setup_logging", lambda self: None)
    non_main_output = tmp_path / "rank1-output"
    parsed = TrainingOrchestrator._parse_config(
        {
            "stages": ["pretrain"],
            "distributed": {"strategy": "ddp", "num_gpus": 2, "bf16": False},
            "logging": {"output_dir": str(non_main_output)},
        }
    )
    TrainingOrchestrator(parsed)
    assert not non_main_output.exists()


def _resume_orchestrator(tmp_path, checkpoint, *, world_size=2):
    parsed = TrainingOrchestrator._parse_config(
        {
            "model": {"backend": "saddle", "config": "qwen-tiny-160m"},
            "stages": ["pretrain"],
            "data": {"sources": [{"type": "local", "path": "unused.txt"}]},
            "training": {"resume_from_checkpoint": str(checkpoint)},
            "distributed": {
                "strategy": "ddp" if world_size > 1 else "single",
                "num_gpus": world_size,
                "bf16": False,
                "fp16": False,
            },
            "logging": {"output_dir": str(tmp_path / "run")},
        }
    )
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.config = parsed
    orchestrator._output_dir = parsed.logging.output_dir
    return orchestrator


def _write_resume_checkpoint(path, *, saved_world_size=2, rng_ranks=(0, 1)):
    path.mkdir()
    for name in (
        "saddle_config.json",
        "pytorch_model.bin",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
    ):
        (path / name).write_bytes(b"test")
    for rank in rng_ranks:
        (path / f"rng_state_{rank}.pth").write_bytes(b"rng")
    (path / "saddle_checkpoint.json").write_text(
        json.dumps({"parallelism": {"strategy": "ddp", "world_size": saved_world_size}}),
        encoding="utf-8",
    )


def test_exact_resume_validates_world_size_and_all_rank_rng_states(tmp_path):
    complete = tmp_path / "complete"
    _write_resume_checkpoint(complete)
    orchestrator = _resume_orchestrator(tmp_path, complete)
    assert orchestrator._resolve_exact_resume_checkpoint() == str(complete.resolve())

    wrong_world_size = tmp_path / "wrong-world-size"
    _write_resume_checkpoint(wrong_world_size, saved_world_size=1)
    with pytest.raises(ValueError, match=r"world-size mismatch.*world_size=1.*world_size=2"):
        _resume_orchestrator(tmp_path, wrong_world_size)._resolve_exact_resume_checkpoint()

    missing_rng = tmp_path / "missing-rng"
    _write_resume_checkpoint(missing_rng, rng_ranks=(0,))
    with pytest.raises(ValueError, match=r"rng_state_1\.pth"):
        _resume_orchestrator(tmp_path, missing_rng)._resolve_exact_resume_checkpoint()


def test_hybrid_planner_and_launcher_fail_closed_without_runtime_adapter(tmp_path):
    plan = FactoryBackendPlanner.recommend_parallelism(
        model_params=70_000_000_000,
        num_gpus=8,
        gpu_memory_gb=80,
        seq_length=16_384,
        prefer_backend="hybrid_parallel",
    )
    assert plan.backend == "hybrid_parallel"

    with pytest.raises((NotImplementedError, RuntimeError), match=r"(?i)hybrid.*runtime|adapter"):
        FactoryBackendPlanner.to_training_orchestrator_distributed(plan)
    with pytest.raises((NotImplementedError, RuntimeError), match=r"(?i)hybrid.*runtime|adapter"):
        BackendAdapterRegistry.create_launch_plan(
            str(tmp_path / "train.yaml"),
            backend="hybrid_parallel",
            num_gpus=8,
        )


def test_fsdp_configuration_constructs_without_gpu_runtime(tmp_path):
    config = DistributedConfig(
        strategy="fsdp",
        num_gpus=2,
        num_nodes=1,
        gradient_accumulation_steps=2,
        bf16=False,
        fp16=False,
        fsdp_sharding_strategy="FULL_SHARD",
        fsdp_offload_params=True,
        fsdp_auto_wrap_policy="transformer_based",
        fsdp_sync_module_states=True,
    )
    args = config.to_training_args()
    assert args["fsdp"] == "FULL_SHARD"
    assert args["gradient_accumulation_steps"] == 2
    assert args["fsdp_config"]["fsdp_offload_params"] is True
    assert args["fsdp_config"]["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
    assert args["fsdp_config"]["fsdp_sync_module_states"] is True

    path = config.save_accelerate_config(str(tmp_path / "accelerate-fsdp.json"))
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["distributed_type"] == "FSDP"
    assert payload["num_processes"] == 2
    assert payload["fsdp_config"] == args["fsdp_config"]
