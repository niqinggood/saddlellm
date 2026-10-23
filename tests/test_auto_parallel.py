from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from saddlellm import cli
from saddlellm.factory.BackendAdapters import BackendAdapterRegistry, LaunchPlan
from saddlellm.training import AutoParallel
from saddlellm.training.AutoParallel import (
    HardwareProfile,
    apply_parallel_decision,
    create_parallel_launch_plan,
    parallel_artifact_directory,
    plan_parallel_training,
)
from saddlellm.training.DistributedRuntime import (
    distributed_barrier,
    validate_distributed_runtime,
)


def _hardware(count: int = 2) -> HardwareProfile:
    return HardwareProfile(
        platform="test",
        accelerator="cuda" if count else "cpu",
        visible_gpus=count,
        gpu_names=tuple(f"gpu-{index}" for index in range(count)),
        gpu_memory_gb=tuple(24.0 for _ in range(count)),
        bf16_supported=bool(count),
    )


def _clear_parallel_env(monkeypatch) -> None:
    for name in (
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "MACHINE_RANK",
        "GROUP_RANK",
        "NODE_RANK",
        "SLURM_NODEID",
        "MASTER_ADDR",
        "MAIN_PROCESS_IP",
        "MASTER_PORT",
        "MAIN_PROCESS_PORT",
        "SADDLELLM_PARALLEL_CHILD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_auto_uses_single_process_for_one_visible_gpu(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    decision = plan_parallel_training(
        {"stages": ["sft"], "model": {"backend": "hf"}},
        strategy="auto",
        launcher="auto",
        hardware=_hardware(1),
    )

    assert decision.strategy == "single"
    assert decision.world_size == 1
    assert decision.launcher == "in_process"
    assert decision.executable


def test_num_gpus_alone_opts_into_auto_ddp(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    decision = plan_parallel_training(
        {
            "stages": ["sft"],
            "model": {"backend": "hf"},
            "distributed": {"strategy": "single", "num_gpus": 1},
        },
        num_gpus=2,
        launcher="torchrun",
        hardware=_hardware(2),
    )

    assert decision.requested_strategy == "auto"
    assert decision.strategy == "ddp"
    assert decision.world_size == 2
    assert decision.executable


def test_post_training_rejects_unverified_sharded_strategies(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    decision = plan_parallel_training(
        {"stages": ["sft"], "model": {"backend": "hf"}},
        strategy="fsdp",
        num_gpus=2,
        launcher="torchrun",
        hardware=_hardware(2),
    )

    assert not decision.executable
    assert any("has not been verified" in item for item in decision.blockers)


def test_deepspeed_selection_fails_closed_when_dependency_is_missing(
    monkeypatch,
) -> None:
    _clear_parallel_env(monkeypatch)
    original_find_spec = AutoParallel.importlib.util.find_spec
    monkeypatch.setattr(
        AutoParallel.importlib.util,
        "find_spec",
        lambda name: None if name == "deepspeed" else original_find_spec(name),
    )
    decision = plan_parallel_training(
        {"stages": ["pretrain"], "model": {"backend": "saddle"}},
        strategy="deepspeed-zero3",
        num_gpus=4,
        launcher="torchrun",
        hardware=_hardware(4),
    )

    assert not decision.executable
    assert any("not installed" in item for item in decision.blockers)


def test_apply_decision_does_not_mutate_user_config(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    config = {
        "stages": ["sft"],
        "model": {"backend": "hf"},
        "distributed": {"strategy": "single"},
    }
    decision = plan_parallel_training(
        config,
        strategy="auto",
        num_gpus=2,
        launcher="torchrun",
        hardware=_hardware(2),
    )

    resolved = apply_parallel_decision(config, decision)

    assert config["distributed"] == {"strategy": "single"}
    assert resolved["distributed"]["strategy"] == "ddp"
    assert resolved["distributed"]["world_size"] == 2


def test_existing_launcher_topology_mismatch_is_not_silently_rewritten(
    monkeypatch,
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("GROUP_WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    decision = plan_parallel_training(
        {
            "stages": ["pretrain"],
            "model": {"backend": "saddle"},
            "distributed": {
                "strategy": "ddp",
                "num_gpus": 2,
                "num_nodes": 2,
            },
        },
        launcher="torchrun",
        hardware=_hardware(4),
    )

    assert decision.already_launched
    assert decision.num_gpus == 2
    assert decision.num_nodes == 2
    assert not decision.executable
    assert any("WORLD_SIZE=8" in item for item in decision.blockers)
    assert any("LOCAL_WORLD_SIZE=4" in item for item in decision.blockers)


def test_existing_launcher_can_fill_missing_topology(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("GROUP_WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    decision = plan_parallel_training(
        {
            "stages": ["pretrain"],
            "model": {"backend": "saddle"},
            "distributed": {"strategy": "ddp"},
        },
        launcher="torchrun",
        hardware=_hardware(4),
    )

    assert decision.executable
    assert (decision.num_gpus, decision.num_nodes, decision.world_size) == (4, 2, 8)


def test_existing_launcher_with_world_size_only_fails_closed(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    decision = plan_parallel_training(
        {
            "stages": ["pretrain"],
            "model": {"backend": "saddle"},
            "distributed": {"strategy": "ddp"},
        },
        launcher="torchrun",
        hardware=_hardware(8),
    )

    assert decision.already_launched
    assert not decision.executable
    assert any("cannot infer per-node topology" in item for item in decision.blockers)


def test_stale_world_size_alone_is_not_treated_as_worker(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    decision = plan_parallel_training(
        {"stages": ["sft"], "model": {"backend": "hf"}},
        strategy="auto",
        num_gpus=2,
        launcher="torchrun",
        hardware=_hardware(2),
    )

    assert not decision.already_launched
    assert decision.world_size == 2


def test_accelerate_uses_global_process_count_for_multi_node(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("MACHINE_RANK", "1")
    monkeypatch.setenv("MAIN_PROCESS_IP", "10.0.0.10")
    plan = BackendAdapterRegistry.create_launch_plan(
        "train config.yaml",
        backend="ddp",
        num_gpus=4,
        num_nodes=2,
        launcher="accelerate",
    )

    assert plan.topology["world_size"] == 8
    assert plan.command[plan.command.index("--num_processes") + 1] == "8"
    assert plan.command[plan.command.index("--machine_rank") + 1] == "1"


def test_powershell_launch_script_quotes_paths_and_preserves_exit_code(
    tmp_path,
) -> None:
    script_path = tmp_path / "launch.ps1"
    plan = LaunchPlan(
        backend="torch_ddp",
        launcher="torchrun",
        command=["python.exe", "-m", "saddlellm.cli", "训练 a'b&$(x).yaml"],
        env={"SADDLE_VALUE": "a'b"},
    )

    BackendAdapterRegistry.save_launch_script(plan, str(script_path))
    script = script_path.read_text(encoding="utf-8-sig")

    assert "$env:SADDLE_VALUE = 'a''b'" in script
    assert "'训练 a''b&$(x).yaml'" in script
    assert script.rstrip().endswith("exit $LASTEXITCODE")


def test_thousand_worker_plan_is_per_node_static_preflight(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("SLURM_NODEID", "3")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    decision = plan_parallel_training(
        {"stages": ["pretrain"], "model": {"backend": "saddle"}},
        strategy="ddp",
        num_gpus=8,
        num_nodes=125,
        launcher="torchrun",
        hardware=_hardware(8),
    )
    plan = create_parallel_launch_plan("train.yaml", decision)

    assert decision.executable
    assert decision.world_size == 1000
    assert decision.scale_tier == "very_large_scale"
    assert decision.to_dict()["launch_scope"].endswith("external_fanout_required")
    assert decision.to_dict()["validation_level"] == "static_preflight"
    assert any("external scheduler" in item for item in decision.warnings)
    assert plan.topology["world_size"] == 1000
    assert plan.command[plan.command.index("--nproc-per-node") + 1] == "8"
    assert plan.command[plan.command.index("--nnodes") + 1] == "125"
    assert plan.env["SADDLELLM_PARALLEL_CHILD"] == "1"
    assert parallel_artifact_directory(
        {"logging": {"output_dir": "outputs"}}, decision
    ).endswith(str(Path("parallel") / "node-00003"))


def test_post_training_multi_node_is_blocked_until_shared_storage_is_verified(
    monkeypatch,
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("MACHINE_RANK", "0")
    monkeypatch.setenv("MAIN_PROCESS_IP", "10.0.0.1")
    decision = plan_parallel_training(
        {"stages": ["sft"], "model": {"backend": "hf"}},
        strategy="ddp",
        num_gpus=8,
        num_nodes=2,
        launcher="torchrun",
        hardware=_hardware(8),
    )

    assert not decision.executable
    assert any("limited to one node" in item for item in decision.blockers)


def test_distributed_runtime_requires_rank_environment(monkeypatch) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    with pytest.raises(RuntimeError, match="RANK, LOCAL_RANK"):
        validate_distributed_runtime("ddp", 2)


def test_initialized_barrier_failure_is_not_swallowed(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def fail_barrier():
        raise RuntimeError("collective failed")

    monkeypatch.setattr(dist, "barrier", fail_barrier)
    with pytest.raises(RuntimeError, match="collective failed"):
        distributed_barrier()


def test_progress_status_retries_a_transient_windows_replace_failure(
    tmp_path, monkeypatch
) -> None:
    from saddlellm.training import TrainingProgress

    callback = TrainingProgress.LocalProgressCallback(str(tmp_path))
    state = SimpleNamespace(
        global_step=0,
        max_steps=1,
        epoch=0.0,
        is_world_process_zero=True,
    )
    real_replace = TrainingProgress.os.replace
    attempts = 0

    def transient_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("transient file lock")
        return real_replace(source, destination)

    monkeypatch.setattr(TrainingProgress.os, "replace", transient_replace)
    callback.on_train_begin(None, state, object())

    assert attempts == 2
    assert json.loads(
        (tmp_path / "training_progress.json").read_text(encoding="utf-8")
    )["status"] == "running"
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_auto_parallel_dry_run_reports_resolved_ddp(
    tmp_path, capsys, monkeypatch
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setattr(AutoParallel, "detect_hardware", lambda: _hardware(2))
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "stages": ["sft"],
                "model": {"backend": "hf"},
                "distributed": {"strategy": "single"},
                "logging": {"output_dir": str(tmp_path / "run")},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "train",
            str(config_path),
            "--parallel",
            "auto",
            "--launcher",
            "torchrun",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["parallel"]["strategy"] == "ddp"
    assert payload["parallel"]["world_size"] == 2
    assert payload["distributed"]["strategy"] == "ddp"
    assert not (tmp_path / "run").exists()


def test_public_launch_training_dry_run_builds_worker_plan(
    tmp_path, monkeypatch
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setattr(AutoParallel, "detect_hardware", lambda: _hardware(2))
    config_path = tmp_path / "train.json"
    output_dir = tmp_path / "run"
    config_path.write_text(
        json.dumps(
            {
                "stages": ["sft"],
                "model": {"backend": "hf"},
                "distributed": {"strategy": "single"},
                "logging": {"output_dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )

    from saddlellm import launch_training

    result = launch_training(
        str(config_path),
        strategy="auto",
        num_gpus=2,
        launcher="torchrun",
        dry_run=True,
    )

    assert result.decision.strategy == "ddp"
    assert result.decision.world_size == 2
    assert result.returncode is None
    assert result.launch_plan["launcher"] == "torchrun"
    assert "torch.distributed.run" in result.launch_plan["command"]
    assert Path(result.config_path).is_file()


def test_cli_controller_preserves_return_code_and_writes_launch_artifacts(
    tmp_path, capsys, monkeypatch
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setattr(AutoParallel, "detect_hardware", lambda: _hardware(2))
    monkeypatch.setattr(AutoParallel, "run_launch_plan", lambda _plan: 42)
    output_dir = tmp_path / "run"
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "stages": ["sft"],
                "model": {"backend": "hf"},
                "distributed": {"strategy": "single"},
                "logging": {"output_dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "train",
            str(config_path),
            "--parallel",
            "auto",
            "--launcher",
            "torchrun",
            "--debug",
        ]
    )
    captured = capsys.readouterr()
    failure = json.loads(captured.out)
    launching = json.loads(captured.err)
    plan_path = output_dir / "parallel" / "launch_plan.json"
    saved_plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert exit_code == 42
    assert launching["status"] == "launching"
    assert failure["status"] == "launcher_failed"
    assert failure["returncode"] == 42
    assert (output_dir / "parallel" / "resolved_config.json").is_file()
    assert (output_dir / "parallel" / "launch.ps1").is_file()
    assert "--parallel" not in saved_plan["command"]
    assert saved_plan["command"][-1] == "--debug"
    assert saved_plan["env"]["SADDLELLM_PARALLEL_CHILD"] == "1"


def test_cli_worker_dry_run_does_not_launch_a_second_controller(
    tmp_path, capsys, monkeypatch
) -> None:
    _clear_parallel_env(monkeypatch)
    monkeypatch.setenv("SADDLELLM_PARALLEL_CHILD", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("GROUP_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(AutoParallel, "detect_hardware", lambda: _hardware(2))

    def unexpected_launch(_plan):
        raise AssertionError("a worker must not relaunch a controller")

    monkeypatch.setattr(AutoParallel, "run_launch_plan", unexpected_launch)
    config_path = tmp_path / "resolved.json"
    config_path.write_text(
        json.dumps(
            {
                "stages": ["sft"],
                "model": {"backend": "hf"},
                "distributed": {
                    "strategy": "ddp",
                    "num_gpus": 2,
                    "num_nodes": 1,
                    "launcher": "torchrun",
                },
                "logging": {"output_dir": str(tmp_path / "run")},
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["train", str(config_path), "--dry-run"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["parallel"]["already_launched"] is True
    assert payload["parallel"]["world_size"] == 2
