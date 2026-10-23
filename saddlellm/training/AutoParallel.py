"""Automatic, fail-closed distributed training planning and launching.

The planner deliberately separates the control plane from the training
runtime.  It may select an already-supported runtime, materialize a resolved
configuration, and start the correct multi-process launcher.  It never claims
that a planned hybrid-parallel topology is executable without a matching
runtime adapter.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

RUNTIME_STRATEGIES = frozenset(
    {"single", "ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"}
)
PARALLEL_STRATEGY_ALIASES = {
    "deepspeed": "deepspeed_zero2",
    "zero2": "deepspeed_zero2",
    "zero3": "deepspeed_zero3",
    "torch_ddp": "ddp",
    "torch_fsdp": "fsdp",
}
SUPPORTED_LAUNCHERS = frozenset({"auto", "accelerate", "torchrun"})


@dataclass(frozen=True)
class HardwareProfile:
    platform: str
    accelerator: str
    visible_gpus: int
    gpu_names: tuple[str, ...] = ()
    gpu_memory_gb: tuple[float, ...] = ()
    bf16_supported: bool = False

    @property
    def minimum_gpu_memory_gb(self) -> float:
        return min(self.gpu_memory_gb) if self.gpu_memory_gb else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParallelDecision:
    requested_strategy: str
    strategy: str
    launcher: str
    num_gpus: int
    num_nodes: int
    world_size: int
    already_launched: bool
    scale_tier: str
    executable: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    hardware: Optional[HardwareProfile] = None

    @property
    def backend(self) -> str:
        return {
            "ddp": "torch_ddp",
            "fsdp": "torch_fsdp",
        }.get(self.strategy, self.strategy)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["backend"] = self.backend
        payload["launch_scope"] = (
            "in_process"
            if self.strategy == "single"
            else "local_node"
            if self.num_nodes == 1
            else "per_node_agent_external_fanout_required"
        )
        payload["deployment_scope"] = (
            "single_process"
            if self.world_size == 1
            else "single_node"
            if self.num_nodes == 1
            else "multi_node"
        )
        payload["worker_scale"] = (
            "very_large"
            if self.world_size >= 512
            else "large"
            if self.world_size >= 64
            else "small"
        )
        payload["validation_level"] = "static_preflight"
        return payload

    def require_executable(self) -> None:
        if self.executable:
            return
        detail = "; ".join(self.blockers) or "parallel plan is not executable"
        raise RuntimeError(detail)


@dataclass
class ParallelLaunchResult:
    decision: ParallelDecision
    config_path: str
    launch_plan: Dict[str, Any]
    returncode: Optional[int] = None
    training_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "config_path": self.config_path,
            "launch_plan": self.launch_plan,
            "returncode": self.returncode,
            "training_results": self.training_results,
        }


def detect_hardware() -> HardwareProfile:
    """Inspect only local, visible devices; ``CUDA_VISIBLE_DEVICES`` is honored."""

    names: List[str] = []
    memory: List[float] = []
    bf16 = False
    accelerator = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            accelerator = "cuda"
            for index in range(torch.cuda.device_count()):
                names.append(str(torch.cuda.get_device_name(index)))
                props = torch.cuda.get_device_properties(index)
                memory.append(round(float(props.total_memory) / (1024**3), 2))
            bf16 = bool(torch.cuda.is_bf16_supported())
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
    except Exception:
        # Planning should remain usable on login nodes that do not import the
        # training runtime. Explicit topology flags can still describe GPUs.
        pass
    return HardwareProfile(
        platform=platform.platform(),
        accelerator=accelerator,
        visible_gpus=len(names),
        gpu_names=tuple(names),
        gpu_memory_gb=tuple(memory),
        bf16_supported=bf16,
    )


def normalize_parallel_strategy(value: Optional[str]) -> str:
    normalized = str(value or "single").strip().lower().replace("-", "_")
    normalized = PARALLEL_STRATEGY_ALIASES.get(normalized, normalized)
    if normalized not in RUNTIME_STRATEGIES | {"auto"}:
        raise ValueError(f"unsupported parallel strategy: {value!r}")
    return normalized


def plan_parallel_training(
    config: Mapping[str, Any],
    *,
    strategy: Optional[str] = None,
    num_gpus: Optional[int] = None,
    num_nodes: Optional[int] = None,
    launcher: str = "auto",
    hardware: Optional[HardwareProfile] = None,
) -> ParallelDecision:
    """Resolve an executable topology without changing training semantics."""

    if not isinstance(config, Mapping):
        raise TypeError("training config must be a mapping")
    dist = config.get("distributed", {})
    if not isinstance(dist, Mapping):
        raise TypeError("distributed must be a mapping")
    hardware = hardware or detect_hardware()
    configured_strategy = normalize_parallel_strategy(dist.get("strategy", "single"))
    topology_override = num_gpus is not None or num_nodes is not None
    if strategy is None and configured_strategy == "single" and topology_override:
        # A topology override is itself an opt-in to automatic parallelism, so
        # ``--num-gpus 8`` does not also need ``--parallel auto``.
        requested = "auto"
    else:
        requested = (
            normalize_parallel_strategy(strategy)
            if strategy is not None
            else configured_strategy
        )

    launched_world_size = _env_int("WORLD_SIZE", 1)
    has_worker_ranks = all(
        os.environ.get(name) not in {None, ""} for name in ("RANK", "LOCAL_RANK")
    )
    child_marker = os.environ.get("SADDLELLM_PARALLEL_CHILD") == "1"
    already_launched = launched_world_size > 1 and (has_worker_ranks or child_marker)
    launched_local_size = _optional_env_int("LOCAL_WORLD_SIZE")
    launched_node_count = _optional_env_int("GROUP_WORLD_SIZE")
    topology_errors: List[str] = []

    configured_gpus = int(dist["num_gpus"]) if "num_gpus" in dist else None
    configured_nodes = int(dist["num_nodes"]) if "num_nodes" in dist else None
    declared_gpus = int(num_gpus) if num_gpus is not None else configured_gpus
    declared_nodes = int(num_nodes) if num_nodes is not None else configured_nodes
    inferred_gpus: Optional[int] = None
    inferred_nodes: Optional[int] = None

    if already_launched:
        inferred_gpus = launched_local_size
        inferred_nodes = launched_node_count
        if inferred_gpus is not None and inferred_nodes is None:
            if launched_world_size % inferred_gpus:
                topology_errors.append(
                    "WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE"
                )
            else:
                inferred_nodes = launched_world_size // inferred_gpus
        elif inferred_nodes is not None and inferred_gpus is None:
            if launched_world_size % inferred_nodes:
                topology_errors.append(
                    "WORLD_SIZE must be divisible by GROUP_WORLD_SIZE"
                )
            else:
                inferred_gpus = launched_world_size // inferred_nodes
        elif inferred_gpus is None and inferred_nodes is None:
            if declared_gpus is not None and declared_nodes is None:
                if launched_world_size % declared_gpus:
                    topology_errors.append(
                        "WORLD_SIZE must be divisible by the declared num_gpus"
                    )
                else:
                    inferred_nodes = launched_world_size // declared_gpus
            elif declared_nodes is not None and declared_gpus is None:
                if launched_world_size % declared_nodes:
                    topology_errors.append(
                        "WORLD_SIZE must be divisible by the declared num_nodes"
                    )
                else:
                    inferred_gpus = launched_world_size // declared_nodes
            elif declared_gpus is None and declared_nodes is None:
                topology_errors.append(
                    "cannot infer per-node topology from WORLD_SIZE alone; provide "
                    "LOCAL_WORLD_SIZE/GROUP_WORLD_SIZE or explicit num_gpus/num_nodes"
                )

    if declared_nodes is not None:
        resolved_nodes = declared_nodes
    elif already_launched and inferred_nodes is not None:
        resolved_nodes = inferred_nodes
    else:
        resolved_nodes = 1
    if resolved_nodes < 1:
        raise ValueError("num_nodes must be >= 1")

    if declared_gpus is not None:
        resolved_gpus = declared_gpus
    elif already_launched:
        if inferred_gpus is not None:
            resolved_gpus = inferred_gpus
        elif launched_world_size % resolved_nodes == 0:
            resolved_gpus = launched_world_size // resolved_nodes
        else:
            topology_errors.append(
                "WORLD_SIZE must be divisible by the resolved num_nodes"
            )
            resolved_gpus = launched_world_size
    elif strategy is not None and requested == "auto":
        # An explicit --parallel auto means "use every visible device". The
        # visibility environment remains the user's natural device limiter.
        resolved_gpus = max(1, hardware.visible_gpus)
    elif configured_strategy == "auto" and "num_gpus" not in dist:
        resolved_gpus = max(1, hardware.visible_gpus)
    else:
        resolved_gpus = int(dist.get("num_gpus", 1))
    if resolved_gpus < 1:
        raise ValueError("num_gpus must be >= 1")

    world_size = (
        launched_world_size if already_launched else resolved_gpus * resolved_nodes
    )
    reasons: List[str] = []
    warnings: List[str] = []
    blockers: List[str] = list(topology_errors)

    if requested == "auto":
        resolved_strategy = _auto_strategy(
            config,
            world_size=world_size,
            hardware=hardware,
            reasons=reasons,
        )
    else:
        resolved_strategy = requested
        reasons.append(f"strategy explicitly resolved as {resolved_strategy}")

    if world_size == 1 and resolved_strategy != "single":
        blockers.append(
            f"strategy {resolved_strategy!r} requires at least two worker processes"
        )
    if world_size > 1 and resolved_strategy == "single":
        blockers.append("strategy 'single' cannot be used with world_size > 1")

    stages = tuple(str(item).strip().lower() for item in config.get("stages", ()))
    from ..framework.stages import get_stage_registry

    stage_registry = get_stage_registry()
    for stage in stages:
        if not stage_registry.contains(stage):
            blockers.append(f"unknown training stage {stage!r}")
            continue
        capabilities = stage_registry.create(stage).capabilities
        if not capabilities.supports_strategy(resolved_strategy):
            blockers.append(
                f"stage {stage!r} has not been verified with strategy "
                f"{resolved_strategy!r}"
            )

    stage_set = set(stages)
    model = config.get("model", {})
    model_backend = (
        str(model.get("backend", "hf")).strip().lower()
        if isinstance(model, Mapping)
        else "hf"
    )
    if (
        resolved_strategy != "single"
        and "pretrain" in stage_set
        and model_backend != "saddle"
    ):
        blockers.append(
            "distributed pretraining is currently verified for "
            "model.backend='saddle' only"
        )
    if resolved_strategy != "single" and stage_set & {"sft", "preference"}:
        try:
            from ..models.ModelAdapter import get_model_adapter

            adapter = get_model_adapter(model_backend)
            if not adapter.capabilities.distributed_post_training:
                blockers.append(
                    f"model backend {model_backend!r} does not support distributed "
                    "post-training"
                )
        except ValueError as exc:
            blockers.append(str(exc))
    if resolved_nodes > 1 and stage_set & {"sft", "preference"}:
        blockers.append(
            "post-training DDP is currently limited to one node; shared-filesystem "
            "and multi-node checkpoint handoff have not been verified"
        )

    if (
        resolved_strategy.startswith("deepspeed")
        and importlib.util.find_spec("deepspeed") is None
    ):
        blockers.append(
            "DeepSpeed was selected but the 'deepspeed' package is not installed "
            "in this runtime"
        )

    requested_launcher = str(launcher or dist.get("launcher", "auto")).strip().lower()
    if requested_launcher not in SUPPORTED_LAUNCHERS:
        raise ValueError(
            f"unsupported launcher {requested_launcher!r}; expected one of "
            f"{sorted(SUPPORTED_LAUNCHERS)}"
        )
    resolved_launcher = (
        "accelerate" if requested_launcher == "auto" else requested_launcher
    )
    if resolved_strategy == "single":
        resolved_launcher = "in_process"
        if requested_launcher != "auto":
            blockers.append(
                f"launcher {requested_launcher!r} requires a distributed strategy"
            )
    elif (
        resolved_launcher == "accelerate"
        and importlib.util.find_spec("accelerate") is None
    ):
        blockers.append(
            "Accelerate launcher was selected but 'accelerate' is not installed"
        )

    if already_launched and world_size != resolved_gpus * resolved_nodes:
        blockers.append(
            "launcher topology does not match the resolved config: "
            f"WORLD_SIZE={world_size}, "
            f"num_gpus*num_nodes={resolved_gpus * resolved_nodes}"
        )
    if (
        already_launched
        and launched_local_size is not None
        and launched_local_size != resolved_gpus
    ):
        blockers.append(
            "launcher local topology does not match the resolved config: "
            f"LOCAL_WORLD_SIZE={launched_local_size}, num_gpus={resolved_gpus}"
        )
    if (
        already_launched
        and launched_node_count is not None
        and launched_node_count != resolved_nodes
    ):
        blockers.append(
            "launcher node topology does not match the resolved config: "
            f"GROUP_WORLD_SIZE={launched_node_count}, num_nodes={resolved_nodes}"
        )

    if world_size >= 64:
        warnings.append(
            "Large-scale topology detected. Pure data/sharded-data parallelism may be "
            "insufficient; TP/PP/CP/EP requires a separately verified "
            "Megatron-style runtime."
        )
    if resolved_nodes > 1:
        warnings.append(
            "Multi-node launch is a per-node agent command: an external scheduler must "
            "fan it out, and training paths/checkpoints must be visible where required."
        )
    if resolved_nodes > 1 and not already_launched:
        if not _multi_node_environment_available(resolved_launcher):
            blockers.append(
                "multi-node execution requires MACHINE_RANK and MAIN_PROCESS_IP "
                "(or RANK/MASTER_ADDR supplied by the scheduler) on every node"
            )
    if (
        not already_launched
        and hardware.visible_gpus
        and resolved_nodes == 1
        and resolved_gpus > hardware.visible_gpus
    ):
        blockers.append(
            f"requested {resolved_gpus} local GPUs but only "
            f"{hardware.visible_gpus} are visible"
        )
    if hardware.accelerator != "cuda" and world_size > 1:
        warnings.append(
            "No local CUDA devices were detected; this plan is only executable "
            "if the launcher runs in a GPU allocation or intentionally uses "
            "CPU workers."
        )

    return ParallelDecision(
        requested_strategy=requested,
        strategy=resolved_strategy,
        launcher=resolved_launcher,
        num_gpus=resolved_gpus,
        num_nodes=resolved_nodes,
        world_size=world_size,
        already_launched=already_launched,
        scale_tier=_scale_tier(world_size, resolved_nodes),
        executable=not blockers,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        blockers=tuple(blockers),
        hardware=hardware,
    )


def apply_parallel_decision(
    config: Mapping[str, Any], decision: ParallelDecision
) -> Dict[str, Any]:
    """Return a copy with only runtime topology fields resolved."""

    resolved = copy.deepcopy(dict(config))
    distributed = resolved.setdefault("distributed", {})
    if not isinstance(distributed, dict):
        raise TypeError("distributed must be a mapping")
    distributed.update(
        {
            "strategy": decision.strategy,
            "num_gpus": decision.num_gpus,
            "num_nodes": decision.num_nodes,
            "launcher": decision.launcher,
            "world_size": decision.world_size,
            "resolved_by": "saddlellm.auto_parallel",
        }
    )
    return resolved


def materialize_parallel_config(
    config: Mapping[str, Any],
    decision: ParallelDecision,
    *,
    output_path: Optional[str] = None,
) -> str:
    """Persist the resolved worker config as an auditable run artifact."""

    resolved = apply_parallel_decision(config, decision)
    if output_path is None:
        output_path = os.path.join(
            parallel_artifact_directory(resolved, decision),
            "resolved_config.json",
        )
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return str(path)


def parallel_artifact_directory(
    config: Mapping[str, Any], decision: ParallelDecision
) -> str:
    """Return a collision-free directory for controller launch artifacts."""

    logging = config.get("logging", {})
    if not isinstance(logging, Mapping):
        raise TypeError("logging must be a mapping")
    root = Path(logging.get("output_dir", "./outputs")).expanduser() / "parallel"
    if decision.num_nodes > 1 and not decision.already_launched:
        machine_rank = _controller_machine_rank()
        suffix = "node-unknown" if machine_rank is None else f"node-{machine_rank:05d}"
        root /= suffix
    return str(root.resolve())


def create_parallel_launch_plan(config_path: str, decision: ParallelDecision):
    """Build a concrete launcher command for a resolved decision."""

    from ..factory.BackendAdapters import BackendAdapterRegistry

    decision.require_executable()
    launcher = "accelerate" if decision.launcher == "in_process" else decision.launcher
    plan = BackendAdapterRegistry.create_launch_plan(
        config_path,
        backend=decision.backend,
        num_gpus=decision.num_gpus,
        num_nodes=decision.num_nodes,
        launcher=launcher,
    )
    plan.env.setdefault("SADDLELLM_PARALLEL_CHILD", "1")
    plan.env.setdefault("PYTHONUTF8", "1")
    return plan


def run_launch_plan(plan: Any, *, check: bool = False) -> int:
    """Execute a launch plan without shell interpolation."""

    env = os.environ.copy()
    env.update(getattr(plan, "env", {}) or {})
    completed = subprocess.run(list(plan.command), env=env, check=False)
    if check and completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, plan.command)
    return int(completed.returncode)


def launch_training(
    config_path: str,
    *,
    strategy: str = "auto",
    num_gpus: Optional[int] = None,
    num_nodes: Optional[int] = None,
    launcher: str = "auto",
    dry_run: bool = False,
    check: bool = False,
) -> ParallelLaunchResult:
    """Python API for one-call planning and multi-process training startup."""

    config = _load_config(config_path)
    decision = plan_parallel_training(
        config,
        strategy=strategy,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        launcher=launcher,
    )
    decision.require_executable()
    resolved_path = materialize_parallel_config(config, decision)
    resolved_config = apply_parallel_decision(config, decision)
    if decision.already_launched or decision.strategy == "single":
        result = ParallelLaunchResult(
            decision=decision,
            config_path=resolved_path,
            launch_plan={
                "launcher": "existing" if decision.already_launched else "in_process",
                "command": [],
                "topology": {
                    "num_gpus_per_node": decision.num_gpus,
                    "num_nodes": decision.num_nodes,
                    "world_size": decision.world_size,
                },
            },
        )
        if not dry_run:
            from .TrainingOrchestrator import TrainingOrchestrator

            result.training_results = TrainingOrchestrator.from_dict(
                resolved_config
            ).run()
            result.returncode = 0
        return result

    plan = create_parallel_launch_plan(resolved_path, decision)
    result = ParallelLaunchResult(
        decision=decision,
        config_path=resolved_path,
        launch_plan=plan.to_dict(),
    )
    if not dry_run:
        result.returncode = run_launch_plan(plan, check=check)
    return result


def _auto_strategy(
    config: Mapping[str, Any],
    *,
    world_size: int,
    hardware: HardwareProfile,
    reasons: List[str],
) -> str:
    if world_size <= 1:
        reasons.append("one worker is available; selected single-process training")
        return "single"

    stages = {str(item).strip().lower() for item in config.get("stages", ())}
    post_training = bool(stages & {"sft", "preference"})
    if post_training:
        reasons.append(
            "post-training currently uses verified data parallelism; selected DDP"
        )
        return "ddp"

    model_params = _model_parameter_hint(config)
    memory_gb = hardware.minimum_gpu_memory_gb or float(
        config.get("distributed", {}).get("gpu_memory_gb", 0) or 0
    )
    if model_params and memory_gb:
        # AdamW mixed-precision training is commonly around 16 bytes/parameter
        # before activations and temporary buffers. Keep a conservative margin.
        estimated_state_gb = model_params * 16 / (1024**3)
        if estimated_state_gb > memory_gb * 0.72:
            if world_size >= 4 and importlib.util.find_spec("deepspeed") is not None:
                reasons.append(
                    "estimated model state does not fit per GPU; selected "
                    "DeepSpeed ZeRO-3"
                )
                return "deepspeed_zero3"
            reasons.append(
                "estimated model state does not fit per GPU; selected PyTorch FSDP"
            )
            return "fsdp"

    reasons.append(
        "multiple workers are available and no sharding need was proven; selected DDP"
    )
    return "ddp"


def _model_parameter_hint(config: Mapping[str, Any]) -> Optional[int]:
    distributed = config.get("distributed", {})
    model = config.get("model", {})
    for source in (distributed, model):
        if not isinstance(source, Mapping):
            continue
        for key in ("model_params", "estimated_params", "num_parameters"):
            value = source.get(key)
            if value is not None:
                parsed = int(value)
                if parsed <= 0:
                    raise ValueError(f"{key} must be positive")
                return parsed
    model_config = model.get("config") if isinstance(model, Mapping) else None
    if isinstance(model_config, str):
        try:
            from ..models.ModelRegistry import ModelRegistry

            return int(ModelRegistry.get(model_config).estimated_params)
        except (KeyError, ImportError, AttributeError, TypeError, ValueError):
            pass
    return None


def _multi_node_environment_available(launcher: str) -> bool:
    del launcher  # Both supported launchers need the same rank/address inputs.
    has_main_address = bool(
        os.environ.get("MAIN_PROCESS_IP") or os.environ.get("MASTER_ADDR")
    )
    if not has_main_address:
        return False
    if any(
        os.environ.get(name) not in {None, ""}
        for name in ("MACHINE_RANK", "GROUP_RANK", "NODE_RANK", "SLURM_NODEID")
    ):
        return True
    return all(
        os.environ.get(name) not in {None, ""} for name in ("RANK", "LOCAL_WORLD_SIZE")
    )


def _controller_machine_rank() -> Optional[int]:
    for name in ("MACHINE_RANK", "GROUP_RANK", "NODE_RANK", "SLURM_NODEID"):
        value = _optional_env_int(name, allow_zero=True)
        if value is not None:
            return value
    rank = _optional_env_int("RANK", allow_zero=True)
    local_world_size = _optional_env_int("LOCAL_WORLD_SIZE")
    if rank is not None and local_world_size is not None:
        return rank // local_world_size
    return None


def _scale_tier(world_size: int, num_nodes: int) -> str:
    if world_size == 1:
        return "single_device"
    if world_size >= 512:
        return "very_large_scale"
    if world_size >= 64:
        return "large_scale"
    if num_nodes > 1:
        return "multi_node"
    return "single_node"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in {None, ""}:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _optional_env_int(name: str, *, allow_zero: bool = False) -> Optional[int]:
    raw = os.environ.get(name)
    if raw in {None, ""}:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = ">= 0" if allow_zero else ">= 1"
        raise ValueError(f"{name} must be {comparator}")
    return value


def _load_config(path: str) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"training config does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            payload = yaml.safe_load(handle) or {}
        else:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("training config must be a YAML/JSON object")
    return payload


__all__ = [
    "HardwareProfile",
    "ParallelDecision",
    "ParallelLaunchResult",
    "apply_parallel_decision",
    "create_parallel_launch_plan",
    "detect_hardware",
    "launch_training",
    "materialize_parallel_config",
    "normalize_parallel_strategy",
    "parallel_artifact_directory",
    "plan_parallel_training",
    "run_launch_plan",
]
