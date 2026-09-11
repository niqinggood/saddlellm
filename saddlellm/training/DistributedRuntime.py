"""Small, dependency-free helpers shared by distributed training paths."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DistributedRuntimeInfo:
    """Describe the process topology visible to the current worker."""

    strategy: str = "single"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    local_world_size: int = 1

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(cls, strategy: str = "single") -> "DistributedRuntimeInfo":
        return cls(
            strategy=str(strategy),
            rank=_env_int("RANK", 0),
            local_rank=_env_int("LOCAL_RANK", 0),
            world_size=_env_int("WORLD_SIZE", 1),
            local_world_size=_env_int("LOCAL_WORLD_SIZE", 1),
        )


def current_distributed_runtime(strategy: str = "single") -> DistributedRuntimeInfo:
    """Return initialized torch.distributed state, falling back to launcher env."""

    info = DistributedRuntimeInfo.from_env(strategy)
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return DistributedRuntimeInfo(
                strategy=str(strategy),
                rank=int(dist.get_rank()),
                local_rank=info.local_rank,
                world_size=int(dist.get_world_size()),
                local_world_size=info.local_world_size,
            )
    except Exception:
        pass
    return info


def validate_distributed_runtime(
    strategy: str,
    num_gpus: int,
    num_nodes: int = 1,
    *,
    require_initialized: bool = False,
) -> DistributedRuntimeInfo:
    """Fail when a distributed config is executed by an incompatible launcher."""

    strategy = str(strategy).lower()
    info = current_distributed_runtime(strategy)
    processes_per_node = int(num_gpus)
    nodes = int(num_nodes)
    if processes_per_node < 1 or nodes < 1:
        raise ValueError("num_gpus and num_nodes must both be >= 1")
    expected = processes_per_node * nodes
    if strategy == "single":
        if info.world_size != 1:
            raise ValueError(
                f"distributed.strategy='single' cannot run with WORLD_SIZE={info.world_size}"
            )
        return info

    if strategy not in {"ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"}:
        raise ValueError(f"unsupported distributed strategy: {strategy!r}")
    if expected < 2:
        raise ValueError(f"distributed.strategy={strategy!r} requires at least 2 processes")
    if require_initialized and info.world_size == 1:
        raise RuntimeError(
            f"distributed.strategy={strategy!r} requires a multi-process launcher; "
            "use the generated launch script or torchrun/accelerate launch"
        )
    if info.world_size > 1 and info.world_size != expected:
        raise ValueError(
            f"configured world size is {expected} ({num_gpus} processes x {num_nodes} nodes), "
            f"but launcher WORLD_SIZE is {info.world_size}"
        )
    return info


def distributed_barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
    except Exception:
        return


def _env_int(name: str, default: int) -> int:
    value: Optional[str] = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0 or (name in {"WORLD_SIZE", "LOCAL_WORLD_SIZE"} and parsed < 1):
        raise ValueError(f"{name} has invalid value {parsed}")
    return parsed


__all__ = [
    "DistributedRuntimeInfo",
    "current_distributed_runtime",
    "distributed_barrier",
    "validate_distributed_runtime",
]
