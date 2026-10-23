"""Dependency-free configuration for optional training and rollout runtimes."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields
from typing import Any, Mapping
from urllib.parse import urlsplit


def _settings(cls, raw):
    if not isinstance(raw, Mapping):
        raise TypeError(f"{cls.__name__} must be a mapping")
    unknown = set(raw) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**raw)


def _positive(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    if (
        not math.isfinite(value)
        or value <= 0
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(
            f"{name} must be a positive {'integer' if integer else 'number'}"
        )


@dataclass
class MegatronBackendConfig:
    recipe: str = ""
    recipes: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    stage_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = ""
    master_port: int = 29500
    timeout_seconds: float | None = None
    hf_export_source: str | None = None
    run_id: str | None = None

    def recipe_for(self, stage):
        return self.recipes.get(stage, self.recipe)

    def validate(self):
        for name in ("nproc_per_node", "nnodes", "master_port"):
            _positive(getattr(self, name), f"backends.megatron.{name}", integer=True)
        if self.master_port > 65535:
            raise ValueError("backends.megatron.master_port must be <= 65535")
        if isinstance(self.node_rank, bool) or not isinstance(self.node_rank, int):
            raise ValueError("backends.megatron.node_rank must be an integer")
        if not 0 <= self.node_rank < self.nnodes:
            raise ValueError("backends.megatron.node_rank must be in [0, nnodes)")
        if self.nnodes > 1 and not self.master_addr:
            raise ValueError(
                "multi-node Megatron requires backends.megatron.master_addr"
            )
        if self.run_id is not None and (
            not isinstance(self.run_id, str)
            or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", self.run_id) is None
        ):
            raise ValueError("Megatron run_id must be a safe, non-empty run identifier")
        if self.nnodes > 1 and not self.run_id:
            raise ValueError("multi-node Megatron requires one shared, unique run_id")
        if self.timeout_seconds is not None:
            _positive(self.timeout_seconds, "backends.megatron.timeout_seconds")
        for name in ("recipes", "overrides", "stage_overrides"):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"backends.megatron.{name} must be a mapping")
        if (set(self.recipes) | set(self.stage_overrides)) - {"pretrain", "sft"}:
            raise ValueError(
                "Megatron recipes/stage_overrides support pretrain and sft only"
            )
        for recipe in [self.recipe, *self.recipes.values()]:
            if not isinstance(recipe, str) or (
                recipe
                and (
                    re.fullmatch(
                        r"megatron\.bridge\.recipes\.[\w.]+:[A-Za-z]\w*", recipe
                    )
                    is None
                )
            ):
                raise ValueError(
                    "Megatron recipe must be megatron.bridge.recipes.MODULE:FUNCTION"
                )
        if any(not isinstance(value, dict) for value in self.stage_overrides.values()):
            raise TypeError("Megatron stage_overrides values must be mappings")
        for block in [self.overrides, *self.stage_overrides.values()]:
            for name in (
                "model",
                "dataset",
                "train",
                "checkpoint",
                "optimizer",
                "scheduler",
            ):
                if name in block and not isinstance(block[name], dict):
                    raise TypeError(
                        f"Megatron overrides.{name} must be a nested mapping"
                    )
        if self.hf_export_source is not None and not str(self.hf_export_source).strip():
            raise ValueError("hf_export_source must be a model/config source or null")


@dataclass
class SGLangBackendConfig:
    mode: str = "http"
    base_url: str = "http://127.0.0.1:30000"
    model_path: str | None = None
    policy_version: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 120.0
    concurrency: int = 4
    max_pending_requests: int = 128
    refresh_weights: bool = True
    return_logprob: bool = False
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if self.mode not in {"http", "engine"}:
            raise ValueError("backends.sglang.mode must be http or engine")
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "SGLang base_url must be HTTP(S), without credentials/query/fragment"
            )
        if url.path.rstrip("/").endswith("/v1"):
            raise ValueError(
                "SGLang requires the native API root, not the /v1 endpoint"
            )
        _positive(self.timeout_seconds, "backends.sglang.timeout_seconds")
        for name in ("concurrency", "max_pending_requests"):
            _positive(getattr(self, name), f"backends.sglang.{name}", integer=True)
        if self.concurrency > self.max_pending_requests:
            raise ValueError("SGLang concurrency cannot exceed max_pending_requests")
        if self.policy_version is not None and (
            not isinstance(self.policy_version, str) or not self.policy_version.strip()
        ):
            raise ValueError("SGLang policy_version must be a non-empty string or null")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str)
            or re.fullmatch(r"[A-Za-z_]\w*", self.api_key_env) is None
        ):
            raise ValueError("SGLang api_key_env must be an environment variable name")
        if self.model_path is not None and (
            not isinstance(self.model_path, str) or not self.model_path.strip()
        ):
            raise ValueError("SGLang model_path must be a non-empty string or null")
        if not isinstance(self.engine_kwargs, dict):
            raise TypeError("SGLang engine_kwargs must be a mapping")
        if set(self.engine_kwargs) & {
            "model_path",
            "weight_version",
            "api_key",
            "admin_api_key",
        }:
            raise ValueError(
                "model_path/weight_version/API keys cannot be set in engine_kwargs"
            )
        for name in ("refresh_weights", "return_logprob"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"backends.sglang.{name} must be a boolean")


@dataclass
class RuntimeBackendConfig:
    training: str = "default"
    rollout: str = "native"
    megatron: MegatronBackendConfig = field(default_factory=MegatronBackendConfig)
    sglang: SGLangBackendConfig = field(default_factory=SGLangBackendConfig)

    @classmethod
    def from_dict(cls, raw=None):
        if raw is not None and not isinstance(raw, Mapping):
            raise TypeError("backends must be a mapping")
        raw = dict(raw or {})
        raw["megatron"] = _settings(MegatronBackendConfig, raw.get("megatron", {}))
        raw["sglang"] = _settings(SGLangBackendConfig, raw.get("sglang", {}))
        result = _settings(cls, raw)
        result.validate()
        return result

    def validate(self):
        if self.training not in {"default", "megatron"}:
            raise ValueError("backends.training must be default or megatron")
        if self.rollout not in {"native", "sglang"}:
            raise ValueError("backends.rollout must be native or sglang")
        self.megatron.validate()
        self.sglang.validate()


def validate_runtime_backends(config):
    """Shared checks used by construction and read-only config validation."""
    backends = config.backends
    backends.validate()
    if backends.rollout == "sglang" and "mopd" in config.stages:
        if config.model_backend != "hf":
            raise ValueError(
                "SGLang MOPD requires a supported HF model, not a Saddle checkpoint"
            )
        if not backends.sglang.policy_version:
            raise ValueError(
                "SGLang MOPD requires policy_version to verify the student policy"
            )
    if backends.training != "megatron":
        return
    if config.distributed.strategy != "single":
        raise ValueError("Megatron owns its launcher; keep distributed.strategy=single")
    if config.distributed.num_gpus != 1 or config.distributed.num_nodes != 1:
        raise ValueError(
            "Megatron topology belongs in backends.megatron, "
            "not distributed.num_gpus/num_nodes"
        )
    if config.model_blueprint or config.model_backend != "hf":
        raise ValueError(
            "Megatron recipes do not support Saddle model blueprints/backends"
        )
    unsupported = set(config.stages) - {"pretrain", "sft", "mopd", "eval", "export"}
    if unsupported:
        raise ValueError(
            f"Megatron runtime does not implement stages: {sorted(unsupported)}"
        )
    for stage in set(config.stages) & {"pretrain", "sft"}:
        if not backends.megatron.recipe_for(stage):
            raise ValueError(f"Megatron requires a recipe for stage {stage}")
    if config.training.resume_from_checkpoint:
        raise ValueError(
            "Megatron resume uses megatron.overrides.checkpoint.load, "
            "not training.resume_from_checkpoint"
        )
    if config.pretrain_mode not in {"scratch", "continue"}:
        raise ValueError("pretrain_mode must be scratch or continue")
    if backends.megatron.nnodes > 1 and set(config.stages) - {"pretrain", "sft"}:
        raise ValueError(
            "multi-node controllers run training only; collect/eval/export separately"
        )
    if config.mopd.teacher_concurrency < 1:
        raise ValueError("mopd.teacher_concurrency must be at least 1")
    if config.eval.gate.enabled:
        from ..evaluation.ReleaseGate import EvaluationReleaseGate

        problems = EvaluationReleaseGate.validate_rules(config.eval.gate.rules)
        if problems or "eval" not in config.stages or not config.eval.enabled:
            raise ValueError(f"Invalid Megatron downstream evaluation gate: {problems}")
    if config.export.require_gate and (
        not config.eval.gate.enabled
        or "eval" not in config.stages
        or "export" not in config.stages
        or config.stages.index("eval") > config.stages.index("export")
    ):
        raise ValueError(
            "export.require_gate requires an enabled eval gate before export"
        )
    if "mopd" in config.stages and not config.mopd.dry_run:
        if not config.mopd.teachers or not config.mopd.prompts_path:
            raise ValueError("MOPD requires teachers and prompts_path")
    trained = False
    for stage in config.stages:
        if stage in {"pretrain", "sft"}:
            trained = True
        elif trained and stage in {"mopd", "eval", "export"}:
            if not backends.megatron.hf_export_source:
                raise ValueError(
                    "Megatron downstream inference/eval/export "
                    "requires hf_export_source"
                )
