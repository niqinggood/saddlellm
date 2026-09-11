"""Dependency-aware execution plans and resumable pipeline state.

The training implementations live in the individual SaddleLLM stages.  This
module deliberately contains no Torch or Transformers imports; it provides the
small, stable control-plane contract that can connect foundation training,
post-training, embodied policies, and world models in one run.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_STATE_API_VERSION = "saddlellm.io/pipeline-state/v1alpha1"


class PipelinePlanError(ValueError):
    """Raised when stages cannot be compiled into a valid dependency graph."""


class PipelineStateError(RuntimeError):
    """Raised when persisted state is incompatible with the requested run."""


@dataclass(frozen=True)
class PipelineNode:
    """One uniquely named stage in a compiled execution graph."""

    name: str
    needs: tuple[str, ...] = ()
    family: str = "extension"
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    level: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "needs": list(self.needs),
            "family": self.family,
            "consumes": list(self.consumes),
            "produces": list(self.produces),
            "tags": list(self.tags),
            "level": self.level,
        }


@dataclass(frozen=True)
class PipelinePlan:
    """A deterministic, topologically sorted stage plan.

    If ``dependencies`` is omitted, the supplied stage list retains the legacy
    linear semantics: each stage depends on the stage immediately before it.
    Supplying a dependency mapping switches to explicit DAG semantics; stages
    omitted from that mapping are roots.  Ties are resolved by the original
    stage order so plans remain reproducible.
    """

    nodes: tuple[PipelineNode, ...]
    dependency_mode: str = "linear"
    _original_order: tuple[str, ...] = field(default=(), repr=False)

    @classmethod
    def compile(
        cls,
        stages: Sequence[str],
        dependencies: Mapping[str, Sequence[str]] | None = None,
        capabilities: Mapping[str, Any] | None = None,
    ) -> PipelinePlan:
        if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
            raise PipelinePlanError("stages must be a sequence of unique stage names")
        original = tuple(str(stage).strip() for stage in stages)
        if any(not stage for stage in original):
            raise PipelinePlanError("stage names must not be empty")
        duplicates = sorted(name for name in set(original) if original.count(name) > 1)
        if duplicates:
            raise PipelinePlanError(
                "pipeline stage names must be unique: " + ", ".join(duplicates)
            )

        dependency_mode = "linear" if dependencies is None else "dag"
        if dependencies is None:
            normalized: dict[str, tuple[str, ...]] = {
                name: ((original[index - 1],) if index else ())
                for index, name in enumerate(original)
            }
        else:
            if not isinstance(dependencies, Mapping):
                raise PipelinePlanError("pipeline.dependencies must be a mapping")
            unknown_nodes = sorted({str(key) for key in dependencies} - set(original))
            if unknown_nodes:
                raise PipelinePlanError(
                    "pipeline.dependencies contains unknown stages: "
                    + ", ".join(unknown_nodes)
                )
            normalized = {name: () for name in original}
            for raw_name, raw_needs in dependencies.items():
                name = str(raw_name)
                if isinstance(raw_needs, str) or not isinstance(raw_needs, Sequence):
                    raise PipelinePlanError(
                        f"pipeline dependency list for {name!r} must be a sequence"
                    )
                needs = tuple(str(item).strip() for item in raw_needs)
                if any(not item for item in needs):
                    raise PipelinePlanError(
                        f"pipeline dependencies for {name!r} contain an empty name"
                    )
                duplicate_needs = sorted(
                    item for item in set(needs) if needs.count(item) > 1
                )
                if duplicate_needs:
                    raise PipelinePlanError(
                        f"pipeline dependencies for {name!r} contain duplicates: "
                        + ", ".join(duplicate_needs)
                    )
                unknown_needs = sorted(set(needs) - set(original))
                if unknown_needs:
                    raise PipelinePlanError(
                        f"pipeline stage {name!r} depends on unknown stages: "
                        + ", ".join(unknown_needs)
                    )
                if name in needs:
                    raise PipelinePlanError(
                        f"pipeline stage {name!r} cannot depend on itself"
                    )
                normalized[name] = needs

        order_index = {name: index for index, name in enumerate(original)}
        remaining = {name: set(needs) for name, needs in normalized.items()}
        ordered: list[str] = []
        levels: dict[str, int] = {}
        while remaining:
            ready = sorted(
                (name for name, needs in remaining.items() if not needs),
                key=order_index.__getitem__,
            )
            if not ready:
                cycle_nodes = sorted(remaining, key=order_index.__getitem__)
                raise PipelinePlanError(
                    "pipeline dependency cycle detected among: "
                    + ", ".join(cycle_nodes)
                )
            for name in ready:
                ordered.append(name)
                needs = normalized[name]
                levels[name] = (
                    max((levels[dependency] for dependency in needs), default=-1) + 1
                )
                remaining.pop(name)
            ready_set = set(ready)
            for needs in remaining.values():
                needs.difference_update(ready_set)

        capability_map = dict(capabilities or {})
        nodes = []
        for name in ordered:
            capability = capability_map.get(name)
            tags = tuple(sorted(str(item) for item in getattr(capability, "tags", ())))
            family = _stage_family(name, tags)
            nodes.append(
                PipelineNode(
                    name=name,
                    needs=tuple(normalized[name]),
                    family=family,
                    consumes=tuple(
                        sorted(
                            str(item) for item in getattr(capability, "consumes", ())
                        )
                    ),
                    produces=tuple(
                        sorted(
                            str(item) for item in getattr(capability, "produces", ())
                        )
                    ),
                    tags=tags,
                    level=levels[name],
                )
            )
        return cls(
            nodes=tuple(nodes),
            dependency_mode=dependency_mode,
            _original_order=original,
        )

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    @property
    def levels(self) -> tuple[tuple[str, ...], ...]:
        if not self.nodes:
            return ()
        maximum = max(node.level for node in self.nodes)
        return tuple(
            tuple(node.name for node in self.nodes if node.level == level)
            for level in range(maximum + 1)
        )

    def descendants(self, stages: Iterable[str]) -> tuple[str, ...]:
        requested = {str(stage) for stage in stages}
        unknown = sorted(requested - set(self.stages))
        if unknown:
            raise PipelinePlanError(
                "cannot select descendants of unknown stages: " + ", ".join(unknown)
            )
        selected = set(requested)
        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                if node.name not in selected and selected.intersection(node.needs):
                    selected.add(node.name)
                    changed = True
        return tuple(name for name in self.stages if name in selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "PipelinePlan",
            "dependency_mode": self.dependency_mode,
            "original_stages": list(self._original_order),
            "execution_order": list(self.stages),
            "levels": [list(level) for level in self.levels],
            "nodes": [node.to_dict() for node in self.nodes],
            "signature": self.signature(),
        }

    def signature(self) -> str:
        payload = {
            "dependency_mode": self.dependency_mode,
            "original_stages": self._original_order,
            "nodes": [
                {
                    "name": node.name,
                    "needs": node.needs,
                    "family": node.family,
                    "consumes": node.consumes,
                    "produces": node.produces,
                }
                for node in self.nodes
            ],
        }
        return _fingerprint(payload)


class PipelineStateStore:
    """Atomically persist stage attempts, results, and artifact lineage."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        run_fingerprint: str,
        plan: PipelinePlan,
    ) -> None:
        self.path = Path(path)
        self.run_fingerprint = str(run_fingerprint)
        self.plan = plan
        self.state: dict[str, Any] = {}

    def prepare(
        self,
        *,
        resume: bool = False,
        rerun: Sequence[str] = (),
        write: bool = True,
    ) -> tuple[str, ...]:
        restored: tuple[str, ...] = ()
        if resume and self.path.is_file():
            self.state = self._load()
            self._validate_loaded_state()
            invalidated = set(self.plan.descendants(rerun)) if rerun else set()
            restored = tuple(
                stage
                for stage in self.plan.stages
                if self.state.get("stages", {}).get(stage, {}).get("status")
                in {"completed", "skipped"}
                and stage not in invalidated
            )
            for stage in invalidated:
                self.state.setdefault("stages", {}).pop(stage, None)
            if invalidated:
                self.state["artifacts"] = {
                    name: value
                    for name, value in self.state.get("artifacts", {}).items()
                    if not isinstance(value, Mapping)
                    or value.get("producer") not in invalidated
                }
            self.state["status"] = "running"
            self.state["resumed_at"] = _utc_now()
            self.state["restored_stages"] = list(restored)
            self.state["invalidated_stages"] = [
                stage for stage in self.plan.stages if stage in invalidated
            ]
        else:
            self.state = {
                "apiVersion": PIPELINE_STATE_API_VERSION,
                "kind": "PipelineState",
                "status": "running",
                "run_fingerprint": self.run_fingerprint,
                "plan_signature": self.plan.signature(),
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "plan": self.plan.to_dict(),
                "stages": {},
                "artifacts": {},
                "restored_stages": [],
                "invalidated_stages": [],
            }
        if write:
            self._write()
        return restored

    def should_skip(self, stage: str) -> bool:
        return stage in set(self.state.get("restored_stages", ()))

    def restored_results(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for stage in self.state.get("restored_stages", ()):
            record = self.state.get("stages", {}).get(stage, {})
            result = record.get("result")
            if isinstance(result, Mapping):
                results[stage] = deepcopy(dict(result))
        return results

    def restored_artifacts(self) -> dict[str, Any]:
        return deepcopy(dict(self.state.get("artifacts", {}) or {}))

    def begin(self, stage: str, *, write: bool = True) -> None:
        previous = self.state.setdefault("stages", {}).get(stage, {})
        self.state["active_stage"] = stage
        self.state["status"] = "running"
        self.state["stages"][stage] = {
            "status": "running",
            "attempt": int(previous.get("attempt", 0)) + 1,
            "started_at": _utc_now(),
        }
        if write:
            self._write()

    def finish(
        self,
        stage: str,
        result: Mapping[str, Any] | None,
        *,
        artifacts: Mapping[str, Any] | None = None,
        write: bool = True,
    ) -> None:
        previous = self.state.setdefault("stages", {}).get(stage, {})
        result_payload = _json_safe(dict(result or {}))
        result_status = str(result_payload.get("status", "completed")).lower()
        state_status = "skipped" if result_status == "skipped" else "completed"
        self.state["stages"][stage] = {
            **previous,
            "status": state_status,
            "result_status": result_status,
            "finished_at": _utc_now(),
            "result": result_payload,
        }
        self.state["active_stage"] = None
        if artifacts is not None:
            self.state["artifacts"] = _json_safe(dict(artifacts))
        if write:
            self._write()

    def fail(
        self, stage: str | None, error: BaseException, *, write: bool = True
    ) -> None:
        if stage:
            previous = self.state.setdefault("stages", {}).get(stage, {})
            self.state["stages"][stage] = {
                **previous,
                "status": "failed",
                "finished_at": _utc_now(),
                "error": str(error),
                "error_type": type(error).__name__,
            }
        self.state["status"] = "failed"
        self.state["active_stage"] = stage
        self.state["error"] = str(error)
        if write:
            self._write()

    def complete(
        self,
        *,
        artifacts: Mapping[str, Any] | None = None,
        write: bool = True,
    ) -> None:
        self.state["status"] = "completed"
        self.state["active_stage"] = None
        self.state["completed_at"] = _utc_now()
        if artifacts is not None:
            self.state["artifacts"] = _json_safe(dict(artifacts))
        if write:
            self._write()

    def _load(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineStateError(
                f"Cannot read pipeline state {self.path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise PipelineStateError(
                f"Pipeline state must contain a JSON object: {self.path}"
            )
        return payload

    def _validate_loaded_state(self) -> None:
        if self.state.get("apiVersion") != PIPELINE_STATE_API_VERSION:
            raise PipelineStateError(
                "Pipeline state version mismatch; start a new run or choose a "
                "compatible state file"
            )
        if self.state.get("run_fingerprint") != self.run_fingerprint:
            raise PipelineStateError(
                "Pipeline state belongs to a different normalized configuration; "
                "disable pipeline.resume, use another output directory, or restore "
                "the original configuration"
            )
        if self.state.get("plan_signature") != self.plan.signature():
            raise PipelineStateError(
                "Pipeline dependency plan changed since the saved run; disable "
                "pipeline.resume or restore the original stage graph"
            )

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = _utc_now()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.{os.getpid()}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    self.state,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(8):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    # Windows indexers and virus scanners may briefly hold the
                    # destination after a rapid preceding state update.
                    time.sleep(0.01 * (2**attempt))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def normalized_run_fingerprint(config: Any) -> str:
    """Fingerprint semantic run settings while ignoring resume controls."""

    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict

        payload = asdict(config)
    elif isinstance(config, Mapping):
        payload = deepcopy(dict(config))
    else:
        raise TypeError("config must be a dataclass or mapping")
    pipeline = payload.get("pipeline")
    if isinstance(pipeline, Mapping):
        pipeline = dict(pipeline)
        for key in ("resume", "rerun", "state_path"):
            pipeline.pop(key, None)
        payload["pipeline"] = pipeline
    return _fingerprint(payload)


def _stage_family(name: str, tags: Sequence[str]) -> str:
    known = {
        "tokenizer": "data",
        "pretrain": "foundation",
        "sft": "post_training",
        "preference": "post_training",
        "rlhf": "post_training",
        "mopd": "post_training",
        "vla_sft": "embodied_policy",
        "mllm_sft": "multimodal",
        "vision_alignment": "multimodal",
        "media_cache": "data",
        "image_generation": "generative",
        "music_generation": "generative",
        "video_generation": "generative",
        "world_model": "world_model",
        "eval": "evaluation",
        "export": "release",
        "operator": "extension",
    }
    for tag in tags:
        if tag.startswith("family:") and len(tag) > len("family:"):
            return tag.split(":", 1)[1]
    return known.get(name, "extension")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


__all__ = [
    "PIPELINE_STATE_API_VERSION",
    "PipelineNode",
    "PipelinePlan",
    "PipelinePlanError",
    "PipelineStateError",
    "PipelineStateStore",
    "normalized_run_fingerprint",
]
