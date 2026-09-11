"""Typed artifact lineage shared by all training stage families."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")


class ArtifactError(ValueError):
    """Raised when a stage publishes an invalid or conflicting artifact."""


@dataclass(frozen=True)
class ArtifactRef:
    """Serializable reference to a stage output.

    ``uri`` intentionally accepts local paths as well as remote/object-store
    identifiers.  Existence is not required because dry-run stages publish
    planned outputs before materialization.
    """

    name: str
    kind: str
    uri: str
    producer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        kind = str(self.kind).strip().lower()
        uri = str(self.uri).strip()
        producer = str(self.producer).strip()
        if not name or not _ARTIFACT_NAME.fullmatch(name):
            raise ArtifactError(f"invalid artifact name: {self.name!r}")
        if not kind or not _ARTIFACT_NAME.fullmatch(kind):
            raise ArtifactError(f"invalid artifact kind: {self.kind!r}")
        if not uri:
            raise ArtifactError("artifact uri must not be empty")
        if not producer:
            raise ArtifactError("artifact producer must not be empty")
        if not isinstance(self.metadata, Mapping):
            raise ArtifactError("artifact metadata must be a mapping")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @property
    def path(self) -> Path:
        return Path(self.uri)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "uri": self.uri,
            "producer": self.producer,
            "metadata": deepcopy(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactRef:
        if not isinstance(data, Mapping):
            raise ArtifactError("artifact reference must be a mapping")
        return cls(
            name=data.get("name", ""),
            kind=data.get("kind", "artifact"),
            uri=data.get("uri", data.get("path", "")),
            producer=data.get("producer", "unknown"),
            metadata=data.get("metadata", {}),
        )


class ArtifactRegistry(Mapping[str, ArtifactRef]):
    """Ordered registry of outputs produced during one pipeline run."""

    def __init__(self) -> None:
        self._items: MutableMapping[str, ArtifactRef] = {}

    def __getitem__(self, name: str) -> ArtifactRef:
        return self._items[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def publish(
        self,
        name: str,
        *,
        kind: str,
        uri: Any,
        producer: str,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> ArtifactRef:
        reference = ArtifactRef(
            name=name,
            kind=kind,
            uri=str(uri),
            producer=producer,
            metadata=metadata or {},
        )
        if reference.name in self._items and not replace:
            current = self._items[reference.name]
            if current == reference:
                return current
            raise ArtifactError(
                f"artifact {reference.name!r} was already published by "
                f"{current.producer!r}"
            )
        self._items[reference.name] = reference
        return reference

    def latest(self, kind: str) -> ArtifactRef | None:
        normalized = str(kind).strip().lower()
        for reference in reversed(tuple(self._items.values())):
            if reference.kind == normalized:
                return reference
        return None

    def by_kind(self, kind: str) -> tuple[ArtifactRef, ...]:
        normalized = str(kind).strip().lower()
        return tuple(item for item in self._items.values() if item.kind == normalized)

    def capture(
        self, stage: str, result: Mapping[str, Any] | None
    ) -> tuple[ArtifactRef, ...]:
        """Capture conventional paths plus a plugin's explicit artifacts block."""

        if not isinstance(result, Mapping):
            return ()
        before = set(self._items)
        explicit = result.get("artifacts")
        if isinstance(explicit, Mapping):
            for raw_name, raw_value in explicit.items():
                self._capture_explicit(stage, str(raw_name), raw_value)

        for field_name in (
            "model_path",
            "tokenizer_path",
            "checkpoint_path",
            "data_path",
            "dataset_path",
            "normalized_data_path",
            "output_path",
            "output_dir",
            "plan_path",
            "action_space_path",
        ):
            value = result.get(field_name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                continue
            name = f"{stage}.{field_name}"
            self.publish(
                name,
                kind=_infer_kind(stage, field_name),
                uri=value,
                producer=stage,
                metadata={"field": field_name, "implicit": True},
                replace=True,
            )
        return tuple(
            reference for name, reference in self._items.items() if name not in before
        )

    def load_dict(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise ArtifactError("persisted artifacts must be a mapping")
        for name, raw in payload.items():
            if not isinstance(raw, Mapping):
                raise ArtifactError(f"persisted artifact {name!r} must be a mapping")
            values = dict(raw)
            values.setdefault("name", name)
            reference = ArtifactRef.from_dict(values)
            self._items[reference.name] = reference

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: reference.to_dict() for name, reference in self._items.items()}

    def _capture_explicit(self, stage: str, raw_name: str, raw_value: Any) -> None:
        name = raw_name if raw_name.startswith(f"{stage}.") else f"{stage}.{raw_name}"
        if isinstance(raw_value, (str, Path)):
            self.publish(
                name,
                kind=_infer_kind(stage, raw_name),
                uri=raw_value,
                producer=stage,
                metadata={"implicit": False},
                replace=True,
            )
            return
        if not isinstance(raw_value, Mapping):
            return
        uri = raw_value.get("uri", raw_value.get("path"))
        if uri in (None, ""):
            return
        self.publish(
            name,
            kind=raw_value.get("kind", _infer_kind(stage, raw_name)),
            uri=uri,
            producer=raw_value.get("producer", stage),
            metadata=raw_value.get("metadata", {}),
            replace=True,
        )


def _infer_kind(stage: str, field_name: str) -> str:
    name = str(field_name).lower()
    if "tokenizer" in name:
        return "tokenizer"
    if "plan" in name or "action_space" in name:
        return "plan"
    if "data" in name or "dataset" in name:
        return "dataset"
    if "checkpoint" in name:
        return "checkpoint"
    if stage == "world_model" and name in {
        "model",
        "model_path",
        "output",
        "output_path",
        "output_dir",
    }:
        return "world_model"
    if stage == "vla_sft" and name in {
        "model",
        "model_path",
        "output",
        "output_path",
        "output_dir",
    }:
        return "control_policy"
    if stage == "export" and name in {"release", "output", "output_path", "output_dir"}:
        return "release"
    if name == "model_path" or name == "model":
        return "language_model"
    if name in {"output", "output_path", "output_dir"}:
        return "stage_output"
    return "artifact"


__all__ = [
    "ArtifactError",
    "ArtifactRef",
    "ArtifactRegistry",
]
