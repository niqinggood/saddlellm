"""Lightweight configuration for Saddle's experimental Choice decision model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping, Optional


def json_object(value: Optional[Mapping[str, Any]], *, name: str) -> dict:
    """Copy metadata without accepting non-JSON objects or non-finite numbers."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON values") from exc


@dataclass(frozen=True)
class DecisionModelConfig:
    """A text-conditioned scorer, with no fixed label IDs or classification width.

    The backbone is an external, frozen artifact. Its reference and identity must
    travel with the head; a head-only save is not a self-contained backbone save.
    """

    backbone_name_or_path: str
    embedding_dim: int = 1024
    projection_rank: int = 16
    initial_scale: float = 20.0
    max_length: int = 512
    max_candidates: int = 256
    encoder_batch_size: int = 32
    max_text_chars: int = 65536
    model_revision: Optional[str] = None
    backbone_identity: dict = field(default_factory=dict)
    template_version: str = "choice-v1"
    supported_types: tuple[str, ...] = ("choice",)

    def __post_init__(self) -> None:
        if not isinstance(self.backbone_name_or_path, str) or not self.backbone_name_or_path.strip():
            raise ValueError("backbone_name_or_path must be a non-empty string")
        for name in ("embedding_dim", "projection_rank", "max_length", "encoder_batch_size", "max_text_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.projection_rank > self.embedding_dim:
            raise ValueError("projection_rank cannot exceed embedding_dim")
        if isinstance(self.max_candidates, bool) or not isinstance(self.max_candidates, int) or self.max_candidates < 2:
            raise ValueError("max_candidates must be an integer of at least two")
        if isinstance(self.initial_scale, bool) or not isinstance(self.initial_scale, (int, float)):
            raise ValueError("initial_scale must be a finite positive number")
        if not math.isfinite(self.initial_scale) or not 1e-3 <= self.initial_scale <= 1e3:
            raise ValueError("initial_scale must be between 0.001 and 1000")
        if self.model_revision is not None and (not isinstance(self.model_revision, str) or not self.model_revision.strip()):
            raise ValueError("model_revision must be a non-empty string or None")
        if self.template_version != "choice-v1":
            raise ValueError("unsupported decision template_version")
        if tuple(self.supported_types) != ("choice",):
            raise ValueError("only Choice is supported; Score and Noul are not trained or implemented")
        object.__setattr__(self, "supported_types", ("choice",))
        object.__setattr__(self, "backbone_identity", json_object(self.backbone_identity, name="backbone_identity"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionModelConfig":
        if not isinstance(data, Mapping):
            raise ValueError("decision config must be an object")
        try:
            return cls(**dict(data))
        except TypeError as exc:
            raise ValueError(f"invalid decision config: {exc}") from exc


__all__ = ["DecisionModelConfig"]
