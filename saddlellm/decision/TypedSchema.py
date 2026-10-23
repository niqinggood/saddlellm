"""Version two decision requests and split-safe, variable-task datasets."""

# ruff: noqa: N999
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from .dataset import normalized, sha256

TYPES = ("choice", "score", "noul")
SPLITS = ("train", "validation", "calibration", "test")


@dataclass(frozen=True)
class TypedDecisionConfig:
    backbone: str = "native"  # native byte Transformer, or local HF AutoModel
    backbone_name_or_path: str | None = None
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    max_length: int = 256
    max_candidates: int = 100
    freeze_backbone: bool = False
    query_pooling: str = "mean"
    instance_question_in_query: bool = False

    def __post_init__(self):
        if self.backbone not in {"native", "hf"}:
            raise ValueError("typed backbone must be native or hf")
        if self.backbone == "hf" and not self.backbone_name_or_path:
            raise ValueError("hf requires a local backbone_name_or_path")
        for name in (
            "hidden_size",
            "num_layers",
            "num_heads",
            "max_length",
            "max_candidates",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads or self.max_candidates < 2:
            raise ValueError(
                "hidden_size must divide by num_heads; max_candidates must be >=2"
            )
        if type(self.freeze_backbone) is not bool:
            raise TypeError("freeze_backbone must be boolean")
        if type(self.instance_question_in_query) is not bool:
            raise TypeError("instance_question_in_query must be boolean")
        if not isinstance(self.query_pooling, str) or self.query_pooling not in {
            "mean",
            "last_token",
        }:
            raise ValueError("query_pooling must be mean or last_token")

    def to_dict(self):
        return asdict(self)


def _text(value, name, limit=16384):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(
            f"{name} must be a nonblank string of at most {limit} characters"
        )
    return value


def validate_question(value, *, max_candidates=100):
    if not isinstance(value, dict):
        raise TypeError("question must be an object")
    kind = value.get("type", "").lower() if isinstance(value.get("type"), str) else ""
    if kind not in TYPES:
        raise ValueError("type must be Choice, Score or Noul")
    permitted = {
        "id",
        "type",
        "question",
        "instance_question",
        "choices" if kind == "choice" else "levels" if kind == "score" else "question",
    }
    if set(value) - permitted:
        raise ValueError("unsupported question fields")
    result = {"type": kind, "question": _text(value.get("question"), "question", 4096)}
    if "instance_question" in value:
        result["instance_question"] = _text(
            value["instance_question"], "instance_question", 4096
        )
    if "id" in value:
        result["id"] = _text(value["id"], "question id", 128)
    if kind == "noul":
        return result
    key = "choices" if kind == "choice" else "levels"
    candidates = value.get(key)
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= max_candidates:
        raise ValueError(f"{key} requires 2..{max_candidates} candidates")
    ids, scores, checked = set(), [], []
    for item in candidates:
        required = {"id", "text"} | ({"value"} if kind == "score" else set())
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"each {key} item requires exactly {sorted(required)}")
        identifier = _text(item["id"], "candidate id", 128)
        if identifier in ids:
            raise ValueError("candidate IDs must be unique")
        ids.add(identifier)
        candidate = {
            "id": identifier,
            "text": _text(item["text"], "candidate text", 4096),
        }
        if kind == "score":
            number = item["value"]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ValueError("Score level values must be finite numbers")
            scores.append(number)
            candidate["value"] = number
        checked.append(candidate)
    if scores and any(a >= b for a, b in pairwise(scores)):
        raise ValueError("Score level values must be strictly increasing")
    result[key] = checked
    return result


def candidates(question):
    if question["type"] == "noul":
        return [{"id": "false", "text": "False"}, {"id": "true", "text": "True"}]
    return question["choices" if question["type"] == "choice" else "levels"]


def candidate_query_text(row, option, *, instance_question_in_query=False):
    """Render one candidate query; variable instance input does not define scope."""
    instance = ""
    if instance_question_in_query and "instance_question" in row:
        text = _text(row["instance_question"], "instance_question", 4096)
        instance = f"\nInstance question: {text}"
    level = f"\nValue: {option['value']}" if row["type"] == "score" else ""
    return (
        f"Type: {row['type']}\nQuestion: {row['question']}{instance}"
        f"\nOption: {option['text']}{level}"
    )


def scope_key(question):
    """Fixed task semantics; presentation order and instance input are irrelevant."""
    items = sorted(candidates(question), key=lambda c: c["id"])
    payload = {
        "type": question["type"],
        "question": question["question"],
        "candidates": items,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_record(row, *, require_target=True, max_candidates=100):
    if not isinstance(row, dict):
        raise TypeError("decision row must be an object")
    allowed = {
        "id",
        "state",
        "state_id",
        "task_id",
        "type",
        "question",
        "instance_question",
        "choices",
        "levels",
        "label",
        "target_probs",
    }
    if set(row) - allowed:
        raise ValueError("unsupported decision row fields")
    question = validate_question(
        {
            k: row[k]
            for k in ("type", "question", "instance_question", "choices", "levels")
            if k in row
        },
        max_candidates=max_candidates,
    )
    result = {**question, "state": _text(row.get("state"), "state")}
    for key in ("id", "state_id", "task_id"):
        if require_target or key in row:
            result[key] = _text(row.get(key), key, 256)
    if not require_target:
        return result
    if ("label" in row) == ("target_probs" in row):
        raise ValueError("provide exactly one label or target_probs")
    ids = [c["id"] for c in candidates(result)]
    if "label" in row:
        label = row["label"]
        if result["type"] == "noul" and type(label) is bool:
            label = "true" if label else "false"
        if label not in ids:
            raise ValueError("label must be a candidate ID (or boolean for Noul)")
        probabilities = [float(identifier == label) for identifier in ids]
    else:
        mapping = row["target_probs"]
        if not isinstance(mapping, dict) or set(mapping) != set(ids):
            raise ValueError("target_probs must map every candidate ID exactly once")
        probabilities = [mapping[identifier] for identifier in ids]
        if any(
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not math.isfinite(p)
            or not 0 <= p <= 1
            for p in probabilities
        ):
            raise ValueError("target probabilities must be finite numbers in [0,1]")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("target probabilities must sum to one")
    result["target"] = probabilities
    return result


def read_typed_dataset(directory):
    root = Path(directory).expanduser().resolve()
    descriptor = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if descriptor.get("schema_version") != 2 or descriptor.get("task_type") != "typed":
        raise ValueError("expected schema_version=2 and task_type=typed")
    files = descriptor.get("splits")
    if not isinstance(files, dict) or set(files) != set(SPLITS):
        raise ValueError(
            "four separate train/validation/calibration/test splits are required"
        )
    hashes, splits = {"dataset.json": sha256(root / "dataset.json")}, {}
    seen_ids, states, groups, task_scopes = set(), {}, {}, {}
    for split in SPLITS:
        if not isinstance(files[split], str):
            raise TypeError("split paths must be strings")
        path = (root / files[split]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("split files must remain inside the dataset directory")
        hashes[split] = sha256(path)
        rows = [
            validate_record(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"empty {split} split")
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError("row IDs must be globally unique")
            seen_ids.add(row["id"])
            state, group = normalized(row["state"]), row["state_id"]
            if (
                state in states
                and states[state] != split
                or group in groups
                and groups[group] != split
            ):
                raise ValueError("state or state_id leakage between splits")
            states[state], groups[group] = split, split
            scope = scope_key(row)
            if row["task_id"] in task_scopes and task_scopes[row["task_id"]] != scope:
                raise ValueError(
                    "one task_id must describe one exact question and candidate set"
                )
            task_scopes[row["task_id"]] = scope
        splits[split] = rows
    manifest = root / "manifest.json"
    if manifest.exists():
        for relative, expected in json.loads(manifest.read_text(encoding="utf-8"))[
            "output_sha256"
        ].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or sha256(path) != expected:
                raise ValueError("dataset manifest hash mismatch")
        hashes["manifest.json"] = sha256(manifest)
    return descriptor, splits, hashes
