"""Decision dataset checks with no model or optional tensor dependencies."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())



def read_decision_dataset(directory: str | Path) -> tuple[dict, dict, dict]:
    directory = Path(directory).expanduser().resolve()
    descriptor = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    if descriptor.get("schema_version") != 1 or descriptor.get("task_type") != "choice":
        raise ValueError("expected schema_version=1 and task_type=choice")
    question, choices = descriptor.get("question"), descriptor.get("choices")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("dataset question must be nonempty")
    if not isinstance(choices, list) or not 2 <= len(choices) <= 100:
        raise ValueError("dataset needs 2..100 choices")
    for choice in choices:
        if not isinstance(choice, dict) or set(choice) != {"id", "text"} or any(not isinstance(v, str) or not v.strip() for v in choice.values()):
            raise ValueError("choices must contain nonempty id and text")
    labels = [choice["id"] for choice in choices]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate IDs must be unique")
    label_indices = {label: index for index, label in enumerate(labels)}
    split_paths = descriptor.get("splits", {})
    if set(split_paths) != {"train", "validation", "calibration", "test"}:
        raise ValueError("four separate train/validation/calibration/test splits are required")
    paths = {}
    for name, relative in split_paths.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("split files must remain inside the dataset directory")
        paths[name] = path
    hashes = {"dataset.json": sha256(directory / "dataset.json"), **{name: sha256(path) for name, path in paths.items()}}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in manifest.get("output_sha256", {}).items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory) or sha256(path) != expected:
                raise ValueError(f"dataset manifest hash mismatch: {relative}")
        hashes["manifest.json"] = sha256(manifest_path)
    splits, seen_text, seen_ids = {}, {}, set()
    for name, path in paths.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty {name} split")
        for row in rows:
            if not isinstance(row.get("state"), str) or not normalized(row["state"]) or row.get("label") not in label_indices:
                raise ValueError(f"invalid state/label in {name}")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen_ids:
                raise ValueError("row IDs must be nonempty and globally unique")
            seen_ids.add(identifier)
            key = normalized(row["state"])
            if key in seen_text and seen_text[key] != name:
                raise ValueError(f"normalized text leakage between {seen_text[key]} and {name}")
            seen_text[key] = name
        splits[name] = rows
    return descriptor, splits, hashes
