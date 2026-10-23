"""Content identity for local model artifacts; no model imports or downloads.

The fingerprint binds inference-relevant bytes, not a model's quality or a
user's permissions. Paths are provenance only and do not enter the digest.
Remote identifiers remain unresolved until their actual artifacts are local.
"""

from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
import fnmatch
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_WEIGHTS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf")
_CONFIGS = {
    "config.json",
    "saddle_config.json",
    "generation_config.json",
    "adapter_config.json",
    "decision_config.json",
    "calibration.json",
    "quantization_config.json",
    "quantization_manifest.json",
    "model_blueprint.json",
}
_TOKENIZERS = (
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template*",
    "sentencepiece*",
)
_EXCLUDED = {
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "training_args.bin",
    "trainer_state.json",
    "training_summary.json",
    "summary.json",
    "release_manifest.json",
    "manifest.json",
    "saddle_checkpoint.json",
}
_SKIP_DIRS = {".git", "__pycache__", "runs", "logs", ".cache"}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _digest_file(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError(f"Model artifact changed while fingerprinting: {path}")
    return digest.hexdigest(), after.st_size


def _file_kind(name: str) -> str | None:
    if name in _EXCLUDED or fnmatch.fnmatch(name, "rng_state*.pth"):
        return None
    if any(fnmatch.fnmatch(name, pattern) for pattern in _WEIGHTS):
        return "weights"
    if name in _CONFIGS or fnmatch.fnmatch(name, "*.index.json"):
        return "config"
    if any(fnmatch.fnmatch(name, pattern) for pattern in _TOKENIZERS):
        return "tokenizer"
    if name.endswith(".py"):
        return "model_code"
    return None


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _component(reference: str | Path) -> dict[str, Any]:
    raw = str(reference)
    source = Path(raw).expanduser()
    if not source.exists():
        return {
            "reference": raw,
            "local": False,
            "status": "unresolved",
            "fingerprint": None,
            "files": [],
        }
    source = source.resolve()
    if not source.is_dir():
        return {
            "reference": str(source),
            "local": True,
            "status": "unresolved",
            "fingerprint": None,
            "files": [],
        }
    paths: list[Path] = []
    visited: set[Path] = set()
    for directory, subdirs, names in os.walk(source, followlinks=True):
        folder = Path(directory)
        canonical = folder.resolve()
        if canonical in visited:
            subdirs[:] = []
            continue
        visited.add(canonical)
        subdirs[:] = sorted(
            name
            for name in subdirs
            if name not in _SKIP_DIRS and not name.startswith("checkpoint-")
        )
        paths.extend(folder / name for name in names if _file_kind(name) is not None)
    files = []
    for path in sorted(paths, key=lambda item: item.relative_to(source).as_posix()):
        digest, size = _digest_file(path)
        files.append(
            {
                "path": path.relative_to(source).as_posix(),
                "kind": _file_kind(path.name),
                "bytes": size,
                "sha256": digest,
            }
        )
    verified = any(item["kind"] == "weights" for item in files)
    return {
        "reference": str(source),
        "local": True,
        "status": "verified" if verified else "unresolved",
        "fingerprint": _canonical_digest(files) if verified else None,
        "files": files,
    }


def compute_model_identity(
    model_path: str | Path,
    adapter_path: str | Path | None = None,
) -> dict[str, Any]:
    """Hash local weights, inference configuration, tokenizer and optional PEFT.

    Passing a local adapter directory alone resolves the base declared in its
    adapter_config.json, using the loader's ordinary current-directory path
    semantics. Passing base + adapter explicitly produces the same identity.
    No registry identifier is downloaded or guessed from a name/revision.
    """
    warnings: list[str] = []
    model_reference: str | Path = model_path
    supplied = Path(model_path).expanduser()
    if (
        adapter_path is None
        and supplied.is_dir()
        and (supplied / "adapter_config.json").is_file()
    ):
        adapter_path = supplied
        try:
            config = json.loads(
                (supplied / "adapter_config.json").read_text(encoding="utf-8")
            )
            model_reference = config.get("base_model_name_or_path") or ""
        except (OSError, ValueError, AttributeError):
            model_reference = ""
            warnings.append(
                "Cannot resolve the adapter's declared base model configuration."
            )
    model = (
        _component(model_reference)
        if str(model_reference)
        else {
            "reference": "",
            "local": False,
            "status": "unresolved",
            "fingerprint": None,
            "files": [],
        }
    )
    adapter = _component(adapter_path) if adapter_path is not None else None
    verified = model["status"] == "verified" and (
        adapter is None or adapter["status"] == "verified"
    )
    if not verified:
        warnings.append(
            "Model content identity is unresolved: local inference weights are required; no remote files were downloaded. Materialize the base/adapter locally and re-run evaluate/compare for a content-bound gate."
        )
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "status": "verified" if verified else "unresolved",
        "fingerprint": _canonical_digest(
            {
                "schema_version": 1,
                "algorithm": "sha256",
                "model": model["fingerprint"],
                "adapter": adapter["fingerprint"] if adapter is not None else None,
            }
        )
        if verified
        else None,
        "model": model,
        "adapter": adapter,
        "warnings": warnings,
        "scope": "local inference-relevant file contents; not quality, authenticity, or permissions",
    }


def identity_is_bound(identity: Any) -> bool:
    return bool(
        isinstance(identity, Mapping)
        and identity.get("schema_version") == 1
        and identity.get("algorithm") == "sha256"
        and identity.get("status") == "verified"
        and isinstance(identity.get("fingerprint"), str)
        and _DIGEST.fullmatch(identity["fingerprint"])
    )


def model_identities_match(left: Any, right: Any) -> bool:
    return (
        identity_is_bound(left)
        and identity_is_bound(right)
        and left["fingerprint"] == right["fingerprint"]
    )


def gate_model_identity(gate: Any) -> Mapping[str, Any] | None:
    """Read evaluation, paired-comparison and nested legacy artifact schemas."""
    visited: set[int] = set()

    def find(value: Any) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping) or id(value) in visited:
            return None
        visited.add(id(value))
        for field in ("candidate_model_identity", "model_identity"):
            identity = value.get(field)
            if isinstance(identity, Mapping):
                return identity
        for field in ("gate", "report", "evaluation"):
            identity = find(value.get(field))
            if identity is not None:
                return identity
        return None

    return find(gate)


def verify_gate_model(
    gate: Mapping[str, Any] | None,
    model_identity: Mapping[str, Any] | None,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Bind a reported gate to the current source, without adding access rules.

    Optional export stays available for legacy/unresolved/mismatched evidence;
    its returned accepted=False prevents attaching another model's pass claim.
    Strict rejection is enabled only by the caller's explicit required option.
    """
    from .ReleaseGate import release_gate_accepted

    reported_accepted = release_gate_accepted(gate)
    identity = gate_model_identity(gate)
    warnings = []
    if gate is None:
        status = "no_gate"
    elif not identity_is_bound(identity) or not identity_is_bound(model_identity):
        status = "unbound"
        warnings.append(
            "Gate evidence is not bound to local model content. Legacy paths and remote identifiers cannot prove a match; re-run evaluate/compare on the local artifact."
        )
    elif model_identities_match(identity, model_identity):
        status = "matched"
    else:
        status = "mismatch"
        warnings.append(
            "Gate candidate model content does not match the export source. The gate's accepted claim is not attached to this model."
        )
    accepted = reported_accepted and status == "matched"
    if required and not accepted:
        if status in {"unbound", "no_gate"}:
            raise ValueError(
                "require_gate requires an accepted gate bound to local model content; this evidence is unbound. Re-run evaluate/compare on the local model (and its base/adapter)."
            )
        if status == "mismatch":
            raise ValueError(
                "require_gate candidate model identity does not match the export source; re-run evaluate/compare for the current model contents."
            )
        raise ValueError(
            "require_gate rejected: the matching model's release gate was not accepted."
        )
    return {
        "status": status,
        "accepted": accepted,
        "reported_accepted": reported_accepted,
        "warnings": warnings,
    }


__all__ = [
    "compute_model_identity",
    "gate_model_identity",
    "identity_is_bound",
    "model_identities_match",
    "verify_gate_model",
]
