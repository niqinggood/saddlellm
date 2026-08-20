"""Resumable raw-media to generation-cache builder and shard reader."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .BuiltinMediaCodecs import (
    build_text_condition_encoder,
    register_builtin_media_codecs,
)
from .ModalityCodec import ModalityCodecRegistry
from .MultimodalData import MultimodalDataAdapter


_MODALITY_CONTRACTS = {
    "image": ("latents", "baseline-rgb-image"),
    "audio": ("codes", "baseline-wav-rvq"),
    "video": ("video_latents", "baseline-frame-video"),
}


@dataclass
class MediaCacheBuildConfig:
    input_path: str
    output_dir: str
    modality: str
    codec_name: Optional[str] = None
    codec_config: Dict[str, Any] = field(default_factory=dict)
    text_encoder: str = "hash"
    text_encoder_config: Dict[str, Any] = field(
        default_factory=lambda: {"dimension": 256}
    )
    prompt_field: Optional[str] = None
    media_field: Optional[str] = None
    shard_size: int = 256
    strict: bool = True
    resume: bool = True
    overwrite: bool = False
    plugin_modules: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.modality = str(self.modality).lower()
        if self.modality not in _MODALITY_CONTRACTS:
            raise ValueError("modality must be image, audio, or video")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite cannot both be true")
        if not self.codec_name:
            self.codec_name = _MODALITY_CONTRACTS[self.modality][1]


class ShardedNpzStore:
    """Random-access reader for one MediaCache manifest or output directory."""

    def __init__(self, path: str, required_arrays: Sequence[str]) -> None:
        source = Path(path)
        manifest_path = source / "manifest.json" if source.is_dir() else source
        if not manifest_path.is_file() or manifest_path.suffix.lower() != ".json":
            raise FileNotFoundError(f"Media cache manifest not found: {manifest_path}")
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "completed":
            raise ValueError("Media cache manifest is not completed")
        self.required_arrays = tuple(required_arrays)
        schema = self.manifest.get("arrays", {})
        missing = set(self.required_arrays) - set(schema)
        if missing:
            raise ValueError(
                "Media cache does not contain required arrays: "
                + ", ".join(sorted(missing))
            )
        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError("Media cache manifest contains no shards")
        self._ends: List[int] = []
        total = 0
        for shard in self.shards:
            samples = int(shard.get("samples", 0))
            if samples <= 0:
                raise ValueError("Each media cache shard must contain samples")
            total += samples
            self._ends.append(total)
        self._length = total
        if int(self.manifest.get("samples", total)) != total:
            raise ValueError("Media cache sample count does not match shard metadata")
        self._cached_index: Optional[int] = None
        self._cached_archive: Any = None

    def __len__(self) -> int:
        return self._length

    def array_shape(self, name: str) -> Tuple[int, ...]:
        try:
            return tuple(int(value) for value in self.manifest["arrays"][name]["shape"])
        except KeyError as exc:
            raise KeyError(f"Unknown cached array {name!r}") from exc

    def array_stat(self, name: str, statistic: str) -> Any:
        try:
            return self.manifest["arrays"][name][statistic]
        except KeyError as exc:
            raise KeyError(
                f"Cached array {name!r} has no {statistic!r} statistic"
            ) from exc

    def get(self, index: int) -> Dict[str, np.ndarray]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_index = bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        archive = self._open_shard(shard_index)
        return {
            name: np.array(archive[name][index - start], copy=True)
            for name in self.required_arrays
        }

    def _open_shard(self, index: int):
        if index == self._cached_index:
            return self._cached_archive
        if self._cached_archive is not None:
            self._cached_archive.close()
        path = self.root / self.shards[index]["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Media cache shard is missing: {path}")
        self._cached_archive = np.load(path, allow_pickle=False)
        self._cached_index = index
        missing = set(self.required_arrays) - set(self._cached_archive.files)
        if missing:
            raise ValueError(
                f"Shard {path.name} is missing arrays: " + ", ".join(sorted(missing))
            )
        return self._cached_archive

    def close(self) -> None:
        if getattr(self, "_cached_archive", None) is not None:
            self._cached_archive.close()
            self._cached_archive = None
            self._cached_index = None

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_cached_index"] = None
        state["_cached_archive"] = None
        return state

    def __del__(self) -> None:
        self.close()


def build_media_cache(
    config: MediaCacheBuildConfig | Dict[str, Any],
) -> Dict[str, Any]:
    """Encode a JSON/JSONL/CSV media manifest into resumable NPZ shards."""

    if isinstance(config, dict):
        config = MediaCacheBuildConfig(**config)
    source = Path(config.input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Media cache input manifest not found: {source}")
    output = Path(config.output_dir).expanduser().resolve()
    manifest_path = output / "manifest.json"
    if output.exists() and config.overwrite:
        _clear_owned_cache(output)
    if output.exists() and not manifest_path.is_file() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to write media cache into non-empty non-cache directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    register_builtin_media_codecs()
    for module_name in config.plugin_modules:
        importlib.import_module(str(module_name))
    codec = ModalityCodecRegistry.build(config.codec_name, **config.codec_config)
    if codec.spec.modality != config.modality:
        raise ValueError(
            f"Codec {codec.spec.name!r} handles {codec.spec.modality}, not {config.modality}"
        )
    text_encoder = build_text_condition_encoder(
        config.text_encoder, **config.text_encoder_config
    )
    input_fingerprint = _file_fingerprint(source)
    identity = {
        "schema_version": 1,
        "modality": config.modality,
        "input": input_fingerprint,
        "codec": codec.fingerprint(),
        "condition_encoder": text_encoder.fingerprint(),
        "array_name": _MODALITY_CONTRACTS[config.modality][0],
        "prompt_field": config.prompt_field,
        "media_field": config.media_field,
        "shard_size": config.shard_size,
    }
    manifest = _initial_manifest(identity, config)
    if manifest_path.exists():
        if not config.resume:
            raise FileExistsError(
                f"Media cache already exists: {output}; use resume or overwrite"
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_identity(previous, identity)
        if previous.get("status") == "completed":
            return previous
        manifest = previous

    records = MultimodalDataAdapter.load_records(str(source))
    processed = int(manifest.get("processed_records", 0))
    if processed > len(records):
        raise ValueError("Manifest progress exceeds current input record count")
    failures = list(manifest.get("failures", []))
    shard_index = len(manifest.get("shards", []))
    media_batch: List[np.ndarray] = []
    condition_batch: List[np.ndarray] = []
    accepted_record_indices: List[int] = []
    attention_batch: List[np.ndarray] = []
    array_name = identity["array_name"]

    for record_index in range(processed, len(records)):
        record = records[record_index]
        try:
            prompt = _extract_prompt(record, config.prompt_field)
            media_path = _extract_media_path(
                record,
                config.modality,
                config.media_field,
                source.parent,
            )
            if config.modality == "audio" and hasattr(
                codec, "encode_with_attention_mask"
            ):
                encoded, attention_mask = codec.encode_with_attention_mask(
                    str(media_path)
                )
                media = np.asarray(encoded)
                attention_mask = np.asarray(attention_mask, dtype=np.int64)
                if attention_mask.shape != (media.shape[-1],):
                    raise ValueError(
                        "Audio codec attention_mask must have shape [frames]"
                    )
            else:
                media = np.asarray(codec.encode(str(media_path)))
                attention_mask = None
            condition = text_encoder.encode([prompt])[0]
            _validate_encoded_sample(config.modality, media, condition)
            if media_batch and media.shape != media_batch[0].shape:
                raise ValueError(
                    f"Encoded media shape {media.shape} differs from {media_batch[0].shape}"
                )
            media_batch.append(media)
            condition_batch.append(np.asarray(condition, dtype=np.float32))
            accepted_record_indices.append(record_index)
            if attention_mask is not None:
                attention_batch.append(attention_mask)
        except Exception as exc:
            failure = {
                "record_index": record_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            if config.strict:
                manifest.update(
                    status="failed",
                    failures=failures,
                )
                _atomic_write_json(manifest_path, manifest)
                raise
        if len(media_batch) >= config.shard_size:
            _flush_shard(
                output,
                manifest,
                shard_index,
                array_name,
                media_batch,
                condition_batch,
                accepted_record_indices,
                attention_batch=attention_batch,
                processed_records=record_index + 1,
                failures=failures,
            )
            shard_index += 1
            media_batch, condition_batch, accepted_record_indices = [], [], []
            attention_batch = []

    if media_batch:
        _flush_shard(
            output,
            manifest,
            shard_index,
            array_name,
            media_batch,
            condition_batch,
            accepted_record_indices,
            attention_batch=attention_batch,
            processed_records=len(records),
            failures=failures,
        )
    if not manifest.get("shards"):
        manifest.update(
            status="failed",
            processed_records=len(records),
            total_records=len(records),
            failures=failures,
        )
        _atomic_write_json(manifest_path, manifest)
        raise ValueError("No records were encoded into the media cache")
    manifest.update(
        status="completed",
        processed_records=len(records),
        total_records=len(records),
        failures=failures,
        samples=sum(int(item["samples"]) for item in manifest["shards"]),
    )
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _initial_manifest(
    identity: Dict[str, Any], config: MediaCacheBuildConfig
) -> Dict[str, Any]:
    return {
        **identity,
        "status": "building",
        "processed_records": 0,
        "total_records": None,
        "samples": 0,
        "arrays": {},
        "shards": [],
        "failures": [],
        "build_config": asdict(config),
    }


def _validate_resume_identity(previous: Dict[str, Any], identity: Dict[str, Any]) -> None:
    for key, value in identity.items():
        if previous.get(key) != value:
            raise ValueError(
                f"Cannot resume media cache: {key} changed from "
                f"{previous.get(key)!r} to {value!r}"
            )


def _extract_prompt(record: Dict[str, Any], field: Optional[str]) -> str:
    if field:
        value = record.get(field)
    else:
        value = next(
            (
                record[key]
                for key in ("prompt", "text", "caption", "instruction", "query")
                if record.get(key) not in (None, "")
            ),
            None,
        )
        if value is None and isinstance(record.get("messages"), list):
            value = "\n".join(
                str(item.get("content", ""))
                for item in record["messages"]
                if item.get("role") in {"user", "system"}
            )
    if value in (None, ""):
        raise ValueError("Record has no text prompt")
    return str(value)


def _extract_media_path(
    record: Dict[str, Any],
    modality: str,
    field: Optional[str],
    root: Path,
) -> Path:
    value: Any = record.get(field) if field else None
    if value is None:
        for key in (modality, f"{modality}_path", f"{modality}_file", f"{modality}s"):
            if record.get(key) is not None:
                value = record[key]
                break
    if value is None and isinstance(record.get("segments"), list):
        candidates = [
            item
            for item in record["segments"]
            if str(item.get("modality", item.get("type", ""))).lower() == modality
            and item.get("role", "target") in {"target", "input", "observation"}
        ]
        if candidates:
            value = candidates[0]
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("uri", value.get("path", value.get("file_name")))
    if value in (None, ""):
        raise ValueError(f"Record has no {modality} media path")
    text = os.fspath(value)
    if text.startswith(("http://", "https://", "s3://", "gs://", "hf://")):
        raise ValueError("Baseline cache builder accepts local media paths only")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")
    return path


def _validate_encoded_sample(
    modality: str, media: np.ndarray, condition: np.ndarray
) -> None:
    expected_ndim = {"image": 3, "audio": 2, "video": 4}[modality]
    if media.ndim != expected_ndim:
        raise ValueError(
            f"{modality} codec must return {expected_ndim}D arrays; got {media.shape}"
        )
    if not media.size:
        raise ValueError("Codec returned an empty media representation")
    if modality == "audio":
        if not np.issubdtype(media.dtype, np.integer) or int(media.min()) < 0:
            raise ValueError("Audio codec must return non-negative integer codes")
    elif not np.isfinite(media).all():
        raise ValueError("Continuous media latents contain NaN or infinite values")
    if condition.ndim != 1 or not condition.size or not np.isfinite(condition).all():
        raise ValueError("Text encoder must return one finite 1D condition vector")


def _flush_shard(
    output: Path,
    manifest: Dict[str, Any],
    shard_index: int,
    array_name: str,
    media_batch: Sequence[np.ndarray],
    condition_batch: Sequence[np.ndarray],
    record_indices: Sequence[int],
    *,
    attention_batch: Sequence[np.ndarray],
    processed_records: int,
    failures: Sequence[Dict[str, Any]],
) -> None:
    media = np.stack(media_batch)
    conditions = np.stack(condition_batch).astype(np.float32, copy=False)
    indices = np.asarray(record_indices, dtype=np.int64)
    filename = f"shard-{shard_index:05d}.npz"
    temporary = output / f".{filename}.tmp.npz"
    final = output / filename
    arrays = {
        array_name: media,
        "conditions": conditions,
        "record_indices": indices,
    }
    if attention_batch:
        if len(attention_batch) != len(media_batch):
            raise ValueError("attention masks must exist for every sample in a shard")
        arrays["attention_mask"] = np.stack(attention_batch).astype(np.int64)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, final)
    shard = {
        "path": filename,
        "samples": len(media),
        "sha256": _sha256_file(final),
        "first_record_index": int(indices[0]),
        "last_record_index": int(indices[-1]),
    }
    manifest.setdefault("shards", []).append(shard)
    previous_media = manifest.get("arrays", {}).get(array_name, {})
    media_min = float(media.min())
    media_max = float(media.max())
    manifest["arrays"] = {
        array_name: {
            "dtype": str(media.dtype),
            "shape": list(media.shape[1:]),
            "min": min(float(previous_media.get("min", media_min)), media_min),
            "max": max(float(previous_media.get("max", media_max)), media_max),
        },
        "conditions": {
            "dtype": str(conditions.dtype),
            "shape": list(conditions.shape[1:]),
        },
    }
    if "attention_mask" in arrays:
        manifest["arrays"]["attention_mask"] = {
            "dtype": str(arrays["attention_mask"].dtype),
            "shape": list(arrays["attention_mask"].shape[1:]),
            "min": int(arrays["attention_mask"].min()),
            "max": int(arrays["attention_mask"].max()),
        }
    manifest.update(
        status="building",
        processed_records=processed_records,
        samples=sum(int(item["samples"]) for item in manifest["shards"]),
        failures=list(failures),
    )
    _atomic_write_json(output / "manifest.json", manifest)


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _clear_owned_cache(output: Path) -> None:
    """Remove only files declared by a SaddleLLM-owned media cache manifest."""

    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        if any(output.iterdir()):
            raise ValueError(
                f"Refusing to overwrite non-cache directory without manifest: {output}"
            )
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or "shards" not in manifest:
        raise ValueError(f"Refusing to overwrite unrecognized cache directory: {output}")
    for shard in manifest.get("shards", []):
        path = (output / shard["path"]).resolve()
        if path.parent != output.resolve():
            raise ValueError("Cache manifest contains an unsafe shard path")
        if path.is_file():
            path.unlink()
    manifest_path.unlink()


__all__ = [
    "MediaCacheBuildConfig",
    "ShardedNpzStore",
    "build_media_cache",
]
