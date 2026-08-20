"""Stable release packaging for Hugging Face and native SaddleLLM models."""
from __future__ import annotations

import fnmatch
import gc
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
)
_TRAINING_ONLY_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
    "training.log",
    "summary.json",
    "training_summary.json",
}


@dataclass
class ModelExportRequest:
    model_path: str
    output_dir: str
    format: str = "hf"
    merge_lora: bool = True
    safe_serialization: bool = True
    trust_remote_code: bool = False
    local_files_only: bool = False
    device: str = "auto"
    dtype: str = "auto"
    hash_weights: bool = False
    overwrite: bool = False


@dataclass
class ModelExportResult:
    status: str
    model_path: str
    output_dir: str
    manifest_path: str
    source_model: str
    source_type: str
    format: str = "hf"
    adapter_merged: bool = False
    release_id: str = ""
    files: int = 0
    total_bytes: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelExporter:
    """Create a self-contained, inference-oriented model directory.

    Local full checkpoints are copied without loading when possible. PEFT
    adapters are merged into their declared base model by default. Native
    SaddleLLM checkpoints remain native because their per-layer architecture
    configuration cannot be losslessly represented as a generic HF config.
    """

    @classmethod
    def export(
        cls,
        request: ModelExportRequest,
        *,
        gate: Optional[Mapping[str, Any]] = None,
    ) -> ModelExportResult:
        cls._validate_request(request)
        export_format = cls._canonical_format(request.format)
        destination = Path(request.output_dir).expanduser().resolve()
        source_path = Path(request.model_path).expanduser()
        source_is_local = source_path.exists()
        source = source_path.resolve() if source_is_local else None
        manifest_source = source.name if source is not None else request.model_path
        if source is not None:
            cls._validate_path_relationship(source, destination)
        native_source = source is not None and cls._is_native_checkpoint(source)
        cls._validate_source_format(
            request,
            export_format=export_format,
            source=source,
            native_source=native_source,
        )
        cls._prepare_destination(destination, overwrite=request.overwrite)

        source_type = "huggingface_repository"
        adapter_merged = False
        warnings: List[str] = []
        try:
            if native_source:
                source_type = "native_checkpoint"
                cls._copy_local_release(source, destination)
            elif source is not None and (source / "adapter_config.json").is_file():
                source_type = "peft_adapter"
                if request.merge_lora:
                    warnings.extend(cls._merge_adapter(source, destination, request))
                    adapter_merged = True
                else:
                    cls._copy_local_release(source, destination)
                    warnings.append(
                        "The release contains an unmerged PEFT adapter and requires its base model at inference time."
                    )
            elif source is not None:
                source_type = "local_checkpoint"
                has_safe_weights = bool(list(source.glob("*.safetensors")))
                has_bin_weights = any(
                    path.name not in _TRAINING_ONLY_FILES for path in source.glob("*.bin")
                )
                if request.safe_serialization and has_bin_weights and not has_safe_weights:
                    cls._materialize_model(str(source), destination, request)
                else:
                    cls._copy_local_release(source, destination)
            else:
                cls._materialize_model(request.model_path, destination, request)

            if cls._normalize_tokenizer_config(destination):
                warnings.append(
                    "Normalized generic TokenizersBackend metadata to PreTrainedTokenizerFast for AutoTokenizer compatibility."
                )
            cls._verify_release(
                destination,
                allow_adapter=not request.merge_lora,
                format=export_format,
            )
            cls._write_release_guide(
                destination,
                adapter_merged,
                source_type,
                format=export_format,
            )
            inventory, total_bytes, hashes = cls._inventory(
                destination,
                hash_weights=request.hash_weights,
            )
            release_id = cls._release_id(inventory, hashes)
            manifest = {
                "schema_version": 1,
                "release_id": release_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "format": export_format,
                "source_model": manifest_source,
                "source_type": source_type,
                "adapter_merged": adapter_merged,
                "safe_serialization_requested": request.safe_serialization,
                "gate": dict(gate) if gate is not None else None,
                "inventory": inventory,
                "hashes": hashes,
                "total_bytes": total_bytes,
                "warnings": warnings,
            }
            manifest_path = destination / "release_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except Exception:
            incomplete = destination / ".saddlellm_export_incomplete"
            try:
                incomplete.write_text("Export failed before verification.\n", encoding="utf-8")
            except OSError:
                pass
            raise

        return ModelExportResult(
            status="completed",
            model_path=str(destination),
            output_dir=str(destination),
            manifest_path=str(manifest_path),
            source_model=request.model_path,
            source_type=source_type,
            format=export_format,
            adapter_merged=adapter_merged,
            release_id=release_id,
            files=len(inventory) + 1,
            total_bytes=total_bytes + manifest_path.stat().st_size,
            warnings=warnings,
        )

    @staticmethod
    def _validate_request(request: ModelExportRequest) -> None:
        if not str(request.model_path).strip():
            raise ValueError("model_path is required for export.")
        if not str(request.output_dir).strip():
            raise ValueError("output_dir is required for export.")
        ModelExporter._canonical_format(request.format)
        if request.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")
        if request.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be one of: auto, float32, float16, bfloat16.")

    @staticmethod
    def _canonical_format(format: str) -> str:
        normalized = str(format).strip().lower().replace("_", "-")
        if normalized == "hf":
            return "hf"
        if normalized in {"saddle", "native", "saddle-native", "saddle/native"}:
            return "saddle"
        raise ValueError(
            "format must be 'hf' or native SaddleLLM ('saddle'/'native'). "
            "ONNX and GGUF export are not enabled yet."
        )

    @staticmethod
    def _is_native_checkpoint(source: Path) -> bool:
        return (
            source.is_dir()
            and (source / "saddle_config.json").is_file()
            and (source / "pytorch_model.bin").is_file()
        )

    @classmethod
    def _validate_source_format(
        cls,
        request: ModelExportRequest,
        *,
        export_format: str,
        source: Optional[Path],
        native_source: bool,
    ) -> None:
        if source is not None and (source / "saddle_config.json").is_file() and not native_source:
            raise ValueError(
                "Incomplete native SaddleLLM checkpoint: saddle_config.json exists "
                "but pytorch_model.bin is missing."
            )
        if native_source and export_format == "hf":
            raise ValueError(
                "A native SaddleLLM checkpoint cannot be exported as format='hf' "
                "without a potentially lossy architecture conversion. Use format='saddle'."
            )
        if native_source and request.merge_lora:
            raise ValueError(
                "merge_lora is not supported for native SaddleLLM checkpoints; "
                "set merge_lora=False."
            )
        if export_format == "saddle" and not native_source:
            if source is None:
                raise ValueError(
                    "format='saddle' currently requires a local native SaddleLLM "
                    "checkpoint; remote/Hugging Face models cannot be converted losslessly."
                )
            raise ValueError(
                "format='saddle' requires saddle_config.json and pytorch_model.bin "
                "in the local source directory."
            )

    @staticmethod
    def _validate_path_relationship(source: Path, destination: Path) -> None:
        if not source.is_dir():
            raise ValueError(f"Local model_path must be a directory: {source}")
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("Export source and destination must be separate, non-nested directories.")

    @staticmethod
    def _prepare_destination(destination: Path, *, overwrite: bool) -> None:
        if destination == Path(destination.anchor) or destination == Path.home().resolve():
            raise ValueError(f"Refusing to use broad export destination: {destination}")
        if destination.exists() and any(destination.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Export destination is not empty: {destination}. Use overwrite=true explicitly."
                )
            resolved = destination.resolve()
            if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
                raise ValueError(f"Refusing to overwrite broad export destination: {resolved}")
            shutil.rmtree(resolved)
        destination.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _copy_local_release(cls, source: Path, destination: Path) -> None:
        def ignore(directory: str, names: List[str]) -> List[str]:
            ignored = []
            for name in names:
                if name in {".git", "runs", "logs", "__pycache__"}:
                    ignored.append(name)
                elif name.startswith("checkpoint-"):
                    ignored.append(name)
                elif name in _TRAINING_ONLY_FILES or fnmatch.fnmatch(name, "rng_state*.pth"):
                    ignored.append(name)
            return ignored

        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=ignore)

    @classmethod
    def _merge_adapter(
        cls,
        source: Path,
        destination: Path,
        request: ModelExportRequest,
    ) -> List[str]:
        from .PostTrainingCompatibility import stabilize_peft_optional_backends

        compatibility_warnings = stabilize_peft_optional_backends()
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("Merging a PEFT adapter requires transformers and peft.") from exc

        with open(source / "adapter_config.json", "r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("adapter_config.json does not declare base_model_name_or_path.")

        load_kwargs, target_device = cls._load_kwargs(request)
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        if target_device == "cpu":
            model = model.to("cpu")
        adapter = PeftModel.from_pretrained(
            model,
            str(source),
            local_files_only=request.local_files_only,
        )
        merged = adapter.merge_and_unload()
        merged.save_pretrained(
            destination,
            safe_serialization=request.safe_serialization,
        )
        tokenizer_source = str(source) if cls._has_tokenizer_files(source) else base_model
        from .TokenizerLoader import load_tokenizer_compatible

        tokenizer = load_tokenizer_compatible(
            tokenizer_source,
            trust_remote_code=request.trust_remote_code,
            local_files_only=request.local_files_only,
        )
        tokenizer.save_pretrained(destination)
        del tokenizer, merged, adapter, model
        cls._release_memory()
        return compatibility_warnings

    @classmethod
    def _materialize_model(
        cls,
        model_path: str,
        destination: Path,
        request: ModelExportRequest,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("Model export requires transformers.") from exc

        load_kwargs, target_device = cls._load_kwargs(request)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if target_device == "cpu":
            model = model.to("cpu")
        from .TokenizerLoader import load_tokenizer_compatible

        tokenizer = load_tokenizer_compatible(
            model_path,
            trust_remote_code=request.trust_remote_code,
            local_files_only=request.local_files_only,
        )
        model.save_pretrained(destination, safe_serialization=request.safe_serialization)
        tokenizer.save_pretrained(destination)
        del tokenizer, model
        cls._release_memory()

    @staticmethod
    def _load_kwargs(request: ModelExportRequest) -> Tuple[Dict[str, Any], str]:
        import torch

        target_device = request.device
        if target_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        if target_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested, but CUDA is unavailable.")

        dtype_name = request.dtype
        if dtype_name == "auto":
            dtype_name = "float16" if target_device == "cuda" else "float32"
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_name]
        kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": request.trust_remote_code,
            "local_files_only": request.local_files_only,
        }
        if target_device == "cuda":
            kwargs["device_map"] = "auto"
        return kwargs, target_device

    @staticmethod
    def _has_tokenizer_files(path: Path) -> bool:
        return any(
            (path / name).is_file()
            for name in (
                "tokenizer.json",
                "tokenizer.model",
                "vocab.json",
                "vocab.txt",
            )
        )

    @staticmethod
    def _normalize_tokenizer_config(destination: Path) -> bool:
        config_path = destination / "tokenizer_config.json"
        tokenizer_path = destination / "tokenizer.json"
        if not config_path.is_file() or not tokenizer_path.is_file():
            return False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(config, dict) or config.get("tokenizer_class") != "TokenizersBackend":
            return False
        config["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return True

    @classmethod
    def _verify_release(
        cls,
        destination: Path,
        *,
        allow_adapter: bool,
        format: str = "hf",
    ) -> None:
        if format == "saddle":
            missing = [
                name
                for name in ("saddle_config.json", "pytorch_model.bin")
                if not (destination / name).is_file()
            ]
            if missing:
                raise RuntimeError(
                    "Native SaddleLLM export verification failed: missing "
                    + ", ".join(missing)
                    + "."
                )
            if not cls._has_tokenizer_files(destination):
                raise RuntimeError(
                    "Native SaddleLLM export verification failed: tokenizer files are missing. "
                    "Save the tokenizer into the checkpoint before creating a self-contained release."
                )
            return
        adapter = (destination / "adapter_config.json").is_file()
        if adapter and allow_adapter:
            has_weights = any(destination.glob("adapter_model.*"))
        else:
            if not (destination / "config.json").is_file():
                raise RuntimeError("Export verification failed: config.json is missing.")
            has_weights = any(
                path.is_file()
                for pattern in _WEIGHT_PATTERNS
                for path in destination.glob(pattern)
                if path.name not in _TRAINING_ONLY_FILES
            )
        if not has_weights:
            raise RuntimeError("Export verification failed: model weights are missing.")

    @staticmethod
    def _write_release_guide(
        destination: Path,
        adapter_merged: bool,
        source_type: str,
        *,
        format: str = "hf",
    ) -> None:
        if format == "saddle":
            format_label = "Native SaddleLLM (`saddle_config.json` + `pytorch_model.bin`)"
            usage = (
                "Load from Python:\n\n"
                "```python\n"
                "from saddlellm.ModelLoader import load_causal_lm\n\n"
                "model = load_causal_lm(\".\", device=\"auto\")\n"
                "```\n"
            )
        else:
            format_label = "Hugging Face `save_pretrained`"
            usage = (
                "Start an OpenAI-compatible server:\n\n"
                "```powershell\n"
                "saddle-llm serve-model . --host 127.0.0.1 --port 8000\n"
                "```\n"
            )
        text = (
            "# SaddleLLM model release\n\n"
            f"- Source type: `{source_type}`\n"
            f"- LoRA adapter merged: `{str(adapter_merged).lower()}`\n"
            f"- Format: {format_label}\n\n"
            f"{usage}"
        )
        (destination / "SADDLELLM_RELEASE.md").write_text(text, encoding="utf-8")

    @classmethod
    def _inventory(
        cls,
        destination: Path,
        *,
        hash_weights: bool,
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, str]]:
        inventory: List[Dict[str, Any]] = []
        hashes: Dict[str, str] = {}
        total_bytes = 0
        for path in sorted(item for item in destination.rglob("*") if item.is_file()):
            relative = path.relative_to(destination).as_posix()
            size = path.stat().st_size
            total_bytes += size
            inventory.append({"path": relative, "bytes": size})
            is_weight = any(fnmatch.fnmatch(path.name, pattern) for pattern in _WEIGHT_PATTERNS)
            if hash_weights or not is_weight:
                hashes[relative] = cls._sha256(path)
        return inventory, total_bytes, hashes

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _release_id(inventory: List[Dict[str, Any]], hashes: Dict[str, str]) -> str:
        payload = json.dumps(
            {"inventory": inventory, "hashes": hashes},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "saddlellm-" + hashlib.sha256(payload).hexdigest()[:16]

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def export_model(
    model_path: str,
    output_dir: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience API returning a JSON-serializable export result."""
    return ModelExporter.export(
        ModelExportRequest(model_path=model_path, output_dir=output_dir, **kwargs)
    ).to_dict()
