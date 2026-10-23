"""Strict safetensors loading, shared by registered native dense adapters.

No HF model is constructed and no remote Python is executed. Configuration,
keys and shapes are checked before allocating real parameters. Tensor-at-a-time
loading avoids a second complete model/state dict, including for sharded 8B.
The Qwen3 entry point stays strict; NativeArchitectures supplies other contracts.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch

from .SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"checkpoint metadata is too large: {path.name}")
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _positive_int(config: dict, name: str, default=None) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"checkpoint {name} must be a positive integer")
    return value


def _positive_float(config: dict, name: str, default: float) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"checkpoint {name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"checkpoint {name} must be a finite positive number")
    return value


def _token_id(value, name: str, vocab_size: int, *, multiple=False):
    if value is None:
        return None
    if multiple and isinstance(value, list):
        if any(item is None for item in value):
            raise ValueError(f"{name} must contain valid vocabulary token IDs")
        return [_token_id(item, name, vocab_size) for item in value]
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < vocab_size
    ):
        raise ValueError(f"{name} must contain valid vocabulary token IDs")
    return value


def qwen3_native_config(raw: dict, generation: dict | None = None) -> SaddleModelConfig:
    """Reject unsupported computation instead of silently approximating it."""
    from .NativeArchitectures import native_architecture_config

    return native_architecture_config(raw, generation, expected_type="qwen3")


def _checkpoint_directory(path, *, local_files_only: bool, config_factory=None) -> Path:
    from .ModelLoader import _local_directory

    config_factory = config_factory or qwen3_native_config
    directory = _local_directory(path)
    if directory is None:
        from huggingface_hub import snapshot_download

        metadata = Path(
            snapshot_download(
                repo_id=str(path),
                local_files_only=local_files_only,
                allow_patterns=["config.json", "generation_config.json"],
            )
        )
        generation_path = metadata / "generation_config.json"
        config_factory(
            _read_json(metadata / "config.json"),
            _read_json(generation_path) if generation_path.is_file() else {},
        )
        # Validate before downloading large weights, and pin the SAME revision
        # for weights/tokenizer rather than racing a moving repository HEAD.
        if not re.fullmatch(r"[0-9a-f]{40}", metadata.name):
            raise ValueError("Hub metadata must resolve to an immutable snapshot")
        directory = Path(
            snapshot_download(
                repo_id=str(path),
                revision=metadata.name,
                local_files_only=local_files_only,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "tokenizer*",
                    "vocab.json",
                    "merges.txt",
                    "special_tokens_map.json",
                    "chat_template*",
                ],
            )
        )
    directory = directory.resolve()
    if (directory / "adapter_config.json").is_file() or (
        directory / "quantization_manifest.json"
    ).is_file():
        raise ValueError(
            "native checkpoint loading requires full, non-quantized weights; "
            "merge/export adapters first"
        )
    return directory


def _weight_files(directory: Path) -> tuple[list[Path], dict | None]:
    index_path = directory / "model.safetensors.index.json"
    single_path = directory / "model.safetensors"
    if index_path.is_file():
        if single_path.is_file():
            raise ValueError(
                "ambiguous checkpoint: both single and sharded model.safetensors exist"
            )
        weight_map = _read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index must contain a nonempty weight_map")
        for key, filename in weight_map.items():
            if (
                not isinstance(key, str)
                or not isinstance(filename, str)
                or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors", filename
                )
            ):
                raise ValueError(
                    "safetensors index must use simple relative shard filenames"
                )
        paths = [directory / name for name in sorted(set(weight_map.values()))]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(
                "checkpoint is missing an indexed safetensors shard"
            )
        return paths, weight_map
    if single_path.is_file():
        return [single_path], None
    raise FileNotFoundError(
        "native loading requires model.safetensors or its complete shard index; "
        "pickle weights are not loaded"
    )


def _native_name(hf_name: str) -> str:
    return hf_name[6:] if hf_name.startswith("model.") else hf_name


def load_qwen3_native(
    path, *, device=None, dtype=None, local_files_only=False, attention_backend="sdpa"
) -> SaddleForCausalLM:
    """Load standard Qwen3 safetensors directly into a frozen Saddle model."""
    return _load_native_checkpoint(
        path, device=device, dtype=dtype, local_files_only=local_files_only,
        attention_backend=attention_backend, config_factory=qwen3_native_config,
    )


def _load_native_checkpoint(
    path, *, device=None, dtype=None, local_files_only=False, attention_backend="sdpa",
    config_factory,
) -> SaddleForCausalLM:
    """Shared tensor-at-a-time loader; config_factory owns architecture semantics."""
    from safetensors import safe_open

    from .ModelLoader import _normalize_device, _normalize_dtype

    directory = _checkpoint_directory(
        path, local_files_only=local_files_only, config_factory=config_factory,
    )
    raw = _read_json(directory / "config.json")
    generation_path = directory / "generation_config.json"
    generation = _read_json(generation_path) if generation_path.is_file() else {}
    config = config_factory(raw, generation)
    label = {"qwen3": "Qwen3", "qwen2": "Qwen2", "llama": "Llama"}[raw["model_type"]]
    if attention_backend not in {None, "sdpa", "sdpa_reference", "eager"}:
        raise ValueError("native attention_backend must be sdpa, sdpa_reference or eager")
    config.attention_backend = attention_backend or "sdpa"
    files, index = _weight_files(directory)
    # Meta construction performs no real initialization or full FP32 allocation.
    with torch.device("meta"):
        model = SaddleForCausalLM(config)
    expected = dict(model.named_parameters(remove_duplicate=False))
    expected_hf = {
        name if name.startswith("lm_head.") else "model." + name for name in expected
    }
    locations, storage_dtypes = {}, {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for key in source.keys():
                if key in locations:
                    raise ValueError(f"duplicate {label} tensor across shards: {key}")
                if key not in expected_hf:
                    raise ValueError(f"unexpected {label} tensor: {key}")
                if index is not None and index.get(key) != path.name:
                    raise ValueError(f"{label} tensor/shard does not match index: {key}")
                view = source.get_slice(key)
                if tuple(view.get_shape()) != tuple(expected[_native_name(key)].shape):
                    raise ValueError(f"{label} tensor shape mismatch: {key}")
                locations[key], storage_dtypes[key] = path, view.get_dtype()
    missing = expected_hf - locations.keys()
    if config.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if missing:
        raise ValueError(f"missing {label} tensors: {', '.join(sorted(missing))}")
    if index is not None and set(index) != set(locations):
        raise ValueError("checkpoint index tensor keys do not match shard headers")
    supported = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    if any(value not in supported for value in storage_dtypes.values()):
        raise ValueError("native checkpoint supports float32/float16/bfloat16 tensors only")
    selected_dtype = _normalize_dtype(dtype)
    if selected_dtype in (None, "auto"):
        selected_dtype = supported[storage_dtypes["model.embed_tokens.weight"]]
    if selected_dtype not in supported.values():
        raise ValueError("native checkpoint dtype must be float32, float16 or bfloat16")
    target = _normalize_device(device, allow_auto=False) or torch.device("cpu")
    if target.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("native checkpoint supports CPU, CUDA or MPS on one device")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for native checkpoint")
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable for native checkpoint")
    element_size = torch.empty((), dtype=selected_dtype).element_size()
    weight_bytes = sum(param.numel() for param in model.parameters()) * element_size
    if target.type == "cuda":
        free, _ = torch.cuda.mem_get_info(target)
        if free < weight_bytes:
            raise MemoryError(
                f"native {label} weights need {weight_bytes} bytes, "
                f"only {free} CUDA bytes free; no implicit CPU offload"
            )
    embedding_source = head_source = None
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for key in source.keys():
                tensor = source.get_tensor(key)
                if config.tie_word_embeddings:
                    if key == "lm_head.weight":
                        head_source = tensor
                        continue
                    if key == "model.embed_tokens.weight":
                        embedding_source = tensor
                name = _native_name(key)
                module_name, attribute = name.rsplit(".", 1)
                module = model.get_submodule(module_name)
                parameter = torch.nn.Parameter(
                    tensor.to(device=target, dtype=selected_dtype), requires_grad=False
                )
                setattr(module, attribute, parameter)
    if config.tie_word_embeddings:
        if head_source is not None and not torch.equal(head_source, embedding_source):
            raise ValueError(
                "tie_word_embeddings declares unequal embedding/lm_head weights"
            )
        model.lm_head.weight = model.embed_tokens.weight
    for layer in model.layers:
        layer.self_attn.rotary.reset_inv_freq(target)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("native checkpoint left unmaterialized parameters")
    # Full-weight Qwen2 BF16 checks found argmax divergence with the automatic
    # efficient/folded SDPA route on CUDA builds without FlashAttention. Match
    # compact-GQA reference dispatch there, unless the caller explicitly opts
    # into another backend. Old Saddle/Qwen3/INT8 policies remain unchanged.
    if attention_backend is None and raw["model_type"] in {"qwen2", "llama"}:
        from .OperatorBackends import _flash_attention_available

        if target.type == "cuda" and selected_dtype == torch.bfloat16 and not _flash_attention_available():
            model.config.attention_backend = "sdpa_reference"
            for layer in model.layers:
                layer.config.attention_backend = "sdpa_reference"
    model._saddle_native_source = {
        "model_type": raw["model_type"],
        "weight_format": "safetensors",
        "shards": len(files),
        "weight_bytes": weight_bytes,
        "dtype": str(selected_dtype),
        "model_forward": "saddle",
        "generation": "native_step",
        "attention_policy": "explicit" if attention_backend is not None else "architecture_default",
    }
    model._saddle_tokenizer_source = str(directory)
    return model.eval()
