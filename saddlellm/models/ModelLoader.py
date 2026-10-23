"""Unified loading for native SaddleLLM and Hugging Face causal LMs.

Native SaddleLLM checkpoints are local directories containing both
``saddle_config.json`` and ``pytorch_model.bin``.  Every other reference is
handled by Hugging Face, including remote model IDs.  Imports of modeling
classes stay inside the public functions to keep this module lightweight and
avoid circular imports during package initialization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

PathLike = Union[str, os.PathLike[str]]

_SADDLE_CONFIG = "saddle_config.json"
_SADDLE_WEIGHTS = "pytorch_model.bin"
_QUANTIZATION_MANIFEST = "quantization_manifest.json"


def _local_directory(path: PathLike) -> Optional[Path]:
    """Return an existing local directory without interpreting remote IDs."""

    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("model path must be a string or os.PathLike value")
    try:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_dir() else None
    except (OSError, ValueError, TypeError):
        # Values such as URLs or otherwise invalid filesystem strings are
        # legitimate inputs to Hugging Face and are therefore not local dirs.
        return None


def is_saddle_checkpoint(path: PathLike) -> bool:
    """Return whether ``path`` is a complete local SaddleLLM checkpoint.

    This function never queries the Hub and only recognizes existing local
    directories.  Consequently a remote ID such as ``"org/model"`` is not
    accidentally classified as a native checkpoint.
    """

    directory = _local_directory(path)
    return bool(
        directory is not None
        and (directory / _SADDLE_CONFIG).is_file()
        and (directory / _SADDLE_WEIGHTS).is_file()
    )


def is_packaged_quantized_checkpoint(path: PathLike) -> bool:
    """Return whether ``path`` is a packaged SaddleLLM quantized model."""

    directory = _local_directory(path)
    return bool(
        directory is not None and (directory / _QUANTIZATION_MANIFEST).is_file()
    )


def packaged_quantization_kind(path: PathLike) -> str | None:
    """Identify the numeric/storage contract without importing a model runtime."""
    directory = _local_directory(path)
    if directory is None or not (directory / _QUANTIZATION_MANIFEST).is_file():
        return None
    manifest = json.loads((directory / _QUANTIZATION_MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("quantization_manifest.json must be an object")
    method = manifest.get("method")
    if method == "dynamic" and manifest.get("precision", "int8") == "int8":
        return "dynamic-int8"
    if method == "qat" and manifest.get("backend") == "portable-int8-weight-only":
        return "portable-int8-weight-only"
    if method in {"awq", "gptq"} and manifest.get("format") == "compressed-tensors" and manifest.get("backend") == "llm-compressor":
        return "compressed-tensors"
    raise ValueError("Unsupported packaged quantization method/backend/format; use its matching loader")


def resolve_model_reference(
    path: PathLike,
    *,
    local_files_only: bool = False,
) -> str:
    """Resolve a model reference to an unambiguous source.

    Existing directories are returned as absolute paths.  In offline mode a
    Hugging Face repository ID is first resolved to its cached snapshot.  This
    prevents Transformers metadata probes from reaching the network even when
    ``local_files_only=True`` was supplied to ``from_pretrained``.
    """

    directory = _local_directory(path)
    if directory is not None:
        return str(directory.resolve())
    if not local_files_only:
        return str(path)

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=str(path), local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            f"model {path!s} is not an existing directory or a complete cached "
            "Hugging Face snapshot"
        ) from exc


def _normalize_dtype(dtype: Any) -> Any:
    if dtype is None or dtype == "auto":
        return dtype

    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError(
            "dtype must be a torch.dtype, a supported dtype name, 'auto', or None"
        )
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
    }
    normalized = dtype.strip().lower().replace("torch.", "")
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(aliases))
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of: {choices}, auto"
        ) from exc


def _normalize_device(device: Any, *, allow_auto: bool = True) -> Any:
    if device is None:
        return None
    if isinstance(device, str) and device.strip().lower() == "auto":
        if allow_auto:
            return "auto"
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torch

    try:
        return torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid device {device!r}") from exc


def _native_checkpoint_error(path: PathLike) -> Optional[FileNotFoundError]:
    """Describe a local directory that advertises an incomplete native save."""

    directory = _local_directory(path)
    if directory is None or not (directory / _SADDLE_CONFIG).is_file():
        return None
    if not (directory / _SADDLE_WEIGHTS).is_file():
        return FileNotFoundError(
            f"incomplete SaddleLLM checkpoint at {directory}: "
            f"{_SADDLE_CONFIG} exists but {_SADDLE_WEIGHTS} is missing"
        )
    return None


def _load_packaged_qat(path, *, map_location, device, selected_dtype, quantized_backend,
                       trust_remote_code, local_files_only, model_kwargs):
    """Translate canonical loader options to the matching QAT execution request."""
    import torch

    if model_kwargs:
        raise TypeError("portable QAT loading does not accept extra Hugging Face model arguments")
    if quantized_backend != "cuda-w8a16" and selected_dtype not in (None, "auto", torch.float32):
        raise ValueError("portable INT8 weight-only QAT requires dtype='float32'")
    if device is not None and map_location is not None:
        raise ValueError("pass either map_location or device")
    target = _normalize_device(device if device is not None else map_location, allow_auto=True)
    if quantized_backend != "cuda-w8a16" and (target is None or target == "auto"):
        target = "cpu"
    from ..compression.QuantizationAwareTraining import load_qat

    options = {}
    if quantized_backend == "cuda-w8a16":
        options = {"inference_backend": "cuda-w8a16", "dtype": selected_dtype}
    return load_qat(path, device=target, trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only, **options)


def load_causal_lm(
    path: PathLike,
    *,
    map_location: Any = None,
    device: Any = None,
    dtype: Any = None,
    quantized_backend: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    **model_kwargs: Any,
) -> Any:
    """Load a native SaddleLLM checkpoint or a Hugging Face causal LM.

    Args:
        path: Local checkpoint path or Hugging Face model ID.
        map_location: PyTorch checkpoint mapping for native checkpoints.  For
            Hugging Face models, a string or ``torch.device`` acts as the
            target device when ``device`` is omitted.
        device: Final device.  For Hugging Face models ``"auto"`` selects
            ``device_map="auto"``; for native models it selects CUDA when
            available, otherwise CPU.
        dtype: A ``torch.dtype``, common name such as ``"bf16"``, ``"auto"``,
            or ``None``.  ``torch_dtype`` is accepted as a compatibility alias
            in ``model_kwargs``.
        quantized_backend: Optional explicit ``"cuda-w8a16"`` execution of a
            portable QAT artifact. Requires explicit CUDA and FP16/BF16 dtype.
            ``None``/``"reference"`` preserve default loading. Other formats
            cannot use this backend and no fallback is performed.
        trust_remote_code: Forwarded to Hugging Face loading.
        local_files_only: Forwarded to Hugging Face loading.
        **model_kwargs: Additional Hugging Face ``from_pretrained`` arguments.
            Native loading intentionally rejects these because its checkpoint
            loader has no equivalent semantics.
    """

    # Validate the public path type even when it is ultimately a remote ID.
    _local_directory(path)
    torch_dtype_alias = model_kwargs.pop("torch_dtype", None)
    if dtype is not None and torch_dtype_alias is not None:
        raise ValueError("pass only one of dtype and torch_dtype")
    selected_dtype = _normalize_dtype(dtype if dtype is not None else torch_dtype_alias)
    if quantized_backend not in (None, "reference", "cuda-w8a16"):
        raise ValueError("quantized_backend must be reference or cuda-w8a16")
    if quantized_backend == "cuda-w8a16" and packaged_quantization_kind(path) != "portable-int8-weight-only":
        raise ValueError("cuda-w8a16 quantized_backend requires a portable QAT artifact")

    if is_packaged_quantized_checkpoint(path):
        kind = packaged_quantization_kind(path)
        if kind == "compressed-tensors":
            checkpoint_config = json.loads((Path(path).expanduser() / "config.json").read_text(encoding="utf-8"))
            quantized = checkpoint_config.get("quantization_config") if isinstance(checkpoint_config, dict) else None
            if not isinstance(quantized, dict) or quantized.get("quant_method") != "compressed-tensors" or quantized.get("quantization_status") != "compressed" or quantized.get("format") not in {"pack-quantized", "int-quantized"}:
                raise ValueError("compressed-tensors manifest requires a genuinely compressed integer quantization_config")
            if model_kwargs:
                raise TypeError("compressed-tensors reference loading accepts device/dtype, not additional Hugging Face model arguments")
            if device is not None and map_location is not None:
                raise ValueError("pass either map_location or device")
            from ..compression.CalibratedPTQ import load_calibrated_ptq

            target = device if device is not None else map_location
            model = load_calibrated_ptq(
                path, device="cpu" if target is None else target,
                dtype=None if selected_dtype in (None, "auto") else selected_dtype,
                trust_remote_code=trust_remote_code, local_files_only=local_files_only,
            )
            model._saddle_quantization_load_mode = "compressed-tensors-decompressed"
            return model
        if kind == "portable-int8-weight-only":
            return _load_packaged_qat(
                path, map_location=map_location, device=device, selected_dtype=selected_dtype,
                quantized_backend=quantized_backend, trust_remote_code=trust_remote_code,
                local_files_only=local_files_only, model_kwargs=model_kwargs,
            )
        if model_kwargs:
            names = ", ".join(sorted(model_kwargs))
            raise TypeError(
                "packaged quantized loading does not support Hugging Face model "
                f"argument(s): {names}"
            )
        import torch

        requested_device = device if device is not None else map_location
        target_device = _normalize_device(requested_device, allow_auto=True)
        if target_device == "auto":
            target_device = torch.device("cpu")
        if target_device is not None and target_device.type != "cpu":
            raise ValueError("packaged dynamic INT8 models require device='cpu'")
        if selected_dtype not in (None, "auto", torch.float32):
            raise ValueError("packaged dynamic INT8 models require dtype='float32'")
        from ..compression.quantizer import ModelQuantizer

        try:
            quantizer = ModelQuantizer.load(
                str(Path(path).expanduser().resolve()),
                device_map="cpu",
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load packaged quantized checkpoint from {path!s}: {exc}"
            ) from exc
        return quantizer.model

    if is_saddle_checkpoint(path):
        if model_kwargs:
            names = ", ".join(sorted(model_kwargs))
            raise TypeError(
                "native SaddleLLM loading does not support Hugging Face model "
                f"argument(s): {names}"
            )
        from .SaddleModeling import SaddleForCausalLM

        # Native state dicts are mapped to CPU by default for portability and
        # moved exactly once after construction when a device was requested.
        load_location = "cpu" if map_location is None else map_location
        try:
            model = SaddleForCausalLM.from_pretrained_saddle(
                str(Path(path).expanduser()), map_location=load_location
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load native SaddleLLM checkpoint from {path!s}: {exc}"
            ) from exc

        target_device = _normalize_device(device, allow_auto=False)
        to_kwargs = {}
        if target_device is not None:
            to_kwargs["device"] = target_device
        if selected_dtype not in (None, "auto"):
            to_kwargs["dtype"] = selected_dtype
        if to_kwargs:
            model = model.to(**to_kwargs)
        return model

    incomplete_error = _native_checkpoint_error(path)
    if incomplete_error is not None:
        raise incomplete_error

    from transformers import AutoModelForCausalLM

    resolved_path = resolve_model_reference(path, local_files_only=local_files_only)

    hf_kwargs = dict(model_kwargs)
    hf_kwargs["trust_remote_code"] = trust_remote_code
    hf_kwargs["local_files_only"] = local_files_only
    if selected_dtype is not None:
        # SaddleLLM's supported Transformers line (>=4.57.6) uses ``dtype``;
        # keep accepting the legacy public ``torch_dtype`` alias above, but do
        # not forward it and trigger a deprecation warning.
        hf_kwargs["dtype"] = selected_dtype

    explicit_device_map = "device_map" in hf_kwargs
    target_device = _normalize_device(device, allow_auto=True)
    if target_device == "auto":
        if explicit_device_map and hf_kwargs["device_map"] != "auto":
            raise ValueError("device='auto' conflicts with the supplied device_map")
        hf_kwargs.setdefault("device_map", "auto")
        target_device = None
    elif target_device is not None and explicit_device_map:
        raise ValueError("pass either device or device_map, not both")

    if map_location is not None:
        if target_device is not None:
            raise ValueError(
                "pass either map_location or device for a Hugging Face model"
            )
        if explicit_device_map:
            raise ValueError("map_location cannot be combined with device_map")
        target_device = _normalize_device(map_location, allow_auto=True)
        if target_device == "auto":
            hf_kwargs["device_map"] = "auto"
            target_device = None

    try:
        model = AutoModelForCausalLM.from_pretrained(resolved_path, **hf_kwargs)
    except Exception as exc:
        source = (
            "local Hugging Face checkpoint"
            if _local_directory(path)
            else "Hugging Face model"
        )
        raise RuntimeError(f"failed to load {source} {path!s}: {exc}") from exc
    if target_device is not None:
        model = model.to(target_device)
    return model


def load_native_causal_lm(
    path: PathLike,
    *,
    device: Any = None,
    dtype: Any = None,
    local_files_only: bool = False,
    attention_backend: Optional[str] = None,
    quantization: Optional[str] = None,
    fuse_projections: bool = True,
    gpu_reserve_bytes: int = 3 * 1024**3,
) -> Any:
    """Inference-only native loader: Saddle or registered dense HF architectures.

    Unlike ``load_causal_lm``, this never falls back to an HF model. Supported
    Qwen3/Qwen2/Llama safetensors stream directly into Saddle's implementation;
    architecture-specific variants are validated before weights are loaded.

    Optional ``quantization='int8'`` is experimental CUDA/Triton W8A16 for dense
    Qwen3 float safetensors. It lossily quantizes hidden linears on CPU before
    moving ONE compressed model to CUDA; activations/embeddings/head/norms remain float.
    FP16/BF16 compute, explicit CUDA device and spare RAM are required. Defaults
    remain unquantized. INT8 defaults to eager attention. No C++ extension is
    built; small Triton kernels are JIT compiled on first use.
    Runtime INT8 models cannot be trained/exported as
    float checkpoints. ``fuse_projections`` affects the INT8 loading path only.
    """
    if attention_backend not in (None, "sdpa", "sdpa_reference", "eager"):
        raise ValueError("native attention_backend must be sdpa, sdpa_reference, eager or None")
    if quantization not in (None, "none", "int8"):
        raise ValueError("native quantization must be none or int8")
    if not isinstance(fuse_projections, bool):
        raise ValueError("fuse_projections must be boolean")
    if (
        isinstance(gpu_reserve_bytes, bool)
        or not isinstance(gpu_reserve_bytes, int)
        or gpu_reserve_bytes < 1024**3
    ):
        raise ValueError("gpu_reserve_bytes must be an integer >= 1 GiB")
    if is_packaged_quantized_checkpoint(path):
        raise ValueError(
            "the native backend currently requires non-quantized checkpoints"
        )
    if quantization == "int8":
        if is_saddle_checkpoint(path):
            raise ValueError(
                "native INT8 currently supports dense Qwen3 float safetensors only"
            )
        return _load_native_qwen3_int8(
            path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            attention_backend=attention_backend or "eager",
            fuse_projections=fuse_projections,
            gpu_reserve_bytes=gpu_reserve_bytes,
        )
    if is_saddle_checkpoint(path):
        model = (
            load_causal_lm(
                path, device=device, dtype=dtype, local_files_only=local_files_only
            )
            .eval()
            .requires_grad_(False)
        )
        if attention_backend is not None:
            if attention_backend != "eager" and any(
                layer.config.attention_kind == "kda" for layer in model.layers
            ):
                raise ValueError("KDA layers require their eager attention backend")
            model.config.attention_backend = attention_backend
            for layer in model.layers:
                layer.config.attention_backend = attention_backend
        return model
    incomplete_error = _native_checkpoint_error(path)
    if incomplete_error is not None:
        raise incomplete_error
    from .NativeArchitectures import load_native_transformer

    return load_native_transformer(
        path,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        attention_backend=attention_backend,
    )


def _load_native_qwen3_int8(
    path,
    *,
    device,
    dtype,
    local_files_only,
    attention_backend,
    fuse_projections,
    gpu_reserve_bytes,
):
    """Validated CPU preparation, conservative GPU budget, no HF fallback."""
    import torch
    from torch import nn

    from .CPULinear import available_cpu_memory
    from .CUDAInt8 import cuda_int8_supported, prepare_cuda_int8
    from .InferenceOptimizer import optimize_native_inference
    from .NativeQwen3 import (
        _checkpoint_directory,
        _read_json,
        load_qwen3_native,
        qwen3_native_config,
    )
    from .SaddleModeling import SaddleForCausalLM

    target = _normalize_device(device, allow_auto=False)
    if target is None or target.type != "cuda":
        raise ValueError("native INT8 requires an explicit CUDA device")
    selected_dtype = _normalize_dtype(dtype)
    if selected_dtype in (None, "auto"):
        selected_dtype = torch.bfloat16
    if selected_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("native INT8 compute dtype must be float16 or bfloat16")
    supported, reason = cuda_int8_supported(target)
    if not supported:
        raise RuntimeError(reason)
    if selected_dtype == torch.bfloat16 and torch.cuda.get_device_capability(target) < (
        8,
        0,
    ):
        raise ValueError(
            "BF16 native INT8 compute requires CUDA capability >= 8.0; use float16"
        )
    directory = _checkpoint_directory(path, local_files_only=local_files_only)
    config = qwen3_native_config(_read_json(directory / "config.json"))
    with torch.device("meta"):
        shape_model = SaddleForCausalLM(config)
    float_bytes = sum(p.numel() for p in shape_model.parameters()) * 2
    estimated = float_bytes
    for name, module in shape_model.named_modules():
        if type(module) is nn.Linear and name != "lm_head":
            padded_in = ((module.in_features + 15) // 16) * 16
            padded_out = ((module.out_features + 15) // 16) * 16
            estimated += (
                padded_in * padded_out + padded_out * 4 - module.weight.numel() * 2
            )
    del shape_model, module
    free, _ = torch.cuda.mem_get_info(target)
    if estimated + gpu_reserve_bytes > free:
        raise MemoryError(
            "native INT8 weights plus GPU reserve exceed available CUDA memory"
        )
    available_ram = available_cpu_memory()
    if available_ram is not None and available_ram < float_bytes + 1024**3:
        raise MemoryError(
            "native INT8 CPU preparation needs float weights plus 1 GiB of RAM"
        )
    model = load_qwen3_native(
        directory,
        device="cpu",
        dtype=selected_dtype,
        local_files_only=True,
        attention_backend=attention_backend,
    )
    if fuse_projections:
        model._saddle_inference_optimization = optimize_native_inference(
            model
        ).to_dict()
    report = prepare_cuda_int8(model)
    required = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    free, _ = torch.cuda.mem_get_info(target)
    if required + gpu_reserve_bytes > free:
        raise MemoryError(
            "GPU budget changed during INT8 preparation; refusing allocation"
        )
    try:
        model.to(target)
        torch.cuda.synchronize(target)
        remaining, _ = torch.cuda.mem_get_info(target)
        if remaining < gpu_reserve_bytes:
            raise MemoryError("remaining GPU memory is below native INT8 reserve")
    except (RuntimeError, MemoryError):
        model.cpu()
        torch.cuda.empty_cache()
        raise
    report.update(
        {"gpu_reserve_bytes": gpu_reserve_bytes, "compute_dtype": str(selected_dtype)}
    )
    return model


def load_model_and_tokenizer(
    path: PathLike,
    *,
    tokenizer_path: Optional[PathLike] = None,
    map_location: Any = None,
    device: Any = None,
    dtype: Any = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Load a causal LM and its compatible tokenizer as ``(model, tokenizer)``.

    ``tokenizer_path`` is useful when a native model checkpoint stores only
    model files.  Tokenizer loading goes through
    :func:`TokenizerLoader.load_tokenizer_compatible`, including its local
    ``tokenizer.json`` fallback.
    """

    from .TokenizerLoader import load_tokenizer_compatible

    resolved_path = resolve_model_reference(path, local_files_only=local_files_only)
    tokenizer_source = tokenizer_path if tokenizer_path is not None else resolved_path
    resolved_tokenizer_source = resolve_model_reference(
        tokenizer_source,
        local_files_only=local_files_only,
    )
    try:
        tokenizer = load_tokenizer_compatible(
            resolved_tokenizer_source,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to load tokenizer from {tokenizer_source!s}: {exc}"
        ) from exc

    model = load_causal_lm(
        resolved_path,
        map_location=map_location,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        **dict(model_kwargs or {}),
    )
    return model, tokenizer


__all__ = [
    "is_packaged_quantized_checkpoint",
    "is_saddle_checkpoint",
    "load_causal_lm",
    "load_native_causal_lm",
    "load_model_and_tokenizer",
    "packaged_quantization_kind",
    "resolve_model_reference",
]
