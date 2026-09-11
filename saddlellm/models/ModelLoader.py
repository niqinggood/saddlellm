"""Unified loading for native SaddleLLM and Hugging Face causal LMs.

Native SaddleLLM checkpoints are local directories containing both
``saddle_config.json`` and ``pytorch_model.bin``.  Every other reference is
handled by Hugging Face, including remote model IDs.  Imports of modeling
classes stay inside the public functions to keep this module lightweight and
avoid circular imports during package initialization.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union


PathLike = Union[str, os.PathLike[str]]

_SADDLE_CONFIG = "saddle_config.json"
_SADDLE_WEIGHTS = "pytorch_model.bin"


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
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of: {choices}, auto") from exc


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


def load_causal_lm(
    path: PathLike,
    *,
    map_location: Any = None,
    device: Any = None,
    dtype: Any = None,
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
    selected_dtype = _normalize_dtype(
        dtype if dtype is not None else torch_dtype_alias
    )

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

    hf_kwargs = dict(model_kwargs)
    hf_kwargs["trust_remote_code"] = trust_remote_code
    hf_kwargs["local_files_only"] = local_files_only
    if selected_dtype is not None:
        # ``torch_dtype`` remains supported across more Transformers releases
        # than the newer ``dtype`` spelling.
        hf_kwargs["torch_dtype"] = selected_dtype

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
            raise ValueError("pass either map_location or device for a Hugging Face model")
        if explicit_device_map:
            raise ValueError("map_location cannot be combined with device_map")
        target_device = _normalize_device(map_location, allow_auto=True)
        if target_device == "auto":
            hf_kwargs["device_map"] = "auto"
            target_device = None

    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), **hf_kwargs)
    except Exception as exc:
        source = "local Hugging Face checkpoint" if _local_directory(path) else "Hugging Face model"
        raise RuntimeError(f"failed to load {source} {path!s}: {exc}") from exc
    if target_device is not None:
        model = model.to(target_device)
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

    tokenizer_source = tokenizer_path if tokenizer_path is not None else path
    try:
        tokenizer = load_tokenizer_compatible(
            str(tokenizer_source),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to load tokenizer from {tokenizer_source!s}: {exc}"
        ) from exc

    model = load_causal_lm(
        path,
        map_location=map_location,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        **dict(model_kwargs or {}),
    )
    return model, tokenizer


__all__ = [
    "is_saddle_checkpoint",
    "load_causal_lm",
    "load_model_and_tokenizer",
]
