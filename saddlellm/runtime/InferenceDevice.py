"""Inference device selection without importing torch at CLI/config import time."""

from __future__ import annotations


def resolve_inference_device(requested, torch_module):
    if not isinstance(requested, str) or requested not in {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }:
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps is not None and mps.is_available())
    if requested == "auto":
        return (
            "cuda"
            if torch_module.cuda.is_available()
            else "mps"
            if mps_available
            else "cpu"
        )
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but CUDA is unavailable.")
    if requested == "mps" and not mps_available:
        raise RuntimeError(
            "device='mps' was requested, but Apple MPS is unavailable in this PyTorch/runtime."
        )
    return requested


def resolve_inference_dtype(requested, device):
    if not isinstance(requested, str) or requested not in {
        "auto",
        "float32",
        "float16",
        "bfloat16",
    }:
        raise ValueError("dtype must be one of: auto, float32, float16, bfloat16.")
    # Preserve the CUDA default. MPS starts conservatively at FP32: explicit
    # reduced precision requires verification on the actual OS/model/operators.
    return (
        "float16"
        if requested == "auto" and device == "cuda"
        else "float32"
        if requested == "auto"
        else requested
    )
