"""Purpose-specific, read-only preflight; never loads weights or installs packages."""
from __future__ import annotations

import importlib
import platform
import shutil
from importlib import metadata

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion


def profile_requirements(profile: str) -> dict[str, str]:
    """Return the supported dependency family for one user workflow."""
    shared = {"torch": ">=2.6,<3", "transformers": ">=4.57.6,<5"}
    web = {"fastapi": ">=0.115,<1", "uvicorn": ">=0.30,<1", "pydantic": ">=2.10,<3"}
    if profile == "train":
        from ..training.PostTrainingCompatibility import POST_TRAINING_REQUIREMENTS

        return {**POST_TRAINING_REQUIREMENTS, **shared}
    if profile in {"serve", "native"}:
        requirements = {**shared, **web, "safetensors": ">=0.4.3,<1"}
        if profile == "serve":
            requirements.update(accelerate=">=1.12,<2", peft=">=0.18,<0.19")
        return requirements
    if profile == "vllm":
        from ..runtime.ServingPlan import VLLM_VERSION_RANGE

        return {"torch": ">=2.6,<3", "vllm": VLLM_VERSION_RANGE, **web}
    raise ValueError("profile must be train, serve, native or vllm")


def environment_report(profile: str = "train", device: str = "auto") -> dict:
    """Check dependencies/device readiness, not model capacity or model quality.

    CUDA free memory is a driver snapshot across processes, not total minus this
    process's tensor allocation. Failure to query memory is reported as unknown.
    vLLM is inspected via distribution metadata, not imported/initialized.
    """
    requirements = profile_requirements(profile)
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda or mps")
    issues, warnings, recommendations = [], [], []
    versions, modules, imports = {}, {}, {}
    for package, constraint in requirements.items():
        try:
            found = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
            issues.append(f"Missing dependency: {package} (required {constraint}).")
            continue
        versions[package] = found
        try:
            compatible = found in SpecifierSet(constraint)
        except InvalidVersion:
            compatible = False
        if not compatible:
            issues.append(f"{package}=={found} is outside supported range {constraint}.")
        if package == "vllm":
            imports[package] = "metadata_only"
            continue
        try:
            modules[package] = importlib.import_module(package)
            imports[package] = "ok"
        except Exception as exc:
            imports[package] = "failed"
            issues.append(f"Cannot import {package} ({type(exc).__name__}); check its binary/dependency installation.")
    packages_ok = not issues
    torch = modules.get("torch")
    cuda_available, mps_available, gpus = False, False, []
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            mps_available = bool(mps and mps.is_available())
            count = torch.cuda.device_count() if cuda_available else 0
            for index in range(count):
                entry = {"index": index, "name": "unknown", "free_gb": None, "total_gb": None}
                try:
                    entry["name"] = torch.cuda.get_device_name(index)
                    free, total = torch.cuda.mem_get_info(index)
                    entry.update(free_gb=free / 1024**3, total_gb=total / 1024**3)
                except Exception as exc:
                    warnings.append(f"GPU {index} memory query failed ({type(exc).__name__}); capacity is unknown.")
                gpus.append(entry)
        except Exception as exc:
            issues.append(f"Device discovery failed ({type(exc).__name__}); check PyTorch and the device driver.")
    selected = device
    if selected == "auto":
        selected = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if selected == "cuda" and not cuda_available:
        issues.append("CUDA was requested but is unavailable; install a compatible CUDA PyTorch build and check the driver.")
    if selected == "mps" and not mps_available:
        issues.append("MPS was requested but is unavailable; use an MPS-capable PyTorch build or select CPU.")
    if profile == "vllm":
        if platform.system() != "Linux" or selected != "cuda" or not cuda_available:
            issues.append("Saddle's vLLM route requires Linux with available CUDA; use local/native on this platform.")
        recommendations.append("vLLM engine startup is not tested by doctor. Validate the chosen model with serve --dry-run, then a real request.")
    elif selected == "cpu":
        warnings.append("CPU is supported for small-model checks; practical capacity and speed require a model-specific test.")
    if selected == "cuda":
        for gpu in gpus:
            if gpu["free_gb"] is not None and gpu["free_gb"] < 4:
                warnings.append(f"GPU {gpu['index']} has only {gpu['free_gb']:.1f} GiB free; check other processes and model/KV/workspace budgets.")
    ram_total, ram_available, disk_free = None, None, None
    try:
        ram = importlib.import_module("psutil").virtual_memory()
        ram_total, ram_available = ram.total / 1024**3, ram.available / 1024**3
    except Exception:
        warnings.append("RAM information unavailable; optional psutil enables RAM diagnostics.")
    try:
        disk_free = shutil.disk_usage(".").free / 1024**3
    except OSError:
        warnings.append("Disk capacity query failed; available disk space is unknown.")
    if ram_available is not None and ram_available < 8:
        warnings.append("Available RAM is below 8 GiB; check model loading/CPU offload requirements.")
    if disk_free is not None and disk_free < 10:
        warnings.append("Available disk space is below 10 GiB; check model download and checkpoint requirements.")
    extra = {"train": "posttrain", "serve": "serve", "native": "serve", "vllm": "serve-vllm"}[profile]
    if not packages_ok:
        from .. import __version__

        recommendations.append(f'Use a separate environment and install the matching release: python -m pip install "saddlellm[{extra}]=={__version__}" (use the supplied wheel if not published).')
    recommendations.append("Readiness covers dependencies/devices only. Budget weights, KV cache, activations/workspace and, for training, gradients/optimizer states separately.")
    first = gpus[0] if gpus else {"name": "", "free_gb": 0.0, "total_gb": 0.0}
    post_training = None
    if profile == "train":
        from ..training.PostTrainingCompatibility import post_training_runtime_report

        post_training = post_training_runtime_report()
    return {
        "schema_version": 1, "profile": profile, "requested_device": device, "selected_device": selected,
        "python_version": platform.python_version(), "platform": platform.system(),
        "cuda_available": cuda_available, "mps_available": mps_available,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None) or "N/A",
        "gpu_count": len(gpus), "gpu_name": first["name"], "gpu_memory_gb": first["free_gb"],
        "gpu_total_memory_gb": first["total_gb"], "gpus": gpus, "memory_unit": "GiB",
        "memory_source": "cuda.mem_get_info; driver snapshot, not a reservation",
        "ram_total_gb": ram_total, "ram_available_gb": ram_available, "disk_free_gb": disk_free,
        "pytorch_version": versions.get("torch", "missing"), "transformers_version": versions.get("transformers", "not_required"),
        "packages": versions, "requirements": requirements, "import_checks": imports,
        "packages_ok": packages_ok, "post_training": post_training,
        "issues": issues, "warnings": warnings, "recommendations": recommendations,
        "ready": not issues, "ready_scope": "dependency_device_preflight_not_model_capacity",
    }
