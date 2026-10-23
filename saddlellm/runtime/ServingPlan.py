"""Transparent serving selection: never imports vLLM or loads model weights."""

from __future__ import annotations

import platform
import socket
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.specifiers import SpecifierSet

from .VLLMConfig import VLLMServingConfig

VLLM_VERSION_RANGE = ">=0.30,<0.31"


def check_serving_address(host: str, port: int):
    """Fail before model allocation on a bad/busy address; not a port reservation."""
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a nonempty bind address.")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in 0..65535.")
    try:
        candidates = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        family, kind, protocol, _, address = candidates[0]
        with socket.socket(family, kind, protocol) as probe:
            # Match asyncio's POSIX listener restart semantics: TIME_WAIT is
            # not an active listener. Do NOT enable Windows SO_REUSEADDR,
            # where it can permit another process's live port to be hijacked.
            if platform.system() != "Windows":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(address)
    except OSError as exc:
        raise RuntimeError(f"Cannot bind {host}:{port}; check the address or choose an unused port before loading the model.") from exc


def probe_serving_environment() -> dict:
    """Inspect platform/dependencies/CUDA availability, without engine creation."""
    result = {"platform": platform.system(), "torch_version": None,
              "cuda_available": False, "vllm_version": None, "vllm_supported": False}
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    try:
        result["vllm_version"] = version("vllm")
        result["vllm_supported"] = result["vllm_version"] in SpecifierSet(VLLM_VERSION_RANGE)
    except PackageNotFoundError:
        pass
    return result


def resolve_serving_settings(settings, *, injected_model: bool = False):
    """Return a new validated settings object; caller's configuration stays intact.

    Auto selects a compatible Linux/CUDA vLLM installation for ordinary full
    HF checkpoints. Explicit native/local controls and native/adapter files
    retain their own routes. A chosen engine's startup failure is never caught
    and retried with a slower engine. No automatic quantization/install/offload.
    """
    requested = settings.backend
    selected = requested
    environment = None
    warnings = []
    reason = "Explicit backend selection."
    if requested == "auto":
        environment = probe_serving_environment()
        native_required = (settings.quantization == "int8" or settings.cpu_prepack_linear
                           or settings.native_attention_backend is not None or settings.native_fused_kernels)
        local_controls = (settings.quantization in {"4bit", "8bit"} or settings.batch_size != 1
                          or settings.max_concurrency != 1 or settings.batch_token_budget is not None
                          or bool(settings.batch_length_bucket_size) or bool(settings.prefix_cache_mb)
                          or settings.prefix_cache_partial or settings.truncate_input)
        vllm_controls = settings.vllm != VLLMServingConfig()
        directory = Path(settings.model_path).expanduser()
        local_checkpoint = any((directory / name).is_file() for name in (
            "saddle_config.json", "quantization_manifest.json", "adapter_config.json",
        ))
        capable = (environment["platform"] == "Linux" and environment["cuda_available"]
                   and environment["vllm_supported"] and settings.device in {"auto", "cuda"})
        if vllm_controls and (native_required or local_controls or local_checkpoint or injected_model):
            raise ValueError("vLLM options conflict with native/local controls or this checkpoint; select one route explicitly.")
        if vllm_controls and not capable:
            raise RuntimeError("vLLM options require Linux CUDA and a supported vLLM installation; install saddlellm[serve-vllm] in its own environment.")
        if injected_model:
            selected, reason = "local", "An injected model stays with its existing execution implementation."
        elif native_required:
            selected, reason = "native", "Explicit native attention/INT8/CPU-prepack controls require the Saddle engine."
        elif local_checkpoint or local_controls:
            selected, reason = "local", "Native/adapter checkpoint or local batching/quantization controls need the local route."
        elif capable:
            selected, reason = "vllm", "Compatible Linux CUDA vLLM installation: asynchronous continuous batching."
        else:
            selected, reason = "local", "Compatible vLLM unavailable for this platform/device/environment; using local serving."
            warnings.append("Local serving does not provide vLLM continuous batching. For high-throughput CUDA deployment use a separate Linux saddlellm[serve-vllm] environment.")
        if environment["torch_version"] is None:
            warnings.append("PyTorch is missing; install serving dependencies before starting a model.")
        if environment["vllm_version"] and not environment["vllm_supported"]:
            warnings.append(f"Installed vLLM {environment['vllm_version']} is outside {VLLM_VERSION_RANGE}; it is not selected automatically.")
    if settings.require_high_throughput and selected != "vllm":
        raise RuntimeError("High-throughput continuous batching is required but no compatible vLLM route was selected. Use a supported full HF checkpoint and saddlellm[serve-vllm] on Linux CUDA; native/local are not silently substituted.")
    selection = {
        "requested": requested, "selected": selected, "reason": reason,
        "continuous_batching": selected == "vllm", "warnings": warnings,
        "environment": environment, "startup_fallback": False,
        "automatic_quantization": False,
    }
    return replace(settings, backend=selected, backend_selection=selection)


def serve(model_path: str, *, backend: str = "auto", performance: str = "auto",
          host: str = "127.0.0.1", port: int = 8000,
          **settings_options):
    """Blocking Python deployment entry, equivalent to CLI serve (one worker).

    Call from a main-guarded script for CUDA workers. No automatic package
    installation, weight quantization or restart with a different backend.
    """
    import uvicorn

    from .InferenceServer import create_inference_app
    from .ServingPerformance import build_serving_settings

    settings = build_serving_settings(
        model_path, backend=backend, performance=performance, **settings_options,
    )
    check_serving_address(host, port)
    app = create_inference_app(settings)
    uvicorn.run(app, host=host, port=port, log_level="info")
