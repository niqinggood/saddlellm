"""Small serving presets over existing backends, with explicit options winning.

Planning loads no weights. Presets never change dtype, quantization, thinking,
or the native numerical mode, and do not claim to reserve GPU memory.
"""
from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path


def _known_saddle_checkpoint(model_path):
    directory = Path(model_path).expanduser()
    return ((directory / "saddle_config.json").is_file()
            and not (directory / "adapter_config.json").is_file()
            and not (directory / "quantization_manifest.json").is_file())


def build_serving_settings(model_path: str, *, backend="auto", performance="auto", **options):
    """Build resolved settings; only omitted options receive preset defaults.

    A vllm mapping preserves its explicitly supplied fields. Passing a complete
    VLLMServingConfig treats every field in that object as explicit.
    """
    from .InferenceServer import InferenceServerSettings
    from .ServingPlan import resolve_serving_settings
    from .VLLMConfig import VLLMServingConfig

    if performance not in {"auto", "latency", "throughput"}:
        raise ValueError("performance must be auto, latency or throughput.")
    if "backend_selection" in options:
        raise ValueError("backend_selection is produced by serving planning; do not supply it.")
    options = dict(options)
    explicit = set(options)
    raw_vllm = options.get("vllm", {})
    if not isinstance(raw_vllm, (dict, VLLMServingConfig)):
        raise TypeError("vllm must be VLLMServingConfig or a settings mapping.")
    explicit_vllm = (set(raw_vllm) if isinstance(raw_vllm, dict)
                     else {item.name for item in fields(VLLMServingConfig)})
    # Match ServingPlan's explicit native routing before merging prerequisites.
    # Arbitrary HF models still require one of these controls or backend=native.
    native_controls = (performance != "auto" and backend == "auto" and (
        options.get("quantization", "none") == "int8"
        or options.get("cpu_prepack_linear", False)
        or options.get("native_attention_backend") is not None
        or options.get("native_fused_kernels", False)
    ))
    if native_controls and (
        VLLMServingConfig(**raw_vllm) if isinstance(raw_vllm, dict) else raw_vllm
    ) != VLLMServingConfig():
        raise ValueError("vLLM options conflict with native/local controls or this checkpoint; select one route explicitly.")
    promoted = (performance != "auto" and backend == "auto"
                and _known_saddle_checkpoint(model_path)
                and options.get("quantization", "none") == "none"
                and not options.get("trust_remote_code", False)
                and not options.get("require_high_throughput", False)
                and (VLLMServingConfig(**raw_vllm) if isinstance(raw_vllm, dict)
                     else raw_vllm) == VLLMServingConfig())
    selected = "native" if native_controls or promoted else backend
    defaults = {item.name: item.default for item in fields(InferenceServerSettings)}
    applied, rollback = {}, {}
    reason = "Existing backend and performance settings retained."

    def supply(name, value):
        if name not in explicit:
            options[name] = value
            if value != defaults[name]:
                applied[name] = value
                rollback[name] = defaults[name]

    # Native presets are merged before dataclass validation, so e.g. chunked
    # prefill can receive its omitted batching prerequisites from throughput.
    if performance != "auto" and selected == "native":
        reason = _native_plan_exclusion(performance, options)
        if reason is None and performance == "latency":
            supply("prefix_cache_mb", 64)
            if options.get("prefix_cache_mb", defaults["prefix_cache_mb"]):
                # Partial snapshots still missed the end-to-end performance
                # gate; use complete prompt matches unless explicitly enabled.
                supply("prefix_cache_partial", False)
                supply("prefix_cache_min_tokens", 256)
            reason = "Bounded reuse of repeated complete prompts; short, unique or growing prompts may not benefit."
        elif reason is None:
            supply("batch_size", 4)
            supply("max_concurrency", 4)
            supply("native_stream_batching", True)
            rows = min(options.get("batch_size", 1), options.get("max_concurrency", 1))
            tokens = (options.get("max_input_tokens", defaults["max_input_tokens"])
                      + options.get("max_new_tokens", defaults["max_new_tokens"]))
            supply("batch_token_budget", rows * tokens)
            reason = "Native step batching with the existing numerical and generation policies."

    settings = resolve_serving_settings(InferenceServerSettings(
        model_path=model_path, backend=selected, **options,
    ))
    if performance != "auto" and settings.backend == "vllm":
        if "enable_prefix_caching" not in explicit_vllm:
            settings = replace(settings, vllm=replace(settings.vllm, enable_prefix_caching=True))
            applied["vllm.enable_prefix_caching"] = True
        reason = "vLLM prefix caching; existing scheduler and memory settings retained."
    elif performance != "auto" and settings.backend != "native":
        reason = "This backend retains its execution settings; no native conversion of HF or adapter models."

    selection = dict(settings.backend_selection)
    if native_controls:
        selection.update(requested=backend, reason="Explicit native attention/INT8/CPU-prepack/fused-kernel controls require the Saddle engine; performance defaults follow that route.")
    elif promoted:
        selection.update(requested=backend, reason="Explicit performance preset uses the existing Saddle checkpoint's native implementation.")
    plan = {
        "requested": performance,
        "effective": performance if applied else "auto",
        "state": "planned" if applied else "default" if performance == "auto" else "not_applied",
        "reason": reason,
        "applied": applied,
        "fallback_values": rollback,
        "explicit_fields": sorted(explicit),
        "memory_policy": "configured_limits_not_vram_admission",
        "capability_checked": False,
    }
    if applied and settings.backend == "vllm":
        plan["state"] = "engine_managed"
    selection["performance"] = plan
    return replace(settings, backend_selection=selection)


def _native_plan_exclusion(performance, options):
    if options.get("quantization", "none") != "none":
        return "Native performance presets require unquantized weights; explicit quantization retained."
    if performance == "latency":
        if options.get("native_prefill_chunk_size", 0):
            return "Prefix snapshots and chunked prefill cannot be combined; explicit prefill settings retained."
        if options.get("prefix_cache_mb") == 0:
            return "Explicit prefix_cache_mb=0 disables the latency preset's prefix cache."
    else:
        if options.get("native_stream_batching") is False:
            return "Explicit native_stream_batching=False disables preset batching."
        if options.get("batch_size") == 1 or options.get("max_concurrency") == 1:
            return "Explicit single-request batch or concurrency limit retained."
        if options.get("max_pending_tokens") is not None or options.get("batch_length_bucket_size", 0):
            return "Explicit fixed-microbatch controls retained; they cannot be combined with step batching."
        if (options.get("device") == "mps" or options.get("dtype") == "float16"
                or (options.get("device") == "cpu" and options.get("dtype") == "bfloat16")):
            return "Native step batching requires FP32 or CUDA BF16; explicit device and dtype retained."
    return None


def _loaded_native_exclusion(settings, model):
    import torch

    from ..models.NativePrefixCache import NativePrefixCache
    from ..models.SaddleModeling import SaddleForCausalLM

    if type(model) is not SaddleForCausalLM:
        return "Loaded model is not a standard native Saddle model."
    weight = model.embed_tokens.weight
    if settings.native_stream_batching and (
        weight.dtype != torch.float32
        and not (weight.dtype == torch.bfloat16 and weight.device.type == "cuda")
    ):
        return "Loaded dtype/device does not support native step batching."
    if (getattr(model, "_saddle_cuda_int8", None) is not None
            or not model.layers
            or any(layer.config.attention_kind not in {"mha", "mqa", "gqa"}
                   or layer.config.attention_backend not in {"sdpa", "sdpa_reference", "eager"}
                   or layer.config.sliding_window or layer.config.num_experts
                   or layer.config.residual_attention_mode != "none" for layer in model.layers)):
        return "Loaded architecture is outside the preset's dense native support."
    if NativePrefixCache._fingerprint(model, weight.device) is None:
        return "Loaded model has unsupported execution hooks, custom modules, training state or tensor layout."
    return None


def finalize_serving_performance(settings, model):
    """Check a loaded native model before engine/cache/batcher construction.

    On incompatibility undo ALL preset-added runtime fields together. Explicit
    settings are never changed. If those cannot stand alone, report an error
    before serving instead of silently changing the user's requested behavior.
    This reads model metadata, with no GPU allocation or memory auto-sizing.
    """
    selection = settings.backend_selection or {}
    plan = selection.get("performance")
    if not plan or plan["state"] != "planned" or settings.backend != "native":
        return settings
    reason = _loaded_native_exclusion(settings, model)
    updated_plan = dict(plan, capability_checked=True, state="ready")
    if reason is not None:
        updated_plan.update(state="fallback", effective="auto", reason=reason, applied={})
    updated_selection = dict(selection, performance=updated_plan)
    try:
        return replace(settings, backend_selection=updated_selection,
                       **(plan["fallback_values"] if reason else {}))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"performance={plan['requested']} unavailable: {reason} "
            "Explicit options require unsupported preset defaults; choose compatible options."
        ) from exc


def performance_settings_summary(settings):
    """Public, credential-free effective settings for dry-run and diagnostics."""
    return {
        "max_concurrency": settings.max_concurrency,
        "batch_size": settings.batch_size,
        "batch_token_budget": settings.resolved_batch_token_budget,
        "prefix_cache_mb": settings.prefix_cache_mb,
        "prefix_cache_partial": settings.prefix_cache_partial,
        "prefix_cache_min_tokens": settings.prefix_cache_min_tokens,
        "prefix_cache_min_store_tokens": settings.prefix_cache_min_store_tokens,
        "vllm": asdict(settings.vllm),
    }
