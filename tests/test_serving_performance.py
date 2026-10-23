"""Preset contracts: no real GPU, model downloads, or speed assertions."""
import json
from dataclasses import asdict

import pytest
import torch

from saddlellm import cli
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime import ServingPlan
from saddlellm.runtime.InferenceServer import InferenceServerSettings
from saddlellm.runtime.ServingPerformance import (
    build_serving_settings,
    finalize_serving_performance,
)
from saddlellm.runtime.VLLMConfig import VLLMServingConfig


@pytest.fixture
def environment(monkeypatch):
    values = {"platform": "Linux", "torch_version": "tested", "cuda_available": True,
              "vllm_version": "0.30.0", "vllm_supported": True}
    monkeypatch.setattr(ServingPlan, "probe_serving_environment", lambda: dict(values))
    return values


def tiny_model(dtype=torch.float32):
    return SaddleForCausalLM(SaddleModelConfig(
        vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, attention_kind="gqa",
    )).to(dtype=dtype).eval()


@pytest.mark.parametrize("backend", ["auto", "local", "native", "vllm"])
def test_auto_preserves_existing_settings_and_quality_policy(environment, backend):
    options = {"dtype": "bfloat16", "enable_thinking": True, "api_key": "private-test-key"}
    old = ServingPlan.resolve_serving_settings(InferenceServerSettings(
        model_path="MODEL", backend=backend, **options,
    ))
    new = build_serving_settings("MODEL", backend=backend, **options)
    old_values, new_values = asdict(old), asdict(new)
    new_selection = new_values.pop("backend_selection")
    assert new_selection.pop("performance")["state"] == "default"
    assert new_selection == old_values.pop("backend_selection")
    assert new_values == old_values
    assert "private-test-key" not in json.dumps(new.backend_selection)


def test_throughput_defaults_cover_four_requests_without_quality_changes():
    settings = build_serving_settings("MODEL", backend="native", performance="throughput",
                                      device="cuda", dtype="bfloat16", enable_thinking=True)
    assert settings.native_stream_batching
    assert settings.batch_size == settings.max_concurrency == 4
    assert settings.batch_token_budget == 4 * (4096 + 512)
    assert settings.dtype == "bfloat16" and settings.enable_thinking is True
    assert settings.native_decode_mode == "compatible" and settings.quantization == "none"
    assert not settings.native_fused_kernels and not settings.prefix_cache_mb


def test_explicit_capacity_and_budget_win_over_profile():
    settings = build_serving_settings("MODEL", backend="native", performance="throughput",
                                      batch_size=2, max_concurrency=3, batch_token_budget=321)
    assert (settings.batch_size, settings.max_concurrency, settings.batch_token_budget) == (2, 3, 321)
    assert settings.native_stream_batching
    assert settings.backend_selection["performance"]["applied"] == {"native_stream_batching": True}
    # None is explicit in the Python API: retain the ordinary token-budget rule.
    settings = build_serving_settings("MODEL", backend="native", performance="throughput",
                                      batch_token_budget=None)
    assert settings.batch_token_budget is None


@pytest.mark.parametrize("options", [
    {"max_concurrency": 1}, {"batch_size": 1}, {"native_stream_batching": False},
    {"max_pending_tokens": 1024}, {"batch_length_bucket_size": 32},
    {"device": "cpu", "dtype": "bfloat16"}, {"dtype": "float16"},
    {"device": "mps"}, {"quantization": "int8", "device": "cuda"},
])
def test_conflicting_explicit_options_are_retained_without_partial_profile(options):
    settings = build_serving_settings("MODEL", backend="native", performance="throughput", **options)
    plan = settings.backend_selection["performance"]
    assert plan["state"] == "not_applied" and plan["applied"] == {}
    assert not settings.native_stream_batching
    assert settings.batch_token_budget is None
    for name, value in options.items():
        assert getattr(settings, name) == value


def test_latency_is_bounded_and_explicit_prefix_controls_win():
    settings = build_serving_settings("MODEL", backend="native", performance="latency")
    assert settings.prefix_cache_mb == 64 and not settings.prefix_cache_partial
    assert settings.prefix_cache_min_tokens == 256
    assert settings.prefix_cache_min_store_tokens == 0
    assert settings.max_concurrency == settings.batch_size == 1
    explicit = build_serving_settings("MODEL", backend="native", performance="latency",
                                     prefix_cache_mb=8, prefix_cache_partial=False,
                                     prefix_cache_min_tokens=512)
    assert explicit.prefix_cache_mb == 8 and not explicit.prefix_cache_partial
    assert explicit.prefix_cache_min_tokens == 512
    disabled = build_serving_settings("MODEL", backend="native", performance="latency", prefix_cache_mb=0)
    assert not disabled.prefix_cache_mb and not disabled.prefix_cache_partial
    assert disabled.prefix_cache_min_tokens == 0
    partial = build_serving_settings("MODEL", backend="native", performance="latency",
                                    prefix_cache_partial=True, prefix_cache_min_store_tokens=256)
    assert partial.prefix_cache_partial and partial.prefix_cache_min_store_tokens == 256


def test_chunked_prefill_excludes_latency_cache_but_can_use_throughput_defaults():
    settings = build_serving_settings("MODEL", backend="native", performance="throughput",
                                      native_prefill_chunk_size=128)
    assert settings.native_stream_batching and not settings.prefix_cache_mb
    existing = build_serving_settings("MODEL", backend="native", performance="latency",
                                     native_prefill_chunk_size=128, native_stream_batching=True,
                                     batch_size=2, max_concurrency=2)
    assert existing.native_prefill_chunk_size == 128 and not existing.prefix_cache_mb
    assert existing.backend_selection["performance"]["state"] == "not_applied"


@pytest.mark.parametrize("profile", ["latency", "throughput"])
def test_only_known_saddle_checkpoint_can_be_promoted(environment, tmp_path, profile):
    (tmp_path / "saddle_config.json").write_text("{}", encoding="utf-8")
    settings = build_serving_settings(str(tmp_path), performance=profile)
    assert settings.backend == "native" and settings.backend_selection["requested"] == "auto"
    # An explicit local backend, HF model, or adapter never becomes native.
    assert build_serving_settings(str(tmp_path), backend="local", performance=profile).backend == "local"
    assert build_serving_settings("ordinary-hf-model", performance=profile).backend == "vllm"
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert build_serving_settings(str(tmp_path), performance=profile).backend == "local"


NATIVE_CONTROLS = [
    pytest.param({"quantization": "int8", "device": "cuda"}, id="int8"),
    pytest.param({"cpu_prepack_linear": True, "device": "cpu"}, id="cpu-prepack"),
    pytest.param({"native_attention_backend": "sdpa"}, id="attention"),
    pytest.param({"native_fused_kernels": True, "device": "cuda"}, id="fused-kernels"),
]


@pytest.mark.parametrize("profile", ["latency", "throughput"])
@pytest.mark.parametrize("options", NATIVE_CONTROLS)
def test_auto_explicit_native_controls_receive_profile_defaults(environment, profile, options):
    settings = build_serving_settings("ordinary-hf-model", performance=profile, **options)
    assert settings.backend == "native"
    assert settings.backend_selection["requested"] == "auto"
    assert settings.backend_selection["selected"] == "native"
    assert "Explicit native" in settings.backend_selection["reason"]
    for name, value in options.items():
        assert getattr(settings, name) == value
    plan = settings.backend_selection["performance"]
    if options.get("quantization") == "int8":
        assert plan["state"] == "not_applied" and not plan["applied"]
        assert not settings.native_stream_batching and settings.prefix_cache_mb == 0
    elif profile == "latency":
        assert plan["state"] == "planned"
        assert settings.prefix_cache_mb == 64 and not settings.prefix_cache_partial
    else:
        assert plan["state"] == "planned" and settings.native_stream_batching
        assert settings.batch_size == settings.max_concurrency == 4
        assert settings.batch_token_budget == 4 * (4096 + 512)


@pytest.mark.parametrize("profile", ["latency", "throughput"])
@pytest.mark.parametrize("options", NATIVE_CONTROLS)
def test_auto_native_controls_keep_backend_conflict_errors(environment, profile, options):
    with pytest.raises(ValueError, match="vLLM options conflict"):
        build_serving_settings("ordinary-hf-model", performance=profile,
                               vllm={"max_num_seqs": 8}, **options)
    with pytest.raises(RuntimeError, match="High-throughput continuous batching is required"):
        build_serving_settings("ordinary-hf-model", performance=profile,
                               require_high_throughput=True, **options)


@pytest.mark.parametrize("options", NATIVE_CONTROLS)
def test_auto_performance_keeps_existing_native_control_planning(environment, options):
    old = ServingPlan.resolve_serving_settings(InferenceServerSettings(
        model_path="ordinary-hf-model", backend="auto", **options,
    ))
    new = build_serving_settings("ordinary-hf-model", performance="auto", **options)
    old_values, new_values = asdict(old), asdict(new)
    assert new_values["backend_selection"].pop("performance")["state"] == "default"
    assert new_values == old_values


@pytest.mark.parametrize("profile", ["latency", "throughput"])
def test_vllm_prefix_default_keeps_scheduler_and_explicit_fields(environment, profile):
    settings = build_serving_settings("MODEL", performance=profile,
                                      vllm={"max_num_seqs": 8, "gpu_memory_utilization": 0.23})
    assert settings.backend == "vllm" and settings.vllm.enable_prefix_caching
    assert settings.backend_selection["performance"]["state"] == "engine_managed"
    assert settings.vllm.max_num_seqs == 8 and settings.vllm.gpu_memory_utilization == 0.23
    assert settings.max_concurrency == settings.batch_size == 1
    explicit = build_serving_settings("MODEL", performance=profile,
                                     vllm={"enable_prefix_caching": False})
    assert not explicit.vllm.enable_prefix_caching
    config = VLLMServingConfig(max_num_seqs=8)
    complete = build_serving_settings("MODEL", performance=profile, vllm=config)
    assert not complete.vllm.enable_prefix_caching and not config.enable_prefix_caching


def test_loaded_incompatible_dtype_atomically_reverts_only_profile_fields():
    planned = build_serving_settings("MODEL", backend="native", performance="throughput")
    effective = finalize_serving_performance(planned, tiny_model(torch.bfloat16))
    assert effective.batch_size == effective.max_concurrency == 1
    assert not effective.native_stream_batching and effective.batch_token_budget is None
    assert effective.dtype == "auto" and planned.native_stream_batching
    assert effective.backend_selection["performance"]["state"] == "fallback"
    assert effective.backend_selection["performance"]["capability_checked"]
    assert finalize_serving_performance(effective, tiny_model()) is effective


def test_loaded_standard_model_accepts_plan_without_changing_weights():
    model = tiny_model()
    versions = [p._version for p in model.parameters()]
    planned = build_serving_settings("MODEL", backend="native", performance="throughput", dtype="float32")
    ready = finalize_serving_performance(planned, model)
    assert ready.native_stream_batching
    assert ready.backend_selection["performance"]["state"] == "ready"
    assert [p._version for p in model.parameters()] == versions


def test_custom_hook_reverts_latency_fields_and_preserves_explicit_cache_budget():
    model = tiny_model()
    handle = model.layers[0].register_forward_hook(lambda *_: None)
    planned = build_serving_settings("MODEL", backend="native", performance="latency", prefix_cache_mb=8)
    try:
        effective = finalize_serving_performance(planned, model)
    finally:
        handle.remove()
    assert effective.prefix_cache_mb == 8
    assert not effective.prefix_cache_partial and effective.prefix_cache_min_tokens == 0
    assert effective.backend_selection["performance"]["state"] == "fallback"


def test_unsafe_fallback_with_explicit_dependent_settings_is_clear():
    planned = build_serving_settings("MODEL", backend="native", performance="throughput",
                                      native_stream_batching=True)
    with pytest.raises(ValueError, match="Explicit options require unsupported preset defaults"):
        finalize_serving_performance(planned, tiny_model(torch.bfloat16))


def test_cli_exposes_effective_limits_and_honors_explicit_disabled_controls(environment, capsys):
    args = ["serve", "MODEL", "--backend", "native", "--performance", "throughput", "--dry-run"]
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["performance"]["max_concurrency"] == 4
    assert report["performance"]["batch_token_budget"] == 18432
    assert report["native_decode_mode"] == "compatible" and report["thinking_policy"] == "model_default"
    assert cli.main(args + ["--max-concurrency", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not report["native_stream_batching"] and report["performance"]["batch_size"] == 1
    assert cli.main(["serve", "MODEL", "--performance", "latency", "--no-vllm-prefix-caching", "--dry-run"]) == 0
    assert not json.loads(capsys.readouterr().out)["performance"]["vllm"]["enable_prefix_caching"]


def test_python_serve_performance_keyword_reaches_app_without_changing_dtype(monkeypatch):
    import uvicorn

    from saddlellm.runtime import InferenceServer

    seen = []
    monkeypatch.setattr(ServingPlan, "check_serving_address", lambda *_: None)
    monkeypatch.setattr(InferenceServer, "create_inference_app", lambda settings: seen.append(settings) or "app")
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    ServingPlan.serve("MODEL", backend="native", performance="latency", dtype="float32", enable_thinking=False)
    assert seen[0].prefix_cache_mb == 64
    assert seen[0].dtype == "float32" and seen[0].enable_thinking is False


def test_invalid_profile_is_rejected_before_model_creation():
    with pytest.raises(ValueError, match="performance must"):
        build_serving_settings("MODEL", performance="fastest")


@pytest.mark.parametrize("value", [None, False, "auto", []])
def test_invalid_vllm_config_is_a_clear_error(value):
    with pytest.raises(TypeError, match="vllm must be"):
        build_serving_settings("MODEL", performance="latency", vllm=value)
