"""CPU decode caches must accelerate only safe paths and retain canonical weights."""

import copy

import pytest
import torch
from torch import nn

from saddlellm.models import CPULinear
from saddlellm.models.CPULinear import (
    CPUPrepackedLinear,
    cpu_prepack_status,
    cpu_prepack_supported,
    prepare_cpu_linears,
)
from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


@pytest.fixture(autouse=True)
def cpu_resources(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(CPULinear, "available_cpu_memory", lambda: 8 * 1024**3)
    yield
    torch.set_num_threads(previous)


def linear(bias=False):
    torch.manual_seed(198)
    return nn.Linear(64, 96, bias=bias).eval()


def native():
    torch.manual_seed(198)
    return SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=97,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_backend="eager",
        )
    ).eval()


@pytest.mark.parametrize("bias", [False, True])
def test_one_row_fastpath_and_prefill_fallback_preserve_values_and_parameters(bias):
    source = linear(bias)
    wrapped = CPUPrepackedLinear(source)
    wrapped.prepare_cpu_cache()
    assert wrapped.weight is source.weight
    assert wrapped.bias is source.bias
    with torch.inference_mode():
        for shape in [(1, 64), (1, 1, 64), (2, 4, 64)]:
            x = torch.randn(shape)
            torch.testing.assert_close(wrapped(x), source(x), atol=1e-5, rtol=2e-4)
    assert wrapped.cpu_prepack_hits == 2  # Multi-token prefill is unchanged.
    assert set(wrapped.state_dict()) == set(source.state_dict())
    for name, weight in wrapped.state_dict().items():
        assert torch.equal(weight, source.state_dict()[name])


def test_parameter_changes_invalidate_without_reallocating_during_request():
    source = linear()
    wrapped = CPUPrepackedLinear(source)
    wrapped.prepare_cpu_cache()
    with torch.no_grad():
        source.weight.add_(0.25)
        x = torch.randn(1, 64)
        assert torch.equal(wrapped(x), source(x))
    assert wrapped._cpu_packed_weight is None
    assert wrapped.cpu_prepack_hits == 0


def test_global_backend_disable_keeps_dense_values_and_reports_actual_status():
    source = linear()
    wrapped = CPUPrepackedLinear(source)
    wrapped.prepare_cpu_cache()
    container = nn.Sequential(wrapped)
    stored_bytes = cpu_prepack_status(container)["cache_bytes"]
    with torch.backends.mkldnn.flags(enabled=False), torch.inference_mode():
        x = torch.randn(1, 64)
        assert torch.equal(wrapped(x), source(x))
        status = cpu_prepack_status(container)
        assert status["active_caches"] == 0
        assert status["backend"] == "torch"
        assert status["cache_bytes"] == stored_bytes
    assert cpu_prepack_status(container)["active_caches"] == 1


def test_benchmark_ab_switch_retains_parameters_cache_and_restores_on_error():
    from saddlellm.models.CPULinear import disable_cpu_prepack

    wrapped = CPUPrepackedLinear(linear())
    wrapped.prepare_cpu_cache()
    model = nn.Sequential(wrapped)
    parameter, cache = wrapped.weight, wrapped._cpu_packed_weight
    with pytest.raises(RuntimeError, match="body failed"):
        with disable_cpu_prepack(model), torch.inference_mode():
            assert cpu_prepack_status(model)["active_caches"] == 0
            x = torch.randn(1, 64)
            assert torch.equal(wrapped(x), nn.functional.linear(x, wrapped.weight))
            raise RuntimeError("body failed")
    assert cpu_prepack_status(model)["active_caches"] == 1
    assert wrapped.weight is parameter and wrapped._cpu_packed_weight is cache


def test_global_hooks_and_root_linear_are_never_replaced():
    source = linear()
    assert prepare_cpu_linears(source)["prepared"] == 0
    assert not source._modules
    model = native()
    hook = nn.modules.module.register_module_forward_hook(lambda *args: None)
    try:
        report = optimize_native_inference(model, cpu_prepack=True)
        assert report.packed_qkv == report.packed_gate_up == 0
        assert report.cpu_prepack["prepared"] == 0
        assert "global" in report.cpu_prepack["skipped"]["<model>"]
    finally:
        hook.remove()


def test_gradients_and_training_use_canonical_dense_weights():
    source = linear(True)
    wrapped = CPUPrepackedLinear(source)
    wrapped.prepare_cpu_cache()
    x = torch.randn(1, 64, requires_grad=True)
    wrapped(x).sum().backward()
    assert source.weight.grad is not None
    assert source.bias.grad is not None
    assert x.grad is not None
    assert wrapped.cpu_prepack_hits == 0
    wrapped.train()
    assert wrapped._cpu_packed_weight is None
    assert wrapped.weight is source.weight


def test_dtype_change_and_state_loading_clear_opaque_caches():
    wrapped = CPUPrepackedLinear(linear()).eval()
    wrapped.prepare_cpu_cache()
    state = {k: v.clone() for k, v in wrapped.state_dict().items()}
    wrapped.load_state_dict(state)
    assert wrapped._cpu_packed_weight is None
    wrapped.prepare_cpu_cache()
    wrapped.double()
    assert wrapped._cpu_packed_weight is None
    with torch.inference_mode():
        x = torch.randn(1, 64, dtype=torch.float64)
        assert torch.equal(wrapped(x), nn.functional.linear(x, wrapped.weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_transfer_clears_opaque_cpu_cache_and_keeps_dense_path():
    wrapped = CPUPrepackedLinear(linear()).eval()
    wrapped.prepare_cpu_cache()
    wrapped.to("cuda")
    assert wrapped._cpu_packed_weight is None
    with torch.inference_mode():
        x = torch.randn(1, 64, device="cuda")
        assert torch.equal(wrapped(x), nn.functional.linear(x, wrapped.weight))


def test_hooks_subclasses_and_budget_are_not_silently_overridden():
    class Adapter(nn.Linear):
        pass

    model = nn.ModuleDict(
        {"plain": linear(), "hooked": linear(), "adapter": Adapter(64, 96)}
    )
    model.eval()
    model["hooked"].register_forward_hook(lambda *args: None)
    report = prepare_cpu_linears(model, max_prepack_bytes=0)
    assert report["prepared"] == 0
    assert type(model["plain"]) is nn.Linear
    assert "budget" in report["skipped"]["plain"]
    assert "unhooked" in report["skipped"]["hooked"]
    report = prepare_cpu_linears(model, max_prepack_bytes=96 * 64 * 4)
    assert report["prepared"] == 1
    assert type(model["hooked"]) is nn.Linear
    assert type(model["adapter"]) is Adapter


def test_unavailable_backend_or_unknown_memory_retains_torch_path(monkeypatch):
    model = nn.Sequential(linear())
    model.eval()
    monkeypatch.setattr(CPULinear, "available_cpu_memory", lambda: None)
    assert prepare_cpu_linears(model)["prepared"] == 0
    monkeypatch.setattr(CPULinear, "available_cpu_memory", lambda: 8 * 1024**3)
    monkeypatch.setattr(
        CPULinear, "cpu_prepack_supported", lambda dtype: (False, "unavailable")
    )
    report = prepare_cpu_linears(model)
    assert report["prepared"] == 0
    assert report["skipped"]["0"] == "unavailable"


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_budget_fails_before_mutating_model(budget):
    model = native()
    with pytest.raises(ValueError, match="non-negative"):
        optimize_native_inference(model, cpu_prepack=True, max_cpu_prepack_bytes=budget)
    assert model.layers[0].self_attn.qkv_packed_proj is None


def test_native_state_keys_tied_weights_and_restore_remain_compatible(tmp_path):
    model = native()
    reference = copy.deepcopy(model)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    report = optimize_native_inference(model, cpu_prepack=True)
    assert report.cpu_prepack["prepared"] == 9
    assert report.cpu_prepack["skipped"] == {}
    assert set(model.state_dict()) == set(original)
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
    prompt = torch.tensor([[4, 6, 8, 10]])
    with torch.inference_mode():
        torch.testing.assert_close(model(prompt).logits, reference(prompt).logits)
        assert torch.equal(
            model.generate(prompt, do_sample=False, max_new_tokens=6, eos_token_id=[]),
            reference.generate(
                prompt, do_sample=False, max_new_tokens=6, eos_token_id=[]
            ),
        )
    status = cpu_prepack_status(model)
    assert status["active_caches"] == status["wrapped_linears"] == 9
    assert status["decode_calls"] > 0
    assert status["cache_bytes"] == report.cpu_prepack["cache_bytes"]
    assert (
        optimize_native_inference(model, cpu_prepack=True).cpu_prepack[
            "already_prepared"
        ]
        == 9
    )
    model.save_pretrained(tmp_path)
    restored = SaddleForCausalLM.from_pretrained_saddle(tmp_path).eval()
    assert cpu_prepack_status(restored)["wrapped_linears"] == 0
    with torch.inference_mode():
        torch.testing.assert_close(restored(prompt).logits, model(prompt).logits)


def test_bfloat16_probe_reports_actual_hardware_support():
    supported, reason = cpu_prepack_supported(torch.bfloat16)
    if supported:
        source = linear().to(torch.bfloat16)
        wrapped = CPUPrepackedLinear(source)
        wrapped.prepare_cpu_cache()
        with torch.inference_mode():
            x = torch.randn(1, 64, dtype=torch.bfloat16)
            torch.testing.assert_close(wrapped(x), source(x), atol=0.004, rtol=0.04)
        assert wrapped.cpu_prepack_hits == 1
    else:
        assert reason
    assert cpu_prepack_supported(torch.float64)[0] is False


def test_prefix_entries_invalidate_when_cpu_execution_cache_changes():
    from saddlellm.models.NativePrefixCache import NativePrefixCache

    model = native()
    optimize_native_inference(model, cpu_prepack=True)
    cache = NativePrefixCache(1024**2, min_prefix_tokens=4)
    prompt = torch.tensor([[4, 6, 8, 10]])
    arguments = dict(
        do_sample=False,
        max_new_tokens=4,
        eos_token_id=[],
        prefix_cache=cache,
    )
    first = model.generate(prompt, **arguments)
    assert torch.equal(model.generate(prompt, **arguments), first)
    assert cache.snapshot()["hits"] == 1
    model.lm_head.clear_cpu_cache()
    assert torch.equal(model.generate(prompt, **arguments), first)
    assert cache.snapshot()["invalidations"] >= 1
    assert cache.snapshot()["hits"] == 1
    assert torch.equal(model.generate(prompt, **arguments), first)
    assert cache.snapshot()["hits"] == 2


def test_cli_and_settings_prepack_loading_option(monkeypatch):
    import uvicorn

    from saddlellm import cli
    from saddlellm.runtime import InferenceServer

    settings = []
    monkeypatch.setattr(
        InferenceServer, "create_inference_app", lambda s: settings.append(s)
    )
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
    assert (
        cli.main(
            [
                "serve",
                "model",
                "--backend",
                "native",
                "--device",
                "cpu",
                "--cpu-prepack",
            ]
        )
        == 0
    )
    assert settings[0].cpu_prepack_linear is True
    with pytest.raises(ValueError, match="backend=native"):
        InferenceServer.InferenceServerSettings(cpu_prepack_linear=True, device="cpu")
    with pytest.raises(ValueError, match="device=cpu"):
        InferenceServer.InferenceServerSettings(
            cpu_prepack_linear=True, backend="native"
        )
    with pytest.raises(ValueError, match="boolean"):
        InferenceServer.InferenceServerSettings(cpu_prepack_linear="yes")
