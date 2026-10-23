import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from saddlellm import optimize_native_inference
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def _model(**kwargs):
    return SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=41,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_backend="eager",
            **kwargs,
        )
    ).eval()


def test_packing_preserves_logits_weights_generation_and_is_idempotent(tmp_path):
    torch.manual_seed(193)
    model = _model(tie_word_embeddings=True)
    reference = copy.deepcopy(model)
    config = copy.deepcopy(model.config.to_dict())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    prompt = torch.tensor([[4, 6, 8, 10], [5, 7, 9, 11]])
    report = optimize_native_inference(model)

    assert report.packed_qkv == report.packed_gate_up == 2
    assert report.skipped == {}
    assert model.config.to_dict() == config
    assert model.layers[0].self_attn.q_proj is None
    assert model.layers[0].mlp.gate_proj is None
    assert set(model.state_dict()) == set(state)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    assert model.lm_head.weight is model.embed_tokens.weight
    with torch.no_grad():
        torch.testing.assert_close(model(prompt).logits, reference(prompt).logits)
    assert torch.equal(
        model.generate(prompt, max_new_tokens=5, do_sample=False, eos_token_id=[]),
        reference.generate(prompt, max_new_tokens=5, do_sample=False, eos_token_id=[]),
    )
    packed_parameter = model.layers[0].self_attn.qkv_packed_proj.weight
    repeated = optimize_native_inference(model)
    assert repeated.already_packed == 4
    assert repeated.packed_qkv == repeated.packed_gate_up == 0
    assert model.layers[0].self_attn.qkv_packed_proj.weight is packed_parameter

    model.save_pretrained(tmp_path)
    restored = SaddleForCausalLM.from_pretrained_saddle(tmp_path).eval()
    assert restored.layers[0].self_attn.qkv_packed_proj is None
    with torch.no_grad():
        torch.testing.assert_close(restored(prompt).logits, model(prompt).logits)
    # We keep normal tensors even when optimization is invoked in inference mode.
    restored.train()
    restored(prompt, labels=prompt).loss.backward()
    assert restored.layers[0].self_attn.q_proj.weight.grad is not None


def test_packing_requires_eval_and_native_model():
    with pytest.raises(TypeError, match="native"):
        optimize_native_inference(nn.Linear(4, 4))
    model = _model().train()
    with pytest.raises(ValueError, match="eval"):
        optimize_native_inference(model)
    model.eval().layers[1].self_attn.train()
    with pytest.raises(ValueError, match="eval"):
        optimize_native_inference(model)
    assert model.layers[0].self_attn.qkv_packed_proj is None


def test_selective_gate_up_packing_preserves_qkv_and_output():
    torch.manual_seed(194)
    model = _model()
    reference = copy.deepcopy(model)
    prompt = torch.tensor([[4, 6, 8, 10]])

    report = optimize_native_inference(model, pack_qkv=False)

    assert report.packed_qkv == 0
    assert report.packed_gate_up == 2
    assert len(report.skipped) == 2
    assert all("numerical fidelity" in reason for reason in report.skipped.values())
    assert all(layer.self_attn.qkv_packed_proj is None for layer in model.layers)
    assert all(layer.mlp.gate_up_packed_proj is not None for layer in model.layers)
    with torch.no_grad():
        torch.testing.assert_close(model(prompt).logits, reference(prompt).logits)
    with pytest.raises(ValueError, match="pack_qkv"):
        optimize_native_inference(_model(), pack_qkv="no")


def test_inference_mode_packing_does_not_create_inference_parameters():
    model = _model()
    with torch.inference_mode():
        optimize_native_inference(model)
    assert not model.layers[0].self_attn.qkv_packed_proj.weight.is_inference()
    assert not model.layers[0].mlp.gate_up_packed_proj.weight.is_inference()


def test_hooks_and_linear_subclasses_are_not_discarded():
    model = _model()
    attention = model.layers[0].self_attn
    called = []
    attention.q_proj.register_forward_hook(lambda *args: called.append(True))

    class AdapterLinear(nn.Linear):
        pass

    mlp = model.layers[1].mlp
    adapter = AdapterLinear(16, 32, bias=False).eval()
    adapter.load_state_dict(mlp.gate_proj.state_dict())
    mlp.gate_proj = adapter
    report = optimize_native_inference(model)
    assert report.packed_qkv == report.packed_gate_up == 1
    assert len(report.skipped) == 2
    assert attention.qkv_packed_proj is None
    assert mlp.gate_up_packed_proj is None
    model(torch.tensor([[4, 5, 6]]))
    assert called


def test_mla_is_skipped_but_moe_experts_are_packed():
    model = _model(
        attention_kind="mla",
        kv_lora_rank=4,
        q_lora_rank=4,
        ffn_kind="moe",
        num_experts=3,
        num_experts_per_tok=1,
        expert_intermediate_size=24,
        shared_expert=True,
    )
    reference = copy.deepcopy(model)
    report = optimize_native_inference(model)
    assert report.packed_qkv == 0
    assert report.packed_gate_up == 8
    assert len(report.skipped) == 2
    with torch.no_grad():
        ids = torch.tensor([[4, 6, 7]])
        torch.testing.assert_close(model(ids).logits, reference(ids).logits)


@pytest.mark.parametrize("learnable", [False, True])
def test_unit_residual_scale_retains_logits_and_learnable_gradients(learnable):
    model = _model(residual_learnable=learnable).train()
    ids = torch.tensor([[4, 5, 6, 7]])
    model(ids, labels=ids).loss.backward()
    if learnable:
        layer = model.layers[0]
        assert layer.attention_residual_scale.grad is not None
        assert layer.ffn_residual_scale.grad is not None
        assert torch.isfinite(layer.attention_residual_scale.grad)


def test_serve_model_loader_packs_by_default_and_can_disable(tmp_path, monkeypatch):
    from saddlellm.models import TokenizerLoader
    from saddlellm.runtime.InferenceServer import (
        InferenceServerSettings,
        load_inference_model,
    )

    class Tokenizer:
        pad_token_id = 0

    monkeypatch.setattr(
        TokenizerLoader, "load_tokenizer_compatible", lambda *a, **kw: Tokenizer()
    )
    _model().save_pretrained(tmp_path)
    model, _ = load_inference_model(
        InferenceServerSettings(model_path=str(tmp_path), device="cpu")
    )
    assert model.layers[0].self_attn.qkv_packed_proj is not None
    model, _ = load_inference_model(
        InferenceServerSettings(
            model_path=str(tmp_path), device="cpu", fuse_projections=False
        )
    )
    assert model.layers[0].self_attn.qkv_packed_proj is None


def test_serve_cli_projection_option(monkeypatch):
    import uvicorn

    from saddlellm.cli import main
    from saddlellm.runtime import InferenceServer

    settings = []
    monkeypatch.setattr(
        InferenceServer, "create_inference_app", lambda value: settings.append(value)
    )
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    assert main(["serve", "model", "--no-fuse-projections"]) == 0
    assert settings[0].fuse_projections is False


def test_local_comparison_benchmark_cpu(tmp_path):
    _model().save_pretrained(tmp_path)
    script = (
        Path(__file__).resolve().parents[1] / "tools/benchmark_inference_compare.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(tmp_path),
            "--batch-size",
            "2",
            "--prompt-tokens",
            "4",
            "--decode-tokens",
            "3",
            "--repeats",
            "1",
            "--hf-reference",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=40,
    )
    report = json.loads(completed.stdout)
    assert report["checkpoint_sha256"]
    assert report["all_outputs_equal"]
    assert report["variants"]["native_packed"]["optimization"]["packed_qkv"] == 2
    assert report["variants"]["transformers_sdpa"]["outputs_equal"]
    assert report["variants"]["native_unpacked"]["median_generation_ms"] > 0
