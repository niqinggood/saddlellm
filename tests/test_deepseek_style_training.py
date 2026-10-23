"""Tiny native training checks, not a reproduction of DeepSeek's model quality."""

from __future__ import annotations

import pytest
import torch

from saddlellm.models.ModelBlueprint import ModelBlueprint, ModelBlueprintLab
from saddlellm.models.ModelLoader import load_causal_lm
from saddlellm.models.SaddleModeling import SaddleMoE


def _blueprint():
    blueprint = ModelBlueprintLab.deepseek_style_moe(
        hidden_size=32,
        layers=2,
        heads=4,
        kv_heads=2,
        experts=4,
        experts_per_token=2,
        seq_length=32,
    )
    blueprint.vocab_size = 37
    blueprint.attention.backend = "eager"
    return blueprint


def test_deepseek_style_preset_really_disables_router_auxiliary_loss():
    blueprint = _blueprint()
    assert blueprint.ffn.aux_loss_free
    assert blueprint.ffn.router_aux_loss_coef == 0.0
    assert blueprint.attention.kind == "mla"
    assert blueprint.ffn.shared_expert
    assert blueprint.objective.multi_token_prediction

    restored = ModelBlueprint.from_dict(blueprint.to_dict())
    model = restored.build_model()
    assert model.config.aux_loss_free
    assert model.config.router_aux_loss_coef == 0.0
    for layer in model.layers:
        assert isinstance(layer.mlp, SaddleMoE)
        assert layer.mlp.shared_expert is not None
        assert layer.mlp.router_aux_loss_coef == 0.0


@pytest.mark.parametrize("loss_chunk_size", [0, 5])
def test_tiny_native_mla_moe_mtp_train_save_reload(tmp_path, loss_chunk_size):
    torch.manual_seed(173)
    model = _blueprint().build_model().train()
    model.config.loss_chunk_size = loss_chunk_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(4, model.config.vocab_size, (2, 12))
    labels = input_ids.clone()
    labels[:, :2] = -100
    original_router = model.layers[0].mlp.router.weight.detach().clone()
    original_mtp = model.mtp_heads[0].weight.detach().clone()

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
        assert torch.isfinite(output.loss)
        assert output.loss_items["aux_loss"].item() == 0.0
        assert {"mtp_loss_2", "mtp_loss_3", "mtp_loss_4"}.issubset(output.loss_items)
        assert (output.logits is None) == bool(loss_chunk_size)
        for layer, stats in zip(model.layers, output.expert_stats):
            assert stats["router_aux_loss"].item() == 0.0
            assert stats["dropped_tokens"].item() == 0.0
            assert stats["tokens_per_expert"].sum().item() == (
                input_ids.numel() * layer.mlp.top_k
            )
        output.loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()

    assert not torch.equal(model.layers[0].mlp.router.weight, original_router)
    assert not torch.equal(model.mtp_heads[0].weight, original_mtp)
    assert any(torch.count_nonzero(layer.mlp.router_bias) for layer in model.layers)

    model.eval()
    checkpoint = tmp_path / "tiny_mla_moe_mtp"
    model.save_pretrained(str(checkpoint))
    restored = load_causal_lm(str(checkpoint), device="cpu").eval()
    assert restored.config.loss_chunk_size == loss_chunk_size
    assert restored.config.aux_loss_free
    assert restored.config.router_aux_loss_coef == 0.0
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name

    with torch.inference_mode():
        reference = model(input_ids=input_ids, use_cache=False).logits
        actual = restored(input_ids=input_ids, use_cache=False).logits
        assert torch.equal(reference, actual)
        prompt = input_ids[:1, :4]
        generation = dict(max_new_tokens=3, do_sample=False, eos_token_id=None)
        assert torch.equal(
            model.generate(prompt, **generation),
            restored.generate(prompt, **generation),
        )

    # Reloaded weights remain trainable; this is NOT exact optimizer/RNG resume.
    restored.train()
    next_output = restored(input_ids=input_ids, labels=labels, use_cache=False)
    assert next_output.loss_items["aux_loss"].item() == 0.0
    next_output.loss.backward()
    assert torch.isfinite(restored.layers[0].mlp.router.weight.grad).all()
