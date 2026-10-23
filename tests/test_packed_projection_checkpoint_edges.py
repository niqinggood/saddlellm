from collections import OrderedDict

import pytest
import torch

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def _tiny_config(
    *,
    fused_qkv: bool,
    fused_gate_up: bool,
    moe: bool = False,
) -> SaddleModelConfig:
    return SaddleModelConfig(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_kind="gqa",
        attention_backend="eager",
        fused_qkv=fused_qkv,
        ffn_kind="moe" if moe else "swiglu",
        fused_gate_up=fused_gate_up,
        num_experts=3 if moe else 0,
        num_experts_per_tok=1 if moe else 0,
        expert_intermediate_size=24 if moe else 0,
        shared_expert=moe,
        router_aux_loss_coef=0.01 if moe else 0.0,
    )


def _model(*, fused_qkv: bool, fused_gate_up: bool, moe: bool = False):
    return SaddleForCausalLM(
        _tiny_config(
            fused_qkv=fused_qkv,
            fused_gate_up=fused_gate_up,
            moe=moe,
        )
    )


def _pack_projection(
    state_dict: OrderedDict,
    *,
    prefix: str,
    packed_name: str,
    logical_names: tuple[str, ...],
) -> None:
    state_dict[prefix + packed_name] = torch.cat(
        [state_dict.pop(prefix + name) for name in logical_names], dim=0
    )


def _raw_packed_dense_state(model: SaddleForCausalLM) -> OrderedDict:
    state_dict = OrderedDict(
        (key, value.detach().clone()) for key, value in model.state_dict().items()
    )
    _pack_projection(
        state_dict,
        prefix="layers.0.self_attn.",
        packed_name="qkv_packed_proj.weight",
        logical_names=("q_proj.weight", "k_proj.weight", "v_proj.weight"),
    )
    _pack_projection(
        state_dict,
        prefix="layers.0.mlp.",
        packed_name="gate_up_packed_proj.weight",
        logical_names=("gate_proj.weight", "up_proj.weight"),
    )
    return state_dict


@pytest.mark.parametrize("target_fused", [False, True])
def test_raw_packed_state_dict_strictly_loads_into_both_layouts(target_fused):
    torch.manual_seed(701)
    reference = _model(fused_qkv=False, fused_gate_up=False).eval()
    raw_packed = _raw_packed_dense_state(reference)
    target = _model(
        fused_qkv=target_fused,
        fused_gate_up=target_fused,
    ).eval()

    incompatible = target.load_state_dict(raw_packed, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    input_ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual = target(input_ids=input_ids).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("target_fused", [False, True])
@pytest.mark.parametrize(
    ("projection", "legacy_names", "message"),
    [
        (
            "layers.0.self_attn.",
            ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
            "both packed and legacy Q/K/V projection weights",
        ),
        (
            "layers.0.mlp.",
            ("gate_proj.weight", "up_proj.weight"),
            "both packed and legacy gate/up projection weights",
        ),
    ],
)
def test_raw_packed_and_legacy_projection_conflict_fails_clearly(
    target_fused,
    projection,
    legacy_names,
    message,
):
    reference = _model(fused_qkv=False, fused_gate_up=False)
    canonical = reference.state_dict()
    conflicting = _raw_packed_dense_state(reference)
    for name in legacy_names:
        conflicting[projection + name] = canonical[projection + name].detach().clone()
    target = _model(
        fused_qkv=target_fused,
        fused_gate_up=target_fused,
    )

    with pytest.raises(RuntimeError, match=message):
        target.load_state_dict(conflicting, strict=True)


@pytest.mark.parametrize("target_fused", [False, True])
@pytest.mark.parametrize(
    "packed_key",
    [
        "layers.0.self_attn.qkv_packed_proj.weight",
        "layers.0.mlp.gate_up_packed_proj.weight",
    ],
)
def test_malformed_raw_packed_projection_shape_fails_clearly(
    target_fused,
    packed_key,
):
    reference = _model(fused_qkv=False, fused_gate_up=False)
    malformed = _raw_packed_dense_state(reference)
    malformed[packed_key] = malformed[packed_key][:-1].clone()
    target = _model(
        fused_qkv=target_fused,
        fused_gate_up=target_fused,
    )

    with pytest.raises(RuntimeError, match=rf"size mismatch.*{packed_key}"):
        target.load_state_dict(malformed, strict=True)


def test_fused_gate_up_covers_moe_routed_and_shared_experts_with_roundtrip():
    torch.manual_seed(709)
    fused = _model(fused_qkv=False, fused_gate_up=True, moe=True).eval()
    moe = fused.layers[0].mlp

    assert len(moe.experts) == 3
    assert all(expert.gate_up_packed_proj is not None for expert in moe.experts)
    assert all(expert.gate_proj is expert.up_proj is None for expert in moe.experts)
    assert moe.shared_expert is not None
    assert moe.shared_expert.gate_up_packed_proj is not None
    assert moe.shared_expert.gate_proj is moe.shared_expert.up_proj is None

    canonical = fused.state_dict()
    assert not any("gate_up_packed_proj" in key for key in canonical)
    for expert_index in range(3):
        prefix = f"layers.0.mlp.experts.{expert_index}."
        assert prefix + "gate_proj.weight" in canonical
        assert prefix + "up_proj.weight" in canonical
    assert "layers.0.mlp.shared_expert.gate_proj.weight" in canonical
    assert "layers.0.mlp.shared_expert.up_proj.weight" in canonical

    legacy = _model(fused_qkv=False, fused_gate_up=False, moe=True).eval()
    incompatible = legacy.load_state_dict(canonical, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    with torch.no_grad():
        fused_logits = fused(input_ids=input_ids).logits
        legacy_logits = legacy(input_ids=input_ids).logits
    assert torch.allclose(fused_logits, legacy_logits, atol=1e-6, rtol=1e-5)

    fused.train()
    output = fused(input_ids=input_ids, labels=input_ids)
    assert torch.isfinite(output.loss)
    output.loss.backward()

    packed_parameters = [
        expert.gate_up_packed_proj.weight for expert in moe.experts
    ] + [moe.shared_expert.gate_up_packed_proj.weight]
    assert all(parameter.grad is not None for parameter in packed_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in packed_parameters)
