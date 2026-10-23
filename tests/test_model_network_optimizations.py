import pytest
import torch
from torch import nn

from saddlellm.models.ModelBlueprint import (
    AttentionBlueprint,
    FFNBlueprint,
    LayerBlueprint,
    ModelBlueprint,
    ModelBlueprintLab,
)
from saddlellm.models.SaddleModeling import (
    SaddleAttention,
    SaddleForCausalLM,
    SaddleModelConfig,
    SaddleRMSNorm,
)
from saddlellm.models.ModelRegistry import MODEL_SPECS
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def test_rms_norm_uses_stable_fp32_reduction_for_half_inputs():
    norm = SaddleRMSNorm(4).to(dtype=torch.float16)
    values = torch.tensor(
        [[300.0, -400.0, 500.0, -600.0]],
        dtype=torch.float16,
        requires_grad=True,
    )

    output = norm(values)
    reference_values = values.detach().float()
    reference = reference_values * torch.rsqrt(
        reference_values.square().mean(dim=-1, keepdim=True) + norm.eps
    )

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, rtol=2e-3, atol=2e-3)

    output.float().sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


def test_rms_norm_preserves_float64_reference_precision():
    norm = SaddleRMSNorm(3).double()
    values = torch.tensor(
        [[1.0, -2.0, 4.0], [0.25, 0.5, -0.75]], dtype=torch.float64
    )
    expected = values * torch.rsqrt(
        values.square().mean(dim=-1, keepdim=True) + norm.eps
    )

    output = norm(values)

    assert output.dtype == torch.float64
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)


def _attention(
    *,
    kind: str = "gqa",
    kv_heads: int = 2,
    qk_norm: bool = False,
    sliding_window: int = 0,
    rope_scaling=None,
    fused_qkv: bool = False,
) -> AttentionBlueprint:
    return AttentionBlueprint(
        kind=kind,
        backend="eager",
        num_heads=4,
        num_kv_heads=kv_heads,
        qk_norm=qk_norm,
        rope_scaling=rope_scaling,
        sliding_window=sliding_window,
        fused_qkv=fused_qkv,
    )


def _tiny_blueprint(
    name: str,
    *,
    attention: AttentionBlueprint,
    layers=None,
    num_layers: int = 2,
    max_position_embeddings: int = 32,
    fused_gate_up: bool = False,
) -> ModelBlueprint:
    return ModelBlueprint(
        name=name,
        hidden_size=16,
        num_layers=num_layers,
        vocab_size=41,
        max_position_embeddings=max_position_embeddings,
        attention=attention,
        ffn=FFNBlueprint(
            kind="swiglu",
            intermediate_size=32,
            fused_gate_up=fused_gate_up,
        ),
        layers=list(layers or []),
    )


def test_qk_norm_propagates_to_config_layers_and_backward(tmp_path):
    blueprint = _tiny_blueprint(
        "qk-norm-mixed",
        attention=_attention(qk_norm=True),
        layers=[
            LayerBlueprint(name="normalized"),
            LayerBlueprint(
                name="legacy",
                attention=_attention(qk_norm=False),
            ),
        ],
    )
    config = SaddleModelConfig.from_blueprint(blueprint)

    assert blueprint.attention.qk_norm is True
    assert config.qk_norm is True
    assert config.for_layer(0).qk_norm is True
    assert config.for_layer(1).qk_norm is False
    assert blueprint.architecture_axes()["qk_norm"] == "hybrid"

    model = SaddleForCausalLM(config).train()
    normalized = model.layers[0].self_attn
    legacy = model.layers[1].self_attn
    assert isinstance(normalized.q_norm, SaddleRMSNorm)
    assert isinstance(normalized.k_norm, SaddleRMSNorm)
    assert normalized.q_norm.weight.shape == (normalized.head_dim,)
    assert normalized.k_norm.weight.shape == (normalized.head_dim,)
    assert isinstance(legacy.q_norm, nn.Identity)
    assert isinstance(legacy.k_norm, nn.Identity)

    state_keys = set(model.state_dict())
    assert "layers.0.self_attn.q_norm.weight" in state_keys
    assert "layers.0.self_attn.k_norm.weight" in state_keys
    assert not any(key.startswith("layers.1.self_attn.q_norm.") for key in state_keys)
    assert not any(key.startswith("layers.1.self_attn.k_norm.") for key in state_keys)

    torch.manual_seed(23)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    output = model(input_ids=input_ids, labels=input_ids)
    assert output.logits.shape == (2, 6, config.vocab_size)
    assert torch.isfinite(output.loss)
    output.loss.backward()

    for parameter in (normalized.q_norm.weight, normalized.k_norm.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    checkpoint = tmp_path / "qk-norm-checkpoint"
    model.save_pretrained(checkpoint)
    restored = SaddleForCausalLM.from_pretrained_saddle(
        checkpoint, map_location="cpu"
    )
    assert restored.config.qk_norm is True
    assert isinstance(restored.layers[0].self_attn.q_norm, SaddleRMSNorm)
    assert isinstance(restored.layers[1].self_attn.q_norm, nn.Identity)
    restored.load_state_dict(model.state_dict(), strict=True)


def test_qk_norm_round_trips_through_model_spec_and_parameter_estimates():
    disabled = _tiny_blueprint(
        "qk-norm-estimate-off",
        attention=_attention(qk_norm=False),
        num_layers=3,
    )
    enabled = _tiny_blueprint(
        "qk-norm-estimate-on",
        attention=_attention(qk_norm=True),
        num_layers=3,
    )

    disabled_spec = disabled.to_model_spec()
    enabled_spec = enabled.to_model_spec()
    expected_extra_parameters = 3 * 2 * (enabled.hidden_size // 4)

    assert enabled_spec.qk_norm is True
    assert enabled_spec.to_dict()["qk_norm"] is True
    assert SaddleModelConfig.from_model_spec(enabled_spec).qk_norm is True
    assert (
        enabled.estimate_total_params() - disabled.estimate_total_params()
        == expected_extra_parameters
    )
    assert (
        enabled_spec.estimated_params - disabled_spec.estimated_params
        == expected_extra_parameters
    )


def test_qk_norm_default_does_not_add_state_dict_parameter_keys():
    blueprint = _tiny_blueprint(
        "qk-norm-default-off",
        attention=_attention(),
    )
    config = SaddleModelConfig.from_blueprint(blueprint)
    model = SaddleForCausalLM(config)

    assert blueprint.attention.qk_norm is False
    assert config.qk_norm is False
    assert all(config.for_layer(index).qk_norm is False for index in range(2))
    assert all(isinstance(layer.self_attn.q_norm, nn.Identity) for layer in model.layers)
    assert all(isinstance(layer.self_attn.k_norm, nn.Identity) for layer in model.layers)
    assert not any(".q_norm." in key or ".k_norm." in key for key in model.state_dict())


def test_rope_cache_is_shared_sliced_grown_and_isolated_by_scaling():
    blueprint = _tiny_blueprint(
        "shared-rope-cache",
        attention=_attention(),
        max_position_embeddings=4,
        layers=[
            LayerBlueprint(name="base-a"),
            LayerBlueprint(name="base-b"),
            LayerBlueprint(
                name="scaled",
                attention=_attention(
                    rope_scaling={"type": "linear", "factor": 2.0}
                ),
            ),
        ],
    )
    model = blueprint.build_model()
    base_a = model.layers[0].self_attn.rotary
    base_b = model.layers[1].self_attn.rotary
    scaled = model.layers[2].self_attn.rotary

    assert base_a._cache is model._rope_cache
    assert base_b._cache is model._rope_cache
    assert scaled._cache is model._rope_cache

    base_cos, base_sin = base_a(4, dtype=torch.float32)
    sliced_cos, sliced_sin = base_b(2, dtype=torch.float32, offset=1)
    assert len(model._rope_cache) == 1
    assert torch.equal(sliced_cos, base_cos[1:3])
    assert torch.equal(sliced_sin, base_sin[1:3])

    prefix_cos = base_cos.clone()
    prefix_sin = base_sin.clone()
    grown_cos, grown_sin = base_b(4, dtype=torch.float32, offset=2)
    assert grown_cos.shape == grown_sin.shape == (4, base_b.dim)
    cached_cos, cached_sin = next(iter(model._rope_cache.values()))
    assert cached_cos.shape[0] == cached_sin.shape[0]
    assert cached_cos.shape[0] >= 6
    assert torch.equal(cached_cos[:4], prefix_cos)
    assert torch.equal(cached_sin[:4], prefix_sin)

    scaled_cos, scaled_sin = scaled(6, dtype=torch.float32)
    assert len(model._rope_cache) == 2
    assert {key[2] for key in model._rope_cache} == {1.0, 2.0}
    assert not torch.allclose(scaled_cos[1], cached_cos[1])
    assert not torch.allclose(scaled_sin[1], cached_sin[1])


def test_minimax_style_long_context_builds_and_runs_tiny_forward():
    blueprint = ModelBlueprintLab.minimax_style_long_context(
        hidden_size=16,
        layers=2,
        heads=4,
        seq_length=8192,
    )
    blueprint.vocab_size = 43
    model = blueprint.build_model().eval()

    assert model.config.max_position_embeddings == 8192
    assert model.config.rope_scaling == {"type": "linear", "factor": 2}
    input_ids = torch.randint(0, model.config.vocab_size, (1, 7))
    with torch.no_grad():
        output = model(input_ids=input_ids)

    assert output.logits.shape == (1, 7, model.config.vocab_size)
    assert torch.isfinite(output.logits).all()


@pytest.mark.parametrize(
    ("attention_kind", "kv_heads"),
    [("gqa", 2), ("mqa", 1)],
)
@pytest.mark.parametrize("sliding_window", [0, 3])
def test_cached_decode_last_token_matches_full_forward_with_padding_and_window(
    attention_kind,
    kv_heads,
    sliding_window,
):
    torch.manual_seed(31)
    blueprint = _tiny_blueprint(
        f"cached-{attention_kind}-window-{sliding_window}",
        attention=_attention(
            kind=attention_kind,
            kv_heads=kv_heads,
            sliding_window=sliding_window,
        ),
    )
    model = blueprint.build_model().eval()
    input_ids = torch.tensor(
        [
            [0, 0, 5, 7, 11, 13],
            [0, 3, 17, 19, 23, 29],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        full_last = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits[:, -1]
        prefix = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=True,
        )
        cached_last = model(
            input_ids=input_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits[:, -1]

    assert len(prefix.past_key_values) == len(model.layers)
    for key, value in prefix.past_key_values:
        assert key.shape[1] == value.shape[1] == kv_heads
        assert key.shape[2] == value.shape[2] == input_ids.shape[1] - 1
    assert torch.allclose(full_last, cached_last, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize(
    ("attention_kind", "kv_heads"),
    [("mha", 4), ("gqa", 2), ("mqa", 1)],
)
def test_packed_projections_are_strictly_checkpoint_compatible_and_equivalent(
    attention_kind,
    kv_heads,
):
    legacy_blueprint = _tiny_blueprint(
        f"legacy-{attention_kind}",
        attention=_attention(
            kind=attention_kind,
            kv_heads=kv_heads,
            qk_norm=True,
        ),
        num_layers=1,
    )
    fused_blueprint = _tiny_blueprint(
        f"fused-{attention_kind}",
        attention=_attention(
            kind=attention_kind,
            kv_heads=kv_heads,
            qk_norm=True,
            fused_qkv=True,
        ),
        num_layers=1,
        fused_gate_up=True,
    )
    torch.manual_seed(101)
    legacy = legacy_blueprint.build_model().eval()
    torch.manual_seed(202)
    fused = fused_blueprint.build_model().eval()

    incompatible = fused.load_state_dict(legacy.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert sum(parameter.numel() for parameter in legacy.parameters()) == sum(
        parameter.numel() for parameter in fused.parameters()
    )

    attention = fused.layers[0].self_attn
    mlp = fused.layers[0].mlp
    assert attention.qkv_packed_proj is not None
    assert attention.q_proj is attention.k_proj is attention.v_proj is None
    assert mlp.gate_up_packed_proj is not None
    assert mlp.gate_proj is mlp.up_proj is None
    # Stable checkpoint keys stay logical, so old readers do not need to know
    # about the in-memory packed layout.
    canonical_state = fused.state_dict()
    assert not any("packed_proj" in key for key in canonical_state)
    assert "layers.0.self_attn.q_proj.weight" in canonical_state
    assert "layers.0.mlp.gate_proj.weight" in canonical_state

    input_ids = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    legacy_logits = legacy(input_ids=input_ids).logits
    fused_logits = fused(input_ids=input_ids).logits
    assert torch.allclose(legacy_logits, fused_logits, atol=1e-6, rtol=1e-5)

    legacy_logits.square().mean().backward()
    fused_logits.square().mean().backward()
    q_size = attention.q_projection_size
    kv_size = attention.kv_projection_size
    q_grad, k_grad, v_grad = attention.qkv_packed_proj.weight.grad.split(
        (q_size, kv_size, kv_size), dim=0
    )
    assert torch.allclose(q_grad, legacy.layers[0].self_attn.q_proj.weight.grad)
    assert torch.allclose(k_grad, legacy.layers[0].self_attn.k_proj.weight.grad)
    assert torch.allclose(v_grad, legacy.layers[0].self_attn.v_proj.weight.grad)
    gate_grad, up_grad = mlp.gate_up_packed_proj.weight.grad.chunk(2, dim=0)
    assert torch.allclose(gate_grad, legacy.layers[0].mlp.gate_proj.weight.grad)
    assert torch.allclose(up_grad, legacy.layers[0].mlp.up_proj.weight.grad)

    restored_legacy = legacy_blueprint.build_model().eval()
    restored_legacy.load_state_dict(canonical_state, strict=True)
    assert torch.allclose(
        restored_legacy(input_ids=input_ids).logits,
        fused_logits.detach(),
        atol=1e-6,
        rtol=1e-5,
    )


def test_optimized_dense_preset_and_fusion_validation():
    blueprint = ModelBlueprintLab.optimized_dense_gqa(
        hidden_size=16,
        layers=2,
        heads=4,
        kv_heads=2,
        vocab_size=41,
        seq_length=64,
    )
    model = blueprint.build_model()

    assert blueprint.attention.fused_qkv is True
    assert blueprint.ffn.fused_gate_up is True
    assert blueprint.attention.qk_norm is True
    assert blueprint.residual.initialization == "depth_scaled"
    assert model.config.fused_qkv is True
    assert model.config.fused_gate_up is True
    assert all(layer.self_attn.qkv_packed_proj is not None for layer in model.layers)
    assert all(layer.mlp.gate_up_packed_proj is not None for layer in model.layers)

    raw = {
        "name": "invalid-mla-fusion",
        "hidden_size": 16,
        "num_layers": 1,
        "vocab_size": 41,
        "attention": {
            "kind": "mla",
            "num_heads": 4,
            "num_kv_heads": 2,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "fused_qkv": True,
        },
        "ffn": {"kind": "swiglu", "intermediate_size": 32},
    }
    with pytest.raises(ValueError, match="fused_qkv is not supported for MLA"):
        ModelBlueprint.from_dict(raw)


def test_packed_projection_names_do_not_collide_with_legacy_lora_suffixes():
    model = ModelBlueprintLab.optimized_dense_gqa(
        hidden_size=16,
        layers=1,
        heads=4,
        kv_heads=2,
        vocab_size=41,
        seq_length=32,
    ).build_model()
    packed_names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "packed" in name
    ]

    assert packed_names == [
        "layers.0.self_attn.qkv_packed_proj",
        "layers.0.mlp.gate_up_packed_proj",
    ]
    legacy_targets = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
    assert all(
        not any(name.endswith(target) for target in legacy_targets)
        for name in packed_names
    )


def test_optimized_registry_preset_survives_blueprint_resolution():
    spec = MODEL_SPECS["saddle-gqa-300m-optimized"]
    config = SaddleModelConfig.from_model_spec(spec)
    blueprint = TrainingOrchestrator.resolve_architecture_blueprint(
        spec.name,
        {"name": "custom-optimized-run", "max_position_embeddings": 8192},
    )

    assert config.qk_norm is config.fused_qkv is config.fused_gate_up is True
    assert blueprint.name == "custom-optimized-run"
    assert blueprint.max_position_embeddings == 8192
    assert blueprint.attention.qk_norm is True
    assert blueprint.attention.fused_qkv is True
    assert blueprint.ffn.fused_gate_up is True


def _forward_with_per_layer_attention_masks(
    model,
    input_ids,
    attention_mask,
    *,
    past_key_values=None,
    use_cache=False,
):
    """Reference path that deliberately lets every layer rebuild its mask."""

    hidden_states = model.embed_tokens(input_ids)
    if past_key_values is None:
        past_key_values = [None] * len(model.layers)
    next_past_key_values = []
    for layer, past_key_value in zip(model.layers, past_key_values):
        hidden_states, _, _, present_key_value = layer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            prepared_attention=None,
        )
        if use_cache:
            next_past_key_values.append(present_key_value)
    logits = model.lm_head(model.norm(hidden_states))
    return logits, next_past_key_values


def _install_attention_mask_counter(monkeypatch):
    original = SaddleAttention._build_attention_mask
    calls = []

    def counted(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        SaddleAttention,
        "_build_attention_mask",
        staticmethod(counted),
    )
    return calls


def test_attention_mask_is_built_once_per_forward_and_matches_per_layer_reference(
    monkeypatch,
):
    torch.manual_seed(307)
    blueprint = _tiny_blueprint(
        "shared-padding-mask",
        attention=_attention(sliding_window=0),
        num_layers=4,
    )
    shared = blueprint.build_model().train()
    reference = blueprint.build_model().train()
    reference.load_state_dict(shared.state_dict(), strict=True)
    input_ids = torch.tensor(
        [[0, 5, 7, 11, 13, 17], [0, 0, 3, 19, 23, 29]],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [[0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]],
        dtype=torch.long,
    )
    calls = _install_attention_mask_counter(monkeypatch)

    shared_logits = shared(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits
    assert len(calls) == 1
    assert calls[0]["q_len"] == calls[0]["kv_len"] == input_ids.shape[1]
    assert calls[0]["past_len"] == 0
    assert calls[0]["sliding_window"] == 0
    shared_logits.square().mean().backward()

    calls.clear()
    reference_logits, _ = _forward_with_per_layer_attention_masks(
        reference,
        input_ids,
        attention_mask,
    )
    assert len(calls) == len(reference.layers)
    reference_logits.square().mean().backward()

    assert torch.allclose(shared_logits, reference_logits, atol=1e-6, rtol=1e-5)
    shared_parameters = dict(shared.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    assert shared_parameters.keys() == reference_parameters.keys()
    for name, parameter in shared_parameters.items():
        reference_gradient = reference_parameters[name].grad
        assert parameter.grad is not None, name
        assert reference_gradient is not None, name
        assert torch.allclose(
            parameter.grad,
            reference_gradient,
            atol=1e-7,
            rtol=1e-5,
        ), name


def test_attention_mask_is_built_once_for_each_heterogeneous_sliding_window(
    monkeypatch,
):
    blueprint = _tiny_blueprint(
        "heterogeneous-shared-padding-mask",
        attention=_attention(sliding_window=0),
        layers=[
            LayerBlueprint(
                name="global",
                repeat=2,
                attention=_attention(sliding_window=0),
            ),
            LayerBlueprint(
                name="local",
                repeat=2,
                attention=_attention(sliding_window=3),
            ),
        ],
    )
    model = blueprint.build_model().eval()
    input_ids = torch.tensor([[0, 2, 3, 5, 7, 11]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.long)
    calls = _install_attention_mask_counter(monkeypatch)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    assert len(calls) == 2
    assert sorted(call["sliding_window"] for call in calls) == [0, 3]
    assert all(call["q_len"] == call["kv_len"] == input_ids.shape[1] for call in calls)
    assert all(call["past_len"] == 0 for call in calls)


def test_attention_mask_is_reused_during_cached_decode(monkeypatch):
    torch.manual_seed(401)
    blueprint = _tiny_blueprint(
        "cached-shared-padding-mask",
        attention=_attention(sliding_window=3),
        num_layers=3,
    )
    model = blueprint.build_model().eval()
    input_ids = torch.tensor(
        [[0, 5, 7, 11, 13, 17], [0, 0, 3, 19, 23, 29]],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [[0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]],
        dtype=torch.long,
    )
    calls = _install_attention_mask_counter(monkeypatch)

    with torch.no_grad():
        prefix = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=True,
        )
    assert len(calls) == 1
    assert calls[0]["q_len"] == calls[0]["kv_len"] == input_ids.shape[1] - 1
    assert calls[0]["past_len"] == 0

    calls.clear()
    with torch.no_grad():
        cached = model(
            input_ids=input_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
    assert len(calls) == 1
    assert calls[0]["q_len"] == 1
    assert calls[0]["kv_len"] == input_ids.shape[1]
    assert calls[0]["past_len"] == input_ids.shape[1] - 1
    assert calls[0]["sliding_window"] == 3

    calls.clear()
    with torch.no_grad():
        reference_logits, reference_cache = _forward_with_per_layer_attention_masks(
            model,
            input_ids[:, -1:],
            attention_mask,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
    assert len(calls) == len(model.layers)
    assert len(reference_cache) == len(model.layers)
    assert torch.allclose(
        cached.logits,
        reference_logits,
        atol=1e-6,
        rtol=1e-5,
    )
