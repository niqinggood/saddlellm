import json
import math

import pytest
import torch
from torch import nn

from saddlellm.models.ModelBlueprint import (
    AttentionBlueprint,
    FFNBlueprint,
    LayerBlueprint,
    ModelBlueprint,
    ResidualBlueprint,
)
from saddlellm.models.ModelBuilder import ModelComponentRegistry, SaddleModelBuilder
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleMoE, SaddleSwiGLU
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def _attention(kind="gqa", **overrides):
    values = {
        "kind": kind,
        "backend": "eager",
        "num_heads": 4,
        "num_kv_heads": 2,
    }
    if kind == "mla":
        values.update(
            {
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "mla_cache_mode": "latent",
            }
        )
    values.update(overrides)
    return AttentionBlueprint(**values)


def _ffn(kind="swiglu", **overrides):
    values = {"kind": kind, "intermediate_size": 32}
    if kind == "moe":
        values.update(
            {
                "num_experts": 2,
                "experts_per_token": 1,
                "expert_intermediate_size": 24,
                "shared_expert": True,
            }
        )
    values.update(overrides)
    return FFNBlueprint(**values)


def _tiny_blueprint(name="tiny-composed", residual=None):
    return ModelBlueprint(
        name=name,
        hidden_size=16,
        vocab_size=31,
        max_position_embeddings=32,
        attention=_attention(),
        ffn=_ffn(),
        residual=residual or ResidualBlueprint(),
        layers=[
            LayerBlueprint(name="dense-gqa", repeat=2),
            LayerBlueprint(
                name="sparse-mla",
                attention=_attention("mla"),
                ffn=_ffn("moe"),
            ),
        ],
    )


def _tiny_mapping(name="tiny-composed"):
    return {
        "name": name,
        "hidden_size": 16,
        "vocab_size": 31,
        "max_position_embeddings": 32,
        "attention": {
            "kind": "gqa",
            "backend": "eager",
            "num_heads": 4,
            "num_kv_heads": 2,
        },
        "ffn": {"kind": "swiglu", "intermediate_size": 32},
        "layers": [
            {"name": "dense-gqa", "repeat": 2},
            {
                "name": "sparse-mla",
                "attention": {
                    "kind": "mla",
                    "q_lora_rank": 4,
                    "kv_lora_rank": 4,
                    "mla_cache_mode": "latent",
                },
                "ffn": {
                    "kind": "moe",
                    "num_experts": 2,
                    "experts_per_token": 1,
                    "expert_intermediate_size": 24,
                    "shared_expert": True,
                },
            },
        ],
    }


def _assert_same_blueprint(left, right, expected_layers=3):
    assert left.to_config_dict() == right.to_config_dict()
    assert left.num_layers == right.num_layers == expected_layers
    assert len(left.expanded_layers()) == len(right.expanded_layers()) == expected_layers
    assert all(layer.repeat == 1 for layer in left.expanded_layers())


def test_python_dict_json_and_yaml_blueprints_have_the_same_normalized_form(tmp_path):
    python_blueprint = _tiny_blueprint()
    mapping_blueprint = ModelBlueprint.from_dict(_tiny_mapping())
    _assert_same_blueprint(python_blueprint, mapping_blueprint)

    json_path = tmp_path / "model.json"
    mapping_blueprint.save(json_path)
    saved_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "analysis" in saved_payload
    _assert_same_blueprint(mapping_blueprint, ModelBlueprint.from_file(json_path))

    yaml = pytest.importorskip("yaml")
    yaml_path = tmp_path / "training.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {"model_blueprint": mapping_blueprint.to_config_dict()},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _assert_same_blueprint(mapping_blueprint, ModelBlueprint.from_file(yaml_path))


def test_partial_layer_overrides_are_identical_in_python_and_mapping_apis():
    mapping = _tiny_mapping("partial-overrides")
    mapping["layers"] = [
        {
            "attention": {
                "kind": "mla",
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
            },
            "ffn": {"kind": "moe", "num_experts": 2, "experts_per_token": 1},
        }
    ]
    from_mapping = ModelBlueprint.from_dict(mapping)
    from_python_dict = ModelBlueprint(
        name="partial-overrides",
        hidden_size=16,
        vocab_size=31,
        max_position_embeddings=32,
        attention=_attention(),
        ffn=_ffn(),
        layers=[mapping["layers"][0]],
    )

    _assert_same_blueprint(from_mapping, from_python_dict, expected_layers=1)
    layer = from_python_dict.expanded_layers()[0]
    assert layer.attention.num_heads == 4
    assert layer.attention.num_kv_heads == 2
    assert layer.attention.backend == "eager"
    assert layer.ffn.intermediate_size == 32


def test_fluent_code_builder_and_configuration_build_the_same_blueprint():
    configured = ModelBlueprint.from_dict(_tiny_mapping("builder-parity"))
    built = (
        SaddleModelBuilder(
            "builder-parity",
            hidden_size=16,
            vocab_size=31,
            max_position_embeddings=32,
        )
        .defaults(
            attention={
                "preset": "gqa",
                "backend": "eager",
                "num_heads": 4,
                "num_kv_heads": 2,
            },
            ffn={"preset": "swiglu", "intermediate_size": 32},
        )
        .add_layers(2, name="dense-gqa")
        .add_layer(
            name="sparse-mla",
            attention={
                "preset": "mla",
                "backend": "eager",
                "num_heads": 4,
                "num_kv_heads": 2,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "mla_cache_mode": "latent",
            },
            ffn={
                "preset": "moe",
                "intermediate_size": 32,
                "num_experts": 2,
                "experts_per_token": 1,
                "expert_intermediate_size": 24,
                "shared_expert": True,
            },
        )
        .build_blueprint()
    )

    _assert_same_blueprint(configured, built)


def test_component_registry_returns_independent_custom_presets_and_rejects_unknown():
    registry = ModelComponentRegistry(include_builtins=False)
    registry.register_attention(
        "tiny-gqa",
        AttentionBlueprint(
            kind="gqa", backend="eager", num_heads=4, num_kv_heads=2
        ),
    )
    first = registry.create_attention("tiny-gqa")
    second = registry.create_attention("tiny-gqa")
    first.num_kv_heads = 1

    assert second.num_kv_heads == 2
    with pytest.raises(KeyError, match="unknown attention preset"):
        registry.create_attention("missing")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_attention("tiny-gqa", {"kind": "mha"})


def test_repeat_expands_to_independent_heterogeneous_decoder_modules():
    blueprint = _tiny_blueprint()
    model = blueprint.build_model()

    assert len(model.layers) == model.config.num_hidden_layers == 3
    assert isinstance(model.layers[0].mlp, SaddleSwiGLU)
    assert isinstance(model.layers[1].mlp, SaddleSwiGLU)
    assert isinstance(model.layers[2].mlp, SaddleMoE)
    assert model.layers[0].self_attn.config.attention_kind == "gqa"
    assert model.layers[0].self_attn.num_kv_heads == 2
    assert model.layers[2].self_attn.config.attention_kind == "mla"
    assert model.layers[2].self_attn.config.mla_cache_mode == "latent"
    assert model.layers[2].self_attn.q_proj is None
    assert model.layers[2].self_attn.kv_a_proj.out_features == 4

    # A repeated segment describes repeated structure, not weight sharing.
    assert model.layers[0] is not model.layers[1]
    assert (
        model.layers[0].self_attn.q_proj.weight.data_ptr()
        != model.layers[1].self_attn.q_proj.weight.data_ptr()
    )


def test_legacy_blueprint_without_layer_specs_remains_homogeneous_and_parameter_free_residual():
    blueprint = ModelBlueprint(
        name="legacy",
        hidden_size=16,
        num_layers=2,
        vocab_size=31,
        max_position_embeddings=32,
        attention=_attention(),
        ffn=_ffn(),
    )
    model = blueprint.build_model()

    assert len(model.layers) == 2
    assert all(isinstance(layer.mlp, SaddleSwiGLU) for layer in model.layers)
    assert all(layer.self_attn.config.attention_kind == "gqa" for layer in model.layers)
    residual_parameters = [name for name, _ in model.named_parameters() if "scale" in name]
    assert residual_parameters == []
    clone = blueprint.build_model()
    incompatible = clone.load_state_dict(model.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_tiny_heterogeneous_model_forward_backward_and_mixed_cache():
    torch.manual_seed(7)
    model = _tiny_blueprint().build_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 6))
    labels = input_ids.clone()
    labels[0, 0] = -100

    model.train()
    output = model(input_ids=input_ids, labels=labels)
    assert output.logits.shape == (2, 6, model.config.vocab_size)
    assert torch.isfinite(output.loss)
    assert len(output.expert_stats) == 1
    assert output.loss_items["aux_loss"].item() >= 0
    output.loss.backward()

    watched_parameters = [
        model.layers[0].self_attn.q_proj.weight,
        model.layers[2].self_attn.kv_a_proj.weight,
        model.layers[2].mlp.router.weight,
    ]
    assert all(parameter.grad is not None for parameter in watched_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in watched_parameters)

    model.eval()
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        full = model(input_ids=input_ids).logits[:, -1]
        prefix = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=True,
        )
        assert isinstance(prefix.past_key_values[0], tuple)
        assert prefix.past_key_values[2]["cache_type"] == "mla_latent"
        cached = model(
            input_ids=input_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits[:, -1]
    assert torch.allclose(full, cached, atol=2e-5, rtol=2e-4)


def test_native_checkpoint_preserves_composition_and_logits(tmp_path):
    torch.manual_seed(11)
    model = _tiny_blueprint().build_model().eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    reloaded = SaddleForCausalLM.from_pretrained_saddle(checkpoint, map_location="cpu").eval()
    with torch.no_grad():
        actual = reloaded(input_ids=input_ids).logits

    assert reloaded.config.to_dict() == model.config.to_dict()
    assert len(reloaded.layers) == 3
    assert isinstance(reloaded.layers[0].mlp, SaddleSwiGLU)
    assert isinstance(reloaded.layers[2].mlp, SaddleMoE)
    assert reloaded.layers[2].self_attn.config.attention_kind == "mla"
    assert torch.allclose(expected, actual, atol=0, rtol=0)


class _IdentityNorm(nn.Module):
    def forward(self, inputs):
        return inputs


class _RecordingAttention(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier
        self.last_input = None

    def forward(self, hidden_states, **_):
        self.last_input = hidden_states.detach().clone()
        return hidden_states * self.multiplier, None


class _RecordingFFN(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier
        self.last_input = None

    def forward(self, hidden_states):
        self.last_input = hidden_states.detach().clone()
        return hidden_states * self.multiplier


def _residual_test_layer(residual):
    blueprint = ModelBlueprint(
        name="residual-test",
        hidden_size=4,
        vocab_size=7,
        attention=AttentionBlueprint(
            kind="mha", backend="eager", num_heads=2, num_kv_heads=2
        ),
        ffn=FFNBlueprint(kind="swiglu", intermediate_size=8),
        layers=[LayerBlueprint(residual=residual)],
    )
    layer = blueprint.build_model().layers[0]
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingAttention(2.0)
    layer.mlp = _RecordingFFN(3.0)
    layer.is_moe = False
    return layer


@pytest.mark.parametrize(
    ("topology", "expected", "expected_ffn_input"),
    [("serial", 3.5, 2.0), ("parallel", 2.75, 1.0)],
)
def test_residual_topology_and_fixed_scales(topology, expected, expected_ffn_input):
    layer = _residual_test_layer(
        ResidualBlueprint(
            topology=topology,
            attention_scale=0.5,
            ffn_scale=0.25,
        )
    ).eval()
    inputs = torch.ones(2, 3, 4)

    output, _, _, _ = layer(inputs)

    assert torch.allclose(layer.self_attn.last_input, inputs)
    assert torch.allclose(
        layer.mlp.last_input, torch.full_like(inputs, expected_ffn_input)
    )
    assert torch.allclose(output, torch.full_like(inputs, expected))
    assert [name for name, _ in layer.named_parameters() if "scale" in name] == []


def test_zero_fixed_scale_isolates_each_residual_branch():
    inputs = torch.ones(1, 2, 4)
    no_attention = _residual_test_layer(
        ResidualBlueprint(attention_scale=0.0, ffn_scale=1.0)
    ).eval()
    no_ffn = _residual_test_layer(
        ResidualBlueprint(attention_scale=1.0, ffn_scale=0.0)
    ).eval()

    attention_off, _, _, _ = no_attention(inputs)
    ffn_off, _, _, _ = no_ffn(inputs)

    assert torch.allclose(attention_off, torch.full_like(inputs, 4.0))
    assert torch.allclose(ffn_off, torch.full_like(inputs, 3.0))


def test_learnable_residual_scales_are_scalar_parameters_with_gradients():
    layer = _residual_test_layer(
        ResidualBlueprint(
            attention_scale=0.25,
            ffn_scale=0.75,
            learnable=True,
        )
    )
    scales = [(name, parameter) for name, parameter in layer.named_parameters() if "scale" in name]

    assert len(scales) == 2
    assert all(parameter.shape == torch.Size([]) for _, parameter in scales)
    assert sorted(parameter.item() for _, parameter in scales) == [0.25, 0.75]

    output, _, _, _ = layer(torch.ones(2, 3, 4))
    output.sum().backward()
    assert all(parameter.grad is not None for _, parameter in scales)
    assert all(torch.isfinite(parameter.grad) and parameter.grad.abs() > 0 for _, parameter in scales)


def test_residual_dropout_is_disabled_in_eval_and_seeded_in_train():
    layer = _residual_test_layer(
        ResidualBlueprint(topology="parallel", dropout=0.5)
    )
    inputs = torch.ones(3, 4, 4)

    layer.eval()
    eval_first = layer(inputs)[0]
    eval_second = layer(inputs)[0]
    assert torch.equal(eval_first, eval_second)
    assert torch.allclose(eval_first, torch.full_like(inputs, 6.0))

    layer.train()
    torch.manual_seed(101)
    train_first = layer(inputs)[0]
    torch.manual_seed(101)
    train_repeated = layer(inputs)[0]
    torch.manual_seed(102)
    train_different_seed = layer(inputs)[0]
    assert torch.equal(train_first, train_repeated)
    assert not torch.equal(train_first, train_different_seed)
    assert not torch.equal(train_first, eval_first)


def _initialization_blueprint(initialization):
    residual = ResidualBlueprint(initialization=initialization)
    return ModelBlueprint(
        name=f"init-{initialization}",
        hidden_size=16,
        vocab_size=31,
        attention=_attention(),
        ffn=_ffn(),
        residual=residual,
        layers=[
            LayerBlueprint(name="dense", residual=residual),
            LayerBlueprint(name="moe", ffn=_ffn("moe"), residual=residual),
        ],
    )


def _residual_projection_weights(model):
    weights = {
        "layers.0.self_attn.o_proj.weight": model.layers[0].self_attn.o_proj.weight,
        "layers.0.mlp.down_proj.weight": model.layers[0].mlp.down_proj.weight,
        "layers.1.self_attn.o_proj.weight": model.layers[1].self_attn.o_proj.weight,
    }
    for index, expert in enumerate(model.layers[1].mlp.experts):
        weights[f"layers.1.mlp.experts.{index}.down_proj.weight"] = expert.down_proj.weight
    weights["layers.1.mlp.shared_expert.down_proj.weight"] = (
        model.layers[1].mlp.shared_expert.down_proj.weight
    )
    return weights


def test_standard_depth_scaled_and_zero_residual_initialization():
    torch.manual_seed(19)
    standard = _initialization_blueprint("standard").build_model()
    torch.manual_seed(19)
    depth_scaled = _initialization_blueprint("depth_scaled").build_model()
    torch.manual_seed(19)
    zero = _initialization_blueprint("zero").build_model()

    standard_outputs = _residual_projection_weights(standard)
    scaled_outputs = _residual_projection_weights(depth_scaled)
    zero_outputs = _residual_projection_weights(zero)
    factor = 1.0 / math.sqrt(2.0 * len(depth_scaled.layers))
    for name in standard_outputs:
        assert torch.count_nonzero(standard_outputs[name]).item() > 0
        assert torch.allclose(
            scaled_outputs[name], standard_outputs[name] * factor, atol=1e-8, rtol=1e-6
        )
        assert torch.count_nonzero(zero_outputs[name]).item() == 0

    hidden_states = torch.randn(2, 3, 16)
    dense_output = zero.layers[0](hidden_states)[0]
    moe_output = zero.layers[1](hidden_states)[0]
    assert torch.allclose(dense_output, hidden_states, atol=0, rtol=0)
    assert torch.allclose(moe_output, hidden_states, atol=0, rtol=0)

    output_names = set(standard_outputs)
    # Initialization policy must not perturb embeddings, norms, routers, or
    # the input/gate projections when the random seed is held constant.
    for name, standard_value in standard.state_dict().items():
        if name in output_names:
            continue
        assert torch.equal(standard_value, depth_scaled.state_dict()[name]), name
        assert torch.equal(standard_value, zero.state_dict()[name]), name


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda raw: raw.update(hidden_size=15), "divisible"),
        (
            lambda raw: raw["attention"].update(num_heads=4, num_kv_heads=3),
            "num_kv_heads",
        ),
        (lambda raw: raw["attention"].update(kind="unknown"), "attention kind"),
        (
            lambda raw: raw["attention"].update(kind="mla", kv_lora_rank=0),
            "kv_lora_rank",
        ),
        (lambda raw: raw["ffn"].update(kind="unknown"), "ffn kind"),
        (
            lambda raw: raw["ffn"].update(
                kind="moe", num_experts=0, experts_per_token=1
            ),
            "num_experts",
        ),
        (
            lambda raw: raw["ffn"].update(
                kind="moe", num_experts=2, experts_per_token=3
            ),
            "experts_per_token",
        ),
        (
            lambda raw: raw.update(residual={"topology": "sideways"}),
            "residual topology",
        ),
        (
            lambda raw: raw.update(residual={"dropout": 1.0}),
            "dropout",
        ),
        (
            lambda raw: raw.update(residual={"initialization": "mystery"}),
            "residual initialization",
        ),
    ],
)
def test_invalid_component_and_shape_configuration_fails_early(mutate, match):
    raw = _tiny_mapping("invalid")
    raw["layers"] = []
    raw["num_layers"] = 1
    mutate(raw)
    with pytest.raises(ValueError, match=match):
        ModelBlueprint.from_dict(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {**_tiny_mapping("unknown-top-level"), "hidden_sze": 16},
        {
            **_tiny_mapping("unknown-layer-field"),
            "layers": [{"repeat": 1, "attentin": {"kind": "gqa"}}],
        },
        {
            **_tiny_mapping("fractional-repeat"),
            "layers": [{"repeat": 1.5}],
        },
    ],
)
def test_typos_and_non_integral_repeat_are_not_silently_normalized(raw):
    with pytest.raises((TypeError, ValueError)):
        ModelBlueprint.from_dict(raw)


def test_non_positive_repeat_is_rejected():
    with pytest.raises(ValueError, match="repeat"):
        LayerBlueprint(repeat=0)


def test_training_orchestrator_parse_accepts_nested_model_blueprint():
    blueprint = _tiny_mapping("nested-training-blueprint")

    parsed = TrainingOrchestrator._parse_config(
        {
            "model": {
                "config": "qwen-tiny-160m",
                "backend": "saddle",
                "blueprint": blueprint,
            },
            "stages": ["eval"],
        }
    )

    assert parsed.model_backend == "saddle"
    assert parsed.model_blueprint == blueprint


def test_training_orchestrator_parse_keeps_legacy_top_level_model_blueprint():
    blueprint = _tiny_mapping("legacy-training-blueprint")

    parsed = TrainingOrchestrator._parse_config(
        {
            "model": {"config": "qwen-tiny-160m", "backend": "saddle"},
            "model_blueprint": blueprint,
            "stages": ["eval"],
        }
    )

    assert parsed.model_blueprint == blueprint


def test_training_orchestrator_rejects_conflicting_blueprint_locations():
    nested = _tiny_mapping("nested")
    legacy = _tiny_mapping("legacy")

    with pytest.raises(ValueError, match=r"model_blueprint.*model\.blueprint"):
        TrainingOrchestrator._parse_config(
            {
                "model": {
                    "config": "qwen-tiny-160m",
                    "backend": "saddle",
                    "blueprint": nested,
                },
                "model_blueprint": legacy,
                "stages": ["eval"],
            }
        )


def test_plain_config_preserves_model_blueprint():
    blueprint = _tiny_mapping("config-blueprint")
    config = {
        "experiment": "composed-model",
        "stages": ["eval"],
        "model": {
            "config": "qwen-tiny-160m",
            "backend": "saddle",
            "blueprint": blueprint,
        },
    }

    parsed = TrainingOrchestrator._parse_config(config)

    assert config["model"]["blueprint"] == blueprint
    assert parsed.model_blueprint == blueprint


def test_layer_residual_aliases_override_inherited_model_defaults():
    raw = _tiny_mapping("layer-residual-aliases")
    raw["residual"] = {
        "topology": "serial",
        "attention_scale": 0.9,
        "ffn_scale": 0.8,
        "learnable": False,
        "initialization": "standard",
    }
    raw["layers"] = [
        {
            "residual": {
                "layout": "parallel",
                "scale": 0.25,
                "gate": "scalar",
                "projection_init": "zero",
            }
        }
    ]

    blueprint = ModelBlueprint.from_dict(raw)
    residual = blueprint.expanded_layers()[0].residual

    assert blueprint.residual.topology == "serial"
    assert residual.topology == "parallel"
    assert residual.attention_scale == pytest.approx(0.25)
    assert residual.ffn_scale == pytest.approx(0.25)
    assert residual.learnable is True
    assert residual.initialization == "zero"


def test_builder_layer_residual_layout_alias_overrides_serial_default():
    blueprint = (
        SaddleModelBuilder(
            "builder-layout-alias",
            hidden_size=16,
            vocab_size=31,
            max_position_embeddings=32,
        )
        .defaults(
            attention={
                "preset": "gqa",
                "backend": "eager",
                "num_heads": 4,
                "num_kv_heads": 2,
            },
            ffn={"preset": "swiglu", "intermediate_size": 32},
            residual={"topology": "serial", "attention_scale": 0.5},
        )
        .add_layer(residual={"layout": "parallel"})
        .build_blueprint()
    )

    residual = blueprint.expanded_layers()[0].residual
    assert blueprint.residual.topology == "serial"
    assert residual.topology == "parallel"
    assert residual.attention_scale == pytest.approx(0.5)


def test_linear_rope_scaling_changes_rotary_positions():
    unscaled_raw = _tiny_mapping("rope-unscaled")
    unscaled_raw["num_layers"] = 1
    unscaled_raw["layers"] = []
    scaled_raw = _tiny_mapping("rope-scaled")
    scaled_raw["num_layers"] = 1
    scaled_raw["layers"] = []
    scaled_raw["attention"]["rope_scaling"] = {"type": "linear", "factor": 2.0}

    unscaled = ModelBlueprint.from_dict(unscaled_raw).build_model()
    scaled = ModelBlueprint.from_dict(scaled_raw).build_model()
    unscaled_rotary = unscaled.layers[0].self_attn.rotary
    scaled_rotary = scaled.layers[0].self_attn.rotary

    assert scaled_rotary.scaling_factor == pytest.approx(2.0)
    base_cos, base_sin = unscaled_rotary(3, dtype=torch.float32)
    scaled_cos, scaled_sin = scaled_rotary(3, dtype=torch.float32)
    assert torch.allclose(scaled_cos[2], base_cos[1])
    assert torch.allclose(scaled_sin[2], base_sin[1])
    assert not torch.allclose(scaled_cos[1], base_cos[1])


def test_preflight_rejects_blueprint_with_hugging_face_backend(tmp_path):
    parsed = TrainingOrchestrator._parse_config(
        {
            "model": {
                "config": "qwen-tiny-160m",
                "backend": "hf",
                "blueprint": _tiny_mapping("hf-cannot-consume-blueprint"),
            },
            "stages": ["eval"],
            "logging": {"output_dir": str(tmp_path)},
        }
    )

    with pytest.raises(ValueError, match=r"backend='saddle'.*HF backend"):
        TrainingOrchestrator(parsed)


def test_unknown_model_spec_requires_a_self_contained_blueprint(tmp_path):
    parsed = TrainingOrchestrator._parse_config(
        {
            "model": {
                "config": "not-a-registered-model-spec",
                "backend": "saddle",
                "blueprint": {"name": "incomplete", "layers": [{"repeat": 1}]},
            },
            "stages": ["eval"],
            "logging": {"output_dir": str(tmp_path)},
        }
    )

    with pytest.raises(ValueError, match=r"without a known model\.config.*explicitly define"):
        TrainingOrchestrator(parsed)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda raw: raw.update(norm="layernorm"), "only rmsnorm"),
        (lambda raw: raw.update(activation="gelu"), "only silu activation"),
        (
            lambda raw: raw["attention"].update(
                rope_scaling={"type": "dynamic", "factor": 2.0}
            ),
            "rope_scaling.*only.*linear",
        ),
        (
            lambda raw: raw["attention"].update(mla_cache_mode="compressed"),
            "mla_cache_mode.*kv or latent",
        ),
    ],
)
def test_unimplemented_native_model_options_fail_during_blueprint_validation(
    mutate, match
):
    raw = _tiny_mapping("unsupported-native-option")
    raw["num_layers"] = 1
    raw["layers"] = []
    mutate(raw)

    with pytest.raises(ValueError, match=match):
        ModelBlueprint.from_dict(raw)
