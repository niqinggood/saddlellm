import torch

from saddlellm.models.ModelBlueprint import ModelBlueprintLab
from saddlellm.models.ModelRegistry import MODEL_SPECS, ModelRegistry
from saddlellm.models.SaddleModeling import SaddleModelConfig
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator

EFFICIENT_V2_NAME = "saddle-gqa-300m-efficient-v2"


def test_efficient_v2_is_an_independent_303m_registry_preset():
    legacy = MODEL_SPECS["saddle-gqa-300m-optimized"]
    efficient = MODEL_SPECS[EFFICIENT_V2_NAME]

    assert legacy.intermediate_size == 4096
    assert legacy.tie_word_embeddings is False
    assert efficient.intermediate_size == 2816
    assert efficient.tie_word_embeddings is True
    assert efficient.residual_initialization == "depth_scaled"
    assert efficient.qk_norm is efficient.fused_qkv is efficient.fused_gate_up is True
    assert efficient.estimated_params == 303_353_856
    assert 300_000_000 <= efficient.estimated_params < 310_000_000
    assert efficient.estimated_params < legacy.estimated_params
    assert efficient.to_dict()["tie_word_embeddings"] is True


def test_efficient_v2_blueprint_matches_registry_and_derives_hardware_width():
    spec = MODEL_SPECS[EFFICIENT_V2_NAME]
    blueprint = ModelBlueprintLab.efficient_dense_gqa_v2()
    resolved = TrainingOrchestrator.resolve_architecture_blueprint(spec.name)

    for candidate in (blueprint, resolved):
        assert candidate.name == spec.name
        assert candidate.hidden_size == spec.hidden_size
        assert candidate.num_layers == spec.num_hidden_layers
        assert candidate.ffn.intermediate_size == spec.intermediate_size
        assert candidate.tie_word_embeddings is True
        assert candidate.attention.qk_norm is True
        assert candidate.attention.fused_qkv is True
        assert candidate.ffn.fused_gate_up is True
        assert candidate.estimate_total_params() == spec.estimated_params

    scaled = ModelBlueprintLab.efficient_dense_gqa_v2(
        hidden_size=768,
        layers=2,
        heads=12,
        kv_heads=4,
        vocab_size=128,
    )
    assert scaled.ffn.intermediate_size == 2048


def test_efficient_v2_estimate_matches_the_instantiated_model_parameter_count():
    spec = MODEL_SPECS[EFFICIENT_V2_NAME]

    # Meta construction exercises the complete module topology without
    # allocating more than a gigabyte merely to verify the 300M preset.
    with torch.device("meta"):
        model = ModelRegistry.create_saddle_model(spec)

    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert actual_parameters == spec.estimated_params
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.config.tie_word_embeddings is True


def test_efficient_v2_config_round_trip_preserves_weight_tying():
    spec = MODEL_SPECS[EFFICIENT_V2_NAME]
    config = SaddleModelConfig.from_model_spec(spec)

    assert config.intermediate_size == 2816
    assert config.tie_word_embeddings is True
    assert config.residual_initialization == "depth_scaled"
    assert config.qk_norm is config.fused_qkv is config.fused_gate_up is True
