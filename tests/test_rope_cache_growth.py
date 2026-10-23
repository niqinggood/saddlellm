import pytest
import torch

from saddlellm.models.SaddleModeling import (
    SaddleForCausalLM,
    SaddleModelConfig,
    SaddleRotaryEmbedding,
)


def _tiny_model(*, max_position_embeddings: int = 8) -> SaddleForCausalLM:
    return SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=37,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=max_position_embeddings,
            attention_kind="gqa",
            attention_backend="eager",
        )
    ).eval()


def _only_cached_pair(cache):
    assert len(cache) == 1
    return next(iter(cache.values()))


def test_rope_cache_grows_geometrically_without_replacing_each_token():
    rotary = SaddleRotaryEmbedding(dim=8, max_position_embeddings=4)
    previous_capacity = 0
    previous_cos = None
    previous_sin = None
    allocations = []

    # One-token cached decoding increases the required range on every call.
    # The backing tensors must remain stable until a capacity boundary is hit.
    for required_length in range(1, 66):
        cos, sin = rotary(
            1,
            dtype=torch.float32,
            offset=required_length - 1,
        )
        cached_cos, cached_sin = _only_cached_pair(rotary._cache)
        capacity = cached_cos.shape[0]

        assert cos.shape == sin.shape == (1, rotary.dim)
        assert capacity == cached_sin.shape[0]
        assert capacity >= required_length

        if capacity == previous_capacity:
            assert cached_cos.data_ptr() == previous_cos.data_ptr()
            assert cached_sin.data_ptr() == previous_sin.data_ptr()
        else:
            allocations.append((capacity, cached_cos, cached_sin))
            if previous_capacity:
                # Token-at-a-time growth crosses the old boundary by one, so
                # geometric reservation should at least double the capacity.
                assert required_length == previous_capacity + 1
                assert capacity >= previous_capacity * 2
                # Rebuilding a larger table must preserve all prior positions.
                assert torch.equal(cached_cos[:previous_capacity], previous_cos)
                assert torch.equal(cached_sin[:previous_capacity], previous_sin)

        previous_capacity = capacity
        previous_cos = cached_cos
        previous_sin = cached_sin

    # This would be 65 allocations under exact-size growth.
    assert 1 < len(allocations) < 10
    assert [capacity for capacity, _, _ in allocations] == sorted(
        capacity for capacity, _, _ in allocations
    )


def test_model_prewarm_and_freeze_support_fixed_length_and_reject_overflow():
    model = _tiny_model()
    model.prewarm_rope_cache(8, dtype=torch.float32, freeze=True)

    assert model._rope_cache_limit == 8
    assert all(layer.self_attn.rotary._cache_frozen for layer in model.layers)
    assert all(
        layer.self_attn.rotary._cache_frozen_length == 8
        for layer in model.layers
    )
    assert {key[-1] for key in model._rope_cache} == {torch.float32}

    cached_cos, cached_sin = _only_cached_pair(model._rope_cache)
    pointers_before = (cached_cos.data_ptr(), cached_sin.data_ptr())
    input_ids = torch.arange(8).remainder(model.config.vocab_size).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids=input_ids)

    assert output.logits.shape == (1, 8, model.config.vocab_size)
    cached_cos, cached_sin = _only_cached_pair(model._rope_cache)
    assert (cached_cos.data_ptr(), cached_sin.data_ptr()) == pointers_before

    overflow_ids = torch.arange(9).remainder(model.config.vocab_size).unsqueeze(0)
    with pytest.raises(
        RuntimeError,
        match=r"RoPE cache is frozen.*frozen_length=8.*required_length=9",
    ):
        model(input_ids=overflow_ids)


@pytest.mark.parametrize("recovery", ["unfreeze_rope_cache", "clear_rope_cache"])
def test_unfreeze_or_clear_allows_rope_cache_to_grow_again(recovery):
    model = _tiny_model()
    model.prewarm_rope_cache(8, dtype=torch.float32, freeze=True)
    old_cos, old_sin = _only_cached_pair(model._rope_cache)
    old_capacity = old_cos.shape[0]
    required_length = old_capacity + 1

    overflow_ids = torch.arange(required_length).remainder(
        model.config.vocab_size
    ).unsqueeze(0)
    with pytest.raises(RuntimeError, match="RoPE cache is frozen"):
        model(input_ids=overflow_ids)

    returned = getattr(model, recovery)()
    assert returned is model
    assert all(not layer.self_attn.rotary._cache_frozen for layer in model.layers)
    if recovery == "clear_rope_cache":
        assert model._rope_cache == {}
        assert model._rope_cache_limit is None

    with torch.no_grad():
        output = model(input_ids=overflow_ids)

    assert output.logits.shape[:2] == (1, required_length)
    grown_cos, grown_sin = _only_cached_pair(model._rope_cache)
    assert grown_cos.shape[0] == grown_sin.shape[0]
    assert grown_cos.shape[0] >= required_length
    assert grown_cos.data_ptr() != old_cos.data_ptr()
    assert grown_sin.data_ptr() != old_sin.data_ptr()
    assert torch.equal(grown_cos[:old_capacity], old_cos)
    assert torch.equal(grown_sin[:old_capacity], old_sin)
