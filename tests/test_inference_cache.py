import pytest
import torch

from saddlellm.models.InferenceCache import InferenceLayerCache
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


@pytest.mark.parametrize('latent', [False, True])
def test_cache_reuses_storage_preserves_history_and_bounds_capacity(latent):
    cache = InferenceLayerCache(9, latent=latent)
    dim = 1 if latent else 2
    shape = (2, 1, 4) if latent else (2, 2, 1, 4)
    history = []
    previous_pointer = None
    previous_capacity = 0
    reallocations = 0
    with torch.inference_mode():
        for step in range(9):
            values = torch.full(shape, float(step))
            history.append(values)
            result = (cache.append(values) if latent
                      else cache.append(values, values + 1))
            torch.testing.assert_close(result[0], torch.cat(history, dim=dim))
            pointer = cache.buffers[0].data_ptr()
            if cache.capacity == previous_capacity:
                assert pointer == previous_pointer
            else:
                reallocations += 1
            previous_pointer, previous_capacity = pointer, cache.capacity
            assert cache.capacity <= 9
        assert reallocations <= 4
        with pytest.raises(ValueError, match='budget'):
            cache.append(values) if latent else cache.append(values, values)


def test_cache_rejects_training_and_incompatible_shapes():
    cache = InferenceLayerCache(8)
    values = torch.randn(1, 2, 1, 4)
    with pytest.raises(RuntimeError, match='gradients disabled'):
        cache.append(values, values)
    with torch.no_grad():
        cache.append(values, values)
        with pytest.raises(ValueError, match='shape'):
            cache.append(values.expand(2, -1, -1, -1), values.expand(2, -1, -1, -1))
    assert cache.length == 1


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
@pytest.mark.parametrize('kind', ['gqa', 'mha', 'mqa', 'mla', 'mla_kv'])
def test_generation_cache_matches_dynamic_and_uncached(backend, kind):
    torch.manual_seed(318)
    config = SaddleModelConfig(
        vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4 if kind == 'mha' else 1 if kind == 'mqa' else 2,
        max_position_embeddings=64,
        attention_kind='mla' if kind == 'mla_kv' else kind,
        attention_backend=backend,
        kv_lora_rank=4 if kind.startswith('mla') else 0,
        mla_cache_mode='kv' if kind == 'mla_kv' else 'latent',
    )
    model = SaddleForCausalLM(config).eval()
    prompt = torch.tensor([[0, 0, 4, 5], [4, 5, 6, 7]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    kwargs = dict(attention_mask=mask, max_new_tokens=7, do_sample=False,
                  eos_token_id=[], output_scores=True, return_dict_in_generate=True)
    cached = model.generate(prompt, **kwargs)
    dynamic = model.generate(prompt, cache_implementation='dynamic', **kwargs)
    uncached = model.generate(prompt, use_cache=False, **kwargs)
    assert torch.equal(cached.sequences, dynamic.sequences)
    assert torch.equal(cached.sequences, uncached.sequences)
    for cached_score, dynamic_score, uncached_score in zip(
        cached.scores, dynamic.scores, uncached.scores
    ):
        torch.testing.assert_close(cached_score, dynamic_score)
        torch.testing.assert_close(cached_score, uncached_score, rtol=1e-4, atol=1e-5)
