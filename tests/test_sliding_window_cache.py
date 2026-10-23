"""Window KV eviction must preserve absolute positions and model semantics."""

import pytest
import torch

from saddlellm.models.InferenceCache import InferenceLayerCache
from saddlellm.models.NativePrefixCache import NativePrefixCache
from saddlellm.models.SaddleModeling import (
    SaddleAttention,
    SaddleForCausalLM,
    SaddleModelConfig,
)


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(kind="gqa", backend="sdpa", window=4, mode="none", mixed=False):
    torch.manual_seed(713)
    config = SaddleModelConfig(
        vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=4 if kind == "mha" else 1 if kind == "mqa" else 2,
        max_position_embeddings=64, attention_kind="mla" if kind == "mla_kv" else kind,
        attention_backend=backend, sliding_window=window,
        kv_lora_rank=4 if kind.startswith("mla") else 0,
        mla_cache_mode="kv" if kind == "mla_kv" else "latent",
        residual_attention_mode=mode, residual_block_size=4,
        eos_token_id=[],
    )
    model = SaddleForCausalLM(config).eval()
    if mixed:
        # Layer configs are independent, as in an expanded blueprint.
        from dataclasses import replace

        layer = model.layers[1]
        layer.config = replace(layer.config, sliding_window=0)
        layer.self_attn.config = layer.config
    return model


@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("window", [1, 3, 4, 31])
def test_ring_matches_reference_for_chunks_wraps_and_short_budgets(latent, window):
    cache = InferenceLayerCache(23, latent=latent, window_size=window)
    dim = 1 if latent else 2
    histories = [[] for _ in range(1 if latent else 2)]
    pointer = None
    with torch.inference_mode():
        for count in [7, 1, 1, 5, 1, 1, 6, 1]:
            past = cache.length
            shape = (2, count, 4) if latent else (2, 2, count, 4)
            values = tuple(torch.randn(shape) for _ in histories)
            key_count, key_start = cache.attention_layout(count)
            result = cache.append(*values)
            for index, value in enumerate(values):
                histories[index].append(value)
                full = torch.cat(histories[index], dim=dim)
                torch.testing.assert_close(result[index], full.narrow(dim, key_start, key_count))
                torch.testing.assert_close(
                    cache._window_views()[index],
                    full.narrow(dim, cache.length - cache.stored_length, cache.stored_length),
                )
            assert cache.length == past + count
            assert cache.stored_length == min(cache.length, window)
            assert cache.capacity == min(window, cache.max_length)
            assert cache.buffers[0].shape[dim] == 2 * cache.capacity
            if pointer is not None:
                assert cache.buffers[0].data_ptr() == pointer
            pointer = cache.buffers[0].data_ptr()
        with pytest.raises(ValueError, match="budget"):
            cache.append(*values)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_window_rejected(value):
    with pytest.raises(ValueError, match="window"):
        InferenceLayerCache(8, window_size=value)
    with pytest.raises(ValueError, match="window"):
        SaddleModelConfig(sliding_window=value)


def test_window_invalid_append_does_not_mutate_state():
    cache = InferenceLayerCache(8, window_size=3)
    value = torch.randn(1, 2, 1, 4)
    with pytest.raises(RuntimeError, match="gradients disabled"):
        cache.append(value, value)
    with torch.no_grad():
        cache.append(value, value)
        saved = [item.clone() for item in cache.buffers]
        for values in ((value, value.double()), (value.expand(2, -1, -1, -1), value),
                       (value[:, :, :0], value[:, :, :0])):
            with pytest.raises(ValueError):
                cache.append(*values)
            assert cache.length == 1
            for actual, reference in zip(cache.buffers, saved):
                torch.testing.assert_close(actual, reference, equal_nan=True)


def test_bad_first_window_append_does_not_allocate_storage():
    cache = InferenceLayerCache(8, window_size=3)
    with torch.no_grad():
        for values in ((torch.ones(1, 2, 1), torch.ones(1, 2, 1)),
                       (torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4, dtype=torch.float64))):
            with pytest.raises(ValueError):
                cache.append(*values)
            assert cache.length == cache.capacity == 0 and not cache.buffers


@pytest.mark.parametrize("kind", ["mha", "mqa", "gqa", "mla", "mla_kv"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("window", [1, 4])
def test_window_generation_scores_match_full_history_and_uncached(kind, backend, window):
    model = _model(kind, backend, window, mixed=True)
    ids = torch.tensor([[0, 0, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 10, 11]])
    mask = ids.ne(0).long()
    kwargs = dict(attention_mask=mask, max_new_tokens=11, do_sample=False,
                  output_scores=True, return_dict_in_generate=True)
    actual = model.generate(ids, **kwargs)
    legacy = model.generate(ids, cache_implementation="dynamic", **kwargs)
    uncached = model.generate(ids, use_cache=False, **kwargs)
    assert torch.equal(actual.sequences, legacy.sequences)
    assert torch.equal(actual.sequences, uncached.sequences)
    for score, old, ref in zip(actual.scores, legacy.scores, uncached.scores):
        torch.testing.assert_close(score, old, atol=2e-6, rtol=1e-4)
        torch.testing.assert_close(score, ref, atol=2e-6, rtol=1e-4)
    with model.create_generation_session(ids, **{k: v for k, v in kwargs.items()
                                               if k != "return_dict_in_generate"}) as session:
        session.prefill()
        initial_bytes = [sum(t.numel() * t.element_size() for t in c.buffers)
                         for c in session.past_key_values]
        list(session)
        for index, cache in enumerate(session.past_key_values):
            assert cache.length == ids.shape[1] + 10
            if index != 1:
                assert cache.stored_length == window
                assert sum(t.numel() * t.element_size() for t in cache.buffers) == initial_bytes[index]
            else:
                assert cache.stored_length == cache.length


@pytest.mark.parametrize("kind", ["gqa", "mla", "mla_kv"])
@pytest.mark.parametrize("mode", ["none", "full", "block"])
def test_packed_document_positions_and_multitoken_chunks(kind, mode):
    model = _model(kind, mode=mode, mixed=True)
    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    docs = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    caches = [InferenceLayerCache(12, latent=kind == "mla", window_size=layer.config.sliding_window)
              for layer in model.layers]
    parts = []
    with torch.inference_mode():
        reference = model(ids, document_ids=docs).logits
        for start, end in [(0, 5), (5, 8), (8, 9), (9, 12)]:
            output = model(ids[:, start:end], document_ids=docs[:, :end],
                           past_key_values=caches, use_cache=True)
            parts.append(output.logits)
            assert {cache.length for cache in caches} == {end}
        torch.testing.assert_close(torch.cat(parts, dim=1), reference, atol=2e-6, rtol=1e-4)


def test_rotary_positions_do_not_reset_when_window_wraps():
    model = _model(window=3)
    ids = torch.arange(4, 23)[None, :]
    with torch.inference_mode():
        reference = model(ids).logits
        caches = [InferenceLayerCache(19, window_size=3) for _ in model.layers]
        for index in range(ids.shape[1]):
            actual = model(ids[:, index:index + 1], past_key_values=caches, use_cache=True).logits
            torch.testing.assert_close(actual, reference[:, index:index + 1], atol=2e-6, rtol=1e-4)


def test_wrong_window_rejected_before_any_layer_cache_is_mutated():
    model = _model(mixed=True)
    caches = [InferenceLayerCache(12, window_size=4) for _ in model.layers]
    with torch.inference_mode(), pytest.raises(ValueError, match="window"):
        model(torch.tensor([[4, 5]]), past_key_values=caches, use_cache=True)
    assert all(cache.length == 0 and not cache.buffers for cache in caches)


def test_window_prefix_cache_fails_closed_without_changing_outputs():
    model = _model()
    prefix = NativePrefixCache(65536, min_prefix_tokens=2, reuse_partial_prefixes=True)
    ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    reference = model.generate(ids, max_new_tokens=5, do_sample=False)
    for _ in range(2):
        actual = model.generate(ids, max_new_tokens=5, do_sample=False, prefix_cache=prefix)
        assert torch.equal(actual, reference)
    assert prefix.snapshot()["entries"] == 0
    assert prefix.snapshot()["skipped_requests"] == 2


def test_window_checkpoint_roundtrip_and_training_gradients_remain_valid(tmp_path):
    model = _model(window=3, mode="block")
    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    expected = model.generate(ids, max_new_tokens=8, do_sample=False)
    model.save_pretrained(str(tmp_path))
    loaded = SaddleForCausalLM.from_pretrained_saddle(str(tmp_path)).eval()
    assert torch.equal(loaded.generate(ids, max_new_tokens=8, do_sample=False), expected)
    loaded.train()
    loss = loaded(ids, labels=ids).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in loaded.parameters()
               if parameter.grad is not None)


def test_kda_plus_window_global_layer_matches_full_history_generation():
    from saddlellm.models.ModelBlueprint import ModelBlueprintLab

    blueprint = ModelBlueprintLab.hybrid_kda(hidden_size=16, layers=4, heads=4,
                                            kv_heads=2, vocab_size=37, seq_length=32,
                                            attention_residuals="block")
    blueprint.layers[-1].attention.sliding_window = 3
    model = blueprint.build_model().eval()
    ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    kwargs = dict(max_new_tokens=9, do_sample=False, eos_token_id=[], output_scores=True,
                  return_dict_in_generate=True)
    actual = model.generate(ids, **kwargs)
    reference = model.generate(ids, cache_implementation="dynamic", **kwargs)
    assert torch.equal(actual.sequences, reference.sequences)
    for new, old in zip(actual.scores, reference.scores):
        torch.testing.assert_close(new, old, atol=2e-6, rtol=1e-4)


def test_window_benchmark_reports_physical_storage_and_correctness(monkeypatch, capsys):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("window_benchmark", Path(__file__).resolve().parents[1] / "tools/benchmark_window_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["benchmark_window_cache.py", "--prompt-tokens", "12",
                                     "--decode-tokens", "2", "--window", "3", "--repeats", "1"])
    module.main()
    report = json.loads(capsys.readouterr().out)
    assert report["greedy_tokens_equal"]
    assert report["window_ring"]["stored_lengths"] == [3, 3, 3, 13]
    assert report["window_ring"]["physical_slots_per_layer"] == [6, 6, 6, 14]
    assert report["persistent_kv_reduction_percent"] > 0


def test_bounded_single_query_window_uses_unmasked_decode_fast_path():
    mask, causal = SaddleAttention._build_attention_mask(
        attention_mask=None, q_len=1, kv_len=4, past_len=17, key_start=14,
        dtype=torch.float32, device="cpu", sliding_window=4,
    )
    assert mask is None and not causal


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_window_generation_matches_full_kv(dtype):
    model = _model(window=4, mixed=True).to(device="cuda", dtype=dtype)
    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]], device="cuda")
    kwargs = dict(max_new_tokens=12, do_sample=False, output_scores=True, return_dict_in_generate=True)
    actual = model.generate(ids, **kwargs)
    legacy = model.generate(ids, cache_implementation="dynamic", **kwargs)
    assert torch.equal(actual.sequences, legacy.sequences)
    for new, old in zip(actual.scores, legacy.scores):
        torch.testing.assert_close(new, old, atol=0.004, rtol=0.02)
