import pytest
import torch

from saddlellm.models.OperatorBackends import AttentionBackendRegistry


@pytest.mark.parametrize("mask", [False, True])
def test_reference_dispatch_does_not_use_speed_folding(monkeypatch, mask):
    import saddlellm.models.OperatorBackends as backends

    AttentionBackendRegistry.ensure_defaults()
    monkeypatch.setattr(backends, "_sdpa_supports_gqa", lambda: True)
    monkeypatch.setattr(backends, "_folded_gqa_decode", lambda *a: pytest.fail("reference used folding"))
    observed = []

    def sdpa(q, k, v, **kwargs):
        observed.append((q.shape, k.shape, kwargs))
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sdpa)
    q, k, v = torch.randn(1, 4, 1, 8), torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
    attention_mask = torch.zeros(1, 1, 1, 4) if mask else None
    AttentionBackendRegistry.run("sdpa_reference", q, k, v, attention_mask, False)
    assert observed[0][0] == q.shape
    assert observed[0][1][1] == (4 if mask else 2)
    assert observed[0][2].get("enable_gqa", False) == (not mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("seq", [1, 7])
def test_reference_matches_compact_sdpa_bf16(seq):
    torch.manual_seed(44)
    q = torch.randn(1, 12, seq, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, seq, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with torch.inference_mode():
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=seq > 1, scale=128 ** -0.5)
        own = AttentionBackendRegistry.run("sdpa_reference", q, k, v, None, seq > 1)
    assert torch.equal(ref, own)


def test_reference_attention_supports_prefix_cache_and_policy_change_invalidates():
    from saddlellm.models.NativePrefixCache import NativePrefixCache
    from test_native_prefix_cache import model, run

    instance = model(backend="sdpa_reference")
    cache = NativePrefixCache(1024 * 1024, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6]])
    first = run(instance, ids, cache)
    second = run(instance, ids, cache)
    assert torch.equal(first[0], second[0])
    assert cache.snapshot()["hits"] == 1
    instance.config.attention_backend = "eager"
    for layer in instance.layers:
        layer.config.attention_backend = "eager"
    run(instance, ids, cache)
    assert cache.snapshot()["hits"] == 1
