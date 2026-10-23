import torch

from saddlellm.models.OperatorBackends import (
    AttentionBackendRegistry,
    AttentionBackendSpec,
    _flash_attention_available,
)


def _compact_gqa_inputs(dtype=torch.float64):
    torch.manual_seed(17)
    q = torch.randn(2, 4, 5, 8, dtype=dtype)
    k = torch.randn(2, 2, 5, 8, dtype=dtype)
    v = torch.randn(2, 2, 5, 8, dtype=dtype)
    return q, k, v


def test_flash_backend_reports_real_runtime_capability():
    AttentionBackendRegistry.ensure_defaults()
    backends = AttentionBackendRegistry.list_backends()

    assert backends["flash"]["available"] is _flash_attention_available()
    if not backends["flash"]["available"]:
        assert AttentionBackendRegistry.resolve("auto").name == "sdpa"
        assert AttentionBackendRegistry.resolve("flash").name == "sdpa"


def test_sdpa_receives_compact_kv_and_enable_gqa(monkeypatch):
    AttentionBackendRegistry.ensure_defaults()
    q, k, v = _compact_gqa_inputs(dtype=torch.float32)
    observed = {}

    def fake_sdpa(query, key, value, **kwargs):
        observed["query_shape"] = query.shape
        observed["key_shape"] = key.shape
        observed["value_shape"] = value.shape
        observed["enable_gqa"] = kwargs.get("enable_gqa", False)
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    output = AttentionBackendRegistry.run(
        "sdpa",
        q,
        k,
        v,
        None,
        True,
        enable_gqa=True,
    )

    assert output.shape == q.shape
    assert observed == {
        "query_shape": q.shape,
        "key_shape": k.shape,
        "value_shape": v.shape,
        "enable_gqa": True,
    }


def test_sdpa_auto_policy_expands_when_flash_gqa_is_unavailable(monkeypatch):
    AttentionBackendRegistry.ensure_defaults()
    q, k, v = _compact_gqa_inputs(dtype=torch.float32)
    observed = {}

    def fake_sdpa(query, key, value, **kwargs):
        observed["heads"] = (query.shape[-3], key.shape[-3], value.shape[-3])
        observed["enable_gqa"] = kwargs.get("enable_gqa", False)
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(
        "saddlellm.models.OperatorBackends._can_use_flash_gqa",
        lambda *_: False,
    )

    output = AttentionBackendRegistry.run("sdpa", q, k, v, None, True)

    assert output.shape == q.shape
    assert observed == {"heads": (4, 4, 4), "enable_gqa": False}


def test_sdpa_auto_policy_keeps_compact_kv_when_flash_gqa_is_available(monkeypatch):
    AttentionBackendRegistry.ensure_defaults()
    q, k, v = _compact_gqa_inputs(dtype=torch.float32)
    observed = {}

    def fake_sdpa(query, key, value, **kwargs):
        observed["heads"] = (query.shape[-3], key.shape[-3], value.shape[-3])
        observed["enable_gqa"] = kwargs.get("enable_gqa", False)
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(
        "saddlellm.models.OperatorBackends._can_use_flash_gqa",
        lambda *_: True,
    )

    output = AttentionBackendRegistry.run("sdpa", q, k, v, None, True)

    assert output.shape == q.shape
    assert observed == {"heads": (4, 2, 2), "enable_gqa": True}


def test_compact_sdpa_matches_eager_gqa():
    q, k, v = _compact_gqa_inputs(dtype=torch.float32)

    sdpa = AttentionBackendRegistry.run(
        "sdpa", q, k, v, None, True, enable_gqa=True
    )
    eager = AttentionBackendRegistry.run(
        "eager", q, k, v, None, True, enable_gqa=True
    )

    assert torch.allclose(sdpa, eager, atol=1e-5, rtol=1e-5)


def test_legacy_custom_handler_keeps_five_argument_signature(monkeypatch):
    AttentionBackendRegistry.ensure_defaults()
    q, k, v = _compact_gqa_inputs(dtype=torch.float32)
    observed = {}

    def legacy_handler(query, key, value, attn_mask, is_causal):
        observed["heads"] = (query.shape[-3], key.shape[-3], value.shape[-3])
        observed["is_causal"] = is_causal
        return torch.zeros_like(query)

    spec = AttentionBackendSpec(
        name="legacy-test",
        family="test",
        available=True,
    )
    monkeypatch.setitem(AttentionBackendRegistry._registry, spec.name, spec)
    monkeypatch.setitem(
        AttentionBackendRegistry._handlers, spec.name, legacy_handler
    )

    output = AttentionBackendRegistry.run(
        spec.name,
        q,
        k,
        v,
        None,
        True,
        enable_gqa=True,
    )

    assert output.shape == q.shape
    assert observed == {"heads": (4, 4, 4), "is_causal": True}
