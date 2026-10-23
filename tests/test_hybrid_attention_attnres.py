"""Mathematics and lifecycle checks; no frontier-quality/kernel-speed claim."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from saddlellm import (
    ModelBlueprintLab,
    SaddleAttentionResidual,
    SaddleKimiDeltaAttention,
)
from saddlellm.models.InferenceCache import InferenceLayerCache, RecurrentAttentionCache
from saddlellm.models.KimiDeltaAttention import kda_recurrence
from saddlellm.models.ModelBlueprint import ModelBlueprint
from saddlellm.models.ModelBuilder import SaddleModelBuilder
from saddlellm.models.ModelLoader import load_causal_lm
from saddlellm.models.ModelMetrics import ModelMetricsExtractor
from saddlellm.models.NativePrefixCache import NativePrefixCache
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _blueprint(mode="none", kind="gqa", *, block_size=4, layers=5):
    return ModelBlueprintLab.hybrid_kda(
        hidden_size=16,
        layers=layers,
        heads=4,
        kv_heads=2,
        vocab_size=37,
        seq_length=32,
        attention_residuals=mode,
        full_attention_kind=kind,
        residual_block_size=block_size,
    )


def _model(mode="none", kind="gqa", **kwargs):
    torch.manual_seed(173)
    return _blueprint(mode, kind, **kwargs).build_model().eval()


def _kda(conv_size=4):
    torch.manual_seed(193)
    return SaddleKimiDeltaAttention(
        SaddleModelConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=4,
            attention_kind="kda",
            attention_backend="eager",
            kda_conv_size=conv_size,
        )
    ).eval()


def _manual_mix(module, sources):
    # Independent scalar-source implementation, no call to the mixer under test.
    scores = [
        (
            (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.eps))
            * module.key_norm_weight
            * module.query
        ).sum(-1)
        for x in sources
    ]
    weights = torch.stack(scores).softmax(0)
    return sum(x * weights[index].unsqueeze(-1) for index, x in enumerate(sources))


def test_attnres_depth_axis_raw_values_and_zero_query():
    mixer = SaddleAttentionResidual(4).double()
    sources = tuple(torch.randn(2, 3, 4, dtype=torch.double) for _ in range(3))
    actual, weights = mixer(*sources, return_weights=True)
    torch.testing.assert_close(actual, sum(sources) / 3)
    torch.testing.assert_close(
        weights, torch.full((3, 2, 3), 1 / 3, dtype=torch.double)
    )
    with torch.no_grad():
        mixer.query.copy_(torch.tensor([0.3, -0.2, 0.4, 0.1]))
    actual, weights = mixer(*sources, return_weights=True)
    torch.testing.assert_close(actual, _manual_mix(mixer, sources))
    torch.testing.assert_close(weights.sum(0), torch.ones(2, 3, dtype=torch.double))
    changed = list(sources)
    changed[0] = changed[0].clone()
    changed[0][:, -1] += 20
    torch.testing.assert_close(mixer(*changed)[:, :-1], actual[:, :-1])
    assert torch.autograd.gradcheck(mixer, tuple(x.requires_grad_() for x in sources))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attnres_reductions_are_fp32_and_preserve_activation_dtype(dtype):
    mixer = SaddleAttentionResidual(4).to(dtype)
    x = torch.full((1, 2, 4), 200.0, dtype=dtype, requires_grad=True)
    output, weights = mixer(x, x * 2, return_weights=True)
    assert output.dtype == dtype and weights.dtype == torch.float32
    assert torch.isfinite(output).all()
    output.float().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attnres_mixed_embedding_and_amp_branch_dtypes(dtype):
    mixer = SaddleAttentionResidual(4)
    embedding = torch.randn(1, 3, 4, requires_grad=True)
    branch = torch.randn(1, 3, 4, dtype=dtype, requires_grad=True)
    output = mixer(embedding, branch)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, (embedding + branch.float()) / 2)
    output.square().sum().backward()
    assert torch.isfinite(embedding.grad).all() and torch.isfinite(branch.grad).all()


@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_cpu_autocast_attnres_train_with_fp32_embeddings(mode, checkpointing):
    model = _model(mode).train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    ids = torch.randint(4, 37, (2, 7))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(ids, labels=ids)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_recurrence_matches_literal_matrix_delta_rule_and_gradcheck():
    torch.manual_seed(11)
    q, k = (
        F.normalize(torch.randn(1, 3, 2, 2, dtype=torch.double), dim=-1)
        for _ in range(2)
    )
    v = torch.randn(1, 3, 2, 3, dtype=torch.double)
    alpha = torch.rand_like(k) * 0.8 + 0.1
    beta = torch.rand(1, 3, 2, dtype=torch.double)
    initial = torch.randn(1, 2, 2, 3, dtype=torch.double)
    state, expected = initial.clone(), []
    identity = torch.eye(2, dtype=torch.double)
    for index in range(3):
        kt = k[:, index]
        correction = identity - beta[:, index].unsqueeze(-1).unsqueeze(
            -1
        ) * kt.unsqueeze(-1) * kt.unsqueeze(-2)
        state = correction @ (alpha[:, index].unsqueeze(-1) * state)
        state = state + beta[:, index].unsqueeze(-1).unsqueeze(-1) * kt.unsqueeze(
            -1
        ) * v[:, index].unsqueeze(-2)
        expected.append((q[:, index].unsqueeze(-2) @ state).squeeze(-2))
    output, actual_state = kda_recurrence(q, k, v, alpha, beta, initial)
    torch.testing.assert_close(output, torch.stack(expected, dim=1))
    torch.testing.assert_close(actual_state, state)
    inputs = tuple(x.requires_grad_() for x in (q, k, v, alpha, beta, initial))
    assert torch.autograd.gradcheck(kda_recurrence, inputs)


@pytest.mark.parametrize("conv_size", [1, 4])
@pytest.mark.parametrize("cache_kind", ["dynamic", "mutable"])
def test_kda_full_chunked_and_single_token_execution_match(conv_size, cache_kind):
    attention = _kda(conv_size)
    x = torch.randn(2, 9, 16)
    with torch.inference_mode():
        full = attention(x)[0]
        cache = RecurrentAttentionCache(9) if cache_kind == "mutable" else None
        parts, byte_counts = [], []
        for start, end in [(0, 3), (3, 8), (8, 9)]:
            part, cache = attention(
                x[:, start:end], past_key_value=cache, use_cache=True
            )
            parts.append(part)
            data = vars(cache) if cache_kind == "mutable" else cache
            assert data["length"] == end
            assert data["state"].shape == (2, 4, 4, 4)
            byte_counts.append(data["state"].numel() * data["state"].element_size())
        torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-6, rtol=1e-5)
        assert len(set(byte_counts)) == 1
        cache, parts = None, []
        for index in range(x.shape[1]):
            part, cache = attention(
                x[:, index : index + 1], past_key_value=cache, use_cache=True
            )
            parts.append(part)
        torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-6, rtol=1e-5)


def test_kda_padding_is_skipped_and_document_boundaries_reset_all_history():
    attention = _kda()
    x = torch.randn(1, 8, 16)
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0]])
    documents = torch.tensor([[-1, 0, 0, 0, 1, 1, 1, -1]])
    with torch.inference_mode():
        full, cache = attention(
            x, attention_mask=mask, document_ids=documents, use_cache=True
        )
        first = attention(x[:, 1:3])[0]
        second = attention(x[:, 4:7])[0]
        torch.testing.assert_close(full[:, 1:3], first, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(full[:, 4:7], second, atol=1e-6, rtol=1e-5)
        assert torch.count_nonzero(full[:, [0, 3, 7]]) == 0
        _, isolated = attention(x[:, 4:7], use_cache=True)
        torch.testing.assert_close(cache["state"], isolated["state"])
        for actual, expected in zip(cache["conv_states"], isolated["conv_states"]):
            torch.testing.assert_close(actual, expected)
        a, prior = attention(
            x[:, :5],
            attention_mask=mask[:, :5],
            document_ids=documents[:, :5],
            use_cache=True,
        )
        b, _ = attention(
            x[:, 5:],
            attention_mask=mask,
            document_ids=documents,
            past_key_value=prior,
            use_cache=True,
        )
        torch.testing.assert_close(torch.cat((a, b), 1), full, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "mode,block_size", [("full", 4), ("block", 2), ("block", 4), ("block", 6)]
)
def test_full_and_block_model_match_independent_depth_transition(mode, block_size):
    model = _model(mode, block_size=block_size)
    for module in model.modules():
        if isinstance(module, SaddleAttentionResidual):
            assert (
                torch.count_nonzero(module.query) == 0
            )  # post_init preserves uniform initialization
            with torch.no_grad():
                module.query.normal_(std=0.2)
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        sources, partial = [model.embed_tokens(ids)], None
        for index, layer in enumerate(model.layers):
            if mode == "block" and index and (2 * index) % block_size == 0:
                sources.append(partial)
                partial = None
            selected = sources + ([] if partial is None else [partial])
            h = layer.input_layernorm(
                _manual_mix(layer.attention_depth_residual, selected)
            )
            attn = layer.self_attn(h)[0]
            if mode == "full":
                sources.append(attn)
                selected = sources
            else:
                partial = attn if partial is None else partial + attn
                selected = sources + [partial]
            mlp = layer.mlp(
                layer.post_attention_layernorm(
                    _manual_mix(layer.ffn_depth_residual, selected)
                )
            )
            if mode == "full":
                sources.append(mlp)
            else:
                partial = partial + mlp
        selected = sources + ([] if partial is None else [partial])
        expected = model.lm_head(
            model.norm(_manual_mix(model.final_attention_residual, selected))
        )
        torch.testing.assert_close(model(ids).logits, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["none", "full", "block"])
@pytest.mark.parametrize("kind", ["gqa", "mla"])
@pytest.mark.parametrize("cache_kind", ["dynamic", "mutable"])
def test_hybrid_model_chunked_cache_matches_full_logits(mode, kind, cache_kind):
    model = _model(mode, kind)
    ids = torch.randint(4, 37, (2, 8))
    documents = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]]).expand(2, -1)
    caches = (
        [
            RecurrentAttentionCache(8)
            if layer.config.attention_kind == "kda"
            else InferenceLayerCache(8, latent=kind == "mla")
            for layer in model.layers
        ]
        if cache_kind == "mutable"
        else None
    )
    with torch.inference_mode():
        full = model(ids, document_ids=documents).logits
        parts = []
        for start, end in [(0, 2), (2, 5), (5, 8)]:
            result = model(
                ids[:, start:end],
                document_ids=documents[:, :end],
                past_key_values=caches,
                use_cache=True,
            )
            caches = result.past_key_values
            parts.append(result.logits)
            assert {model._past_key_value_length(cache) for cache in caches} == {end}
        torch.testing.assert_close(torch.cat(parts, 1), full, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("mode", ["none", "full", "block"])
@pytest.mark.parametrize("kind", ["gqa", "mla"])
def test_generation_and_interleaved_request_sessions_match_uncached(mode, kind):
    model = _model(mode, kind)
    ids = torch.tensor([[0, 0, 4, 5], [4, 5, 6, 7]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    kwargs = dict(
        max_new_tokens=4, do_sample=False, eos_token_id=[], attention_mask=mask
    )
    reference = model.generate(ids, use_cache=False, **kwargs)
    for cache_kind in ("geometric", "dynamic"):
        torch.testing.assert_close(
            model.generate(ids, cache_implementation=cache_kind, **kwargs),
            reference,
            rtol=0,
            atol=0,
        )
    with model.create_generation_session(ids, **kwargs) as session:
        session.prefill()
        first_state = session.past_key_values[0].state.clone()
        model.generate(
            torch.tensor([[8, 9, 10]]),
            max_new_tokens=2,
            eos_token_id=[],
            do_sample=False,
        )
        torch.testing.assert_close(
            session.past_key_values[0].state, first_state, rtol=0, atol=0
        )
        list(session)
        torch.testing.assert_close(session.sequences(), reference, rtol=0, atol=0)
    assert session.past_key_values is None


@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.parametrize("moe", [False, True])
def test_checkpointing_chunked_loss_and_mtp_gradients_match(mode, moe):
    raw = _blueprint(mode, "mla").to_dict()
    raw["objective"].update(multi_token_prediction=True, mtp_extra_tokens=2)
    if moe:
        raw["ffn"].update(
            kind="moe",
            num_experts=4,
            experts_per_token=2,
            expert_intermediate_size=16,
            aux_loss_free=True,
            router_aux_loss_coef=0.0,
        )
        for layer in raw["layers"]:
            layer["ffn"] = raw["ffn"].copy()
    torch.manual_seed(177)
    reference = ModelBlueprint.from_dict(raw).build_model().train()
    for module in reference.modules():
        if isinstance(module, SaddleAttentionResidual):
            with torch.no_grad():
                module.query.normal_(std=0.2)
    checked = copy.deepcopy(reference)
    reference.config.loss_chunk_size = checked.config.loss_chunk_size = 3
    checked.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    ids = torch.randint(4, 37, (2, 7))
    labels = ids.clone()
    labels[:, :2] = -100
    a = reference(ids, labels=labels)
    b = checked(ids, labels=labels)
    torch.testing.assert_close(a.loss, b.loss)
    assert a.logits is None and b.logits is None
    a.loss.backward()
    b.loss.backward()
    for (name, actual), (other, expected) in zip(
        checked.named_parameters(), reference.named_parameters()
    ):
        assert (
            name == other and actual.grad is not None and expected.grad is not None
        ), name
        assert torch.isfinite(actual.grad).all(), name
        torch.testing.assert_close(
            actual.grad, expected.grad, atol=2e-6, rtol=2e-4, msg=name
        )
    if moe:
        for left, right in zip(checked.layers, reference.layers):
            torch.testing.assert_close(
                left.mlp.router_bias, right.mlp.router_bias, rtol=0, atol=0
            )


@pytest.mark.parametrize("mode", ["none", "full", "block"])
@pytest.mark.parametrize("kind", ["gqa", "mla"])
def test_train_save_reload_continue_and_cache_estimates(tmp_path, mode, kind):
    blueprint = _blueprint(mode, kind)
    model = blueprint.build_model().train()
    assert blueprint.estimate_total_params() == sum(
        parameter.numel() for parameter in model.parameters()
    )
    ids = torch.randint(4, 37, (2, 7))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    query_before = (
        None
        if mode == "none"
        else model.final_attention_residual.query.detach().clone()
    )
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(ids, labels=ids)
        output.loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
    if mode != "none":
        assert not torch.equal(model.final_attention_residual.query, query_before)
    model.eval().save_pretrained(str(tmp_path))
    restored = load_causal_lm(tmp_path, device="cpu").eval()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, atol=0, rtol=0)
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(ids).logits, model(ids).logits, atol=0, rtol=0
        )
    restored.train()(ids, labels=ids).loss.backward()
    assert torch.isfinite(restored.layers[0].self_attn.A_log.grad).all()
    metrics = ModelMetricsExtractor.cache_profile(model)
    assert metrics["mode"] == "mixed"
    assert (
        metrics["bytes_per_token_total"]
        == blueprint.estimate_kv_cache_bytes_per_token()
    )
    assert (
        metrics["recurrent_state_bytes_per_sequence"]
        == blueprint.estimate_recurrent_cache_bytes()
    )
    assert metrics["bytes_per_token_by_layer"] == [
        0,
        0,
        0,
        32 if kind == "gqa" else 2,
        0,
    ]


def test_default_attnres_off_retains_legacy_state_keys_and_legacy_reload():
    config = SaddleModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=37,
    )
    model = SaddleForCausalLM(config)
    assert not any(
        "depth_residual" in key or "final_attention_residual" in key
        for key in model.state_dict()
    )
    raw = config.to_dict()
    for key in (
        "residual_attention_mode",
        "residual_block_size",
        "kda_conv_size",
        "kda_gate_rank",
    ):
        raw.pop(key)
    legacy = SaddleForCausalLM(SaddleModelConfig(**raw))
    legacy.load_state_dict(model.state_dict(), strict=True)


def test_pure_kda_inference_prewarm_and_freeze_do_not_require_rope():
    model = SaddleForCausalLM(
        SaddleModelConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=37,
            attention_kind="kda",
            attention_backend="eager",
        )
    ).eval()
    model.prewarm_rope_cache(16, freeze=True)
    assert not model._rope_cache
    model.unfreeze_rope_cache().clear_rope_cache()
    assert model.generate(
        torch.tensor([[4, 5]]), max_new_tokens=2, eos_token_id=[], do_sample=False
    ).shape == (1, 4)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda raw: raw["residual"].update(attention_mode="unknown"),
            "attention_mode",
        ),
        (
            lambda raw: raw["residual"].update(attention_mode="block", block_size=3),
            "even",
        ),
        (
            lambda raw: raw["residual"].update(
                attention_mode="full", topology="parallel"
            ),
            "serial",
        ),
        (lambda raw: raw["attention"].update(kind="kda", backend="sdpa"), "eager"),
        (
            lambda raw: raw["attention"].update(
                kind="kda", backend="eager", fused_qkv=True
            ),
            "fused_qkv",
        ),
        (
            lambda raw: raw["attention"].update(
                kind="kda", backend="eager", qk_norm=True
            ),
            "qk_norm",
        ),
        (
            lambda raw: raw["attention"].update(
                kind="kda", backend="eager", sliding_window=2
            ),
            "sliding_window",
        ),
        (lambda raw: raw["attention"].update(kda_conv_size=0), "kda_conv_size"),
    ],
)
def test_invalid_architectures_are_rejected(mutation, match):
    raw = _blueprint().to_dict()
    raw["layers"] = []
    mutation(raw)
    with pytest.raises(ValueError, match=match):
        ModelBlueprint.from_dict(raw)


def test_per_layer_attnres_mode_conflict_and_legacy_conversion_fail_explicitly():
    raw = _blueprint("block").to_dict()
    raw["layers"][0]["residual"]["attention_mode"] = "full"
    with pytest.raises(ValueError, match="match the model-level"):
        ModelBlueprint.from_dict(raw)
    for blueprint in (_blueprint(), _blueprint("block")):
        with pytest.raises(ValueError, match="KDA|AttnRes"):
            blueprint.to_model_spec()
        with pytest.raises(ValueError, match="KDA|AttnRes"):
            blueprint.to_transformers_config_dict()


def test_builder_supports_kda_and_attnres_named_presets():
    blueprint = (
        SaddleModelBuilder(
            "named-presets", hidden_size=16, vocab_size=37, max_position_embeddings=32
        )
        .defaults(
            attention={"preset": "gqa", "num_heads": 4, "num_kv_heads": 2},
            ffn={"intermediate_size": 32},
            residual="attnres_block",
        )
        .add_layers(3, attention={"preset": "kda", "num_heads": 4})
        .add_layer(attention={"preset": "gqa", "num_heads": 4, "num_kv_heads": 2})
        .build_blueprint()
    )
    assert blueprint.residual.attention_mode == "block"
    assert [layer.attention.kind for layer in blueprint.expanded_layers()] == [
        "kda",
        "kda",
        "kda",
        "gqa",
    ]
    assert blueprint.expanded_layers()[0].attention.backend == "eager"
    assert blueprint.build_model()(torch.tensor([[4, 5]])).logits.shape == (1, 2, 37)


def test_reports_do_not_call_experimental_hybrid_low_risk_dense_baseline():
    blueprint = _blueprint("block")
    assert any("KDA" in risk for risk in blueprint.risk_report())
    assert any("Attention Residuals" in risk for risk in blueprint.risk_report())
    assert not any("low-risk" in risk for risk in blueprint.risk_report())
    assert any("hybrid KDA" in idea for idea in blueprint.training_hypotheses())
    assert any("AttnRes" in idea for idea in blueprint.training_hypotheses())


@pytest.mark.parametrize("mode", ["full", "block"])
def test_attnres_layers_preserve_standard_module_call_hooks(mode):
    model = _model(mode)
    calls = []
    handles = []
    for index, layer in enumerate(model.layers):
        handles.append(
            layer.register_forward_pre_hook(
                lambda module, args, layer_index=index: calls.append(
                    ("pre", layer_index)
                )
            )
        )
        handles.append(
            layer.register_forward_hook(
                lambda module, args, output, layer_index=index: calls.append(
                    ("post", layer_index)
                )
            )
        )
    try:
        with torch.inference_mode():
            model(torch.tensor([[4, 5]]))
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [
        (event, index)
        for index in range(len(model.layers))
        for event in ("pre", "post")
    ]


def test_direct_attnres_layer_call_requires_explicit_depth_history():
    layer = _model("block").layers[0]
    with pytest.raises(ValueError, match="depth source history"):
        layer(torch.randn(1, 3, 16))


@pytest.mark.parametrize("value", [0, -1, True, 4.0])
def test_zero_or_invalid_explicit_head_dimensions_are_not_silently_ignored(value):
    raw = _blueprint().to_dict()
    raw["layers"] = []
    raw["attention"]["head_dim"] = value
    with pytest.raises((ValueError, TypeError), match="head_dim"):
        ModelBlueprint.from_dict(raw)
    with pytest.raises(ValueError, match="head_dim"):
        SaddleModelConfig(head_dim=value)


def test_unknown_direct_config_attention_is_not_silently_gqa():
    with pytest.raises(ValueError, match="unsupported attention_kind"):
        SaddleModelConfig(attention_kind="kda_typo")


def test_kda_odd_head_dimension_does_not_require_rope_even_dimensions():
    raw = _blueprint().to_dict()
    raw["layers"] = []
    raw["attention"].update(kind="kda", backend="eager", head_dim=3)
    blueprint = ModelBlueprint.from_dict(raw)
    model = blueprint.build_model()
    assert blueprint.estimate_total_params() == sum(
        p.numel() for p in model.parameters()
    )
    assert torch.isfinite(model(torch.tensor([[4, 5]])).logits).all()


def test_cached_mla_document_position_failure_does_not_mutate_hybrid_caches():
    model = _model("block", "mla")
    ids = torch.tensor([[4, 5]])
    documents = torch.tensor([[0, 1]])
    caches = [
        RecurrentAttentionCache(8)
        if layer.config.attention_kind == "kda"
        else InferenceLayerCache(8, latent=True)
        for layer in model.layers
    ]
    with torch.inference_mode():
        with pytest.raises(ValueError, match="must follow document boundaries"):
            model(
                ids,
                document_ids=documents,
                position_ids=torch.tensor([[0, 1]]),
                past_key_values=caches,
                use_cache=True,
            )
    assert all(cache.length == 0 for cache in caches)


def test_cache_budget_invalid_state_and_hybrid_length_fail_before_mutation():
    attention = _kda()
    x = torch.randn(1, 3, 16)
    with pytest.raises(RuntimeError, match="gradients disabled"):
        attention(x, use_cache=True)
    with torch.inference_mode():
        cache = RecurrentAttentionCache(3)
        attention(x, past_key_value=cache, use_cache=True)
        state = cache.state.clone()
        with pytest.raises(ValueError, match="budget"):
            attention(x[:, :1], past_key_value=cache, use_cache=True)
        torch.testing.assert_close(cache.state, state, atol=0, rtol=0)
        bad = {"cache_type": "kda", "length": 2}
        with pytest.raises(ValueError, match="matrix and convolution"):
            attention(x[:, :1], past_key_value=bad, use_cache=True)
        bad = dict(cache_type="kda", **vars(cache))
        bad["state"] = torch.zeros(1, 4, 4, 5)
        with pytest.raises(ValueError, match="state shape"):
            attention(x[:, :1], past_key_value=bad, use_cache=True)
        with pytest.raises(ValueError, match="2D padding"):
            attention(x, attention_mask=torch.zeros(1, 1, 3, 3))
        with pytest.raises(ValueError, match="only 0 or 1"):
            attention(x, attention_mask=torch.full((1, 3), 0.5))
        _, doc_cache = attention(
            x, document_ids=torch.zeros(1, 3, dtype=torch.long), use_cache=True
        )
        with pytest.raises(ValueError, match="segmentation cannot change"):
            attention(x[:, :1], past_key_value=doc_cache, use_cache=True)
        model = _model()
        caches = [RecurrentAttentionCache(8) for _ in model.layers]
        caches[0].length = 1
        with pytest.raises(ValueError, match="equal input-slot"):
            model(torch.tensor([[4, 5]]), past_key_values=caches, use_cache=True)


@pytest.mark.parametrize("mode", ["full", "block"])
def test_attnres_without_kda_trains_and_generates_and_bypasses_prefix_snapshots(mode):
    raw = _blueprint(mode).to_dict()
    raw["layers"] = []
    raw["num_layers"] = 3
    model = ModelBlueprint.from_dict(raw).build_model().train()
    ids = torch.tensor([[4, 5, 6, 7]])
    model(ids, labels=ids).loss.backward()
    assert torch.isfinite(model.final_attention_residual.query.grad).all()
    model.eval()
    kwargs = dict(max_new_tokens=3, do_sample=False, eos_token_id=[])
    expected = model.generate(ids, use_cache=False, **kwargs)
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    for _ in range(2):
        with model.create_generation_session(
            ids, prefix_cache=cache, **kwargs
        ) as session:
            session.prefill()
            assert session.prefix_cached_tokens == 0
            list(session)
            torch.testing.assert_close(session.sequences(), expected, rtol=0, atol=0)
    assert cache.snapshot()["entries"] == 0


def test_kda_cache_cannot_be_accidentally_reused_for_autograd_training():
    attention = _kda()
    x = torch.randn(1, 3, 16)
    with torch.inference_mode():
        _, cache = attention(x, use_cache=True)
    with pytest.raises(RuntimeError, match="gradients disabled"):
        attention(x[:, :1].requires_grad_(), past_key_value=cache, use_cache=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_low_precision_kda_cache_and_backward_are_finite(dtype):
    attention = _kda().cuda().to(dtype)
    x = torch.randn(2, 6, 16, device="cuda", dtype=dtype, requires_grad=True)
    attention(x)[0].float().square().sum().backward()
    assert torch.isfinite(x.grad).all()
    with torch.inference_mode():
        full = attention(x)[0]
        a, cache = attention(x[:, :3], use_cache=True)
        b, cache = attention(x[:, 3:], past_key_value=cache, use_cache=True)
        assert cache["state"].dtype == torch.float32
        assert cache["conv_states"][0].dtype == dtype
        torch.testing.assert_close(torch.cat((a, b), 1), full, atol=2e-3, rtol=2e-2)
