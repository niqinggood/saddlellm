"""Model-level tests for activation checkpointing and DDP-safe MoE graphs."""

from copy import deepcopy

import pytest
import torch

from saddlellm.SaddleModeling import SaddleForCausalLM, SaddleModelConfig, SaddleMoE


def _attention(kind="gqa"):
    return {
        "kind": kind,
        "backend": "eager",
        "mla_cache_mode": "kv",
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": None,
        "q_lora_rank": 0,
        "kv_lora_rank": 0,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "sliding_window": 0,
    }


def _residual(topology="serial"):
    return {
        "topology": topology,
        "attention_scale": 1.0,
        "ffn_scale": 1.0,
        "learnable": False,
        "dropout": 0.0,
        "initialization": "standard",
    }


def _runtime_config(*, aux_loss_free=False, router_aux_loss_coef=0.01):
    return SaddleModelConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_backend="eager",
        layer_configs=[
            {
                "name": "dense-parallel",
                "attention": _attention(),
                "ffn": {"kind": "swiglu", "intermediate_size": 32},
                "residual": _residual("parallel"),
            },
            {
                "name": "sparse-serial",
                "attention": _attention(),
                "ffn": {
                    "kind": "moe",
                    "intermediate_size": 32,
                    "num_experts": 4,
                    "experts_per_token": 1,
                    "expert_intermediate_size": 16,
                    "shared_expert": False,
                    "aux_loss_free": aux_loss_free,
                    "router_aux_loss_coef": router_aux_loss_coef,
                },
                "residual": _residual("serial"),
            },
        ],
    )


def _run_backward(model, input_ids, labels):
    model.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    output.loss.backward()
    gradients = {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }
    return output, gradients


def test_activation_checkpoint_matches_forward_gradients_and_moe_outputs():
    torch.manual_seed(29)
    baseline = SaddleForCausalLM(_runtime_config()).train()
    checkpointed = SaddleForCausalLM(deepcopy(baseline.config)).train()
    checkpointed.load_state_dict(baseline.state_dict(), strict=True)
    checkpointed.gradient_checkpointing_enable()

    input_ids = torch.randint(0, baseline.config.vocab_size, (2, 7))
    labels = input_ids.clone()
    labels[0, 0] = -100
    baseline_output, baseline_gradients = _run_backward(baseline, input_ids, labels)
    checkpoint_output, checkpoint_gradients = _run_backward(
        checkpointed, input_ids, labels
    )

    assert torch.equal(baseline_output.logits, checkpoint_output.logits)
    assert torch.equal(baseline_output.loss, checkpoint_output.loss)
    assert baseline_output.loss_items.keys() == checkpoint_output.loss_items.keys()
    assert len(baseline_output.expert_stats) == len(checkpoint_output.expert_stats) == 1
    for key in baseline_output.expert_stats[0]:
        assert torch.equal(
            baseline_output.expert_stats[0][key],
            checkpoint_output.expert_stats[0][key],
        )
    assert baseline_gradients.keys() == checkpoint_gradients.keys()
    for name in baseline_gradients:
        assert baseline_gradients[name] is not None, name
        assert checkpoint_gradients[name] is not None, name
        assert torch.allclose(
            baseline_gradients[name],
            checkpoint_gradients[name],
            atol=1e-7,
            rtol=1e-6,
        ), name


def test_activation_checkpoint_recomputes_layers_and_rejects_training_cache():
    torch.manual_seed(31)
    model = SaddleForCausalLM(_runtime_config()).train()
    calls = [0 for _ in model.layers]
    hooks = [
        layer.input_layernorm.register_forward_pre_hook(
            lambda _module, _inputs, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, layer in enumerate(model.layers)
    ]
    model.gradient_checkpointing_enable()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 6))
    output = model(input_ids=input_ids, labels=input_ids)
    output.loss.backward()
    for hook in hooks:
        hook.remove()

    assert all(count >= 2 for count in calls)
    with pytest.raises(ValueError, match="use_cache=True.*activation checkpointing"):
        model(input_ids=input_ids, use_cache=True)

    # Checkpointing is training-only, so cached evaluation/generation remains
    # available without requiring callers to toggle the feature off.
    model.eval()
    with torch.no_grad():
        cached = model(input_ids=input_ids, use_cache=True)
    assert len(cached.past_key_values) == len(model.layers)


def test_checkpoint_configuration_requires_non_reentrant_mode():
    model = SaddleForCausalLM(_runtime_config())
    with pytest.raises(ValueError, match="requires use_reentrant=False"):
        model.gradient_checkpointing_enable({"use_reentrant": True})
    with pytest.raises(ValueError, match="context_fn is managed internally"):
        model.gradient_checkpointing_enable({"context_fn": lambda: None})

    model.gradient_checkpointing_enable({"preserve_rng_state": True})
    assert model.gradient_checkpointing
    model.gradient_checkpointing_disable()
    assert not model.gradient_checkpointing


def test_aux_loss_free_router_bias_updates_once_with_checkpoint_recompute():
    torch.manual_seed(37)
    baseline = SaddleForCausalLM(
        _runtime_config(aux_loss_free=True, router_aux_loss_coef=0.0)
    ).train()
    checkpointed = SaddleForCausalLM(deepcopy(baseline.config)).train()
    checkpointed.load_state_dict(baseline.state_dict(), strict=True)
    checkpointed.gradient_checkpointing_enable()
    input_ids = torch.randint(0, baseline.config.vocab_size, (2, 7))

    baseline_output, baseline_gradients = _run_backward(baseline, input_ids, input_ids)
    checkpoint_output, checkpoint_gradients = _run_backward(
        checkpointed, input_ids, input_ids
    )

    baseline_moe = baseline.layers[1].mlp
    checkpoint_moe = checkpointed.layers[1].mlp
    assert torch.equal(baseline_moe.router_bias, checkpoint_moe.router_bias)
    assert torch.equal(baseline_moe.last_expert_load, checkpoint_moe.last_expert_load)
    assert torch.equal(baseline_output.logits, checkpoint_output.logits)
    for name in baseline_gradients:
        assert baseline_gradients[name] is not None, name
        assert checkpoint_gradients[name] is not None, name
        assert torch.allclose(
            baseline_gradients[name],
            checkpoint_gradients[name],
            atol=1e-7,
            rtol=1e-6,
        ), name


def test_moe_unused_experts_keep_zero_gradient_connections_for_ddp_static_graph():
    torch.manual_seed(41)
    model = SaddleForCausalLM(
        _runtime_config(aux_loss_free=False, router_aux_loss_coef=0.0)
    ).train()
    moe = model.layers[1].mlp
    with torch.no_grad():
        # Equal router logits with top_k=1 deterministically route every token
        # to one expert, leaving the remaining experts empty on this rank.
        moe.router.weight.zero_()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    output, _ = _run_backward(model, input_ids, input_ids)

    assert torch.isfinite(output.loss)
    assert int((moe.last_expert_load == 0).sum()) >= 3
    for expert_index, expert in enumerate(moe.experts):
        load = int(moe.last_expert_load[expert_index])
        for name, parameter in expert.named_parameters():
            assert parameter.grad is not None, f"expert {expert_index}.{name}"
            assert torch.isfinite(parameter.grad).all()
            if load == 0:
                assert torch.count_nonzero(parameter.grad) == 0


def test_moe_synchronizes_global_load_for_bias_and_metrics(monkeypatch):
    config = SaddleModelConfig(
        hidden_size=8,
        intermediate_size=16,
        ffn_kind="moe",
        num_experts=4,
        num_experts_per_tok=1,
        expert_intermediate_size=8,
        aux_loss_free=True,
        router_aux_loss_coef=0.0,
    )
    moe = SaddleMoE(config).train()
    peer_load = torch.tensor([1.0, 2.0, 3.0, 4.0])
    observed_local_load = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(value, op=None):
        assert op == torch.distributed.ReduceOp.SUM
        observed_local_load.append(value.clone())
        value.add_(peer_load.to(value))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    _, _, stats = moe(torch.randn(2, 3, config.hidden_size))

    assert len(observed_local_load) == 1
    expected_load = observed_local_load[0] + peer_load
    assert torch.equal(moe.last_expert_load, expected_load)
    assert torch.equal(stats["expert_load"], expected_load)
    expected_bias = -1e-3 * (
        expected_load / expected_load.mean().clamp_min(1.0) - 1.0
    )
    assert torch.allclose(moe.router_bias, expected_bias)

    # Recompute uses the already synchronized snapshot: no second collective
    # and no second mutable router-bias update.
    bias_after_forward = moe.router_bias.clone()
    moe._checkpoint_recomputing = True
    try:
        moe(torch.randn(2, 3, config.hidden_size))
    finally:
        moe._checkpoint_recomputing = False
    assert len(observed_local_load) == 1
    assert torch.equal(moe.router_bias, bias_after_forward)
