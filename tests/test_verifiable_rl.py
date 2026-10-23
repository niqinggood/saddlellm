"""Independent numerical checks for DAPO weighting and PPO value targets."""

from types import SimpleNamespace

import pytest
import torch

from saddlellm.alignment.DAPOTrainer import DAPOConfig, DAPOTrainer
from saddlellm.alignment.VerifiablePPO import (
    VerifiablePPOConfig,
    VerifiablePPOTrainer,
    generalized_advantages,
)
from saddlellm.experiments.TrainOnlyRetrieval import RetrievalDocument, TrainOnlyBM25
from saddlellm.training.PeftDispatch import unquantized_lora_dispatch


def test_dense_peft_dispatch_restores_optional_backend_after_error():
    import peft.tuners.lora.awq as dispatch
    previous = dispatch.is_auto_awq_available
    with pytest.raises(RuntimeError, match="test"):
        with unquantized_lora_dispatch(SimpleNamespace(config=SimpleNamespace())):
            assert not dispatch.is_auto_awq_available()
            raise RuntimeError("test")
    assert dispatch.is_auto_awq_available is previous
    with unquantized_lora_dispatch(SimpleNamespace(is_quantized=True)):
        assert dispatch.is_auto_awq_available is previous


def test_dpo_warm_start_reference_is_frozen_initial_adapter_not_disabled_adapter():
    from saddlellm.training.Preference import _initial_adapter_reference
    model = torch.nn.Linear(2, 1)
    reference = _initial_adapter_reference(model, method="dpo", continuing_adapter=True)
    initial = reference.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1)
    assert torch.equal(reference.weight, initial)
    assert not reference.training
    assert not any(p.requires_grad for p in reference.parameters())
    assert _initial_adapter_reference(model, method="dpo", continuing_adapter=False) is None


def test_dapo_asymmetric_clip_and_token_weighted_gradient():
    trainer = object.__new__(DAPOTrainer)
    trainer.config = DAPOConfig(max_new_tokens=4, overlong_buffer=2)
    assert trainer._clip_policy_ratio(
        torch.tensor([0.5, 1.25, 1.5])
    ).tolist() == pytest.approx([0.8, 1.25, 1.28])
    short, long = (
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(4.0, requires_grad=True),
    )
    loss = trainer._aggregate_sequence_losses([short, long], [1, 3])
    assert float(loss) == 3.5
    loss.backward()
    assert float(short.grad) == 0.25 and float(long.grad) == 0.75
    assert trainer._shape_rollout_rewards(
        [1, 1, 1], [range(2), range(3), range(4)]
    ) == [1, 0.5, 0]


def test_dapo_rejects_grpo_settings_disguised_as_dapo():
    with pytest.raises(ValueError, match="beta"):
        DAPOConfig(beta=0.1).validate()
    with pytest.raises(ValueError, match="dynamic_sampling"):
        DAPOConfig(dynamic_sampling=False).validate()


def test_gae_matches_hand_calculated_terminal_returns():
    values = torch.tensor([0.2, 0.3])
    advantages, returns = generalized_advantages(
        torch.tensor([0.0, 1.0]), values, gamma=1.0, gae_lambda=1.0
    )
    assert advantages.tolist() == pytest.approx([0.8, 0.7])
    assert returns.tolist() == pytest.approx([1.0, 1.0])
    advantages, returns = generalized_advantages(
        torch.tensor([0.0, 1.0]), values, gamma=0.9, gae_lambda=0.0
    )
    assert advantages.tolist() == pytest.approx([0.07, 0.7])
    assert returns.tolist() == pytest.approx([0.27, 1.0])


def test_retrieval_rejects_heldout_data_and_excludes_self():
    documents = [
        RetrievalDocument("a", "group-a", "climate remuneration"),
        RetrievalDocument("b", "group-b", "climate scenario"),
    ]
    with pytest.raises(ValueError, match="held-out"):
        TrainOnlyBM25(documents, heldout_groups=["group-b"])
    with pytest.raises(ValueError, match="held-out"):
        TrainOnlyBM25([RetrievalDocument("x", "g", "climate", split="test")])
    index = TrainOnlyBM25(documents)
    assert index.search("remuneration")[0][1].document_id == "a"
    assert index.search("climate", exclude_groups=["group-a"])[0][1].document_id == "b"


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(6, 4)
        self.head = torch.nn.Linear(4, 6)
        self.config = SimpleNamespace(hidden_size=4)

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.head(hidden), hidden_states=[hidden])


def test_ppo_reuses_frozen_rollouts_and_updates_actor_and_learned_critic(tmp_path):
    policy = TinyPolicy()
    config = VerifiablePPOConfig(
        beta=0.0,
        max_prompt_length=4,
        max_new_tokens=2,
        max_sequence_length=6,
        num_generations=2,
        ppo_epochs=2,
        learning_rate=0.01,
    )
    trainer = VerifiablePPOTrainer(
        policy, None, SimpleNamespace(pad_token_id=0, eos_token_id=2), config
    )
    rollout_calls = []

    def rollout(prompts):
        rollout_calls.append(prompts)
        tokens = [torch.tensor([4, 2]), torch.tensor([5, 2])]
        prefix = torch.tensor([[1, 3], [1, 3]])
        logs = []
        for row in tokens:
            ids = torch.cat([prefix[0], row]).unsqueeze(0)
            logs.append(
                trainer._completion_log_probs(policy(ids).logits, ids, 2)
                .squeeze(0)
                .detach()
            )
        return ["right", "wrong"], tokens, logs, prefix, torch.ones_like(prefix), 2

    trainer._generate_responses_with_log_probs = rollout
    before = policy.head.weight.detach().clone()
    metrics = trainer.train(["question"], lambda p, r: float(r == "right"), num_steps=1)
    assert len(rollout_calls) == 1  # Both PPO epochs use the same old policy.
    assert metrics[-1]["optimizer_step"] == 2
    assert metrics[-1]["value_loss"] > 0
    assert trainer.value_head.weight.detach().abs().sum() > 0
    assert not torch.equal(before, policy.head.weight.detach())
