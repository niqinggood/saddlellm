"""Hand-derived DPO/GRPO loss, gradient and rollout EOS correctness."""
import math
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from saddlellm.alignment.GRPOTrainer import GRPOConfig, GRPOTrainer
from saddlellm.training.NativePostTraining import NativeDPOTrainer, _sequence_log_probs


class TableLM(torch.nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        values = torch.arange(36, dtype=torch.float32).reshape(6, 6)
        self.table = torch.nn.Parameter(torch.sin(values + offset))
        self.masks = []

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        if attention_mask is not None:
            self.masks.append(attention_mask.detach().clone())
        return SimpleNamespace(logits=self.table[input_ids])


def _log_probability(values, target):
    return values[target] - math.log(sum(math.exp(value) for value in values))


def test_dpo_matches_manual_log_ratio_loss_and_gradient_with_prompt_padding_masks():
    model, reference = TableLM(), TableLM(1.5)
    trainer = object.__new__(NativeDPOTrainer)
    trainer.reference_model, trainer.beta = reference, .2
    inputs = {
        'chosen_input_ids': torch.tensor([[1, 2, 4, 0]]),
        'rejected_input_ids': torch.tensor([[1, 3, 5, 0]]),
        'chosen_labels': torch.tensor([[-100, -100, 4, -100]]),
        'rejected_labels': torch.tensor([[-100, -100, 5, -100]]),
        'chosen_attention_mask': torch.tensor([[1, 1, 1, 0]]),
        'rejected_attention_mask': torch.tensor([[1, 1, 1, 0]]),
    }
    actual = trainer.compute_loss(model, inputs)
    pc = _log_probability(model.table[2].detach().tolist(), 4)
    pr = _log_probability(model.table[3].detach().tolist(), 5)
    rc = _log_probability(reference.table[2].detach().tolist(), 4)
    rr = _log_probability(reference.table[3].detach().tolist(), 5)
    expected_number = math.log1p(math.exp(-.2 * (pc - pr - rc + rr)))
    assert float(actual) == pytest.approx(expected_number, abs=1e-7)
    expected_table = model.table.detach().clone().requires_grad_()
    c = expected_table[2, 4] - expected_table[2].logsumexp(0)
    r = expected_table[3, 5] - expected_table[3].logsumexp(0)
    expected = -torch.nn.functional.logsigmoid(.2 * (c - r - rc + rr))
    expected.backward()
    actual.backward()
    assert torch.allclose(model.table.grad, expected_table.grad, atol=1e-7)
    assert reference.table.grad is None
    assert model.table.grad[[0, 1, 4, 5]].count_nonzero() == 0
    assert model.table.grad[2, 4] < 0 and model.table.grad[3, 5] > 0


def test_sequence_log_probability_sums_only_shifted_completion_and_real_eos():
    logits = torch.randn(2, 5, 6, requires_grad=True)
    labels = torch.tensor([[-100, -100, 4, 2, -100], [-100, -100, 2, -100, -100]])
    actual = _sequence_log_probs(logits, labels)
    expected = torch.stack([logits[0, 1].log_softmax(0)[4] + logits[0, 2].log_softmax(0)[2],
                            logits[1, 1].log_softmax(0)[2]])
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert logits.grad[0, [0, 3, 4]].count_nonzero() == 0
    assert logits.grad[1, [0, 2, 3, 4]].count_nonzero() == 0
    assert logits.grad[0, 2, 2] > 0


def _config(**kwargs):
    return GRPOConfig(max_prompt_length=8, max_new_tokens=4, max_sequence_length=16, **kwargs)


@pytest.mark.parametrize('advantage', [.7, -.7])
def test_grpo_clipped_ratio_and_k3_match_manual_loss_and_gradient(advantage):
    model, reference = TableLM(), TableLM(.9)
    trainer = GRPOTrainer(model, reference, SimpleNamespace(pad_token_id=0, eos_token_id=2),
                          _config(beta=.15, epsilon=.2))
    response = torch.tensor([4, 2])
    positions, targets = [3, 4], [4, 2]
    lp = [_log_probability(model.table[p].detach().tolist(), target) for p, target in zip(positions, targets)]
    rp = [_log_probability(reference.table[p].detach().tolist(), target) for p, target in zip(positions, targets)]
    ratios = [1.5, .5]
    old = torch.tensor([p - math.log(ratio) for p, ratio in zip(lp, ratios)])
    actual, reported_kl = trainer._compute_grpo_loss('', '', advantage, response, old,
        torch.tensor([1, 3]), torch.tensor([1, 1]), 2)
    policy_expected = -sum(min(r * advantage, min(1.2, max(.8, r)) * advantage) for r in ratios) / 2
    kl_expected = sum(math.exp(r - p) - (r - p) - 1 for p, r in zip(lp, rp)) / 2
    assert reported_kl == pytest.approx(kl_expected, abs=1e-7)
    assert float(actual) == pytest.approx(policy_expected + .15 * kl_expected, abs=1e-6)
    expected_table = model.table.detach().clone().requires_grad_()
    logs = torch.stack([expected_table[p, target] - expected_table[p].logsumexp(0) for p, target in zip(positions, targets)])
    ratio = (logs - old).exp()
    difference = torch.tensor(rp) - logs
    expected = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
    expected = expected + .15 * (difference.exp() - difference - 1).mean()
    expected.backward()
    actual.backward()
    assert torch.allclose(model.table.grad, expected_table.grad, atol=1e-6)
    assert reference.table.grad is None
    assert model.table.grad[[0, 1, 2, 5]].count_nonzero() == 0


@pytest.mark.parametrize('estimator', ['k1', 'k2', 'k3'])
def test_sampled_kl_estimators_match_formula_and_gradient(estimator):
    trainer = object.__new__(GRPOTrainer)
    trainer.config = _config(kl_estimator=estimator)
    ratios = torch.tensor([-.5, 0., .7], requires_grad=True)
    actual = trainer._estimate_kl(ratios)
    expected_values = {'k1': [.5, 0, -.7], 'k2': [.125, 0, .245],
                       'k3': [math.exp(-.5) + .5 - 1, 0, math.exp(.7) - .7 - 1]}
    assert actual.tolist() == pytest.approx(expected_values[estimator], abs=1e-7)
    actual.sum().backward()
    expected_gradients = {'k1': [-1, -1, -1], 'k2': [-.5, 0, .7],
                          'k3': [math.exp(-.5) - 1, 0, math.exp(.7) - 1]}
    assert ratios.grad.tolist() == pytest.approx(expected_gradients[estimator], abs=1e-7)


def test_equal_reward_group_has_zero_advantage_and_no_policy_gradient():
    model = TableLM()
    trainer = GRPOTrainer(model, None, SimpleNamespace(pad_token_id=0, eos_token_id=2), _config(beta=0))
    assert trainer._compute_group_advantages([5, 5, 5]) == [0, 0, 0]
    assert trainer._compute_group_advantages([1, 3]) == pytest.approx([-1, 1])
    loss, kl = trainer._compute_grpo_loss('', '', 0, torch.tensor([4, 2]), None,
        torch.tensor([1, 3]), torch.tensor([1, 1]), 2)
    loss.backward()
    assert kl == 0 and float(loss) == 0 and model.table.grad.count_nonzero() == 0


class RolloutTokenizer:
    def __init__(self, pad):
        self.pad_token_id, self.eos_token_id = pad, 2
        self.padding_side = 'right'

    def __call__(self, prompts, **kwargs):
        return BatchEncoding({'input_ids': torch.tensor([[self.pad_token_id, 1, 3], [1, 3, 5]]),
                              'attention_mask': torch.tensor([[0, 1, 1], [1, 1, 1]])})

    def decode(self, tokens, **kwargs):
        return ' '.join(str(token) for token in tokens.tolist() if token not in {self.pad_token_id, self.eos_token_id})


class RolloutLM(TableLM):
    def generate(self, input_ids, **kwargs):
        pad = kwargs['pad_token_id']
        completions = torch.tensor([[4, 2, pad, pad], [2, pad, pad, pad]])
        return SimpleNamespace(sequences=torch.cat([input_ids, completions], dim=1))


@pytest.mark.parametrize('pad', [0, 2])
def test_rollout_keeps_sampled_eos_and_masks_post_eos_padding_in_loss(pad):
    model, tokenizer = RolloutLM(), RolloutTokenizer(pad)
    trainer = GRPOTrainer(model, None, tokenizer, _config(beta=0))
    texts, ids, old, prompt, prompt_mask, width = trainer._generate_responses_with_log_probs(['short', 'long'])
    assert [row.tolist() for row in ids] == [[4, 2], [2]]
    assert [len(row) for row in old] == [2, 1]
    assert old[0].tolist() == pytest.approx([
        _log_probability(model.table[3].detach().tolist(), 4),
        _log_probability(model.table[4].detach().tolist(), 2),
    ], abs=5e-7)
    assert old[1].tolist() == pytest.approx([
        _log_probability(model.table[5].detach().tolist(), 2),
    ], abs=5e-7)
    assert texts == ['4', '']
    assert tokenizer.padding_side == 'right' and model.training
    assert model.masks[0].tolist() == [[0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]]
    for index in range(2):
        loss, _ = trainer._compute_grpo_loss('', texts[index], 1, ids[index], old[index],
            prompt[index], prompt_mask[index], width)
        # Ratio is exactly one: scoring and update must see identical context.
        assert float(loss) == pytest.approx(-1, abs=1e-7)
        loss.backward()
    assert model.table.grad[4, 2] < 0  # genuine terminal EOS remains supervised
    assert model.table.grad[5, 2] < 0  # empty text can still be a sampled EOS
    assert model.table.grad[2].count_nonzero() == 0  # no post-EOS target
