"""Reference KL masks and gradients must match causal SFT supervision."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from saddlellm.training.Retention import (
    ReferenceKLMixin,
    masked_reference_kl,
    validate_reference_tokenizers,
    validate_retention_config,
)


def test_disabled_retention_accepts_unloaded_reference_path():
    assert validate_retention_config("/does/not/exist", 0.0) is False
    assert validate_retention_config(None, 0.2, 2.0) is True


@pytest.mark.parametrize("weight,temperature", [
    (-1, 1), (float("nan"), 1), (float("inf"), 1), (True, 1),
    (0.1, 0), (0.1, -1), (0.1, float("nan")), (0.1, True),
])
def test_retention_rejects_invalid_loss_controls(weight, temperature):
    with pytest.raises(ValueError):
        validate_retention_config(retention_kl_weight=weight, retention_temperature=temperature)


def test_kl_uses_next_token_completion_mask_and_detaches_teacher():
    student = torch.zeros((1, 5, 3), requires_grad=True)
    teacher = torch.tensor([[[3.0, 0.0, -1.0]] * 5], requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, -100, 2]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    actual = masked_reference_kl(student, teacher, labels, attention_mask=attention, temperature=2.0)
    expected = torch.nn.functional.kl_div(
        student[0, 1].log_softmax(-1), (teacher[0, 1].detach() / 2).log_softmax(-1),
        reduction="sum", log_target=True,
    ) * 4
    assert actual.item() == pytest.approx(expected.item())
    actual.backward()
    assert student.grad[0, 1].abs().sum() > 0
    assert torch.equal(student.grad[0, [0, 2, 3, 4]], torch.zeros((4, 3)))
    assert teacher.grad is None


def test_left_padding_and_empty_supervision_have_zero_kl_gradient():
    student = torch.randn((1, 3, 4), requires_grad=True)
    teacher = torch.randn_like(student)
    labels = torch.tensor([[-100, 2, -100]])
    attention = torch.tensor([[0, 1, 1]])
    loss = masked_reference_kl(student, teacher, labels, attention_mask=attention)
    assert loss.item() == 0.0
    loss.backward()
    assert student.grad is not None and student.grad.count_nonzero() == 0


def test_kl_requires_matching_output_vocabulary_and_sequence_shape():
    with pytest.raises(ValueError, match="matching"):
        masked_reference_kl(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), torch.ones(1, 2))
    with pytest.raises(ValueError, match="labels"):
        masked_reference_kl(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 3))


def test_equal_sized_vocabularies_with_different_token_ids_are_rejected():
    class Tokenizer:
        bos_token_id = 0
        eos_token_id = 1
        pad_token_id = 2

        def __init__(self, vocabulary):
            self.vocabulary = vocabulary

        def get_vocab(self):
            return self.vocabulary

    with pytest.raises(ValueError, match="token-to-ID"):
        validate_reference_tokenizers(Tokenizer({"a": 0, "b": 1}), Tokenizer({"a": 1, "b": 0}))


class _ToyModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.values = torch.nn.Parameter(logits.clone())
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, labels=None, return_logits=False):
        self.calls += 1
        return SimpleNamespace(logits=self.values, loss=self.values.sum() * 0 + 2.5)


class _BaseTrainer:
    def __init__(self, model, sharded=False):
        self.model = model
        self.is_fsdp_enabled = sharded
        self.model_accepts_loss_kwargs = False
        self.compute_loss_func = None
        self.args = SimpleNamespace(average_tokens_across_devices=False)
        self.accelerator = SimpleNamespace(num_processes=1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = model(**inputs)
        return (result.loss, result) if return_outputs else result.loss


class _RetentionTrainer(ReferenceKLMixin, _BaseTrainer):
    pass


def _toy_trainer():
    student = _ToyModel(torch.zeros(1, 4, 3))
    teacher = _ToyModel(torch.tensor([[[2.0, -1.0, 0.0]] * 4]))
    trainer = _RetentionTrainer(model=student, reference_model=teacher, retention_kl_weight=0.3)
    inputs = {"input_ids": torch.tensor([[0, 1, 2, 1]]),
              "labels": torch.tensor([[-100, -100, 2, 1]]), "attention_mask": torch.ones(1, 4)}
    return trainer, student, teacher, inputs


def test_mixin_freezes_teacher_adds_training_kl_and_evaluates_plain_ce():
    trainer, student, teacher, inputs = _toy_trainer()
    loss, output = trainer.compute_loss(student, inputs, return_outputs=True)
    assert output.logits is student.values
    assert loss.item() > 2.5
    assert teacher.training is False and teacher.values.requires_grad is False
    loss.backward()
    assert teacher.values.grad is None
    assert student.values.grad[0, 1:].abs().sum() > 0
    student.eval()
    assert trainer.compute_loss(student, inputs).item() == 2.5
    assert teacher.calls == 1
    assert trainer.retention_metrics()["loss_calls"] == 1


def test_kl_respects_accumulated_token_denominator_and_ddp_average():
    trainer, student, teacher, inputs = _toy_trainer()
    kl = masked_reference_kl(student.values, teacher.values, inputs["labels"])
    trainer.model_accepts_loss_kwargs = True
    loss = trainer.compute_loss(student, inputs, num_items_in_batch=8)
    assert loss.item() == pytest.approx(2.5 + 0.3 * kl.item() * 2 / 8)
    trainer.args.average_tokens_across_devices = True
    trainer.accelerator.num_processes = 2
    loss = trainer.compute_loss(student, inputs, num_items_in_batch=8)
    assert loss.item() == pytest.approx(2.5 + 0.3 * kl.item() * 2 / 8 * 2)


def test_sharded_teacher_training_is_rejected_explicitly():
    with pytest.raises(ValueError, match="FSDP or DeepSpeed"):
        _RetentionTrainer(
            model=_ToyModel(torch.zeros(1, 2, 3)), reference_model=_ToyModel(torch.zeros(1, 2, 3)),
            retention_kl_weight=0.1, sharded=True,
        )


def test_data_parallel_teacher_training_is_rejected_explicitly(monkeypatch):
    original = _BaseTrainer.__init__

    def configure_data_parallel(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.args.n_gpu = 2

    monkeypatch.setattr(_BaseTrainer, "__init__", configure_data_parallel)
    with pytest.raises(ValueError, match="DataParallel"):
        _toy_trainer()
