from types import SimpleNamespace

import torch

from saddlellm.distillation._model_compat import inputs_requiring_logits
from saddlellm.distillation.distill import DistillationTrainer as TrainerDistiller
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def _native_model() -> SaddleForCausalLM:
    return SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=19,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            attention_backend="eager",
            loss_chunk_size=2,
        )
    )


class _StrictHuggingFaceStyleModel(torch.nn.Module):
    """Small stand-in that rejects unknown forward keyword arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 7)

    def forward(self, input_ids, labels=None):
        logits = self.projection(torch.nn.functional.one_hot(input_ids, 4).float())
        loss = logits.sum() if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


class _ForwardingWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _ClosedWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, input_ids, labels=None):
        return self.module(input_ids=input_ids, labels=labels)


def test_logits_request_is_native_only_and_survives_common_wrapper():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    labels = input_ids.clone()
    native = _ForwardingWrapper(_native_model().train())

    native_inputs = inputs_requiring_logits(
        native, {"input_ids": input_ids, "labels": labels}
    )
    assert native_inputs["return_logits"] is True
    assert native(**native_inputs).logits.shape == (1, 4, 19)

    closed = _ClosedWrapper(_native_model().train())
    closed_inputs = inputs_requiring_logits(
        closed, {"input_ids": input_ids, "labels": labels}
    )
    assert "return_logits" not in closed_inputs

    strict_hf = _StrictHuggingFaceStyleModel().train()
    hf_inputs = inputs_requiring_logits(
        strict_hf, {"input_ids": input_ids, "labels": labels}
    )
    assert "return_logits" not in hf_inputs
    assert strict_hf(**hf_inputs).logits.shape == (1, 4, 7)


def test_trainer_distillation_keeps_native_logits_with_chunked_loss_enabled():
    torch.manual_seed(457)
    student = _native_model().train()
    teacher = _native_model().eval()
    distiller = object.__new__(TrainerDistiller)
    distiller.teacher = teacher
    distiller.temperature = 2.0
    distiller.alpha = 0.5
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    loss, outputs = distiller.compute_loss(
        student,
        {"input_ids": input_ids, "labels": input_ids.clone()},
        return_outputs=True,
    )

    assert outputs.logits.shape == (2, 4, student.config.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.lm_head.weight.grad is not None


def test_trainer_distillation_preserves_labels_through_forwarding_wrappers():
    student = _ForwardingWrapper(_native_model().train())
    teacher = _ForwardingWrapper(_native_model().eval())
    distiller = object.__new__(TrainerDistiller)
    distiller.teacher = teacher
    distiller.temperature = 2.0
    distiller.alpha = 0.5
    input_ids = torch.tensor([[1, 2, 3, 4]])

    loss, outputs = distiller.compute_loss(
        student,
        {"input_ids": input_ids, "labels": input_ids.clone()},
        return_outputs=True,
    )

    assert outputs.logits.shape == (1, 4, student.module.config.vocab_size)
    assert torch.isfinite(loss)


def test_trainer_distillation_does_not_send_saddle_keyword_to_hf_model():
    student = _StrictHuggingFaceStyleModel().train()
    teacher = _StrictHuggingFaceStyleModel().eval()
    distiller = object.__new__(TrainerDistiller)
    distiller.teacher = teacher
    distiller.temperature = 2.0
    distiller.alpha = 0.5
    input_ids = torch.tensor([[0, 1, 2, 3]])

    loss, outputs = distiller.compute_loss(
        student,
        {"input_ids": input_ids, "labels": input_ids.clone()},
        return_outputs=True,
    )

    assert outputs.logits.shape == (1, 4, 7)
    assert torch.isfinite(loss)
