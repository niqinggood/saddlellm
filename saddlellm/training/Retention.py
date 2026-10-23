"""Optional reference-model KL for SFT capability retention.

The module is cheap to import for configuration validation. Torch and model
helpers are loaded only when a retention loss is actually computed.
"""
from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
import math
import os
from numbers import Real
from typing import Any


def validate_retention_config(
    reference_model_path: str | None = None,
    retention_kl_weight: float = 0.0,
    retention_temperature: float = 1.0,
) -> bool:
    """Return enabled state; a zero weight never loads a reference model."""
    if (isinstance(retention_kl_weight, bool) or not isinstance(retention_kl_weight, Real)
            or not math.isfinite(retention_kl_weight) or retention_kl_weight < 0):
        raise ValueError("retention_kl_weight must be a finite nonnegative number")
    if (isinstance(retention_temperature, bool) or not isinstance(retention_temperature, Real)
            or not math.isfinite(retention_temperature) or retention_temperature <= 0):
        raise ValueError("retention_temperature must be a finite positive number")
    if reference_model_path is not None and (
        not isinstance(reference_model_path, (str, os.PathLike))
        or not os.fspath(reference_model_path).strip()
    ):
        raise ValueError("reference_model_path must be a nonempty model path")
    return retention_kl_weight > 0


def validate_reference_tokenizers(student: Any, reference: Any) -> None:
    """KL requires identical token IDs, not merely equal vocabulary sizes."""
    if student.get_vocab() != reference.get_vocab():
        raise ValueError("Retention reference and student tokenizers must have identical token-to-ID vocabularies")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(student, name, None) != getattr(reference, name, None):
            raise ValueError(f"Retention reference tokenizer has a different {name}")


def masked_reference_kl(student_logits, reference_logits, labels, *, attention_mask=None, temperature=1.0):
    """Mean KL(reference || student) on shifted supervised nonpadding tokens.

    Prompt labels (-100), padded targets and targets following padding are
    excluded. Teacher probabilities are detached even for direct API callers.
    The T**2 scale preserves the conventional distillation gradient scale.
    """
    import torch.nn.functional as F

    validate_retention_config(retention_temperature=temperature)
    if student_logits.ndim != 3 or reference_logits.shape != student_logits.shape:
        raise ValueError("Retention requires matching [batch, sequence, vocabulary] student/reference logits")
    if tuple(labels.shape) != tuple(student_logits.shape[:2]):
        raise ValueError("Retention labels must match the batch and sequence dimensions of logits")
    mask = labels[:, 1:].ne(-100)
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("Retention attention_mask must match labels")
        mask = mask & attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    if not bool(mask.any()):
        return student_logits[..., :0].float().sum()
    student = student_logits[:, :-1][mask].float() / temperature
    reference = reference_logits.detach()[:, :-1][mask].float() / temperature
    return F.kl_div(
        F.log_softmax(student, dim=-1), F.log_softmax(reference, dim=-1),
        reduction="batchmean", log_target=True,
    ) * temperature ** 2


class ReferenceKLMixin:
    """Compose with an existing HF/native trainer without changing eval loss.

    A separate frozen teacher lives on each process's local device for single
    device and DDP runs. Sharded training must prepare the teacher explicitly
    and is rejected rather than silently moving/sharding it incorrectly.
    """

    def __init__(
        self, *args, reference_model, retention_kl_weight: float,
        retention_temperature: float = 1.0, **kwargs,
    ):
        validate_retention_config(
            retention_kl_weight=retention_kl_weight,
            retention_temperature=retention_temperature,
        )
        super().__init__(*args, **kwargs)
        if getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_deepspeed_enabled", False):
            raise ValueError("SFT reference KL currently supports single-device/DDP, not FSDP or DeepSpeed")
        if getattr(self.args, "n_gpu", 0) > 1:
            raise ValueError("SFT reference KL requires one device per process; use DDP instead of DataParallel")
        self.retention_reference_model = reference_model.eval()
        self.retention_reference_model.requires_grad_(False)
        self.retention_kl_weight = float(retention_kl_weight)
        self.retention_temperature = float(retention_temperature)
        self._retention_kl_sum = 0.0
        self._retention_kl_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not model.training or not self.retention_kl_weight:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        import torch

        from ..distillation._model_compat import inputs_requiring_logits

        if "labels" not in inputs:
            raise ValueError("SFT retention KL requires completion-masked labels")
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask")
        # SaddleTrainer honors this explicit request instead of its chunked
        # loss-only path. Ordinary HF forwards never receive return_logits.
        loss, outputs = super().compute_loss(
            model, inputs_requiring_logits(model, inputs), return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")
        if logits is None:
            raise ValueError("SFT retention KL requires full student logits; disable fused loss-only kernels")
        reference = self.retention_reference_model
        if next(reference.parameters()).device != logits.device:
            reference.to(logits.device)
        reference.eval()
        teacher_inputs = {
            key: value for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids", "token_type_ids", "inputs_embeds"}
        }
        with torch.no_grad():
            teacher_outputs = reference(**inputs_requiring_logits(reference, teacher_inputs))
        teacher_logits = (
            teacher_outputs.logits if hasattr(teacher_outputs, "logits") else teacher_outputs.get("logits")
        )
        if teacher_logits is None:
            raise ValueError("SFT retention KL requires reference model logits")
        kl = masked_reference_kl(
            logits, teacher_logits, labels, attention_mask=attention_mask,
            temperature=self.retention_temperature,
        )
        self._retention_kl_sum += float(kl.detach())
        self._retention_kl_steps += 1
        scaled_kl = kl
        # HF can normalize CE across an entire accumulated batch. Match that
        # denominator so the KL coefficient does not grow with accumulation.
        if num_items_in_batch is not None and (
            getattr(self, "model_accepts_loss_kwargs", False)
            or getattr(self, "compute_loss_func", None) is not None
        ):
            mask = labels[:, 1:].ne(-100)
            if attention_mask is not None:
                mask = mask & attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
            denominator = torch.as_tensor(num_items_in_batch, device=kl.device).clamp_min(1)
            scaled_kl = kl * mask.sum() / denominator
            if getattr(self.args, "average_tokens_across_devices", False):
                scaled_kl = scaled_kl * self.accelerator.num_processes
        loss = loss + self.retention_kl_weight * scaled_kl
        return (loss, outputs) if return_outputs else loss

    def retention_metrics(self) -> dict[str, Any]:
        return {
            "mean_kl": self._retention_kl_sum / max(1, self._retention_kl_steps),
            "loss_calls": self._retention_kl_steps,
            "metrics_scope": "local_rank",
        }


__all__ = [
    "ReferenceKLMixin",
    "masked_reference_kl",
    "validate_reference_tokenizers",
    "validate_retention_config",
]
