# ruff: noqa: N999 -- Existing public module name.
"""Public batched causal-LM distillation with shared recovery-loss semantics."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.optim import AdamW

from ._model_compat import inputs_requiring_logits


def _positive(value, name, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            or (integer and not isinstance(value, int))):
        kind = "integer" if integer else "finite number"
        raise ValueError(f"{name} must be a positive {kind}")


def _device(device_map):
    if isinstance(device_map, dict):
        devices = set(device_map.values())
        if len(devices) != 1:
            raise ValueError("Public distillation supports one device, not a sharded device_map")
        device_map = devices.pop()
    if device_map == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device_map, int):
        device_map = f"cuda:{device_map}"
    return torch.device(device_map)


def _field(outputs, name):
    return outputs.get(name) if isinstance(outputs, dict) else getattr(outputs, name, None)


def _hidden_size(model):
    return getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))


class DistillationTrainer:
    """Distill compatible causal LMs, preserving the original public API.

    ``alpha_ce`` historically weights **teacher KL**, despite its name; it
    retains that meaning. CE has weight ``1 - alpha_ce - alpha_hidden``.
    KL and CE use supervised next-token means, with the same prompt/pad mask.
    Hidden matching uses those same predictor positions and requires equal
    hidden dimensions; set ``alpha_hidden=0`` for unlike architectures.

    Strings load local/HF checkpoints; injected models require their own
    ``tokenizer`` / ``teacher_tokenizer``. ``mask_prompt=True`` supervises
    assistant turns in conversation/prompt-completion records. Raw ``text``
    trains all nonpadding targets; explicit ``labels=-100`` are always kept.
    """

    def __init__(
        self, teacher, student, temperature: float = 2.0,
        alpha_ce: float = 0.5, alpha_hidden: float = 0.3,
        max_length: int = 1024, device_map="auto", *, tokenizer=None,
        teacher_tokenizer=None, mask_prompt: bool = True,
        trust_remote_code: bool = False, local_files_only: bool = False,
        dtype=torch.float32,
    ):
        _positive(temperature, "temperature")
        _positive(max_length, "max_length", integer=True)
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        for name, value in (("alpha_ce", alpha_ce), ("alpha_hidden", alpha_hidden)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if alpha_ce + alpha_hidden > 1:
            raise ValueError("alpha_ce + alpha_hidden must not exceed 1")
        if type(mask_prompt) is not bool:
            raise TypeError("mask_prompt must be boolean")
        self.temperature, self.alpha_ce, self.alpha_hidden = temperature, alpha_ce, alpha_hidden
        self.max_length, self.mask_prompt = max_length, mask_prompt
        device = _device(device_map)
        self.student, self.tokenizer = self._load(
            student, tokenizer, device, dtype, trust_remote_code, local_files_only)
        self.teacher, self.teacher_tokenizer = None, None
        self._ensure_pad(self.tokenizer)
        if teacher is not None:
            self.teacher, self.teacher_tokenizer = self._load(
                teacher, teacher_tokenizer, device, dtype, trust_remote_code, local_files_only)
            self._ensure_pad(self.teacher_tokenizer)
            from ..compression.RecoveryTrainer import _check_tokenizers

            _check_tokenizers(self.tokenizer, self.teacher_tokenizer)
            if self.tokenizer.pad_token_id != self.teacher_tokenizer.pad_token_id:
                raise ValueError("teacher/student pad_token_id differs")
            student_storage = {(p.device, p.untyped_storage().data_ptr())
                               for p in self.student.parameters() if p.numel()}
            if (self.student is self.teacher or any(
                    (p.device, p.untyped_storage().data_ptr()) in student_storage
                    for p in self.teacher.parameters() if p.numel())):
                raise ValueError("teacher and student must have independent parameter storage")
            student_vocab = getattr(self.student.config, "vocab_size", None)
            teacher_vocab = getattr(self.teacher.config, "vocab_size", None)
            if student_vocab is not None and teacher_vocab is not None and student_vocab != teacher_vocab:
                raise ValueError("teacher/student output vocabulary sizes must agree")
            if (alpha_hidden and _hidden_size(self.student) is not None
                    and _hidden_size(self.teacher) is not None
                    and _hidden_size(self.student) != _hidden_size(self.teacher)):
                raise ValueError("hidden dimensions differ; set alpha_hidden=0 or use explicit projected distillation")
            self.teacher.eval().requires_grad_(False)
        self.history_ = []
        self.metrics_ = {}

    @staticmethod
    def _load(source, tokenizer, device, dtype, trust_remote_code, local_files_only):
        if isinstance(source, torch.nn.Module):
            if tokenizer is None:
                raise ValueError("an injected model requires its tokenizer (teacher_tokenizer for teacher)")
            return source.to(device), tokenizer
        from ..models.ModelLoader import load_causal_lm
        from ..models.TokenizerLoader import load_tokenizer_compatible

        if tokenizer is None:
            tokenizer = load_tokenizer_compatible(source, trust_remote_code=trust_remote_code,
                                                 local_files_only=local_files_only)
        model = load_causal_lm(source, device=device, dtype=dtype,
                              trust_remote_code=trust_remote_code, local_files_only=local_files_only)
        return model, tokenizer

    @staticmethod
    def _ensure_pad(tokenizer):
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("distillation tokenizer requires a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token

    def compute_distillation_loss(self, student_outputs, teacher_outputs, labels, attention_mask=None):
        """Return legacy weighted CE/KL/hidden loss on valid causal targets."""
        from ..compression.RecoveryTrainer import recovery_loss

        logits, teacher_logits = _field(student_outputs, "logits"), _field(teacher_outputs, "logits")
        if logits is None or teacher_logits is None:
            raise ValueError("distillation requires full student and teacher logits")
        teacher_logits = teacher_logits.to(logits.device)
        base_weight = 1 - self.alpha_hidden
        base, components = recovery_loss(
            logits, labels, attention_mask=attention_mask, teacher_logits=teacher_logits,
            temperature=self.temperature, distill_weight=self.alpha_ce / base_weight if base_weight else 0.0)
        loss = base_weight * base
        if self.alpha_hidden:
            student_hidden, teacher_hidden = _field(student_outputs, "hidden_states"), _field(teacher_outputs, "hidden_states")
            if not student_hidden or not teacher_hidden:
                raise ValueError("alpha_hidden > 0 requires student and teacher hidden_states")
            left, right = student_hidden[-1], teacher_hidden[-1].detach().to(logits.device)
            if left.shape != right.shape or left.shape[:2] != labels.shape:
                raise ValueError("hidden-state dimensions must match; set alpha_hidden=0 for unlike architectures")
            valid = labels[:, 1:].ne(-100)
            if attention_mask is not None:
                valid &= attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
            hidden_loss = (left[:, :-1][valid].float() - right[:, :-1][valid].float()).square().mean()
            loss = loss + self.alpha_hidden * hidden_loss
        self._last_tokens = components["tokens"]
        return loss

    def _prepare(self, dataset):
        from ..compression.RecoveryTrainer import (
            _features,
            _format_messages,
            _normalize_record,
        )

        records = [_normalize_record(dict(row)) for row in dataset]
        if not records:
            raise ValueError("distillation dataset must be nonempty")
        if not self.mask_prompt:
            records = [{"text": _format_messages(self.tokenizer, row["messages"], False)}
                       if "messages" in row else row for row in records]
        return _features(records, self.tokenizer, self.max_length)

    def _forward(self, model, batch):
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        return model(**inputs_requiring_logits(model, inputs), output_hidden_states=bool(self.alpha_hidden), use_cache=False)

    def _evaluate_features(self, features, batch_size):
        from ..compression.RecoveryTrainer import _batch, recovery_loss

        was_training = self.student.training
        self.student.eval()
        total, ce_total, tokens = 0.0, 0.0, 0
        try:
            with torch.no_grad():
                for start in range(0, len(features), batch_size):
                    batch = _batch(features[start:start + batch_size], self.tokenizer.pad_token_id,
                                   next(self.student.parameters()).device)
                    student = self._forward(self.student, batch)
                    teacher = self._forward(self.teacher, batch)
                    loss = self.compute_distillation_loss(student, teacher, batch["labels"], batch["attention_mask"])
                    ce, info = recovery_loss(_field(student, "logits"), batch["labels"], attention_mask=batch["attention_mask"])
                    count = info["tokens"]
                    total += float(loss) * count
                    ce_total += float(ce) * count
                    tokens += count
        finally:
            self.student.train(was_training)
        return {"eval_loss": total / tokens, "eval_ce_loss": ce_total / tokens, "eval_tokens": tokens,
                "eval_samples": len(features)}

    def fit(self, train_dataset, eval_dataset=None, epochs: int = 3, batch_size: int = 2,
            learning_rate: float = 5e-5, output_dir: str = "./distill_output", logging_steps: int = 10):
        """Train real batches, evaluate held-out samples before/after each epoch."""
        if self.teacher is None:
            raise ValueError("fit requires a teacher; load() creates an inference-only student")
        for name, value in (("epochs", epochs), ("batch_size", batch_size), ("logging_steps", logging_steps)):
            _positive(value, name, integer=True)
        _positive(learning_rate, "learning_rate")
        train_records = [dict(row) for row in train_dataset]
        validation_records = [dict(row) for row in eval_dataset] if eval_dataset is not None else None
        train = self._prepare(train_records)
        validation = self._prepare(validation_records) if validation_records is not None else None
        self.split_audit_ = None
        if validation is not None:
            def identity(row):
                return tuple(token for token, active in zip(row["input_ids"], row["attention_mask"]) if active)
            if set(map(identity, train)) & set(map(identity, validation)):
                raise ValueError("training and validation overlap after tokenization/truncation")
            from ..data.TrainingSplitAudit import audit_record_splits

            self.split_audit_ = audit_record_splits(
                train_records, validation_records, task="sft", train_features=train,
                validation_features=validation,
                report_path=Path(output_dir) / "split_audit.json",
            )
        from ..compression.RecoveryTrainer import _batch, _check_trainable

        _check_trainable(self.student)
        self.teacher.eval().requires_grad_(False)
        parameters = [p for p in self.student.parameters() if p.requires_grad]
        optimizer = AdamW(parameters, lr=learning_rate)
        self.history_ = []
        before = self._evaluate_features(validation, batch_size) if validation is not None else None
        steps = 0
        for epoch in range(epochs):
            self.student.train()
            weighted_loss, tokens = 0.0, 0
            for start in range(0, len(train), batch_size):
                batch = _batch(train[start:start + batch_size], self.tokenizer.pad_token_id,
                               next(self.student.parameters()).device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_outputs = self._forward(self.teacher, batch)
                student_outputs = self._forward(self.student, batch)
                loss = self.compute_distillation_loss(student_outputs, teacher_outputs,
                                                       batch["labels"], batch["attention_mask"])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite distillation loss")
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                    raise RuntimeError("non-finite distillation gradient")
                optimizer.step()
                count = self._last_tokens
                tokens += count
                weighted_loss += float(loss.detach()) * count
                steps += 1
                if steps % logging_steps == 0:
                    print(f"Distillation epoch {epoch + 1}, step {steps}: loss={float(loss.detach()):.4f}")
            row = {"epoch": epoch + 1, "train_loss": weighted_loss / tokens, "train_tokens": tokens}
            if validation is not None:
                row.update(self._evaluate_features(validation, batch_size))
            self.history_.append(row)
        self.metrics_ = {"steps": steps, "batch_size": batch_size, "train_samples": len(train),
                         "split_audit": self.split_audit_,
                         "validation_before": before, **self.history_[-1]}
        self.save(output_dir)
        (Path(output_dir) / "distillation_metrics.json").write_text(
            json.dumps({"metrics": self.metrics_, "history": self.history_}, indent=2), encoding="utf-8")
        return self

    def evaluate(self, eval_dataset, batch_size: int = 2):
        """Evaluate held-out data without optimizer updates."""
        if self.teacher is None:
            raise ValueError("distillation evaluation requires a teacher")
        _positive(batch_size, "batch_size", integer=True)
        self.teacher.eval()
        return self._evaluate_features(self._prepare(eval_dataset), batch_size)

    def predict(self, text: str, max_new_tokens: int = 100) -> str:
        _positive(max_new_tokens, "max_new_tokens", integer=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {key: value.to(next(self.student.parameters()).device) for key, value in inputs.items()}
        was_training = self.student.training
        self.student.eval()
        try:
            with torch.no_grad():
                outputs = self.student.generate(**inputs, max_new_tokens=max_new_tokens,
                                                pad_token_id=self.tokenizer.pad_token_id)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        finally:
            self.student.train(was_training)

    def save(self, path: str):
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Reload an inference-only student without a dummy teacher download."""
        return cls(teacher=kwargs.pop("teacher", None), student=path, **kwargs)
