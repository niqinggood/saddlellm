"""Behavioral checks for typed optimizer phases and held-out test isolation."""

import importlib
import random
import sys

import numpy as np
import pytest
import torch

from saddlellm.decision.TypedModel import TypedDecisionModel
from saddlellm.decision.TypedSchema import (
    TypedDecisionConfig,
    read_typed_dataset,
    scope_key,
    validate_record,
)
from tools.prepare_decision_typed import prepare

training = importlib.import_module("saddlellm.decision.TypedTraining")


def model_config(**overrides):
    return TypedDecisionConfig(hidden_size=8, num_layers=1, num_heads=2, **overrides)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backbone_learning_rate", 0),
        ("backbone_learning_rate", -1),
        ("backbone_learning_rate", True),
        ("backbone_learning_rate", float("nan")),
        ("head_learning_rate", float("inf")),
        ("head_learning_rate", "0.01"),
        ("head_warmup_epochs", -1),
        ("head_warmup_epochs", True),
        ("head_warmup_epochs", 1.5),
        ("early_stopping_patience", -1),
        ("early_stopping_patience", False),
        ("evaluate_test", 1),
        ("evaluate_test", "false"),
    ],
)
def test_controls_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        training.TypedTrainingConfig(**{field: value})


def legacy_one_epoch_state(directory, model_cfg, config):
    """Execute the former flat-AdamW path as an independent numerical reference."""
    _, splits, _ = read_typed_dataset(directory)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    model = TypedDecisionModel(model_cfg)
    parameters = [p for p in model.parameters() if p.requires_grad]
    masters = (
        [torch.nn.Parameter(p.detach().float().cpu().clone()) for p in parameters]
        if config.optimizer_device == "cpu"
        else parameters
    )
    optimizer = torch.optim.AdamW(
        masters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    initial = training.collect_logits(model, splits["validation"], config.batch_size)
    best_nll = training._nll(initial, splits["validation"])
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.train()
    indices = torch.randperm(
        len(splits["train"]), generator=torch.Generator().manual_seed(config.seed)
    ).tolist()
    for start in range(0, len(indices), config.batch_size):
        batch = [
            splits["train"][index]
            for index in indices[start : start + config.batch_size]
        ]
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        loss = training.supervised_loss(
            model(batch),
            batch,
            brier_weight=config.brier_weight,
            ordinal_weight=config.ordinal_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if config.optimizer_device == "cpu":
            for parameter, master in zip(parameters, masters):
                master.grad = (
                    parameter.grad.detach().float().cpu()
                    if parameter.grad is not None
                    else None
                )
        optimizer.step()
        if config.optimizer_device == "cpu":
            with torch.no_grad():
                for parameter, master in zip(parameters, masters):
                    parameter.copy_(master)
    final = training.collect_logits(model, splits["validation"], config.batch_size)
    if training._nll(final, splits["validation"]) < best_nll:
        state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
    return state


@pytest.mark.parametrize("optimizer_device", ["model", "cpu"])
def test_defaults_preserve_flat_optimizer_weights_exactly(tmp_path, optimizer_device):
    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    cfg = training.TypedTrainingConfig(
        epochs=1,
        batch_size=3,
        learning_rate=0.005,
        cpu_threads=1,
        optimizer_device=optimizer_device,
    )
    expected = legacy_one_epoch_state(directory, model_config(), cfg)
    report = training.train_typed(
        directory,
        tmp_path / "run",
        model_config=model_config(),
        config=cfg,
        device="cpu",
    )
    restored = TypedDecisionModel.from_pretrained(tmp_path / "run/model")
    for key, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert report["optimizer_groups"][0]["name"] == "all_trainable"
    assert report["epochs_completed"] == 1
    assert report["test_evaluated"] is True
    assert report["artifact_reload_split"] == "test"


@pytest.mark.parametrize("optimizer_device", ["model", "cpu"])
def test_separate_rates_change_actual_adam_steps(
    tmp_path, monkeypatch, optimizer_device
):
    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    original = torch.optim.AdamW
    first_updates = {}

    def factory(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        step = optimizer.step

        def capture(*step_args, **step_kwargs):
            observed = []
            for group in optimizer.param_groups:
                if group["name"] in first_updates:
                    continue
                parameter = next(
                    p
                    for p in group["params"]
                    if p.grad is not None and (p.grad.abs() > 1e-6).any()
                )
                observed.append(
                    (
                        group["name"],
                        parameter,
                        parameter.detach().clone(),
                        parameter.grad.abs() > 1e-6,
                    )
                )
            result = step(*step_args, **step_kwargs)
            for name, parameter, before, mask in observed:
                first_updates[name] = float(
                    (parameter.detach() - before).abs()[mask].mean()
                )
            return result

        optimizer.step = capture
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", factory)
    report = training.train_typed(
        directory,
        tmp_path / "run",
        model_config=model_config(),
        device="cpu",
        config=training.TypedTrainingConfig(
            epochs=1,
            batch_size=9,
            cpu_threads=1,
            weight_decay=0,
            backbone_learning_rate=2e-5,
            head_learning_rate=0.02,
            optimizer_device=optimizer_device,
        ),
    )
    assert {
        group["name"]: group["learning_rate"] for group in report["optimizer_groups"]
    } == {"backbone": 2e-5, "head": 0.02}
    assert first_updates["backbone"] > 0
    assert first_updates["head"] > first_updates["backbone"] * 100


@pytest.mark.parametrize("optimizer_device", ["model", "cpu"])
def test_warmup_freezes_then_unfreezes_backbone(
    tmp_path, monkeypatch, optimizer_device
):
    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    original = TypedDecisionModel.forward
    training_modes = []

    def capture(self, records):
        if self.training:
            training_modes.append(
                (
                    self.backbone.training,
                    any(p.requires_grad for p in self.backbone.parameters()),
                )
            )
        return original(self, records)

    monkeypatch.setattr(TypedDecisionModel, "forward", capture)
    report = training.train_typed(
        directory,
        tmp_path / "run",
        model_config=model_config(),
        device="cpu",
        config=training.TypedTrainingConfig(
            epochs=2,
            batch_size=9,
            cpu_threads=1,
            learning_rate=0.005,
            head_warmup_epochs=1,
            backbone_learning_rate=0.001,
            head_learning_rate=0.005,
            optimizer_device=optimizer_device,
        ),
    )
    history = report["selection"]["history"]
    assert [row["phase"] for row in history] == ["head_warmup", "joint"]
    assert history[0]["backbone_probe_delta_l2"] == 0
    assert history[1]["backbone_probe_delta_l2"] > 0
    assert training_modes == [(False, False), (True, True)]
    assert history[0]["train_by_type"]["choice"]["nll"] > 0
    assert history[0]["validation_by_type"]["score"]["score"]["mae_argmax_value"] >= 0


def test_permanent_freeze_is_never_reversed_by_warmup(tmp_path):
    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    report = training.train_typed(
        directory,
        tmp_path / "run",
        model_config=model_config(freeze_backbone=True),
        device="cpu",
        config=training.TypedTrainingConfig(
            epochs=2,
            batch_size=9,
            cpu_threads=1,
            head_warmup_epochs=1,
            head_learning_rate=0.005,
        ),
    )
    assert [row["phase"] for row in report["selection"]["history"]] == [
        "head_only",
        "head_only",
    ]
    assert report["backbone_trainable_parameters"] == 0
    assert report["backbone_update_probe"]["optimizer_delta_l2"] == 0
    assert [group["name"] for group in report["optimizer_groups"]] == ["head"]


def test_cpu_groups_match_regular_optimizer_across_warmup_transition(tmp_path):
    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    for device in ("model", "cpu"):
        training.train_typed(
            directory,
            tmp_path / device,
            model_config=model_config(),
            device="cpu",
            config=training.TypedTrainingConfig(
                epochs=2,
                batch_size=9,
                cpu_threads=1,
                head_warmup_epochs=1,
                backbone_learning_rate=0.001,
                head_learning_rate=0.005,
                optimizer_device=device,
                evaluate_test=False,
            ),
        )
    regular = TypedDecisionModel.from_pretrained(tmp_path / "model/model")
    cpu = TypedDecisionModel.from_pretrained(tmp_path / "cpu/model")
    for key, value in regular.state_dict().items():
        torch.testing.assert_close(value, cpu.state_dict()[key], rtol=0, atol=0)


def test_early_stopping_retains_initial_checkpoint_when_validation_worsens(
    tmp_path, monkeypatch
):
    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    original = training.collect_logits
    validation_calls = 0

    def worsening(model, rows, batch_size):
        nonlocal validation_calls
        values = original(model, rows, batch_size)
        if rows[0]["id"].startswith("validation-"):
            strength = 2 - validation_calls
            validation_calls += 1
            return [np.asarray(row["target"]) * strength for row in rows]
        return values

    monkeypatch.setattr(training, "collect_logits", worsening)
    report = training.train_typed(
        directory,
        tmp_path / "run",
        model_config=model_config(),
        device="cpu",
        config=training.TypedTrainingConfig(
            epochs=5,
            batch_size=9,
            cpu_threads=1,
            early_stopping_patience=2,
            evaluate_test=False,
        ),
    )
    assert report["epochs_completed"] == 2
    assert report["early_stopping"]["stopped_early"] is True
    assert report["selection"]["best_epoch"] == 0
    assert report["backbone_update_probe"]["selected_checkpoint_delta_l2"] == 0


def test_early_stopping_waits_until_requested_head_warmup_finishes(
    tmp_path, monkeypatch
):
    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    original = training.collect_logits
    validation_calls = 0

    def worsening(model, rows, batch_size):
        nonlocal validation_calls
        values = original(model, rows, batch_size)
        if rows[0]["id"].startswith("validation-"):
            strength = 2 - validation_calls
            validation_calls += 1
            return [np.asarray(row["target"]) * strength for row in rows]
        return values

    monkeypatch.setattr(training, "collect_logits", worsening)
    report = training.train_typed(
        directory,
        tmp_path / "run",
        model_config=model_config(),
        device="cpu",
        config=training.TypedTrainingConfig(
            epochs=5,
            batch_size=9,
            cpu_threads=1,
            head_warmup_epochs=2,
            early_stopping_patience=1,
            evaluate_test=False,
        ),
    )
    assert report["epochs_completed"] == 3
    assert [row["phase"] for row in report["selection"]["history"]] == [
        "head_warmup",
        "head_warmup",
        "joint",
    ]
    assert report["selection"]["history"][2]["backbone_probe_delta_l2"] > 0


def test_no_test_evaluation_does_not_tokenize_or_forward_test(tmp_path, monkeypatch):
    import json

    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    test_path = directory / "test.jsonl"
    raw = [
        json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()
    ]
    raw[0]["state"] = "x" * 300
    test_path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8"
    )
    original = TypedDecisionModel.forward

    def prohibit(self, records):
        assert not any(row["id"].startswith("test-") for row in records)
        return original(self, records)

    monkeypatch.setattr(TypedDecisionModel, "forward", prohibit)
    report = training.train_typed(
        directory,
        tmp_path / "run",
        model_config=model_config(),
        device="cpu",
        config=training.TypedTrainingConfig(
            epochs=1, batch_size=9, cpu_threads=1, evaluate_test=False
        ),
    )
    assert report["test_evaluated"] is False
    assert (
        report["test"] == report["test_by_type"] == report["test_by_task_scope"] == {}
    )
    assert report["artifact_reload_split"] == "validation"
    assert report["artifact_reload_verified"] is True
    assert report["validation"]["trained_uncalibrated"]["samples"] == 9


def test_class_diagnostics_align_permuted_candidates_and_score_mae():
    base = {
        "state": "text",
        "state_id": "s",
        "task_id": "t",
        "question": "Which?",
        "type": "choice",
    }
    rows = [
        validate_record(
            {
                **base,
                "id": "a",
                "choices": [{"id": "r", "text": "red"}, {"id": "b", "text": "blue"}],
                "label": "r",
            }
        ),
        validate_record(
            {
                **base,
                "id": "b",
                "choices": [{"id": "b", "text": "blue"}, {"id": "r", "text": "red"}],
                "label": "b",
            }
        ),
        validate_record(
            {
                "id": "c",
                "state": "text",
                "state_id": "s",
                "task_id": "score",
                "question": "Rate",
                "type": "score",
                "levels": [
                    {"id": "low", "text": "low", "value": 0},
                    {"id": "mid", "text": "mid", "value": 1},
                    {"id": "high", "text": "high", "value": 2},
                ],
                "label": "mid",
            }
        ),
    ]
    result = training.typed_metrics([[-4, 4], [3, 0], [3, 0, 0]], rows)
    item = result["class_diagnostics"][scope_key(rows[0])]
    assert item["prediction_counts"] == {"b": 2, "r": 0}
    assert item["recall_by_label"] == {"b": 1.0, "r": 0.0}
    assert item["confusion"]["matrix"] == [[1, 0], [1, 0]]
    assert result["score"]["mae_argmax_value"] == 1
    assert result["score"]["within_one_level_accuracy"] == 1


def test_module_cli_parses_optional_rates_boolean_and_query_pooling(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "typed",
            "--dataset",
            "unused",
            "--output",
            "unused",
            "--head-learning-rate",
            "0.002",
            "--no-evaluate-test",
            "--query-pooling",
            "last_token",
        ],
    )
    monkeypatch.setattr(
        training, "train_typed", lambda **kwargs: captured.update(kwargs)
    )
    training.main()
    assert captured["config"].head_learning_rate == 0.002
    assert captured["config"].backbone_learning_rate is None
    assert captured["config"].evaluate_test is False
    assert captured["model_config"].query_pooling == "last_token"
