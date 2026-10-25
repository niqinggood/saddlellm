"""Tie decisions must agree across the prediction API and metric reports."""

import math

import numpy as np
import pytest
import torch

from saddlellm.decision.Evaluation import metrics as evaluation_metrics
from saddlellm.decision.TypedModel import TypedDecisionModel
from saddlellm.decision.TypedSchema import (
    TypedDecisionConfig,
    scope_key,
    validate_record,
)
from saddlellm.decision.TypedTraining import typed_metrics


@pytest.fixture
def tied_model():
    # Zero heads produce actual tied logits through the complete native forward
    # path, independently of candidate order and the randomly initialized encoder.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = TypedDecisionModel(
        TypedDecisionConfig(hidden_size=8, num_layers=1, num_heads=2, max_length=128)
    ).eval()
    with torch.no_grad():
        for head in model.heads.values():
            for parameter in head.parameters():
                parameter.zero_()
    yield model
    torch.set_num_threads(threads)


def record(**fields):
    return validate_record({"id": "fixture", "state_id": "fixture-state",
                            "task_id": "fixture-task", "state": "Local fixture.", **fields})


def compare_api_and_metrics(model, row):
    prediction = model.predict([row])[0]
    with torch.inference_mode():
        logits = [model([row])[0].cpu().numpy()]
    assert np.count_nonzero(logits[0]) == 0
    raw = typed_metrics(logits, [row])
    calibrated = typed_metrics(logits, [row], {scope_key(row): {"temperature": 2.0}})
    assert calibrated["accuracy"] == raw["accuracy"]
    assert calibrated["expected_accuracy"] == raw["expected_accuracy"]
    assert calibrated["top_label_ece"] == pytest.approx(raw["top_label_ece"])
    evaluated = evaluation_metrics(logits, [row])
    assert evaluated["accuracy"] == raw["accuracy"]
    assert evaluated["top_label_ece"] == raw["top_label_ece"]
    return prediction, raw, evaluated


@pytest.mark.parametrize("order", [("z", "a", "m"), ("m", "z", "a"), ("a", "m", "z")])
def test_choice_permutation_tie_matches_api_accuracy_ece_and_class_diagnostics(tied_model, order):
    row = record(type="choice", question="Choose an answer.",
                 choices=[{"id": label, "text": label} for label in order], label="a")
    prediction, result, evaluated = compare_api_and_metrics(tied_model, row)
    assert prediction["choice_id"] == "a"
    assert result["accuracy"] == result["expected_accuracy"] == 1.0
    assert result["nll"] == pytest.approx(math.log(3))
    assert result["top_label_ece"] == pytest.approx(2 / 3)
    diagnostic = result["class_diagnostics"][scope_key(row)]
    assert diagnostic["candidate_ids"] == ["a", "m", "z"]
    assert diagnostic["prediction_counts"] == {"a": 1, "m": 0, "z": 0}
    assert diagnostic["hard_target_counts"] == {"a": 1, "m": 0, "z": 0}
    assert diagnostic["confusion"]["matrix"] == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert evaluated["confusion"]["matrix"] == diagnostic["confusion"]["matrix"]


@pytest.mark.parametrize("gold,mae,within_one", [("z", 4.0, 0.0), ("m", 3.0, 1.0), ("a", 0.0, 1.0)])
def test_score_value_order_with_arbitrary_ids_tie_matches_api_mae_and_within_one(
    tied_model, gold, mae, within_one
):
    row = record(type="score", question="Which rating?", levels=[
        {"id": "z", "text": "low", "value": 1},
        {"id": "m", "text": "medium", "value": 2},
        {"id": "a", "text": "high", "value": 5},
    ], label=gold)
    prediction, result, evaluated = compare_api_and_metrics(tied_model, row)
    assert prediction["level_id"] == "a"
    assert next(option["value"] for option in prediction["probabilities"]
                if option["id"] == prediction["level_id"]) == 5
    assert result["accuracy"] == float(gold == "a")
    assert result["score"]["mae_argmax_value"] == mae
    assert result["score"]["within_one_level_accuracy"] == within_one
    assert prediction["expected_score"] == pytest.approx(8 / 3)
    truth_value = next(level["value"] for level in row["levels"] if level["id"] == gold)
    assert result["score"]["mae_expected_score"] == pytest.approx(abs(8 / 3 - truth_value))
    assert evaluated["score"]["mae_argmax_value"] == mae
    assert evaluated["score"]["within_one_level_accuracy"] == within_one
    # Equivalent floating-point reductions can differ in their final bits.
    assert evaluated["score"]["mae_expected_score"] == pytest.approx(
        result["score"]["mae_expected_score"], rel=1e-12, abs=1e-12
    )
    diagnostic = result["class_diagnostics"][scope_key(row)]
    assert diagnostic["prediction_counts"] == {"a": 1, "m": 0, "z": 0}
    assert diagnostic["confusion"]["matrix"] == evaluated["confusion"]["matrix"]


@pytest.mark.parametrize("order", [("z", "a", "m"), ("m", "z", "a")])
def test_soft_target_tie_expected_accuracy_and_ece_follow_api_winner(tied_model, order):
    row = record(type="choice", question="Choose an answer.",
                 choices=[{"id": label, "text": label} for label in order],
                 target_probs={"z": .1, "a": .6, "m": .3})
    prediction, result, evaluated = compare_api_and_metrics(tied_model, row)
    assert prediction["choice_id"] == "a"
    assert result["accuracy"] is None
    assert result["hard_label_samples"] == 0
    assert result["expected_accuracy"] == .6
    assert result["top_label_ece"] == pytest.approx(abs(1 / 3 - .6))
    assert result["reliability"][0]["mean_target_mass"] == .6
    diagnostic = result["class_diagnostics"][scope_key(row)]
    assert diagnostic["prediction_counts"] == {"a": 1, "m": 0, "z": 0}
    assert diagnostic["confusion"]["matrix"] == [[0, 0, 0]] * 3
    assert evaluated["expected_accuracy"] == result["expected_accuracy"]


@pytest.mark.parametrize("temperature", [1.0, 20.0])
@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_small_logit_margin_matches_public_prediction_and_offline_metrics(
    tied_model, monkeypatch, temperature, kind
):
    if kind == "choice":
        row = record(type=kind, question="Choose.", choices=[
            {"id": "a", "text": "First"}, {"id": "b", "text": "Second"},
        ], label="b")
    elif kind == "score":
        row = record(type=kind, question="Rate.", levels=[
            {"id": "a", "text": "Low", "value": 1},
            {"id": "b", "text": "High", "value": 2},
        ], label="b")
    else:
        row = record(type=kind, question="Is it true?", label=True)
    logits = torch.tensor([0.0, 1e-8], dtype=torch.float32)
    monkeypatch.setattr(tied_model, "forward", lambda rows: [logits for _ in rows])
    calibration = {scope_key(row): {"temperature": temperature}}
    tied_model.calibration = calibration
    public = tied_model.predict([row])[0]
    raw = typed_metrics([logits.numpy()], [row], calibration)
    offline = evaluation_metrics([logits.numpy()], [row], calibration)
    assert public["probabilities"][1]["probability"] > public["probabilities"][0]["probability"]
    assert raw["accuracy"] == offline["accuracy"] == 1.0
    if kind == "choice":
        assert public["choice_id"] == "b"
    elif kind == "score":
        assert public["level_id"] == "b"
        assert public["expected_score"] > 1.5
    else:
        assert public["probability_true"] > 0.5
