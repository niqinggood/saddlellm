import numpy as np
import pytest

from saddlellm.decision.Calibration import (
    fit_temperature,
    probabilities,
    probability_metrics,
)


def test_temperature_fit_reduces_overconfident_calibration_loss():
    logits = np.array([[8.0, 0.0]] * 10)
    labels = np.array([0] * 7 + [1] * 3)
    result = fit_temperature(logits, labels)
    assert result["temperature"] > 1
    assert result["nll_after"] < result["nll_before"]
    assert probabilities(logits, result["temperature"])[0, 0] == pytest.approx(0.7)
    assert np.array_equal(logits.argmax(1), probabilities(logits, result["temperature"]).argmax(1))


def test_metrics_hand_computable_uniform_distribution_and_empty_coverage():
    result = probability_metrics([[0, 0], [0, 0]], [0, 1], bins=2)
    assert result["accuracy"] == 0.5
    assert result["nll"] == pytest.approx(np.log(2))
    assert result["multiclass_brier"] == 0.5
    assert result["top_label_ece"] == 0
    assert result["reliability"][0]["count"] == 0
    assert result["selective_risk"][-1]["coverage"] == 0
    assert result["selective_risk"][-1]["error_rate"] is None


def test_probability_column_permutation_preserves_metrics():
    logits = np.array([[1, 5, 2], [4, 1, 2], [1, 0, 3]], dtype=float)
    labels = np.array([1, 0, 0])
    permutation = [2, 0, 1]
    remapped = np.argsort(permutation)[labels]
    before = probability_metrics(logits, labels)
    after = probability_metrics(logits[:, permutation], remapped)
    for key in ["accuracy", "nll", "multiclass_brier", "top_label_ece", "macro_f1"]:
        assert before[key] == pytest.approx(after[key])


def test_extreme_finite_logits_remain_stable():
    probs = probabilities([[10000, -10000], [-10000, 10000]])
    assert np.isfinite(probs).all()
    assert np.array_equal(probs, [[1, 0], [0, 1]])
    result = probability_metrics([[10000, -10000]], [1])
    assert result["nll"] == 20000


@pytest.mark.parametrize("logits,labels", [
    ([[float("nan"), 0]], [0]), ([[0, 1]], [2]), ([[0, 1]], [0.5]),
    ([], []), ([[0]], [0]), ([[0, 1]], []),
])
def test_invalid_calibration_data_is_rejected(logits, labels):
    with pytest.raises(ValueError):
        fit_temperature(logits, labels)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError):
        probabilities([[1, 0]], temperature)
