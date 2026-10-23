"""Real artifact evaluation, requested-split isolation and legacy CLI parity."""

import importlib
import json
from pathlib import Path

import pytest
import torch

from saddlellm.decision.dataset import sha256
from saddlellm.decision.TypedModel import TypedDecisionModel
from saddlellm.decision.TypedSchema import (
    TypedDecisionConfig,
    candidate_query_text,
    candidates,
    read_typed_dataset,
    scope_key,
)
from saddlellm.decision.TypedTraining import collect_logits, fit_typed_temperature
from tools.prepare_decision_typed import prepare

evaluation = importlib.import_module("saddlellm.decision.Evaluation")
training = importlib.import_module("saddlellm.decision.TypedTraining")


@pytest.fixture
def saved_fixture(tmp_path):
    data = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = TypedDecisionModel(
        TypedDecisionConfig(hidden_size=8, num_layers=1, num_heads=2)
    ).eval()
    model.save_pretrained(tmp_path / "raw_model")
    _, splits, _ = read_typed_dataset(data)
    logits = collect_logits(model, splits["calibration"], 2)
    for scope in {scope_key(row) for row in splits["calibration"]}:
        indices = [
            index
            for index, row in enumerate(splits["calibration"])
            if scope_key(row) == scope
        ]
        model.calibration[scope] = fit_typed_temperature(
            [logits[index] for index in indices],
            [splits["calibration"][index] for index in indices],
        )
    model.save_pretrained(tmp_path / "model")
    return data, tmp_path / "model", tmp_path / "raw_model"


def evaluate(data, model, output, **kwargs):
    return evaluation.evaluate_typed(
        data,
        model,
        output,
        device="cpu",
        batch_size=2,
        cpu_threads=1,
        progress=False,
        **kwargs,
    )


def test_only_requested_split_gets_model_logits_and_artifact_stays_immutable(
    saved_fixture, tmp_path, monkeypatch
):
    data, model, _ = saved_fixture
    manifest_before = sha256(model / "manifest.json")
    calibration_before = sha256(model / "calibration.json")
    forward = TypedDecisionModel.forward
    requests = []

    def validation_only(self, records):
        assert all(row["id"].startswith("validation-") for row in records)
        requests.extend(row["id"] for row in records)
        return forward(self, records)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Evaluation must neither train/save a model nor refit calibration"
        )

    monkeypatch.setattr(TypedDecisionModel, "forward", validation_only)
    monkeypatch.setattr(TypedDecisionModel, "save_pretrained", forbidden)
    monkeypatch.setattr(
        importlib.import_module("saddlellm.decision.TypedTraining"),
        "fit_typed_temperature",
        forbidden,
    )
    report = evaluate(data, model, tmp_path / "eval", splits=["validation"])
    assert set(report["splits"]) == {"validation"}
    assert len(requests) == 9
    assert report["evaluation_only"] is True
    assert report["calibration_fit_performed"] is False
    assert report["test_used_for_model_selection"] is False
    assert (
        report["splits"]["validation"]["overall"]["calibrated"]["calibration_coverage"]
        == 1
    )
    preflight = json.loads((tmp_path / "eval/token_preflight.json").read_text())
    assert preflight["all_four_splits_checked"] is True
    assert set(preflight["by_split_task"]) == {
        "train",
        "validation",
        "calibration",
        "test",
    }
    rows = [
        json.loads(line)
        for line in (tmp_path / "eval/predictions.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 9 and all(row["split"] == "validation" for row in rows)
    assert sha256(model / "manifest.json") == manifest_before
    assert sha256(model / "calibration.json") == calibration_before


def test_saved_artifact_reload_reference_is_exact_for_selected_splits(
    saved_fixture, tmp_path
):
    data, model, _ = saved_fixture
    first = evaluate(data, model, tmp_path / "first", splits=["validation", "test"])
    second = evaluate(
        data,
        model,
        tmp_path / "second",
        splits=["validation", "test"],
        logits_reference=tmp_path / "first/logits.jsonl",
    )
    check = second["reload_consistency"]
    assert check["passed"] is True
    assert check["exact_logits_samples"] == 21
    assert check["max_absolute_error"] == 0
    assert first["splits"] == second["splits"]
    assert (tmp_path / "first/predictions.jsonl").read_bytes() == (
        tmp_path / "second/predictions.jsonl"
    ).read_bytes()
    assert (
        second["splits"]["test"]["overall"]["calibrated"]["calibration_coverage"]
        == 0.75
    )


def test_raw_artifact_without_temperatures_is_explicitly_uncalibrated(
    saved_fixture, tmp_path
):
    data, _, raw_model = saved_fixture
    report = evaluate(data, raw_model, tmp_path / "raw", splits="test")
    metrics = report["splits"]["test"]["overall"]
    assert metrics["calibrated"] is None
    assert metrics["calibration_status"] == "unavailable_no_saved_calibration"
    assert metrics["raw"]["samples"] == 12
    assert (
        report["splits"]["test"]["by_type"]["score"]["raw"]["score"]["mae_argmax_value"]
        >= 0
    )


def test_majority_baseline_cannot_choose_labels_using_test_distribution(
    saved_fixture, tmp_path
):
    data, model, _ = saved_fixture
    path = data / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["type"] == "score":
            row["label"] = "high"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = evaluate(data, model, tmp_path / "majority", splits=["test"])
    score = report["splits"]["test"]["by_task"]["intensity"]
    assert score["training_majority"]["accuracy"] == 0
    _, parsed, _ = read_typed_dataset(data)
    scope = scope_key(next(row for row in parsed["train"] if row["type"] == "score"))
    assert report["training_majorities"][scope]["label"] == "low"


def test_legacy_tool_and_output_dir_alias_preserve_row_outputs(
    saved_fixture, tmp_path, capsys
):
    from tools.evaluate_jev_real import main as legacy_main

    data, model, _ = saved_fixture
    common = [
        "--data",
        str(data),
        "--model",
        str(model),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--cpu-threads",
        "1",
        "--splits",
        "test",
    ]
    assert legacy_main(common + ["--output", str(tmp_path / "legacy")]) == 0
    assert evaluation.main(common + ["--output-dir", str(tmp_path / "native")]) == 0
    legacy = json.loads((tmp_path / "legacy/report.json").read_text())
    native = json.loads((tmp_path / "native/report.json").read_text())
    assert legacy["splits"] == native["splits"]
    for filename in ("predictions.jsonl", "logits.jsonl", "token_preflight.json"):
        assert (tmp_path / "legacy" / filename).read_bytes() == (
            tmp_path / "native" / filename
        ).read_bytes()
    assert '"completed": true' in capsys.readouterr().out


def test_reference_mismatch_is_recorded_and_cli_returns_failure(
    saved_fixture, tmp_path
):
    data, model, _ = saved_fixture
    evaluate(data, model, tmp_path / "first", splits=["test"])
    references = [
        json.loads(line)
        for line in (tmp_path / "first/logits.jsonl").read_text().splitlines()
    ]
    references[0]["logits"][0] += 1
    reference = tmp_path / "bad_reference.jsonl"
    reference.write_text("".join(json.dumps(row) + "\n" for row in references))
    code = evaluation.main(
        [
            "--data",
            str(data),
            "--model",
            str(model),
            "--output",
            str(tmp_path / "second"),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--cpu-threads",
            "1",
            "--splits",
            "test",
            "--logits-reference",
            str(reference),
        ]
    )
    report = json.loads((tmp_path / "second/report.json").read_text())
    assert code == 1
    assert report["reload_consistency"]["passed"] is False
    assert report["reload_consistency"]["mismatches"][0]["reason"] == "logits_mismatch"


def test_complete_preflight_rejects_long_unrequested_input_before_prediction(
    saved_fixture, tmp_path, monkeypatch
):
    data, model, _ = saved_fixture
    path = data / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["state"] = "z" * 300
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        TypedDecisionModel,
        "forward",
        lambda *args: pytest.fail("Preflight must finish before any prediction"),
    )
    with pytest.raises(ValueError, match="overlong"):
        evaluate(data, model, tmp_path / "long", splits=["validation"])
    failure = json.loads((tmp_path / "long/failure.json").read_text())
    assert failure["overlong_samples"][0]["split"] == "test"
    assert not (tmp_path / "long/predictions.jsonl").exists()


def add_instance_question(data, split, text):
    path = data / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["instance_question"] = text
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def save_instance_model(raw_model, output):
    torch.manual_seed(42)
    model = TypedDecisionModel(
        TypedDecisionConfig(
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            instance_question_in_query=True,
        )
    ).eval()
    model.load_state_dict(TypedDecisionModel.from_pretrained(raw_model).state_dict())
    model.save_pretrained(output)
    return output


def test_default_ignores_instance_input_without_changing_logits_or_preflight(
    saved_fixture, tmp_path
):
    data, model, _ = saved_fixture
    before = evaluate(data, model, tmp_path / "before", splits=["validation"])
    add_instance_question(data, "validation", "q" * 300)
    add_instance_question(data, "test", "r" * 300)
    after = evaluate(data, model, tmp_path / "after", splits=["validation"])
    assert before["splits"] == after["splits"]
    for filename in ("logits.jsonl", "token_preflight.json"):
        assert (tmp_path / "before" / filename).read_bytes() == (
            tmp_path / "after" / filename
        ).read_bytes()


def test_enabled_instance_preflight_checks_unrequested_test_without_forward(
    saved_fixture, tmp_path, monkeypatch
):
    data, _, raw_model = saved_fixture
    enabled_model = save_instance_model(raw_model, tmp_path / "instance_model")
    add_instance_question(data, "test", "q" * 300)
    _, splits, _ = read_typed_dataset(data)
    row = splits["test"][0]
    expected_lengths = [
        len(
            candidate_query_text(row, option, instance_question_in_query=True).encode(
                "utf-8"
            )
        )
        + 1
        for option in candidates(row)
    ]
    monkeypatch.setattr(
        TypedDecisionModel,
        "forward",
        lambda *args: pytest.fail("Full preflight must reject before any forward"),
    )
    with pytest.raises(ValueError, match="overlong"):
        evaluate(data, enabled_model, tmp_path / "instance_long", splits=["validation"])
    failure = json.loads((tmp_path / "instance_long/failure.json").read_text())
    error = failure["overlong_samples"][0]
    assert error["split"] == "test"
    assert error["formatted_query_tokens"] == expected_lengths
    assert min(expected_lengths) > 256
    assert not (tmp_path / "instance_long/predictions.jsonl").exists()


def test_enabled_instance_evaluation_only_predicts_requested_validation(
    saved_fixture, tmp_path, monkeypatch
):
    data, _, raw_model = saved_fixture
    enabled_model = save_instance_model(raw_model, tmp_path / "instance_model")
    for split in ("train", "validation", "calibration", "test"):
        add_instance_question(data, split, f"What does this {split} text support?")
    forward = TypedDecisionModel.forward
    requests = []

    def validation_only(self, records):
        assert self.config.instance_question_in_query is True
        assert all(row["id"].startswith("validation-") for row in records)
        requests.extend(row["id"] for row in records)
        return forward(self, records)

    monkeypatch.setattr(TypedDecisionModel, "forward", validation_only)
    report = evaluate(
        data, enabled_model, tmp_path / "instance_eval", splits="validation"
    )
    assert set(report["splits"]) == {"validation"}
    assert len(requests) == 9
    _, splits, _ = read_typed_dataset(data)
    preflight = json.loads(
        (tmp_path / "instance_eval/token_preflight.json").read_text()
    )
    for split, rows in splits.items():
        row = rows[0]
        group = preflight["by_split_task"][split][row["task_id"]]["formatted_query"]
        longest = max(
            len(
                candidate_query_text(
                    item, option, instance_question_in_query=True
                ).encode("utf-8")
            )
            + 1
            for item in rows
            if item["task_id"] == row["task_id"]
            for option in candidates(item)
        )
        assert group["max"] == longest


@pytest.mark.parametrize("split", ["train", "calibration"])
def test_training_preflight_rejects_long_instance_before_updates(
    saved_fixture, tmp_path, monkeypatch, split
):
    data, _, _ = saved_fixture
    add_instance_question(data, split, "q" * 300)
    monkeypatch.setattr(
        training,
        "supervised_loss",
        lambda *args, **kwargs: pytest.fail("Overlong input must fail before training"),
    )
    with pytest.raises(ValueError, match="no truncation performed"):
        training.train_typed(
            data,
            tmp_path / "instance_train",
            device="cpu",
            model_config=TypedDecisionConfig(
                hidden_size=8,
                num_layers=1,
                num_heads=2,
                instance_question_in_query=True,
            ),
            config=training.TypedTrainingConfig(
                epochs=1, batch_size=9, cpu_threads=1, evaluate_test=False
            ),
        )
    assert not (tmp_path / "instance_train").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": True},
        {"cpu_threads": 0},
        {"splits": []},
        {"splits": ["train"]},
        {"splits": ["test", "test"]},
        {"reference_atol": float("nan")},
        {"reference_rtol": -1},
    ],
)
def test_public_api_rejects_invalid_settings(saved_fixture, tmp_path, options):
    data, model, _ = saved_fixture
    values = {"device": "cpu", "progress": False, **options}
    with pytest.raises(ValueError):
        evaluation.evaluate_typed(data, model, tmp_path / "invalid", **values)
    assert not (tmp_path / "invalid").exists()


def test_public_output_dir_alias_and_existing_output_protection(
    saved_fixture, tmp_path
):
    data, model, _ = saved_fixture
    output = tmp_path / "alias"
    report = evaluation.evaluate_typed(
        data,
        model,
        output_dir=output,
        device="cpu",
        batch_size=2,
        cpu_threads=1,
        splits=["validation"],
        progress=False,
    )
    assert Path(report["settings"]["output"]) == output
    with pytest.raises(FileExistsError):
        evaluate(data, model, output, splits=["validation"])
    with pytest.raises(ValueError, match="same directory"):
        evaluation.evaluate_typed(
            data, model, output, output_dir=tmp_path / "different"
        )
