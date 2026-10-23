"""Checkpoint ranking must preserve the existing content-bound quality gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from saddlellm.evaluation.CheckpointSelection import (
    CheckpointSelectionError,
    select_checkpoint,
)
from saddlellm.evaluation.EvaluationRunner import _records_identity
from saddlellm.evaluation.ModelIdentity import compute_model_identity, verify_gate_model


@pytest.fixture
def selection_setup(tmp_path):
    models = {}
    for name in ("baseline", "best-new-but-forgets", "retained", "alternative"):
        path = tmp_path / name
        path.mkdir()
        (path / "model.safetensors").write_bytes(name.encode())
        models[name] = str(path)
    data = {}
    for name in ("general", "new-task"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text('{"id":"one","instruction":"Q","answer":"A"}\n', encoding="utf-8")
        data[name] = str(path)
    suite = {"name": "checkpoint-retention", "tasks": [
        {"name": "general", "data": data["general"], "task": "multiple-choice",
         "metrics": {"accuracy": {"max_drop": 0.02, "higher_is_better": True}}},
        {"name": "new-task", "data": data["new-task"], "task": "multiple-choice",
         "metrics": {"accuracy": {"max_drop": 0.02, "higher_is_better": True, "min": 0.7}}},
    ]}
    scores = {
        "baseline": {"general": 0.8, "new-task": 0.5},
        "best-new-but-forgets": {"general": 0.7, "new-task": 0.99},
        "retained": {"general": 0.79, "new-task": 0.9},
        "alternative": {"general": 0.8, "new-task": 0.8},
    }
    calls = []
    def evaluate(model, source, *, task, split=None, **options):
        name = Path(model).name
        task_name = Path(source).stem
        calls.append((name, task_name))
        data_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        report = {
            "task_type": task, "model": model,
            "model_identity": compute_model_identity(model),
            "data_sha256": data_hash, "samples": 1,
            "evaluation_identity": {
                "task": task, "data_sha256": data_hash, "split": split,
                "limit": options.get("limit", 0), "samples": 1,
                **_records_identity(source, split, options.get("limit", 0)),
            },
        }
        if "accuracy" not in scores[name]:
            # Task-keyed values are the ordinary fixture; metric-keyed values
            # permit malformed evaluator output without faking gate results.
            value = scores[name].get(task_name)
        else:
            value = scores[name]["accuracy"]
        if value != "missing":
            report["accuracy"] = value
        return report
    return models, data, suite, scores, calls, evaluate


def run_selection(setup, names=("best-new-but-forgets", "retained", "alternative"), **kwargs):
    models, _, suite, _, _, evaluator = setup
    return select_checkpoint(
        models["baseline"], [models[name] for name in names], suite,
        selection_task="new-task", selection_metric="accuracy", evaluator=evaluator, **kwargs,
    )


def test_higher_new_task_score_cannot_bypass_old_capability_gate(selection_setup, tmp_path):
    output = tmp_path / "selection.json"
    result = run_selection(selection_setup, output=output)
    models = selection_setup[0]
    assert result["accepted"] is True
    assert result["selected_model"] == models["retained"]
    assert result["selected_report"]["accepted"] is True
    assert verify_gate_model(result["selected_report"], compute_model_identity(models["retained"]), required=True)["accepted"]
    assert result["candidate_reports"][0]["status"] == "failed"
    assert [row["model"] for row in result["ranking"]] == [models["retained"], models["alternative"]]
    assert json.loads(output.read_text(encoding="utf-8")) == result
    files = sorted((tmp_path / "selection_candidates").glob("*.json"))
    assert len(files) == 3
    for path, entry in zip(files, result["candidate_reports"]):
        assert json.loads(path.read_text(encoding="utf-8")) == entry["regression_report"]


def test_absolute_new_task_floor_is_also_mandatory(selection_setup):
    selection_setup[3]["retained"]["new-task"] = 0.69
    with pytest.raises(CheckpointSelectionError) as error:
        run_selection(selection_setup, names=("retained",))
    assert error.value.report["candidate_reports"][0]["status"] == "failed"


def test_all_failed_candidates_save_rejection_before_raising(selection_setup, tmp_path):
    output = tmp_path / "rejected.json"
    with pytest.raises(CheckpointSelectionError, match="No checkpoint passed") as error:
        run_selection(selection_setup, names=("best-new-but-forgets",), output=output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == error.value.report
    assert report["accepted"] is False and report["selected_model"] is None
    assert report["selected_report"] is None and report["ranking"] == []


@pytest.mark.parametrize("value", ["missing", None, float("nan"), float("inf"), True])
def test_missing_or_nonfinite_metric_cannot_participate_in_ranking(selection_setup, value, tmp_path):
    selection_setup[3]["retained"]["new-task"] = value
    report = run_selection(selection_setup, names=("retained", "alternative"), output=tmp_path / "selection.json")
    assert report["selected_model"] == selection_setup[0]["alternative"]
    assert report["candidate_reports"][0]["status"] == "invalid"
    assert not report["candidate_reports"][0]["accepted"]
    json.dumps(report, allow_nan=False)


def test_ties_use_explicit_candidate_order(selection_setup):
    selection_setup[3]["alternative"]["new-task"] = 0.9
    report = run_selection(selection_setup, names=("alternative", "retained"))
    assert report["selected_model"] == selection_setup[0]["alternative"]
    assert report["selection"]["tie_break"] == "candidate_input_order"


def test_lower_is_better_uses_same_declared_gate_direction(selection_setup):
    for task in selection_setup[2]["tasks"]:
        task["metrics"]["accuracy"] = {"max_drop": 0.5, "higher_is_better": False, "max": 1}
    report = run_selection(selection_setup, names=("retained", "alternative"), higher_is_better=False)
    assert report["selected_model"] == selection_setup[0]["alternative"]


def test_undeclared_metric_or_direction_fails_before_evaluation(selection_setup):
    models, _, suite, _, calls, evaluator = selection_setup
    for options in [
        {"selection_task": "not-in-suite", "selection_metric": "accuracy"},
        {"selection_task": "new-task", "selection_metric": "not-gated"},
        {"selection_task": "new-task", "selection_metric": "accuracy", "higher_is_better": False},
    ]:
        with pytest.raises(ValueError):
            select_checkpoint(models["baseline"], [models["retained"]], suite, evaluator=evaluator, **options)
    assert calls == []


@pytest.mark.parametrize("candidates", [[], "checkpoint-*", {"path": "model"}, ["same", "same"]])
def test_candidates_are_an_explicit_nonempty_unique_sequence(selection_setup, candidates):
    with pytest.raises(ValueError):
        select_checkpoint(selection_setup[0]["baseline"], candidates, selection_setup[2],
                          selection_task="new-task", selection_metric="accuracy")


def test_model_mutated_after_its_evaluation_cannot_win(selection_setup):
    models, _, suite, _, _, evaluate = selection_setup
    def mutate(model, source, **options):
        report = evaluate(model, source, **options)
        if model == models["alternative"]:
            (Path(models["retained"]) / "model.safetensors").write_bytes(b"changed after evaluation")
        return report
    report = select_checkpoint(models["baseline"], [models["retained"], models["alternative"]], suite,
                               selection_task="new-task", selection_metric="accuracy", evaluator=mutate)
    assert report["selected_model"] == models["alternative"]
    assert report["candidate_reports"][0]["status"] == "invalid"
    assert "content changed" in " ".join(report["candidate_reports"][0]["errors"])


def test_baseline_changed_between_candidates_invalidates_selection(selection_setup, tmp_path):
    models, _, suite, _, _, evaluate = selection_setup
    def mutate(model, source, **options):
        report = evaluate(model, source, **options)
        if model == models["alternative"]:
            (Path(models["baseline"]) / "model.safetensors").write_bytes(b"changed baseline")
        return report
    output = tmp_path / "selection.json"
    with pytest.raises(CheckpointSelectionError) as error:
        select_checkpoint(models["baseline"], [models["retained"], models["alternative"]], suite,
                          selection_task="new-task", selection_metric="accuracy", evaluator=mutate, output=output)
    assert error.value.report["status"] == "invalid"
    assert not any(row["accepted"] for row in error.value.report["candidate_reports"])
    assert "Baseline model content changed" in " ".join(error.value.report["errors"])


def test_data_changed_between_candidates_invalidates_all_selection(selection_setup, tmp_path):
    models, data, suite, _, _, evaluate = selection_setup
    def mutate(model, source, **options):
        report = evaluate(model, source, **options)
        if model == models["alternative"] and source == data["new-task"]:
            Path(source).write_text('{"id":"two","instruction":"changed","answer":"B"}\n', encoding="utf-8")
        return report
    output = tmp_path / "selection.json"
    with pytest.raises(CheckpointSelectionError) as error:
        select_checkpoint(models["baseline"], [models["retained"], models["alternative"]], suite,
                          selection_task="new-task", selection_metric="accuracy", evaluator=mutate, output=output)
    assert error.value.report["status"] == "invalid"
    assert "Evaluation data/settings changed" in " ".join(error.value.report["errors"])
    assert json.loads(output.read_text(encoding="utf-8"))["selected_model"] is None


@pytest.mark.parametrize("target", ["data", "inside-baseline", "inside-other-candidate", "colliding-report-dir"])
def test_outputs_cannot_overwrite_any_model_or_evaluation_data(selection_setup, target, tmp_path):
    models, data, _, _, calls, _ = selection_setup
    if target == "data":
        output = Path(data["general"])
    elif target == "inside-baseline":
        output = Path(models["baseline"]) / "selection.json"
    elif target == "inside-other-candidate":
        output = Path(models["alternative"]) / "selection.json"
    else:
        # The generated per-candidate report directory is itself a checkpoint.
        output = Path(models["alternative"]).with_name("alternative.json")
        Path(models["alternative"]).rename(tmp_path / "alternative_candidates")
        models["alternative"] = str(tmp_path / "alternative_candidates")
    original_data = Path(data["general"]).read_bytes()
    with pytest.raises(ValueError, match="overwrite|outside"):
        run_selection(selection_setup, output=output)
    assert calls == []
    assert Path(data["general"]).read_bytes() == original_data


def test_output_cannot_overwrite_the_suite_file(selection_setup, tmp_path):
    models, _, suite, _, calls, evaluator = selection_setup
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        select_checkpoint(models["baseline"], [models["retained"]], path, selection_task="new-task",
                          selection_metric="accuracy", evaluator=evaluator, output=path)
    assert path.read_bytes() == original and calls == []


def test_unresolved_remote_or_missing_models_are_not_selectable(selection_setup):
    models, _, suite, _, _, evaluator = selection_setup
    with pytest.raises(CheckpointSelectionError) as error:
        select_checkpoint(models["baseline"], ["nonexistent/checkpoint"], suite,
                          selection_task="new-task", selection_metric="accuracy", evaluator=evaluator)
    assert not error.value.report["accepted"]
    assert error.value.report["candidate_reports"][0]["status"] == "invalid"


def test_adapter_directory_is_rejected_before_evaluation(selection_setup):
    models, _, _, _, calls, _ = selection_setup
    (Path(models["retained"]) / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": models["baseline"]}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Adapter checkpoints are not supported"):
        run_selection(selection_setup, names=("retained",))
    assert calls == []


def test_output_with_a_file_parent_fails_before_evaluation(selection_setup, tmp_path):
    parent = tmp_path / "ordinary-file"
    parent.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="parents must be directories"):
        run_selection(selection_setup, output=parent / "selection.json")
    assert selection_setup[4] == []
    assert parent.read_text(encoding="utf-8") == "preserve"


def test_candidate_runtime_failure_does_not_hide_other_qualifying_candidates(selection_setup):
    models, _, suite, _, _, evaluate = selection_setup
    def broken(model, source, **options):
        if model == models["retained"]:
            raise RuntimeError("candidate failed to load")
        return evaluate(model, source, **options)
    report = select_checkpoint(models["baseline"], [models["retained"], models["alternative"]], suite,
                               selection_task="new-task", selection_metric="accuracy", evaluator=broken)
    assert report["selected_model"] == models["alternative"]
    assert report["candidate_reports"][0]["status"] == "invalid"
