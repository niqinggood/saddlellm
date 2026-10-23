"""The pipeline must export the gated winner and reject stale/failed choices."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from saddlellm.evaluation.EvaluationRunner import _records_identity
from saddlellm.evaluation.ModelIdentity import compute_model_identity
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


@pytest.fixture
def setup(tmp_path, monkeypatch):
    models = {}
    for name in ("baseline", "forgets", "retained"):
        path = tmp_path / name
        path.mkdir()
        (path / "model.safetensors").write_bytes(name.encode())
        models[name] = str(path)
    tasks = []
    for name in ("general", "new-task"):
        data = tmp_path / (name + ".jsonl")
        data.write_text('{"id":"one","instruction":"Q","answer":"A"}\n')
        tasks.append({"name": name, "data": str(data), "task": "multiple-choice",
                      "metrics": {"accuracy": {"max_drop": .02, "higher_is_better": True}}})
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"tasks": tasks}))
    scores = {"baseline": [.8, .5], "forgets": [.6, 1.0], "retained": [.8, .9]}
    def evaluator(model, source, *, task, split=None, **options):
        data_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        return {"task_type": task, "model": model, "model_identity": compute_model_identity(model),
                "data_sha256": data_hash, "samples": 1,
                "accuracy": scores[Path(model).name][Path(source).stem == "new-task"],
                "evaluation_identity": {"task": task, "data_sha256": data_hash, "split": split,
                    "limit": 0, "samples": 1, **_records_identity(source, split, 0)}}
    from saddlellm.evaluation import CheckpointSelection
    actual_selection = CheckpointSelection.select_checkpoint
    monkeypatch.setattr(CheckpointSelection, "select_checkpoint",
                        lambda *a, **k: actual_selection(*a, **k, evaluator=evaluator))
    config = {"model": {"name_or_path": models["baseline"]}, "stages": ["eval", "export"],
              "logging": {"output_dir": str(tmp_path / "run")},
              "eval": {"regression_suite": str(suite), "checkpoint_candidates": [models["forgets"], models["retained"]],
                       "selection_task": "new-task", "selection_metric": "accuracy"},
              "export": {"enabled": True, "merge_lora": False}}
    return config, models, scores, suite


def test_default_export_uses_selected_checkpoint_and_bound_gate(setup, monkeypatch):
    from saddlellm.runtime.ModelExporter import ModelExporter
    config, models, _, _ = setup
    requests = []
    def export(request, *, gate):
        requests.append((request, gate))
        return SimpleNamespace(to_dict=lambda: {"status": "completed"})
    monkeypatch.setattr(ModelExporter, "export", export)
    runner = TrainingOrchestrator.from_dict(config)
    try:
        runner._run_eval_stage()
        runner._run_export_stage()
        assert runner._stage_results["eval"]["selected_checkpoint"] == models["retained"]
        assert requests[0][0].model_path == models["retained"]
        assert requests[0][1]["model_identity"] == compute_model_identity(models["retained"])
    finally:
        runner._close_logging_handlers()


def test_all_rejected_cannot_export_old_upstream_model(setup):
    config, _, scores, _ = setup
    scores["retained"] = [.5, .99]
    config["eval"]["regression_fail_on_rejection"] = False
    runner = TrainingOrchestrator.from_dict(config)
    try:
        runner._run_eval_stage()
        assert runner._stage_results["eval"]["status"] == "rejected"
        assert "selected_checkpoint" not in runner._stage_results["eval"]
        with pytest.raises(RuntimeError, match="gate was not accepted"):
            runner._run_export_stage()
    finally:
        runner._close_logging_handlers()


def test_changed_winner_is_rejected_before_export(setup):
    config, models, _, _ = setup
    runner = TrainingOrchestrator.from_dict(config)
    try:
        runner._run_eval_stage()
        Path(models["retained"], "model.safetensors").write_bytes(b"changed after selection")
        with pytest.raises((ValueError, RuntimeError), match="identity|content|match"):
            runner._run_export_stage()
    finally:
        runner._close_logging_handlers()


def test_derived_selection_report_cannot_overwrite_suite(setup, tmp_path):
    config, _, _, suite = setup
    output = tmp_path / "evaluation.json"
    collision = tmp_path / "evaluation.selection.json"
    original = suite.read_bytes()
    collision.write_bytes(original)
    config["eval"].update(regression_suite=str(collision), output_path=str(output))
    runner = TrainingOrchestrator.from_dict(config)
    try:
        with pytest.raises(ValueError, match="overwrite|suite|output"):
            runner._run_eval_stage()
        assert collision.read_bytes() == original
    finally:
        runner._close_logging_handlers()


def test_gate_cannot_collide_with_per_candidate_report(setup, tmp_path):
    config, _, _, _ = setup
    output = tmp_path / "evaluation.json"
    config["eval"].update(output_path=str(output), gate={"output_path": str(
        tmp_path / "evaluation.selection_candidates/candidate-0000.json")})
    runner = TrainingOrchestrator.from_dict(config)
    try:
        with pytest.raises(ValueError, match="distinct files"):
            runner._run_eval_stage()
    finally:
        runner._close_logging_handlers()


def test_new_configuration_controls_are_parsed(setup):
    config, _, _, _ = setup
    config["data"] = {"packing_attention": "concat"}
    parsed = TrainingOrchestrator._parse_config(config)
    assert parsed.data.packing_attention == "concat"
    assert parsed.eval.checkpoint_candidates == config["eval"]["checkpoint_candidates"]
    assert parsed.eval.selection_task == "new-task"


def test_standalone_candidate_selection_is_validated_as_export_source(setup):
    from saddlellm.training.TrainingConfigValidator import validate_training_config

    config, models, _, _ = setup
    config["model"] = {}
    config["eval"]["regression_baseline"] = models["baseline"]
    report = validate_training_config(config, inspect_data=False)
    assert report["valid"], report["issues"]
    runner = TrainingOrchestrator.from_dict(config)
    try:
        runner._run_eval_stage()
        assert runner._stage_results["eval"]["selected_checkpoint"] == models["retained"]
    finally:
        runner._close_logging_handlers()


def test_wrong_selection_direction_fails_during_preflight_and_initialization(setup):
    from saddlellm.training.TrainingConfigValidator import validate_training_config

    config, _, _, _ = setup
    config["eval"]["selection_higher_is_better"] = False
    report = validate_training_config(config, inspect_data=False)
    assert not report["valid"] and any("direction" in issue for issue in report["issues"])
    with pytest.raises(ValueError, match="direction"):
        TrainingOrchestrator.from_dict(config)
