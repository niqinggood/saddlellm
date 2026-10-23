"""Regression pipeline planning and export must preserve artifact identity."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from saddlellm.evaluation.ModelIdentity import compute_model_identity
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


@pytest.fixture
def artifacts(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"evaluated-base")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(model)}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"evaluated-adapter")
    data = tmp_path / "eval.jsonl"
    data.write_text('{"instruction":"question","answer":"A"}\n', encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "general",
                        "task": "multiple-choice",
                        "data": str(data),
                        "metrics": {
                            "accuracy": {"max_drop": 0.02, "higher_is_better": True}
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return model, adapter, suite


def pipeline(tmp_path, artifacts, *, dry_run=False, adapter=False, require_gate=False):
    model, adapter_path, suite = artifacts
    return {
        "model": {"name_or_path": str(model)},
        "stages": ["eval", "export"],
        "training": {"dry_run": dry_run},
        "logging": {"output_dir": str(tmp_path / "run"), "logging_backend": "local"},
        "eval": {
            "regression_suite": str(suite),
            "adapter": str(adapter_path) if adapter else None,
        },
        "export": {"enabled": True, "require_gate": require_gate, "merge_lora": False},
    }


def accepted_gate(monkeypatch, model, adapter=None):
    import saddlellm.evaluation.RegressionSuite as regression

    seen = []

    def evaluate(*args, **kwargs):
        seen.append((args, kwargs))
        return {
            "accepted": True,
            "status": "accepted",
            "candidate_model_identity": compute_model_identity(model, adapter),
        }

    monkeypatch.setattr(regression, "run_regression_suite", evaluate)
    return seen


def metric_pipeline(tmp_path, artifacts, monkeypatch, *, adapter=False):
    """Exercise the ordinary metric gate without a model runtime or suite."""
    from saddlellm.evaluation import EvaluationRunner

    raw = pipeline(tmp_path, artifacts, adapter=adapter)
    raw["eval"].pop("regression_suite")
    raw["eval"].update({
        "task": "multiple-choice",
        "eval_dataset": str(tmp_path / "eval.jsonl"),
        "gate": {"enabled": True, "rules": {"accuracy": {"min": 0.8}}},
    })

    def evaluate(model, data, **options):
        return {
            "accuracy": 0.9,
            "model_identity": compute_model_identity(model, options.get("adapter")),
        }

    monkeypatch.setattr(EvaluationRunner, "run_evaluation", evaluate)
    return raw


@pytest.mark.parametrize("changed_file", ["model.safetensors", "quantization_manifest.json"])
def test_enabled_metric_gate_blocks_changed_content_without_explicit_require_gate(
    tmp_path, artifacts, monkeypatch, changed_file,
):
    from saddlellm.runtime.ModelExporter import ModelExporter

    model, _, _ = artifacts
    if changed_file.endswith(".json"):
        (model / changed_file).write_text(
            json.dumps({"format": "compressed-tensors", "method": "gptq", "bits": 4}),
            encoding="utf-8",
        )
    raw = metric_pipeline(tmp_path, artifacts, monkeypatch)
    destination = tmp_path / "existing-release"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    raw["export"].update(output_dir=str(destination), overwrite=True)
    monkeypatch.setattr(ModelExporter, "export", lambda *a, **k: pytest.fail("stale metric gate reached export"))
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        runner._run_eval_stage()
        if changed_file.endswith(".json"):
            value = json.loads((model / changed_file).read_text(encoding="utf-8"))
            value["bits"] = 8
            (model / changed_file).write_text(json.dumps(value), encoding="utf-8")
        else:
            (model / changed_file).write_bytes(b"changed-after-metric-evaluation")
        with pytest.raises(ValueError, match="identity|content|match"):
            runner._run_export_stage()
        assert runner._stage_results["export"]["status"] == "blocked"
        assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"
    finally:
        runner._close_logging_handlers()


@pytest.mark.parametrize("adapter", [False, True])
def test_enabled_metric_gate_forces_strict_export_of_the_evaluated_artifact(
    tmp_path, artifacts, monkeypatch, adapter,
):
    from pathlib import Path

    from saddlellm.runtime.ModelExporter import ModelExporter

    raw = metric_pipeline(tmp_path, artifacts, monkeypatch, adapter=adapter)
    original_export = ModelExporter.export
    requests = []

    def export(request, *, gate):
        requests.append(request)
        return original_export(request, gate=gate)

    monkeypatch.setattr(ModelExporter, "export", export)
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        runner._run_eval_stage()
        runner._run_export_stage()
        expected_source = artifacts[1] if adapter else artifacts[0]
        assert requests[0].require_gate is True
        assert requests[0].model_path == str(expected_source)
        result = runner._stage_results["export"]
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["gate"]["accepted"] is True
        assert manifest["gate_binding"]["status"] == "matched"
        component = "adapter" if adapter else "model"
        assert manifest["model_identity"][component]["reference"] == result["model_path"]
        if adapter:
            assert manifest["model_identity"]["model"]["reference"] == str(artifacts[0])
    finally:
        runner._close_logging_handlers()


def test_regression_dry_run_plans_export_with_pending_gate(
    tmp_path, artifacts, monkeypatch
):
    import saddlellm.evaluation.RegressionSuite as regression

    monkeypatch.setattr(
        regression,
        "run_regression_suite",
        lambda *a, **k: pytest.fail("dry-run evaluated a model"),
    )
    runner = TrainingOrchestrator.from_dict(
        pipeline(tmp_path, artifacts, dry_run=True, require_gate=True)
    )
    try:
        runner._run_eval_stage()
        runner._run_export_stage()
        assert runner._stage_results["eval"]["status"] == "planned"
        assert runner._stage_results["export"]["status"] == "planned"
        assert runner._stage_results["export"]["gate_pending"] is True
    finally:
        runner._close_logging_handlers()


def test_adapter_evaluation_exports_the_evaluated_adapter_by_default(
    tmp_path, artifacts, monkeypatch
):
    from saddlellm.runtime.ModelExporter import ModelExporter

    model, adapter, _ = artifacts
    seen = accepted_gate(monkeypatch, model, adapter)
    exported = []

    def export(request, *, gate):
        exported.append((request, gate))
        return SimpleNamespace(to_dict=lambda: {"status": "completed"})

    monkeypatch.setattr(ModelExporter, "export", export)
    runner = TrainingOrchestrator.from_dict(pipeline(tmp_path, artifacts, adapter=True))
    try:
        runner._run_eval_stage()
        runner._run_export_stage()
        assert seen[0][0][1] == str(model)
        assert seen[0][1]["candidate_adapter"] == str(adapter)
        assert exported[0][0].model_path == str(adapter)
        assert runner._stage_results["eval"]["model"] == str(adapter)
    finally:
        runner._close_logging_handlers()


def test_enabled_suite_blocks_changed_model_even_without_require_gate(
    tmp_path, artifacts, monkeypatch
):
    from saddlellm.runtime.ModelExporter import ModelExporter

    model, _, _ = artifacts
    accepted_gate(monkeypatch, model)
    monkeypatch.setattr(
        ModelExporter,
        "export",
        lambda *a, **k: pytest.fail("export accepted stale model evidence"),
    )
    runner = TrainingOrchestrator.from_dict(
        pipeline(tmp_path, artifacts, require_gate=False)
    )
    try:
        runner._run_eval_stage()
        (model / "model.safetensors").write_bytes(b"different-after-evaluation")
        with pytest.raises((ValueError, RuntimeError), match="identity|content|match"):
            runner._run_export_stage()
    finally:
        runner._close_logging_handlers()


def test_enabled_suite_blocks_changed_adapter_even_without_require_gate(
    tmp_path, artifacts, monkeypatch
):
    from saddlellm.runtime.ModelExporter import ModelExporter

    model, adapter, _ = artifacts
    accepted_gate(monkeypatch, model, adapter)
    monkeypatch.setattr(
        ModelExporter,
        "export",
        lambda *a, **k: pytest.fail("export accepted stale adapter evidence"),
    )
    runner = TrainingOrchestrator.from_dict(
        pipeline(tmp_path, artifacts, adapter=True, require_gate=False)
    )
    try:
        runner._run_eval_stage()
        (adapter / "adapter_model.safetensors").write_bytes(
            b"different-after-evaluation"
        )
        with pytest.raises((ValueError, RuntimeError), match="identity|content|match"):
            runner._run_export_stage()
    finally:
        runner._close_logging_handlers()


@pytest.mark.parametrize("destination", ["evaluation", "gate"])
def test_pipeline_report_rejection_does_not_overwrite_source_data(
    tmp_path, artifacts, monkeypatch, destination
):
    import saddlellm.evaluation.RegressionSuite as regression

    data = tmp_path / "eval.jsonl"
    original = data.read_bytes()
    raw = pipeline(tmp_path, artifacts)
    if destination == "evaluation":
        raw["eval"]["output_path"] = str(data)
    else:
        raw["eval"]["gate"] = {"output_path": str(data)}
    monkeypatch.setattr(
        regression,
        "run_regression_suite",
        lambda *a, **k: pytest.fail("unsafe output was not rejected before evaluation"),
    )
    runner = None
    try:
        with pytest.raises(
            (ValueError, RuntimeError), match="output|Report|report|overwrite"
        ):
            runner = TrainingOrchestrator.from_dict(raw)
            runner._run_eval_stage()
    finally:
        if runner is not None:
            runner._close_logging_handlers()
    assert data.read_bytes() == original
