"""Regression coverage for release provenance, resume, and stage aliases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from saddlellm.evaluation.LLModelEvalute import Evaluator
from saddlellm.framework import PipelinePlan, PipelineStateStore
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def _checkpoint(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    # The exporter copies this fixture; no model inference is performed.
    (path / "model.safetensors").write_bytes(b"export-control-flow-fixture")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return path


def _config(root: Path, model: Path, stages: list[str]) -> dict:
    return {
        "model": {"name_or_path": str(model)},
        "stages": stages,
        "distributed": {"bf16": False, "fp16": False},
        "logging": {"backend": "local", "output_dir": str(root / "run")},
        "eval": {
            "enabled": True,
            "dataset": "mocked-evaluation",
            "gate": {"enabled": True, "rules": {"perplexity": {"max": 20}}},
        },
        "export": {"enabled": True, "output_dir": str(root / "release")},
    }


@pytest.fixture
def passing_evaluation(monkeypatch):
    calls = []

    def evaluate(self, **kwargs):
        calls.append(kwargs["model_path"])
        return {
            "model": kwargs["model_path"],
            "metrics": {"perplexity": {"score": 10}},
            "error": None,
        }

    monkeypatch.setattr(Evaluator, "evaluate", evaluate)
    return calls


def test_required_export_rejects_disabled_gate(tmp_path, passing_evaluation):
    cfg = _config(tmp_path, _checkpoint(tmp_path, "model"), ["eval", "export"])
    cfg["eval"]["gate"]["enabled"] = False
    cfg["export"]["require_gate"] = True

    with pytest.raises(RuntimeError, match="accepted evaluation release gate"):
        TrainingOrchestrator.from_dict(cfg).run()
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("require_gate", [False, True])
def test_gated_export_rejects_other_model(tmp_path, passing_evaluation, require_gate):
    model_a = _checkpoint(tmp_path, "model_a")
    model_b = _checkpoint(tmp_path, "model_b")
    cfg = _config(tmp_path, model_a, ["eval", "export"])
    cfg["export"].update(model_path=str(model_b), require_gate=require_gate)

    with pytest.raises(RuntimeError, match="evaluated model"):
        TrainingOrchestrator.from_dict(cfg).run()
    assert passing_evaluation == [str(model_a)]
    assert not (tmp_path / "release").exists()


def test_gated_export_accepts_same_model_path_alias(
    tmp_path, monkeypatch, passing_evaluation
):
    model = _checkpoint(tmp_path, "model")
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path, model, ["eval", "export"])
    cfg["export"].update(model_path="./model", require_gate=True)

    results = TrainingOrchestrator.from_dict(cfg).run()
    assert results["export"]["status"] == "completed"
    gate = json.loads(
        (tmp_path / "run" / "release_gate.json").read_text(encoding="utf-8")
    )
    assert Path(gate["model"]).samefile(model)
    manifest = json.loads(
        (tmp_path / "release" / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert all(manifest["gate"][key] == value for key, value in gate.items())
    assert manifest["gate"]["model_binding"] == "matched"
    assert manifest["gate"]["accepted"] is True


def test_ungated_export_needs_no_evaluation(tmp_path, passing_evaluation):
    cfg = _config(tmp_path, _checkpoint(tmp_path, "model"), ["export"])
    cfg["eval"]["enabled"] = False
    cfg["eval"]["gate"]["enabled"] = False

    results = TrainingOrchestrator.from_dict(cfg).run()
    assert results["export"]["status"] == "completed"
    assert passing_evaluation == []


def test_optional_export_drops_unrelated_gate(tmp_path):
    model_a = _checkpoint(tmp_path, "model_a")
    cfg = _config(tmp_path, _checkpoint(tmp_path, "model_b"), ["export"])
    cfg["eval"]["gate"]["enabled"] = False
    orchestrator = TrainingOrchestrator.from_dict(cfg)
    orchestrator._stage_results["eval"] = {
        "model": str(model_a),
        "release_gate": {"enabled": True, "accepted": True},
    }

    results = orchestrator.run()
    manifest = json.loads(
        Path(results["export"]["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["gate"] is None


def test_required_export_rejects_legacy_path_only_evaluation_binding(tmp_path):
    model = _checkpoint(tmp_path, "model")
    cfg = _config(tmp_path, model, ["export"])
    cfg["export"]["require_gate"] = True
    orchestrator = TrainingOrchestrator.from_dict(cfg)
    # Older state files store model identity beside the gate, not inside it.
    orchestrator._stage_results["eval"] = {
        "model": str(model),
        "release_gate": {"enabled": True, "accepted": True},
    }

    with pytest.raises(ValueError, match="unbound"):
        orchestrator.run()
    assert not (tmp_path / "release").exists()


def test_failed_eval_is_retried_on_resume(tmp_path, monkeypatch):
    cfg = _config(tmp_path, tmp_path / "mocked-model", ["eval"])
    cfg["eval"]["fail_on_error"] = False
    calls = []

    def evaluate(self, **kwargs):
        calls.append(kwargs["model_path"])
        if len(calls) == 1:
            return {"error": "transient evaluation failure"}
        return {"model": kwargs["model_path"], "metrics": {"perplexity": {"score": 10}}}

    monkeypatch.setattr(Evaluator, "evaluate", evaluate)
    assert TrainingOrchestrator.from_dict(cfg).run()["eval"]["status"] == "failed"
    state = json.loads(
        (tmp_path / "run" / "pipeline_state.json").read_text(encoding="utf-8")
    )
    assert state["stages"]["eval"]["status"] == "failed"

    cfg["pipeline"] = {"resume": True}
    assert TrainingOrchestrator.from_dict(cfg).run()["eval"]["status"] == "completed"
    assert len(calls) == 2


@pytest.mark.parametrize("status", ["failed", "rejected", "blocked"])
@pytest.mark.parametrize("legacy", [False, True])
def test_resume_retries_unsuccessful_stage_and_invalidates_descendants(
    tmp_path, status, legacy
):
    plan = PipelinePlan.compile(
        ["eval", "independent", "release"],
        dependencies={"eval": [], "independent": [], "release": ["eval"]},
    )
    path = tmp_path / "state.json"
    store = PipelineStateStore(path, run_fingerprint="stable", plan=plan)
    store.prepare()
    artifacts = {name: {"producer": name} for name in plan.stages}
    for name in plan.stages:
        store.begin(name)
        store.finish(
            name,
            {"status": status if name == "eval" else "completed"},
            artifacts=artifacts,
        )
    if legacy:
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["stages"]["eval"]["status"] = "completed"
        saved["stages"]["eval"].pop("result_status")
        path.write_text(json.dumps(saved), encoding="utf-8")

    resumed = PipelineStateStore(path, run_fingerprint="stable", plan=plan)
    assert resumed.prepare(resume=True) == ("independent",)
    assert set(resumed.restored_artifacts()) == {"independent"}
    assert resumed.state["invalidated_stages"] == ["eval", "release"]
    resumed.begin("eval", write=False)
    assert resumed.state["stages"]["eval"]["attempt"] == 2


@pytest.mark.parametrize(
    "stages", [["vision_alignment"], ["vision_alignment", "mllm_sft"]]
)
@pytest.mark.parametrize("with_data", [False, True])
def test_multimodal_stage_results_artifacts_and_resume_use_actual_stage(
    tmp_path, stages, with_data
):
    cfg = _config(tmp_path, tmp_path / "unused-model", stages)
    cfg["eval"]["enabled"] = False
    cfg["eval"]["gate"]["enabled"] = False
    if with_data:
        data = tmp_path / "samples.jsonl"
        data.write_text(
            json.dumps(
                {
                    "images": ["scene.png"],
                    "messages": [
                        {"role": "user", "content": "describe"},
                        {"role": "assistant", "content": "a scene"},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cfg["multimodal"] = {"enabled": True, "data_path": str(data)}
    orchestrator = TrainingOrchestrator.from_dict(cfg)
    results = orchestrator.run()
    assert set(results) == set(stages)
    paths = []
    for stage in stages:
        result = results[stage]
        plan_path = Path(result["plan_path"])
        paths.append(plan_path)
        assert json.loads(plan_path.read_text(encoding="utf-8"))["stage"] == stage
        assert orchestrator.artifacts[f"{stage}.plan_path"].producer == stage
        if with_data:
            normalized = json.loads(
                Path(result["normalized_data_path"]).read_text(encoding="utf-8")
            )
            assert normalized["task"] == stage
    assert len(set(paths)) == len(stages)

    cfg["pipeline"] = {"resume": True}
    resumed = TrainingOrchestrator.from_dict(cfg)
    assert resumed.run() == results
    assert resumed._restored_stages == tuple(stages)
