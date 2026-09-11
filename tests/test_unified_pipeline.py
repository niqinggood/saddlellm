from __future__ import annotations

import json

import pytest

import saddlellm.framework.pipeline as pipeline_module
from saddlellm.framework import (
    ArtifactRegistry,
    FunctionStage,
    PipelinePlan,
    PipelinePlanError,
    PipelineStateError,
    PipelineStateStore,
    StageCapabilities,
    StageRegistry,
)
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def test_pipeline_plan_compiles_stable_dag_levels_and_contracts():
    capabilities = {
        "foundation": StageCapabilities(
            produces=frozenset({"language_model"}),
            tags=frozenset({"family:foundation"}),
        ),
        "dynamics": StageCapabilities(
            consumes=frozenset({"trajectory"}),
            produces=frozenset({"world_model"}),
            tags=frozenset({"family:world_model"}),
        ),
        "release": StageCapabilities(
            consumes=frozenset({"language_model", "world_model"}),
            produces=frozenset({"release"}),
            tags=frozenset({"family:release"}),
        ),
    }
    plan = PipelinePlan.compile(
        ["release", "dynamics", "foundation"],
        dependencies={
            "dynamics": [],
            "foundation": [],
            "release": ["foundation", "dynamics"],
        },
        capabilities=capabilities,
    )

    assert plan.stages == ("dynamics", "foundation", "release")
    assert plan.levels == (("dynamics", "foundation"), ("release",))
    assert plan.nodes[-1].family == "release"
    assert plan.nodes[-1].consumes == ("language_model", "world_model")
    assert plan.descendants(["dynamics"]) == ("dynamics", "release")
    assert len(plan.signature()) == 64


@pytest.mark.parametrize(
    "stages,dependencies,match",
    [
        (["a", "a"], None, "unique"),
        (["a"], {"a": ["missing"]}, "unknown stages"),
        (["a", "b"], {"a": ["b"], "b": ["a"]}, "cycle"),
        (["a"], {"a": ["a"]}, "cannot depend on itself"),
    ],
)
def test_pipeline_plan_rejects_ambiguous_or_cyclic_graphs(stages, dependencies, match):
    with pytest.raises(PipelinePlanError, match=match):
        PipelinePlan.compile(stages, dependencies=dependencies)


def test_artifact_registry_captures_implicit_and_explicit_outputs(tmp_path):
    registry = ArtifactRegistry()
    registry.capture(
        "world_model",
        {
            "output_dir": str(tmp_path / "world"),
            "plan_path": str(tmp_path / "plan.json"),
            "artifacts": {
                "metrics": {
                    "kind": "evaluation",
                    "path": str(tmp_path / "metrics.json"),
                    "metadata": {"split": "validation"},
                }
            },
        },
    )

    assert registry["world_model.output_dir"].kind == "world_model"
    assert registry["world_model.plan_path"].kind == "plan"
    assert registry["world_model.metrics"].kind == "evaluation"
    assert registry.latest("world_model").producer == "world_model"
    assert registry["world_model.metrics"].metadata["split"] == "validation"


def _recording_registry(calls, tmp_path):
    registry = StageRegistry(entry_point_group=None)

    def make_stage(name, family, kind):
        def run(context):
            calls.append(name)
            return {
                "status": "completed",
                "output_path": str(tmp_path / name),
                "seen_results": sorted(context.results),
                "artifacts": {
                    name: {
                        "kind": kind,
                        "path": str(tmp_path / name),
                    }
                },
            }

        return FunctionStage(
            name,
            run,
            capabilities=StageCapabilities(
                produces=frozenset({kind}),
                tags=frozenset({f"family:{family}"}),
            ),
        )

    registry.register(
        "foundation",
        make_stage("foundation", "foundation", "language_model"),
    )
    registry.register(
        "dynamics",
        make_stage("dynamics", "world_model", "world_model"),
    )
    registry.register(
        "release",
        make_stage("release", "release", "release"),
    )
    return registry


def _pipeline_document(tmp_path, *, resume=False, rerun=None):
    return {
        "project": "unified-test",
        "stages": ["release", "dynamics", "foundation"],
        "pipeline": {
            "dependencies": {
                "dynamics": [],
                "foundation": [],
                "release": ["foundation", "dynamics"],
            },
            "resume": resume,
            "rerun": list(rerun or []),
        },
        "distributed": {"bf16": False, "fp16": False},
        "logging": {"backend": "local", "output_dir": str(tmp_path / "run")},
    }


def test_orchestrator_resumes_completed_dag_and_reruns_descendants(tmp_path):
    calls = []
    registry = _recording_registry(calls, tmp_path)

    first = TrainingOrchestrator.from_dict(
        _pipeline_document(tmp_path), stage_registry=registry
    )
    first_results = first.run()
    assert calls == ["dynamics", "foundation", "release"]
    assert first_results["release"]["seen_results"] == ["dynamics", "foundation"]
    assert first.pipeline_plan.levels == (("dynamics", "foundation"), ("release",))
    assert first.artifacts["dynamics.dynamics"].kind == "world_model"

    state_path = tmp_path / "run" / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["plan"]["dependency_mode"] == "dag"
    assert set(state["artifacts"]) >= {
        "foundation.foundation",
        "dynamics.dynamics",
        "release.release",
    }

    resumed = TrainingOrchestrator.from_dict(
        _pipeline_document(tmp_path, resume=True), stage_registry=registry
    )
    resumed_results = resumed.run()
    assert calls == ["dynamics", "foundation", "release"]
    assert resumed._restored_stages == ("dynamics", "foundation", "release")
    assert resumed_results["release"]["status"] == "completed"

    rerun = TrainingOrchestrator.from_dict(
        _pipeline_document(tmp_path, resume=True, rerun=["dynamics"]),
        stage_registry=registry,
    )
    rerun.run()
    assert calls == [
        "dynamics",
        "foundation",
        "release",
        "dynamics",
        "release",
    ]
    assert rerun._restored_stages == ("foundation",)


def test_pipeline_state_refuses_configuration_drift(tmp_path):
    path = tmp_path / "state.json"
    plan = PipelinePlan.compile(["a"])
    first = PipelineStateStore(path, run_fingerprint="first", plan=plan)
    first.prepare(write=True)
    first.finish("a", {"status": "completed"}, write=True)

    changed = PipelineStateStore(path, run_fingerprint="changed", plan=plan)
    with pytest.raises(PipelineStateError, match="different normalized configuration"):
        changed.prepare(resume=True, write=False)


def test_pipeline_state_retries_transient_windows_replace_lock(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    plan = PipelinePlan.compile(["a"])
    store = PipelineStateStore(path, run_fingerprint="stable", plan=plan)
    original_replace = pipeline_module.os.replace
    attempts = []

    def flaky_replace(source, destination):
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError("temporarily locked")
        return original_replace(source, destination)

    monkeypatch.setattr(pipeline_module.os, "replace", flaky_replace)
    monkeypatch.setattr(pipeline_module.time, "sleep", lambda _: None)

    store.prepare(write=True)

    assert path.is_file()
    assert len(attempts) == 3
