"""Evaluation must use the declared cohort and retain model-byte identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from saddlellm.evaluation.EvaluationRunner import run_evaluation, summarize_evaluation
from saddlellm.evaluation.LLModelEvalute import Evaluator
from saddlellm.evaluation.MultipleChoiceEvaluator import load_evaluation_records
from saddlellm.training.TrainingOrchestrator import EvalConfig, TrainingOrchestrator


def corpus(tmp_path):
    path = tmp_path / "cohorts.json"
    path.write_text(
        json.dumps(
            {
                "train": [
                    {
                        "id": "train-only",
                        "instruction": "train question",
                        "output": "train answer",
                        "answer": "A",
                    }
                ],
                "validation": [
                    {
                        "id": "validation-only",
                        "instruction": "validation question",
                        "output": "validation answer",
                        "answer": "A",
                        "rubric": [{"type": "contains_any", "values": ["answer"]}],
                    }
                ],
                "test": [
                    {
                        "id": "sealed-test-only",
                        "instruction": "test question",
                        "output": "test answer",
                        "answer": "B",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def model_artifact(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"immutable-model-byte-fixture")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return path


def stub_task(monkeypatch, task):
    if task == "completion":
        from saddlellm.evaluation import CompletionEvaluator as backend

        function, report_task, scores = (
            "evaluate_completions",
            "completion",
            {
                "completion_perplexity": 2.0,
                "completion_nll": 0.7,
                "completion_tokens": 1,
            },
        )
    elif task == "rubric":
        from saddlellm.evaluation import RubricEvaluator as backend

        function, report_task, scores = (
            "evaluate_rubric",
            "rubric",
            {"pass_rate": 1.0, "mean_score": 1.0, "critical_failure_count": 0},
        )
    else:
        from saddlellm.evaluation import MultipleChoiceEvaluator as backend

        function, report_task, scores = (
            "evaluate_multiple_choice",
            "multiple_choice",
            {"accuracy": 1.0, "correct": 1},
        )
    seen = []

    def evaluate(request):
        rows = load_evaluation_records(request.data, request.limit, split=request.split)
        seen.append((request, rows))
        return {
            "schema_version": 1,
            "task_type": report_task,
            "model": request.model,
            "adapter": request.adapter,
            "data": request.data,
            "data_sha256": hashlib.sha256(Path(request.data).read_bytes()).hexdigest(),
            "samples": len(rows),
            "details": [
                {"id": row[request.id_field], "prompt": row[request.prompt_field]}
                for row in rows
            ],
            **scores,
        }

    monkeypatch.setattr(backend, function, evaluate)
    return seen


def pipeline_config(
    tmp_path, model, data, *, task="completion", split="validation", fail_on_error=True
):
    return {
        "model": {"name_or_path": str(model)},
        "stages": ["eval"],
        "distributed": {"bf16": False, "fp16": False},
        "logging": {"output_dir": str(tmp_path / "run")},
        "eval": {
            "task": task,
            "data": str(data),
            "split": split,
            "max_samples": 2,
            "fail_on_error": fail_on_error,
        },
    }


def test_legacy_remote_missing_validation_does_not_request_test(monkeypatch):
    import saddlellm.evaluation.LLModelEvalute as legacy

    calls = []

    def missing(dataset, config, *, split):
        calls.append(split)
        raise ValueError("validation not available")

    monkeypatch.setattr(legacy, "load_dataset", missing)
    monkeypatch.setattr(
        Evaluator,
        "_load_model",
        lambda *args: pytest.fail("must not load a model after split failure"),
    )
    result = Evaluator().evaluate("unused", "remote-example", split="validation")
    assert calls == ["validation"]
    assert result["error"] == "validation not available"
    assert result["data_split"] == "validation"


def test_local_named_split_never_selects_first_train_cohort(tmp_path):
    source = corpus(tmp_path)
    assert [
        row["id"]
        for row in Evaluator()._load_records(str(source), None, "validation", 10)
    ] == ["validation-only"]
    assert [
        row["id"] for row in load_evaluation_records(str(source), split="test")
    ] == ["sealed-test-only"]
    with pytest.raises(ValueError, match="multiple splits"):
        load_evaluation_records(str(source))
    with pytest.raises(ValueError, match="missing"):
        Evaluator()._load_records(str(source), None, "absent", 10)


def test_single_file_scope_declaration_preserves_exact_rows(tmp_path):
    source = tmp_path / "validation.jsonl"
    source.write_text('{"id":"one","instruction":"q","output":"a"}\n', encoding="utf-8")
    assert load_evaluation_records(
        str(source), split="validation"
    ) == load_evaluation_records(str(source))


def test_requested_missing_validation_does_not_use_single_test_array(tmp_path):
    source = tmp_path / "test-only.json"
    source.write_text('{"test":[{"id":"sealed-only"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="validation.*missing"):
        load_evaluation_records(str(source), split="validation")
    with pytest.raises(ValueError, match="validation.*missing"):
        Evaluator()._load_records(str(source), None, "validation", 10)


@pytest.mark.parametrize("task", ["multiple-choice", "completion", "rubric"])
def test_public_runner_reuses_task_reports_on_selected_split(
    tmp_path, monkeypatch, task
):
    source, model = corpus(tmp_path), model_artifact(tmp_path)
    seen = stub_task(monkeypatch, task)
    target = tmp_path / "report.json"
    result = run_evaluation(
        str(model), str(source), task=task, split="validation", output=str(target)
    )
    assert [row["id"] for row in seen[0][1]] == ["validation-only"]
    assert result["details"][0]["id"] == "validation-only"
    assert result["data"] == str(source)
    assert result["data_split"] == "validation"
    assert result["split_selection"] == "named_split"
    assert result["model_identity"]["status"] == "verified"
    assert (
        json.loads(target.read_text(encoding="utf-8"))["model_identity"]
        == result["model_identity"]
    )
    assert summarize_evaluation(result)["output"] == str(target.resolve())


@pytest.mark.parametrize("task", ["multiple-choice", "completion", "rubric"])
def test_task_evaluator_passes_its_split_to_real_loader(tmp_path, monkeypatch, task):
    source = corpus(tmp_path)
    if task == "completion":
        from saddlellm.evaluation import CompletionEvaluator as backend

        request_class, function = (
            backend.CompletionEvaluationRequest,
            backend.evaluate_completions,
        )
    elif task == "rubric":
        from saddlellm.evaluation import RubricEvaluator as backend

        request_class, function = (
            backend.RubricEvaluationRequest,
            backend.evaluate_rubric,
        )
    else:
        from saddlellm.evaluation import MultipleChoiceEvaluator as backend

        request_class, function = (
            backend.MultipleChoiceEvaluationRequest,
            backend.evaluate_multiple_choice,
        )

    class CohortReached(Exception):
        pass

    def inspect(path, limit=0, split=None):
        assert [row["id"] for row in load_evaluation_records(path, limit, split)] == [
            "validation-only"
        ]
        raise CohortReached

    monkeypatch.setattr(backend, "load_evaluation_records", inspect)
    with pytest.raises(CohortReached):
        function(
            request_class(model="never-loaded", data=str(source), split="validation")
        )


@pytest.mark.parametrize("task", ["multiple-choice", "completion", "rubric"])
def test_pipeline_and_public_api_share_report_contract(tmp_path, monkeypatch, task):
    source, model = corpus(tmp_path), model_artifact(tmp_path)
    seen = stub_task(monkeypatch, task)
    config = pipeline_config(tmp_path, model, source, task=task)
    config["eval"]["gate"] = {
        "enabled": True,
        "rules": {
            "accuracy"
            if task == "multiple-choice"
            else "perplexity"
            if task == "completion"
            else "pass_rate": {"max": 3} if task == "completion" else {"min": 0.5}
        },
    }
    results = TrainingOrchestrator.from_dict(config).run()
    report = results["eval"]
    assert seen[0][0].split == "validation"
    assert report["details"][0]["id"] == "validation-only"
    assert report["release_gate"]["accepted"] is True
    assert report["release_gate"]["model_identity"] == report["model_identity"]
    disk = json.loads(
        (tmp_path / "run/eval/evaluation_results.json").read_text(encoding="utf-8")
    )
    assert disk["task_type"] == report["task_type"]
    assert disk["details"] == report["details"]


@pytest.mark.parametrize("fail_on_error", [True, False])
def test_pipeline_split_failure_policy_is_explicit(
    tmp_path, monkeypatch, fail_on_error
):
    source, model = corpus(tmp_path), model_artifact(tmp_path)
    seen = stub_task(monkeypatch, "completion")
    pipeline = TrainingOrchestrator.from_dict(
        pipeline_config(
            tmp_path, model, source, split="absent", fail_on_error=fail_on_error
        )
    )
    if fail_on_error:
        with pytest.raises(ValueError, match="missing"):
            pipeline.run()
    else:
        assert pipeline.run()["eval"]["status"] == "failed"
    assert seen == []
    failure = json.loads(
        (tmp_path / "run/eval/evaluation_results.json").read_text(encoding="utf-8")
    )
    assert failure["data_split"] == "absent"
    assert "missing" in failure["error"]


def test_legacy_config_retains_metrics_and_validation_default(tmp_path, monkeypatch):
    assert EvalConfig().split == "validation"
    source, model = corpus(tmp_path), model_artifact(tmp_path)
    calls = []

    def legacy(self, **kwargs):
        calls.append(kwargs)
        return {"metrics": {"perplexity": {"score": 2}}, "error": None}

    monkeypatch.setattr(Evaluator, "evaluate", legacy)
    cfg = pipeline_config(tmp_path, model, source)
    cfg["eval"].pop("task")
    cfg["eval"]["tasks"] = ["perplexity"]
    result = TrainingOrchestrator.from_dict(cfg).run()["eval"]
    assert calls[0]["split"] == "validation"
    assert calls[0]["metrics"] == ["perplexity"]
    assert result["metrics"]["perplexity"]["score"] == 2
    assert result["model_identity"]["status"] == "verified"


def test_same_path_model_mutation_changes_evaluation_binding(tmp_path, monkeypatch):
    source, model = corpus(tmp_path), model_artifact(tmp_path)
    stub_task(monkeypatch, "completion")
    before = run_evaluation(
        str(model), str(source), task="completion", split="validation"
    )
    (model / "model.safetensors").write_bytes(b"different-model-byte-fixture")
    after = run_evaluation(
        str(model), str(source), task="completion", split="validation"
    )
    assert (
        before["model_identity"]["fingerprint"]
        != after["model_identity"]["fingerprint"]
    )


def test_same_container_has_distinct_selected_cohort_identity(tmp_path, monkeypatch):
    source, model = corpus(tmp_path), model_artifact(tmp_path)
    stub_task(monkeypatch, "completion")
    validation = run_evaluation(
        str(model), str(source), task="completion", split="validation"
    )
    test = run_evaluation(str(model), str(source), task="completion", split="test")
    assert validation["data_sha256"] == test["data_sha256"]
    assert (
        validation["evaluation_identity"]["records_sha256"]
        != test["evaluation_identity"]["records_sha256"]
    )
    assert validation["evaluation_identity"]["source_records"] == 1


def test_pipeline_configurable_task_data_fields_reach_public_request(
    tmp_path, monkeypatch
):
    model = model_artifact(tmp_path)
    source = tmp_path / "custom.jsonl"
    source.write_text(
        '{"sample_key":"custom-one","source_text":"custom question","gold_answer":"custom answer"}\n',
        encoding="utf-8",
    )
    seen = stub_task(monkeypatch, "completion")
    cfg = pipeline_config(tmp_path, model, source)
    cfg["eval"].update(
        prompt_field="source_text",
        completion_field="gold_answer",
        id_field="sample_key",
        max_length=256,
        generation_all=True,
    )
    result = TrainingOrchestrator.from_dict(cfg).run()["eval"]
    request = seen[0][0]
    assert (request.prompt_field, request.completion_field, request.id_field) == (
        "source_text",
        "gold_answer",
        "sample_key",
    )
    assert request.max_length == 256
    assert request.generation_samples > 1000
    assert result["details"] == [{"id": "custom-one", "prompt": "custom question"}]


def test_remote_rubric_does_not_claim_local_weight_identity(tmp_path, monkeypatch):
    source, model = corpus(tmp_path), model_artifact(tmp_path)
    stub_task(monkeypatch, "rubric")
    report = run_evaluation(
        str(model),
        str(source),
        task="rubric",
        split="validation",
        endpoint="http://localhost:9999/v1",
    )
    assert report["model_identity"]["status"] == "unresolved"
    assert report["model_identity"]["fingerprint"] is None
