import json

import pytest

from saddlellm.evaluation import DistillationEvaluator as module
from saddlellm.evaluation.DistillationEvaluator import (
    DistillationEvaluationRequest,
    evaluate_distillation_pair,
    infer_distillation_evaluation_task,
)


def test_distillation_evaluation_task_auto_detection(tmp_path) -> None:
    multiple_choice = tmp_path / "mcq.jsonl"
    multiple_choice.write_text(
        '{"question":"q","instruction":"prompt","answer":"B"}\n',
        encoding="utf-8",
    )
    completion = tmp_path / "completion.jsonl"
    completion.write_text(
        '{"instruction":"q","output":"answer"}\n', encoding="utf-8"
    )

    assert infer_distillation_evaluation_task(str(multiple_choice)) == "multiple-choice"
    assert infer_distillation_evaluation_task(str(completion)) == "completion"


def test_multiple_choice_distillation_evaluation_compares_baseline_and_adapter(
    tmp_path, monkeypatch
) -> None:
    def fake_evaluate(request):
        candidate = bool(request.adapter)
        return {
            "model": request.model,
            "adapter": request.adapter,
            "accuracy": 1.0 if candidate else 0.0,
            "details": [
                {
                    "record_id": "1",
                    "question": "q",
                    "answer": "A",
                    "prediction": "A" if candidate else "B",
                    "correct": candidate,
                }
            ],
        }

    monkeypatch.setattr(module, "evaluate_multiple_choice", fake_evaluate)
    monkeypatch.setattr(module, "save_evaluation", lambda *_args: None)
    result = evaluate_distillation_pair(
        DistillationEvaluationRequest(
            baseline_model="base",
            candidate_model="base",
            candidate_adapter="adapter",
            data=str(tmp_path / "eval.jsonl"),
            output_dir=str(tmp_path / "evaluation"),
            task="multiple-choice",
            generation_samples=0,
        )
    )

    assert result["comparison"]["outcome"] == "improved"
    assert result["comparison"]["absolute_gain_percentage_points"] == 100.0
    assert result["candidate"]["adapter"] == "adapter"
    saved = json.loads(
        (tmp_path / "evaluation" / "comparison.json").read_text(encoding="utf-8")
    )
    assert saved["comparison_result"].endswith("comparison.json")


def test_completion_distillation_evaluation_reports_lower_ppl_as_improvement(
    tmp_path, monkeypatch
) -> None:
    def fake_evaluate(request):
        candidate = bool(request.adapter)
        return {
            "model": request.model,
            "adapter": request.adapter,
            "samples": 2,
            "completion_tokens": 10,
            "completion_nll": 1.0 if candidate else 2.0,
            "completion_perplexity": 3.0 if candidate else 6.0,
        }

    monkeypatch.setattr(module, "evaluate_completions", fake_evaluate)
    monkeypatch.setattr(module, "save_completion_evaluation", lambda *_args: None)
    result = evaluate_distillation_pair(
        DistillationEvaluationRequest(
            baseline_model="base",
            candidate_model="base",
            candidate_adapter="adapter",
            data=str(tmp_path / "eval.jsonl"),
            output_dir=str(tmp_path / "evaluation"),
            task="completion",
            generation_samples=0,
        )
    )

    comparison = result["comparison"]
    assert comparison["outcome"] == "improved"
    assert comparison["absolute_improvement"] == 3.0
    assert comparison["relative_improvement_percent"] == 50.0


def test_completion_distillation_rejects_prompt_variant_options(tmp_path) -> None:
    with pytest.raises(ValueError, match="only supported for multiple-choice"):
        evaluate_distillation_pair(
            DistillationEvaluationRequest(
                baseline_model="base",
                candidate_model="candidate",
                data=str(tmp_path / "eval.jsonl"),
                output_dir=str(tmp_path / "evaluation"),
                task="completion",
                prompt_variants="variants.json",
            )
        )
