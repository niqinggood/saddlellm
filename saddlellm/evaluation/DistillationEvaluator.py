"""Paired capability evaluation for distilled models.

The distillation trainer answers whether optimization completed.  This module
answers the separate question that matters to users: how the trained student
performs against its pre-distillation baseline on independent data.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .CompletionEvaluator import (
    CompletionEvaluationRequest,
    evaluate_completions,
    save_completion_evaluation,
)
from .EvaluationComparison import paired_metrics
from .MultipleChoiceEvaluator import (
    MultipleChoiceEvaluationRequest,
    evaluate_multiple_choice,
    load_evaluation_records,
    save_evaluation,
)


@dataclass
class DistillationEvaluationRequest:
    baseline_model: str
    candidate_model: str
    data: str
    output_dir: str
    baseline_adapter: str | None = None
    candidate_adapter: str | None = None
    task: str = "auto"
    choices: str = "ABCDE"
    case_field: str | None = None
    variant_field: str | None = None
    prompt_variants: str | None = None
    limit: int = 0
    batch_size: int = 16
    max_length: int = 512
    max_new_tokens: int = 96
    generation_samples: int = 3
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False


def infer_distillation_evaluation_task(
    data: str, requested: str = "auto", choices: str = "ABCDE"
) -> str:
    """Resolve ``auto`` using the first record while keeping failures explicit."""

    if requested not in {"auto", "multiple-choice", "completion"}:
        raise ValueError("evaluation task must be auto, multiple-choice, or completion")
    if requested != "auto":
        return requested
    rows = load_evaluation_records(data, limit=1)
    if not rows:
        raise ValueError("distillation evaluation dataset is empty")
    row = rows[0]
    answer = str(row.get("answer", "")).strip().upper()
    if answer in set(choices) and (row.get("question") or row.get("instruction")):
        return "multiple-choice"
    if row.get("instruction") and row.get("output"):
        return "completion"
    raise ValueError(
        "Cannot auto-detect evaluation task. Use question/answer records for "
        "multiple-choice, instruction/output records for completion, or pass --eval-task."
    )


def _release_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _direction(change: float, tolerance: float = 1e-12) -> str:
    if change > tolerance:
        return "improved"
    if change < -tolerance:
        return "degraded"
    return "unchanged"


def _multiple_choice_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    paired = paired_metrics(baseline, candidate)
    baseline_accuracy = float(paired["base_accuracy"])
    candidate_accuracy = float(paired["candidate_accuracy"])
    gain = float(paired["absolute_gain_percentage_points"])
    result = {
        "primary_metric": "accuracy",
        "higher_is_better": True,
        "baseline": baseline_accuracy,
        "candidate": candidate_accuracy,
        "absolute_gain_percentage_points": gain,
        "relative_gain_percent": paired["relative_gain_percent"],
        "capability_retention_percent": (
            candidate_accuracy / baseline_accuracy * 100
            if baseline_accuracy
            else None
        ),
        "outcome": _direction(gain),
        "mcnemar_exact_two_sided_p": paired["mcnemar_exact_two_sided_p"],
        "statistically_significant_at_0_05": (
            paired["mcnemar_exact_two_sided_p"] < 0.05
        ),
        "base_wrong_candidate_right": paired["base_wrong_candidate_right"],
        "base_right_candidate_wrong": paired["base_right_candidate_wrong"],
        "samples": paired["samples"],
    }
    baseline_robustness = baseline.get("prompt_robustness")
    candidate_robustness = candidate.get("prompt_robustness")
    if baseline_robustness and candidate_robustness:
        result["prompt_robustness"] = {
            "baseline": {
                "prediction_flip_rate": baseline_robustness["prediction_flip_rate"],
                "all_variants_correct_rate": baseline_robustness[
                    "all_variants_correct_rate"
                ],
                "worst_variant_accuracy": baseline_robustness[
                    "worst_variant_accuracy"
                ],
            },
            "candidate": {
                "prediction_flip_rate": candidate_robustness["prediction_flip_rate"],
                "all_variants_correct_rate": candidate_robustness[
                    "all_variants_correct_rate"
                ],
                "worst_variant_accuracy": candidate_robustness[
                    "worst_variant_accuracy"
                ],
            },
            "all_variants_correct_gain_percentage_points": (
                candidate_robustness["all_variants_correct_rate"]
                - baseline_robustness["all_variants_correct_rate"]
            )
            * 100,
            "prediction_flip_reduction_percentage_points": (
                baseline_robustness["prediction_flip_rate"]
                - candidate_robustness["prediction_flip_rate"]
            )
            * 100
            if baseline_robustness["prediction_flip_rate"] is not None
            and candidate_robustness["prediction_flip_rate"] is not None
            else None,
        }
    return result


def _completion_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_ppl = float(baseline["completion_perplexity"])
    candidate_ppl = float(candidate["completion_perplexity"])
    # Positive means the student improved because lower perplexity is better.
    improvement = baseline_ppl - candidate_ppl
    return {
        "primary_metric": "completion_perplexity",
        "higher_is_better": False,
        "baseline": baseline_ppl,
        "candidate": candidate_ppl,
        "absolute_improvement": improvement,
        "relative_improvement_percent": (
            improvement / baseline_ppl * 100 if baseline_ppl else None
        ),
        "baseline_completion_nll": baseline["completion_nll"],
        "candidate_completion_nll": candidate["completion_nll"],
        "outcome": _direction(improvement),
        "evidence": "descriptive; no significance test is inferred from aggregate PPL",
        "samples": candidate["samples"],
        "completion_tokens": candidate["completion_tokens"],
    }


def _report_markdown(result: Mapping[str, Any]) -> str:
    comparison = result["comparison"]
    lines = [
        "# SaddleLLM Distillation Evaluation",
        "",
        f"- Task: `{result['task']}`",
        f"- Outcome: **{comparison['outcome']}**",
        f"- Independent data: `{result['data']}`",
        f"- Samples: {comparison['samples']}",
        "",
        "| Model | Reference | Primary metric |",
        "|---|---|---:|",
        f"| Before distillation | `{result['baseline']['reference']}` | "
        f"{comparison['baseline']:.6f} |",
        f"| Distilled student | `{result['candidate']['reference']}` | "
        f"{comparison['candidate']:.6f} |",
        "",
    ]
    if result["task"] == "multiple-choice":
        lines.extend(
            [
                f"- Accuracy gain: {comparison['absolute_gain_percentage_points']:.2f} pp",
                f"- Capability retention: {comparison['capability_retention_percent']:.2f}%"
                if comparison["capability_retention_percent"] is not None
                else "- Capability retention: unavailable because baseline accuracy is zero",
                f"- Paired McNemar p: {comparison['mcnemar_exact_two_sided_p']:.6f}",
            ]
        )
        robustness = comparison.get("prompt_robustness")
        if robustness:
            lines.extend(
                [
                    f"- All-variants-correct gain: "
                    f"{robustness['all_variants_correct_gain_percentage_points']:.2f} pp",
                    f"- Prediction-flip reduction: "
                    f"{robustness['prediction_flip_reduction_percentage_points']:.2f} pp"
                    if robustness["prediction_flip_reduction_percentage_points"]
                    is not None
                    else "- Prediction-flip reduction: unavailable",
                ]
            )
    else:
        lines.extend(
            [
                f"- PPL improvement (positive is better): {comparison['absolute_improvement']:.6f}",
                f"- Relative PPL improvement: {comparison['relative_improvement_percent']:.2f}%"
                if comparison["relative_improvement_percent"] is not None
                else "- Relative PPL improvement: unavailable",
            ]
        )
    lines.extend(
        [
            "",
            "> This comparison measures one declared capability on one fixed dataset. "
            "It does not establish clinical safety or authorize release.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_distillation_pair(
    request: DistillationEvaluationRequest,
) -> dict[str, Any]:
    """Evaluate the baseline and student sequentially and save paired evidence."""

    if request.limit < 0 or request.generation_samples < 0:
        raise ValueError("evaluation limits cannot be negative")
    if request.batch_size < 1 or request.max_length < 1 or request.max_new_tokens < 1:
        raise ValueError("evaluation batch and token limits must be positive")
    output_dir = Path(request.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task = infer_distillation_evaluation_task(
        request.data, request.task, request.choices
    )
    if task != "multiple-choice" and (
        request.case_field or request.variant_field or request.prompt_variants
    ):
        raise ValueError(
            "prompt-variant evaluation options are only supported for multiple-choice data"
        )

    common = {
        "data": request.data,
        "limit": request.limit,
        "max_length": request.max_length,
        "max_new_tokens": request.max_new_tokens,
        "generation_samples": request.generation_samples,
        "device": request.device,
        "dtype": request.dtype,
        "trust_remote_code": request.trust_remote_code,
        "local_files_only": request.local_files_only,
    }
    baseline_path = output_dir / "baseline.json"
    candidate_path = output_dir / "candidate.json"
    baseline_markdown = output_dir / "baseline.md"
    candidate_markdown = output_dir / "candidate.md"

    if task == "multiple-choice":
        baseline_request = MultipleChoiceEvaluationRequest(
            model=request.baseline_model,
            adapter=request.baseline_adapter,
            choices=request.choices,
            case_field=request.case_field,
            variant_field=request.variant_field,
            prompt_variants=request.prompt_variants,
            batch_size=request.batch_size,
            **common,
        )
        candidate_request = MultipleChoiceEvaluationRequest(
            model=request.candidate_model,
            adapter=request.candidate_adapter,
            choices=request.choices,
            case_field=request.case_field,
            variant_field=request.variant_field,
            prompt_variants=request.prompt_variants,
            batch_size=request.batch_size,
            **common,
        )
        baseline = evaluate_multiple_choice(baseline_request)
        save_evaluation(baseline, str(baseline_path), str(baseline_markdown))
        _release_accelerator_cache()
        candidate = evaluate_multiple_choice(candidate_request)
        save_evaluation(candidate, str(candidate_path), str(candidate_markdown))
        comparison = _multiple_choice_comparison(baseline, candidate)
    else:
        baseline_request = CompletionEvaluationRequest(
            model=request.baseline_model,
            adapter=request.baseline_adapter,
            **common,
        )
        candidate_request = CompletionEvaluationRequest(
            model=request.candidate_model,
            adapter=request.candidate_adapter,
            **common,
        )
        baseline = evaluate_completions(baseline_request)
        save_completion_evaluation(
            baseline, str(baseline_path), str(baseline_markdown)
        )
        _release_accelerator_cache()
        candidate = evaluate_completions(candidate_request)
        save_completion_evaluation(
            candidate, str(candidate_path), str(candidate_markdown)
        )
        comparison = _completion_comparison(baseline, candidate)

    _release_accelerator_cache()
    comparison_path = output_dir / "comparison.json"
    report_path = output_dir / "report.md"
    result = {
        "schema_version": 1,
        "status": "completed",
        "task": task,
        "data": str(Path(request.data).expanduser().resolve()),
        "baseline": {
            "model": request.baseline_model,
            "adapter": request.baseline_adapter,
            "reference": request.baseline_adapter or request.baseline_model,
            "result": str(baseline_path),
        },
        "candidate": {
            "model": request.candidate_model,
            "adapter": request.candidate_adapter,
            "reference": request.candidate_adapter or request.candidate_model,
            "result": str(candidate_path),
        },
        "comparison": comparison,
        "comparison_result": str(comparison_path),
        "report": str(report_path),
        "limitations": [
            "Training data must not be reused as independent evaluation data.",
            "A capability improvement does not establish factual or clinical safety.",
            "Medical release still requires domain-specific safety tests and expert review.",
        ],
    }
    comparison_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(_report_markdown(result), encoding="utf-8")
    return result


__all__ = [
    "DistillationEvaluationRequest",
    "evaluate_distillation_pair",
    "infer_distillation_evaluation_task",
]
