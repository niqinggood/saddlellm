"""Evaluation, release gates, benchmarks, and smoke tests."""

from .CompletionEvaluator import (
    CompletionEvaluationRequest,
    completion_markdown,
    completion_summary,
    evaluate_completions,
)
from .CheckpointSelection import select_checkpoint
from .DistillationEvaluator import (
    DistillationEvaluationRequest,
    evaluate_distillation_pair,
    infer_distillation_evaluation_task,
)
from .EvaluationComparison import (
    build_evaluation_comparison,
    comparison_markdown,
    paired_metrics,
)
from .EvaluationRunner import run_evaluation, summarize_evaluation
from .MultipleChoiceEvaluator import (
    MultipleChoiceEvaluationRequest,
    evaluate_multiple_choice,
    evaluation_markdown,
    evaluation_summary,
)
from .RegressionSuite import (
    compare_regression_suite,
    load_regression_suite,
    run_regression_suite,
)
from .ReleaseGate import release_gate_accepted
from .RubricEvaluator import (
    RubricEvaluationRequest,
    evaluate_rubric,
    rubric_markdown,
    rubric_summary,
)

__all__ = [
    "CompletionEvaluationRequest",
    "DistillationEvaluationRequest",
    "MultipleChoiceEvaluationRequest",
    "RubricEvaluationRequest",
    "build_evaluation_comparison",
    "compare_regression_suite",
    "comparison_markdown",
    "completion_markdown",
    "completion_summary",
    "evaluate_completions",
    "evaluate_distillation_pair",
    "evaluate_multiple_choice",
    "evaluate_rubric",
    "evaluation_markdown",
    "evaluation_summary",
    "infer_distillation_evaluation_task",
    "load_regression_suite",
    "paired_metrics",
    "release_gate_accepted",
    "select_checkpoint",
    "rubric_markdown",
    "rubric_summary",
    "run_evaluation",
    "run_regression_suite",
    "summarize_evaluation",
]
