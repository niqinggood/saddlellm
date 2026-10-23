"""Shared, lazy entry point for CLI and pipeline task evaluation.

Importing this module does not load model runtimes or inspect a dataset.
Existing task evaluators remain available as direct library entry points.
"""

from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any


def _task_backend(task: str):
    task = task.replace("_", "-")
    if task == "multiple-choice":
        from .MultipleChoiceEvaluator import (
            MultipleChoiceEvaluationRequest,
            evaluate_multiple_choice,
            evaluation_summary,
            save_evaluation,
        )

        return (
            MultipleChoiceEvaluationRequest,
            evaluate_multiple_choice,
            evaluation_summary,
            save_evaluation,
        )
    if task == "completion":
        from .CompletionEvaluator import (
            CompletionEvaluationRequest,
            completion_summary,
            evaluate_completions,
            save_completion_evaluation,
        )

        return (
            CompletionEvaluationRequest,
            evaluate_completions,
            completion_summary,
            save_completion_evaluation,
        )
    if task == "rubric":
        from .RubricEvaluator import (
            RubricEvaluationRequest,
            evaluate_rubric,
            rubric_summary,
            save_rubric_evaluation,
        )

        return (
            RubricEvaluationRequest,
            evaluate_rubric,
            rubric_summary,
            save_rubric_evaluation,
        )
    raise ValueError("evaluation task must be multiple-choice, completion, or rubric")


def _data_scope(data: str, split: str | None) -> tuple[str | None, str]:
    if split is not None and (not isinstance(split, str) or not split.strip()):
        raise ValueError("evaluation split must be a nonempty name")
    source = Path(data).expanduser()
    if not source.is_file():
        # Task evaluators own file validation; this keeps their direct APIs and
        # programmatic test doubles usable without downloading anything here.
        return split, "declared_file" if split else "file_rows"
    if source.suffix.lower() != ".json":
        return split, "declared_file" if split else "file_rows"
    from .MultipleChoiceEvaluator import select_evaluation_json

    value = json.loads(source.read_text(encoding="utf-8"))
    _, selected = select_evaluation_json(value, split)
    mapping = value.get("splits", value) if isinstance(value, dict) else None
    named = (
        isinstance(mapping, dict)
        and selected is not None
        and isinstance(mapping.get(selected), list)
    )
    return (
        selected,
        "named_split" if named else "declared_file" if split else "file_rows",
    )


def _records_identity(data: str, split: str | None, limit: int) -> dict[str, Any]:
    source = Path(data).expanduser()
    if not source.is_file():
        return {}
    from .MultipleChoiceEvaluator import load_evaluation_records

    rows = load_evaluation_records(str(source), limit, split=split)
    payload = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {
        "records_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "source_records": len(rows),
    }


def run_evaluation(
    model: str,
    data: str,
    *,
    task: str = "multiple-choice",
    split: str | None = None,
    output: str | None = None,
    markdown_output: str | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Evaluate with the same task contract used by the CLI.

    A named JSON split must exist; multi-split JSON without a split fails.
    A flat file's split is an explicit scope declaration, not an inferred
    membership claim. Errors propagate so callers can choose their failure
    policy. Reports keep the original source and task-specific detail schema.
    """
    task = task.replace("_", "-")
    request_class, evaluate, _, save = _task_backend(task)
    options = dict(options)
    generation_all = options.pop("generation_all", False)
    raw_indices = options.get("generation_indices", ())
    if isinstance(raw_indices, str):
        options["generation_indices"] = tuple(
            int(item.strip()) for item in raw_indices.split(",") if item.strip()
        )
    if generation_all:
        options["generation_samples"] = sys.maxsize
    if task != "multiple-choice" and any(
        options.get(key)
        for key in ("case_field", "variant_field", "prompt_variants", "group_field")
    ):
        raise ValueError(
            "case, variant, and group options require multiple-choice evaluation"
        )
    if task != "rubric" and any(
        options.get(key) for key in ("endpoint", "api_key_env")
    ):
        raise ValueError("endpoint and api_key_env require rubric evaluation")
    if task == "rubric" and (
        generation_all
        or options.get("generation_samples")
        or options.get("generation_indices")
    ):
        raise ValueError(
            "rubric evaluation generates every record; generation selection is not supported"
        )
    selected_split, split_selection = _data_scope(data, split)
    records_identity = _records_identity(data, selected_split, options.get("limit", 0))
    allowed = {item.name for item in fields(request_class)}
    known_options = {
        "adapter",
        "choices",
        "prompt_field",
        "question_field",
        "answer_field",
        "completion_field",
        "rubric_field",
        "reference_field",
        "id_field",
        "group_field",
        "case_field",
        "variant_field",
        "prompt_variants",
        "limit",
        "batch_size",
        "max_length",
        "generation_samples",
        "generation_indices",
        "max_new_tokens",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "device",
        "dtype",
        "enable_thinking",
        "trust_remote_code",
        "local_files_only",
        "endpoint",
        "api_key_env",
        "request_timeout",
        "pass_threshold",
    }
    unknown = set(options) - known_options
    if unknown:
        raise ValueError("Unknown evaluation options: " + ", ".join(sorted(unknown)))
    request = request_class(
        model=model,
        data=data,
        split=selected_split,
        **{key: value for key, value in options.items() if key in allowed},
    )
    from .ModelIdentity import compute_model_identity

    identity = compute_model_identity(model, adapter_path=options.get("adapter"))
    if task == "rubric" and options.get("endpoint"):
        # Local files cannot identify the weights actually loaded by a remote
        # service. Preserve the endpoint declaration without asserting binding.
        identity = {
            "schema_version": 1,
            "algorithm": "sha256",
            "status": "unresolved",
            "fingerprint": None,
            "model": {"reference": model},
            "endpoint": options["endpoint"],
            "warnings": [
                "Remote service model content was not verified by this evaluation."
            ],
        }
    result = evaluate(request)
    result["model_identity"] = identity
    result["data_split"] = selected_split
    result["split_selection"] = split_selection
    result["evaluation_identity"] = {
        "task": result.get("task_type", task),
        "data_sha256": result.get("data_sha256"),
        "split": selected_split,
        "limit": options.get("limit", 0),
        "samples": result.get("samples"),
        **records_identity,
    }
    if result.get("details") is not None:
        payload = json.dumps(
            result["details"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        result["evaluation_identity"]["details_sha256"] = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()
    if output:
        result["output"] = str(Path(output).expanduser().resolve())
        result["markdown_output"] = (
            str(Path(markdown_output).expanduser().resolve())
            if markdown_output
            else None
        )
        save(result, output, markdown_output)
    elif markdown_output:
        raise ValueError("markdown_output requires an evaluation JSON output path")
    return result


def summarize_evaluation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the existing task summary plus lifecycle provenance fields."""
    _, _, summarize, _ = _task_backend(str(result.get("task_type", "multiple-choice")))
    summary = summarize(result)
    for key in (
        "output",
        "markdown_output",
        "data_split",
        "split_selection",
        "model_identity",
        "evaluation_identity",
    ):
        if key in result:
            summary[key] = result[key]
    return summary


def evaluation_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Expose task report metrics to the existing optional pipeline gate."""
    keys = (
        "accuracy",
        "generation_accuracy",
        "completion_nll",
        "completion_perplexity",
        "pass_rate",
        "mean_score",
        "critical_failure_count",
        "critical_failure_rate",
        "by_group",
        "by_dimension",
        "prompt_robustness",
    )
    metrics = {key: result[key] for key in keys if key in result}
    if "completion_perplexity" in result:
        metrics["perplexity"] = {
            "score": result["completion_perplexity"],
            "scope": "completion_only",
        }
    return metrics


__all__ = ["evaluation_metrics", "run_evaluation", "summarize_evaluation"]
