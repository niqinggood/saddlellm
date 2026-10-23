"""Select explicit checkpoint candidates only after the shared regression gate."""

from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
# ruff: noqa: TRY004 -- Public configuration validation consistently raises ValueError.
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ModelIdentity import compute_model_identity, model_identities_match
from .RegressionSuite import (
    _digest,
    _provenance,
    _save,
    load_regression_suite,
    run_regression_suite,
    validate_regression_output,
)


class CheckpointSelectionError(ValueError):
    """No valid checkpoint qualified; ``report`` retains the rejection evidence."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _validate_destinations(
    suite: Mapping[str, Any] | str | Path,
    config: Mapping[str, Any],
    references: Sequence[str],
    destinations: Sequence[Path | None],
) -> None:
    resolved: set[Path] = set()
    for destination in destinations:
        if destination is None:
            continue
        path = Path(destination).expanduser().resolve()
        if path in resolved or path.is_dir():
            raise ValueError(
                "Checkpoint-selection report destinations must be distinct files"
            )
        if any(parent.exists() and not parent.is_dir() for parent in path.parents):
            raise ValueError("Checkpoint-selection report parents must be directories")
        resolved.add(path)
        for reference in references:
            validate_regression_output(
                config,
                path,
                candidate_model=reference,
                source_suite=suite if isinstance(suite, (str, Path)) else None,
            )


def _validate_selection_request(
    baseline_model: str | Path,
    candidates: Sequence[str | Path],
    suite: Mapping[str, Any] | str | Path,
    *,
    selection_task: str,
    selection_metric: str,
    higher_is_better: bool,
    evaluator: Callable[..., dict[str, Any]] | None,
) -> tuple[list[str], dict[str, Any]]:
    """Validate caller inputs before resolving outputs or reading model content."""
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not candidates
    ):
        raise ValueError(
            "candidates must be a nonempty explicit list of checkpoint paths"
        )
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be boolean")
    if evaluator is not None and not callable(evaluator):
        raise ValueError("evaluator must be callable")
    if any(
        not isinstance(model, (str, Path)) or not str(model).strip()
        for model in (baseline_model, *candidates)
    ):
        raise ValueError("Model references must be nonempty paths")

    candidate_models = [str(model) for model in candidates]
    canonical = [Path(model).expanduser().resolve() for model in candidate_models]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Candidate checkpoint paths must be unique")

    config = load_regression_suite(suite)
    selected_task = next(
        (task for task in config["tasks"] if task["name"] == selection_task),
        None,
    )
    if selected_task is None or selection_metric not in selected_task["metrics"]:
        raise ValueError(
            "selection_task and selection_metric must be declared in the regression suite"
        )
    if (
        selected_task["metrics"][selection_metric]["higher_is_better"]
        != higher_is_better
    ):
        raise ValueError(
            "Selection direction must match the regression-suite metric rule"
        )
    return candidate_models, config


def _evaluate_candidates(
    baseline_model: str | Path,
    candidate_models: Sequence[str],
    config: Mapping[str, Any],
    report_paths: Sequence[Path | None],
    identities: Sequence[dict[str, Any]],
    initial_data: Mapping[str, Any],
    *,
    selection_task: str,
    selection_metric: str,
    evaluator: Callable[..., dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the shared gate and collect only its accepted selection metrics."""
    entries = []
    selection_errors = []
    for index, (model, path) in enumerate(zip(candidate_models, report_paths)):
        regression = run_regression_suite(
            str(baseline_model),
            model,
            config,
            output=path,
            evaluator=evaluator,
        )
        errors = []
        if not model_identities_match(
            identities[0],
            regression.get("baseline_model_identity"),
        ):
            errors.append(
                "Baseline identity changed or is unbound during checkpoint selection"
            )
        if not model_identities_match(
            identities[index + 1],
            regression.get("candidate_model_identity"),
        ):
            errors.append(
                "Candidate identity changed or is unbound during checkpoint selection"
            )
        for task in regression.get("tasks", []):
            provenance = task.get("provenance")
            if provenance is not None and provenance != initial_data.get(task["name"]):
                selection_errors.append(
                    f"Evaluation data/settings changed between candidates: {task['name']}"
                )

        passed_regression = (
            regression.get("accepted") is True
            and regression.get("status") == "accepted"
        )
        value = None
        if passed_regression and not errors:
            task_result = next(
                (
                    task
                    for task in regression["tasks"]
                    if task["name"] == selection_task
                ),
                {},
            )
            value = (
                task_result.get("metrics", {})
                .get(selection_metric, {})
                .get("candidate")
            )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                errors.append(
                    "Selection metric is missing or nonfinite in the accepted regression report"
                )
                value = None

        entries.append(
            {
                "index": index,
                "model": model,
                "model_identity": identities[index + 1],
                "accepted": passed_regression and not errors,
                "status": "invalid" if errors else regression["status"],
                "selection_value": value,
                "errors": errors,
                "regression_report": regression,
            }
        )
    return entries, selection_errors


def _recheck_selection_inputs(
    baseline_model: str | Path,
    suite: Mapping[str, Any] | str | Path,
    config: Mapping[str, Any],
    identities: Sequence[dict[str, Any]],
    initial_data: Mapping[str, Any],
    entries: list[dict[str, Any]],
    selection_errors: Sequence[str],
) -> list[str]:
    """Invalidate stale candidates; shared-input changes invalidate every choice."""
    errors = list(selection_errors)
    try:
        if load_regression_suite(suite) != config:
            errors.append("Regression suite changed during checkpoint selection")
    except (OSError, ValueError, TypeError) as error:
        errors.append(
            f"Regression suite became invalid during checkpoint selection: {error}"
        )
    for task in config["tasks"]:
        try:
            if _provenance(task) != initial_data[task["name"]]:
                errors.append(
                    f"Evaluation data/settings changed during selection: {task['name']}"
                )
        except (OSError, ValueError, TypeError) as error:
            errors.append(
                f"Evaluation data/settings became invalid: {task['name']}: {error}"
            )

    if not model_identities_match(
        identities[0], compute_model_identity(baseline_model)
    ):
        errors.append(
            "Baseline model content changed or is unbound during checkpoint selection"
        )
    for entry, initial_identity in zip(entries, identities[1:]):
        current_identity = compute_model_identity(entry["model"])
        if not model_identities_match(initial_identity, current_identity):
            entry["accepted"] = False
            entry["status"] = "invalid"
            entry["errors"].append(
                "Candidate model content changed or is unbound during checkpoint selection"
            )

    if errors:
        for entry in entries:
            entry["accepted"] = False
            entry["status"] = "invalid"
    return errors


def select_checkpoint(
    baseline_model: str | Path,
    candidates: Sequence[str | Path],
    suite: Mapping[str, Any] | str | Path,
    *,
    selection_task: str,
    selection_metric: str,
    higher_is_better: bool = True,
    output: str | Path | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Gate candidates on the full suite, then rank an explicitly gated metric.

    Candidate order breaks exact ties. No checkpoint directories are scanned.
    The selection task/metric and direction must be declared in the suite, so
    ranking cannot bypass its retention or absolute quality requirements.
    Models, datasets and task settings must retain their content identities
    throughout selection. Standalone local checkpoints are required; merge
    adapters before using this entry point.

    With ``output``, save the final report and each full regression report in
    the sibling ``<output-stem>_candidates`` directory. Without it, all reports
    remain embedded in the returned mapping. No eligible candidate raises
    ``CheckpointSelectionError`` after saving the rejection report, if requested.
    """
    candidate_models, config = _validate_selection_request(
        baseline_model,
        candidates,
        suite,
        selection_task=selection_task,
        selection_metric=selection_metric,
        higher_is_better=higher_is_better,
        evaluator=evaluator,
    )

    destination = None
    report_paths: list[Path | None] = [None] * len(candidate_models)
    if output is not None:
        destination = Path(output).expanduser().resolve()
        report_directory = destination.parent / f"{destination.stem}_candidates"
        report_paths = [
            report_directory / f"candidate-{index:04d}.json"
            for index in range(len(candidate_models))
        ]
    references = [str(baseline_model), *candidate_models]
    _validate_destinations(suite, config, references, [destination, *report_paths])

    # Freeze every input before evaluating any candidate. Otherwise a later
    # evaluator could change the evidence used to accept an earlier one.
    identities = [compute_model_identity(model) for model in references]
    if any(identity.get("adapter") is not None for identity in identities):
        raise ValueError(
            "Adapter checkpoints are not supported for checkpoint selection; merge into standalone models"
        )
    initial_data = {task["name"]: _provenance(task) for task in config["tasks"]}
    entries, selection_errors = _evaluate_candidates(
        baseline_model,
        candidate_models,
        config,
        report_paths,
        identities,
        initial_data,
        selection_task=selection_task,
        selection_metric=selection_metric,
        evaluator=evaluator,
    )
    errors = _recheck_selection_inputs(
        baseline_model,
        suite,
        config,
        identities,
        initial_data,
        entries,
        selection_errors,
    )

    eligible = [entry for entry in entries if entry["accepted"]]
    # Python's stable sort preserves candidate input order for exact ties.
    ranking = sorted(
        eligible,
        key=lambda entry: entry["selection_value"],
        reverse=higher_is_better,
    )
    winner = ranking[0] if ranking else None
    selected_model = None
    selected_report = None
    selected_identity = None
    if winner:
        status = "accepted"
        selected_model = winner["model"]
        selected_report = winner["regression_report"]
        selected_identity = winner["model_identity"]
    elif errors:
        status = "invalid"
    else:
        status = "rejected"

    report = {
        "schema_version": 1,
        "report_type": "checkpoint_selection",
        "accepted": winner is not None,
        "status": status,
        "baseline_model": str(baseline_model),
        "baseline_model_identity": identities[0],
        "suite": config,
        "suite_sha256": _digest(config),
        "selection": {
            "task": selection_task,
            "metric": selection_metric,
            "higher_is_better": higher_is_better,
            "tie_break": "candidate_input_order",
        },
        "selected_model": selected_model,
        "selected_report": selected_report,
        "selected_model_identity": selected_identity,
        "candidate_reports": entries,
        "ranking": [
            {key: entry[key] for key in ("index", "model", "selection_value")}
            for entry in ranking
        ],
        "errors": errors,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "output": None,
    }
    # Recheck destinations immediately before writing the final decision.
    _validate_destinations(suite, config, references, [destination, *report_paths])
    report = _save(report, destination)
    if winner is None:
        raise CheckpointSelectionError(
            "No checkpoint passed the complete regression suite and identity checks",
            report,
        )
    return report


__all__ = ["CheckpointSelectionError", "select_checkpoint"]
