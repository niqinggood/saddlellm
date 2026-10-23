"""Paired validation/test comparisons and release decisions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .ModelIdentity import identity_is_bound, model_identities_match


def load_evaluation(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "details" not in value:
        raise ValueError(f"not a SaddleLLM evaluation result: {path}")
    return value


def exact_two_sided_binomial_p(successes: int, trials: int) -> float:
    """Exact two-sided binomial p-value used by paired McNemar tests."""

    if not trials:
        return 1.0
    observed = math.comb(trials, successes) / (2**trials)
    return min(
        1.0,
        sum(
            math.comb(trials, value) / (2**trials)
            for value in range(trials + 1)
            if math.comb(trials, value) / (2**trials) <= observed + 1e-15
        ),
    )


def _identity(row: Mapping[str, Any]):
    if "rubric_sha256" in row:
        return (
            row.get("record_id"),
            row.get("instruction"),
            row.get("rubric_sha256"),
        )
    return row.get("record_id"), row.get("question"), row.get("answer")


def paired_metrics(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    section: str = "details",
    outcome_field: str = "correct",
) -> dict[str, Any]:
    base_rows: Sequence[Mapping[str, Any]] = base.get(section) or []
    candidate_rows: Sequence[Mapping[str, Any]] = candidate.get(section) or []
    if not base_rows or not candidate_rows:
        raise ValueError(f"both results require non-empty {section!r}")
    if len(base_rows) != len(candidate_rows):
        raise ValueError(f"base and candidate {section} counts differ")
    for index, (left, right) in enumerate(zip(base_rows, candidate_rows)):
        if _identity(left) != _identity(right):
            raise ValueError(f"base and candidate {section} row {index} are not aligned")
    if any(outcome_field not in row for row in (*base_rows, *candidate_rows)):
        raise ValueError(f"evaluation rows are missing outcome field {outcome_field!r}")
    base_correct = sum(bool(row[outcome_field]) for row in base_rows)
    candidate_correct = sum(bool(row[outcome_field]) for row in candidate_rows)
    wrong_right = sum(
        (not bool(left[outcome_field])) and bool(right[outcome_field])
        for left, right in zip(base_rows, candidate_rows)
    )
    right_wrong = sum(
        bool(left[outcome_field]) and (not bool(right[outcome_field]))
        for left, right in zip(base_rows, candidate_rows)
    )
    samples = len(base_rows)
    base_accuracy = base_correct / samples
    candidate_accuracy = candidate_correct / samples
    return {
        "section": section,
        "outcome_field": outcome_field,
        "samples": samples,
        "base_correct": base_correct,
        "candidate_correct": candidate_correct,
        "base_accuracy": base_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "absolute_gain_percentage_points": (candidate_accuracy - base_accuracy) * 100,
        "relative_gain_percent": (
            (candidate_accuracy / base_accuracy - 1) * 100 if base_accuracy else None
        ),
        "base_wrong_candidate_right": wrong_right,
        "base_right_candidate_wrong": right_wrong,
        "net_gain": wrong_right - right_wrong,
        "mcnemar_exact_two_sided_p": exact_two_sided_binomial_p(
            min(wrong_right, right_wrong), wrong_right + right_wrong
        ),
    }


def completion_metrics(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare aligned completion losses where lower perplexity is better."""

    base_rows: Sequence[Mapping[str, Any]] = base.get("details") or []
    candidate_rows: Sequence[Mapping[str, Any]] = candidate.get("details") or []
    if not base_rows or not candidate_rows:
        raise ValueError("both completion results require non-empty 'details'")
    if len(base_rows) != len(candidate_rows):
        raise ValueError("base and candidate completion detail counts differ")

    improvements = 0
    regressions = 0
    ties = 0
    compared_rows = 0
    for index, (left, right) in enumerate(zip(base_rows, candidate_rows)):
        left_identity = (left.get("record_id"), left.get("source_index"))
        right_identity = (right.get("record_id"), right.get("source_index"))
        if left_identity != right_identity:
            raise ValueError(
                f"base and candidate completion detail row {index} are not aligned"
            )
        left_tokens = int(left.get("completion_tokens") or 0)
        right_tokens = int(right.get("completion_tokens") or 0)
        if left_tokens != right_tokens:
            raise ValueError(
                f"base and candidate completion token counts differ at row {index}"
            )
        left_nll = left.get("completion_nll")
        right_nll = right.get("completion_nll")
        if left_nll is None or right_nll is None:
            if left_nll != right_nll:
                raise ValueError(
                    f"base and candidate completion NLL availability differs at row {index}"
                )
            continue
        left_nll = float(left_nll)
        right_nll = float(right_nll)
        if not math.isfinite(left_nll) or not math.isfinite(right_nll):
            raise ValueError(f"completion NLL at row {index} must be finite")
        compared_rows += 1
        if math.isclose(left_nll, right_nll, rel_tol=0.0, abs_tol=1e-12):
            ties += 1
        elif right_nll < left_nll:
            improvements += 1
        else:
            regressions += 1

    if not compared_rows:
        raise ValueError("completion results contain no comparable per-record NLL values")
    base_tokens = int(base.get("completion_tokens") or 0)
    candidate_tokens = int(candidate.get("completion_tokens") or 0)
    if base_tokens <= 0 or candidate_tokens <= 0:
        raise ValueError("completion results require positive completion token counts")
    if base_tokens != candidate_tokens:
        raise ValueError("base and candidate aggregate completion token counts differ")
    base_nll = float(base["completion_nll"])
    candidate_nll = float(candidate["completion_nll"])
    base_ppl = float(base["completion_perplexity"])
    candidate_ppl = float(candidate["completion_perplexity"])
    if not all(
        math.isfinite(value) for value in (base_nll, candidate_nll, base_ppl, candidate_ppl)
    ):
        raise ValueError("completion aggregate metrics must be finite")
    if base_ppl <= 0 or candidate_ppl <= 0:
        raise ValueError("completion perplexity must be positive")
    absolute_improvement = base_ppl - candidate_ppl
    discordant = improvements + regressions
    return {
        "samples": len(base_rows),
        "compared_rows": compared_rows,
        "completion_tokens": base_tokens,
        "base_completion_nll": base_nll,
        "candidate_completion_nll": candidate_nll,
        "nll_improvement": base_nll - candidate_nll,
        "base_completion_perplexity": base_ppl,
        "candidate_completion_perplexity": candidate_ppl,
        "absolute_perplexity_improvement": absolute_improvement,
        "relative_perplexity_improvement_percent": absolute_improvement
        / base_ppl
        * 100,
        "records_improved": improvements,
        "records_regressed": regressions,
        "records_tied": ties,
        "paired_exact_two_sided_sign_p": exact_two_sided_binomial_p(
            min(improvements, regressions), discordant
        ),
        "higher_is_better": False,
    }


def _nested(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _set_comparison(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    base_task = base.get("task_type")
    candidate_task = candidate.get("task_type")
    if base_task != candidate_task:
        raise ValueError("base and candidate evaluation task types differ")
    if (
        base.get("data_sha256")
        and candidate.get("data_sha256")
        and base["data_sha256"] != candidate["data_sha256"]
    ):
        raise ValueError("base and candidate evaluation data hashes differ")
    if base_task == "completion":
        return {"completion_perplexity": completion_metrics(base, candidate)}
    if base_task == "rubric":
        metrics = paired_metrics(base, candidate, outcome_field="passed")
        base_rows = base.get("details") or []
        candidate_rows = candidate.get("details") or []
        base_mean_score = sum(float(row["score"]) for row in base_rows) / len(base_rows)
        candidate_mean_score = sum(float(row["score"]) for row in candidate_rows) / len(
            candidate_rows
        )
        base_critical = sum(bool(row.get("critical_failures")) for row in base_rows)
        candidate_critical = sum(
            bool(row.get("critical_failures")) for row in candidate_rows
        )
        return {
            "rubric_pass": metrics,
            "rubric_safety": {
                "base_mean_score": base_mean_score,
                "candidate_mean_score": candidate_mean_score,
                "mean_score_gain_percentage_points": (
                    candidate_mean_score - base_mean_score
                )
                * 100,
                "base_critical_failure_count": base_critical,
                "candidate_critical_failure_count": candidate_critical,
                "critical_failure_change": candidate_critical - base_critical,
            },
        }

    result = {"label_probability": paired_metrics(base, candidate)}
    base_generations = base.get("generations") or []
    candidate_generations = candidate.get("generations") or []
    if base_generations or candidate_generations:
        result["generation"] = paired_metrics(
            base, candidate, section="generations"
        )
        base_details = {
            row.get("source_index"): row for row in base["details"]
        }
        candidate_details = {
            row.get("source_index"): row for row in candidate["details"]
        }
        result["base_probability_generation_disagreements"] = sum(
            row["parsed_prediction"]
            != base_details[row.get("source_index")]["prediction"]
            for row in base_generations
        )
        result["candidate_probability_generation_disagreements"] = sum(
            row["parsed_prediction"]
            != candidate_details[row.get("source_index")]["prediction"]
            for row in candidate_generations
        )
    base_robustness = base.get("prompt_robustness")
    candidate_robustness = candidate.get("prompt_robustness")
    if base_robustness or candidate_robustness:
        if not base_robustness or not candidate_robustness:
            raise ValueError(
                "base and candidate must both contain prompt_robustness"
            )
        base_templates = base.get("prompt_variant_templates") or {}
        candidate_templates = candidate.get("prompt_variant_templates") or {}
        if base_templates.get("sha256") != candidate_templates.get("sha256"):
            raise ValueError("base and candidate prompt variant hashes differ")
        base_cases = base_robustness.get("case_details") or []
        candidate_cases = candidate_robustness.get("case_details") or []
        if [row.get("case_id") for row in base_cases] != [
            row.get("case_id") for row in candidate_cases
        ]:
            raise ValueError("base and candidate robustness cases are not aligned")
        wrong_right = sum(
            not bool(left["all_variants_correct"])
            and bool(right["all_variants_correct"])
            for left, right in zip(base_cases, candidate_cases)
        )
        right_wrong = sum(
            bool(left["all_variants_correct"])
            and not bool(right["all_variants_correct"])
            for left, right in zip(base_cases, candidate_cases)
        )
        variants = sorted(
            set(base_robustness.get("by_variant", {}))
            | set(candidate_robustness.get("by_variant", {}))
        )
        if any(
            name not in base_robustness.get("by_variant", {})
            or name not in candidate_robustness.get("by_variant", {})
            for name in variants
        ):
            raise ValueError("base and candidate prompt variant names differ")
        result["prompt_robustness"] = {
            "cases": len(base_cases),
            "baseline": {
                "prediction_flip_rate": base_robustness["prediction_flip_rate"],
                "all_variants_correct_rate": base_robustness[
                    "all_variants_correct_rate"
                ],
                "worst_variant_accuracy": base_robustness[
                    "worst_variant_accuracy"
                ],
            },
            "candidate": {
                "prediction_flip_rate": candidate_robustness[
                    "prediction_flip_rate"
                ],
                "all_variants_correct_rate": candidate_robustness[
                    "all_variants_correct_rate"
                ],
                "worst_variant_accuracy": candidate_robustness[
                    "worst_variant_accuracy"
                ],
            },
            "all_variants_correct_gain_percentage_points": (
                candidate_robustness["all_variants_correct_rate"]
                - base_robustness["all_variants_correct_rate"]
            )
            * 100,
            "prediction_flip_reduction_percentage_points": (
                base_robustness["prediction_flip_rate"]
                - candidate_robustness["prediction_flip_rate"]
            )
            * 100
            if base_robustness["prediction_flip_rate"] is not None
            and candidate_robustness["prediction_flip_rate"] is not None
            else None,
            "worst_variant_accuracy_gain_percentage_points": (
                candidate_robustness["worst_variant_accuracy"]
                - base_robustness["worst_variant_accuracy"]
            )
            * 100,
            "all_variants_base_wrong_candidate_right": wrong_right,
            "all_variants_base_right_candidate_wrong": right_wrong,
            "all_variants_mcnemar_exact_two_sided_p": exact_two_sided_binomial_p(
                min(wrong_right, right_wrong), wrong_right + right_wrong
            ),
            "by_variant": {
                name: {
                    "baseline_accuracy": base_robustness["by_variant"][name][
                        "accuracy"
                    ],
                    "candidate_accuracy": candidate_robustness["by_variant"][name][
                        "accuracy"
                    ],
                    "gain_percentage_points": (
                        candidate_robustness["by_variant"][name]["accuracy"]
                        - base_robustness["by_variant"][name]["accuracy"]
                    )
                    * 100,
                }
                for name in variants
            },
        }
    return result


def build_evaluation_comparison(
    *,
    base_validation: Mapping[str, Any],
    candidate_validation: Mapping[str, Any],
    base_test: Mapping[str, Any] | None = None,
    candidate_test: Mapping[str, Any] | None = None,
    minimum_gain_percentage_points: float = 0.0,
    maximum_p_exclusive: float = 0.05,
    maximum_candidate_critical_failures: int = 0,
    manifest: Mapping[str, Any] | None = None,
    overlap_key: str = "exact_question_overlap_after_exclusion",
    experiment: str = "model-comparison",
    scope: str = "multiple-choice evaluation",
) -> dict[str, Any]:
    if bool(base_test) != bool(candidate_test):
        raise ValueError("base_test and candidate_test must be provided together")
    def consistent_identity(validation, test, label):
        identity = validation.get("model_identity")
        if test is None:
            return identity
        test_identity = test.get("model_identity")
        if identity_is_bound(identity) != identity_is_bound(test_identity):
            raise ValueError(
                f"{label} validation/test model identity is only bound in one report; "
                "re-run evaluate for both sets on the same model contents"
            )
        if identity_is_bound(identity) and not model_identities_match(identity, test_identity):
            raise ValueError(
                f"{label} validation/test model identities differ; "
                "a shared model path cannot substitute for matching contents"
            )
        return identity

    base_identity = consistent_identity(base_validation, base_test, "base")
    candidate_identity = consistent_identity(candidate_validation, candidate_test, "candidate")
    sets = {
        "validation": _set_comparison(base_validation, candidate_validation),
    }
    if base_test is not None and candidate_test is not None:
        sets["test"] = _set_comparison(base_test, candidate_test)

    def accuracy_passed(metrics: Mapping[str, Any]) -> bool:
        return (
            metrics["absolute_gain_percentage_points"]
            >= minimum_gain_percentage_points
            and metrics["mcnemar_exact_two_sided_p"] < maximum_p_exclusive
        )

    def completion_passed(metrics: Mapping[str, Any]) -> bool:
        return (
            metrics["relative_perplexity_improvement_percent"]
            >= minimum_gain_percentage_points
            and metrics["paired_exact_two_sided_sign_p"] < maximum_p_exclusive
        )

    if maximum_candidate_critical_failures < 0:
        raise ValueError("maximum_candidate_critical_failures cannot be negative")
    set_gate = {}
    for name, values in sets.items():
        if "completion_perplexity" in values:
            accepted = completion_passed(values["completion_perplexity"])
        else:
            metric_name = (
                "rubric_pass" if "rubric_pass" in values else "label_probability"
            )
            accepted = accuracy_passed(values[metric_name])
        if "rubric_safety" in values:
            accepted = accepted and (
                values["rubric_safety"]["candidate_critical_failure_count"]
                <= maximum_candidate_critical_failures
            )
        set_gate[name] = accepted
    leakage = {"status": "not_checked", "passed": True}
    if manifest is not None:
        overlaps = _nested(manifest, overlap_key)
        if not isinstance(overlaps, Mapping):
            raise ValueError(f"manifest overlap key {overlap_key!r} must be a mapping")
        leakage = {
            "status": "checked",
            "passed": not any(bool(value) for value in overlaps.values()),
            "overlap_key": overlap_key,
            "exact_overlap": dict(overlaps),
        }
    validation_passed = leakage["passed"] and set_gate["validation"]
    sealed_test_provided = "test" in sets
    promoted = (
        validation_passed
        and sealed_test_provided
        and set_gate["test"]
    )
    decision = (
        "promoted"
        if promoted
        else "validation_passed"
        if validation_passed and not sealed_test_provided
        else "not_promoted"
    )
    limitations = [
        "Statistical significance does not establish domain safety.",
        "Test results must not be used for further checkpoint selection.",
    ]
    identity_bound = identity_is_bound(base_identity) and identity_is_bound(candidate_identity)
    if not identity_bound:
        limitations.append(
            "Some model evidence is not content-bound. Legacy paths or remote identifiers "
            "are not model fingerprints; strict export requires local evaluate/compare evidence."
        )
    if any("rubric_pass" in values for values in sets.values()):
        limitations.append(
            "Deterministic rubric checks are an engineering screen and require independent expert review."
        )
    if any("completion_perplexity" in values for values in sets.values()):
        limitations.extend(
            [
                "Completion PPL is comparable only with the same tokenizer, template, and data.",
                "Lower completion PPL does not establish factual correctness or generation safety.",
                "The paired sign test uses per-record mean-NLL direction while aggregate PPL is token-weighted.",
            ]
        )
    if not sealed_test_provided:
        limitations.insert(
            0,
            "Validation-only evidence may select a checkpoint but cannot approve a release; "
            "paired sealed-test results are required for promotion.",
        )
    return {
        "schema_version": 2,
        "experiment": experiment,
        "accepted": promoted,
        "decision": decision,
        "selection_eligible": validation_passed,
        "release_evidence_complete": sealed_test_provided,
        "scope": scope,
        "base_model": base_validation.get("model"),
        "candidate_model": candidate_validation.get("model"),
        "candidate_adapter": candidate_validation.get("adapter"),
        "base_model_identity": base_identity,
        "candidate_model_identity": candidate_identity,
        "model_identity_binding": {
            "status": "bound" if identity_bound else "unbound",
            "validation_test_consistency": "checked" if sealed_test_provided else "validation_only",
        },
        "criteria": {
            "minimum_absolute_gain_percentage_points": minimum_gain_percentage_points,
            "minimum_relative_perplexity_improvement_percent": minimum_gain_percentage_points,
            "maximum_mcnemar_exact_two_sided_p_exclusive": maximum_p_exclusive,
            "maximum_paired_sign_exact_two_sided_p_exclusive": maximum_p_exclusive,
            "maximum_candidate_critical_failures": maximum_candidate_critical_failures,
            "set_gate": set_gate,
        },
        "data_leakage_check": leakage,
        "sets": sets,
        "limitations": limitations,
    }


def comparison_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# SaddleLLM Paired Evaluation",
        "",
        f"- Experiment: `{result['experiment']}`",
        f"- Decision: **{result['decision']}**",
        f"- Selection eligible: **{str(bool(result.get('selection_eligible'))).lower()}**",
        f"- Release accepted: **{str(bool(result.get('accepted'))).lower()}**",
        f"- Sealed-test evidence complete: "
        f"**{str(bool(result.get('release_evidence_complete'))).lower()}**",
        f"- Scope: {result['scope']}",
        f"- Model content binding: {result.get('model_identity_binding', {}).get('status', 'unbound')}",
    ]
    accuracy_rows = []
    for set_name, values in result["sets"].items():
        for method, metrics in values.items():
            if not isinstance(metrics, Mapping) or "base_accuracy" not in metrics:
                continue
            accuracy_rows.append(
                f"| {set_name} | {method} | {metrics['base_accuracy'] * 100:.2f}% | "
                f"{metrics['candidate_accuracy'] * 100:.2f}% | "
                f"{metrics['absolute_gain_percentage_points']:.2f} | "
                f"{metrics['mcnemar_exact_two_sided_p']:.6f} |"
            )
    if accuracy_rows:
        lines.extend(
            [
                "",
                "| Set | Method | Base | Candidate | Gain (pp) | McNemar p |",
                "|---|---|---:|---:|---:|---:|",
                *accuracy_rows,
            ]
        )
    completion_sections = [
        (set_name, values["completion_perplexity"])
        for set_name, values in result["sets"].items()
        if values.get("completion_perplexity")
    ]
    if completion_sections:
        lines.extend(
            [
                "",
                "## Completion perplexity",
                "",
                "| Set | Base PPL | Candidate PPL | Relative improvement | Improved / regressed / tied | Sign-test p |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for set_name, metrics in completion_sections:
            lines.append(
                f"| {set_name} | {metrics['base_completion_perplexity']:.6f} | "
                f"{metrics['candidate_completion_perplexity']:.6f} | "
                f"{metrics['relative_perplexity_improvement_percent']:.4f}% | "
                f"{metrics['records_improved']} / {metrics['records_regressed']} / "
                f"{metrics['records_tied']} | "
                f"{metrics['paired_exact_two_sided_sign_p']:.6f} |"
            )
    robustness_sections = [
        (set_name, values["prompt_robustness"])
        for set_name, values in result["sets"].items()
        if values.get("prompt_robustness")
    ]
    for set_name, robustness in robustness_sections:
        flip_reduction = robustness["prediction_flip_reduction_percentage_points"]
        lines.extend(
            [
                "",
                f"## {set_name.title()} prompt robustness",
                "",
                "| Metric | Base | Candidate | Improvement |",
                "|---|---:|---:|---:|",
                f"| All variants correct | "
                f"{robustness['baseline']['all_variants_correct_rate'] * 100:.2f}% | "
                f"{robustness['candidate']['all_variants_correct_rate'] * 100:.2f}% | "
                f"{robustness['all_variants_correct_gain_percentage_points']:.2f} pp |",
                f"| Worst variant accuracy | "
                f"{robustness['baseline']['worst_variant_accuracy'] * 100:.2f}% | "
                f"{robustness['candidate']['worst_variant_accuracy'] * 100:.2f}% | "
                f"{robustness['worst_variant_accuracy_gain_percentage_points']:.2f} pp |",
                f"| Prediction flip rate (lower is better) | "
                f"{robustness['baseline']['prediction_flip_rate'] * 100:.2f}% | "
                f"{robustness['candidate']['prediction_flip_rate'] * 100:.2f}% | "
                f"{flip_reduction:.2f} pp reduction |"
                if flip_reduction is not None
                else "| Prediction flip rate | unavailable | unavailable | unavailable |",
                "",
                f"All-variants-correct paired McNemar p: "
                f"{robustness['all_variants_mcnemar_exact_two_sided_p']:.6f}",
            ]
        )
    rubric_sections = [
        (set_name, values["rubric_safety"])
        for set_name, values in result["sets"].items()
        if values.get("rubric_safety")
    ]
    for set_name, safety in rubric_sections:
        lines.extend(
            [
                "",
                f"## {set_name.title()} rubric safety screen",
                "",
                "| Metric | Base | Candidate | Change |",
                "|---|---:|---:|---:|",
                f"| Mean weighted score | {safety['base_mean_score'] * 100:.2f}% | "
                f"{safety['candidate_mean_score'] * 100:.2f}% | "
                f"{safety['mean_score_gain_percentage_points']:.2f} pp |",
                f"| Samples with critical failures (lower is better) | "
                f"{safety['base_critical_failure_count']} | "
                f"{safety['candidate_critical_failure_count']} | "
                f"{safety['critical_failure_change']:+d} |",
            ]
        )
    lines.extend(
        [
            "",
            "> Validation-only evidence can select a checkpoint but cannot approve a release. "
            "Promotion additionally requires paired sealed-test evidence; statistical promotion "
            "still does not establish domain or clinical safety.",
            "",
        ]
    )
    return "\n".join(lines)


def save_comparison(
    result: Mapping[str, Any],
    output: str,
    markdown_output: str | None = None,
) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if markdown_output:
        report = Path(markdown_output)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(comparison_markdown(result), encoding="utf-8")


__all__ = [
    "build_evaluation_comparison",
    "comparison_markdown",
    "completion_metrics",
    "exact_two_sided_binomial_p",
    "load_evaluation",
    "paired_metrics",
    "save_comparison",
]
