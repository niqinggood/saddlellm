"""Evaluation gates for deciding whether a trained checkpoint may be released."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class ReleaseGateCheck:
    metric: str
    passed: bool
    status: str
    score: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    required: bool = True
    reason: str = ""


@dataclass
class ReleaseGateResult:
    enabled: bool
    accepted: bool
    status: str
    require_all: bool = True
    checks: List[ReleaseGateCheck] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return path


class EvaluationReleaseGate:
    """Apply min/max rules to evaluator metric payloads.

    Rules use a deliberately small schema::

        {
            "perplexity": {"max": 30.0},
            "exact_match": {"min": 0.75, "required": True},
        }

    Evaluator metrics may be raw numbers or dictionaries containing ``score``.
    Dot-separated metric names are supported for nested custom metric payloads.
    """

    @classmethod
    def validate_rules(cls, rules: Mapping[str, Any]) -> List[str]:
        issues: List[str] = []
        if not isinstance(rules, Mapping) or not rules:
            return ["evaluation.gate.rules must contain at least one metric rule."]
        for metric, raw_rule in rules.items():
            if not str(metric).strip():
                issues.append("evaluation.gate.rules contains an empty metric name.")
                continue
            if not isinstance(raw_rule, Mapping):
                issues.append(f"Gate rule for {metric!r} must be a mapping with min and/or max.")
                continue
            minimum = raw_rule.get("min")
            maximum = raw_rule.get("max")
            if minimum is None and maximum is None:
                issues.append(f"Gate rule for {metric!r} requires min and/or max.")
                continue
            for label, value in (("min", minimum), ("max", maximum)):
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    issues.append(f"Gate rule {metric!r}.{label} must be numeric.")
                    continue
                if not math.isfinite(numeric):
                    issues.append(f"Gate rule {metric!r}.{label} must be finite.")
            if minimum is not None and maximum is not None:
                try:
                    if float(minimum) > float(maximum):
                        issues.append(f"Gate rule {metric!r} has min greater than max.")
                except (TypeError, ValueError):
                    pass
        return issues

    @classmethod
    def evaluate(
        cls,
        metrics: Mapping[str, Any],
        rules: Mapping[str, Any],
        *,
        enabled: bool = True,
        require_all: bool = True,
    ) -> ReleaseGateResult:
        if not enabled:
            return ReleaseGateResult(
                enabled=False,
                accepted=True,
                status="disabled",
                require_all=require_all,
            )

        issues = cls.validate_rules(rules)
        if issues:
            return ReleaseGateResult(
                enabled=True,
                accepted=False,
                status="invalid",
                require_all=require_all,
                failures=issues,
            )

        checks: List[ReleaseGateCheck] = []
        for metric, raw_rule in rules.items():
            required = bool(raw_rule.get("required", True))
            minimum = cls._optional_float(raw_rule.get("min"))
            maximum = cls._optional_float(raw_rule.get("max"))
            raw_score = cls._find_metric(metrics, str(metric))
            score = cls._score(raw_score)
            if score is None:
                passed = not required
                checks.append(
                    ReleaseGateCheck(
                        metric=str(metric),
                        passed=passed,
                        status="missing" if required else "skipped",
                        minimum=minimum,
                        maximum=maximum,
                        required=required,
                        reason=(
                            "required metric is missing or non-numeric"
                            if required
                            else "optional metric is missing"
                        ),
                    )
                )
                continue

            failures = []
            if minimum is not None and score < minimum:
                failures.append(f"{score} < min {minimum}")
            if maximum is not None and score > maximum:
                failures.append(f"{score} > max {maximum}")
            checks.append(
                ReleaseGateCheck(
                    metric=str(metric),
                    score=score,
                    minimum=minimum,
                    maximum=maximum,
                    required=required,
                    passed=not failures,
                    status="passed" if not failures else "failed",
                    reason="; ".join(failures),
                )
            )

        considered = [check for check in checks if check.status != "skipped"]
        required_metric_missing = any(
            check.required and check.status == "missing" for check in considered
        )
        if required_metric_missing:
            accepted = False
        elif not considered:
            accepted = False
        elif require_all:
            accepted = all(check.passed for check in considered)
        else:
            accepted = any(check.passed for check in considered)
        failures = [
            f"{check.metric}: {check.reason or check.status}"
            for check in considered
            if not check.passed
        ]
        return ReleaseGateResult(
            enabled=True,
            accepted=accepted,
            status="accepted" if accepted else "rejected",
            require_all=require_all,
            checks=checks,
            failures=failures,
        )

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        return None if value is None else float(value)

    @staticmethod
    def _find_metric(metrics: Mapping[str, Any], path: str) -> Any:
        if path in metrics:
            return metrics[path]
        current: Any = metrics
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _score(value: Any) -> Optional[float]:
        if isinstance(value, Mapping):
            value = value.get("score")
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None


def evaluate_release_gate(
    metrics: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    enabled: bool = True,
    require_all: bool = True,
) -> Dict[str, Any]:
    return EvaluationReleaseGate.evaluate(
        metrics,
        rules,
        enabled=enabled,
        require_all=require_all,
    ).to_dict()
