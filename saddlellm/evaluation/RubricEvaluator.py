"""Deterministic, domain-owned rubric checks for generated model responses.

The framework implements a small generic rule schema.  Medical, legal, or
other domain rules stay in the evaluation data owned and reviewed by that
domain; SaddleLLM does not embed clinical policy in its core package.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .CompletionEvaluator import _generate
from .MultipleChoiceEvaluator import (
    _load_model,
    _render_prompt,
    load_evaluation_records,
)


SUPPORTED_CHECK_TYPES = {
    "contains_any",
    "contains_all",
    "not_contains_any",
    "regex",
    "not_regex",
}


@dataclass
class RubricEvaluationRequest:
    model: str
    data: str
    adapter: str | None = None
    prompt_field: str = "instruction"
    rubric_field: str = "rubric"
    reference_field: str = "reference"
    id_field: str = "id"
    pass_threshold: float = 0.8
    limit: int = 0
    max_length: int = 512
    max_new_tokens: int = 192
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    device: str = "auto"
    dtype: str = "auto"
    enable_thinking: bool = False
    trust_remote_code: bool = False
    local_files_only: bool = False
    endpoint: str | None = None
    api_key_env: str | None = None
    request_timeout: float = 120.0
    split: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _rubric_for_record(
    row: Mapping[str, Any],
    *,
    index: int,
    rubric_field: str,
    default_threshold: float,
) -> tuple[list[dict[str, Any]], float, str]:
    rubric = row.get(rubric_field)
    if isinstance(rubric, Sequence) and not isinstance(rubric, (str, bytes)):
        checks = rubric
        threshold = default_threshold
    elif isinstance(rubric, Mapping):
        checks = rubric.get("checks")
        threshold = rubric.get("pass_threshold", default_threshold)
    else:
        raise ValueError(
            f"record {index} field {rubric_field!r} must be a rubric object or check list"
        )
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)) or not checks:
        raise ValueError(f"record {index} rubric requires a non-empty checks list")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {index} rubric pass_threshold must be numeric") from exc
    if not 0 <= threshold <= 1:
        raise ValueError(f"record {index} rubric pass_threshold must be in [0, 1]")

    normalized_checks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for check_index, raw in enumerate(checks):
        if not isinstance(raw, Mapping):
            raise ValueError(f"record {index} rubric check {check_index} must be an object")
        check_id = str(raw.get("id", f"check-{check_index + 1}")).strip()
        if not check_id or check_id in seen_ids:
            raise ValueError(f"record {index} has an empty or duplicate rubric check id")
        seen_ids.add(check_id)
        check_type = str(raw.get("type", "")).strip()
        if check_type not in SUPPORTED_CHECK_TYPES:
            raise ValueError(
                f"record {index} rubric check {check_id!r} has unsupported type "
                f"{check_type!r}; expected one of {sorted(SUPPORTED_CHECK_TYPES)}"
            )
        raw_patterns = raw.get("patterns")
        if isinstance(raw_patterns, str):
            raw_patterns = [raw_patterns]
        if (
            not isinstance(raw_patterns, Sequence)
            or isinstance(raw_patterns, (str, bytes))
            or not raw_patterns
        ):
            raise ValueError(
                f"record {index} rubric check {check_id!r} requires patterns"
            )
        patterns = [str(pattern) for pattern in raw_patterns if str(pattern)]
        if not patterns:
            raise ValueError(
                f"record {index} rubric check {check_id!r} requires non-empty patterns"
            )
        if check_type in {"regex", "not_regex"}:
            for pattern in patterns:
                try:
                    re.compile(pattern, re.IGNORECASE)
                except re.error as exc:
                    raise ValueError(
                        f"record {index} rubric check {check_id!r} has invalid regex: {exc}"
                    ) from exc
        try:
            weight = float(raw.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"record {index} rubric check {check_id!r} weight must be numeric"
            ) from exc
        if weight <= 0:
            raise ValueError(
                f"record {index} rubric check {check_id!r} weight must be positive"
            )
        normalized_checks.append(
            {
                "id": check_id,
                "type": check_type,
                "patterns": patterns,
                "weight": weight,
                "critical": bool(raw.get("critical", False)),
                "dimension": str(raw.get("dimension", "general")).strip() or "general",
                "description": str(raw.get("description", "")).strip(),
            }
        )
    canonical = {"pass_threshold": threshold, "checks": normalized_checks}
    rubric_sha256 = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return normalized_checks, threshold, rubric_sha256


def _evaluate_check(output: str, check: Mapping[str, Any]) -> tuple[bool, list[str]]:
    check_type = str(check["type"])
    patterns = list(check["patterns"])
    normalized_output = _normalized_text(output)
    if check_type in {"contains_any", "contains_all", "not_contains_any"}:
        matches = [pattern for pattern in patterns if _normalized_text(pattern) in normalized_output]
        if check_type == "contains_any":
            return bool(matches), matches
        if check_type == "contains_all":
            return len(matches) == len(patterns), matches
        return not matches, matches
    matches = [
        pattern
        for pattern in patterns
        if re.search(pattern, unicodedata.normalize("NFKC", output), re.IGNORECASE)
    ]
    if check_type == "regex":
        return bool(matches), matches
    return not matches, matches


def _score_output(
    output: str, checks: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    total_weight = sum(float(check["weight"]) for check in checks)
    passed_weight = 0.0
    check_results = []
    critical_failures = []
    for check in checks:
        passed, matches = _evaluate_check(output, check)
        if passed:
            passed_weight += float(check["weight"])
        elif check["critical"]:
            critical_failures.append(str(check["id"]))
        check_results.append(
            {
                "id": check["id"],
                "type": check["type"],
                "dimension": check["dimension"],
                "description": check["description"],
                "weight": check["weight"],
                "critical": check["critical"],
                "passed": passed,
                "matched_patterns": matches,
            }
        )
    score = passed_weight / total_weight
    empty_output = not output.strip()
    if empty_output:
        critical_failures.insert(0, "__empty_output__")
    return {
        "score": score,
        "passed": not empty_output and score >= threshold and not critical_failures,
        "empty_output": empty_output,
        "critical_failures": critical_failures,
        "checks": check_results,
    }


def _chat_completions_url(endpoint: str) -> str:
    value = endpoint.strip().rstrip("/")
    if not value:
        raise ValueError("endpoint cannot be empty")
    if value.endswith("/v1/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def _endpoint_generate(prompt: str, request: RubricEvaluationRequest) -> str:
    headers = {"Content-Type": "application/json"}
    if request.api_key_env:
        api_key = os.getenv(request.api_key_env)
        if not api_key:
            raise ValueError(
                f"environment variable {request.api_key_env!r} is not set"
            )
        headers["Authorization"] = f"Bearer {api_key}"
    payload = json.dumps(
        {
            "model": request.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": request.max_new_tokens,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    http_request = urllib.request.Request(
        _chat_completions_url(str(request.endpoint)),
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=request.request_timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"evaluation endpoint returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach evaluation endpoint: {exc.reason}") from exc
    try:
        return str(result["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("evaluation endpoint returned an invalid chat completion") from exc


def evaluate_rubric(request: RubricEvaluationRequest) -> dict[str, Any]:
    """Generate every response deterministically and score declarative checks."""

    if not 0 <= request.pass_threshold <= 1:
        raise ValueError("pass_threshold must be in [0, 1]")
    if request.max_length < 1 or request.max_new_tokens < 1:
        raise ValueError("max_length and max_new_tokens must be positive")
    if request.repetition_penalty <= 0 or request.no_repeat_ngram_size < 0:
        raise ValueError("invalid repetition control values")
    if request.request_timeout <= 0:
        raise ValueError("request_timeout must be positive")
    source = Path(request.data).expanduser().resolve()
    rows = load_evaluation_records(str(source), request.limit, split=request.split)
    if not rows:
        raise ValueError("evaluation dataset is empty")

    prepared = []
    for index, row in enumerate(rows):
        prompt = str(row.get(request.prompt_field, "")).strip()
        if not prompt:
            raise ValueError(f"record {index} is missing {request.prompt_field!r}")
        checks, threshold, rubric_sha256 = _rubric_for_record(
            row,
            index=index,
            rubric_field=request.rubric_field,
            default_threshold=request.pass_threshold,
        )
        prepared.append((row, prompt, checks, threshold, rubric_sha256))

    model = tokenizer = None
    device = "endpoint"
    if request.endpoint:
        if request.adapter:
            raise ValueError("adapter cannot be used with endpoint evaluation")
        if request.api_key_env and not os.getenv(request.api_key_env):
            raise ValueError(
                f"environment variable {request.api_key_env!r} is not set"
            )
        if request.repetition_penalty != 1.0 or request.no_repeat_ngram_size != 0:
            raise ValueError(
                "repetition controls are only supported for local rubric evaluation"
            )
    else:
        model, tokenizer, device = _load_model(request)

    torch = None
    if device == "cuda":
        import torch as torch_module

        torch = torch_module
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    details = []
    dimensions: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"checks": 0, "passed_checks": 0, "weight": 0.0, "passed_weight": 0.0}
    )
    for index, (row, prompt_text, checks, threshold, rubric_sha256) in enumerate(prepared):
        if request.endpoint:
            generated = _endpoint_generate(prompt_text, request)
        else:
            rendered = _render_prompt(
                tokenizer, prompt_text, enable_thinking=request.enable_thinking
            )
            generated = _generate(model, tokenizer, device, rendered, request)
        scored = _score_output(generated, checks, threshold)
        for check in scored["checks"]:
            dimension = dimensions[str(check["dimension"])]
            dimension["checks"] += 1
            dimension["passed_checks"] += int(check["passed"])
            dimension["weight"] += float(check["weight"])
            dimension["passed_weight"] += float(check["weight"]) * int(check["passed"])
        details.append(
            {
                "record_id": str(row.get(request.id_field, index)),
                "source_index": index,
                "instruction": prompt_text,
                "reference": str(row.get(request.reference_field, "")),
                "generated": generated,
                "rubric_sha256": rubric_sha256,
                "pass_threshold": threshold,
                **scored,
            }
        )

    samples = len(details)
    passed_samples = sum(bool(item["passed"]) for item in details)
    critical_failure_count = sum(bool(item["critical_failures"]) for item in details)
    by_dimension = {
        name: {
            **values,
            "check_pass_rate": values["passed_checks"] / values["checks"],
            "weighted_pass_rate": values["passed_weight"] / values["weight"],
        }
        for name, values in sorted(dimensions.items())
    }
    return {
        "schema_version": 1,
        "task_type": "rubric",
        "model": request.model,
        "adapter": request.adapter,
        "endpoint": request.endpoint,
        "data": str(source),
        "data_sha256": _sha256(source),
        "samples": samples,
        "passed_samples": passed_samples,
        "pass_rate": passed_samples / samples,
        "mean_score": sum(float(item["score"]) for item in details) / samples,
        "critical_failure_count": critical_failure_count,
        "critical_failure_rate": critical_failure_count / samples,
        "by_dimension": by_dimension,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024**3) if torch is not None else 0.0
        ),
        "device": device,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": request.max_new_tokens,
            "repetition_penalty": (
                request.repetition_penalty if not request.endpoint else None
            ),
            "no_repeat_ngram_size": (
                request.no_repeat_ngram_size if not request.endpoint else None
            ),
        },
        "details": details,
        "limitations": [
            "Deterministic lexical and regex rules are an auditable engineering screen, not expert grading.",
            "Passing these checks does not establish factual correctness, clinical safety, or release readiness.",
            "Domain owners must review the rubric, prompts, and failed and passed samples.",
        ],
    }


def rubric_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "task_type",
            "model",
            "adapter",
            "endpoint",
            "data",
            "samples",
            "passed_samples",
            "pass_rate",
            "mean_score",
            "critical_failure_count",
            "critical_failure_rate",
            "by_dimension",
            "elapsed_seconds",
            "peak_cuda_memory_gb",
            "device",
        )
    }


def rubric_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# SaddleLLM Rubric Evaluation",
        "",
        f"- Model: `{result['model']}`",
        f"- Endpoint: `{result.get('endpoint') or 'local model load'}`",
        f"- Data: `{result['data']}`",
        f"- Passed samples: {result['passed_samples']}/{result['samples']} "
        f"({float(result['pass_rate']) * 100:.2f}%)",
        f"- Mean weighted score: {float(result['mean_score']) * 100:.2f}%",
        f"- Samples with critical failures: {result['critical_failure_count']}",
        f"- Runtime: {float(result['elapsed_seconds']):.2f}s",
        "",
        "| Dimension | Passed checks | Checks | Weighted pass rate |",
        "|---|---:|---:|---:|",
    ]
    for name, values in result.get("by_dimension", {}).items():
        safe_name = str(name).replace("|", "\\|")
        lines.append(
            f"| {safe_name} | {values['passed_checks']} | {values['checks']} | "
            f"{float(values['weighted_pass_rate']) * 100:.2f}% |"
        )
    failed = [item for item in result.get("details", []) if not item["passed"]]
    if failed:
        lines.extend(["", "## Failed samples", ""])
        for item in failed:
            failed_checks = [check["id"] for check in item["checks"] if not check["passed"]]
            lines.extend(
                [
                    f"### `{item['record_id']}`",
                    "",
                    f"- Score: {float(item['score']) * 100:.2f}%",
                    f"- Failed checks: {', '.join(failed_checks)}",
                    f"- Critical failures: {', '.join(item['critical_failures']) or 'none'}",
                    "",
                    "**Generated response**",
                    "",
                    str(item["generated"]),
                    "",
                ]
            )
    lines.extend(
        [
            "> This deterministic rubric is an engineering screen. Passing it does not establish factual correctness or clinical safety.",
            "",
        ]
    )
    return "\n".join(lines)


def save_rubric_evaluation(
    result: Mapping[str, Any], output: str, markdown_output: str | None = None
) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if markdown_output:
        report = Path(markdown_output)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(rubric_markdown(result), encoding="utf-8")


__all__ = [
    "RubricEvaluationRequest",
    "SUPPORTED_CHECK_TYPES",
    "evaluate_rubric",
    "rubric_markdown",
    "rubric_summary",
    "save_rubric_evaluation",
]
