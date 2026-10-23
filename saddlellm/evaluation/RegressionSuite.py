"""Content-bound capability regression checks over the shared task evaluators.

Metric tolerances are absolute units: ``max_drop: 0.02`` on accuracy permits
two percentage points. A suite pass is an engineering regression gate, not a
substitute for the sealed-test promotion workflow in EvaluationComparison.
Importing/loading a suite never loads model runtimes or downloads artifacts.
"""

from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
# ruff: noqa: TRY004 -- Public configuration validation consistently raises ValueError.
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .EvaluationRunner import (
    _data_scope,
    _records_identity,
    evaluation_metrics,
    run_evaluation,
)
from .ModelIdentity import (
    compute_model_identity,
    identity_is_bound,
    model_identities_match,
)
from .ReleaseGate import EvaluationReleaseGate

_TASKS = {"multiple-choice", "completion", "rubric"}
_RESERVED_OPTIONS = {
    "model",
    "data",
    "task",
    "split",
    "output",
    "markdown_output",
    "adapter",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def load_regression_suite(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate JSON/YAML or an in-memory suite; resolve data relative to its file.

    Each task has ``name``, ``data``, ``task``, optional ``split``/``options``,
    and a nonempty ``metrics`` mapping. Every metric declares a nonnegative
    ``max_drop`` and boolean ``higher_is_better``; optional ``min``/``max``
    impose absolute candidate limits. No datasets/models are opened here.
    """
    base = Path.cwd()
    if isinstance(config, (str, Path)):
        source = Path(config).expanduser().resolve()
        base = source.parent
        content = source.read_text(encoding="utf-8-sig")
        if source.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            config = yaml.safe_load(content)
        else:
            config = json.loads(content)
    if not isinstance(config, Mapping):
        raise ValueError("regression suite must be a mapping")
    unknown = set(config) - {"schema_version", "name", "tasks"}
    if unknown:
        raise ValueError(f"unknown regression suite keys: {sorted(unknown)}")
    if config.get("schema_version", 1) != 1:
        raise ValueError("regression suite schema_version must be 1")
    name = config.get("name", "capability-regression")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("regression suite name must be nonempty")
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("regression suite tasks must be a nonempty list")
    normalized = []
    seen = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ValueError(f"regression task {index} must be a mapping")
        unknown = set(task) - {"name", "data", "task", "split", "options", "metrics"}
        if unknown:
            raise ValueError(f"unknown regression task keys: {sorted(unknown)}")
        task_name = task.get("name")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError(f"regression task {index} requires a nonempty name")
        if task_name in seen:
            raise ValueError(f"duplicate regression task name: {task_name}")
        seen.add(task_name)
        kind = str(task.get("task", "multiple-choice")).replace("_", "-")
        if kind not in _TASKS:
            raise ValueError(f"unsupported regression task: {kind}")
        data = task.get("data")
        if not isinstance(data, str) or not data.strip():
            raise ValueError(f"regression task {task_name} requires data")
        data_path = Path(data).expanduser()
        if not data_path.is_absolute():
            data_path = base / data_path
        split = task.get("split")
        if split is not None and (not isinstance(split, str) or not split.strip()):
            raise ValueError(f"regression task {task_name} split must be nonempty")
        options = task.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError(f"regression task {task_name} options must be a mapping")
        options = dict(options)
        if set(options) & _RESERVED_OPTIONS:
            raise ValueError(
                "regression options cannot override model/data/task/split/output/adapter"
            )
        if options.get("prompt_variants"):
            variants = Path(options["prompt_variants"]).expanduser()
            options["prompt_variants"] = str(
                (variants if variants.is_absolute() else base / variants).resolve()
            )
        # JSON normalization makes tuple/list representations compare identically.
        options = json.loads(_canonical(options))
        raw_metrics = task.get("metrics")
        if not isinstance(raw_metrics, Mapping) or not raw_metrics:
            raise ValueError(f"regression task {task_name} metrics must be nonempty")
        metrics = {}
        for metric, rule in raw_metrics.items():
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError("regression metric names must be nonempty strings")
            if not isinstance(rule, Mapping):
                raise ValueError(f"metric rule {metric} must be a mapping")
            if set(rule) - {"max_drop", "higher_is_better", "min", "max"}:
                raise ValueError(f"unknown metric rule keys for {metric}")
            tolerance = _number(rule.get("max_drop"), f"{metric}.max_drop")
            if tolerance < 0:
                raise ValueError(f"{metric}.max_drop must be nonnegative")
            if not isinstance(rule.get("higher_is_better"), bool):
                raise ValueError(f"{metric}.higher_is_better must be boolean")
            normalized_rule = {
                "max_drop": tolerance,
                "higher_is_better": rule["higher_is_better"],
            }
            bounds = {}
            for bound in ("min", "max"):
                if bound in rule:
                    bounds[bound] = _number(rule[bound], f"{metric}.{bound}")
            if bounds:
                issues = EvaluationReleaseGate.validate_rules({metric: bounds})
                if issues:
                    raise ValueError("; ".join(issues))
            metrics[metric] = {**normalized_rule, **bounds}
        normalized.append(
            {
                "name": task_name,
                "data": str(data_path.resolve()),
                "task": kind,
                "split": split,
                "options": options,
                "metrics": metrics,
            }
        )
    return {"schema_version": 1, "name": name, "tasks": normalized}


def _provenance(task: Mapping[str, Any]) -> dict[str, Any]:
    settings = {key: task[key] for key in ("name", "task", "split", "options")}
    split, _ = _data_scope(task["data"], task["split"])
    return {
        "schema_version": 1,
        "task_name": task["name"],
        "settings": settings,
        "settings_sha256": _digest(settings),
        "data_sha256": _file_digest(task["data"]),
        "selected_split": split,
        "records_identity": _records_identity(
            task["data"], split, task["options"].get("limit", 0)
        ),
        "prompt_variants_sha256": _file_digest(task["options"]["prompt_variants"])
        if task["options"].get("prompt_variants")
        else None,
    }


def _read_report(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError("evaluation report must be a mapping or JSON report path")
    return dict(value)


def _score(report: Mapping[str, Any], metric: str) -> float:
    values = {**report, **evaluation_metrics(report)}
    value: Any = values
    if metric in values:
        value = values[metric]
    else:
        for component in metric.split("."):
            if not isinstance(value, Mapping) or component not in value:
                raise ValueError(f"required metric {metric!r} is missing")
            value = value[component]
    if isinstance(value, Mapping):
        value = value.get("score")
    return _number(value, f"metric {metric!r}")


def _tokenizer_signature(
    identity: Mapping[str, Any], source: str, *, evaluation_max_length: int
) -> dict[str, Any]:
    """Bind tokenizer semantics to the exact raw files in a model fingerprint."""
    component_name = "model"
    adapter = identity.get("adapter")
    if (
        isinstance(adapter, Mapping)
        and Path(source).expanduser().resolve()
        == Path(adapter["reference"]).expanduser().resolve()
    ):
        component_name = "adapter"
    component = identity[component_name]
    files = []
    for entry in component.get("files", []):
        if entry.get("kind") != "tokenizer":
            continue
        path = Path(component["reference"]) / entry["path"]
        content = path.read_bytes()
        raw_sha = hashlib.sha256(content).hexdigest()
        if raw_sha != entry["sha256"]:
            raise ValueError("tokenizer content changed since model identity binding")
        normalized_sha = raw_sha
        normalization = "raw"
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(
                    content.decode("utf-8-sig"), object_pairs_hook=_unique_json_keys
                )
                normalization = "json"
                if path.name == "tokenizer.json" and isinstance(payload, dict):
                    payload.pop("padding", None)
                    truncation = payload.pop("truncation", None)
                    if isinstance(truncation, Mapping):
                        residual = dict(truncation)
                        residual.pop("max_length", None)
                        for key, defaults in (
                            ("direction", ("Right", "right")),
                            ("strategy", ("LongestFirst", "longest_first")),
                            ("stride", (0,)),
                        ):
                            if residual.get(key) in defaults:
                                residual.pop(key, None)
                        if residual:
                            payload["truncation"] = residual
                    elif truncation is not None:
                        payload["truncation"] = truncation
                    normalization = "tokenizer-json-evaluation-runtime-defaults-v2"
                if path.name == "tokenizer_config.json" and isinstance(payload, dict):
                    for key, default in (
                        ("truncation_side", "right"),
                        ("stride", 0),
                        ("truncation_strategy", "longest_first"),
                    ):
                        if payload.get(key) == default:
                            payload.pop(key, None)
                    sibling = str(
                        Path(entry["path"]).with_name("tokenizer.json")
                    ).replace("\\", "/")
                    related = next(
                        (
                            item
                            for item in component.get("files", [])
                            if item.get("path") == sibling
                        ),
                        None,
                    )
                    cached_length = payload.get("max_length")
                    if (
                        related
                        and type(cached_length) is int
                        and cached_length > 0
                        and type(evaluation_max_length) is int
                        and evaluation_max_length > 0
                    ):
                        backend_bytes = (
                            Path(component["reference"]) / sibling
                        ).read_bytes()
                        if (
                            hashlib.sha256(backend_bytes).hexdigest()
                            != related["sha256"]
                        ):
                            raise RuntimeError(
                                "tokenizer content changed since model identity binding"
                            )
                        backend = json.loads(
                            backend_bytes.decode("utf-8-sig"),
                            object_pairs_hook=_unique_json_keys,
                        )
                        runtime = (
                            backend.get("truncation")
                            if isinstance(backend, Mapping)
                            else None
                        )
                        if (
                            isinstance(runtime, Mapping)
                            and type(runtime.get("max_length")) is int
                            and runtime["max_length"] == cached_length
                        ):
                            payload.pop("max_length")
                    normalization = (
                        "tokenizer-config-json-evaluation-runtime-defaults-v2"
                    )
                if path.name == "special_tokens_map.json" and isinstance(payload, dict):
                    payload = {
                        key: (
                            [_default_special_token(item) for item in value]
                            if isinstance(value, list)
                            else _default_special_token(value)
                        )
                        for key, value in payload.items()
                    }
                    normalization = "special-tokens-map-default-added-tokens-v1"
                normalized_sha = _digest(payload)
            except (UnicodeDecodeError, ValueError):
                normalization = "raw"
        files.append(
            {
                "path": entry["path"],
                "source_sha256": raw_sha,
                "semantic_sha256": normalized_sha,
                "normalization": normalization,
            }
        )
    signature = {
        "schema_version": 1,
        "algorithm": "sha256",
        "model_fingerprint": identity["fingerprint"],
        "component": component_name,
        "files": files,
    }
    signature["fingerprint"] = _digest(
        [
            {key: value for key, value in entry.items() if key != "source_sha256"}
            for entry in files
        ]
    )
    return signature


def _default_special_token(value: Any) -> Any:
    """Collapse only the default AddedToken representation of a special string.

    In special_tokens_map.json a string has normalized=False and special=True.
    HF's serializer omits the special flag. Other attributes, token contents,
    unknown fields and list ordering remain evidence rather than being dropped.
    """
    if not isinstance(value, Mapping) or not isinstance(value.get("content"), str):
        return value
    defaults = {
        "lstrip": False,
        "rstrip": False,
        "single_word": False,
        "normalized": False,
        "special": True,
    }
    if set(value) - {"content", "__type", *defaults}:
        return value
    if "__type" in value and value["__type"] != "AddedToken":
        return value
    if any(value.get(key, default) is not default for key, default in defaults.items()):
        return value
    return value["content"]


def _unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(
                "duplicate tokenizer JSON keys require raw-byte comparison"
            )
        value[key] = item
    return value


def _validated_tokenizer_signature(report: Mapping[str, Any]) -> str:
    signature = report.get("tokenizer_semantic_identity")
    identity = report["model_identity"]
    if not isinstance(signature, Mapping) or set(signature) != {
        "schema_version",
        "algorithm",
        "model_fingerprint",
        "component",
        "files",
        "fingerprint",
    }:
        raise ValueError("tokenizer semantic signature schema is invalid")
    if (
        type(signature["schema_version"]) is not int
        or signature["schema_version"] != 1
        or signature["algorithm"] != "sha256"
        or signature["model_fingerprint"] != identity["fingerprint"]
    ):
        raise ValueError(
            "tokenizer semantic signature is not bound to the evaluated model"
        )
    component = (
        identity.get(signature["component"])
        if signature["component"] in {"model", "adapter"}
        else None
    )
    if not isinstance(component, Mapping):
        raise ValueError("tokenizer semantic signature component is invalid")
    expected = {
        entry["path"]: entry["sha256"]
        for entry in component.get("files", [])
        if entry.get("kind") == "tokenizer"
    }
    files = signature["files"]
    if (
        not isinstance(files, list)
        or not files
        or any(not isinstance(entry, Mapping) for entry in files)
    ):
        raise ValueError("tokenizer semantic signature requires tokenizer files")
    seen = set()
    for entry in files:
        if (
            set(entry) != {"path", "source_sha256", "semantic_sha256", "normalization"}
            or not isinstance(entry["path"], str)
            or entry["path"] in seen
        ):
            raise ValueError("tokenizer semantic signature file schema is invalid")
        seen.add(entry["path"])
        if expected.get(entry["path"]) != entry["source_sha256"]:
            raise ValueError("tokenizer semantic signature source hash mismatch")
        digest = entry["semantic_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("tokenizer semantic signature digest is invalid")
        allowed = {"raw"}
        if Path(entry["path"]).suffix.lower() == ".json":
            allowed.add("json")
        if Path(entry["path"]).name == "tokenizer.json":
            allowed.add("tokenizer-json-without-runtime-padding-truncation")
            allowed.add("tokenizer-json-evaluation-runtime-defaults-v2")
        if Path(entry["path"]).name == "tokenizer_config.json":
            allowed.add("tokenizer-config-json-evaluation-runtime-defaults-v2")
        if Path(entry["path"]).name == "special_tokens_map.json":
            allowed.add("special-tokens-map-default-added-tokens-v1")
        if entry["normalization"] not in allowed or (
            entry["normalization"] == "raw" and digest != entry["source_sha256"]
        ):
            raise ValueError("tokenizer semantic signature normalization is invalid")
    if seen != set(expected):
        raise ValueError("tokenizer semantic signature file set mismatch")
    fingerprint = _digest(
        [
            {key: value for key, value in entry.items() if key != "source_sha256"}
            for entry in files
        ]
    )
    if signature["fingerprint"] != fingerprint:
        raise ValueError("tokenizer semantic signature fingerprint mismatch")
    return fingerprint


def _validate_report(
    report: Mapping[str, Any], task: Mapping[str, Any], provenance: Mapping[str, Any]
) -> None:
    if report.get("error"):
        raise ValueError(f"evaluation failed: {report['error']}")
    if report.get("regression_provenance") != provenance:
        raise ValueError(
            "evaluation data/settings provenance mismatch or missing; rerun using run_regression_suite"
        )
    if str(report.get("task_type", "")).replace("_", "-") != task["task"]:
        raise ValueError("evaluation task type mismatch")
    if report.get("data_sha256") != provenance["data_sha256"]:
        raise ValueError("evaluation data hash mismatch")
    if not identity_is_bound(report.get("model_identity")):
        raise ValueError(
            "evaluation model identity is unbound; local model content is required"
        )
    if (
        not isinstance(report.get("samples"), int)
        or isinstance(report["samples"], bool)
        or report["samples"] <= 0
    ):
        raise ValueError("evaluation requires a positive sample count")
    identity = report.get("evaluation_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("evaluation identity is missing")
    expected = {
        "data_sha256": provenance["data_sha256"],
        "split": provenance["selected_split"],
        "limit": task["options"].get("limit", 0),
        "samples": report["samples"],
        **provenance["records_identity"],
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"evaluation identity {key} mismatch")
    if str(identity.get("task", "")).replace("_", "-") != task["task"]:
        raise ValueError("evaluation identity task mismatch")


def _json_safe(value: Any) -> Any:
    """Keep invalid evaluator outputs inspectable in strict, portable JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _protect_output(
    output: str | Path | None,
    suite_source: Mapping[str, Any] | str | Path,
    config: Mapping[str, Any],
    *,
    identities: tuple[Any, ...] = (),
    report_sources: tuple[Any, ...] = (),
) -> None:
    """A report destination must preserve the inputs that justify its gate."""
    if output is None:
        return
    target = Path(output).expanduser().resolve()
    protected = []
    if isinstance(suite_source, (str, Path)):
        protected.append(Path(suite_source).expanduser().resolve())
    for task in config["tasks"]:
        protected.append(Path(task["data"]).resolve())
        if task["options"].get("prompt_variants"):
            protected.append(Path(task["options"]["prompt_variants"]).resolve())
    protected.extend(
        Path(source).expanduser().resolve()
        for source in report_sources
        if isinstance(source, (str, Path))
    )
    for identity in identities:
        if not isinstance(identity, Mapping):
            continue
        for key in ("model", "adapter"):
            component = identity.get(key)
            if not isinstance(component, Mapping) or not component.get("local"):
                continue
            root = Path(component["reference"]).expanduser().resolve()
            protected.extend(
                (root / item["path"]).resolve() for item in component.get("files", [])
            )
    for source in protected:
        if target == source or (
            target.exists() and source.exists() and os.path.samefile(target, source)
        ):
            raise ValueError(
                f"regression output must not overwrite an input artifact: {source}"
            )


def _save(report: dict[str, Any], output: str | Path | None) -> dict[str, Any]:
    report = _json_safe(report)
    if output is not None:
        target = Path(output).expanduser().resolve()
        report["output"] = str(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def validate_regression_output(
    suite: Mapping[str, Any] | str | Path,
    output: str | Path | None,
    *,
    baseline_model: str | Path | None = None,
    candidate_model: str | Path | None = None,
    baseline_adapter: str | Path | None = None,
    candidate_adapter: str | Path | None = None,
    source_suite: str | Path | None = None,
) -> None:
    """Validate pipeline report destinations before error-writing handlers.

    This planning check never loads models, fingerprints weights or evaluates
    data. It preserves suite/data/variant inputs and rejects destinations inside
    supplied local checkpoint directories. ``source_suite`` preserves the
    original configuration path when ``suite`` has already been normalized.
    """
    config = load_regression_suite(suite)
    _protect_output(output, source_suite if source_suite is not None else suite, config)
    if output is None:
        return
    destination = Path(output).expanduser().resolve()
    references = [baseline_model, candidate_model, baseline_adapter, candidate_adapter]
    for reference in tuple(references):
        if reference is None:
            continue
        adapter_config = Path(reference).expanduser() / "adapter_config.json"
        if adapter_config.is_file():
            value = json.loads(adapter_config.read_text(encoding="utf-8"))
            if isinstance(value, Mapping) and value.get("base_model_name_or_path"):
                references.append(value["base_model_name_or_path"])
    for reference in references:
        if reference is None:
            continue
        source = Path(reference).expanduser().resolve()
        if source == destination or (source.is_dir() and source in destination.parents):
            raise ValueError(
                "regression output must be outside input checkpoint directories"
            )


def compare_regression_suite(
    suite: Mapping[str, Any] | str | Path,
    baseline_reports: Mapping[str, Mapping[str, Any] | str | Path],
    candidate_reports: Mapping[str, Mapping[str, Any] | str | Path],
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Compare suite-generated reports, keyed by task name, without inference.

    Full reports are embedded in the result. Missing/nonfinite metrics,
    unbound/changing model identities and differing data/settings invalidate
    the gate. Tolerance violations produce ``failed`` rather than ``invalid``.
    """
    config = load_regression_suite(suite)
    _protect_output(
        output,
        suite,
        config,
        report_sources=tuple(baseline_reports.values())
        + tuple(candidate_reports.values()),
    )
    expected_names = {task["name"] for task in config["tasks"]}
    failures = []
    if (
        set(baseline_reports) != expected_names
        or set(candidate_reports) != expected_names
    ):
        failures.append(
            "baseline/candidate report task names must exactly match the suite"
        )
    identities: dict[str, Any] = {"baseline": None, "candidate": None}
    tasks = []
    for task in config["tasks"]:
        name = task["name"]
        result: dict[str, Any] = {
            "name": name,
            "task": task["task"],
            "data": task["data"],
            "split": task["split"],
            "accepted": False,
            "status": "invalid",
            "metrics": {},
            "errors": [],
        }
        try:
            provenance = _provenance(task)
            reports = {
                "baseline": _read_report(baseline_reports[name]),
                "candidate": _read_report(candidate_reports[name]),
            }
            result.update(
                {
                    "baseline_report": reports["baseline"],
                    "candidate_report": reports["candidate"],
                    "provenance": provenance,
                }
            )
            for role, report in reports.items():
                _validate_report(report, task, provenance)
                if identities[role] is None:
                    identities[role] = report["model_identity"]
                elif not model_identities_match(
                    identities[role], report["model_identity"]
                ):
                    raise ValueError(
                        f"{role} model identity differs across regression tasks"
                    )
            baseline, candidate = reports["baseline"], reports["candidate"]
            if baseline["samples"] != candidate["samples"]:
                raise ValueError("baseline/candidate sample counts differ")
            # The runner binds selected records; comparing its outcome-dependent
            # details_sha256 would incorrectly reject every changed prediction.
            for key in (
                "task",
                "data_sha256",
                "split",
                "limit",
                "samples",
                "records_sha256",
                "source_records",
            ):
                left = baseline.get("evaluation_identity", {}).get(key)
                right = candidate.get("evaluation_identity", {}).get(key)
                if left != right:
                    raise ValueError(f"evaluation identity {key} mismatch")
            for key in ("generation_config", "prompt_variant_templates"):
                if baseline.get(key) != candidate.get(key):
                    raise ValueError(f"evaluation {key} mismatch")
            if task["task"] == "completion":
                signatures = [
                    report.get("tokenizer_semantic_identity")
                    for report in (baseline, candidate)
                ]
                for report, signature in zip((baseline, candidate), signatures):
                    if signature is not None:
                        _validated_tokenizer_signature(report)
                if all(signature is not None for signature in signatures):
                    matched = _validated_tokenizer_signature(
                        baseline
                    ) == _validated_tokenizer_signature(candidate)
                else:
                    tokenizers = [
                        [
                            item
                            for item in report["model_identity"]
                            .get("model", {})
                            .get("files", [])
                            if item.get("kind") == "tokenizer"
                        ]
                        for report in (baseline, candidate)
                    ]
                    matched = bool(tokenizers[0]) and tokenizers[0] == tokenizers[1]
                if not matched:
                    raise ValueError(
                        "completion evaluation requires matching content-bound tokenizers/templates"
                    )
            values, rules = {}, {}
            for metric, rule in task["metrics"].items():
                base_score = _score(baseline, metric)
                candidate_score = _score(candidate, metric)
                delta = candidate_score - base_score
                _number(delta, f"metric {metric!r} delta")
                drop = -delta if rule["higher_is_better"] else delta
                bounds = {key: rule[key] for key in ("min", "max") if key in rule}
                if rule["higher_is_better"]:
                    bounds["min"] = max(
                        bounds.get("min", -math.inf), base_score - rule["max_drop"]
                    )
                else:
                    bounds["max"] = min(
                        bounds.get("max", math.inf), base_score + rule["max_drop"]
                    )
                values[metric] = candidate_score
                rules[metric] = bounds
                result["metrics"][metric] = {
                    "baseline": base_score,
                    "candidate": candidate_score,
                    "delta": delta,
                    "drop": drop,
                    **rule,
                }
            gate = EvaluationReleaseGate.evaluate(values, rules).to_dict()
            result["gate"] = gate
            result["accepted"] = gate["accepted"]
            result["status"] = "accepted" if gate["accepted"] else "failed"
            for check in gate["checks"]:
                result["metrics"][check["metric"]]["passed"] = check["passed"]
        except (KeyError, ValueError, TypeError, OSError) as error:
            result["errors"].append(str(error))
        tasks.append(result)
    invalid = bool(failures) or any(task["status"] == "invalid" for task in tasks)
    accepted = not invalid and all(task["accepted"] for task in tasks)
    _protect_output(output, suite, config, identities=tuple(identities.values()))
    return _save(
        {
            "schema_version": 1,
            "report_type": "capability_regression_suite",
            "name": config["name"],
            "suite": config,
            "suite_sha256": _digest(config),
            "enabled": True,
            "accepted": accepted,
            "status": "invalid" if invalid else "accepted" if accepted else "failed",
            "baseline_model_identity": identities["baseline"],
            "candidate_model_identity": identities["candidate"],
            "tasks": tasks,
            "errors": failures,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "output": None,
            "limitations": [
                "A regression-suite pass does not replace independent sealed-test promotion or domain review."
            ],
        },
        output,
    )


def run_regression_suite(
    baseline_model: str,
    candidate_model: str,
    suite: Mapping[str, Any] | str | Path,
    *,
    baseline_adapter: str | None = None,
    candidate_adapter: str | None = None,
    output: str | Path | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate both artifacts with identical task/data/options and apply gates.

    ``evaluator`` follows ``run_evaluation(model, data, task=..., split=...,
    **options)`` and permits deterministic CPU tests without model downloads.
    Exceptions are recorded as invalid task evidence; other tasks still run.
    """
    config = load_regression_suite(suite)
    evaluate = evaluator or run_evaluation
    identities = {
        "baseline": compute_model_identity(baseline_model, baseline_adapter),
        "candidate": compute_model_identity(candidate_model, candidate_adapter),
    }
    _protect_output(output, suite, config, identities=tuple(identities.values()))
    reports: dict[str, dict[str, Any]] = {"baseline": {}, "candidate": {}}
    for task in config["tasks"]:
        for role, model, adapter in (
            ("baseline", baseline_model, baseline_adapter),
            ("candidate", candidate_model, candidate_adapter),
        ):
            try:
                provenance = _provenance(task)
                options = dict(task["options"])
                if adapter is not None:
                    options["adapter"] = adapter
                report = dict(
                    evaluate(
                        model,
                        task["data"],
                        task=task["task"],
                        split=task["split"],
                        **options,
                    )
                )
                if _provenance(task) != provenance:
                    raise ValueError(
                        "evaluation source data/settings changed while evaluating"
                    )
                if not model_identities_match(
                    identities[role], report.get("model_identity")
                ):
                    raise ValueError(
                        f"{role} evaluator model identity does not match the requested artifact"
                    )
                if not model_identities_match(
                    identities[role], compute_model_identity(model, adapter)
                ):
                    raise ValueError(f"{role} model content changed while evaluating")
                if (
                    "regression_provenance" in report
                    and report["regression_provenance"] != provenance
                ):
                    raise ValueError(
                        "evaluator returned inconsistent regression provenance"
                    )
                report["regression_provenance"] = provenance
                if task["task"] == "completion":
                    from .CompletionEvaluator import CompletionEvaluationRequest

                    signature = _tokenizer_signature(
                        report["model_identity"],
                        model,
                        evaluation_max_length=task["options"].get(
                            "max_length", CompletionEvaluationRequest.max_length
                        ),
                    )
                    if (
                        report.get("tokenizer_semantic_identity", signature)
                        != signature
                    ):
                        raise ValueError(
                            "evaluator tokenizer semantic signature mismatch"
                        )
                    report["tokenizer_semantic_identity"] = signature
            except Exception as error:  # noqa: BLE001 -- Isolate optional evaluator/runtime failures per task.
                report = {"error": str(error)}
            reports[role][task["name"]] = report
    return compare_regression_suite(
        config, reports["baseline"], reports["candidate"], output=output
    )


__all__ = [
    "compare_regression_suite",
    "load_regression_suite",
    "run_regression_suite",
    "validate_regression_output",
]
