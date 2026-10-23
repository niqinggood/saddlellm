"""Evaluate saved SaddleLLM typed decision artifacts without selecting a model.

Every training input is token-checked, but only requested held-out splits are
predicted. The majority baseline uses training targets only. A previously saved
logits.jsonl can independently check artifact reload consistency.
"""

# ruff: noqa: N999
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .dataset import sha256
from .TypedModel import TypedDecisionModel
from .TypedSchema import (
    SPLITS,
    TYPES,
    candidate_query_text,
    candidates,
    read_typed_dataset,
    scope_key,
)
from .TypedTraining import collect_logits, typed_metrics


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def probabilities(values, temperature=1.0):
    values = np.asarray(values, dtype=np.float64) / temperature
    values -= values.max()
    result = np.exp(values)
    return result / result.sum()


def selected_index(probs, row):
    options = candidates(row)
    return min(
        range(len(options)), key=lambda index: (-probs[index], options[index]["id"])
    )


def hard_target(row):
    values = np.asarray(row["target"])
    return int(values.argmax()) if np.count_nonzero(values) == 1 else None


def classification_metrics(rows, predicted):
    scopes = {scope_key(row) for row in rows}
    labels = set()
    observed = []
    score_errors, score_distances, expected_score_errors = [], [], []
    for row, prediction in zip(rows, predicted):
        options = candidates(row)
        prefix = "" if len(scopes) == 1 else scope_key(row)[:12] + ":"
        labels.update(prefix + option["id"] for option in options)
        truth = hard_target(row)
        if truth is None:
            continue
        chosen = prediction["index"]
        observed.append((prefix + options[truth]["id"], prefix + options[chosen]["id"]))
        if row["type"] == "score":
            score_errors.append(abs(options[truth]["value"] - options[chosen]["value"]))
            score_distances.append(abs(truth - chosen))
            if prediction.get("probabilities") is not None:
                mean = sum(
                    p * option["value"]
                    for p, option in zip(prediction["probabilities"], options)
                )
                expected_score_errors.append(abs(mean - options[truth]["value"]))
    labels = sorted(labels)
    indices = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, prediction in observed:
        confusion[indices[truth], indices[prediction]] += 1
    per_class = []
    for index, label in enumerate(labels):
        tp = int(confusion[index, index])
        support, predicted_count = (
            int(confusion[index].sum()),
            int(confusion[:, index].sum()),
        )
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class.append(
            {
                "label": label,
                "support": support,
                "predicted": predicted_count,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    result = {
        "hard_label_samples": len(observed),
        "correct": int(np.trace(confusion)),
        "accuracy": int(np.trace(confusion)) / len(observed) if observed else None,
        "macro_f1": float(np.mean([row["f1"] for row in per_class]))
        if observed
        else None,
        "macro_f1_labels": "All candidate labels, zero_division=0; scopes qualified when mixed",
        "confusion": {
            "labels": labels,
            "rows": "gold",
            "columns": "prediction",
            "matrix": confusion.tolist(),
        },
        "per_class": per_class,
    }
    if score_errors:
        result["score"] = {
            "samples": len(score_errors),
            "mae_argmax_value": float(np.mean(score_errors)),
            "mae_argmax_level_index": float(np.mean(score_distances)),
            "within_one_level_accuracy": float(
                np.mean(np.asarray(score_distances) <= 1)
            ),
            "mae_expected_score": float(np.mean(expected_score_errors))
            if expected_score_errors
            else None,
        }
    return result


def metrics(logits, rows, calibration=None):
    if not rows:
        return {"samples": 0}
    calibration = calibration or {}
    predicted = []
    ties = 0
    for values, row in zip(logits, rows):
        temperature = calibration.get(scope_key(row), {}).get("temperature", 1.0)
        probs = probabilities(values, temperature)
        ties += int(np.count_nonzero(probs == probs.max()) > 1)
        predicted.append({"index": selected_index(probs, row), "probabilities": probs})
    result = typed_metrics(logits, rows, calibration)
    result.update(classification_metrics(rows, predicted))
    result["argmax_tie_samples"] = ties
    result["prediction_tie_policy"] = (
        "Highest probability, then lexicographically smallest candidate id; matches TypedDecisionModel.predict"
    )
    result["calibration_coverage"] = result["calibration_in_scope_samples"] / len(rows)
    return result


def training_majorities(rows):
    counts, task_names = {}, defaultdict(set)
    for row in rows:
        scope = scope_key(row)
        task_names[scope].add(row["task_id"])
        counts.setdefault(scope, {})
        for option, mass in zip(candidates(row), row["target"]):
            counts[scope][option["id"]] = counts[scope].get(option["id"], 0.0) + mass
    return {
        scope: {
            "label": min(mass, key=lambda label: (-mass[label], label)),
            "training_target_mass": mass,
            "training_samples": sum(mass.values()),
            "task_ids": sorted(task_names[scope]),
        }
        for scope, mass in counts.items()
    }


def majority_metrics(rows, majorities):
    supported, predicted = [], []
    for row in rows:
        baseline = majorities.get(scope_key(row))
        if baseline is None:
            continue
        labels = [option["id"] for option in candidates(row)]
        supported.append(row)
        predicted.append(
            {"index": labels.index(baseline["label"]), "probabilities": None}
        )
    return {
        "rule": "Select majority candidate using training target mass only; no held-out label used to choose baseline",
        "samples": len(rows),
        "covered_samples": len(supported),
        "unsupported_scope_samples": len(rows) - len(supported),
        **classification_metrics(supported, predicted),
    }


def grouped_metrics(logits, rows, calibration, majorities):
    def summarize(indices):
        selected_rows = [rows[index] for index in indices]
        selected_logits = [logits[index] for index in indices]
        calibrated = (
            metrics(selected_logits, selected_rows, calibration)
            if calibration
            else None
        )
        return {
            "raw": metrics(selected_logits, selected_rows),
            "calibrated": calibrated,
            "calibration_status": "artifact_temperatures_used_for_matching_scopes"
            if calibration
            else "unavailable_no_saved_calibration",
            "training_majority": majority_metrics(selected_rows, majorities),
        }

    by_task = defaultdict(list)
    by_scope = defaultdict(list)
    for index, row in enumerate(rows):
        by_task[row["task_id"]].append(index)
        by_scope[scope_key(row)].append(index)
    return {
        "overall": summarize(list(range(len(rows)))),
        "by_type": {
            kind: summarize(
                [index for index, row in enumerate(rows) if row["type"] == kind]
            )
            for kind in TYPES
        },
        "by_task": {
            name: summarize(indices) for name, indices in sorted(by_task.items())
        },
        "by_scope": {
            name: summarize(indices) for name, indices in sorted(by_scope.items())
        },
    }


def token_preflight(model, splits):
    def token_length(text):
        if model.tokenizer is None:
            return len(text.encode("utf-8")) + 1
        return len(
            model.tokenizer(
                text, truncation=False, padding=False, add_special_tokens=True
            )["input_ids"]
        )

    summaries, errors = {}, []
    for split, rows in splits.items():
        groups = defaultdict(lambda: {"state": [], "formatted_query": []})
        for row in rows:
            lengths = {"state": token_length(row["state"]), "formatted_query": []}
            for item in candidates(row):
                text = candidate_query_text(
                    row,
                    item,
                    instance_question_in_query=model.config.instance_question_in_query,
                )
                lengths["formatted_query"].append(token_length(text))
            groups[row["task_id"]]["state"].append(lengths["state"])
            groups[row["task_id"]]["formatted_query"].extend(lengths["formatted_query"])
            if (
                lengths["state"] > model.config.max_length
                or max(lengths["formatted_query"]) > model.config.max_length
            ):
                errors.append(
                    {
                        "split": split,
                        "id": row["id"],
                        "state_id": row["state_id"],
                        "task_id": row["task_id"],
                        "state_tokens": lengths["state"],
                        "formatted_query_tokens": lengths["formatted_query"],
                    }
                )
        summaries[split] = {
            name: {
                component: {
                    "count": len(values),
                    "min": min(values),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "max": max(values),
                }
                for component, values in components.items()
            }
            for name, components in sorted(groups.items())
        }
    return {
        "max_length": model.config.max_length,
        "truncation_performed": False,
        "all_four_splits_checked": True,
        "by_split_task": summaries,
        "overlong_samples": errors,
    }


def prediction_row(split, row, values, calibration):
    options = candidates(row)
    scope = scope_key(row)
    truth = hard_target(row)

    def decision(temperature):
        probs = probabilities(values, temperature)
        selected = selected_index(probs, row)
        result = {
            "pred": options[selected]["id"],
            "probabilities": probs.tolist(),
            "max_probability": float(probs[selected]),
            "correct": selected == truth if truth is not None else None,
            "temperature": temperature,
        }
        if row["type"] == "score":
            result["expected_score"] = sum(
                float(p) * option["value"] for p, option in zip(probs, options)
            )
        if row["type"] == "noul":
            result["probability_true"] = float(probs[1])
        return result

    result = {
        "split": split,
        "id": row["id"],
        "source_id": row["id"],
        "state_id": row["state_id"],
        "task_id": row["task_id"],
        "type": row["type"],
        "scope_key": scope,
        "candidate_ids": [option["id"] for option in options],
        "label": options[truth]["id"] if truth is not None else None,
        "target_probs": row["target"],
        "raw": decision(1.0),
        "calibrated": decision(calibration[scope]["temperature"])
        if scope in calibration
        else None,
        "calibration_status": "calibrated"
        if scope in calibration
        else "out_of_scope"
        if calibration
        else "uncalibrated",
    }
    if row["type"] == "score":
        result["candidate_values"] = [option["value"] for option in options]
    return result


def compare_reference(reference_path, actual, *, atol, rtol):
    reference_path = Path(reference_path)
    reference = {}
    for line in Path(reference_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["split"], row["id"])
        if key in reference:
            raise ValueError(f"duplicate logits reference key: {key}")
        reference[key] = row
    mismatches, exact, maximum = [], 0, 0.0
    for key, row in actual.items():
        expected = reference.get(key)
        if expected is None:
            mismatches.append(
                {"split": key[0], "id": key[1], "reason": "missing_reference"}
            )
            continue
        left, right = np.asarray(row["logits"]), np.asarray(expected["logits"])
        if (
            row["candidate_ids"] != expected["candidate_ids"]
            or left.shape != right.shape
        ):
            mismatches.append(
                {"split": key[0], "id": key[1], "reason": "candidate_or_shape_mismatch"}
            )
            continue
        error = float(np.max(np.abs(left - right)))
        maximum = max(maximum, error)
        exact += int(np.array_equal(left, right))
        if not np.allclose(left, right, rtol=rtol, atol=atol):
            mismatches.append(
                {
                    "split": key[0],
                    "id": key[1],
                    "reason": "logits_mismatch",
                    "max_absolute_error": error,
                }
            )
        for field in ("raw_pred", "calibrated_pred"):
            if field in expected and row.get(field) != expected[field]:
                mismatches.append(
                    {"split": key[0], "id": key[1], "reason": field + "_mismatch"}
                )
    extra = sorted(set(reference) - set(actual))
    return {
        "reference": str(Path(reference_path).resolve()),
        "reference_sha256": sha256(reference_path),
        "samples": len(actual),
        "reference_samples": len(reference),
        "exact_logits_samples": exact,
        "max_absolute_error": maximum,
        "atol": atol,
        "rtol": rtol,
        "extra_reference_rows": [
            {"split": split, "id": identifier} for split, identifier in extra
        ],
        "mismatches": mismatches,
        "passed": not mismatches and not extra,
        "used_for_model_selection": False,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output",
        "--output-dir",
        required=True,
        help="New directory for report.json and held-out predictions/logits",
    )
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--run-label", default="model", choices=("baseline", "trained", "model")
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "calibration", "test"),
        default=["validation", "calibration", "test"],
    )
    parser.add_argument("--logits-reference")
    parser.add_argument("--reference-atol", type=float, default=0.0)
    parser.add_argument("--reference-rtol", type=float, default=0.0)
    return parser


def evaluate_typed(
    data,
    model,
    output=None,
    *,
    output_dir=None,
    device="cuda",
    batch_size=4,
    cpu_threads=4,
    run_label="model",
    splits=("validation", "calibration", "test"),
    logits_reference=None,
    reference_atol=0.0,
    reference_rtol=0.0,
    progress=True,
):
    """Evaluate an immutable typed artifact using its already saved temperatures.

    Return the report also saved as ``report.json`` in a new output directory.
    Full data/token preflight includes all four splits; model logits and metrics
    use only the requested held-out splits. Test and calibration labels never
    choose a model or the train-only majority baseline. A reference mismatch is
    recorded in ``reload_consistency.passed`` rather than silently accepted.
    """
    if output is None:
        output = output_dir
    elif (
        output_dir is not None and Path(output).resolve() != Path(output_dir).resolve()
    ):
        raise ValueError("output and output_dir must identify the same directory")
    if output is None:
        raise ValueError("output or output_dir is required")
    requested_splits = [splits] if isinstance(splits, str) else list(splits)
    if (
        type(batch_size) is not int
        or batch_size < 1
        or type(cpu_threads) is not int
        or cpu_threads < 1
        or not requested_splits
        or len(set(requested_splits)) != len(requested_splits)
        or any(
            split not in ("validation", "calibration", "test")
            for split in requested_splits
        )
    ):
        raise ValueError(
            "batch_size/cpu_threads must be positive integers; request distinct nonempty held-out splits"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in (reference_atol, reference_rtol)
    ):
        raise ValueError("reference tolerances must be finite and nonnegative")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if run_label not in {"baseline", "trained", "model"}:
        raise ValueError("run_label must be baseline, trained or model")
    args = argparse.Namespace(
        data=str(data),
        model=str(model),
        output=str(output),
        device=device,
        batch_size=batch_size,
        cpu_threads=cpu_threads,
        run_label=run_label,
        splits=requested_splits,
        logits_reference=str(logits_reference)
        if logits_reference is not None
        else None,
        reference_atol=reference_atol,
        reference_rtol=reference_rtol,
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    started = time.perf_counter()
    torch.set_num_threads(args.cpu_threads)
    descriptor, splits, hashes = read_typed_dataset(args.data)
    model = TypedDecisionModel.from_pretrained(args.model, device=args.device)
    model.eval()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    preflight = token_preflight(model, splits)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "token_preflight.json", preflight)
    if preflight["overlong_samples"]:
        write_json(
            output / "failure.json",
            {
                "stage": "all_split_token_preflight",
                "error": "Overlong samples found; no inputs truncated or predicted",
                "overlong_samples": preflight["overlong_samples"],
            },
        )
        raise ValueError(
            f"{len(preflight['overlong_samples'])} overlong rows; see {output / 'token_preflight.json'}"
        )
    majorities = training_majorities(splits["train"])
    report = {
        "schema_version": 1,
        "run_label": args.run_label,
        "model": str(Path(args.model).resolve()),
        "model_manifest_sha256": sha256(Path(args.model) / "manifest.json"),
        "data": str(Path(args.data).resolve()),
        "dataset_hashes": hashes,
        "dataset_descriptor": descriptor,
        "counts": {name: len(splits[name]) for name in SPLITS},
        "settings": vars(args),
        "model_config": model.config.to_dict(),
        "training_majorities": majorities,
        "saved_calibration": model.calibration,
        "test_used_for_model_selection": False,
        "calibration_used_for_model_selection": False,
        "evaluation_only": True,
        "calibration_fit_performed": False,
        "splits": {},
    }
    report["task_schemas_by_scope"] = {
        scope_key(row): {
            "type": row["type"],
            "task_id": row["task_id"],
            "question": row["question"],
            "candidates": candidates(row),
        }
        for rows in splits.values()
        for row in rows
    }
    actual_logits = {}
    with (
        (output / "predictions.jsonl").open("w", encoding="utf-8") as prediction_stream,
        (output / "logits.jsonl").open("w", encoding="utf-8") as logit_stream,
    ):
        for split in args.splits:
            logits = collect_logits(model, splits[split], args.batch_size)
            report["splits"][split] = grouped_metrics(
                logits, splits[split], model.calibration, majorities
            )
            for row, values in zip(splits[split], logits):
                prediction = prediction_row(split, row, values, model.calibration)
                saved_logits = {
                    "split": split,
                    "id": row["id"],
                    "candidate_ids": prediction["candidate_ids"],
                    "logits": values.tolist(),
                    "raw_pred": prediction["raw"]["pred"],
                    "calibrated_pred": prediction["calibrated"]["pred"]
                    if prediction["calibrated"]
                    else None,
                }
                prediction_stream.write(
                    json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n"
                )
                logit_stream.write(
                    json.dumps(saved_logits, ensure_ascii=False, allow_nan=False) + "\n"
                )
                actual_logits[(split, row["id"])] = saved_logits
            if progress:
                print(
                    json.dumps(
                        {
                            "split": split,
                            "samples": len(logits),
                            "raw_accuracy": report["splits"][split]["overall"]["raw"][
                                "accuracy"
                            ],
                        }
                    ),
                    flush=True,
                )
    report["reload_consistency"] = (
        compare_reference(
            args.logits_reference,
            actual_logits,
            atol=args.reference_atol,
            rtol=args.reference_rtol,
        )
        if args.logits_reference
        else {
            "checked": False,
            "reason": "No --logits-reference supplied; artifact manifest hashes were verified by from_pretrained",
        }
    )
    report["artifacts"] = {
        name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
        for name in ("token_preflight.json", "predictions.jsonl", "logits.jsonl")
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    if args.device == "cuda":
        report["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["gpu_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(output / "report.json", report)
    if progress:
        print(
            json.dumps(
                {
                    "completed": True,
                    "report": str(output / "report.json"),
                    "run_label": args.run_label,
                    "elapsed_seconds": report["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    return report


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1
        or args.cpu_threads < 1
        or len(set(args.splits)) != len(args.splits)
    ):
        parser.error(
            "batch-size/cpu-threads must be positive and splits must not repeat"
        )
    if any(
        not math.isfinite(value) or value < 0
        for value in (args.reference_atol, args.reference_rtol)
    ):
        parser.error("reference tolerances must be finite and nonnegative")
    report = evaluate_typed(**vars(args))
    return (
        0 if not args.logits_reference or report["reload_consistency"]["passed"] else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
