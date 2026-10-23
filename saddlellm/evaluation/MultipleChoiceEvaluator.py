"""Reproducible local multiple-choice evaluation for base and PEFT models."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass
class MultipleChoiceEvaluationRequest:
    model: str
    data: str
    adapter: str | None = None
    choices: str = "ABCDE"
    prompt_field: str = "instruction"
    question_field: str = "question"
    answer_field: str = "answer"
    id_field: str = "id"
    group_field: str | None = None
    case_field: str | None = None
    variant_field: str | None = None
    prompt_variants: str | None = None
    limit: int = 0
    batch_size: int = 16
    max_length: int = 512
    generation_samples: int = 0
    generation_indices: Sequence[int] = field(default_factory=tuple)
    max_new_tokens: int = 96
    device: str = "auto"
    dtype: str = "auto"
    enable_thinking: bool = False
    trust_remote_code: bool = False
    local_files_only: bool = False
    split: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_evaluation_json(value: Any, split: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
    """Select a named split exactly; flat files only declare their scope."""
    selected_split = split
    if split is not None and (not isinstance(split, str) or not split.strip()):
        raise ValueError("evaluation split must be a nonempty name")
    if isinstance(value, dict):
        if isinstance(value.get("splits"), dict):
            value = value["splits"]
        named = [key for key in ("train", "validation", "test", "calibration") if isinstance(value.get(key), list)]
        if split is not None and isinstance(value.get(split), list):
            value = value[split]
        elif named:
            if split is not None:
                raise ValueError(f"Requested evaluation split {split!r} is missing; available splits: {named}")
            if len(named) != 1:
                raise ValueError(f"Evaluation JSON contains multiple splits {named}; choose --split explicitly")
            selected_split = named[0]
            value = value[named[0]]
        else:
            wrappers = [key for key in ("data", "records") if isinstance(value.get(key), list)]
            if len(wrappers) == 1:
                value = value[wrappers[0]]
            elif len(wrappers) > 1:
                raise ValueError("Evaluation JSON contains multiple record containers; choose one explicitly")
            else:
                raise ValueError("Evaluation JSON must contain an array of records or the requested split")
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("Evaluation data must contain a JSON array of objects")
    return value, selected_split


def load_evaluation_records(path: str, limit: int = 0, split: str | None = None) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"evaluation data file not found: {source}")
    if source.suffix.lower() == ".jsonl":
        rows = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON on line {line_number} of {source}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise TypeError(
                        f"line {line_number} of {source} must contain a JSON object"
                    )
                rows.append(value)
                if limit > 0 and len(rows) >= limit:
                    break
        return rows
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        value, _ = select_evaluation_json(value, split)
        return list(value[:limit] if limit > 0 else value)
    raise ValueError("evaluation data must be a .jsonl or .json file")


def load_prompt_variant_templates(path: str) -> list[dict[str, str]]:
    """Load domain-owned prompt templates without embedding domain logic here."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"prompt variant file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_variants = payload.get("variants") if isinstance(payload, Mapping) else None
    if isinstance(raw_variants, Mapping):
        raw_variants = [
            {"name": str(name), "template": str(template)}
            for name, template in raw_variants.items()
        ]
    if not isinstance(raw_variants, list) or len(raw_variants) < 2:
        raise ValueError("prompt variant file requires at least two variants")
    variants = []
    names = set()
    for index, value in enumerate(raw_variants):
        if not isinstance(value, Mapping):
            raise TypeError(f"prompt variant {index} must be an object")
        name = str(value.get("name", "")).strip()
        template = str(value.get("template", "")).strip()
        if not name or not template:
            raise ValueError(f"prompt variant {index} requires name and template")
        if name in names:
            raise ValueError(f"duplicate prompt variant name: {name}")
        names.add(name)
        variants.append({"name": name, "template": template})
    return variants


class _TemplateValues(dict):
    def __missing__(self, key):
        raise KeyError(key)


def _template_value(value: Any) -> str:
    """Render common structured fields as readable prompt text."""

    if isinstance(value, Mapping):
        return "\n".join(f"{key}. {item}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "\n".join(str(item) for item in value)
    return str(value)


def expand_prompt_variants(
    rows: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, str]],
    *,
    id_field: str,
    prompt_field: str,
) -> list[dict[str, Any]]:
    """Expand each case with declarative templates owned by the domain project."""

    expanded = []
    seen_case_ids = set()
    for index, source in enumerate(rows):
        case_id = str(source.get(id_field, index)).strip()
        if not case_id:
            raise ValueError(f"record {index} has an empty case id in {id_field!r}")
        if case_id in seen_case_ids:
            raise ValueError(
                f"prompt variant expansion requires unique source ids; duplicate {case_id!r}"
            )
        seen_case_ids.add(case_id)
        values = _TemplateValues(
            {key: _template_value(value) for key, value in source.items()}
        )
        for variant in variants:
            try:
                instruction = variant["template"].format_map(values).strip()
            except KeyError as exc:
                raise ValueError(
                    f"prompt variant {variant['name']!r} references missing field {exc.args[0]!r}"
                ) from exc
            if not instruction:
                raise ValueError(
                    f"prompt variant {variant['name']!r} produced an empty prompt"
                )
            row = dict(source)
            row[prompt_field] = instruction
            row[id_field] = f"{case_id}::{variant['name']}"
            row["_saddle_case_id"] = case_id
            row["_saddle_prompt_variant"] = variant["name"]
            expanded.append(row)
    return expanded


def _render_prompt(tokenizer, instruction: str, *, enable_thinking: bool) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return instruction
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], **kwargs
        )
    except TypeError:
        kwargs.pop("enable_thinking")
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}], **kwargs
            )
        except ValueError as exc:
            if "chat template" not in str(exc).lower():
                raise
            return instruction
    except ValueError as exc:
        if "chat template" not in str(exc).lower():
            raise
        return instruction


def _instruction(row: Mapping[str, Any], request: MultipleChoiceEvaluationRequest) -> str:
    value = str(row.get(request.prompt_field, "")).strip()
    if value:
        return value
    question = str(row.get(request.question_field, "")).strip()
    options = row.get("options", "")
    if isinstance(options, Mapping):
        options = "\n".join(f"{key}. {value}" for key, value in options.items())
    options = str(options).strip()
    if not question:
        raise ValueError(
            f"record has neither {request.prompt_field!r} nor {request.question_field!r}"
        )
    return f"{question}\n{options}" if options else question


def _validate_records(
    rows: Sequence[Mapping[str, Any]], request: MultipleChoiceEvaluationRequest
) -> None:
    if not rows:
        raise ValueError("evaluation dataset is empty")
    if len(set(request.choices)) != len(request.choices) or len(request.choices) < 2:
        raise ValueError("choices must contain at least two unique label characters")
    allowed = set(request.choices)
    case_answers: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        answer = str(row.get(request.answer_field, "")).strip().upper()
        if answer not in allowed:
            raise ValueError(
                f"record {index} has invalid {request.answer_field!r}: {answer!r}; "
                f"expected one of {request.choices}"
            )
        _instruction(row, request)
        if request.case_field:
            case_id = str(row.get(request.case_field, "")).strip()
            if not case_id:
                raise ValueError(
                    f"record {index} is missing case field {request.case_field!r}"
                )
            case_answers[case_id].add(answer)
        if request.variant_field and not str(
            row.get(request.variant_field, "")
        ).strip():
            raise ValueError(
                f"record {index} is missing variant field {request.variant_field!r}"
            )
    if request.variant_field and not request.case_field:
        raise ValueError("variant_field requires case_field")
    inconsistent = [
        case_id for case_id, answers in case_answers.items() if len(answers) > 1
    ]
    if inconsistent:
        preview = ", ".join(inconsistent[:3])
        raise ValueError(
            "prompt variants for the same case must share one answer; "
            f"inconsistent case(s): {preview}"
        )


def _prompt_robustness(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize accuracy and prediction stability across prompt variants."""

    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_variant: dict[str, Counter] = defaultdict(Counter)
    for row in details:
        by_case[str(row["case_id"])].append(row)
        variant = str(row.get("variant", "unlabeled"))
        by_variant[variant]["total"] += 1
        by_variant[variant]["correct"] += int(bool(row["correct"]))

    multi_variant_cases = 0
    prediction_flip_cases = 0
    all_correct_cases = 0
    case_summaries = []
    for case_id, rows in sorted(by_case.items()):
        predictions = [str(row["prediction"]) for row in rows]
        correct = sum(bool(row["correct"]) for row in rows)
        if len(rows) > 1:
            multi_variant_cases += 1
            prediction_flip_cases += int(len(set(predictions)) > 1)
        all_correct_cases += int(correct == len(rows))
        case_summaries.append(
            {
                "case_id": case_id,
                "answer": rows[0]["answer"],
                "variants": [str(row.get("variant", "unlabeled")) for row in rows],
                "predictions": predictions,
                "accuracy": correct / len(rows),
                "all_variants_correct": correct == len(rows),
                "prediction_flipped": len(set(predictions)) > 1,
            }
        )

    variant_metrics = {
        name: {
            "samples": counts["total"],
            "correct": counts["correct"],
            "accuracy": counts["correct"] / counts["total"],
        }
        for name, counts in sorted(by_variant.items())
    }
    variant_accuracies = [value["accuracy"] for value in variant_metrics.values()]
    cases = len(by_case)
    return {
        "case_field": "case_id",
        "cases": cases,
        "multi_variant_cases": multi_variant_cases,
        "singleton_cases": cases - multi_variant_cases,
        "mean_variants_per_case": len(details) / cases if cases else 0.0,
        "prediction_flip_cases": prediction_flip_cases,
        "prediction_flip_rate": (
            prediction_flip_cases / multi_variant_cases
            if multi_variant_cases
            else None
        ),
        "all_variants_correct_cases": all_correct_cases,
        "all_variants_correct_rate": all_correct_cases / cases if cases else 0.0,
        "worst_variant_accuracy": min(variant_accuracies) if variant_accuracies else None,
        "best_variant_accuracy": max(variant_accuracies) if variant_accuracies else None,
        "by_variant": variant_metrics,
        "case_details": case_summaries,
        "definitions": {
            "prediction_flip_rate": "cases with different predicted labels across prompts / multi-variant cases",
            "all_variants_correct_rate": "cases answered correctly under every supplied prompt / all cases",
        },
    }


def _select_runtime(request: MultipleChoiceEvaluationRequest):
    import torch
    from ..models.ModelLoader import packaged_quantization_kind

    kind = packaged_quantization_kind(request.model)
    device = request.device
    if device == "auto":
        device = (
            "cpu"
            if kind in {"dynamic-int8", "portable-int8-weight-only"}
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    dtype: Any = request.dtype
    if dtype == "auto":
        if kind in {"dynamic-int8", "portable-int8-weight-only"}:
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
                torch.float16 if device == "cuda" else torch.float32
            )
    return device, dtype


def _load_model(request: MultipleChoiceEvaluationRequest):
    from ..models.ModelLoader import load_model_and_tokenizer

    device, dtype = _select_runtime(request)
    if request.adapter:
        from ..training.PostTrainingCompatibility import (
            stabilize_peft_optional_backends,
        )

        stabilize_peft_optional_backends()
    model, tokenizer = load_model_and_tokenizer(
        request.model,
        device=device,
        dtype=dtype,
        trust_remote_code=request.trust_remote_code,
        local_files_only=request.local_files_only,
    )
    if request.adapter:
        from peft import PeftModel

        try:
            model = PeftModel.from_pretrained(
                model,
                request.adapter,
                local_files_only=request.local_files_only,
            )
        except Exception as exc:
            from ..training.PostTrainingCompatibility import (
                format_post_training_runtime_error,
                post_training_runtime_report,
            )

            if post_training_runtime_report()["issues"]:
                raise RuntimeError(
                    format_post_training_runtime_error("PEFT adapter evaluation", exc)
                ) from exc
            raise
        model.to(device)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer, device


def _label_token_ids(tokenizer, choices: str) -> list[int]:
    token_ids = []
    for label in choices:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"choice label {label!r} is not one token for this tokenizer: {encoded}"
            )
        token_ids.append(encoded[0])
    return token_ids


def _predict_batches(model, tokenizer, device: str, rows, request):
    import torch

    label_ids = _label_token_ids(tokenizer, request.choices)
    details = []
    with torch.inference_mode():
        for start in range(0, len(rows), request.batch_size):
            batch = rows[start : start + request.batch_size]
            prompts = [
                _render_prompt(
                    tokenizer,
                    _instruction(row, request),
                    enable_thinking=request.enable_thinking,
                )
                for row in batch
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=request.max_length,
            ).to(device)
            logits = model(**inputs).logits[:, -1, :].float()
            option_logits = logits[:, label_ids]
            probabilities = torch.softmax(option_logits, dim=-1)
            predictions = option_logits.argmax(dim=-1).tolist()
            for offset, (row, prediction_index) in enumerate(zip(batch, predictions)):
                index = start + offset
                prediction = request.choices[prediction_index]
                answer = str(row[request.answer_field]).strip().upper()
                detail = {
                    "record_id": str(row.get(request.id_field, index)),
                    "source_index": index,
                    "question": str(row.get(request.question_field, "")),
                    "answer": answer,
                    "prediction": prediction,
                    "option_probabilities": {
                        label: round(float(probabilities[offset, label_index]), 8)
                        for label_index, label in enumerate(request.choices)
                    },
                    "correct": prediction == answer,
                }
                if request.group_field:
                    detail["group"] = str(row.get(request.group_field, "")) or "unknown"
                if request.case_field:
                    detail["case_id"] = str(row[request.case_field])
                    detail["variant"] = (
                        str(row[request.variant_field])
                        if request.variant_field
                        else "unlabeled"
                    )
                details.append(detail)
    return details


def _generate(model, tokenizer, device: str, rows, indices, request):
    import torch

    pattern = re.compile(f"[{re.escape(request.choices)}]", re.IGNORECASE)
    generated = []
    with torch.inference_mode():
        for index in indices:
            row = rows[index]
            prompt = _render_prompt(
                tokenizer,
                _instruction(row, request),
                enable_thinking=request.enable_thinking,
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=request.max_length,
            ).to(device)
            output = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=request.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(
                output[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            match = pattern.search(text.upper())
            prediction = match.group(0).upper() if match else None
            answer = str(row[request.answer_field]).strip().upper()
            generated.append(
                {
                    "record_id": str(row.get(request.id_field, index)),
                    "source_index": index,
                    "question": str(row.get(request.question_field, "")),
                    "answer": answer,
                    "generated": text,
                    "parsed_prediction": prediction,
                    "correct": prediction == answer,
                }
            )
    return generated


def evaluate_multiple_choice(
    request: MultipleChoiceEvaluationRequest,
) -> dict[str, Any]:
    """Evaluate one base/full/adapter model and return an auditable payload."""

    source = Path(request.data).expanduser().resolve()
    rows = load_evaluation_records(str(source), request.limit, split=request.split)
    effective_request = request
    variant_metadata = None
    if request.prompt_variants:
        if request.case_field or request.variant_field:
            raise ValueError(
                "prompt_variants cannot be combined with case_field/variant_field; "
                "the generated fields are managed automatically"
            )
        variant_path = Path(request.prompt_variants).expanduser().resolve()
        variants = load_prompt_variant_templates(str(variant_path))
        source_samples = len(rows)
        rows = expand_prompt_variants(
            rows,
            variants,
            id_field=request.id_field,
            prompt_field=request.prompt_field,
        )
        effective_request = replace(
            request,
            case_field="_saddle_case_id",
            variant_field="_saddle_prompt_variant",
        )
        variant_metadata = {
            "path": str(variant_path),
            "sha256": _sha256(variant_path),
            "names": [variant["name"] for variant in variants],
            "source_samples": source_samples,
            "expanded_samples": len(rows),
        }
    _validate_records(rows, effective_request)
    if request.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if request.max_length < 1 or request.max_new_tokens < 1:
        raise ValueError("max_length and max_new_tokens must be positive")
    if request.generation_samples < 0:
        raise ValueError("generation_samples cannot be negative")
    model, tokenizer, device = _load_model(effective_request)

    import torch

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    details = _predict_batches(model, tokenizer, device, rows, effective_request)
    if request.generation_indices:
        indices = list(request.generation_indices)
    else:
        indices = list(range(min(request.generation_samples, len(rows))))
    if any(index < 0 or index >= len(rows) for index in indices):
        raise ValueError("generation index is outside the evaluated dataset")
    generations = _generate(
        model, tokenizer, device, rows, indices, effective_request
    )
    elapsed = time.perf_counter() - started

    correct = sum(item["correct"] for item in details)
    groups: dict[str, Counter] = defaultdict(Counter)
    if request.group_field:
        for item in details:
            groups[item["group"]]["total"] += 1
            groups[item["group"]]["correct"] += int(item["correct"])
    by_group = {
        key: {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": counts["correct"] / counts["total"],
        }
        for key, counts in sorted(groups.items())
    }
    generation_correct = sum(item["correct"] for item in generations)
    result = {
        "schema_version": 1,
        "task_type": "multiple_choice",
        "model": request.model,
        "adapter": request.adapter,
        "data": str(source),
        "data_sha256": _sha256(source),
        "evaluation": f"next-token normalized label probability among {request.choices}",
        "choices": request.choices,
        "samples": len(details),
        "correct": correct,
        "accuracy": correct / len(details) if details else math.nan,
        "generation_samples": len(generations),
        "generation_correct": generation_correct,
        "generation_accuracy": (
            generation_correct / len(generations) if generations else None
        ),
        "unparsed_generations": sum(
            item["parsed_prediction"] is None for item in generations
        ),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
        ),
        "device": device,
        "by_group": by_group,
        "generations": generations,
        "details": details,
        "limitations": [
            "Normalized label probability can differ from unconstrained generation.",
            "Multiple-choice accuracy does not establish free-form or safety quality.",
        ],
    }
    if effective_request.case_field:
        result["prompt_robustness"] = _prompt_robustness(details)
        result["prompt_robustness"]["source_case_field"] = (
            request.case_field or request.id_field
        )
        result["prompt_robustness"]["source_variant_field"] = (
            request.variant_field or "prompt_variants.name"
        )
        if result["prompt_robustness"]["singleton_cases"]:
            result["limitations"].append(
                "Singleton cases do not contribute to prediction_flip_rate."
            )
    if variant_metadata:
        result["prompt_variant_templates"] = variant_metadata
    return result


def evaluation_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        key: result.get(key)
        for key in (
            "task_type",
            "model",
            "adapter",
            "data",
            "samples",
            "correct",
            "accuracy",
            "generation_samples",
            "generation_correct",
            "generation_accuracy",
            "unparsed_generations",
            "elapsed_seconds",
            "peak_cuda_memory_gb",
            "device",
            "by_group",
            "prompt_variant_templates",
        )
    }
    robustness = result.get("prompt_robustness")
    if robustness:
        summary["prompt_robustness"] = {
            key: value
            for key, value in robustness.items()
            if key not in {"case_details", "definitions"}
        }
    return summary


def evaluation_markdown(result: Mapping[str, Any]) -> str:
    accuracy = float(result["accuracy"]) * 100
    lines = [
        "# SaddleLLM Model Evaluation",
        "",
        f"- Model: `{result['model']}`",
        f"- Adapter: `{result.get('adapter') or 'none'}`",
        f"- Data: `{result['data']}`",
        f"- Label score: {result['correct']}/{result['samples']} ({accuracy:.2f}%)",
    ]
    if result.get("generation_samples"):
        generation_accuracy = float(result["generation_accuracy"]) * 100
        lines.append(
            f"- Greedy generation: {result['generation_correct']}/"
            f"{result['generation_samples']} ({generation_accuracy:.2f}%)"
        )
        lines.append(f"- Unparsed generations: {result['unparsed_generations']}")
    robustness = result.get("prompt_robustness")
    if robustness:
        flip_rate = robustness.get("prediction_flip_rate")
        lines.extend(
            [
                "",
                "## Prompt robustness",
                "",
                f"- Cases: {robustness['cases']}",
                f"- Multi-variant cases: {robustness['multi_variant_cases']}",
                f"- Prediction flip rate: {flip_rate * 100:.2f}%"
                if flip_rate is not None
                else "- Prediction flip rate: unavailable",
                f"- All variants correct: {robustness['all_variants_correct_rate'] * 100:.2f}%",
                f"- Worst variant accuracy: {robustness['worst_variant_accuracy'] * 100:.2f}%",
                "",
                "| Variant | Correct | Samples | Accuracy |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in robustness["by_variant"].items():
            lines.append(
                f"| {name} | {values['correct']} | {values['samples']} | "
                f"{values['accuracy'] * 100:.2f}% |"
            )
        templates = result.get("prompt_variant_templates")
        if templates:
            lines.extend(
                [
                    "",
                    f"- Template file: `{templates['path']}`",
                    f"- Template SHA256: `{templates['sha256']}`",
                ]
            )
    lines.extend(
        [
            f"- Runtime: {float(result['elapsed_seconds']):.2f}s",
            f"- Peak CUDA memory: {float(result['peak_cuda_memory_gb']):.2f} GiB",
            "",
            "> Multiple-choice scores do not establish free-form or clinical safety quality.",
            "",
        ]
    )
    return "\n".join(lines)


def save_evaluation(
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
        report.write_text(evaluation_markdown(result), encoding="utf-8")


__all__ = [
    "MultipleChoiceEvaluationRequest",
    "evaluate_multiple_choice",
    "evaluation_markdown",
    "evaluation_summary",
    "expand_prompt_variants",
    "load_evaluation_records",
    "load_prompt_variant_templates",
    "save_evaluation",
]
