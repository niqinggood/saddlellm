"""Completion-only perplexity and deterministic generation evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .MultipleChoiceEvaluator import (
    _load_model,
    _render_prompt,
    load_evaluation_records,
)


@dataclass
class CompletionEvaluationRequest:
    model: str
    data: str
    adapter: str | None = None
    prompt_field: str = "instruction"
    completion_field: str = "output"
    id_field: str = "id"
    limit: int = 0
    max_length: int = 512
    generation_samples: int = 0
    generation_indices: Sequence[int] = field(default_factory=tuple)
    max_new_tokens: int = 96
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
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


def _render_pair(tokenizer, prompt_text: str, completion_text: str, enable_thinking: bool):
    prompt = _render_prompt(
        tokenizer,
        prompt_text,
        enable_thinking=enable_thinking,
    )
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": completion_text},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": False}
        try:
            full_text = tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:
            try:
                full_text = tokenizer.apply_chat_template(messages, **kwargs)
            except ValueError as exc:
                if "chat template" not in str(exc).lower():
                    raise
                full_text = ""
        except ValueError as exc:
            if "chat template" not in str(exc).lower():
                raise
            full_text = ""
        if full_text.startswith(prompt):
            return prompt, full_text[len(prompt) :]
    eos = tokenizer.eos_token or ""
    return prompt, f" {completion_text}{eos}"


def _completion_nll(model, tokenizer, device, prompt, completion, max_length):
    import torch
    from torch.nn import functional

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    encoded = tokenizer(
        f"{prompt}{completion}",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
    valid_tokens = int((labels[:, 1:] != -100).sum().item())
    if valid_tokens == 0:
        return 0.0, 0
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[
            :, :-1, :
        ].float()
    shifted_labels = labels[:, 1:]
    loss_sum = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(loss_sum.item()), valid_tokens


def _generate(model, tokenizer, device, prompt, request):
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=request.max_length,
    ).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=request.repetition_penalty,
            no_repeat_ngram_size=request.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def evaluate_completions(request: CompletionEvaluationRequest) -> dict[str, Any]:
    """Compute completion-token NLL/PPL and optional deterministic generations."""

    if request.max_length < 1 or request.max_new_tokens < 1:
        raise ValueError("max_length and max_new_tokens must be positive")
    if request.generation_samples < 0:
        raise ValueError("generation_samples cannot be negative")
    if request.repetition_penalty <= 0 or request.no_repeat_ngram_size < 0:
        raise ValueError("invalid repetition control values")
    source = Path(request.data).expanduser().resolve()
    rows = load_evaluation_records(str(source), request.limit, split=request.split)
    if not rows:
        raise ValueError("evaluation dataset is empty")
    for index, row in enumerate(rows):
        if not str(row.get(request.prompt_field, "")).strip():
            raise ValueError(f"record {index} is missing {request.prompt_field!r}")
        if not str(row.get(request.completion_field, "")).strip():
            raise ValueError(f"record {index} is missing {request.completion_field!r}")

    model, tokenizer, device = _load_model(request)
    import torch

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    total_nll = 0.0
    total_tokens = 0
    details = []
    rendered_prompts = []
    for index, row in enumerate(rows):
        prompt_text = str(row[request.prompt_field])
        completion_text = str(row[request.completion_field])
        prompt, completion = _render_pair(
            tokenizer,
            prompt_text,
            completion_text,
            request.enable_thinking,
        )
        rendered_prompts.append(prompt)
        loss_sum, tokens = _completion_nll(
            model,
            tokenizer,
            device,
            prompt,
            completion,
            request.max_length,
        )
        total_nll += loss_sum
        total_tokens += tokens
        details.append(
            {
                "record_id": str(row.get(request.id_field, index)),
                "source_index": index,
                "completion_tokens": tokens,
                "completion_nll": loss_sum / tokens if tokens else None,
            }
        )
    if total_tokens == 0:
        raise ValueError("no reference completion tokens remain after truncation")

    if request.generation_indices:
        indices = list(request.generation_indices)
    else:
        indices = list(range(min(request.generation_samples, len(rows))))
    if any(index < 0 or index >= len(rows) for index in indices):
        raise ValueError("generation index is outside the evaluated dataset")
    generations = []
    for index in indices:
        row = rows[index]
        generations.append(
            {
                "record_id": str(row.get(request.id_field, index)),
                "source_index": index,
                "instruction": str(row[request.prompt_field]),
                "reference": str(row[request.completion_field]),
                "generated": _generate(
                    model, tokenizer, device, rendered_prompts[index], request
                ),
            }
        )
    mean_nll = total_nll / total_tokens
    return {
        "schema_version": 1,
        "task_type": "completion",
        "model": request.model,
        "adapter": request.adapter,
        "data": str(source),
        "data_sha256": _sha256(source),
        "samples": len(rows),
        "completion_tokens": total_tokens,
        "completion_nll": mean_nll,
        "completion_perplexity": math.exp(min(mean_nll, 20.0)),
        "generation_samples": len(generations),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
        ),
        "device": device,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": request.max_new_tokens,
            "repetition_penalty": request.repetition_penalty,
            "no_repeat_ngram_size": request.no_repeat_ngram_size,
        },
        "details": details,
        "generations": generations,
        "limitations": [
            "Completion PPL is comparable only with the same tokenizer, template, and data.",
            "Lower PPL does not establish factual correctness or generation safety.",
        ],
    }


def completion_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "task_type",
            "model",
            "adapter",
            "data",
            "samples",
            "completion_tokens",
            "completion_nll",
            "completion_perplexity",
            "generation_samples",
            "elapsed_seconds",
            "peak_cuda_memory_gb",
            "device",
        )
    }


def completion_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# SaddleLLM Completion Evaluation",
        "",
        f"- Model: `{result['model']}`",
        f"- Adapter: `{result.get('adapter') or 'none'}`",
        f"- Data: `{result['data']}`",
        f"- Samples: {result['samples']}",
        f"- Completion tokens: {result['completion_tokens']}",
        f"- Completion NLL: {float(result['completion_nll']):.6f}",
        f"- Completion PPL: {float(result['completion_perplexity']):.6f}",
        f"- Generated samples: {result['generation_samples']}",
        f"- Runtime: {float(result['elapsed_seconds']):.2f}s",
        "",
        "> Lower PPL does not establish factual correctness or safety.",
        "",
    ]
    generations = result.get("generations", [])
    if generations:
        lines.extend(["## Deterministic generation samples", ""])
        for position, sample in enumerate(generations, start=1):
            lines.extend(
                [
                    f"### Sample {position} — record `{sample['record_id']}`",
                    "",
                    "**Instruction**",
                    "",
                    str(sample["instruction"]),
                    "",
                    "**Generated**",
                    "",
                    str(sample["generated"]),
                    "",
                    "<details><summary>Reference answer</summary>",
                    "",
                    str(sample["reference"]),
                    "",
                    "</details>",
                    "",
                ]
            )
    return "\n".join(lines)


def save_completion_evaluation(
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
        report.write_text(completion_markdown(result), encoding="utf-8")


__all__ = [
    "CompletionEvaluationRequest",
    "completion_markdown",
    "completion_summary",
    "evaluate_completions",
    "save_completion_evaluation",
]
