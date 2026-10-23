"""Shared runtime holdout checks for direct HF and native training APIs."""

from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
import hashlib
import json
import os
import tempfile
from pathlib import Path


def _digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_identities(records, task):
    from .PostTrainingData import PostTrainingDataAdapter
    from .ReplayData import _normalized_text, _record_identities

    identities = set()
    for record in records:
        record_identities = _record_identities(record)
        if task != "sft":
            prompt = PostTrainingDataAdapter._prompt_from_record(record)
            if prompt:
                record_identities.add(("prompt", _digest(_normalized_text(prompt))))
        identities.update(record_identities)
    return identities


def _active_tokens(input_ids, attention_mask):
    """Remove padding before comparing token sequences."""
    return [int(token) for token, keep in zip(input_ids, attention_mask) if keep]


def _input_identity(feature, prefix, active_tokens, prompt_tokens, task):
    """Build an identity for the complete model input or a standalone prompt."""
    if prefix == "prompt_":
        return "token_prompt", _digest(active_tokens)

    # DPO without labels stores the prompt and completion separately. ORPO,
    # KTO and labelled alternatives already contain the complete model input.
    separate_completion = (
        task == "dpo"
        and prefix in {"chosen_", "rejected_"}
        and prefix + "labels" not in feature
    )
    full_input = active_tokens
    if separate_completion and prompt_tokens:
        full_input = prompt_tokens + active_tokens
    return "tokens", _digest(full_input)


def _supervised_prompt_identities(feature, prefix, input_ids, attention_mask):
    """Find contexts preceding supervised answers, including later chat turns."""
    labels = feature.get(prefix + "labels")
    if labels is not None:
        supervised = [label != -100 for label in labels]
    else:
        # TRL may defer label construction until collation. Its prepared
        # completion/assistant masks carry the same supervision boundaries.
        supervised = feature.get(
            prefix + "completion_mask", feature.get(prefix + "assistant_masks")
        )
    if supervised is None:
        return set()
    if len(supervised) != len(input_ids):
        raise ValueError("prepared supervision masks must match input_ids")

    assistant_mask = prefix + "assistant_masks" in feature
    if not assistant_mask and supervised and supervised[0]:
        return set()
    answer_starts = [
        index for index, keep in enumerate(supervised)
        if keep and index > 0 and not supervised[index - 1]
    ]
    if not assistant_mask:
        answer_starts = answer_starts[:1]

    identities = set()
    for start in answer_starts:
        prompt_tokens = _active_tokens(input_ids[:start], list(attention_mask)[:start])
        if prompt_tokens:
            identities.add(("token_prompt", _digest(prompt_tokens)))
    return identities


def _input_identities(features, task):
    identities = set()
    sequence_count = 0
    for feature in features:
        prompt_ids = list(feature.get("prompt_input_ids", []))
        prompt_mask = feature.get("prompt_attention_mask", [1] * len(prompt_ids))
        prompt_tokens = _active_tokens(prompt_ids, prompt_mask)

        for prefix in ("", "chosen_", "rejected_", "completion_", "prompt_"):
            input_ids = feature.get(prefix + "input_ids")
            if input_ids is None:
                continue
            input_ids = list(input_ids)
            attention_mask = feature.get(prefix + "attention_mask", [1] * len(input_ids))
            active_tokens = _active_tokens(input_ids, attention_mask)
            if active_tokens:
                identities.add(_input_identity(feature, prefix, active_tokens, prompt_tokens, task))
                sequence_count += 1
            identities.update(
                _supervised_prompt_identities(feature, prefix, input_ids, attention_mask)
            )
    return identities, sequence_count


def _near_duplicate_count(train_records, validation_records):
    """Reuse the existing approximate screen for advisory cross-split matches."""
    from .TrainingDataInspector import (
        DEFAULT_CONTENT_FIELDS,
        _content_simhash,
        _near_duplicate_pairs,
        _normalized_identity_text,
        _record_content,
    )

    def prepare(records):
        prepared = []
        for index, record in enumerate(records):
            text = _record_content(record, DEFAULT_CONTENT_FIELDS)
            normalized = _normalized_identity_text(text)
            if normalized:
                prepared.append({
                    "index": index,
                    "content": text,
                    "normalized_content": normalized,
                    "signature": _digest(normalized),
                    "simhash": _content_simhash(normalized),
                })
        return prepared

    count, _ = _near_duplicate_pairs(
        prepare(train_records), prepare(validation_records), .96, 0, False
    )
    return count


def _write_report(report, report_path, protected_paths):
    """Publish the report atomically without overwriting a protected input."""
    path = Path(report_path)
    if any(
        path.resolve() == Path(source).expanduser().resolve()
        for source in protected_paths if source
    ):
        raise ValueError("split audit report must not overwrite a source data file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report["report_path"] = str(path)


def audit_record_splits(train_records, validation_records, *, task="sft",
                        train_features=None, validation_features=None, report_path=None,
                        protected_paths=()):
    """Block shared cases/prompts/content and identical prepared model inputs.

    Input records may be datasets or lists. Reports contain counts and hashed
    examples, never raw prompts or IDs. Near-duplicate matching is advisory;
    a clean result cannot establish absence of benchmark contamination.
    """
    train_records = list(train_records)
    validation_records = list(validation_records)
    train_identities = _source_identities(train_records, task)
    validation_identities = _source_identities(validation_records, task)
    source_overlap = train_identities & validation_identities

    prepared_overlap = set()
    prepared_counts = [0, 0]
    if train_features is not None and validation_features is not None:
        train_inputs, prepared_counts[0] = _input_identities(train_features, task)
        validation_inputs, prepared_counts[1] = _input_identities(validation_features, task)
        prepared_overlap = train_inputs & validation_inputs

    near_count = _near_duplicate_count(train_records, validation_records)
    issues = []
    if source_overlap:
        issues.append(f"training and validation share {len(source_overlap)} source/group/prompt/content identity(s)")
    if prepared_overlap:
        issues.append(f"training and validation share {len(prepared_overlap)} prepared token identity(s)")
    report = {
        "schema_version": 1,
        "kind": "runtime_training_split_audit",
        "task": task,
        "passed": not issues,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "source_overlap": len(source_overlap),
        "prepared_overlap": len(prepared_overlap),
        "prepared_sequences_checked": prepared_counts,
        "near_duplicate_pairs": near_count,
        "near_duplicate_threshold": .96,
        "issues": issues,
        "warnings": ([f"{near_count} likely near-duplicate prompt pair(s); inspect the split"] if near_count else []),
        "examples": [
            {"kind": kind, "identity_sha256": _digest((kind, value))}
            for kind, value in sorted(source_overlap | prepared_overlap)[:20]
        ],
        "limitations": [
            "Near-duplicate screening is advisory, not a contamination guarantee.",
            "Prepared-token checks apply only to supplied tokenized features.",
        ],
    }
    if report_path is not None:
        _write_report(report, report_path, protected_paths)
    if issues:
        raise ValueError("Training/validation leakage detected: " + "; ".join(issues)
                         + (f". See {report_path}" if report_path else ""))
    return report
