"""Deterministic, training-only replay for supervised fine-tuning.

Replay is selected after validation is held out. Source identities are checked
before formatting; exact formatted/tokenized samples are checked again before
mixing. Ratios describe the final training sample mixture, not source records.
"""
from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
import hashlib
import json
import math
import os
import random
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from numbers import Real
from typing import Any

from .GroupSplit import record_group_identities
from .PostTrainingData import PostTrainingDataAdapter


def validate_replay_config(
    replay_data_path: str | None,
    replay_ratio: float = 0.0,
    replay_count: int | None = None,
) -> bool:
    """Validate without loading data, returning whether replay is enabled."""
    if (isinstance(replay_ratio, bool) or not isinstance(replay_ratio, Real)
            or not math.isfinite(replay_ratio) or not 0 <= replay_ratio < 1):
        raise ValueError("replay_ratio must be a finite fraction in [0, 1)")
    if replay_count is not None and (
        isinstance(replay_count, bool) or not isinstance(replay_count, int) or replay_count <= 0
    ):
        raise ValueError("replay_count must be a positive integer")
    if replay_ratio > 0 and replay_count is not None:
        raise ValueError("Use either replay_ratio or replay_count, not both")
    if replay_data_path is not None and (
        not isinstance(replay_data_path, (str, os.PathLike)) or not os.fspath(replay_data_path).strip()
    ):
        raise ValueError("replay_data_path must be a nonempty file path")
    requested = replay_ratio > 0 or replay_count is not None
    if requested and replay_data_path is None:
        raise ValueError("replay_data_path is required when requesting replay")
    if replay_data_path is not None and not requested:
        raise ValueError("replay_data_path requires a positive replay_ratio or replay_count")
    return requested


def _normalized_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sample_fingerprint(sample: Mapping[str, Any]) -> str:
    # Identical model inputs disclose the same held-out answer even if one
    # schema uses completion-only labels and another trains all tokens.
    if "input_ids" in sample:
        return _fingerprint({"input_ids": sample["input_ids"]})
    if "prompt" in sample and "completion" in sample:
        return _fingerprint({"text": str(sample["prompt"]) + str(sample["completion"])})
    if "text" in sample:
        return _fingerprint({"text": sample["text"]})
    return _fingerprint(dict(sample))


def _record_identities(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    identities = record_group_identities(record)
    normalized = PostTrainingDataAdapter.normalize_record(dict(record), task="sft", keep_metadata=False)
    if normalized is None:
        return identities
    if normalized.get("messages"):
        messages = [
            {"role": item["role"], "content": _normalized_text(item["content"])}
            for item in normalized["messages"]
        ]
        identities.add(("content", _fingerprint(messages)))
        # Match each assistant's complete context, so a replay conversation
        # cannot reintroduce a held-out turn through a different schema.
        for index, message in enumerate(messages):
            if message["role"] == "assistant" and index:
                identities.add(("prompt", _fingerprint(messages[:index])))
    elif normalized.get("text"):
        identities.add(("content", _fingerprint(_normalized_text(normalized["text"]))))
    return identities


def _identity_groups(identities: Sequence[set[tuple[str, str]]]) -> list[int]:
    parents = list(range(len(identities)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[tuple[str, str], int] = {}
    for index, values in enumerate(identities):
        for identity in values:
            if identity in owners:
                left, right = find(index), find(owners[identity])
                parents[max(left, right)] = min(left, right)
            else:
                owners[identity] = index
    return [find(index) for index in range(len(identities))]


def sft_source_groups(records: Sequence[Mapping[str, Any]]) -> list[int]:
    """Join shared SFT prompts, content and explicit source IDs transitively."""
    return _identity_groups([_record_identities(record) for record in records])


def filter_replay_records(
    replay_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Exclude all replay groups connected to held-out source identities.

    Group links are transitive across explicit IDs, prompts, and normalized
    content. Unknown schemas are dropped before expensive tokenization.
    """
    records = list(validation_records) + list(replay_records)
    identities = [_record_identities(record) for record in records]
    groups = _identity_groups(identities)
    valid = [any(kind in {"content", "prompt"} for kind, _ in values) for values in identities]
    held_out = set(groups[:len(validation_records)])
    selected = []
    excluded, invalid = 0, 0
    for offset, record in enumerate(replay_records, start=len(validation_records)):
        if not valid[offset]:
            invalid += 1
        elif groups[offset] in held_out:
            excluded += 1
        else:
            selected.append(dict(record))
    return selected, {
        "source_records": len(replay_records),
        "excluded_validation_records": excluded,
        "invalid_records": invalid,
        "eligible_source_records": len(selected),
    }


def mix_replay_samples(
    train_samples: Sequence[Mapping[str, Any]],
    replay_samples: Sequence[Mapping[str, Any]],
    *,
    replay_ratio: float = 0.0,
    replay_count: int | None = None,
    seed: int = 42,
    validation_samples: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mix an exact number of distinct, non-validation replay samples.

    ``ceil(n * ratio / (1 - ratio))`` rounds the target final-mixture ratio up
    to a whole sample. Insufficient eligible data is an error, never implicit
    resampling. Duplicate task samples are also removed from the replay pool.
    """
    validate_replay_config("<in-memory>", replay_ratio, replay_count)
    if not train_samples:
        raise ValueError("Replay requires at least one primary training sample")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Replay seed must be an integer")
    requested_count = replay_count
    if requested_count is None:
        # Decimal avoids ceil(3.0000000000000004) becoming four samples for
        # ordinary settings such as seven task samples and ratio=0.3.
        ratio = Decimal(str(replay_ratio))
        requested_count = int(
            (Decimal(len(train_samples)) * ratio / (1 - ratio)).to_integral_value(rounding=ROUND_CEILING)
        )
    validation_ids = {_sample_fingerprint(item) for item in validation_samples}
    training_ids = {_sample_fingerprint(item) for item in train_samples}
    seen = set()
    eligible = []
    excluded_validation = 0
    excluded_duplicates = 0
    for item in replay_samples:
        item = dict(item)
        identity = _sample_fingerprint(item)
        if identity in validation_ids:
            excluded_validation += 1
        elif identity in training_ids or identity in seen:
            excluded_duplicates += 1
        else:
            eligible.append(item)
            seen.add(identity)
    if requested_count > len(eligible):
        raise ValueError(
            f"Replay requested {requested_count} distinct samples but only {len(eligible)} "
            "remain after normalization, validation exclusion, and deduplication; "
            "supply more replay data or reduce replay_ratio/replay_count"
        )
    generator = random.Random(seed)
    chosen = generator.sample(eligible, requested_count)
    mixed = [dict(item) for item in train_samples] + chosen
    generator.shuffle(mixed)
    return mixed, {
        "enabled": True,
        "seed": seed,
        "primary_samples": len(train_samples),
        "replay_samples": len(chosen),
        "train_samples": len(mixed),
        "requested_ratio": float(replay_ratio),
        "requested_count": replay_count,
        "actual_ratio": len(chosen) / len(mixed),
        "eligible_samples": len(eligible),
        "excluded_validation_samples": excluded_validation,
        "excluded_duplicate_samples": excluded_duplicates,
        "selected_fingerprint": _fingerprint(chosen),
    }


def replay_source_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    """Stable content identity for recording which replay corpus was used."""
    return _fingerprint([dict(item) for item in records])


__all__ = [
    "filter_replay_records",
    "mix_replay_samples",
    "replay_source_fingerprint",
    "sft_source_groups",
    "validate_replay_config",
]
