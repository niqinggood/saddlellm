"""Dependency-free grouping of source records before training expansion.

Explicit record, case, conversation and domain IDs identify groups. Generic
source or dataset identifiers must not accidentally collapse a whole corpus.
"""

from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
import hashlib
import math
import random
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from .TrainingDataInspector import DEFAULT_GROUP_FIELDS, _record_groups

GROUP_FIELDS = DEFAULT_GROUP_FIELDS


def record_group_identities(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Use the same declared domain IDs in split, replay and runtime audits."""
    metadata = record.get("metadata")
    identities = set()
    for values in (record, metadata if isinstance(metadata, Mapping) else {}):
        identities.update(
            ("group", value) for value in _record_groups(values, GROUP_FIELDS)
        )
    return identities


def source_record_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    prompts: Sequence[str] | None = None,
) -> list[int]:
    """Join records sharing an explicit group or an optional normalized prompt.

    Each record without a shared identity is its own group. Prompt identities
    preserve case and merge Unicode/whitespace presentation differences only.
    Multiple declared identities are linked transitively within a record.
    """
    if prompts is not None and len(prompts) != len(records):
        raise ValueError("prompt identities must match source records")
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)

    owners: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        identities = list(record_group_identities(record))
        if prompts is not None:
            normalized = " ".join(unicodedata.normalize("NFKC", prompts[index]).split())
            if normalized:
                identities.append(
                    ("prompt", hashlib.sha256(normalized.encode("utf-8")).hexdigest())
                )
        for identity in identities:
            if identity in owners:
                join(index, owners[identity])
            else:
                owners[identity] = index
    return [find(index) for index in range(len(records))]


def split_group_indices(
    groups: Sequence[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split independent groups reproducibly while leaving training records."""
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(validation_fraction)
        or not 0 <= validation_fraction < 1
    ):
        raise ValueError("validation_split must be a finite number in [0, 1)")
    if validation_fraction == 0:
        return list(range(len(groups))), []
    unique_groups = list(dict.fromkeys(groups))
    if len(unique_groups) < 2:
        raise ValueError(
            "validation_split requires at least two independent groups of source "
            "records (conversations/cases/prompts)"
        )
    random.Random(seed).shuffle(unique_groups)
    validation_count = min(
        len(unique_groups) - 1,
        max(1, math.ceil(len(unique_groups) * validation_fraction)),
    )
    validation_groups = set(unique_groups[:validation_count])
    train = [
        index for index, group in enumerate(groups) if group not in validation_groups
    ]
    validation = [
        index for index, group in enumerate(groups) if group in validation_groups
    ]
    return train, validation
