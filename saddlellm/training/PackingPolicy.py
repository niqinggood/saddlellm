"""Resolve packed-document attention before model loading or training."""

from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.


def resolve_packing_attention(mode, model_backend, pack_sequences=True):
    if mode not in {"auto", "concat", "isolated"}:
        raise ValueError("data.packing_attention must be auto, concat or isolated")
    if not pack_sequences:
        return "unpacked"
    if mode == "auto":
        return "isolated" if model_backend == "saddle" else "concat"
    if mode == "isolated" and model_backend != "saddle":
        raise ValueError(
            "isolated packed-document attention requires model.backend=saddle; "
            "the generic HF/Megatron trainer does not enforce document_ids"
        )
    return mode


def apply_packing_attention(dataset, mode):
    """Concat uses ordinary positions and causal attention within each block.

    Existing boundary label masks remain: they define the loss, independently
    from whether earlier documents are visible to attention.
    """
    if mode != "concat":
        return dataset
    columns = getattr(dataset, "column_names", None)
    removed = [
        name
        for name in ("document_ids", "position_ids")
        if columns is None or name in columns
    ]
    return dataset.remove_columns(removed) if removed else dataset
