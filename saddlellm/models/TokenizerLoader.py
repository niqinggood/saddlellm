"""Compatibility loading for standard and generic fast-tokenizer artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_tokenizer_compatible(
    model_path: str,
    *,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> Any:
    """Load with AutoTokenizer, falling back to a local tokenizer.json.

    ``transformers.TokenizersBackend`` can save a valid ``tokenizer.json``
    while recording a tokenizer class that ``AutoTokenizer`` cannot import.
    The fallback reconstructs that artifact as ``PreTrainedTokenizerFast``.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except (ValueError, ImportError, AttributeError) as original_error:
        local_path = Path(model_path).expanduser()
        tokenizer_file = local_path / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise original_error

        from transformers import PreTrainedTokenizerFast

        config_path = local_path / "tokenizer_config.json"
        config = {}
        if config_path.is_file():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                config = payload if isinstance(payload, dict) else {}
            except (OSError, json.JSONDecodeError):
                config = {}

        special_tokens = {}
        for name in (
            "unk_token",
            "bos_token",
            "eos_token",
            "pad_token",
            "sep_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens",
        ):
            value = config.get(name)
            if isinstance(value, dict):
                value = value.get("content")
            if value is not None:
                special_tokens[name] = value
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            **special_tokens,
        )
        for name in (
            "model_max_length",
            "padding_side",
            "truncation_side",
            "chat_template",
            "clean_up_tokenization_spaces",
        ):
            if name in config:
                setattr(tokenizer, name, config[name])
        return tokenizer
