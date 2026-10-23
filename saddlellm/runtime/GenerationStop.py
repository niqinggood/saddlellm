"""Request-local text stops with a conservative ByteLevel suffix fast path."""

from __future__ import annotations

from dataclasses import dataclass, field


def _byte_level_backend(tokenizer):
    """Only bypass full history for an unchanged, cleanup-free fast decoder.

    ByteLevel concatenates token bytes before UTF-8 decoding. Keeping four
    tokens per stop character, plus a UTF-8 boundary margin, is sufficient
    provided skipped/empty tokens do not count towards the look-behind.
    Other decoders and custom HF decode overrides retain their exact behavior.
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel
        from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
    except ImportError:
        return None
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return None
    if getattr(tokenizer, "clean_up_tokenization_spaces", True):
        return None
    if (
        getattr(tokenizer.decode, "__func__", None)
        is not PreTrainedTokenizerBase.decode
    ):
        return None
    if (
        getattr(tokenizer._decode, "__func__", None)
        is not PreTrainedTokenizerFast._decode
    ):
        return None
    backend = tokenizer.backend_tokenizer
    if not isinstance(backend, Tokenizer) or not isinstance(backend.decoder, ByteLevel):
        return None
    return backend


@dataclass
class _StopRow:
    stops: tuple[str, ...]
    incremental: bool
    window_tokens: int
    tail: list[int] = field(default_factory=list)
    matched: bool = False


class StopStringRows:
    """Check generated text without interpreting stop strings in the prompt.

    ``allow_incremental`` requires append-only rows with a fixed row order.
    Native generation satisfies that contract; arbitrary/custom/HF generation
    defaults to full-history checks. Even when enabled, unsupported decoders,
    cleanup and stops containing U+FFFD use the exact full-history path.
    """

    def __init__(
        self, tokenizer, prompt_tokens, stops_by_row, *, allow_incremental=False
    ):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self._backend = _byte_level_backend(tokenizer) if allow_incremental else None
        self._visible_ids: dict[int, bool] = {}
        if self._backend is not None:
            added_tokens = self._backend.get_added_tokens_decoder()
            self._visible_ids.update(
                {
                    token_id: False
                    for token_id, token in added_tokens.items()
                    if token.special or not token.content
                }
            )
        self.rows = []
        for stops in stops_by_row:
            stops = tuple(dict.fromkeys(item for item in stops if item))
            incremental = (
                bool(stops)
                and self._backend is not None
                and all("\ufffd" not in item for item in stops)
            )
            self.rows.append(
                _StopRow(
                    stops,
                    incremental,
                    4 * max((len(item) for item in stops), default=0) + 4,
                )
            )
        self._active = [index for index, row in enumerate(self.rows) if row.stops]
        self._all_incremental = bool(self._active) and all(
            self.rows[index].incremental for index in self._active
        )
        self._processed_tokens = 0

    def _decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _visible(self, token_id):
        visible = self._visible_ids.get(token_id)
        if visible is None:
            visible = bool(self._backend.id_to_token(token_id))
            self._visible_ids[token_id] = visible
        return visible

    def __call__(self, input_ids):
        import torch

        if input_ids.ndim != 2 or input_ids.shape[0] != len(self.rows):
            raise ValueError("stop checker requires a fixed batch of 2D token IDs")
        if not self._active:
            return torch.zeros(
                input_ids.shape[0], dtype=torch.bool, device=input_ids.device
            )
        count = max(0, input_ids.shape[1] - self.prompt_tokens)
        if count <= self._processed_tokens:
            # Repeated calls and shortened sequences are evaluated afresh.
            self._processed_tokens = 0
            for row in self.rows:
                row.tail.clear()
                row.matched = False
        start = self._processed_tokens if self._all_incremental else 0
        # One host transfer for all rows, not one transfer per stop or row.
        values = input_ids[:, self.prompt_tokens + start :].tolist()
        stopped = [False] * len(self.rows)
        for index in self._active:
            row, ids = self.rows[index], values[index]
            if row.incremental:
                if not row.matched:
                    new_ids = ids[self._processed_tokens - start :]
                    visible_ids = [token for token in new_ids if self._visible(token)]
                    if visible_ids:
                        decode_ids = row.tail + visible_ids
                        text = self._decode(decode_ids)
                        row.matched = any(item in text for item in row.stops)
                        row.tail = decode_ids[-row.window_tokens :]
                stopped[index] = row.matched
            else:
                # Generic decode may rewrite punctuation or incomplete UTF-8.
                text = self._decode(ids)
                stopped[index] = any(item in text for item in row.stops)
        self._processed_tokens = count
        return torch.tensor(stopped, dtype=torch.bool, device=input_ids.device)
