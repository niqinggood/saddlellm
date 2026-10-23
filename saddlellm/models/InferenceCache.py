"""Request-local, amortized linear storage for native inference caches."""
from __future__ import annotations

import torch


class _IndependentLayerCache:
    """Internal decode-only view of request-owned KV; no history concatenation.

    ``length`` is only the largest rotary-table range. Attention must use each
    row's actual length and cache, never interpret it as a shared KV length.
    """

    def __init__(self, rows, positions):
        self.rows = tuple(rows)
        self.lengths = tuple(row.length for row in self.rows)
        self.positions = positions

    @property
    def length(self):
        return max(row.length for row in self.rows)


class RecurrentAttentionCache:
    """Request-private KDA matrix/short-convolution state, not a KV history.

    Length counts input slots (including padding) for alignment with the other
    layers. Only valid tokens update the recurrent state. Publication is owned
    by the attention module and requires gradients disabled.
    """

    def __init__(self, max_length: int):
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("cache max_length must be a positive integer")
        self.max_length = max_length
        self.length = 0
        self.state = None
        self.conv_states = None
        self.last_document_id = None
        self.has_document = None


class GenerationBuffers:
    """Fixed request-budget storage for token IDs and padding masks."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_new_tokens: int):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("generation inputs and mask must have matching 2D shapes")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        self.length = input_ids.shape[1]
        self.capacity = self.length + max_new_tokens
        shape = (input_ids.shape[0], self.capacity)
        self.tokens = input_ids.new_empty(shape)
        self.mask = attention_mask.new_empty(shape)
        self.tokens[:, :self.length].copy_(input_ids)
        self.mask[:, :self.length].copy_(attention_mask)

    def views(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[:, :self.length], self.mask[:, :self.length]

    def append(self, token: torch.Tensor, active: torch.Tensor) -> None:
        if self.length >= self.capacity:
            raise ValueError("generation token budget exceeded")
        if token.shape != (self.tokens.shape[0], 1):
            raise ValueError("next token must have shape [batch_size, 1]")
        if active.shape != (self.tokens.shape[0],):
            raise ValueError("active must have shape [batch_size]")
        self.tokens[:, self.length:self.length + 1].copy_(token)
        self.mask[:, self.length].copy_(active)
        self.length += 1

    def sequences(self) -> torch.Tensor:
        # A short response must not retain a large reserved request buffer.
        return self.tokens[:, :self.length].clone()


class InferenceLayerCache:
    """Append projected KV (or MLA latents) without copying history each step.

    Storage grows geometrically and never exceeds the request token budget.
    Only initialized slices are exposed to attention. A positive window_size
    selects a mirrored, fixed-capacity ring (2*W physical slots, W logical keys)
    instead of full-history geometric growth. This is inference-only mutable
    storage; a cache belongs to exactly one generation request.
    """

    def __init__(self, max_length: int, *, latent: bool = False, window_size: int = 0):
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("cache max_length must be a positive integer")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 0:
            raise ValueError("cache window_size must be a non-negative integer")
        self.max_length = max_length
        self.latent = latent
        self.window_size = window_size
        # length is the absolute input-slot count, NEVER the retained KV count.
        # This keeps RoPE and mixed global/window layers aligned after eviction.
        self.length = 0
        self.capacity = 0
        self.buffers: list[torch.Tensor] = []

    @property
    def stored_length(self) -> int:
        return min(self.length, self.window_size) if self.window_size else self.length

    def attention_layout(self, query_length: int) -> tuple[int, int]:
        """Return (visible key count, absolute first key) before appending.

        A multi-token chunk needs the previous W-1 keys plus ALL new keys:
        evicting to W before computing that chunk would corrupt early queries.
        """
        history = min(self.length, self.window_size - 1) if self.window_size else self.length
        return history + query_length, self.length - history

    def _window_views(self) -> tuple[torch.Tensor, ...]:
        dim = 1 if self.latent else 2
        start = (self.length - self.stored_length) % self.capacity
        return tuple(value.narrow(dim, start, self.stored_length) for value in self.buffers)

    def _append_window(self, values, dim, count, end):
        """Mirrored ring: O(1) single-token writes, contiguous chronological reads.

        Two physical copies of each slot avoid concatenating/rolling W keys at
        every decode step. Persistent tensor storage is at most 2*W slots;
        only W logical keys are retained. Prefill/chunk workspaces are transient.
        """
        if count < 1:
            raise ValueError("window cache requires at least one new token")
        context = values
        if count > 1 and self.length:
            history = min(self.length, self.window_size - 1)
            if history:
                context = tuple(
                    torch.cat((old.narrow(dim, self.stored_length - history, history), new), dim=dim)
                    for old, new in zip(self._window_views(), values)
                )
        if not self.buffers:
            capacity = min(self.window_size, self.max_length)
            buffers = []
            for value in values:
                shape = list(value.shape)
                shape[dim] = 2 * capacity
                buffers.append(value.new_empty(shape))
            self.buffers, self.capacity = buffers, capacity
        retained = min(count, self.capacity)
        start = (end - retained) % self.capacity
        first = min(retained, self.capacity - start)
        for buffer, value in zip(self.buffers, values):
            tail = value.narrow(dim, count - retained, retained)
            for mirror in (0, self.capacity):
                buffer.narrow(dim, mirror + start, first).copy_(tail.narrow(dim, 0, first))
                if first < retained:
                    buffer.narrow(dim, mirror, retained - first).copy_(
                        tail.narrow(dim, first, retained - first)
                    )
        self.length = end
        return self._window_views() if count == 1 else context

    def append(self, *values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("mutable inference caches require gradients disabled")
        dim = 1 if self.latent else 2
        expected = 1 if self.latent else 2
        if len(values) != expected:
            raise ValueError(f"cache requires {expected} tensors")
        if self.window_size:
            rank = 3 if self.latent else 4
            if any(not isinstance(value, torch.Tensor) or value.ndim != rank for value in values):
                raise ValueError(f"window cache tensors must have rank {rank}")
            if any(value.shape != values[0].shape or value.dtype != values[0].dtype
                   or value.device != values[0].device for value in values):
                raise ValueError("window cache tensors must share shape, device and dtype")
        count = values[0].shape[dim]
        end = self.length + count
        if end > self.max_length:
            raise ValueError("inference cache token budget exceeded")
        for index, value in enumerate(values):
            if value.shape[dim] != count:
                raise ValueError("cache tensors must have equal sequence lengths")
            if self.buffers:
                buffer = self.buffers[index]
                if (value.shape[:dim] + value.shape[dim + 1:] !=
                        buffer.shape[:dim] + buffer.shape[dim + 1:] or
                        value.device != buffer.device or
                        value.dtype != buffer.dtype):
                    raise ValueError(
                        "cache shape, device and dtype must remain unchanged"
                    )
        if self.window_size:
            return self._append_window(values, dim, count, end)
        if end > self.capacity:
            capacity = min(self.max_length, max(end, max(1, self.capacity) * 2))
            buffers = []
            for index, value in enumerate(values):
                shape = list(value.shape)
                shape[dim] = capacity
                buffer = value.new_empty(shape)
                if self.length:
                    buffer.narrow(dim, 0, self.length).copy_(
                        self.buffers[index].narrow(dim, 0, self.length)
                    )
                buffers.append(buffer)
            self.buffers = buffers
            self.capacity = capacity
        for buffer, value in zip(self.buffers, values):
            buffer.narrow(dim, self.length, count).copy_(value)
        self.length = end
        return tuple(buffer.narrow(dim, 0, end) for buffer in self.buffers)
