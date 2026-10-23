"""Differentiable PyTorch KDA reference; no chunkwise/Triton performance claim."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .InferenceCache import RecurrentAttentionCache


def kda_recurrence(
    query,
    key,
    value,
    decay,
    beta,
    initial_state=None,
    *,
    active_mask=None,
    reset_mask=None,
):
    """Channel-gated delta rule from the Kimi Linear paper.

    q/k/v: [B,T,H,D]; decay: [B,T,H,Dk]; beta: [B,T,H]. Inputs q/k
    are expected to be L2-normalized by the caller. Matrix orientation is
    [B,H,Dk,Dv]. FP16/BF16 inputs accumulate in FP32; FP64 stays FP64.
    """
    if query.ndim != 4 or query.shape != key.shape or decay.shape != key.shape:
        raise ValueError("q, k and decay must have matching [B,T,H,Dk] shapes")
    if (
        value.ndim != 4
        or value.shape[:3] != query.shape[:3]
        or beta.shape != query.shape[:3]
    ):
        raise ValueError("value/beta batch, token and head dimensions must match q")
    batch, tokens, heads, dim = query.shape
    if tokens == 0:
        raise ValueError("KDA requires a nonempty sequence")
    if any(
        not tensor.is_floating_point() or tensor.device != query.device
        for tensor in (query, key, value, decay, beta)
    ):
        raise ValueError("KDA inputs must be floating tensors on the same device")
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    query, key, value, decay, beta = (
        tensor.to(dtype) for tensor in (query, key, value, decay, beta)
    )
    shape = (batch, heads, dim, value.shape[-1])
    if initial_state is None:
        state = query.new_zeros(shape)
    elif (
        initial_state.shape != shape
        or initial_state.device != query.device
        or initial_state.dtype != dtype
    ):
        raise ValueError("KDA state shape, device or accumulation dtype changed")
    else:
        state = initial_state
    for mask in (active_mask, reset_mask):
        if mask is not None and (
            mask.shape != (batch, tokens)
            or mask.device != query.device
            or mask.dtype != torch.bool
        ):
            raise ValueError("KDA masks must be boolean [B,T] on the input device")
    outputs = []
    for index in range(tokens):
        previous = state
        if reset_mask is not None:
            state = torch.where(
                reset_mask[:, index, None, None, None], torch.zeros_like(state), state
            )
        decayed = decay[:, index].unsqueeze(-1) * state
        predicted = (key[:, index].unsqueeze(-1) * decayed).sum(-2)
        error = beta[:, index].unsqueeze(-1) * (value[:, index] - predicted)
        updated = decayed + key[:, index].unsqueeze(-1) * error.unsqueeze(-2)
        output = (query[:, index].unsqueeze(-1) * updated).sum(-2)
        if active_mask is not None:
            active = active_mask[:, index]
            state = torch.where(active[:, None, None, None], updated, previous)
            output = torch.where(
                active[:, None, None], output, torch.zeros_like(output)
            )
        else:
            state = updated
        outputs.append(output)
    return torch.stack(outputs, dim=1), state


class SaddleKimiDeltaAttention(nn.Module):
    """KDA reference with short conv, channel decay, beta and output gating.

    This implements a sequential reference, not Kimi's optimized chunkwise
    kernels, pretrained weight format or full model. Padding is skipped;
    packed document boundaries reset BOTH matrix and short-conv history.
    Only 2D binary padding masks are supported. Cached execution is inference
    only and carries a fixed-size FP32 (FP64 for double) matrix per head.
    """

    def __init__(self, config):
        super().__init__()
        from .SaddleModeling import SaddleRMSNorm

        self.config = config
        self.heads = config.num_attention_heads
        if self.heads < 1:
            raise ValueError("KDA heads must be positive")
        self.dim = config.head_dim or config.hidden_size // self.heads
        self.inner = self.heads * self.dim
        self.conv_size = config.kda_conv_size
        rank = config.kda_gate_rank or self.dim
        if self.heads < 1 or self.dim < 1 or self.conv_size < 1 or rank < 1:
            raise ValueError(
                "KDA heads, head dimension, conv size and gate rank must be positive"
            )
        self.q_proj = nn.Linear(config.hidden_size, self.inner, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.inner, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.inner, bias=False)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.inner,
                    self.inner,
                    self.conv_size,
                    groups=self.inner,
                    bias=False,
                )
                for _ in range(3)
            ]
        )
        self.decay_down = nn.Linear(config.hidden_size, rank, bias=False)
        self.decay_up = nn.Linear(rank, self.inner, bias=False)
        self.beta_proj = nn.Linear(config.hidden_size, self.heads, bias=False)
        self.gate_down = nn.Linear(config.hidden_size, rank, bias=False)
        self.gate_up = nn.Linear(rank, self.inner, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.heads))
        self.dt_bias = nn.Parameter(torch.full((self.heads, self.dim), -2.0))
        self.output_norm = SaddleRMSNorm(self.dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(self.inner, config.hidden_size, bias=False)
        self.rotary = None  # KDA recurrence carries order; there is no RoPE.

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        document_ids=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        prepared_attention=None,
    ):
        del position_ids, prepared_attention
        batch, tokens, _ = hidden_states.shape
        if tokens < 1:
            raise ValueError("KDA requires a nonempty sequence")
        mutable = isinstance(past_key_value, RecurrentAttentionCache)
        if past_key_value is None:
            cached = {}
        elif mutable:
            cached = vars(past_key_value)
        elif (
            isinstance(past_key_value, dict)
            and past_key_value.get("cache_type") == "kda"
        ):
            cached = past_key_value
        else:
            raise ValueError("KDA requires a recurrent attention cache, not KV tensors")
        past_length = cached.get("length", 0)
        if (
            isinstance(past_length, bool)
            or not isinstance(past_length, int)
            or past_length < 0
        ):
            raise ValueError("invalid KDA cache length")
        if (use_cache or past_length > 0) and torch.is_grad_enabled():
            raise RuntimeError("KDA cached execution requires gradients disabled")
        if past_length and (
            cached.get("state") is None or cached.get("conv_states") is None
        ):
            raise ValueError("nonempty KDA caches require matrix and convolution state")
        if mutable and use_cache and past_length + tokens > past_key_value.max_length:
            raise ValueError("inference cache token budget exceeded")
        if attention_mask is None:
            active = torch.ones(
                (batch, tokens), dtype=torch.bool, device=hidden_states.device
            )
        else:
            if attention_mask.ndim != 2 or attention_mask.shape != (
                batch,
                past_length + tokens,
            ):
                raise ValueError(
                    "KDA supports only a 2D padding mask covering "
                    "past and current input"
                )
            if not bool(torch.all((attention_mask == 0) | (attention_mask == 1))):
                raise ValueError("KDA padding masks must contain only 0 or 1")
            active = attention_mask[:, -tokens:].to(
                device=hidden_states.device, dtype=torch.bool
            )
        last_doc = cached.get("last_document_id")
        has_doc = cached.get("has_document")
        if past_length and (document_ids is None) != (last_doc is None):
            raise ValueError(
                "KDA document segmentation cannot change while reusing a cache"
            )
        resets = torch.zeros_like(active)
        if document_ids is not None:
            if document_ids.shape != (batch, past_length + tokens):
                raise ValueError("KDA document_ids must cover past and current input")
            ids = document_ids[:, -tokens:].to(
                device=hidden_states.device, dtype=torch.long
            )
            if last_doc is None:
                last_doc = ids.new_zeros(batch)
                has_doc = torch.zeros(batch, dtype=torch.bool, device=ids.device)
            elif (
                last_doc.shape != (batch,)
                or has_doc is None
                or has_doc.shape != (batch,)
                or last_doc.device != ids.device
                or has_doc.device != ids.device
                or last_doc.dtype != torch.long
                or has_doc.dtype != torch.bool
            ):
                raise ValueError("invalid KDA document cache shape, dtype or device")
            reset_parts = []
            for index in range(tokens):
                valid = active[:, index]
                reset_parts.append(valid & has_doc & (last_doc != ids[:, index]))
                last_doc = torch.where(valid, ids[:, index], last_doc)
                has_doc = has_doc | valid
            resets = torch.stack(reset_parts, dim=1)
        projections = [
            projection(hidden_states)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        conv_states = cached.get("conv_states")
        if conv_states is None:
            conv_states = tuple(
                raw.new_zeros(batch, self.inner, self.conv_size - 1)
                for raw in projections
            )
        if len(conv_states) != 3:
            raise ValueError("KDA cache requires three short-convolution histories")
        convolved, next_conv = [], []
        for raw, conv, history in zip(projections, self.convs, conv_states):
            if (
                history.shape != (batch, self.inner, self.conv_size - 1)
                or history.dtype != raw.dtype
                or history.device != raw.device
            ):
                raise ValueError(
                    "KDA short-convolution cache shape, dtype or device changed"
                )
            pieces = []
            for index in range(tokens):
                previous = history
                history = torch.where(
                    resets[:, index, None, None], torch.zeros_like(history), history
                )
                window = torch.cat((history, raw[:, index].unsqueeze(-1)), dim=-1)
                result = (window * conv.weight[:, 0].to(raw.dtype).unsqueeze(0)).sum(-1)
                result = torch.where(
                    active[:, index, None], F.silu(result), torch.zeros_like(result)
                )
                history = torch.where(
                    active[:, index, None, None], window[:, :, 1:], previous
                )
                pieces.append(result)
            convolved.append(
                torch.stack(pieces, dim=1).reshape(batch, tokens, self.heads, self.dim)
            )
            next_conv.append(history)
        dtype = (
            torch.float64 if projections[0].dtype == torch.float64 else torch.float32
        )
        query, key, value = convolved
        query = F.normalize(query.to(dtype), dim=-1, eps=1e-6)
        key = F.normalize(key.to(dtype), dim=-1, eps=1e-6)
        decay_logits = (
            self.decay_up(self.decay_down(hidden_states))
            .reshape(batch, tokens, self.heads, self.dim)
            .to(dtype)
        )
        log_decay = -self.A_log.to(dtype).clamp(-20, 20).exp()[
            None, None, :, None
        ] * F.softplus(decay_logits + self.dt_bias.to(dtype))
        decay = log_decay.exp()
        beta = self.beta_proj(hidden_states).to(dtype).sigmoid()
        output, state = kda_recurrence(
            query,
            key,
            value,
            decay,
            beta,
            cached.get("state"),
            active_mask=active,
            reset_mask=resets,
        )
        gate = self.gate_up(self.gate_down(hidden_states)).sigmoid().reshape_as(output)
        output = self.output_norm(output.to(projections[0].dtype)) * gate.to(
            projections[0].dtype
        )
        output = self.o_proj(output.reshape(batch, tokens, self.inner))
        present = None
        if use_cache:
            values = dict(
                state=state,
                conv_states=tuple(next_conv),
                length=past_length + tokens,
                last_document_id=last_doc,
                has_document=has_doc,
            )
            if mutable:
                for name, value in values.items():
                    setattr(past_key_value, name, value)
                present = past_key_value
            else:
                present = dict(cache_type="kda", **values)
        return output, present
