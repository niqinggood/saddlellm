"""Modular SaddleLLM model components.

This is the model layer: reusable blocks for attention, FFN/MoE, decoder
layers, and causal-LM objectives.  The goal is to make architecture research
explicit instead of hiding model ideas inside trainer code.
"""
import json
import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.utils import ModelOutput

from .AttentionResiduals import SaddleAttentionResidual
from .InferenceCache import GenerationBuffers, InferenceLayerCache, RecurrentAttentionCache, _IndependentLayerCache
from .KimiDeltaAttention import SaddleKimiDeltaAttention
from .NativeGeneration import NativeGenerationSession
from .NativeFusedKernels import can_fuse_norm, can_fuse_rotary


@dataclass
class SaddleModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    attention_kind: str = "gqa"  # mha | mqa | gqa | mla | kda
    attention_backend: str = "sdpa"  # auto | sdpa | flash | eager | xformers | transformer_engine
    qk_norm: bool = False
    qkv_bias: bool = False
    fused_qkv: bool = False
    mla_cache_mode: str = "kv"  # kv | latent
    ffn_kind: str = "swiglu"  # swiglu | moe
    fused_gate_up: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    expert_intermediate_size: int = 0
    shared_expert: bool = False
    aux_loss_free: bool = False
    router_aux_loss_coef: float = 0.001
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    sliding_window: int = 0
    kda_conv_size: int = 4
    kda_gate_rank: int = 0
    multi_token_prediction: bool = False
    mtp_extra_tokens: int = 0
    mtp_loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.3, 0.1, 0.05])
    # When positive, training with labels computes the LM projection and
    # cross-entropy in token chunks and omits logits by default.  Existing
    # checkpoints keep their historical output contract because zero disables
    # the fast path.  Callers that need logits (for example distillation) can
    # always pass ``return_logits=True`` to ``forward``.
    loss_chunk_size: int = 0
    initializer: str = "llama"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    head_dim: Optional[int] = None
    rope_scaling: Optional[Dict] = None
    residual_topology: str = "serial"
    residual_attention_scale: float = 1.0
    residual_ffn_scale: float = 1.0
    residual_learnable: bool = False
    residual_dropout: float = 0.0
    residual_initialization: str = "standard"
    residual_attention_mode: str = "none"
    residual_block_size: int = 4
    layer_name: str = ""
    layer_configs: List[Dict[str, Any]] = field(default_factory=list)
    model_blueprint: Optional[Dict[str, Any]] = None
    schema_version: int = 2
    # External architectures can require a different low-precision rounding
    # order. Defaults preserve historical native checkpoint behavior.
    rms_norm_weight_in_fp32: bool = True
    rope_compute_fp32: bool = False

    def __post_init__(self) -> None:
        if self.attention_kind not in {"mha", "mqa", "gqa", "mla", "kda"}:
            raise ValueError(f"unsupported attention_kind: {self.attention_kind!r}")
        if isinstance(self.sliding_window, bool) or not isinstance(self.sliding_window, int) or self.sliding_window < 0:
            raise ValueError("sliding_window must be a non-negative integer")
        if self.head_dim is not None and (
            isinstance(self.head_dim, bool) or not isinstance(self.head_dim, int) or self.head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive integer or None")
        if self.residual_attention_mode not in {"none", "full", "block"}:
            raise ValueError("residual_attention_mode must be none, full or block")
        if isinstance(self.residual_block_size, bool) or not isinstance(self.residual_block_size, int) or self.residual_block_size < 1:
            raise ValueError("residual_block_size must be a positive integer")
        if self.residual_attention_mode == "block" and (self.residual_block_size < 2 or self.residual_block_size % 2):
            raise ValueError("block AttnRes residual_block_size must be even and >= 2")
        if self.residual_attention_mode != "none" and self.residual_topology != "serial":
            raise ValueError("AttnRes requires serial residual topology")
        for name, minimum in (("kda_conv_size", 1), ("kda_gate_rank", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.attention_kind == "kda":
            if self.attention_backend not in {"eager", "auto"}:
                raise ValueError("KDA reference requires attention_backend eager or auto")
            if self.fused_qkv or self.qk_norm or self.sliding_window:
                raise ValueError("KDA does not support fused_qkv, qk_norm or sliding_window")
        for name in (
            "qk_norm", "qkv_bias", "fused_qkv", "fused_gate_up",
            "rms_norm_weight_in_fp32", "rope_compute_fp32",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.qkv_bias and (self.fused_qkv or self.attention_kind in {"mla", "kda"}):
            raise ValueError("qkv_bias requires unfused MHA/GQA/MQA projections")
        if isinstance(self.loss_chunk_size, bool) or not isinstance(
            self.loss_chunk_size, int
        ):
            raise TypeError("loss_chunk_size must be an integer")
        if self.loss_chunk_size < 0:
            raise ValueError("loss_chunk_size must be non-negative")
        self.layer_configs = [dict(item) for item in (self.layer_configs or [])]
        if self.layer_configs:
            # An explicit layer topology is authoritative, just as it is in
            # ModelBlueprint.  This also makes saved configs self-contained.
            self.num_hidden_layers = len(self.layer_configs)

    @staticmethod
    def _layer_blueprint_dict(layer) -> Dict[str, Any]:
        return {
            "name": layer.name,
            "attention": asdict(layer.attention),
            "ffn": asdict(layer.ffn),
            "residual": asdict(layer.residual),
        }

    @classmethod
    def from_blueprint(cls, blueprint) -> "SaddleModelConfig":
        expanded_layers = blueprint.expanded_layers()
        return cls(
            vocab_size=blueprint.vocab_size,
            hidden_size=blueprint.hidden_size,
            intermediate_size=blueprint.ffn.intermediate_size,
            num_hidden_layers=blueprint.num_layers,
            num_attention_heads=blueprint.attention.num_heads,
            num_key_value_heads=blueprint.attention.num_kv_heads or blueprint.attention.num_heads,
            max_position_embeddings=blueprint.max_position_embeddings,
            rope_theta=blueprint.attention.rope_theta,
            rms_norm_eps=blueprint.norm_eps,
            tie_word_embeddings=blueprint.tie_word_embeddings,
            attention_kind=blueprint.attention.kind,
            attention_backend=getattr(blueprint.attention, "backend", "sdpa"),
            qk_norm=getattr(blueprint.attention, "qk_norm", False),
            fused_qkv=getattr(blueprint.attention, "fused_qkv", False),
            mla_cache_mode=getattr(blueprint.attention, "mla_cache_mode", "kv"),
            ffn_kind=blueprint.ffn.kind,
            fused_gate_up=getattr(blueprint.ffn, "fused_gate_up", False),
            num_experts=blueprint.ffn.num_experts,
            num_experts_per_tok=blueprint.ffn.experts_per_token,
            expert_intermediate_size=blueprint.ffn.expert_intermediate_size,
            shared_expert=blueprint.ffn.shared_expert,
            aux_loss_free=blueprint.ffn.aux_loss_free,
            router_aux_loss_coef=blueprint.ffn.router_aux_loss_coef,
            q_lora_rank=blueprint.attention.q_lora_rank,
            kv_lora_rank=blueprint.attention.kv_lora_rank,
            sliding_window=blueprint.attention.sliding_window,
            kda_conv_size=blueprint.attention.kda_conv_size,
            kda_gate_rank=blueprint.attention.kda_gate_rank,
            multi_token_prediction=blueprint.objective.multi_token_prediction,
            mtp_extra_tokens=blueprint.objective.mtp_extra_tokens,
            initializer=blueprint.initializer,
            head_dim=blueprint.attention.head_dim,
            rope_scaling=blueprint.attention.rope_scaling,
            residual_topology=blueprint.residual.topology,
            residual_attention_scale=blueprint.residual.attention_scale,
            residual_ffn_scale=blueprint.residual.ffn_scale,
            residual_learnable=blueprint.residual.learnable,
            residual_dropout=blueprint.residual.dropout,
            residual_initialization=blueprint.residual.initialization,
            residual_attention_mode=blueprint.residual.attention_mode,
            residual_block_size=blueprint.residual.block_size,
            layer_configs=[cls._layer_blueprint_dict(layer) for layer in expanded_layers],
            model_blueprint=blueprint.to_config_dict(),
        )

    @classmethod
    def from_model_spec(cls, spec) -> "SaddleModelConfig":
        attention_kind = "mla" if spec.kv_lora_rank > 0 else ("gqa" if spec.num_kv_heads else "mha")
        ffn_kind = "moe" if spec.num_experts > 0 else "swiglu"
        return cls(
            vocab_size=spec.vocab_size,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_kv_heads or spec.num_attention_heads,
            max_position_embeddings=spec.max_position_embeddings,
            rope_theta=spec.rope_theta,
            rms_norm_eps=spec.norm_eps,
            tie_word_embeddings=spec.tie_word_embeddings,
            residual_initialization=getattr(
                spec, "residual_initialization", "standard"
            ),
            attention_kind=attention_kind,
            attention_backend="sdpa",
            qk_norm=getattr(spec, "qk_norm", False),
            fused_qkv=getattr(spec, "fused_qkv", False),
            mla_cache_mode="kv",
            ffn_kind=ffn_kind,
            fused_gate_up=getattr(spec, "fused_gate_up", False),
            num_experts=spec.num_experts,
            num_experts_per_tok=spec.num_experts_per_tok,
            expert_intermediate_size=spec.expert_intermediate_size,
            shared_expert=spec.shared_expert,
            aux_loss_free=spec.router_aux_loss_coef == 0 and spec.num_experts > 0,
            router_aux_loss_coef=spec.router_aux_loss_coef,
            q_lora_rank=spec.q_lora_rank,
            kv_lora_rank=spec.kv_lora_rank,
            sliding_window=spec.sliding_window,
            multi_token_prediction=spec.multi_token_prediction,
            mtp_extra_tokens=spec.num_mtp_layers,
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    def for_layer(self, index: int) -> "SaddleModelConfig":
        """Return a standalone config resolved for one decoder layer."""

        if not self.layer_configs:
            return replace(
                self,
                layer_name=f"layer-{index}",
                num_hidden_layers=1,
                layer_configs=[],
                model_blueprint=None,
            )
        if not 0 <= index < len(self.layer_configs):
            raise IndexError(f"decoder layer index out of range: {index}")
        layer = self.layer_configs[index]
        attention = dict(layer.get("attention") or {})
        ffn = dict(layer.get("ffn") or {})
        residual = dict(layer.get("residual") or {})
        overrides = {
            "layer_name": str(layer.get("name", f"layer-{index}")),
            "attention_kind": attention.get("kind", self.attention_kind),
            "attention_backend": attention.get("backend", self.attention_backend),
            "qk_norm": attention.get("qk_norm", self.qk_norm),
            "fused_qkv": attention.get("fused_qkv", self.fused_qkv),
            "mla_cache_mode": attention.get("mla_cache_mode", self.mla_cache_mode),
            "num_attention_heads": attention.get("num_heads", self.num_attention_heads),
            "num_key_value_heads": attention.get("num_kv_heads", self.num_key_value_heads),
            "head_dim": attention.get("head_dim", self.head_dim),
            "rope_theta": attention.get("rope_theta", self.rope_theta),
            "rope_scaling": attention.get("rope_scaling", self.rope_scaling),
            "sliding_window": attention.get("sliding_window", self.sliding_window),
            "kda_conv_size": attention.get("kda_conv_size", self.kda_conv_size),
            "kda_gate_rank": attention.get("kda_gate_rank", self.kda_gate_rank),
            "q_lora_rank": attention.get("q_lora_rank", self.q_lora_rank),
            "kv_lora_rank": attention.get("kv_lora_rank", self.kv_lora_rank),
            "ffn_kind": ffn.get("kind", self.ffn_kind),
            "fused_gate_up": ffn.get("fused_gate_up", self.fused_gate_up),
            "intermediate_size": ffn.get("intermediate_size", self.intermediate_size),
            "num_experts": ffn.get("num_experts", self.num_experts),
            "num_experts_per_tok": ffn.get("experts_per_token", self.num_experts_per_tok),
            "expert_intermediate_size": ffn.get(
                "expert_intermediate_size", self.expert_intermediate_size
            ),
            "shared_expert": ffn.get("shared_expert", self.shared_expert),
            "aux_loss_free": ffn.get("aux_loss_free", self.aux_loss_free),
            "router_aux_loss_coef": ffn.get(
                "router_aux_loss_coef", self.router_aux_loss_coef
            ),
            "residual_topology": residual.get("topology", self.residual_topology),
            "residual_attention_scale": residual.get(
                "attention_scale", self.residual_attention_scale
            ),
            "residual_ffn_scale": residual.get("ffn_scale", self.residual_ffn_scale),
            "residual_learnable": residual.get("learnable", self.residual_learnable),
            "residual_dropout": residual.get("dropout", self.residual_dropout),
            "residual_initialization": residual.get(
                "initialization", self.residual_initialization
            ),
            "residual_attention_mode": residual.get("attention_mode", self.residual_attention_mode),
            "residual_block_size": residual.get("block_size", self.residual_block_size),
        }
        if overrides["residual_attention_mode"] != self.residual_attention_mode or overrides["residual_block_size"] != self.residual_block_size:
            raise ValueError("all layers must share the model-level AttnRes mode and block size")
        return replace(
            self,
            **overrides,
            num_hidden_layers=1,
            layer_configs=[],
            model_blueprint=None,
        )

    def save_pretrained(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "saddle_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        if self.model_blueprint is not None:
            with open(os.path.join(path, "model_blueprint.json"), "w", encoding="utf-8") as f:
                json.dump(self.model_blueprint, f, ensure_ascii=False, indent=2)
        return path


@dataclass
class SaddleCausalLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Any]] = None
    hidden_states: Optional[torch.Tensor] = None
    expert_stats: Optional[List[Dict]] = None
    loss_items: Optional[Dict] = None


@dataclass
class SaddleGenerationOutput(ModelOutput):
    sequences: Optional[torch.Tensor] = None
    scores: Optional[Tuple[torch.Tensor, ...]] = None


def _rowwise_decode_linear(projection, inputs):
    """Preserve B1 Linear shapes for a low-precision batch decode call.

    CUDA GEMM algorithms may round differently at M=1 and M=B. Do not change
    weights, cast to FP32, mutate modules, or set process-wide backend flags.
    Attention/norm/activation/cache work can still run as a batch. The caller
    validates this inference-only, single-query path before changing any KV.
    """
    return torch.cat([projection(inputs[index:index + 1])
                      for index in range(inputs.shape[0])], dim=0)


class SaddleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, *, weight_in_fp32: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.weight_in_fp32 = weight_in_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = getattr(self, "_fused_decode", None)
        if fused is not None and not self.training:
            if can_fuse_norm(x, self.weight):
                return fused(x, self.weight, self.eps, self.weight_in_fp32)
        # Accumulating the variance in fp16/bf16 can overflow even when the
        # normalized result itself is perfectly representable.  Keep the
        # reduction (and the scale multiplication) in fp32, then restore the
        # activation dtype so this remains friendly to mixed-precision fused
        # kernels and residual connections.
        input_dtype = x.dtype
        compute_dtype = (
            torch.float32
            if not self.weight_in_fp32 or input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        normalized = x.to(compute_dtype)
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        if not self.weight_in_fp32:
            # Qwen3 restores activation precision BEFORE applying the scale.
            return self.weight * normalized.to(input_dtype)
        return (normalized * self.weight.to(compute_dtype)).to(input_dtype)


class SaddleRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        compute_fp32: bool = False,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = int(dim)
        self.base = float(base)
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = float(scaling_factor)
        self.compute_fp32 = compute_fp32
        self._cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_frozen = False
        self._cache_frozen_length: Optional[int] = None

    def reset_inv_freq(self, device) -> None:
        """Materialize constants after meta loading, without low-precision loss."""
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self.compute_fp32 and not self.inv_freq.is_meta:
            self.reset_inv_freq(self.inv_freq.device)
            self._cache.clear()
        return result

    def set_cache(
        self,
        cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Share a non-persistent RoPE cache with other compatible layers."""

        self._cache = cache

    def freeze_cache(self, max_length: int) -> None:
        """Disallow cache mutations, useful while compiling a fixed shape."""

        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._cache_frozen = True
        self._cache_frozen_length = int(max_length)

    def unfreeze_cache(self) -> None:
        self._cache_frozen = False
        self._cache_frozen_length = None

    def forward(self, seq_len: int, device=None, dtype=None, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len < 0 or offset < 0:
            raise ValueError("RoPE seq_len and offset must be non-negative")
        target_device = torch.device(device) if device is not None else self.inv_freq.device
        target_dtype = dtype or self.inv_freq.dtype
        cache_key = (
            self.dim,
            self.base,
            self.scaling_factor,
            self.compute_fp32,
            target_device.type,
            target_device.index,
            target_dtype,
        )
        required_length = offset + seq_len
        cached = self._cache.get(cache_key)
        if (
            self._cache_frozen
            and self._cache_frozen_length is not None
            and required_length > self._cache_frozen_length
        ):
            raise RuntimeError(
                "RoPE cache is frozen and cannot satisfy the requested range: "
                f"frozen_length={self._cache_frozen_length}, "
                f"required_length={required_length}, device={target_device}, "
                f"dtype={target_dtype}"
            )
        if cached is not None and cached[0].shape[0] >= required_length:
            return cached[0][offset:required_length], cached[1][offset:required_length]

        if self._cache_frozen:
            cached_length = 0 if cached is None else cached[0].shape[0]
            raise RuntimeError(
                "RoPE cache is frozen and cannot satisfy the requested range: "
                f"cached_length={cached_length}, required_length={required_length}, "
                f"device={target_device}, dtype={target_dtype}"
            )

        # Exact-size growth turns token-by-token decoding into repeated
        # arange/outer/sin/cos work with O(T^2) total cost.  Reserve a small
        # first block and double existing capacity so each cache key grows only
        # O(log T) times while retaining less than 2x capacity overhead.
        cached_length = 0 if cached is None else cached[0].shape[0]
        initial_capacity = max(
            16,
            min(max(1, int(self.max_position_embeddings)), 256),
        )
        cache_length = max(
            required_length,
            initial_capacity if cached_length == 0 else cached_length * 2,
        )

        # This cache is model-shared, not request-local. Constants created
        # inside inference_mode must still be usable by a later training
        # forward: autograd cannot save an inference tensor for backward.
        with torch.inference_mode(False), torch.no_grad():
            frequency_dtype = torch.float32 if self.compute_fp32 else self.inv_freq.dtype
            t = torch.arange(cache_length, device=target_device, dtype=frequency_dtype)
            if self.scaling_factor != 1.0:
                t = t / self.scaling_factor
            freqs = torch.outer(t, self.inv_freq.to(device=target_device, dtype=frequency_dtype))
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype=target_dtype)
            sin = emb.sin().to(dtype=target_dtype)
        self._cache[cache_key] = (cos, sin)
        return cos[offset:required_length], sin[offset:required_length]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, pack_decode: bool = False):
    # Cached contiguous positions are [T, D]; packed document positions are
    # gathered per sample and arrive as [B, T, D].
    cos = cos[None, None, :, :] if cos.ndim == 2 else cos[:, None, :, :]
    sin = sin[None, None, :, :] if sin.ndim == 2 else sin[:, None, :, :]
    if pack_decode and q.is_cuda and q.shape[-2] == 1 and not torch.is_grad_enabled():
        # Single-token decode is launch-bound: rotate Q/K together with the
        # SAME elementwise operations/rounding, not a differently shaped GEMM.
        # Restrict to decode so a long prefill never copies large Q/K tensors.
        joined = torch.cat((q, k), dim=1)
        rotated = (joined * cos) + (_rotate_half(joined) * sin)
        return rotated.split((q.shape[1], k.shape[1]), dim=1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def apply_rotary_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :] if cos.ndim == 2 else cos[:, None, :, :]
    sin = sin[None, None, :, :] if sin.ndim == 2 else sin[:, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


def _canonicalize_qkv_state_dict(module, state_dict, prefix, _local_metadata) -> None:
    """Save packed QKV parameters under the legacy logical projection keys."""

    packed_key = prefix + "qkv_packed_proj.weight"
    if packed_key not in state_dict:
        return
    packed = state_dict.pop(packed_key)
    q, k, v = packed.split(
        (
            module.q_projection_size,
            module.kv_projection_size,
            module.kv_projection_size,
        ),
        dim=0,
    )
    # Keep views into one shared storage.  ``torch.save`` preserves that
    # sharing, so canonical checkpoint names do not require a second full
    # model-sized allocation while the packed runtime parameter is resident.
    state_dict[prefix + "q_proj.weight"] = q
    state_dict[prefix + "k_proj.weight"] = k
    state_dict[prefix + "v_proj.weight"] = v


def _canonicalize_gate_up_state_dict(module, state_dict, prefix, _local_metadata) -> None:
    """Save packed SwiGLU parameters under the legacy logical keys."""

    packed_key = prefix + "gate_up_packed_proj.weight"
    if packed_key not in state_dict:
        return
    packed = state_dict.pop(packed_key)
    gate, up = packed.split(module.intermediate_size, dim=0)
    state_dict[prefix + "gate_proj.weight"] = gate
    state_dict[prefix + "up_proj.weight"] = up


class SaddleAttention(nn.Module):
    def __init__(self, config: SaddleModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        if config.head_dim is None and self.head_dim * self.num_heads != config.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("attention head_dim must be a positive even integer")

        self.q_projection_size = self.num_heads * self.head_dim
        self.kv_projection_size = self.num_kv_heads * self.head_dim
        use_mla = config.attention_kind == "mla" and config.kv_lora_rank > 0
        if use_mla and config.fused_qkv:
            raise ValueError(
                "fused_qkv is not supported for MLA; MLA uses latent projections"
            )
        self.qkv_packed_proj = None
        if use_mla:
            q_rank = config.q_lora_rank or max(1, config.hidden_size // 8)
            self.q_a_proj = nn.Linear(config.hidden_size, q_rank, bias=False)
            self.q_b_proj = nn.Linear(q_rank, self.q_projection_size, bias=False)
            self.kv_a_proj = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
            self.k_b_proj = nn.Linear(config.kv_lora_rank, self.kv_projection_size, bias=False)
            self.v_b_proj = nn.Linear(config.kv_lora_rank, self.kv_projection_size, bias=False)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        elif config.fused_qkv:
            self.qkv_packed_proj = nn.Linear(
                config.hidden_size,
                self.q_projection_size + 2 * self.kv_projection_size,
                bias=False,
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.q_proj = nn.Linear(
                config.hidden_size, self.q_projection_size, bias=config.qkv_bias
            )
            self.k_proj = nn.Linear(
                config.hidden_size, self.kv_projection_size, bias=config.qkv_bias
            )
            self.v_proj = nn.Linear(
                config.hidden_size, self.kv_projection_size, bias=config.qkv_bias
            )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        if config.qk_norm:
            self.q_norm = SaddleRMSNorm(
                self.head_dim, config.rms_norm_eps,
                weight_in_fp32=config.rms_norm_weight_in_fp32,
            )
            self.k_norm = SaddleRMSNorm(
                self.head_dim, config.rms_norm_eps,
                weight_in_fp32=config.rms_norm_weight_in_fp32,
            )
        else:
            # Identity keeps legacy checkpoint state dictionaries unchanged.
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.rotary = SaddleRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=(config.rope_scaling or {}).get("factor", 1.0),
            compute_fp32=config.rope_compute_fp32,
        )
        self.register_state_dict_post_hook(_canonicalize_qkv_state_dict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Translate packed and legacy projection layouts during strict load."""

        fused_key = prefix + "qkv_packed_proj.weight"
        legacy_keys = tuple(
            prefix + name for name in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
            )
        )
        if self.qkv_packed_proj is not None:
            if fused_key in state_dict and any(
                key in state_dict for key in legacy_keys
            ):
                error_msgs.append(
                    f"{prefix[:-1]}: checkpoint contains both packed and legacy "
                    "Q/K/V projection weights"
                )
            if fused_key not in state_dict and all(
                key in state_dict for key in legacy_keys
            ):
                try:
                    state_dict[fused_key] = torch.cat(
                        [state_dict[key] for key in legacy_keys], dim=0
                    )
                except RuntimeError as error:
                    error_msgs.append(
                        f"{prefix[:-1]}: cannot pack legacy Q/K/V weights: {error}"
                    )
            if fused_key in state_dict:
                for key in legacy_keys:
                    state_dict.pop(key, None)
        elif self.q_proj is not None and fused_key in state_dict:
            if any(key in state_dict for key in legacy_keys):
                error_msgs.append(
                    f"{prefix[:-1]}: checkpoint contains both packed and legacy "
                    "Q/K/V projection weights"
                )
            packed = state_dict.pop(fused_key)
            expected_rows = self.q_projection_size + 2 * self.kv_projection_size
            if packed.ndim != 2 or packed.shape[0] != expected_rows:
                error_msgs.append(
                    f"size mismatch for {fused_key}: expected first dimension "
                    f"{expected_rows}, got {tuple(packed.shape)}"
                )
            else:
                parts = packed.split(
                    (
                        self.q_projection_size,
                        self.kv_projection_size,
                        self.kv_projection_size,
                    ),
                    dim=0,
                )
                for key, value in zip(legacy_keys, parts):
                    state_dict.setdefault(key, value)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _document_position_ids(document_ids: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(document_ids.shape[1], device=document_ids.device).expand_as(document_ids)
        starts = torch.ones_like(document_ids, dtype=torch.bool)
        starts[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
        last_start = torch.cummax(torch.where(starts, indices, torch.zeros_like(indices)), dim=-1).values
        return indices - last_start

    def _rotary_for_position_ids(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather shared RoPE tables for reset positions in packed samples."""

        if position_ids.ndim != 2 or position_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                "position_ids must have shape [batch_size, sequence_length]"
            )
        position_ids = position_ids.to(device=hidden_states.device, dtype=torch.long)
        if position_ids.numel() == 0:
            empty = hidden_states.new_empty(
                (*position_ids.shape, self.head_dim)
            )
            return empty, empty
        cos, sin = self.rotary(
            cache_length,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return cos[position_ids], sin[position_ids]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        use_cache: bool = False,
        prepared_attention: Optional[
            Tuple[Optional[torch.Tensor], bool]
        ] = None,
        rowwise_linear: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        bsz, q_len, _ = hidden_states.shape
        if isinstance(past_key_value, _IndependentLayerCache) and (
                self.training or torch.is_grad_enabled() or not use_cache or q_len != 1
                or attention_mask is not None or document_ids is not None or position_ids is not None
                or self.config.attention_kind not in {"mha", "mqa", "gqa"}
                or self.config.sliding_window or len(past_key_value.rows) != bsz
                or tuple(row.length for row in past_key_value.rows) != past_key_value.lengths
                or (prepared_attention is not None and
                    (prepared_attention[0] is not None or prepared_attention[1]))):
            raise ValueError("independent KV attention requires single-query dense inference")
        use_mla = self.config.attention_kind == "mla" and self.config.kv_lora_rank > 0
        use_latent_cache = use_mla and self.config.mla_cache_mode == "latent"
        if isinstance(past_key_value, InferenceLayerCache):
            if not use_cache or past_key_value.latent != use_latent_cache:
                raise ValueError("inference cache mode must match attention and use_cache=True")
            if past_key_value.window_size and past_key_value.window_size != self.config.sliding_window:
                raise ValueError("inference cache window must match the attention layer")

        if use_latent_cache:
            # Packed documents reset RoPE positions. Latents are unrotated, so
            # reconstruct ALL key positions from the full document-id prefix.
            # Custom cached positions remain unsupported: they are not stored
            # in the compressed cache and cannot be recovered from latents.
            past_len = SaddleForCausalLM._past_key_value_length(past_key_value)
            document_positions = None
            if position_ids is not None:
                if document_ids is not None:
                    document_positions = self._document_position_ids(document_ids)
                    if use_cache and not torch.equal(position_ids, document_positions[:, -q_len:]):
                        raise ValueError("cached MLA document positions must follow document boundaries")
                elif past_len:
                    raise ValueError("explicit position_ids with an MLA latent cache are not supported")
            q = self.q_b_proj(self.q_a_proj(hidden_states))
            latent_kv = self.kv_a_proj(hidden_states)
            if isinstance(past_key_value, InferenceLayerCache):
                past_len = past_key_value.length
                latent_kv_all, = past_key_value.append(latent_kv)
            elif isinstance(past_key_value, dict) and past_key_value.get("cache_type") == "mla_latent":
                past_latent = past_key_value["latent_kv"]
                past_len = past_latent.shape[1]
                latent_kv_all = torch.cat([past_latent, latent_kv], dim=1)
            else:
                past_len = 0
                latent_kv_all = latent_kv
            kv_len = latent_kv_all.shape[1]
            key_start = past_len + q_len - kv_len
            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_b_proj(latent_kv_all).view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_b_proj(latent_kv_all).view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            if position_ids is None:
                cos_q, sin_q = self.rotary(q_len, device=hidden_states.device, dtype=hidden_states.dtype, offset=past_len)
                cos_k, sin_k = self.rotary(kv_len, device=hidden_states.device, dtype=hidden_states.dtype, offset=key_start)
                q = apply_rotary_single(q, cos_q, sin_q)
                k = apply_rotary_single(k, cos_k, sin_k)
            else:
                cos, sin = self._rotary_for_position_ids(
                    position_ids, hidden_states, past_len + q_len
                )
                q = apply_rotary_single(q, cos, sin)
                if past_len:
                    cos_table, sin_table = self.rotary(
                        past_len + q_len, device=hidden_states.device, dtype=hidden_states.dtype
                    )
                    key_positions = document_positions[:, key_start:key_start + kv_len]
                    k = apply_rotary_single(k, cos_table[key_positions], sin_table[key_positions])
                else:
                    k = apply_rotary_single(k, cos, sin)
            present_key_value = {"cache_type": "mla_latent", "latent_kv": latent_kv_all} if use_cache else None
            if isinstance(past_key_value, InferenceLayerCache):
                present_key_value = past_key_value
        else:
            past_len = (past_key_value.length if isinstance(past_key_value, (InferenceLayerCache, _IndependentLayerCache))
                        else past_key_value[0].shape[2] if past_key_value is not None else 0)
            if use_mla:
                q = self.q_b_proj(self.q_a_proj(hidden_states))
                latent_kv = self.kv_a_proj(hidden_states)
                k = self.k_b_proj(latent_kv)
                v = self.v_b_proj(latent_kv)
            elif self.qkv_packed_proj is not None:
                packed = (_rowwise_decode_linear(self.qkv_packed_proj, hidden_states)
                          if rowwise_linear else self.qkv_packed_proj(hidden_states))
                q, k, v = packed.split(
                    (
                        self.q_projection_size,
                        self.kv_projection_size,
                        self.kv_projection_size,
                    ),
                    dim=-1,
                )
            else:
                q = _rowwise_decode_linear(self.q_proj, hidden_states) if rowwise_linear else self.q_proj(hidden_states)
                k = _rowwise_decode_linear(self.k_proj, hidden_states) if rowwise_linear else self.k_proj(hidden_states)
                v = _rowwise_decode_linear(self.v_proj, hidden_states) if rowwise_linear else self.v_proj(hidden_states)

            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            if isinstance(past_key_value, _IndependentLayerCache):
                # Every query has its own absolute position. The shared table
                # is only a read-only lookup, not padded/mixed request history.
                cos_table, sin_table = self.rotary(past_len + 1, device=hidden_states.device,
                                                   dtype=hidden_states.dtype)
                cos = cos_table.index_select(0, past_key_value.positions).unsqueeze(1)
                sin = sin_table.index_select(0, past_key_value.positions).unsqueeze(1)
            elif position_ids is None:
                cos, sin = self.rotary(q_len, device=hidden_states.device, dtype=hidden_states.dtype, offset=past_len)
            else:
                cos, sin = self._rotary_for_position_ids(
                    position_ids, hidden_states, past_len + q_len
                )
            fused_rotary = getattr(self, "_fused_rotary_decode", None)
            if fused_rotary is not None and not self.training and can_fuse_rotary(q, k, cos, sin):
                q, k = fused_rotary(q, k, cos, sin)
            else:
                q, k = apply_rotary_pos_emb(
                    q, k, cos, sin, pack_decode=getattr(self, "_pack_decode_rotary", False),
                )

            if isinstance(past_key_value, _IndependentLayerCache):
                outputs = []
                for index, cache in enumerate(past_key_value.rows):
                    row_k, row_v = cache.append(k[index:index + 1], v[index:index + 1])
                    outputs.append(self._attention(q[index:index + 1], row_k, row_v,
                                                   attn_mask=None, is_causal=False))
                out = torch.cat(outputs, dim=0).transpose(1, 2).contiguous().view(
                    bsz, q_len, self.num_heads * self.head_dim)
                out = _rowwise_decode_linear(self.o_proj, out) if rowwise_linear else self.o_proj(out)
                return out, past_key_value
            if isinstance(past_key_value, InferenceLayerCache):
                k, v = past_key_value.append(k, v)
            elif past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            present_key_value = (k, v) if use_cache else None
            if isinstance(past_key_value, InferenceLayerCache):
                present_key_value = past_key_value
            kv_len = k.shape[2]
            key_start = past_len + q_len - kv_len

        if prepared_attention is None:
            attn_mask, is_causal = self._build_attention_mask(
                attention_mask=attention_mask,
                document_ids=document_ids,
                q_len=q_len,
                kv_len=kv_len,
                past_len=past_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                sliding_window=self.config.sliding_window,
                key_start=key_start,
            )
        else:
            attn_mask, is_causal = prepared_attention
        out = self._attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim
        )
        out = _rowwise_decode_linear(self.o_proj, out) if rowwise_linear else self.o_proj(out)
        return out, present_key_value

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        from .OperatorBackends import AttentionBackendRegistry

        return AttentionBackendRegistry.run(
            self.config.attention_backend,
            q,
            k,
            v,
            attn_mask,
            is_causal,
        )

    @staticmethod
    def _build_attention_mask(
        attention_mask: Optional[torch.Tensor],
        q_len: int,
        kv_len: int,
        past_len: int,
        dtype: torch.dtype,
        device: torch.device,
        document_ids: Optional[torch.Tensor] = None,
        sliding_window: int = 0,
        key_start: int = 0,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        if (attention_mask is None and document_ids is None and q_len == 1 and sliding_window
                and key_start + kv_len == past_len + q_len
                and (not sliding_window or kv_len <= sliding_window)):
            # A physically bounded window has no excluded/future keys at decode.
            return None, False
        if attention_mask is None and document_ids is None and not sliding_window:
            if past_len == 0:
                return None, True
            if q_len == 1:
                # A single appended query has no future cached keys.  Passing
                # is_causal=True with unequal Q/K lengths has backend-specific
                # alignment semantics, while an explicit all-zero mask only
                # wastes memory and blocks fused kernels.
                return None, False
        query_pos = torch.arange(past_len, past_len + q_len, device=device)[:, None]
        key_pos = torch.arange(key_start, key_start + kv_len, device=device)[None, :]
        blocked = key_pos > query_pos
        if sliding_window and sliding_window > 0:
            blocked = blocked | (key_pos <= query_pos - int(sliding_window))

        if document_ids is not None:
            if document_ids.ndim != 2 or document_ids.shape[-1] != past_len + q_len:
                raise ValueError(
                    "document_ids must have shape [batch_size, key_value_length]"
                )
            document_ids = document_ids.to(device=device)
            query_documents = document_ids[:, past_len : past_len + q_len]
            if query_documents.shape[-1] != q_len:
                raise ValueError("document_ids do not cover every query position")
            blocked = blocked[None, :, :] | (
                query_documents[:, :, None] != document_ids[:, None, key_start:key_start + kv_len]
            )

        if attention_mask is not None and attention_mask.shape[-1] != kv_len:
            if attention_mask.shape[-1] < kv_len:
                pad = torch.ones(
                    attention_mask.shape[0],
                    kv_len - attention_mask.shape[-1],
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([pad, attention_mask], dim=-1)
            else:
                attention_mask = attention_mask[:, -kv_len:]
        if attention_mask is not None:
            padding = attention_mask.to(device=device) == 0
            if blocked.ndim == 2:
                blocked = blocked[None, :, :]
            if blocked.shape[0] not in (1, padding.shape[0]):
                raise ValueError("attention_mask and document_ids batch sizes must match")
            blocked = blocked | padding[:, None, :]

        if blocked.ndim == 2:
            blocked = blocked[None, :, :]
        additive = torch.zeros(blocked.shape, device=device, dtype=dtype)
        additive.masked_fill_(blocked, torch.finfo(dtype).min)
        return additive[:, None, :, :], False


class SaddleSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        fused_gate_up: bool = False,
    ):
        super().__init__()
        self.intermediate_size = int(intermediate_size)
        self.gate_up_packed_proj = None
        if fused_gate_up:
            self.gate_up_packed_proj = nn.Linear(
                hidden_size, 2 * intermediate_size, bias=False
            )
            self.gate_proj = None
            self.up_proj = None
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.register_state_dict_post_hook(_canonicalize_gate_up_state_dict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Translate packed and legacy SwiGLU input projections on load."""

        packed_key = prefix + "gate_up_packed_proj.weight"
        legacy_keys = (
            prefix + "gate_proj.weight",
            prefix + "up_proj.weight",
        )
        if self.gate_up_packed_proj is not None:
            if packed_key in state_dict and any(
                key in state_dict for key in legacy_keys
            ):
                error_msgs.append(
                    f"{prefix[:-1]}: checkpoint contains both packed and legacy "
                    "gate/up projection weights"
                )
            if packed_key not in state_dict and all(
                key in state_dict for key in legacy_keys
            ):
                try:
                    state_dict[packed_key] = torch.cat(
                        [state_dict[key] for key in legacy_keys], dim=0
                    )
                except RuntimeError as error:
                    error_msgs.append(
                        f"{prefix[:-1]}: cannot pack legacy gate/up weights: {error}"
                    )
            if packed_key in state_dict:
                for key in legacy_keys:
                    state_dict.pop(key, None)
        elif self.gate_proj is not None and packed_key in state_dict:
            if any(key in state_dict for key in legacy_keys):
                error_msgs.append(
                    f"{prefix[:-1]}: checkpoint contains both packed and legacy "
                    "gate/up projection weights"
                )
            packed = state_dict.pop(packed_key)
            expected_rows = 2 * self.intermediate_size
            if packed.ndim != 2 or packed.shape[0] != expected_rows:
                error_msgs.append(
                    f"size mismatch for {packed_key}: expected first dimension "
                    f"{expected_rows}, got {tuple(packed.shape)}"
                )
            else:
                gate, up = packed.split(self.intermediate_size, dim=0)
                state_dict.setdefault(legacy_keys[0], gate)
                state_dict.setdefault(legacy_keys[1], up)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor, *, rowwise_linear: bool = False) -> torch.Tensor:
        if self.gate_up_packed_proj is not None:
            packed = (_rowwise_decode_linear(self.gate_up_packed_proj, x)
                      if rowwise_linear else self.gate_up_packed_proj(x))
            gate, up = packed.chunk(2, dim=-1)
        else:
            gate = _rowwise_decode_linear(self.gate_proj, x) if rowwise_linear else self.gate_proj(x)
            up = _rowwise_decode_linear(self.up_proj, x) if rowwise_linear else self.up_proj(x)
        activated = F.silu(gate) * up
        return (_rowwise_decode_linear(self.down_proj, activated)
                if rowwise_linear else self.down_proj(activated))


class SaddleMoE(nn.Module):
    def __init__(self, config: SaddleModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok or 2
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.aux_loss_free = config.aux_loss_free
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SaddleSwiGLU(
                config.hidden_size,
                config.expert_intermediate_size or config.intermediate_size,
                fused_gate_up=config.fused_gate_up,
            )
            for _ in range(config.num_experts)
        ])
        self.shared_expert = (
            SaddleSwiGLU(
                config.hidden_size,
                config.expert_intermediate_size or config.intermediate_size,
                fused_gate_up=config.fused_gate_up,
            )
            if config.shared_expert
            else None
        )
        self.register_buffer("router_bias", torch.zeros(config.num_experts), persistent=True)
        self.register_buffer("last_expert_load", torch.zeros(config.num_experts), persistent=False)
        self._checkpoint_recomputing = False

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        bsz, seq_len, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        logits = self.router(flat)
        # Router decisions and statistics are especially sensitive to BF16 /
        # FP16 ties and probability underflow.  Keep the small [tokens,
        # experts] tensors in FP32 while expert activations retain model dtype.
        router_logits = logits.float()
        routed_logits = (
            router_logits + self.router_bias.float()
            if self.aux_loss_free
            else router_logits
        )
        topk_scores, topk_indices = torch.topk(routed_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1).to(flat.dtype)
        router_probabilities = F.softmax(router_logits, dim=-1)

        output = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            matches = topk_indices == expert_idx
            token_mask = matches.any(dim=-1)
            token_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            # Empty Linear/SwiGLU batches are valid and produce zero parameter
            # gradients.  Running that empty graph avoids one tensor-to-Python
            # synchronization per expert while keeping DDP's used-parameter
            # set stable across ranks.
            expert_out = expert(flat[token_idx])
            weights = (topk_weights[token_idx] * matches[token_idx].to(topk_weights.dtype)).sum(dim=-1)
            output.index_add_(0, token_idx, expert_out * weights.unsqueeze(-1))

        if self.shared_expert is not None:
            output = output + self.shared_expert(flat)

        load = torch.bincount(
            topk_indices.reshape(-1), minlength=self.num_experts
        ).to(dtype=torch.float32)
        load = self._global_expert_load(load)
        self.last_expert_load = load.detach()
        aux_loss = (
            self._router_aux_loss(router_probabilities, load)
            if self.router_aux_loss_coef > 0
            else flat.new_zeros(())
        )
        if self.training and self.aux_loss_free and not self._checkpoint_recomputing:
            self._update_router_bias(load)
        stats = {
            "expert_load": load.detach(),
            "router_aux_loss": aux_loss.detach(),
            "router_entropy": self._router_entropy(router_probabilities).detach(),
            "load_balance": self._load_balance(load).detach(),
            "tokens_per_expert": load.detach(),
            "dropped_tokens": torch.zeros((), device=flat.device, dtype=flat.dtype),
        }
        return output.view(bsz, seq_len, hidden), aux_loss, stats

    def _global_expert_load(self, load: torch.Tensor) -> torch.Tensor:
        """Sum routing counts across DDP ranks when a process group is active."""

        if self._checkpoint_recomputing:
            # The original forward already synchronized this value.  Avoid a
            # redundant backward-time collective and preserve identical MoE
            # state for non-reentrant checkpoint recomputation.
            return self.last_expert_load.detach().clone()
        distributed = torch.distributed
        if not distributed.is_available() or not distributed.is_initialized():
            return load
        global_load = load.detach().clone()
        distributed.all_reduce(global_load, op=distributed.ReduceOp.SUM)
        return global_load

    @staticmethod
    def _global_router_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
        """Average differentiable router probabilities across active ranks."""

        distributed = torch.distributed
        if not distributed.is_available() or not distributed.is_initialized():
            return probabilities
        world_size = distributed.get_world_size()
        if world_size <= 1:
            return probabilities
        # all_reduce is not autograd-aware, so preserve the local gradient and
        # replace only the value with the global mean.
        global_value = probabilities.detach().clone()
        distributed.all_reduce(global_value, op=distributed.ReduceOp.SUM)
        global_value.div_(world_size)
        return probabilities + (global_value - probabilities.detach())

    def _router_aux_loss(
        self,
        router_probabilities: torch.Tensor,
        load: torch.Tensor,
    ) -> torch.Tensor:
        probs = router_probabilities.mean(dim=0)
        probs = self._global_router_probabilities(probs)
        load_fraction = load / load.sum().clamp_min(1.0)
        return self.router_aux_loss_coef * self.num_experts * torch.sum(
            probs * load_fraction
        )

    def _update_router_bias(self, load: torch.Tensor, rate: float = 1e-3):
        with torch.no_grad():
            avg = load.mean().clamp_min(1.0)
            self.router_bias -= rate * (load / avg - 1.0)

    def _router_entropy(self, router_probabilities: torch.Tensor) -> torch.Tensor:
        entropy = -(
            router_probabilities
            * torch.log(router_probabilities.clamp_min(1e-8))
        ).sum(dim=-1).mean()
        return entropy / math.log(max(2, self.num_experts))

    def _load_balance(self, load: torch.Tensor) -> torch.Tensor:
        probs = load / load.sum().clamp_min(1.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        return entropy / math.log(max(2, self.num_experts))


class SaddleDecoderLayer(nn.Module):
    def __init__(
        self,
        config: SaddleModelConfig,
        layer_index: Optional[int] = None,
        total_layers: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.total_layers = total_layers
        self.input_layernorm = SaddleRMSNorm(
            config.hidden_size, config.rms_norm_eps,
            weight_in_fp32=config.rms_norm_weight_in_fp32,
        )
        self.self_attn = SaddleKimiDeltaAttention(config) if config.attention_kind == "kda" else SaddleAttention(config)
        self.post_attention_layernorm = SaddleRMSNorm(
            config.hidden_size, config.rms_norm_eps,
            weight_in_fp32=config.rms_norm_weight_in_fp32,
        )
        if config.ffn_kind == "moe" and config.num_experts > 0:
            self.mlp = SaddleMoE(config)
            self.is_moe = True
        else:
            self.mlp = SaddleSwiGLU(
                config.hidden_size,
                config.intermediate_size,
                fused_gate_up=config.fused_gate_up,
            )
            self.is_moe = False
        if config.residual_learnable:
            self.attention_residual_scale = nn.Parameter(
                torch.tensor(float(config.residual_attention_scale))
            )
            self.ffn_residual_scale = nn.Parameter(
                torch.tensor(float(config.residual_ffn_scale))
            )
        else:
            # Plain attributes intentionally create no state-dict keys.  This
            # keeps legacy checkpoints strictly loadable with the defaults.
            self.attention_residual_scale = float(config.residual_attention_scale)
            self.ffn_residual_scale = float(config.residual_ffn_scale)
        self.residual_dropout = nn.Dropout(float(config.residual_dropout))
        self.attention_depth_residual = (
            SaddleAttentionResidual(config.hidden_size, config.rms_norm_eps)
            if config.residual_attention_mode != "none" else None
        )
        self.ffn_depth_residual = (
            SaddleAttentionResidual(config.hidden_size, config.rms_norm_eps)
            if config.residual_attention_mode != "none" else None
        )

    def forward_attention_residuals(self, sources, partial_block=None, **attention_kwargs):
        """Pure depth-state transition; no shared mutable history for checkpointing."""
        if self.attention_depth_residual is None:
            raise ValueError("enable Attention Residuals before supplying depth sources")
        sources = tuple(sources)
        attn_sources = sources + (() if partial_block is None else (partial_block,))
        attn_input = self.attention_depth_residual(*attn_sources)
        attn_out, present = self.self_attn(self.input_layernorm(attn_input), **attention_kwargs)
        attn_out = self.residual_dropout(attn_out)
        if not isinstance(self.attention_residual_scale, float) or self.attention_residual_scale != 1.0:
            attn_out = attn_out * self.attention_residual_scale
        if self.config.residual_attention_mode == "full":
            ffn_sources = sources + (attn_out,)
        else:
            partial_block = attn_out if partial_block is None else partial_block + attn_out
            ffn_sources = sources + (partial_block,)
        mlp_input = self.post_attention_layernorm(self.ffn_depth_residual(*ffn_sources))
        if self.is_moe:
            mlp_out, aux_loss, stats = self.mlp(mlp_input)
        else:
            mlp_out, aux_loss, stats = self.mlp(mlp_input), attn_input.new_zeros(()), {}
        mlp_out = self.residual_dropout(mlp_out)
        if not isinstance(self.ffn_residual_scale, float) or self.ffn_residual_scale != 1.0:
            mlp_out = mlp_out * self.ffn_residual_scale
        output = mlp_out if self.config.residual_attention_mode == "full" else partial_block + mlp_out
        return output, aux_loss, stats, present, attn_out, mlp_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        use_cache: bool = False,
        prepared_attention: Optional[
            Tuple[Optional[torch.Tensor], bool]
        ] = None,
        depth_sources: Optional[Tuple[torch.Tensor, ...]] = None,
        partial_block: Optional[torch.Tensor] = None,
        rowwise_linear: bool = False,
        _skip_aux_loss: bool = False,
    ):
        if self.attention_depth_residual is not None:
            if depth_sources is None:
                raise ValueError(
                    "AttnRes layers require depth source history; call the full "
                    "SaddleForCausalLM or provide depth_sources explicitly"
                )
            return self.forward_attention_residuals(
                depth_sources, partial_block=partial_block,
                attention_mask=attention_mask, document_ids=document_ids,
                position_ids=position_ids, past_key_value=past_key_value,
                use_cache=use_cache, prepared_attention=prepared_attention,
            )
        if depth_sources is not None or partial_block is not None:
            raise ValueError("depth sources require Attention Residuals enabled")
        residual = hidden_states
        attn_out, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            document_ids=document_ids,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            prepared_attention=prepared_attention,
            **({"rowwise_linear": True} if rowwise_linear else {}),
        )
        attn_out = self.residual_dropout(attn_out)
        # Multiplication by a plain 1.0 launches a useless CUDA kernel during
        # every decode step. Learnable tensor scales must retain their gradient.
        if (
            not isinstance(self.attention_residual_scale, float)
            or self.attention_residual_scale != 1.0
        ):
            attn_out = attn_out * self.attention_residual_scale
        if self.config.residual_topology == "parallel":
            mlp_in = self.post_attention_layernorm(residual)
        else:
            hidden_states = residual + attn_out
            mlp_in = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            mlp_out, aux_loss, stats = self.mlp(mlp_in)
        else:
            mlp_out = self.mlp(mlp_in, **({"rowwise_linear": True} if rowwise_linear else {}))
            aux_loss = None if _skip_aux_loss else residual.new_zeros(())
            stats = {}
        mlp_out = self.residual_dropout(mlp_out)
        if (
            not isinstance(self.ffn_residual_scale, float)
            or self.ffn_residual_scale != 1.0
        ):
            mlp_out = mlp_out * self.ffn_residual_scale
        if self.config.residual_topology == "parallel":
            hidden_states = residual + attn_out + mlp_out
        else:
            hidden_states = hidden_states + mlp_out
        return hidden_states, aux_loss, stats, present_key_value


class SaddleForCausalLM(nn.Module):
    # ``forward`` intentionally accepts extra generation/integration kwargs,
    # but the native loss is a per-microbatch mean and does not consume
    # Transformers' ``num_items_in_batch`` contract.  Declaring this explicitly
    # prevents Trainer from mistaking ``**kwargs`` support for token-summed loss
    # support and skipping gradient-accumulation scaling.
    accepts_loss_kwargs = False

    supports_gradient_checkpointing = True

    def __init__(self, config: SaddleModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs: Dict[str, Any] = {
            "use_reentrant": False,
        }
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        layer_configs = [
            config.for_layer(index) for index in range(config.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(
            [
                SaddleDecoderLayer(
                    layer_config,
                    layer_index=index,
                    total_layers=config.num_hidden_layers,
                )
                for index, layer_config in enumerate(layer_configs)
            ]
        )
        self._rope_cache: Dict[
            Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._rope_cache_limit: Optional[int] = None
        for layer in self.layers:
            if layer.self_attn.rotary is not None:
                layer.self_attn.rotary.set_cache(self._rope_cache)
        self.final_attention_residual = (
            SaddleAttentionResidual(config.hidden_size, config.rms_norm_eps)
            if config.residual_attention_mode != "none" else None
        )
        self.norm = SaddleRMSNorm(
            config.hidden_size, config.rms_norm_eps,
            weight_in_fp32=config.rms_norm_weight_in_fp32,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.mtp_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(max(0, config.mtp_extra_tokens))
        ])
        self.post_init()

    def train(self, mode: bool = True):
        if mode and getattr(self, "_saddle_cuda_int8", None) is not None:
            raise ValueError(
                "native INT8 is inference-only; reload original float weights for training"
            )
        return super().train(mode)

    def prewarm_rope_cache(
        self,
        seq_len: int,
        *,
        device: Optional[Any] = None,
        dtype: Optional[torch.dtype] = None,
        freeze: bool = False,
    ) -> "SaddleForCausalLM":
        """Populate every distinct RoPE table before a fixed-shape run.

        Call this after moving the model to its final device/dtype.  ``freeze``
        turns accidental shape, dtype, or device changes into a clear error
        instead of a graph-breaking Python-dict mutation.
        """

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        reference = self.embed_tokens.weight
        target_device = reference.device if device is None else torch.device(device)
        target_dtype = reference.dtype if dtype is None else dtype
        for layer in self.layers:
            if layer.self_attn.rotary is None:
                continue
            layer.self_attn.rotary.unfreeze_cache()
            layer.self_attn.rotary(
                int(seq_len),
                device=target_device,
                dtype=target_dtype,
            )
        self._rope_cache_limit = int(seq_len)
        if freeze:
            self.freeze_rope_cache()
        return self

    def freeze_rope_cache(
        self,
        max_length: Optional[int] = None,
    ) -> "SaddleForCausalLM":
        limit = self._rope_cache_limit if max_length is None else int(max_length)
        if not any(layer.self_attn.rotary is not None for layer in self.layers):
            return self  # pure recurrent models have no RoPE state to freeze
        if not self._rope_cache or limit is None:
            raise RuntimeError("prewarm_rope_cache must be called before freezing")
        for layer in self.layers:
            if layer.self_attn.rotary is not None:
                layer.self_attn.rotary.freeze_cache(limit)
        self._rope_cache_limit = limit
        return self

    def unfreeze_rope_cache(self) -> "SaddleForCausalLM":
        for layer in self.layers:
            if layer.self_attn.rotary is not None:
                layer.self_attn.rotary.unfreeze_cache()
        return self

    def clear_rope_cache(self) -> "SaddleForCausalLM":
        self._rope_cache.clear()
        self._rope_cache_limit = None
        return self.unfreeze_rope_cache()

    @classmethod
    def from_blueprint(cls, blueprint) -> "SaddleForCausalLM":
        return cls(SaddleModelConfig.from_blueprint(blueprint))

    @classmethod
    def from_model_spec(cls, spec) -> "SaddleForCausalLM":
        return cls(SaddleModelConfig.from_model_spec(spec))

    def post_init(self):
        std = 0.02 if self.config.initializer != "deepseek" else 0.006
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.config.hidden_size))
        depth_scale = 1.0 / math.sqrt(2.0 * max(1, len(self.layers)))
        with torch.no_grad():
            for layer in self.layers:
                policy = layer.config.residual_initialization
                if policy == "standard":
                    continue
                if policy not in {"depth_scaled", "zero"}:
                    raise ValueError(f"unsupported residual initialization: {policy!r}")
                for projection in self._residual_output_projections(layer):
                    if policy == "zero":
                        projection.weight.zero_()
                    else:
                        projection.weight.mul_(depth_scale)

    @staticmethod
    def _residual_output_projections(layer: SaddleDecoderLayer) -> List[nn.Linear]:
        projections = [layer.self_attn.o_proj]
        if layer.is_moe:
            projections.extend(expert.down_proj for expert in layer.mlp.experts)
            if layer.mlp.shared_expert is not None:
                projections.append(layer.mlp.shared_expert.down_proj)
        else:
            projections.append(layer.mlp.down_proj)
        return projections

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        return_logits: Optional[bool] = None,
        logits_to_keep: int = 0,
        _decode_rowwise_linear: bool = False,
        _prefill_only: bool = False,
        _generation_only: bool = False,
        **_,
    ):
        if not isinstance(_generation_only, bool):
            raise TypeError("generation-only flag must be boolean")
        if _generation_only and (self.training or torch.is_grad_enabled() or labels is not None):
            raise ValueError("generation-only forward requires eval, no gradients and no labels")
        if not isinstance(_prefill_only, bool):
            raise TypeError("_prefill_only must be boolean")
        if _prefill_only and (self.training or torch.is_grad_enabled() or labels is not None
                              or not use_cache or return_logits is True):
            raise ValueError("prefill-only forward requires eval, no gradients, cache and no labels/logits")
        if isinstance(logits_to_keep, bool) or not isinstance(logits_to_keep, int):
            raise TypeError("logits_to_keep must be a non-negative integer")
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a non-negative integer")
        if labels is not None and logits_to_keep:
            raise ValueError("logits_to_keep must be zero when labels are supplied")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        # Packed pretraining batches commonly carry an all-one 2D mask.  It
        # conveys no padding information, but retaining it would make every
        # decoder layer allocate a dense QxK additive mask and would disable
        # SDPA's causal fast path.  Collapse it once before entering layers.
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and attention_mask.numel() > 0
            and not torch.compiler.is_compiling()
            and bool(torch.all(attention_mask == 1))
        ):
            attention_mask = None
        checkpointing = bool(self.gradient_checkpointing and self.training)
        if checkpointing and use_cache:
            raise ValueError(
                "use_cache=True is incompatible with activation checkpointing "
                "during training; call with use_cache=False or disable gradient checkpointing"
            )
        aux_losses = []
        expert_stats = []
        next_past_key_values = [] if use_cache else None
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        elif len(past_key_values) != len(self.layers):
            raise ValueError(
                "past_key_values must contain exactly one entry per decoder layer"
            )
        batch_size, query_length = hidden_states.shape[:2]
        independent_rows = any(isinstance(cache, _IndependentLayerCache) for cache in past_key_values)
        if independent_rows:
            if (self.training or torch.is_grad_enabled() or labels is not None or not use_cache
                    or query_length != 1 or attention_mask is not None or document_ids is not None
                    or position_ids is not None or _prefill_only
                    or self.config.residual_attention_mode != "none"
                    or any(layer.config.attention_kind not in {"mha", "mqa", "gqa"}
                           or layer.config.sliding_window or layer.config.num_experts
                           or layer.config.residual_attention_mode != "none" for layer in self.layers)
                    or any(not isinstance(cache, _IndependentLayerCache) or len(cache.rows) != batch_size
                           or tuple(row.length for row in cache.rows) != cache.lengths
                           or cache.positions.shape != (batch_size,)
                           or cache.positions.device != hidden_states.device or cache.positions.dtype != torch.long
                           or any(not isinstance(row, InferenceLayerCache) or row.latent or row.window_size
                                  or row.length < 1 or len(row.buffers) != 2 for row in cache.rows)
                           for cache in past_key_values)):
                raise ValueError("independent KV decode requires unpadded, single-query dense cached inference")
        first_past_length = self._past_key_value_length(past_key_values[0])
        if _decode_rowwise_linear:
            if (self.training or torch.is_grad_enabled() or labels is not None
                    or not use_cache or query_length != 1 or first_past_length < 1
                    or hidden_states.dtype != torch.bfloat16 or hidden_states.device.type != "cuda"
                    or self.config.residual_attention_mode != "none"
                    or any(layer.config.attention_kind not in {"mha", "mqa", "gqa"}
                           or layer.config.num_experts or layer.config.sliding_window
                           or layer.config.residual_attention_mode != "none" for layer in self.layers)):
                raise ValueError("rowwise Linear decode requires cached CUDA BF16 dense inference")
        for layer, cache in zip(self.layers, past_key_values):
            if isinstance(cache, InferenceLayerCache) and cache.window_size:
                if not use_cache or cache.window_size != layer.config.sliding_window:
                    raise ValueError("inference cache window must match the attention layer and use_cache=True")
        if (any(layer.config.attention_kind == "kda" for layer in self.layers)
                or any(isinstance(cache, InferenceLayerCache) and cache.window_size for cache in past_key_values)) and any(
            self._past_key_value_length(cache) != first_past_length for cache in past_key_values
        ):
            raise ValueError("hybrid caches must have equal input-slot lengths in every layer")
        context_length = first_past_length + query_length
        if document_ids is not None:
            if document_ids.ndim != 2 or document_ids.shape != (
                batch_size,
                context_length,
            ):
                raise ValueError(
                    "document_ids must have shape [batch_size, key_value_length]"
                )
            document_ids = document_ids.to(
                device=hidden_states.device,
                dtype=torch.long,
            )
            if position_ids is None:
                position_ids = SaddleAttention._document_position_ids(document_ids)[:, -query_length:]
        if position_ids is not None:
            if position_ids.ndim != 2 or position_ids.shape != (
                batch_size,
                query_length,
            ):
                raise ValueError(
                    "position_ids must have shape [batch_size, sequence_length]"
                )
            position_ids = position_ids.to(
                device=hidden_states.device,
                dtype=torch.long,
            )
            if position_ids.numel() and not torch.compiler.is_compiling():
                min_position, max_position = torch.aminmax(position_ids)
                if int(min_position.item()) < 0 or int(max_position.item()) >= context_length:
                    raise ValueError(
                        "position_ids must be within the current key/value context"
                    )
        if use_cache and position_ids is not None and any(
            layer.config.attention_kind == "mla" and layer.config.mla_cache_mode == "latent"
            for layer in self.layers
        ):
            if document_ids is None and first_past_length:
                raise ValueError("explicit position_ids with an MLA latent cache are not supported")
            if document_ids is not None and not torch.equal(
                position_ids, SaddleAttention._document_position_ids(document_ids)[:, -query_length:]
            ):
                raise ValueError("cached MLA document positions must follow document boundaries")
        prepared_attention_masks: Dict[
            Tuple[int, int, int, int], Tuple[Optional[torch.Tensor], bool]
        ] = {}
        depth_mode = self.config.residual_attention_mode
        depth_sources = (hidden_states,) if depth_mode != "none" else ()
        partial_block = None
        for layer_index, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if depth_mode == "block" and layer_index > 0 and (2 * layer_index) % self.config.residual_block_size == 0:
                depth_sources = depth_sources + (partial_block,)
                partial_block = None
            past_len = self._past_key_value_length(past_key_value)
            q_len = hidden_states.shape[1]
            kv_len, key_start = (past_key_value.attention_layout(q_len)
                                 if isinstance(past_key_value, InferenceLayerCache)
                                 else (past_len + q_len, 0))
            mask_key = (q_len, past_len, int(layer.config.sliding_window), key_start)
            if layer.config.attention_kind != "kda" and mask_key not in prepared_attention_masks:
                prepared_attention_masks[mask_key] = (
                    SaddleAttention._build_attention_mask(
                        attention_mask=attention_mask,
                        document_ids=document_ids,
                        q_len=q_len,
                        kv_len=kv_len,
                        past_len=past_len,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                        sliding_window=layer.config.sliding_window,
                        key_start=key_start,
                    )
                )
            prepared_attention = None if layer.config.attention_kind == "kda" else prepared_attention_masks[mask_key]
            if checkpointing:
                checkpoint_kwargs = dict(self._gradient_checkpointing_kwargs)
                if layer.is_moe:
                    checkpoint_kwargs["context_fn"] = (
                        lambda current_layer=layer: self._moe_checkpoint_contexts(
                            current_layer.mlp
                        )
                    )

                def custom_forward(
                    states: torch.Tensor,
                    *sources: torch.Tensor,
                    current_layer: SaddleDecoderLayer = layer,
                    current_past_key_value: Any = past_key_value,
                    current_prepared_attention: Any = prepared_attention,
                    current_has_partial: bool = partial_block is not None,
                ):
                    if depth_mode != "none":
                        return current_layer(
                            states, depth_sources=sources,
                            partial_block=states if current_has_partial else None,
                            attention_mask=attention_mask,
                            document_ids=document_ids,
                            position_ids=position_ids,
                            past_key_value=current_past_key_value,
                            use_cache=False,
                            prepared_attention=current_prepared_attention,
                        )
                    return current_layer(
                        states,
                        attention_mask=attention_mask,
                        document_ids=document_ids,
                        position_ids=position_ids,
                        past_key_value=current_past_key_value,
                        use_cache=False,
                        prepared_attention=current_prepared_attention,
                    )

                layer_output = checkpoint(
                    custom_forward,
                    hidden_states,
                    *depth_sources,
                    **checkpoint_kwargs,
                )
            else:
                if depth_mode != "none":
                    layer_output = layer(
                        hidden_states, depth_sources=depth_sources, partial_block=partial_block,
                        attention_mask=attention_mask, document_ids=document_ids,
                        position_ids=position_ids, past_key_value=past_key_value,
                        use_cache=use_cache, prepared_attention=prepared_attention,
                    )
                else:
                    layer_output = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        document_ids=document_ids,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        prepared_attention=prepared_attention,
                        **({"rowwise_linear": True} if _decode_rowwise_linear else {}),
                        **({"_skip_aux_loss": True} if _generation_only
                           and type(layer) is SaddleDecoderLayer and not layer.is_moe
                           and getattr(layer.forward, "__func__", None) is SaddleDecoderLayer.forward
                           and not layer._forward_hooks and not layer._forward_pre_hooks
                           and not nn.modules.module._global_forward_hooks
                           and not nn.modules.module._global_forward_pre_hooks else {}),
                    )
            hidden_states, aux_loss, stats, present_key_value = layer_output[:4]
            if depth_mode == "full":
                depth_sources = depth_sources + (layer_output[4], layer_output[5])
            elif depth_mode == "block":
                partial_block = hidden_states
            aux_losses.append(aux_loss)
            if stats:
                expert_stats.append(stats)
            if use_cache:
                next_past_key_values.append(present_key_value)
        if depth_mode != "none":
            readout_sources = depth_sources + (() if partial_block is None else (partial_block,))
            hidden_states = self.final_attention_residual(*readout_sources)
        # The public forward keeps full normalized hidden states. Generation
        # consumes only logits/KV, so a built-in, unhooked norm can operate on
        # the selected positions, or be omitted for a cache-only prompt chunk.
        trim_readout = (
            _generation_only and type(self.norm) is SaddleRMSNorm
            and getattr(self.norm.forward, "__func__", None) is SaddleRMSNorm.forward
            and not self.norm._forward_hooks and not self.norm._forward_pre_hooks
            and not nn.modules.module._global_forward_hooks
            and not nn.modules.module._global_forward_pre_hooks
        )
        if trim_readout and logits_to_keep and not _prefill_only:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        if not (trim_readout and _prefill_only):
            hidden_states = self.norm(hidden_states)

        configured_chunked_loss = bool(
            labels is not None
            and self.training
            and self.config.loss_chunk_size > 0
        )
        if _prefill_only:
            return_logits = False
        elif return_logits is None:
            # Preserve the historical output contract unless the checkpoint
            # explicitly opts into the memory-saving training fast path.
            return_logits = not configured_chunked_loss
        elif not isinstance(return_logits, bool):
            raise TypeError("return_logits must be a boolean or None")
        if not return_logits and labels is None and not _prefill_only:
            raise ValueError("return_logits=False requires labels")

        projection_states = (
            hidden_states[:, -logits_to_keep:, :] if logits_to_keep else hidden_states
        )
        logits = ((_rowwise_decode_linear(self.lm_head, projection_states)
                   if _decode_rowwise_linear else self.lm_head(projection_states)) if return_logits else None)

        loss = None
        loss_items = {}
        if labels is not None:
            if logits is None:
                chunk_size = self.config.loss_chunk_size or 1024
                lm_loss = self._chunked_causal_lm_loss(
                    hidden_states,
                    labels,
                    self.lm_head.weight,
                    chunk_size=chunk_size,
                )
            else:
                lm_loss = self._causal_lm_loss(logits, labels)
            aux_loss = (
                torch.stack(aux_losses).sum()
                if aux_losses
                else hidden_states.new_zeros(())
            )
            loss = lm_loss + aux_loss
            loss_items = {"lm_loss": lm_loss.detach(), "aux_loss": aux_loss.detach()}
            if self.config.multi_token_prediction and self.mtp_heads:
                mtp_loss, mtp_items = self._mtp_loss(
                    hidden_states,
                    labels,
                    chunk_size=(
                        (self.config.loss_chunk_size or 1024)
                        if logits is None
                        else None
                    ),
                )
                loss = loss + mtp_loss
                loss_items.update(mtp_items)

        output = SaddleCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=next_past_key_values,
            hidden_states=None if _generation_only else hidden_states,
            expert_stats=expert_stats,
            loss_items=loss_items,
        )
        if return_dict:
            return output
        return (loss, logits) if loss is not None else (logits,)

    @staticmethod
    def _past_key_value_length(past_key_value: Optional[Any]) -> int:
        if past_key_value is None:
            return 0
        if isinstance(past_key_value, (InferenceLayerCache, RecurrentAttentionCache, _IndependentLayerCache)):
            return past_key_value.length
        if isinstance(past_key_value, dict):
            if past_key_value.get("cache_type") == "kda":
                length = past_key_value.get("length")
                if isinstance(length, bool) or not isinstance(length, int) or length < 0:
                    raise ValueError("invalid KDA cache length")
                return length
            latent = past_key_value.get("latent_kv")
            if latent is None:
                raise ValueError("MLA latent cache is missing latent_kv")
            return int(latent.shape[1])
        if len(past_key_value) != 2:
            raise ValueError("past_key_value must contain key and value tensors")
        return int(past_key_value[0].shape[2])

    @staticmethod
    def _moe_checkpoint_contexts(moe: SaddleMoE):
        router_bias_snapshot = moe.router_bias.detach().clone()
        return (
            nullcontext(),
            SaddleForCausalLM._moe_checkpoint_recompute_context(
                moe, router_bias_snapshot
            ),
        )

    @staticmethod
    @contextmanager
    def _moe_checkpoint_recompute_context(
        moe: SaddleMoE,
        router_bias_snapshot: torch.Tensor,
    ):
        previous = moe._checkpoint_recomputing
        current_router_bias = moe.router_bias.detach().clone()
        current_expert_load = moe.last_expert_load.detach().clone()
        with torch.no_grad():
            moe.router_bias.copy_(router_bias_snapshot)
        moe._checkpoint_recomputing = True
        try:
            yield
        finally:
            with torch.no_grad():
                moe.router_bias.copy_(current_router_bias)
                moe.last_expert_load.copy_(current_expert_load)
            moe._checkpoint_recomputing = previous

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict] = None):
        kwargs = dict(gradient_checkpointing_kwargs or {})
        use_reentrant = kwargs.pop("use_reentrant", False)
        if use_reentrant:
            raise ValueError(
                "SaddleLLM activation checkpointing requires use_reentrant=False "
                "to preserve MoE auxiliary outputs and heterogeneous layer semantics"
            )
        if "context_fn" in kwargs:
            raise ValueError(
                "context_fn is managed internally for checkpoint-safe MoE state updates"
            )
        self._gradient_checkpointing_kwargs = {
            "use_reentrant": False,
            **kwargs,
        }
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {"use_reentrant": False}

    def enable_input_require_grads(self):
        def make_inputs_require_grad(_module, _input, output):
            output.requires_grad_(True)

        self.embed_tokens.register_forward_hook(make_inputs_require_grad)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def resize_token_embeddings(self, new_num_tokens: int):
        old_embed = self.embed_tokens
        old_lm_head = self.lm_head
        self.config.vocab_size = int(new_num_tokens)
        self.embed_tokens = nn.Embedding(new_num_tokens, self.config.hidden_size, device=old_embed.weight.device, dtype=old_embed.weight.dtype)
        self.lm_head = nn.Linear(self.config.hidden_size, new_num_tokens, bias=False, device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        num = min(old_embed.weight.shape[0], new_num_tokens)
        with torch.no_grad():
            self.embed_tokens.weight[:num].copy_(old_embed.weight[:num])
            self.lm_head.weight[:num].copy_(old_lm_head.weight[:num])
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        return self.embed_tokens

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict:
        if past_key_values is not None and self._past_key_value_length(past_key_values[0]) > 0:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @torch.no_grad()
    def router_metrics(self) -> Dict[str, float]:
        """Aggregate latest MoE router metrics across layers."""
        balances = []
        loads = []
        for layer in self.layers:
            if getattr(layer, "is_moe", False):
                load = layer.mlp.last_expert_load.detach().float()
                loads.append(load.cpu().tolist())
                balances.append(float(layer.mlp._load_balance(load).cpu()))
        if not balances:
            return {"num_moe_layers": 0}
        return {
            "num_moe_layers": len(balances),
            "mean_load_balance": sum(balances) / len(balances),
            "min_load_balance": min(balances),
            "max_load_balance": max(balances),
            "tokens_per_expert_by_layer": loads,
        }

    def _causal_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Shift the small integer labels instead of materializing a contiguous
        # copy of logits[:, :-1, :].  At long context and large vocabularies
        # that legacy copy can consume gigabytes per micro-batch.
        shifted_labels = torch.full_like(labels, -100)
        shifted_labels[:, :-1] = labels[:, 1:]
        flat_labels = shifted_labels.reshape(-1)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            flat_labels,
            ignore_index=-100,
            reduction="sum",
        )
        # ``cross_entropy(..., reduction="mean")`` returns NaN when every
        # target is ignored.  A differentiable zero is both safer for empty
        # packed samples and consistent with those tokens contributing no loss.
        valid_tokens = flat_labels.ne(-100).sum()
        return loss_sum / valid_tokens.clamp_min(1).to(loss_sum.dtype)

    @staticmethod
    def _linear_cross_entropy_sum(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Projection plus summed CE kept as one checkpointable region."""

        logits = F.linear(hidden_states, weight)
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=-100,
            reduction="sum",
        )

    def _chunked_linear_cross_entropy(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        """Compute exact mean CE without retaining a full ``tokens x vocab`` tensor.

        Each projection/CE region is non-reentrant-checkpointed.  Forward logits
        are therefore released after every chunk and recreated one chunk at a
        time during backward, bounding saved-logit memory by ``chunk_size x
        vocab_size``.  This uses ordinary PyTorch operators, so it remains a
        portable fallback when a fused linear-cross-entropy kernel is absent.
        """

        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if hidden_states.shape[:-1] != labels.shape:
            raise ValueError(
                "labels must match the leading dimensions of hidden_states: "
                f"got hidden_states={tuple(hidden_states.shape)}, "
                f"labels={tuple(labels.shape)}"
            )

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_labels = labels.reshape(-1)
        loss_sum = hidden_states.new_zeros(())
        should_checkpoint = bool(
            torch.is_grad_enabled()
            and (hidden_states.requires_grad or weight.requires_grad)
        )
        for start in range(0, flat_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, flat_hidden.shape[0])
            hidden_chunk = flat_hidden[start:end]
            label_chunk = flat_labels[start:end]
            if should_checkpoint:
                chunk_loss = checkpoint(
                    self._linear_cross_entropy_sum,
                    hidden_chunk,
                    weight,
                    label_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_loss = self._linear_cross_entropy_sum(
                    hidden_chunk,
                    weight,
                    label_chunk,
                )
            loss_sum = loss_sum + chunk_loss

        valid_tokens = flat_labels.ne(-100).sum()
        return loss_sum / valid_tokens.clamp_min(1).to(loss_sum.dtype)

    def _chunked_causal_lm_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        shifted_labels = torch.full_like(labels, -100)
        shifted_labels[..., :-1] = labels[..., 1:]
        return self._chunked_linear_cross_entropy(
            hidden_states,
            shifted_labels,
            weight,
            chunk_size=chunk_size,
        )

    def _mtp_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        total = hidden_states.new_zeros(())
        items = {}
        for idx, head in enumerate(self.mtp_heads):
            offset = idx + 2
            if labels.size(1) <= offset:
                continue
            if chunk_size is not None:
                shifted_labels = torch.full_like(labels, -100)
                shifted_labels[:, :-offset] = labels[:, offset:]
                loss = self._chunked_linear_cross_entropy(
                    hidden_states,
                    shifted_labels,
                    head.weight,
                    chunk_size=chunk_size,
                )
            else:
                logits = head(hidden_states[:, :-offset, :])
                target = labels[:, offset:]
                flat_target = target.reshape(-1)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    flat_target,
                    ignore_index=-100,
                    reduction="sum",
                )
                loss = loss_sum / flat_target.ne(-100).sum().clamp_min(1).to(
                    loss_sum.dtype
                )
            weight = self.config.mtp_loss_weights[min(idx + 1, len(self.config.mtp_loss_weights) - 1)]
            total = total + weight * loss
            items[f"mtp_loss_{offset}"] = loss.detach()
        return total, items

    @torch.no_grad()
    def create_generation_session(
        self, input_ids: torch.Tensor, **generation_kwargs,
    ) -> NativeGenerationSession:
        """Create request-local prefill/step state for advanced inference callers."""
        return NativeGenerationSession(
            self, input_ids, buffer_factory=GenerationBuffers, **generation_kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        do_sample: Optional[bool] = None,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Any] = None,
        min_new_tokens: int = 0,
        use_cache: bool = True,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        stopping_criteria: Optional[Any] = None,
        streamer: Optional[Any] = None,
        cache_implementation: str = "geometric",
        logits_to_keep: int = 1,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Run the same resumable session used by native streaming inference."""
        with self.create_generation_session(
            input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, pad_token_id=pad_token_id,
            eos_token_id=eos_token_id, min_new_tokens=min_new_tokens, use_cache=use_cache,
            output_scores=output_scores, stopping_criteria=stopping_criteria,
            streamer=streamer, cache_implementation=cache_implementation,
            logits_to_keep=logits_to_keep, **generation_kwargs,
        ) as session:
            while not session.done:
                session.step()
            out = session.sequences()
            if return_dict_in_generate:
                return SaddleGenerationOutput(sequences=out, scores=session.scores)
            return out

    @torch.no_grad()
    def draft_mtp_tokens(self, input_ids: torch.Tensor, max_draft_tokens: Optional[int] = None) -> torch.Tensor:
        """Return greedy MTP draft tokens predicted from the final hidden state."""
        self.eval()
        if not self.mtp_heads:
            return input_ids.new_empty((input_ids.shape[0], 0))
        outputs = self(input_ids, use_cache=False)
        hidden = outputs.hidden_states[:, -1:, :]
        draft_logits = [head(hidden).squeeze(1) for head in self.mtp_heads]
        draft = torch.stack([torch.argmax(logits, dim=-1) for logits in draft_logits], dim=1)
        if max_draft_tokens is not None:
            draft = draft[:, :max_draft_tokens]
        return draft

    @torch.no_grad()
    def generate_mtp_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        max_draft_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Experimental full-forward MTP draft/verify decoding.

        This is intentionally conservative: MTP heads propose future tokens,
        and the main LM head verifies them before acceptance.
        """
        self.eval()
        out = input_ids
        while out.shape[1] - input_ids.shape[1] < max_new_tokens:
            logits = self(out).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            out = torch.cat([out, next_token], dim=1)
            remaining = max_new_tokens - (out.shape[1] - input_ids.shape[1])
            if remaining <= 0 or not self.mtp_heads:
                continue
            draft = self.draft_mtp_tokens(out[:, :-1], max_draft_tokens=max_draft_tokens or remaining)
            if draft.numel() == 0:
                continue
            draft = draft[:, :remaining]
            candidate = torch.cat([out, draft], dim=1)
            verify_logits = self(candidate[:, :-1]).logits[:, out.shape[1] - 1 :, :]
            verify_tokens = torch.argmax(verify_logits, dim=-1)
            accepted = 0
            for idx in range(draft.shape[1]):
                if torch.equal(verify_tokens[:, idx], draft[:, idx]):
                    accepted += 1
                else:
                    break
            if accepted > 0:
                out = torch.cat([out, draft[:, :accepted]], dim=1)
        return out[:, : input_ids.shape[1] + max_new_tokens]

    def _canonicalize_projection_state_dict(self, state_dict: Dict) -> Dict:
        """Return a checkpoint dict with stable logical projection names."""

        canonical = state_dict.copy()
        for module_name, module in self.named_modules():
            prefix = f"{module_name}." if module_name else ""
            if isinstance(module, SaddleAttention):
                packed_key = prefix + "qkv_packed_proj.weight"
                legacy_keys = tuple(
                    prefix + name
                    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight")
                )
                if packed_key in canonical:
                    if any(key in canonical for key in legacy_keys):
                        raise ValueError(
                            f"state_dict contains both {packed_key} and legacy Q/K/V keys"
                        )
                    packed = canonical.pop(packed_key)
                    parts = packed.split(
                        (
                            module.q_projection_size,
                            module.kv_projection_size,
                            module.kv_projection_size,
                        ),
                        dim=0,
                    )
                    for key, value in zip(legacy_keys, parts):
                        canonical[key] = value
            elif isinstance(module, SaddleSwiGLU):
                packed_key = prefix + "gate_up_packed_proj.weight"
                legacy_keys = (
                    prefix + "gate_proj.weight",
                    prefix + "up_proj.weight",
                )
                if packed_key in canonical:
                    if any(key in canonical for key in legacy_keys):
                        raise ValueError(
                            f"state_dict contains both {packed_key} and legacy gate/up keys"
                        )
                    packed = canonical.pop(packed_key)
                    gate, up = packed.split(module.intermediate_size, dim=0)
                    canonical[legacy_keys[0]] = gate
                    canonical[legacy_keys[1]] = up
        return canonical

    def save_pretrained(
        self,
        path: str,
        state_dict: Optional[Dict] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        **_,
    ) -> str:
        if getattr(self, "_saddle_cuda_int8", None) is not None:
            raise ValueError(
                "native runtime INT8 cannot be exported as a float Saddle checkpoint; "
                "keep and reload the original checkpoint"
            )
        os.makedirs(path, exist_ok=True)
        state = (
            self.state_dict()
            if state_dict is None
            else self._canonicalize_projection_state_dict(state_dict)
        )
        torch.save(state, os.path.join(path, "pytorch_model.bin"))
        self.config.save_pretrained(path)
        manifest = {
            "schema_version": 1,
            "format": "saddle",
            "model_class": type(self).__name__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_file": "saddle_config.json",
            "weights_file": "pytorch_model.bin",
            "blueprint_file": (
                "model_blueprint.json" if self.config.model_blueprint is not None else None
            ),
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "metadata": dict(metadata or {}),
            "parallelism": dict(parallelism or {
                "strategy": "single",
                "world_size": 1,
                "rank": 0,
                "state_dict_type": "full",
            }),
        }
        with open(os.path.join(path, "saddle_checkpoint.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return path

    @classmethod
    def from_pretrained_saddle(cls, path: str, map_location: Optional[str] = None) -> "SaddleForCausalLM":
        config_path = os.path.join(path, "saddle_config.json")
        weights_path = os.path.join(path, "pytorch_model.bin")
        if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                "native SaddleLLM checkpoint requires saddle_config.json and pytorch_model.bin"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config = SaddleModelConfig(**json.load(f))
        model = cls(config)
        try:
            from transformers.utils.import_utils import check_torch_load_is_safe

            check_torch_load_is_safe()
        except ImportError:
            from packaging.version import Version

            torch_version = Version(torch.__version__.split("+")[0])
            if torch_version < Version("2.6"):
                raise RuntimeError(
                    "Loading native .bin checkpoints requires torch>=2.6 for safe "
                    "weights-only deserialization"
                )
        state = torch.load(
            weights_path,
            map_location=map_location or "cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        return model
