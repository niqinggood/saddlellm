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
    attention_kind: str = "gqa"  # mha | mqa | gqa | mla
    attention_backend: str = "sdpa"  # auto | sdpa | flash | eager | xformers | transformer_engine
    mla_cache_mode: str = "kv"  # kv | latent
    ffn_kind: str = "swiglu"  # swiglu | moe
    num_experts: int = 0
    num_experts_per_tok: int = 0
    expert_intermediate_size: int = 0
    shared_expert: bool = False
    aux_loss_free: bool = False
    router_aux_loss_coef: float = 0.001
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    sliding_window: int = 0
    multi_token_prediction: bool = False
    mtp_extra_tokens: int = 0
    mtp_loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.3, 0.1, 0.05])
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
    layer_name: str = ""
    layer_configs: List[Dict[str, Any]] = field(default_factory=list)
    model_blueprint: Optional[Dict[str, Any]] = None
    schema_version: int = 2

    def __post_init__(self) -> None:
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
            mla_cache_mode=getattr(blueprint.attention, "mla_cache_mode", "kv"),
            ffn_kind=blueprint.ffn.kind,
            num_experts=blueprint.ffn.num_experts,
            num_experts_per_tok=blueprint.ffn.experts_per_token,
            expert_intermediate_size=blueprint.ffn.expert_intermediate_size,
            shared_expert=blueprint.ffn.shared_expert,
            aux_loss_free=blueprint.ffn.aux_loss_free,
            router_aux_loss_coef=blueprint.ffn.router_aux_loss_coef,
            q_lora_rank=blueprint.attention.q_lora_rank,
            kv_lora_rank=blueprint.attention.kv_lora_rank,
            sliding_window=blueprint.attention.sliding_window,
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
            attention_kind=attention_kind,
            attention_backend="sdpa",
            mla_cache_mode="kv",
            ffn_kind=ffn_kind,
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
            "mla_cache_mode": attention.get("mla_cache_mode", self.mla_cache_mode),
            "num_attention_heads": attention.get("num_heads", self.num_attention_heads),
            "num_key_value_heads": attention.get("num_kv_heads", self.num_key_value_heads),
            "head_dim": attention.get("head_dim", self.head_dim),
            "rope_theta": attention.get("rope_theta", self.rope_theta),
            "rope_scaling": attention.get("rope_scaling", self.rope_scaling),
            "sliding_window": attention.get("sliding_window", self.sliding_window),
            "q_lora_rank": attention.get("q_lora_rank", self.q_lora_rank),
            "kv_lora_rank": attention.get("kv_lora_rank", self.kv_lora_rank),
            "ffn_kind": ffn.get("kind", self.ffn_kind),
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
        }
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


class SaddleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class SaddleRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = float(scaling_factor)

    def forward(self, seq_len: int, device=None, dtype=None, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def apply_rotary_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


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

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        if config.attention_kind == "mla" and config.kv_lora_rank > 0:
            q_rank = config.q_lora_rank or max(1, config.hidden_size // 8)
            self.q_a_proj = nn.Linear(config.hidden_size, q_rank, bias=False)
            self.q_b_proj = nn.Linear(q_rank, self.num_heads * self.head_dim, bias=False)
            self.kv_a_proj = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
            self.k_b_proj = nn.Linear(config.kv_lora_rank, self.num_kv_heads * self.head_dim, bias=False)
            self.v_b_proj = nn.Linear(config.kv_lora_rank, self.num_kv_heads * self.head_dim, bias=False)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary = SaddleRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=(config.rope_scaling or {}).get("factor", 1.0),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        bsz, q_len, _ = hidden_states.shape
        use_mla = self.config.attention_kind == "mla" and self.config.kv_lora_rank > 0
        use_latent_cache = use_mla and self.config.mla_cache_mode == "latent"

        if use_latent_cache:
            q = self.q_b_proj(self.q_a_proj(hidden_states))
            latent_kv = self.kv_a_proj(hidden_states)
            if isinstance(past_key_value, dict) and past_key_value.get("cache_type") == "mla_latent":
                past_latent = past_key_value["latent_kv"]
                past_len = past_latent.shape[1]
                latent_kv_all = torch.cat([past_latent, latent_kv], dim=1)
            else:
                past_len = 0
                latent_kv_all = latent_kv
            kv_len = latent_kv_all.shape[1]
            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_b_proj(latent_kv_all).view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_b_proj(latent_kv_all).view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            cos_q, sin_q = self.rotary(q_len, device=hidden_states.device, dtype=hidden_states.dtype, offset=past_len)
            cos_k, sin_k = self.rotary(kv_len, device=hidden_states.device, dtype=hidden_states.dtype, offset=0)
            q = apply_rotary_single(q, cos_q, sin_q)
            k = apply_rotary_single(k, cos_k, sin_k)
            present_key_value = {"cache_type": "mla_latent", "latent_kv": latent_kv_all} if use_cache else None
        else:
            past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
            if use_mla:
                q = self.q_b_proj(self.q_a_proj(hidden_states))
                latent_kv = self.kv_a_proj(hidden_states)
                k = self.k_b_proj(latent_kv)
                v = self.v_b_proj(latent_kv)
            else:
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)

            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.rotary(q_len, device=hidden_states.device, dtype=hidden_states.dtype, offset=past_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            present_key_value = (k, v) if use_cache else None
            kv_len = k.shape[2]

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_mask, is_causal = self._build_attention_mask(
            attention_mask=attention_mask,
            q_len=q_len,
            kv_len=kv_len,
            past_len=past_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            sliding_window=self.config.sliding_window,
        )
        out = self._attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim
        )
        return self.o_proj(out), present_key_value

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
        sliding_window: int = 0,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        if attention_mask is None and past_len == 0 and not sliding_window:
            return None, True
        min_value = torch.finfo(dtype).min
        query_pos = torch.arange(past_len, past_len + q_len, device=device)[:, None]
        key_pos = torch.arange(kv_len, device=device)[None, :]
        causal = torch.zeros((q_len, kv_len), device=device, dtype=dtype)
        causal = causal.masked_fill(key_pos > query_pos, min_value)
        if sliding_window and sliding_window > 0:
            causal = causal.masked_fill(
                key_pos <= query_pos - int(sliding_window), min_value
            )
        if attention_mask is None:
            return causal[None, None, :, :], False
        if attention_mask.shape[-1] != kv_len:
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
        padding = (1.0 - attention_mask[:, None, None, :].to(dtype=dtype, device=device)) * min_value
        return padding + causal[None, None, :, :], False


class SaddleSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SaddleMoE(nn.Module):
    def __init__(self, config: SaddleModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok or 2
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.aux_loss_free = config.aux_loss_free
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SaddleSwiGLU(config.hidden_size, config.expert_intermediate_size or config.intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.shared_expert = SaddleSwiGLU(config.hidden_size, config.expert_intermediate_size or config.intermediate_size) if config.shared_expert else None
        self.register_buffer("router_bias", torch.zeros(config.num_experts), persistent=True)
        self.register_buffer("last_expert_load", torch.zeros(config.num_experts), persistent=False)
        self._checkpoint_recomputing = False

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        bsz, seq_len, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        logits = self.router(flat)
        routed_logits = logits + self.router_bias if self.aux_loss_free else logits
        topk_scores, topk_indices = torch.topk(routed_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)

        output = torch.zeros_like(flat)
        unused_expert_connection = flat.new_zeros(())
        for expert_idx, expert in enumerate(self.experts):
            matches = topk_indices == expert_idx
            token_mask = matches.any(dim=-1)
            if not token_mask.any():
                # DDP must see every expert parameter in the autograd graph on
                # every rank, even when this rank routes no tokens to an
                # expert.  Connecting one scalar from each parameter creates
                # a full zero gradient without running the empty expert or
                # reducing over its large weight matrices.  This makes the
                # used-parameter set stable and works with static_graph=True.
                unused_expert_connection = (
                    unused_expert_connection
                    + self._zero_parameter_connection(expert, flat)
                )
                continue
            token_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            expert_out = expert(flat[token_idx])
            weights = (topk_weights[token_idx] * matches[token_idx].to(topk_weights.dtype)).sum(dim=-1)
            output[token_idx] += expert_out * weights.unsqueeze(-1)
        output = output + unused_expert_connection

        if self.shared_expert is not None:
            output = output + self.shared_expert(flat)

        load = torch.bincount(
            topk_indices.reshape(-1), minlength=self.num_experts
        ).to(flat.dtype)
        load = self._global_expert_load(load)
        self.last_expert_load = load.detach()
        aux_loss = (
            self._router_aux_loss(logits, load)
            if self.router_aux_loss_coef > 0
            else flat.new_zeros(())
        )
        if self.training and self.aux_loss_free and not self._checkpoint_recomputing:
            self._update_router_bias(load)
        stats = {
            "expert_load": load.detach(),
            "router_aux_loss": aux_loss.detach(),
            "router_entropy": self._router_entropy(logits).detach(),
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

    @staticmethod
    def _zero_parameter_connection(module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
        connection = reference.new_zeros(())
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.numel() > 0:
                connection = connection + parameter.reshape(-1)[0] * 0.0
        return connection

    def _router_aux_loss(self, logits: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1).mean(dim=0)
        probs = self._global_router_probabilities(probs)
        load_fraction = load.float() / load.float().sum().clamp_min(1.0)
        return self.router_aux_loss_coef * self.num_experts * torch.sum(
            probs * load_fraction
        )

    def _update_router_bias(self, load: torch.Tensor, rate: float = 1e-3):
        with torch.no_grad():
            avg = load.mean().clamp_min(1.0)
            self.router_bias -= rate * (load / avg - 1.0)

    def _router_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
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
        self.input_layernorm = SaddleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SaddleAttention(config)
        self.post_attention_layernorm = SaddleRMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.ffn_kind == "moe" and config.num_experts > 0:
            self.mlp = SaddleMoE(config)
            self.is_moe = True
        else:
            self.mlp = SaddleSwiGLU(config.hidden_size, config.intermediate_size)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        attn_out, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        attn_out = self.residual_dropout(attn_out) * self.attention_residual_scale
        if self.config.residual_topology == "parallel":
            mlp_in = self.post_attention_layernorm(residual)
        else:
            hidden_states = residual + attn_out
            mlp_in = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            mlp_out, aux_loss, stats = self.mlp(mlp_in)
        else:
            mlp_out = self.mlp(mlp_in)
            aux_loss = residual.new_zeros(())
            stats = {}
        mlp_out = self.residual_dropout(mlp_out) * self.ffn_residual_scale
        if self.config.residual_topology == "parallel":
            hidden_states = residual + attn_out + mlp_out
        else:
            hidden_states = hidden_states + mlp_out
        return hidden_states, aux_loss, stats, present_key_value


class SaddleForCausalLM(nn.Module):
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
        self.norm = SaddleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.mtp_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(max(0, config.mtp_extra_tokens))
        ])
        self.post_init()

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
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        **_,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
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
        for layer, past_key_value in zip(self.layers, past_key_values):
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
                    current_layer: SaddleDecoderLayer = layer,
                    current_past_key_value: Any = past_key_value,
                ):
                    return current_layer(
                        states,
                        attention_mask=attention_mask,
                        past_key_value=current_past_key_value,
                        use_cache=False,
                    )

                hidden_states, aux_loss, stats, present_key_value = checkpoint(
                    custom_forward,
                    hidden_states,
                    **checkpoint_kwargs,
                )
            else:
                hidden_states, aux_loss, stats, present_key_value = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            aux_losses.append(aux_loss)
            if stats:
                expert_stats.append(stats)
            if use_cache:
                next_past_key_values.append(present_key_value)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        loss_items = {}
        if labels is not None:
            lm_loss = self._causal_lm_loss(logits, labels)
            aux_loss = torch.stack(aux_losses).sum() if aux_losses else logits.new_zeros(())
            loss = lm_loss + aux_loss
            loss_items = {"lm_loss": lm_loss.detach(), "aux_loss": aux_loss.detach()}
            if self.config.multi_token_prediction and self.mtp_heads:
                mtp_loss, mtp_items = self._mtp_loss(hidden_states, labels)
                loss = loss + mtp_loss
                loss_items.update(mtp_items)

        output = SaddleCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=next_past_key_values,
            hidden_states=hidden_states,
            expert_stats=expert_stats,
            loss_items=loss_items,
        )
        if return_dict:
            return output
        return (loss, logits) if loss is not None else (logits,)

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
        if past_key_values is not None:
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
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

    def _mtp_loss(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        total = hidden_states.new_zeros(())
        items = {}
        for idx, head in enumerate(self.mtp_heads):
            offset = idx + 2
            if labels.size(1) <= offset:
                continue
            logits = head(hidden_states[:, :-offset, :])
            target = labels[:, offset:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100)
            weight = self.config.mtp_loss_weights[min(idx + 1, len(self.config.mtp_loss_weights) - 1)]
            total = total + weight * loss
            items[f"mtp_loss_{offset}"] = loss.detach()
        return total, items

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
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate with the standard greedy/sampling arguments used by callers."""
        # Tokenizers may emit these, but this decoder does not need them.
        generation_kwargs.pop("token_type_ids", None)
        generation_kwargs.pop("position_ids", None)
        generation_kwargs.pop("bos_token_id", None)
        num_beams = generation_kwargs.pop("num_beams", 1)
        if num_beams != 1:
            raise NotImplementedError("native SaddleLLM generation does not support beam search")
        if generation_kwargs:
            names = ", ".join(sorted(generation_kwargs))
            raise TypeError(f"unsupported native generation argument(s): {names}")
        if max_new_tokens < 0 or min_new_tokens < 0:
            raise ValueError("max_new_tokens and min_new_tokens must be non-negative")
        if min_new_tokens > max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if do_sample is None:
            do_sample = temperature > 0
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=True")

        self.eval()
        out = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(out)
        elif attention_mask.shape != out.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        else:
            attention_mask = attention_mask.to(device=out.device)
        pad_token_id = self.config.pad_token_id if pad_token_id is None else int(pad_token_id)
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if isinstance(eos_token_id, torch.Tensor):
            eos_ids = [int(item) for item in eos_token_id.flatten().tolist()]
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids = [int(item) for item in eos_token_id]
        elif eos_token_id is None:
            eos_ids = []
        else:
            eos_ids = [int(eos_token_id)]
        finished = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
        collected_scores = [] if output_scores else None
        past_key_values = None
        for step in range(max_new_tokens):
            model_inputs = self.prepare_inputs_for_generation(
                out,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            outputs = self(**model_inputs)
            past_key_values = outputs.past_key_values if use_cache else None
            logits = outputs.logits[:, -1, :]

            if repetition_penalty != 1.0:
                for batch_index in range(out.shape[0]):
                    token_ids = torch.unique(out[batch_index])
                    token_scores = logits[batch_index, token_ids]
                    logits[batch_index, token_ids] = torch.where(
                        token_scores < 0,
                        token_scores * repetition_penalty,
                        token_scores / repetition_penalty,
                    )
            if eos_ids and step < min_new_tokens:
                logits[:, eos_ids] = -torch.inf
            if output_scores:
                collected_scores.append(logits.detach())

            if not do_sample:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                sample_logits = logits / temperature
                if top_k > 0:
                    keep = min(int(top_k), sample_logits.shape[-1])
                    threshold = torch.topk(sample_logits, keep, dim=-1).values[:, -1, None]
                    sample_logits = sample_logits.masked_fill(sample_logits < threshold, -torch.inf)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(sample_logits, descending=True, dim=-1)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
                    sample_logits = torch.full_like(sample_logits, -torch.inf).scatter(
                        -1, sorted_indices, sorted_logits
                    )
                probs = F.softmax(sample_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            active = ~finished
            next_token = torch.where(
                active[:, None], next_token, torch.full_like(next_token, pad_token_id)
            )
            out = torch.cat([out, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, active.to(attention_mask.dtype)[:, None]], dim=1
            )
            if eos_ids:
                is_eos = torch.zeros_like(finished)
                for token_id in eos_ids:
                    is_eos |= next_token.squeeze(-1).eq(token_id)
                finished |= is_eos & active
                if bool(finished.all()):
                    break
        if return_dict_in_generate:
            return SaddleGenerationOutput(
                sequences=out,
                scores=tuple(collected_scores) if collected_scores is not None else None,
            )
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

    def save_pretrained(
        self,
        path: str,
        state_dict: Optional[Dict] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        **_,
    ) -> str:
        os.makedirs(path, exist_ok=True)
        state = self.state_dict() if state_dict is None else state_dict
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
