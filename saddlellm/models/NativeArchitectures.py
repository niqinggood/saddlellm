"""Explicit architecture contracts for shared native checkpoint loading.

Brand/repository names never select mathematical semantics. Each adapter
validates config.json before weight allocation, then uses Saddle's common
forward, KV cache and generation implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .SaddleModeling import SaddleModelConfig


@dataclass(frozen=True)
class NativeArchitecture:
    model_type: str
    architecture: str
    qk_norm: bool = False
    qkv_bias: bool = False


_ARCHITECTURES = {
    "qwen3": NativeArchitecture("qwen3", "Qwen3ForCausalLM", qk_norm=True),
    "qwen2": NativeArchitecture("qwen2", "Qwen2ForCausalLM", qkv_bias=True),
    "llama": NativeArchitecture("llama", "LlamaForCausalLM"),
}


def native_architecture_config(
    raw: dict, generation: dict | None = None, *, expected_type: str | None = None,
) -> SaddleModelConfig:
    """Map only validated dense, full-attention, default-RoPE architectures."""
    from .NativeQwen3 import _positive_float, _positive_int, _token_id

    model_type = raw.get("model_type")
    if not isinstance(model_type, str) or model_type not in _ARCHITECTURES:
        raise ValueError(
            f"native dense checkpoint adapters support {', '.join(_ARCHITECTURES)}; "
            f"unsupported model_type={model_type!r}. Use a compatible local/vllm "
            "backend for other architectures. No HF fallback is performed."
        )
    if expected_type is not None and model_type != expected_type:
        raise ValueError(f"this native dense loader requires model_type={expected_type!r}")
    spec = _ARCHITECTURES[model_type]
    if raw.get("architectures", [spec.architecture]) != [spec.architecture]:
        raise ValueError(f"native {model_type} requires {spec.architecture} weights")
    if raw.get("quantization_config"):
        raise ValueError("native dense adapters require non-quantized checkpoints")
    if raw.get("hidden_act", "silu") != "silu":
        raise ValueError("native dense adapters require hidden_act='silu'")
    if model_type != "qwen2" and raw.get("attention_bias", False) is not False:
        raise ValueError(f"native {model_type} does not support attention_bias")
    if raw.get("mlp_bias", False) is not False:
        raise ValueError("native dense adapters do not support mlp_bias")
    if raw.get("use_sliding_window", False) is not False:
        raise ValueError("native dense adapters currently require full attention, not sliding windows")
    if raw.get("rope_scaling") or raw.get("rope_parameters"):
        raise ValueError("native dense adapters currently require unscaled default RoPE")
    if raw.get("use_mrope", False) is not False:
        raise ValueError("native dense adapters do not support multimodal RoPE")
    if raw.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("native dense adapters require full rotary head dimensions")
    if raw.get("attention_dropout", 0.0) != 0.0 or raw.get("pretraining_tp", 1) != 1:
        raise ValueError("native dense adapters require attention_dropout=0 and pretraining_tp=1")
    if not isinstance(raw.get("tie_word_embeddings", False), bool):
        raise ValueError("tie_word_embeddings must be boolean")
    vocab = _positive_int(raw, "vocab_size")
    hidden = _positive_int(raw, "hidden_size")
    heads = _positive_int(raw, "num_attention_heads")
    kv_heads = _positive_int(raw, "num_key_value_heads", heads)
    layers = _positive_int(raw, "num_hidden_layers")
    head_dim = _positive_int(raw, "head_dim", hidden // heads)
    if heads % kv_heads or head_dim % 2:
        raise ValueError("native attention needs an integer Q/KV head ratio and an even head_dim")
    if "head_dim" not in raw and hidden % heads:
        raise ValueError("hidden_size/num_attention_heads needs an explicit head_dim")
    layer_types = raw.get("layer_types")
    if layer_types is not None and layer_types != ["full_attention"] * layers:
        raise ValueError("native dense adapters support full_attention in every layer only")
    generation = generation or {}
    ids = {
        name: _token_id(generation.get(name, raw.get(name)), name, vocab,
                        multiple=name == "eos_token_id")
        for name in ("pad_token_id", "bos_token_id", "eos_token_id")
    }
    if ids["pad_token_id"] is None:
        eos = ids["eos_token_id"]
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        ids["pad_token_id"] = eos if eos is not None else 0
    return SaddleModelConfig(
        vocab_size=vocab, hidden_size=hidden,
        intermediate_size=_positive_int(raw, "intermediate_size"),
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv_heads, head_dim=head_dim,
        max_position_embeddings=_positive_int(raw, "max_position_embeddings"),
        rope_theta=_positive_float(raw, "rope_theta", 10000.0),
        rms_norm_eps=_positive_float(raw, "rms_norm_eps", 1e-6),
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
        attention_kind="gqa" if heads != kv_heads else "mha",
        qk_norm=spec.qk_norm, qkv_bias=spec.qkv_bias, attention_backend="sdpa",
        rms_norm_weight_in_fp32=False, rope_compute_fp32=True, **ids,
    )


def load_native_transformer(path, **kwargs):
    """Reuse strict safetensors validation/loading across registered adapters."""
    from .NativeQwen3 import _load_native_checkpoint

    return _load_native_checkpoint(path, config_factory=native_architecture_config, **kwargs)
