"""Model architecture blueprints and design-time analysis.

This module is intentionally model-focused.  DeepSpeed and Megatron-LM are
execution backends; SaddleLLM's differentiating layer should be the model
idea: architecture variants, active-parameter budgets, long-context choices,
MoE/MLA/MTP experiments, and domain-specific small-model families.
"""
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class AttentionBlueprint:
    kind: str = "gqa"  # mha | mqa | gqa | mla
    backend: str = "sdpa"  # auto | sdpa | flash | eager | xformers | transformer_engine
    mla_cache_mode: str = "kv"  # kv | latent
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: Optional[int] = None
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    sliding_window: int = 0

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()
        if self.kind == "mha":
            self.num_kv_heads = self.num_heads
        elif self.kind == "mqa":
            self.num_kv_heads = 1


@dataclass
class FFNBlueprint:
    kind: str = "swiglu"  # swiglu | moe
    intermediate_size: int = 4096
    num_experts: int = 0
    experts_per_token: int = 0
    expert_intermediate_size: int = 0
    shared_expert: bool = False
    aux_loss_free: bool = False
    router_aux_loss_coef: float = 0.001

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()


@dataclass
class ResidualBlueprint:
    """Residual topology, scaling and branch-output initialization.

    ``topology="serial"`` implements the conventional Transformer sequence
    ``x + attention`` followed by ``x + ffn``.  ``parallel`` evaluates both
    branches from the same input and adds them together.  Scales can be fixed
    constants or trainable scalar parameters.
    """

    topology: str = "serial"  # serial | parallel
    attention_scale: float = 1.0
    ffn_scale: float = 1.0
    learnable: bool = False
    dropout: float = 0.0
    initialization: str = "standard"  # standard | depth_scaled | zero

    def __post_init__(self) -> None:
        topology = str(self.topology).lower().replace("_", "-")
        self.topology = {
            "sequential": "serial",
            "pre-norm": "serial",
            "prenorm": "serial",
        }.get(topology, topology)
        self.initialization = str(self.initialization).lower().replace("-", "_")


@dataclass
class LayerBlueprint:
    """One repeatable decoder-layer segment.

    ``attention`` and ``ffn`` are complete component descriptions after a
    blueprint is normalized.  Configuration files may provide partial
    dictionaries; :meth:`ModelBlueprint.from_dict` merges those dictionaries
    with the model-level defaults before constructing this object.
    """

    repeat: int = 1
    attention: Optional[AttentionBlueprint] = None
    ffn: Optional[FFNBlueprint] = None
    residual: Optional[ResidualBlueprint] = None
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int):
            raise TypeError("layer repeat must be an integer")
        if self.repeat <= 0:
            raise ValueError("layer repeat must be positive")

    def to_dict(self) -> Dict[str, Any]:
        def component_dict(value):
            if value is None:
                return None
            return dict(value) if isinstance(value, dict) else asdict(value)

        return {
            "name": self.name,
            "repeat": self.repeat,
            "attention": component_dict(self.attention),
            "ffn": component_dict(self.ffn),
            "residual": component_dict(self.residual),
        }


@dataclass
class ObjectiveBlueprint:
    causal_lm: bool = True
    multi_token_prediction: bool = False
    mtp_extra_tokens: int = 0
    reasoning_trace_policy: str = "mixed"  # keep | strip | mixed
    domain_objectives: List[str] = field(default_factory=list)


@dataclass
class VisionBlueprint:
    enabled: bool = False
    backbone: str = "tiny_patch"  # tiny_patch | clip | siglip | dinov2 | hf model id
    pretrained_name_or_path: Optional[str] = None
    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 768
    output_tokens: int = 64
    freeze: bool = True


@dataclass
class ProjectorBlueprint:
    kind: str = "mlp"  # linear | mlp | gated_mlp | resampler
    intermediate_size: int = 0
    num_query_tokens: int = 64
    dropout: float = 0.0


@dataclass
class ModelBlueprint:
    name: str
    family: str = "llama"
    hidden_size: int = 1024
    num_layers: int = 24
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    norm: str = "rmsnorm"
    norm_eps: float = 1e-5
    activation: str = "silu"
    tie_word_embeddings: bool = False
    initializer: str = "llama"
    attention: AttentionBlueprint = field(default_factory=AttentionBlueprint)
    ffn: FFNBlueprint = field(default_factory=FFNBlueprint)
    residual: ResidualBlueprint = field(default_factory=ResidualBlueprint)
    layers: List[LayerBlueprint] = field(default_factory=list)
    objective: ObjectiveBlueprint = field(default_factory=ObjectiveBlueprint)
    vision: VisionBlueprint = field(default_factory=VisionBlueprint)
    projector: ProjectorBlueprint = field(default_factory=ProjectorBlueprint)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.attention, dict):
            self.attention = AttentionBlueprint(**self.attention)
        if isinstance(self.ffn, dict):
            self.ffn = FFNBlueprint(**self.ffn)
        if isinstance(self.residual, dict):
            self.residual = self._residual_from_dict(self.residual)
        if isinstance(self.objective, dict):
            self.objective = ObjectiveBlueprint(**self.objective)
        if isinstance(self.vision, dict):
            self.vision = VisionBlueprint(**self.vision)
        if isinstance(self.projector, dict):
            self.projector = ProjectorBlueprint(**self.projector)
        for label, value, expected in (
            ("attention", self.attention, AttentionBlueprint),
            ("ffn", self.ffn, FFNBlueprint),
            ("residual", self.residual, ResidualBlueprint),
            ("objective", self.objective, ObjectiveBlueprint),
            ("vision", self.vision, VisionBlueprint),
            ("projector", self.projector, ProjectorBlueprint),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{label} must be {expected.__name__} or a mapping")
        if not isinstance(self.layers, list):
            raise TypeError("layers must be a list")
        normalized_layers: List[LayerBlueprint] = []
        for item in self.layers:
            if isinstance(item, LayerBlueprint):
                layer = self._layer_from_dict(
                    item.to_dict(), self.attention, self.ffn, self.residual
                )
            elif isinstance(item, dict):
                layer = self._layer_from_dict(
                    item, self.attention, self.ffn, self.residual
                )
            else:
                raise TypeError(
                    "layers must contain LayerBlueprint objects or dictionaries"
                )
            if layer.attention is None:
                layer.attention = AttentionBlueprint(**asdict(self.attention))
            if layer.ffn is None:
                layer.ffn = FFNBlueprint(**asdict(self.ffn))
            if layer.residual is None:
                layer.residual = ResidualBlueprint(**asdict(self.residual))
            normalized_layers.append(layer)
        self.layers = normalized_layers
        if self.layers:
            # Layer segments are the authoritative topology when supplied.
            self.num_layers = sum(layer.repeat for layer in self.layers)
        self.validate()

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        defaults: Optional["ModelBlueprint"] = None,
    ) -> "ModelBlueprint":
        """Create a blueprint from JSON/YAML-compatible data.

        The derived ``analysis`` key is ignored so files written by
        :meth:`save` can be loaded again.  Other unknown keys fail fast.  A
        defaults blueprint is useful to apply a registered model size before
        overriding selected components.
        """

        raw = dict(data or {})
        if "model_blueprint" in raw and isinstance(raw["model_blueprint"], dict):
            raw = dict(raw["model_blueprint"])
        elif isinstance(raw.get("model"), dict) and isinstance(
            raw["model"].get("blueprint"), dict
        ):
            raw = dict(raw["model"]["blueprint"])
        raw.pop("analysis", None)
        cls._reject_unknown_keys(raw, cls, "model blueprint")
        base = defaults.to_config_dict() if defaults is not None else {}
        merged = {**base, **raw}
        cls._ensure_mapping(raw.get("attention"), "attention")
        cls._ensure_mapping(raw.get("ffn"), "ffn")
        cls._ensure_mapping(raw.get("residual"), "residual")
        cls._reject_unknown_keys(raw.get("attention") or {}, AttentionBlueprint, "attention")
        cls._reject_unknown_keys(raw.get("ffn") or {}, FFNBlueprint, "ffn")
        default_attention = AttentionBlueprint(
            **{
                **(asdict(defaults.attention) if defaults is not None else {}),
                **dict(raw.get("attention") or {}),
            }
        )
        default_ffn = FFNBlueprint(
            **{
                **(asdict(defaults.ffn) if defaults is not None else {}),
                **dict(raw.get("ffn") or {}),
            }
        )
        residual_overrides = cls._residual_values(raw.get("residual") or {})
        default_residual = ResidualBlueprint(
            **{
                **(asdict(defaults.residual) if defaults is not None else {}),
                **residual_overrides,
            }
        )
        layers = [
            cls._layer_from_dict(
                item, default_attention, default_ffn, default_residual
            )
            if isinstance(item, dict)
            else item
            for item in raw.get("layers", merged.get("layers", []))
        ]
        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in merged.items() if key in allowed}
        values["attention"] = default_attention
        values["ffn"] = default_ffn
        values["residual"] = default_residual
        values["layers"] = layers
        for key, component in (
            ("objective", ObjectiveBlueprint),
            ("vision", VisionBlueprint),
            ("projector", ProjectorBlueprint),
        ):
            value = raw.get(key, merged.get(key))
            if isinstance(value, dict):
                cls._reject_unknown_keys(value, component, key)
                default_value = asdict(getattr(defaults, key)) if defaults is not None else {}
                values[key] = component(**{**default_value, **value})
            elif value is not None and not isinstance(value, component):
                raise TypeError(f"{key} must be a mapping or {component.__name__}")
        return cls(**values)

    @classmethod
    def from_file(cls, path: str) -> "ModelBlueprint":
        """Load a standalone blueprint or a training config containing one."""

        path_string = os.fspath(path)
        with open(path_string, "r", encoding="utf-8") as file:
            if path_string.lower().endswith((".yaml", ".yml")):
                import yaml

                payload = yaml.safe_load(file) or {}
            else:
                payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("model blueprint file must contain a mapping")
        return cls.from_dict(payload)

    @staticmethod
    def _layer_from_dict(
        data: Dict[str, Any],
        default_attention: AttentionBlueprint,
        default_ffn: FFNBlueprint,
        default_residual: ResidualBlueprint,
    ) -> LayerBlueprint:
        raw = dict(data or {})
        ModelBlueprint._reject_unknown_keys(raw, LayerBlueprint, "layer")
        ModelBlueprint._ensure_mapping(raw.get("attention"), "layer.attention")
        ModelBlueprint._ensure_mapping(raw.get("ffn"), "layer.ffn")
        ModelBlueprint._ensure_mapping(raw.get("residual"), "layer.residual")
        ModelBlueprint._reject_unknown_keys(
            raw.get("attention") or {}, AttentionBlueprint, "layer.attention"
        )
        ModelBlueprint._reject_unknown_keys(raw.get("ffn") or {}, FFNBlueprint, "layer.ffn")
        repeat = raw.get("repeat", 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int):
            raise TypeError("layer repeat must be an integer")
        attention = AttentionBlueprint(
            **{**asdict(default_attention), **dict(raw.get("attention") or {})}
        )
        ffn = FFNBlueprint(
            **{**asdict(default_ffn), **dict(raw.get("ffn") or {})}
        )
        residual = ResidualBlueprint(
            **{
                **asdict(default_residual),
                **ModelBlueprint._residual_values(raw.get("residual") or {}),
            }
        )
        return LayerBlueprint(
            name=str(raw.get("name", "")),
            repeat=repeat,
            attention=attention,
            ffn=ffn,
            residual=residual,
        )

    @staticmethod
    def _residual_from_dict(data: Dict[str, Any]) -> ResidualBlueprint:
        return ResidualBlueprint(**ModelBlueprint._residual_values(data))

    @staticmethod
    def _residual_values(data: Dict[str, Any]) -> Dict[str, Any]:
        values = dict(data or {})
        ModelBlueprint._consume_alias(values, "layout", "topology")
        ModelBlueprint._consume_alias(values, "mode", "topology")
        ModelBlueprint._consume_alias(values, "projection_init", "initialization")
        ModelBlueprint._consume_alias(values, "init", "initialization")
        gate = values.pop("gate", None)
        if gate is not None:
            gate_kind = str(gate).lower()
            if gate_kind not in {"none", "fixed", "false", "scalar", "learnable", "true"}:
                raise ValueError(
                    "residual gate must be one of none, fixed, or scalar"
                )
            gate_value = gate_kind in {"scalar", "learnable", "true"}
            if "learnable" in values and values["learnable"] != gate_value:
                raise ValueError("residual gate conflicts with learnable")
            values.setdefault("learnable", gate_value)
        ModelBlueprint._consume_alias(values, "learnable_scale", "learnable")
        gate_init = values.pop("gate_init", None)
        if gate_init is not None:
            if isinstance(gate_init, dict):
                unknown = sorted(set(gate_init) - {"attention", "ffn"})
                if unknown:
                    raise ValueError(
                        "unknown residual gate_init field(s): " + ", ".join(unknown)
                    )
                values.setdefault("attention_scale", gate_init.get("attention", 1.0))
                values.setdefault("ffn_scale", gate_init.get("ffn", 1.0))
            else:
                values.setdefault("attention_scale", gate_init)
                values.setdefault("ffn_scale", gate_init)
        if "scale" in values:
            scale = values.pop("scale")
            values.setdefault("attention_scale", scale)
            values.setdefault("ffn_scale", scale)
        ModelBlueprint._reject_unknown_keys(values, ResidualBlueprint, "residual")
        return values

    @staticmethod
    def _ensure_mapping(value: Any, label: str) -> None:
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"{label} must be a mapping")

    @staticmethod
    def _consume_alias(values: Dict[str, Any], alias: str, canonical: str) -> None:
        if alias not in values:
            return
        alias_value = values.pop(alias)
        if canonical in values and values[canonical] != alias_value:
            raise ValueError(f"residual {alias} conflicts with {canonical}")
        values.setdefault(canonical, alias_value)

    @staticmethod
    def _reject_unknown_keys(data: Dict[str, Any], component, label: str) -> None:
        allowed = {item.name for item in fields(component)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")

    def validate(self) -> None:
        if not self.name:
            raise ValueError("model blueprint name is required")
        if self.hidden_size <= 0 or self.vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("model must contain at least one decoder layer")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if str(self.norm).lower() not in {"rmsnorm", "rms_norm"}:
            raise ValueError("native Saddle models currently support only rmsnorm")
        if str(self.activation).lower() not in {"silu", "swish"}:
            raise ValueError("native Saddle models currently support only silu activation")
        if self.initializer not in {"llama", "deepseek"}:
            raise ValueError("initializer must be llama or deepseek")
        for index, layer in enumerate(self.expanded_layers()):
            attention = layer.attention or self.attention
            ffn = layer.ffn or self.ffn
            residual = layer.residual or self.residual
            for label, value in (
                ("num_heads", attention.num_heads),
                ("num_kv_heads", attention.num_kv_heads),
            ):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"layer {index}: {label} must be an integer")
            heads = attention.num_heads
            kv_heads = attention.num_kv_heads or heads
            if heads <= 0 or (
                attention.head_dim is None and self.hidden_size % heads
            ):
                raise ValueError(
                    f"layer {index}: hidden_size must be divisible by num_heads"
                )
            if attention.head_dim is not None and (
                isinstance(attention.head_dim, bool)
                or not isinstance(attention.head_dim, int)
            ):
                raise TypeError(f"layer {index}: head_dim must be an integer")
            head_dim = attention.head_dim or self.hidden_size // heads
            if head_dim <= 0 or head_dim % 2:
                raise ValueError(
                    f"layer {index}: head_dim must be a positive even integer"
                )
            if kv_heads <= 0 or heads % kv_heads:
                raise ValueError(
                    f"layer {index}: num_heads must be divisible by num_kv_heads"
                )
            if attention.kind not in {"mha", "mqa", "gqa", "mla"}:
                raise ValueError(f"layer {index}: unsupported attention kind {attention.kind!r}")
            if attention.kind == "mha" and kv_heads != heads:
                raise ValueError(f"layer {index}: mha requires num_kv_heads == num_heads")
            if attention.kind == "mqa" and kv_heads != 1:
                raise ValueError(f"layer {index}: mqa requires num_kv_heads == 1")
            if attention.kind == "mla" and attention.kv_lora_rank <= 0:
                raise ValueError(f"layer {index}: mla requires a positive kv_lora_rank")
            if attention.mla_cache_mode not in {"kv", "latent"}:
                raise ValueError(
                    f"layer {index}: mla_cache_mode must be kv or latent"
                )
            if attention.backend not in {
                "auto", "sdpa", "flash", "eager", "xformers",
                "transformer_engine", "te", "transformer-engine",
            }:
                raise ValueError(
                    f"layer {index}: unsupported attention backend {attention.backend!r}"
                )
            if attention.rope_theta <= 0:
                raise ValueError(f"layer {index}: rope_theta must be positive")
            if isinstance(attention.sliding_window, bool) or not isinstance(
                attention.sliding_window, int
            ) or attention.sliding_window < 0:
                raise ValueError(
                    f"layer {index}: sliding_window must be a non-negative integer"
                )
            if attention.rope_scaling is not None:
                if not isinstance(attention.rope_scaling, dict):
                    raise TypeError(f"layer {index}: rope_scaling must be a mapping")
                scaling_type = attention.rope_scaling.get(
                    "type", attention.rope_scaling.get("rope_type")
                )
                factor = attention.rope_scaling.get("factor")
                if scaling_type != "linear" or not isinstance(factor, (int, float)) or isinstance(factor, bool) or factor <= 0:
                    raise ValueError(
                        f"layer {index}: native rope_scaling currently supports "
                        "only {type: linear, factor: positive_number}"
                    )
            if ffn.kind not in {"swiglu", "moe"}:
                raise ValueError(f"layer {index}: unsupported ffn kind {ffn.kind!r}")
            if isinstance(ffn.intermediate_size, bool) or not isinstance(
                ffn.intermediate_size, int
            ):
                raise TypeError(f"layer {index}: intermediate_size must be an integer")
            if ffn.intermediate_size <= 0:
                raise ValueError(f"layer {index}: intermediate_size must be positive")
            if ffn.kind == "moe":
                for label, value in (
                    ("num_experts", ffn.num_experts),
                    ("experts_per_token", ffn.experts_per_token),
                    ("expert_intermediate_size", ffn.expert_intermediate_size),
                ):
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise TypeError(f"layer {index}: {label} must be an integer")
                if ffn.num_experts <= 0:
                    raise ValueError(f"layer {index}: moe requires num_experts > 0")
                if not 0 < ffn.experts_per_token <= ffn.num_experts:
                    raise ValueError(
                        f"layer {index}: experts_per_token must be in [1, num_experts]"
                    )
                if ffn.expert_intermediate_size < 0:
                    raise ValueError(
                        f"layer {index}: expert_intermediate_size must be non-negative"
                    )
            if residual.topology not in {"serial", "parallel"}:
                raise ValueError(
                    f"layer {index}: unsupported residual topology {residual.topology!r}"
                )
            if residual.initialization not in {"standard", "depth_scaled", "zero"}:
                raise ValueError(
                    f"layer {index}: unsupported residual initialization "
                    f"{residual.initialization!r}"
                )
            if not 0.0 <= residual.dropout < 1.0:
                raise ValueError(f"layer {index}: residual dropout must be in [0, 1)")
            if not isinstance(residual.learnable, bool):
                raise TypeError(f"layer {index}: residual learnable must be boolean")
            for scale_name, scale in (
                ("attention_scale", residual.attention_scale),
                ("ffn_scale", residual.ffn_scale),
            ):
                if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
                    raise ValueError(
                        f"layer {index}: residual {scale_name} must be finite"
                    )

    def expanded_layers(self) -> List[LayerBlueprint]:
        """Return one fully resolved blueprint per decoder layer."""

        if not self.layers:
            return [
                LayerBlueprint(
                    name=f"layer-{index}",
                    repeat=1,
                    attention=AttentionBlueprint(**asdict(self.attention)),
                    ffn=FFNBlueprint(**asdict(self.ffn)),
                    residual=ResidualBlueprint(**asdict(self.residual)),
                )
                for index in range(self.num_layers)
            ]
        expanded: List[LayerBlueprint] = []
        for segment in self.layers:
            for offset in range(segment.repeat):
                expanded.append(
                    LayerBlueprint(
                        name=segment.name or f"layer-{len(expanded)}",
                        repeat=1,
                        attention=AttentionBlueprint(**asdict(segment.attention or self.attention)),
                        ffn=FFNBlueprint(**asdict(segment.ffn or self.ffn)),
                        residual=ResidualBlueprint(**asdict(segment.residual or self.residual)),
                    )
                )
        return expanded

    def to_config_dict(self) -> Dict[str, Any]:
        """Return only the serializable architecture source of truth."""

        data = asdict(self)
        data["layers"] = [layer.to_dict() for layer in self.layers]
        return data

    def to_dict(self, include_analysis: bool = True) -> Dict:
        data = self.to_config_dict()
        if include_analysis:
            data["analysis"] = self.analyze()
        return data

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def analyze(self) -> Dict:
        total = self.estimate_total_params()
        active = self.estimate_active_params()
        cost = self.cost_profile()
        return {
            "total_params": total,
            "active_params": active,
            "total_params_human": self._human_number(total),
            "active_params_human": self._human_number(active),
            "chinchilla_tokens": active * 20,
            "flops_per_token": active * 2.0,
            "kv_cache_bytes_per_token": self.estimate_kv_cache_bytes_per_token(),
            "cost_profile": cost,
            "architecture_axes": self.architecture_axes(),
            "risk_report": self.risk_report(),
            "training_hypotheses": self.training_hypotheses(),
            "multimodal": self.multimodal_analysis(),
        }

    def to_model_spec(self):
        """Convert to the existing ModelSpec format for scratch experiments."""
        from .ModelRegistry import ModelSpec

        return ModelSpec(
            name=self.name,
            architecture=self.family,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.attention.num_heads,
            num_kv_heads=self.attention.num_kv_heads if self.attention.kind != "mha" else 0,
            intermediate_size=self.ffn.intermediate_size,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.attention.rope_theta,
            norm_eps=self.norm_eps,
            tie_word_embeddings=self.tie_word_embeddings,
            use_sliding_window=self.attention.sliding_window > 0,
            sliding_window=self.attention.sliding_window,
            num_experts=self.ffn.num_experts,
            num_experts_per_tok=self.ffn.experts_per_token,
            expert_intermediate_size=self.ffn.expert_intermediate_size,
            router_aux_loss_coef=self.ffn.router_aux_loss_coef,
            shared_expert=self.ffn.shared_expert,
            kv_lora_rank=self.attention.kv_lora_rank,
            q_lora_rank=self.attention.q_lora_rank,
            multi_token_prediction=self.objective.multi_token_prediction,
            num_mtp_layers=self.objective.mtp_extra_tokens,
        )

    def to_transformers_config_dict(self) -> Dict:
        """Return a runnable HF-style config dict for dense fallback models."""
        return {
            "model_type": "qwen2" if self.family.startswith("qwen") else "llama",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.ffn.intermediate_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.attention.num_heads,
            "num_key_value_heads": self.attention.num_kv_heads or self.attention.num_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.attention.rope_theta,
            "rms_norm_eps": self.norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": False,
            "use_sliding_window": self.attention.sliding_window > 0,
            "sliding_window": self.attention.sliding_window or None,
            "saddle_blueprint": {
                "attention_kind": self.attention.kind,
                "ffn_kind": self.ffn.kind,
                "q_lora_rank": self.attention.q_lora_rank,
                "kv_lora_rank": self.attention.kv_lora_rank,
                "multi_token_prediction": self.objective.multi_token_prediction,
                "mtp_extra_tokens": self.objective.mtp_extra_tokens,
            },
        }

    def to_saddle_config(self):
        """Convert this blueprint into SaddleLLM's modular model config."""
        from .SaddleModeling import SaddleModelConfig

        return SaddleModelConfig.from_blueprint(self)

    def build_model(self):
        """Instantiate a modular PyTorch SaddleForCausalLM."""
        from .SaddleModeling import SaddleForCausalLM

        return SaddleForCausalLM.from_blueprint(self)

    def is_multimodal(self) -> bool:
        return bool(self.vision.enabled)

    def build_multimodal_model(self, image_token_id: Optional[int] = None):
        """Instantiate a LLaVA-style multimodal model."""
        from ..multimodal.MultimodalProjector import MultimodalProjectorConfig
        from ..multimodal.MultimodalModeling import MultimodalForCausalLM
        from ..multimodal.VisionBackbones import VisionBackboneConfig

        vision_config = VisionBackboneConfig(
            name=self.vision.backbone,
            image_size=self.vision.image_size,
            patch_size=self.vision.patch_size,
            hidden_size=self.vision.hidden_size,
            output_tokens=self.vision.output_tokens,
            freeze=self.vision.freeze,
            pretrained_name_or_path=self.vision.pretrained_name_or_path,
        )
        projector_config = MultimodalProjectorConfig(
            kind=self.projector.kind,
            vision_hidden_size=self.vision.hidden_size,
            llm_hidden_size=self.hidden_size,
            intermediate_size=self.projector.intermediate_size,
            num_query_tokens=self.projector.num_query_tokens,
            dropout=self.projector.dropout,
        )
        return MultimodalForCausalLM.from_text_blueprint(
            self,
            vision_config=vision_config,
            projector_config=projector_config,
            image_token_id=image_token_id,
        )

    def estimate_total_params(self) -> int:
        total, _ = self._estimate_parameter_counts()
        return total

    def estimate_active_params(self) -> int:
        _, active = self._estimate_parameter_counts()
        return active

    def _estimate_parameter_counts(self) -> tuple[int, int]:
        """Estimate heterogeneous decoder parameters and active parameters."""

        hidden = self.hidden_size
        embedding = self.vocab_size * hidden
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * hidden
        total = embedding + lm_head + hidden
        active = embedding + lm_head + hidden
        for layer in self.expanded_layers():
            attention = layer.attention or self.attention
            ffn = layer.ffn or self.ffn
            heads = attention.num_heads
            kv_heads = attention.num_kv_heads or heads
            head_dim = attention.head_dim or hidden // heads
            if attention.kind == "mla" and attention.kv_lora_rank > 0:
                q_rank = attention.q_lora_rank or max(1, hidden // 8)
                attention_params = (
                    hidden * attention.kv_lora_rank
                    + attention.kv_lora_rank * (kv_heads + heads) * head_dim
                    + hidden * q_rank
                    + q_rank * heads * head_dim
                    + heads * head_dim * hidden
                )
            else:
                attention_params = (
                    hidden * heads * head_dim
                    + hidden * kv_heads * head_dim * 2
                    + heads * head_dim * hidden
                )
            if ffn.kind == "moe":
                expert_hidden = ffn.expert_intermediate_size or ffn.intermediate_size
                per_expert = 3 * hidden * expert_hidden
                shared = per_expert if ffn.shared_expert else 0
                router = hidden * ffn.num_experts
                total_ffn = per_expert * ffn.num_experts + shared + router
                active_ffn = (
                    per_expert * ffn.experts_per_token + shared + router
                )
            else:
                total_ffn = active_ffn = 3 * hidden * ffn.intermediate_size
            residual_parameters = 2 if (layer.residual or self.residual).learnable else 0
            norms = 2 * hidden
            total += attention_params + total_ffn + norms + residual_parameters
            active += attention_params + active_ffn + norms + residual_parameters
        mtp = len(range(max(0, self.objective.mtp_extra_tokens))) * hidden * self.vocab_size
        return total + mtp, active + mtp

    def estimate_kv_cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        total = 0
        for layer in self.expanded_layers():
            attention = layer.attention or self.attention
            head_dim = attention.head_dim or self.hidden_size // max(
                1, attention.num_heads
            )
            kv_heads = attention.num_kv_heads or attention.num_heads
            if attention.kind == "mla" and attention.kv_lora_rank > 0:
                total += attention.kv_lora_rank * dtype_bytes
            else:
                total += 2 * kv_heads * head_dim * dtype_bytes
        return total

    def estimate_training_flops(self, token_budget: int) -> float:
        """Rough dense-equivalent training FLOPs for architecture comparisons."""
        objective_multiplier = 1.0
        if self.objective.multi_token_prediction and self.objective.mtp_extra_tokens > 0:
            objective_multiplier += min(0.30, 0.08 * self.objective.mtp_extra_tokens)
        return 6.0 * float(self.estimate_active_params()) * float(token_budget) * objective_multiplier

    def estimate_kv_cache_gb(
        self,
        batch_size: int = 1,
        sequence_length: Optional[int] = None,
        dtype_bytes: int = 2,
    ) -> float:
        seq = sequence_length or self.max_position_embeddings
        cache_bytes = self.estimate_kv_cache_bytes_per_token(dtype_bytes=dtype_bytes)
        total_bytes = cache_bytes * max(1, int(batch_size)) * max(1, int(seq))
        return total_bytes / (1024 ** 3)

    def cost_profile(
        self,
        token_budget: Optional[int] = None,
        batch_size: int = 1,
        sequence_length: Optional[int] = None,
        dtype_bytes: int = 2,
    ) -> Dict:
        tokens = int(token_budget or self.estimate_active_params() * 20)
        train_flops = self.estimate_training_flops(tokens)
        kv_gb = self.estimate_kv_cache_gb(
            batch_size=batch_size,
            sequence_length=sequence_length or self.max_position_embeddings,
            dtype_bytes=dtype_bytes,
        )
        params_gb_bf16 = self.estimate_total_params() * dtype_bytes / (1024 ** 3)
        active_ratio = self.estimate_active_params() / max(1, self.estimate_total_params())
        return {
            "token_budget": tokens,
            "training_flops": train_flops,
            "training_peta_flops": train_flops / 1e15,
            "kv_cache_gb": round(kv_gb, 4),
            "params_gb_bf16": round(params_gb_bf16, 4),
            "active_to_total_ratio": round(active_ratio, 4),
            "assumptions": {
                "training_flops_formula": "6 * active_params * tokens, with a small MTP objective multiplier",
                "dtype_bytes": dtype_bytes,
                "batch_size": batch_size,
                "sequence_length": sequence_length or self.max_position_embeddings,
            },
        }

    def architecture_axes(self) -> Dict:
        expanded = self.expanded_layers()
        attention_kinds = list(dict.fromkeys(
            (layer.attention or self.attention).kind for layer in expanded
        ))
        ffn_kinds = list(dict.fromkeys(
            (layer.ffn or self.ffn).kind for layer in expanded
        ))
        return {
            "attention": attention_kinds[0] if len(attention_kinds) == 1 else "hybrid",
            "attention_kinds": attention_kinds,
            "ffn": ffn_kinds[0] if len(ffn_kinds) == 1 else "hybrid",
            "ffn_kinds": ffn_kinds,
            "sparsity": "sparse_moe" if "moe" in ffn_kinds else "dense",
            "context": "long" if self.max_position_embeddings >= 32768 else "standard",
            "objective": "mtp" if self.objective.multi_token_prediction else "next_token",
            "init": self.initializer,
            "modality": "vision_language" if self.vision.enabled else "text",
        }

    def multimodal_analysis(self) -> Dict:
        if not self.vision.enabled:
            return {"enabled": False}
        image_tokens = self.vision.output_tokens or (self.vision.image_size // max(1, self.vision.patch_size)) ** 2
        return {
            "enabled": True,
            "vision_backbone": self.vision.backbone,
            "projector": self.projector.kind,
            "image_tokens": image_tokens,
            "projector_trainable_by_default": True,
            "vision_frozen": self.vision.freeze,
        }

    def risk_report(self) -> List[str]:
        risks: List[str] = []
        expanded = self.expanded_layers()
        ffn_kinds = {(layer.ffn or self.ffn).kind for layer in expanded}
        attention_kinds = {
            (layer.attention or self.attention).kind for layer in expanded
        }
        if "moe" in ffn_kinds:
            risks.append("MoE needs router stability, expert load monitoring, and more tokens than dense smoke tests.")
        if "mla" in attention_kinds:
            risks.append("MLA uses SaddleLLM's native model path; validate native checkpoint consumers separately from Hugging Face AutoModel workflows.")
        if self.objective.multi_token_prediction:
            risks.append("MTP changes loss accounting and checkpoint compatibility; evaluate quality and speed separately.")
        if self.max_position_embeddings >= 32768:
            risks.append("Long context requires attention kernel, RoPE schedule, and eval coverage; memory grows quickly.")
        if self.vision.enabled:
            risks.append("VLM training needs image preprocessing, projector warmup, visual instruction tuning, and hallucination eval.")
        if not risks:
            risks.append("Dense GQA blueprint is low-risk and suitable for first scratch runs.")
        return risks

    def training_hypotheses(self) -> List[str]:
        ideas = []
        expanded = self.expanded_layers()
        ffn_kinds = {(layer.ffn or self.ffn).kind for layer in expanded}
        attention_kinds = {
            (layer.attention or self.attention).kind for layer in expanded
        }
        if "moe" in ffn_kinds:
            ideas.append("Measure active-parameter quality against dense models at matched FLOPs.")
            ideas.append("Track expert entropy, dropped tokens, and per-expert token counts.")
        if "mla" in attention_kinds:
            ideas.append("Compare KV-cache savings and long-context perplexity against GQA.")
        if self.objective.multi_token_prediction:
            ideas.append("Compare sample efficiency at fixed token budget and fixed wall-clock budget.")
        if self.max_position_embeddings >= 32768:
            ideas.append("Run short-context regression eval before and after long-context extension.")
        if self.vision.enabled:
            ideas.append("Warm up projector with frozen vision encoder and frozen LLM before visual instruction tuning.")
            ideas.append("Evaluate OCR, VQA, chart/document QA, and visual hallucination separately from text eval.")
        if not ideas:
            ideas.append("Use this dense baseline as the control arm for architecture experiments.")
        return ideas

    @staticmethod
    def _human_number(value: int) -> str:
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        return str(value)


class ModelBlueprintLab:
    """Factory for model-first architecture experiments."""

    @staticmethod
    def dense_gqa(
        name: str = "saddle-dense-gqa-300m",
        hidden_size: int = 1024,
        layers: int = 24,
        heads: int = 16,
        kv_heads: int = 4,
        vocab_size: int = 32000,
        seq_length: int = 4096,
    ) -> ModelBlueprint:
        return ModelBlueprint(
            name=name,
            family="qwen",
            hidden_size=hidden_size,
            num_layers=layers,
            vocab_size=vocab_size,
            max_position_embeddings=seq_length,
            attention=AttentionBlueprint(kind="gqa", num_heads=heads, num_kv_heads=kv_heads),
            ffn=FFNBlueprint(kind="swiglu", intermediate_size=hidden_size * 4),
            notes=["Dense GQA baseline: use as control arm before trying sparse or MLA variants."],
        )

    @staticmethod
    def deepseek_style_moe(
        name: str = "saddle-deepseek-style-moe",
        hidden_size: int = 1536,
        layers: int = 24,
        heads: int = 24,
        kv_heads: int = 4,
        experts: int = 16,
        experts_per_token: int = 2,
        seq_length: int = 8192,
    ) -> ModelBlueprint:
        return ModelBlueprint(
            name=name,
            family="deepseek",
            hidden_size=hidden_size,
            num_layers=layers,
            max_position_embeddings=seq_length,
            initializer="deepseek",
            attention=AttentionBlueprint(
                kind="mla",
                num_heads=heads,
                num_kv_heads=kv_heads,
                q_lora_rank=max(128, hidden_size // 8),
                kv_lora_rank=max(64, hidden_size // 16),
                rope_theta=1000000.0,
            ),
            ffn=FFNBlueprint(
                kind="moe",
                intermediate_size=hidden_size * 4,
                num_experts=experts,
                experts_per_token=experts_per_token,
                expert_intermediate_size=hidden_size * 2,
                shared_expert=True,
                aux_loss_free=True,
            ),
            objective=ObjectiveBlueprint(multi_token_prediction=True, mtp_extra_tokens=3),
            notes=["Research blueprint for MLA + aux-loss-free MoE + MTP experiments."],
        )

    @staticmethod
    def minimax_style_long_context(
        name: str = "saddle-minimax-style-long-context",
        hidden_size: int = 1024,
        layers: int = 24,
        heads: int = 16,
        seq_length: int = 65536,
    ) -> ModelBlueprint:
        return ModelBlueprint(
            name=name,
            family="minimax",
            hidden_size=hidden_size,
            num_layers=layers,
            max_position_embeddings=seq_length,
            attention=AttentionBlueprint(
                kind="gqa",
                num_heads=heads,
                num_kv_heads=max(1, heads // 4),
                rope_theta=1000000.0,
                rope_scaling={"type": "longrope", "factor": max(1, seq_length // 4096)},
            ),
            ffn=FFNBlueprint(kind="swiglu", intermediate_size=hidden_size * 4),
            notes=["Long-context dense baseline; custom linear/flash attention can replace GQA later."],
        )

    @staticmethod
    def glm_style_reasoning(
        name: str = "saddle-glm-style-reasoning",
        hidden_size: int = 1024,
        layers: int = 24,
        heads: int = 16,
        seq_length: int = 8192,
    ) -> ModelBlueprint:
        return ModelBlueprint(
            name=name,
            family="glm",
            hidden_size=hidden_size,
            num_layers=layers,
            max_position_embeddings=seq_length,
            attention=AttentionBlueprint(kind="gqa", num_heads=heads, num_kv_heads=max(1, heads // 4)),
            ffn=FFNBlueprint(kind="swiglu", intermediate_size=hidden_size * 4),
            objective=ObjectiveBlueprint(
                reasoning_trace_policy="mixed",
                domain_objectives=["instruction_following", "reasoning_trace_mixture"],
            ),
            notes=["Reasoning-oriented dense architecture baseline with GLM-style data/objective emphasis."],
        )

    @staticmethod
    def llava_style_vlm(
        name: str = "saddle-llava-style-vlm",
        hidden_size: int = 1024,
        layers: int = 24,
        heads: int = 16,
        kv_heads: int = 4,
        vocab_size: int = 32000,
        seq_length: int = 4096,
        vision_backbone: str = "tiny_patch",
        projector: str = "mlp",
        image_tokens: int = 64,
    ) -> ModelBlueprint:
        return ModelBlueprint(
            name=name,
            family="qwen-vl",
            hidden_size=hidden_size,
            num_layers=layers,
            vocab_size=vocab_size,
            max_position_embeddings=seq_length,
            attention=AttentionBlueprint(kind="gqa", num_heads=heads, num_kv_heads=kv_heads),
            ffn=FFNBlueprint(kind="swiglu", intermediate_size=hidden_size * 4),
            vision=VisionBlueprint(
                enabled=True,
                backbone=vision_backbone,
                hidden_size=768,
                output_tokens=image_tokens,
                freeze=True,
            ),
            projector=ProjectorBlueprint(kind=projector, num_query_tokens=image_tokens),
            objective=ObjectiveBlueprint(domain_objectives=["visual_instruction_following", "vqa", "ocr"]),
            notes=["LLaVA-style VLM: frozen vision encoder + projector + Saddle causal decoder."],
        )

    @staticmethod
    def compare(blueprints: List[ModelBlueprint]) -> Dict:
        return {
            "models": [
                {
                    "name": bp.name,
                    "family": bp.family,
                    **bp.analyze(),
                }
                for bp in blueprints
            ]
        }

    @staticmethod
    def save_comparison(blueprints: List[ModelBlueprint], path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ModelBlueprintLab.compare(blueprints), f, ensure_ascii=False, indent=2)
        return path
