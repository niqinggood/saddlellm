"""Code-first, composable construction of SaddleLLM decoder models.

Configuration files and :class:`~saddlellm.models.ModelBlueprint.ModelBlueprint`
remain the serializable source of truth.  This module adds a small fluent API
for experiments that are easier to express in Python, while deliberately
building the same blueprint objects before a model is instantiated.

Example::

    builder = (
        SaddleModelBuilder(
            "tiny-hybrid",
            hidden_size=256,
            vocab_size=8_000,
            max_position_embeddings=2_048,
        )
        .defaults(
            attention={"preset": "gqa", "num_heads": 8, "num_kv_heads": 2},
            ffn={"preset": "swiglu", "intermediate_size": 768},
            residual={"preset": "depth_scaled"},
        )
        .add_layers(4)
        .add_layers(
            2,
            name="moe-tail",
            ffn={
                "preset": "moe",
                "num_experts": 8,
                "experts_per_token": 2,
                "expert_intermediate_size": 384,
            },
            residual="parallel",
        )
        .objective(multi_token_prediction=True, mtp_extra_tokens=2)
    )
    blueprint = builder.build_blueprint()
    model = builder.build()
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from .ModelBlueprint import (
    AttentionBlueprint,
    FFNBlueprint,
    LayerBlueprint,
    ModelBlueprint,
    ObjectiveBlueprint,
    ResidualBlueprint,
)


ComponentBlueprint = Union[
    AttentionBlueprint,
    FFNBlueprint,
    ResidualBlueprint,
]
ComponentInput = Union[ComponentBlueprint, Mapping[str, Any], str]
PresetFactory = Callable[[], Union[ComponentBlueprint, Mapping[str, Any]]]
PresetValue = Union[ComponentBlueprint, Mapping[str, Any], PresetFactory]


class ModelComponentRegistry:
    """Registry of reusable attention, FFN and residual presets.

    A preset may be a blueprint dataclass, a mapping, or a zero-argument
    factory returning either form.  Objects are copied on creation so callers
    can safely customize a returned component without mutating the preset.
    Preset names are case-insensitive.
    """

    _COMPONENT_TYPES = {
        "attention": AttentionBlueprint,
        "ffn": FFNBlueprint,
        "residual": ResidualBlueprint,
    }

    def __init__(self, include_builtins: bool = True) -> None:
        self._presets: Dict[str, Dict[str, PresetValue]] = {
            component: {} for component in self._COMPONENT_TYPES
        }
        if include_builtins:
            self._register_builtins()

    @staticmethod
    def _normalize_name(name: str, label: str = "preset") -> str:
        normalized = str(name).strip().lower()
        if not normalized:
            raise ValueError(f"{label} name must not be empty")
        return normalized

    @classmethod
    def _normalize_component(cls, component: str) -> str:
        normalized = str(component).strip().lower()
        if normalized not in cls._COMPONENT_TYPES:
            choices = ", ".join(sorted(cls._COMPONENT_TYPES))
            raise ValueError(
                f"unsupported component type {component!r}; expected one of: {choices}"
            )
        return normalized

    def register(
        self,
        component: str,
        name: str,
        preset: PresetValue,
        *,
        overwrite: bool = False,
    ) -> "ModelComponentRegistry":
        """Register one component preset and return the registry for chaining."""

        component_name = self._normalize_component(component)
        preset_name = self._normalize_name(name)
        if not callable(preset) and not isinstance(preset, Mapping):
            expected_type = self._COMPONENT_TYPES[component_name]
            if not isinstance(preset, expected_type):
                raise TypeError(
                    f"{component_name} preset must be {expected_type.__name__}, "
                    "a mapping, or a zero-argument factory"
                )
        if preset_name in self._presets[component_name] and not overwrite:
            raise ValueError(
                f"{component_name} preset {preset_name!r} is already registered"
            )
        self._presets[component_name][preset_name] = preset
        return self

    def register_attention(
        self,
        name: str,
        preset: PresetValue,
        *,
        overwrite: bool = False,
    ) -> "ModelComponentRegistry":
        return self.register("attention", name, preset, overwrite=overwrite)

    def register_ffn(
        self,
        name: str,
        preset: PresetValue,
        *,
        overwrite: bool = False,
    ) -> "ModelComponentRegistry":
        return self.register("ffn", name, preset, overwrite=overwrite)

    def register_residual(
        self,
        name: str,
        preset: PresetValue,
        *,
        overwrite: bool = False,
    ) -> "ModelComponentRegistry":
        return self.register("residual", name, preset, overwrite=overwrite)

    def create(self, component: str, name: str, **overrides: Any) -> ComponentBlueprint:
        """Create a fresh component from a named preset plus field overrides."""

        component_name = self._normalize_component(component)
        preset_name = self._normalize_name(name)
        try:
            preset = self._presets[component_name][preset_name]
        except KeyError as exc:
            available = ", ".join(self.available(component_name)) or "<none>"
            raise KeyError(
                f"unknown {component_name} preset {preset_name!r}; available: {available}"
            ) from exc

        value = preset() if callable(preset) else preset
        if isinstance(value, Mapping):
            values = deepcopy(dict(value))
        elif is_dataclass(value):
            values = deepcopy(asdict(value))
        else:
            raise TypeError(
                f"factory for {component_name} preset {preset_name!r} returned "
                f"unsupported value {type(value).__name__}"
            )
        values.update(deepcopy(overrides))
        return self._construct(component_name, values)

    def create_attention(self, name: str, **overrides: Any) -> AttentionBlueprint:
        return self.create("attention", name, **overrides)  # type: ignore[return-value]

    def create_ffn(self, name: str, **overrides: Any) -> FFNBlueprint:
        return self.create("ffn", name, **overrides)  # type: ignore[return-value]

    def create_residual(self, name: str, **overrides: Any) -> ResidualBlueprint:
        return self.create("residual", name, **overrides)  # type: ignore[return-value]

    def available(self, component: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
        """List preset names for one component, or all components."""

        if component is not None:
            component_name = self._normalize_component(component)
            return sorted(self._presets[component_name])
        return {
            name: sorted(presets)
            for name, presets in self._presets.items()
        }

    @classmethod
    def _construct(
        cls, component: str, values: Mapping[str, Any]
    ) -> ComponentBlueprint:
        payload = deepcopy(dict(values))
        if component == "residual":
            # Keep code-first and configuration aliases (layout, init, gate,
            # scale) exactly aligned.
            return ModelBlueprint._residual_from_dict(payload)
        component_type = cls._COMPONENT_TYPES[component]
        return component_type(**payload)

    def _register_builtins(self) -> None:
        attention_presets = {
            "mha": {"kind": "mha"},
            "mqa": {"kind": "mqa"},
            "gqa": {"kind": "gqa"},
            "mla": {"kind": "mla", "kv_lora_rank": 64},
        }
        ffn_presets = {
            "swiglu": {"kind": "swiglu"},
            "dense": {"kind": "swiglu"},
            "moe": {
                "kind": "moe",
                "num_experts": 8,
                "experts_per_token": 2,
                "expert_intermediate_size": 2048,
            },
        }
        residual_presets = {
            "serial": {"topology": "serial"},
            "sequential": {"topology": "serial"},
            "parallel": {"topology": "parallel"},
            "depth_scaled": {
                "topology": "serial",
                "initialization": "depth_scaled",
            },
            "zero": {"topology": "serial", "initialization": "zero"},
            "learnable": {"topology": "serial", "learnable": True},
            "rezero": {
                "topology": "serial",
                "attention_scale": 0.0,
                "ffn_scale": 0.0,
                "learnable": True,
            },
        }
        for name, preset in attention_presets.items():
            self.register_attention(name, preset)
        for name, preset in ffn_presets.items():
            self.register_ffn(name, preset)
        for name, preset in residual_presets.items():
            self.register_residual(name, preset)


class SaddleModelBuilder:
    """Fluent Python builder for a validated :class:`ModelBlueprint`.

    Layer segments added with :meth:`add_layers` are authoritative: their
    repeat counts determine the final number of decoder layers.  If no segment
    is added, ``num_layers`` from the constructor/configuration is used for a
    homogeneous model.
    """

    _RESERVED_MODEL_FIELDS = {"attention", "ffn", "residual", "layers", "objective"}
    _MODEL_FIELDS = {item.name for item in fields(ModelBlueprint)}

    def __init__(
        self,
        name: str,
        *,
        family: str = "llama",
        hidden_size: int = 1024,
        num_layers: int = 24,
        vocab_size: int = 32000,
        max_position_embeddings: int = 4096,
        registry: Optional[ModelComponentRegistry] = None,
        **model_options: Any,
    ) -> None:
        self.registry = registry or ModelComponentRegistry()
        self._model_options: Dict[str, Any] = {
            "name": name,
            "family": family,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_position_embeddings,
        }
        self._attention = AttentionBlueprint()
        self._ffn = FFNBlueprint()
        self._residual = ResidualBlueprint()
        self._objective = ObjectiveBlueprint()
        self._layers: List[LayerBlueprint] = []
        if model_options:
            self.configure(**model_options)

    def configure(self, **model_options: Any) -> "SaddleModelBuilder":
        """Set model-level blueprint fields such as norm or initializer."""

        unknown = set(model_options) - self._MODEL_FIELDS
        if unknown:
            raise TypeError(f"unknown model option(s): {', '.join(sorted(unknown))}")
        reserved = set(model_options) & self._RESERVED_MODEL_FIELDS
        if reserved:
            raise TypeError(
                "use defaults(), add_layers(), or objective() for: "
                + ", ".join(sorted(reserved))
            )
        self._model_options.update(deepcopy(model_options))
        return self

    def defaults(
        self,
        *,
        attention: Optional[ComponentInput] = None,
        ffn: Optional[ComponentInput] = None,
        residual: Optional[ComponentInput] = None,
    ) -> "SaddleModelBuilder":
        """Set components inherited by layer segments that omit overrides.

        Mappings are merged with the current default.  A mapping containing a
        ``preset`` key, or a string value, replaces it with a registry preset.
        Dataclass values are copied and used as-is.
        """

        if attention is not None:
            self._attention = self._resolve_component(
                "attention", attention, base=self._attention
            )  # type: ignore[assignment]
        if ffn is not None:
            self._ffn = self._resolve_component(
                "ffn", ffn, base=self._ffn
            )  # type: ignore[assignment]
        if residual is not None:
            self._residual = self._resolve_component(
                "residual", residual, base=self._residual
            )  # type: ignore[assignment]
        return self

    def add_layers(
        self,
        repeat: int = 1,
        *,
        name: str = "",
        attention: Optional[ComponentInput] = None,
        ffn: Optional[ComponentInput] = None,
        residual: Optional[ComponentInput] = None,
    ) -> "SaddleModelBuilder":
        """Append a repeatable decoder-layer segment."""

        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
            raise ValueError("repeat must be a positive integer")
        layer_attention = (
            self._resolve_component("attention", attention, base=self._attention)
            if attention is not None
            else None
        )
        layer_ffn = (
            self._resolve_component("ffn", ffn, base=self._ffn)
            if ffn is not None
            else None
        )
        layer_residual = (
            self._resolve_component("residual", residual, base=self._residual)
            if residual is not None
            else None
        )
        self._layers.append(
            LayerBlueprint(
                name=str(name),
                repeat=repeat,
                attention=layer_attention,  # type: ignore[arg-type]
                ffn=layer_ffn,  # type: ignore[arg-type]
                residual=layer_residual,  # type: ignore[arg-type]
            )
        )
        return self

    def add_layer(
        self,
        *,
        name: str = "",
        attention: Optional[ComponentInput] = None,
        ffn: Optional[ComponentInput] = None,
        residual: Optional[ComponentInput] = None,
    ) -> "SaddleModelBuilder":
        """Convenience form of :meth:`add_layers` with ``repeat=1``."""

        return self.add_layers(
            1,
            name=name,
            attention=attention,
            ffn=ffn,
            residual=residual,
        )

    def objective(
        self,
        value: Optional[Union[ObjectiveBlueprint, Mapping[str, Any]]] = None,
        **overrides: Any,
    ) -> "SaddleModelBuilder":
        """Set or update the training objective blueprint."""

        if value is None:
            payload = asdict(self._objective)
        elif isinstance(value, ObjectiveBlueprint):
            payload = asdict(value)
        elif isinstance(value, Mapping):
            payload = {**asdict(self._objective), **deepcopy(dict(value))}
        else:
            raise TypeError("objective must be ObjectiveBlueprint, a mapping, or None")
        payload.update(deepcopy(overrides))
        self._objective = ObjectiveBlueprint(**payload)
        return self

    def build_blueprint(self) -> ModelBlueprint:
        """Build and validate the serializable architecture blueprint."""

        payload = deepcopy(self._model_options)
        payload.update(
            {
                "attention": asdict(self._attention),
                "ffn": asdict(self._ffn),
                "residual": asdict(self._residual),
                "objective": asdict(self._objective),
                "layers": [layer.to_dict() for layer in self._layers],
            }
        )
        # ModelBlueprint.from_dict normalizes component aliases, expands
        # defaults into layer segments, and runs ModelBlueprint.validate().
        return ModelBlueprint.from_dict(payload)

    def build(self):
        """Instantiate :class:`SaddleForCausalLM` from the validated blueprint."""

        return self.build_blueprint().build_model()

    def to_dict(self) -> Dict[str, Any]:
        """Return the same configuration mapping accepted by file-based use."""

        return self.build_blueprint().to_config_dict()

    def _resolve_component(
        self,
        component: str,
        value: ComponentInput,
        *,
        base: Optional[ComponentBlueprint] = None,
    ) -> ComponentBlueprint:
        expected_type = ModelComponentRegistry._COMPONENT_TYPES[component]
        if isinstance(value, str):
            return self.registry.create(component, value)
        if isinstance(value, expected_type):
            return ModelComponentRegistry._construct(component, asdict(value))
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{component} must be {expected_type.__name__}, a mapping, or a preset name"
            )

        payload = deepcopy(dict(value))
        preset = payload.pop("preset", None)
        if preset is not None:
            return self.registry.create(component, str(preset), **payload)
        if component == "residual":
            # Canonicalize aliases before inherited defaults are merged, so a
            # layer-level ``layout: parallel`` legitimately overrides the
            # default ``topology: serial`` instead of looking like a conflict.
            payload = ModelBlueprint._residual_values(payload)
        merged = asdict(base) if base is not None else {}
        merged.update(payload)
        return ModelComponentRegistry._construct(component, merged)


__all__ = ["ModelComponentRegistry", "SaddleModelBuilder"]
