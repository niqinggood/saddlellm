"""Stable Stage contract and built-in orchestrator Stage registry."""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
import threading
from typing import Any, Callable, FrozenSet, Mapping, MutableMapping, Optional, Protocol, runtime_checkable

from .registry import PluginRegistry, PluginRegistryError


SINGLE_PROCESS_STRATEGIES = frozenset({"single"})
PRETRAIN_STRATEGIES = frozenset(
    {"single", "ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"}
)
STAGE_ENTRY_POINT_GROUP = "saddlellm.stages"


@dataclass(frozen=True)
class StageCapabilities:
    """Execution constraints declared before a Stage is scheduled."""

    distributed_strategies: FrozenSet[str] = SINGLE_PROCESS_STRATEGIES
    uses_model_adapter: bool = False
    tags: FrozenSet[str] = field(default_factory=frozenset)
    consumes: FrozenSet[str] = field(default_factory=frozenset)
    produces: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        strategies = frozenset(
            str(item).strip().lower() for item in self.distributed_strategies
        )
        if not strategies:
            raise ValueError("distributed_strategies must not be empty")
        object.__setattr__(self, "distributed_strategies", strategies)
        object.__setattr__(
            self,
            "consumes",
            frozenset(str(item).strip().lower() for item in self.consumes if str(item).strip()),
        )
        object.__setattr__(
            self,
            "produces",
            frozenset(str(item).strip().lower() for item in self.produces if str(item).strip()),
        )
        object.__setattr__(
            self,
            "tags",
            frozenset(str(item).strip() for item in self.tags if str(item).strip()),
        )

    def supports_strategy(self, strategy: str) -> bool:
        return str(strategy).strip().lower() in self.distributed_strategies


@dataclass
class StageContext:
    """Runtime-owned state exposed to a Stage plugin.

    Plugins should prefer ``config``, ``output_dir``, ``results`` and named
    services over importing the concrete orchestrator.  The built-in migration
    adapter uses the ``orchestrator`` service to invoke existing private stage
    methods without changing their behavior.
    """

    name: str
    config: Any
    output_dir: Path
    results: MutableMapping[str, Any]
    distributed: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    services: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Any = None

    def service(self, name: str) -> Any:
        try:
            return self.services[name]
        except KeyError as exc:
            raise KeyError(f"Stage service {name!r} is unavailable") from exc

    def artifact(self, name: str) -> Any:
        """Resolve a previously published artifact by its stable name."""

        if self.artifacts is None:
            raise KeyError(f"Stage artifact {name!r} is unavailable")
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise KeyError(f"Stage artifact {name!r} is unavailable") from exc


@runtime_checkable
class StagePlugin(Protocol):
    """Protocol implemented by independently distributable Stage plugins."""

    name: str
    capabilities: StageCapabilities

    def validate(self, context: StageContext) -> None:
        """Validate configuration without allocating models or datasets."""

    def run(self, context: StageContext) -> Optional[Mapping[str, Any]]:
        """Execute the Stage and optionally return its JSON-compatible result."""


class BaseStage:
    """Convenient base class for third-party Stage implementations."""

    name = ""
    capabilities = StageCapabilities()

    def validate(self, context: StageContext) -> None:
        return None

    def run(self, context: StageContext) -> Optional[Mapping[str, Any]]:
        raise NotImplementedError


class FunctionStage(BaseStage):
    """Adapt a ``StageContext -> result`` callable to :class:`StagePlugin`."""

    def __init__(
        self,
        name: str,
        function: Callable[[StageContext], Optional[Mapping[str, Any]]],
        *,
        capabilities: Optional[StageCapabilities] = None,
    ):
        self.name = PluginRegistry.normalize_name(name)
        self.function = function
        self.capabilities = capabilities or StageCapabilities()

    def run(self, context: StageContext) -> Optional[Mapping[str, Any]]:
        return self.function(context)


class OrchestratorMethodStage(BaseStage):
    """Compatibility adapter around an existing orchestrator stage method."""

    def __init__(
        self,
        name: str,
        method_name: str,
        *,
        capabilities: StageCapabilities,
    ):
        self.name = name
        self.method_name = method_name
        self.capabilities = capabilities

    def run(self, context: StageContext) -> None:
        orchestrator = context.service("orchestrator")
        method = getattr(orchestrator, self.method_name, None)
        if not callable(method):
            raise AttributeError(
                f"Built-in Stage {self.name!r} requires orchestrator method "
                f"{self.method_name!r}"
            )
        method()
        # Existing methods already write the authoritative result key.  None
        # tells the scheduler not to duplicate or rename it during migration.
        return None


class StageRegistry(PluginRegistry[Any]):
    """Plugin registry that materializes and validates Stage providers."""

    def __init__(self, *, entry_point_group: Optional[str] = STAGE_ENTRY_POINT_GROUP):
        super().__init__(kind="training stage", entry_point_group=entry_point_group)

    def create(self, name: str) -> StagePlugin:
        normalized = self.normalize_name(name)
        provider = self.resolve(normalized)
        plugin: Any
        if inspect.isclass(provider):
            try:
                plugin = provider()
            except TypeError as exc:
                raise PluginRegistryError(
                    f"Stage class {provider!r} must have a zero-argument constructor"
                ) from exc
        elif callable(provider) and not hasattr(provider, "run"):
            plugin = FunctionStage(normalized, provider)
        else:
            plugin = provider

        plugin_name = self.normalize_name(getattr(plugin, "name", ""))
        if plugin_name != normalized:
            raise PluginRegistryError(
                f"Stage registered as {normalized!r} declares name {plugin_name!r}"
            )
        capabilities = getattr(plugin, "capabilities", None)
        if not isinstance(capabilities, StageCapabilities):
            raise PluginRegistryError(
                f"Stage {normalized!r} must declare StageCapabilities"
            )
        if not callable(getattr(plugin, "validate", None)):
            raise PluginRegistryError(f"Stage {normalized!r} must define validate(context)")
        if not callable(getattr(plugin, "run", None)):
            raise PluginRegistryError(f"Stage {normalized!r} must define run(context)")
        return plugin


_BUILTIN_METHODS = {
    "tokenizer": "_run_tokenizer_stage",
    "pretrain": "_run_pretrain_stage",
    "sft": "_run_sft_stage",
    "preference": "_run_preference_stage",
    "rlhf": "_run_rlhf_stage",
    "mopd": "_run_mopd_stage",
    "vla_sft": "_run_vla_stage",
    "mllm_sft": "_run_multimodal_stage",
    "vision_alignment": "_run_multimodal_stage",
    "media_cache": "_run_media_cache_stage",
    "image_generation": "_run_image_generation_stage",
    "music_generation": "_run_music_generation_stage",
    "video_generation": "_run_video_generation_stage",
    "world_model": "_run_world_model_stage",
    "eye_control": "_run_eye_control_stage",
    "eval": "_run_eval_stage",
    "export": "_run_export_stage",
    "operator": "_run_operator_stage",
}
BUILTIN_STAGE_NAMES = frozenset(_BUILTIN_METHODS)

_MODEL_ADAPTER_STAGES = frozenset(
    {
        "tokenizer",
        "pretrain",
        "sft",
        "preference",
        "rlhf",
        "mopd",
        "vla_sft",
        "mllm_sft",
        "vision_alignment",
        "eval",
        "export",
    }
)

_BUILTIN_CONTRACTS = {
    "tokenizer": ((), ("tokenizer",), "data"),
    "pretrain": (("dataset", "tokenizer"), ("language_model",), "foundation"),
    "sft": (("dataset", "language_model"), ("language_model",), "post_training"),
    "preference": (("dataset", "language_model"), ("language_model",), "post_training"),
    "rlhf": (("dataset", "language_model"), ("language_model",), "post_training"),
    "mopd": (("language_model", "prompts"), ("dataset",), "post_training"),
    "vla_sft": (("robot_trajectory",), ("control_policy",), "embodied_policy"),
    "mllm_sft": (("multimodal_dataset",), ("language_model",), "multimodal"),
    "vision_alignment": (("multimodal_dataset",), ("language_model",), "multimodal"),
    "media_cache": (("raw_media",), ("encoded_media",), "data"),
    "image_generation": (("encoded_media",), ("generative_model",), "generative"),
    "music_generation": (("encoded_media",), ("generative_model",), "generative"),
    "video_generation": (("encoded_media",), ("generative_model",), "generative"),
    "world_model": (("trajectory",), ("world_model",), "world_model"),
    "eye_control": (("world_model",), ("control_report",), "embodied_control"),
    "eval": (("model",), ("evaluation",), "evaluation"),
    "export": (("model",), ("release",), "release"),
    "operator": ((), ("artifact",), "extension"),
}


def register_builtin_stages(registry: StageRegistry) -> StageRegistry:
    for name, method_name in _BUILTIN_METHODS.items():
        strategies = PRETRAIN_STRATEGIES if name == "pretrain" else SINGLE_PROCESS_STRATEGIES
        consumes, produces, family = _BUILTIN_CONTRACTS[name]
        registry.register(
            name,
            OrchestratorMethodStage(
                name,
                method_name,
                capabilities=StageCapabilities(
                    distributed_strategies=strategies,
                    uses_model_adapter=name in _MODEL_ADAPTER_STAGES,
                    consumes=frozenset(consumes),
                    produces=frozenset(produces),
                    tags=frozenset({"builtin", f"family:{family}"}),
                ),
            ),
            source="builtin",
        )
    return registry


def create_stage_registry(
    *,
    include_builtins: bool = True,
    discover: bool = True,
) -> StageRegistry:
    registry = StageRegistry()
    if include_builtins:
        register_builtin_stages(registry)
    if discover:
        registry.discover()
    return registry


_DEFAULT_STAGE_REGISTRY: Optional[StageRegistry] = None
_DEFAULT_STAGE_REGISTRY_LOCK = threading.Lock()


def get_stage_registry() -> StageRegistry:
    global _DEFAULT_STAGE_REGISTRY
    if _DEFAULT_STAGE_REGISTRY is None:
        with _DEFAULT_STAGE_REGISTRY_LOCK:
            if _DEFAULT_STAGE_REGISTRY is None:
                _DEFAULT_STAGE_REGISTRY = create_stage_registry()
    return _DEFAULT_STAGE_REGISTRY


def register_stage(
    provider: Any,
    *,
    name: Optional[str] = None,
    replace: bool = False,
    registry: Optional[StageRegistry] = None,
) -> Any:
    target = registry or get_stage_registry()
    stage_name = name or getattr(provider, "name", "")
    return target.register(stage_name, provider, replace=replace)


def list_stages(*, registry: Optional[StageRegistry] = None) -> tuple[str, ...]:
    return (registry or get_stage_registry()).names()


__all__ = [
    "BUILTIN_STAGE_NAMES",
    "BaseStage",
    "FunctionStage",
    "OrchestratorMethodStage",
    "PRETRAIN_STRATEGIES",
    "SINGLE_PROCESS_STRATEGIES",
    "STAGE_ENTRY_POINT_GROUP",
    "StageCapabilities",
    "StageContext",
    "StagePlugin",
    "StageRegistry",
    "create_stage_registry",
    "get_stage_registry",
    "list_stages",
    "register_builtin_stages",
    "register_stage",
]
