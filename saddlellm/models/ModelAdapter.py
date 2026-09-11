"""Stable model-backend capabilities used by the training lifecycle.

The adapter deliberately stays smaller than a trainer.  It owns checkpoint
loading, tokenizer loading, persistence and capability validation so stage
orchestration does not need to guess whether a Hugging Face or native Saddle
model is being handled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Optional


@dataclass(frozen=True)
class ModelBackendCapabilities:
    backend: str
    stages: FrozenSet[str]
    preference_methods: FrozenSet[str]
    lora: bool
    qlora: bool
    distributed_post_training: bool

    def supports_stage(self, stage: str) -> bool:
        return stage in self.stages


class ModelAdapter:
    """Checkpoint lifecycle contract shared by all causal-LM backends."""

    backend: str
    capabilities: ModelBackendCapabilities

    def load_model(self, path: str, **kwargs: Any) -> Any:
        from .ModelLoader import load_causal_lm

        return load_causal_lm(path, **kwargs)

    def load_tokenizer(self, path: str, **kwargs: Any) -> Any:
        from .TokenizerLoader import load_tokenizer_compatible

        return load_tokenizer_compatible(path, **kwargs)

    def trainer_class(self):
        from transformers import Trainer

        return Trainer

    def validate_stage(
        self,
        stage: str,
        *,
        method: Optional[str] = None,
        use_lora: bool = False,
        use_qlora: bool = False,
    ) -> None:
        if not self.capabilities.supports_stage(stage):
            supported = ", ".join(sorted(self.capabilities.stages))
            raise ValueError(
                f"Model backend {self.backend!r} does not support stage {stage!r}; "
                f"supported stages: {supported}"
            )
        if method and method not in self.capabilities.preference_methods:
            supported = ", ".join(sorted(self.capabilities.preference_methods)) or "none"
            raise ValueError(
                f"Model backend {self.backend!r} does not support {stage} method "
                f"{method!r}; supported methods: {supported}"
            )
        if use_lora and not self.capabilities.lora:
            raise ValueError(
                f"Model backend {self.backend!r} does not support LoRA for {stage}; "
                "set use_lora=false for full-parameter training"
            )
        if use_qlora and not self.capabilities.qlora:
            raise ValueError(
                f"Model backend {self.backend!r} does not support QLoRA for {stage}; "
                "set use_qlora=false"
            )


class SaddleModelAdapter(ModelAdapter):
    backend = "saddle"
    capabilities = ModelBackendCapabilities(
        backend=backend,
        stages=frozenset({
            "tokenizer",
            "pretrain",
            "sft",
            "preference",
            "rlhf",
            "media_cache",
            "image_generation",
            "music_generation",
            "video_generation",
            "world_model",
            "eval",
            "export",
        }),
        preference_methods=frozenset({"dpo"}),
        lora=False,
        qlora=False,
        distributed_post_training=False,
    )

    def load_model(self, path: str, **kwargs: Any) -> Any:
        from .ModelLoader import is_saddle_checkpoint

        if not is_saddle_checkpoint(path):
            raise ValueError(
                "Native Saddle post-training requires a complete local checkpoint "
                "containing saddle_config.json and pytorch_model.bin"
            )
        return super().load_model(path, **kwargs)

    def trainer_class(self):
        from ..training.NativeTrainer import SaddleTrainer

        return SaddleTrainer


class HuggingFaceModelAdapter(ModelAdapter):
    backend = "hf"
    capabilities = ModelBackendCapabilities(
        backend=backend,
        stages=frozenset({
            "tokenizer",
            "pretrain",
            "sft",
            "preference",
            "rlhf",
            "mopd",
            "mllm_sft",
            "vision_alignment",
            "media_cache",
            "image_generation",
            "music_generation",
            "video_generation",
            "world_model",
            "vla_sft",
            "eval",
            "export",
        }),
        preference_methods=frozenset({"dpo", "orpo", "kto"}),
        lora=True,
        qlora=True,
        distributed_post_training=False,
    )


_ADAPTERS = {
    "saddle": SaddleModelAdapter(),
    "hf": HuggingFaceModelAdapter(),
}


def get_model_adapter(backend: str) -> ModelAdapter:
    """Resolve a configured backend without inspecting a remote model ID."""

    key = str(backend or "").strip().lower()
    try:
        return _ADAPTERS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model backend {backend!r}; expected one of: "
            + ", ".join(sorted(_ADAPTERS))
        ) from exc


__all__ = [
    "ModelBackendCapabilities",
    "ModelAdapter",
    "SaddleModelAdapter",
    "HuggingFaceModelAdapter",
    "get_model_adapter",
]
