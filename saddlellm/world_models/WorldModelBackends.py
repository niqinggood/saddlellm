"""Registry, factory, and loader for SaddleLLM-native world models."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ._CategoricalWorldModel import (
    CategoricalWorldModel,
    CategoricalWorldModelConfig,
)
from ._WorldModel import WorldModel, WorldModelConfig


NATIVE_WORLD_MODEL_BACKENDS: Dict[str, Dict[str, Any]] = {
    "rssm": {
        "name": "rssm",
        "kind": "native",
        "description": "Gaussian recurrent state-space model with balanced KL.",
        "license": "SaddleLLM MIT",
        "available": True,
        "supports": ["vector", "image", "continuous_action", "discrete_action"],
    },
    "categorical_rssm": {
        "name": "categorical_rssm",
        "kind": "native",
        "description": "Categorical recurrent state-space model with split KL objectives.",
        "license": "SaddleLLM MIT",
        "available": True,
        "supports": [
            "vector",
            "image",
            "continuous_action",
            "discrete_action",
            "categorical_latent",
            "symlog",
            "twohot_reward",
        ],
    },
}


def normalize_world_model_backend(name: Optional[str]) -> str:
    aliases = {
        "": "rssm",
        "native": "rssm",
        "gaussian": "rssm",
        "gaussian_rssm": "rssm",
        "categorical": "categorical_rssm",
        "discrete_rssm": "categorical_rssm",
    }
    normalized = str(name or "").strip().lower()
    return aliases.get(normalized, normalized)


def create_world_model(backend: str, config: Dict[str, Any]):
    """Instantiate a native model from a backend name and inferred config."""

    normalized = normalize_world_model_backend(backend)
    values = dict(config)
    values.pop("backend", None)
    if normalized == "rssm":
        return WorldModel(WorldModelConfig.from_dict(values))
    if normalized == "categorical_rssm":
        return CategoricalWorldModel(CategoricalWorldModelConfig.from_dict(values))
    raise ValueError(
        f"Unknown native world-model backend {backend!r}; choose from "
        + ", ".join(sorted(NATIVE_WORLD_MODEL_BACKENDS))
    )


def load_world_model(path: str, map_location: Any = "cpu"):
    """Load any native SaddleLLM world model from one output directory."""

    config_path = os.path.join(path, WorldModel.CONFIG_NAME)
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    backend = normalize_world_model_backend(config.get("backend", "rssm"))
    if backend == "rssm":
        return WorldModel.from_pretrained(path, map_location=map_location)
    if backend == "categorical_rssm":
        return CategoricalWorldModel.from_pretrained(path, map_location=map_location)
    raise ValueError(f"Unsupported saved world-model backend: {backend}")


def list_world_model_backends() -> List[Dict[str, Any]]:
    """List the world-model implementations owned and trained by SaddleLLM."""

    return [dict(value) for value in NATIVE_WORLD_MODEL_BACKENDS.values()]
