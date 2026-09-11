"""Compatibility module for early multimodal development snapshots.

New code should import from :mod:`saddlellm.multimodal.MultimodalModeling`.
"""
from .MultimodalModeling import (
    MultimodalConfig,
    MultimodalForCausalLM,
    SaddleMultimodalConfig,
    SaddleMultimodalForCausalLM,
)

__all__ = [
    "MultimodalConfig",
    "MultimodalForCausalLM",
    "SaddleMultimodalConfig",
    "SaddleMultimodalForCausalLM",
]
