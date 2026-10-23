"""Compatibility import for :mod:`saddlellm.data.pipeline`.

New code should import from :mod:`saddlellm.data`.
"""

from .data.pipeline import (
    CausalLMDataCollator,
    DataMixConfig,
    DataMixer,
    DataPipeline,
    PipelineConfig,
)

__all__ = [
    "CausalLMDataCollator",
    "DataMixConfig",
    "DataMixer",
    "DataPipeline",
    "PipelineConfig",
]
