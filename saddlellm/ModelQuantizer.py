"""Compatibility import for :mod:`saddlellm.compression.quantizer`.

New code should import from :mod:`saddlellm.compression`.
"""

from .compression.quantizer import (
    QUANTIZATION_MANIFEST_FILENAME,
    QUANTIZED_STATE_FILENAME,
    ModelQuantizer,
)

__all__ = [
    "QUANTIZATION_MANIFEST_FILENAME",
    "QUANTIZED_STATE_FILENAME",
    "ModelQuantizer",
]
