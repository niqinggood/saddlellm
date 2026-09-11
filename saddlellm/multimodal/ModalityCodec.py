"""Codec contracts for text, image, audio, and video latent spaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .MultimodalData import SUPPORTED_MODALITIES


@dataclass(frozen=True)
class ModalityCodecSpec:
    name: str
    modality: str
    latent_kind: str  # discrete | continuous
    trainable: bool = False
    supports_decode: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported codec modality: {self.modality!r}")
        if self.latent_kind not in {"discrete", "continuous"}:
            raise ValueError("latent_kind must be 'discrete' or 'continuous'")


class ModalityCodec(ABC):
    """Transforms raw modality samples to and from model latent space."""

    spec: ModalityCodecSpec

    @abstractmethod
    def encode(self, value: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latents: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def fingerprint(self) -> Dict[str, Any]:
        """Serializable identity stored beside cached latents/checkpoints."""

        return {
            "name": self.spec.name,
            "modality": self.spec.modality,
            "latent_kind": self.spec.latent_kind,
            "trainable": self.spec.trainable,
            "supports_decode": self.spec.supports_decode,
            "metadata": dict(self.spec.metadata),
        }


class CallableModalityCodec(ModalityCodec):
    """Small adapter for externally provided tokenizer/VAE/audio-codec calls."""

    def __init__(
        self,
        spec: ModalityCodecSpec,
        encoder: Callable[..., Any],
        decoder: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.spec = spec
        self._encoder = encoder
        self._decoder = decoder
        if spec.supports_decode and decoder is None:
            raise ValueError(f"Codec {spec.name!r} declares decode support but has no decoder")

    def encode(self, value: Any, **kwargs: Any) -> Any:
        return self._encoder(value, **kwargs)

    def decode(self, latents: Any, **kwargs: Any) -> Any:
        if self._decoder is None:
            raise NotImplementedError(f"Codec {self.spec.name!r} is encode-only")
        return self._decoder(latents, **kwargs)


class ModalityCodecRegistry:
    """Explicit registry; no heavyweight modality dependency is imported eagerly."""

    _factories: Dict[str, Callable[..., ModalityCodec]] = {}
    _specs: Dict[str, ModalityCodecSpec] = {}

    @classmethod
    def register(
        cls,
        spec: ModalityCodecSpec,
        factory: Callable[..., ModalityCodec],
        *,
        overwrite: bool = False,
    ) -> None:
        name = spec.name.strip().lower()
        if not name:
            raise ValueError("codec name cannot be empty")
        if name in cls._factories and not overwrite:
            raise ValueError(f"Codec {name!r} is already registered")
        cls._specs[name] = spec
        cls._factories[name] = factory

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> ModalityCodec:
        key = str(name).strip().lower()
        try:
            codec = cls._factories[key](**kwargs)
        except KeyError as exc:
            raise KeyError(
                f"Unknown modality codec {name!r}; available: "
                + ", ".join(sorted(cls._factories))
            ) from exc
        if not isinstance(codec, ModalityCodec):
            raise TypeError(f"Codec factory {key!r} did not return ModalityCodec")
        if codec.spec.name.lower() != key:
            raise ValueError(
                f"Codec factory {key!r} returned mismatched spec {codec.spec.name!r}"
            )
        return codec

    @classmethod
    def get_spec(cls, name: str) -> ModalityCodecSpec:
        try:
            return cls._specs[str(name).strip().lower()]
        except KeyError as exc:
            raise KeyError(f"Unknown modality codec {name!r}") from exc

    @classmethod
    def list_specs(cls, modality: Optional[str] = None) -> List[ModalityCodecSpec]:
        specs = list(cls._specs.values())
        if modality is not None:
            normalized = modality.lower()
            if normalized not in SUPPORTED_MODALITIES:
                raise ValueError(f"Unsupported modality: {modality!r}")
            specs = [spec for spec in specs if spec.modality == normalized]
        return sorted(specs, key=lambda spec: spec.name)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a registration; primarily useful for isolated plugin tests."""

        key = str(name).strip().lower()
        cls._factories.pop(key, None)
        cls._specs.pop(key, None)


__all__ = [
    "ModalityCodecSpec",
    "ModalityCodec",
    "CallableModalityCodec",
    "ModalityCodecRegistry",
]
