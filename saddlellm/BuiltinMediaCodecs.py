"""Dependency-light baseline codecs and text condition encoders.

These codecs are intentionally simple reference implementations.  They make
the raw-media pipeline executable in a default SaddleLLM installation while
the registry remains the extension point for production VAEs, EnCodec-like
audio tokenizers, and causal video codecs.
"""
from __future__ import annotations

import hashlib
import math
import os
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .ModalityCodec import ModalityCodec, ModalityCodecRegistry, ModalityCodecSpec


class TextConditionEncoder(ABC):
    """Convert one or more prompts into fixed-width floating point vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def fingerprint(self) -> Dict[str, Any]:
        raise NotImplementedError


class HashTextConditionEncoder(TextConditionEncoder):
    """Deterministic UTF-8 n-gram hashing baseline with no model download."""

    def __init__(self, dimension: int = 256, ngram: int = 3, seed: int = 0) -> None:
        if dimension <= 0 or ngram <= 0:
            raise ValueError("dimension and ngram must be positive")
        self._dimension = int(dimension)
        self.ngram = int(ngram)
        self.seed = int(seed)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        salt = self.seed.to_bytes(8, byteorder="little", signed=True)
        for row, text in enumerate(texts):
            payload = str(text).encode("utf-8")
            tokens = [payload[index : index + self.ngram] for index in range(max(1, len(payload) - self.ngram + 1))]
            if not tokens:
                tokens = [b""]
            for token in tokens:
                digest = hashlib.blake2b(token, digest_size=16, key=salt).digest()
                index = int.from_bytes(digest[:8], "little") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                result[row, index] += sign
            norm = float(np.linalg.norm(result[row]))
            if norm:
                result[row] /= norm
        return result

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "name": "hash",
            "dimension": self.dimension,
            "ngram": self.ngram,
            "seed": self.seed,
            "implementation": "blake2b-signed-feature-hashing-v1",
        }


class HuggingFaceTextConditionEncoder(TextConditionEncoder):
    """Mean-pooled hidden states from a local or Hugging Face encoder."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "auto",
        max_length: int = 256,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required for the HF text encoder")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.max_length = int(max_length)
        self.local_files_only = bool(local_files_only)
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else (
                "cpu" if device == "auto" else device
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        ).to(self.device).eval()
        self._dimension = int(
            getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "d_model", 0)
        )
        if self._dimension <= 0:
            raise ValueError("Could not infer hidden dimension from text encoder config")

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            output = self.model(**batch)
            hidden = output.last_hidden_state
            mask = batch.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device))
            pooled = (hidden * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled.float().cpu().numpy()

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "name": "huggingface",
            "model_name_or_path": self.model_name_or_path,
            "dimension": self.dimension,
            "max_length": self.max_length,
            "local_files_only": self.local_files_only,
        }


def build_text_condition_encoder(
    name: str = "hash", **config: Any
) -> TextConditionEncoder:
    normalized = str(name or "hash").strip().lower()
    if normalized in {"hash", "local-hash", "deterministic-hash"}:
        return HashTextConditionEncoder(**config)
    if normalized in {"hf", "huggingface", "transformers"}:
        return HuggingFaceTextConditionEncoder(**config)
    raise ValueError("Unknown text condition encoder; expected hash or huggingface")


class RGBImageCodec(ModalityCodec):
    """Resize RGB pixels and expose normalized CHW continuous latents."""

    spec = ModalityCodecSpec(
        name="baseline-rgb-image",
        modality="image",
        latent_kind="continuous",
        trainable=False,
        supports_decode=True,
        metadata={"quality": "baseline", "range": "[-1,1]"},
    )

    def __init__(self, height: int = 32, width: int = 32) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("image height and width must be positive")
        self.height = int(height)
        self.width = int(width)

    def encode(self, value: Any, **kwargs: Any) -> np.ndarray:
        from PIL import Image

        image = value if isinstance(value, Image.Image) else Image.open(os.fspath(value))
        image = image.convert("RGB").resize((self.width, self.height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return np.transpose(array, (2, 0, 1))

    def decode(self, latents: Any, **kwargs: Any):
        from PIL import Image

        array = np.asarray(latents, dtype=np.float32)
        if array.shape != (3, self.height, self.width):
            raise ValueError(
                f"image latents must have shape {(3, self.height, self.width)}"
            )
        pixels = np.clip((np.transpose(array, (1, 2, 0)) + 1.0) * 127.5, 0, 255)
        image = Image.fromarray(pixels.astype(np.uint8), mode="RGB")
        output_path = kwargs.get("output_path")
        if output_path:
            image.save(output_path)
        return image

    def fingerprint(self) -> Dict[str, Any]:
        return {**super().fingerprint(), "height": self.height, "width": self.width}


class WAVResidualCodec(ModalityCodec):
    """Residual scalar quantizer over fixed-rate mono WAV frame averages."""

    spec = ModalityCodecSpec(
        name="baseline-wav-rvq",
        modality="audio",
        latent_kind="discrete",
        trainable=False,
        supports_decode=True,
        metadata={"quality": "baseline", "input": "PCM WAV"},
    )

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 320,
        max_frames: int = 256,
        num_codebooks: int = 4,
        codebook_size: int = 256,
    ) -> None:
        for name, value in {
            "sample_rate": sample_rate,
            "frame_size": frame_size,
            "max_frames": max_frames,
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.max_frames = int(max_frames)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)

    def encode(self, value: Any, **kwargs: Any) -> np.ndarray:
        codes, _ = self.encode_with_attention_mask(value)
        return codes

    def encode_with_attention_mask(self, value: Any) -> tuple[np.ndarray, np.ndarray]:
        samples, source_rate = _read_wav_mono(os.fspath(value))
        samples = _resample_linear(samples, source_rate, self.sample_rate)
        if not samples.size:
            raise ValueError("WAV input contains no audio samples")
        required = self.frame_size * self.max_frames
        padded = np.zeros(required, dtype=np.float32)
        copied = min(required, samples.size)
        padded[:copied] = samples[:copied]
        valid_frames = min(self.max_frames, int(math.ceil(copied / self.frame_size)))
        attention_mask = np.zeros(self.max_frames, dtype=np.int64)
        attention_mask[:valid_frames] = 1
        frames = padded.reshape(self.max_frames, self.frame_size).mean(axis=1)
        residual = np.clip(frames, -1.0, 1.0)
        codes = np.empty((self.num_codebooks, self.max_frames), dtype=np.int64)
        scale = 1.0
        for index in range(self.num_codebooks):
            normalized = np.clip(residual / scale, -1.0, 1.0)
            code = np.rint((normalized + 1.0) * 0.5 * (self.codebook_size - 1))
            code = code.astype(np.int64)
            level = (code.astype(np.float32) / (self.codebook_size - 1) * 2.0 - 1.0) * scale
            codes[index] = code
            residual = residual - level
            scale /= self.codebook_size - 1
        return codes, attention_mask

    def decode(self, latents: Any, **kwargs: Any) -> np.ndarray:
        codes = np.asarray(latents)
        expected = (self.num_codebooks, self.max_frames)
        if codes.shape != expected:
            raise ValueError(f"audio codes must have shape {expected}")
        if codes.min() < 0 or codes.max() >= self.codebook_size:
            raise ValueError("audio codes exceed the codec vocabulary")
        frames = np.zeros(self.max_frames, dtype=np.float32)
        scale = 1.0
        for codebook in codes:
            frames += (
                codebook.astype(np.float32) / (self.codebook_size - 1) * 2.0 - 1.0
            ) * scale
            scale /= self.codebook_size - 1
        samples = np.repeat(np.clip(frames, -1.0, 1.0), self.frame_size)
        output_path = kwargs.get("output_path")
        if output_path:
            _write_wav_mono(output_path, samples, self.sample_rate)
        return samples

    def fingerprint(self) -> Dict[str, Any]:
        return {
            **super().fingerprint(),
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "max_frames": self.max_frames,
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
        }


class FrameVideoCodec(ModalityCodec):
    """Fixed-length RGB video latents from a frame directory or animated image."""

    spec = ModalityCodecSpec(
        name="baseline-frame-video",
        modality="video",
        latent_kind="continuous",
        trainable=False,
        supports_decode=True,
        metadata={"quality": "baseline", "input": "frame directory or GIF/WebP"},
    )

    def __init__(self, frames: int = 8, height: int = 32, width: int = 32) -> None:
        if frames <= 0 or height <= 0 or width <= 0:
            raise ValueError("frames, height, and width must be positive")
        self.frames = int(frames)
        self.height = int(height)
        self.width = int(width)

    def encode(self, value: Any, **kwargs: Any) -> np.ndarray:
        images = _load_video_frames(os.fspath(value))
        if not images:
            raise ValueError(f"No decodable frames found in {value}")
        indices = np.linspace(0, len(images) - 1, self.frames).round().astype(int)
        encoded = []
        for index in indices:
            image = images[int(index)].convert("RGB").resize(
                (self.width, self.height), image_resampling()
            )
            encoded.append(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
        array = np.stack(encoded, axis=0)
        return np.transpose(array, (3, 0, 1, 2))

    def decode(self, latents: Any, **kwargs: Any) -> List[Any]:
        from PIL import Image

        array = np.asarray(latents, dtype=np.float32)
        expected = (3, self.frames, self.height, self.width)
        if array.shape != expected:
            raise ValueError(f"video latents must have shape {expected}")
        frames = []
        for index in range(self.frames):
            pixels = np.clip((np.transpose(array[:, index], (1, 2, 0)) + 1.0) * 127.5, 0, 255)
            frames.append(Image.fromarray(pixels.astype(np.uint8), mode="RGB"))
        output_path = kwargs.get("output_path")
        if output_path:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(kwargs.get("duration_ms", 100)),
                loop=0,
            )
        return frames

    def fingerprint(self) -> Dict[str, Any]:
        return {
            **super().fingerprint(),
            "frames": self.frames,
            "height": self.height,
            "width": self.width,
        }


def image_resampling():
    from PIL import Image

    return Image.Resampling.BICUBIC


def register_builtin_media_codecs() -> None:
    """Idempotently register the dependency-light baseline implementations."""

    registrations = (
        (RGBImageCodec.spec, RGBImageCodec),
        (WAVResidualCodec.spec, WAVResidualCodec),
        (FrameVideoCodec.spec, FrameVideoCodec),
    )
    existing = {spec.name for spec in ModalityCodecRegistry.list_specs()}
    for spec, factory in registrations:
        if spec.name not in existing:
            ModalityCodecRegistry.register(spec, factory)


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM WAV sample width: {sample_width}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32, copy=False), int(sample_rate)


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size < 2:
        return samples
    target_length = max(1, int(round(samples.size * target_rate / source_rate)))
    source_positions = np.linspace(0.0, 1.0, samples.size)
    target_positions = np.linspace(0.0, 1.0, target_length)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _write_wav_mono(path: str, samples: np.ndarray, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def _load_video_frames(path: str) -> List[Any]:
    from PIL import Image, ImageSequence

    source = Path(path)
    if source.is_dir():
        suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return [
            Image.open(item).copy()
            for item in sorted(source.iterdir())
            if item.is_file() and item.suffix.lower() in suffixes
        ]
    with Image.open(source) as image:
        return [frame.copy() for frame in ImageSequence.Iterator(image)]


__all__ = [
    "TextConditionEncoder",
    "HashTextConditionEncoder",
    "HuggingFaceTextConditionEncoder",
    "build_text_condition_encoder",
    "RGBImageCodec",
    "WAVResidualCodec",
    "FrameVideoCodec",
    "register_builtin_media_codecs",
]
