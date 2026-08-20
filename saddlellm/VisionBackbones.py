"""Vision backbone registry for SaddleLLM multimodal models."""
import importlib.util
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class VisionBackboneConfig:
    name: str = "tiny_patch"
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 768
    output_tokens: int = 0
    freeze: bool = True
    pretrained_name_or_path: Optional[str] = None
    trust_remote_code: bool = True
    local_files_only: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VisionBackboneSpec:
    name: str
    family: str
    available: bool
    output_dim: int
    required_packages: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class TinyPatchVisionBackbone(nn.Module):
    """Small patch encoder for tests and low-cost projector warmup experiments."""

    def __init__(self, config: VisionBackboneConfig):
        super().__init__()
        self.config = config
        self.output_dim = config.hidden_size
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(pixel_values)
        features = features.flatten(2).transpose(1, 2).contiguous()
        if self.config.output_tokens and features.shape[1] > self.config.output_tokens:
            features = features[:, : self.config.output_tokens]
        return self.norm(features)


class HFVisionBackbone(nn.Module):
    """Thin wrapper around Transformers vision encoders."""

    def __init__(self, config: VisionBackboneConfig):
        super().__init__()
        if importlib.util.find_spec("transformers") is None:
            raise ImportError("transformers is required for HF vision backbones")
        from transformers import AutoModel

        model_name = config.pretrained_name_or_path or config.name
        self.config = config
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
        hidden_size = hidden_size or getattr(getattr(self.model, "config", None), "vision_config", None).hidden_size
        self.output_dim = int(hidden_size)
        if config.freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        if hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            features = outputs[0]
        else:
            raise ValueError("HF vision backbone did not return hidden states")
        if self.config.output_tokens and features.shape[1] > self.config.output_tokens:
            features = features[:, : self.config.output_tokens]
        return features


class VisionBackboneRegistry:
    """Build and inspect vision encoders without hard dependency on one provider."""

    @staticmethod
    def list_backbones() -> Dict[str, Dict]:
        has_transformers = importlib.util.find_spec("transformers") is not None
        return {
            "tiny_patch": VisionBackboneSpec(
                name="tiny_patch",
                family="saddle",
                available=True,
                output_dim=768,
                required_packages=["torch"],
                notes=["No-download patch encoder for tests and projector warmup."],
            ).to_dict(),
            "clip": VisionBackboneSpec(
                name="clip",
                family="huggingface",
                available=has_transformers,
                output_dim=768,
                required_packages=["transformers"],
                notes=["Use pretrained_name_or_path such as openai/clip-vit-large-patch14."],
            ).to_dict(),
            "siglip": VisionBackboneSpec(
                name="siglip",
                family="huggingface",
                available=has_transformers,
                output_dim=1152,
                required_packages=["transformers"],
                notes=["Use pretrained_name_or_path for SigLIP/SigLIP2 style encoders."],
            ).to_dict(),
            "dinov2": VisionBackboneSpec(
                name="dinov2",
                family="huggingface",
                available=has_transformers,
                output_dim=768,
                required_packages=["transformers"],
                notes=["Use pretrained_name_or_path for DINOv2 encoders when visual features are not CLIP-aligned."],
            ).to_dict(),
        }

    @staticmethod
    def build(config: VisionBackboneConfig) -> nn.Module:
        name = (config.name or "tiny_patch").lower()
        if name in {"tiny", "tiny_patch", "patch"}:
            model = TinyPatchVisionBackbone(config)
        else:
            model = HFVisionBackbone(config)
        if config.freeze:
            for param in model.parameters():
                param.requires_grad_(False)
        return model


def list_vision_backbones() -> Dict[str, Dict]:
    return VisionBackboneRegistry.list_backbones()

