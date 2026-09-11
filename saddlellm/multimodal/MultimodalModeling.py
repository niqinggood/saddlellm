"""LLaVA-style multimodal wrapper for decoder-only language models."""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from .MultimodalProjector import MultimodalProjectorConfig, MultimodalProjectorFactory
from ..models.SaddleModeling import SaddleCausalLMOutput, SaddleForCausalLM, SaddleModelConfig
from .VisionBackbones import VisionBackboneConfig, VisionBackboneRegistry


@dataclass
class MultimodalConfig:
    llm_config: SaddleModelConfig = field(default_factory=SaddleModelConfig)
    vision_config: VisionBackboneConfig = field(default_factory=VisionBackboneConfig)
    projector_config: Optional[MultimodalProjectorConfig] = None
    image_token_id: int = 32000
    image_token_strategy: str = "replace"  # replace | prepend
    freeze_vision: bool = True
    freeze_llm: bool = False
    train_projector_only: bool = False

    def __post_init__(self):
        if self.projector_config is None:
            self.projector_config = MultimodalProjectorConfig(
                vision_hidden_size=self.vision_config.hidden_size,
                llm_hidden_size=self.llm_config.hidden_size,
            )

    def to_dict(self) -> Dict:
        return {
            "llm_config": self.llm_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "projector_config": self.projector_config.to_dict(),
            "image_token_id": self.image_token_id,
            "image_token_strategy": self.image_token_strategy,
            "freeze_vision": self.freeze_vision,
            "freeze_llm": self.freeze_llm,
            "train_projector_only": self.train_projector_only,
        }

    @classmethod
    def from_dict(cls, raw: Dict) -> "MultimodalConfig":
        return cls(
            llm_config=SaddleModelConfig(**raw.get("llm_config", {})),
            vision_config=VisionBackboneConfig(**raw.get("vision_config", {})),
            projector_config=MultimodalProjectorConfig(**raw.get("projector_config", {})),
            image_token_id=raw.get("image_token_id", 32000),
            image_token_strategy=raw.get("image_token_strategy", "replace"),
            freeze_vision=raw.get("freeze_vision", True),
            freeze_llm=raw.get("freeze_llm", False),
            train_projector_only=raw.get("train_projector_only", False),
        )


class MultimodalForCausalLM(nn.Module):
    """Vision encoder + projector + SaddleForCausalLM decoder."""

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.language_model = SaddleForCausalLM(config.llm_config)
        self.vision_encoder = VisionBackboneRegistry.build(config.vision_config)
        vision_dim = getattr(self.vision_encoder, "output_dim", config.vision_config.hidden_size)
        projector_config = config.projector_config or MultimodalProjectorConfig()
        projector_config.vision_hidden_size = vision_dim
        projector_config.llm_hidden_size = config.llm_config.hidden_size
        self.projector = MultimodalProjectorFactory.build(projector_config)
        self.config.projector_config = projector_config
        self._apply_freezing()

    @classmethod
    def from_text_blueprint(
        cls,
        blueprint,
        vision_config: Optional[VisionBackboneConfig] = None,
        projector_config: Optional[MultimodalProjectorConfig] = None,
        image_token_id: Optional[int] = None,
    ) -> "MultimodalForCausalLM":
        llm_config = SaddleModelConfig.from_blueprint(blueprint)
        image_token = image_token_id if image_token_id is not None else llm_config.vocab_size
        if image_token >= llm_config.vocab_size:
            llm_config.vocab_size = image_token + 1
        return cls(MultimodalConfig(
            llm_config=llm_config,
            vision_config=vision_config or VisionBackboneConfig(),
            projector_config=projector_config,
            image_token_id=image_token,
        ))

    def _apply_freezing(self):
        if self.config.freeze_vision or self.config.train_projector_only:
            for param in self.vision_encoder.parameters():
                param.requires_grad_(False)
        if self.config.freeze_llm or self.config.train_projector_only:
            for param in self.language_model.parameters():
                param.requires_grad_(False)
        for param in self.projector.parameters():
            param.requires_grad_(True)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(pixel_values)
        return self.projector(features)

    def prepare_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is None:
            return {"inputs_embeds": text_embeds, "attention_mask": attention_mask, "labels": labels}

        image_embeds = self.encode_images(pixel_values).to(dtype=text_embeds.dtype, device=text_embeds.device)
        if self.config.image_token_strategy == "prepend":
            return self._prepend_image_tokens(text_embeds, image_embeds, attention_mask, labels)
        return self._replace_image_tokens(input_ids, text_embeds, image_embeds, attention_mask, labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> SaddleCausalLMOutput:
        prepared = self.prepare_multimodal_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )
        return self.language_model(
            input_ids=None,
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared.get("attention_mask"),
            labels=prepared.get("labels"),
            return_dict=return_dict,
            **kwargs,
        )

    def _prepend_image_tokens(self, text_embeds, image_embeds, attention_mask, labels):
        batch, image_tokens, _ = image_embeds.shape
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        image_mask = torch.ones(batch, image_tokens, device=text_embeds.device, dtype=attention_mask.dtype if attention_mask is not None else torch.long)
        if attention_mask is None:
            attention_mask = torch.ones(text_embeds.shape[:2], device=text_embeds.device, dtype=image_mask.dtype)
        attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        if labels is not None:
            image_labels = torch.full((batch, image_tokens), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([image_labels, labels], dim=1)
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "labels": labels}

    def _replace_image_tokens(self, input_ids, text_embeds, image_embeds, attention_mask, labels):
        inputs_embeds = text_embeds.clone()
        labels = labels.clone() if labels is not None else None
        batch = input_ids.shape[0]
        for row in range(batch):
            positions = (input_ids[row] == self.config.image_token_id).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            count = min(int(positions.numel()), image_embeds.shape[1])
            inputs_embeds[row, positions[:count]] = image_embeds[row, :count].to(inputs_embeds.dtype)
            if labels is not None:
                labels[row, positions] = -100
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.long)
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "labels": labels}

    def save_pretrained(self, path: str, state_dict: Optional[Dict] = None, **_) -> str:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "multimodal_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)
        torch.save(state_dict or self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        return path

    @classmethod
    def from_pretrained_saddle_multimodal(cls, path: str, map_location: Optional[str] = None) -> "MultimodalForCausalLM":
        config_path = os.path.join(path, "multimodal_config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(path, "saddle_multimodal_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = MultimodalConfig.from_dict(json.load(f))
        model = cls(config)
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=map_location)
        model.load_state_dict(state)
        return model


# Backward-compatible aliases. Prefer MultimodalConfig and MultimodalForCausalLM
# in new code because the package name already carries the SaddleLLM identity.
SaddleMultimodalConfig = MultimodalConfig
SaddleMultimodalForCausalLM = MultimodalForCausalLM
