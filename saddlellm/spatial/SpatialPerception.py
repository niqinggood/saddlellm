"""Image-to-space adapters for the native spatial world-model stack."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .SpatialPlanner import OccupancyGrid


@dataclass
class SpatialEntity:
    name: str
    category: str
    bbox: Tuple[float, float, float, float]
    confidence: float = 0.5
    traversable: Optional[bool] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox = tuple(float(value) for value in self.bbox)
        if len(self.bbox) != 4:
            raise ValueError("SpatialEntity.bbox must contain x1, y1, x2, y2")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("SpatialEntity.confidence must be in [0, 1]")

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialEntity":
        values = dict(data)
        values.setdefault("name", values.get("label", "unnamed"))
        values.setdefault("category", "object")
        values.setdefault("bbox", [0.0, 0.0, 0.0, 0.0])
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class SpatialAnalysis:
    """Structured semantic interpretation produced by a vision-language model."""

    image_type: str = "unknown"
    summary: str = ""
    entities: List[SpatialEntity] = field(default_factory=list)
    connectivity: List[Dict[str, Any]] = field(default_factory=list)
    hazards: List[str] = field(default_factory=list)
    unknown_regions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    caveats: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.entities = [
            value if isinstance(value, SpatialEntity) else SpatialEntity.from_dict(value)
            for value in self.entities
        ]
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("SpatialAnalysis.confidence must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialAnalysis":
        values = dict(data or {})
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MapExtractionConfig:
    """Threshold-based extractor for clean top-down maps and floor plans."""

    free_threshold: float = 0.72
    free_is_bright: bool = True
    uncertainty_band: float = 0.0
    obstacle_dilation: int = 1
    max_dimension: int = 512
    block_border: bool = True
    resolution: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.free_threshold <= 1.0:
            raise ValueError("free_threshold must be in [0, 1]")
        if not 0.0 <= self.uncertainty_band <= 0.5:
            raise ValueError("uncertainty_band must be in [0, 0.5]")
        if self.obstacle_dilation < 0:
            raise ValueError("obstacle_dilation cannot be negative")
        if self.max_dimension <= 0:
            raise ValueError("max_dimension must be positive")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")


class TopDownMapExtractor:
    """Convert a clean floor plan/map image into an occupancy grid."""

    def __init__(self, config: Optional[MapExtractionConfig] = None) -> None:
        self.config = config or MapExtractionConfig()

    def extract(self, image_path: str) -> OccupancyGrid:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("Pillow is required for spatial image extraction") from error
        with Image.open(image_path) as source:
            source_width, source_height = source.size
            image = source.convert("RGBA")
            if max(image.size) > self.config.max_dimension:
                scale = self.config.max_dimension / max(image.size)
                size = (
                    max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))),
                )
                resampling = getattr(Image, "Resampling", Image)
                image = image.resize(size, resampling.BILINEAR)
            rgba = np.asarray(image, dtype=np.float32) / 255.0
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
        luminance = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )
        if self.config.free_is_bright:
            free = luminance >= self.config.free_threshold
        else:
            free = luminance <= self.config.free_threshold
        cells = np.where(free, OccupancyGrid.FREE, OccupancyGrid.BLOCKED).astype(
            np.uint8
        )
        if self.config.uncertainty_band:
            uncertain = (
                np.abs(luminance - self.config.free_threshold)
                <= self.config.uncertainty_band
            )
            cells[uncertain] = OccupancyGrid.UNKNOWN
        cells[alpha < 0.5] = OccupancyGrid.UNKNOWN
        if self.config.obstacle_dilation:
            blocked = cells == OccupancyGrid.BLOCKED
            blocked = _dilate(blocked, self.config.obstacle_dilation)
            cells[blocked] = OccupancyGrid.BLOCKED
        if self.config.block_border:
            cells[0, :] = OccupancyGrid.BLOCKED
            cells[-1, :] = OccupancyGrid.BLOCKED
            cells[:, 0] = OccupancyGrid.BLOCKED
            cells[:, -1] = OccupancyGrid.BLOCKED
        return OccupancyGrid(
            cells=cells,
            resolution=self.config.resolution,
            source_size=(source_width, source_height),
        )


class CallableSpatialAnalyzer:
    """Adapt an application-provided vision function to the spatial protocol."""

    def __init__(self, analyze_fn: Callable[[str, str], Union[SpatialAnalysis, Dict[str, Any]]]):
        self.analyze_fn = analyze_fn

    def analyze(self, image_path: str, instruction: str = "") -> SpatialAnalysis:
        result = self.analyze_fn(image_path, instruction)
        return result if isinstance(result, SpatialAnalysis) else SpatialAnalysis.from_dict(result)


class QwenVLSpatialAnalyzer:
    """Optional Qwen-VL semantic grounding adapter.

    The adapter only supplies semantic composition. Pixel-accurate navigation
    remains the responsibility of :class:`TopDownMapExtractor` and the native
    path planner.
    """

    DEFAULT_PROMPT = """Analyze this navigation image as spatial evidence.
Return one JSON object only, using coordinates normalized to 0..1000:
{
  "image_type": "top_down|perspective|floor_plan|unknown",
  "summary": "short factual description",
  "entities": [{
    "name": "entity name",
    "category": "room|door|walkable|obstacle|stairs|hazard|landmark|unknown",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.0,
    "traversable": true,
    "attributes": {}
  }],
  "connectivity": [{"from": "name", "to": "name", "via": "door/path"}],
  "hazards": [],
  "unknown_regions": [],
  "confidence": 0.0,
  "caveats": []
}
Do not invent hidden geometry. Perspective images must mark occluded and
out-of-frame space as unknown. Do not calculate routes."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        max_new_tokens: int = 1200,
    ) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        max_new_tokens: int = 1200,
    ) -> "QwenVLSpatialAnalyzer":
        try:
            from transformers import AutoProcessor
            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError as error:
            raise ImportError(
                "A recent transformers release is required for Qwen-VL spatial analysis"
            ) from error
        model = ModelClass.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        return cls(model, processor, max_new_tokens=max_new_tokens)

    def analyze(self, image_path: str, instruction: str = "") -> SpatialAnalysis:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("Pillow is required for Qwen-VL image loading") from error
        prompt = self.DEFAULT_PROMPT
        if instruction:
            prompt += "\nUser navigation context: " + str(instruction)
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        input_length = inputs["input_ids"].shape[1]
        generated = generated[:, input_length:]
        response = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return SpatialAnalysis.from_dict(_extract_json_object(response))


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Vision model did not return a JSON object")
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Vision model spatial output must be a JSON object")
    return value


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            source_y_start = max(0, -dy)
            source_y_end = min(height, height - dy)
            source_x_start = max(0, -dx)
            source_x_end = min(width, width - dx)
            target_y_start = source_y_start + dy
            target_y_end = source_y_end + dy
            target_x_start = source_x_start + dx
            target_x_end = source_x_end + dx
            result[target_y_start:target_y_end, target_x_start:target_x_end] |= mask[
                source_y_start:source_y_end, source_x_start:source_x_end
            ]
    return result
