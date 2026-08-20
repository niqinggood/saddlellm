"""Stateful agent runtime joining perception, planning, dynamics, and feedback.

The native world model deliberately stays independent from any one VLM or
language model.  A vision adapter turns an image into semantic evidence, the
spatial stack builds a deterministic world state, the learned RSSM can score
or imagine routes when a compatible checkpoint is configured, and feedback is
stored as replay data for later training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .SpatialPerception import (
    MapExtractionConfig,
    QwenVLSpatialAnalyzer,
    SpatialAnalysis,
    TopDownMapExtractor,
)
from .SpatialPlanner import (
    GridPathPlanner,
    GridPoint,
    OccupancyGrid,
    SpatialPlannerConfig,
)
from .SpatialWorldModel import (
    SpatialObservationEncoder,
    SpatialTensorObservationEncoder,
    SpatialWorldModelCoordinator,
    WorldModelRouteScorer,
)


WORLD_STATE_SCHEMA = "saddle.world-state.v1"
WORLD_PLAN_SCHEMA = "saddle.world-plan.v1"
WORLD_SIMULATION_SCHEMA = "saddle.world-simulation.v1"
WORLD_FEEDBACK_SCHEMA = "saddle.world-feedback.v1"
WORLD_REPLAY_SCHEMA = "spatial_features_v1"

PointLike = Union[str, Sequence[float]]


class WorldAgentError(RuntimeError):
    """Structured runtime error shared by the Python and HTTP interfaces."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str = "invalid_request",
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)


@dataclass
class WorldAgentSettings:
    """Small, serializable configuration for a local WorldAgent runtime."""

    workspace: str = "outputs/world_agent"
    qwen_model: Optional[str] = None
    world_model_checkpoint: Optional[str] = None
    device: str = "auto"
    allow_perspective: bool = False
    max_upload_mb: int = 25
    max_image_pixels: int = 40_000_000
    map_config: MapExtractionConfig = field(default_factory=MapExtractionConfig)
    planner_config: SpatialPlannerConfig = field(
        default_factory=lambda: SpatialPlannerConfig(clearance_weight=0.25)
    )

    def __post_init__(self) -> None:
        if self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be positive")
        if self.max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")
        if not isinstance(self.map_config, MapExtractionConfig):
            self.map_config = MapExtractionConfig(**dict(self.map_config))
        if not isinstance(self.planner_config, SpatialPlannerConfig):
            self.planner_config = SpatialPlannerConfig(**dict(self.planner_config))

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WorldAgentSettings":
        values = dict(data or {})
        if "map" in values and "map_config" not in values:
            values["map_config"] = values.pop("map")
        if "planner" in values and "planner_config" not in values:
            values["planner_config"] = values.pop("planner")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown WorldAgent settings: {', '.join(unknown)}")
        return cls(**values)

    @classmethod
    def from_file(cls, path: Union[str, os.PathLike]) -> "WorldAgentSettings":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"WorldAgent config not found: {source}")
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            values = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as error:
                raise ImportError("PyYAML is required for WorldAgent YAML configs") from error
            values = yaml.safe_load(text)
        if not isinstance(values, dict):
            raise ValueError("WorldAgent config must contain an object")
        settings = cls.from_dict(values)
        config_root = source.resolve().parent
        if not Path(settings.workspace).is_absolute():
            settings.workspace = str((config_root / settings.workspace).resolve())
        if settings.world_model_checkpoint and not Path(
            settings.world_model_checkpoint
        ).is_absolute():
            settings.world_model_checkpoint = str(
                (config_root / settings.world_model_checkpoint).resolve()
            )
        if settings.qwen_model and not Path(settings.qwen_model).is_absolute():
            local_qwen = config_root / settings.qwen_model
            if local_qwen.exists():
                settings.qwen_model = str(local_qwen.resolve())
        return settings

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["map"] = result.pop("map_config")
        result["planner"] = result.pop("planner_config")
        return result


@dataclass
class WorldObservation:
    observation_id: str
    modality: str
    source: str
    source_name: str
    sha256: str
    instruction: str = ""
    captured_at: str = field(default_factory=lambda: _utc_now())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldObservation":
        return cls(**{item.name: data.get(item.name) for item in fields(cls) if item.name in data})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorldState:
    state_id: str
    observation: WorldObservation
    grid: OccupancyGrid
    analysis: SpatialAnalysis
    confidence: Dict[str, Any]
    limitations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utc_now())
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = WORLD_STATE_SCHEMA

    def to_dict(self, include_grid: bool = True) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "state_id": self.state_id,
            "observation": self.observation.to_dict(),
            "map": _grid_to_dict(self.grid, include_grid=include_grid),
            "analysis": self.analysis.to_dict(),
            "confidence": dict(self.confidence),
            "limitations": list(self.limitations),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldState":
        return cls(
            schema=str(data.get("schema", WORLD_STATE_SCHEMA)),
            state_id=str(data["state_id"]),
            observation=WorldObservation.from_dict(data["observation"]),
            grid=_grid_from_dict(data["map"]),
            analysis=SpatialAnalysis.from_dict(data.get("analysis", {})),
            confidence=dict(data.get("confidence", {})),
            limitations=list(data.get("limitations", [])),
            created_at=str(data.get("created_at", _utc_now())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class WorldPlan:
    plan_id: str
    state_id: str
    instruction: str
    start: GridPoint
    goal: GridPoint
    routes: List[Dict[str, Any]]
    selected_route_id: str
    ranking: Dict[str, Any]
    explanation: Dict[str, Any]
    confidence: Dict[str, Any]
    limitations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utc_now())
    schema: str = WORLD_PLAN_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_id": self.plan_id,
            "state_id": self.state_id,
            "instruction": self.instruction,
            "start": list(self.start),
            "goal": list(self.goal),
            "routes": self.routes,
            "selected_route_id": self.selected_route_id,
            "ranking": self.ranking,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "limitations": self.limitations,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldPlan":
        return cls(
            schema=str(data.get("schema", WORLD_PLAN_SCHEMA)),
            plan_id=str(data["plan_id"]),
            state_id=str(data["state_id"]),
            instruction=str(data.get("instruction", "")),
            start=_grid_point(data["start"]),
            goal=_grid_point(data["goal"]),
            routes=list(data.get("routes", [])),
            selected_route_id=str(data["selected_route_id"]),
            ranking=dict(data.get("ranking", {})),
            explanation=dict(data.get("explanation", {})),
            confidence=dict(data.get("confidence", {})),
            limitations=list(data.get("limitations", [])),
            created_at=str(data.get("created_at", _utc_now())),
        )


@dataclass
class WorldSimulation:
    simulation_id: str
    plan_id: str
    state_id: str
    route_id: str
    mode: str
    geometry: Dict[str, Any]
    learned_prediction: Optional[Dict[str, Any]]
    confidence: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utc_now())
    schema: str = WORLD_SIMULATION_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldSimulation":
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class WorldFeedback:
    feedback_id: str
    plan_id: str
    state_id: str
    route_id: str
    outcome: str
    actual_path: List[GridPoint]
    comparison: Dict[str, Any]
    replay: Dict[str, Any]
    simulation_id: Optional[str] = None
    note: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    schema: str = WORLD_FEEDBACK_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["actual_path"] = [list(point) for point in self.actual_path]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldFeedback":
        values = dict(data)
        values["actual_path"] = [_grid_point(point) for point in values.get("actual_path", [])]
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


class WorldAgentStore:
    """Filesystem-backed state/event store with append-only training replay."""

    COLLECTIONS = {
        "states": "state_id",
        "plans": "plan_id",
        "simulations": "simulation_id",
        "feedback": "feedback_id",
    }

    def __init__(self, workspace: Union[str, os.PathLike]) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        for collection in (*self.COLLECTIONS, "observations", "replay"):
            (self.workspace / collection).mkdir(parents=True, exist_ok=True)
        self.events_path = self.workspace / "events.jsonl"
        self.replay_path = self.workspace / "replay" / "spatial_feedback.jsonl"
        self._lock = threading.RLock()

    def save(self, collection: str, entity_id: str, payload: Dict[str, Any]) -> Path:
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Unknown WorldAgent collection: {collection}")
        entity_id = _entity_id(entity_id)
        destination = self.workspace / collection / f"{entity_id}.json"
        temporary = destination.with_suffix(".json.tmp")
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        with self._lock:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(destination)
            self._append_jsonl(
                self.events_path,
                {
                    "event": collection.rstrip("s"),
                    "entity_id": entity_id,
                    "schema": payload.get("schema"),
                    "created_at": payload.get("created_at", _utc_now()),
                },
            )
        return destination

    def load(self, collection: str, entity_id: str) -> Dict[str, Any]:
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Unknown WorldAgent collection: {collection}")
        entity_id = _entity_id(entity_id)
        path = self.workspace / collection / f"{entity_id}.json"
        if not path.is_file():
            raise WorldAgentError(
                f"WorldAgent {collection.rstrip('s')} not found: {entity_id}",
                status_code=404,
                code="not_found",
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, collection: str, limit: int = 50) -> List[Dict[str, Any]]:
        if collection not in self.COLLECTIONS:
            raise ValueError(f"Unknown WorldAgent collection: {collection}")
        limit = max(1, min(int(limit), 500))
        paths = sorted(
            (self.workspace / collection).glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return [json.loads(path.read_text(encoding="utf-8")) for path in paths[:limit]]

    def append_replay(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._append_jsonl(self.replay_path, record)

    def summary(self) -> Dict[str, Any]:
        counts = {
            collection: sum(1 for _ in (self.workspace / collection).glob("*.json"))
            for collection in self.COLLECTIONS
        }
        outcomes: Dict[str, int] = {}
        for path in (self.workspace / "feedback").glob("*.json"):
            try:
                outcome = str(json.loads(path.read_text(encoding="utf-8")).get("outcome", "unknown"))
            except (OSError, json.JSONDecodeError):
                continue
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        replay_records = 0
        if self.replay_path.is_file():
            with self.replay_path.open("r", encoding="utf-8") as handle:
                replay_records = sum(1 for line in handle if line.strip())
        return {
            "workspace": str(self.workspace),
            "counts": counts,
            "feedback_outcomes": outcomes,
            "replay_records": replay_records,
            "replay_path": str(self.replay_path),
        }

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


class WorldAgentRuntime:
    """Unified analyze -> plan -> simulate -> feedback runtime."""

    def __init__(
        self,
        settings: Optional[WorldAgentSettings] = None,
        store: Optional[WorldAgentStore] = None,
        analyzer_loader: Optional[Callable[[], Any]] = None,
        world_model_loader: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.settings = settings or WorldAgentSettings()
        self.store = store or WorldAgentStore(self.settings.workspace)
        self._analyzer_loader = analyzer_loader
        self._world_model_loader = world_model_loader
        self._analyzer: Optional[Any] = None
        self._world_model: Optional[Any] = None
        self._analyzer_lock = threading.Lock()
        self._world_model_lock = threading.Lock()
        self.observation_encoder = SpatialObservationEncoder()

    def capabilities(self) -> Dict[str, Any]:
        return {
            "protocol_version": 1,
            "stages": ["analyze", "plan", "simulate", "feedback"],
            "semantic": {
                "geometry": True,
                "qwen_vl": {
                    "configured": bool(self.settings.qwen_model or self._analyzer_loader),
                    "loaded": self._analyzer is not None,
                    "model": _display_name(self.settings.qwen_model),
                },
            },
            "planning": {
                "occupancy_grid": True,
                "top_k_routes": True,
                "entity_endpoints": True,
            },
            "world_model": {
                "configured": self._world_model_available,
                "loaded": self._world_model is not None,
                "checkpoint": _display_name(self.settings.world_model_checkpoint),
                "geometric_simulation": True,
                "learned_imagination": self._world_model_available,
            },
            "memory": self.store.summary(),
            "limits": {
                "max_upload_mb": self.settings.max_upload_mb,
                "max_image_pixels": self.settings.max_image_pixels,
                "max_routes": 16,
            },
        }

    @property
    def _world_model_available(self) -> bool:
        return bool(self.settings.world_model_checkpoint or self._world_model_loader)

    def analyze_image(
        self,
        image_path: Union[str, os.PathLike],
        instruction: str = "",
        semantic_backend: Literal["disabled", "qwen-vl"] = "disabled",
        map_config: Optional[Union[MapExtractionConfig, Dict[str, Any]]] = None,
        observation_id: Optional[str] = None,
        source_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorldState:
        path = Path(image_path).resolve()
        if not path.is_file():
            raise WorldAgentError(f"Spatial image not found: {path}", 404, "not_found")
        if semantic_backend not in {"disabled", "qwen-vl"}:
            raise WorldAgentError(f"Unsupported semantic backend: {semantic_backend}")
        extraction = _coerce_map_config(map_config or self.settings.map_config)
        grid = TopDownMapExtractor(extraction).extract(str(path))
        if semantic_backend == "qwen-vl":
            analyzer = self._get_analyzer()
            analysis = analyzer.analyze(str(path), str(instruction))
        else:
            analysis = SpatialAnalysis(
                image_type="top_down",
                summary="Geometry extracted from image luminance.",
                confidence=0.5,
                caveats=[
                    "No vision-language analyzer was configured; semantic labels are unavailable."
                ],
            )

        counts = grid.to_dict()["cell_counts"]
        total = max(grid.width * grid.height, 1)
        known_ratio = (counts["free"] + counts["blocked"]) / total
        semantic_confidence = float(analysis.confidence)
        overall = (
            0.55 * known_ratio + 0.45 * semantic_confidence
            if semantic_backend == "qwen-vl"
            else 0.70 * known_ratio + 0.30 * semantic_confidence
        )
        limitations = list(dict.fromkeys(analysis.caveats))
        if semantic_backend == "disabled":
            limitations.append("Object names, room semantics, and hazards were not visually grounded.")
        if counts["unknown"]:
            limitations.append("Unknown map cells require additional observation before safe execution.")
        if analysis.image_type == "perspective":
            limitations.append(
                "A single perspective image does not reveal occluded or out-of-frame geometry."
            )
        if analysis.unknown_regions:
            limitations.append("The semantic analyzer reported unresolved spatial regions.")

        observation = WorldObservation(
            observation_id=_entity_id(observation_id or uuid.uuid4().hex),
            modality="image",
            source=str(path),
            source_name=Path(source_name or path.name).name,
            sha256=_sha256(path),
            instruction=str(instruction),
            metadata=dict(metadata or {}),
        )
        state = WorldState(
            state_id=uuid.uuid4().hex,
            observation=observation,
            grid=grid,
            analysis=analysis,
            confidence={
                "overall": round(_clamp(overall), 6),
                "geometry_coverage": round(float(known_ratio), 6),
                "semantic": round(semantic_confidence, 6),
                "calibrated": False,
            },
            limitations=list(dict.fromkeys(limitations)),
            metadata={
                "semantic_backend": (
                    "qwen-vl" if semantic_backend == "qwen-vl" else "geometry-only"
                ),
                "map_extraction": asdict(extraction),
            },
        )
        self.store.save("states", state.state_id, state.to_dict(include_grid=True))
        return state

    def analyze_bytes(
        self,
        payload: bytes,
        instruction: str = "",
        semantic_backend: Literal["disabled", "qwen-vl"] = "disabled",
        map_config: Optional[Union[MapExtractionConfig, Dict[str, Any]]] = None,
        source_name: str = "upload",
    ) -> WorldState:
        maximum = self.settings.max_upload_mb * 1024 * 1024
        if len(payload) > maximum:
            raise WorldAgentError(
                f"Image exceeds {self.settings.max_upload_mb} MB upload limit",
                status_code=413,
                code="upload_too_large",
            )
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as error:
            raise WorldAgentError("Pillow is required for image uploads", 500, "dependency_error") from error
        import io

        observation_id = uuid.uuid4().hex
        destination = self.store.workspace / "observations" / f"{observation_id}.png"
        temporary = destination.with_suffix(".png.tmp")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                width, height = source.size
                if width <= 0 or height <= 0:
                    raise WorldAgentError("Image dimensions must be positive")
                if width * height > self.settings.max_image_pixels:
                    raise WorldAgentError(
                        "Image pixel count exceeds the configured limit",
                        status_code=413,
                        code="image_too_large",
                    )
                source.convert("RGB").save(temporary, format="PNG", optimize=True)
            temporary.replace(destination)
        except WorldAgentError:
            temporary.unlink(missing_ok=True)
            raise
        except (UnidentifiedImageError, OSError) as error:
            temporary.unlink(missing_ok=True)
            raise WorldAgentError("Upload is not a supported image") from error
        try:
            return self.analyze_image(
                destination,
                instruction=instruction,
                semantic_backend=semantic_backend,
                map_config=map_config,
                observation_id=observation_id,
                source_name=source_name,
                metadata={"width": int(width), "height": int(height)},
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def plan(
        self,
        state_id: str,
        start: PointLike,
        goal: PointLike,
        instruction: str = "",
        route_count: Optional[int] = None,
        use_world_model: bool = False,
        planner_config: Optional[Union[SpatialPlannerConfig, Dict[str, Any]]] = None,
        allow_perspective: Optional[bool] = None,
        language: str = "zh-CN",
    ) -> WorldPlan:
        state = self.get_state(state_id)
        requested_routes = int(route_count or self.settings.planner_config.route_count)
        if not 1 <= requested_routes <= 16:
            raise WorldAgentError("route_count must be between 1 and 16")
        config = _coerce_planner_config(planner_config or self.settings.planner_config)
        config = replace(config, route_count=requested_routes)
        planner = GridPathPlanner(config)
        perspective_enabled = (
            self.settings.allow_perspective
            if allow_perspective is None
            else bool(allow_perspective)
        )
        coordinator = SpatialWorldModelCoordinator(
            planner=planner,
            allow_perspective=perspective_enabled,
        )
        try:
            spatial = coordinator.plan_grid(
                state.grid,
                state.analysis,
                start=start,
                goal=goal,
                instruction=str(instruction),
                route_count=requested_routes,
                image_path=state.observation.source,
            )
        except ValueError as error:
            raise WorldAgentError(str(error)) from error

        ranking: Dict[str, Any] = {
            "mode": "geometry",
            "semantic_backend": state.metadata.get("semantic_backend", "unknown"),
            "world_model": None,
            "observation_schema": None,
            "calibrated": False,
        }
        if use_world_model:
            runtime = self._get_world_model()
            observation, observation_schema = self._encode_for_world_model(
                runtime, state.grid, spatial.start, spatial.goal
            )
            latent_state = runtime.encode_observation(observation)
            try:
                WorldModelRouteScorer(runtime).score(latent_state, spatial.routes)
            except ValueError as error:
                raise WorldAgentError(str(error), code="world_model_schema_mismatch") from error
            ranking.update(
                {
                    "mode": "world-model",
                    "world_model": getattr(runtime, "backend", "rssm"),
                    "observation_schema": observation_schema,
                }
            )

        routes: List[Dict[str, Any]] = []
        for index, route in enumerate(spatial.routes, start=1):
            item = route.to_dict()
            item["route_id"] = f"route-{index}"
            item["source_points"] = [
                list(state.grid.scale_to_source(point)) for point in route.points
            ]
            routes.append(item)
        if not routes:
            raise WorldAgentError("The planner returned no route")
        selected = routes[0]
        state_confidence = float(state.confidence.get("overall", 0.0))
        route_confidence = _clamp(1.0 - float(selected.get("risk", 1.0)))
        limitations = list(state.limitations)
        if use_world_model:
            limitations.append(
                "Learned route scores are checkpoint-dependent and are not calibrated safety guarantees."
            )
        else:
            limitations.append(
                "Routes were checked against the extracted map only; no learned dynamics were used."
            )
        plan = WorldPlan(
            plan_id=uuid.uuid4().hex,
            state_id=state.state_id,
            instruction=str(instruction),
            start=spatial.start,
            goal=spatial.goal,
            routes=routes,
            selected_route_id=str(selected["route_id"]),
            ranking=ranking,
            explanation=_route_explanation(
                selected,
                route_count=len(routes),
                ranking_mode=str(ranking["mode"]),
                language=language,
            ),
            confidence={
                "overall": round(_clamp(state_confidence * route_confidence), 6),
                "state": round(state_confidence, 6),
                "route_geometry": round(route_confidence, 6),
                "calibrated": False,
            },
            limitations=list(dict.fromkeys(limitations)),
        )
        self.store.save("plans", plan.plan_id, plan.to_dict())
        return plan

    def simulate(
        self,
        plan_id: str,
        route_id: Optional[str] = None,
        mode: Literal["auto", "geometry", "world_model"] = "auto",
        max_world_model_steps: int = 64,
    ) -> WorldSimulation:
        if mode not in {"auto", "geometry", "world_model"}:
            raise WorldAgentError(f"Unsupported simulation mode: {mode}")
        if max_world_model_steps <= 0:
            raise WorldAgentError("max_world_model_steps must be positive")
        plan = self.get_plan(plan_id)
        state = self.get_state(plan.state_id)
        selected_route_id = str(route_id or plan.selected_route_id)
        route = _find_route(plan, selected_route_id)
        geometry = _simulate_geometry(state.grid, route, plan.goal)
        learned: Optional[Dict[str, Any]] = None
        warnings: List[str] = []
        resolved_mode = "geometry"

        should_use_model = mode == "world_model" or (
            mode == "auto" and self._world_model_available
        )
        if should_use_model:
            try:
                learned = self._simulate_learned(
                    state,
                    plan,
                    route,
                    max_steps=int(max_world_model_steps),
                )
                resolved_mode = "world-model"
            except Exception as error:
                if mode == "world_model":
                    if isinstance(error, WorldAgentError):
                        raise
                    raise WorldAgentError(
                        str(error), code="world_model_simulation_failed"
                    ) from error
                warnings.append(
                    f"World-model prediction was unavailable; geometry-only validation was used: {error}"
                )
        elif mode == "auto":
            warnings.append(
                "No world-model checkpoint is configured; this is deterministic geometry validation."
            )

        if state.grid.to_dict()["cell_counts"]["unknown"]:
            warnings.append("The map contains unknown cells that may change real execution outcomes.")
        confidence = {
            "overall": round(
                _clamp(
                    float(plan.confidence.get("overall", 0.0))
                    * (1.0 if geometry["route_valid"] else 0.0)
                ),
                6,
            ),
            "state": float(state.confidence.get("overall", 0.0)),
            "prediction_type": (
                "learned-open-loop" if learned is not None else "deterministic-map-check"
            ),
            "calibrated": False,
        }
        simulation = WorldSimulation(
            simulation_id=uuid.uuid4().hex,
            plan_id=plan.plan_id,
            state_id=state.state_id,
            route_id=selected_route_id,
            mode=resolved_mode,
            geometry=geometry,
            learned_prediction=learned,
            confidence=confidence,
            warnings=warnings,
        )
        self.store.save("simulations", simulation.simulation_id, simulation.to_dict())
        return simulation

    def record_feedback(
        self,
        plan_id: str,
        outcome: Literal["success", "collision", "blocked", "cancelled", "unknown"],
        route_id: Optional[str] = None,
        actual_path: Optional[Sequence[Sequence[float]]] = None,
        coordinate_space: Literal["grid", "source"] = "grid",
        simulation_id: Optional[str] = None,
        note: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> WorldFeedback:
        allowed_outcomes = {"success", "collision", "blocked", "cancelled", "unknown"}
        if outcome not in allowed_outcomes:
            raise WorldAgentError(f"Unsupported feedback outcome: {outcome}")
        if coordinate_space not in {"grid", "source"}:
            raise WorldAgentError("coordinate_space must be 'grid' or 'source'")
        plan = self.get_plan(plan_id)
        state = self.get_state(plan.state_id)
        selected_route_id = str(route_id or plan.selected_route_id)
        route = _find_route(plan, selected_route_id)
        if simulation_id is not None:
            simulation = self.get_simulation(simulation_id)
            if simulation.plan_id != plan.plan_id:
                raise WorldAgentError("simulation_id does not belong to plan_id")

        actual: List[GridPoint] = []
        for raw_point in actual_path or []:
            point = (
                state.grid.scale_from_source(raw_point)
                if coordinate_space == "source"
                else _grid_point(raw_point)
            )
            if not state.grid.in_bounds(point):
                raise WorldAgentError(f"Actual path point is outside the map: {point}")
            actual.append(point)
        comparison = _compare_paths(
            [_grid_point(point) for point in route["points"]],
            actual,
            plan.goal,
        )
        feedback_id = uuid.uuid4().hex
        replay: Dict[str, Any] = {
            "recorded": False,
            "path": str(self.store.replay_path),
            "reason": "actual_path with at least two points is required",
        }
        if len(actual) >= 2:
            record = self._feedback_replay_record(
                feedback_id,
                state,
                plan,
                selected_route_id,
                outcome,
                actual,
            )
            self.store.append_replay(record)
            replay = {
                "recorded": True,
                "path": str(self.store.replay_path),
                "episode_id": record["episode_id"],
                "schema": WORLD_REPLAY_SCHEMA,
                "transitions": len(record["actions"]),
            }
        feedback = WorldFeedback(
            feedback_id=feedback_id,
            plan_id=plan.plan_id,
            state_id=state.state_id,
            route_id=selected_route_id,
            outcome=outcome,
            actual_path=actual,
            comparison=comparison,
            replay=replay,
            simulation_id=simulation_id,
            note=str(note),
            metrics=dict(metrics or {}),
        )
        self.store.save("feedback", feedback.feedback_id, feedback.to_dict())
        return feedback

    def run_image(
        self,
        image_path: Union[str, os.PathLike],
        start: PointLike,
        goal: PointLike,
        instruction: str = "",
        semantic_backend: Literal["disabled", "qwen-vl"] = "disabled",
        route_count: int = 3,
        use_world_model: bool = False,
        simulation_mode: Literal["auto", "geometry", "world_model"] = "auto",
    ) -> Dict[str, Any]:
        state = self.analyze_image(
            image_path,
            instruction=instruction,
            semantic_backend=semantic_backend,
        )
        plan = self.plan(
            state.state_id,
            start=start,
            goal=goal,
            instruction=instruction,
            route_count=route_count,
            use_world_model=use_world_model,
        )
        simulation = self.simulate(plan.plan_id, mode=simulation_mode)
        return {
            "state": state.to_dict(include_grid=True),
            "plan": plan.to_dict(),
            "simulation": simulation.to_dict(),
            "memory": self.memory_summary(),
        }

    def get_state(self, state_id: str) -> WorldState:
        return WorldState.from_dict(self.store.load("states", state_id))

    def get_plan(self, plan_id: str) -> WorldPlan:
        return WorldPlan.from_dict(self.store.load("plans", plan_id))

    def get_simulation(self, simulation_id: str) -> WorldSimulation:
        return WorldSimulation.from_dict(self.store.load("simulations", simulation_id))

    def get_feedback(self, feedback_id: str) -> WorldFeedback:
        return WorldFeedback.from_dict(self.store.load("feedback", feedback_id))

    def memory_summary(self) -> Dict[str, Any]:
        return self.store.summary()

    def _get_analyzer(self) -> Any:
        if not (self.settings.qwen_model or self._analyzer_loader):
            raise WorldAgentError(
                "Qwen-VL is not configured for this WorldAgent runtime",
                status_code=409,
                code="model_unavailable",
            )
        with self._analyzer_lock:
            if self._analyzer is None:
                self._analyzer = (
                    self._analyzer_loader()
                    if self._analyzer_loader is not None
                    else QwenVLSpatialAnalyzer.from_pretrained(
                        str(self.settings.qwen_model),
                        device_map=self.settings.device,
                    )
                )
        return self._analyzer

    def _get_world_model(self) -> Any:
        if not self._world_model_available:
            raise WorldAgentError(
                "A spatial world-model checkpoint is not configured",
                status_code=409,
                code="model_unavailable",
            )
        with self._world_model_lock:
            if self._world_model is None:
                if self._world_model_loader is not None:
                    self._world_model = self._world_model_loader()
                else:
                    from .WorldModelInference import WorldModelRuntime

                    self._world_model = WorldModelRuntime.from_pretrained(
                        str(self.settings.world_model_checkpoint),
                        device=self.settings.device,
                    )
        return self._world_model

    def _encode_for_world_model(
        self,
        runtime: Any,
        grid: OccupancyGrid,
        start: GridPoint,
        goal: GridPoint,
    ) -> Tuple[np.ndarray, str]:
        shape = tuple(int(value) for value in runtime.model.config.observation_shape)
        if len(shape) == 1:
            return (
                self.observation_encoder.encode(
                    grid,
                    start,
                    goal,
                    shape,
                    clearance=GridPathPlanner().clearance_map(grid),
                ),
                self.observation_encoder.schema,
            )
        if len(shape) == 3 and shape[0] == len(SpatialTensorObservationEncoder.channel_names):
            if shape[-2] != shape[-1]:
                raise WorldAgentError(
                    f"Spatial tensor checkpoint needs a square crop, got {shape}",
                    code="world_model_schema_mismatch",
                )
            encoder = SpatialTensorObservationEncoder(
                crop_size=shape[-1],
                sensor_radius=min(6, max(1, shape[-1] // 2 - 1)),
            )
            return (
                encoder.encode(grid, start, goal, observation_shape=shape),
                encoder.schema,
            )
        raise WorldAgentError(
            f"Unsupported spatial world-model observation shape: {shape}",
            code="world_model_schema_mismatch",
        )

    def _simulate_learned(
        self,
        state: WorldState,
        plan: WorldPlan,
        route: Dict[str, Any],
        max_steps: int,
    ) -> Dict[str, Any]:
        runtime = self._get_world_model()
        config = runtime.model.config
        if config.action_type != "continuous" or config.action_dim < 2:
            raise WorldAgentError(
                "Automatic spatial simulation requires a continuous world model with action_dim >= 2",
                code="world_model_schema_mismatch",
            )
        observation, schema = self._encode_for_world_model(
            runtime, state.grid, plan.start, plan.goal
        )
        latent_state = runtime.encode_observation(observation)
        actions, sampled_points = _route_actions(
            [_grid_point(point) for point in route["points"]],
            int(config.action_dim),
            max_steps,
        )
        imagination = runtime.rollout(latent_state, actions, deterministic=True)
        rewards = _batch_vector(imagination.rewards)
        continuation = _batch_vector(imagination.continuation)
        discounts = np.power(0.99, np.arange(rewards.size, dtype=np.float32))
        survival = np.cumprod(np.concatenate(([1.0], continuation[:-1])))
        learned: Dict[str, Any] = {
            "backend": getattr(runtime, "backend", "rssm"),
            "observation_schema": schema,
            "prediction": "open-loop-latent-imagination",
            "calibrated": False,
            "sampled_points": [list(point) for point in sampled_points],
            "actions": actions.tolist(),
            "predicted_rewards": rewards.tolist(),
            "continuation_probability": continuation.tolist(),
            "discounted_return": float(np.sum(rewards * discounts * survival)),
        }
        collision = getattr(imagination, "collision_probability", None)
        if collision is not None:
            collision_values = _batch_step_means(collision)
            learned["collision_probability"] = collision_values.tolist()
            learned["route_collision_probability"] = float(
                1.0 - np.prod(1.0 - np.clip(collision_values, 0.0, 1.0))
            )
            learned["collision_aggregation"] = "independent-step approximation"
        else:
            learned["collision_probability"] = None
            learned["route_collision_probability"] = None
        occupancy = getattr(imagination, "occupancy_logits", None)
        if occupancy is not None:
            learned["future_occupancy"] = _summarize_occupancy(occupancy)
        ego_motion = getattr(imagination, "ego_motion", None)
        if ego_motion is not None:
            learned["predicted_ego_motion"] = ego_motion.detach().cpu()[0].tolist()
        return learned

    def _feedback_replay_record(
        self,
        feedback_id: str,
        state: WorldState,
        plan: WorldPlan,
        route_id: str,
        outcome: str,
        actual_path: Sequence[GridPoint],
    ) -> Dict[str, Any]:
        clearance = GridPathPlanner().clearance_map(state.grid)
        observations = [
            self.observation_encoder.encode(
                state.grid,
                point,
                plan.goal,
                (16,),
                clearance=clearance,
            ).tolist()
            for point in actual_path
        ]
        actions: List[List[float]] = []
        rewards: List[float] = []
        collisions: List[float] = []
        for index, (current, following) in enumerate(zip(actual_path, actual_path[1:])):
            dx = float(following[0] - current[0])
            dy = float(following[1] - current[1])
            norm = max(math.hypot(dx, dy), 1.0)
            actions.append([dx / norm, dy / norm])
            collided = not state.grid.traversable(following)
            if index == len(actual_path) - 2 and outcome == "collision":
                collided = True
            collisions.append(float(collided))
            before = _normalized_goal_distance(state.grid, current, plan.goal)
            after = _normalized_goal_distance(state.grid, following, plan.goal)
            reward = before - after - float(collided)
            if index == len(actual_path) - 2 and outcome == "success":
                reward += 1.0
            rewards.append(float(reward))
        dones = [False] * len(actions)
        dones[-1] = True
        return {
            "episode_id": f"feedback-{feedback_id}",
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "collisions": collisions,
            "metadata": {
                "schema": WORLD_REPLAY_SCHEMA,
                "source": "world-agent-feedback",
                "feedback_id": feedback_id,
                "state_id": state.state_id,
                "plan_id": plan.plan_id,
                "route_id": route_id,
                "outcome": outcome,
                "map": state.grid.to_dict(include_cells=False),
            },
        }


def _grid_to_dict(grid: OccupancyGrid, include_grid: bool) -> Dict[str, Any]:
    result = grid.to_dict(include_cells=False)
    if include_grid:
        result["cell_encoding"] = "rle-v1"
        result["cell_runs"] = _rle_encode(grid.cells.reshape(-1))
    return result


def _grid_from_dict(data: Dict[str, Any]) -> OccupancyGrid:
    if "cells" in data:
        cells = np.asarray(data["cells"], dtype=np.uint8)
    elif data.get("cell_encoding") == "rle-v1" and "cell_runs" in data:
        flat = _rle_decode(data["cell_runs"], int(data["width"]) * int(data["height"]))
        cells = flat.reshape(int(data["height"]), int(data["width"]))
    else:
        raise ValueError("Serialized WorldState does not contain occupancy cells")
    return OccupancyGrid(
        cells,
        resolution=float(data.get("resolution", 1.0)),
        source_size=(
            tuple(int(value) for value in data["source_size"])
            if data.get("source_size") is not None
            else None
        ),
    )


def _rle_encode(values: np.ndarray) -> List[List[int]]:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if not flat.size:
        return []
    runs: List[List[int]] = []
    current = int(flat[0])
    count = 1
    for raw in flat[1:]:
        value = int(raw)
        if value == current:
            count += 1
        else:
            runs.append([current, count])
            current, count = value, 1
    runs.append([current, count])
    return runs


def _rle_decode(runs: Iterable[Sequence[int]], expected: int) -> np.ndarray:
    chunks: List[np.ndarray] = []
    total = 0
    for run in runs:
        if len(run) != 2:
            raise ValueError("Every occupancy RLE run must contain value and count")
        value, count = int(run[0]), int(run[1])
        if value not in {OccupancyGrid.FREE, OccupancyGrid.BLOCKED, OccupancyGrid.UNKNOWN}:
            raise ValueError(f"Invalid occupancy RLE value: {value}")
        if count <= 0:
            raise ValueError("Occupancy RLE counts must be positive")
        chunks.append(np.full(count, value, dtype=np.uint8))
        total += count
    if total != expected:
        raise ValueError(f"Occupancy RLE size mismatch: expected {expected}, got {total}")
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)


def _simulate_geometry(
    grid: OccupancyGrid,
    route: Dict[str, Any],
    goal: GridPoint,
) -> Dict[str, Any]:
    points = [_grid_point(point) for point in route.get("points", [])]
    if not points:
        raise WorldAgentError("Route has no points")
    steps: List[Dict[str, Any]] = []
    collisions = 0
    distance = 0.0
    for index, point in enumerate(points):
        in_bounds = grid.in_bounds(point)
        traversable = in_bounds and grid.traversable(point)
        corner_clear = True
        action = [0, 0]
        step_distance = 0.0
        if index:
            previous = points[index - 1]
            dx, dy = point[0] - previous[0], point[1] - previous[1]
            action = [dx, dy]
            step_distance = math.hypot(dx, dy) * grid.resolution
            distance += step_distance
            if dx and dy:
                corner_clear = grid.traversable((previous[0] + dx, previous[1])) and grid.traversable(
                    (previous[0], previous[1] + dy)
                )
        collided = not (in_bounds and traversable and corner_clear)
        collisions += int(collided)
        steps.append(
            {
                "index": index,
                "position": list(point),
                "action": action,
                "traversable": bool(traversable),
                "corner_clear": bool(corner_clear),
                "collision": bool(collided),
                "cumulative_distance": float(distance),
            }
        )
    return {
        "validation": "deterministic-occupancy-grid",
        "route_valid": collisions == 0,
        "reached_goal": points[-1] == goal and collisions == 0,
        "collisions": collisions,
        "steps": steps,
        "step_count": max(len(points) - 1, 0),
        "distance": float(distance),
        "success_probability": None,
    }


def _route_actions(
    points: Sequence[GridPoint],
    action_dim: int,
    max_steps: int,
) -> Tuple[np.ndarray, List[GridPoint]]:
    if len(points) < 2:
        raise WorldAgentError("World-model simulation requires at least two route points")
    if len(points) - 1 > max_steps:
        indices = np.linspace(0, len(points) - 1, max_steps + 1).round().astype(int)
        sampled = [points[int(index)] for index in indices]
    else:
        sampled = list(points)
    actions = np.zeros((len(sampled) - 1, action_dim), dtype=np.float32)
    for index, (first, second) in enumerate(zip(sampled, sampled[1:])):
        dx = float(second[0] - first[0])
        dy = float(second[1] - first[1])
        norm = max(math.hypot(dx, dy), 1.0)
        actions[index, 0] = dx / norm
        actions[index, 1] = dy / norm
    return actions, sampled


def _route_explanation(
    route: Dict[str, Any],
    route_count: int,
    ranking_mode: str,
    language: str,
) -> Dict[str, Any]:
    learned = ranking_mode == "world-model"
    facts = {
        "length": float(route.get("length", 0.0)),
        "turns": int(route.get("turns", 0)),
        "minimum_clearance": float(route.get("minimum_clearance", 0.0)),
        "risk": float(route.get("risk", 1.0)),
        "predicted_return": route.get("predicted_return"),
        "combined_score": route.get("combined_score"),
    }
    if str(language).lower().startswith("zh"):
        basis = "世界模型综合评分最高" if learned else "几何代价与障碍风险最优"
        text = (
            f"推荐 {route['route_id']}：在 {route_count} 条可行路线中{basis}。"
            f"路线长度 {facts['length']:.2f}，转弯 {facts['turns']} 次，"
            f"最小净空 {facts['minimum_clearance']:.2f}，几何风险 {facts['risk']:.3f}。"
        )
    else:
        basis = "highest combined world-model score" if learned else "best geometric cost and obstacle risk"
        text = (
            f"Recommended {route['route_id']}: {basis} among {route_count} feasible routes. "
            f"Length {facts['length']:.2f}, {facts['turns']} turns, minimum clearance "
            f"{facts['minimum_clearance']:.2f}, geometric risk {facts['risk']:.3f}."
        )
    return {"text": text, "basis": ranking_mode, "facts": facts, "generated_by": "rules-v1"}


def _compare_paths(
    planned: Sequence[GridPoint],
    actual: Sequence[GridPoint],
    goal: GridPoint,
) -> Dict[str, Any]:
    if not actual:
        return {
            "actual_path_provided": False,
            "reached_goal": None,
            "endpoint_error_cells": None,
            "planned_cell_recall": None,
            "actual_cell_precision": None,
        }
    planned_cells = set(planned)
    actual_cells = set(actual)
    overlap = planned_cells & actual_cells
    return {
        "actual_path_provided": True,
        "reached_goal": actual[-1] == goal,
        "endpoint_error_cells": float(math.dist(actual[-1], goal)),
        "planned_cell_recall": len(overlap) / max(len(planned_cells), 1),
        "actual_cell_precision": len(overlap) / max(len(actual_cells), 1),
        "actual_steps": max(len(actual) - 1, 0),
        "planned_steps": max(len(planned) - 1, 0),
    }


def _find_route(plan: WorldPlan, route_id: str) -> Dict[str, Any]:
    for route in plan.routes:
        if str(route.get("route_id")) == str(route_id):
            return route
    raise WorldAgentError(f"Route not found in plan: {route_id}", 404, "not_found")


def _batch_vector(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim < 2 or array.shape[0] != 1:
        raise ValueError(f"Expected a single-batch prediction, got {array.shape}")
    return np.asarray(array[0], dtype=np.float32).reshape(-1)


def _batch_step_means(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim < 2 or array.shape[0] != 1:
        raise ValueError(f"Expected a single-batch prediction, got {array.shape}")
    steps = array[0]
    return np.asarray(steps, dtype=np.float32).reshape(steps.shape[0], -1).mean(axis=1)


def _summarize_occupancy(logits: Any) -> Dict[str, Any]:
    probabilities = logits.detach().cpu().softmax(dim=2)
    classes = probabilities.argmax(dim=2)[0].numpy()
    confidence = probabilities.max(dim=2).values[0].numpy()
    summaries = []
    for step, (class_map, confidence_map) in enumerate(zip(classes, confidence)):
        values, counts = np.unique(class_map, return_counts=True)
        summaries.append(
            {
                "step": step,
                "class_counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
                "mean_confidence": float(np.mean(confidence_map)),
            }
        )
    return {"classes": ["free", "blocked", "unknown"], "steps": summaries}


def _coerce_map_config(value: Union[MapExtractionConfig, Dict[str, Any]]) -> MapExtractionConfig:
    return value if isinstance(value, MapExtractionConfig) else MapExtractionConfig(**dict(value))


def _coerce_planner_config(
    value: Union[SpatialPlannerConfig, Dict[str, Any]],
) -> SpatialPlannerConfig:
    return value if isinstance(value, SpatialPlannerConfig) else SpatialPlannerConfig(**dict(value))


def _grid_point(value: Sequence[Any]) -> GridPoint:
    if len(value) != 2:
        raise WorldAgentError("A point must contain x and y")
    return int(round(float(value[0]))), int(round(float(value[1])))


def _normalized_goal_distance(
    grid: OccupancyGrid,
    point: GridPoint,
    goal: GridPoint,
) -> float:
    dx = (goal[0] - point[0]) / max(grid.width - 1, 1)
    dy = (goal[1] - point[1]) / max(grid.height - 1, 1)
    return math.hypot(dx, dy)


def _entity_id(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 32 or any(character not in "0123456789abcdef" for character in normalized):
        raise WorldAgentError("WorldAgent ids must be 32 lowercase hexadecimal characters")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_name(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    normalized = str(path).rstrip("/\\")
    return os.path.basename(normalized) or normalized


def _clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "WORLD_STATE_SCHEMA",
    "WORLD_PLAN_SCHEMA",
    "WORLD_SIMULATION_SCHEMA",
    "WORLD_FEEDBACK_SCHEMA",
    "WorldAgentError",
    "WorldAgentSettings",
    "WorldObservation",
    "WorldState",
    "WorldPlan",
    "WorldSimulation",
    "WorldFeedback",
    "WorldAgentStore",
    "WorldAgentRuntime",
]
