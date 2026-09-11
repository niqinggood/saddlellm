"""FastAPI surface for the stateful SaddleLLM WorldAgent runtime."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from fastapi import APIRouter, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .SpatialPerception import MapExtractionConfig
from .SpatialPlanner import SpatialPlannerConfig
from .WorldAgent import (
    WorldAgentError,
    WorldAgentRuntime,
    WorldAgentSettings,
)


Coordinate = Tuple[float, float]
Endpoint = Union[str, Coordinate]


class WorldAnalyzeRequest(BaseModel):
    instruction: str = Field(default="", max_length=4000)
    semantic_backend: Literal["disabled", "qwen-vl"] = "disabled"
    free_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    free_is_bright: Optional[bool] = None
    uncertainty_band: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    obstacle_dilation: Optional[int] = Field(default=None, ge=0, le=32)
    max_dimension: Optional[int] = Field(default=None, ge=32, le=2048)
    block_border: Optional[bool] = None
    resolution: Optional[float] = Field(default=None, gt=0.0, le=100.0)

    def map_config(self, base: MapExtractionConfig) -> MapExtractionConfig:
        values = asdict(base)
        for name in values:
            override = getattr(self, name)
            if override is not None:
                values[name] = override
        return MapExtractionConfig(**values)


class WorldPlanRequest(BaseModel):
    state_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    start: Endpoint
    goal: Endpoint
    instruction: str = Field(default="", max_length=4000)
    route_count: int = Field(default=3, ge=1, le=16)
    use_world_model: bool = False
    allow_perspective: Optional[bool] = None
    language: str = Field(default="zh-CN", max_length=16)
    diagonal: Optional[bool] = None
    allow_unknown: Optional[bool] = None
    unknown_cost: Optional[float] = Field(default=None, ge=1.0, le=100.0)
    clearance_weight: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    diversity_weight: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    diversity_radius: Optional[int] = Field(default=None, ge=0, le=64)
    max_detour_ratio: Optional[float] = Field(default=None, ge=1.0, le=20.0)

    @field_validator("start", "goal")
    @classmethod
    def validate_endpoint(cls, value: Endpoint) -> Endpoint:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("entity endpoint names cannot be empty")
            return normalized
        if len(value) != 2 or not all(math.isfinite(float(item)) for item in value):
            raise ValueError("coordinate endpoints must contain two finite values")
        return float(value[0]), float(value[1])

    def planner_config(self, base: SpatialPlannerConfig) -> SpatialPlannerConfig:
        values = asdict(base)
        for name in (
            "diagonal",
            "allow_unknown",
            "unknown_cost",
            "clearance_weight",
            "diversity_weight",
            "diversity_radius",
            "max_detour_ratio",
        ):
            override = getattr(self, name)
            if override is not None:
                values[name] = override
        values["route_count"] = self.route_count
        return SpatialPlannerConfig(**values)


class WorldSimulationRequest(BaseModel):
    plan_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    route_id: Optional[str] = Field(default=None, pattern=r"^route-[1-9][0-9]*$")
    mode: Literal["auto", "geometry", "world_model"] = "auto"
    max_world_model_steps: int = Field(default=64, ge=1, le=512)


class WorldFeedbackRequest(BaseModel):
    plan_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    outcome: Literal["success", "collision", "blocked", "cancelled", "unknown"]
    route_id: Optional[str] = Field(default=None, pattern=r"^route-[1-9][0-9]*$")
    actual_path: List[Coordinate] = Field(default_factory=list, max_length=10000)
    coordinate_space: Literal["grid", "source"] = "grid"
    simulation_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    note: str = Field(default="", max_length=4000)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("actual_path")
    @classmethod
    def validate_actual_path(cls, value: List[Coordinate]) -> List[Coordinate]:
        for point in value:
            if len(point) != 2 or not all(math.isfinite(float(item)) for item in point):
                raise ValueError("actual_path points must contain two finite values")
        return [(float(point[0]), float(point[1])) for point in value]


async def _world_agent_error_handler(
    _request: Any,
    error: WorldAgentError,
) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": str(error)}},
    )


def create_world_agent_router(
    runtime: WorldAgentRuntime,
    prefix: str = "/v1/world",
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["WorldAgent"])

    @router.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "saddle-world-agent",
            "protocol_version": 1,
        }

    @router.get("/capabilities")
    async def capabilities() -> Dict[str, Any]:
        return runtime.capabilities()

    @router.get("/memory")
    async def memory() -> Dict[str, Any]:
        return runtime.memory_summary()

    @router.post("/analyze")
    async def analyze(
        image: UploadFile = File(...),
        request: str = Form("{}"),
    ) -> Dict[str, Any]:
        try:
            parsed = WorldAnalyzeRequest.model_validate_json(request or "{}")
        except Exception as error:
            raise WorldAgentError(f"Invalid analyze request: {error}", 422, "validation_error") from error
        maximum = runtime.settings.max_upload_mb * 1024 * 1024
        payload = await image.read(maximum + 1)
        state = await run_in_threadpool(
            runtime.analyze_bytes,
            payload,
            parsed.instruction,
            parsed.semantic_backend,
            parsed.map_config(runtime.settings.map_config),
            image.filename or "upload",
        )
        return {"status": "completed", "state": state.to_dict(include_grid=True)}

    @router.post("/plan")
    async def plan(request: WorldPlanRequest) -> Dict[str, Any]:
        result = await run_in_threadpool(
            runtime.plan,
            request.state_id,
            request.start,
            request.goal,
            request.instruction,
            request.route_count,
            request.use_world_model,
            request.planner_config(runtime.settings.planner_config),
            request.allow_perspective,
            request.language,
        )
        return {"status": "completed", "plan": result.to_dict()}

    @router.post("/simulate")
    async def simulate(request: WorldSimulationRequest) -> Dict[str, Any]:
        result = await run_in_threadpool(
            runtime.simulate,
            request.plan_id,
            request.route_id,
            request.mode,
            request.max_world_model_steps,
        )
        return {"status": "completed", "simulation": result.to_dict()}

    @router.post("/feedback")
    async def feedback(request: WorldFeedbackRequest) -> Dict[str, Any]:
        result = await run_in_threadpool(
            runtime.record_feedback,
            request.plan_id,
            request.outcome,
            request.route_id,
            request.actual_path,
            request.coordinate_space,
            request.simulation_id,
            request.note,
            request.metrics,
        )
        return {
            "status": "recorded",
            "feedback": result.to_dict(),
            "memory": runtime.memory_summary(),
        }

    @router.get("/states/{state_id}")
    async def get_state(state_id: str) -> Dict[str, Any]:
        return runtime.get_state(state_id).to_dict(include_grid=True)

    @router.get("/plans/{plan_id}")
    async def get_plan(plan_id: str) -> Dict[str, Any]:
        return runtime.get_plan(plan_id).to_dict()

    @router.get("/simulations/{simulation_id}")
    async def get_simulation(simulation_id: str) -> Dict[str, Any]:
        return runtime.get_simulation(simulation_id).to_dict()

    @router.get("/feedback/{feedback_id}")
    async def get_feedback(feedback_id: str) -> Dict[str, Any]:
        return runtime.get_feedback(feedback_id).to_dict()

    @router.get("/records/{collection}")
    async def list_records(collection: str, limit: int = 50) -> Dict[str, Any]:
        if collection not in runtime.store.COLLECTIONS:
            raise WorldAgentError(f"Unknown collection: {collection}", 404, "not_found")
        return {
            "collection": collection,
            "records": runtime.store.list(collection, limit=limit),
        }

    return router


def install_world_agent_routes(
    app: FastAPI,
    runtime: WorldAgentRuntime,
    prefix: str = "/v1/world",
) -> None:
    app.state.world_agent_runtime = runtime
    app.add_exception_handler(WorldAgentError, _world_agent_error_handler)
    app.include_router(create_world_agent_router(runtime, prefix=prefix))


def create_world_agent_app(
    settings: Optional[WorldAgentSettings] = None,
    runtime: Optional[WorldAgentRuntime] = None,
) -> FastAPI:
    settings = settings or WorldAgentSettings()
    runtime = runtime or WorldAgentRuntime(settings)
    app = FastAPI(
        title="SaddleLLM WorldAgent API",
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    install_world_agent_routes(app, runtime)

    @app.get("/")
    async def root() -> Dict[str, Any]:
        return {
            "service": "SaddleLLM WorldAgent API",
            "health": "/v1/world/health",
            "docs": "/docs",
        }

    return app


__all__ = [
    "WorldAnalyzeRequest",
    "WorldPlanRequest",
    "WorldSimulationRequest",
    "WorldFeedbackRequest",
    "create_world_agent_router",
    "install_world_agent_routes",
    "create_world_agent_app",
]
