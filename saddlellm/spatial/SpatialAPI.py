"""FastAPI application for the end-to-end spatial world-model studio."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .SpatialPerception import MapExtractionConfig, QwenVLSpatialAnalyzer, TopDownMapExtractor
from .SpatialPlanner import GridPathPlanner, SpatialPlannerConfig
from .SpatialVisualization import (
    render_spatial_plan_html,
    render_spatial_plan_png,
    save_occupancy_mask_png,
    save_spatial_plan_json,
)
from .SpatialWorldModel import (
    SpatialObservationEncoder,
    SpatialTensorObservationEncoder,
    SpatialWorldModelCoordinator,
    WorldModelRouteScorer,
)
from .SpatialWorldModelControl import (
    SpatialCEMPlannerConfig,
    SpatialClosedLoopPlanner,
)


Coordinate = Tuple[float, float]
_JOB_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_ARTIFACTS = {
    "image": ("source.png", "image/png"),
    "json": ("plan.json", "application/json"),
    "html": ("plan.html", "text/html; charset=utf-8"),
    "png": ("plan.png", "image/png"),
    "mask": ("mask.png", "image/png"),
}


class SpatialPlanRequest(BaseModel):
    start: Coordinate
    goal: Coordinate
    route_count: int = Field(default=3, ge=1, le=6)
    instruction: str = Field(default="", max_length=2000)
    free_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    free_is_bright: bool = True
    uncertainty_band: float = Field(default=0.0, ge=0.0, le=0.5)
    obstacle_dilation: int = Field(default=1, ge=0, le=24)
    max_dimension: int = Field(default=768, ge=32, le=2048)
    resolution: float = Field(default=1.0, gt=0.0, le=100.0)
    diagonal: bool = True
    allow_unknown: bool = False
    clearance_weight: float = Field(default=0.25, ge=0.0, le=100.0)
    diversity_weight: float = Field(default=2.0, ge=0.0, le=100.0)
    diversity_radius: int = Field(default=2, ge=0, le=32)
    max_detour_ratio: float = Field(default=2.5, ge=1.0, le=10.0)
    semantic_backend: Literal["disabled", "qwen-vl"] = "disabled"
    use_world_model: bool = False
    allow_perspective: bool = False

    @field_validator("start", "goal")
    @classmethod
    def finite_coordinate(cls, value: Coordinate) -> Coordinate:
        if not all(float(item) == float(item) and abs(float(item)) != float("inf") for item in value):
            raise ValueError("coordinates must be finite")
        return float(value[0]), float(value[1])


@dataclass
class SpatialStudioSettings:
    workspace: str = "outputs/spatial_studio"
    frontend_dist: Optional[str] = None
    demo_image: str = "data/spatial_floorplan_example.pbm"
    qwen_model: Optional[str] = None
    world_model_checkpoint: Optional[str] = None
    device: str = "auto"
    max_upload_mb: int = 25
    max_image_pixels: int = 40_000_000

    @classmethod
    def from_env(cls) -> "SpatialStudioSettings":
        package_frontend = Path(__file__).resolve().parent / "spatial_studio_web"
        return cls(
            workspace=os.environ.get("SADDLE_SPATIAL_WORKSPACE", "outputs/spatial_studio"),
            frontend_dist=os.environ.get(
                "SADDLE_SPATIAL_FRONTEND", str(package_frontend)
            ),
            demo_image=os.environ.get(
                "SADDLE_SPATIAL_DEMO_IMAGE", "data/spatial_floorplan_example.pbm"
            ),
            qwen_model=os.environ.get("SADDLE_SPATIAL_QWEN_MODEL") or None,
            world_model_checkpoint=os.environ.get("SADDLE_SPATIAL_WORLD_MODEL") or None,
            device=os.environ.get("SADDLE_SPATIAL_DEVICE", "auto"),
            max_upload_mb=int(os.environ.get("SADDLE_SPATIAL_MAX_UPLOAD_MB", "25")),
            max_image_pixels=int(
                os.environ.get("SADDLE_SPATIAL_MAX_IMAGE_PIXELS", "40000000")
            ),
        )

    def __post_init__(self) -> None:
        if self.frontend_dist is None:
            self.frontend_dist = str(
                Path(__file__).resolve().parent / "spatial_studio_web"
            )
        if self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be positive")
        if self.max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")


class SpatialStudioError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)


class SpatialStudioService:
    """Thread-safe local service coordinating perception, planning, and RSSM."""

    def __init__(
        self,
        settings: Optional[SpatialStudioSettings] = None,
        analyzer_factory: Optional[Callable[[str], Any]] = None,
        runtime_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.settings = settings or SpatialStudioSettings.from_env()
        self.workspace = Path(self.settings.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._analyzer_factory = analyzer_factory
        self._runtime_factory = runtime_factory
        self._analyzer: Optional[Any] = None
        self._runtime: Optional[Any] = None
        self._analyzer_lock = threading.Lock()
        self._runtime_lock = threading.Lock()
        self._demo_lock = threading.Lock()
        self._demo_response: Optional[Dict[str, Any]] = None
        self.observation_encoder = SpatialObservationEncoder()
        self.tensor_observation_encoder = SpatialTensorObservationEncoder()

    def capabilities(self) -> Dict[str, Any]:
        return {
            "api_version": 1,
            "world_agent": {
                "enabled": True,
                "base_url": "/v1/world",
                "stages": ["analyze", "plan", "simulate", "feedback"],
            },
            "semantic": {
                "qwen_vl": {
                    "configured": bool(self.settings.qwen_model),
                    "loaded": self._analyzer is not None,
                    "model": _display_model_name(self.settings.qwen_model),
                }
            },
            "world_model": {
                "configured": bool(self.settings.world_model_checkpoint),
                "loaded": self._runtime is not None,
                "checkpoint": _display_model_name(self.settings.world_model_checkpoint),
                "observation_schema": self.observation_encoder.describe(),
                "observation_schemas": [
                    self.observation_encoder.describe(),
                    self.tensor_observation_encoder.describe(),
                ],
            },
            "limits": {
                "max_upload_mb": self.settings.max_upload_mb,
                "max_image_pixels": self.settings.max_image_pixels,
                "max_routes": 6,
            },
        }

    def plan_bytes(
        self,
        payload: bytes,
        request: SpatialPlanRequest,
        original_filename: str = "upload",
    ) -> Dict[str, Any]:
        if len(payload) > self.settings.max_upload_mb * 1024 * 1024:
            raise SpatialStudioError(
                f"Image exceeds {self.settings.max_upload_mb} MB upload limit",
                status_code=413,
                code="upload_too_large",
            )
        job_id = uuid.uuid4().hex
        final_directory = self.workspace / job_id
        final_directory.mkdir(parents=False, exist_ok=False)
        temporary = final_directory
        try:
            source_path = temporary / "source.png"
            image_size = self._decode_image(payload, source_path)
            self._validate_coordinates(request, image_size)
            analyzer = self._get_analyzer(request.semantic_backend)
            extractor = TopDownMapExtractor(
                MapExtractionConfig(
                    free_threshold=request.free_threshold,
                    free_is_bright=request.free_is_bright,
                    uncertainty_band=request.uncertainty_band,
                    obstacle_dilation=request.obstacle_dilation,
                    max_dimension=request.max_dimension,
                    resolution=request.resolution,
                )
            )
            planner = GridPathPlanner(
                SpatialPlannerConfig(
                    diagonal=request.diagonal,
                    allow_unknown=request.allow_unknown,
                    clearance_weight=request.clearance_weight,
                    route_count=request.route_count,
                    diversity_weight=request.diversity_weight,
                    diversity_radius=request.diversity_radius,
                    max_detour_ratio=request.max_detour_ratio,
                )
            )
            coordinator = SpatialWorldModelCoordinator(
                extractor=extractor,
                planner=planner,
                analyzer=analyzer,
                allow_perspective=request.allow_perspective,
            )
            result = coordinator.plan_image(
                str(source_path),
                start=request.start,
                goal=request.goal,
                instruction=request.instruction,
                route_count=request.route_count,
            )
            model_info: Dict[str, Any] = {
                "semantic_backend": (
                    "qwen-vl" if analyzer is not None else "geometry-only"
                ),
                "world_model": "disabled",
                "route_ranking": "geometry",
            }
            runtime = self._get_runtime(request.use_world_model)
            if runtime is not None:
                model_shape = tuple(runtime.model.config.observation_shape)
                if len(model_shape) == 1:
                    observation = self.observation_encoder.encode(
                        result.grid,
                        result.start,
                        result.goal,
                        model_shape,
                        clearance=planner.clearance_map(result.grid),
                    )
                    observation_schema = self.observation_encoder.schema
                    local_plan = None
                elif len(model_shape) == 3:
                    tensor_encoder = SpatialTensorObservationEncoder(
                        crop_size=model_shape[-1],
                        sensor_radius=min(6, max(1, model_shape[-1] // 2 - 1)),
                    )
                    observation = tensor_encoder.encode(
                        result.grid,
                        result.start,
                        result.goal,
                        observation_shape=model_shape,
                    )
                    observation_schema = tensor_encoder.schema
                    local_plan = SpatialClosedLoopPlanner(
                        runtime,
                        encoder=tensor_encoder,
                        config=SpatialCEMPlannerConfig(
                            horizon=12,
                            num_candidates=64,
                            iterations=3,
                            diagonal=request.diagonal,
                            allow_unknown=request.allow_unknown,
                            seed=0,
                        ),
                    )
                else:
                    raise SpatialStudioError(
                        f"Unsupported spatial observation shape: {model_shape}",
                        code="world_model_schema_mismatch",
                    )
                state = runtime.encode_observation(observation)
                WorldModelRouteScorer(runtime).score(state, result.routes)
                if local_plan is not None:
                    result.world_model = {
                        "schema": observation_schema,
                        "planner": "closed-loop-cem",
                        "forecast": local_plan.plan(
                            state, result.grid, result.start, result.goal
                        ).to_dict(include_occupancy=True),
                    }
                model_info.update(
                    {
                        "world_model": runtime.backend,
                        "route_ranking": "world-model",
                        "observation_schema": observation_schema,
                        "planning": (
                            "closed-loop-cem" if local_plan is not None else "route-rescoring"
                        ),
                    }
                )

            render_spatial_plan_html(result, str(temporary / "plan.html"))
            render_spatial_plan_png(result, str(temporary / "plan.png"))
            save_occupancy_mask_png(result.grid, str(temporary / "mask.png"))
            result.image_path = str(final_directory / "source.png")
            save_spatial_plan_json(
                result,
                str(temporary / "plan.json"),
                include_grid=True,
            )
            response = {
                "status": "completed",
                "job_id": job_id,
                "source_name": Path(original_filename or "upload").name,
                "image": {
                    "width": image_size[0],
                    "height": image_size[1],
                    "url": f"/api/jobs/{job_id}/image",
                },
                "artifacts": {
                    "json": f"/api/jobs/{job_id}/json",
                    "html": f"/api/jobs/{job_id}/html",
                    "png": f"/api/jobs/{job_id}/png",
                    "mask": f"/api/jobs/{job_id}/mask",
                },
                "models": model_info,
                "result": result.to_dict(include_grid=True),
            }
            (temporary / "response.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return response
        except SpatialStudioError:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except (ValueError, FileNotFoundError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            raise SpatialStudioError(str(error)) from error
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def demo(self) -> Dict[str, Any]:
        with self._demo_lock:
            if self._demo_response is not None:
                return self._demo_response
            demo_path = Path(self.settings.demo_image)
            if not demo_path.is_absolute():
                demo_path = Path.cwd() / demo_path
            if not demo_path.is_file():
                raise SpatialStudioError(
                    f"Demo image not found: {demo_path}",
                    status_code=500,
                    code="demo_unavailable",
                )
            request = SpatialPlanRequest(
                start=(2.0, 2.0),
                goal=(21.0, 13.0),
                route_count=3,
                instruction="避开墙体并给出三条明显不同的路线",
                obstacle_dilation=0,
                max_dimension=256,
                resolution=0.5,
                clearance_weight=0.0,
            )
            self._demo_response = self.plan_bytes(
                demo_path.read_bytes(), request, original_filename=demo_path.name
            )
            return self._demo_response

    def job_response(self, job_id: str) -> Dict[str, Any]:
        path = self._job_directory(job_id) / "response.json"
        if not path.is_file():
            raise SpatialStudioError("Spatial job not found", status_code=404, code="not_found")
        return json.loads(path.read_text(encoding="utf-8"))

    def artifact_path(self, job_id: str, artifact: str) -> Tuple[Path, str]:
        if artifact not in _ARTIFACTS:
            raise SpatialStudioError("Unknown artifact", status_code=404, code="not_found")
        filename, media_type = _ARTIFACTS[artifact]
        path = self._job_directory(job_id) / filename
        if not path.is_file():
            raise SpatialStudioError("Spatial artifact not found", status_code=404, code="not_found")
        return path, media_type

    def delete_job(self, job_id: str) -> None:
        directory = self._job_directory(job_id)
        if not directory.is_dir():
            raise SpatialStudioError("Spatial job not found", status_code=404, code="not_found")
        shutil.rmtree(directory)
        if self._demo_response and self._demo_response.get("job_id") == job_id:
            self._demo_response = None

    def _decode_image(self, payload: bytes, destination: Path) -> Tuple[int, int]:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as error:
            raise SpatialStudioError("Pillow is required for spatial uploads", 500) from error
        try:
            with Image.open(io.BytesIO(payload)) as source:
                width, height = source.size
                if width <= 0 or height <= 0:
                    raise SpatialStudioError("Image dimensions must be positive")
                if width * height > self.settings.max_image_pixels:
                    raise SpatialStudioError(
                        "Image pixel count exceeds the configured limit",
                        status_code=413,
                        code="image_too_large",
                    )
                source.seek(0)
                source.convert("RGB").save(destination, format="PNG", optimize=True)
        except (UnidentifiedImageError, OSError) as error:
            raise SpatialStudioError("Upload is not a supported image") from error
        return int(width), int(height)

    @staticmethod
    def _validate_coordinates(
        request: SpatialPlanRequest, image_size: Tuple[int, int]
    ) -> None:
        width, height = image_size
        for name, point in (("start", request.start), ("goal", request.goal)):
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise SpatialStudioError(
                    f"{name} coordinate must be inside the source image"
                )

    def _get_analyzer(self, backend: str) -> Optional[Any]:
        if backend == "disabled":
            return None
        if not self.settings.qwen_model:
            raise SpatialStudioError(
                "Qwen-VL is not configured on this server",
                status_code=409,
                code="model_unavailable",
            )
        with self._analyzer_lock:
            if self._analyzer is None:
                factory = self._analyzer_factory or (
                    lambda model: QwenVLSpatialAnalyzer.from_pretrained(model)
                )
                self._analyzer = factory(self.settings.qwen_model)
        return self._analyzer

    def _get_runtime(self, requested: bool) -> Optional[Any]:
        if not requested:
            return None
        if not self.settings.world_model_checkpoint:
            raise SpatialStudioError(
                "A spatial RSSM checkpoint is not configured on this server",
                status_code=409,
                code="model_unavailable",
            )
        with self._runtime_lock:
            if self._runtime is None:
                if self._runtime_factory is not None:
                    self._runtime = self._runtime_factory(
                        self.settings.world_model_checkpoint
                    )
                else:
                    from ..world_models.WorldModelInference import WorldModelRuntime

                    self._runtime = WorldModelRuntime.from_pretrained(
                        self.settings.world_model_checkpoint,
                        device=self.settings.device,
                    )
        return self._runtime

    def _job_directory(self, job_id: str) -> Path:
        if not _JOB_PATTERN.fullmatch(str(job_id)):
            raise SpatialStudioError("Invalid spatial job id", status_code=404, code="not_found")
        path = (self.workspace / job_id).resolve()
        if path.parent != self.workspace:
            raise SpatialStudioError("Invalid spatial job path", status_code=404, code="not_found")
        return path


def create_spatial_studio_app(
    settings: Optional[SpatialStudioSettings] = None,
    service: Optional[SpatialStudioService] = None,
) -> FastAPI:
    settings = settings or SpatialStudioSettings.from_env()
    service = service or SpatialStudioService(settings)
    app = FastAPI(
        title="Saddle Spatial World Model Studio",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.spatial_service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # The studio and the headless agent share lazy model instances.  Existing
    # /api routes remain stable while /v1/world exposes the decoupled
    # analyze -> plan -> simulate -> feedback protocol.
    from .WorldAgent import WorldAgentRuntime, WorldAgentSettings
    from .WorldAgentAPI import install_world_agent_routes

    agent_settings = WorldAgentSettings(
        workspace=str(service.workspace / "world_agent"),
        qwen_model=service.settings.qwen_model,
        world_model_checkpoint=service.settings.world_model_checkpoint,
        device=service.settings.device,
        max_upload_mb=service.settings.max_upload_mb,
        max_image_pixels=service.settings.max_image_pixels,
    )
    world_agent = WorldAgentRuntime(
        agent_settings,
        analyzer_loader=(
            (lambda: service._get_analyzer("qwen-vl"))
            if service.settings.qwen_model
            else None
        ),
        world_model_loader=(
            (lambda: service._get_runtime(True))
            if service.settings.world_model_checkpoint
            else None
        ),
    )
    install_world_agent_routes(app, world_agent)

    @app.exception_handler(SpatialStudioError)
    async def handle_spatial_error(_request: Any, error: SpatialStudioError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": str(error)}},
        )

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "service": "saddle-spatial-studio", "api_version": 1}

    @app.get("/api/capabilities")
    async def capabilities() -> Dict[str, Any]:
        return service.capabilities()

    @app.get("/api/demo")
    async def demo() -> Dict[str, Any]:
        return await run_in_threadpool(service.demo)

    @app.post("/api/plan")
    async def plan(
        image: UploadFile = File(...),
        request: str = Form(...),
    ) -> Dict[str, Any]:
        try:
            parsed = SpatialPlanRequest.model_validate_json(request)
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"Invalid planning request: {error}") from error
        maximum = settings.max_upload_mb * 1024 * 1024
        payload = await image.read(maximum + 1)
        return await run_in_threadpool(
            service.plan_bytes,
            payload,
            parsed,
            image.filename or "upload",
        )

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str) -> Dict[str, Any]:
        return service.job_response(job_id)

    @app.get("/api/jobs/{job_id}/{artifact}")
    async def artifact(job_id: str, artifact: str) -> FileResponse:
        path, media_type = service.artifact_path(job_id, artifact)
        download_name = None
        # Keep the standalone visualizer inline; JSON/PNG remain convenient downloads.
        if artifact in {"json", "png"}:
            download_name = path.name
        return FileResponse(path, media_type=media_type, filename=download_name)

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> Dict[str, str]:
        service.delete_job(job_id)
        return {"status": "deleted", "job_id": job_id}

    frontend = Path(settings.frontend_dist or "")
    if (frontend / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="studio")
    else:
        @app.get("/")
        async def root() -> Dict[str, Any]:
            return {
                "service": "Saddle Spatial World Model Studio",
                "status": "frontend-not-built",
                "api_docs": "/api/docs",
            }

    return app


def _display_model_name(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    normalized = str(path).rstrip("/\\")
    return os.path.basename(normalized) or normalized


app = create_spatial_studio_app()


__all__ = [
    "SpatialPlanRequest",
    "SpatialStudioSettings",
    "SpatialStudioError",
    "SpatialStudioService",
    "create_spatial_studio_app",
    "app",
]
