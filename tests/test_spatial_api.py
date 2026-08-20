import json
from pathlib import Path

from fastapi.testclient import TestClient

from saddlellm import WorldModel, WorldModelConfig, WorldModelRuntime
from saddlellm.SpatialAPI import (
    SpatialStudioService,
    SpatialStudioSettings,
    create_spatial_studio_app,
)


def _settings(tmp_path, **overrides):
    values = {
        "workspace": str(tmp_path / "jobs"),
        "frontend_dist": str(tmp_path / "missing-frontend"),
        "demo_image": "data/spatial_floorplan_example.pbm",
        "max_upload_mb": 1,
    }
    values.update(overrides)
    return SpatialStudioSettings(**values)


def _client(tmp_path, service=None, **overrides):
    settings = _settings(tmp_path, **overrides)
    app = create_spatial_studio_app(settings, service=service)
    return TestClient(app)


def _request(**overrides):
    values = {
        "start": [2, 2],
        "goal": [21, 13],
        "route_count": 3,
        "obstacle_dilation": 0,
        "max_dimension": 256,
        "resolution": 0.5,
    }
    values.update(overrides)
    return values


def test_health_capabilities_and_frontend_fallback(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/health").json()["status"] == "ok"
    capabilities = client.get("/api/capabilities").json()
    assert capabilities["world_agent"]["base_url"] == "/v1/world"
    assert capabilities["semantic"]["qwen_vl"]["configured"] is False
    assert capabilities["world_model"]["observation_schema"]["minimum_size"] == 16
    assert client.get("/v1/world/health").json()["service"] == "saddle-world-agent"
    assert client.get("/").json()["status"] == "frontend-not-built"


def test_built_spatial_studio_is_served_by_fastapi(tmp_path):
    frontend = Path(__file__).resolve().parents[1] / "saddlellm" / "spatial_studio_web"
    assert (frontend / "index.html").is_file()
    client = _client(tmp_path, frontend_dist=str(frontend))
    response = client.get("/")
    assert response.status_code == 200
    assert '<div id="root"></div>' in response.text
    assert "Saddle 空间世界模型" in response.text


def test_demo_returns_grid_routes_and_downloadable_artifacts(tmp_path):
    client = _client(tmp_path)
    response = client.get("/api/demo")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert len(payload["result"]["routes"]) == 3
    assert len(payload["result"]["map"]["cells"]) == 16

    image = client.get(payload["image"]["url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")
    plan = client.get(payload["artifacts"]["json"]).json()
    assert len(plan["routes"]) == 3
    html = client.get(payload["artifacts"]["html"])
    assert "Route comparison" in html.text
    assert "content-disposition" not in html.headers
    assert client.get(payload["artifacts"]["png"]).content.startswith(b"\x89PNG")
    assert client.get(payload["artifacts"]["mask"]).content.startswith(b"\x89PNG")

    job_id = payload["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["job_id"] == job_id
    assert client.delete(f"/api/jobs/{job_id}").json()["status"] == "deleted"
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_uploaded_image_can_be_planned_and_validation_errors_are_structured(tmp_path):
    client = _client(tmp_path)
    image = Path("data/spatial_floorplan_example.pbm").read_bytes()
    response = client.post(
        "/api/plan",
        files={"image": ("floorplan.pbm", image, "image/x-portable-bitmap")},
        data={"request": json.dumps(_request())},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["source_name"] == "floorplan.pbm"
    assert payload["models"]["route_ranking"] == "geometry"
    assert len(payload["result"]["routes"]) == 3

    invalid = client.post(
        "/api/plan",
        files={"image": ("floorplan.pbm", image, "image/x-portable-bitmap")},
        data={"request": json.dumps(_request(start=[999, 999]))},
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_request"

    missing_model = client.post(
        "/api/plan",
        files={"image": ("floorplan.pbm", image, "image/x-portable-bitmap")},
        data={"request": json.dumps(_request(semantic_backend="qwen-vl"))},
    )
    assert missing_model.status_code == 409
    assert missing_model.json()["error"]["code"] == "model_unavailable"


def test_configured_world_model_rescores_api_routes(tmp_path):
    runtime = WorldModelRuntime(
        WorldModel(
            WorldModelConfig(
                observation_shape=(16,),
                action_dim=2,
                embedding_size=12,
                deterministic_size=12,
                stochastic_size=4,
                hidden_size=12,
                free_nats=0.0,
            )
        ),
        device="cpu",
    )
    settings = _settings(
        tmp_path,
        world_model_checkpoint="fake-spatial-checkpoint",
    )
    service = SpatialStudioService(settings, runtime_factory=lambda _path: runtime)
    client = _client(tmp_path, service=service, world_model_checkpoint="fake-spatial-checkpoint")
    image = Path("data/spatial_floorplan_example.pbm").read_bytes()
    response = client.post(
        "/api/plan",
        files={"image": ("floorplan.pbm", image, "image/x-portable-bitmap")},
        data={"request": json.dumps(_request(use_world_model=True))},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"]["route_ranking"] == "world-model"
    assert all(route["predicted_return"] is not None for route in payload["result"]["routes"])


def test_v2_world_model_exposes_safe_cem_forecast(tmp_path):
    runtime = WorldModelRuntime(
        WorldModel(
            WorldModelConfig(
                observation_shape=(11, 8, 8),
                action_dim=2,
                embedding_size=12,
                deterministic_size=16,
                stochastic_size=4,
                hidden_size=16,
                cnn_channels=(4, 8),
                free_nats=0.0,
                spatial_occupancy_channels=3,
                spatial_occupancy_weight=1.0,
                spatial_ego_motion_dim=3,
                spatial_ego_motion_weight=1.0,
                spatial_collision_weight=1.0,
            )
        ),
        device="cpu",
    )
    settings = _settings(tmp_path, world_model_checkpoint="fake-spatial-v2")
    service = SpatialStudioService(settings, runtime_factory=lambda _path: runtime)
    client = _client(
        tmp_path, service=service, world_model_checkpoint="fake-spatial-v2"
    )
    image = Path("data/spatial_floorplan_example.pbm").read_bytes()
    response = client.post(
        "/api/plan",
        files={"image": ("floorplan.pbm", image, "image/x-portable-bitmap")},
        data={"request": json.dumps(_request(use_world_model=True))},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"]["observation_schema"] == "spatial_sequence_v2"
    assert payload["models"]["planning"] == "closed-loop-cem"
    forecast = payload["result"]["world_model"]["forecast"]
    assert forecast["geometry_collisions"] == 0
    assert forecast["predicted_points"]
    assert forecast["future_occupancy"]["classes"]
