import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from saddlellm import WorldModel, WorldModelConfig, WorldModelRuntime
from saddlellm.spatial.SpatialPerception import SpatialAnalysis
from saddlellm.spatial.WorldAgent import (
    WORLD_PLAN_SCHEMA,
    WORLD_STATE_SCHEMA,
    WorldAgentError,
    WorldAgentRuntime,
    WorldAgentSettings,
)
from saddlellm.spatial.WorldAgentAPI import create_world_agent_app
from saddlellm.world_models.WorldModelData import load_world_model_trajectories
from saddlellm.cli import main as cli_main


IMAGE = Path("data/spatial_floorplan_example.pbm")


def _runtime(tmp_path, **kwargs):
    settings = WorldAgentSettings(workspace=str(tmp_path / "world-agent"))
    return WorldAgentRuntime(settings, **kwargs)


def test_world_state_is_rle_persisted_and_restored(tmp_path):
    runtime = _runtime(tmp_path)
    state = runtime.analyze_image(IMAGE)
    stored = json.loads(
        (tmp_path / "world-agent" / "states" / f"{state.state_id}.json").read_text(
            encoding="utf-8"
        )
    )

    assert stored["schema"] == WORLD_STATE_SCHEMA
    assert stored["map"]["cell_encoding"] == "rle-v1"
    assert "cells" not in stored["map"]
    restored = runtime.get_state(state.state_id)
    assert np.array_equal(restored.grid.cells, state.grid.cells)
    assert restored.observation.sha256 == state.observation.sha256
    assert restored.confidence["calibrated"] is False


def test_geometry_plan_simulation_feedback_and_training_replay(tmp_path):
    runtime = _runtime(tmp_path)
    state = runtime.analyze_image(IMAGE)
    plan = runtime.plan(state.state_id, [2, 2], [21, 13], route_count=3)
    simulation = runtime.simulate(plan.plan_id, mode="geometry")
    feedback = runtime.record_feedback(
        plan.plan_id,
        "success",
        simulation_id=simulation.simulation_id,
        actual_path=plan.routes[0]["points"],
    )

    assert plan.schema == WORLD_PLAN_SCHEMA
    assert len(plan.routes) == 3
    assert plan.ranking["mode"] == "geometry"
    assert plan.explanation["generated_by"] == "rules-v1"
    assert simulation.geometry["route_valid"] is True
    assert simulation.geometry["reached_goal"] is True
    assert simulation.geometry["success_probability"] is None
    assert feedback.comparison["reached_goal"] is True
    assert feedback.replay["recorded"] is True

    trajectories = load_world_model_trajectories(runtime.store.replay_path)
    assert len(trajectories) == 1
    assert len(trajectories[0]["observations"]) == len(trajectories[0]["actions"]) + 1
    assert trajectories[0]["metadata"]["source"] == "world-agent-feedback"
    assert runtime.memory_summary()["replay_records"] == 1


def test_world_model_can_rank_and_imagine_selected_route(tmp_path):
    model_runtime = WorldModelRuntime(
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
    runtime = _runtime(tmp_path, world_model_loader=lambda: model_runtime)
    state = runtime.analyze_image(IMAGE)
    plan = runtime.plan(
        state.state_id,
        [2, 2],
        [21, 13],
        route_count=2,
        use_world_model=True,
    )
    simulation = runtime.simulate(plan.plan_id, mode="world_model")

    assert plan.ranking["mode"] == "world-model"
    assert all(route["predicted_return"] is not None for route in plan.routes)
    assert simulation.mode == "world-model"
    assert simulation.learned_prediction["prediction"] == "open-loop-latent-imagination"
    assert simulation.learned_prediction["predicted_rewards"]
    assert simulation.confidence["calibrated"] is False


def test_perspective_state_can_be_analyzed_but_not_silently_planned(tmp_path):
    class Analyzer:
        def analyze(self, _path, _instruction):
            return SpatialAnalysis(
                image_type="perspective",
                summary="single camera view",
                confidence=0.8,
                caveats=["occluded space is unknown"],
            )

    settings = WorldAgentSettings(
        workspace=str(tmp_path / "world-agent"),
        qwen_model="fake-qwen-vl",
    )
    runtime = WorldAgentRuntime(settings, analyzer_loader=Analyzer)
    state = runtime.analyze_image(IMAGE, semantic_backend="qwen-vl")

    assert state.analysis.image_type == "perspective"
    with pytest.raises(WorldAgentError, match="single perspective image"):
        runtime.plan(state.state_id, [2, 2], [21, 13])

    approximate = runtime.plan(
        state.state_id,
        [2, 2],
        [21, 13],
        allow_perspective=True,
    )
    assert approximate.routes
    assert any("perspective" in item.lower() for item in approximate.limitations)


def test_world_agent_http_protocol_runs_all_four_stages(tmp_path):
    runtime = _runtime(tmp_path)
    client = TestClient(create_world_agent_app(runtime=runtime))
    image = IMAGE.read_bytes()

    analyzed = client.post(
        "/v1/world/analyze",
        files={"image": ("floorplan.pbm", image, "image/x-portable-bitmap")},
        data={"request": json.dumps({"obstacle_dilation": 0})},
    )
    assert analyzed.status_code == 200
    state_id = analyzed.json()["state"]["state_id"]

    planned = client.post(
        "/v1/world/plan",
        json={
            "state_id": state_id,
            "start": [2, 2],
            "goal": [21, 13],
            "route_count": 3,
        },
    )
    assert planned.status_code == 200
    plan = planned.json()["plan"]

    simulated = client.post(
        "/v1/world/simulate",
        json={"plan_id": plan["plan_id"], "mode": "geometry"},
    )
    assert simulated.status_code == 200
    simulation_id = simulated.json()["simulation"]["simulation_id"]

    feedback = client.post(
        "/v1/world/feedback",
        json={
            "plan_id": plan["plan_id"],
            "simulation_id": simulation_id,
            "outcome": "success",
            "actual_path": plan["routes"][0]["points"],
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["feedback"]["replay"]["recorded"] is True
    assert client.get("/v1/world/memory").json()["counts"] == {
        "states": 1,
        "plans": 1,
        "simulations": 1,
        "feedback": 1,
    }
    assert client.get(f"/v1/world/states/{state_id}").status_code == 200
    assert client.get("/v1/world/states/not-an-id").status_code == 400


def test_feedback_without_observed_path_is_stored_but_not_used_for_training(tmp_path):
    runtime = _runtime(tmp_path)
    state = runtime.analyze_image(IMAGE)
    plan = runtime.plan(state.state_id, [2, 2], [21, 13])
    feedback = runtime.record_feedback(plan.plan_id, "cancelled", note="operator stopped")

    assert feedback.replay["recorded"] is False
    assert runtime.memory_summary()["replay_records"] == 0


def test_world_agent_config_and_end_to_end_cli(tmp_path):
    settings = WorldAgentSettings.from_file("configs/world_agent.yaml")
    assert Path(settings.workspace).is_absolute()
    assert settings.planner_config.route_count == 3

    output = tmp_path / "result.json"
    exit_code = cli_main(
        [
            "world-agent-run",
            str(IMAGE),
            "--start",
            "2,2",
            "--goal",
            "21,13",
            "--workspace",
            str(tmp_path / "cli-memory"),
            "--simulation",
            "geometry",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["plan"]["selected_route_id"] == "route-1"
    assert result["simulation"]["geometry"]["reached_goal"] is True
