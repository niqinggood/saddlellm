import json

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from saddlellm import (
    CallableSpatialAnalyzer,
    GridPathPlanner,
    MapExtractionConfig,
    OccupancyGrid,
    SpatialAnalysis,
    SpatialObservationEncoder,
    SpatialTensorObservationEncoder,
    SpatialSequenceConfig,
    SpatialSequenceTrajectoryBuilder,
    SpatialCEMPlannerConfig,
    SpatialClosedLoopPlanner,
    SpatialTrajectoryBuilder,
    SpatialTrajectoryConfig,
    SpatialPlannerConfig,
    SpatialWorldModelCoordinator,
    TopDownMapExtractor,
    WorldModel,
    WorldModelConfig,
    WorldModelRouteScorer,
    WorldModelRuntime,
    generate_spatial_world_model_dataset,
    generate_spatial_sequence_dataset,
    evaluate_spatial_world_model,
    infer_world_model_dimensions,
    load_world_model_trajectories,
    split_world_model_trajectories,
    render_spatial_plan_html,
    render_spatial_plan_png,
    save_spatial_plan_json,
)
from saddlellm.cli import main as cli_main


def _grid():
    cells = np.zeros((14, 18), dtype=np.uint8)
    cells[0, :] = OccupancyGrid.BLOCKED
    cells[-1, :] = OccupancyGrid.BLOCKED
    cells[:, 0] = OccupancyGrid.BLOCKED
    cells[:, -1] = OccupancyGrid.BLOCKED
    cells[1:-1, 8] = OccupancyGrid.BLOCKED
    cells[3:6, 8] = OccupancyGrid.FREE
    cells[9:12, 8] = OccupancyGrid.FREE
    return OccupancyGrid(cells, resolution=0.25, source_size=(180, 140))


def _floorplan(path):
    image = Image.new("RGB", (180, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 179, 139), outline="black", width=5)
    draw.rectangle((84, 5, 95, 45), fill="black")
    draw.rectangle((84, 70, 95, 105), fill="black")
    draw.rectangle((84, 128, 95, 139), fill="black")
    draw.rectangle((95, 65, 135, 74), fill="black")
    draw.rectangle((155, 65, 179, 74), fill="black")
    image.save(path)


def _analysis(image_type="floor_plan"):
    return {
        "image_type": image_type,
        "summary": "Two connected rooms with visible doors and walls.",
        "entities": [
            {
                "name": "entrance",
                "category": "door",
                "bbox": [60, 80, 140, 180],
                "confidence": 0.9,
                "traversable": True,
            },
            {
                "name": "office",
                "category": "room",
                "bbox": [760, 720, 920, 900],
                "confidence": 0.85,
                "traversable": True,
            },
        ],
        "connectivity": [{"from": "entrance", "to": "office", "via": "door"}],
        "confidence": 0.82,
        "caveats": ["Synthetic test map."],
    }


def test_astar_and_diverse_top_k_routes_are_traversable_and_distinct():
    grid = _grid()
    planner = GridPathPlanner(
        SpatialPlannerConfig(route_count=4, diagonal=True, clearance_weight=0.1)
    )
    routes = planner.plan(grid, (2, 7), (15, 7))

    assert len(routes) == 4
    assert len({tuple(route.points) for route in routes}) == 4
    first_cells = set(routes[0].points[1:-1])
    second_cells = set(routes[1].points[1:-1])
    overlap = len(first_cells & second_cells) / min(len(first_cells), len(second_cells))
    assert overlap < 0.8
    for route in routes:
        assert route.points[0] == (2, 7)
        assert route.points[-1] == (15, 7)
        assert all(grid.traversable(point) for point in route.points)
        assert route.length > 0
        assert route.minimum_clearance > 0
        assert 0 <= route.risk <= 1


def test_image_extraction_semantic_coordination_and_visualization(tmp_path):
    image_path = tmp_path / "floorplan.png"
    _floorplan(image_path)
    analyzer = CallableSpatialAnalyzer(lambda _path, _instruction: _analysis())
    coordinator = SpatialWorldModelCoordinator(
        extractor=TopDownMapExtractor(
            MapExtractionConfig(
                free_threshold=0.7,
                obstacle_dilation=1,
                max_dimension=180,
                resolution=0.1,
            )
        ),
        planner=GridPathPlanner(SpatialPlannerConfig(route_count=3)),
        analyzer=analyzer,
    )
    result = coordinator.plan_image(
        str(image_path),
        start=[20, 20],
        goal=[165, 120],
        instruction="Avoid walls and provide alternatives.",
    )

    assert result.analysis.confidence == pytest.approx(0.82)
    assert len(result.routes) == 3
    assert result.grid.source_size == (180, 140)
    assert all(result.grid.traversable(point) for route in result.routes for point in route.points)

    json_path = tmp_path / "plan.json"
    html_path = tmp_path / "plan.html"
    png_path = tmp_path / "plan.png"
    save_spatial_plan_json(result, str(json_path))
    render_spatial_plan_html(result, str(html_path))
    render_spatial_plan_png(result, str(png_path), width=900)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    page = html_path.read_text(encoding="utf-8")
    assert len(payload["routes"]) == 3
    assert "data:image/png;base64," in page
    assert "Route comparison" in page
    assert "data-route-toggle" in page
    assert "https://" not in page
    assert "fetch(" not in page
    assert png_path.is_file()
    with Image.open(png_path) as preview:
        assert preview.size[0] == 900


def test_semantic_entity_points_and_perspective_safety(tmp_path):
    image_path = tmp_path / "floorplan.png"
    _floorplan(image_path)
    extractor = TopDownMapExtractor(
        MapExtractionConfig(obstacle_dilation=0, max_dimension=180)
    )
    named = SpatialWorldModelCoordinator(
        extractor=extractor,
        analyzer=CallableSpatialAnalyzer(lambda _path, _instruction: _analysis()),
    )
    result = named.plan_image(
        str(image_path), start="entrance", goal="office", route_count=1
    )
    assert result.routes

    unsafe = SpatialWorldModelCoordinator(
        extractor=extractor,
        analyzer=CallableSpatialAnalyzer(
            lambda _path, _instruction: _analysis("perspective")
        ),
    )
    with pytest.raises(ValueError, match="single perspective image"):
        unsafe.plan_image(str(image_path), [20, 20], [160, 120])


def test_world_model_can_rescore_geometric_routes():
    planner = GridPathPlanner(SpatialPlannerConfig(route_count=2))
    routes = planner.plan(_grid(), (2, 7), (15, 7))
    model = WorldModel(
        WorldModelConfig(
            observation_shape=(3,),
            action_dim=2,
            embedding_size=12,
            deterministic_size=12,
            stochastic_size=4,
            hidden_size=12,
            free_nats=0.0,
        )
    )
    runtime = WorldModelRuntime(model, device="cpu")
    state = runtime.encode_observation([0.0, 0.0, 0.0])
    scorer = WorldModelRouteScorer(runtime, max_steps=12)
    scorer.score(state, routes)

    assert all(route.predicted_return is not None for route in routes)
    assert all(np.isfinite(route.combined_score) for route in routes)
    assert routes[0].combined_score >= routes[1].combined_score


def test_spatial_observation_encoder_has_stable_schema():
    grid = _grid()
    encoded = SpatialObservationEncoder().encode(grid, (2, 7), (15, 7), (20,))
    assert encoded.shape == (20,)
    assert np.isfinite(encoded).all()
    assert encoded[16:].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_spatial_trajectory_builder_matches_native_world_model_contract():
    grid = _grid()
    route = GridPathPlanner(SpatialPlannerConfig(route_count=1)).plan(
        grid, (2, 7), (15, 7)
    )[0]
    episode = SpatialTrajectoryBuilder(
        SpatialTrajectoryConfig(observation_size=20, action_dim=3, max_steps=12)
    ).build(grid, route, "route-1")

    assert len(episode["observations"]) == len(episode["actions"]) + 1
    assert len(episode["observations"][0]) == 20
    assert len(episode["actions"][0]) == 3
    assert episode["dones"][-1] is True
    assert not any(episode["dones"][:-1])
    assert episode["metadata"]["schema"] == "spatial_features_v1"


def test_spatial_sequence_encoder_persists_partial_belief():
    grid = _grid()
    encoder = SpatialTensorObservationEncoder(crop_size=16, sensor_radius=3)
    belief = encoder.new_belief(grid)

    first = encoder.encode(grid, (2, 7), (15, 7), belief=belief, heading=0.0)
    known_after_first = int(np.count_nonzero(belief.cells != grid.UNKNOWN))
    second = encoder.encode(grid, (5, 7), (15, 7), belief=belief, heading=0.0)

    assert first.shape == (11, 16, 16)
    assert np.allclose(first[:3].sum(axis=0), 1.0)
    assert np.isfinite(first).all()
    assert int(np.count_nonzero(belief.cells != grid.UNKNOWN)) > known_after_first
    assert not np.array_equal(first, second)
    assert encoder.describe()["schema"] == "spatial_sequence_v2"


def test_spatial_occupancy_kinematic_prior_shifts_with_agent_motion():
    model = WorldModel(
        WorldModelConfig(
            observation_shape=(3, 8, 8),
            action_dim=2,
            embedding_size=12,
            deterministic_size=16,
            stochastic_size=4,
            hidden_size=16,
            cnn_channels=(4, 8),
            free_nats=0.0,
            spatial_occupancy_channels=3,
            spatial_occupancy_weight=1.0,
            spatial_occupancy_kinematic_prior=True,
        )
    )
    context = torch.zeros(1, 3, 8, 8)
    context[:, 2] = 1.0
    context[:, 2, 4, 6] = 0.0
    context[:, 1, 4, 6] = 1.0
    feature = torch.zeros(
        1, model.config.deterministic_size + model.config.stochastic_size
    )

    stationary = model._spatial_occupancy_logits(
        feature, context, torch.tensor([[0.0, 0.0]])
    ).argmax(dim=1)
    moved_right = model._spatial_occupancy_logits(
        feature, context, torch.tensor([[1.0, 0.0]])
    ).argmax(dim=1)

    assert stationary[0, 4, 6].item() == 1
    # The agent moves right, so a fixed obstacle moves left in the local crop.
    assert moved_right[0, 4, 5].item() == 1

    collision_context = context.clone()
    collision_context[:, 2, 4, 5] = 0.0
    collision_context[:, 1, 4, 5] = 1.0
    collision_attempt = model._spatial_occupancy_logits(
        feature, collision_context, torch.tensor([[1.0, 0.0]])
    ).argmax(dim=1)
    assert collision_attempt[0, 4, 5].item() == 1
    assert collision_attempt[0, 4, 4].item() != 1
    assert model._spatial_geometry_collision(
        collision_context, torch.tensor([[1.0, 0.0]])
    ).item()


def test_spatial_sequence_builder_emits_collision_and_motion_targets():
    grid = _grid()
    route = GridPathPlanner(SpatialPlannerConfig(route_count=1)).plan(
        grid, (1, 1), (7, 1)
    )[0]
    episode = SpatialSequenceTrajectoryBuilder(
        SpatialSequenceConfig(
            crop_size=8,
            sensor_radius=3,
            max_steps=16,
            collision_probability=1.0,
        )
    ).build(grid, route, "sequence-1", rng=np.random.default_rng(4))

    assert episode["metadata"]["schema"] == "spatial_sequence_v2"
    assert np.asarray(episode["observations"]).shape[1:] == (11, 8, 8)
    assert len(episode["observations"]) == len(episode["actions"]) + 1
    assert len(episode["ego_motions"]) == len(episode["actions"])
    assert sum(episode["collisions"]) >= 1
    assert episode["dones"][-1] is True


def test_generate_multimap_spatial_sequence_dataset_has_group_holdout(tmp_path):
    first_image = tmp_path / "map-a.png"
    second_image = tmp_path / "map-b.png"
    _floorplan(first_image)
    _floorplan(second_image)
    output_path = tmp_path / "spatial-v2.jsonl"
    summary = generate_spatial_sequence_dataset(
        [str(first_image), str(second_image)],
        str(output_path),
        episodes=8,
        routes_per_pair=1,
        seed=7,
        minimum_distance=0.1,
        map_config=MapExtractionConfig(
            obstacle_dilation=0,
            max_dimension=48,
            resolution=0.1,
        ),
        sequence_config=SpatialSequenceConfig(
            crop_size=8,
            sensor_radius=3,
            max_steps=12,
            collision_probability=0.5,
        ),
    )
    trajectories = load_world_model_trajectories(str(output_path))
    dimensions = infer_world_model_dimensions(trajectories)
    train, validation = split_world_model_trajectories(
        trajectories, 0.25, seed=5, group_key="map_id"
    )

    assert summary["schema"] == "spatial_sequence_v2"
    assert summary["maps"] == 2
    assert dimensions["observation_shape"] == (11, 8, 8)
    train_maps = {item["metadata"]["map_id"] for item in train}
    validation_maps = {item["metadata"]["map_id"] for item in validation}
    assert train_maps
    assert validation_maps
    assert train_maps.isdisjoint(validation_maps)


def test_spatial_auxiliary_heads_and_safe_cem_plan():
    grid = _grid()
    encoder = SpatialTensorObservationEncoder(crop_size=8, sensor_radius=3)
    model = WorldModel(
        WorldModelConfig(
            observation_shape=encoder.observation_shape,
            action_dim=2,
            embedding_size=16,
            deterministic_size=20,
            stochastic_size=6,
            hidden_size=20,
            cnn_channels=(8, 16),
            free_nats=0.0,
            spatial_occupancy_channels=3,
            spatial_occupancy_weight=1.0,
            spatial_ego_motion_dim=3,
            spatial_ego_motion_weight=1.0,
            spatial_collision_weight=1.0,
        )
    )
    observations = torch.rand(2, 4, *encoder.observation_shape)
    occupancy_ids = torch.randint(0, 3, (2, 4, 8, 8))
    observations[:, :, :3] = torch.nn.functional.one_hot(
        occupancy_ids, num_classes=3
    ).permute(0, 1, 4, 2, 3)
    batch = {
        "observations": observations,
        "actions": torch.rand(2, 3, 2) * 2 - 1,
        "rewards": torch.zeros(2, 3),
        "dones": torch.zeros(2, 3),
        "mask": torch.ones(2, 3),
        "ego_motions": torch.zeros(2, 3, 3),
        "collisions": torch.zeros(2, 3),
    }
    losses = model.compute_loss(batch, sample_state=False)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert {"occupancy_loss", "ego_motion_loss", "collision_loss"} <= set(losses)
    runtime = WorldModelRuntime(model, device="cpu")
    observation = encoder.encode(grid, (2, 7), (15, 7))
    state = runtime.encode_observation(observation)
    controller = SpatialClosedLoopPlanner(
        runtime,
        encoder=encoder,
        config=SpatialCEMPlannerConfig(
            horizon=5,
            num_candidates=8,
            iterations=1,
            goal_progress_weight=1000.0,
            terminal_distance_weight=100.0,
            learned_collision_weight=0.0,
            seed=3,
        ),
    )
    plan = controller.plan(state, grid, (2, 7), (15, 7))

    assert plan.geometry_collisions == 0
    assert all(grid.traversable(point) for point in plan.predicted_points)
    assert plan.occupancy_logits is not None
    assert plan.learned_collision_probability is not None
    assert plan.to_dict()["future_occupancy"]["classes"]
    closed_loop = controller.run_closed_loop(grid, (2, 7), (15, 7), max_steps=24)
    assert closed_loop.reached_goal is True
    assert closed_loop.collisions == 0
    assert closed_loop.points[-1] == (15, 7)


def test_spatial_rollout_evaluator_reports_prediction_and_calibration_metrics(tmp_path):
    grid = _grid()
    route = GridPathPlanner(SpatialPlannerConfig(route_count=1)).plan(
        grid, (1, 1), (7, 1)
    )[0]
    builder = SpatialSequenceTrajectoryBuilder(
        SpatialSequenceConfig(
            crop_size=8,
            sensor_radius=3,
            max_steps=12,
            collision_probability=1.0,
        )
    )
    episode = builder.build(
        grid, route, "eval-sequence", rng=np.random.default_rng(2)
    )
    data_path = tmp_path / "eval.jsonl"
    data_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    model = WorldModel(
        WorldModelConfig(
            observation_shape=builder.encoder.observation_shape,
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
    )
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint))

    metrics = evaluate_spatial_world_model(
        str(checkpoint),
        str(data_path),
        device="cpu",
        sequence_length=20,
        max_windows=1,
    )

    expected_steps = min(len(episode["actions"]), 20)
    assert metrics["valid_transitions"] > 0
    assert 0.0 <= metrics["occupancy"]["mean_iou"] <= 1.0
    assert metrics["ego_motion"]["rmse"] >= 0.0
    assert 0.0 <= metrics["collision"]["ece"] <= 1.0
    assert metrics["termination"]["count"] == metrics["valid_transitions"]
    assert metrics["termination"]["positive_count"] == 1
    assert [item["step"] for item in metrics["horizon_curve"]] == list(
        range(1, expected_steps + 1)
    )
    assert all(
        item["valid_transitions"] == 1 for item in metrics["horizon_curve"]
    )
    assert metrics["rollout_drift"]["first_step"] == 1
    assert metrics["rollout_drift"]["last_step"] == expected_steps
    assert (
        metrics["horizon_curve"][-1]["termination"]["positive_count"] == 1
    )


def test_generate_spatial_dataset_is_directly_trainable(tmp_path):
    image_path = tmp_path / "floorplan.png"
    output_path = tmp_path / "spatial.jsonl"
    _floorplan(image_path)
    summary = generate_spatial_world_model_dataset(
        str(image_path),
        str(output_path),
        episodes=8,
        routes_per_pair=1,
        minimum_distance=0.1,
        map_config=MapExtractionConfig(
            obstacle_dilation=0,
            max_dimension=180,
            resolution=0.1,
        ),
    )

    trajectories = load_world_model_trajectories(str(output_path))
    dimensions = infer_world_model_dimensions(trajectories)
    assert summary["episodes"] == 8
    assert summary["transitions"] > 8
    assert dimensions["observation_shape"] == (16,)
    assert dimensions["action_dim"] == 2


def test_spatial_route_cli_writes_json_and_html(tmp_path):
    image_path = tmp_path / "floorplan.png"
    _floorplan(image_path)
    json_path = tmp_path / "route.json"
    html_path = tmp_path / "route.html"
    png_path = tmp_path / "route.png"
    exit_code = cli_main(
        [
            "plan-spatial-route",
            str(image_path),
            "--start",
            "20,20",
            "--goal",
            "165,120",
            "--routes",
            "2",
            "--max-dimension",
            "180",
            "--output-json",
            str(json_path),
            "--output-html",
            str(html_path),
            "--output-png",
            str(png_path),
        ]
    )
    assert exit_code == 0
    assert json_path.is_file()
    assert html_path.is_file()
    assert png_path.is_file()
    assert len(json.loads(json_path.read_text(encoding="utf-8"))["routes"]) == 2
