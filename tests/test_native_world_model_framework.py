import json

import torch

from saddlellm.cli import main as cli_main
from saddlellm import (
    CategoricalWorldModel,
    CategoricalWorldModelConfig,
    WorldModel,
    WorldModelConfig,
    WorldModelPlannerConfig,
    WorldModelRuntime,
    list_world_model_backends,
    load_world_model,
    run_world_model_inference,
    train_world_model_from_config,
)


def _categorical_config():
    return CategoricalWorldModelConfig(
        observation_shape=(3,),
        action_dim=2,
        embedding_size=16,
        deterministic_size=16,
        stochastic_size=4,
        stochastic_classes=4,
        hidden_size=16,
        recurrent_blocks=4,
        reward_bins=15,
        free_nats=0.0,
    )


def _batch(batch_size=2, transitions=4):
    observations = torch.randn(batch_size, transitions + 1, 3)
    actions = torch.randn(batch_size, transitions, 2).clamp(-1.0, 1.0)
    return {
        "observations": observations,
        "actions": actions,
        "rewards": torch.randn(batch_size, transitions),
        "dones": torch.zeros(batch_size, transitions),
        "mask": torch.ones(batch_size, transitions),
    }


def _trajectories(num_episodes=3, transitions=4):
    episodes = []
    for episode in range(num_episodes):
        observations = [[float(episode), 0.0, 0.0]]
        actions = []
        for step in range(transitions):
            action = [0.05 * (step + 1), -0.02]
            previous = observations[-1]
            observations.append(
                [previous[0], previous[1] + action[0], previous[2] + action[1]]
            )
            actions.append(action)
        episodes.append(
            {
                "episode_id": f"ep-{episode}",
                "observations": observations,
                "actions": actions,
                "rewards": [0.0] * transitions,
                "dones": [0.0] * (transitions - 1) + [1.0],
            }
        )
    return episodes


def test_categorical_rssm_loss_imagination_and_generic_load(tmp_path):
    model = CategoricalWorldModel(_categorical_config())
    batch = _batch()
    losses = model.compute_loss(batch, sample_state=True)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["dynamics_kl_loss"])
    assert torch.isfinite(losses["representation_kl_loss"])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    model.eval()
    filtered = model(batch["observations"], batch["actions"], sample_state=False)
    imagined = model.imagine(filtered.final_state, batch["actions"], deterministic=True)
    assert filtered.prior_logits.shape == (2, 4, 4, 4)
    assert filtered.stochastic_states.shape == (2, 4, 16)
    assert imagined.observations.shape == (2, 4, 3)

    model.save_pretrained(str(tmp_path))
    saved_config = json.loads((tmp_path / model.CONFIG_NAME).read_text(encoding="utf-8"))
    assert saved_config["backend"] == "categorical_rssm"
    reloaded = load_world_model(str(tmp_path))
    assert isinstance(reloaded, CategoricalWorldModel)
    reloaded.eval()
    reloaded_output = reloaded(
        batch["observations"], batch["actions"], sample_state=False
    )
    assert torch.allclose(filtered.observations, reloaded_output.observations)


def test_categorical_config_entry_point_trains_on_cpu(tmp_path):
    data_path = tmp_path / "trajectories.json"
    data_path.write_text(json.dumps(_trajectories()), encoding="utf-8")
    output_path = tmp_path / "output"
    config = {
        "backend": "categorical_rssm",
        "model": _categorical_config().to_dict(),
        "data": {
            "train_path": str(data_path),
            "sequence_length": 4,
            "validation_split": 1 / 3,
        },
        "training": {
            "output_dir": str(output_path),
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "max_steps": 1,
            "device": "cpu",
            "mixed_precision": "no",
            "log_steps": 1,
            "checkpoint_steps": 0,
            "seed": 11,
        },
    }

    dry_run = train_world_model_from_config(config, dry_run=True)
    assert dry_run["status"] == "dry_run"
    assert dry_run["backend"] == "categorical_rssm"
    assert not output_path.exists()

    result = train_world_model_from_config(config)
    assert result["status"] == "completed"
    assert result["backend"] == "categorical_rssm"
    assert result["global_step"] == 1
    assert isinstance(load_world_model(str(output_path)), CategoricalWorldModel)


def test_categorical_rssm_supports_image_observations_and_discrete_actions():
    model = CategoricalWorldModel(
        CategoricalWorldModelConfig(
            observation_shape=(3, 8, 10),
            action_dim=5,
            action_type="discrete",
            embedding_size=16,
            deterministic_size=16,
            stochastic_size=4,
            stochastic_classes=4,
            hidden_size=16,
            recurrent_blocks=4,
            cnn_channels=(8, 16),
            reward_bins=15,
            free_nats=0.0,
        )
    )
    observations = torch.rand(2, 4, 3, 8, 10)
    actions = torch.tensor([[0, 1, 2], [4, 3, 2]])
    output = model(observations, actions, sample_state=False)
    losses = model.compute_loss({"observations": observations, "actions": actions})

    assert output.observations.shape == (2, 3, 3, 8, 10)
    assert output.prior_logits.shape == (2, 3, 4, 4)
    assert torch.isfinite(losses["loss"])


def test_native_runtime_rollout_scoring_planning_and_json_inference(tmp_path):
    model = CategoricalWorldModel(_categorical_config())
    model.save_pretrained(str(tmp_path))
    runtime = WorldModelRuntime.from_pretrained(str(tmp_path), device="cpu")
    state = runtime.encode_observation([0.0, 0.0, 0.0])

    rollout = runtime.rollout(state, [[0.1, 0.0], [0.0, -0.1]])
    assert rollout.observations.shape == (1, 2, 3)
    candidates = torch.zeros(12, 3, 2)
    candidates[:, :, 0] = torch.linspace(-1.0, 1.0, 12).unsqueeze(1)
    evaluation = runtime.score_action_sequences(state, candidates)
    assert evaluation.scores.shape == (12,)
    assert torch.isfinite(evaluation.scores).all()

    planner = WorldModelPlannerConfig(
        horizon=3,
        num_candidates=16,
        iterations=2,
        elite_fraction=0.25,
        seed=4,
    )
    plan = runtime.plan(state, planner)
    assert plan.actions.shape == (3, 2)
    assert torch.isfinite(plan.score)
    assert torch.all(plan.actions >= -1.0)
    assert torch.all(plan.actions <= 1.0)

    result = run_world_model_inference(
        str(tmp_path),
        {
            "observation": [0.0, 0.0, 0.0],
            "future_actions": [[0.1, 0.0], [0.0, 0.1]],
            "planner": planner.to_dict(),
        },
        device="cpu",
    )
    assert result["backend"] == "categorical_rssm"
    assert len(result["rollout"]["rewards"][0]) == 2
    assert len(result["plan"]["actions"]) == 3


def test_discrete_action_planner_and_native_backend_registry():
    model = WorldModel(
        WorldModelConfig(
            observation_shape=(3,),
            action_dim=4,
            action_type="discrete",
            embedding_size=12,
            deterministic_size=12,
            stochastic_size=4,
            hidden_size=12,
            free_nats=0.0,
        )
    )
    runtime = WorldModelRuntime(model, device="cpu")
    state = runtime.encode_observation([0.0, 0.0, 0.0])
    plan = runtime.plan(
        state,
        WorldModelPlannerConfig(
            horizon=3,
            num_candidates=12,
            iterations=2,
            elite_fraction=0.25,
            seed=3,
        ),
    )
    assert plan.actions.shape == (3,)
    assert plan.actions.dtype == torch.long
    assert int(plan.actions.min()) >= 0
    assert int(plan.actions.max()) < 4

    backends = list_world_model_backends()
    assert {item["name"] for item in backends} == {"rssm", "categorical_rssm"}
    assert all(item["kind"] == "native" for item in backends)


def test_native_inference_cli(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    CategoricalWorldModel(_categorical_config()).save_pretrained(str(checkpoint))
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "observation": [0.0, 0.0, 0.0],
                "future_actions": [[0.1, 0.0], [0.0, 0.1]],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"

    exit_code = cli_main(
        [
            "infer-world-model",
            str(checkpoint),
            str(request_path),
            "--device",
            "cpu",
            "--output",
            str(result_path),
        ]
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["status"] == "completed"
    assert result["backend"] == "categorical_rssm"
