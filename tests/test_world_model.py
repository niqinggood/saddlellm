import json

import pytest
import torch

from saddlellm import (
    WorldModel,
    WorldModelConfig,
    WorldModelTrajectoryDataset,
    load_world_model_trajectories,
    train_world_model_from_config,
)


def _tiny_trajectories(num_episodes=3, transitions=4):
    trajectories = []
    for episode in range(num_episodes):
        observations = [[float(episode), 0.0, 0.0]]
        actions = []
        rewards = []
        for step in range(transitions):
            action = [0.05 * (step + 1), -0.02 * (episode + 1)]
            previous = observations[-1]
            next_observation = [
                previous[0],
                previous[1] + action[0],
                previous[2] + action[1],
            ]
            observations.append(next_observation)
            actions.append(action)
            rewards.append(-sum(value * value for value in next_observation) / 3.0)
        trajectories.append(
            {
                "episode_id": f"episode-{episode}",
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "dones": [0.0] * (transitions - 1) + [1.0],
            }
        )
    return trajectories


def _tiny_model_config():
    return WorldModelConfig(
        observation_shape=(3,),
        action_dim=2,
        embedding_size=16,
        deterministic_size=16,
        stochastic_size=4,
        hidden_size=16,
        free_nats=0.0,
    )


def test_world_model_loss_imagination_and_save_round_trip(tmp_path):
    dataset = WorldModelTrajectoryDataset(
        _tiny_trajectories(), sequence_length=4, stride=2
    )
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    model = WorldModel(_tiny_model_config())

    losses = model.compute_loss(batch, sample_state=True)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    model.eval()
    filtered = model(batch["observations"], batch["actions"], sample_state=False)
    imagined = model.imagine(filtered.initial_state, batch["actions"], deterministic=True)
    assert filtered.observations.shape == (2, 4, 3)
    assert filtered.rewards.shape == (2, 4)
    assert imagined.observations.shape == (2, 4, 3)
    assert imagined.continuation.min() >= 0
    assert imagined.continuation.max() <= 1

    model.save_pretrained(str(tmp_path))
    reloaded = WorldModel.from_pretrained(str(tmp_path))
    reloaded.eval()
    reloaded_output = reloaded(
        batch["observations"], batch["actions"], sample_state=False
    )
    assert reloaded.config.to_dict() == model.config.to_dict()
    assert torch.allclose(filtered.observations, reloaded_output.observations)


def test_image_world_model_accepts_discrete_actions():
    model = WorldModel(
        WorldModelConfig(
            observation_shape=(3, 8, 10),
            action_dim=5,
            action_type="discrete",
            embedding_size=16,
            deterministic_size=16,
            stochastic_size=4,
            hidden_size=16,
            cnn_channels=(8, 16),
            free_nats=0.0,
        )
    )
    observations = torch.rand(2, 4, 3, 8, 10)
    actions = torch.tensor([[0, 1, 2], [4, 3, 2]])
    output = model(observations, actions, sample_state=False)
    losses = model.compute_loss({"observations": observations, "actions": actions})

    assert output.observations.shape == (2, 3, 3, 8, 10)
    assert torch.isfinite(losses["loss"])
    with pytest.raises(ValueError, match="Discrete action ids"):
        model(observations, torch.tensor([[0, 1, 5], [0, 1, 2]]))


def test_transition_jsonl_is_grouped_and_short_windows_are_masked(tmp_path):
    path = tmp_path / "transitions.jsonl"
    rows = [
        {
            "episode_id": "a",
            "step_id": 0,
            "observation": [0, 0],
            "action": [0.1],
            "next_observation": [0.1, 0],
            "reward": 0.0,
            "done": False,
        },
        {
            "episode_id": "a",
            "step_id": 1,
            "observation": [0.1, 0],
            "action": [0.1],
            "next_observation": [0.2, 0],
            "reward": 1.0,
            "done": True,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in reversed(rows)) + "\n",
        encoding="utf-8",
    )

    trajectories = load_world_model_trajectories(str(path))
    dataset = WorldModelTrajectoryDataset(trajectories, sequence_length=4)
    item = dataset[0]

    assert len(trajectories) == 1
    assert item["observations"].shape == (5, 2)
    assert item["actions"].shape == (4, 1)
    assert item["mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert item["dones"].tolist() == [0.0, 1.0, 1.0, 1.0]


def test_config_entry_point_dry_run_and_tiny_cpu_training(tmp_path):
    data_path = tmp_path / "trajectories.json"
    data_path.write_text(json.dumps(_tiny_trajectories()), encoding="utf-8")
    output_path = tmp_path / "output"
    config = {
        "model": {
            **_tiny_model_config().to_dict(),
            "kl_weight": 0.1,
        },
        "data": {
            "train_path": str(data_path),
            "sequence_length": 4,
            "validation_split": 1 / 3,
        },
        "training": {
            "output_dir": str(output_path),
            "num_epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "max_steps": 2,
            "device": "cpu",
            "mixed_precision": "no",
            "log_steps": 1,
            "checkpoint_steps": 1,
            "seed": 7,
        },
    }

    dry_run = train_world_model_from_config(config, dry_run=True)
    assert dry_run["status"] == "dry_run"
    assert dry_run["train_data"]["episodes"] == 2
    assert dry_run["validation_data"]["episodes"] == 1
    assert not output_path.exists()

    result = train_world_model_from_config(config)
    assert result["status"] == "completed"
    assert result["global_step"] == 2
    assert (output_path / WorldModel.CONFIG_NAME).is_file()
    assert (output_path / WorldModel.WEIGHTS_NAME).is_file()
    assert (output_path / "training_metrics.json").is_file()
    assert (output_path / "checkpoints" / "step-00000001" / "trainer_state.pt").is_file()

