import numpy as np
import torch

from saddlellm.VideoLatentFlowModel import (
    ConditionalVideoLatentFlowTransformer,
    VideoLatentFlowConfig,
)
from saddlellm.VideoLatentFlowTrainer import (
    CachedVideoLatentDataset,
    VideoLatentFlowTrainingConfig,
    load_video_latent_flow_checkpoint,
    train_video_latent_flow,
)
from saddlellm.TrainingOrchestrator import TrainingOrchestrator


def _config():
    return VideoLatentFlowConfig(
        latent_channels=2,
        latent_frames=2,
        latent_height=4,
        latent_width=4,
        temporal_patch_size=1,
        patch_size=2,
        condition_dim=6,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
    )


def test_video_flow_loss_and_sampling():
    model = ConditionalVideoLatentFlowTransformer(_config())
    latents = torch.randn(2, 2, 2, 4, 4)
    conditions = torch.randn(2, 6)
    loss = model.compute_flow_loss(latents, conditions)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    with torch.inference_mode():
        sample = model.sample(torch.zeros(1, 6), num_steps=1)
    assert sample.shape == (1, 2, 2, 4, 4)


def test_video_flow_training_checkpoint_and_reload(tmp_path):
    path = tmp_path / "video_latents.npz"
    rng = np.random.default_rng(41)
    np.savez(
        path,
        video_latents=rng.normal(size=(2, 2, 2, 4, 4)).astype("float32"),
        conditions=rng.normal(size=(2, 6)).astype("float32"),
    )
    output = tmp_path / "video_flow"
    result = train_video_latent_flow(
        _config(),
        VideoLatentFlowTrainingConfig(
            data_path=str(path),
            output_dir=str(output),
            batch_size=1,
            max_steps=1,
            checkpoint_steps=0,
            device="cpu",
        ),
    )
    assert result["status"] == "completed"
    assert result["global_step"] == 1
    loaded = load_video_latent_flow_checkpoint(str(output)).eval()
    with torch.inference_mode():
        sample = loaded.sample(torch.zeros(1, 6), num_steps=1)
    assert sample.shape == (1, 2, 2, 4, 4)


def test_video_dataset_infers_spatiotemporal_dimensions(tmp_path):
    path = tmp_path / "cached_video.npz"
    np.savez(
        path,
        video_latents=np.zeros((2, 3, 4, 8, 8), dtype="float32"),
        conditions=np.zeros((2, 12), dtype="float32"),
    )
    config = CachedVideoLatentDataset(str(path)).infer_model_config(
        temporal_patch_size=2,
        patch_size=2,
        hidden_size=12,
        num_layers=1,
        num_heads=3,
    )
    assert config.latent_channels == 3
    assert config.latent_frames == 4
    assert config.condition_dim == 12


def test_orchestrator_preflights_video_generation(tmp_path):
    path = tmp_path / "orchestrated_video.npz"
    np.savez(
        path,
        video_latents=np.zeros((2, 2, 2, 4, 4), dtype="float32"),
        conditions=np.zeros((2, 6), dtype="float32"),
    )
    results = TrainingOrchestrator.from_dict(
        {
            "stages": ["video_generation"],
            "training": {"dry_run": True},
            "video_generation": {
                "enabled": True,
                "data_path": str(path),
                "temporal_patch_size": 1,
                "patch_size": 2,
                "hidden_size": 8,
                "num_layers": 1,
                "num_heads": 2,
            },
            "logging": {
                "output_dir": str(tmp_path / "pipeline"),
                "backend": "local",
            },
            "eval": {"enabled": False},
        }
    ).run()
    assert results["video_generation"]["status"] == "planned"
    assert results["video_generation"]["model_config"]["latent_frames"] == 2


def test_orchestrator_trains_native_world_model(tmp_path):
    trajectory_path = tmp_path / "world.json"
    trajectory_path.write_text(
        '[{"observations":[[0.0,0.0],[0.1,0.0],[0.2,0.0]],'
        '"actions":[[0.1],[0.1]],"rewards":[0.0,1.0],'
        '"dones":[0.0,1.0]}]',
        encoding="utf-8",
    )
    results = TrainingOrchestrator.from_dict(
        {
            "stages": ["world_model"],
            "world_model": {
                "enabled": True,
                "backend": "rssm",
                "model": {
                    "embedding_size": 8,
                    "deterministic_size": 8,
                    "stochastic_size": 4,
                    "hidden_size": 8,
                    "free_nats": 0.0,
                },
                "data": {
                    "train_path": str(trajectory_path),
                    "sequence_length": 2,
                    "validation_split": 0.0,
                },
                "training": {
                    "output_dir": "world",
                    "num_epochs": 1,
                    "max_steps": 1,
                    "batch_size": 1,
                    "device": "cpu",
                    "mixed_precision": "no",
                },
            },
            "logging": {
                "output_dir": str(tmp_path / "pipeline"),
                "backend": "local",
            },
            "eval": {"enabled": False},
        }
    ).run()
    assert results["world_model"]["status"] == "completed"
    assert results["world_model"]["backend"] == "rssm"
    assert results["world_model"]["global_step"] == 1
    assert results["world_model"]["data"]["train"]["episodes"] == 1
    assert (tmp_path / "pipeline" / "world" / "world_model.pt").is_file()
