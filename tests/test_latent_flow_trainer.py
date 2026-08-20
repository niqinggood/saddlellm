import numpy as np
import torch

from saddlellm.LatentFlowModel import LatentFlowConfig
from saddlellm.LatentFlowTrainer import (
    CachedLatentDataset,
    LatentFlowTrainingConfig,
    load_latent_flow_checkpoint,
    train_latent_flow,
)
from saddlellm.TrainingOrchestrator import TrainingOrchestrator


def test_cached_latent_training_and_reload(tmp_path):
    data_path = tmp_path / "latents.npz"
    rng = np.random.default_rng(9)
    np.savez(
        data_path,
        latents=rng.normal(size=(4, 2, 4, 4)).astype("float32"),
        conditions=rng.normal(size=(4, 6)).astype("float32"),
    )
    model_config = LatentFlowConfig(
        latent_channels=2,
        latent_height=4,
        latent_width=4,
        patch_size=2,
        condition_dim=6,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
    )
    output = tmp_path / "flow"
    result = train_latent_flow(
        model_config,
        LatentFlowTrainingConfig(
            data_path=str(data_path),
            output_dir=str(output),
            batch_size=2,
            epochs=1,
            max_steps=1,
            warmup_steps=0,
            checkpoint_steps=0,
            device="cpu",
        ),
    )

    assert result["status"] == "completed"
    assert result["global_step"] == 1
    assert np.isfinite(result["final_loss"])
    model = load_latent_flow_checkpoint(str(output)).eval()
    with torch.inference_mode():
        sample = model.sample(torch.zeros(1, 6), num_steps=1)
    assert sample.shape == (1, 2, 4, 4)
    assert torch.isfinite(sample).all()


def test_cached_dataset_infers_codec_dimensions(tmp_path):
    path = tmp_path / "cached.npz"
    np.savez(
        path,
        latents=np.zeros((2, 3, 8, 8), dtype="float32"),
        conditions=np.zeros((2, 12), dtype="float32"),
    )
    config = CachedLatentDataset(str(path)).infer_model_config(
        patch_size=2, hidden_size=16, num_layers=1, num_heads=4
    )
    assert config.latent_channels == 3
    assert config.condition_dim == 12


def test_orchestrator_runs_image_generation_stage(tmp_path):
    data_path = tmp_path / "orchestrated.npz"
    np.savez(
        data_path,
        latents=np.zeros((2, 2, 4, 4), dtype="float32"),
        conditions=np.zeros((2, 6), dtype="float32"),
    )
    results = TrainingOrchestrator.from_dict(
        {
            "stages": ["image_generation"],
            "image_generation": {
                "enabled": True,
                "data_path": str(data_path),
                "output_dir": "flow",
                "patch_size": 2,
                "hidden_size": 8,
                "num_layers": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "batch_size": 1,
                "epochs": 1,
                "max_steps": 1,
                "warmup_steps": 0,
                "checkpoint_steps": 0,
                "device": "cpu",
            },
            "logging": {
                "output_dir": str(tmp_path / "pipeline"),
                "backend": "local",
            },
            "eval": {"enabled": False},
        }
    ).run()

    assert results["image_generation"]["status"] == "completed"
    assert results["image_generation"]["global_step"] == 1
    assert (tmp_path / "pipeline" / "flow" / "latent_flow_model.bin").is_file()
