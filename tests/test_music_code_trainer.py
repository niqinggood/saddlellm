import numpy as np
import torch

from saddlellm.MusicCodeModel import MusicCodeConfig
from saddlellm.MusicCodeTrainer import (
    CachedMusicCodeDataset,
    MusicCodeTrainingConfig,
    load_music_code_checkpoint,
    train_music_code,
)
from saddlellm.TrainingOrchestrator import TrainingOrchestrator


def test_music_code_training_checkpoint_and_reload(tmp_path):
    data_path = tmp_path / "music_codes.npz"
    rng = np.random.default_rng(31)
    np.savez(
        data_path,
        codes=rng.integers(0, 8, size=(4, 2, 5), dtype=np.int64),
        conditions=rng.normal(size=(4, 6)).astype("float32"),
        attention_mask=np.ones((4, 5), dtype=np.int64),
    )
    model_config = MusicCodeConfig(
        num_codebooks=2,
        codebook_size=8,
        max_sequence_length=5,
        condition_dim=6,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
    )
    output = tmp_path / "music"
    result = train_music_code(
        model_config,
        MusicCodeTrainingConfig(
            data_path=str(data_path),
            output_dir=str(output),
            batch_size=2,
            max_steps=1,
            warmup_steps=0,
            checkpoint_steps=0,
            device="cpu",
        ),
    )
    assert result["status"] == "completed"
    assert result["global_step"] == 1
    model = load_music_code_checkpoint(str(output)).eval()
    with torch.inference_mode():
        codes = model.generate(torch.zeros(1, 6), max_frames=4, temperature=0)
    assert codes.shape == (1, 2, 4)


def test_music_dataset_infers_dimensions(tmp_path):
    path = tmp_path / "music.npz"
    np.savez(
        path,
        codes=np.full((2, 3, 7), 11, dtype=np.int64),
        conditions=np.zeros((2, 9), dtype="float32"),
    )
    config = CachedMusicCodeDataset(str(path)).infer_model_config(
        hidden_size=12, num_layers=1, num_heads=3
    )
    assert config.num_codebooks == 3
    assert config.codebook_size == 12
    assert config.max_sequence_length == 7
    assert config.condition_dim == 9


def test_orchestrator_runs_music_generation_stage(tmp_path):
    data_path = tmp_path / "orchestrated_music.npz"
    rng = np.random.default_rng(37)
    np.savez(
        data_path,
        codes=rng.integers(0, 8, size=(2, 2, 4), dtype=np.int64),
        conditions=np.zeros((2, 6), dtype="float32"),
    )
    results = TrainingOrchestrator.from_dict(
        {
            "stages": ["music_generation"],
            "music_generation": {
                "enabled": True,
                "data_path": str(data_path),
                "output_dir": "music",
                "codebook_size": 16,
                "max_sequence_length": 8,
                "hidden_size": 8,
                "num_layers": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "batch_size": 1,
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

    assert results["music_generation"]["status"] == "completed"
    assert results["music_generation"]["global_step"] == 1
    assert results["music_generation"]["model_config"]["codebook_size"] == 16
    assert (tmp_path / "pipeline" / "music" / "music_code_model.bin").is_file()
