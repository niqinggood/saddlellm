import json
from pathlib import Path

import pytest
import torch

from saddlellm.decision.Model import ChoiceDecisionModel
from saddlellm.decision.training import (
    ChoiceTrainingConfig,
    load_dataset,
    sha256,
    train_choice,
)


def dataset(directory: Path):
    directory.mkdir()
    descriptor = {
        "schema_version": 1, "task_type": "choice", "question": "Choose the category",
        "choices": [{"id": "alpha", "text": "alpha"}, {"id": "beta", "text": "beta"}],
        "splits": {name: name + ".jsonl" for name in ("train", "validation", "calibration", "test")},
    }
    (directory / "dataset.json").write_text(json.dumps(descriptor), encoding="utf-8")
    for name, filename in descriptor["splits"].items():
        rows = [{"id": f"{name}:{i}", "state": f"{label} {name} example {i}", "label": label}
                for i, label in enumerate(["alpha", "beta"] * 4)]
        (directory / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return directory


def test_rejects_cross_split_normalized_leakage(tmp_path):
    data = dataset(tmp_path / "data")
    path = data / "calibration.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["state"] = "  ALPHA\tTRAIN example 0  "
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="leakage"):
        load_dataset(data)


def test_manifest_modification_and_missing_calibration_fail(tmp_path):
    data = dataset(tmp_path / "data")
    (data / "manifest.json").write_text(json.dumps({"output_sha256": {"train.jsonl": sha256(data / "train.jsonl")}}))
    with (data / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(data)
    descriptor_path = data / "dataset.json"
    descriptor = json.loads(descriptor_path.read_text())
    del descriptor["splits"]["calibration"]
    descriptor_path.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="four separate"):
        load_dataset(data)


def test_split_cannot_escape_dataset(tmp_path):
    data = dataset(tmp_path / "data")
    descriptor_path = data / "dataset.json"
    descriptor = json.loads(descriptor_path.read_text())
    descriptor["splits"]["test"] = "../outside.jsonl"
    descriptor_path.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="inside"):
        load_dataset(data)


def test_training_calibration_save_reload_and_phase_order(tmp_path, monkeypatch, capsys):
    data = dataset(tmp_path / "data")
    backbone = tmp_path / "backbone"
    backbone.mkdir()
    (backbone / "config.json").write_text(json.dumps({"hidden_size": 4}))

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            vectors = []
            for text in texts:
                option = "Option:" in text
                alpha = "alpha" in text
                first = alpha if option else not alpha
                vectors.append([1., 0., 0., 0.] if first else [0., 1., 0., 0.])
            return torch.tensor(vectors)

    monkeypatch.setattr("saddlellm.decision.Model.FrozenTextEncoder", Encoder)
    config = ChoiceTrainingConfig(epochs=20, batch_size=8, projection_rank=2, learning_rate=.05, cpu_threads=1)
    output = tmp_path / "run"
    report = train_choice(data, backbone, output, config=config, encoder_device="cpu", encoder_dtype="float32")
    assert report["selection"]["best_epoch"] > 0
    assert report["test"]["trained_calibrated"]["accuracy"] > report["test"]["cosine_baseline"]["accuracy"]
    assert report["artifact_reload_verified"] is True
    assert report["test_used_for_selection"] is False
    restored = ChoiceDecisionModel.from_pretrained(output / "model")
    assert restored.calibration["metadata"]["split_sha256"] == sha256(data / "calibration.jsonl")
    assert restored.calibration["metadata"]["scope"]["choices"][0]["id"] == "alpha"
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    phases = [line.get("split", "training") for line in lines if line.get("phase") != "completed"]
    assert phases[:2] == ["train", "validation"]
    assert phases[-2:] == ["calibration", "test"]
    with pytest.raises(FileExistsError):
        train_choice(data, backbone, output, config=config, encoder_device="cpu")


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"learning_rate": float("nan")}, {"weight_decay": -1}])
def test_bad_training_config(kwargs):
    with pytest.raises(ValueError):
        ChoiceTrainingConfig(**kwargs)
