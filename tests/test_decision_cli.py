import json
import subprocess
import sys

import pytest
import torch

from saddlellm import cli
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def prepared(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    descriptor = {"schema_version": 1, "task_type": "choice", "question": "Select a category",
                  "choices": [{"id": "alpha", "text": "alpha"}, {"id": "beta", "text": "beta"}],
                  "splits": {name: name + ".jsonl" for name in ("train", "validation", "calibration", "test")}}
    (data / "dataset.json").write_text(json.dumps(descriptor), encoding="utf-8")
    for name, filename in descriptor["splits"].items():
        rows = [{"id": f"{name}:{i}", "state": f"{label} {name} {i}", "label": label}
                for i, label in enumerate(["alpha", "beta"] * 4)]
        (data / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    backbone = tmp_path / "backbone"
    backbone.mkdir()
    (backbone / "config.json").write_text('{"hidden_size": 4}', encoding="utf-8")
    return {"stages": ["decision"], "model": {"name_or_path": str(backbone)},
            "decision": {"dataset_dir": str(data), "projection_rank": 2, "epochs": 6, "learning_rate": .05, "encoder_device": "cpu", "cpu_threads": 1},
            "logging": {"output_dir": str(tmp_path / "output"), "logging_backend": "local"}, "eval": {"enabled": False}}


def test_direct_dry_run_is_registered_and_has_no_side_effects(tmp_path, capsys):
    output = tmp_path / "new-run"
    assert cli.main(["train", "decision", "--model", "local-backbone", "--data", "decision-data", "--output-dir", str(output), "--epochs", "3", "--projection-rank", "8", "--precision", "bf16", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["pipeline_plan"]["nodes"][0]["family"] == "decision"
    assert "decision_model" in result["pipeline_plan"]["nodes"][0]["produces"]
    assert result["decision"]["settings"]["projection_rank"] == 8
    assert result["decision"]["settings"]["encoder_dtype"] == "bfloat16"
    assert result["decision"]["rlcd_implemented"] is False
    assert not output.exists()


def test_dry_run_loads_no_tensor_libraries_or_weights():
    code = "from saddlellm.cli import main; import sys; assert main(['train','--stage','decision','--model','fake','--data','fake','--dry-run'])==0; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'saddlellm.decision.training' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("extra", [["--method", "lora"], ["--resume-from-checkpoint"], ["--max-steps", "10"], ["--backend", "saddle"]])
def test_direct_decision_rejects_unimplemented_options(tmp_path, capsys, extra):
    output = tmp_path / "untouched"
    assert cli.main(["train", "--stage", "decision", "--model", "fake", "--data", "fake", "--output-dir", str(output), "--dry-run", *extra]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert not output.exists()


@pytest.mark.parametrize("change", [{"method": "rlcd"}, {"task_type": "score"}, {"task_type": "noul"}, {"freeze_backbone": False}, {"epochs": 0}, {"learning_rate": float("nan")}])
def test_config_validator_rejects_unsupported_semantics(tmp_path, change):
    raw = prepared(tmp_path)
    raw["decision"].update(change)
    report = TrainingConfigValidator.validate(raw, inspect_data=False)
    assert report.valid is False
    assert any("Decision training" in issue for issue in report.issues)
    assert not (tmp_path / "output").exists()


def test_config_validation_audits_splits_without_creating_output(tmp_path):
    raw = prepared(tmp_path)
    report = TrainingConfigValidator.validate(raw)
    assert report.valid, report.issues
    assert report.data_inspections["decision"]["counts"] == dict.fromkeys(["train", "validation", "calibration", "test"], 8)
    assert not (tmp_path / "output").exists()
    test_path = tmp_path / "data/test.jsonl"
    rows = [json.loads(line) for line in test_path.read_text().splitlines()]
    rows[0]["state"] = "alpha train 0"
    test_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert TrainingConfigValidator.validate(raw).valid is False


def test_direct_cli_trains_and_registers_artifacts_then_resumes(tmp_path, capsys, monkeypatch):
    raw = prepared(tmp_path)

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            return torch.tensor([[1., 0., 0., 0.] if ("alpha" in text) == ("Option:" in text) else [0., 1., 0., 0.] for text in texts])

    monkeypatch.setattr("saddlellm.decision.Model.FrozenTextEncoder", Encoder)
    assert cli.main(["train", "--stage", "decision", "--model", raw["model"]["name_or_path"], "--data", raw["decision"]["dataset_dir"], "--output-dir", raw["logging"]["output_dir"], "--encoder-device", "cpu", "--projection-rank", "2", "--epochs", "6", "--learning-rate", ".05", "--cpu-threads", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    stage = result["results"]["decision"]
    assert stage["status"] == "completed"
    assert "serve-decision" in result["next"]["serve"]
    from pathlib import Path

    model_path = Path(stage["model_path"])
    assert (model_path / "decision_head.safetensors").is_file()
    state_path = tmp_path / "output/pipeline_state.json"
    state = json.loads(state_path.read_text())
    assert state["artifacts"]["decision.model"]["kind"] == "decision_model"
    assert state["artifacts"]["decision.model_path"]["kind"] == "decision_model"
    # Use the exact saved canonical config; resume must not invoke training.
    saved = json.loads((tmp_path / "output/command_config.json").read_text())
    saved["pipeline"] = {"resume": True}
    monkeypatch.setattr("saddlellm.decision.training.train_choice", lambda *a, **k: pytest.fail("resumed completed decision stage retrained"))
    restored = TrainingOrchestrator.from_dict(saved).run()
    assert restored["decision"]["model_path"] == str(model_path)
    assert json.loads(state_path.read_text())["restored_stages"] == ["decision"]


def test_failed_attempt_retries_into_new_directory(tmp_path, monkeypatch):
    raw = prepared(tmp_path)
    calls = []

    def train(data, backbone, destination, **kwargs):
        destination.mkdir(parents=True)
        calls.append(destination)
        raise RuntimeError("interrupted encoding")

    monkeypatch.setattr("saddlellm.decision.training.train_choice", train)
    with pytest.raises(RuntimeError, match="interrupted"):
        TrainingOrchestrator.from_dict(raw).run()
    raw["pipeline"] = {"resume": True}
    with pytest.raises(RuntimeError, match="interrupted"):
        TrainingOrchestrator.from_dict(raw).run()
    assert [path.name for path in calls] == ["attempt-001", "attempt-002"]
    assert all(path.is_dir() for path in calls)


def test_choice_does_not_enter_causal_lm_or_distributed_stages(tmp_path):
    raw = prepared(tmp_path)
    raw["stages"] = ["decision", "eval"]
    assert not TrainingConfigValidator.validate(raw, inspect_data=False).valid
    raw["stages"] = ["decision"]
    raw["distributed"] = {"strategy": "ddp", "num_gpus": 2}
    assert not TrainingConfigValidator.validate(raw, inspect_data=False).valid


def test_serve_decision_dry_run_does_not_load_model_or_expose_key(monkeypatch, capsys):
    monkeypatch.setenv("DECISION_TEST_KEY", "do-not-print-this")
    assert cli.main(["serve-decision", "fake", "--backbone", "external", "--dtype", "bfloat16", "--api-key-env", "DECISION_TEST_KEY", "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "do-not-print-this" not in text
    assert json.loads(text)["requires_auth"] is True


def test_optional_decision_settings_preserve_old_run_fingerprints():
    from dataclasses import asdict

    from saddlellm.framework.pipeline import normalized_run_fingerprint

    config = TrainingOrchestrator._parse_config({"stages": ["operator"], "operator": {"enabled": False}})
    previous = asdict(config)
    previous.pop("decision")
    assert normalized_run_fingerprint(config) == normalized_run_fingerprint(previous)
    config.decision.epochs = 99
    assert normalized_run_fingerprint(config) == normalized_run_fingerprint(previous)


def test_relocated_backbone_requires_matching_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from saddlellm.decision.dataset import sha256
    from saddlellm.decision.Server import ChoiceDecisionEngine, DecisionServerSettings

    backbone = tmp_path / "backbone"
    backbone.mkdir()
    config = backbone / "config.json"
    config.write_text("{}")
    fake = SimpleNamespace(config=SimpleNamespace(backbone_identity={"config.json": sha256(config)}),
                           metadata={"question": "Pick one", "choices": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]},
                           calibration={"calibrated": False}, calibration_temperature=1.0)
    monkeypatch.setattr("saddlellm.decision.Model.ChoiceDecisionModel.from_pretrained", lambda *args, **kwargs: fake)
    settings = DecisionServerSettings(model_path="fake", backbone_name_or_path=str(backbone))
    assert ChoiceDecisionEngine(settings).model is fake
    config.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity"):
        ChoiceDecisionEngine(settings)
