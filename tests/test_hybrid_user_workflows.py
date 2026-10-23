"""CLI/import configuration regressions, without model downloads or training."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from saddlellm import cli
from saddlellm.models.ModelBlueprint import ModelBlueprint
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator


def _sample_config():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "hybrid_kda_attnres_scratch.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_checked_in_sample_config_has_a_valid_hybrid_architecture():
    config = _sample_config()
    report = TrainingConfigValidator.validate(config, inspect_data=False)
    assert report.valid, report.issues
    assert report.architecture_analysis["architecture_axes"]["attention"] == "hybrid"
    assert (
        report.architecture_analysis["architecture_axes"]["attention_residuals"]
        == "block"
    )
    blueprint = ModelBlueprint.from_dict(config["model"]["blueprint"])
    assert [layer.attention.kind for layer in blueprint.expanded_layers()] == [
        "kda",
        "kda",
        "kda",
        "gqa",
    ]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda config: config["model"]["blueprint"]["residual"].update(
                block_size=3
            ),
            "even",
        ),
        (
            lambda config: config["model"]["blueprint"]["layers"][0][
                "attention"
            ].update(backend="sdpa"),
            "eager",
        ),
        (
            lambda config: config["model"].update(backend="hf"),
            "requires model.backend='saddle'",
        ),
    ],
)
def test_dry_run_rejects_bad_blueprints_before_touching_output(
    tmp_path, capsys, mutation, match
):
    config = copy.deepcopy(_sample_config())
    config["logging"]["output_dir"] = str(tmp_path / "must-not-exist")
    mutation(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert cli.main(["train", str(path), "--dry-run"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert match in payload["error"]
    assert not (tmp_path / "must-not-exist").exists()


def test_dry_run_valid_blueprint_never_builds_model_or_creates_outputs(
    tmp_path, monkeypatch, capsys
):
    config = _sample_config()
    config["logging"]["output_dir"] = str(tmp_path / "must-not-exist")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    def unexpected_build(*args, **kwargs):
        raise AssertionError("dry-run must not allocate a model")

    monkeypatch.setattr(ModelBlueprint, "build_model", unexpected_build)
    assert cli.main(["train", str(path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == ["tokenizer", "pretrain"]
    assert not (tmp_path / "must-not-exist").exists()
