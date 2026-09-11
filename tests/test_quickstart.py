from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
import yaml

import saddlellm
from saddlellm.cli import main as cli_main
from saddlellm.factory.Quickstart import create_quickstart
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator
from saddlellm.data.TrainingDataInspector import inspect_training_data


@pytest.mark.parametrize(
    ("stages", "preference_task"),
    [("sft", None), ("sft,preference", "dpo"), ("sft,preference", "kto")],
)
def test_quickstart_creates_valid_beginner_project(tmp_path, stages, preference_task):
    workspace = tmp_path / f"case-{preference_task or 'sft'}"
    result = create_quickstart(
        str(workspace),
        stages=stages,
        preference_method=preference_task or "dpo",
    )

    config = yaml.safe_load(Path(result.config).read_text(encoding="utf-8"))
    validation = TrainingConfigValidator.validate(config, inspect_data=False)
    assert validation.valid, validation.issues
    assert config["stages"] == result.stages
    assert config["sft"]["use_lora"] is True
    assert config["sft"]["use_qlora"] is False
    assert config["training"]["max_steps"] == 10
    assert config["eval"]["enabled"] is False

    assert (workspace / config["sft"]["data_path"]).is_file()
    assert inspect_training_data(result.data_files["sft"], task="sft")["ready"]

    if preference_task:
        assert inspect_training_data(
            result.data_files["preference"], task=preference_task
        )["ready"]
        assert config["preference"]["method"] == preference_task
    else:
        assert "preference" not in result.data_files
    assert "saddle-llm train \"config.yaml\" --dry-run" in result.next_steps
    assert Path(workspace / "README.md").is_file()


def test_quickstart_refuses_conflicts_and_force_preserves_unrelated_files(tmp_path):
    workspace = tmp_path / "starter"
    create_quickstart(str(workspace))
    unrelated = workspace / "my-notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        create_quickstart(str(workspace))

    create_quickstart(str(workspace), stages="sft,preference", overwrite=True)
    assert unrelated.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("stages", ["", "preference", "preference,sft", "sft,sft"])
def test_quickstart_only_accepts_the_two_documented_stage_lists(tmp_path, stages):
    with pytest.raises(ValueError, match="must be `sft` or `sft,preference`"):
        create_quickstart(str(tmp_path / "starter"), stages=stages)


def test_quickstart_references_user_data_without_modifying_it(tmp_path):
    source = tmp_path / "customer-data.jsonl"
    original = json.dumps({"instruction": "hello", "output": "world"}) + "\n"
    source.write_text(original, encoding="utf-8")

    result = create_quickstart(str(tmp_path / "starter"), sft_data=str(source))
    config = yaml.safe_load(Path(result.config).read_text(encoding="utf-8"))

    assert source.read_text(encoding="utf-8") == original
    assert str(source) == result.data_files["sft"]
    assert str(source) not in result.created_files
    assert (Path(result.workspace) / config["sft"]["data_path"]).resolve() == source


def test_cli_without_command_shows_a_short_getting_started_path(capsys):
    assert cli_main([]) == 0
    output = capsys.readouterr().out
    assert "saddle-llm quickstart" in output
    assert "saddle-llm --help" in output


def test_quickstart_is_available_from_the_top_level_python_api(tmp_path):
    result = saddlellm.create_quickstart(str(tmp_path / "python-api"))
    assert isinstance(result, saddlellm.factory.QuickstartResult)
    assert Path(result.config).is_file()


def test_quickstart_cli_emits_artifacts_and_next_steps(tmp_path, capsys):
    workspace = tmp_path / "cli-starter"
    assert cli_main(["quickstart", str(workspace), "--stages", "sft,preference"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == ["sft", "preference"]
    assert Path(payload["config"]).is_file()
    assert payload["next_steps"][0].startswith("cd ")


def test_quickstart_cli_reports_conflict_without_traceback(tmp_path, capsys):
    workspace = tmp_path / "cli-starter"
    assert cli_main(["quickstart", str(workspace)]) == 0
    capsys.readouterr()

    assert cli_main(["quickstart", str(workspace)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_type"] == "FileExistsError"
    assert "--force" in payload["error"]


def test_basic_config_validation_does_not_require_external_agent_package(
    tmp_path, monkeypatch
):
    result = create_quickstart(str(tmp_path / "starter"))
    config = yaml.safe_load(Path(result.config).read_text(encoding="utf-8"))
    real_import = builtins.__import__

    def import_without_saddle_ml(name, *args, **kwargs):
        if name == "saddle_ml" or name.startswith("saddle_ml."):
            raise ImportError("saddle_ml intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_saddle_ml)
    report = TrainingConfigValidator.validate(config)

    assert report.valid, report.issues
