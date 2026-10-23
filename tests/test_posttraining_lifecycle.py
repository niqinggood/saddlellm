"""Public lifecycle entry points must preserve configuration and gate failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from saddlellm import cli
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def suite_file(tmp_path):
    data = tmp_path / "general.jsonl"
    data.write_text('{"instruction":"question","output":"answer"}\n', encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"tasks": [{
        "name": "general", "task": "completion", "data": "general.jsonl",
        "metrics": {"completion_nll": {"max_drop": 0.1, "higher_is_better": False}},
    }]}), encoding="utf-8")
    return suite


def config(tmp_path, suite, *, reject=True):
    return {
        "model": {"name_or_path": "original-model"},
        "stages": ["eval", "export"],
        "logging": {"output_dir": str(tmp_path / "run"), "logging_backend": "local"},
        "eval": {"regression_suite": str(suite), "regression_fail_on_rejection": reject},
        "export": {"enabled": True, "require_gate": True},
    }


def test_direct_sft_passes_replay_config_and_seed(monkeypatch):
    seen = {}

    def train(args):
        seen.update(cli._build_direct_train_config(args, ["sft"]))
        return 0

    monkeypatch.setattr(cli, "_cmd_train", train)
    assert cli.main([
        "train", "--stage", "sft", "--model", "local", "--data", "business.jsonl",
        "--replay-data", "history.jsonl", "--replay-ratio", "0.2", "--seed", "17",
    ]) == 0
    parsed = TrainingOrchestrator._parse_config(seen)
    assert parsed.sft.replay_data_path == "history.jsonl"
    assert parsed.sft.replay_ratio == 0.2
    assert parsed.seed == 17


def test_direct_preference_does_not_silently_ignore_replay(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cmd_train", lambda args: cli._build_direct_train_config(args, ["dpo"]))
    assert cli.main([
        "train", "--stage", "dpo", "--model", "local", "--data", "prefs.jsonl",
        "--replay-data", "history.jsonl", "--replay-count", "3",
    ]) == 1
    assert "Replay options require sft" in capsys.readouterr().out


@pytest.mark.parametrize("replay", [
    {"replay_ratio": 0.1},
    {"replay_data_path": "history", "replay_ratio": float("nan")},
    {"replay_data_path": "history", "replay_ratio": 1.0},
    {"replay_data_path": "history", "replay_ratio": 0.2, "replay_count": 3},
    {"replay_data_path": "history", "replay_count": True},
])
def test_bad_replay_fails_during_config_parse(replay):
    with pytest.raises(ValueError):
        TrainingOrchestrator._parse_config({"sft": replay})


def test_regression_dry_run_does_not_load_any_model(tmp_path, monkeypatch, capsys):
    import saddlellm.evaluation.RegressionSuite as regression

    suite = suite_file(tmp_path)
    monkeypatch.setattr(regression, "run_regression_suite", lambda *a, **k: pytest.fail("dry-run loaded models"))
    output = tmp_path / "must-not-be-created.json"
    assert cli.main([
        "regression", str(suite), "--baseline", "old", "--candidate", "new",
        "--output", str(output), "--dry-run",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "planned"
    assert Path(result["suite"]["tasks"][0]["data"]).is_absolute()
    assert not output.exists()


@pytest.mark.parametrize("accepted,expected", [(True, 0), (False, 1)])
def test_regression_cli_exit_code_reflects_gate(tmp_path, monkeypatch, capsys, accepted, expected):
    import saddlellm.evaluation.RegressionSuite as regression

    monkeypatch.setattr(regression, "run_regression_suite", lambda *a, **k: {
        "accepted": accepted, "status": "accepted" if accepted else "failed",
    })
    assert cli.main([
        "regression", str(suite_file(tmp_path)), "--baseline", "old", "--candidate", "new",
    ]) == expected
    assert json.loads(capsys.readouterr().out)["ok"] is accepted


def test_regression_suite_is_valid_release_gate_configuration(tmp_path):
    report = TrainingConfigValidator.validate(config(tmp_path, suite_file(tmp_path)), inspect_data=False)
    assert report.valid, report.issues


def test_regression_must_precede_export(tmp_path):
    raw = config(tmp_path, suite_file(tmp_path))
    raw["stages"] = ["export", "eval"]
    report = TrainingConfigValidator.validate(raw, inspect_data=False)
    assert not report.valid
    assert any("eval before export" in issue for issue in report.issues)


def test_regression_does_not_silently_discard_legacy_gate_rules(tmp_path):
    raw = config(tmp_path, suite_file(tmp_path))
    raw["eval"]["gate"] = {"enabled": True, "rules": {"accuracy": {"min": 0.5}}}
    report = TrainingConfigValidator.validate(raw, inspect_data=False)
    assert not report.valid
    assert any("instead of also enabling" in issue for issue in report.issues)


def test_rejected_regression_blocks_export_even_when_eval_continues(tmp_path, monkeypatch):
    import saddlellm.evaluation.RegressionSuite as regression

    monkeypatch.setattr(regression, "run_regression_suite", lambda *a, **k: {
        "accepted": False, "status": "failed", "candidate_model_identity": None,
    })
    runner = TrainingOrchestrator.from_dict(config(tmp_path, suite_file(tmp_path), reject=False))
    try:
        runner._run_eval_stage()
        gate = json.loads(Path(runner._stage_results["eval"]["release_gate_path"]).read_text(encoding="utf-8"))
        assert gate["accepted"] is False
        with pytest.raises(RuntimeError, match="Export blocked"):
            runner._run_export_stage()
        assert runner._stage_results["export"]["status"] == "blocked"
    finally:
        runner._close_logging_handlers()


def test_regression_runtime_error_is_saved_as_invalid_gate(tmp_path, monkeypatch):
    import saddlellm.evaluation.RegressionSuite as regression

    def broken(*args, **kwargs):
        raise OSError("missing candidate checkpoint")

    monkeypatch.setattr(regression, "run_regression_suite", broken)
    runner = TrainingOrchestrator.from_dict(config(tmp_path, suite_file(tmp_path)))
    try:
        with pytest.raises(RuntimeError, match="Capability regression rejected"):
            runner._run_eval_stage()
        gate = runner._stage_results["eval"]["release_gate"]
        assert gate["status"] == "invalid"
        assert gate["accepted"] is False
        assert "missing candidate checkpoint" in gate["error"]
    finally:
        runner._close_logging_handlers()


def test_regression_report_cannot_replace_its_suite(tmp_path, capsys):
    suite = suite_file(tmp_path)
    original = suite.read_bytes()
    assert cli.main([
        "regression", str(suite), "--baseline", "old", "--candidate", "new", "--output", str(suite),
    ]) == 1
    assert suite.read_bytes() == original
    assert "must differ" in capsys.readouterr().out


def test_recovery_dry_run_preserves_sources_and_uses_explicit_validation(tmp_path, monkeypatch, capsys):
    import importlib

    recovery = importlib.import_module("saddlellm.compression.RecoveryTrainer")
    monkeypatch.setattr(recovery, "run_recovery", lambda *a, **k: pytest.fail("dry-run trained"))
    model = tmp_path / "source"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"text":"training example"}\n', encoding="utf-8")
    validation.write_text('{"text":"heldout example"}\n', encoding="utf-8")
    output = tmp_path / "recovered"
    assert cli.main([
        "recover", str(model), str(train), str(output), "--validation-data", str(validation), "--dry-run",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["config"]["validation_split"] == 0
    assert result["model_loaded"] is False
    assert result["qat"] is False
    assert not output.exists()


def test_recovery_report_cannot_replace_training_data(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"training example"}\n', encoding="utf-8")
    original = data.read_bytes()
    assert cli.main([
        "recover", "source", str(data), str(tmp_path / "recovered"), "--output", str(data),
    ]) == 1
    assert data.read_bytes() == original
    assert "must differ" in capsys.readouterr().out


def test_direct_sft_carries_capability_retention_options(monkeypatch):
    seen = {}

    def train(args):
        seen.update(cli._build_direct_train_config(args, ["sft"]))
        return 0

    monkeypatch.setattr(cli, "_cmd_train", train)
    assert cli.main([
        "train", "--stage", "sft", "--model", "local", "--data", "business.jsonl",
        "--reference-model", "original", "--retention-kl-weight", "0.1", "--retention-temperature", "2",
    ]) == 0
    parsed = TrainingOrchestrator._parse_config(seen)
    assert parsed.sft.reference_model_path == "original"
    assert parsed.sft.retention_kl_weight == 0.1
    assert parsed.sft.retention_temperature == 2


def test_offline_cli_sft_prune_recover_quantize_and_regression(tmp_path, monkeypatch):
    """Exercise real command handoffs, serialization, masks and final PTQ evaluation."""
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(23)
    base = tmp_path / "base"
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "task", "general", "answer", "validation", "zero", "one", "two", "three", "four", "five", "user", "assistant", ":", "#", "Question", "Answer"]
    backend = tokenizers.Tokenizer(WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.save_pretrained(base)
    GPT2LMHeadModel(GPT2Config(
        vocab_size=len(words), n_positions=64, n_ctx=64, n_embd=8, n_layer=1, n_head=1,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )).save_pretrained(base)

    def rows(name, values):
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(value) for value in values) + "\n", encoding="utf-8")
        return str(path)

    train = rows("train.jsonl", [{"instruction": f"task {value}", "output": f"answer {value}"} for value in ("one", "two")])
    replay = rows("replay.jsonl", [{"instruction": "general three", "output": "answer three"}])
    validation = rows("validation.jsonl", [{"instruction": "validation four", "output": "answer four"}])
    heldout = rows("heldout.jsonl", [{"instruction": "general five", "output": "answer five"}])
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"tasks": [{
        "name": "general", "task": "completion", "data": heldout,
        "options": {"max_length": 32, "device": "cpu", "local_files_only": True},
        "metrics": {"completion_nll": {"max_drop": 0.5, "higher_is_better": False}},
    }]}), encoding="utf-8")
    run = tmp_path / "sft-run"
    assert cli.main([
        "train", "--stage", "sft", "--model", str(base), "--data", train,
        "--validation-data", validation, "--replay-data", replay, "--replay-count", "1",
        "--retention-kl-weight", "0.1", "--method", "full", "--precision", "fp32",
        "--max-steps", "1", "--batch-size", "1", "--gradient-accumulation-steps", "1",
        "--max-seq-length", "32", "--save-steps", "1", "--eval-steps", "1",
        "--warmup-steps", "0", "--no-gradient-checkpointing", "--local-files-only", "--output-dir", str(run),
    ]) == 0
    trained = run / "sft_checkpoints"
    assert (trained / "replay_manifest.json").is_file()
    assert (trained / "retention_manifest.json").is_file()
    pruned, recovered, quantized = (tmp_path / name for name in ("pruned", "recovered", "quantized"))
    assert cli.main(["prune", str(trained), str(pruned), "--ratio", "0.15", "--target-modules", "c_attn,c_proj", "--local-files-only"]) == 0
    assert cli.main([
        "recover", str(pruned), train, str(recovered), "--teacher", str(trained),
        "--validation-data", validation, "--max-steps", "1", "--max-seq-length", "32",
        "--device", "cpu", "--local-files-only", "--suite", str(suite),
    ]) == 0
    recovery = json.loads((recovered / "recovery_report.json").read_text(encoding="utf-8"))
    assert recovery["actual_steps"] == 1
    assert recovery["sparsity_before"]["preserved_pruned_parameters"] == recovery["sparsity_after"]["preserved_pruned_parameters"]
    assert recovery["sparsity_after"]["pruned_weights_nonzero"] == 0
    assert cli.main(["quantize", str(recovered), str(quantized), "--local-files-only"]) == 0
    report = tmp_path / "final-regression.json"
    assert cli.main([
        "regression", str(suite), "--baseline", str(trained), "--candidate", str(quantized), "--output", str(report),
    ]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["accepted"] is True
    release = tmp_path / "release"
    runner = TrainingOrchestrator.from_dict({
        "model": {"name_or_path": str(recovered)}, "stages": ["eval", "export"],
        "logging": {"output_dir": str(tmp_path / "release-run"), "logging_backend": "local"},
        "eval": {"regression_suite": str(suite), "regression_baseline": str(trained)},
        "export": {"enabled": True, "require_gate": True, "output_dir": str(release), "local_files_only": True},
    })
    try:
        runner.run()
        assert runner._stage_results["eval"]["release_gate"]["accepted"] is True
        assert (release / "release_manifest.json").is_file()
    finally:
        runner._close_logging_handlers()
