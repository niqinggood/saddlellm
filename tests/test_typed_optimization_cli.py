import json

import pytest

from saddlellm import cli
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator
from tools.prepare_decision_typed import prepare


def test_direct_optimization_controls_reach_the_plan(tmp_path, capsys):
    data = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    assert cli.main([
        "train", "decision", "--data", str(data), "--dry-run",
        "--query-pooling", "last_token", "--backbone-learning-rate", ".00001",
        "--instance-question-in-query",
        "--head-learning-rate", ".0002", "--head-warmup-epochs", "1",
        "--early-stopping-patience", "2", "--no-evaluate-test",
    ]) == 0
    plan = json.loads(capsys.readouterr().out)["decision"]
    assert plan["model_config"]["query_pooling"] == "last_token"
    assert plan["model_config"]["instance_question_in_query"] is True
    assert plan["settings"]["head_learning_rate"] == .0002
    assert plan["settings"]["backbone_learning_rate"] == .00001
    assert plan["settings"]["head_warmup_epochs"] == 1
    assert plan["settings"]["early_stopping_patience"] == 2
    assert plan["settings"]["evaluate_test"] is False


def test_config_and_false_direct_flag_cannot_be_mixed(tmp_path, capsys):
    data = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"stages": ["decision"], "decision": {
        "task_type": "typed", "dataset_dir": str(data)}, "eval": {"enabled": False}}))
    assert cli.main(["train", str(config), "--no-evaluate-test", "--dry-run"]) == 1
    assert "either" in json.loads(capsys.readouterr().out)["error"].lower()


@pytest.mark.parametrize("field,value", [
    ("head_learning_rate", 0), ("backbone_learning_rate", True),
    ("head_learning_rate", float("nan")), ("head_warmup_epochs", -1),
    ("early_stopping_patience", True), ("evaluate_test", "false"),
    ("query_pooling", "arbitrary"),
    ("instance_question_in_query", "yes"),
])
def test_invalid_controls_fail_before_creating_output(tmp_path, field, value):
    data = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    raw = {"stages": ["decision"], "decision": {"task_type": "typed",
        "dataset_dir": str(data), field: value}, "eval": {"enabled": False},
        "logging": {"output_dir": str(tmp_path / "output")}}
    report = TrainingConfigValidator.validate(raw, inspect_data=False)
    assert not report.valid
    assert not (tmp_path / "output").exists()


def test_direct_cli_without_test_registers_validation_metrics(tmp_path, capsys):
    data = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    assert cli.main([
        "train", "decision", "--data", str(data), "--output-dir", str(tmp_path / "run"),
        "--decision-hidden-size", "16", "--decision-heads", "2", "--max-seq-length", "256",
        "--epochs", "1", "--batch-size", "3", "--cpu-threads", "1", "--encoder-device", "cpu",
        "--query-pooling", "last_token", "--head-learning-rate", ".001", "--no-evaluate-test",
    ]) == 0
    result = json.loads(capsys.readouterr().out)["results"]["decision"]
    assert result["metrics_split"] == "validation"
    assert result["test_evaluated"] is False
    assert result["metrics"]["nll"] >= 0


def test_choice_contract_rejects_typed_optimization_flags(capsys):
    assert cli.main(["train", "decision", "--data", "fake", "--model", "fake",
                     "--head-learning-rate", ".001", "--dry-run"]) == 1
    assert "typed" in json.loads(capsys.readouterr().out)["error"].lower()


def test_native_evaluation_command_writes_requested_validation_only(tmp_path, capsys):
    from saddlellm.decision.TypedModel import TypedDecisionModel
    from saddlellm.decision.TypedSchema import TypedDecisionConfig

    data = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    model = TypedDecisionModel(TypedDecisionConfig(hidden_size=16, num_heads=2, num_layers=1)).eval()
    saved = tmp_path / "model"
    model.save_pretrained(saved)
    out = tmp_path / "evaluation"
    assert cli.main(["evaluate-decision", "--model", str(saved), "--data", str(data),
                     "--output-dir", str(out), "--splits", "validation", "--device", "cpu",
                     "--batch-size", "4", "--cpu-threads", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["command"] == "evaluate-decision"
    assert set(result["splits"]) == {"validation"}
    report = json.loads((out / "report.json").read_text())
    assert set(report["splits"]) == {"validation"}
    preflight = json.loads((out / "token_preflight.json").read_text())
    assert preflight["all_four_splits_checked"] is True
    reference_rows = [json.loads(line) for line in (out / "logits.jsonl").read_text().splitlines()]
    reference_rows[0]["logits"][0] += 1
    reference = tmp_path / "incorrect_reference.jsonl"
    reference.write_text("".join(json.dumps(row) + "\n" for row in reference_rows))
    assert cli.main(["evaluate-decision", "--model", str(saved), "--data", str(data),
                     "--output-dir", str(tmp_path / "mismatch"), "--splits", "validation",
                     "--device", "cpu", "--cpu-threads", "1", "--logits-reference", str(reference)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
