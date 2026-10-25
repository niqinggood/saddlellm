import copy
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from saddlellm.decision.Server import DecisionServerSettings, create_decision_app
from saddlellm.decision.TypedModel import TypedDecisionModel
from saddlellm.decision.TypedSchema import (
    TypedDecisionConfig,
    read_typed_dataset,
    validate_record,
)
from saddlellm.decision.TypedTraining import (
    TypedTrainingConfig,
    reinforce_brier_loss,
    supervised_loss,
    train_typed,
)
from tools.prepare_decision_typed import prepare


def rows():
    return [
        {
            "state": "Color: red",
            "question": "Which color?",
            "type": "choice",
            "choices": [
                {"id": "r", "text": "red"},
                {"id": "b", "text": "blue"},
                {"id": "g", "text": "green"},
            ],
        },
        {"state": "Color: red", "question": "Is it red?", "type": "noul"},
        {
            "state": "Color: red",
            "question": "Rate intensity",
            "type": "score",
            "levels": [
                {"id": "low", "text": "low", "value": -1},
                {"id": "high", "text": "high", "value": 5},
            ],
        },
    ]


def cfg(**kw):
    return TypedDecisionConfig(hidden_size=16, num_layers=1, num_heads=2, **kw)


def test_save_recovers_from_windows_directory_lock_and_preserves_weights(tmp_path, monkeypatch):
    model = TypedDecisionModel(cfg()).eval()
    destination = tmp_path / "saved"
    original_rename = Path.rename
    locked = False

    def briefly_locked(path, target):
        nonlocal locked
        if Path(target) == destination and not locked:
            locked = True
            error = PermissionError(13, "temporary Windows directory lock")
            error.winerror = 5
            raise error
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", briefly_locked)
    model.save_pretrained(destination)
    restored = TypedDecisionModel.from_pretrained(destination)
    assert locked
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    assert not list(tmp_path.glob(".typed-save-*"))
    with pytest.raises(FileExistsError):
        model.save_pretrained(destination)


@pytest.mark.parametrize("winerror", [5, None])
def test_save_preserves_directory_permission_errors_without_partial_output(
    tmp_path, monkeypatch, winerror
):
    model = TypedDecisionModel(cfg())
    destination = tmp_path / "saved"
    attempts = []

    def denied(path, target):
        attempts.append(target)
        error = PermissionError(13, "directory access denied")
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(Path, "rename", denied)
    monkeypatch.setattr("saddlellm.decision.TypedModel.time.sleep", lambda _: None)
    with pytest.raises(PermissionError, match="directory access denied"):
        model.save_pretrained(destination)
    assert 1 <= len(attempts) <= 5
    if winerror is None:
        assert len(attempts) == 1
    assert not destination.exists()
    assert not list(tmp_path.glob(".typed-save-*"))


def test_shared_state_and_question_isolation_and_choice_permutation():
    torch.manual_seed(1)
    model = TypedDecisionModel(cfg()).eval()
    calls = []
    hook = model.backbone.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[0])
    )
    with torch.inference_mode():
        together = model(rows())
    hook.remove()
    assert calls == [1, 7]  # One shared state; 3+2+2 query-candidate encodings.
    with torch.inference_mode():
        for row, values in zip(rows(), together):
            torch.testing.assert_close(model([row])[0], values, atol=1e-6, rtol=1e-5)
        permuted = copy.deepcopy(rows()[0])
        permuted["choices"] = [permuted["choices"][i] for i in (2, 0, 1)]
        torch.testing.assert_close(
            model([permuted])[0],
            together[0][torch.tensor([2, 0, 1])],
            atol=1e-6,
            rtol=1e-5,
        )
    predictions = model.predict(rows())
    assert predictions[0]["choice_id"] in {"r", "b", "g"}
    assert 0 <= predictions[1]["probability_true"] <= 1
    assert -1 <= predictions[2]["expected_score"] <= 5
    assert all(
        abs(sum(p["probability"] for p in r["probabilities"]) - 1) < 1e-6
        for r in predictions
    )


def test_joint_and_frozen_backbone_gradients():
    for frozen in (False, True):
        model = TypedDecisionModel(cfg(freeze_backbone=frozen)).train()
        examples = [
            {
                **row,
                "target": [1.0]
                + [0.0] * (len(row.get("choices", row.get("levels", [0, 1]))) - 1),
            }
            for row in rows()
        ]
        before = model.backbone.embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        supervised_loss(model(examples), examples).backward()
        assert (model.backbone.embedding.weight.grad is None) == frozen
        assert all(
            any(p.grad is not None for p in head.parameters())
            for head in model.heads.values()
        )
        optimizer.step()
        assert torch.equal(before, model.backbone.embedding.weight) == frozen


def test_score_loss_uses_ordinal_distance():
    row = {"type": "score", "target": [1.0, 0.0, 0.0]}
    nearby, far = torch.tensor([0.0, 2.0, 0.0]), torch.tensor([0.0, 0.0, 2.0])
    a = supervised_loss([nearby], [row], brier_weight=0, ordinal_weight=1)
    b = supervised_loss([far], [row], brier_weight=0, ordinal_weight=1)
    assert a < b


def test_reinforce_is_score_function_and_improves_brier():
    torch.manual_seed(14)
    logits = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    optimizer = torch.optim.Adam([logits], lr=0.05)
    target = {"target": [0.8, 0.2]}
    before = (logits.softmax(-1) - torch.tensor(target["target"])).square().sum().item()
    for _ in range(80):
        optimizer.zero_grad()
        loss, reward = reinforce_brier_loss([logits], [target], samples=64, std=0.25)
        assert not reward.requires_grad
        loss.backward()
        optimizer.step()
    after = (logits.softmax(-1) - torch.tensor(target["target"])).square().sum().item()
    assert after < before * 0.1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(target_probs={"r": 0.5, "b": 0.4, "g": 0.4}),
        lambda r: r.update(label="absent"),
        lambda r: r["choices"].append(r["choices"][0]),
        lambda r: r.update(type="unknown"),
    ],
)
def test_invalid_records(mutation):
    row = {
        **copy.deepcopy(rows()[0]),
        "id": "a",
        "state_id": "s",
        "task_id": "t",
        "label": "r",
    }
    mutation(row)
    with pytest.raises(ValueError):
        validate_record(row)


def test_leakage_by_group_and_text_and_soft_label_alignment(tmp_path):
    directory = prepare(tmp_path / "data", train_states=12, heldout_states=12)
    _, splits, _ = read_typed_dataset(directory)
    assert any(len(r["target"]) == 3 and max(r["target"]) < 1 for r in splits["train"])
    path = directory / "test.jsonl"
    raw = [json.loads(s) for s in path.read_text().splitlines()]
    for field in ("state_id", "state"):
        changed = copy.deepcopy(raw)
        changed[0][field] = splits["train"][0][field]
        path.write_text("".join(json.dumps(r) + "\n" for r in changed))
        with pytest.raises(ValueError, match="leakage"):
            read_typed_dataset(directory)
    descriptor = json.loads((directory / "dataset.json").read_text())
    descriptor["splits"]["test"] = "../escape.jsonl"
    (directory / "dataset.json").write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="inside"):
        read_typed_dataset(directory)


def test_training_roundtrip_scope_and_http(tmp_path):
    directory = prepare(tmp_path / "data", train_states=6, heldout_states=3)
    run = tmp_path / "run"
    report = train_typed(
        directory,
        run,
        model_config=cfg(),
        device="cpu",
        config=TypedTrainingConfig(
            epochs=2, batch_size=6, learning_rate=0.005, cpu_threads=1
        ),
    )
    assert report["artifact_reload_verified"]
    assert report["backbone_trainable_parameters"] > 0
    assert (
        not report["test_used_for_selection"]
        and not report["calibration_used_for_selection"]
    )
    assert report["test_by_task_scope"]["unseen"]["samples"] == 3
    model = TypedDecisionModel.from_pretrained(run / "model")
    _, splits, _ = read_typed_dataset(directory)
    assert model.predict(splits["test"][:1])[0]["calibration_status"] == "calibrated"
    assert model.predict(splits["test"][3:4])[0]["calibration_status"] == "out_of_scope"
    client = TestClient(
        create_decision_app(
            DecisionServerSettings(
                model_path=str(run / "model"), api_key="test", web_ui=False
            )
        )
    )
    payload = {
        "state": "Color: red",
        "questions": [{k: v for k, v in r.items() if k != "state"} for r in rows()],
    }
    assert client.post("/v1/decisions", json=payload).status_code == 401
    response = client.post(
        "/v1/decisions", json=payload, headers={"Authorization": "Bearer test"}
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 3
    assert client.get("/health").json()["supported_types"] == [
        "Choice",
        "Score",
        "Noul",
    ]
    payload["questions"][2]["levels"][1]["value"] = -2
    assert (
        client.post(
            "/v1/decisions", json=payload, headers={"Authorization": "Bearer test"}
        ).status_code
        == 422
    )
    with pytest.raises(FileExistsError):
        train_typed(directory, run, model_config=cfg())
    weights = run / "model/typed_decision.safetensors"
    with weights.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        TypedDecisionModel.from_pretrained(run / "model")


def test_reinforce_training_and_initial_artifact(tmp_path):
    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    model_config = cfg()
    initial = TypedDecisionModel(model_config).eval()
    initial.save_pretrained(tmp_path / "initial")
    report = train_typed(
        directory,
        tmp_path / "rl",
        model_config=model_config,
        device="cpu",
        initial_model=tmp_path / "initial",
        config=TypedTrainingConfig(
            epochs=1,
            batch_size=3,
            cpu_threads=1,
            method="reinforce_brier",
            supervised_weight=0,
        ),
    )
    assert report["training"]["method"] == "reinforce_brier"
    assert "sampled_negative_brier_reward" in report["selection"]["history"][0]
    assert report["rlcd_reproduction"] is False


def test_hf_joint_training_save_and_offline_reload(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    directory = tmp_path / "backbone"
    tokenizer = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "red": 2, "blue": 3, "color": 4}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    )
    wrapped.save_pretrained(directory)
    BertModel(
        BertConfig(
            vocab_size=5,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    ).save_pretrained(directory)
    config = TypedDecisionConfig(
        backbone="hf", backbone_name_or_path=str(directory), hidden_size=16, num_heads=2
    )
    model = TypedDecisionModel(config).train()
    loss = model(rows())[0].sum()
    loss.backward()
    assert any(p.grad is not None for p in model.backbone.parameters())
    model.eval()
    with torch.inference_mode():
        original = model(rows())
    model.save_pretrained(tmp_path / "saved")
    # Prove reload does not depend on the source directory.
    moved = tmp_path / "relocated_backbone"
    directory.rename(moved)
    restored = TypedDecisionModel.from_pretrained(tmp_path / "saved")
    with torch.inference_mode():
        for a, b in zip(original, restored(rows())):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_no_silent_truncation_before_backbone_call():
    model = TypedDecisionModel(cfg(max_length=16)).eval()
    calls = []
    model.backbone.register_forward_pre_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError, match="no truncation"):
        model(rows())
    assert not calls


def test_cpu_master_optimizer_matches_regular_adamw(tmp_path):
    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    for device in ("model", "cpu"):
        train_typed(
            directory,
            tmp_path / device,
            model_config=cfg(),
            device="cpu",
            config=TypedTrainingConfig(
                epochs=1,
                batch_size=3,
                learning_rate=0.005,
                cpu_threads=1,
                optimizer_device=device,
            ),
        )
    first = TypedDecisionModel.from_pretrained(tmp_path / "model/model")
    second = TypedDecisionModel.from_pretrained(tmp_path / "cpu/model")
    for key, values in first.state_dict().items():
        torch.testing.assert_close(values, second.state_dict()[key], rtol=0, atol=0)


def test_typed_cli_infers_dataset_and_uses_joint_training_defaults(tmp_path, capsys):
    from saddlellm import cli

    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    assert cli.main(["train", "decision", "--data", str(directory), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)["decision"]
    assert plan["type"] == "Typed"
    assert plan["settings"]["freeze_backbone"] is False
    assert plan["settings"]["batch_size"] == 16
    assert plan["settings"]["learning_rate"] == 0.001
    assert plan["settings"]["epochs"] == 20


def test_hf_defaults_do_not_inherit_head_hyperparameters(tmp_path):
    from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator

    path = tmp_path / "hf"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"hidden_size": 1024}))
    parsed = TrainingOrchestrator._parse_config(
        {
            "stages": ["decision"],
            "model": {"name_or_path": str(path)},
            "decision": {
                "task_type": "typed",
                "typed_backbone": "hf",
                "dataset_dir": "data",
            },
        }
    )
    assert parsed.decision.hidden_size == 1024
    assert parsed.decision.learning_rate == 0.00002
    assert parsed.decision.batch_size == 1
    assert parsed.decision.epochs == 3
    assert parsed.decision.freeze_backbone is False
    overridden = TrainingOrchestrator._parse_config(
        {
            "stages": ["decision"],
            "model": {"name_or_path": str(path)},
            "decision": {
                "task_type": "typed",
                "typed_backbone": "hf",
                "dataset_dir": "data",
                "learning_rate": 0.00001,
                "batch_size": 2,
                "freeze_backbone": True,
            },
        }
    )
    assert (
        overridden.decision.learning_rate == 0.00001
        and overridden.decision.batch_size == 2
    )
    assert overridden.decision.freeze_backbone is True


def test_typed_cli_simplified_command_trains_and_registers_complete_model(
    tmp_path, capsys
):
    from saddlellm import cli

    directory = prepare(tmp_path / "data", train_states=3, heldout_states=3)
    run = tmp_path / "run"
    assert (
        cli.main(
            [
                "train",
                "decision",
                "--data",
                str(directory),
                "--output-dir",
                str(run),
                "--epochs",
                "1",
                "--decision-hidden-size",
                "16",
                "--decision-heads",
                "2",
                "--batch-size",
                "3",
                "--cpu-threads",
                "1",
                "--encoder-device",
                "cpu",
                "--decision-method",
                "reinforce_brier",
                "--optimizer-device",
                "cpu",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    stage = result["results"]["decision"]
    assert stage["method"] == "reinforce_brier" and stage["backbone_bundled"] is True
    assert cli.main(["serve-decision", stage["model_path"], "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["supported_types"] == [
        "Choice",
        "Score",
        "Noul",
    ]


def test_typed_dry_run_does_not_import_tensor_libraries():
    import subprocess
    import sys

    script = "from saddlellm.cli import main; import sys; assert main(['train','decision','--decision-type','typed','--data','fake','--dry-run'])==0; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    run = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert run.returncode == 0, run.stderr
