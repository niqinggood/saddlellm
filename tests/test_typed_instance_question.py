"""Variable instance input, fixed task scope, and isolated candidate queries."""

import copy
import json

import pytest
import torch
from fastapi.testclient import TestClient

from saddlellm.decision.dataset import sha256
from saddlellm.decision.Server import DecisionServerSettings
from saddlellm.decision.TypedModel import TypedDecisionModel
from saddlellm.decision.TypedSchema import (
    TypedDecisionConfig,
    candidate_query_text,
    candidates,
    read_typed_dataset,
    scope_key,
    validate_question,
    validate_record,
)
from saddlellm.decision.TypedServer import create_typed_decision_app
from saddlellm.decision.TypedTraining import supervised_loss


def cfg(**values):
    return TypedDecisionConfig(hidden_size=16, num_layers=1, num_heads=2, **values)


def examples():
    return [
        {
            "state": "The sky is blue and the review is excellent.",
            "question": "Choose a color.",
            "instance_question": "Which color is the sky?",
            "type": "choice",
            "choices": [{"id": "b", "text": "blue"}, {"id": "r", "text": "red"}],
        },
        {
            "state": "The sky is blue and the review is excellent.",
            "question": "Answer the instance question.",
            "instance_question": "Is the sky blue?",
            "type": "noul",
        },
        {
            "state": "The sky is blue and the review is excellent.",
            "question": "Rate the review.",
            "instance_question": "How good is the service?",
            "type": "score",
            "levels": [
                {"id": "1", "text": "1 star", "value": 1},
                {"id": "5", "text": "5 stars", "value": 5},
            ],
        },
    ]


@pytest.mark.parametrize("kind_index", range(3))
def test_instance_input_survives_question_and_record_validation_without_changing_target(
    kind_index,
):
    row = examples()[kind_index]
    question = {key: value for key, value in row.items() if key != "state"}
    assert validate_question(question)["instance_question"] == row["instance_question"]
    assert (
        validate_record(row, require_target=False)["instance_question"]
        == row["instance_question"]
    )
    labeled = {
        **row,
        "id": "row",
        "state_id": "state",
        "task_id": "task",
        "label": candidates(row)[0]["id"],
    }
    checked = validate_record(labeled)
    changed = validate_record(
        {**labeled, "instance_question": "A different input question?"}
    )
    assert checked["target"] == changed["target"] == [1.0, 0.0]
    assert checked["question"] == changed["question"]
    assert scope_key(checked) == scope_key(changed)


@pytest.mark.parametrize(
    "value", [None, True, 123, 1.5, [], {}, "", " \n\t ", "a" * 4097]
)
def test_instance_question_rejects_nonstring_blank_or_oversized_input(value):
    row = {**examples()[1], "instance_question": value}
    with pytest.raises(ValueError, match="instance_question"):
        validate_record(row, require_target=False)
    with pytest.raises(ValueError, match="instance_question"):
        validate_question({key: item for key, item in row.items() if key != "state"})


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_instance_query_config_requires_a_boolean(value):
    with pytest.raises(TypeError, match="instance_question_in_query"):
        cfg(instance_question_in_query=value)


@pytest.mark.parametrize("kind_index", range(3))
def test_shared_query_helper_keeps_exact_legacy_template_and_optional_instance_line(
    kind_index,
):
    row = examples()[kind_index]
    for option in candidates(row):
        level = f"\nValue: {option['value']}" if row["type"] == "score" else ""
        legacy = f"Type: {row['type']}\nQuestion: {row['question']}\nOption: {option['text']}{level}"
        assert candidate_query_text(row, option).encode("utf-8") == legacy.encode(
            "utf-8"
        )
        assert (
            candidate_query_text(row, option, instance_question_in_query=False)
            == legacy
        )
        enabled = f"Type: {row['type']}\nQuestion: {row['question']}\nInstance question: {row['instance_question']}\nOption: {option['text']}{level}"
        assert (
            candidate_query_text(row, option, instance_question_in_query=True)
            == enabled
        )
        without_instance = {
            key: value for key, value in row.items() if key != "instance_question"
        }
        assert (
            candidate_query_text(
                without_instance, option, instance_question_in_query=True
            )
            == legacy
        )
        # Candidate IDs remain output handles, even with instance input enabled.
        assert (
            candidate_query_text(
                row, {**option, "id": "different-id"}, instance_question_in_query=True
            )
            == enabled
        )


def write_dataset(directory):
    directory.mkdir()
    descriptor = {
        "schema_version": 2,
        "task_type": "typed",
        "splits": {
            split: f"{split}.jsonl"
            for split in ("train", "validation", "calibration", "test")
        },
    }
    (directory / "dataset.json").write_text(json.dumps(descriptor), encoding="utf-8")
    for index, split in enumerate(descriptor["splits"]):
        row = {
            **examples()[0],
            "id": split,
            "state_id": split,
            "task_id": "same_task",
            "state": f"Distinct passage {index}.",
            "instance_question": f"Different instance {index}?",
            "label": "b",
        }
        (directory / f"{split}.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
    return directory


@pytest.mark.parametrize("changed_scope", [None, "question", "candidate"])
def test_dataset_task_scope_excludes_instance_input_but_still_enforces_fixed_semantics(
    tmp_path, changed_scope
):
    directory = write_dataset(tmp_path / "data")
    _, splits, _ = read_typed_dataset(directory)
    assert len({scope_key(rows[0]) for rows in splits.values()}) == 1
    assert len({rows[0]["instance_question"] for rows in splits.values()}) == 4
    if changed_scope is None:
        return
    path = directory / "validation.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    if changed_scope == "question":
        row["question"] = "A different fixed task?"
    else:
        row["choices"][0]["text"] = "A different candidate meaning"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one task_id"):
        read_typed_dataset(directory)


def test_disabled_instance_input_and_enabled_missing_field_keep_exact_legacy_logits():
    torch.manual_seed(37)
    legacy = TypedDecisionModel(cfg()).eval()
    enabled = TypedDecisionModel(cfg(instance_question_in_query=True)).eval()
    enabled.load_state_dict(legacy.state_dict())
    without = [
        {key: value for key, value in row.items() if key != "instance_question"}
        for row in examples()
    ]
    with torch.inference_mode():
        original = legacy(without)
        for expected, ignored, absent in zip(
            original, legacy(examples()), enabled(without)
        ):
            torch.testing.assert_close(ignored, expected, rtol=0, atol=0)
            torch.testing.assert_close(absent, expected, rtol=0, atol=0)
    assert legacy.predict(examples()) == legacy.predict(without)


@pytest.mark.parametrize("pooling", ["mean", "last_token"])
def test_enabled_instance_query_changes_only_its_own_logits_and_preserves_permutation(
    monkeypatch, pooling
):
    torch.manual_seed(41)
    model = TypedDecisionModel(
        cfg(instance_question_in_query=True, query_pooling=pooling)
    ).eval()
    rows = examples()
    captured = []
    original_tokens = model._tokens

    def capture(texts):
        captured.append(list(texts))
        return original_tokens(texts)

    monkeypatch.setattr(model, "_tokens", capture)
    with torch.inference_mode():
        baseline = model(rows)
    assert captured[0] == [rows[0]["state"]]  # Only one shared, unchanged state.
    assert captured[1] == [
        candidate_query_text(row, option, instance_question_in_query=True)
        for row in rows
        for option in candidates(row)
    ]
    changed = copy.deepcopy(rows)
    changed[1]["instance_question"] = (
        "Is the sky pink?"  # Same length; same tensor shapes.
    )
    with torch.inference_mode():
        new_logits = model(changed)
        assert not torch.equal(new_logits[1], baseline[1])
        for index in (0, 2):
            torch.testing.assert_close(
                new_logits[index], baseline[index], rtol=0, atol=0
            )
        for row, expected in zip(rows, baseline):
            torch.testing.assert_close(model([row])[0], expected, rtol=1e-5, atol=1e-6)
        permuted = copy.deepcopy(rows[0])
        permuted["choices"].reverse()
        torch.testing.assert_close(
            model([permuted])[0], baseline[0].flip(0), rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("pooling", ["mean", "last_token"])
def test_enabled_instance_tokens_and_every_head_receive_finite_gradients(pooling):
    torch.manual_seed(43)
    model = TypedDecisionModel(
        cfg(instance_question_in_query=True, query_pooling=pooling)
    ).train()
    rows = [
        {**row, "instance_question": "Zebra input?", "target": [1.0, 0.0]}
        for row in examples()
    ]
    loss = supervised_loss(model(rows), rows)
    loss.backward()
    parameters = [
        model.backbone.embedding.weight,
        model.cross_attention.in_proj_weight,
        *(head[0].weight for head in model.heads.values()),
    ]
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in parameters
    )
    # Capital Z appears only in the new instance input, not state/fixed question/options.
    assert torch.count_nonzero(model.backbone.embedding.weight.grad[ord("Z") + 1]) > 0


@pytest.mark.parametrize("enabled", [False, True])
def test_instance_query_artifact_round_trip_preserves_config_and_logits_exactly(
    tmp_path, enabled
):
    torch.manual_seed(47)
    model = TypedDecisionModel(cfg(instance_question_in_query=enabled)).eval()
    model.calibration[scope_key(examples()[1])] = {"temperature": 2.0}
    with torch.inference_mode():
        expected = model(examples())
    destination = tmp_path / "saved"
    model.save_pretrained(destination)
    restored = TypedDecisionModel.from_pretrained(destination)
    assert restored.config == model.config
    assert restored.config.instance_question_in_query is enabled
    with torch.inference_mode():
        for before, after in zip(expected, restored(examples())):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert restored.predict(examples()) == model.predict(examples())
    changed = {**examples()[1], "instance_question": "A different instance question?"}
    assert restored.predict([changed])[0]["calibration_status"] == "calibrated"
    assert restored.predict([changed])[0]["temperature_used"] == 2.0


def test_legacy_artifact_missing_instance_query_flag_defaults_false_and_keeps_exact_logits(
    tmp_path,
):
    torch.manual_seed(53)
    model = TypedDecisionModel(cfg()).eval()
    with torch.inference_mode():
        expected = model(examples())
    destination = tmp_path / "legacy"
    model.save_pretrained(destination)
    config_path = destination / "decision_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    del payload["config"]["instance_question_in_query"]
    config_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decision_config.json"] = sha256(config_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    restored = TypedDecisionModel.from_pretrained(destination)
    assert restored.config.instance_question_in_query is False
    with torch.inference_mode():
        for before, after in zip(expected, restored(examples())):
            torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_instance_query_length_is_rejected_before_any_backbone_forward():
    row = examples()[1]
    max_length = max(
        len(candidate_query_text(row, option).encode("utf-8")) + 1
        for option in candidates(row)
    )
    model = TypedDecisionModel(
        cfg(max_length=max_length, instance_question_in_query=True)
    ).eval()
    calls = []
    model.backbone.register_forward_pre_hook(lambda *args: calls.append(True))
    with pytest.raises(ValueError, match="no truncation"):
        model([row])
    assert not calls


def test_http_multiple_instances_are_preserved_and_isolated(monkeypatch):
    torch.manual_seed(59)
    model = TypedDecisionModel(cfg(instance_question_in_query=True)).eval()
    client = TestClient(
        create_typed_decision_app(
            DecisionServerSettings(model_path="injected", web_ui=False), model=model
        )
    )
    row = examples()[1]
    questions = [
        {key: value for key, value in row.items() if key != "state"} for _ in range(2)
    ]
    questions[0].update(id="first", instance_question="Is the sky blue?")
    questions[1].update(id="second", instance_question="Is the sky gray?")
    payload = {"state": row["state"], "questions": questions}
    captured = []
    original_predict = model.predict

    def capture(rows):
        captured.append(copy.deepcopy(rows))
        return original_predict(rows)

    monkeypatch.setattr(model, "predict", capture)
    response = client.post("/v1/decisions", json=payload)
    assert response.status_code == 200, response.text
    assert [item["instance_question"] for item in captured[0]] == [
        item["instance_question"] for item in questions
    ]
    assert [item["id"] for item in response.json()["data"]] == ["first", "second"]
    expected = original_predict([{**item, "state": row["state"]} for item in questions])
    assert response.json()["data"] == expected
    changed = copy.deepcopy(payload)
    changed["questions"][0]["instance_question"] = "Is the sky pink?"
    another = client.post("/v1/decisions", json=changed)
    assert another.status_code == 200
    assert another.json()["data"][0] != response.json()["data"][0]
    assert another.json()["data"][1] == response.json()["data"][1]
    assert "instance_question" in client.get("/v1/decisions/schema").json()["input"]


@pytest.mark.parametrize("value", [None, True, "", " \n ", "x" * 33])
def test_http_rejects_invalid_instance_question_and_applies_question_limit(value):
    model = TypedDecisionModel(cfg(instance_question_in_query=True)).eval()
    settings = DecisionServerSettings(
        model_path="injected", max_question_chars=32, web_ui=False
    )
    client = TestClient(create_typed_decision_app(settings, model=model))
    question = {
        "type": "noul",
        "question": "Answer the question.",
        "instance_question": value,
    }
    response = client.post(
        "/v1/decisions", json={"state": "A passage.", "questions": [question]}
    )
    assert response.status_code == 422, response.text


def test_http_counts_instance_input_toward_total_request_limit():
    model = TypedDecisionModel(cfg(instance_question_in_query=True)).eval()
    settings = DecisionServerSettings(
        model_path="injected", max_total_chars=40, web_ui=False
    )
    client = TestClient(create_typed_decision_app(settings, model=model))
    payload = {
        "state": "A passage.",
        "questions": [{"type": "noul", "question": "Answer the question."}],
    }
    assert client.post("/v1/decisions", json=payload).status_code == 200
    payload["questions"][0]["instance_question"] = "x" * 20
    response = client.post("/v1/decisions", json=payload)
    assert response.status_code == 422
    assert "total character limit" in response.text
