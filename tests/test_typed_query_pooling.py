"""Behavior checks for typed query pooling and existing artifact compatibility."""

import copy
import json

import pytest
import torch

from saddlellm.decision.dataset import sha256
from saddlellm.decision.TypedModel import TypedDecisionModel
from saddlellm.decision.TypedSchema import TypedDecisionConfig
from saddlellm.decision.TypedTraining import supervised_loss


def config(**values):
    return TypedDecisionConfig(hidden_size=16, num_layers=1, num_heads=2, **values)


def records():
    return [
        {
            "state": "The color is red.",
            "question": "Which color?",
            "type": "choice",
            "choices": [
                {"id": "r", "text": "red"},
                {"id": "b", "text": "blue"},
                {"id": "g", "text": "green"},
            ],
        },
        {
            "state": "The color is red.",
            "question": "Is the color red?",
            "type": "noul",
        },
        {
            "state": "The review describes excellent service and food.",
            "question": "Which rating?",
            "type": "score",
            "levels": [
                {"id": "1", "text": "1 star", "value": 1},
                {"id": "5", "text": "5 stars", "value": 5},
            ],
        },
    ]


@pytest.mark.parametrize("value", ["first_token", "", None, 1, False, []])
def test_query_pooling_rejects_unsupported_values(value):
    with pytest.raises(ValueError, match="query_pooling"):
        config(query_pooling=value)


@pytest.mark.parametrize("pooling", ["mean", "last_token"])
def test_pooling_respects_right_left_and_mixed_padding(pooling):
    model = TypedDecisionModel(config(query_pooling=pooling))
    hidden = torch.tensor(
        [
            [[2.0, 20.0], [6.0, 60.0], [1000.0, 1000.0], [2000.0, 2000.0]],
            [[1000.0, 1000.0], [2000.0, 2000.0], [2.0, 20.0], [6.0, 60.0]],
            [[1000.0, 1000.0], [2.0, 20.0], [6.0, 60.0], [2000.0, 2000.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [[True, True, False, False], [False, False, True, True], [False, True, True, False]]
    )
    pooled = model._pool_query(hidden, mask)
    expected = [4.0, 40.0] if pooling == "mean" else [6.0, 60.0]
    torch.testing.assert_close(pooled, torch.tensor([expected] * 3), rtol=0, atol=0)
    pooled.sum().backward()
    assert torch.count_nonzero(hidden.grad[~mask]) == 0
    if pooling == "last_token":
        # Only final valid tokens receive the gather's gradient, independently
        # of whether the sequence is padded on the left or right.
        expected_grad = torch.zeros_like(hidden)
        expected_grad[0, 1] = 1
        expected_grad[1, 3] = 1
        expected_grad[2, 2] = 1
        torch.testing.assert_close(hidden.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("pooling", ["mean", "last_token"])
def test_query_pooling_preserves_choice_permutation_and_full_state_sequence(pooling):
    torch.manual_seed(19)
    model = TypedDecisionModel(config(query_pooling=pooling)).eval()
    attention_shapes = []
    hook = model.cross_attention.register_forward_pre_hook(
        lambda module, args: attention_shapes.append((args[0].shape, args[1].shape))
    )
    examples = records()
    with torch.inference_mode():
        original = model(examples)
    hook.remove()
    query_shape, state_shape = attention_shapes[0]
    assert query_shape == (7, 1, 16)
    assert state_shape == (7, 1 + len(examples[2]["state"].encode("utf-8")), 16)
    permuted = copy.deepcopy(examples[0])
    order = [2, 0, 1]
    permuted["choices"] = [permuted["choices"][index] for index in order]
    with torch.inference_mode():
        torch.testing.assert_close(
            model([permuted])[0], original[0][order], rtol=1e-5, atol=1e-6
        )
        for row, logits in zip(examples, original):
            torch.testing.assert_close(model([row])[0], logits, rtol=1e-5, atol=1e-6)


def test_changing_pooling_only_changes_query_summary_with_identical_weights():
    torch.manual_seed(21)
    mean = TypedDecisionModel(config()).eval()
    last = TypedDecisionModel(config(query_pooling="last_token")).eval()
    last.load_state_dict(mean.state_dict())
    encoded, attention, outputs = [], [], []
    for model in (mean, last):
        hidden, inputs = [], []
        encoder_hook = model.backbone.register_forward_hook(
            lambda module, args, output, captured=hidden: captured.append(output.detach().clone())
        )
        attention_hook = model.cross_attention.register_forward_pre_hook(
            lambda module, args, captured=inputs: captured.append(
                (args[0].detach().clone(), args[1].detach().clone())
            )
        )
        with torch.inference_mode():
            outputs.append(torch.cat(model(records())))
        encoder_hook.remove()
        attention_hook.remove()
        encoded.append(hidden)
        attention.append(inputs[0])
    # Both full state and full query encodings are unchanged. Only the query
    # summary supplied to cross-attention changes, and it changes real logits.
    for before, after in zip(encoded[0], encoded[1]):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(attention[0][1], attention[1][1], rtol=0, atol=0)
    assert not torch.equal(attention[0][0], attention[1][0])
    assert not torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("pooling", ["mean", "last_token"])
def test_pooling_model_backpropagates_through_backbone_attention_and_every_head(pooling):
    torch.manual_seed(23)
    model = TypedDecisionModel(config(query_pooling=pooling)).train()
    examples = [
        {**row, "target": [1.0, 0.0, 0.0] if row["type"] == "choice" else [0.0, 1.0]}
        for row in records()
    ]
    loss = supervised_loss(model(examples), examples)
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


@pytest.mark.parametrize("pooling", ["mean", "last_token"])
def test_pooling_artifact_round_trip_preserves_logits_exactly(tmp_path, pooling):
    torch.manual_seed(29)
    model = TypedDecisionModel(config(query_pooling=pooling)).eval()
    with torch.inference_mode():
        expected = model(records())
    destination = tmp_path / "saved"
    model.save_pretrained(destination)
    payload = json.loads((destination / "decision_config.json").read_text(encoding="utf-8"))
    assert payload["config"]["query_pooling"] == pooling
    restored = TypedDecisionModel.from_pretrained(destination)
    assert restored.config == model.config
    with torch.inference_mode():
        for before, after in zip(expected, restored(records())):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert restored.predict(records()) == model.predict(records())


def test_legacy_artifact_without_pooling_field_keeps_mean_and_exact_outputs(tmp_path):
    torch.manual_seed(31)
    model = TypedDecisionModel(config()).eval()
    assert model.config.query_pooling == "mean"
    with torch.inference_mode():
        expected = model(records())
    destination = tmp_path / "legacy"
    model.save_pretrained(destination)
    config_path = destination / "decision_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    del payload["config"]["query_pooling"]
    config_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decision_config.json"] = sha256(config_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    restored = TypedDecisionModel.from_pretrained(destination)
    assert restored.config.query_pooling == "mean"
    with torch.inference_mode():
        for before, after in zip(expected, restored(records())):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
