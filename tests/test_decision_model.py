from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from saddlellm.decision.Model import (
    ChoiceDecisionModel,
    DecisionModelConfig,
    FrozenTextEncoder,
    last_valid_token_pool,
)


def config(**kwargs):
    return DecisionModelConfig(
        **{"backbone_name_or_path": "offline-fake", "embedding_dim": 4,
           "projection_rank": 2, **kwargs}
    )


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return self.vector.unsqueeze(0).expand(len(texts), -1)


class TinyTokenizer:
    def __init__(self, padding_side="left"):
        self.padding_side = padding_side

    def __call__(self, texts, *, padding, truncation, add_special_tokens):
        assert padding is False and truncation is False and add_special_tokens is True
        ids = [[index + 1 for index in range(len(text))] for text in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    def pad(self, rows, *, padding, return_tensors):
        assert padding is True and return_tensors == "pt"
        length = max(map(len, rows["input_ids"]))
        ids, masks = [], []
        for row in rows["input_ids"]:
            extra = length - len(row)
            ids.append(([0] * extra + row) if self.padding_side == "left" else (row + [0] * extra))
            masks.append(([0] * extra + [1] * len(row)) if self.padding_side == "left" else ([1] * len(row) + [0] * extra))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(16, 4)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(64).reshape(16, 4).float() + 1)
        self.calls = 0

    def forward(self, input_ids, attention_mask, *, use_cache, return_dict):
        assert use_cache is False and return_dict is True
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_last_valid_pool_handles_both_padding_sides_and_interior_gaps():
    hidden = torch.arange(3 * 4 * 2).reshape(3, 4, 2).float()
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0]])
    torch.testing.assert_close(last_valid_token_pool(hidden, mask), hidden[torch.arange(3), torch.tensor([3, 1, 2])])
    with pytest.raises(ValueError, match="at least one"):
        last_valid_token_pool(hidden[:1], torch.zeros(1, 4))
    with pytest.raises(ValueError, match="zeros and ones"):
        last_valid_token_pool(hidden[:1], torch.tensor([[1, 1, 1, 2]]))


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_auto_model_adapter_pools_without_generate_or_vocabulary_logits(padding_side):
    backbone = TinyBackbone()
    encoder = FrozenTextEncoder(config(), model=backbone, tokenizer=TinyTokenizer(padding_side))
    result = encoder.encode(["abc", "a"])
    expected = F.normalize(backbone.embedding.weight[torch.tensor([3, 1])], dim=-1)
    torch.testing.assert_close(result, expected)
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert not result.requires_grad
    assert all(not p.requires_grad for p in backbone.parameters())
    assert backbone.calls == 1


def test_overlength_rejected_before_any_encoder_batch_runs():
    backbone = TinyBackbone()
    encoder = FrozenTextEncoder(config(max_length=3, encoder_batch_size=1), model=backbone, tokenizer=TinyTokenizer())
    with pytest.raises(ValueError, match="no truncation"):
        encoder.encode(["a", "abcd"])
    assert backbone.calls == 0


def test_encoder_width_must_match_manifest():
    with pytest.raises(ValueError, match="hidden_size"):
        FrozenTextEncoder(config(embedding_dim=8), model=TinyBackbone(), tokenizer=TinyTokenizer())


def test_encode_api_templates_normalization_and_frozen_encoder():
    encoder = FakeEncoder()
    model = ChoiceDecisionModel(config(), encoder=encoder)
    model.train()
    states = model.encode_states(["first", "second"], "Route this input")
    choices = model.encode_choices(["first kind", "second kind"], "Route this input")
    assert states.shape == choices.shape == (2, 4)
    assert states.dtype == torch.float32 and not states.requires_grad
    torch.testing.assert_close(states.norm(dim=-1), torch.ones(2))
    assert encoder.calls[0] == ["Instruct: Route this input\nQuery: first", "Instruct: Route this input\nQuery: second"]
    assert encoder.calls[1][0] == "Question: Route this input\nOption: first kind"
    assert not encoder.training and not encoder.vector.requires_grad
    model.to(dtype=torch.float64)
    assert encoder.vector.dtype == torch.float32
    assert all(name.startswith("head.") for name in model.state_dict())
    assert set(map(id, model.parameters())) == set(map(id, model.head_parameters()))


def test_lazy_head_training_never_loads_backbone(monkeypatch):
    module = importlib.import_module("saddlellm.decision.Model")
    monkeypatch.setattr(module, "FrozenTextEncoder", lambda *a, **k: pytest.fail("backbone unexpectedly loaded"))
    model = ChoiceDecisionModel(config())
    assert not model.encoder_loaded
    loss = F.cross_entropy(model(torch.eye(4)[:2], torch.eye(4)), torch.tensor([0, 1]))
    loss.backward()
    assert model.head.up.weight.grad is not None
    assert not model.encoder_loaded


def test_zero_residual_preserves_cosine_baseline_and_candidate_permutation():
    torch.manual_seed(24)
    model = ChoiceDecisionModel(config(initial_scale=7.0))
    states, choices = torch.randn(3, 4), torch.randn(5, 4)
    expected = 7.0 * (F.normalize(states, dim=-1) @ F.normalize(choices, dim=-1).T)
    torch.testing.assert_close(model.score(states, choices), expected)
    order = torch.tensor([3, 0, 4, 1, 2])
    torch.testing.assert_close(model.score(states, choices[order]), expected[:, order])
    assert model.calibration == {"temperature": 1.0, "calibrated": False, "metadata": {}}


def test_batched_variable_candidate_mask_matches_individual_requests():
    model = ChoiceDecisionModel(config())
    states = torch.eye(4)[:2]
    candidates = torch.stack([torch.eye(4)[:3], torch.eye(4)[:3]])
    candidates[1, 2] = 0  # A padded slot is allowed to have a zero embedding.
    mask = torch.tensor([[True, True, True], [True, True, False]])
    scores = model.score(states, candidates, mask)
    torch.testing.assert_close(scores[0], model.score(states[:1], candidates[0])[0])
    torch.testing.assert_close(scores[1, :2], model.score(states[1:], candidates[1, :2])[0])
    assert torch.isneginf(scores[1, 2])
    probabilities = model.predict_proba(states, candidates, mask)
    assert probabilities[1, 2].item() == 0
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))


def test_shared_candidate_mask_and_calibration_temperature():
    model = ChoiceDecisionModel(config())
    states, choices = torch.eye(4)[:2], torch.eye(4)
    mask = torch.tensor([True, True, False, True])
    logits = model.score(states, choices, mask)
    model.set_calibration(2.5, {"split": "calibration-only", "scope": {"question": "q"}})
    torch.testing.assert_close(model.predict_proba(states, choices, mask), torch.softmax(logits / 2.5, dim=-1))
    copied = model.calibration
    copied["metadata"]["scope"]["question"] = "changed"
    assert model.calibration["metadata"]["scope"]["question"] == "q"
    assert "temperature" not in " ".join(model.state_dict())
    assert not any(name.startswith("calibration") for name, _ in model.named_parameters())


def test_cached_feature_training_can_learn_a_changed_candidate_mapping():
    torch.manual_seed(7)
    model = ChoiceDecisionModel(config(projection_rank=4, initial_scale=3.0))
    states = choices = torch.eye(4)
    targets = torch.tensor([1, 2, 3, 0])
    optimizer = torch.optim.Adam(model.head_parameters(), lr=0.05)
    before = F.cross_entropy(model.score(states, choices), targets).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = F.cross_entropy(model.score(states, choices), targets)
        loss.backward()
        optimizer.step()
    after_logits = model.score(states, choices)
    assert F.cross_entropy(after_logits, targets).item() < before * 0.1
    assert torch.equal(after_logits.argmax(dim=-1), targets)
    assert not model.encoder_loaded


def test_head_calibration_roundtrip_and_identity_check(tmp_path):
    model = ChoiceDecisionModel(config(backbone_identity={"revision": "immutable", "config_sha256": "abc"}), encoder=FakeEncoder())
    with torch.no_grad():
        model.head.up.weight.fill_(0.01)
    model.set_calibration(1.5, {"scope": {"question": "route", "choices": [["a", "A"], ["b", "B"]]}})
    model.save_pretrained(tmp_path, metadata={"question": "route", "choices": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]})
    restored = ChoiceDecisionModel.from_pretrained(tmp_path, expected_backbone_identity=model.config.backbone_identity)
    states, choices = torch.eye(4)[:2], torch.eye(4)
    torch.testing.assert_close(restored.predict_proba(states, choices), model.predict_proba(states, choices))
    assert restored.calibration == model.calibration
    assert restored.metadata["question"] == "route"
    assert not restored.encoder_loaded
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calibration.json", "decision_config.json", "decision_head.safetensors"]
    with pytest.raises(ValueError, match="identity"):
        ChoiceDecisionModel.from_pretrained(tmp_path, expected_backbone_identity={"revision": "wrong"})
    moved = ChoiceDecisionModel.from_pretrained(tmp_path, backbone_name_or_path="/local/relocated")
    assert moved.config.backbone_name_or_path == "/local/relocated"
    assert moved.config.backbone_identity == model.config.backbone_identity


@pytest.mark.parametrize("filename", ["decision_head.safetensors", "calibration.json"])
def test_tampered_artifact_rejected(tmp_path, filename):
    ChoiceDecisionModel(config()).save_pretrained(tmp_path)
    with (tmp_path / filename).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="integrity"):
        ChoiceDecisionModel.from_pretrained(tmp_path)


def test_manifest_cannot_redirect_weight_loading(tmp_path):
    ChoiceDecisionModel(config()).save_pretrained(tmp_path)
    path = tmp_path / "decision_config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["head_file"] = "../other.safetensors"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        ChoiceDecisionModel.from_pretrained(tmp_path)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True, 1e-10])
def test_calibration_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        ChoiceDecisionModel(config()).set_calibration(temperature)


@pytest.mark.parametrize("overrides", [
    {"supported_types": ("score",)}, {"supported_types": ("noul",)},
    {"embedding_dim": 0}, {"projection_rank": 5}, {"max_candidates": 1},
    {"initial_scale": float("nan")}, {"template_version": "unknown"},
    {"backbone_identity": {"invalid": float("nan")}},
])
def test_config_rejects_unsupported_or_invalid_contracts(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


def test_encoding_invalid_requests_never_calls_encoder():
    encoder = FakeEncoder()
    model = ChoiceDecisionModel(config(max_candidates=3, max_text_chars=64), encoder=encoder)
    for states, questions in [([" "], "q"), (["a", "b"], ["q"]), (["a"], "q" * 65)]:
        with pytest.raises(ValueError):
            model.encode_states(states, questions)
    for choices in [["a"], ["a", "b", "c", "d"], ["a", " "]]:
        with pytest.raises(ValueError):
            model.encode_choices(choices, "q")
    assert not encoder.calls


def test_score_rejects_invalid_features_and_mask():
    model = ChoiceDecisionModel(config())
    states, choices = torch.eye(4)[:1], torch.eye(4)
    with pytest.raises(ValueError, match="finite"):
        model.score(states * float("nan"), choices)
    with pytest.raises(ValueError, match="non-zero"):
        model.score(states * 0, choices)
    with pytest.raises(ValueError, match="at least two"):
        model.score(states, choices, torch.tensor([True, False, False, False]))
    with pytest.raises(ValueError, match="boolean"):
        model.score(states, choices, torch.ones(4))
    with pytest.raises(ValueError, match="dimension"):
        model.score(torch.ones(1, 2), choices)


def test_malformed_encoder_features_rejected():
    class WrongEncoder:
        def encode(self, texts):
            return torch.ones(len(texts), 5)

    with pytest.raises(ValueError, match="encoder features"):
        ChoiceDecisionModel(config(), encoder=WrongEncoder()).encode_states("state", "question")
