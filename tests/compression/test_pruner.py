"""Tests for the canonical pruning API and its legacy import path."""

import json

import pytest
import torch

import saddlellm.compression.pruner as pruner_module
from saddlellm.compression import Lightweight
from saddlellm.ModelPruner import ModelPruner


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.fc = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 4, bias=False)
        self.saved = []
        with torch.no_grad():
            self.q_proj.weight.copy_(
                torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
            )
            self.fc.weight.copy_(
                torch.arange(17, 33, dtype=torch.float32).reshape(4, 4)
            )
            self.lm_head.weight.fill_(1)

    def save_pretrained(self, path, **kwargs):
        self.saved.append((path, kwargs))


class DummyTokenizer:
    def __init__(self):
        self.saved = []

    def save_pretrained(self, path):
        self.saved.append(path)


def test_empty_pruner_does_not_download_default_model(monkeypatch):
    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("constructor should not load a default model")

    monkeypatch.setattr(
        pruner_module.AutoModelForCausalLM, "from_pretrained", unexpected_load
    )

    wrapper = ModelPruner()

    assert wrapper.model is None
    assert wrapper.model_name is None
    with pytest.raises(ValueError, match="in-memory model"):
        wrapper.prune()


@pytest.mark.parametrize("ratio", [-0.01, 1, 1.1, "invalid"])
def test_pruning_ratio_is_validated(ratio):
    with pytest.raises(ValueError, match="range"):
        ModelPruner(model=TinyModel(), pruning_ratio=ratio)


def test_unknown_method_and_empty_targets_are_rejected():
    with pytest.raises(ValueError, match="unsupported pruning method"):
        ModelPruner(model=TinyModel(), pruning_method="made-up")
    with pytest.raises(ValueError, match="target_modules"):
        ModelPruner(model=TinyModel(), target_modules=[])


def test_global_unstructured_pruning_uses_one_global_budget():
    model = TinyModel()
    wrapper = ModelPruner(
        model=model, pruning_method="unstructured", pruning_ratio=0.25
    )

    result = wrapper.prune()

    assert result is model
    assert wrapper.pruning_method == "l1_unstructured"
    masks = [model.q_proj.weight_mask, model.fc.weight_mask]
    assert sum(int((mask == 0).sum()) for mask in masks) == 8
    assert not hasattr(model.lm_head, "weight_mask")
    assert set(wrapper.pruning_masks) == {"q_proj.weight", "fc.weight"}


def test_structured_alias_prunes_complete_output_channels():
    model = TinyModel()
    wrapper = ModelPruner(
        model=model,
        pruning_method="structured",
        pruning_ratio=0.5,
        target_modules=["q_proj"],
    )

    wrapper.prune()

    row_is_zero = model.q_proj.weight_mask.sum(dim=1) == 0
    assert int(row_is_zero.sum()) == 2


def test_architecture_specific_methods_fail_closed():
    wrapper = ModelPruner(model=TinyModel(), pruning_method="head_structured")

    with pytest.raises(NotImplementedError, match="architecture-specific"):
        wrapper.prune()


def test_no_matching_module_fails_instead_of_silently_succeeding():
    wrapper = ModelPruner(model=TinyModel(), target_modules=["does_not_exist"])

    with pytest.raises(ValueError, match="no torch.nn.Linear modules matched"):
        wrapper.prune()


def test_remove_masks_materializes_zero_weights():
    model = TinyModel()
    wrapper = ModelPruner(model=model, pruning_ratio=0.25)
    wrapper.prune()

    result = wrapper.remove_masks()

    assert result is model
    assert not hasattr(model.q_proj, "weight_orig")
    assert not hasattr(model.fc, "weight_mask")
    assert (
        sum(int((layer.weight == 0).sum()) for layer in (model.q_proj, model.fc)) == 8
    )


def test_save_emits_portable_state_without_mutating_live_masks(tmp_path):
    model = TinyModel()
    tokenizer = DummyTokenizer()
    wrapper = ModelPruner(model=model, tokenizer=tokenizer, pruning_ratio=0.25)
    wrapper.prune()
    output_dir = tmp_path / "pruned"

    wrapper.save(str(output_dir))

    saved_state = model.saved[0][1]["state_dict"]
    assert "q_proj.weight" in saved_state
    assert "q_proj.weight_orig" not in saved_state
    assert "q_proj.weight_mask" not in saved_state
    assert hasattr(model.q_proj, "weight_mask")
    assert tokenizer.saved == [output_dir]
    manifest = json.loads(
        (output_dir / pruner_module.PRUNING_MANIFEST_FILENAME).read_text()
    )
    assert manifest["masks_materialized"] is True


def test_lightweight_prunes_in_memory_model_and_records_applied_status(tmp_path):
    model = TinyModel()
    tokenizer = DummyTokenizer()

    result = Lightweight.optimize(
        model,
        tokenizer,
        output_dir=str(tmp_path),
        target="300m",
        apply_quantization=False,
        apply_kv_cache_quant=False,
    )

    assert result is model
    assert not hasattr(model.q_proj, "weight_mask")
    config = json.loads((tmp_path / "lightweight_config.json").read_text())
    assert config["pruning"]["resolved_method"] == "l1_unstructured"
    assert config["quantization"] is None
    assert config["errors"] == []


def test_lightweight_does_not_claim_failed_pruning_was_applied(tmp_path):
    class NoMatchingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2)

        def save_pretrained(self, path, **_kwargs):
            return path

    model = NoMatchingModel()
    tokenizer = DummyTokenizer()

    Lightweight.optimize(
        model,
        tokenizer,
        output_dir=str(tmp_path),
        target="300m",
        apply_quantization=False,
        apply_kv_cache_quant=False,
    )

    config = json.loads((tmp_path / "lightweight_config.json").read_text())
    assert config["pruning"] is None
    assert config["errors"][0]["stage"] == "pruning"
