"""Tests for the canonical pruning API and its legacy import path."""

import json

import pytest
import torch
from transformers.pytorch_utils import Conv1D

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


class DiskTinyModel(TinyModel):
    def forward(self, values):
        return self.fc(self.q_proj(values))

    def save_pretrained(self, path, **kwargs):
        super().save_pretrained(path, **kwargs)
        torch.save(kwargs.get("state_dict", self.state_dict()), path / "weights.pt")


def _load_disk_model(path):
    model = DiskTinyModel()
    model.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
    return model


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


def test_gpt_style_conv1d_projection_can_be_pruned():
    class GPTProjectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = Conv1D(12, 4)

    model = GPTProjectionModel()
    wrapper = ModelPruner(
        model=model,
        target_modules=["c_attn"],
        pruning_ratio=0.25,
    )

    wrapper.prune()

    assert int((model.c_attn.weight_mask == 0).sum()) == 12


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


@pytest.mark.parametrize("remove_masks", [False, True])
def test_optimizer_step_then_immediate_save_keeps_current_weights(tmp_path, remove_masks):
    model = DiskTinyModel()
    wrapper = ModelPruner(model=model, target_modules=["q_proj"], pruning_ratio=.5)
    wrapper.prune()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    expected = model.q_proj.weight_orig.detach() * model.q_proj.weight_mask
    assert not torch.equal(model.q_proj.weight.detach(), expected)

    wrapper.save(tmp_path, remove_masks=remove_masks)

    saved = _load_disk_model(tmp_path)
    torch.testing.assert_close(saved.q_proj.weight, expected, rtol=0, atol=0)
    exact = torch.load(tmp_path / "pruning_masks.pt", weights_only=True)
    assert exact["q_proj.weight"].dtype == torch.bool
    assert torch.equal(exact["q_proj.weight"], expected.ne(0))
    assert hasattr(model.q_proj, "weight_orig") is not remove_masks


@pytest.mark.parametrize("remove_masks", [False, True])
def test_saved_exact_masks_preserve_natural_zero_as_trainable(tmp_path, remove_masks):
    from torch.nn.utils import prune

    from saddlellm.compression.RecoveryTrainer import _enforce_masks, _pruning_masks

    model = DiskTinyModel()
    with torch.no_grad():
        model.q_proj.weight[3, 3] = 0  # A real zero outside the pruning mask.
    mask = torch.ones_like(model.q_proj.weight)
    mask[0] = 0
    prune.custom_from_mask(model.q_proj, "weight", mask)
    ModelPruner(model=model, target_modules=["q_proj"]).save(tmp_path, remove_masks=remove_masks)

    dense = _load_disk_model(tmp_path)
    restored = ModelPruner.load(str(tmp_path), model=dense, tokenizer=DummyTokenizer())
    assert restored.pruning_masks["q_proj.weight"][3, 3]
    masks, _, info = _pruning_masks(dense, str(tmp_path), torch.device("cpu"))
    assert info["source"] == "saved_exact_masks"
    assert info["inferred_from_materialized_zeros"] is False
    optimizer = torch.optim.SGD(dense.parameters(), lr=.01)
    dense(torch.ones(1, 4)).sum().backward()
    _enforce_masks(masks, gradients=True)
    optimizer.step()
    _enforce_masks(masks)
    assert dense.q_proj.weight_orig[3, 3] != 0
    assert torch.count_nonzero(dense.q_proj.weight_orig[mask == 0]) == 0


@pytest.mark.parametrize("remove_masks", [False, True])
def test_load_repeated_pruning_keeps_old_targets_and_replaces_stale_sidecar(tmp_path, remove_masks):
    first = ModelPruner(model=DiskTinyModel(), target_modules=["q_proj"], pruning_ratio=.25)
    first.prune()
    first.save(tmp_path, remove_masks=remove_masks)
    first_mask = torch.load(tmp_path / "pruning_masks.pt", weights_only=True)["q_proj.weight"]
    second = ModelPruner.load(str(tmp_path), model=_load_disk_model(tmp_path), tokenizer=DummyTokenizer(),
                             target_modules=["fc"], pruning_method="random_unstructured", pruning_ratio=.5)
    assert hasattr(second.model.q_proj, "weight_mask")
    second.prune()
    assert set(second.pruning_masks) == {"q_proj.weight", "fc.weight"}
    assert torch.equal(second.pruning_masks["q_proj.weight"], first_mask)
    # Saving into the same directory must replace the first pass's exact mask.
    second.save(tmp_path, remove_masks=remove_masks)
    current = torch.load(tmp_path / "pruning_masks.pt", weights_only=True)
    assert set(current) == {"q_proj.weight", "fc.weight"}
    assert int((~current["fc.weight"]).sum()) == 8
    third = ModelPruner.load(str(tmp_path), model=_load_disk_model(tmp_path), tokenizer=DummyTokenizer(),
                            target_modules=["q_proj"], pruning_method="random_unstructured", pruning_ratio=.5)
    third.prune()
    assert torch.all(third.pruning_masks["q_proj.weight"][~first_mask] == 0)
    assert torch.equal(third.pruning_masks["fc.weight"], current["fc.weight"])
    third.remove_masks()
    assert not hasattr(third.model.q_proj, "weight_orig")
    assert not hasattr(third.model.fc, "weight_orig")


def test_remove_masks_then_prune_new_targets_reattaches_old_masks():
    wrapper = ModelPruner(model=DiskTinyModel(), target_modules=["q_proj"], pruning_ratio=.25)
    wrapper.prune()
    old = wrapper.pruning_masks["q_proj.weight"].clone()
    wrapper.remove_masks()
    wrapper.target_modules = ("fc",)
    wrapper.prune()
    assert torch.equal(wrapper.model.q_proj.weight_mask.bool(), old)
    assert set(wrapper.pruning_masks) == {"q_proj.weight", "fc.weight"}


@pytest.mark.parametrize("payload,match", [
    ({}, "nonempty"),
    ({"q_proj.weight": [1, 0]}, "binary"),
    ({"q_proj.weight": torch.ones(2, 2)}, "shape"),
    ({"q_proj.weight_orig": torch.ones(4, 4)}, "noncanonical"),
    ({"unknown.weight": torch.ones(4, 4)}, "unknown"),
    ({"q_proj.weight": torch.full((4, 4), float("nan"))}, "binary"),
    ({"q_proj.weight": torch.zeros(4, 4)}, "disagrees"),
])
def test_load_rejects_invalid_or_inconsistent_masks(tmp_path, payload, match):
    torch.save(payload, tmp_path / "pruning_masks.pt")
    with pytest.raises(ValueError, match=match):
        ModelPruner.load(str(tmp_path), model=DiskTinyModel(), tokenizer=DummyTokenizer())


def test_load_validates_all_masks_before_installing_any_hook(tmp_path):
    torch.save({"q_proj.weight": torch.ones(4, 4), "fc.weight": torch.ones(2, 2)}, tmp_path / "pruning_masks.pt")
    model = DiskTinyModel()
    with pytest.raises(ValueError, match="shape"):
        ModelPruner.load(str(tmp_path), model=model, tokenizer=DummyTokenizer())
    assert not hasattr(model.q_proj, "weight_orig")


def test_load_rejects_incomplete_sidecar_against_saved_manifest(tmp_path):
    wrapper = ModelPruner(model=DiskTinyModel(), pruning_ratio=.25)
    wrapper.prune()
    wrapper.save(tmp_path)
    saved = torch.load(tmp_path / "pruning_masks.pt", weights_only=True)
    saved.pop("fc.weight")
    torch.save(saved, tmp_path / "pruning_masks.pt")
    model = _load_disk_model(tmp_path)
    with pytest.raises(ValueError, match="masked_parameters"):
        ModelPruner.load(str(tmp_path), model=model, tokenizer=DummyTokenizer())
    assert not hasattr(model.q_proj, "weight_orig")


@pytest.mark.parametrize("masks_file,exception", [("pruning_masks.pt", FileNotFoundError), ("other.pt", ValueError)])
def test_load_rejects_missing_or_unsupported_declared_sidecar(tmp_path, masks_file, exception):
    (tmp_path / "pruning_manifest.json").write_text(json.dumps({"masks_file": masks_file}), encoding="utf-8")
    with pytest.raises(exception, match="masks"):
        ModelPruner.load(str(tmp_path), model=DiskTinyModel(), tokenizer=DummyTokenizer())


def test_new_dense_save_cannot_retain_an_unrelated_old_sidecar(tmp_path):
    torch.save({"old.weight": torch.ones(1)}, tmp_path / "pruning_masks.pt")
    ModelPruner(model=DiskTinyModel()).save(tmp_path)
    assert not (tmp_path / "pruning_masks.pt").exists()


@pytest.mark.parametrize("architecture", ["hf", "native"])
def test_real_checkpoint_save_reload_and_repeat_pruning(tmp_path, architecture):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from saddlellm.models.ModelLoader import load_causal_lm

    if architecture == "hf":
        model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=16, n_embd=8, n_layer=1, n_head=1))
        target = "c_attn"
    else:
        from saddlellm.models.ModelBlueprint import (
            AttentionBlueprint,
            FFNBlueprint,
            ModelBlueprint,
        )

        model = ModelBlueprint(name="exact-pruner-mask", vocab_size=8, hidden_size=8, num_layers=1,
            max_position_embeddings=16, attention=AttentionBlueprint(kind="gqa", num_heads=2, num_kv_heads=1, backend="eager"),
            ffn=FFNBlueprint(kind="swiglu", intermediate_size=16)).build_model()
        target = "q_proj"
    backend = Tokenizer(WordLevel({str(index): index for index in range(8)}, unk_token="0"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="0", pad_token="1", eos_token="2")
    wrapper = ModelPruner(model=model, tokenizer=tokenizer, target_modules=[target], pruning_ratio=.25)
    wrapper.prune()
    input_ids = torch.tensor([[1, 4, 5, 2]])
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    optimizer.step()
    expected = {name: parameter.detach().clone() for name, parameter in wrapper._materialized_state_dict().items()}
    wrapper.save(tmp_path)
    dense = load_causal_lm(tmp_path, device="cpu", local_files_only=True)
    for name, value in dense.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    if architecture == "hf":
        # Exercise the unchanged default HF loader, not just injected loading.
        restored = ModelPruner.load(str(tmp_path), device_map="cpu", local_files_only=True)
        assert next(restored.model.parameters()).dtype == torch.float32
        assert next(restored.model.parameters()).device.type == "cpu"
    else:
        restored = ModelPruner.load(str(tmp_path), model=dense, tokenizer=tokenizer)
    before = {name: mask.clone() for name, mask in restored.pruning_masks.items()}
    restored.prune(method="random_unstructured", sparsity=.25)
    for name, mask in before.items():
        assert torch.count_nonzero(restored.pruning_masks[name][~mask]) == 0
    restored.save(tmp_path / "second", remove_masks=True)
    final = load_causal_lm(tmp_path / "second", device="cpu", local_files_only=True)
    assert torch.isfinite(final(input_ids=input_ids).logits).all()


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
