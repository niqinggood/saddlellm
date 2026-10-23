"""Configuration, calibration, and export contracts for opt-in AWQ/GPTQ."""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest
import torch

import saddlellm.compression.CalibratedPTQ as ptq


def enabled(**kwargs):
    return ptq.PTQConfig(enabled=True, calibration_data=["one two three", "four five six"], group_size=4, **kwargs)


@pytest.mark.parametrize("kwargs,match", [
    ({"enabled": "false"}, "boolean"),
    ({"method": "rtn"}, "awq"),
    ({"bits": 4.0}, "bits"),
    ({"bits": True}, "bits"),
    ({"group_size": 0}, "group_size"),
    ({"method": "awq", "group_size": -1}, "positive group_size"),
    ({"max_samples": 0}, "max_samples"),
    ({"max_seq_length": 1}, "max_seq_length"),
    ({"seed": -1}, "seed"),
    ({"dampening_frac": float("nan")}, "dampening_frac"),
    ({"device": "mps"}, "device"),
    ({"actorder": "wrong"}, "actorder"),
    ({"pipeline": "unknown"}, "pipeline"),
    ({"targets": []}, "targets"),
    ({"targets": ["re:["]}, "regular expression"),
    ({"calibration_data": "https://example.com/data.jsonl"}, "local file"),
    ({"calibration_data": "builtin-c4"}, "jsonl"),
    ({"enabled": True}, "explicit local"),
    ({"enabled": True, "calibration_data": []}, "empty"),
    ({"calibration_data": [], "calibration_path": "data.jsonl"}, "only one"),
])
def test_invalid_configuration_fails_before_model_load(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ptq.validate_ptq_config(kwargs)


def test_plan_is_serializable_and_does_not_read_files_or_import_stack(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("dry run imported a backend or read a file")

    monkeypatch.setattr(ptq.importlib, "import_module", unexpected)
    monkeypatch.setattr(Path, "open", unexpected)
    plan = ptq.validate_ptq_config({"enabled": True, "method": "gptq", "calibration_path": "missing.jsonl"})
    assert plan["format"] == "compressed-tensors"
    assert plan["model_loaded"] is False
    json.dumps(plan)


def test_disabled_is_identity_without_output_or_optional_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(ptq, "_load_backend", lambda *args: pytest.fail("backend imported"))
    model, tokenizer = object(), object()
    result = ptq.run_calibrated_ptq(model, tmp_path / "unused", {}, tokenizer=tokenizer)
    assert result.model is model
    assert result.tokenizer is tokenizer
    assert result.report["status"] == "disabled"
    assert not (tmp_path / "unused").exists()


def test_sampling_deduplicates_strips_and_preserves_global_random_state():
    config = enabled(max_samples=3)
    config.calibration_data = [" same ", "same", " "] + [f"row {index}" for index in range(40)]
    state = random.getstate()
    first = ptq.load_calibration_texts(config)
    assert first == ptq.load_calibration_texts(config)
    assert len(first) == len(set(first)) == 3
    assert random.getstate() == state
    config.seed += 1
    assert first != ptq.load_calibration_texts(config)


@pytest.mark.parametrize("suffix", [".jsonl", ".json", ".txt"])
def test_local_calibration_formats_and_custom_column(tmp_path, suffix):
    path = tmp_path / f"calibration{suffix}"
    records = [{"content": "one two"}, {"content": "three four"}, {"content": "one two"}]
    if suffix == ".jsonl":
        value = "\n".join(json.dumps(item) for item in records)
    elif suffix == ".json":
        value = json.dumps({"records": records})
    else:
        value = "one two\nthree four\none two\n"
    path.write_text(value, encoding="utf-8-sig")
    config = ptq.PTQConfig(enabled=True, calibration_path=str(path), text_column="content")
    assert ptq.load_calibration_texts(config) == ["one two", "three four"]


def test_invalid_json_identifies_file_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"text":"valid"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        ptq.load_calibration_texts(ptq.PTQConfig(enabled=True, calibration_path=str(path)))


@pytest.mark.parametrize("rows,match", [([{}], "string field"), ([" ", ""], "nonempty unique")])
def test_invalid_calibration_is_not_silently_replaced(rows, match):
    config = enabled()
    config.calibration_data = rows
    with pytest.raises(ValueError, match=match):
        ptq.load_calibration_texts(config)


class TinyConfig:
    def __init__(self):
        self._name_or_path = "original-model-name"
        self.quantization_config = None

    def save_pretrained(self, output):
        payload = {"model_type": "llama"}
        if self.quantization_config:
            payload["quantization_config"] = self.quantization_config
        (Path(output) / "config.json").write_text(json.dumps(payload), encoding="utf-8")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.lm_head = torch.nn.Linear(8, 8)
        self.config = TinyConfig()

    def save_pretrained(self, output, **kwargs):
        from safetensors.torch import save_file

        assert kwargs["save_compressed"] is True
        self.config.save_pretrained(output)
        save_file({"proj.weight_packed": torch.zeros(8, 1, dtype=torch.int32), "proj.weight_scale": torch.ones(8, 2)},
                  str(Path(output) / "model.safetensors"))


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        tokens = [index + 1 for index, _ in enumerate(text.split())][:kwargs["max_length"]]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    def save_pretrained(self, output):
        (Path(output) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakeDataset:
    @staticmethod
    def from_list(rows):
        return rows


class FakeModifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_backend(monkeypatch, *, transform_only=False, mutate=True, fail=False):
    calls = []

    def oneshot(**kwargs):
        calls.append(kwargs)
        local_name = Path(kwargs["model"].config._name_or_path)
        assert (local_name / "config.json").is_file()
        assert kwargs["shuffle_calibration_samples"] is False
        assert kwargs["pipeline"] == "basic"
        if fail:
            raise RuntimeError("calibration failed")
        if mutate:
            kwargs["model"].config.quantization_config = {
                "quant_method": "compressed-tensors", "format": "pack-quantized", "quantization_status": "compressed",
                "config_groups": {"group_0": {"targets": ["Linear"]}},
            }
        return kwargs["model"]

    monkeypatch.setattr(ptq, "_load_backend", lambda method: (oneshot, FakeDataset, FakeModifier, FakeModifier if transform_only else None))
    return calls


@pytest.mark.parametrize("method,transform_only", [("awq", False), ("awq", True), ("gptq", False)])
def test_backend_recipe_uses_requested_quantization_and_exports_manifest(tmp_path, monkeypatch, method, transform_only):
    calls = fake_backend(monkeypatch, transform_only=transform_only)
    model = TinyModel()
    output = tmp_path / method
    config = enabled(method=method, device="cpu", max_seq_length=2, symmetric=False)
    result = ptq.run_calibrated_ptq(None, output, config, model=model, tokenizer=TinyTokenizer())
    assert result.model is model
    assert model.config._name_or_path == "original-model-name"
    assert result.report["calibration_tokens"] == 4
    assert result.report["quantized_modules"] == ["proj"]
    assert result.report["accelerated_inference_verified"] is False
    manifest = json.loads((output / ptq.MANIFEST_FILENAME).read_text())
    assert manifest == {"method": method, "backend": "llm-compressor", "format": "compressed-tensors", "state_file": None,
                        "self_contained": True, "bits": 4, "group_size": 4, "precision": "int4", "source_model": None,
                        "report_file": "ptq_report.json"}
    recipe = calls[0]["recipe"]
    quantizer = recipe[1] if isinstance(recipe, list) else recipe
    weights = quantizer.kwargs["config_groups"]["group_0"]["weights"]
    assert weights == {"num_bits": 4, "type": "int", "symmetric": False, "strategy": "group", "group_size": 4}
    if method == "gptq":
        assert quantizer.kwargs["actorder"] == "static"


def test_per_channel_and_target_exclusions(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    config = enabled(method="gptq")
    config.group_size = -1
    config.targets = [r"re:^proj$"]
    result = ptq.run_calibrated_ptq(None, tmp_path / "channel", config, model=TinyModel(), tokenizer=TinyTokenizer())
    assert result.report["quantized_modules"] == ["proj"]
    weights = calls[0]["recipe"].kwargs["config_groups"]["group_0"]["weights"]
    assert weights["strategy"] == "channel"
    assert "group_size" not in weights


@pytest.mark.parametrize("config_changes,match", [({"group_size": 3}, "must divide"), ({"ignore": ["Linear"]}, "no supported")])
def test_unsupported_targets_fail_before_mutation(tmp_path, monkeypatch, config_changes, match):
    calls = fake_backend(monkeypatch)
    config = enabled()
    for key, value in config_changes.items():
        setattr(config, key, value)
    model = TinyModel()
    with pytest.raises(ValueError, match=match):
        ptq.run_calibrated_ptq(None, tmp_path / "out", config, model=model, tokenizer=TinyTokenizer())
    assert calls == []
    assert not (tmp_path / "out").exists()


def test_requested_cuda_fails_explicitly_on_cpu_host(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        ptq.run_calibrated_ptq(None, tmp_path / "out", enabled(device="cuda"), model=TinyModel(), tokenizer=TinyTokenizer())


def test_backend_failure_restores_source_name_and_publishes_no_success(tmp_path, monkeypatch):
    fake_backend(monkeypatch, fail=True)
    model = TinyModel()
    with pytest.raises(RuntimeError, match="calibration failed"):
        ptq.run_calibrated_ptq(None, tmp_path / "failed", enabled(device="cpu"), model=model, tokenizer=TinyTokenizer())
    assert model.config._name_or_path == "original-model-name"
    assert not (tmp_path / "failed").exists()
    assert list(tmp_path.iterdir()) == []


def test_noop_backend_cannot_report_success(tmp_path, monkeypatch):
    fake_backend(monkeypatch, mutate=False)
    with pytest.raises(RuntimeError, match="did not export a compressed-tensors"):
        ptq.run_calibrated_ptq(None, tmp_path / "fake", enabled(device="cpu"), model=TinyModel(), tokenizer=TinyTokenizer())
    assert not (tmp_path / "fake").exists()


@pytest.mark.parametrize("format_name", ["fakequant", "dense"])
def test_fake_quantized_weights_are_not_packed_exports(tmp_path, format_name):
    model = TinyModel()
    model.config.quantization_config = {"quant_method": "compressed-tensors", "format": format_name,
                                       "quantization_status": "compressed", "config_groups": {"group_0": {}}}
    model.save_pretrained(tmp_path, save_compressed=True)
    with pytest.raises(RuntimeError, match="integer quantization_config"):
        ptq._verify_export(tmp_path, ["proj"])


def test_export_must_contain_every_target_as_integer_weights_with_scales(tmp_path):
    model = TinyModel()
    model.config.quantization_config = {"quant_method": "compressed-tensors", "format": "pack-quantized",
                                       "quantization_status": "compressed", "config_groups": {"group_0": {}}}
    model.save_pretrained(tmp_path, save_compressed=True)
    with pytest.raises(RuntimeError, match="target lm_head"):
        ptq._verify_export(tmp_path, ["proj", "lm_head"])


def test_nonempty_output_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    monkeypatch.setattr(ptq, "_load_backend", lambda *args: pytest.fail("backend loaded"))
    with pytest.raises(ValueError, match="empty directory"):
        ptq.run_calibrated_ptq(None, output, enabled(), model=TinyModel(), tokenizer=TinyTokenizer())
    assert (output / "keep.txt").read_text() == "keep"


def test_optional_dependency_error_is_actionable(monkeypatch):
    real_import = ptq.importlib.import_module

    def import_module(name):
        if name == "llmcompressor":
            raise ModuleNotFoundError("No module named llmcompressor", name="llmcompressor")
        return real_import(name)

    monkeypatch.setattr(ptq.importlib, "import_module", import_module)
    with pytest.raises(ImportError, match=r"saddlellm\[ptq\]"):
        ptq._load_backend("awq")


@pytest.mark.skipif(importlib.util.find_spec("llmcompressor") is None, reason="optional real PTQ backend is not installed")
@pytest.mark.parametrize("method", ["awq", "gptq"])
@pytest.mark.parametrize("bits", [4, 8])
def test_real_tiny_llama_calibration_export_and_reload_on_cpu(tmp_path, method, bits):
    """Actual backend algorithm and HF roundtrip; creates all fixtures locally."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    tokenizer_impl = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "one": 2, "two": 3, "three": 4, "four": 5}, unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl, unk_token="[UNK]", pad_token="[PAD]",
                                       model_input_names=["input_ids", "attention_mask"])
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                       num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                                       pad_token_id=1, tie_word_embeddings=False))
    config = ptq.PTQConfig(enabled=True, method=method, bits=bits, group_size=8, device="cpu", local_files_only=True,
                          max_seq_length=8, calibration_data=["one two three four one two", "four three two one four three"])
    result = ptq.run_calibrated_ptq(None, tmp_path / method, config, model=model, tokenizer=tokenizer)
    assert result.report["status"] == "completed"
    assert result.report["calibration_pipeline"] == "saddlellm_single_device_basic"
    assert all(parameter.device.type == "cpu" for parameter in result.model.parameters())
    from safetensors.torch import load_file

    saved = load_file(str(Path(result.output_dir) / "model.safetensors"))
    assert any(key.endswith((".weight_packed", ".weight")) and value.dtype in {torch.int32, torch.int8} for key, value in saved.items())
    loaded = ptq.load_calibrated_ptq(result.output_dir, local_files_only=True)
    batch = tokenizer("one two three", return_tensors="pt")
    with torch.no_grad():
        logits = loaded(**batch).logits
        returned_logits = result.model(**batch).logits
    assert logits.shape == (1, 3, 8)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(returned_logits, logits)
    from saddlellm.models.ModelLoader import load_model_and_tokenizer

    repository_model, repository_tokenizer = load_model_and_tokenizer(result.output_dir, local_files_only=True, device="cpu")
    with torch.no_grad():
        repository_logits = repository_model(**repository_tokenizer("one two three", return_tensors="pt")).logits
    assert repository_model._saddle_quantization_load_mode == "compressed-tensors-decompressed"
    torch.testing.assert_close(repository_logits, logits)
    generated = repository_model.generate(**repository_tokenizer("one two", return_tensors="pt"), max_new_tokens=2,
                                          do_sample=False, pad_token_id=1)
    assert generated.shape[0] == 1
    assert 2 < generated.shape[1] <= 4
