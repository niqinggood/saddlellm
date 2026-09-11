"""Tests for the canonical quantization API and its legacy import path."""

import pytest
import torch

import saddlellm.compression.quantizer as quantizer_module
from saddlellm.ModelQuantizer import ModelQuantizer


class DummyModel:
    def __init__(self):
        self.qconfig = None
        self.eval_called = False
        self.loaded_state = None

    def eval(self):
        self.eval_called = True
        return self

    def state_dict(self):
        return {"weight": torch.tensor([1.0])}

    def load_state_dict(self, state):
        self.loaded_state = state


class DummyTokenizer:
    def __init__(self):
        self.saved = []

    def save_pretrained(self, path):
        self.saved.append(path)


def test_empty_quantizer_does_not_download_default_model(monkeypatch):
    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("constructor should not load a default model")

    monkeypatch.setattr(
        quantizer_module.AutoModelForCausalLM, "from_pretrained", unexpected_load
    )

    quantizer = ModelQuantizer()

    assert quantizer.model is None
    assert quantizer.model_name is None


def test_dynamic_quantization_accepts_an_in_memory_model(monkeypatch):
    source = DummyModel()
    quantized = DummyModel()
    calls = []

    def fake_quantize(model, layers, dtype):
        calls.append((model, layers, dtype))
        return quantized

    monkeypatch.setattr(quantizer_module, "quantize_dynamic", fake_quantize)
    wrapper = ModelQuantizer(model=source)

    result = wrapper.quantize(bits=8)

    assert result is quantized
    assert wrapper.model is quantized
    assert calls[0][0] is source


def test_four_bit_quantization_rejects_loaded_model_before_reloading():
    wrapper = ModelQuantizer(model=DummyModel())

    with pytest.raises(ValueError, match="cannot convert an already-loaded model"):
        wrapper.quantize(model=wrapper.model, bits=4)


def test_static_quantization_requires_calibration_data():
    wrapper = ModelQuantizer(model=DummyModel(), method="static")

    with pytest.raises(ValueError, match="requires calibration_data"):
        wrapper.quantize()


def test_static_quantization_uses_custom_qconfig_and_calibrates(monkeypatch):
    model = DummyModel()
    wrapper = ModelQuantizer(model=model, method="static")
    qconfig = object()
    calibration_data = [{"text": "sample"}]
    calibrated = []

    monkeypatch.setattr(
        quantizer_module.torch.quantization, "prepare", lambda item, inplace: item
    )
    monkeypatch.setattr(
        quantizer_module.torch.quantization, "convert", lambda item, inplace: item
    )
    monkeypatch.setattr(
        wrapper,
        "_calibrate",
        lambda item, data: calibrated.append((item, data)),
    )

    result = wrapper.quantize(qconfig_spec=qconfig, calibration_data=calibration_data)

    assert result is model
    assert model.eval_called
    assert model.qconfig is qconfig
    assert calibrated == [(model, calibration_data)]


def test_qat_cannot_skip_training_via_quantize():
    wrapper = ModelQuantizer(model=DummyModel(), method="qat")

    with pytest.raises(ValueError, match=r"requires fit\(train_dataset\)"):
        wrapper.quantize()


def test_save_uses_a_directory_contract_for_state_and_manifest(tmp_path):
    tokenizer = DummyTokenizer()
    wrapper = ModelQuantizer(model=DummyModel(), tokenizer=tokenizer)
    output_dir = tmp_path / "quantized"

    result = wrapper.save(str(output_dir))

    assert result == str(output_dir)
    assert (output_dir / quantizer_module.QUANTIZED_STATE_FILENAME).is_file()
    assert (output_dir / quantizer_module.QUANTIZATION_MANIFEST_FILENAME).is_file()
    assert tokenizer.saved == [output_dir]


def test_dynamic_checkpoint_requires_explicit_model_when_source_is_unknown(
    tmp_path, monkeypatch
):
    source = ModelQuantizer(model=DummyModel())
    output_dir = tmp_path / "quantized"
    source.save(str(output_dir))
    monkeypatch.setattr(
        quantizer_module,
        "quantize_dynamic",
        lambda model, _layers, dtype: model,
    )

    with pytest.raises(ValueError, match="pass model"):
        ModelQuantizer.load(str(output_dir))

    target = DummyModel()
    loaded = ModelQuantizer.load(str(output_dir), model=target)

    assert loaded.model is target
    assert torch.equal(target.loaded_state["weight"], torch.tensor([1.0]))


def test_static_checkpoint_load_fails_closed(tmp_path):
    wrapper = ModelQuantizer(model=DummyModel(), method="static")
    output_dir = tmp_path / "quantized"
    wrapper.save(str(output_dir))

    with pytest.raises(NotImplementedError, match="static/QAT checkpoints"):
        ModelQuantizer.load(str(output_dir))
