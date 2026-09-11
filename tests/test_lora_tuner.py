import json

import pytest
import torch

import saddlellm.tuning.LoRATuner as lora_module
from saddlellm.tuning.LoRATuner import LoRATuner


class DummyModel:
    def __init__(self):
        self.printed = False
        self.device = "cpu"
        self.saved = []

    def print_trainable_parameters(self):
        self.printed = True

    def save_pretrained(self, path):
        self.saved.append(path)


class DummyTokenizer:
    eos_token = "<eos>"

    def __init__(self):
        self.pad_token = None
        self.saved = []

    def save_pretrained(self, path):
        self.saved.append(path)

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": [1]}


def _patch_loading(monkeypatch):
    model_calls = []
    tokenizer_calls = []
    adapter_configs = []
    dependency_calls = []

    def load_model(name, **kwargs):
        model_calls.append((name, kwargs))
        return DummyModel()

    def load_tokenizer(name, **kwargs):
        tokenizer_calls.append((name, kwargs))
        return DummyTokenizer()

    monkeypatch.setattr(lora_module.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(lora_module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(lora_module, "LoraConfig", lambda **kwargs: adapter_configs.append(kwargs) or kwargs)
    monkeypatch.setattr(lora_module, "get_peft_model", lambda model, _config: model)
    monkeypatch.setattr(lora_module, "prepare_model_for_kbit_training", lambda model: model)
    monkeypatch.setattr(lora_module, "BitsAndBytesConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        lora_module,
        "require_distribution",
        lambda *args, **kwargs: dependency_calls.append((args, kwargs)) or "test",
    )
    return model_calls, tokenizer_calls, adapter_configs, dependency_calls


def test_plain_lora_does_not_require_bitsandbytes_or_remote_code(monkeypatch):
    model_calls, tokenizer_calls, adapter_configs, dependency_calls = _patch_loading(monkeypatch)

    tuner = LoRATuner(model_name="local-base", use_4bit=False)

    assert dependency_calls == []
    assert model_calls[0][0] == "local-base"
    assert "load_in_4bit" not in model_calls[0][1]
    assert model_calls[0][1]["quantization_config"] is None
    assert model_calls[0][1]["trust_remote_code"] is False
    assert tokenizer_calls[0][1]["trust_remote_code"] is False
    assert adapter_configs[0]["target_modules"] == ["q_proj", "v_proj"]
    assert tuner.tokenizer.pad_token == "<eos>"


def test_lora_never_downloads_an_implicit_default_model(monkeypatch):
    model_calls, _, _, dependency_calls = _patch_loading(monkeypatch)

    with pytest.raises(ValueError, match="model_name is required"):
        LoRATuner()

    assert model_calls == []
    assert dependency_calls == []


def test_plain_lora_uses_float32_on_cpu(monkeypatch):
    monkeypatch.setattr(lora_module.torch.cuda, "is_available", lambda: False)
    model_calls, _, _, _ = _patch_loading(monkeypatch)

    LoRATuner(model_name="local-base", use_4bit=False)

    assert model_calls[0][1]["torch_dtype"] is torch.float32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_length": 0}, "max_length"),
        ({"lora_rank": 0}, "lora_rank"),
        ({"lora_alpha": 0}, "lora_alpha"),
        ({"lora_dropout": 1}, "lora_dropout"),
        ({"target_modules": []}, "target_modules"),
    ],
)
def test_invalid_lora_configuration_fails_before_loading(monkeypatch, kwargs, message):
    model_calls, _, _, _ = _patch_loading(monkeypatch)

    with pytest.raises(ValueError, match=message):
        LoRATuner(model_name="local-base", use_4bit=False, **kwargs)

    assert model_calls == []


def test_qlora_uses_one_quantization_config_without_legacy_flag(monkeypatch):
    model_calls, _, _, dependency_calls = _patch_loading(monkeypatch)

    LoRATuner(model_name="local-base", use_4bit=True)

    assert dependency_calls
    kwargs = model_calls[0][1]
    assert "load_in_4bit" not in kwargs
    assert kwargs["quantization_config"]["load_in_4bit"] is True


def test_load_uses_adapter_base_model_without_creating_second_adapter(monkeypatch, tmp_path):
    model_calls, _, adapter_configs, _ = _patch_loading(monkeypatch)
    attached = []
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "base-from-config"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lora_module.PeftModel,
        "from_pretrained",
        lambda model, path, **kwargs: attached.append((model, path, kwargs)) or model,
    )

    tuner = LoRATuner.load(str(tmp_path), use_4bit=False)

    assert model_calls[0][0] == "base-from-config"
    assert adapter_configs == []
    assert attached == [(tuner.model, str(tmp_path), {"is_trainable": True})]


def test_plain_lora_training_disables_half_precision_on_cpu(monkeypatch):
    _patch_loading(monkeypatch)
    tuner = LoRATuner(model_name="local-base", use_4bit=False)
    training_kwargs = []
    trained = []

    class Dataset:
        def map(self, _function, batched):
            assert batched
            return self

    class Trainer:
        def __init__(self, **_kwargs):
            pass

        def train(self):
            trained.append(True)

    monkeypatch.setattr(lora_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(lora_module.torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(
        lora_module,
        "TrainingArguments",
        lambda **kwargs: training_kwargs.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(lora_module, "DataCollatorForLanguageModeling", lambda **_kwargs: object())
    monkeypatch.setattr(lora_module, "Trainer", Trainer)

    tuner.fit(Dataset(), epochs=1)

    assert trained == [True]
    assert training_kwargs[0]["bf16"] is False
    assert training_kwargs[0]["fp16"] is False
    assert training_kwargs[0]["optim"] == "adamw_torch"


def test_save_persists_adapter_and_tokenizer(monkeypatch, tmp_path):
    _patch_loading(monkeypatch)
    tuner = LoRATuner(model_name="local-base", use_4bit=False)

    tuner.save(str(tmp_path))

    assert tuner.model.saved == [str(tmp_path)]
    assert tuner.tokenizer.saved == [str(tmp_path)]
