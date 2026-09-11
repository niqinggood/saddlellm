from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from saddlellm import easy


class _Tokenizer:
    pad_token = None
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, size=32):
        self.size = size

    def __len__(self):
        return self.size


class _Embeddings:
    def __init__(self, size):
        self.num_embeddings = size


class _Model:
    def __init__(self, vocab_size=64, scratch=False):
        self.config = SimpleNamespace(vocab_size=vocab_size, _name_or_path="")
        self.embeddings = _Embeddings(vocab_size)
        self.resize_calls = []
        if scratch:
            self._saddlellm_scratch_model = True

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, size):
        self.resize_calls.append(size)
        self.embeddings.num_embeddings = size
        self.config.vocab_size = size


def test_chat_keeps_system_prompt_for_normal_generation(monkeypatch):
    captured = {}

    class FakeSafeGenerate:
        def __init__(self, model, tokenizer):
            pass

        def generate(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return {"text": "ok"}

    import saddlellm.alignment.SafeGenerate as safe_generate_module

    monkeypatch.setattr(safe_generate_module, "SafeGenerate", FakeSafeGenerate)
    answer = easy.chat(
        _Model(),
        "解释一下",
        tokenizer=_Tokenizer(),
        system_prompt="回答要简洁",
    )

    assert answer == "ok"
    assert captured["prompt"] == "系统: 回答要简洁\n\n解释一下\n\n"


def test_chat_honors_self_consistency_without_cot_flag(monkeypatch):
    captured = {}

    class FakeSelfConsistency:
        def __init__(self, model, tokenizer, **kwargs):
            captured["temperature"] = kwargs["temperature"]

        def solve(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["cot_prompt"] = kwargs["cot_prompt"]
            return {"answer": "多数答案"}

    import saddlellm.alignment.AdvancedTechniques as techniques

    monkeypatch.setattr(techniques, "SelfConsistency", FakeSelfConsistency)
    answer = easy.chat(
        _Model(),
        "1+1=?",
        tokenizer=_Tokenizer(),
        system_prompt="认真计算",
        temperature=0.4,
        use_self_consistency=True,
    )

    assert answer == "多数答案"
    assert "系统: 认真计算" in captured["prompt"]
    assert captured["temperature"] == pytest.approx(0.4)


def test_scratch_model_vocab_is_safely_resized_for_tokenizer():
    model = _Model(vocab_size=32, scratch=True)
    tokenizer = _Tokenizer(size=50)

    assert easy._resolve_tokenizer(model, tokenizer) is tokenizer
    assert model.resize_calls == [50]
    assert tokenizer.pad_token == tokenizer.eos_token


def test_pretrained_model_vocab_mismatch_has_actionable_error():
    with pytest.raises(ValueError, match="与模型匹配的 tokenizer"):
        easy._resolve_tokenizer(_Model(vocab_size=32), _Tokenizer(size=50))


def test_auto_config_uses_real_torch_total_memory_field(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(index):
            return "RTX 4090"

        @staticmethod
        def get_device_properties(index):
            return SimpleNamespace(total_memory=24 * 1024**3)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))
    config = easy.auto_config(model_size="100m")

    assert config["gpu_memory_gb"] == 24.0
    assert config["strategy"] == "single"


def test_train_reports_missing_local_data_before_loading_pipeline():
    with pytest.raises(FileNotFoundError, match="训练数据不存在"):
        easy.train(
            _Model(),
            tokenizer=_Tokenizer(),
            data="definitely-missing.jsonl",
            steps=1,
        )
