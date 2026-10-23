"""Custom detokenizers must preserve sample/callback ordering."""

from functools import wraps
from types import MethodType

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel, Decoder, Sequence
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from saddlellm.runtime.InferenceServer import _can_batch_tokenizer_delivery


def tokenizer():
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]")))


def test_standard_fast_tokenizer_can_batch_delivery():
    instance = tokenizer()
    assert instance.backend_tokenizer.decoder is None
    assert _can_batch_tokenizer_delivery(instance)


@pytest.mark.parametrize("sequence", [False, True])
def test_standard_rust_decoder_can_batch_delivery(sequence):
    instance = tokenizer()
    decoder = ByteLevel()
    instance.backend_tokenizer.decoder = Sequence([decoder]) if sequence else decoder
    assert _can_batch_tokenizer_delivery(instance)


@pytest.mark.parametrize("sequence", [False, True])
def test_python_decoder_falls_back_without_executing_callback(sequence):
    class PythonDecoder:
        def decode_chain(self, tokens):
            pytest.fail("eligibility must not execute a custom decoder")

    instance = tokenizer()
    decoder = Decoder.custom(PythonDecoder())
    if sequence:
        try:
            decoder = Sequence([ByteLevel(), decoder])
        except BaseException as exc:
            # PyO3 exposes this BaseException without an importable Python
            # class. Only skip this exact unsupported-construction failure;
            # all ordinary errors, interrupts and other panics must propagate.
            if (type(exc).__module__ != "pyo3_runtime"
                    or type(exc).__name__ != "PanicException"
                    or str(exc) != "not implemented"):
                raise
            pytest.skip("Installed tokenizers cannot construct Sequence with a Python decoder")
    instance.backend_tokenizer.decoder = decoder
    assert not _can_batch_tokenizer_delivery(instance)


def test_decoder_changed_after_admission_is_rechecked():
    instance = tokenizer()
    assert _can_batch_tokenizer_delivery(instance)
    instance.backend_tokenizer.decoder = Decoder.custom(object())
    assert not _can_batch_tokenizer_delivery(instance)


@pytest.mark.parametrize("before_first_check", [False, True])
@pytest.mark.parametrize("owner,name", [
    (PreTrainedTokenizerBase, "decode"),
    (PreTrainedTokenizerFast, "_decode"),
    (PreTrainedTokenizerBase, "clean_up_tokenization"),
])
def test_class_method_override_falls_back(monkeypatch, before_first_check, owner, name):
    instance = tokenizer()
    monkeypatch.delattr(_can_batch_tokenizer_delivery, "_original_methods", raising=False)
    if not before_first_check:
        assert _can_batch_tokenizer_delivery(instance)

    @wraps(getattr(owner, name))
    def custom(*args, **kwargs):
        pytest.fail("eligibility must not execute a patched method")

    monkeypatch.setattr(owner, name, staticmethod(custom) if name == "clean_up_tokenization" else custom)
    assert not _can_batch_tokenizer_delivery(instance)


def test_custom_decode_falls_back():
    instance = tokenizer()
    instance.decode = MethodType(lambda self, *args, **kwargs: "custom", instance)
    assert not _can_batch_tokenizer_delivery(instance)


def test_custom_cleanup_falls_back():
    instance = tokenizer()
    instance.clean_up_tokenization = lambda value: value
    assert not _can_batch_tokenizer_delivery(instance)


def test_custom_subclass_and_plain_tokenizers_fall_back():
    class Custom(PreTrainedTokenizerFast):
        pass

    instance = Custom(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    assert not _can_batch_tokenizer_delivery(instance)
    assert not _can_batch_tokenizer_delivery(object())
