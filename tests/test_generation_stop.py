import random
import threading
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.GenerationStop import StopStringRows
from saddlellm.runtime.InferenceServer import (
    CompletionRequest,
    InferenceServerSettings,
    LocalGenerationEngine,
    _generation_control,
)


@pytest.fixture
def tokenizer():
    vocab = {"[UNK]": 0, "[PAD]": 1}
    vocab.update(
        {char: index + 2 for index, char in enumerate(bytes_to_unicode().values())}
    )
    vocab[""] = len(vocab)
    vocab["<STOP>" + "x" * 1000] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["[UNK]", "[PAD]"])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        clean_up_tokenization_spaces=False,
    )


def _ids(tokenizer, text):
    alphabet = bytes_to_unicode()
    return [
        tokenizer.backend_tokenizer.token_to_id(alphabet[byte])
        for byte in text.encode()
    ]


def _check_prefixes(tokenizer, generated, stops, *, incremental=True):
    prompt = _ids(tokenizer, "prompt<STOP>")
    checker = StopStringRows(
        tokenizer, len(prompt), [stops], allow_incremental=incremental
    )
    for count in range(1, len(generated) + 1):
        prefix = generated[:count]
        expected = any(
            stop in tokenizer.decode(prefix, skip_special_tokens=True) for stop in stops
        )
        actual = checker(torch.tensor([prompt + prefix]))
        assert actual.tolist() == [expected]
    return checker


@pytest.mark.parametrize(
    "text,stops",
    [
        ("hello <STOP> ignored", ["<STOP>", "end"]),
        ("你好🙂 世界结束", ["世界结束"]),
        ("hello\n\n world", ["\n\n"]),
        ("I'm fine . end", ["fine ."]),
        ("x" * 200 + "a b c", ["a b"]),
        ("你好", ["\ufffd"]),
    ],
)
def test_suffix_checks_match_full_decode_at_every_token(tokenizer, text, stops):
    checker = _check_prefixes(tokenizer, _ids(tokenizer, text), stops)
    assert checker.rows[0].incremental == all("\ufffd" not in stop for stop in stops)


def test_skipped_empty_and_unknown_tokens_do_not_shorten_effective_lookbehind(
    tokenizer,
):
    ignored = [
        tokenizer.pad_token_id,
        tokenizer.backend_tokenizer.token_to_id(""),
        999999,
    ]
    generated = _ids(tokenizer, "你a") + ignored * 100 + _ids(tokenizer, "b")
    checker = _check_prefixes(tokenizer, generated, ["ab"])
    assert checker.rows[0].matched
    assert len(checker.rows[0].tail) <= checker.rows[0].window_tokens


def test_invalid_utf8_prefixes_and_decoder_fallback_tokens_match_full_decode(tokenizer):
    rng = random.Random(173)
    alphabet = bytes_to_unicode()
    tokenizer.backend_tokenizer.add_tokens(["「🙂」"])
    literal = tokenizer.backend_tokenizer.token_to_id("「🙂」")
    for stop in ["<STOP>", "好🙂", "\nend", "「🙂」"]:
        for _ in range(4):
            raw = bytes(rng.randrange(256) for _ in range(40))
            generated = [
                tokenizer.backend_tokenizer.token_to_id(alphabet[byte])
                for byte in raw
            ]
            generated += [literal] + _ids(tokenizer, "x" * 40 + stop)
            _check_prefixes(tokenizer, generated, [stop])


def test_long_new_token_is_checked_in_full_before_trimming(tokenizer):
    token = tokenizer.backend_tokenizer.token_to_id("<STOP>" + "x" * 1000)
    _check_prefixes(tokenizer, [token], ["<STOP>"])


def test_delayed_or_chunked_callbacks_check_all_new_tokens(tokenizer):
    checker = StopStringRows(tokenizer, 1, [["<STOP>"]], allow_incremental=True)
    prompt = _ids(tokenizer, "A")
    first = _ids(tokenizer, "x" * 100)
    assert not checker(torch.tensor([prompt + first])).any()
    new = _ids(tokenizer, "<STOP>" + "x" * 200)
    assert checker(torch.tensor([prompt + first + new])).all()


def test_byte_level_check_keeps_decode_work_bounded(tokenizer, monkeypatch):
    checker = StopStringRows(tokenizer, 1, [["STOP"]], allow_incremental=True)
    original = checker._decode
    lengths = []

    def decode(ids):
        lengths.append(len(ids))
        return original(ids)

    monkeypatch.setattr(checker, "_decode", decode)
    token = _ids(tokenizer, "x")[0]
    prompt = _ids(tokenizer, "A")[0]
    for count in range(1, 1001):
        assert not checker(torch.tensor([[prompt] + [token] * count])).any()
    assert max(lengths) <= 4 * len("STOP") + 5
    assert sum(lengths) < 22000


def test_different_rows_use_independent_stops_and_mixed_fallback(tokenizer):
    checker = StopStringRows(
        tokenizer, 1, [["ab"], ["\ufffd"], []], allow_incremental=True
    )
    for count in range(1, 4):
        rows = [
            _ids(tokenizer, "Aabc")[: count + 1],
            _ids(tokenizer, "A你")[: count + 1],
            _ids(tokenizer, "Aabc")[: count + 1],
        ]
        expected = [
            "ab" in tokenizer.decode(rows[0][1:]),
            "\ufffd" in tokenizer.decode(rows[1][1:]),
            False,
        ]
        assert checker(torch.tensor(rows)).tolist() == expected


def test_repeated_and_shortened_sequences_reset_fast_state(tokenizer):
    checker = StopStringRows(tokenizer, 1, [["ab"]], allow_incremental=True)
    assert checker(torch.tensor([_ids(tokenizer, "Aab")])).all()
    assert not checker(torch.tensor([_ids(tokenizer, "Axy")])).any()
    assert not checker(torch.tensor([_ids(tokenizer, "Ax")])).any()


def test_cleanup_and_other_decoder_use_exact_full_history(tokenizer):
    tokenizer.clean_up_tokenization_spaces = True
    checker = _check_prefixes(tokenizer, _ids(tokenizer, "a ."), ["a."])
    assert not checker.rows[0].incremental
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.backend_tokenizer.decoder = decoders.WordPiece(cleanup=True)
    checker = _check_prefixes(tokenizer, _ids(tokenizer, "a b"), ["a b"])
    assert not checker.rows[0].incremental


def test_custom_decode_override_uses_full_history(tokenizer, monkeypatch):
    original = tokenizer.decode
    monkeypatch.setattr(
        tokenizer, "decode", lambda ids, **kw: "prefix:" + original(ids, **kw)
    )
    checker = _check_prefixes(tokenizer, _ids(tokenizer, "abc"), ["prefix:"])
    assert not checker.rows[0].incremental


def test_generic_control_decodes_once_per_row_not_per_stop():
    calls = []

    class CustomTokenizer:
        def decode(self, ids, **kwargs):
            calls.append(ids)
            return "abc"

    control = _generation_control(CustomTokenizer(), 2, ["x", "y", "z"])
    assert control(torch.tensor([[4, 5, 6], [7, 8, 9]]), None).tolist() == [
        False,
        False,
    ]
    assert calls == [[6], [9]]


def test_cancel_and_empty_stops_need_no_text_decode():
    class CustomTokenizer:
        def decode(self, *args, **kwargs):
            pytest.fail("unexpected text decode")

    tokens = torch.tensor([[4, 5, 6]])
    event = threading.Event()
    control = _generation_control(CustomTokenizer(), 2, ["stop"], event)
    event.set()
    assert control(tokens, None).all()
    assert not _generation_control(CustomTokenizer(), 2, [""])(tokens, None).any()


def test_invalid_row_shape_is_rejected(tokenizer):
    checker = StopStringRows(tokenizer, 1, [["stop"]], allow_incremental=True)
    with pytest.raises(ValueError, match="fixed batch"):
        checker(torch.tensor([[4], [5]]))


def test_complete_stream_and_batch_share_exact_native_stops(tokenizer, monkeypatch):
    output_ids = _ids(tokenizer, "你好 <STOP> ignored")
    model = SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            eos_token_id=[],
            pad_token_id=tokenizer.pad_token_id,
        )
    )

    def planned(input_ids, **kwargs):
        scores = torch.full((input_ids.shape[0], 1, len(tokenizer)), -100.0)
        scores[:, :, output_ids[input_ids.shape[1] - 1]] = 10
        return SimpleNamespace(logits=scores, past_key_values=None)

    monkeypatch.setattr(model, "forward", planned)
    engine = LocalGenerationEngine(
        model,
        tokenizer,
        InferenceServerSettings(
            batch_size=2,
            default_max_new_tokens=32,
            max_new_tokens=64,
        ),
    )
    request = CompletionRequest(prompt="A", temperature=0, max_tokens=32, stop="<STOP>")
    complete = engine.complete(request)
    state = {}
    streamed = "".join(
        engine._stream_generate(
            "A",
            max_tokens=32,
            temperature=0,
            top_p=1,
            repetition_penalty=1,
            stop="<STOP>",
            stream_state=state,
        )
    )
    batched = engine.complete_batch(
        [
            request,
            CompletionRequest(prompt="B", temperature=0, max_tokens=32, stop="好"),
        ]
    )
    assert (
        complete["choices"][0]["text"]
        == streamed
        == batched[0]["choices"][0]["text"]
        == "你好 "
    )
    assert (
        state["completion_tokens"]
        == complete["usage"]["completion_tokens"]
        == len("你好 <STOP>".encode())
    )
    assert state["finish_reason"] == "stop"
    assert batched[1]["choices"][0]["text"] == "你"
    assert batched[1]["usage"]["completion_tokens"] == len("你好".encode())
