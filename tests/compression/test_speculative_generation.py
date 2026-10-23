"""Correctness of shared-vocabulary speculative causal decoding, not speed claims."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from saddlellm.compression.lightweight import Lightweight


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Tokenizer:
    eos_token_id = None

    def __init__(self, vocabulary=5):
        self.vocabulary = vocabulary

    def get_vocab(self):
        return {str(index): index for index in range(self.vocabulary)}

    def __call__(self, prompt, return_tensors="pt"):
        ids = torch.tensor([[int(value) for value in prompt.split()]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in ids)


class TransitionModel(torch.nn.Module):
    """Independent, exactly specified next-token laws indexed by previous token."""

    def __init__(self, probabilities, *, context=4096):
        super().__init__()
        matrix = torch.tensor(probabilities, dtype=torch.float64)
        if matrix.ndim == 1:
            matrix = matrix.repeat(matrix.numel(), 1)
        self.register_buffer("scores", matrix.log())
        self.config = SimpleNamespace(max_position_embeddings=context, is_encoder_decoder=False)
        self.calls = []

    def forward(self, input_ids):
        self.calls.append(input_ids.detach().cpu().tolist())
        return SimpleNamespace(logits=self.scores[input_ids])


def run(target, draft=None, **kwargs):
    options = {"max_new_tokens": 5, "num_draft_tokens": 3, "eos_token_id": []}
    options.update(kwargs)
    return Lightweight.speculative_generate(target, Tokenizer(), "1", draft_model=draft, **options)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_original_audit_mismatch_emits_only_target_and_obeys_one_token_budget(temperature):
    target = TransitionModel([0, 0, 1, 0, 0])
    draft = TransitionModel([0, 0, 0, 1, 0])
    result = run(target, draft, max_new_tokens=1, num_draft_tokens=1, temperature=temperature)
    assert result["token_ids"] == [2]
    assert result["text"] == "2"
    assert result["generated_tokens"] == 1
    assert result["accepted_tokens"] == 0
    assert result["acceptance_rate"] == 0


@pytest.mark.parametrize("draft_width", [1, 2, 4, 9])
def test_greedy_rejection_after_an_accepted_prefix_has_no_rejected_suffix(draft_width):
    target = TransitionModel([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    draft = TransitionModel([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    result = run(target, draft, max_new_tokens=11, num_draft_tokens=draft_width)
    assert result["token_ids"] == [2, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0]
    assert all(len(ids) == 1 for ids in target.calls + draft.calls)
    assert result["accepted_tokens"] <= result["draft_tokens"]


def test_all_accepted_draft_gets_one_target_bonus_and_still_obeys_budget():
    result = run(TransitionModel([0, 0, 1]), TransitionModel([0, 0, 1]),
                 max_new_tokens=7, num_draft_tokens=2)
    assert result["token_ids"] == [2] * 7
    assert result["accepted_tokens"] == result["draft_tokens"] == 5
    assert result["target_forwards"] == 4  # Includes one target vocabulary probe.
    assert result["draft_forwards"] == 5


@pytest.mark.parametrize("eos", [0, [0], [0, 4]])
@pytest.mark.parametrize("draft_law,accepted", [([1, 0, 0, 0, 0], 1), ([0, 0, 0, 1, 0], 0)])
def test_eos_from_accepted_or_replacement_token_stops_without_extra_tokens(eos, draft_law, accepted):
    result = run(TransitionModel([1, 0, 0, 0, 0]), TransitionModel(draft_law), eos_token_id=eos)
    assert result["token_ids"] == [0]
    assert result["accepted_tokens"] == accepted
    assert result["finish_reason"] == "eos"


def test_eos_from_bonus_token_stops():
    target = TransitionModel([[1, 0, 0], [0, 0, 1], [1, 0, 0]])
    result = run(target, TransitionModel([0, 0, 1]), num_draft_tokens=1, eos_token_id=0)
    assert result["token_ids"] == [2, 0]
    assert result["accepted_tokens"] == 1
    assert result["finish_reason"] == "eos"


def test_rejected_draft_eos_does_not_end_target_sequence():
    result = run(TransitionModel([0, 0, 1, 0]), TransitionModel([0, 0, 0, 1]),
                 eos_token_id=3, max_new_tokens=4)
    assert result["token_ids"] == [2] * 4
    assert result["finish_reason"] == "length"


def test_target_eos_precedence_and_explicit_disable():
    target = TransitionModel([0, 0, 1])
    target.generation_config = SimpleNamespace(eos_token_id=[2])
    target.config.eos_token_id = 1
    result = Lightweight.speculative_generate(target, Tokenizer(), "1", max_new_tokens=3)
    assert result["token_ids"] == [2]
    assert run(target, max_new_tokens=3)["token_ids"] == [2] * 3


def test_no_draft_uses_same_raw_logits_temperature_contract():
    result = run(TransitionModel([0, 0, 1]), max_new_tokens=3, temperature=1,
                 generator=torch.Generator().manual_seed(56))
    assert result["token_ids"] == [2] * 3
    assert result["mode"] == "target-only"
    assert result["draft_forwards"] == result["accepted_tokens"] == 0


def test_zero_budget_does_not_invoke_either_model():
    target, draft = TransitionModel([0, 1]), TransitionModel([1, 0])
    result = run(target, draft, max_new_tokens=0)
    assert result["token_ids"] == []
    assert result["text"] == ""
    assert not target.calls and not draft.calls


def test_context_above_2048_is_not_silently_truncated():
    target, draft = TransitionModel([0, 0, 1]), TransitionModel([0, 0, 1])
    prompt = " ".join(["1"] * 2300)
    result = Lightweight.speculative_generate(target, Tokenizer(), prompt, draft_model=draft,
                                               max_new_tokens=4, num_draft_tokens=1, eos_token_id=[])
    assert result["token_ids"] == [2] * 4
    assert len(draft.calls[0][0]) == 2300
    assert target.calls[0][0][:2300] == [1] * 2300
    assert len(target.calls[-1][0]) > 2300


def test_context_budget_is_explicitly_rejected_before_model_forward():
    target, draft = TransitionModel([0, 1], context=3), TransitionModel([1, 0])
    with pytest.raises(ValueError, match="never truncated"):
        run(target, draft, max_new_tokens=3)
    assert not target.calls and not draft.calls


def test_rejection_uses_hand_computed_normalized_positive_difference(monkeypatch):
    sampled = []

    def draw(probabilities, count, generator=None):
        sampled.append(probabilities.clone())
        return torch.tensor([0 if len(sampled) == 1 else 2])

    monkeypatch.setattr(torch, "multinomial", draw)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.8, dtype=torch.float64))
    result = run(TransitionModel([0.2, 0.3, 0.5]), TransitionModel([0.6, 0.2, 0.2]),
                 max_new_tokens=1, num_draft_tokens=1, temperature=1)
    assert result["token_ids"] == [2]
    assert len(sampled) == 2
    torch.testing.assert_close(sampled[0], torch.tensor([0.6, 0.2, 0.2], dtype=torch.float64))
    torch.testing.assert_close(sampled[1], torch.tensor([0, 0.25, 0.75], dtype=torch.float64))


@pytest.mark.parametrize("uniform,expected,accepted", [(0.499, 0, 1), (0.501, 1, 0)])
def test_acceptance_threshold_is_p_over_q_not_target_probability(monkeypatch, uniform, expected, accepted):
    draws = []

    def draw(probabilities, count, generator=None):
        draws.append(probabilities)
        return torch.tensor([0 if len(draws) == 1 else 1])

    monkeypatch.setattr(torch, "multinomial", draw)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(uniform, dtype=torch.float64))
    result = run(TransitionModel([0.4, 0.6]), TransitionModel([0.8, 0.2]),
                 max_new_tokens=1, temperature=1)
    assert result["token_ids"] == [expected]
    assert result["accepted_tokens"] == accepted


def test_sampling_matches_target_marginal_with_strongly_biased_draft():
    target, draft = TransitionModel([0.1, 0.25, 0.65]), TransitionModel([0.8, 0.15, 0.05])
    generator = torch.Generator().manual_seed(661)
    counts = Counter(run(target, draft, max_new_tokens=1, temperature=1, generator=generator)["token_ids"][0]
                     for _ in range(6000))
    assert all(abs(counts[index] / 6000 - expected) < 0.025
               for index, expected in enumerate([0.1, 0.25, 0.65]))


@pytest.mark.parametrize("draft_width", [1, 2])
def test_sampling_matches_conditional_two_token_joint_law(draft_width):
    laws = [[0.7, 0.1, 0.2], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]
    target = TransitionModel(laws)
    draft = TransitionModel([[0.1, 0.8, 0.1], [0.7, 0.1, 0.2], [0.7, 0.2, 0.1]])
    generator = torch.Generator().manual_seed(81)
    counts = Counter(tuple(run(target, draft, max_new_tokens=2, num_draft_tokens=draft_width,
                               temperature=1, generator=generator)["token_ids"]) for _ in range(6000))
    for first in range(3):
        for second in range(3):
            assert abs(counts[(first, second)] / 6000 - laws[1][first] * laws[first][second]) < 0.022


def test_sampling_seed_reproduces_tokens_and_temperature_applies_to_both_laws(monkeypatch):
    target, draft = TransitionModel([0.1, 0.3, 0.6]), TransitionModel([0.4, 0.3, 0.3])
    values = [run(target, draft, max_new_tokens=11, temperature=0.7,
                  generator=torch.Generator().manual_seed(822))["token_ids"] for _ in range(2)]
    assert values[0] == values[1]
    captured = []

    def draw(probabilities, count, generator=None):
        captured.append(probabilities.clone())
        return torch.tensor([2])

    monkeypatch.setattr(torch, "multinomial", draw)
    run(target, temperature=2, max_new_tokens=1)
    expected = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64).sqrt()
    torch.testing.assert_close(captured[0], expected / expected.sum())


def test_real_hf_and_native_greedy_match_their_target_only_generation():
    from transformers import GPT2Config, GPT2LMHeadModel

    from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

    torch.manual_seed(123)
    hf = GPT2LMHeadModel(GPT2Config(vocab_size=19, n_positions=32, n_embd=16, n_layer=1,
                                  n_head=2, eos_token_id=None, bos_token_id=1, pad_token_id=0)).eval()
    native = SaddleForCausalLM(SaddleModelConfig(vocab_size=19, hidden_size=16, intermediate_size=32,
                                                num_hidden_layers=1, num_attention_heads=2,
                                                num_key_value_heads=1, max_position_embeddings=32,
                                                attention_backend="eager")).eval()
    for target in (hf, native):
        with torch.inference_mode():
            expected = target.generate(torch.tensor([[1, 4, 6]]), do_sample=False,
                                       max_new_tokens=9, eos_token_id=[], pad_token_id=0)[0, 3:].tolist()
        result = Lightweight.speculative_generate(target, Tokenizer(19), "1 4 6",
                                                  draft_model=hf if target is native else native,
                                                  max_new_tokens=9, num_draft_tokens=3, eos_token_id=[])
        assert result["token_ids"] == expected


def test_training_modes_restored_even_when_forward_fails():
    target = TransitionModel([0, 0, 1])
    target.child = torch.nn.Dropout().eval()
    draft = TransitionModel([0, 0, 1]).eval()
    run(target, draft)
    assert target.training and not target.child.training and not draft.training
    target.forward = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected"))
    with pytest.raises(RuntimeError, match="injected"):
        run(target, draft)
    assert target.training and not target.child.training and not draft.training


@pytest.mark.parametrize("field,value", [("max_new_tokens", -1), ("max_new_tokens", True),
                                         ("num_draft_tokens", 0), ("num_draft_tokens", 1.5),
                                         ("temperature", float("nan")), ("temperature", float("inf")),
                                         ("temperature", -0.1), ("temperature", True),
                                         ("eos_token_id", [True]), ("eos_token_id", -1),
                                         ("generator", object())])
def test_invalid_controls_rejected(field, value):
    with pytest.raises((ValueError, TypeError)):
        run(TransitionModel([0, 1]), **{field: value})


def test_incompatible_tokenizer_vocab_or_logits_vocab_is_explicitly_rejected():
    with pytest.raises(ValueError, match="identical"):
        run(TransitionModel([0, 1]), TransitionModel([1, 0]), draft_tokenizer=Tokenizer(9))
    with pytest.raises(ValueError, match="vocabulary size"):
        run(TransitionModel([0, 1]), TransitionModel([0, 0, 1]))
    run(TransitionModel([0, 1]), TransitionModel([1, 0]), draft_tokenizer=Tokenizer(5))


@pytest.mark.parametrize("mode", ["encoder_decoder", "padding", "extra_inputs", "trimmed_logits", "bad_logits", "offload"])
def test_unsupported_model_and_input_contracts_fail_explicitly(mode):
    target = TransitionModel([0, 0, 1])
    tokenizer = Tokenizer()
    if mode == "encoder_decoder":
        target.config.is_encoder_decoder = True
    elif mode in {"padding", "extra_inputs"}:
        base = tokenizer

        def altered(prompt, **kwargs):
            values = base(prompt, **kwargs)
            values["attention_mask" if mode == "padding" else "token_type_ids"] = torch.zeros_like(values["input_ids"])
            return values

        tokenizer = altered
    elif mode == "trimmed_logits":
        target.forward = lambda input_ids: SimpleNamespace(logits=torch.zeros(1, 1, 3))
    elif mode == "bad_logits":
        target.scores.fill_(float("nan"))
    elif mode == "offload":
        target.hf_device_map = {"": "disk"}
    with pytest.raises(ValueError):
        Lightweight.speculative_generate(target, tokenizer, "1 1", max_new_tokens=2)


def test_timing_reports_measured_generation_without_invented_speedup(monkeypatch):
    clock = iter([100.0, 102.0])
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    result = run(TransitionModel([0, 1]), max_new_tokens=4)
    assert result["elapsed_seconds"] == 2
    assert result["tokens_per_second"] == 2
    assert result["estimated_speedup"] is None
    assert result["baseline_measured"] is False


def test_continuation_decoding_does_not_slice_by_decoded_prompt_character_count():
    class ContextSensitiveTokenizer(Tokenizer):
        def decode(self, ids, skip_special_tokens=True):
            return "continuation" if list(ids) == [2] else "not-a-prefix"

    result = Lightweight.speculative_generate(TransitionModel([0, 0, 1]), ContextSensitiveTokenizer(),
                                               "1", max_new_tokens=1, eos_token_id=[])
    assert result["text"] == "continuation"
