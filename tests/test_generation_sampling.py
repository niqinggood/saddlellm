from types import SimpleNamespace

import pytest
import torch

from saddlellm.models.NativeGeneration import NativeGenerationSession
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def _model():
    torch.manual_seed(173)
    return SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=37,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()


def _legacy_penalty(self, logits, out):
    for index in range(out.shape[0]):
        ids = torch.unique(out[index])
        values = logits[index, ids]
        logits[index, ids] = torch.where(
            values < 0,
            values * self.repetition_penalty,
            values / self.repetition_penalty,
        )


@pytest.mark.parametrize("penalty", [0.7, 1.1, 2.0])
@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("cache_mode", ["geometric", "dynamic"])
def test_cached_penalty_matches_independent_legacy_sampler(
    monkeypatch, penalty, sample, cache_mode
):
    model = _model()
    prompt = torch.tensor([[0, 0, 4, 4], [4, 5, 6, 7]])

    def stops(ids, scores):
        return torch.tensor([ids.shape[1] >= 5, ids.shape[1] >= 8])

    kwargs = dict(
        attention_mask=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
        max_new_tokens=7,
        do_sample=sample,
        seed=511,
        top_p=0.9,
        top_k=8,
        repetition_penalty=penalty,
        eos_token_id=[],
        stopping_criteria=stops,
        cache_implementation=cache_mode,
        output_scores=True,
        return_dict_in_generate=True,
    )
    actual = model.generate(prompt, **kwargs)
    monkeypatch.setattr(
        NativeGenerationSession, "_apply_repetition_penalty", _legacy_penalty
    )
    expected = model.generate(prompt, **kwargs)
    assert torch.equal(actual.sequences, expected.sequences)
    for observed, reference in zip(actual.scores, expected.scores):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("penalty", [0.7, 1.1, 2.0])
def test_cached_indices_preserve_signed_and_unseen_scores(dtype, penalty):
    model = _model()
    prompt = torch.tensor([[0, 4, 4, 6], [1, 3, 5, 7]])
    with model.create_generation_session(prompt, repetition_penalty=penalty) as session:
        with torch.inference_mode():
            scores = torch.linspace(-6, 6, 37, dtype=dtype).repeat(2, 1)
            expected = scores.clone()
            _legacy_penalty(session, expected, prompt)
            session._apply_repetition_penalty(scores, prompt)
            torch.testing.assert_close(scores, expected, rtol=0, atol=0)
        assert session._penalty_token_ids.shape == (2, 4 + session.max_new_tokens)
        assert session._penalty_token_ids.dtype == torch.long
    assert session._penalty_token_ids is None


def test_penalty_tracks_prompt_generated_tokens_and_finished_row_padding(monkeypatch):
    instance = _model()
    prompt = torch.tensor([[4, 5], [6, 7]])
    original_unique = torch.unique
    scanned = []

    def unique(values):
        scanned.append(values.shape)
        return original_unique(values)

    monkeypatch.setattr(torch, "unique", unique)

    def stops(ids, scores):
        return torch.tensor([ids.shape[1] >= 3, ids.shape[1] >= 5])

    with instance.create_generation_session(
        prompt,
        repetition_penalty=1.1,
        do_sample=False,
        eos_token_id=[],
        max_new_tokens=5,
        stopping_criteria=stops,
    ) as session:
        pointer = None
        for _ in session:
            expected = torch.zeros((2, 37), dtype=torch.bool)
            expected.scatter_(1, session.sequences(), True)
            observed = torch.zeros_like(expected)
            observed.scatter_(
                1, session._penalty_token_ids[:, : session._penalty_token_length], True
            )
            assert torch.equal(observed, expected)
            assert scanned == [torch.Size([2]), torch.Size([2])]
            if pointer is not None:
                assert session._penalty_token_ids.data_ptr() == pointer
            pointer = session._penalty_token_ids.data_ptr()
        assert session.sequences()[0, -1] == instance.config.pad_token_id


def test_default_penalty_has_no_index_buffer_allocation():
    with _model().create_generation_session(
        torch.tensor([[4, 5]]), eos_token_id=[]
    ) as session:
        session.step()
        assert session._penalty_token_ids is None


def test_int32_input_ids_keep_long_penalty_indices(monkeypatch):
    model = _model()
    prompt = torch.tensor([[4, 4, 5], [6, 7, 8]], dtype=torch.int32)
    kwargs = dict(
        max_new_tokens=4,
        repetition_penalty=1.1,
        eos_token_id=[],
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    actual = model.generate(prompt, **kwargs)
    monkeypatch.setattr(
        NativeGenerationSession, "_apply_repetition_penalty", _legacy_penalty
    )
    expected = model.generate(prompt, **kwargs)
    assert actual.sequences.dtype == torch.int32
    assert torch.equal(actual.sequences, expected.sequences)
    for observed, reference in zip(actual.scores, expected.scores):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


def test_no_eos_or_stop_does_not_synchronize_finished_count(monkeypatch):
    instance = _model()

    def planned(input_ids, **kwargs):
        return SimpleNamespace(logits=torch.ones((1, 1, 37)), past_key_values=None)

    monkeypatch.setattr(instance, "forward", planned)
    monkeypatch.setattr(
        torch.Tensor, "item", lambda *a: pytest.fail("unnecessary scalar sync")
    )
    with instance.create_generation_session(
        torch.tensor([[4, 5]]),
        max_new_tokens=3,
        eos_token_id=[],
        do_sample=False,
    ) as session:
        list(session)
        assert session.done and session.generated_tokens == 3
