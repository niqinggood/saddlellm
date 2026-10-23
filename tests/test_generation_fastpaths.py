import pytest
import torch

from saddlellm.models.InferenceCache import GenerationBuffers
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
            attention_backend="sdpa",
        )
    ).eval()


class _LegacyGenerationBuffers:
    """The pre-optimization concatenation path used as an independent reference."""

    def __init__(self, tokens, mask, max_new_tokens):
        self.tokens, self.mask = tokens, mask

    def views(self):
        return self.tokens, self.mask

    def append(self, token, active):
        self.tokens = torch.cat([self.tokens, token], dim=1)
        self.mask = torch.cat([self.mask, active.to(self.mask.dtype)[:, None]], dim=1)

    def sequences(self):
        return self.tokens


@pytest.mark.parametrize("keep", [1, 2, 20])
def test_partial_logits_match_full_forward_without_changing_hidden_states(keep):
    model = _model()
    prompt = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    with torch.no_grad():
        full = model(prompt)
        partial = model(prompt, logits_to_keep=keep)
    torch.testing.assert_close(partial.logits, full.logits[:, -keep:])
    torch.testing.assert_close(partial.hidden_states, full.hidden_states)


@pytest.mark.parametrize(
    "keep,error", [(-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_partial_logits_reject_invalid_limits(keep, error):
    with pytest.raises(error, match="logits_to_keep"):
        _model()(torch.tensor([[4, 5]]), logits_to_keep=keep)


def test_partial_logits_reject_labels_before_computation():
    model = _model()
    prompt = torch.tensor([[4, 5, 6]])
    with pytest.raises(ValueError, match="labels"):
        model(prompt, labels=prompt, logits_to_keep=1)
    output = model(prompt, labels=prompt)
    assert output.logits.shape == (1, 3, 37)
    output.loss.backward()
    assert model.lm_head.weight.grad is not None


@pytest.mark.parametrize("use_cache", [True, False])
def test_generate_projects_only_last_position(use_cache):
    model = _model()
    positions = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda _, inputs: positions.append(inputs[0].shape[1])
    )
    try:
        output = model.generate(
            torch.tensor([[4, 5, 6]]),
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=[],
            use_cache=use_cache,
        )
    finally:
        handle.remove()
    assert output.shape == (1, 8)
    assert positions == [1] * 5


def test_buffers_keep_storage_and_release_unused_output_capacity():
    prompt = torch.tensor([[0, 4], [5, 6]])
    mask = torch.tensor([[0, 1], [1, 1]], dtype=torch.bool)
    buffers = GenerationBuffers(prompt, mask, 512)
    pointers = (buffers.tokens.data_ptr(), buffers.mask.data_ptr())
    buffers.append(torch.tensor([[7], [8]]), torch.tensor([True, False]))
    buffers.append(torch.tensor([[9], [0]]), torch.tensor([True, False]))
    assert pointers == (buffers.tokens.data_ptr(), buffers.mask.data_ptr())
    tokens, full_mask = buffers.views()
    assert tokens.tolist() == [[0, 4, 7, 9], [5, 6, 8, 0]]
    assert full_mask.tolist() == [[False, True, True, True], [True, True, False, False]]
    assert prompt.tolist() == [[0, 4], [5, 6]]
    assert mask.tolist() == [[False, True], [True, True]]
    output = buffers.sequences()
    assert output.untyped_storage().nbytes() == output.numel() * output.element_size()
    assert output.data_ptr() != buffers.tokens.data_ptr()


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("early_stop", [False, True])
@pytest.mark.parametrize("cache_mode", ["geometric", "dynamic"])
def test_generation_matches_legacy_buffers_and_full_logits(
    monkeypatch, sample, early_stop, cache_mode
):
    model = _model()
    prompt = torch.tensor([[0, 0, 4, 5], [4, 5, 6, 7]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    def stop_rows(ids, scores):
        return torch.tensor([ids.shape[1] >= 5, ids.shape[1] >= 7])

    kwargs = dict(
        attention_mask=mask,
        max_new_tokens=7,
        do_sample=sample,
        top_p=0.9,
        top_k=8,
        repetition_penalty=1.1,
        eos_token_id=[],
        stopping_criteria=stop_rows if early_stop else None,
        cache_implementation=cache_mode,
        output_scores=True,
        return_dict_in_generate=True,
    )
    torch.manual_seed(511)
    optimized = model.generate(prompt, **kwargs)
    monkeypatch.setattr(
        "saddlellm.models.SaddleModeling.GenerationBuffers", _LegacyGenerationBuffers
    )
    torch.manual_seed(511)
    reference = model.generate(prompt, logits_to_keep=0, **kwargs)
    assert torch.equal(optimized.sequences, reference.sequences)
    assert len(optimized.scores) == len(reference.scores)
    for actual, expected in zip(optimized.scores, reference.scores):
        torch.testing.assert_close(actual, expected)
    assert prompt.tolist() == [[0, 0, 4, 5], [4, 5, 6, 7]]
    assert mask.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]


def test_zero_token_generation_preserves_prompt_and_has_no_scores():
    model = _model()
    prompt = torch.tensor([[4, 5]])
    output = model.generate(
        prompt, max_new_tokens=0, output_scores=True, return_dict_in_generate=True
    )
    assert torch.equal(output.sequences, prompt)
    assert output.scores == ()


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_unpadded_generation_checks_initial_mask_at_most_once(
    monkeypatch, explicit_mask
):
    model = _model()
    prompt = torch.tensor([[4, 5, 6], [7, 8, 9]])
    original_all = torch.all
    scans = []

    def scanned(value, *args, **kwargs):
        scans.append(value.shape)
        return original_all(value, *args, **kwargs)

    monkeypatch.setattr(torch, "all", scanned)
    model.generate(
        prompt,
        attention_mask=torch.ones_like(prompt) if explicit_mask else None,
        max_new_tokens=5,
        do_sample=False,
        eos_token_id=[],
    )
    assert scans == ([prompt.shape] if explicit_mask else [])


def test_mask_fastpath_restores_padding_after_a_batch_row_finishes(monkeypatch):
    model = _model()
    prompt = torch.tensor([[4, 5, 6], [7, 8, 9]])
    masks = []

    def capture(_, args, kwargs):
        mask = kwargs["attention_mask"]
        masks.append(None if mask is None else mask.clone())

    def stop_rows(ids, scores):
        return torch.tensor([ids.shape[1] >= 4, ids.shape[1] >= 6])

    handle = model.register_forward_pre_hook(capture, with_kwargs=True)
    kwargs = dict(
        max_new_tokens=5,
        do_sample=False,
        eos_token_id=[],
        stopping_criteria=stop_rows,
        output_scores=True,
        return_dict_in_generate=True,
    )
    try:
        actual = model.generate(prompt, **kwargs)
    finally:
        handle.remove()
    assert masks[0] is None
    assert masks[1].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert masks[2].tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]

    prepare = model.prepare_inputs_for_generation

    def force_mask(ids, attention_mask=None, **kw):
        if attention_mask is None:
            attention_mask = torch.ones_like(ids)
        return prepare(ids, attention_mask=attention_mask, **kw)

    monkeypatch.setattr(model, "prepare_inputs_for_generation", force_mask)
    reference = model.generate(prompt, **kwargs)
    assert torch.equal(actual.sequences, reference.sequences)
    for observed, expected in zip(actual.scores, reference.scores):
        torch.testing.assert_close(observed, expected)
