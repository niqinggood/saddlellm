import threading

import pytest
import torch

from saddlellm.models.NativeGeneration import (
    NativeGenerationSession,
    NativeGenerationStep,
)
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def model(kind="gqa", backend="sdpa"):
    torch.manual_seed(173)
    return SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=37,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4 if kind == "mha" else 1 if kind == "mqa" else 2,
            max_position_embeddings=64,
            attention_kind=kind,
            attention_backend=backend,
            kv_lora_rank=4 if kind == "mla" else 0,
            mla_cache_mode="latent",
        )
    ).eval()


@pytest.mark.parametrize("kind", ["mha", "mqa", "gqa", "mla"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_prefill_and_resumed_decode_match_regular_generate(kind, backend):
    instance = model(kind, backend)
    prompt = torch.tensor([[0, 0, 4, 5], [4, 5, 6, 7]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    kwargs = dict(
        attention_mask=mask,
        max_new_tokens=5,
        do_sample=False,
        eos_token_id=[],
        output_scores=True,
    )
    reference = instance.generate(prompt, return_dict_in_generate=True, **kwargs)
    with instance.create_generation_session(prompt, **kwargs) as session:
        peek = session.prefill()
        assert peek.shape == (2, 37)
        assert session.generated_tokens == 0
        assert session.past_key_values[0].length == prompt.shape[1]
        first = session.step()
        assert isinstance(first, NativeGenerationStep)
        assert first.generated_tokens == 1
        assert not first.done
        assert session.past_key_values[0].length == prompt.shape[1]
        remaining = list(session)
        assert remaining[-1].done
        assert session.finish_reason == "length"
        assert torch.equal(session.sequences(), reference.sequences)
        for actual, expected in zip(session.scores, reference.scores):
            torch.testing.assert_close(actual, expected)
    assert session.closed and session.past_key_values is None


def test_prefill_is_idempotent_and_first_token_does_not_run_prompt_twice():
    instance = model()
    shapes = []
    handle = instance.register_forward_pre_hook(
        lambda _, args, kwargs: shapes.append(kwargs["input_ids"].shape[1]),
        with_kwargs=True,
    )
    try:
        with instance.create_generation_session(
            torch.tensor([[4, 5, 6]]),
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=[],
        ) as session:
            first = session.prefill()
            assert session.prefill() is first
            assert session.generated_tokens == 0
            session.step()
            assert shapes == [3]
            with pytest.raises(RuntimeError, match="before the first"):
                session.prefill()
            list(session)
            assert shapes == [3, 1, 1]
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "use_cache,cache_mode",
    [
        (True, "geometric"),
        (True, "dynamic"),
        (False, "geometric"),
    ],
)
def test_session_preserves_sampling_and_stop_rows(use_cache, cache_mode):
    instance = model()
    prompt = torch.tensor([[4, 5], [6, 7]])

    def stops(ids, scores):
        return torch.tensor([ids.shape[1] >= 3, ids.shape[1] >= 5])

    kwargs = dict(
        max_new_tokens=7,
        use_cache=use_cache,
        cache_implementation=cache_mode,
        do_sample=True,
        top_k=8,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=[],
        stopping_criteria=stops,
        output_scores=True,
    )
    torch.manual_seed(511)
    reference = instance.generate(prompt, return_dict_in_generate=True, **kwargs)
    torch.manual_seed(511)
    with instance.create_generation_session(prompt, **kwargs) as session:
        session.prefill()
        for result in session:
            assert result.token_ids.shape == (2, 1)
            assert result.scores.shape == (2, 37)
        assert session.finish_reason == "stop"
        assert torch.equal(session.sequences(), reference.sequences)
        for actual, expected in zip(session.scores, reference.scores):
            torch.testing.assert_close(actual, expected)


def test_seeded_sessions_are_independent_of_other_requests_and_global_rng():
    instance = model()
    prompt = torch.tensor([[4, 5]])
    kwargs = dict(max_new_tokens=5, do_sample=True, top_p=0.9, eos_token_id=[])
    state = torch.random.get_rng_state().clone()
    reference = instance.generate(prompt, seed=511, **kwargs)
    assert torch.equal(torch.random.get_rng_state(), state)
    with instance.create_generation_session(prompt, seed=511, **kwargs) as session:
        session.prefill()
        session.step()
        instance.generate(torch.tensor([[8, 9]]), seed=123, **kwargs)
        torch.rand(50)
        list(session)
        assert torch.equal(session.sequences(), reference)


def test_seed_and_generator_are_mutually_exclusive():
    with pytest.raises(ValueError, match="only one"):
        model().create_generation_session(
            torch.tensor([[4, 5]]), seed=1, generator=torch.Generator()
        )
    with pytest.raises(TypeError, match="seed"):
        model().create_generation_session(torch.tensor([[4, 5]]), seed=True)


def test_close_releases_owned_state_and_keeps_copied_results():
    instance = model()
    with instance.create_generation_session(
        torch.tensor([[4, 5]]),
        max_new_tokens=8,
        do_sample=False,
        eos_token_id=[],
        output_scores=True,
    ) as session:
        session.prefill()
        session.step()
        result = session.sequences()
    assert result.shape == (1, 3)
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()
    assert session.buffers is None and session.past_key_values is None
    assert session.model is None and session.scores is None
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.step()


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_step_and_prefill_do_not_leak_gradient_or_inference_context(grad_enabled):
    instance = model()
    with torch.set_grad_enabled(grad_enabled):
        with instance.create_generation_session(
            torch.tensor([[4, 5]]), eos_token_id=[]
        ) as session:
            assert not torch.is_inference_mode_enabled()
            session.prefill()
            assert torch.is_grad_enabled() == grad_enabled
            assert not torch.is_inference_mode_enabled()
            next(session)
            assert torch.is_grad_enabled() == grad_enabled
            assert not torch.is_inference_mode_enabled()


def test_training_backward_still_works_after_inference_initialized_rope_cache():
    instance = model()
    prompt = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        instance.generate(prompt, max_new_tokens=2, do_sample=False, eos_token_id=[])
    instance.train()
    output = instance(prompt, labels=prompt)
    output.loss.backward()
    assert instance.lm_head.weight.grad is not None
    assert torch.isfinite(instance.lm_head.weight.grad).all()


def test_empty_budget_does_not_run_model_and_iterator_is_exhausted(monkeypatch):
    instance = model()
    monkeypatch.setattr(
        instance, "forward", lambda *args, **kwargs: pytest.fail("unexpected forward")
    )
    prompt = torch.tensor([[4, 5]])
    with instance.create_generation_session(prompt, max_new_tokens=0) as session:
        assert session.prefill() is None
        assert list(session) == []
        assert torch.equal(session.sequences(), prompt)


def test_failed_forward_cannot_resume_a_partially_written_cache(monkeypatch):
    instance = model()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(instance, "forward", fail)
    session = instance.create_generation_session(torch.tensor([[4, 5]]))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        session.step()
    with pytest.raises(RuntimeError, match="session failed"):
        session.step()
    session.close()
    assert session.buffers is None and session.past_key_values is None


def test_session_rejects_concurrent_step_or_close(monkeypatch):
    instance = model()
    original = instance.forward
    entered, release = threading.Event(), threading.Event()
    errors = []

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(instance, "forward", blocked)
    session = instance.create_generation_session(
        torch.tensor([[4, 5]]), eos_token_id=[]
    )

    def run():
        try:
            session.step()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="already executing"):
            session.step()
        with pytest.raises(RuntimeError, match="executing step"):
            session.close()
    finally:
        release.set()
        worker.join(5)
        session.close()
    assert not errors


def test_public_lazy_imports_expose_session_and_step():
    from saddlellm import NativeGenerationSession as PublicSession
    from saddlellm import NativeGenerationStep as PublicStep

    assert PublicSession is NativeGenerationSession
    assert PublicStep is NativeGenerationStep
