"""Generation-only readout keeps public, training, and hooked forwards intact."""
import pytest
import torch

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model():
    torch.manual_seed(173)
    return SaddleForCausalLM(SaddleModelConfig(
        vocab_size=37, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_kind="gqa", attention_backend="sdpa",
        eos_token_id=[], qk_norm=True,
    )).eval()


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32),
    pytest.param("cuda", torch.float32, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
    pytest.param("cuda", torch.bfloat16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
    pytest.param("cuda", torch.float16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
])
def test_generation_readout_matches_public_logits_and_kv(batch, device, dtype):
    m = model().to(device=device, dtype=dtype)
    ids = torch.randint(4, 37, (batch, 17), device=device)
    with torch.inference_mode():
        expected = m(ids, use_cache=True, logits_to_keep=1)
        actual = m(ids, use_cache=True, logits_to_keep=1, _generation_only=True)
    assert expected.hidden_states.shape == (batch, 17, 32)
    assert actual.hidden_states is None
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    for observed, reference in zip(actual.past_key_values, expected.past_key_values):
        for a, b in zip(observed, reference):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cache_only_generation_skips_final_norm_and_dense_aux_allocations():
    m = model()
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        # Warm caches outside the profiler. Compare operations, not timings.
        m(ids, use_cache=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as baseline:
            expected = m(ids, use_cache=True, _prefill_only=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as candidate:
            actual = m(ids, use_cache=True, _prefill_only=True, _generation_only=True)
    def counts(profile):
        return {event.key: event.count for event in profile.key_averages()}
    before, after = counts(baseline), counts(candidate)
    assert before["aten::new_zeros"] == 2
    assert after.get("aten::new_zeros", 0) == 0
    assert before["aten::mean"] == after["aten::mean"] + 1
    assert actual.logits is None and actual.hidden_states is None
    for observed, reference in zip(actual.past_key_values, expected.past_key_values):
        for a, b in zip(observed, reference):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_norm_hooks_keep_full_inputs_and_layer_hooks_keep_tensor_aux():
    m = model()
    norm_inputs, aux = [], []
    handles = [m.norm.register_forward_pre_hook(lambda _, args: norm_inputs.append(args[0].shape)),
               m.layers[0].register_forward_hook(lambda _, args, output: aux.append(output[1]))]
    try:
        with m.create_generation_session(torch.tensor([[4, 5, 6, 7]]),
                                         max_new_tokens=2, do_sample=False) as session:
            session.step()
    finally:
        for handle in handles:
            handle.remove()
    assert norm_inputs == [(1, 4, 32)]
    assert len(aux) == 1 and isinstance(aux[0], torch.Tensor) and aux[0].item() == 0


def test_model_hook_keeps_full_hidden_states_and_custom_norm_keeps_sequence_shape(monkeypatch):
    m = model()
    observed = []
    handle = m.register_forward_hook(lambda _, args, output: observed.append(output.hidden_states.shape))
    try:
        m.generate(torch.tensor([[4, 5, 6]]), max_new_tokens=2, do_sample=False)
    finally:
        handle.remove()
    assert observed == [(1, 3, 32), (1, 1, 32)]
    original = m.norm.forward
    norm_inputs = []
    def sequence_norm(inputs):
        norm_inputs.append(inputs.shape)
        return original(inputs + inputs.mean(dim=1, keepdim=True))
    monkeypatch.setattr(m.norm, "forward", sequence_norm)
    ids = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        expected = m(ids, logits_to_keep=1)
        actual = m(ids, logits_to_keep=1, _generation_only=True)
    assert norm_inputs == [(1, 3, 32), (1, 3, 32)]
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)


def test_generation_readout_rejects_gradients_training_and_labels():
    m = model()
    ids = torch.tensor([[4, 5, 6]])
    with pytest.raises(ValueError, match="generation-only"):
        m(ids, _generation_only=True)
    with torch.inference_mode():
        with pytest.raises(ValueError, match="generation-only"):
            m(ids, labels=ids, _generation_only=True)
        m.train()
        with pytest.raises(ValueError, match="generation-only"):
            m(ids, _generation_only=True)
    m.train()
    output = m(ids, labels=ids)
    assert output.loss_items["aux_loss"].item() == 0
    output.loss.backward()
    assert m.norm.weight.grad is not None
