"""Unequal-length decode uses batched projections and request-private attention."""

from contextlib import ExitStack

import pytest
import torch

from saddlellm.models.InferenceCache import _IndependentLayerCache
from saddlellm.models.NativeBatchDecode import NativeBatchDecodeRunner
from test_native_batch_decode import model


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["compatible", "throughput"])
@pytest.mark.parametrize(
    "architecture", ["gqa", "mha", "mqa", "packed", "qkv_bias", "qk_norm", "parallel"]
)
def test_mixed_lengths_match_same_history_reference_and_never_join_kv(
    device, mode, architecture
):
    config = {}
    if architecture == "packed":
        config.update(fused_qkv=True, fused_gate_up=True)
    if architecture == "qkv_bias":
        config.update(qkv_bias=True)
    if architecture == "qk_norm":
        config.update(qk_norm=True)
    if architecture == "parallel":
        config.update(
            residual_topology="parallel",
            residual_attention_scale=0.7,
            residual_ffn_scale=1.2,
        )
    instance = model(
        kind=architecture if architecture in {"mha", "mqa"} else "gqa", **config
    ).to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    prompts = [[4, 5], [7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20]]
    with ExitStack() as stack, torch.inference_mode():
        shapes = []
        attention = instance.layers[0].self_attn
        projection = (
            attention.qkv_packed_proj
            if attention.qkv_packed_proj is not None
            else attention.q_proj
        )
        hook = projection.register_forward_pre_hook(
            lambda module, args: shapes.append(tuple(args[0].shape))
        )
        stack.callback(hook.remove)

        def create(ids):
            return stack.enter_context(
                instance.create_generation_session(
                    torch.tensor([ids], device=device),
                    max_new_tokens=10,
                    do_sample=False,
                    eos_token_id=[],
                    output_scores=True,
                )
            )

        references, sessions = (
            [create(p) for p in prompts],
            [create(p) for p in prompts],
        )
        runner = stack.enter_context(
            NativeBatchDecodeRunner(instance, decode_mode=mode)
        )
        for tick in range(10):
            for reference in references:
                reference.step()
            runner.step(sessions if tick % 2 else list(reversed(sessions)))
            for actual, reference in zip(sessions, references):
                exact = device == "cuda" and mode == "compatible"
                torch.testing.assert_close(
                    actual.scores[-1],
                    reference.scores[-1],
                    rtol=0 if exact else 0.02 if device == "cuda" else 1e-4,
                    atol=0 if exact else 0.002 if device == "cuda" else 1e-6,
                )
                for left, right in zip(
                    actual.past_key_values, reference.past_key_values
                ):
                    assert left.length == right.length
                    for k, v in zip(left.buffers, right.buffers):
                        torch.testing.assert_close(
                            k[:, :, : left.length],
                            v[:, :, : right.length],
                            rtol=0 if exact else 0.02 if device == "cuda" else 1e-4,
                            atol=0 if exact else 0.002 if device == "cuda" else 1e-6,
                        )
                actual.buffers.tokens[:, actual.buffers.length - 1].copy_(
                    reference.buffers.tokens[:, reference.buffers.length - 1]
                )
        status = runner.snapshot()
        assert status["independent_kv_forwards"] == 9
        assert (
            status["cohort_builds"]
            == status["cohort_history_copy_bytes"]
            == status["retained_cohorts"]
            == 0
        )
        assert status["batch_size_histogram"] == {3: 9}
        assert ((3, 1, 32) in shapes) == (device == "cpu" or mode == "throughput")
        assert (
            len(
                {
                    s.past_key_values[0].buffers[0].untyped_storage().data_ptr()
                    for s in sessions
                }
            )
            == 3
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["eager", "sdpa", "sdpa_reference"])
def test_seeded_mixed_length_samples_and_stops_match_serial(device, backend):
    instance = model(backend=backend).to(
        device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    plans = [([4, 5], 3), ([7, 8, 9, 10], 8), ([11, 12, 13, 14, 15, 16], 6)]

    def create(ids, count, index):
        return instance.create_generation_session(
            torch.tensor([ids], device=device),
            max_new_tokens=count,
            do_sample=True,
            seed=70 + index,
            temperature=0.7 + index * 0.2,
            top_p=0.9,
            top_k=11,
            repetition_penalty=1.1,
            eos_token_id=[],
        )

    refs = []
    for index, (ids, count) in enumerate(plans):
        with create(ids, count, index) as ref:
            list(ref)
            refs.append(ref.sequences().clone())
    with ExitStack() as stack:
        sessions = [
            stack.enter_context(create(ids, count, index))
            for index, (ids, count) in enumerate(plans)
        ]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        while live := [s for s in sessions if not s.done]:
            runner.step(list(reversed(live)))
        assert runner.independent_kv_forwards > 0
        assert all(torch.equal(s.sequences(), ref) for s, ref in zip(sessions, refs))


def test_dense_cohort_to_private_rows_then_rejoin_and_cancel():
    instance = model()
    expected = instance.generate(
        torch.tensor([[4, 5]]), max_new_tokens=9, do_sample=False
    )
    with ExitStack() as stack:
        sessions = [
            stack.enter_context(
                instance.create_generation_session(
                    torch.tensor([[4, 5]]),
                    max_new_tokens=9,
                    do_sample=False,
                    eos_token_id=[],
                )
            )
            for _ in range(2)
        ]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions)
        runner.step(sessions)
        sessions[0].step()
        runner.step(sessions)
        assert runner.independent_kv_forwards == 1
        sessions[1].step()
        runner.step(sessions)
        assert runner.cohort_builds == 2
        runner.forget(sessions[0])
        sessions[0].close()
        while not sessions[1].done:
            runner.step(sessions[1:])
        assert torch.equal(sessions[1].sequences(), expected)


def test_partial_independent_forward_failure_invalidates_every_row(monkeypatch):
    instance = model()
    with ExitStack() as stack:
        sessions = [
            stack.enter_context(
                instance.create_generation_session(
                    torch.tensor([ids]),
                    max_new_tokens=5,
                    do_sample=False,
                    eos_token_id=[],
                )
            )
            for ids in ([4, 5], [6, 7, 8])
        ]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions)

        def failed(*args, **kwargs):
            raise RuntimeError("injected independent attention failure")

        monkeypatch.setattr(instance.layers[1].self_attn, "_attention", failed)
        with pytest.raises(RuntimeError, match="injected"):
            runner.step(sessions)
        assert all(s._failed for s in sessions)
        for item in sessions:
            with pytest.raises(RuntimeError, match="failed"):
                item.step()


@pytest.mark.parametrize(
    "invalid", ["training", "gradients", "prefill", "mask", "labels", "no_cache"]
)
def test_internal_cache_cannot_silently_change_public_forward_semantics(invalid):
    instance = model()
    with ExitStack() as stack:
        sessions = [
            stack.enter_context(
                instance.create_generation_session(
                    torch.tensor([ids]),
                    max_new_tokens=5,
                    do_sample=False,
                    eos_token_id=[],
                )
            )
            for ids in ([4, 5], [6, 7, 8])
        ]
        for item in sessions:
            item.step()
        lengths = [s.past_key_values[0].length for s in sessions]
        caches = [
            _IndependentLayerCache(
                [s.past_key_values[layer] for s in sessions], torch.tensor(lengths)
            )
            for layer in range(2)
        ]
        kwargs = dict(
            input_ids=torch.tensor([[4], [5]]), past_key_values=caches, use_cache=True
        )
        if invalid == "training":
            instance.train()
        if invalid == "prefill":
            kwargs["_prefill_only"] = True
        if invalid == "mask":
            kwargs["attention_mask"] = torch.tensor([[0], [1]])
        if invalid == "labels":
            kwargs["labels"] = torch.tensor([[4], [5]])
        if invalid == "no_cache":
            kwargs["use_cache"] = False
        with torch.set_grad_enabled(invalid == "gradients"), pytest.raises(ValueError):
            instance(**kwargs)
        assert [s.past_key_values[0].length for s in sessions] == lengths


def test_independent_cache_view_is_single_use_and_rejects_stale_positions():
    instance = model()
    with ExitStack() as stack, torch.inference_mode():
        sessions = [
            stack.enter_context(
                instance.create_generation_session(
                    torch.tensor([ids]),
                    max_new_tokens=5,
                    do_sample=False,
                    eos_token_id=[],
                )
            )
            for ids in ([4, 5], [6, 7, 8])
        ]
        for item in sessions:
            item.step()
        positions = torch.tensor([s.past_key_values[0].length for s in sessions])
        caches = [
            _IndependentLayerCache(
                [s.past_key_values[layer] for s in sessions], positions
            )
            for layer in range(2)
        ]
        ids = torch.tensor([[4], [5]])
        instance(ids, past_key_values=caches, use_cache=True)
        lengths = [s.past_key_values[0].length for s in sessions]
        with pytest.raises(ValueError, match="independent KV"):
            instance(ids, past_key_values=caches, use_cache=True)
        assert [s.past_key_values[0].length for s in sessions] == lengths
