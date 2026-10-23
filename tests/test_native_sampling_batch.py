"""Sampling state, numerical execution modes and diagnostic counters."""

from contextlib import ExitStack

import pytest
import torch

from saddlellm.models.NativeBatchDecode import NativeBatchDecodeRunner
from saddlellm.runtime.InferenceServer import InferenceServerSettings
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
@pytest.mark.parametrize("packed", [False, True])
def test_seeded_samples_match_serial_despite_arrivals_departures_and_order(
    device, packed
):
    instance = model(fused_qkv=packed, fused_gate_up=packed).to(
        device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    plans = [
        (
            [4, 5],
            2,
            dict(seed=71, temperature=0.7, top_k=7, top_p=0.85, repetition_penalty=1.1),
        ),
        (
            [6, 7],
            9,
            dict(seed=82, temperature=1.2, top_k=0, top_p=1, repetition_penalty=1),
        ),
        (
            [8, 9, 10, 11],
            7,
            dict(
                seed=93, temperature=0.5, top_k=12, top_p=0.95, repetition_penalty=1.2
            ),
        ),
    ]

    def create(plan):
        ids, count, kwargs = plan
        return instance.create_generation_session(
            torch.tensor([ids], device=device),
            max_new_tokens=count,
            do_sample=True,
            eos_token_id=[],
            output_scores=True,
            **kwargs,
        )

    references = []
    for plan in plans:
        with create(plan) as item:
            list(item)
            references.append(
                (item.sequences().clone(), [score.clone() for score in item.scores])
            )
    with ExitStack() as stack:
        sessions = [stack.enter_context(create(plan)) for plan in plans]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions[:2])
        runner.step(sessions[:2])
        assert sessions[0].done
        runner.forget(sessions[0])
        runner.step(sessions[1:])  # New prompt joins at the survivor's KV length.
        tick = 0
        while live := [item for item in sessions if not item.done]:
            runner.step(live if tick % 2 else list(reversed(live)))
            tick += 1
        assert runner.snapshot()["sampling_batched_tokens"] > 0
        for actual, (tokens, scores) in zip(sessions, references):
            assert torch.equal(actual.sequences(), tokens)
            for observed, expected in zip(actual.scores, scores):
                torch.testing.assert_close(
                    observed,
                    expected,
                    rtol=0 if device == "cuda" else 1e-4,
                    atol=0 if device == "cuda" else 1e-6,
                )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shared", [False, True])
def test_global_or_shared_rng_consumes_in_submitted_order(device, shared):
    instance = model().to(
        device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )

    def run(batched):
        torch.manual_seed(451)
        generator = torch.Generator(device=device).manual_seed(451) if shared else None
        with ExitStack() as stack:
            sessions = [
                stack.enter_context(
                    instance.create_generation_session(
                        torch.tensor([[4 + i, 8 + i]], device=device),
                        max_new_tokens=7,
                        do_sample=True,
                        generator=generator,
                        temperature=0.7,
                        eos_token_id=[],
                    )
                )
                for i in range(3)
            ]
            runner = stack.enter_context(NativeBatchDecodeRunner(instance))
            for tick in range(7):
                order = sessions[tick % 3 :] + sessions[: tick % 3]
                if batched:
                    runner.step(order)
                else:
                    for item in order:
                        item.step()
            return [item.sequences().clone() for item in sessions]

    assert all(torch.equal(a, b) for a, b in zip(run(False), run(True)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mixed", [False, True])
def test_sampled_stop_eos_and_greedy_peer_remain_private(device, mixed):
    instance = model().to(
        device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    ids = torch.tensor([[4, 5]], device=device)
    with instance.create_generation_session(
        ids, max_new_tokens=3, do_sample=True, seed=3, eos_token_id=[]
    ) as probe:
        list(probe)
        eos = int(probe.sequences()[0, -1])
    policies = [
        dict(do_sample=True, seed=3, eos_token_id=[eos], min_new_tokens=1),
        dict(
            do_sample=True,
            seed=4,
            eos_token_id=[],
            stopping_criteria=lambda ids, scores: ids.shape[1] >= 6,
        ),
        dict(do_sample=False, eos_token_id=[]),
    ]
    prompts = (
        [
            ids,
            torch.tensor([[4, 5, 6]], device=device),
            torch.tensor([[4, 5, 6, 7]], device=device),
        ]
        if mixed
        else [ids] * 3
    )
    expected = [
        instance.generate(prompt, max_new_tokens=9, **policy)
        for prompt, policy in zip(prompts, policies)
    ]
    with ExitStack() as stack:
        sessions = [
            stack.enter_context(
                instance.create_generation_session(prompt, max_new_tokens=9, **p)
            )
            for prompt, p in zip(prompts, policies)
        ]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        while live := [item for item in sessions if not item.done]:
            runner.step(live)
        assert runner.sampling_batched_tokens > 0
        assert (
            sessions[1].generated_tokens == (3 if mixed else 4)
            and sessions[2].generated_tokens == 9
        )
        assert all(
            torch.equal(item.sequences(), ref) for item, ref in zip(sessions, expected)
        )


@pytest.mark.parametrize("decode_mode", ["compatible", "throughput"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_mode_controls_real_projection_shapes_without_changing_weights(
    decode_mode, packed
):
    instance = model(fused_qkv=packed, fused_gate_up=packed).cuda().bfloat16()
    weights = [(id(p), p.clone()) for p in instance.parameters()]
    policy = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    projection = (
        instance.layers[0].self_attn.qkv_packed_proj
        if packed
        else instance.layers[0].self_attn.q_proj
    )
    shapes = []
    hook = projection.register_forward_pre_hook(
        lambda module, args: shapes.append(tuple(args[0].shape))
    )
    try:
        with ExitStack() as stack:
            sessions = [
                stack.enter_context(
                    instance.create_generation_session(
                        torch.tensor([[4 + i, 8 + i]], device="cuda"),
                        max_new_tokens=4,
                        do_sample=True,
                        seed=10 + i,
                        temperature=0.7,
                        eos_token_id=[],
                        output_scores=True,
                    )
                )
                for i in range(3)
            ]
            runner = stack.enter_context(
                NativeBatchDecodeRunner(instance, decode_mode=decode_mode)
            )
            runner.step(sessions)
            shapes.clear()
            runner.step(sessions)
            assert shapes == (
                [(1, 1, 32)] * 3 if decode_mode == "compatible" else [(3, 1, 32)]
            )
            assert all(torch.isfinite(item.scores[-1]).all() for item in sessions)
            status = runner.snapshot()
            assert status["rowwise_linear_forwards"] == int(decode_mode == "compatible")
            assert status["batched_linear_forwards"] == int(decode_mode == "throughput")
            assert status["sampling_batched_tokens"] == 3
    finally:
        hook.remove()
    assert policy == torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    assert all(
        id(p) == ident and torch.equal(p, old)
        for p, (ident, old) in zip(instance.parameters(), weights)
    )


def test_fallback_counters_explain_length_and_capacity_constraints():
    instance = model()
    with ExitStack() as stack:
        # Model hooks keep the dense-cache forward contract, so this model
        # deliberately cannot use the independent-KV path for unequal lengths.
        hook = instance.register_forward_pre_hook(lambda *args: None)
        stack.callback(hook.remove)
        sessions = [
            stack.enter_context(
                instance.create_generation_session(
                    torch.tensor([ids]),
                    max_new_tokens=4,
                    do_sample=True,
                    seed=5,
                    eos_token_id=[],
                )
            )
            for ids in ([4, 5], [4, 5, 6])
        ]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions)
        runner.step(sessions)
        assert runner.snapshot()["individual_reasons"] == {
            "prefill_or_pending_logits": 2,
            "no_compatible_peer": 2,
        }
        assert runner.snapshot()["sampling_batched_tokens"] == 0
        limited = stack.enter_context(
            NativeBatchDecodeRunner(instance, max_batch_tokens=1)
        )
        sessions[0].step()
        limited.step(sessions)
        assert limited.snapshot()["individual_reasons"] == {"batch_token_budget": 2}


@pytest.mark.parametrize("mode", [None, True, "fast", "", 1])
def test_invalid_modes_fail_before_execution(mode):
    with pytest.raises(ValueError, match="decode_mode"):
        NativeBatchDecodeRunner(model(), decode_mode=mode)
    with pytest.raises(ValueError, match="native_decode_mode"):
        InferenceServerSettings(native_decode_mode=mode)


def test_throughput_service_requires_native_step_batching():
    with pytest.raises(ValueError, match="requires native_stream_batching"):
        InferenceServerSettings(native_decode_mode="throughput")
