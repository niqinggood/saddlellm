"""Cohort decode correctness, membership changes, KV ownership and fallbacks."""
from contextlib import ExitStack

import pytest
import torch

from saddlellm.models.NativeBatchDecode import NativeBatchDecodeRunner
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def model(kind="gqa", backend="sdpa", **kwargs):
    torch.manual_seed(173)
    return SaddleForCausalLM(SaddleModelConfig(
        vocab_size=37, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4 if kind == "mha" else 1 if kind == "mqa" else 2,
        max_position_embeddings=128, attention_kind=kind, attention_backend=backend,
        eos_token_id=[], **kwargs,
    )).eval()


def session(instance, ids, **kwargs):
    return instance.create_generation_session(
        torch.tensor([ids]), max_new_tokens=kwargs.pop("max_new_tokens", 8),
        do_sample=kwargs.pop("do_sample", False), output_scores=True, **kwargs,
    )


def drain(runner, sessions):
    while live := [item for item in sessions if not item.done]:
        runner.step(live)


def test_permuted_arrivals_reuse_the_same_cohort():
    instance = model()
    prompts = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
    reference = [instance.generate(torch.tensor([prompt]), max_new_tokens=8, do_sample=False)
                 for prompt in prompts]
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, prompt)) for prompt in prompts]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        for tick in range(8):
            order = sessions[tick % 3:] + sessions[:tick % 3]
            runner.step(order if tick % 2 else list(reversed(order)))
        assert runner.snapshot()["cohort_builds"] == 1
        assert runner.snapshot()["batched_forwards"] == 7
        assert all(torch.equal(item.sequences(), ref) for item, ref in zip(sessions, reference))


@pytest.mark.parametrize("kind", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("backend", ["eager", "sdpa", "sdpa_reference"])
def test_batch_decode_matches_individual_tokens_and_logits(kind, backend):
    instance = model(kind, backend)
    prompts = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
    references = [instance.generate(torch.tensor([prompt]), max_new_tokens=8, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True) for prompt in prompts]
    shapes = []
    hook = instance.register_forward_pre_hook(
        lambda _, args, kwargs: shapes.append(tuple(kwargs["input_ids"].shape)), with_kwargs=True)
    try:
        with ExitStack() as stack:
            sessions = [stack.enter_context(session(instance, prompt)) for prompt in prompts]
            runner = stack.enter_context(NativeBatchDecodeRunner(instance))
            drain(runner, sessions)
            assert shapes.count((1, 3)) == 3  # Prefill remains individual.
            assert shapes.count((3, 1)) == 7  # Genuine B3 decode forwards.
            assert len(shapes) == 10
            assert runner.snapshot()["cohort_builds"] == 1
            assert runner.snapshot()["batched_tokens"] == 21
            for actual, expected in zip(sessions, references):
                assert torch.equal(actual.sequences(), expected.sequences)
                assert actual.finish_reason == "length"
                for score, reference_score in zip(actual.scores, expected.scores):
                    torch.testing.assert_close(score, reference_score, rtol=1e-4, atol=1e-6)
    finally:
        hook.remove()


def test_stable_cohort_reuses_views_and_does_not_recopy_history_each_token():
    instance = model()
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, ids)) for ids in ([4, 5, 6, 7], [8, 9, 10, 11])]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions)
        runner.step(sessions)
        views = [[value for layer in s.past_key_values for value in layer.buffers] for s in sessions]
        copied = runner.snapshot()["cohort_history_copy_bytes"]
        runner.step(sessions)
        runner.step(sessions)
        assert runner.snapshot()["cohort_history_copy_bytes"] == copied > 0
        for s, old_views in zip(sessions, views):
            assert all(new is old for new, old in zip(
                [value for layer in s.past_key_values for value in layer.buffers], old_views))
        for layer in range(2):
            a, b = (s.past_key_values[layer].buffers[0] for s in sessions)
            assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
            assert a.storage_offset() != b.storage_offset()  # Disjoint row ownership.


def test_finished_rows_leave_new_session_joins_and_request_policies_stay_private():
    instance = model()
    plans = [([4, 5], 2, 1.0), ([6, 7], 7, 1.1), ([8, 9, 10, 11], 5, 1.0)]
    references = [instance.generate(torch.tensor([ids]), max_new_tokens=count, do_sample=False,
                                   repetition_penalty=penalty) for ids, count, penalty in plans]
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, ids, max_new_tokens=count, repetition_penalty=penalty))
                    for ids, count, penalty in plans]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions[:2])
        runner.step(sessions[:2])
        assert sessions[0].done
        runner.step(sessions[1:])  # Old row decodes; new row prefills 4 tokens.
        assert sessions[1].past_key_values[0].length == sessions[2].past_key_values[0].length == 4
        runner.step(sessions[1:])  # Membership has changed, B2 joins again.
        assert runner.snapshot()["cohort_builds"] == 2
        drain(runner, sessions[1:])
        for actual, expected in zip(sessions, references):
            assert torch.equal(actual.sequences(), expected)


def test_per_request_stop_and_eos_are_not_shared():
    instance = model()
    ids = [[4, 5], [6, 7], [8, 9]]
    eos = int(instance.generate(torch.tensor([ids[0]]), max_new_tokens=2, do_sample=False)[0, -1])
    policies = [{"eos_token_id": [eos]}, {"stopping_criteria": lambda tokens, scores: tokens.shape[1] >= 5}, {}]
    reference = [instance.generate(torch.tensor([prompt]), max_new_tokens=8, do_sample=False, **policy)
                 for prompt, policy in zip(ids, policies)]
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, prompt, **policy)) for prompt, policy in zip(ids, policies)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        drain(runner, sessions)
        for actual, expected in zip(sessions, reference):
            assert torch.equal(actual.sequences(), expected)
        assert sessions[1].generated_tokens == 3
        assert sessions[2].generated_tokens == 8


@pytest.mark.parametrize("mode", ["padding", "no_cache", "dynamic", "window", "mla", "small_budget"])
def test_unsupported_workloads_keep_individual_semantics(mode):
    options = {"sliding_window": 4} if mode == "window" else {}
    instance = model(kind="mla" if mode == "mla" else "gqa", **options)
    kwargs = {}
    if mode == "padding":
        kwargs["attention_mask"] = torch.tensor([[0, 1, 1]])
    if mode == "no_cache":
        kwargs["use_cache"] = False
    if mode == "dynamic":
        kwargs["cache_implementation"] = "dynamic"
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, [4, 5, 6], **kwargs)) for _ in range(2)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance, max_batch_tokens=1 if mode == "small_budget" else 32768))
        drain(runner, sessions)
        for item in sessions:
            with session(instance, [4, 5, 6], **kwargs) as reference:
                list(reference)
                assert torch.equal(item.sequences(), reference.sequences())
        assert runner.snapshot()["batched_forwards"] == 0


def test_runner_failure_poisons_all_rows_not_partially_mutated_kv(monkeypatch):
    instance = model()
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, [4, 5])) for _ in range(2)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions)
        original = instance.layers[1].forward

        def broken(*args, **kwargs):
            raise RuntimeError("injected layer failure")

        monkeypatch.setattr(instance.layers[1], "forward", broken)
        with pytest.raises(RuntimeError, match="injected"):
            runner.step(sessions)
        monkeypatch.setattr(instance.layers[1], "forward", original)
        for item in sessions:
            with pytest.raises(RuntimeError, match="failed"):
                item.step()


def test_external_individual_step_invalidates_cohort_without_corrupting_peers():
    instance = model()
    reference = instance.generate(torch.tensor([[4, 5]]), max_new_tokens=8, do_sample=False)
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, [4, 5])) for _ in range(2)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions)
        runner.step(sessions)
        sessions[0].step()
        runner.step(sessions)  # Divergent lengths must not reuse the old cohort.
        sessions[1].step()
        drain(runner, sessions)
        for item in sessions:
            assert torch.equal(item.sequences(), reference)


@pytest.mark.parametrize("split", [1, 2])
@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32),
    pytest.param("cuda", torch.bfloat16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
])
def test_interleaved_disjoint_groups_reuse_history(split, device, dtype):
    instance = model().to(device=device, dtype=dtype)
    prompts = [[4 + index, 8 + index, 12 + index] for index in range(4)]
    references = [instance.generate(torch.tensor([ids], device=device), max_new_tokens=8, do_sample=False)
                  for ids in prompts]
    with ExitStack() as stack:
        sessions = [stack.enter_context(instance.create_generation_session(
            torch.tensor([ids], device=device), max_new_tokens=8, do_sample=False)) for ids in prompts]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance, max_batch_size=4))
        for item in sessions:
            item.step()
        groups = [sessions[:split], sessions[split:]]
        for group in groups:
            runner.step(group)
        copied = runner.history_copy_bytes
        for _ in range(6):
            for group in groups:
                runner.step(list(reversed(group)))
            assert runner.history_copy_bytes == copied
        assert runner.cohort_builds == (1 if split == 1 else 2)
        assert runner.snapshot()["retained_cohorts"] == 0
        for actual, expected in zip(sessions, references):
            assert torch.equal(actual.sequences(), expected)


def test_retained_cohorts_are_bounded_and_eviction_keeps_sessions_valid():
    instance = model()
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, [4, 5, 6])) for _ in range(6)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance, max_batch_size=4,
                                                              max_batch_tokens=24))
        for item in sessions:
            item.step()
        for group in (sessions[:2], sessions[2:4], sessions[4:]):
            runner.step(group)
            assert runner.snapshot()["retained_cohort_token_slots"] <= 24
            assert sum(len(key) for key in runner._cohorts) <= 4
        assert runner.snapshot()["retained_cohorts"] == 2
        assert not any(sessions[0] in key for key in runner._cohorts)
        for _ in range(6):
            for group in (sessions[:2], sessions[2:4], sessions[4:]):
                runner.step(group)
                assert runner.snapshot()["retained_cohort_token_slots"] <= 24
        expected = instance.generate(torch.tensor([[4, 5, 6]]), max_new_tokens=8, do_sample=False)
        assert all(torch.equal(item.sequences(), expected) for item in sessions)


def test_cross_group_rejoin_forget_and_closed_peer_release_retained_groups():
    instance = model()
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, [4, 5, 6])) for _ in range(4)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance, max_batch_size=4))
        for item in sessions:
            item.step()
        runner.step(sessions[:2])
        runner.step(sessions[2:])
        runner.step([sessions[0], sessions[2]])
        assert runner.snapshot()["retained_cohorts"] == 1
        runner.forget(sessions[0])
        assert runner.snapshot()["retained_cohorts"] == 0
        runner.step([sessions[1], sessions[3]])
        sessions[1].close()
        runner.step([sessions[0]])
        assert runner.snapshot()["retained_cohorts"] == 0
        for item in (sessions[0], sessions[2], sessions[3]):
            list(item)
            expected = instance.generate(torch.tensor([[4, 5, 6]]), max_new_tokens=8, do_sample=False)
            assert torch.equal(item.sequences(), expected)


def test_rejections_and_close_preserve_session_ownership():
    instance = model()
    with session(instance, [4, 5], max_new_tokens=3) as item:
        runner = NativeBatchDecodeRunner(instance)
        with pytest.raises(ValueError, match="unique"):
            runner.step([item, item])
        assert item.generated_tokens == 0
        assert runner.step([]) == []
        item._lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="already executing"):
                runner.step([item])
        finally:
            item._lock.release()
        runner.step([item])
        runner.close()
        runner.close()
        with pytest.raises(RuntimeError, match="closed"):
            runner.step([item])
        assert not item.closed
        list(item)
        with NativeBatchDecodeRunner(instance) as other:
            with pytest.raises(ValueError, match="finished"):
                other.step([item])


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits(value):
    with pytest.raises(ValueError):
        NativeBatchDecodeRunner(model(), max_batch_size=value)
    with pytest.raises(ValueError):
        NativeBatchDecodeRunner(model(), max_batch_tokens=value)


def test_public_lazy_export():
    from saddlellm import NativeBatchDecodeRunner as exported

    assert exported is NativeBatchDecodeRunner


def test_different_context_lengths_keep_outputs_and_use_no_padding_batch():
    instance = model()
    prompts = [[4], [5, 6, 7, 8]]
    references = [instance.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False) for ids in prompts]
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, ids)) for ids in prompts]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        drain(runner, sessions)
        assert runner.snapshot()["independent_kv_forwards"] == 7
        assert runner.snapshot()["cohort_history_copy_bytes"] == 0
        for actual, expected in zip(sessions, references):
            assert torch.equal(actual.sequences(), expected)


@pytest.mark.parametrize("mode", ["moe", "custom_forward", "custom_prepare", "int8_marker"])
def test_unvalidated_model_paths_are_not_batched(mode, monkeypatch):
    instance = model(**({"num_experts": 2, "num_experts_per_tok": 1} if mode == "moe" else {}))
    if mode == "custom_forward":
        original = instance.forward
        monkeypatch.setattr(instance, "forward", lambda *args, **kwargs: original(*args, **kwargs))
    if mode == "custom_prepare":
        original = instance.prepare_inputs_for_generation
        monkeypatch.setattr(instance, "prepare_inputs_for_generation", lambda *args, **kwargs: original(*args, **kwargs))
    if mode == "int8_marker":
        # Do not claim W8A16 coverage from a float test: only check routing here.
        monkeypatch.setattr(instance, "_saddle_cuda_int8", {}, raising=False)
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, [4, 5])) for _ in range(2)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        drain(runner, sessions)
        assert runner.snapshot()["batched_forwards"] == 0


def test_cancelled_row_can_close_while_its_peer_keeps_decoding():
    instance = model()
    expected = instance.generate(torch.tensor([[6, 7]]), max_new_tokens=8, do_sample=False)
    with ExitStack() as stack:
        cancelled = stack.enter_context(session(instance, [4, 5]))
        survivor = stack.enter_context(session(instance, [6, 7]))
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step([cancelled, survivor])
        runner.step([cancelled, survivor])
        cancelled.close()
        drain(runner, [survivor])
        assert runner.snapshot()["retained_cohorts"] == 0
        assert torch.equal(survivor.sequences(), expected)


def test_streamer_failure_poisoning_and_end_once():
    instance = model()

    class Sink:
        puts = 0
        ends = 0

        def put(self, value):
            self.puts += 1
            if self.puts == 3:  # Prompt, first generated token, then batched decode.
                raise RuntimeError("stream delivery failed")

        def end(self):
            self.ends += 1

    sink = Sink()
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, [4, 5], streamer=sink)),
                    stack.enter_context(session(instance, [6, 7]))]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions)
        with pytest.raises(RuntimeError, match="stream delivery failed"):
            runner.step(sessions)
        for item in sessions:
            with pytest.raises(RuntimeError, match="failed"):
                item.step()
    assert sink.ends == 1


def test_invalid_member_releases_previously_acquired_session_locks():
    instance = model()
    other_model = model()
    with session(instance, [4, 5]) as first, session(other_model, [6, 7]) as other:
        with NativeBatchDecodeRunner(instance, max_batch_size=2) as runner:
            with pytest.raises(ValueError, match="runner model"):
                runner.step([first, other])
            assert first.generated_tokens == 0
            first.step()  # The first lock must have been released on validation failure.
            with pytest.raises(ValueError, match="max_batch_size"):
                runner.step([first, other, object()])
            assert first.generated_tokens == 1


def test_cohort_decode_does_not_mutate_reusable_prefix_snapshots():
    from saddlellm.models.NativePrefixCache import NativePrefixCache

    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    prompt = [4, 5, 6]
    with session(instance, prompt, prefix_cache=cache) as warmup:
        list(warmup)
        expected = warmup.sequences()
    entry = cache._entries[tuple(prompt)]
    originals = [value.clone() for layer in entry.keys_values for value in layer]
    original_logits = entry.logits.clone()
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, prompt, prefix_cache=cache)) for _ in range(2)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        drain(runner, sessions)
        assert runner.snapshot()["batched_forwards"] == 7
        for item in sessions:
            assert item.prefix_cached_tokens == len(prompt)
            assert torch.equal(item.sequences(), expected)
    for actual, original in zip([value for layer in entry.keys_values for value in layer], originals):
        assert torch.equal(actual, original)
    assert torch.equal(entry.logits, original_logits)


def test_performance_gate_requires_stable_speed_and_correctness():
    from tools.benchmark_native_batch_decode import summarize

    def records(speed=2.0, mismatch=False, memory=None):
        return [{"pair": pair, "variant": kind, "seconds": 2.0 if kind == "serial" else 2.0 / speed,
                 "output_tokens": 10, "first_round_seconds": 0.1, "peak_allocated_bytes": memory,
                 "runner": {"batched_forwards": 7} if kind == "cohort" else None,
                 "tokens": [[1, 2, 3 if mismatch and kind == "cohort" else 2]]}
                for pair in range(7) for kind in ("serial", "cohort")]

    result = summarize(records())
    assert result["development_gate_passed"]
    assert result["checks"]["cuda_allocated_growth_within_five_percent"] is None
    assert result["serial"]["max_peak_allocated_bytes"] is None
    assert not summarize(records(mismatch=True))["development_gate_passed"]
    assert not summarize(records(speed=1.02))["development_gate_passed"]
    assert not summarize(records()[:12])["development_gate_passed"]
    inflated = records(memory=100)
    for row in inflated:
        if row["variant"] == "cohort":
            row["peak_allocated_bytes"] = 106
    assert not summarize(inflated)["development_gate_passed"]
    tail_slow = records()
    tail_slow[-1]["seconds"] = 2.11
    assert not summarize(tail_slow)["development_gate_passed"]
    no_batch = records()
    no_batch[-1]["runner"]["batched_forwards"] = 0
    assert not summarize(no_batch)["development_gate_passed"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_low_precision_keeps_fallback(dtype):
    instance = model().to(dtype=dtype)
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, [4, 5])) for _ in range(2)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        drain(runner, sessions)
        assert runner.snapshot()["batched_forwards"] == 0
        assert runner.snapshot()["rowwise_linear_forwards"] == 0
        expected = instance.generate(torch.tensor([[4, 5]]), max_new_tokens=8, do_sample=False)
        for item in sessions:
            assert torch.equal(item.sequences(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["eager", "sdpa", "sdpa_reference"])
@pytest.mark.parametrize("architecture", ["gqa", "mha", "mqa", "qk_norm", "qkv_bias", "packed"])
def test_cuda_bf16_uses_rowwise_linear_inside_batched_forward(backend, architecture):
    options = {"qk_norm": True} if architecture == "qk_norm" else {}
    if architecture == "qkv_bias":
        options["qkv_bias"] = True
    if architecture == "packed":
        options.update(fused_qkv=True, fused_gate_up=True)
    instance = model(kind=architecture if architecture in {"mha", "mqa"} else "gqa",
                     backend=backend, **options).to(device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[4, 5, 6], [7, 8, 9], [11, 12, 13]], device="cuda")
    expected = [instance.generate(row[None], max_new_tokens=12, do_sample=False) for row in ids]
    parameter_ids = [id(parameter) for parameter in instance.parameters()]
    matmul_policy = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    inputs, attention_inputs = [], []
    original_linear_forward = instance.layers[0].self_attn.qkv_packed_proj if architecture == "packed" else instance.layers[0].self_attn.q_proj
    hooks = [original_linear_forward.register_forward_pre_hook(lambda _, args: inputs.append(tuple(args[0].shape))),
             instance.layers[0].self_attn.register_forward_pre_hook(lambda _, args: attention_inputs.append(tuple(args[0].shape)))]
    try:
        with ExitStack() as stack:
            sessions = [stack.enter_context(instance.create_generation_session(
                row[None], max_new_tokens=12, do_sample=False)) for row in ids]
            runner = stack.enter_context(NativeBatchDecodeRunner(instance))
            drain(runner, sessions)
            assert runner.snapshot()["rowwise_linear_forwards"] == 11
            assert inputs.count((1, 1, 32)) == 33
            assert (3, 1, 32) not in inputs
            assert attention_inputs.count((3, 1, 32)) == 11
            for actual, reference in zip(sessions, expected):
                assert torch.equal(actual.sequences(), reference)
            assert original_linear_forward.weight.dtype == torch.bfloat16
            assert parameter_ids == [id(parameter) for parameter in instance.parameters()]
            assert matmul_policy == torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    finally:
        for hook in hooks:
            hook.remove()


def test_rowwise_inference_flag_rejects_training_or_cpu_without_changing_cache():
    instance = model()
    with session(instance, [4, 5]) as item:
        item.step()
        lengths = [cache.length for cache in item.past_key_values]
        with torch.inference_mode(), pytest.raises(ValueError, match="CUDA BF16"):
            instance(torch.tensor([[6]]), past_key_values=item.past_key_values,
                     use_cache=True, _decode_rowwise_linear=True)
        assert [cache.length for cache in item.past_key_values] == lengths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autocast_is_not_mistaken_for_pure_bf16_model():
    instance = model().cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16), ExitStack() as stack:
        sessions = [stack.enter_context(instance.create_generation_session(
            torch.tensor([[4, 5]], device="cuda"), max_new_tokens=4, do_sample=False)) for _ in range(2)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        drain(runner, sessions)
        assert runner.snapshot()["batched_forwards"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_membership_changes_keep_request_penalties_and_stops():
    instance = model().to(device="cuda", dtype=torch.bfloat16)
    plans = [([4, 5], 2, 1.0), ([6, 7], 7, 1.1), ([8, 9, 10, 11], 5, 1.0)]
    expected = [instance.generate(torch.tensor([ids], device="cuda"), max_new_tokens=count,
                                 repetition_penalty=penalty, do_sample=False) for ids, count, penalty in plans]
    with ExitStack() as stack:
        sessions = [stack.enter_context(instance.create_generation_session(
            torch.tensor([ids], device="cuda"), max_new_tokens=count, repetition_penalty=penalty,
            do_sample=False)) for ids, count, penalty in plans]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions[:2])
        runner.step(sessions[:2])
        assert sessions[0].done
        runner.step(sessions[1:])
        drain(runner, sessions[1:])
        assert runner.snapshot()["cohort_builds"] == 2
        assert runner.snapshot()["rowwise_linear_forwards"] > 1
        for actual, reference in zip(sessions, expected):
            assert torch.equal(actual.sequences(), reference)


def test_partial_join_allocation_failure_preserves_valid_histories(monkeypatch):
    instance = model()
    prompt = [4, 5, 6]
    expected = instance.generate(torch.tensor([prompt]), max_new_tokens=8, do_sample=False)
    with ExitStack() as stack:
        sessions = [stack.enter_context(session(instance, prompt)) for _ in range(2)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions)
        histories = [[tuple(value[:, :, :cache.length].clone() for value in cache.buffers)
                      for cache in item.past_key_values] for item in sessions]
        original, allocations = torch.Tensor.new_empty, 0

        def fail_second_layer(tensor, shape, *args, **kwargs):
            nonlocal allocations
            if len(shape) == 4 and shape[0] == 2:
                allocations += 1
                if allocations == 3:
                    raise torch.OutOfMemoryError("injected join OOM")
            return original(tensor, shape, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "new_empty", fail_second_layer)
            with pytest.raises(torch.OutOfMemoryError, match="join OOM"):
                runner.step(sessions)
        for item, history in zip(sessions, histories):
            assert item.generated_tokens == 1 and not item._failed
            for cache, old in zip(item.past_key_values, history):
                assert cache.length == len(prompt)
                for value, previous in zip(cache.buffers, old):
                    assert torch.equal(value[:, :, :cache.length], previous)
        drain(runner, sessions)
        for item in sessions:
            assert torch.equal(item.sequences(), expected)
