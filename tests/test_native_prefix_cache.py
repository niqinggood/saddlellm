import asyncio
import json
import random
import subprocess
import sys
import threading
from pathlib import Path

import httpx
import pytest
import torch

from saddlellm.models.NativePrefixCache import NativePrefixCache
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.InferenceServer import (
    CompletionRequest,
    InferenceServerSettings,
    LocalGenerationEngine,
    create_inference_app,
)


def model(kind="gqa", backend="sdpa", **settings):
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
            eos_token_id=[],
            **settings,
        )
    ).eval()


def run(instance, ids, cache=None, **kwargs):
    with instance.create_generation_session(
        ids,
        max_new_tokens=4,
        do_sample=False,
        prefix_cache=cache,
        output_scores=True,
        **kwargs,
    ) as session:
        session.prefill()
        reused, computed = session.prefix_cached_tokens, session.prefill_computed_tokens
        list(session)
        return session.sequences(), session.scores, reused, computed


@pytest.mark.parametrize("kind", ["gqa", "mha", "mqa"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_full_partial_and_shorter_prefix_preserve_outputs(kind, backend):
    instance = model(kind, backend)
    cache = NativePrefixCache(65536, min_prefix_tokens=2, reuse_partial_prefixes=True)
    original = torch.tensor([[4, 5, 6, 7, 8]])
    assert run(instance, original, cache)[2:] == (0, 5)
    for ids, expected in (
        (original, (5, 0)),
        (torch.tensor([[4, 5, 6, 9, 10]]), (3, 2)),
        (torch.tensor([[4, 5, 6]]), (2, 1)),
    ):
        reference = run(instance, ids)
        actual = run(instance, ids, cache)
        assert actual[2:] == expected
        assert torch.equal(actual[0], reference[0])
        for score, ref_score in zip(actual[1], reference[1]):
            torch.testing.assert_close(score, ref_score, atol=2e-6, rtol=1e-4)


def test_snapshot_is_compact_immutable_and_request_kv_is_private():
    instance = model(qk_norm=True, fused_qkv=True, fused_gate_up=True)
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7, 8]])
    run(instance, ids, cache)
    entry = cache._entries[tuple(ids[0].tolist())]
    originals = [value.clone() for row in entry.keys_values for value in row]
    original_logits = entry.logits.clone()
    storage_bytes = sum(
        value.untyped_storage().nbytes() for row in entry.keys_values for value in row
    )
    storage_bytes += entry.logits.untyped_storage().nbytes()
    assert (
        storage_bytes == entry.size_bytes == cache.snapshot()["retained_tensor_bytes"]
    )
    with instance.create_generation_session(
        ids,
        max_new_tokens=8,
        do_sample=False,
        prefix_cache=cache,
        repetition_penalty=1.3,
    ) as session:
        session.prefill()
        for request_layer, saved_values in zip(
            session.past_key_values, entry.keys_values
        ):
            assert request_layer.buffers[0].data_ptr() != saved_values[0].data_ptr()
        session.step()
        list(session)
    for original, value in zip(
        originals, (v for row in entry.keys_values for v in row)
    ):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    torch.testing.assert_close(entry.logits, original_logits, rtol=0, atol=0)
    assert session.prefix_cache is None and session.past_key_values is None


def test_sampling_and_eos_processing_do_not_poison_cached_logits():
    instance = model()
    ids = torch.tensor([[4, 5, 6, 7]])
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    kwargs = dict(
        max_new_tokens=5,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        min_new_tokens=2,
        eos_token_id=[4, 5],
        seed=311,
    )
    reference = instance.generate(ids, **kwargs)
    for _ in range(3):
        actual = instance.generate(ids, prefix_cache=cache, **kwargs)
        assert torch.equal(actual, reference)
    assert cache.snapshot()["hits"] == 2


def test_lru_respects_bytes_and_entry_limit():
    instance = model()
    ids = [torch.tensor([[first, 5, 6, 7]]) for first in (4, 8, 9)]
    probe = NativePrefixCache(65536, min_prefix_tokens=2)
    run(instance, ids[0], probe)
    size = probe.snapshot()["retained_tensor_bytes"]
    cache = NativePrefixCache(2 * size, min_prefix_tokens=2, max_entries=2)
    run(instance, ids[0], cache)
    run(instance, ids[1], cache)
    run(instance, ids[0], cache)  # touch the oldest entry
    run(instance, ids[2], cache)
    assert tuple(ids[0][0].tolist()) in cache._entries
    assert tuple(ids[1][0].tolist()) not in cache._entries
    stats = cache.snapshot()
    assert stats["retained_tensor_bytes"] == 2 * size and stats["entries"] == 2
    assert stats["evictions"] == 1 and stats["stores"] == 3
    tiny = NativePrefixCache(size - 1, min_prefix_tokens=2)
    run(instance, ids[0], tiny)
    assert tiny.snapshot()["entries"] == 0
    assert tiny.snapshot()["rejected_entries"] == 1
    one_entry = NativePrefixCache(65536, min_prefix_tokens=2, max_entries=1)
    for prompt in ids:
        run(instance, prompt, one_entry)
    assert one_entry.snapshot()["entries"] == 1


@pytest.mark.parametrize(
    "change", ["parameter", "load", "dtype", "config", "other_model"]
)
def test_model_updates_invalidate_without_reusing_old_kv(change):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7]])
    run(instance, ids, cache)
    if change == "parameter":
        with torch.no_grad():
            instance.layers[0].self_attn.v_proj.weight.add_(0.2)
    elif change == "load":
        instance.load_state_dict(instance.state_dict())
    elif change == "dtype":
        instance.double()
    elif change == "config":
        instance.layers[0].config.sliding_window = 3
    else:
        instance = model()
    actual = run(instance, ids, cache)
    reference = run(instance, ids)
    assert torch.equal(actual[0], reference[0])
    assert actual[2] == 0 and cache.snapshot()["invalidations"] == 1


@pytest.mark.parametrize(
    "mode", ["padded", "batch", "dynamic", "uncached", "hook", "mla", "moe"]
)
def test_unsupported_paths_fall_back_without_changing_generation(mode):
    instance = (
        model("mla", kv_lora_rank=4, mla_cache_mode="latent")
        if mode == "mla"
        else model(ffn_kind="moe", num_experts=2, num_experts_per_tok=1)
        if mode == "moe"
        else model()
    )
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7]])
    kwargs = {}
    hook = None
    if mode == "padded":
        kwargs["attention_mask"] = torch.tensor([[0, 1, 1, 1]])
    elif mode == "batch":
        ids = ids.expand(2, -1)
    elif mode == "dynamic":
        kwargs["cache_implementation"] = "dynamic"
    elif mode == "uncached":
        kwargs["use_cache"] = False
    elif mode == "hook":
        hook = instance.norm.register_forward_hook(lambda *args: None)
    try:
        actual = run(instance, ids, cache, **kwargs)
        reference = run(instance, ids, **kwargs)
        assert torch.equal(actual[0], reference[0])
        assert actual[2] == 0 and cache.snapshot()["entries"] == 0
        assert cache.snapshot()["skipped_requests"] == 1
    finally:
        if hook is not None:
            hook.remove()


def test_clear_during_publication_and_failed_forward_do_not_store(monkeypatch):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    original_store = cache.store

    def clear_then_store(*args):
        cache.clear()
        original_store(*args)

    monkeypatch.setattr(cache, "store", clear_then_store)
    run(instance, torch.tensor([[4, 5, 6]]), cache)
    assert cache.snapshot()["entries"] == 0
    monkeypatch.setattr(cache, "store", original_store)
    session = instance.create_generation_session(
        torch.tensor([[4, 5, 999]]), prefix_cache=cache
    )
    with pytest.raises((IndexError, RuntimeError)):
        session.prefill()
    assert cache.snapshot()["entries"] == 0
    session.close()


def test_restore_oom_drops_cached_entries_and_rolls_back_partial_kv(monkeypatch):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7]])
    reference = run(instance, ids)[0]
    run(instance, ids, cache)
    with instance.create_generation_session(
        ids, max_new_tokens=4, do_sample=False, prefix_cache=cache
    ) as session:
        second = session.past_key_values[1]
        original_append = second.append
        calls = []

        def once_oom(*args):
            if not calls:
                calls.append(True)
                raise torch.cuda.OutOfMemoryError("synthetic restore pressure")
            return original_append(*args)

        monkeypatch.setattr(second, "append", once_oom)
        session.prefill()
        assert session.prefix_cached_tokens == 0
        list(session)
        assert torch.equal(session.sequences(), reference)
    assert cache.snapshot()["allocation_failures"] == 1
    assert cache.snapshot()["entries"] == 0


def test_clear_and_close_do_not_corrupt_an_active_request():
    instance = model()
    ids = torch.tensor([[4, 5, 6, 7]])
    reference = run(instance, ids)[0]
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    run(instance, ids, cache)
    with instance.create_generation_session(
        ids, max_new_tokens=4, do_sample=False, prefix_cache=cache
    ) as session:
        session.prefill()
        cache.clear()
        cache.close()
        list(session)
        assert torch.equal(session.sequences(), reference)
    cache.close()
    assert cache.snapshot()["entries"] == 0 and not cache.snapshot()["enabled"]
    assert run(instance, ids, cache)[2] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_cross_stream_reuse_and_partial_outputs(dtype):
    instance = model().to(device="cuda", dtype=dtype)
    cache = NativePrefixCache(65536, min_prefix_tokens=2, reuse_partial_prefixes=True)
    ids = torch.tensor([[4, 5, 6, 7, 8]], device="cuda")
    reference = run(instance, ids)[0]
    writer, reader = torch.cuda.Stream(), torch.cuda.Stream()
    # New streams must first wait for model/input initialization.
    writer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(writer):
        run(instance, ids, cache)
    reader.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(reader):
        actual = run(instance, ids, cache)
        partial_ids = torch.tensor([[4, 5, 6, 9, 10]], device="cuda")
        partial = run(instance, partial_ids, cache)
        partial_reference = run(instance, partial_ids)
    torch.cuda.synchronize()
    assert actual[2] == 5 and torch.equal(actual[0], reference)
    assert partial[2] == 3 and torch.equal(partial[0], partial_reference[0])


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        ids = [int(item) for item in text.split()][: kwargs["max_length"]]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def decode(self, ids, **kwargs):
        return " ".join(str(int(item)) for item in ids if int(item) not in {0, 2})


def test_service_nonstream_stream_health_and_shutdown():
    instance = model()
    settings = InferenceServerSettings(
        prefix_cache_mb=1, default_max_new_tokens=4, max_new_tokens=8
    )
    app = create_inference_app(settings, model=instance, tokenizer=Tokenizer())
    engine = app.state.engine
    prompt = " ".join(str(item) for item in [4, 5, 6, 7] * 5)
    request = CompletionRequest(prompt=prompt, temperature=0, max_tokens=4)
    reference = engine.complete(request)
    actual = engine.complete(request)
    assert (
        actual["choices"] == reference["choices"]
        and actual["usage"] == reference["usage"]
    )
    state = {}
    pieces = list(
        engine._stream_generate(
            prompt,
            max_tokens=4,
            temperature=0,
            top_p=1,
            repetition_penalty=1,
            stop=None,
            stream_state=state,
        )
    )
    assert "".join(pieces) == reference["choices"][0]["text"]
    assert state["completion_tokens"] == 4 and state["worker_done"].is_set()
    cancelled = {"cancel_event": threading.Event()}
    cancelled["cancel_event"].set()
    assert (
        list(
            engine._stream_generate(
                prompt,
                max_tokens=4,
                temperature=0,
                top_p=1,
                repetition_penalty=1,
                stop=None,
                stream_state=cancelled,
            )
        )
        == []
    )

    async def check():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                health = (await client.get("/health")).json()
                assert health["prefix_cache"]["hits"] == 2
                assert "tokens" not in health["prefix_cache"]
        assert engine.prefix_cache.snapshot()["entries"] == 0
        assert not engine.prefix_cache.snapshot()["enabled"]

    asyncio.run(check())


@pytest.mark.parametrize("value", [-1, True, 1.5, "64"])
def test_settings_reject_invalid_budget(value):
    with pytest.raises(ValueError, match="prefix_cache_mb"):
        InferenceServerSettings(prefix_cache_mb=value)


def test_default_disabled_and_cli_opt_in(monkeypatch):
    import uvicorn

    from saddlellm import NativePrefixCache as PublicCache
    from saddlellm.cli import main
    from saddlellm.runtime import InferenceServer

    assert PublicCache is NativePrefixCache
    assert (
        LocalGenerationEngine(
            model(), Tokenizer(), InferenceServerSettings()
        ).prefix_cache
        is None
    )
    settings = []
    monkeypatch.setattr(
        InferenceServer, "create_inference_app", lambda value: settings.append(value)
    )
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    assert main(["serve", "model", "--prefix-cache-mb", "64"]) == 0
    assert settings[0].prefix_cache_mb == 64


@pytest.mark.parametrize("args", [(True,), (-1,), (1, 0), (1, 2, 0)])
def test_cache_budget_validation(args):
    values = dict(zip(("max_bytes", "min_prefix_tokens", "max_entries"), args))
    with pytest.raises(ValueError):
        NativePrefixCache(**values)


def test_cpu_benchmark_uses_correct_hit_paths_and_checks_tokens(tmp_path):
    report_path = tmp_path / "prefix_benchmark.json"
    script = Path(__file__).resolve().parents[1] / "tools/benchmark_prefix_cache.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--prompt-tokens",
            "32",
            "--shared-tokens",
            "16",
            "--decode-tokens",
            "3",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["all_output_tokens_equal"]
    assert report["variants"]["full_hit"]["computed_prompt_tokens"] == 0
    assert report["variants"]["partial_hit"]["computed_prompt_tokens"] == 16
    assert not report["priming_included_in_timing"]


def test_store_oom_is_optional_and_never_breaks_generation(monkeypatch):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7]])
    reference = run(instance, ids)[0]
    original_clone = torch.Tensor.clone
    calls = []

    def fail_snapshot_once(tensor, *args, **kwargs):
        if tensor.ndim == 4 and not calls:
            calls.append(True)
            raise torch.cuda.OutOfMemoryError("synthetic snapshot pressure")
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", fail_snapshot_once)
    actual = run(instance, ids, cache)
    assert torch.equal(actual[0], reference)
    assert cache.snapshot()["entries"] == 0
    assert cache.snapshot()["allocation_failures"] == 1


def test_global_hooks_and_full_logits_bypass_prefix_reuse():
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7]])
    run(instance, ids, cache)
    hook = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
    try:
        assert run(instance, ids, cache)[2] == 0
        assert cache.snapshot()["entries"] == 0
    finally:
        hook.remove()
    assert run(instance, ids, cache, logits_to_keep=0)[2] == 0


def test_nested_config_and_norm_attribute_changes_invalidate():
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7]])
    run(instance, ids, cache)
    instance.norm.eps = 0.001
    assert run(instance, ids, cache)[2] == 0
    instance.config.rope_scaling = {"factor": 1.0}
    run(instance, ids, cache)
    instance.config.rope_scaling["factor"] = 2.0
    assert run(instance, ids, cache)[2] == 0
    assert cache.snapshot()["invalidations"] == 3


def test_hf_or_custom_generation_never_receives_native_cache():
    class HFModel:
        pass

    settings = InferenceServerSettings(prefix_cache_mb=1)
    assert LocalGenerationEngine(HFModel(), Tokenizer(), settings).prefix_cache is None
    instance = model()
    original = instance.generate
    instance.generate = lambda *args, **kwargs: original(*args, **kwargs)
    assert LocalGenerationEngine(instance, Tokenizer(), settings).prefix_cache is None


def test_partial_reuse_requires_explicit_opt_in():
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    run(instance, torch.tensor([[4, 5, 6, 7]]), cache)
    assert run(instance, torch.tensor([[4, 5, 6, 8]]), cache)[2] == 0
    assert not cache.snapshot()["partial_reuse_enabled"]
    with pytest.raises(ValueError, match="reuse_partial_prefixes"):
        NativePrefixCache(65536, reuse_partial_prefixes=1)
    with pytest.raises(ValueError, match="prefix_cache_partial"):
        InferenceServerSettings(prefix_cache_partial=True)


@pytest.mark.parametrize("change", ["replace_parameter", "add_buffer", "replace_module", "class_attribute"])
def test_fingerprint_invalidates_structural_and_class_changes(change, monkeypatch):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7]])
    run(instance, ids, cache)
    if change == "replace_parameter":
        target = instance.layers[0].self_attn.v_proj
        target.weight = torch.nn.Parameter(target.weight.detach().clone())
    elif change == "add_buffer":
        instance.register_buffer("additional_state", torch.zeros(1))
    elif change == "replace_module":
        instance.layers[0].self_attn.v_proj = torch.nn.Linear(16, 8, bias=False)
        instance.eval()
    else:
        value = instance.norm.eps
        del instance.norm.eps
        monkeypatch.setattr(type(instance.norm), "eps", value * 2, raising=False)
    actual = run(instance, ids, cache)
    assert actual[2] == 0
    assert torch.equal(actual[0], run(instance, ids)[0])
    assert cache.snapshot()["invalidations"] == 1


def test_long_partial_prefix_matching_at_block_boundaries():
    from saddlellm.models.NativePrefixCache import _common_prefix_length

    rng = random.Random(173)
    for shared in (0, 1, 16, 127, 128, 129, 255, 256, 257, 2048):
        prefix = tuple(rng.randrange(32) for _ in range(shared))
        for left, right in (
            (prefix + (32,), prefix + (33,)),
            (prefix, prefix + (33,)),
            (prefix + (32,), prefix),
            (prefix, prefix),
        ):
            for start in {0, min(16, shared), shared}:
                assert _common_prefix_length(left, right, start) == shared


def test_tied_weight_fingerprint_tracks_shared_updates():
    instance = model()
    instance.lm_head.weight = instance.embed_tokens.weight
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7]])
    run(instance, ids, cache)
    assert run(instance, ids, cache)[2] == 4
    with torch.no_grad():
        instance.embed_tokens.weight.add_(0.01)
    assert run(instance, ids, cache)[2] == 0
    assert cache.snapshot()["invalidations"] == 1


def test_prompt_below_minimum_skips_fingerprint_and_token_copy(monkeypatch):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=256, reuse_partial_prefixes=True)
    ids = torch.tensor([[4, 5, 6, 7]])
    reference = run(instance, ids)[0]

    def unexpected_fingerprint(*args):
        raise AssertionError("short prompts must bypass mutation scanning")

    monkeypatch.setattr(cache, "_fingerprint", unexpected_fingerprint)
    actual = run(instance, ids, cache)
    assert actual[2] == 0 and torch.equal(actual[0], reference)
    assert cache.snapshot()["skipped_requests"] == 1
    assert cache.snapshot()["lookups"] == cache.snapshot()["stores"] == 0


def test_store_obeys_increased_minimum_without_fingerprinting(monkeypatch):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2)
    ids = torch.tensor([[4, 5, 6, 7]])
    with instance.create_generation_session(ids, max_new_tokens=1) as session:
        ticket, _, _ = cache.restore(instance, ids, session.past_key_values)
        logits = session.prefill()
        cache.min_prefix_tokens = 256

        def unexpected_fingerprint(*args):
            raise AssertionError("publication must obey the current minimum")

        monkeypatch.setattr(cache, "_fingerprint", unexpected_fingerprint)
        cache.store(instance, ticket, session.past_key_values, logits)
    assert cache.snapshot()["entries"] == 0


def test_cache_comparison_benchmark_checks_same_model_and_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "comparison.json"
    completed = subprocess.run(
        [sys.executable, str(root / "tools/benchmark_prefix_cache_compare.py"),
         "--baseline-source", str(root / "saddlellm/models/NativePrefixCache.py"),
         "--prompt-tokens", "32", "--shared-tokens", "16", "--decode-tokens", "3",
         "--layers", "2", "--warmup", "0", "--repeats", "1", "--output", str(destination)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["same_model_instance"] and report["fingerprint_equal"]
    assert report["all_output_tokens_equal"]
    assert report["baseline_vs_current"]["full_hit"]["prefill_ms"]["n"] == 1
    for implementation in ("baseline", "current"):
        assert report["variants"][f"{implementation}/full_hit"]["reused_prompt_tokens"] == 32
        assert report["variants"][f"{implementation}/partial_hit"]["reused_prompt_tokens"] == 16


@pytest.mark.parametrize("value", [-1, True, 1.5, "256"])
def test_minimum_store_tokens_validation(value):
    with pytest.raises(ValueError, match="min_store_tokens"):
        NativePrefixCache(65536, min_store_tokens=value)


def test_partial_publication_waits_for_enough_new_tokens(monkeypatch):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2,
                              reuse_partial_prefixes=True, min_store_tokens=3)
    prefix = torch.tensor([[4, 5, 6, 7]])
    run(instance, prefix, cache)
    assert cache.snapshot()["stores"] == 1  # Cold miss publishes even a short prompt.
    fingerprint = cache._fingerprint
    calls = []

    def counted(*args):
        calls.append(True)
        return fingerprint(*args)

    monkeypatch.setattr(cache, "_fingerprint", counted)
    for suffix in ([8], [8, 9]):
        ids = torch.tensor([[4, 5, 6, 7, *suffix]])
        actual = run(instance, ids, cache)
        assert actual[2] == 4
        assert torch.equal(actual[0], run(instance, ids)[0])
    assert len(calls) == 2  # Restore scans; deferred stores never scan again.
    assert cache.snapshot()["stores"] == cache.snapshot()["entries"] == 1
    assert cache.snapshot()["deferred_stores"] == 2
    assert cache.snapshot()["min_store_tokens"] == 3
    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10]])
    assert run(instance, ids, cache)[2] == 4
    assert len(calls) == 4  # Threshold reached: restore and publication both scan.
    assert cache.snapshot()["stores"] == 2
    assert run(instance, ids, cache)[2] == 7  # Exact hits retain their usual contract.
    assert cache.snapshot()["deferred_stores"] == 2


def test_default_partial_publication_is_unchanged():
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2, reuse_partial_prefixes=True)
    run(instance, torch.tensor([[4, 5, 6, 7]]), cache)
    ids = torch.tensor([[4, 5, 6, 7, 8]])
    assert run(instance, ids, cache)[2] == 4
    assert run(instance, ids, cache)[2] == 5
    assert cache.snapshot()["stores"] == 2 and cache.snapshot()["deferred_stores"] == 0


@pytest.mark.parametrize("change", ["weight", "hook", "clear", "close"])
def test_deferred_publication_never_revives_invalid_cache(change):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2,
                              reuse_partial_prefixes=True, min_store_tokens=256)
    prefix = torch.tensor([[4, 5, 6, 7]])
    run(instance, prefix, cache)
    ids = torch.tensor([[4, 5, 6, 7, 8]])
    assert run(instance, ids, cache)[2] == 4
    assert cache.snapshot()["stores"] == 1
    handle = None
    if change == "weight":
        with torch.no_grad():
            instance.layers[0].self_attn.v_proj.weight.add_(0.2)
    elif change == "hook":
        handle = instance.norm.register_forward_hook(lambda *args: None)
    else:
        getattr(cache, change)()
    try:
        actual = run(instance, ids, cache)
        assert actual[2] == 0
        assert torch.equal(actual[0], run(instance, ids)[0])
        if change in {"hook", "close"}:
            assert cache.snapshot()["entries"] == 0
        else:
            assert cache.snapshot()["stores"] == 2  # The new cold miss still publishes.
    finally:
        if handle is not None:
            handle.remove()


def test_clear_between_partial_restore_and_deferred_store_keeps_cache_empty():
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2,
                              reuse_partial_prefixes=True, min_store_tokens=256)
    run(instance, torch.tensor([[4, 5, 6, 7]]), cache)
    ids = torch.tensor([[4, 5, 6, 7, 8]])
    with instance.create_generation_session(ids, max_new_tokens=1) as session:
        ticket, reused, _ = cache.restore(instance, ids, session.past_key_values)
        assert reused == ticket.reused_tokens == 4
        cache.clear()
        cache.store(instance, ticket, session.past_key_values, torch.zeros(1, 37))
    assert cache.snapshot()["entries"] == 0
    assert cache.snapshot()["deferred_stores"] == 0


def test_evicted_partial_source_does_not_defer_replacement(monkeypatch):
    instance = model()
    cache = NativePrefixCache(65536, min_prefix_tokens=2, max_entries=1,
                              reuse_partial_prefixes=True, min_store_tokens=256)
    prefix = torch.tensor([[4, 5, 6, 7]])
    run(instance, prefix, cache)
    original_store = cache.store
    before_epoch = cache._epoch
    evicted_sources = []

    def publish_after_another_request(model, ticket, caches, logits):
        if ticket is not None and ticket.reused_tokens:
            # Simulate an independent request publishing between restore and
            # store. LRU eviction keeps the epoch unchanged.
            assert ticket.reused_key == tuple(prefix[0].tolist())
            run(instance, torch.tensor([[10, 11, 12, 13]]), cache)
            assert ticket.reused_key not in cache._entries
            assert cache._epoch == before_epoch
            evicted_sources.append(ticket.reused_key)
        original_store(model, ticket, caches, logits)

    monkeypatch.setattr(cache, "store", publish_after_another_request)
    ids = torch.tensor([[4, 5, 6, 7, 8]])
    actual = run(instance, ids, cache)
    assert actual[2] == 4 and torch.equal(actual[0], run(instance, ids)[0])
    assert len(evicted_sources) == 1
    assert tuple(ids[0].tolist()) in cache._entries
    assert cache.snapshot()["deferred_stores"] == 0
    assert cache.snapshot()["stores"] == 3 and cache.snapshot()["evictions"] == 2
    assert cache.snapshot()["retained_tensor_bytes"] <= cache.max_bytes
    assert run(instance, ids, cache)[2] == 5
