"""Token-budget admission, padding-aware batching, fairness and API compatibility."""

import asyncio
import threading

import httpx
import pytest
import torch

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.InferenceBatcher import InferenceBatcher, InferenceQueueFull
from saddlellm.runtime.InferenceServer import (
    CompletionRequest,
    InferenceServerSettings,
    LocalGenerationEngine,
    _BatchGenerationControl,
    create_inference_app,
)


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class NumericTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        ids = [int(item) for item in text.split()][:kwargs["max_length"]]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(item) for item in ids if item not in {0, 2})


def _engine(kind="gqa", window=0, mode="none", **settings):
    torch.manual_seed(173)
    config = SaddleModelConfig(
        vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attention_kind=kind, attention_backend="sdpa", sliding_window=window,
        kv_lora_rank=4 if kind == "mla" else 0, mla_cache_mode="latent",
        residual_attention_mode=mode, eos_token_id=[],
    )
    model = SaddleForCausalLM(config).eval()
    return LocalGenerationEngine(model, NumericTokenizer(), InferenceServerSettings(
        batch_size=4, default_max_new_tokens=5, max_new_tokens=8, **settings,
    ))


def _content(response):
    return response["choices"], response["usage"]


@pytest.mark.parametrize("kind,window,mode", [("gqa", 0, "none"), ("gqa", 3, "none"),
                                             ("mla", 4, "none"), ("gqa", 4, "full"),
                                             ("mla", 3, "block")])
def test_neighboring_length_batches_match_individual_outputs_and_usage(monkeypatch, kind, window, mode):
    engine = _engine(kind, window, mode, batch_length_bucket_size=8)
    requests = [CompletionRequest(prompt=text, temperature=0, max_tokens=6)
                for text in ["4 5", "6 7 8 9 10", "11", "12 13 14"]]
    expected = [engine.complete(row) for row in requests]
    original, seen = engine.generate_batch, []

    def generate(rows, *args):
        seen.append([item.generation.prompt_tokens for item in rows])
        return original(rows, *args)

    monkeypatch.setattr(engine, "generate_batch", generate)
    actual = engine.complete_batch(requests)
    assert seen == [[2, 5, 1, 3]]
    assert [_content(row) for row in actual] == [_content(row) for row in expected]
    assert [row["usage"]["prompt_tokens"] for row in actual] == [2, 5, 1, 3]


def test_import_batch_budget_counts_padding_not_unpadded_token_sum(monkeypatch):
    engine = _engine(batch_length_bucket_size=8, batch_token_budget=12)
    rows = [CompletionRequest(prompt=text, temperature=0, max_tokens=2)
            for text in ["4", "5 6 7 8 9 10", "11"]]
    expected = [engine.complete(row) for row in rows]
    original, sizes = engine.generate_batch, []

    def generate(prepared, *args):
        sizes.append(len(prepared))
        assert engine._batch_token_slots([item.generation for item in prepared]) <= 12
        return original(prepared, *args)

    monkeypatch.setattr(engine, "generate_batch", generate)
    assert [_content(row) for row in engine.complete_batch(rows)] == [_content(row) for row in expected]
    assert sizes == [1, 1, 1]


def test_default_exact_length_grouping_and_repetition_penalty_fallback():
    engine = _engine(batch_length_bucket_size=8)
    rows = [engine.prepare_batch_request(CompletionRequest(prompt=text, temperature=0, repetition_penalty=1.1))
            for text in ["4", "4 5"]]
    assert engine.batch_key(rows[0]) != engine.batch_key(rows[1])
    engine.settings.batch_length_bucket_size = 0
    rows = [engine.prepare_batch_request(CompletionRequest(prompt=text, temperature=0)) for text in ["4", "4 5"]]
    assert engine.batch_key(rows[0]) != engine.batch_key(rows[1])


def test_bucket_width_does_not_merge_distant_lengths_or_custom_generate(monkeypatch):
    engine = _engine(batch_length_bucket_size=2)
    rows = [engine.prepare_batch_request(CompletionRequest(prompt=text, temperature=0))
            for text in ["4", "4 5", "4 5 6"]]
    assert engine.batch_key(rows[0]) == engine.batch_key(rows[1])
    assert engine.batch_key(rows[1]) != engine.batch_key(rows[2])
    original = engine.model.generate
    monkeypatch.setattr(engine.model, "generate", lambda **kwargs: original(**kwargs))
    assert engine.batch_key(rows[0]) != engine.batch_key(rows[1])


def test_stop_control_counts_generated_tokens_after_padded_prompt():
    engine = _engine(batch_length_bucket_size=8)
    rows = [engine.prepare_batch_request(CompletionRequest(prompt=text, temperature=0, stop="4")).generation
            for text in ["4 5", "6 7 8 9 10"]]
    events = [threading.Event(), threading.Event()]
    control = _BatchGenerationControl(NumericTokenizer(), rows, [], events, "cpu", prompt_width=5)
    prompt = torch.tensor([[0, 0, 0, 4, 5], [6, 7, 8, 9, 10]])
    step = torch.cat((prompt, torch.tensor([[7], [8]])), dim=1)
    assert not control(step, None).any()
    step = torch.cat((step, torch.tensor([[9], [4]])), dim=1)
    assert control(step, None).tolist() == [False, True]
    assert control.lengths.tolist() == [0, 2]


def test_kda_hybrid_length_buckets_preserve_padding_and_completion_semantics():
    from saddlellm.models.ModelBlueprint import ModelBlueprintLab

    blueprint = ModelBlueprintLab.hybrid_kda(hidden_size=16, layers=4, heads=4,
                                            kv_heads=2, vocab_size=37, seq_length=32,
                                            attention_residuals="block")
    model = blueprint.build_model().eval()
    engine = LocalGenerationEngine(model, NumericTokenizer(), InferenceServerSettings(
        batch_size=4, batch_length_bucket_size=8, default_max_new_tokens=6, max_new_tokens=8,
    ))
    rows = [CompletionRequest(prompt=text, temperature=0, max_tokens=6)
            for text in ["4 5", "6 7 8 9 10", "11"]]
    expected = [engine.complete(row) for row in rows]
    assert [_content(row) for row in engine.complete_batch(rows)] == [_content(row) for row in expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_mixed_length_bucket_matches_individual_greedy_results(dtype):
    engine = _engine(window=4, batch_length_bucket_size=8)
    engine.model.to(device="cuda", dtype=dtype)
    rows = [CompletionRequest(prompt=text, temperature=0, max_tokens=8)
            for text in ["4 5", "6 7 8 9 10", "11"]]
    expected = [engine.complete(row) for row in rows]
    assert [_content(row) for row in engine.complete_batch(rows)] == [_content(row) for row in expected]


def test_scheduler_bounds_padded_batch_tokens_and_preserves_result_mapping():
    async def run():
        seen = []
        batcher = InferenceBatcher(lambda rows, events: seen.append(rows) or rows,
                                  asyncio.Semaphore(1), max_batch_size=4, max_batch_tokens=12)
        result = await asyncio.gather(
            batcher.submit("short1", "same", prompt_tokens=1, max_new_tokens=1),
            batcher.submit("long", "same", prompt_tokens=6, max_new_tokens=1),
            batcher.submit("short2", "same", prompt_tokens=1, max_new_tokens=1),
        )
        assert result == ["short1", "long", "short2"]
        assert seen == [["short1", "short2"], ["long"]]
        assert batcher.snapshot()["largest_batch_tokens"] <= 12
        assert batcher.snapshot()["queued_reserved_tokens"] == 0
        await batcher.close()

    asyncio.run(run())


@pytest.mark.parametrize("aged,expected", [(False, ["short", "long"]), (True, ["long", "short"])])
def test_shortest_prompt_policy_has_oldest_request_starvation_guard(aged, expected):
    async def run():
        now = [0.0]
        seen, semaphore = [], asyncio.Semaphore(0)
        batcher = InferenceBatcher(lambda rows, events: seen.extend(rows) or rows,
                                  semaphore, max_batch_size=1, wait_ms=0,
                                  scheduling_policy="shortest_prompt", starvation_ms=1000,
                                  clock=lambda: now[0])
        long = asyncio.create_task(batcher.submit("long", "long", prompt_tokens=100))
        await asyncio.sleep(0)
        now[0] = 0.5
        short = asyncio.create_task(batcher.submit("short", "short", prompt_tokens=1))
        await asyncio.sleep(0)
        now[0] = 1.1 if aged else 0.5
        semaphore.release()
        assert await asyncio.gather(long, short) == ["long", "short"]
        assert seen == expected
        await batcher.close()

    asyncio.run(run())


def test_queue_token_admission_and_cancel_release_reservation():
    async def run():
        semaphore = asyncio.Semaphore(0)
        batcher = InferenceBatcher(lambda rows, events: rows, semaphore,
                                  max_batch_size=2, wait_ms=0, max_pending_tokens=10)
        first = asyncio.create_task(batcher.submit("cancel", "a", prompt_tokens=6, max_new_tokens=2))
        await asyncio.sleep(0)
        assert batcher.snapshot()["queued_reserved_tokens"] == 8
        with pytest.raises(InferenceQueueFull, match="token"):
            await batcher.submit("reject", "a", prompt_tokens=2, max_new_tokens=1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert batcher.snapshot()["queued_reserved_tokens"] == 0
        second = asyncio.create_task(batcher.submit("ok", "a", prompt_tokens=2, max_new_tokens=1))
        await asyncio.sleep(0)
        semaphore.release()
        assert await second == "ok"
        assert batcher.snapshot()["admission_rejections"] == 1
        await batcher.close()

    asyncio.run(run())


def test_cancelled_active_job_keeps_token_accounting_until_model_finishes():
    async def run():
        entered, release = threading.Event(), threading.Event()

        def generate(rows, events):
            entered.set()
            assert release.wait(5)
            return rows

        semaphore = asyncio.Semaphore(1)
        batcher = InferenceBatcher(generate, semaphore, max_batch_size=1, wait_ms=0)
        job = asyncio.create_task(batcher.submit("a", "a", prompt_tokens=6, max_new_tokens=2))
        assert await asyncio.to_thread(entered.wait, 5)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert batcher.snapshot()["active_reserved_tokens"] == 8
        assert semaphore.locked()
        release.set()
        await batcher.close()
        assert batcher.snapshot()["active_reserved_tokens"] == 0

    asyncio.run(run())


def test_oversized_request_is_rejected_before_worker_launch():
    async def run():
        batcher = InferenceBatcher(lambda rows, events: pytest.fail("must not generate"),
                                  asyncio.Semaphore(1), max_batch_size=2, max_batch_tokens=4)
        with pytest.raises(ValueError, match="budget"):
            await batcher.submit("oversize", "a", prompt_tokens=3, max_new_tokens=2)
        assert batcher.worker is None and not batcher.pending
        assert batcher.snapshot()["admission_rejections"] == 1
        await batcher.close()

    asyncio.run(run())


@pytest.mark.parametrize("settings", [
    {"batch_token_budget": 0}, {"batch_token_budget": True}, {"batch_token_budget": 1.5},
    {"max_pending_tokens": -1}, {"max_pending_tokens": False},
    {"batch_length_bucket_size": -1}, {"batch_length_bucket_size": True},
    {"batch_starvation_ms": 0}, {"batch_starvation_ms": float("nan")},
    {"batch_starvation_ms": float("inf")}, {"batch_wait_ms": True},
    {"max_input_tokens": True}, {"max_concurrency": 1.5},
])
def test_server_rejects_invalid_budget_and_scheduler_settings(settings):
    with pytest.raises(ValueError):
        InferenceServerSettings(**settings)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_http_budget_errors_health_and_recovery(batch_size):
    async def run():
        engine = _engine(batch_token_budget=5, max_pending_tokens=4, batch_wait_ms=0)
        engine.settings.batch_size = batch_size
        app = create_inference_app(engine.settings, model=engine.model, tokenizer=engine.tokenizer)
        batcher = app.state.batcher
        await batcher.semaphore.acquire()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/v1/completions", json={"prompt": "4 5", "temperature": 0, "max_tokens": 2}))
                for _ in range(100):
                    if batcher.pending:
                        break
                    await asyncio.sleep(0.001)
                assert batcher.snapshot()["queued_reserved_tokens"] == 4
                rejected = await client.post("/v1/completions", json={"prompt": "6 7", "temperature": 0, "max_tokens": 2})
                assert rejected.status_code == 503 and rejected.headers["Retry-After"] == "1"
                oversized = await client.post("/v1/completions", json={"prompt": "6 7 8 9", "temperature": 0, "max_tokens": 2})
                assert oversized.status_code == 400
                batcher.semaphore.release()
                assert (await first).status_code == 200
                health = (await client.get("/health")).json()["batching"]
                assert health["queued_reserved_tokens"] == health["active_reserved_tokens"] == 0
                assert health["admission_rejections"] == 2
        finally:
            batcher.semaphore.release()
            await batcher.close()

    asyncio.run(run())


def test_empty_tokenized_prompt_is_rejected_before_queueing():
    engine = _engine()
    with pytest.raises(ValueError, match="at least one token"):
        engine.prepare_batch_request(CompletionRequest(prompt=" ", temperature=0))


def test_cli_budget_options_reach_server_settings(monkeypatch):
    import uvicorn

    from saddlellm import cli
    from saddlellm.runtime import InferenceServer

    seen = []
    monkeypatch.setattr(InferenceServer, "create_inference_app", lambda settings: seen.append(settings) or object())
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    assert cli.main(["serve", "release", "--batch-size", "4", "--batch-token-budget", "1024",
                     "--batch-length-bucket-size", "32", "--max-queued-tokens", "4096"]) == 0
    assert seen[0].resolved_batch_token_budget == 1024
    assert seen[0].batch_length_bucket_size == 32
    assert seen[0].max_pending_tokens == 4096
