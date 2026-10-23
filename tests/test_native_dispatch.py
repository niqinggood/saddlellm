"""Scheduler regression: a busy GPU must not consume the next collection window."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from test_native_stream_batching import coordinator, new_session
from test_native_streaming import engine


@pytest.mark.parametrize("wait_ms", [0, 100])
def test_queued_request_rejoins_returning_peer_after_busy_forward(monkeypatch, wait_ms):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    instance = engine()
    batcher = coordinator(instance, max_batch_size=2, max_resident=2, wait_ms=wait_ms)
    first = new_session(instance, tokens=5)
    # Different prompt lengths avoid the separate equal-length alignment policy.
    with torch.inference_mode():
        second = instance.model.create_generation_session(
            input_ids=torch.tensor([[4, 5, 6]]),
            max_new_tokens=5,
            do_sample=False,
        )
    for session in (first, second):
        session.step()
        batcher.register(session)
    entered, release = threading.Event(), threading.Event()
    completed, resubmit = threading.Event(), threading.Event()
    batches = []
    original_step = batcher._runner.step
    original_complete = batcher._complete
    original_wait = batcher._condition.wait

    def execute(sessions):
        batches.append(tuple(sessions))
        if len(batches) == 1:
            entered.set()
            assert release.wait(5)
        else:
            # Also release the peer on the old/zero-wait path so a failed
            # assertion never leaves a test worker waiting indefinitely.
            resubmit.set()
        return original_step(sessions)

    def complete(jobs, *args, **kwargs):
        if any(job[0] is first for job in jobs):
            completed.set()
        return original_complete(jobs, *args, **kwargs)

    def wait(timeout=None):
        if completed.is_set() and batcher._queue:
            # A returning HTTP consumer can enqueue while collection waits.
            resubmit.set()
        return original_wait(timeout)

    monkeypatch.setattr(batcher._runner, "step", execute)
    monkeypatch.setattr(batcher, "_complete", complete)
    monkeypatch.setattr(batcher._condition, "wait", wait)

    def returning_peer():
        batcher.step(first, threading.Event())
        assert resubmit.wait(5)
        batcher.step(first, threading.Event())

    try:
        with ThreadPoolExecutor(2) as pool:
            returning = pool.submit(returning_peer)
            assert entered.wait(5)
            queued = pool.submit(batcher.step, second, threading.Event())
            deadline = time.perf_counter() + 5
            while batcher.snapshot()["pending"] != 1 and time.perf_counter() < deadline:
                time.sleep(0.001)
            assert batcher.snapshot()["pending"] == 1
            # Simulate a forward outlasting the configured collection budget.
            time.sleep(0.15)
            release.set()
            returning.result(5)
            queued.result(5)
        assert batches[0] == (first,)
        if wait_ms:
            assert set(batches[1]) == {first, second}
            assert len(batches) == 2
        else:
            assert batches[1] == (second,)
        assert first.generated_tokens == 3 and second.generated_tokens == 2
        assert batcher.snapshot()["pending"] == batcher.snapshot()["inflight"] == 0
    finally:
        release.set()
        resubmit.set()
        for session in (first, second):
            batcher.unregister(session)
            session.close()
        batcher.close()
        torch.set_num_threads(previous_threads)


def test_shutdown_interrupts_collection_without_advancing_queued_request(monkeypatch):
    instance = engine()
    batcher = coordinator(instance, max_batch_size=2, max_resident=2, wait_ms=1000)
    first, peer = new_session(instance), new_session(instance)
    for session in (first, peer):
        batcher.register(session)
    collecting = threading.Event()
    original_wait = batcher._condition.wait

    def wait(timeout=None):
        if batcher._queue:
            collecting.set()
        return original_wait(timeout)

    monkeypatch.setattr(batcher._condition, "wait", wait)
    try:
        with ThreadPoolExecutor(2) as pool:
            queued = pool.submit(batcher.step, first, threading.Event())
            assert collecting.wait(5)
            closing = pool.submit(batcher.close)
            closing.result(5)
            with pytest.raises(RuntimeError, match="closed"):
                queued.result(5)
        assert first.generated_tokens == peer.generated_tokens == 0
        status = batcher.snapshot()
        assert status["pending"] == status["inflight"] == 0
        assert not status["worker_alive"]
    finally:
        for session in (first, peer):
            batcher.unregister(session)
            session.close()
        batcher.close()
