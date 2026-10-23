"""Bounded, request-local microbatch scheduling for synchronous generation."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable, Hashable


class InferenceQueueFull(RuntimeError):
    """The service cannot admit another queued inference request."""


@dataclass
class _BatchJob:
    payload: Any
    key: Hashable
    future: asyncio.Future
    cancel_event: Event = field(default_factory=Event)
    prompt_tokens: int = 0
    max_new_tokens: int = 0
    admitted_at: float = field(default_factory=time.perf_counter)

    @property
    def reserved_tokens(self) -> int:
        return max(1, self.prompt_tokens + self.max_new_tokens)


class InferenceBatcher:
    """Combine compatible jobs without owning model weights or cache state.

    One worker drains the queue. Each batch holds the HTTP layer's semaphore
    until its model thread finishes, even if clients cancel their awaits.
    """

    def __init__(
        self,
        generate: Callable,
        semaphore: asyncio.Semaphore,
        *,
        max_batch_size: int,
        wait_ms: float = 5.0,
        max_pending: int = 64,
        max_batch_tokens: int | None = None,
        max_pending_tokens: int | None = None,
        scheduling_policy: str = "fifo",
        starvation_ms: float = 1000.0,
        clock: Callable = time.perf_counter,
    ):
        for name, value in (("max_batch_size", max_batch_size), ("max_pending", max_pending)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(wait_ms, bool) or not isinstance(wait_ms, (int, float)) or not math.isfinite(wait_ms) or wait_ms < 0:
            raise ValueError("wait_ms must be non-negative and finite")
        for name, value in (("max_batch_tokens", max_batch_tokens), ("max_pending_tokens", max_pending_tokens)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if not isinstance(scheduling_policy, str) or scheduling_policy not in {"fifo", "shortest_prompt", "shortest_work"}:
            raise ValueError("scheduling_policy must be fifo, shortest_prompt or shortest_work")
        if isinstance(starvation_ms, bool) or not isinstance(starvation_ms, (int, float)) or not math.isfinite(starvation_ms) or starvation_ms <= 0:
            raise ValueError("starvation_ms must be positive and finite")
        self.generate = generate
        self.semaphore = semaphore
        self.max_batch_size = max_batch_size
        self.wait_seconds = wait_ms / 1000
        self.max_pending = max_pending
        self.max_batch_tokens = max_batch_tokens
        self.max_pending_tokens = max_pending_tokens
        self.scheduling_policy = scheduling_policy
        self.starvation_seconds = starvation_ms / 1000
        self.clock = clock
        self.pending: list[_BatchJob] = []
        self.active: list[_BatchJob] = []
        self.worker: asyncio.Task | None = None
        self.closed = False
        self.batches = 0
        self.requests = 0
        self.largest_batch = 0
        self.largest_batch_tokens = 0
        self.admission_rejections = 0
        self._wake = asyncio.Event()
        # Bounded aggregates only: never retain prompts, tokens or payloads.
        self._queue_wait_ms = deque(maxlen=1024)
        self._batch_service_ms = deque(maxlen=1024)
        self._batch_padding_tokens = deque(maxlen=1024)

    @staticmethod
    def _batch_tokens(jobs) -> int:
        if not jobs:
            return 0
        # Include left padding, not just the sum of unpadded prompt lengths.
        return len(jobs) * max(1, max(job.prompt_tokens for job in jobs)
                               + max(job.max_new_tokens for job in jobs))

    def _select_batch(self, live):
        anchor = live[0]
        if (self.scheduling_policy != "fifo"
                and self.clock() - anchor.admitted_at < self.starvation_seconds):
            anchor = min(live, key=lambda job: (
                job.reserved_tokens if self.scheduling_policy == "shortest_work" else job.prompt_tokens,
                job.admitted_at,
            ))
        batch = [anchor]
        for job in live:
            if job is anchor or job.key != anchor.key:
                continue
            if len(batch) == self.max_batch_size:
                break
            candidate = batch + [job]
            if self.max_batch_tokens is None or self._batch_tokens(candidate) <= self.max_batch_tokens:
                batch.append(job)
        return batch

    async def _coalesce(self) -> None:
        """Wait at most the oldest job's deadline, waking as soon as a batch fills."""
        while not self.closed:
            live = [job for job in self.pending if not job.future.cancelled()]
            if not live:
                return
            batch = self._select_batch(live)
            if (len(batch) >= self.max_batch_size
                    or (self.max_batch_tokens is not None
                        and self._batch_tokens(batch) >= self.max_batch_tokens)):
                return
            remaining = self.wait_seconds - (self.clock() - live[0].admitted_at)
            if remaining <= 0:
                return
            # No await between observing the queue and clearing the event:
            # submissions cannot be lost on this single event-loop owner.
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return

    async def submit(self, payload: Any, key: Hashable, *, prompt_tokens: int = 0,
                     max_new_tokens: int = 0) -> Any:
        if self.closed:
            raise RuntimeError("inference batcher is closed")
        self.pending = [job for job in self.pending if not job.future.cancelled()]
        for name, value in (("prompt_tokens", prompt_tokens), ("max_new_tokens", max_new_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        reserved = max(1, prompt_tokens + max_new_tokens)
        if self.max_batch_tokens is not None and reserved > self.max_batch_tokens:
            self.admission_rejections += 1
            raise ValueError("Request exceeds the inference batch token budget")
        if len(self.pending) >= self.max_pending:
            self.admission_rejections += 1
            raise InferenceQueueFull("Inference queue is full; retry later.")
        if (self.max_pending_tokens is not None
                and sum(job.reserved_tokens for job in self.pending) + reserved > self.max_pending_tokens):
            self.admission_rejections += 1
            raise InferenceQueueFull("Inference queue token budget is full; retry later.")
        job = _BatchJob(payload, key, asyncio.get_running_loop().create_future(),
                        prompt_tokens=prompt_tokens, max_new_tokens=max_new_tokens,
                        admitted_at=self.clock())
        self.pending.append(job)
        self._wake.set()
        if self.worker is None:
            self.worker = asyncio.create_task(self._drain())
        try:
            return await job.future
        except asyncio.CancelledError:
            job.cancel_event.set()
            self.pending = [item for item in self.pending if item is not job]
            self._wake.set()
            raise

    async def _drain(self) -> None:
        first_batch = True
        try:
            while self.pending and not self.closed:
                # Only the first batch needs a coalescing delay. Jobs that
                # arrived while generating have already waited long enough.
                if first_batch and self.wait_seconds:
                    await self._coalesce()
                first_batch = False
                async with self.semaphore:
                    live = [job for job in self.pending if not job.future.cancelled()]
                    if self.closed or not live:
                        self.pending = live
                        continue
                    batch = self._select_batch(live)
                    selected = {id(job) for job in batch}
                    self.pending = [job for job in live if id(job) not in selected]
                    self.active = batch
                    started = self.clock()
                    self._queue_wait_ms.extend(
                        max(0.0, started - job.admitted_at) * 1000 for job in batch
                    )
                    self._batch_padding_tokens.append(
                        self._batch_tokens(batch) - sum(job.reserved_tokens for job in batch)
                    )
                    thread = asyncio.create_task(
                        asyncio.to_thread(
                            self.generate,
                            [job.payload for job in batch],
                            [job.cancel_event for job in batch],
                        )
                    )
                    try:
                        results = await asyncio.shield(thread)
                        if len(results) != len(batch):
                            raise RuntimeError(
                                "batch generation returned an invalid result count"
                            )
                        self.batches += 1
                        self.requests += len(batch)
                        self.largest_batch = max(self.largest_batch, len(batch))
                        self.largest_batch_tokens = max(self.largest_batch_tokens, self._batch_tokens(batch))
                        for job, result in zip(batch, results):
                            if not job.future.done():
                                job.future.set_result(result)
                    except asyncio.CancelledError:
                        for job in batch:
                            job.cancel_event.set()
                            job.future.cancel()
                        # Do not release the permit while a cancelled model
                        # invocation is still using memory in another thread.
                        try:
                            await asyncio.shield(thread)
                        except Exception:
                            pass
                        raise
                    except Exception as exc:
                        for job in batch:
                            if not job.future.done():
                                job.future.set_exception(exc)
                    finally:
                        self._batch_service_ms.append(max(0.0, self.clock() - started) * 1000)
                        self.active = []
        finally:
            self.worker = None
            # Propagate unexpected worker termination to pending clients.
            if self.pending:
                for job in self.pending:
                    if not job.future.done():
                        job.future.set_exception(
                            RuntimeError("inference batch worker stopped")
                        )
                self.pending.clear()

    async def close(self) -> None:
        self.closed = True
        self._wake.set()
        for job in self.pending + self.active:
            job.cancel_event.set()
            if not job.future.done():
                job.future.set_exception(
                    RuntimeError("inference service is shutting down")
                )
        self.pending.clear()
        if self.worker is not None:
            await asyncio.shield(self.worker)

    @staticmethod
    def _timing_summary(samples) -> dict:
        ordered = sorted(samples)
        if not ordered:
            return {"samples": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}

        def percentile(fraction):
            position = (len(ordered) - 1) * fraction
            low = int(position)
            high = min(low + 1, len(ordered) - 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

        return {"samples": len(ordered), "p50_ms": percentile(0.5),
                "p95_ms": percentile(0.95), "p99_ms": percentile(0.99)}

    def snapshot(self) -> dict:
        return {
            "max_batch_size": self.max_batch_size,
            "queued_requests": len(self.pending),
            "active_requests": len(self.active),
            "completed_batches": self.batches,
            "completed_requests": self.requests,
            "largest_batch": self.largest_batch,
            "max_batch_tokens": self.max_batch_tokens,
            "max_pending_tokens": self.max_pending_tokens,
            "queued_reserved_tokens": sum(job.reserved_tokens for job in self.pending),
            "active_reserved_tokens": sum(job.reserved_tokens for job in self.active),
            "active_batch_tokens": self._batch_tokens(self.active),
            "largest_batch_tokens": self.largest_batch_tokens,
            "admission_rejections": self.admission_rejections,
            "scheduling_policy": self.scheduling_policy,
            "starvation_ms": self.starvation_seconds * 1000,
            "queue_wait": self._timing_summary(self._queue_wait_ms),
            "batch_service": self._timing_summary(self._batch_service_ms),
            "recent_padding_tokens": sum(self._batch_padding_tokens),
            "padding_samples": len(self._batch_padding_tokens),
            "metrics_window_size": 1024,
        }
