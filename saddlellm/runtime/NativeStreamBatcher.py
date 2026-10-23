"""Opt-in, bounded native SSE token-step coordination (not paged attention)."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from concurrent.futures import Future

from ..models.NativeBatchDecode import NativeBatchDecodeRunner


class NativeStreamBatcher:
    """One model worker, at most one outstanding step per resident request.

    Callers retain ownership of sessions, cancellation and text delivery. A
    step returns only after its actual model work has finished; no speculative
    output queue is filled behind a slow client. Unsupported sessions take the
    runner's individual path. Dense decode can share projections while attention
    uses independent request histories when KV lengths differ.
    """

    def __init__(self, model, *, max_batch_size, max_resident, max_batch_tokens, wait_ms=5.0,
                 prefill_chunk_size=0, max_step_tokens=None, decode_mode="compatible"):
        from .NativeTokenScheduler import NativeTokenScheduler

        if isinstance(prefill_chunk_size, bool) or not isinstance(prefill_chunk_size, int) or prefill_chunk_size < 0:
            raise ValueError("prefill_chunk_size must be a non-negative integer")
        if max_step_tokens is not None and not prefill_chunk_size:
            raise ValueError("max_step_tokens requires chunked prefill")
        self._scheduler = (NativeTokenScheduler(
            max_sequences=max_batch_size, max_step_tokens=(max_step_tokens if max_step_tokens is not None
                                                         else max(prefill_chunk_size, max_batch_size)),
            prefill_chunk_size=prefill_chunk_size,
        ) if prefill_chunk_size else None)
        if isinstance(max_resident, bool) or not isinstance(max_resident, int) or max_resident < 1:
            raise ValueError("max_resident must be a positive integer")
        if (isinstance(wait_ms, bool) or not isinstance(wait_ms, (int, float))
                or not math.isfinite(wait_ms) or not 0 <= wait_ms <= 1000):
            raise ValueError("wait_ms must be between 0 and 1000")
        self._runner = NativeBatchDecodeRunner(
            model, max_batch_size=max_batch_size, max_batch_tokens=max_batch_tokens,
            decode_mode=decode_mode,
        )
        if self._scheduler is not None and (
            getattr(model, "_saddle_cuda_int8", None) is not None
            or any(layer.config.attention_kind not in {"mha", "mqa", "gqa"}
                   or layer.config.sliding_window or layer.config.num_experts
                   or layer.config.residual_attention_mode != "none" for layer in model.layers)
        ):
            raise ValueError("chunked prefill requires dense MHA/MQA/GQA float models without windows or AttnRes")
        self.max_batch_size, self.max_resident = max_batch_size, max_resident
        self.wait_seconds = wait_ms / 1000
        self._condition = threading.Condition()
        # Housekeeping may arrive on a caller thread while the worker starts
        # another cohort. Serialize it too: the runner rejects any concurrent
        # owner, including a brief forget() holding its internal lock.
        self._execution = threading.Lock()
        self._registered = set()
        self._outstanding = set()
        self._queue = deque()
        self._thread = None
        self.closed = False
        self._inflight = 0
        self._cancelled_steps = 0
        self._single_request_steps = 0
        self._alignment_deferrals = 0
        self._prefill_chunks = self._prefill_tokens = self._scheduler_steps = 0
        self._scheduled_tokens = self._peak_step_tokens = 0

    def register(self, session):
        with self._condition:
            if self.closed:
                raise RuntimeError("native stream batcher is closed")
            if session.model is not self._runner.model or session.closed or session.done:
                raise ValueError("register a live session belonging to this model")
            if self._scheduler is not None and not session.supports_chunked_prefill():
                raise ValueError("chunked prefill requires unpadded dense sessions without prefix caching")
            if session in self._registered:
                raise ValueError("session is already registered")
            if len(self._registered) >= self.max_resident:
                raise RuntimeError("native stream resident limit reached")
            # Lazy startup: constructing an app never leaks a model thread.
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="saddle-native-batch", daemon=True)
                self._thread.start()
            self._registered.add(session)
            self._condition.notify_all()

    def step(self, session, cancel_event):
        with self._condition:
            if self.closed:
                raise RuntimeError("native stream batcher is closed")
            if session not in self._registered:
                raise ValueError("session is not registered")
            if session in self._outstanding:
                raise RuntimeError("session already has an outstanding step")
            if cancel_event.is_set():
                return None
            future = Future()
            self._outstanding.add(session)
            job = (session, cancel_event, future, time.monotonic(), 0)
            direct = (self._scheduler is None and len(self._registered) == 1
                      and not self._queue and not self._inflight)
            if direct:
                self._inflight += 1
                self._single_request_steps += 1
            else:
                self._queue.append(job)
                self._condition.notify_all()
        # A lone request cannot batch: avoid collection delay and a round trip
        # through another thread. The same execution lock still serializes it
        # against arrivals/cleanup; sessions remain independent and migratable.
        if direct:
            self._execute([job])
        # Never return early on cancellation while a CUDA invocation is live.
        return future.result()

    def unregister(self, session):
        with self._condition:
            if session in self._outstanding:
                raise RuntimeError("wait for the actual step before unregistering")
            self._registered.discard(session)
            self._condition.notify_all()
        with self._execution:
            self._runner.forget(session)

    def _complete(self, jobs, results=None, error=None, *, assigned=True):
        with self._condition:
            for index, (session, _, future, _, _) in enumerate(jobs):
                self._outstanding.discard(session)
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result(None if results is None else results[index])
            if assigned:
                self._inflight -= len(jobs)

    def _execute(self, jobs):
        try:
            with self._execution:
                results = self._runner.step([job[0] for job in jobs])
        except BaseException as exc:
            self._complete(jobs, error=exc)
        else:
            self._complete(jobs, results)

    def _run(self):
        if self._scheduler is not None:
            return self._run_chunked()
        try:
            while True:
                with self._condition:
                    while not self._queue and not self.closed:
                        self._condition.wait()
                    if self.closed:
                        pending = list(self._queue)
                        self._queue.clear()
                        self._complete(pending, error=RuntimeError("native stream batcher is closed"), assigned=False)
                        return
                    # Start one bounded collection window when the worker is
                    # ready. Queue age also includes the previous GPU batch:
                    # using its enqueue deadline can dispatch a waiting B1
                    # before the just-completed B3 callers can resubmit, keeping
                    # both groups permanently apart. Notifications never extend
                    # this window. Ready batches and lone decodes do not wait.
                    deadline = time.monotonic() + self.wait_seconds
                    while not self.closed:
                        first_prefill = not self._queue[0][0].generated_tokens
                        target = (self.max_batch_size if first_prefill else
                                  min(self.max_batch_size, len(self._registered)))
                        if len(self._queue) >= target:
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(remaining)
                    if self.closed:
                        continue
                    jobs = [self._queue.popleft() for _ in range(min(self.max_batch_size, len(self._queue)))]
                    cancelled, active = [], []
                    for job in jobs:
                        (cancelled if job[1].is_set() else active).append(job)
                    del job
                    jobs = active
                    self._cancelled_steps += len(cancelled)
                    self._complete(cancelled, assigned=False)
                    cancelled.clear()
                    # A lone fast-path request can be one tick ahead of new
                    # arrivals with the SAME prompt length. Let ready laggards
                    # catch up, instead of permanently running B1 + B3. Never
                    # wait for an unqueued slow client, and defer any submitted
                    # step at most twice. Different prompt lengths are unchanged.
                    minimum = {}
                    for job in jobs:
                        session = job[0]
                        minimum[session.prompt_tokens] = min(
                            minimum.get(session.prompt_tokens, session.generated_tokens), session.generated_tokens,
                        )
                    ready, deferred = [], []
                    for job in jobs:
                        session = job[0]
                        gap = session.generated_tokens - minimum[session.prompt_tokens]
                        if 0 < gap <= 2 and job[4] < 2:
                            deferred.append((*job[:3], time.monotonic(), job[4] + 1))
                        else:
                            ready.append(job)
                    if jobs:
                        del job, session
                    jobs.clear()
                    jobs = ready
                    self._alignment_deferrals += len(deferred)
                    self._queue.extendleft(reversed(deferred))
                    deferred.clear()
                    self._inflight += len(jobs)
                if not jobs:
                    continue
                self._execute(jobs)
                # Do not retain the previous Future/exception traceback while
                # idle: a failed forward may otherwise pin CUDA intermediates.
                jobs.clear()
        finally:
            with self._execution:
                self._runner.close()

    def _execute_chunked(self, jobs, plan):
        decode = [jobs[item.index] for item in plan if item.stage == "decode"]
        if decode:
            # Publish decoded tokens BEFORE processing any prompt chunk.
            self._execute(decode)
        for item in plan:
            if item.stage != "prefill":
                continue
            job = jobs[item.index]
            session, cancel, future, queued_at, deferrals = job
            if cancel.is_set():
                with self._condition:
                    self._cancelled_steps += 1
                self._complete([job])
                continue
            try:
                with self._execution:
                    progress = session.prefill_chunk(item.tokens)
                    with self._condition:
                        self._prefill_chunks += 1
                        self._prefill_tokens += progress.computed_tokens
                    # First token sampling uses prepared logits; no new forward.
                    result = session.step() if progress.ready and not cancel.is_set() else None
            except BaseException as exc:
                self._complete([job], error=exc)
                continue
            if progress.ready or cancel.is_set():
                if cancel.is_set():
                    with self._condition:
                        self._cancelled_steps += 1
                self._complete([job], [result])
            else:
                with self._condition:
                    # Ownership and the original future remain outstanding.
                    self._inflight -= 1
                    self._queue.append(job)
                    self._condition.notify_all()

    def _run_chunked(self):
        try:
            while True:
                with self._condition:
                    while not self._queue and not self.closed:
                        self._condition.wait()
                    if self.closed:
                        pending = list(self._queue)
                        self._queue.clear()
                        self._complete(pending, error=RuntimeError("native stream batcher is closed"), assigned=False)
                        pending.clear()
                        return
                    deadline = self._queue[0][3] + self.wait_seconds
                    target = min(self.max_batch_size, len(self._registered))
                    while not self.closed and len(self._queue) < target:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(remaining)
                    if self.closed:
                        continue
                    jobs, cancelled = [], []
                    while self._queue:
                        job = self._queue.popleft()
                        (cancelled if job[1].is_set() else jobs).append(job)
                    del job
                    self._cancelled_steps += len(cancelled)
                    self._complete(cancelled, assigned=False)
                    cancelled.clear()
                    if not jobs:
                        continue
                    plan = self._scheduler.plan([job[0] for job in jobs])
                    selected = {item.index for item in plan}
                    self._queue.extend(job for index, job in enumerate(jobs) if index not in selected)
                    self._inflight += len(plan)
                    tokens = sum(item.tokens for item in plan)
                    self._scheduler_steps += 1
                    self._scheduled_tokens += tokens
                    self._peak_step_tokens = max(self._peak_step_tokens, tokens)
                self._execute_chunked(jobs, plan)
                # Never retain a cancelled session or Future traceback while idle.
                jobs.clear()
        finally:
            with self._execution:
                self._runner.close()

    def snapshot(self):
        with self._condition:
            return {"enabled": True, "mode": ("native_token_budget_chunked_prefill" if self._scheduler
                                               else "native_dense_decode"),
                    "scheduler": {"enabled": self._scheduler is not None,
                                  "max_step_tokens": self._scheduler.max_step_tokens if self._scheduler else None,
                                  "prefill_chunk_size": self._scheduler.prefill_chunk_size if self._scheduler else 0,
                                  "steps": self._scheduler_steps, "scheduled_tokens": self._scheduled_tokens,
                                  "peak_step_tokens": self._peak_step_tokens,
                                  "prefill_chunks": self._prefill_chunks, "prefill_tokens": self._prefill_tokens},
                    "closed": self.closed, "max_batch_size": self.max_batch_size,
                    "max_resident": self.max_resident, "registered": len(self._registered),
                    "pending": len(self._queue), "inflight": self._inflight,
                    "worker_alive": self._thread is not None and self._thread.is_alive(),
                    "cancelled_steps": self._cancelled_steps,
                    "single_request_steps": self._single_request_steps,
                    "alignment_deferrals": self._alignment_deferrals, **self._runner.snapshot()}

    def close(self):
        with self._condition:
            self.closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join()
        else:
            with self._execution:
                self._runner.close()
