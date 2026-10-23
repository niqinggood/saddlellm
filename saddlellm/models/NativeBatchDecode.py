"""Experimental batched decode for independently owned dense sessions.

Not paged attention or a serving scheduler. Prefill and unsupported sessions
retain their normal path. Compatible active requests share a model forward;
KV history is copied on cohort changes/growth, not on every decode tick.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from threading import Lock

import torch

from .InferenceCache import InferenceLayerCache, _IndependentLayerCache
from .NativeGeneration import NativeGenerationSession


class _DecodeCohort:
    def __init__(self, sessions):
        self.sessions = tuple(sessions)
        self.caches = []
        self.row_caches = [tuple(session.past_key_values) for session in sessions]
        self.bound_buffers = None
        self.copied_bytes = 0
        for layer_index in range(len(sessions[0].past_key_values)):
            rows = [session.past_key_values[layer_index] for session in sessions]
            first = rows[0]
            cache = InferenceLayerCache(max(row.max_length for row in rows))
            cache.length = first.length
            cache.capacity = min(cache.max_length, max(first.length + 1, first.length * 2))
            for tensor_index in (0, 1):
                sample = first.buffers[tensor_index]
                storage = sample.new_empty((len(rows), sample.shape[1], cache.capacity, sample.shape[3]))
                for row_index, row in enumerate(rows):
                    source = row.buffers[tensor_index][:, :, :cache.length, :]
                    storage[row_index:row_index + 1, :, :cache.length, :].copy_(source)
                    self.copied_bytes += source.numel() * source.element_size()
                cache.buffers.append(storage)
            self.caches.append(cache)
            # Publish one complete layer at a time so joining does not retain
            # every original layer AND the full joined history simultaneously.
            # All session locks are held and no forward has begun. On OOM,
            # published and untouched layers both still contain the same valid
            # history/length: the caller may close or retry these sessions.
            self._bind_layer(layer_index, cache)
        self._remember_buffers()

    def matches(self):
        for row_index, session in enumerate(self.sessions):
            if session.past_key_values is None or len(session.past_key_values) != len(self.caches):
                return False
            for layer_index, (row, cache) in enumerate(zip(session.past_key_values, self.caches)):
                if (row is not self.row_caches[row_index][layer_index]
                        or row.length != cache.length
                        or tuple(id(value) for value in row.buffers) != self.bound_buffers[row_index][layer_index]):
                    return False
        return True

    def _bind_layer(self, index, cache):
        rows = [layers[index] for layers in self.row_caches]
        # Build every complete K/V view pair before publishing any row. A
        # failed allocation cannot leave one row with new K but missing V.
        buffers = [[value[row_index:row_index + 1, :, :min(row.max_length, cache.capacity), :]
                    for value in cache.buffers] for row_index, row in enumerate(rows)]
        for row, pair in zip(rows, buffers):
            row.length, row.capacity = cache.length, min(row.max_length, cache.capacity)
            row.buffers = pair

    def _remember_buffers(self):
        self.storage_identities = tuple(tuple(id(value) for value in cache.buffers) for cache in self.caches)
        self.bound_buffers = [[tuple(id(value) for value in row.buffers) for row in rows]
                              for rows in self.row_caches]

    def bind_rows(self):
        identities = tuple(tuple(id(value) for value in cache.buffers) for cache in self.caches)
        if identities != self.storage_identities:
            for index, cache in enumerate(self.caches):
                self._bind_layer(index, cache)
            self._remember_buffers()
        else:
            for rows in self.row_caches:
                for row, cache in zip(rows, self.caches):
                    row.length = cache.length

    @property
    def token_slots(self):
        return len(self.sessions) * max(cache.capacity for cache in self.caches)


class NativeBatchDecodeRunner:
    """Advance one token per session, batching compatible decode work.

    Advanced opt-in API; HTTP uses it only with native_stream_batching=True.
    Call with live sessions from ONE immutable eval model. The caller owns
    admission, cancellation, session.close(), and output delivery. Membership
    may change between ticks. Unpadded dense MHA/MQA/GQA geometric caches batch.
    Equal lengths share a cohort; supported standard layers can keep independent
    KV for unequal lengths, batch projections, and run attention per request.
    Other sessions transparently step individually.
    FP32 uses batched Linear. ``decode_mode="compatible"`` keeps B1-shaped
    CUDA BF16 Linear; ``"throughput"`` opts into batched Linear and its possible
    rounding/output differences. Sampling stays request-local, including supplied
    generators. Unseeded/shared-generator draws remain dependent on scheduling.
    This is not a promise of bitwise invariance for every model/kernel version.
    """

    _batch_dtypes = frozenset({torch.float32, torch.bfloat16})
    _bf16_rowwise_linear = True

    def __init__(self, model, *, max_batch_size=8, max_batch_tokens=32768,
                 decode_mode="compatible"):
        from .SaddleModeling import SaddleForCausalLM

        if not isinstance(model, SaddleForCausalLM):
            raise TypeError("NativeBatchDecodeRunner requires a native Saddle model")
        for name, value in (("max_batch_size", max_batch_size), ("max_batch_tokens", max_batch_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if decode_mode not in ("compatible", "throughput"):
            raise ValueError("decode_mode must be compatible or throughput")
        self.model = model
        self.decode_mode = decode_mode
        self._bf16_rowwise_linear = decode_mode == "compatible"
        self.max_batch_size, self.max_batch_tokens = max_batch_size, max_batch_tokens
        self._cohorts = {}
        self._lock = Lock()
        self.closed = False
        self.batched_forwards = self.batched_tokens = self.individual_steps = 0
        self.cohort_builds = self.history_copy_bytes = self.largest_batch = 0
        self.rowwise_linear_forwards = 0
        self.sampling_batched_tokens = 0
        self.batched_linear_forwards = 0
        self.batch_size_histogram = Counter()
        self.individual_reasons = Counter()
        self.independent_kv_forwards = 0
        self.batched_host_transfers = 0
        self.batched_host_tokens = 0
        self.batched_sampling_forwards = 0

    def _supports_independent_kv(self):
        from .SaddleModeling import SaddleAttention, SaddleDecoderLayer, SaddleSwiGLU

        # A custom layer/attention contract may assume tuple or dense-batch KV.
        # Keep those on the existing path; standard layer residual/MLP logic is
        # reused by the new path rather than reimplemented in the runner.
        return (not self.model._forward_hooks and not self.model._forward_pre_hooks
                and not torch.nn.modules.module._global_forward_hooks
                and not torch.nn.modules.module._global_forward_pre_hooks
                and self.model.config.residual_attention_mode == "none"
                and all(type(layer) is SaddleDecoderLayer and not layer.is_moe
                        and getattr(layer.forward, "__func__", None) is SaddleDecoderLayer.forward
                        and not layer._forward_hooks and not layer._forward_pre_hooks
                        and type(layer.self_attn) is SaddleAttention
                        and getattr(layer.self_attn.forward, "__func__", None) is SaddleAttention.forward
                        and not layer.self_attn._forward_hooks and not layer.self_attn._forward_pre_hooks
                        and type(layer.mlp) is SaddleSwiGLU
                        and getattr(layer.mlp.forward, "__func__", None) is SaddleSwiGLU.forward
                        and layer.config.residual_attention_mode == "none" for layer in self.model.layers))

    def _retain(self, identity, cohort):
        # Retain only live, disjoint groups. Bound runner-held references by
        # both row count and physical KV token slots, evicting the oldest group.
        # Session-owned row views remain valid after an eviction.
        if any(session.done or session.closed or session._failed for session in identity):
            return
        self._cohorts[identity] = cohort
        while (sum(len(key) for key in self._cohorts) > self.max_batch_size
               or sum(value.token_slots for value in self._cohorts.values()) > self.max_batch_tokens):
            self._cohorts.pop(next(iter(self._cohorts)))

    def _key(self, session):
        return self._eligibility(session)[0]

    def _eligibility(self, session):
        from .SaddleModeling import SaddleForCausalLM

        caches = session.past_key_values
        if (getattr(self.model.forward, "__func__", None) is not SaddleForCausalLM.forward
                or getattr(self.model.prepare_inputs_for_generation, "__func__", None)
                is not SaddleForCausalLM.prepare_inputs_for_generation
                or getattr(self.model, "_saddle_cuda_int8", None) is not None):
            return None, "unsupported_model"
        if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
            return None, "autocast"
        if not session.generated_tokens or session._pending_logits is not None:
            return None, "prefill_or_pending_logits"
        if not session.use_cache or session.mask_has_padding or session.buffers.tokens.shape[0] != 1:
            return None, "unsupported_request"
        if not isinstance(caches, list) or not caches or len(caches) != len(self.model.layers):
            return None, "unsupported_cache"
        if any(layer.config.attention_kind not in {"mha", "mqa", "gqa"}
                       or layer.config.attention_backend not in {"eager", "sdpa", "sdpa_reference"}
                       or layer.config.sliding_window or layer.config.num_experts > 0
                       for layer in self.model.layers):
            return None, "unsupported_architecture"
        length = caches[0].length if isinstance(caches[0], InferenceLayerCache) else -1
        if any(not isinstance(cache, InferenceLayerCache) or cache.latent or cache.window_size
               or cache.length != length or len(cache.buffers) != 2 for cache in caches):
            return None, "unsupported_cache"
        if session.buffers.length != length + 1:
            return None, "cache_position_mismatch"
        tensor = caches[0].buffers[0]
        if tensor.dtype not in self._batch_dtypes:
            return None, "unsupported_dtype"
        if tensor.dtype == torch.bfloat16:
            from .SaddleModeling import SaddleAttention, SaddleDecoderLayer, SaddleSwiGLU

            if (not tensor.is_cuda or self.model.config.residual_attention_mode != "none"
                    or self.model.embed_tokens.weight.dtype != torch.bfloat16
                    or type(self.model.lm_head) is not torch.nn.Linear
                    or any(type(layer) is not SaddleDecoderLayer
                           or layer.config.residual_attention_mode != "none"
                           or type(layer.self_attn) is not SaddleAttention
                           or type(layer.mlp) is not SaddleSwiGLU for layer in self.model.layers)
                    or any(type(module) is not torch.nn.Linear for module in self.model.modules()
                           if isinstance(module, torch.nn.Linear))):
                return None, "unsupported_bf16_model"
        return (length, tensor.device, tensor.dtype), None

    @torch.inference_mode()
    def step(self, sessions):
        """Return aligned NativeGenerationSteps; sessions must be live and unique."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("batch decode runner is already executing")
        try:
            if self.closed:
                raise RuntimeError("batch decode runner is closed")
            sessions = list(sessions)
            if len({id(session) for session in sessions}) != len(sessions):
                raise ValueError("sessions must be unique")
            if len(sessions) > self.max_batch_size:
                raise ValueError("too many sessions for max_batch_size")
            with ExitStack() as locks:
                for session in sessions:
                    if not isinstance(session, NativeGenerationSession) or session.model is not self.model:
                        raise ValueError("all sessions must belong to the runner model")
                    locks.enter_context(session._access())
                    if session.done:
                        raise ValueError("remove finished sessions before the next batch step")
                groups = {}
                reasons = {}
                keys = {}
                independent_supported = len(sessions) > 1 and self._supports_independent_kv()
                for index, session in enumerate(sessions):
                    key, reasons[index] = self._eligibility(session)
                    keys[index] = key
                    group_key = key[1:] if key is not None and independent_supported else key
                    groups.setdefault(group_key if key is not None else ("single", index), []).append(index)
                results = [None] * len(sessions)
                self._cohorts = {
                    key: value for key, value in self._cohorts.items()
                    if not any(item.done or item.closed or item._failed for item in key)
                }
                for indices in groups.values():
                    # Arrival order is not row identity. HTTP consumers may
                    # request their next token in a different order each tick;
                    # keep stable rows so that does not recopy all KV history.
                    indices = sorted(indices, key=lambda index: id(sessions[index]))
                    group = [sessions[index] for index in indices]
                    independent = (len(group) > 1 and len({keys[index][0] for index in indices}) > 1)
                    identity = tuple(group)
                    cohort = self._cohorts.pop(identity, None)
                    touched = set(group)
                    # A subset advancing alone or moving to a new group makes
                    # the old cohort invalid. Unrelated groups keep their KV.
                    self._cohorts = {key: value for key, value in self._cohorts.items()
                                     if touched.isdisjoint(key)}
                    # Limit reserved padded token slots for a shared cohort;
                    # this is NOT a global GPU memory/admission guarantee.
                    slots = (sum(s.buffers.capacity for s in group) if independent
                             else len(group) * max(s.buffers.capacity for s in group))
                    if len(group) == 1 or slots > self.max_batch_tokens:
                        cohort = None
                        for index in indices:
                            results[index] = sessions[index]._step_locked()
                            self.individual_steps += 1
                            self.individual_reasons[reasons[index] or (
                                "no_compatible_peer" if len(group) == 1 else "batch_token_budget")] += 1
                        continue
                    if not independent and (cohort is None or not cohort.matches()):
                        cohort = _DecodeCohort(group)
                        self.cohort_builds += 1
                        self.history_copy_bytes += cohort.copied_bytes
                    try:
                        input_ids = torch.cat([s.buffers.tokens[:, s.buffers.length - 1:s.buffers.length]
                                               for s in group], dim=0)
                        if independent:
                            positions = torch.tensor([s.past_key_values[0].length for s in group],
                                                     device=input_ids.device, dtype=torch.long)
                            caches = [_IndependentLayerCache([s.past_key_values[layer] for s in group], positions)
                                      for layer in range(len(self.model.layers))]
                        else:
                            caches = cohort.caches
                        rowwise = self._bf16_rowwise_linear and keys[indices[0]][2] == torch.bfloat16
                        output = group[0]._model_forward(input_ids=input_ids, past_key_values=caches,
                                            use_cache=True, logits_to_keep=1,
                                            **({"_decode_rowwise_linear": True} if rowwise else {}))
                        if not independent:
                            cohort.bind_rows()
                        self.independent_kv_forwards += int(independent)
                        self.batched_forwards += 1
                        self.batched_tokens += len(group)
                        self.rowwise_linear_forwards += int(rowwise)
                        self.batched_linear_forwards += int(not rowwise)
                        self.sampling_batched_tokens += sum(session.do_sample for session in group)
                        self.batch_size_histogram[len(group)] += 1
                        self.largest_batch = max(self.largest_batch, len(group))
                        for row_index, index in enumerate(indices):
                            # Each session keeps its own penalties, EOS/stop,
                            # streamer, score history and generation budget.
                            sessions[index]._pending_logits = output.logits[row_index:row_index + 1, -1, :]
                        # Storage rows stay stable, but shared/global RNG draws
                        # within a group follow the caller's submitted order.
                        submitted = sorted(indices)
                        # User callbacks can consume RNG or observe peers. Delay
                        # delivery only when the ENTIRE group has the built-in
                        # contract, preserving arbitrary callback ordering on the
                        # ordinary path. Sampling retains submitted request order.
                        if all(s._supports_batched_delivery() for s in group):
                            common_sampling = len({(s.do_sample, s.temperature, s.top_k, s.top_p)
                                                   for s in group}) == 1
                            probabilities = None
                            if common_sampling:
                                # Share pointwise sampling work, retaining B1
                                # sort/reductions and multinomial for exact tie,
                                # rounding and private/shared/global RNG behavior.
                                for session in group:
                                    logits = session._pending_logits
                                    session._prepare_sampling_logits(logits, session.buffers.views()[0])
                                logits = output.logits[:, -1, :]
                                probabilities = (group[0]._sampling_probabilities(logits, _preserve_single_row=True)
                                                 if group[0].do_sample else logits.argmax(dim=-1, keepdim=True))
                                self.batched_sampling_forwards += 1
                            row_indices = {index: row for row, index in enumerate(indices)}
                            pending = []
                            for index in submitted:
                                session = sessions[index]
                                token = None
                                if common_sampling:
                                    row = row_indices[index]
                                    values = probabilities[row:row + 1]
                                    token = (torch.multinomial(values, num_samples=1, generator=session.generator)
                                             if session.do_sample else values)
                                pending.append(session._step_locked(_defer_delivery=True, _sampled_token=token))
                            needs_host = any(s.eos_ids or s.streamer is not None for s in group)
                            host_tokens = None
                            if needs_host:
                                host_tokens = torch.cat([step.token_ids for step in pending], dim=0).cpu()
                                self.batched_host_transfers += 1
                                self.batched_host_tokens += len(pending)
                            for row, (index, step) in enumerate(zip(submitted, pending)):
                                results[index] = sessions[index]._deliver_batched_step(
                                    step, host_tokens[row:row + 1] if needs_host else None)
                        else:
                            for index in submitted:
                                results[index] = sessions[index]._step_locked()
                        if not independent:
                            self._retain(identity, cohort)
                    except BaseException:
                        # A forward can mutate only some layers before failing.
                        # Never resume any row from that partially advanced KV.
                        for session in group:
                            session._failed = True
                        self._cohorts.clear()
                        raise
                return results
        finally:
            self._lock.release()

    def snapshot(self):
        cohorts = tuple(self._cohorts.values())
        return {"experimental": True, "batched_forwards": self.batched_forwards,
                "decode_mode": self.decode_mode, "sampling_supported": True,
                "independent_kv_forwards": self.independent_kv_forwards,
                "sampling_batched_tokens": self.sampling_batched_tokens,
                "batched_host_transfers": self.batched_host_transfers,
                "batched_host_tokens": self.batched_host_tokens,
                "batched_sampling_forwards": self.batched_sampling_forwards,
                "batched_linear_forwards": self.batched_linear_forwards,
                "batch_size_histogram": dict(self.batch_size_histogram),
                "individual_reasons": dict(self.individual_reasons),
                "batch_dtypes": sorted(str(dtype) for dtype in self._batch_dtypes),
                "rowwise_linear_forwards": self.rowwise_linear_forwards,
                "batched_tokens": self.batched_tokens, "individual_steps": self.individual_steps,
                "cohort_builds": self.cohort_builds, "cohort_history_copy_bytes": self.history_copy_bytes,
                "largest_batch": self.largest_batch, "retained_cohorts": len(cohorts),
                "retained_cohort_token_slots": sum(value.token_slots for value in cohorts)}

    def close(self):
        """Release runner references, not caller-owned sessions or model weights."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("wait for batch execution before closing the runner")
        try:
            self._cohorts.clear()
            self.model = None
            self.closed = True
        finally:
            self._lock.release()

    def forget(self, session):
        """Drop retained cohorts containing a departed caller after active work.

        Row views still belong to live sessions. This does not close them or
        free weights; it prevents an idle runner retaining a finished request.
        """
        with self._lock:
            self._cohorts = {key: value for key, value in self._cohorts.items() if session not in key}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
