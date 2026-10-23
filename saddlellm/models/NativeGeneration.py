"""Request-local, resumable native generation without a background thread."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any
from weakref import WeakSet

import torch
import torch.nn.functional as F

from .InferenceCache import GenerationBuffers, InferenceLayerCache, RecurrentAttentionCache
from .NativePrefixCache import NativePrefixCache


# Only framework-owned callbacks opt into delayed delivery. Check exact class
# identity, never a flag supplied by an arbitrary streamer or its subclasses.
_batch_streamer_types = WeakSet()


def _register_batch_streamer_type(streamer_type):
    """Internal: register a framework streamer whose callbacks do not sample."""
    _batch_streamer_types.add(streamer_type)


def _rowwise_sampling_operator(operator, values):
    """Keep B1 reduction kernels while writing into one batched allocation."""
    result = torch.empty_like(values)
    for index in range(values.shape[0]):
        operator(values[index:index + 1], dim=-1, out=result[index:index + 1])
    return result


@dataclass(frozen=True)
class NativeGenerationStep:
    token_ids: torch.Tensor
    generated_tokens: int
    done: bool
    scores: torch.Tensor | None = None


@dataclass(frozen=True)
class NativePrefillStep:
    computed_tokens: int
    remaining_tokens: int
    ready: bool


class NativeGenerationSession:
    """Own one request's cache, sampling settings and token/mask buffers.

    ``prefill()`` may be run separately to prepare next-token logits. Each
    ``step()`` emits one token per row; unfinished sessions can be paused and
    resumed. A session must have one owner, not concurrent step/close calls.
    No gradient context or model thread remains active between calls.
    """

    def __init__(
        self,
        model: Any,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        do_sample: bool | None = None,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        pad_token_id: int | None = None,
        eos_token_id: Any = None,
        min_new_tokens: int = 0,
        use_cache: bool = True,
        output_scores: bool = False,
        stopping_criteria: Any = None,
        streamer: Any = None,
        cache_implementation: str = "geometric",
        logits_to_keep: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        prefix_cache: NativePrefixCache | None = None,
        buffer_factory=GenerationBuffers,
        **generation_kwargs,
    ):
        generation_kwargs.pop("token_type_ids", None)
        generation_kwargs.pop("position_ids", None)
        generation_kwargs.pop("bos_token_id", None)
        if generation_kwargs.pop("num_beams", 1) != 1:
            raise NotImplementedError(
                "native SaddleLLM generation does not support beam search"
            )
        if generation_kwargs:
            names = ", ".join(sorted(generation_kwargs))
            raise TypeError(f"unsupported native generation argument(s): {names}")
        if max_new_tokens < 0 or min_new_tokens < 0:
            raise ValueError("max_new_tokens and min_new_tokens must be non-negative")
        if min_new_tokens > max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")
        if cache_implementation not in {"geometric", "dynamic"}:
            raise ValueError("cache_implementation must be geometric or dynamic")
        if isinstance(logits_to_keep, bool) or not isinstance(logits_to_keep, int):
            raise TypeError("logits_to_keep must be a non-negative integer")
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a non-negative integer")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        do_sample = temperature > 0 if do_sample is None else do_sample
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=True")
        if seed is not None and generator is not None:
            raise ValueError("pass only one of seed and generator")
        if prefix_cache is not None and not isinstance(prefix_cache, NativePrefixCache):
            raise TypeError("prefix_cache must be a NativePrefixCache or None")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be an integer or None")
            generator = torch.Generator(device=input_ids.device).manual_seed(seed)

        model.eval()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            mask_has_padding = False
        elif attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        else:
            attention_mask = attention_mask.to(device=input_ids.device)
            mask_has_padding = not bool(torch.all(attention_mask == 1))
        self.buffers = buffer_factory(input_ids, attention_mask, max_new_tokens)
        self.model = model
        self.prompt_tokens = int(input_ids.shape[1])
        self.generated_tokens = 0
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.temperature, self.do_sample = temperature, do_sample
        self.top_p, self.top_k = top_p, top_k
        self.repetition_penalty = repetition_penalty
        self._penalty_token_ids = None
        self._penalty_token_length = 0
        self.generator = generator
        self.prefix_cache = prefix_cache
        self.prefix_cached_tokens = 0
        self.prefill_computed_tokens = 0
        self.pad_token_id = (
            model.config.pad_token_id if pad_token_id is None else int(pad_token_id)
        )
        eos_token_id = (
            model.config.eos_token_id if eos_token_id is None else eos_token_id
        )
        if isinstance(eos_token_id, torch.Tensor):
            eos_token_id = eos_token_id.flatten().tolist()
        self.eos_ids = (
            [int(item) for item in eos_token_id]
            if isinstance(eos_token_id, (list, tuple, set))
            else []
            if eos_token_id is None
            else [int(eos_token_id)]
        )
        self.use_cache, self.logits_to_keep = use_cache, logits_to_keep
        self.finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        self.finished_count = 0
        self._scores = [] if output_scores else None
        self.stopping_criteria, self.streamer = stopping_criteria, streamer
        # A single unfinished row needs no per-token padding/reduction. Reuse
        # the streamer's host token for EOS instead of synchronizing a second
        # time for finished.sum(). Custom stopping criteria keep the general path.
        # Development A/B found no repeatable whole-model gain (2026-09-28).
        # Keep the candidate disabled by default; benchmarks may explicitly
        # enable it only for one row without custom stopping criteria.
        self._single_row_fastpath = False
        self._single_active = torch.ones_like(self.finished) if input_ids.shape[0] == 1 and stopping_criteria is None else None
        self.mask_has_padding = mask_has_padding
        self.past_key_values = None
        if use_cache and max_new_tokens > 0 and cache_implementation == "geometric":
            self.past_key_values = [
                RecurrentAttentionCache(self.prompt_tokens + max_new_tokens)
                if layer.config.attention_kind == "kda" else InferenceLayerCache(
                    self.prompt_tokens + max_new_tokens,
                    window_size=layer.config.sliding_window,
                    latent=(
                        layer.config.attention_kind == "mla"
                        and layer.config.kv_lora_rank > 0
                        and layer.config.mla_cache_mode == "latent"
                    ),
                )
                for layer in model.layers
            ]
        self._pending_logits = None
        self._prefill_cursor = 0
        self._chunked_prefill_started = False
        self._done = max_new_tokens == 0
        self._closed = False
        self._failed = False
        self._streamer_ended = False
        self._lock = Lock()
        if streamer is not None:
            streamer.put(input_ids.detach().cpu())
            if self.done:
                self._end_streamer()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def scores(self) -> tuple | None:
        return tuple(self._scores) if self._scores is not None else None

    @property
    def finish_reason(self) -> str:
        if not self.done:
            raise RuntimeError("generation session has not finished")
        return "stop" if self.finished_count == self.finished.shape[0] else "length"

    @contextmanager
    def _access(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("generation session is already executing")
        try:
            if self.closed:
                raise RuntimeError("generation session is closed")
            if self._failed:
                raise RuntimeError(
                    "generation session failed; close it before starting a new request"
                )
            yield
        finally:
            self._lock.release()

    def _model_forward(self, **inputs):
        from .SaddleModeling import SaddleForCausalLM

        # Only the built-in forward understands the generation-only contract.
        # Hooks and wrappers retain the ordinary full hidden-state output.
        if (getattr(self.model.forward, "__func__", None) is SaddleForCausalLM.forward
                and not self.model._forward_hooks and not self.model._forward_pre_hooks
                and not torch.nn.modules.module._global_forward_hooks
                and not torch.nn.modules.module._global_forward_pre_hooks):
            inputs["_generation_only"] = True
        return self.model(**inputs)

    def _forward_logits(self) -> torch.Tensor:
        out, mask = self.buffers.views()
        inputs = self.model.prepare_inputs_for_generation(
            out,
            past_key_values=self.past_key_values,
            attention_mask=mask if self.mask_has_padding else None,
            use_cache=self.use_cache,
        )
        output = self._model_forward(**inputs, logits_to_keep=self.logits_to_keep)
        self.past_key_values = output.past_key_values if self.use_cache else None
        return output.logits[:, -1, :]

    def _prefill_logits(self) -> torch.Tensor:
        if self._chunked_prefill_started:
            if self._prefill_cursor < self.prompt_tokens:
                self._prefill_chunk_locked(self.prompt_tokens - self._prefill_cursor)
            return self._pending_logits
        ticket, reused, logits = None, 0, None
        out, _ = self.buffers.views()
        if self.prefix_cache is not None:
            if (self.use_cache and self.logits_to_keep == 1
                    and not self.mask_has_padding and out.shape[0] == 1
                    and isinstance(self.past_key_values, list)
                    and all(isinstance(cache, InferenceLayerCache)
                            for cache in self.past_key_values)):
                ticket, reused, logits = self.prefix_cache.restore(
                    self.model, out, self.past_key_values
                )
            else:
                self.prefix_cache.skip()
        self.prefix_cached_tokens = reused
        self.prefill_computed_tokens = self.prompt_tokens - reused
        if logits is None:
            if reused:
                # The ordinary prepare_inputs_for_generation slices to ONE
                # token when past exists. A partial prefix needs ALL uncached
                # prompt tokens, with positions inferred from the restored KV.
                output = self._model_forward(
                    input_ids=out[:, reused:], past_key_values=self.past_key_values,
                    use_cache=True, logits_to_keep=self.logits_to_keep,
                )
                self.past_key_values = output.past_key_values
                logits = output.logits[:, -1, :]
            else:
                logits = self._forward_logits()
        if (self.prefix_cache is not None and ticket is not None
                and reused != self.prompt_tokens):
            # Publish before sampling mutates repetition/EOS logits. Copies
            # isolate the entry from decode writes, cancellation and close().
            self.prefix_cache.store(self.model, ticket, self.past_key_values, logits)
        return logits

    @property
    def prefill_remaining_tokens(self) -> int:
        if self.closed or self.done or self.generated_tokens or self._pending_logits is not None:
            return 0
        return self.prompt_tokens - self._prefill_cursor

    def supports_chunked_prefill(self) -> bool:
        """Initial supported contract; no padding/prefix/window/recurrent fallback."""
        from .SaddleModeling import SaddleForCausalLM

        return (
            not self.closed and isinstance(self.model, SaddleForCausalLM)
            and getattr(self.model.forward, "__func__", None) is SaddleForCausalLM.forward
            and self.use_cache and not self.mask_has_padding and self.prefix_cache is None
            and self.buffers.tokens.shape[0] == 1 and self.prompt_tokens > 0
            and isinstance(self.past_key_values, list)
            and len(self.past_key_values) == len(self.model.layers)
            and all(isinstance(cache, InferenceLayerCache) and not cache.latent
                    and not cache.window_size for cache in self.past_key_values)
            and all(layer.config.attention_kind in {"mha", "mqa", "gqa"}
                    and not layer.config.sliding_window and not layer.config.num_experts
                    and layer.config.residual_attention_mode == "none" for layer in self.model.layers)
            and getattr(self.model, "_saddle_cuda_int8", None) is None
        )

    def _prefill_chunk_locked(self, max_tokens: int) -> NativePrefillStep:
        if self._pending_logits is not None:
            return NativePrefillStep(0, 0, True)
        start = self._prefill_cursor
        end = min(start + max_tokens, self.prompt_tokens)
        final = end == self.prompt_tokens
        output = self._model_forward(
            input_ids=self.buffers.tokens[:, start:end],
            past_key_values=self.past_key_values, use_cache=True, logits_to_keep=1,
            **({"_prefill_only": True} if not final else {}),
        )
        self.past_key_values = output.past_key_values
        self._prefill_cursor = end
        self.prefill_computed_tokens += end - start
        if final:
            self._pending_logits = output.logits[:, -1, :]
        return NativePrefillStep(end - start, self.prompt_tokens - end, final)

    @torch.inference_mode()
    def prefill_chunk(self, max_tokens: int) -> NativePrefillStep:
        """Advance bounded prompt work without sampling or emitting a token.

        The caller may schedule other requests between calls. A final chunk
        prepares logits for step(); prefill()/step() can also finish a partial
        prompt. Different GEMM/attention shapes may change low-precision output.
        """
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("prefill chunk token budget must be a positive integer")
        with self._access():
            if self.generated_tokens:
                raise RuntimeError("chunked prefill must run before the first generated token")
            if self.done:
                return NativePrefillStep(0, 0, True)
            if not self.supports_chunked_prefill():
                raise NotImplementedError("chunked prefill requires an unpadded single-row dense geometric cache without prefix caching")
            if self._pending_logits is not None:
                return NativePrefillStep(0, 0, True)
            self._chunked_prefill_started = True
            try:
                return self._prefill_chunk_locked(max_tokens)
            except BaseException:
                self._failed = True
                raise

    @torch.inference_mode()
    def prefill(self) -> torch.Tensor | None:
        """Prepare prompt KV and next-token logits, without emitting a token."""
        with self._access():
            if self.generated_tokens:
                raise RuntimeError(
                    "prefill must be called before the first generated token"
                )
            if self.done:
                return None
            try:
                if self._pending_logits is None:
                    self._pending_logits = self._prefill_logits()
                return self._pending_logits
            except BaseException:
                self._failed = True
                raise

    def _apply_repetition_penalty(self, logits, out):
        if self._penalty_token_ids is None:
            unique_ids = [torch.unique(row) for row in out]
            width = max((ids.numel() for ids in unique_ids), default=0)
            self._penalty_token_ids = out.new_empty(
                (out.shape[0], width + self.max_new_tokens), dtype=torch.long,
            )
            for index, ids in enumerate(unique_ids):
                if width:
                    # Repeating an already-seen ID pads rows without introducing
                    # a penalty for an unseen PAD token. Duplicate scatter values
                    # are identical, so an ID is still penalized exactly once.
                    self._penalty_token_ids[index, :width].copy_(ids[0].expand(width))
                    self._penalty_token_ids[index, :ids.numel()].copy_(ids)
            self._penalty_token_length = width
        ids = self._penalty_token_ids[:, :self._penalty_token_length]
        values = logits.gather(1, ids)
        penalized = torch.where(
            values < 0, values * self.repetition_penalty,
            values / self.repetition_penalty,
        )
        # Preserve the score tensor's existing in-place processing contract.
        logits.scatter_(1, ids, penalized)

    def _record_repetition_tokens(self, token):
        if self._penalty_token_ids is not None:
            start = self._penalty_token_length
            self._penalty_token_ids[:, start:start + 1].copy_(token)
            self._penalty_token_length += 1

    def _prepare_sampling_logits(self, logits: torch.Tensor, out: torch.Tensor) -> None:
        if self.repetition_penalty != 1.0:
            self._apply_repetition_penalty(logits, out)
        if self.eos_ids and self.generated_tokens < self.min_new_tokens:
            logits[:, self.eos_ids] = -torch.inf
        if self._scores is not None:
            self._scores.append(logits.detach())

    def _sampling_probabilities(self, logits: torch.Tensor, *, _preserve_single_row: bool = False) -> torch.Tensor:
        """Filter logits, optionally preserving independently sampled B1 rows.

        Sort tie order and CUDA reduction rounding can change with batch shape.
        Serving batches share pointwise kernels only; B1 sort/softmax/cumsum
        calls retain existing probability values and per-request RNG mapping.
        Ordinary multi-row generate keeps its original batch-shaped operators.
        """
        sample_logits = logits / self.temperature
        if self.top_k > 0:
            keep = min(int(self.top_k), sample_logits.shape[-1])
            threshold = torch.topk(sample_logits, keep, dim=-1).values[:, -1, None]
            sample_logits = sample_logits.masked_fill(
                sample_logits < threshold, -torch.inf
            )
        if self.top_p < 1.0:
            if _preserve_single_row:
                sorted_logits = torch.empty_like(sample_logits)
                indices = torch.empty_like(sample_logits, dtype=torch.long)
                for index in range(sample_logits.shape[0]):
                    torch.sort(sample_logits[index:index + 1], descending=True, dim=-1,
                               out=(sorted_logits[index:index + 1], indices[index:index + 1]))
                cumulative = _rowwise_sampling_operator(
                    torch.cumsum, _rowwise_sampling_operator(torch.softmax, sorted_logits))
            else:
                sorted_logits, indices = torch.sort(sample_logits, descending=True, dim=-1)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > self.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            sample_logits = torch.full_like(sample_logits, -torch.inf).scatter(
                -1, indices, sorted_logits
            )
        return (_rowwise_sampling_operator(torch.softmax, sample_logits)
                if _preserve_single_row else F.softmax(sample_logits, dim=-1))

    def _sample(self, logits: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        self._prepare_sampling_logits(logits, out)
        if not self.do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)
        return torch.multinomial(
            self._sampling_probabilities(logits),
            num_samples=1,
            generator=self.generator,
        )

    @torch.inference_mode()
    def step(self) -> NativeGenerationStep:
        with self._access():
            return self._step_locked()

    def _supports_batched_delivery(self) -> bool:
        """Whether sampling may precede this request's host token delivery."""
        return (
            type(self) is NativeGenerationSession
            and type(self.buffers) is GenerationBuffers
            and self.buffers.tokens.shape[0] == 1
            and self.buffers.tokens.is_cuda
            and self.stopping_criteria is None
            and not self._single_row_fastpath
            and (self.streamer is None or type(self.streamer) in _batch_streamer_types)
            and all(getattr(getattr(self, name), "__func__", None) is original
                    for name, original in _batch_delivery_methods)
        )

    def _step_locked(self, *, _defer_delivery: bool = False,
                     _sampled_token: torch.Tensor | None = None) -> NativeGenerationStep:
        """Internal: caller holds _access() and inference_mode, including batch runners."""
        if self.done:
            raise StopIteration
        if _defer_delivery and not self._supports_batched_delivery():
            raise ValueError("deferred token delivery requires a supported built-in session")
        if _sampled_token is not None and not _defer_delivery:
            raise ValueError("pre-sampled tokens require deferred batch delivery")
        try:
            logits = self._pending_logits
            if logits is None:
                logits = (self._prefill_logits() if not self.generated_tokens
                          else self._forward_logits())
            self._pending_logits = None
            out, _ = self.buffers.views()
            token = self._sample(logits, out) if _sampled_token is None else _sampled_token
            active = self._single_active if self._single_row_fastpath else ~self.finished
            if not self._single_row_fastpath:
                token = torch.where(
                    active[:, None], token, torch.full_like(token, self.pad_token_id)
                )
            self.buffers.append(token, active)
            if self.repetition_penalty != 1.0:
                # Include PAD in finished rows, matching the old history scan.
                self._record_repetition_tokens(token)
            self.generated_tokens += 1
            if _defer_delivery:
                # The runner copies the complete batch once and completes every
                # row before returning. No callbacks/CPU synchronization here.
                return NativeGenerationStep(
                    token, self.generated_tokens, False,
                    logits.detach() if self._scores is not None else None,
                )
            host_token = None
            if self.streamer is not None:
                host_token = token.detach().cpu()
                self.streamer.put(host_token)
            if self._single_row_fastpath:
                if self.eos_ids:
                    if host_token is None:
                        host_token = token.detach().cpu()
                    if int(host_token.item()) in self.eos_ids:
                        self.finished_count = 1
                        self.finished.fill_(True)
            elif self.eos_ids:
                is_eos = torch.zeros_like(self.finished)
                for token_id in self.eos_ids:
                    is_eos |= token.squeeze(-1).eq(token_id)
                self.finished |= is_eos & active
            if self.stopping_criteria is not None:
                out, _ = self.buffers.views()
                stop = self.stopping_criteria(out, self._scores)
                self.finished |= torch.as_tensor(
                    stop, dtype=torch.bool, device=out.device
                )
            if not self._single_row_fastpath and (self.eos_ids or self.stopping_criteria is not None):
                self.finished_count = int(self.finished.sum().item())
            self._done = (
                self.finished_count == token.shape[0]
                or self.generated_tokens >= self.max_new_tokens
            )
            self.mask_has_padding |= self.finished_count > 0
            if self.done:
                self._end_streamer()
            return NativeGenerationStep(
                token,
                self.generated_tokens,
                self.done,
                logits.detach() if self._scores is not None else None,
            )
        except BaseException:
            self._failed = True
            raise

    def _deliver_batched_step(self, step: NativeGenerationStep, host_token: torch.Tensor | None) -> NativeGenerationStep:
        """Finish a deferred B1 step using the runner's shared host transfer."""
        if self.streamer is not None:
            self.streamer.put(host_token)
        if self.eos_ids and int(host_token.item()) in self.eos_ids:
            self.finished_count = 1
            self.finished.fill_(True)
        self._done = bool(self.finished_count) or self.generated_tokens >= self.max_new_tokens
        self.mask_has_padding |= bool(self.finished_count)
        if self.done:
            self._end_streamer()
        return NativeGenerationStep(step.token_ids, step.generated_tokens, self.done, step.scores)

    def sequences(self) -> torch.Tensor:
        with self._access():
            return self.buffers.sequences()

    def _end_streamer(self) -> None:
        if self.streamer is not None and not self._streamer_ended:
            self._streamer_ended = True
            self.streamer.end()

    def close(self) -> None:
        """Release request cache/buffers. Copy needed results before closing."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("wait for the executing step before closing the session")
        try:
            if self.closed:
                return
            self._closed = True
            self._done = True
            self.past_key_values = None
            self._pending_logits = None
            self._single_active = None
            self._penalty_token_ids = None
            self._penalty_token_length = 0
            self.buffers = None
            self.model = None
            self.generator = None
            self.prefix_cache = None
            self._scores = None
            try:
                self._end_streamer()
            finally:
                self.streamer = None
                self.stopping_criteria = None
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        return self

    def __next__(self) -> NativeGenerationStep:
        return self.step()


# Capture once: patching a class method must not silently opt custom behavior
# into the callback-free contract, any more than patching an instance does.
_batch_delivery_methods = tuple(
    (name, getattr(NativeGenerationSession, name)) for name in (
        "_step_locked", "_sample", "_prepare_sampling_logits", "_sampling_probabilities",
        "_apply_repetition_penalty", "_record_repetition_tokens", "_deliver_batched_step",
        "_end_streamer",
    )
)
