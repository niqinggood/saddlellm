"""Bounded, process-local prompt KV reuse for ordinary native dense models."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from weakref import ref

import torch
from torch import nn

from .InferenceCache import InferenceLayerCache

_EXECUTION_ATTRIBUTES = (
    "eps", "base", "dim", "scaling_factor", "p",
    "attention_residual_scale", "ffn_residual_scale", "_cache_frozen",
    "_cache_frozen_length", "_rope_cache_limit", "_cpu_signature",
    "_cpu_prepack_enabled", "compile_kernel", "activation_bits",
    "in_features", "out_features",
)


def _common_prefix_length(left, right, start):
    """Compare long token runs in C, examining only the first unequal block."""
    limit = min(len(left), len(right))
    while start < limit:
        end = min(start + 128, limit)
        if left[start:end] != right[start:end]:
            for offset in range(start, end):
                if left[offset] != right[offset]:
                    return offset
        start = end
    return limit


@dataclass(frozen=True)
class _PrefixTicket:
    tokens: tuple[int, ...]
    state: tuple
    epoch: int
    reused_tokens: int = 0
    reused_key: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _PrefixEntry:
    keys_values: tuple[tuple[torch.Tensor, ...], ...]
    logits: torch.Tensor
    size_bytes: int
    ready: torch.cuda.Event | None


class NativePrefixCache:
    """Opt-in LRU of immutable prompt snapshots, bounded by tensor bytes.

    Pass the same instance as ``prefix_cache=cache`` to native ``generate`` or
    ``create_generation_session``. Only unpadded B=1 dense native requests with
    geometric KV storage qualify. Hits COPY KV into request-owned storage; this
    is not paged attention or zero-copy sharing. Sampling/stop state is never
    shared, and generated answers are not stored.

    Normal parameter updates, load_state_dict, device/dtype and execution
    changes invalidate entries on the next eligible lookup. Do not modify a
    serving model concurrently. After untracked edits (such as .data writes),
    call clear() explicitly. Share only within a trusted process/tenant.
    The budget excludes active requests, weights, allocator reserve and small
    CPU key metadata; max_entries separately bounds the number of keys.
    min_store_tokens optionally defers publishing a partial hit's full snapshot
    until its uncached suffix is large enough. Cold misses still publish normally.
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        min_prefix_tokens: int = 16,
        max_entries: int = 128,
        reuse_partial_prefixes: bool = False,
        min_store_tokens: int = 0,
    ):
        for name, value, lower in (
            ("max_bytes", max_bytes, 0),
            ("min_prefix_tokens", min_prefix_tokens, 1),
            ("max_entries", max_entries, 1),
            ("min_store_tokens", min_store_tokens, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        self.max_bytes = max_bytes
        self.min_prefix_tokens = min_prefix_tokens
        self.max_entries = max_entries
        self.min_store_tokens = min_store_tokens
        if not isinstance(reuse_partial_prefixes, bool):
            raise ValueError("reuse_partial_prefixes must be a boolean")
        self.reuse_partial_prefixes = reuse_partial_prefixes
        self._entries: OrderedDict[tuple[int, ...], _PrefixEntry] = OrderedDict()
        self._bytes = 0
        self._state = None
        self._model = None
        self._epoch = 0
        self._closed = False
        self._lock = RLock()
        self._stats = dict.fromkeys(
            (
                "lookups",
                "hits",
                "misses",
                "reused_tokens",
                "stores",
                "evictions",
                "invalidations",
                "skipped_requests",
                "rejected_entries",
                "allocation_failures",
                "deferred_stores",
            ),
            0,
        )

    @staticmethod
    def _fingerprint(model, input_device) -> tuple | None:
        # Local import avoids a cycle with NativeGeneration/SaddleModeling.
        from .CPULinear import CPUPrepackedLinear
        from .CUDAInt8 import CUDAInt8Linear
        from .OperatorBackends import (
            AttentionBackendRegistry,
            _eager_attention,
            _sdpa_attention,
            _sdpa_reference_attention,
        )
        from .SaddleModeling import (
            SaddleAttention,
            SaddleDecoderLayer,
            SaddleForCausalLM,
            SaddleRMSNorm,
            SaddleRotaryEmbedding,
            SaddleSwiGLU,
        )

        if type(model) is not SaddleForCausalLM:
            return None
        if (
            nn.modules.module._global_forward_hooks
            or nn.modules.module._global_forward_pre_hooks
        ):
            return None
        allowed = {
            SaddleForCausalLM,
            SaddleAttention,
            SaddleDecoderLayer,
            SaddleRMSNorm,
            SaddleRotaryEmbedding,
            SaddleSwiGLU,
            nn.ModuleList,
            nn.Linear,
            nn.Embedding,
            nn.Identity,
            nn.Dropout,
            CPUPrepackedLinear,
            CUDAInt8Linear,
        }
        structure = []
        parameters, buffers = [], []
        parameter_ids, buffer_ids = set(), set()
        config_states = {}
        class_attributes = {}
        for module_name, module in model.named_modules():
            if (
                type(module) not in allowed
                or module.training
                or module._forward_hooks
                or module._forward_pre_hooks
                or getattr(module.forward, "__func__", None) is not type(module).forward
                or getattr(module, "parametrizations", None)
            ):
                return None
            config = getattr(module, "config", None)
            if config is not None and id(config) not in config_states:
                # Configs contain nested blueprints. asdict()/to_dict() would
                # deepcopy them repeatedly on every lookup; repr of the live
                # attributes detects nested edits without building copies.
                config_states[id(config)] = repr(vars(config))
            if isinstance(module, SaddleAttention):
                if config.attention_kind == "mla" or config.sliding_window:
                    return None
                spec = AttentionBackendRegistry.resolve(config.attention_backend)
                handler = AttentionBackendRegistry._handlers.get(spec.name)
                if handler not in {_eager_attention, _sdpa_attention, _sdpa_reference_attention}:
                    return None
                structure.append((spec.name, id(handler), spec.supports_compact_gqa))
            module_type = type(module)
            if module_type not in class_attributes:
                # Preserve class-level attributes/properties, while avoiding
                # nn.Module.__getattr__ for the many absent optional fields.
                class_attributes[module_type] = {
                    name for name in _EXECUTION_ATTRIBUTES
                    if any(name in vars(base) for base in module_type.__mro__)
                }
            attributes = tuple(
                (name, value if not isinstance(value, torch.Tensor) else id(value))
                for name in _EXECUTION_ATTRIBUTES
                if (name in module.__dict__ or name in module._parameters
                    or name in module._buffers or name in class_attributes[module_type])
                and (value := getattr(module, name, None)) is not None
            )
            structure.append(
                (
                    id(module),
                    id(type(module).forward),
                    attributes,
                    config_states.get(id(config)),
                )
            )
            # Scan the tree once, retaining named_parameters/named_buffers'
            # first-name and tied-tensor deduplication semantics separately.
            for members, seen, destination in (
                (module._parameters, parameter_ids, parameters),
                (module._buffers, buffer_ids, buffers),
            ):
                for attribute, tensor in members.items():
                    if tensor is None or id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    name = f"{module_name}.{attribute}" if module_name else attribute
                    destination.append((name, tensor, module_type))
        tensors = []
        for name, tensor, owner_type in parameters + buffers:
            attribute = name.rpartition(".")[2]
            cpu_weight_cache = (
                attribute == "_cpu_packed_weight"
                and owner_type is CPUPrepackedLinear
                and tensor.layout == torch._mkldnn
                and input_device.type == "cpu"
            )
            if (
                tensor.is_meta
                or tensor.is_inference()
                or (tensor.layout != torch.strided and not cpu_weight_cache)
                or tensor.device != input_device
                or type(tensor) not in {torch.Tensor, nn.Parameter}
            ):
                return None
            tensors.append(
                (
                    name,
                    id(tensor),
                    tensor._version,
                    tensor.device,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()) if tensor.layout == torch.strided else None,
                )
            )
        device_type = input_device.type
        execution = (
            torch.is_autocast_enabled(device_type),
            torch.get_autocast_dtype(device_type),
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
            torch.are_deterministic_algorithms_enabled(),
            torch.backends.mkldnn.enabled,
            torch.backends.mkldnn.deterministic,
        )
        return tuple(structure), tuple(tensors), execution

    def skip(self) -> None:
        with self._lock:
            self._stats["skipped_requests"] += 1

    def _clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
        self._epoch += 1

    def clear(self) -> None:
        """Drop snapshots; in-flight requests keep their private KV copies."""
        with self._lock:
            self._clear()
            self._stats["invalidations"] += 1

    def close(self) -> None:
        """Release entries and reject future lookups/publication, idempotently."""
        with self._lock:
            self._closed = True
            self._clear()

    @torch.inference_mode()
    def restore(self, model, input_ids, caches):
        """Internal: return publication ticket, reused count and optional logits."""
        if any(
            not isinstance(cache, InferenceLayerCache) or cache.window_size
            for cache in caches
        ):
            # Snapshot slicing assumes full history. A window-aware prefix tree
            # must prove that every layer's required tail is still available.
            with self._lock:
                if self._entries:
                    self._clear()
                    self._stats["invalidations"] += 1
                self.skip()
            return None, 0, None
        if not self.max_bytes or input_ids.shape[1] < self.min_prefix_tokens:
            self.skip()
            return None, 0, None
        state = self._fingerprint(model, input_ids.device)
        if state is None:
            # A newly installed hook/adapter must not leave old entries usable.
            with self._lock:
                if self._entries:
                    self._clear()
                    self._stats["invalidations"] += 1
                self.skip()
            return None, 0, None
        tokens = tuple(input_ids[0].tolist())
        with self._lock:
            if self._closed:
                self.skip()
                return None, 0, None
            if (
                self._state != state
                or self._model is None
                or self._model() is not model
            ):
                if self._state is not None:
                    self._stats["invalidations"] += 1
                self._clear()
                self._state, self._model = state, ref(model)
            self._stats["lookups"] += 1
            best_key, best_count = None, 0
            if tokens in self._entries:
                best_key, best_count = tokens, len(tokens)
            candidates = (
                reversed(self._entries)
                if self.reuse_partial_prefixes and best_key is None
                else ()
            )
            required_prefix = tokens[: self.min_prefix_tokens]
            for key in candidates:
                if key[: self.min_prefix_tokens] != required_prefix:
                    continue
                count = _common_prefix_length(key, tokens, self.min_prefix_tokens)
                # We only store logits at the END of an old prompt. If a new
                # prompt ends earlier, recompute its last token to obtain logits.
                if count == len(tokens) and count < len(key):
                    count -= 1
                if count >= self.min_prefix_tokens and count > best_count:
                    best_key, best_count = key, count
                if count == len(tokens):
                    break
            if best_key is None:
                self._stats["misses"] += 1
                return _PrefixTicket(tokens, state, self._epoch), 0, None
            ticket = _PrefixTicket(tokens, state, self._epoch, best_count, best_key)
            entry = self._entries[best_key]
            stream = (
                torch.cuda.current_stream(input_ids.device)
                if input_ids.is_cuda
                else None
            )
            if stream is not None:
                stream.wait_event(entry.ready)
            try:
                for cache, values in zip(caches, entry.keys_values):
                    if stream is not None:
                        for value in values:
                            value.record_stream(stream)
                    cache.append(*(value[:, :, :best_count, :] for value in values))
                logits = None
                if best_count == len(tokens):
                    if stream is not None:
                        entry.logits.record_stream(stream)
                    logits = entry.logits.clone()
            except torch.cuda.OutOfMemoryError:
                for cache in caches:
                    cache.buffers = []
                    cache.length = cache.capacity = 0
                self._clear()
                self._stats["allocation_failures"] += 1
                return None, 0, None
            self._entries.move_to_end(best_key)
            self._stats["hits"] += 1
            self._stats["reused_tokens"] += best_count
            return ticket, best_count, logits

    @torch.inference_mode()
    def store(self, model, ticket, caches, logits) -> None:
        """Internal: publish a complete successful prompt, never decode state."""
        if ticket is None or len(ticket.tokens) < self.min_prefix_tokens:
            return
        with self._lock:
            if (
                self._closed
                or ticket.epoch != self._epoch
                or ticket.state != self._state
                or self._model() is not model
            ):
                return
            length = len(ticket.tokens)
            if (0 < ticket.reused_tokens < length
                    and length - ticket.reused_tokens < self.min_store_tokens
                    and ticket.reused_key in self._entries):
                # No new cache state is published, so a second model scan and
                # copying the entire extended prompt are unnecessary. The next
                # restore still validates every model/execution change before
                # using the retained prefix. Request KV always remains private.
                # LRU eviction does not change the epoch: if another request
                # evicted that prefix, publish this request instead of losing
                # both opportunities for future reuse.
                self._stats["deferred_stores"] += 1
                return
            if self._fingerprint(model, logits.device) != ticket.state:
                return
            if len(caches) != len(model.layers) or any(
                not isinstance(cache, InferenceLayerCache)
                or cache.window_size
                or cache.latent
                or cache.length != length
                or len(cache.buffers) != 2
                for cache in caches
            ):
                return
            size = logits.numel() * logits.element_size()
            size += sum(
                value[:, :, :length, :].numel() * value.element_size()
                for cache in caches
                for value in cache.buffers
            )
            if size > self.max_bytes:
                self._stats["rejected_entries"] += 1
                return
            # Existing exact entry is already immutable and up to date. Do not
            # clone/re-publish it for every hot request.
            if ticket.tokens in self._entries:
                self._entries.move_to_end(ticket.tokens)
                return
            while self._entries and (
                self._bytes + size > self.max_bytes
                or len(self._entries) >= self.max_entries
            ):
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed.size_bytes
                self._stats["evictions"] += 1
                del removed
            try:
                snapshots = tuple(
                    tuple(value[:, :, :length, :].clone() for value in cache.buffers)
                    for cache in caches
                )
                saved_logits = logits.detach().clone()
                ready = None
                if logits.is_cuda:
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(logits.device))
            except torch.cuda.OutOfMemoryError:
                self._clear()
                self._stats["allocation_failures"] += 1
                return
            self._entries[ticket.tokens] = _PrefixEntry(
                snapshots, saved_logits, size, ready
            )
            self._bytes += size
            self._stats["stores"] += 1

    def snapshot(self) -> dict:
        """Health counters only: no prompts, tokens, hashes or model parameters."""
        with self._lock:
            return {
                "enabled": bool(self.max_bytes) and not self._closed,
                "max_bytes": self.max_bytes,
                "retained_tensor_bytes": self._bytes,
                "max_entries": self.max_entries,
                "entries": len(self._entries),
                "min_prefix_tokens": self.min_prefix_tokens,
                "min_store_tokens": self.min_store_tokens,
                "partial_reuse_enabled": self.reuse_partial_prefixes,
                **self._stats,
            }
