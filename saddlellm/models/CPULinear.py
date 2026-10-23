"""Optional decode-only CPU weight preordering, with dense checkpoint weights.

Uses PyTorch's bundled oneDNN operators, not a custom compiler extension. The
opaque weight is an extra, nonpersistent inference cache. Ordinary prefill,
training, autocast and unsupported devices retain the original Linear path.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def available_cpu_memory() -> int | None:
    """Best-effort available RAM, clamped to the process's Linux v2 cgroup."""
    available = None
    try:
        import psutil

        available = psutil.virtual_memory().available
    except ImportError:
        try:
            lines = Path("/proc/meminfo").read_text().splitlines()
            for line in lines:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError):
            pass
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                relative = line.split("::", 1)[1].lstrip("/")
                root = Path("/sys/fs/cgroup")
                group = root / relative
                # Check ancestors too: the effective limit is the tightest.
                while group == root or root in group.parents:
                    limit_file = group / "memory.max"
                    usage_file = group / "memory.current"
                    if limit_file.is_file() and usage_file.is_file():
                        limit = limit_file.read_text().strip()
                        if limit != "max":
                            remaining = max(
                                0, int(limit) - int(usage_file.read_text().strip())
                            )
                            available = (
                                remaining
                                if available is None
                                else min(available, remaining)
                            )
                    if group == root:
                        break
                    group = group.parent
    except (OSError, ValueError):
        pass
    return available


def cpu_prepack_supported(dtype: torch.dtype) -> tuple[bool, str]:
    if dtype != torch.float32:
        return False, "CPU prepacking currently requires float32; BF16 failed the real 8B equivalence gate"
    if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled:
        return False, "oneDNN is unavailable or disabled"
    try:
        with torch.inference_mode():
            weight = torch.ones((64, 64), dtype=dtype)
            packed = torch.ops.mkldnn._reorder_linear_weight(weight, 1)
            result = torch.ops.mkldnn._linear_pointwise(
                torch.ones((1, 64), dtype=dtype), packed, None, "none", [], None
            )
            if not torch.equal(result, torch.full_like(result, 64)):
                return False, "oneDNN capability probe returned unexpected values"
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        return False, f"oneDNN CPU/dtype capability probe failed: {type(exc).__name__}"
    return True, ""


class CPUPrepackedLinear(nn.Linear):
    """Keep canonical dense parameters and one M=1 oneDNN inference cache.

    An exclusively owned, plain unhooked Linear can be wrapped before serving.
    No adapter, hook or training parameter is rewritten. Weight version changes,
    state loading, training and device/dtype changes invalidate the extra cache.
    """

    def __init__(self, source: nn.Linear):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device="meta",
            dtype=source.weight.dtype,
        )
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer("_cpu_packed_weight", None, persistent=False)
        self._cpu_signature = None
        self.cpu_prepack_hits = 0
        self._cpu_prepack_enabled = True
        self.train(source.training)

    def _signature(self):
        weight = self.weight
        if weight.is_meta or weight.is_inference():
            return None
        return (
            id(weight),
            weight.data_ptr(),
            weight._version,
            weight.device,
            weight.dtype,
            tuple(weight.shape),
        )

    def clear_cpu_cache(self):
        self._cpu_packed_weight = None
        self._cpu_signature = None

    def prepare_cpu_cache(self):
        if self.training or self.weight.device.type != "cpu":
            raise ValueError("CPU cache preparation requires eval mode on CPU")
        if self.weight.dtype != torch.float32:
            raise ValueError("CPU prepacking currently requires float32")
        signature = self._signature()
        if signature is None:
            raise ValueError("CPU cache requires ordinary materialized parameters")
        with torch.inference_mode(False), torch.no_grad():
            self._cpu_packed_weight = torch.ops.mkldnn._reorder_linear_weight(
                self.weight.detach(), 1
            )
        self._cpu_signature = signature

    def _apply(self, fn, recurse=True):
        # Opaque CPU tensors cannot be moved to CUDA/MPS by generic _apply.
        self.clear_cpu_cache()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode=True):
        if mode:
            self.clear_cpu_cache()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cpu_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, inputs):
        if (
            not self._cpu_prepack_enabled
            or self.training
            or torch.is_grad_enabled()
            or inputs.device.type != "cpu"
            or self.weight.device.type != "cpu"
            or inputs.dtype != self.weight.dtype
            or inputs.dtype != torch.float32
            or inputs.layout != torch.strided
            or not inputs.is_contiguous()
            or inputs.numel() != self.in_features
            or torch.is_autocast_enabled("cpu")
            or torch.compiler.is_compiling()
            or not torch.backends.mkldnn.enabled
        ):
            return F.linear(inputs, self.weight, self.bias)
        # Only caches prepared at installation are used. Invalidation falls
        # back rather than allocating another large cache during a request.
        if self._cpu_packed_weight is None or self._cpu_signature != self._signature():
            self.clear_cpu_cache()
            return F.linear(inputs, self.weight, self.bias)
        result = torch.ops.mkldnn._linear_pointwise(
            inputs.reshape(1, self.in_features),
            self._cpu_packed_weight,
            self.bias,
            "none",
            [],
            None,
        )
        self.cpu_prepack_hits += 1
        return result.reshape(*inputs.shape[:-1], self.out_features)


def cpu_prepack_status(model: nn.Module) -> dict:
    layers = [m for m in model.modules() if isinstance(m, CPUPrepackedLinear)]
    stored = [m for m in layers if m._cpu_packed_weight is not None]
    active = [
        m
        for m in stored
        if m._cpu_signature == m._signature()
        and not m.training
        and torch.backends.mkldnn.enabled
        and m._cpu_prepack_enabled
    ]
    return {
        "wrapped_linears": len(layers),
        "active_caches": len(active),
        "cache_bytes": sum(m.weight.numel() * m.weight.element_size() for m in stored),
        "decode_calls": sum(m.cpu_prepack_hits for m in layers),
        "backend": "onednn_m1" if active else "torch",
    }


@contextmanager
def disable_cpu_prepack(model: nn.Module):
    """Benchmark-only A/B switch; requires exclusive ownership of the model.

    Keep both original parameters and prepared caches resident and untouched,
    changing only which Linear operator executes. Never switch a serving model
    concurrently with requests. A finally block restores every previous flag.
    """
    previous = [
        (module, module._cpu_prepack_enabled)
        for module in model.modules()
        if isinstance(module, CPUPrepackedLinear)
    ]
    try:
        for module, _ in previous:
            module._cpu_prepack_enabled = False
        yield
    finally:
        for module, value in previous:
            module._cpu_prepack_enabled = value


def prepare_cpu_linears(model: nn.Module, *, max_prepack_bytes=None) -> dict:
    """Prepare bounded inference caches, never changing the original weights."""
    if max_prepack_bytes is not None and (
        isinstance(max_prepack_bytes, bool)
        or not isinstance(max_prepack_bytes, int)
        or max_prepack_bytes < 0
    ):
        raise ValueError("max_prepack_bytes must be a non-negative integer or None")
    if any(m.training for m in model.modules()):
        raise ValueError("call model.eval() before preparing CPU linears")
    report = {"prepared": 0, "already_prepared": 0, "cache_bytes": 0, "skipped": {}}
    if (
        nn.modules.module._global_forward_hooks
        or nn.modules.module._global_forward_pre_hooks
    ):
        report["skipped"]["<model>"] = "global module hooks are installed"
        return report
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, CPUPrepackedLinear):
            report["already_prepared"] += 1
        elif type(module) is nn.Linear:
            if not name:
                report["skipped"]["<root>"] = "wrap a root Linear explicitly"
                continue
            if (
                module.weight.device.type != "cpu"
                or module.weight.is_meta
                or module.weight.is_inference()
                or module.weight.layout != torch.strided
                or module._forward_hooks
                or module._forward_pre_hooks
                or module._backward_hooks
                or module._backward_pre_hooks
                or module._state_dict_hooks
                or module._state_dict_pre_hooks
                or module._load_state_dict_pre_hooks
                or module._load_state_dict_post_hooks
                or module.weight._backward_hooks
            ):
                report["skipped"][name] = "requires plain, unhooked CPU parameters"
            else:
                candidates.append((name, module))
    supported = {
        dtype: cpu_prepack_supported(dtype)
        for dtype in {m.weight.dtype for _, m in candidates}
    }
    available = available_cpu_memory()
    # Keep a margin; the allocator and oneDNN may require reorder workspace.
    budget = max(0, available - 1024**3) if available is not None else 0
    if max_prepack_bytes is not None:
        budget = min(budget, max_prepack_bytes)
    for name, module in candidates:
        valid, reason = supported[module.weight.dtype]
        size = module.weight.numel() * module.weight.element_size()
        if not valid:
            report["skipped"][name] = reason
        elif size > budget:
            report["skipped"][name] = "CPU prepack memory budget exhausted or unknown"
        else:
            replacement = CPUPrepackedLinear(module)
            try:
                replacement.prepare_cpu_cache()
            except (NotImplementedError, RuntimeError) as exc:
                report["skipped"][name] = (
                    f"oneDNN preparation failed: {type(exc).__name__}"
                )
                continue
            parent_name, _, attribute = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attribute, replacement)
            report["prepared"] += 1
            report["cache_bytes"] += size
            budget -= size
    return report
