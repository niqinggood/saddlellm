"""Scope PEFT optional AWQ probing to models that actually use AWQ."""

# ruff: noqa: N999 -- Matches the public module naming convention.

from contextlib import contextmanager
from threading import RLock

_DISPATCH_LOCK = RLock()


@contextmanager
def unquantized_lora_dispatch(model):
    """Avoid importing an unrelated AutoAWQ install for ordinary LoRA layers.

    PEFT 0.18 dispatches AWQ before standard Linear based on package presence.
    An incompatible optional AutoAWQ install can therefore break dense LoRA.
    Restore PEFT's probe even on failure; quantized models retain normal probing.
    The lock serializes callers of this helper, including nested adapter loads.
    """
    quantized = (
        getattr(model, "quantization_method", None) is not None
        or getattr(model, "is_quantized", False)
        or getattr(getattr(model, "config", None), "quantization_config", None) is not None
    )
    if quantized:
        yield
        return
    import peft.tuners.lora.awq as dispatch

    with _DISPATCH_LOCK:
        previous = dispatch.is_auto_awq_available
        dispatch.is_auto_awq_available = lambda: False
        try:
            yield
        finally:
            dispatch.is_auto_awq_available = previous
