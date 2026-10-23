"""Compatibility helpers for model forwards used by distillation loops."""
from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any


def _forward_declares_argument(model: Any, argument: str) -> bool:
    """Return whether ``model`` (or a common wrapper) declares ``argument``.

    A generic ``**kwargs`` is deliberately not treated as support. Many
    Hugging Face models expose one while still rejecting unknown keywords in a
    nested module. Native Saddle models declare ``return_logits`` explicitly,
    so checking the concrete signature keeps the compatibility opt-in safe.
    """

    pending = [model]
    visited = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        try:
            parameters = inspect.signature(current.forward).parameters
        except (TypeError, ValueError, AttributeError):
            parameters = {}
        if argument in parameters:
            return True
        if not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            # An inner model may support the option, but it cannot safely pass
            # through a wrapper with a closed forward signature.
            continue
        # Cover distributed/compiled wrappers plus adapter wrappers such as
        # PEFT. Traversal remains conservative because every outer layer must
        # expose **kwargs and the innermost model must declare the option.
        pending.extend(
            getattr(current, attribute, None)
            for attribute in (
                "module",
                "_orig_mod",
                "_fsdp_wrapped_module",
                "base_model",
                "model",
            )
        )
    return False


def inputs_requiring_logits(model: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Copy inputs and request logits only when the model explicitly supports it."""

    forwarded = dict(inputs)
    if _forward_declares_argument(model, "return_logits"):
        forwarded["return_logits"] = True
    return forwarded


__all__ = ["inputs_requiring_logits"]
