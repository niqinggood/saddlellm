"""Serving cache thresholds keep short prompts outside snapshot work."""

import pytest
import torch
from test_native_prefix_cache import model

from saddlellm.runtime.InferenceServer import (
    InferenceServerSettings,
    LocalGenerationEngine,
)


@pytest.mark.parametrize("value", [-1, True, 1.5, "256", None])
@pytest.mark.parametrize("field", ["prefix_cache_min_tokens", "prefix_cache_min_store_tokens"])
def test_invalid_cache_threshold_is_rejected(value, field):
    with pytest.raises(ValueError, match=field):
        InferenceServerSettings(prefix_cache_mb=1, **{field: value})


@pytest.mark.parametrize("field", ["prefix_cache_min_tokens", "prefix_cache_min_store_tokens"])
def test_threshold_requires_a_cache_budget(field):
    with pytest.raises(ValueError, match="positive prefix_cache_mb"):
        InferenceServerSettings(**{field: 256})


@pytest.mark.parametrize("requested,effective", [(0, 16), (1, 16), (32, 32), (256, 256)])
def test_serving_uses_threshold_and_preserves_the_original_minimum(requested, effective):
    instance = model()
    settings = InferenceServerSettings(prefix_cache_mb=1, prefix_cache_min_tokens=requested,
                                       prefix_cache_min_store_tokens=256)
    engine = LocalGenerationEngine(instance, object(), settings)
    try:
        assert engine.prefix_cache.min_prefix_tokens == effective
        assert engine.prefix_cache.min_store_tokens == 256
        with instance.create_generation_session(
            torch.tensor([[4] * 8]), max_new_tokens=1, do_sample=False,
            prefix_cache=engine.prefix_cache,
        ) as session:
            session.step()
        state = engine.prefix_cache.snapshot()
        assert state["entries"] == 0
        assert state["lookups"] == 0
        assert state["skipped_requests"] == 1
    finally:
        engine.prefix_cache.close()
