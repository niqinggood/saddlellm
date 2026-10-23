import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from saddlellm.models import OperatorBackends as backends


@pytest.mark.parametrize(
    "batch,heads,kv_heads,history,value_dim",
    [
        (1, 6, 2, 1, 8),
        (4, 8, 2, 137, 8),
        (2, 8, 1, 33, 12),
    ],
)
def test_grouped_query_rows_match_repeated_kv_cpu(
    batch, heads, kv_heads, history, value_dim
):
    torch.manual_seed(713)
    q = torch.randn(batch, heads, 1, 8, dtype=torch.float64)
    k = torch.randn(batch, kv_heads, history, 8, dtype=torch.float64)
    v = torch.randn(batch, kv_heads, history, value_dim, dtype=torch.float64)
    # Strided head slices also occur after packed projections.
    q = torch.stack((q, q), dim=-1)[..., 0]
    reference = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(heads // kv_heads, dim=1),
        v.repeat_interleave(heads // kv_heads, dim=1),
    )
    output = backends._folded_gqa_decode(q, k, v)
    assert output.shape == (batch, heads, 1, value_dim)
    torch.testing.assert_close(output, reference, atol=1e-12, rtol=1e-12)


def test_folded_decode_auto_gate_does_not_change_cpu_or_gradients():
    q, k, v = torch.randn(1, 6, 1, 8), torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8)
    with torch.no_grad():
        assert not backends._can_fold_gqa_decode(q, k, v, None, False)


def test_gqa_microbenchmark_runs_offline_and_checks_outputs():
    script = Path(__file__).resolve().parents[1] / "tools/benchmark_gqa_decode.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--heads",
            "6",
            "--kv-heads",
            "2",
            "--head-dim",
            "8",
            "--history",
            "17",
            "--iterations",
            "2",
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    report = json.loads(result.stdout)
    assert report["outputs_close"]
    assert report["expanded_kv"]["median_per_call_ms"] > 0
    assert report["folded_query"]["median_per_call_ms"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 4])
def test_cuda_registry_folds_without_expanding_and_retains_numerics(
    batch, dtype, monkeypatch
):
    torch.manual_seed(719)
    q = torch.randn(batch, 6, 1, 64, device="cuda", dtype=dtype)
    k = torch.randn(batch, 2, 129, 64, device="cuda", dtype=dtype)
    v = torch.randn(batch, 2, 129, 64, device="cuda", dtype=dtype)
    reference = F.scaled_dot_product_attention(
        q, k.repeat_interleave(3, dim=1), v.repeat_interleave(3, dim=1)
    )
    backends.AttentionBackendRegistry.ensure_defaults()
    monkeypatch.setattr(backends, "_can_use_flash_gqa", lambda *args: False)

    def unexpected(*args):
        raise AssertionError("decode must not expand K/V heads")

    monkeypatch.setattr(backends, "_expand_grouped_kv", unexpected)
    with torch.no_grad():
        output = backends.AttentionBackendRegistry.run("sdpa", q, k, v, None, False)
    tolerance = (
        1e-5 if dtype is torch.float32 else 0.003 if dtype is torch.float16 else 0.02
    )
    torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fold_gate_rejects_mask_multi_query_causal_and_gradients():
    q = torch.randn(1, 6, 1, 8, device="cuda")
    k = torch.randn(1, 2, 7, 8, device="cuda")
    v = torch.randn_like(k)
    assert not backends._can_fold_gqa_decode(q, k, v, None, False)
    with torch.no_grad():
        assert backends._can_fold_gqa_decode(q, k, v, None, False)
        assert not backends._can_fold_gqa_decode(q, k, v, None, True)
        assert not backends._can_fold_gqa_decode(
            q.expand(-1, -1, 2, -1), k, v, None, False
        )
        assert not backends._can_fold_gqa_decode(
            q, k, v, torch.zeros(1, 1, 1, 7), False
        )
        assert not backends._can_fold_gqa_decode(q, k[:, :1], v, None, False)
