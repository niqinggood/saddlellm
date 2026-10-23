"""Synthetic single-query GQA only: repeated KV versus grouped query rows.

Not a whole-model, trained-weight or service benchmark. Uses existing SDPA,
with synchronization at timing boundaries and output-tolerance verification.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from saddlellm.models.OperatorBackends import _folded_gqa_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--history", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    if (
        min(
            args.batch_size,
            args.heads,
            args.kv_heads,
            args.head_dim,
            args.history,
            args.iterations,
            args.repeats,
        )
        < 1
        or args.heads % args.kv_heads
    ):
        parser.error("sizes must be positive; heads must be divisible by kv-heads")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(719)
    dtype = getattr(torch, args.dtype)
    query = torch.randn(
        args.batch_size, args.heads, 1, args.head_dim, device=args.device, dtype=dtype
    )
    key = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.history,
        args.head_dim,
        device=args.device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    groups = args.heads // args.kv_heads
    functions = {
        "expanded_kv": lambda: F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(groups, dim=1),
            value.repeat_interleave(groups, dim=1),
        ),
        "folded_query": lambda: _folded_gqa_decode(query, key, value),
    }

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    samples = {name: [] for name in functions}
    peaks = {name: [] for name in functions}
    report = {
        "scope": "synthetic single-query GQA op, including KV expansion",
        "not_measured": ["model tokens/s", "HTTP latency", "quality"],
        "settings": vars(args),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
    }
    with torch.inference_mode():
        reference = functions["expanded_kv"]()
        candidate = functions["folded_query"]()
        tolerance = 1e-5 if dtype is torch.float32 else 0.02
        report["outputs_close"] = torch.allclose(
            reference, candidate, atol=tolerance, rtol=tolerance
        )
        report["max_abs_difference"] = (
            (reference.float() - candidate.float()).abs().max().item()
        )
        report["tolerance"] = {"atol": tolerance, "rtol": tolerance}
        del reference, candidate
        for fn in functions.values():
            for _ in range(5):
                fn()
        synchronize()
        for index in range(args.repeats):
            names = list(functions)
            if index % 2:
                names.reverse()
            for name in names:
                synchronize()
                base = None
                if args.device == "cuda":
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for _ in range(args.iterations):
                    result = functions[name]()
                synchronize()
                samples[name].append(
                    (time.perf_counter() - started) * 1000 / args.iterations
                )
                peaks[name].append(
                    torch.cuda.max_memory_allocated() - base
                    if base is not None
                    else None
                )
                del result
    for name in functions:
        report[name] = {
            "median_per_call_ms": statistics.median(samples[name]),
            "samples_ms": samples[name],
            "max_extra_cuda_bytes": max(peaks[name])
            if peaks[name][0] is not None
            else None,
        }
    report["speedup"] = (
        report["expanded_kv"]["median_per_call_ms"]
        / report["folded_query"]["median_per_call_ms"]
    )
    text = json.dumps(report, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["outputs_close"]:
        raise SystemExit("Attention parity failed")


if __name__ == "__main__":
    main()
