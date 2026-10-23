"""Compare W8A16 projection tile choices, not end-to-end model throughput.

Run on an otherwise idle CUDA device. Synthetic INT8 weights and BF16 inputs
exercise Qwen3-8B-sized projection matrices without loading model weights.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.testing

from saddlellm.models.Int8WeightKernels import _matmul, _matmul_tile, weight_only_linear

CASES = (
    (8, 4096, 6144),
    (8, 4096, 24576),
    (8, 12288, 4096),  # Control: unchanged tile.
)


def measure_case(rows: int, columns: int, outputs: int, repeats: int) -> dict:
    weights = torch.randint(-127, 128, (columns, outputs), device="cuda", dtype=torch.int8)
    scales = torch.rand(outputs, device="cuda", dtype=torch.float32)
    inputs = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)

    def original():
        result = torch.empty((rows, outputs), device=inputs.device, dtype=inputs.dtype)
        with torch.cuda.device(inputs.device):
            _matmul[(triton.cdiv(rows, 32), triton.cdiv(outputs, 64))](
                inputs, weights, scales, scales, result,
                rows, columns, outputs, outputs, False,
                32, 64, 64, num_warps=4, num_stages=3,
            )
        return result

    def selected():
        return weight_only_linear(inputs, weights, scales, None, outputs)

    expected = original()
    actual = selected()
    torch.cuda.synchronize()
    exact_match = bool(torch.equal(expected, actual))
    timings = {"original": [], "selected": []}
    for index in range(repeats):
        order = (("original", original), ("selected", selected))
        if index % 2 == 0:
            order = order[::-1]
        for name, callback in order:
            timings[name].append(
                triton.testing.do_bench(callback, warmup=30, rep=100)
            )
    original_ms = statistics.median(timings["original"])
    selected_ms = statistics.median(timings["selected"])
    return {
        "rows": rows,
        "input_features": columns,
        "output_features": outputs,
        "selected_tile": _matmul_tile(rows, columns, outputs),
        "exact_output_match": exact_match,
        "original_median_ms": original_ms,
        "selected_median_ms": selected_ms,
        "projection_speedup": original_ms / selected_ms,
        "samples_ms": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA device is required")
    torch.manual_seed(211)
    report = {
        "scope": "Synthetic W8A16 projection calls only; not full 8B or HTTP throughput",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "rows": [measure_case(*case, args.repeats) for case in CASES],
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if all(row["exact_output_match"] for row in report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
