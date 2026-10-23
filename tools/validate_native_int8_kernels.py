"""Exercise 8B projection geometries/variable lengths, NOT model throughput."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from saddlellm.models.CUDAInt8 import CUDAInt8Linear, _compiled_linear, _quantize_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    torch.set_num_threads(1)
    torch.manual_seed(37)
    rows = []
    compiled_quant, _ = _compiled_linear()
    for in_features, out_features in (
        (4096, 4096),
        (4096, 1024),
        (4096, 12288),
        (12288, 4096),
        (4096, 6144),
        (4096, 24576),
        (35, 49),
    ):
        source = (
            nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)
            .eval()
            .requires_grad_(False)
        )
        layer = CUDAInt8Linear(source, compile_kernel=True, activation_bits=8).cuda()
        del source
        for count in (128, 1, 7, 129, 53):
            inputs = torch.randn(
                (count, in_features), device="cuda", dtype=torch.bfloat16
            )
            with torch.inference_mode():
                quant_ref, scale_ref = _quantize_tokens(inputs)
                quant_compiled, scale_compiled = compiled_quant(inputs)
                actual = layer(inputs)
                layer.compile_kernel = False
                reference = layer(inputs)
                layer.compile_kernel = True
            quant_differences = int((quant_ref != quant_compiled).sum())
            within = bool(torch.allclose(actual, reference, rtol=0.008, atol=0.008))
            rows.append(
                {
                    "in_features": in_features,
                    "out_features": out_features,
                    "tokens": count,
                    "activation_quantization_differences": quant_differences,
                    "scale_max_error": float((scale_ref - scale_compiled).abs().max()),
                    "output_max_error": float(
                        (actual.float() - reference.float()).abs().max()
                    ),
                    "output_within_bf16_tolerance": within,
                }
            )
            print(
                f"Verified geometry {in_features}x{out_features}, tokens={count}, quant differences={quant_differences}, within={within}",
                flush=True,
            )
        del layer
        torch.cuda.empty_cache()
    report = {
        "scope": "Kernel geometry correctness, NOT 8B inference speed or model quality",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "rows": rows,
        "all_output_checks_passed": all(
            row["output_within_bf16_tolerance"] for row in rows
        ),
    }
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["all_output_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
