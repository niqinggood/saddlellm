"""Summarize sequential backend runs without treating quantization as equal precision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(results_dir):
    reports = {}
    for path in sorted(Path(results_dir).glob("*/*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(body, dict) and isinstance(body.get("targets"), dict):
            for label, stats in body["targets"].items():
                reports[(label, body.get("phase", path.stem))] = (body, stats)
    rows = []
    quality = {}
    for (label, phase), (body, stats) in sorted(reports.items()):
        reference = reports.get(("raw", phase))
        same_workload = same_asserted_model_precision = False
        text_mismatches = count_mismatches = speed_ratio = None
        if reference:
            ref_body, ref_stats = reference
            same_workload = all(
                body.get(k) == ref_body.get(k)
                for k in (
                    "prompt_sha256",
                    "max_tokens",
                    "concurrency",
                    "requests_per_round",
                    "repeats",
                )
            )
            same_asserted_model_precision = (
                body.get("protocol") == ref_body.get("protocol") == "openai"
                and body.get("precision_asserted_by_operator")
                == ref_body.get("precision_asserted_by_operator")
                and body.get("checkpoint_sha256_asserted_by_operator") is not None
                and body.get("checkpoint_sha256_asserted_by_operator")
                == ref_body.get("checkpoint_sha256_asserted_by_operator")
            )
            samples, ref_samples = stats["samples"], ref_stats["samples"]
            if same_workload and len(samples) == len(ref_samples):
                text_mismatches = sum(
                    s["text_sha256"] != r["text_sha256"]
                    for s, r in zip(samples, ref_samples)
                )
                count_mismatches = sum(
                    (s["prompt_tokens"], s["completion_tokens"])
                    != (r["prompt_tokens"], r["completion_tokens"])
                    for s, r in zip(samples, ref_samples)
                )
                speed_ratio = (
                    stats["generated_tokens_per_second"]
                    / ref_stats["generated_tokens_per_second"]
                )
        rows.append(
            {
                "backend": label,
                "phase": phase,
                "precision": body.get("precision_asserted_by_operator"),
                "tokens_per_second": stats["generated_tokens_per_second"],
                "p50_response_ms": stats["p50_response_ms"],
                "p95_response_ms": stats["p95_response_ms"],
                "p99_response_ms": stats["p99_response_ms"],
                "p50_ttft_ms": stats.get("p50_ttft_ms"),
                "same_workload_as_raw": same_workload,
                "same_asserted_model_precision_as_raw": same_asserted_model_precision,
                "text_mismatches_to_raw": text_mismatches,
                "token_count_mismatches_to_raw": count_mismatches,
                "observed_throughput_ratio_to_raw": speed_ratio,
                "samples": len(stats["samples"]),
            }
        )
        if phase == "quality":
            quality[label] = [
                {k: s.get(k) for k in ["text", "completion_tokens", "finish_reason"]}
                for s in stats["samples"]
            ]
    return {
        "scope": "Same-load deployment observations; not a hardware-isolated capacity or quality benchmark",
        "warning": "Precision/checkpoint fields are operator assertions. GGUF/quantized ratios are practical deployment differences, not same-precision engine gains. Different EOS accounting can change token counts.",
        "rows": rows,
        "quality_outputs": quality,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(args.results_dir)
    path = Path(args.output)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["rows"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
