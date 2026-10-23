"""Same-weight paired native decode A/B: serial sessions vs persistent cohorts.

Local checkpoint, synthetic distinct equal-length prompts, greedy fixed-length
output. Includes prefill, decode, sampling, cache joins and result copies; no
HTTP or tokenizer. Not a model-quality test or other-engine comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from saddlellm.models.ModelLoader import load_native_causal_lm
from saddlellm.models.NativeBatchDecode import NativeBatchDecodeRunner


def summarize(records):
    variants = {key: [row for row in records if row["variant"] == key]
                for key in ("serial", "cohort")}
    result = {}
    for key, rows in variants.items():
        times = sorted(row["seconds"] for row in rows)
        result[key] = {
            "median_seconds": statistics.median(times),
            "p95_seconds": times[math.ceil(0.95 * len(times)) - 1],
            "median_output_tokens_per_second": statistics.median(row["output_tokens"] / row["seconds"] for row in rows),
            "median_first_round_seconds": statistics.median(row["first_round_seconds"] for row in rows),
            "max_peak_allocated_bytes": max((row["peak_allocated_bytes"] for row in rows
                                             if row["peak_allocated_bytes"] is not None), default=None),
        }
    pairs = list(zip(variants["serial"], variants["cohort"]))
    ratios = [old["seconds"] / new["seconds"] for old, new in pairs]
    result["paired_median_speed_ratio"] = statistics.median(ratios)
    result["faster_pairs"] = sum(ratio > 1 for ratio in ratios)
    result["pairs"] = len(pairs)
    result["checks"] = {
        "identical_token_ids": all(old["tokens"] == new["tokens"] for old, new in pairs),
        "batched_decode_exercised": all(row["runner"]["batched_forwards"] > 0 for row in variants["cohort"]),
        "at_least_seven_pairs": len(pairs) >= 7,
        "at_least_three_percent_speedup": statistics.median(ratios) >= 1.03,
        "at_least_six_sevenths_pairs_faster": result["faster_pairs"] * 7 >= len(pairs) * 6,
        "p95_regression_within_five_percent": result["cohort"]["p95_seconds"] <= result["serial"]["p95_seconds"] * 1.05,
        "first_round_regression_within_five_percent": result["cohort"]["median_first_round_seconds"] <= result["serial"]["median_first_round_seconds"] * 1.05,
        "cuda_allocated_growth_within_five_percent": (
            result["cohort"]["max_peak_allocated_bytes"] <= result["serial"]["max_peak_allocated_bytes"] * 1.05
            if result["serial"]["max_peak_allocated_bytes"] is not None else None),
    }
    result["development_gate_passed"] = all(value for value in result["checks"].values() if value is not None)
    result["gate_scope"] = "measured checks only; CPU memory and visible TTFT unmeasured; independent repeat required"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--attention-backend", choices=["eager", "sdpa", "sdpa_reference"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-speedup", action="store_true")
    parser.add_argument("--probe-low-precision", action="store_true",
                        help="Research only: raw BF16/FP16 batched Linear without compatibility protection")
    parser.add_argument("--probe-rowwise-linear", action="store_true",
                        help="Benchmark-only B1 Linear probe; other model operations remain batched")
    args = parser.parse_args()
    if not args.model.is_dir() or args.output.exists():
        parser.error("model must be local and output path must be new")
    if min(args.batch_size, args.prompt_tokens, args.tokens, args.repeats, args.cpu_threads) < 1:
        parser.error("counts must be positive")
    if args.probe_low_precision and args.dtype == "float32":
        parser.error("--probe-low-precision is only for BF16/FP16 numerical investigation")
    torch.set_num_threads(args.cpu_threads)
    model = load_native_causal_lm(str(args.model), device=args.device, dtype=args.dtype,
                                attention_backend=args.attention_backend, local_files_only=True)
    if args.probe_rowwise_linear:
        for module in model.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            original = module.forward

            def rowwise(inputs, _original=original):
                if inputs.ndim == 3 and inputs.shape[0] > 1 and inputs.shape[1] == 1:
                    return torch.cat([_original(row[None]) for row in inputs], dim=0)
                return _original(inputs)

            module.forward = rowwise
    total = args.prompt_tokens + args.tokens
    if total > model.config.max_position_embeddings:
        parser.error("workload exceeds model context limit")
    model.prewarm_rope_cache(total, freeze=True)
    torch.manual_seed(args.seed)
    prompts = torch.randint(3, model.config.vocab_size, (args.batch_size, args.prompt_tokens), device=args.device)
    repository = Path(__file__).resolve().parents[1]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "native model invocation; NOT HTTP, visible TTFT, quality or cross-engine throughput",
        "settings": vars(args) | {"model": str(args.model), "output": str(args.output)},
        "eos_disabled": True, "greedy": True, "prefix_cache": False,
        "resolved_attention_backend": model.config.attention_backend,
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "external_load_isolation": "not guaranteed; existing services are not changed",
        "source_sha256": {path: hashlib.sha256((repository / path).read_bytes()).hexdigest() for path in (
            "saddlellm/models/NativeGeneration.py", "saddlellm/models/NativeBatchDecode.py",
            "saddlellm/models/SaddleModeling.py", "saddlellm/models/OperatorBackends.py",
            "saddlellm/models/InferenceCache.py", "tools/benchmark_native_batch_decode.py")},
        "records": [],
    }
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
        report["free_total_bytes"] = list(torch.cuda.mem_get_info())
        report["cuda_matmul_policy"] = {
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def run(kind):
        sync()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        first_round = None
        with ExitStack() as stack:
            sessions = [stack.enter_context(model.create_generation_session(
                prompt[None, :], max_new_tokens=args.tokens, do_sample=False, eos_token_id=[],
            )) for prompt in prompts]
            runner = stack.enter_context(NativeBatchDecodeRunner(
                model, max_batch_size=args.batch_size, max_batch_tokens=args.batch_size * total,
            )) if kind == "cohort" else None
            if runner is not None and args.probe_low_precision:
                # Research-only override, recorded in settings. The library
                # deliberately falls back for these unvalidated cache dtypes.
                runner._batch_dtypes = frozenset({torch.float32, torch.float16, torch.bfloat16})
                runner._bf16_rowwise_linear = False
            while live := [session for session in sessions if not session.done]:
                if runner is not None:
                    runner.step(live)
                else:
                    for session in live:
                        session.step()
                if first_round is None:
                    sync()
                    first_round = time.perf_counter() - started
            output_tokens = sum(session.generated_tokens for session in sessions)
            tokens = [session.sequences().tolist() for session in sessions]
            stats = runner.snapshot() if runner else None
        sync()
        return {"variant": kind, "seconds": time.perf_counter() - started,
                "first_round_seconds": first_round, "output_tokens": output_tokens,
                "tokens": tokens, "runner": stats,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None}

    for _ in range(2):
        run("serial")
        run("cohort")
    for pair in range(args.repeats):
        for kind in (("serial", "cohort") if pair % 2 == 0 else ("cohort", "serial")):
            report["records"].append({"pair": pair, **run(kind)})
    report["summary"] = summarize(report["records"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["checks"]["identical_token_ids"]:
        return 1
    return 2 if args.require_speedup and not report["summary"]["development_gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
