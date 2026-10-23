"""Same-model A/B timing of native single-row control or decode rotary packing.

Both variants use identical weights, prompts, precision and greedy decoding.
Only the selected optimization differs. Includes tokenizer
stream callbacks, prefill and decode, but excludes HTTP and model loading.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import TextStreamer

from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.ModelLoader import load_native_causal_lm
from saddlellm.models.TokenizerLoader import load_tokenizer_compatible


def acceptance_gate(summary, variants, identical, repeats):
    baseline, candidate = (summary[name] for name in variants)
    checks = {
        "identical_outputs": identical,
        "at_least_seven_pairs": repeats >= 7,
        "median_speedup_at_least_three_percent": summary["speed_ratio"] >= 1.03,
        "at_least_six_sevenths_pairs_faster": summary["faster_pairs"] * 7 >= repeats * 6,
        "ttft_regression_within_five_percent": (
            candidate["median_ttft_s"] is not None and baseline["median_ttft_s"] is not None
            and candidate["median_ttft_s"] <= baseline["median_ttft_s"] * 1.05
        ),
        "memory_growth_within_five_percent": candidate["max_peak_allocated_bytes"] <= baseline["max_peak_allocated_bytes"] * 1.05,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "scope": "development screening only; requires independent repeat and broader workload validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimization", choices=["single-row", "rotary"], default="single-row")
    parser.add_argument("--require-speedup", action="store_true", help="Exit 2 if the development performance gate fails")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new path")
    if args.tokens < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("tokens/repeats must be positive; warmup must be non-negative")
    source_hashes = {
        name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
        for name in ("saddlellm/models/NativeGeneration.py", "saddlellm/models/SaddleModeling.py", "tools/benchmark_native_single_row.py")
    }
    model = load_native_causal_lm(args.model, device=args.device, dtype=args.dtype, local_files_only=True)
    packing = optimize_native_inference(model, pack_qkv=args.dtype != "bfloat16").to_dict()
    tokenizer = load_tokenizer_compatible(getattr(model, "_saddle_tokenizer_source", args.model), local_files_only=True)
    prompt = "Write a numbered list of 20 practical tips for reading technical documentation. Keep each tip to one sentence."
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    encoded = tokenizer(formatted, return_tensors="pt").to(args.device)
    records = []
    variants = ("general", "single_row") if args.optimization == "single-row" else ("separate_rotary", "packed_rotary")

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def run(fast):
        first = None
        pieces = []

        class Stream(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                nonlocal first
                if text:
                    if first is None:
                        first = time.perf_counter() - start
                    pieces.append(text)

        streamer = Stream(tokenizer, skip_prompt=True, skip_special_tokens=True)
        if args.optimization == "rotary":
            for layer in model.layers:
                layer.self_attn._pack_decode_rotary = fast
        sync()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with model.create_generation_session(
            **encoded, max_new_tokens=args.tokens, do_sample=False, streamer=streamer,
        ) as session:
            session._single_row_fastpath = fast if args.optimization == "single-row" else False
            while not session.done:
                session.step()
            sync()
            elapsed = time.perf_counter() - start
            ids = session.sequences()[0, encoded["input_ids"].shape[1]:].tolist()
            return {
                "variant": variants[int(fast)],
                "elapsed_s": elapsed, "ttft_s": first,
                "output_tokens": session.generated_tokens,
                "output_token_s": session.generated_tokens / elapsed,
                "finish_reason": session.finish_reason,
                "token_ids": ids,
                "text_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            }

    for _ in range(args.warmup):
        for fast in (False, True):
            run(fast)
    for index in range(args.repeats):
        for fast in ((False, True) if index % 2 == 0 else (True, False)):
            record = run(fast)
            record["round"] = index + 1
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "token_ids"}), flush=True)
    identical = all(
        row["token_ids"] == records[0]["token_ids"]
        and row["text_sha256"] == records[0]["text_sha256"]
        and row["finish_reason"] == records[0]["finish_reason"] for row in records
    )
    summary = {}
    for variant in variants:
        rows = [r for r in records if r["variant"] == variant]
        visible_ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
        summary[variant] = {
            "median_elapsed_s": statistics.median(r["elapsed_s"] for r in rows),
            "median_output_token_s": statistics.median(r["output_token_s"] for r in rows),
            "median_ttft_s": statistics.median(visible_ttfts) if visible_ttfts else None,
            "max_peak_allocated_bytes": max((r["peak_allocated_bytes"] or 0) for r in rows),
        }
    summary["speed_ratio"] = summary[variants[0]]["median_elapsed_s"] / summary[variants[1]]["median_elapsed_s"]
    paired_ratios = []
    for index in range(args.repeats):
        pair = {r["variant"]: r for r in records if r["round"] == index + 1}
        paired_ratios.append(pair[variants[0]]["elapsed_s"] / pair[variants[1]]["elapsed_s"])
    summary["paired_speed_ratios"] = paired_ratios
    summary["faster_pairs"] = sum(ratio > 1 for ratio in paired_ratios)
    gate = acceptance_gate(summary, variants, identical, args.repeats)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "source": getattr(model, "_saddle_native_source", None),
        "optimization": args.optimization,
        "torch_version": torch.__version__, "device": args.device, "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "packing": packing, "prompt": prompt, "prompt_ids": encoded["input_ids"].tolist(),
        "warmup_per_variant": args.warmup, "repeats": args.repeats,
        "identical_outputs": identical, "summary": summary, "records": records,
        "acceptance_gate": gate,
        "scope": "model/session/stream callbacks; excludes HTTP and loading",
        "source_sha256": source_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"identical_outputs": identical, "summary": summary, "acceptance_gate": gate}), flush=True)
    return 1 if not identical else 2 if args.require_speedup and not gate["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
