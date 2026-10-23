"""Paired A/B for a reusable native microbatch stop mask.

Control-only and complete local model invocation are reported separately.
Not HTTP throughput, answer quality, or a comparison against other engines.
Local checkpoints only; never connects to or restarts existing services.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

runtime = importlib.import_module("saddlellm.runtime.InferenceServer")


class LegacyControl(runtime._BatchGenerationControl):
    def __call__(self, input_ids, scores):
        stop = torch.tensor([event.is_set() for event in self.cancel_events],
                            dtype=torch.bool, device=input_ids.device)
        for token_id in self.eos_ids:
            stop |= input_ids[:, -1].eq(token_id)
        if self.stop_checker is not None:
            stop |= self.stop_checker(input_ids)
        newly_finished = stop & ~self.finished
        self.lengths = torch.where(newly_finished,
                                  input_ids.shape[1] - self.prompt_tokens, self.lengths)
        self.finished |= stop
        return self.finished


class ReuseControl(runtime._BatchGenerationControl):
    """Benchmark-only candidate: complete-model speed gate has not passed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_flags = torch.zeros_like(self.finished)
        self._cancelled_rows = [False] * len(self.cancel_events)

    def __call__(self, input_ids, scores):
        stop = self._stop_flags
        for index, event in enumerate(self.cancel_events):
            if not self._cancelled_rows[index] and event.is_set():
                stop[index] = True
                self._cancelled_rows[index] = True
        for token_id in self.eos_ids:
            stop |= input_ids[:, -1].eq(token_id)
        if self.stop_checker is not None:
            stop |= self.stop_checker(input_ids)
        newly_finished = stop & ~self.finished
        self.lengths = torch.where(newly_finished,
                                  input_ids.shape[1] - self.prompt_tokens, self.lengths)
        self.finished |= stop
        return self.finished


@contextmanager
def benchmark_variant(model, kind):
    """Single-threaded benchmark scope only; never patch a running service."""
    original_control = runtime._BatchGenerationControl
    original_session = model.create_generation_session

    def candidate_session(*args, **kwargs):
        # Candidate batch control owns EOS and exact row lengths.
        kwargs["eos_token_id"] = []
        return original_session(*args, **kwargs)

    runtime._BatchGenerationControl = ReuseControl if kind == "reuse" else LegacyControl
    if kind == "reuse":
        model.create_generation_session = candidate_session
    try:
        yield
    finally:
        runtime._BatchGenerationControl = original_control
        model.create_generation_session = original_session


def summarize(records):
    by_kind = {kind: [row for row in records if row["variant"] == kind]
               for kind in ("legacy", "reuse")}
    result = {kind: {"median_ms": statistics.median(row["ms"] for row in rows),
                     "max_ms": max(row["ms"] for row in rows)} for kind, rows in by_kind.items()}
    ratios = [old["ms"] / new["ms"] for old, new in zip(by_kind["legacy"], by_kind["reuse"])]
    result["paired_median_speed_ratio"] = statistics.median(ratios)
    result["faster_pairs"] = sum(ratio > 1 for ratio in ratios)
    result["pairs"] = len(ratios)
    result["identical_outputs"] = all(old["signature"] == new["signature"]
                                      for old, new in zip(by_kind["legacy"], by_kind["reuse"]))
    result["development_speed_gate"] = (
        result["identical_outputs"] and len(ratios) >= 7
        and statistics.median(ratios) >= 1.03
        and result["faster_pairs"] * 7 >= len(ratios) * 6
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Optional real native checkpoint; omitted = control-only")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--control-iterations", type=int, default=1000)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--require-speedup", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new path; preserve previous evidence")
    if min(args.batch_size, args.tokens, args.repeats, args.control_iterations, args.cpu_threads) < 1:
        parser.error("counts must be positive")
    if args.model and not Path(args.model).is_dir():
        parser.error("model must be an existing local checkpoint directory")
    torch.set_num_threads(args.cpu_threads)
    candidate = ReuseControl
    variants = {"legacy": LegacyControl, "reuse": candidate}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "settings": vars(args) | {"output": str(args.output)},
              "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
              "scope": "same-process paired control/model invocation; not HTTP throughput or quality",
              "source_sha256": hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "external_gpu_workload_isolation": "not guaranteed; existing services untouched"}
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
        report["free_total_bytes_before"] = list(torch.cuda.mem_get_info())

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def measure(function, kind):
        sync()
        started = time.perf_counter()
        result = function(kind)
        sync()
        elapsed = (time.perf_counter() - started) * 1000
        signature = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        record = {"variant": kind, "ms": elapsed, "signature": signature}
        if isinstance(result, dict) and "output_tokens" in result:
            record["output_tokens"] = result["output_tokens"]
        return record

    events = [threading.Event() for _ in range(args.batch_size)]
    prepared = [SimpleNamespace(prompt_tokens=2, stop=None) for _ in events]
    tokens = torch.full((args.batch_size, 3), 4, device=args.device)

    @torch.inference_mode()
    def control(kind):
        checker = variants[kind](None, prepared, [2], events, args.device)
        for _ in range(args.control_iterations):
            checker(tokens, None)
        return [checker.finished.tolist(), checker.lengths.tolist()]

    def paired(function):
        for _ in range(2):
            for kind in variants:
                measure(function, kind)
        records = []
        for repeat in range(args.repeats):
            for kind in (list(variants) if repeat % 2 == 0 else list(reversed(variants))):
                records.append({"pair": repeat, **measure(function, kind)})
        return {"summary": summarize(records), "records": records}

    report["control_only"] = paired(control)
    if args.model:
        from saddlellm.models.ModelLoader import load_native_causal_lm
        from saddlellm.models.TokenizerLoader import load_tokenizer_compatible

        model = load_native_causal_lm(args.model, device=args.device, dtype=args.dtype, local_files_only=True)
        tokenizer = load_tokenizer_compatible(args.model, local_files_only=True)
        engine = runtime.LocalGenerationEngine(model, tokenizer, runtime.InferenceServerSettings(
            batch_size=args.batch_size, web_ui=False, max_input_tokens=256,
            max_new_tokens=args.tokens, default_max_new_tokens=args.tokens,
        ))
        prompt = "请介绍人工智能的常见应用，并给出一些具体的例子。"
        requests = [engine.prepare_batch_request(runtime.CompletionRequest(
            prompt=prompt, temperature=0, max_tokens=args.tokens,
        )) for _ in events]
        report["model_policy"] = {"eos_disabled": False, "eos_token_id": model.config.eos_token_id,
                                  "greedy": True, "prefix_cache": False,
                                  "prompt_tokens": requests[0].generation.prompt_tokens,
                                  "loading_excluded": True, "tokenization_excluded": True,
                                  "response_decoding_included": True}

        @torch.inference_mode()
        def generate(kind):
            with benchmark_variant(model, kind):
                result = engine.generate_batch(requests)
                return {"responses": [{"choices": row["choices"], "usage": row["usage"]} for row in result],
                        "output_tokens": sum(row["usage"]["completion_tokens"] for row in result)}

        report["model_invocation"] = paired(generate)
        for kind in variants:
            summary = report["model_invocation"]["summary"][kind]
            summary["output_tokens_per_second"] = statistics.median(
                row["output_tokens"] * 1000 / row["ms"]
                for row in report["model_invocation"]["records"] if row["variant"] == kind)
    if args.device == "cuda":
        report["free_total_bytes_after"] = list(torch.cuda.mem_get_info())
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value["summary"] for key, value in report.items()
                      if key in {"control_only", "model_invocation"}}, indent=2))
    summaries = [report[key]["summary"] for key in ("control_only", "model_invocation") if key in report]
    if not all(row["identical_outputs"] for row in summaries):
        return 1
    if args.require_speedup and not all(row["development_speed_gate"] for row in summaries):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
