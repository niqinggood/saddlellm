"""Paired before/after token-delivery benchmark with identical weights and RNG.

Measures complete native sessions including prefill, sampling, EOS checks and
host token delivery. This is not an HTTP/visible-TTFT or model-quality test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from saddlellm.models.ModelLoader import load_native_causal_lm
from saddlellm.models.NativeBatchDecode import NativeBatchDecodeRunner
from saddlellm.models.NativeGeneration import NativeGenerationSession, _register_batch_streamer_type


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Recorder:
    def __init__(self):
        self.tokens = []
        self.ended = 0

    def put(self, value):
        self.tokens.extend(value.flatten().tolist())

    def end(self):
        self.ended += 1


_register_batch_streamer_type(Recorder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--decode-mode", choices=["compatible", "throughput"], default="throughput")
    parser.add_argument("--mixed-lengths", action="store_true")
    parser.add_argument("--no-streamer", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new path")
    if min(args.batch_size, args.tokens, args.prompt_tokens, args.repeats, args.cpu_threads) < 1:
        parser.error("counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    old_generation = load_module(args.baseline / "NativeGeneration.py", "saddlellm.models._delivery_baseline_generation")
    old_batch = load_module(args.baseline / "NativeBatchDecode.py", "saddlellm.models._delivery_baseline_batch")
    current_step = NativeGenerationSession._step_locked
    current_sample = NativeGenerationSession._sample

    @contextmanager
    def variant(name):
        NativeGenerationSession._step_locked = (
            old_generation.NativeGenerationSession._step_locked if name == "baseline" else current_step
        )
        NativeGenerationSession._sample = (
            old_generation.NativeGenerationSession._sample if name == "baseline" else current_sample
        )
        try:
            yield old_batch.NativeBatchDecodeRunner if name == "baseline" else NativeBatchDecodeRunner
        finally:
            NativeGenerationSession._step_locked = current_step
            NativeGenerationSession._sample = current_sample

    model = load_native_causal_lm(str(args.model), device="cuda", dtype="bfloat16", local_files_only=True)
    lengths = [args.prompt_tokens + (7 * index if args.mixed_lengths else 0) for index in range(args.batch_size)]
    model.prewarm_rope_cache(max(lengths) + args.tokens, freeze=True)
    generator = torch.Generator(device="cuda").manual_seed(8321)
    prompts = [torch.randint(3, model.config.vocab_size, (1, length), device="cuda", generator=generator)
               for length in lengths]

    @torch.inference_mode()
    def run(name):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        first = None
        with variant(name) as runner_class, ExitStack() as stack:
            recorders = [None if args.no_streamer else Recorder() for _ in prompts]
            sessions = [stack.enter_context(model.create_generation_session(
                prompt, max_new_tokens=args.tokens, min_new_tokens=args.tokens,
                eos_token_id=[2], do_sample=not args.greedy, seed=200 + index,
                temperature=0.7, top_p=0.9, streamer=recorder,
            )) for index, (prompt, recorder) in enumerate(zip(prompts, recorders))]
            runner = stack.enter_context(runner_class(model, max_batch_size=max(2, args.batch_size),
                                                       decode_mode=args.decode_mode))
            while live := [session for session in sessions if not session.done]:
                runner.step(live)
                if first is None:
                    torch.cuda.synchronize()
                    first = time.perf_counter() - started
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            tokens = [session.sequences().cpu().tolist() for session in sessions]
            return dict(variant=name, seconds=elapsed, first_round_seconds=first,
                        output_tokens=sum(session.generated_tokens for session in sessions),
                        tokens=tokens, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        streamer_tokens=[item.tokens if item is not None else None for item in recorders],
                        ended=[item.ended if item is not None else None for item in recorders],
                        runner=runner.snapshot())

    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  scope="native session prefill + decode + sampling + EOS + recording host streamer; not HTTP or quality",
                  settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  torch=torch.__version__, python=platform.python_version(), gpu=torch.cuda.get_device_name(),
                  external_load_isolation="GPU tests coordinated within this task; other machine activity not controlled",
                  baseline_contract="saved _sample + _step_locked + NativeBatchDecodeRunner; same loaded model and constructor",
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                      Path("saddlellm/models/NativeGeneration.py"), Path("saddlellm/models/NativeBatchDecode.py"),
                      Path(__file__),
                      args.baseline / "NativeGeneration.py", args.baseline / "NativeBatchDecode.py")}, records=[])
    for _ in range(args.warmups):
        run("baseline")
        run("candidate")
    for index in range(args.repeats):
        pair = {}
        for name in (("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")):
            record = run(name)
            record["pair"] = index
            pair[name] = record
            report["records"].append(record)
        print(json.dumps(dict(pair=index, speed_ratio=pair["baseline"]["seconds"] / pair["candidate"]["seconds"],
                              identical_tokens=pair["baseline"]["tokens"] == pair["candidate"]["tokens"])), flush=True)
    pairs = [{record["variant"]: record for record in report["records"] if record["pair"] == index}
             for index in range(args.repeats)]
    ratios = [pair["baseline"]["seconds"] / pair["candidate"]["seconds"] for pair in pairs]
    report["summary"] = dict(median_speed_ratio=statistics.median(ratios), faster_pairs=sum(ratio > 1 for ratio in ratios),
                             pairs=len(pairs), identical_tokens=all(pair["baseline"]["tokens"] == pair["candidate"]["tokens"] for pair in pairs),
                             identical_delivery=all(pair["baseline"]["streamer_tokens"] == pair["candidate"]["streamer_tokens"]
                                                    and pair["baseline"]["ended"] == pair["candidate"]["ended"] for pair in pairs))
    for name in ("baseline", "candidate"):
        rows = [pair[name] for pair in pairs]
        report["summary"][name] = dict(
            median_seconds=statistics.median(row["seconds"] for row in rows),
            median_tokens_per_second=statistics.median(row["output_tokens"] / row["seconds"] for row in rows),
            median_first_round_seconds=statistics.median(row["first_round_seconds"] for row in rows),
            peak_allocated_bytes=max(row["peak_allocated_bytes"] for row in rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
