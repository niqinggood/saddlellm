"""Profile warmed native decode; diagnostic overhead is NOT throughput."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.ModelLoader import load_native_causal_lm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new path")
    model = load_native_causal_lm(args.model, device="cuda", dtype="bfloat16", local_files_only=True)
    optimize_native_inference(model, pack_qkv=False)
    prompt = torch.tensor([[42] * 32], device="cuda")
    with torch.inference_mode(), model.create_generation_session(
        prompt, max_new_tokens=24, do_sample=False, eos_token_id=[],
    ) as session:
        for _ in range(8):
            session.step()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(16):
                session.step()
            torch.cuda.synchronize()
    events = profile.key_averages()
    report = {
        "scope": "16 warmed decode steps, fixed synthetic prompt; profiler timing is not a performance benchmark",
        "cpu": events.table(sort_by="self_cpu_time_total", row_limit=35),
        "cuda": events.table(sort_by="self_cuda_time_total", row_limit=35),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(report["cpu"])
    print(report["cuda"])


if __name__ == "__main__":
    main()
