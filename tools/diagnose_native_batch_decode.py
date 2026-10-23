"""Replay a local batch benchmark with identical prefixes to inspect numerical drift.

This is a research-only low-precision override, not a supported runtime mode.
Baseline tokens are teacher-forced into the cohort so later logit differences
cannot be explained merely by different generated histories. No speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from saddlellm import NativeBatchDecodeRunner, load_native_causal_lm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-steps", type=int, nargs="*", default=[],
                        help="1-based generated steps to compare module inputs and outputs")
    parser.add_argument("--rowwise-linear", choices=["none", "head", "hidden", "all"], default="none",
                        help="Diagnosis only: keep B1 Linear execution within the cohort")
    parser.add_argument("--native-linear-path", action="store_true",
                        help="Use the library BF16 compatibility policy instead of the unsafe raw probe")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    settings = report["settings"]
    checkpoint = Path(settings["model"])
    if not checkpoint.is_dir():
        parser.error("recorded model must be a local directory")
    torch.set_num_threads(settings["cpu_threads"])
    model = load_native_causal_lm(
        checkpoint, device=settings["device"], dtype=settings["dtype"], local_files_only=True,
        attention_backend=settings.get("attention_backend"),
    )
    prompt_length, count = settings["prompt_tokens"], settings["tokens"]
    model.prewarm_rope_cache(prompt_length + count, freeze=True)
    if any(step < 1 or step > count for step in args.trace_steps):
        parser.error("trace steps must fit the recorded output budget")
    if args.rowwise_linear != "none":
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if args.rowwise_linear == "head" and name != "lm_head":
                continue
            if args.rowwise_linear == "hidden" and name == "lm_head":
                continue
            original = module.forward

            def rowwise(inputs, _original=original):
                if inputs.ndim == 3 and inputs.shape[0] > 1 and inputs.shape[1] == 1:
                    return torch.cat([_original(row[None]) for row in inputs], dim=0)
                return _original(inputs)

            module.forward = rowwise
    serial = next(row for row in report["records"] if row["variant"] == "serial")
    prompts = [torch.tensor([row[0][:prompt_length]], device=settings["device"]) for row in serial["tokens"]]
    rows, traces = [], []
    with torch.inference_mode(), ExitStack() as stack:
        def create():
            return [stack.enter_context(model.create_generation_session(
                prompt, max_new_tokens=count, do_sample=False, eos_token_id=[], output_scores=True,
            )) for prompt in prompts]

        baseline, cohort = create(), create()
        runner = stack.enter_context(NativeBatchDecodeRunner(model, max_batch_size=len(prompts)))
        runner._batch_dtypes = frozenset({torch.float32, torch.float16, torch.bfloat16})
        runner._bf16_rowwise_linear = args.native_linear_path
        for step in range(count):
            captured, hooks = {}, []
            phase = "baseline"
            if step + 1 in args.trace_steps:
                def record(name):
                    def hook(module, inputs, output):
                        value = output[0] if isinstance(output, tuple) else output
                        if isinstance(value, torch.Tensor) and inputs and isinstance(inputs[0], torch.Tensor):
                            entry = captured.setdefault(name, {"baseline": [], "cohort": []})
                            entry[phase].append((inputs[0].detach().float().cpu().clone(),
                                                 value.detach().float().cpu().clone()))
                    return hook

                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) or type(module).__name__ in {
                        "SaddleRMSNorm", "SaddleAttention", "SaddleSwiGLU", "SaddleDecoderLayer",
                    }:
                        hooks.append(module.register_forward_hook(record(name)))
            try:
                references = [session.step() for session in baseline]
                phase = "cohort"
                actual = runner.step(cohort)
            finally:
                for hook in hooks:
                    hook.remove()
            for name, values in captured.items():
                if not values["baseline"] or not values["cohort"]:
                    continue
                old_inputs, old_outputs = [torch.cat([row[i] for row in values["baseline"]]) for i in (0, 1)]
                new_inputs, new_outputs = [torch.cat([row[i] for row in values["cohort"]]) for i in (0, 1)]
                traces.append({"generated_step": step + 1, "module": name,
                               "input_equal": torch.equal(old_inputs, new_inputs),
                               "output_equal": torch.equal(old_outputs, new_outputs),
                               "input_max_error": float((old_inputs - new_inputs).abs().max()),
                               "output_max_error": float((old_outputs - new_outputs).abs().max())})
            for index, (reference, result, session) in enumerate(zip(references, actual, cohort)):
                old, new = reference.scores.float().cpu(), result.scores.float().cpu()
                top = old.topk(2)
                rows.append({"generated_step": step + 1, "request": index,
                             "baseline_top": int(old.argmax()), "cohort_top": int(new.argmax()),
                             "top_equal": bool(old.argmax() == new.argmax()),
                             "baseline_top_margin": float(top.values[0, 0] - top.values[0, 1]),
                             "max_absolute_logit_error": float((old - new).abs().max()),
                             "mean_absolute_logit_error": float((old - new).abs().mean())})
                # Only greedy/no-penalty/no-stop sessions are used here; replacing
                # this slot aligns the next forward's input without touching KV.
                end = session.buffers.length
                session.buffers.tokens[:, end - 1:end].copy_(reference.token_ids)
    mismatches = [row for row in rows if not row["top_equal"]]
    result = {"scope": "teacher-forced numerical diagnosis, not quality or performance",
              "input_report": str(args.report), "input_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "dtype": settings["dtype"], "attention_backend": model.config.attention_backend,
              "rowwise_linear_probe": args.rowwise_linear,
              "native_linear_path": args.native_linear_path,
              "module_traces": traces,
              "first_top_mismatch": mismatches[0] if mismatches else None,
              "top_mismatch_count": len(mismatches), "steps": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in {"steps", "module_traces"}}, indent=2))
    print(json.dumps({"first_equal_input_different_output": next((row for row in traces
                     if row["input_equal"] and not row["output_equal"]), None)}, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
