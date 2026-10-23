"""Sequential same-prefix HF/native checks without two GPU-resident models."""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM

from saddlellm.models.ModelLoader import load_native_causal_lm
from saddlellm.models.TokenizerLoader import load_tokenizer_compatible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--native-attention", choices=["eager", "sdpa", "sdpa_reference"])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1:
        parser.error("output must be new and steps positive")
    tokenizer = load_tokenizer_compatible(args.model, local_files_only=True)
    prompt = "Write a numbered list of 20 practical tips for reading technical documentation. Keep each tip to one sentence."
    encoded = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    references, forced, rows = [], [], []
    for native in (False, True):
        if native:
            model = load_native_causal_lm(args.model, device="cuda", dtype="bfloat16", local_files_only=True, attention_backend=args.native_attention or args.attention)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attention, local_files_only=True).eval().cuda()
        current, cache = encoded, None
        with torch.inference_mode():
            for index in range(args.steps):
                output = model(current, past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits = output.logits[:, -1].float().cpu()
                cache = output.past_key_values
                if not native:
                    references.append(logits)
                    forced.append(logits.argmax(-1).tolist())
                else:
                    reference = references[index]
                    top = reference.topk(2)
                    rows.append({
                        "step": index, "max_absolute_error": float((reference - logits).abs().max()),
                        "mean_absolute_error": float((reference - logits).abs().mean()),
                        "allclose": torch.allclose(reference, logits, atol=0.004, rtol=0.04),
                        "hf_top": forced[index], "native_top": logits.argmax(-1).tolist(),
                        "hf_top_margin": (top.values[:, 0] - top.values[:, 1]).tolist(),
                    })
                current = torch.tensor(forced[index], device="cuda").unsqueeze(-1)
        del model, cache, output, current
        gc.collect()
        torch.cuda.empty_cache()
    report = {"model": args.model, "attention": args.attention, "dtype": "bfloat16",
              "native_attention": args.native_attention or args.attention,
              "scope": "same-prefix teacher-forced logits; no packing; no speed/quality claim",
              "prompt": prompt, "prompt_ids": encoded.cpu().tolist(), "steps": rows,
              "allclose": all(row["allclose"] for row in rows),
              "all_argmax_equal": all(row["hf_top"] == row["native_top"] for row in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))
    return 0 if report["allclose"] and report["all_argmax_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
