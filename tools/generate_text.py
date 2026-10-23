"""Generate a continuation from a local SaddleLLM or Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# ``python tools/generate_text.py`` otherwise puts only ``tools/`` at the
# front of sys.path and can accidentally import an older site-package build.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from saddlellm.models.ModelLoader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local checkpoint directory or Hugging Face model id")
    parser.add_argument("--prompt", required=True, help="Text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=device,
        dtype=dtype,
        local_files_only=not args.allow_download,
    )
    model.eval()
    torch.manual_seed(args.seed)
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    continuation_ids = output_ids[0, input_ids.shape[1] :].tolist()
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    print(
        json.dumps(
            {
                "model": args.model,
                "device": device,
                "mode": "greedy" if args.greedy else "sample",
                "seed": None if args.greedy else args.seed,
                "prompt": args.prompt,
                "continuation": continuation,
                "text": args.prompt + continuation,
                "new_tokens": len(continuation_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
