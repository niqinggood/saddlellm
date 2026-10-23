"""Offline native inference throughput: individual calls versus one fixed batch.

Uses synthetic token IDs even when --model points to a trained checkpoint.
Measures model generation, not HTTP queuing, tokenization or model quality.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from saddlellm.models.ModelLoader import load_causal_lm
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Optional local native checkpoint (offline)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch_size, args.prompt_tokens, args.decode_tokens, args.repeats) < 1:
        parser.error("batch size, token counts and repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = (
            ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
            if args.device == "cuda"
            else "float32"
        )
    dtype = getattr(torch, dtype_name)
    torch.set_num_threads(1)
    torch.manual_seed(173)
    if args.model:
        model = load_causal_lm(
            args.model, device=args.device, dtype=dtype, local_files_only=True
        )
        if not isinstance(model, SaddleForCausalLM):
            parser.error("--model must be a native SaddleLLM checkpoint")
    else:
        model = SaddleForCausalLM(
            SaddleModelConfig(
                vocab_size=8192,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=args.prompt_tokens + args.decode_tokens,
                attention_backend="sdpa",
            )
        ).to(device=args.device, dtype=dtype)
    model.eval()
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if (
        isinstance(context_limit, int)
        and args.prompt_tokens + args.decode_tokens > context_limit
    ):
        parser.error("prompt + decode tokens exceeds the checkpoint context budget")
    prompt = torch.randint(
        3,
        model.config.vocab_size,
        (args.batch_size, args.prompt_tokens),
        device=args.device,
    )
    report = {
        "scope": "fixed-length model generation, synthetic tokens, no HTTP queue",
        "model": args.model or "random untrained 2-layer model",
        "device": args.device,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "repeats": args.repeats,
    }

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def generate(mode):
        kwargs = dict(
            max_new_tokens=args.decode_tokens, do_sample=False, eos_token_id=[]
        )
        if mode == "individual":
            return torch.cat([model.generate(row[None], **kwargs) for row in prompt])
        return model.generate(prompt, **kwargs)

    outputs = {}
    with torch.inference_mode():
        for mode in ("individual", "batched"):
            generate(mode)
            synchronize()
            base_memory = None
            if args.device == "cuda":
                base_memory = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(args.repeats):
                result = generate(mode)
            synchronize()
            elapsed = (time.perf_counter() - started) / args.repeats
            peak_extra = (
                torch.cuda.max_memory_allocated() - base_memory
                if base_memory is not None
                else None
            )
            outputs[mode] = result.cpu()
            del result
            report[mode] = {
                "workload_wall_ms": elapsed * 1000,
                "aggregate_decode_tokens_per_second": (
                    args.batch_size * args.decode_tokens / elapsed
                ),
                "peak_extra_cuda_bytes": peak_extra,
            }
    # Synchronize each phase separately for diagnosis, not inside the
    # throughput runs above. These exclude session setup and HTTP latency.
    phases = {"prefill_ms": [], "first_sample_ms": [], "decode_step_ms": []}
    session_output = None
    for _ in range(args.repeats):
        with model.create_generation_session(
            prompt,
            max_new_tokens=args.decode_tokens,
            do_sample=False,
            eos_token_id=[],
        ) as session:
            synchronize()
            started = time.perf_counter()
            session.prefill()
            synchronize()
            phases["prefill_ms"].append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            session.step()
            synchronize()
            phases["first_sample_ms"].append((time.perf_counter() - started) * 1000)
            while not session.done:
                started = time.perf_counter()
                session.step()
                synchronize()
                phases["decode_step_ms"].append((time.perf_counter() - started) * 1000)
            session_output = session.sequences().cpu()
    report["session_phases"] = {
        key: sum(values) / len(values) if values else None
        for key, values in phases.items()
    }
    report["session_outputs_equal"] = torch.equal(outputs["batched"], session_output)
    report["outputs_equal"] = torch.equal(outputs["individual"], outputs["batched"])
    report["throughput_ratio"] = (
        report["batched"]["aggregate_decode_tokens_per_second"]
        / report["individual"]["aggregate_decode_tokens_per_second"]
    )
    print(json.dumps(report, indent=2))
    if not report["outputs_equal"] or not report["session_outputs_equal"]:
        raise SystemExit(
            "Token parity failed; inspect numerical differences before deployment."
        )


if __name__ == "__main__":
    main()
