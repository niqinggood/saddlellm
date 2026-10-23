"""Native prompt KV cache: cold/full/partial hits, same weights and target tokens.

Synthetic token IDs; model-only prefill and full generation, not HTTP TTFT.
Cache priming/clearing is outside timed sections. Cold timings include publishing
the prompt snapshot. Hits copy KV into private request buffers, not zero-copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch

from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.ModelLoader import load_causal_lm
from saddlellm.models.NativePrefixCache import NativePrefixCache
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def benchmark(args):
    torch.set_num_threads(1)
    torch.manual_seed(173)
    dtype = getattr(torch, args.dtype)
    if args.model:
        model = load_causal_lm(
            args.model, device=args.device, dtype=dtype, local_files_only=True
        ).eval()
        if type(model) is not SaddleForCausalLM:
            raise ValueError("--model must be a native checkpoint")
    else:
        model = (
            SaddleForCausalLM(
                SaddleModelConfig(
                    vocab_size=1024,
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=args.prompt_tokens + args.decode_tokens,
                )
            )
            .to(device=args.device, dtype=dtype)
            .eval()
        )
    if args.prompt_tokens + args.decode_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt + decode exceeds the model context budget")
    packing = optimize_native_inference(model).to_dict()
    cache = NativePrefixCache(args.cache_mb * 1024 * 1024, reuse_partial_prefixes=True)
    cpu_generator = torch.Generator().manual_seed(173)
    prompt = torch.randint(
        4, model.config.vocab_size, (1, args.prompt_tokens), generator=cpu_generator
    ).to(args.device)
    old_prompt = prompt.clone()
    old_prompt[:, args.shared_tokens :] = (
        old_prompt[:, args.shared_tokens :] - 4 + 1
    ) % (model.config.vocab_size - 4) + 4
    modes = ("uncached", "cold_cache", "full_hit", "partial_hit")
    records = {mode: {"prefill_ms": [], "generation_ms": []} for mode in modes}
    expected_reuse = dict(
        uncached=0,
        cold_cache=0,
        full_hit=args.prompt_tokens,
        partial_hit=args.shared_tokens,
    )

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def prepare(mode):
        cache.clear()
        if mode in {"full_hit", "partial_hit"}:
            source = prompt if mode == "full_hit" else old_prompt
            with model.create_generation_session(
                source,
                max_new_tokens=1,
                do_sample=False,
                eos_token_id=[],
                prefix_cache=cache,
            ) as session:
                session.prefill()
            if cache.snapshot()["entries"] != 1:
                raise ValueError("cache budget/model is ineligible for priming")
        synchronize()

    def generate(mode):
        return model.generate(
            prompt,
            max_new_tokens=args.decode_tokens,
            do_sample=False,
            eos_token_id=[],
            prefix_cache=None if mode == "uncached" else cache,
        )

    with torch.inference_mode():
        reference = generate("uncached")
        with model.create_generation_session(prompt, max_new_tokens=1) as session:
            reference_logits = session.prefill().clone()
        for _ in range(args.warmup):
            for mode in modes:
                prepare(mode)
                generate(mode)
        order_rng = random.Random(173)
        for _ in range(args.repeats):
            order = list(modes)
            order_rng.shuffle(order)
            for mode in order:
                row = records[mode]
                prepare(mode)
                with model.create_generation_session(
                    prompt,
                    max_new_tokens=args.decode_tokens,
                    do_sample=False,
                    eos_token_id=[],
                    prefix_cache=None if mode == "uncached" else cache,
                ) as session:
                    synchronize()
                    started = time.perf_counter()
                    logits = session.prefill()
                    synchronize()
                    row["prefill_ms"].append((time.perf_counter() - started) * 1000)
                    if session.prefix_cached_tokens != expected_reuse[mode]:
                        raise RuntimeError(
                            "benchmark did not use the intended cache path"
                        )
                    row["reused_prompt_tokens"] = session.prefix_cached_tokens
                    row["computed_prompt_tokens"] = session.prefill_computed_tokens
                    row["prefill_max_abs_logits_difference"] = float(
                        (logits.float() - reference_logits.float()).abs().max().item()
                    )
                prepare(mode)
                base_memory = None
                if args.device == "cuda":
                    base_memory = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                output = generate(mode)
                synchronize()
                row["generation_ms"].append((time.perf_counter() - started) * 1000)
                if base_memory is not None:
                    extra = max(0, torch.cuda.max_memory_allocated() - base_memory)
                    row["max_extra_cuda_bytes"] = max(
                        row.get("max_extra_cuda_bytes", 0), extra
                    )
                row["all_output_tokens_equal"] = row.get(
                    "all_output_tokens_equal", True
                ) and torch.equal(output, reference)
                row["retained_cache_tensor_bytes"] = cache.snapshot()[
                    "retained_tensor_bytes"
                ]
    for row in records.values():
        row["median_prefill_ms"] = statistics.median(row["prefill_ms"])
        row["median_generation_ms"] = statistics.median(row["generation_ms"])
        row["generated_tokens_per_second"] = (
            args.decode_tokens * 1000 / row["median_generation_ms"]
        )
    weight_hashes = {}
    if args.model:
        paths = sorted(
            path
            for path in Path(args.model).iterdir()
            if path.is_file()
            and path.suffix in {".bin", ".pt", ".safetensors"}
            and path.name != "training_args.bin"
        )
        for path in paths:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            weight_hashes[path.name] = digest.hexdigest()
    report = {
        "scope": "model-only prefill and complete greedy generation; not HTTP TTFT",
        "model": args.model or "random untrained 2-layer model",
        "weight_sha256": weight_hashes,
        "torch": torch.__version__,
        "device": args.device,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": 1,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "shared_tokens": args.shared_tokens,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt.cpu().tolist()).encode()
        ).hexdigest(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cache_max_bytes": cache.max_bytes,
        "partial_reuse_opt_in_for_benchmark": True,
        "packing": packing,
        "priming_included_in_timing": False,
        "cold_snapshot_publication_included_in_timing": True,
        "all_output_tokens_equal": all(
            row["all_output_tokens_equal"] for row in records.values()
        ),
        "variants": records,
    }
    cache.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Optional local native checkpoint, offline")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--prompt-tokens", type=int, default=384)
    parser.add_argument("--shared-tokens", type=int, default=320)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--cache-mb", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    if (
        min(args.prompt_tokens, args.decode_tokens, args.cache_mb, args.repeats) < 1
        or args.warmup < 0
    ):
        parser.error("positive budgets/repeats and non-negative warmup are required")
    if not 16 <= args.shared_tokens < args.prompt_tokens:
        parser.error("shared-tokens must be >=16 and less than prompt-tokens")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    report = benchmark(args)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not report["all_output_tokens_equal"]:
        raise SystemExit(
            "Output tokens differ; do not treat this as an equivalent speedup."
        )


if __name__ == "__main__":
    main()
