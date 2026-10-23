"""Compare a saved prefix-cache implementation on the SAME loaded model.

Times include session creation, prefill and complete fixed-length greedy output.
Priming is excluded; cold publication and hit KV copies are included. A local HF
checkpoint is converted by Saddle's native loader. No network access is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

from saddlellm.models.NativePrefixCache import NativePrefixCache


def load_baseline(path):
    name = "saddlellm.models._benchmark_prefix_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # NativeGeneration validates isinstance against the current public class.
    # Baseline methods remain first in the MRO, including restore/store/state.
    return type("BaselinePrefixCache", (module.NativePrefixCache, NativePrefixCache), {})


def benchmark(args):
    from saddlellm.models.InferenceOptimizer import optimize_native_inference
    from saddlellm.models.ModelLoader import load_native_causal_lm
    from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

    torch.set_num_threads(args.threads)
    torch.manual_seed(173)
    dtype = getattr(torch, args.dtype)
    if args.model:
        from saddlellm.models.TokenizerLoader import load_tokenizer_compatible

        model = load_native_causal_lm(args.model, device=args.device, dtype=dtype,
                                     local_files_only=True).eval()
        tokenizer = load_tokenizer_compatible(args.model, local_files_only=True)
        passage = (Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file
                   else "请阅读以下项目资料，解释大模型训练、评估、部署和推理优化之间的关系。"
                   "系统应当保持接口简单，逐项验证准确率、首字延迟、吞吐和内存占用。\n")
        tokens = tokenizer.encode(passage, add_special_tokens=False)
        if not tokens:
            raise ValueError("prompt text must produce tokens")
        tokens = (tokens * (args.prompt_tokens // len(tokens) + 1))[:args.prompt_tokens]
        prompt_source = args.prompt_file or "repeated Chinese lifecycle project passage"
    else:
        model = SaddleForCausalLM(SaddleModelConfig(
            vocab_size=1024, hidden_size=64, intermediate_size=128,
            num_hidden_layers=args.layers, num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=args.prompt_tokens + args.decode_tokens,
        )).to(device=args.device, dtype=dtype).eval()
        tokens = torch.randint(4, 1024, (args.prompt_tokens,)).tolist()
        prompt_source = "synthetic token fixture; not a trained model"
    if args.prompt_tokens + args.decode_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt + decode exceeds the model context budget")
    # Match serving's BF16 Qwen3 projection policy.
    source = getattr(model, "_saddle_native_source", None) or {}
    packing = optimize_native_inference(model, pack_qkv=not (
        source.get("model_type") == "qwen3" and dtype == torch.bfloat16
    )).to_dict()
    prompt = torch.tensor([tokens], device=args.device)
    old_prompt = prompt.clone()
    old_prompt[:, args.shared_tokens:] = (old_prompt[:, args.shared_tokens:] + 1) % model.config.vocab_size
    classes = {"baseline": load_baseline(args.baseline_source), "current": NativePrefixCache}
    caches = {name: cls(args.cache_mb * 1024 * 1024, reuse_partial_prefixes=True,
                        **({"min_store_tokens": args.min_store_tokens} if name == "current" else {}))
              for name, cls in classes.items()}
    modes = ("uncached", "cold_cache", "full_hit", "partial_hit")
    records = {f"{name}/{mode}": {"prefill_ms": [], "generation_ms": []}
               for name in classes for mode in modes}
    expected_reuse = {"uncached": 0, "cold_cache": 0,
                      "full_hit": args.prompt_tokens, "partial_hit": args.shared_tokens}

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def session(ids, cache):
        return model.create_generation_session(ids, max_new_tokens=args.decode_tokens,
                                               do_sample=False, eos_token_id=[], prefix_cache=cache)

    def prepare(cache, mode):
        cache.clear()
        if mode in {"full_hit", "partial_hit"}:
            with session(prompt if mode == "full_hit" else old_prompt, cache) as prime:
                prime.prefill()
            if cache.snapshot()["entries"] != 1:
                raise RuntimeError("cache budget/model does not permit priming")
        sync()

    with torch.inference_mode():
        with session(prompt, None) as reference:
            list(reference)
            reference_tokens = reference.sequences().clone()
        fingerprints = {name: cls._fingerprint(model, prompt.device) for name, cls in classes.items()}
        if fingerprints["baseline"] is None or fingerprints["baseline"] != fingerprints["current"]:
            raise RuntimeError("baseline/current mutation fingerprints must be identical")
        fingerprint_ms = {}
        for name, cls in classes.items():
            samples = []
            for _ in range(30):
                started = time.perf_counter()
                cls._fingerprint(model, prompt.device)
                samples.append((time.perf_counter() - started) * 1000)
            fingerprint_ms[name] = statistics.median(samples)
        rng = random.Random(173)
        for repeat in range(args.warmup + args.repeats):
            order = list(records)
            rng.shuffle(order)
            for key in order:
                name, mode = key.split("/")
                cache = caches[name]
                prepare(cache, mode)
                started = time.perf_counter()
                with session(prompt, None if mode == "uncached" else cache) as work:
                    work.prefill()
                    sync()
                    prefill_ms = (time.perf_counter() - started) * 1000
                    if work.prefix_cached_tokens != expected_reuse[mode]:
                        raise RuntimeError(f"wrong cache path: {key}")
                    list(work)
                    sync()
                    generation_ms = (time.perf_counter() - started) * 1000
                    equal_tokens = torch.equal(reference_tokens, work.sequences())
                    records[key]["all_output_tokens_equal"] = (
                        records[key].get("all_output_tokens_equal", True) and equal_tokens
                    )
                if repeat >= args.warmup:
                    records[key]["prefill_ms"].append(prefill_ms)
                    records[key]["generation_ms"].append(generation_ms)
                    records[key]["reused_prompt_tokens"] = expected_reuse[mode]
        for record in records.values():
            for metric in ("prefill_ms", "generation_ms"):
                record[f"median_{metric}"] = statistics.median(record[metric])

        def paired(before, after):
            ratios = [first / second for first, second in zip(before, after)]
            return {"n": len(ratios), "after_wins": sum(value > 1 for value in ratios),
                    "median_pair_speedup": statistics.median(ratios), "pair_speedups": ratios}

        paired_speedups = {
            mode: {metric: paired(records[f"baseline/{mode}"][metric], records[f"current/{mode}"][metric])
                   for metric in ("prefill_ms", "generation_ms")}
            for mode in modes
        }
        cache_vs_uncached = {
            mode: {metric: paired(records["current/uncached"][metric], records[f"current/{mode}"][metric])
                   for metric in ("prefill_ms", "generation_ms")}
            for mode in modes if mode != "uncached"
        }
        report = {"scope": "same loaded model; session + prefill + fixed-length greedy decode; not HTTP",
                      "model": args.model or "untrained fixture", "prompt_source": prompt_source,
                      "device": args.device, "dtype": args.dtype, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                      "threads": args.threads, "layers": len(model.layers), "packing": packing,
                      "prompt_tokens": args.prompt_tokens, "shared_tokens": args.shared_tokens,
                      "decode_tokens": args.decode_tokens, "repeats": args.repeats, "warmup": args.warmup,
                      "current_min_store_tokens": args.min_store_tokens,
                      "baseline_source": str(Path(args.baseline_source).resolve()),
                      "baseline_source_sha256": hashlib.sha256(Path(args.baseline_source).read_bytes()).hexdigest(),
                      "current_source_sha256": hashlib.sha256(
                          Path(sys.modules[NativePrefixCache.__module__].__file__).read_bytes()).hexdigest(),
                      "same_model_instance": True, "fingerprint_equal": True,
                      "all_output_tokens_equal": all(row["all_output_tokens_equal"] for row in records.values()),
                      "median_fingerprint_ms": fingerprint_ms, "variants": records,
                      "pairing": "same repeat, randomized variant order; ratio=before/after, >1 favors after",
                      "baseline_vs_current": paired_speedups, "current_cache_vs_uncached": cache_vs_uncached}
    for cache in caches.values():
        cache.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--model")
    parser.add_argument("--prompt-file")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    for name, default in (("prompt-tokens", 1024), ("shared-tokens", 768),
                          ("decode-tokens", 8), ("cache-mb", 256), ("layers", 8),
                          ("threads", 1), ("warmup", 2), ("repeats", 7)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-store-tokens", type=int, default=0,
                        help="Current cache only: defer partial snapshots below this uncached suffix size")
    args = parser.parse_args()
    if (min(args.prompt_tokens, args.decode_tokens, args.cache_mb, args.layers, args.threads, args.repeats) < 1
            or args.warmup < 0 or args.min_store_tokens < 0
            or not 16 <= args.shared_tokens < args.prompt_tokens):
        parser.error("positive budgets/repeats, nonnegative warmup, and 16 <= shared < prompt required")
    report = benchmark(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "fingerprint_ms": report["median_fingerprint_ms"],
                      "all_output_tokens_equal": report["all_output_tokens_equal"],
                      "variants": {key: {name: value for name, value in row.items() if name.startswith("median_")}
                                   for key, row in report["variants"].items()}}, indent=2))
    if not report["all_output_tokens_equal"]:
        raise SystemExit("Output tokens differ; retained timings must not be treated as an equivalent speedup.")


if __name__ == "__main__":
    main()
