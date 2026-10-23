"""Compare bounded SWA KV with full-history geometric KV on IDENTICAL weights.

Uses a random, untrained native hybrid SWA/global model and synthetic tokens.
Reports persistent KV bytes separately from transient peak memory and latency.
No checkpoint is modified, no download occurs, and no model-quality claim is made.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from saddlellm.models.InferenceCache import InferenceLayerCache
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--global-every", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.decode_tokens, args.window, args.layers, args.global_every, args.repeats) < 1:
        parser.error("lengths, layers, period and repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(173)
    model = SaddleForCausalLM(SaddleModelConfig(
        vocab_size=1024, hidden_size=128, intermediate_size=256,
        num_hidden_layers=args.layers, num_attention_heads=8, num_key_value_heads=2,
        max_position_embeddings=args.prompt_tokens + args.decode_tokens,
        attention_backend="sdpa", sliding_window=args.window, eos_token_id=[],
        layer_configs=[{"attention": {"sliding_window": 0 if (index + 1) % args.global_every == 0 else args.window}}
                       for index in range(args.layers)],
    )).to(device=args.device, dtype=getattr(torch, args.dtype)).eval()
    ids = torch.randint(3, 1024, (1, args.prompt_tokens), device=args.device)

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def run(mode):
        with model.create_generation_session(ids, max_new_tokens=args.decode_tokens, do_sample=False) as session:
            if mode == "full_history_geometric":
                # The baseline retains full history but still uses amortized
                # storage: no dynamic torch.cat penalty is charged to it.
                session.past_key_values = [InferenceLayerCache(args.prompt_tokens + args.decode_tokens)
                                           for _ in model.layers]
            session.prefill()
            list(session)
            caches = session.past_key_values
            metadata = {
                "persistent_kv_bytes": sum(value.untyped_storage().nbytes() for cache in caches for value in cache.buffers),
                "logical_lengths": [cache.length for cache in caches],
                "stored_lengths": [cache.stored_length for cache in caches],
                "physical_slots_per_layer": [cache.buffers[0].shape[2] for cache in caches],
            }
            return session.sequences(), metadata

    modes = ("full_history_geometric", "window_ring")
    outputs, metadata, durations, peaks = {}, {}, {mode: [] for mode in modes}, {mode: [] for mode in modes}
    # Warm both paths before timing, including RoPE tables and allocator setup.
    for mode in modes:
        output, metadata[mode] = run(mode)
        outputs[mode] = output.cpu()
        del output
    assert torch.equal(outputs[modes[0]], outputs[modes[1]]), "greedy outputs differ"
    for repeat in range(args.repeats):
        for mode in (modes if repeat % 2 == 0 else tuple(reversed(modes))):
            synchronize()
            base = None
            if args.device == "cuda":
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            output, metadata[mode] = run(mode)
            synchronize()
            durations[mode].append((time.perf_counter() - started) * 1000)
            if base is not None:
                peaks[mode].append(torch.cuda.max_memory_allocated() - base)
            assert torch.equal(output.cpu(), outputs[modes[0]]), "greedy outputs differ"
            del output
    report = {
        "scope": "same untrained hybrid model, B=1, synthetic tokens; session setup/prefill/decode, no HTTP/tokenizer",
        "seed": 173, "torch_version": torch.__version__, "python_version": platform.python_version(),
        "device": args.device, "hardware": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
        "dtype": args.dtype, "prompt_tokens": args.prompt_tokens, "decode_tokens": args.decode_tokens,
        "window": args.window, "layers": args.layers, "global_every": args.global_every, "repeats": args.repeats,
        "greedy_tokens_equal": True,
    }
    for mode in modes:
        report[mode] = {
            **metadata[mode], "wall_ms_samples": durations[mode],
            "median_wall_ms": statistics.median(durations[mode]),
            "peak_extra_cuda_bytes": max(peaks[mode]) if peaks[mode] else None,
        }
    report["persistent_kv_reduction_percent"] = 100 * (1 - metadata[modes[1]]["persistent_kv_bytes"] / metadata[modes[0]]["persistent_kv_bytes"])
    report["median_wall_speedup"] = statistics.median(durations[modes[0]]) / statistics.median(durations[modes[1]])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
