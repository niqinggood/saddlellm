"""Same-checkpoint native/Transformers inference comparison, offline.

Synthetic fixed-length token IDs, greedy decoding, EOS disabled. This measures
model calls, NOT HTTP TTFT, tokenizer cost or natural-language quality. The HF
reference is an in-memory, strictly checked Llama-compatible subset mapping;
it is NOT a general native-model exporter. vLLM/SGLang are not measured here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from saddlellm.models import OperatorBackends
from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.ModelLoader import load_causal_lm
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleSwiGLU


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_hf_reference(native: SaddleForCausalLM):
    """Reject non-equivalent architectures instead of benchmarking other weights."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = native.config
    for layer in native.layers:
        current = layer.config
        if (
            current.attention_kind not in {"gqa", "mha"}
            or current.qk_norm
            or current.sliding_window
            or current.rope_scaling
            or current.residual_topology != "serial"
            or current.residual_learnable
            or current.residual_attention_scale != 1.0
            or current.residual_ffn_scale != 1.0
            or current.residual_dropout != 0.0
            or type(layer.mlp) is not SaddleSwiGLU
            or current.intermediate_size != config.intermediate_size
            or current.num_attention_heads != config.num_attention_heads
            or current.num_key_value_heads != config.num_key_value_heads
            or current.head_dim != config.head_dim
            or current.rope_theta != config.rope_theta
            or current.rms_norm_eps != config.rms_norm_eps
        ):
            raise ValueError(
                "HF reference requires uniform dense Llama-compatible layers: "
                "no MLA/MoE/QK norm/SWA/scaled RoPE/custom residuals"
            )
    hf_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads or config.num_attention_heads,
        head_dim=config.head_dim or config.hidden_size // config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=config.tie_word_embeddings,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        eos_token_id=None,
        bos_token_id=config.bos_token_id,
        pad_token_id=config.pad_token_id,
    )
    hf_config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(hf_config)
    state = {
        key if key.startswith("lm_head.") else f"model.{key}": value
        for key, value in native.state_dict().items()
    }
    model.load_state_dict(state, strict=True)
    return model.eval()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local native checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--hf-reference", action="store_true")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if (
        min(
            args.batch_size,
            args.prompt_tokens,
            args.decode_tokens,
            args.warmup,
            args.repeats,
        )
        < 1
    ):
        parser.error("batch size, token counts, warmup and repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(173)
    dtype = getattr(torch, args.dtype)
    original = load_causal_lm(args.model, device="cpu", local_files_only=True).eval()
    if not isinstance(original, SaddleForCausalLM):
        parser.error("--model must be a native SaddleLLM checkpoint")
    if (
        args.prompt_tokens + args.decode_tokens
        > original.config.max_position_embeddings
    ):
        parser.error("prompt + decode exceeds checkpoint context budget")

    unpacked = copy.deepcopy(original).to(device=args.device, dtype=dtype)
    packed = copy.deepcopy(original).to(device=args.device, dtype=dtype)
    optimization = optimize_native_inference(packed).to_dict()
    variants = {
        "native_dynamic_full_logits": (unpacked, "dynamic", 0, None),
        "native_unpacked": (unpacked, "geometric", 1, None),
        "native_packed_expanded_kv": (packed, "geometric", 1, None),
        "native_packed": (packed, "geometric", 1, None),
    }
    if args.hf_reference:
        hf = build_hf_reference(original).to(device=args.device, dtype=dtype)
        variants["transformers_eager"] = (hf, None, 1, "eager")
        variants["transformers_sdpa"] = (hf, None, 1, "sdpa")
    prompt = torch.randint(
        4,
        original.config.vocab_size,
        (args.batch_size, args.prompt_tokens),
        device=args.device,
    )
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "offline fixed-budget greedy model calls; EOS disabled; synthetic inputs"
        ),
        "excluded": ["HTTP queue/network", "tokenization", "quality evaluation"],
        "external_engines": {"vllm": "not measured", "sglang": "not measured"},
        "model": str(Path(args.model).resolve()),
        "checkpoint_sha256": checkpoint_hash(Path(args.model) / "pytorch_model.bin"),
        "prompt_sha256": hashlib.sha256(prompt.cpu().numpy().tobytes()).hexdigest(),
        "parameters": sum(p.numel() for p in original.parameters()),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "os": platform.platform(),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_order": (
            "seeded shuffled variants each round, synchronization at boundaries"
        ),
        "memory_note": "reference and optimized models are simultaneously resident",
        "variants": {},
    }

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def generate(name, budget):
        model, cache, keep, implementation = variants[name]
        kwargs = {
            "max_new_tokens": budget,
            "do_sample": False,
            "logits_to_keep": keep,
            "use_cache": True,
        }
        if cache is not None:
            kwargs.update(eos_token_id=[], cache_implementation=cache)
        else:
            model.config._attn_implementation = implementation
            kwargs.update(eos_token_id=None, pad_token_id=original.config.pad_token_id)
        # Earlier native paths retain the old decode expansion as a control.
        # No precision, weights or sampling parameters change across modes.
        context = (
            patch.object(OperatorBackends, "_can_fold_gqa_decode", new=lambda *a: False)
            if name
            in {
                "native_dynamic_full_logits",
                "native_unpacked",
                "native_packed_expanded_kv",
            }
            else nullcontext()
        )
        with context:
            return model.generate(prompt, **kwargs)

    timings = {name: [] for name in variants}
    first_token = {name: [] for name in variants}
    peaks = {name: [] for name in variants}
    outputs = {}
    logit_error = {}
    with torch.inference_mode():
        reference_logits = unpacked(prompt, logits_to_keep=1).logits.float()
        for name, (model, _, _, implementation) in variants.items():
            if implementation:
                model.config._attn_implementation = implementation
            candidate_logits = model(prompt, logits_to_keep=1).logits.float()
            logit_error[name] = (candidate_logits - reference_logits).abs().max().item()
            for _ in range(args.warmup):
                generate(name, args.decode_tokens)
                generate(name, 1)
            synchronize()
        del reference_logits, candidate_logits
        rng = random.Random(173)
        for index in range(args.repeats):
            names = list(variants)
            rng.shuffle(names)
            for name in names:
                synchronize()
                base_memory = None
                if args.device == "cuda":
                    base_memory = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                result = generate(name, args.decode_tokens)
                synchronize()
                timings[name].append((time.perf_counter() - started) * 1000)
                peaks[name].append(
                    torch.cuda.max_memory_allocated() - base_memory
                    if base_memory is not None
                    else None
                )
                outputs[name] = result.cpu()
                del result
                synchronize()
                started = time.perf_counter()
                result = generate(name, 1)
                synchronize()
                first_token[name].append((time.perf_counter() - started) * 1000)
                del result
            print(f"comparison round {index + 1}/{args.repeats}", file=sys.stderr)
    reference_output = outputs["native_unpacked"]
    for name in variants:
        median = statistics.median(timings[name])
        report["variants"][name] = {
            "median_generation_ms": median,
            "mean_generation_ms": statistics.mean(timings[name]),
            "p95_generation_ms": _percentile(timings[name], 0.95),
            "samples_ms": timings[name],
            "aggregate_generated_tokens_per_second": (
                args.batch_size * args.decode_tokens * 1000 / median
            ),
            "median_first_token_model_call_ms": statistics.median(first_token[name]),
            "max_extra_cuda_bytes": max(peaks[name])
            if peaks[name][0] is not None
            else None,
            "outputs_equal": torch.equal(outputs[name], reference_output),
            "max_abs_prefill_logit_difference": logit_error[name],
            "speedup_vs_native_unpacked": statistics.median(timings["native_unpacked"])
            / median,
        }
    report["variants"]["native_packed"]["optimization"] = optimization
    with torch.inference_mode(), patch.object(
        OperatorBackends,
        "_folded_gqa_decode",
        wraps=OperatorBackends._folded_gqa_decode,
    ) as probe:
        generate("native_packed", args.decode_tokens)
        synchronize()
        report["variants"]["native_packed"]["folded_decode_attention_calls"] = (
            probe.call_count
        )
    report["variants"]["native_packed"]["speedup_vs_packed_expanded_kv"] = (
        statistics.median(timings["native_packed_expanded_kv"])
        / statistics.median(timings["native_packed"])
    )
    report["all_outputs_equal"] = all(
        item["outputs_equal"] for item in report["variants"].values()
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["all_outputs_equal"]:
        raise SystemExit(
            "Token parity failed: do not present speed alone as an equivalent result."
        )


if __name__ == "__main__":
    main()
