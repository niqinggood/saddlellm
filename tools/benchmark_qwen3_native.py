"""Offline same-weight HF versus Saddle-native Qwen3 model-call benchmark.

Fixed synthetic inputs, greedy decoding, EOS disabled, no prefix reuse. This
includes prefill and decode but not HTTP, tokenization or model quality. Two
models remain resident; this is NOT a low-memory large-model loading recipe.
Numerical/output differences are retained and make the command fail unless
the operator explicitly requests an observational (non-equivalence) run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

import saddlellm
from saddlellm.models.CPULinear import cpu_prepack_status, disable_cpu_prepack
from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.ModelLoader import load_native_causal_lm
from saddlellm.models.NativePrefixCache import NativePrefixCache
from saddlellm.models.NativeQwen3 import qwen3_native_config
from saddlellm.models.SaddleModeling import SaddleForCausalLM


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_fixture(path: Path) -> None:
    """Create a NEW untrained fixture, never overwrite a user's checkpoint."""
    if path.exists():
        raise FileExistsError("fixture output must not already exist")
    path.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(173)
    config = Qwen3Config(
        vocab_size=32768,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3ForCausalLM(config).eval()
    model.save_pretrained(path)
    info = {
        "scope": "Untrained synthetic Qwen3 fixture, NOT Qwen3-8B or a quality model",
        "seed": 173,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (path / "benchmark_fixture.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def check_logits(hf, native, tokens, steps: int, dtype) -> dict:
    """Teacher forcing isolates cached-forward errors after argmax divergence."""
    hf_cache = native_cache = None
    current = tokens
    rows = []
    atol, rtol = (
        (2e-5, 2e-4)
        if dtype == torch.float32
        else (0.001, 0.02)
        if dtype == torch.float16
        else (0.004, 0.04)
    )
    for index in range(steps):
        ref = hf(current, past_key_values=hf_cache, use_cache=True, logits_to_keep=1)
        own = native(
            current, past_key_values=native_cache, use_cache=True, logits_to_keep=1
        )
        ref_values, own_values = ref.logits[:, -1].float(), own.logits[:, -1].float()
        top = ref_values.topk(2)
        ref_ids = ref_values.argmax(-1)
        own_ids = own_values.argmax(-1)
        rows.append(
            {
                "step": index,
                "max_absolute_logit_error": float(
                    (ref_values - own_values).abs().max()
                ),
                "within_tolerance": bool(
                    torch.allclose(own_values, ref_values, atol=atol, rtol=rtol)
                ),
                "reference_top_ids": ref_ids.tolist(),
                "native_top_ids": own_ids.tolist(),
                "reference_top_margin": (top.values[:, 0] - top.values[:, 1]).tolist(),
            }
        )
        hf_cache, native_cache = ref.past_key_values, own.past_key_values
        current = ref_ids[:, None]
    return {
        "atol": atol,
        "rtol": rtol,
        "steps": rows,
        "all_within_tolerance": all(row["within_tolerance"] for row in rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Local full Qwen3 safetensors directory"
    )
    parser.add_argument(
        "--prepare-fixture",
        action="store_true",
        help="Create NEW small untrained fixture at --model first",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument(
        "--attention-backend", choices=["eager", "sdpa"], default="sdpa"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--correctness-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--cpu-prepack",
        action="store_true",
        help="Opt into extra-RAM oneDNN M=1 native CPU weight caches",
    )
    parser.add_argument(
        "--include-native-baseline",
        action="store_true",
        help="With --cpu-prepack, measure the same native model "
        "with caches disabled too",
    )
    parser.add_argument(
        "--observe-output-differences",
        action="store_true",
        help="Retain/report output differences without "
        "treating them as equivalent outputs",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="NEW report path; never overwrite previous evidence",
    )
    args = parser.parse_args(argv)
    if (
        min(
            args.batch_size,
            args.prompt_tokens,
            args.decode_tokens,
            args.correctness_steps,
            args.warmup,
            args.repeats,
            args.cpu_threads,
        )
        < 1
    ):
        parser.error("batch, token, thread, warmup and repeat counts must be positive")
    report_path, model_path = Path(args.output), Path(args.model)
    if report_path.exists():
        parser.error("--output must not already exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.cpu_prepack and args.device != "cpu":
        parser.error("--cpu-prepack requires --device cpu")
    if args.include_native_baseline and not args.cpu_prepack:
        parser.error("--include-native-baseline requires --cpu-prepack")
    torch.set_num_threads(args.cpu_threads)
    if args.prepare_fixture:
        prepare_fixture(model_path)
    if not model_path.is_dir():
        parser.error("--model must be a local checkpoint directory")
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if (
        args.prompt_tokens + max(args.decode_tokens, args.correctness_steps)
        > raw["max_position_embeddings"]
    ):
        parser.error("prompt + decode/correctness steps exceeds the model context")
    dtype = getattr(torch, args.dtype)
    if args.device == "cuda":
        config = qwen3_native_config(raw)
        with torch.device("meta"):
            estimated = sum(p.numel() for p in SaddleForCausalLM(config).parameters())
        required = 2 * estimated * torch.empty((), dtype=dtype).element_size()
        free, _ = torch.cuda.mem_get_info()
        if free < required + 512 * 1024 * 1024:
            parser.error(
                "comparison needs TWO resident models plus workspace; "
                "available CUDA memory is insufficient"
            )
    native = load_native_causal_lm(
        model_path, device=args.device, dtype=dtype, local_files_only=True
    )
    for layer in native.layers:
        layer.config.attention_backend = args.attention_backend
    optimization = optimize_native_inference(
        native, cpu_prepack=args.cpu_prepack
    ).to_dict()
    if args.cpu_prepack and not optimization["cpu_prepack"].get("prepared"):
        parser.error("CPU prepacking did not enable any cache; inspect CPU/RAM support")
    hf = (
        Qwen3ForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=args.attention_backend,
        )
        .to(args.device)
        .eval()
    )
    generator = torch.Generator().manual_seed(173)
    tokens = torch.randint(
        3, raw["vocab_size"], (args.batch_size, args.prompt_tokens), generator=generator
    ).to(args.device)
    # Do not inherit checkpoint sampling/logits processors during a greedy
    # equivalence benchmark. EOS disabled means actual counts are fixed.
    generation = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.decode_tokens,
        eos_token_id=[],
        bos_token_id=native.config.bos_token_id,
        pad_token_id=native.config.pad_token_id,
        use_cache=True,
        repetition_penalty=1.0,
    )
    variants = {
        # Newer HF versions may refill global-default values (including
        # do_sample=False) from checkpoint-specific sampling defaults. Disable
        # that refill explicitly so both branches really use greedy decoding.
        "transformers": lambda: hf.generate(
            tokens, generation_config=generation, use_model_defaults=False
        ),
        "saddle_native": lambda: native.generate(
            tokens,
            do_sample=False,
            max_new_tokens=args.decode_tokens,
            eos_token_id=[],
            repetition_penalty=1.0,
        ),
    }
    if args.include_native_baseline:

        def native_without_prepack():
            with disable_cpu_prepack(native):
                return native.generate(
                    tokens,
                    do_sample=False,
                    max_new_tokens=args.decode_tokens,
                    eos_token_id=[],
                    repetition_penalty=1.0,
                )

        variants["native_without_prepack"] = native_without_prepack
    samples = {name: [] for name in variants}
    outputs = {name: [] for name in variants}
    cuda_extra = {name: [] for name in variants}
    correctness = check_logits(hf, native, tokens, args.correctness_steps, dtype)
    baseline_correctness = None
    if args.include_native_baseline:
        with disable_cpu_prepack(native):
            baseline_correctness = check_logits(
                hf, native, tokens, args.correctness_steps, dtype
            )
    within_tolerance = correctness["all_within_tolerance"] and (
        baseline_correctness is None or baseline_correctness["all_within_tolerance"]
    )
    randomizer = random.Random(173)
    with torch.inference_mode():
        for method in variants.values():
            for _ in range(args.warmup):
                method()
        synchronize(args.device)
        for _ in range(args.repeats):
            order = list(variants)
            randomizer.shuffle(order)
            for name in order:
                if args.device == "cuda":
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                synchronize(args.device)
                start = time.perf_counter()
                result = variants[name]()
                synchronize(args.device)
                samples[name].append((time.perf_counter() - start) * 1000)
                if args.device == "cuda":
                    cuda_extra[name].append(
                        torch.cuda.max_memory_allocated() - baseline
                    )
                assert result.shape == (
                    args.batch_size,
                    args.prompt_tokens + args.decode_tokens,
                )
                outputs[name].append(result[:, args.prompt_tokens :].cpu().tolist())
    fixture_path = model_path / "benchmark_fixture.json"
    fixture = (
        json.loads(fixture_path.read_text(encoding="utf-8"))
        if fixture_path.is_file()
        else None
    )
    same_outputs = all(value == outputs["transformers"] for value in outputs.values())
    generated = args.batch_size * args.decode_tokens
    report = {
        "scope": "Same-weight model-call benchmark; NOT HTTP throughput, "
        "TTFT, quality or an 8B proxy",
        "fixture": fixture,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "saddlellm": saddlellm.__version__,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "conditions": vars(args),
        "generation_policy": {
            "do_sample": False,
            "hf_use_model_defaults": False,
            "eos_token_ids": [],
            "prefix_cache": False,
        },
        "parameters": sum(p.numel() for p in native.parameters()),
        "checkpoint_sha256": {
            path.name: file_sha256(path)
            for path in sorted(model_path.glob("*.safetensors"))
        },
        "metadata_sha256": {
            path.name: file_sha256(path)
            for path in (
                model_path / "config.json",
                model_path / "generation_config.json",
            )
            if path.is_file()
        },
        "implementation_sha256": {
            "CPULinear.py": file_sha256(Path(cpu_prepack_status.__code__.co_filename)),
            "NativePrefixCache.py": file_sha256(
                Path(NativePrefixCache._fingerprint.__code__.co_filename)
            ),
            "ModelLoader.py": file_sha256(
                Path(load_native_causal_lm.__code__.co_filename)
            ),
            "NativeQwen3.py": file_sha256(
                Path(qwen3_native_config.__code__.co_filename)
            ),
            "SaddleModeling.py": file_sha256(
                Path(SaddleForCausalLM.forward.__code__.co_filename)
            ),
            "InferenceOptimizer.py": file_sha256(
                Path(optimize_native_inference.__code__.co_filename)
            ),
            "benchmark_qwen3_native.py": file_sha256(Path(__file__)),
        },
        "prompt_sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
        "optimization": optimization,
        "cpu_prepack": cpu_prepack_status(native),
        "logit_comparison": correctness,
        "native_baseline_logit_comparison": baseline_correctness,
        "all_generated_tokens_equal": same_outputs,
        "equivalence_accepted": same_outputs and within_tolerance,
        "results": {},
    }
    for name in variants:
        median = statistics.median(samples[name])
        report["results"][name] = {
            "samples_ms": samples[name],
            "median_ms": median,
            "generated_tokens_per_call": generated,
            "generated_tokens_per_second": generated * 1000 / median,
            "generated_token_ids": outputs[name],
            "max_extra_cuda_allocated_bytes": max(cuda_extra[name], default=None),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False))
    for name, result in report["results"].items():
        print(
            f"{name}: {result['generated_tokens_per_second']:.2f} generated tokens/s, "
            f"median {result['median_ms']:.2f} ms",
            flush=True,
        )
    print(
        f"Same generated tokens: {same_outputs}; logits within tolerance: "
        f"{within_tolerance}; report: {report_path}",
        flush=True,
    )
    return (
        0
        if within_tolerance and (same_outputs or args.observe_output_differences)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
