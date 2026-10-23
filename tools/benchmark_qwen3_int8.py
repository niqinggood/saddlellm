"""Sequential same-quantization HF/native CUDA benchmark, one GPU model at a time.

This measures lossy W8A16 (or experimental W8A8), not a lossless BF16 speedup. Both engines
use identical quantization/operator implementations; native may pack QKV and
gate/up. Original floating-point checkpoints remain untouched. The optional
quality probe compares CPU float logits, NOT a general model-quality evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, GenerationConfig, Qwen3ForCausalLM

import saddlellm
from saddlellm.models.CUDAInt8 import (
    cuda_int8_status,
    cuda_int8_supported,
    prepare_cuda_int8,
)
from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.ModelLoader import load_native_causal_lm
from saddlellm.models.NativeQwen3 import qwen3_native_config
from saddlellm.models.SaddleModeling import SaddleForCausalLM

QUALITY_TEXTS = [
    "大型语言模型推理通常分为预填充和逐词解码两个阶段。预填充同时处理输入提示，"
    "解码阶段利用 KV 缓存，避免重复计算已经处理过的词。缓存占用会随上下文长度增长。",
    "若每本书价格为十五元，买四本书需要六十元。付款一百元后，应找回四十元。"
    "计算总价时，可以先乘以数量，再从付款金额中减去总价。",
    "所有哺乳动物都是动物。鲸鱼是哺乳动物，因此鲸鱼是动物。"
    "但是，不能从某个生物是动物，直接推出它一定是哺乳动物。",
    "A language model predicts the next token from the preceding context. "
    "A key-value cache reuses attention states during autoregressive decoding. "
    "Quantization reduces storage, but numerical quality must be checked separately.",
    "The capital of France is Paris. Water freezes at zero degrees Celsius "
    "under standard atmospheric pressure. There are seven days in a week.",
    "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n"
    "# A function receives its arguments and returns the sum.\n",
    "数据库索引能加快查找，但通常也会增加存储空间和写入成本。"
    "事务的原子性表示一组操作要么全部成功，要么全部回滚，不能只完成一部分。",
    "A fair performance comparison fixes the model, device, precision, input "
    "length, output length and concurrency. Warmup and compilation time are "
    "reported separately from steady-state inference latency.",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def teacher_trace(model, tokens, steps, forced_ids=None):
    cache = None
    current = tokens
    logits, ids = [], []
    for step in range(steps):
        output = model(current, past_key_values=cache, use_cache=True, logits_to_keep=1)
        values = output.logits[:, -1].float()
        next_ids = values.argmax(-1)
        logits.append(values.cpu())
        ids.append(next_ids.cpu().tolist())
        cache = output.past_key_values
        current = (
            torch.tensor(forced_ids[step], device=tokens.device)
            if forced_ids is not None
            else next_ids
        )[:, None]
    return logits, ids


def compare_traces(reference, own, reference_ids, own_ids, dtype):
    atol, rtol = (0.001, 0.02) if dtype == torch.float16 else (0.004, 0.04)
    rows = [
        {
            "step": step,
            "max_absolute_logit_error": float((ref - actual).abs().max()),
            "within_tolerance": bool(torch.allclose(actual, ref, atol=atol, rtol=rtol)),
            "reference_top_ids": reference_ids[step],
            "native_top_ids": own_ids[step],
        }
        for step, (ref, actual) in enumerate(zip(reference, own))
    ]
    return {
        "atol": atol,
        "rtol": rtol,
        "steps": rows,
        "all_within_tolerance": all(r["within_tolerance"] for r in rows),
        "all_top_ids_equal": reference_ids == own_ids,
    }


def tensor_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )


def to_cuda_guarded(model, reserve_bytes):
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    required = tensor_bytes(model)
    if required + reserve_bytes > free:
        raise RuntimeError(
            f"refusing GPU allocation: model={required}, reserve={reserve_bytes}, free={free}"
        )
    model.cuda()
    torch.cuda.synchronize()
    remaining, _ = torch.cuda.mem_get_info()
    if remaining < reserve_bytes:
        model.cpu()
        torch.cuda.empty_cache()
        raise RuntimeError("remaining GPU headroom is below the requested reserve")
    return {
        "free_before": free,
        "free_after": remaining,
        "total": total,
        "model_tensor_bytes": required,
        "reserved_headroom_bytes": reserve_bytes,
    }


def offload(model):
    if hasattr(model, "clear_rope_cache"):
        model.clear_rope_cache()
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.inference_mode()
def quality_trace(model, inputs, device):
    result = []
    for tokens in inputs:
        values = model(tokens.to(device), use_cache=False).logits[:, :-1].float()
        labels = tokens[:, 1:].to(device)
        losses = F.cross_entropy(
            values.reshape(-1, values.shape[-1]), labels.reshape(-1), reduction="none"
        )
        result.append(
            {
                "nll": losses.cpu().tolist(),
                "top_ids": values.argmax(-1).cpu().flatten().tolist(),
            }
        )
    return result


def quality_comparison(reference, own):
    tokens = sum(len(row["nll"]) for row in reference)
    float_nll = sum(sum(row["nll"]) for row in reference) / tokens
    int8_nll = sum(sum(row["nll"]) for row in own) / tokens
    matches = sum(
        sum(a == b for a, b in zip(ref["top_ids"], quant["top_ids"]))
        for ref, quant in zip(reference, own)
    )
    return {
        "scope": "Eight self-authored teacher-forced text probes, NOT a standard quality benchmark",
        "reference_device": "cpu",
        "quantized_device": "cuda",
        "tokens": tokens,
        "float_mean_nll": float_nll,
        "int8_mean_nll": int8_nll,
        "float_perplexity": math.exp(float_nll),
        "int8_perplexity": math.exp(int8_nll),
        "perplexity_ratio": math.exp(int8_nll - float_nll),
        "next_token_top1_agreement": matches / tokens,
        "precision_equivalent": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output", required=True, help="NEW evidence path; no overwrite"
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--attention-backend", choices=["eager", "sdpa"], default="eager"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--correctness-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Reverse block order on alternating rounds",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--reserve-gpu-gib", type=float, default=3.0)
    parser.add_argument("--compile-kernel", action="store_true")
    parser.add_argument("--activation-bits", choices=[8, 16], type=int, default=16)
    parser.add_argument("--no-fuse-projections", action="store_true")
    parser.add_argument(
        "--include-native-unfused",
        action="store_true",
        help="Add same-quantization Saddle without projection packing",
    )
    parser.add_argument("--check-float-quality", action="store_true")
    parser.add_argument("--quality-max-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    if (
        min(
            args.batch_size,
            args.prompt_tokens,
            args.decode_tokens,
            args.correctness_steps,
            args.warmup,
            args.repeats,
            args.rounds,
            args.cpu_threads,
            args.quality_max_tokens,
        )
        < 1
    ):
        parser.error("counts must be positive")
    if not math.isfinite(args.reserve_gpu_gib) or args.reserve_gpu_gib < 1:
        parser.error("reserve-gpu-gib must be finite and >= 1")
    if args.compile_kernel and args.activation_bits != 8:
        parser.error("compile-kernel experiment requires activation-bits=8")
    if args.check_float_quality and args.quality_max_tokens < 2:
        parser.error("quality-max-tokens must be >= 2 for next-token loss")
    output_path = Path(args.output)
    if output_path.exists():
        parser.error("output already exists")
    valid, reason = cuda_int8_supported(activation_bits=args.activation_bits)
    if not valid:
        parser.error(reason)
    torch.set_num_threads(args.cpu_threads)
    model_path = Path(args.model)
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config = qwen3_native_config(raw)
    if (
        args.prompt_tokens + max(args.decode_tokens, args.correctness_steps)
        > config.max_position_embeddings
    ):
        parser.error("tokens exceed context limit")
    dtype = getattr(torch, args.dtype)
    print("Loading and quantizing native CPU model", flush=True)
    native = load_native_causal_lm(
        model_path,
        device="cpu",
        dtype=dtype,
        local_files_only=True,
        attention_backend=args.attention_backend,
    )
    parameters = sum(p.numel() for p in native.parameters())
    optimization = (
        {} if args.no_fuse_projections else optimize_native_inference(native).to_dict()
    )
    native_quantization = prepare_cuda_int8(
        native, compile_kernel=args.compile_kernel, activation_bits=args.activation_bits
    )
    print("Loading and quantizing identical HF CPU model", flush=True)
    hf = (
        Qwen3ForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=args.attention_backend,
        )
        .eval()
        .requires_grad_(False)
    )
    hf_quantization = prepare_cuda_int8(
        hf, compile_kernel=args.compile_kernel, activation_bits=args.activation_bits
    )
    models = {"transformers_int8": hf, "saddle_native_int8": native}
    if args.include_native_unfused:
        print("Loading same-quantization unfused Saddle CPU baseline", flush=True)
        unfused = load_native_causal_lm(
            model_path,
            device="cpu",
            dtype=dtype,
            local_files_only=True,
            attention_backend=args.attention_backend,
        )
        prepare_cuda_int8(
            unfused,
            compile_kernel=args.compile_kernel,
            activation_bits=args.activation_bits,
        )
        models["saddle_native_int8_unfused"] = unfused
    tokens_cpu = torch.randint(
        3,
        config.vocab_size,
        (args.batch_size, args.prompt_tokens),
        generator=torch.Generator().manual_seed(173),
    )
    generation = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.decode_tokens,
        eos_token_id=[],
        bos_token_id=config.bos_token_id,
        pad_token_id=config.pad_token_id,
        use_cache=True,
        repetition_penalty=1.0,
    )
    quality_inputs = None
    if args.check_float_quality:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        quality_inputs = [
            tokenizer(text, return_tensors="pt").input_ids[:, : args.quality_max_tokens]
            for text in QUALITY_TEXTS
        ]
    samples = {key: [] for key in models}
    outputs = {key: [] for key in models}
    blocks = []
    trace_ref = trace_ref_ids = comparison = quant_quality = None
    comparisons = {}
    tokens = None
    reserve = int(args.reserve_gpu_gib * 1024**3)
    try:
        for round_index in range(args.rounds):
            order = list(models)
            if round_index % 2:
                order.reverse()
            for name in order:
                model = models[name]
                memory = to_cuda_guarded(model, reserve)
                tokens = tokens_cpu.cuda()
                if name == "transformers_int8":

                    def generate():
                        return model.generate(
                            tokens,
                            generation_config=generation,
                            use_model_defaults=False,
                        )
                else:

                    def generate():
                        return model.generate(
                            tokens,
                            do_sample=False,
                            max_new_tokens=args.decode_tokens,
                            eos_token_id=[],
                            repetition_penalty=1.0,
                        )

                with torch.inference_mode():
                    print(f"Warming {name}, block {round_index + 1}", flush=True)
                    start = time.perf_counter()
                    for _ in range(args.warmup):
                        generate()
                    torch.cuda.synchronize()
                    warmup_ms = (time.perf_counter() - start) * 1000
                    if round_index == 0:
                        trace, trace_ids = teacher_trace(
                            model, tokens, args.correctness_steps, trace_ref_ids
                        )
                        if name == "transformers_int8":
                            trace_ref, trace_ref_ids = trace, trace_ids
                        else:
                            comparisons[name] = compare_traces(
                                trace_ref, trace, trace_ref_ids, trace_ids, dtype
                            )
                            if name == "saddle_native_int8":
                                comparison = comparisons[name]
                        if name == "saddle_native_int8" and quality_inputs is not None:
                            quant_quality = quality_trace(model, quality_inputs, "cuda")
                    block_samples = []
                    for _ in range(args.repeats):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        result = generate()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000
                        if result.shape != (
                            args.batch_size,
                            args.prompt_tokens + args.decode_tokens,
                        ):
                            raise RuntimeError(
                                "generation did not produce the fixed token count"
                            )
                        block_samples.append(elapsed)
                        samples[name].append(elapsed)
                        outputs[name].append(
                            result[:, args.prompt_tokens :].cpu().tolist()
                        )
                        del result
                blocks.append(
                    {
                        "round": round_index,
                        "engine": name,
                        "memory": memory,
                        "warmup_ms": warmup_ms,
                        "samples_ms": block_samples,
                        "quantization_status": cuda_int8_status(model),
                    }
                )
                tokens = None
                offload(model)
                print(
                    f"Completed {name}: {args.batch_size * args.decode_tokens * 1000 / statistics.median(block_samples):.2f} tokens/s",
                    flush=True,
                )
    finally:
        tokens = None
        for model in models.values():
            offload(model)
    quality = None
    if quality_inputs is not None:
        print("Checking small float quality probe on CPU", flush=True)
        reference = (
            Qwen3ForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=dtype,
                attn_implementation=args.attention_backend,
            )
            .eval()
            .requires_grad_(False)
        )
        float_quality = quality_trace(reference, quality_inputs, "cpu")
        quality = quality_comparison(float_quality, quant_quality)
        del reference
    same = all(
        value == outputs["transformers_int8"][0]
        for values in outputs.values()
        for value in values
    )
    equivalent = same and all(
        row["all_within_tolerance"] and row["all_top_ids_equal"]
        for row in comparisons.values()
    )
    generated = args.batch_size * args.decode_tokens
    results = {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "generated_tokens_per_second": generated * 1000 / statistics.median(values),
            "generated_token_ids": outputs[name],
        }
        for name, values in samples.items()
    }
    report = {
        "scope": f"Full-checkpoint, same-W8A{args.activation_bits} model-call comparison; NOT HTTP, lossless BF16 or general quality",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": vars(args),
        "versions": {
            "saddlellm": saddlellm.__version__,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(),
        },
        "original_parameters": parameters,
        "precision_equivalent_to_float": False,
        "checkpoint_sha256": {
            p.name: file_sha256(p) for p in sorted(model_path.glob("*.safetensors"))
        },
        "metadata_sha256": {
            name: file_sha256(model_path / name)
            for name in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            )
            if (model_path / name).is_file()
        },
        "implementation_sha256": {
            name: file_sha256(Path(function.__code__.co_filename))
            for name, function in (
                ("CUDAInt8.py", prepare_cuda_int8),
                ("ModelLoader.py", load_native_causal_lm),
                ("SaddleModeling.py", SaddleForCausalLM.forward),
                ("NativeQwen3.py", qwen3_native_config),
                ("InferenceOptimizer.py", optimize_native_inference),
                ("benchmark_qwen3_int8.py", main),
            )
        },
        "prompt_sha256": hashlib.sha256(tokens_cpu.numpy().tobytes()).hexdigest(),
        "generation_policy": {
            "do_sample": False,
            "hf_use_model_defaults": False,
            "eos_token_ids": [],
            "prefix_cache": False,
        },
        "native_projection_optimization": optimization,
        "native_quantization": native_quantization,
        "hf_quantization": hf_quantization,
        "logit_comparison": comparison,
        "logit_comparisons": comparisons,
        "all_generated_tokens_equal": same,
        "same_quantization_equivalence_accepted": equivalent,
        "float_quality_probe": quality,
        "blocks": blocks,
        "results": results,
        "native_vs_same_int8_hf_speedup": results["saddle_native_int8"][
            "generated_tokens_per_second"
        ]
        / results["transformers_int8"]["generated_tokens_per_second"],
    }
    if args.activation_bits == 16:
        from saddlellm.models.Int8WeightKernels import weight_only_linear

        report["implementation_sha256"]["Int8WeightKernels.py"] = file_sha256(
            Path(weight_only_linear.__code__.co_filename)
        )
    if args.include_native_unfused:
        report["native_vs_unfused_native_speedup"] = (
            results["saddle_native_int8"]["generated_tokens_per_second"]
            / results["saddle_native_int8_unfused"]["generated_tokens_per_second"]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False))
    print(
        f"Equivalence accepted: {equivalent}; speedup: {report['native_vs_same_int8_hf_speedup']:.4f}; report: {output_path}",
        flush=True,
    )
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
