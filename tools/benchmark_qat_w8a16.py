"""Serial GPU comparison of one portable QAT artifact's execution routes.

This measures cached single-request generation, not HTTP serving or production
quality. The dense comparison materializes the same quantized weights in the
activation dtype; the W8A32 comparison is the unchanged portable reference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
from itertools import pairwise
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saddlellm.compression.QuantizationAwareTraining import (
    QATConfig,
    _module_types,
    _set_module,
    load_qat,
    run_qat,
)
from saddlellm.distillation._model_compat import inputs_requiring_logits
from saddlellm.models.ModelLoader import load_causal_lm
from saddlellm.models.QATW8A16 import qat_cuda_status


def hashes(path):
    return {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
            for item in path.iterdir() if item.is_file()}


def create_tiny_artifact(path, architecture):
    """One real QAT optimization step; its quality is deliberately unclaimed."""
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    train = path.parent / f"{architecture}-train.jsonl"
    train.write_text('\n'.join(json.dumps({"input_ids": row}) for row in
                               [[1, 4, 6, 8, 10, 12, 14, 2], [1, 5, 7, 9, 11, 13, 15, 2]]) + '\n', encoding="utf-8")
    torch.manual_seed(20261019)
    if architecture == "native":
        from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

        model = SaddleForCausalLM(SaddleModelConfig(vocab_size=256, hidden_size=128, intermediate_size=384,
                                                    num_hidden_layers=2, num_attention_heads=4,
                                                    num_key_value_heads=2, max_position_embeddings=256,
                                                    attention_backend="sdpa"))
    else:
        from transformers import GPT2Config, GPT2LMHeadModel

        model = GPT2LMHeadModel(GPT2Config(vocab_size=256, n_positions=256, n_embd=128, n_layer=2,
                                          n_head=4, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                                          eos_token_id=2, pad_token_id=0))

    class Tokenizer:
        pad_token_id, bos_token_id, eos_token_id, unk_token_id = 0, 1, 2, 3

        def get_vocab(self):
            return {str(index): index for index in range(256)}

        def save_pretrained(self, output):
            (Path(output) / "tokenizer_stub.json").write_text("{}", encoding="utf-8")

    run_qat("synthetic-tiny-correctness-fixture", path,
            QATConfig(enabled=True, train_data=str(train), validation_split=0, max_steps=1,
                      batch_size=1, max_seq_length=16, device="cpu"), model=model, tokenizer=Tokenizer())


def dense_same_codes(path, device, dtype):
    """Dense W16A16 comparison with no re-quantization or changed code values."""
    model = load_qat(path)
    _, packed_type = _module_types()
    for name, module in list(model.named_modules()):
        if type(module) is not packed_type:
            continue
        dense = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        with torch.no_grad():
            dense.weight.copy_(module.qweight.float() * module.scales[:, None])
            if module.bias is not None:
                dense.bias.copy_(module.bias)
        _set_module(model, name, dense)
    return model.to(device=device, dtype=dtype).requires_grad_(False).eval()


@torch.inference_mode()
def generate_cached(model, ids, tokens):
    """Greedy cached generation with first-token and inter-token host timing."""
    torch.cuda.synchronize(ids.device)
    start = time.perf_counter()
    arrivals, generated, cache = [], [], None
    current = ids
    for _ in range(tokens):
        inputs = {"input_ids": current, "use_cache": True}
        if cache is not None:
            inputs["past_key_values"] = cache
        output = model(**inputs_requiring_logits(model, inputs))
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(int(next_token.item()))  # Materialize actual token; synchronizes its work.
        arrivals.append(time.perf_counter() - start)
        cache, current = output.past_key_values, next_token
        if cache is None:
            raise RuntimeError("benchmark requires actual model KV cache outputs")
    torch.cuda.synchronize(ids.device)
    elapsed = time.perf_counter() - start
    gaps = [right - left for left, right in pairwise(arrivals)]
    return {"token_ids": generated, "generated_tokens": len(generated),
            "latency_ms": elapsed * 1000, "ttft_ms": arrivals[0] * 1000,
            "decode_tpot_ms": statistics.mean(gaps) * 1000 if gaps else None,
            "tokens_per_second": len(generated) / elapsed}


def compare_logits(model, ids, expected):
    with torch.inference_mode():
        actual = model(**inputs_requiring_logits(model, {"input_ids": ids})).logits.float().cpu()
    delta = actual - expected
    return {"max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt()),
            "next_token_equal": bool(torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1)))}


def benchmark(args):
    if args.repeats < 3 or args.warmup < 1 or args.decode_tokens < 2 or any(value < 1 for value in args.prompt_lengths):
        raise ValueError("use repeats>=3, warmup>=1, decode-tokens>=2, positive prompt lengths")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this benchmark requires explicit CUDA")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("refusing to overwrite existing benchmark output")
    if args.create_tiny:
        artifact = output.parent / "qat-w8a16-artifacts" / args.create_tiny
        create_tiny_artifact(artifact, args.create_tiny)
    else:
        artifact = args.artifact.resolve()
    before = hashes(artifact)
    reference_logits, reference_tokens, rows = {}, {}, []
    for route in ("portable-w8a32-reference", "same-codes-dense-w16a16", "cuda-w8a16"):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        if route == "portable-w8a32-reference":
            model = load_qat(artifact, device=device)
        elif route == "same-codes-dense-w16a16":
            model = dense_same_codes(artifact, device, dtype)
        else:
            model = load_causal_lm(artifact, device=device, dtype=dtype,
                                   local_files_only=True, quantized_backend="cuda-w8a16")
        resident = torch.cuda.memory_allocated(device) - allocated_before
        for length in args.prompt_lengths:
            vocab_size = model.config.vocab_size
            ids = ((torch.arange(length, device=device) * 7 + 4) % vocab_size).reshape(1, -1)
            if route == "portable-w8a32-reference":
                with torch.inference_mode():
                    reference_logits[length] = model(**inputs_requiring_logits(model, {"input_ids": ids})).logits.float().cpu()
            correctness = compare_logits(model, ids, reference_logits[length])
            for _ in range(args.warmup):
                generate_cached(model, ids, args.decode_tokens)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            samples = [generate_cached(model, ids, args.decode_tokens) for _ in range(args.repeats)]
            if route == "portable-w8a32-reference":
                reference_tokens[length] = samples[0]["token_ids"]
            rows.append({"route": route, "prompt_tokens": length, "decode_tokens": args.decode_tokens,
                         "resident_allocated_bytes": resident,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) - allocated_before,
                         "logits_vs_w8a32": correctness,
                         "greedy_tokens_equal_w8a32": all(sample["token_ids"] == reference_tokens[length] for sample in samples),
                         "median_ttft_ms": statistics.median(sample["ttft_ms"] for sample in samples),
                         "median_decode_tpot_ms": statistics.median(sample["decode_tpot_ms"] for sample in samples),
                         "median_tokens_per_second": statistics.median(sample["tokens_per_second"] for sample in samples),
                         "median_latency_ms": statistics.median(sample["latency_ms"] for sample in samples),
                         "samples": samples,
                         "execution": qat_cuda_status(model) if route == "cuda-w8a16" else None})
            del ids
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert hashes(artifact) == before, "benchmark modified source artifact"
    payload = {"schema_version": 1, "artifact": str(artifact), "artifact_sha256": before,
               "source_artifact_unchanged": True,
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                               "platform": platform.platform(), "dtype": args.dtype},
               "command": [sys.executable, *sys.argv], "repeats": args.repeats, "warmup": args.warmup,
               "timing": "serial cached greedy single-request generation; CUDA synchronized; TTFT is first token, not SSE text",
               "quality_scope": "Synthetic tiny fixture if --create-tiny; token/logit regression only, no production or domain quality claim.",
               "comparison": "W8A32 default portable reference vs same-code dequantized W16A16 vs direct INT8-code Triton W8A16. Mixed precision may change logits/tokens.",
               "results": rows}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    print(json.dumps({"output": str(output), "results": [{key: value for key, value in row.items()
                                                          if key not in {"samples", "execution"}} for row in rows]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", type=Path)
    source.add_argument("--create-tiny", choices=["native", "hf"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[16, 128])
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
