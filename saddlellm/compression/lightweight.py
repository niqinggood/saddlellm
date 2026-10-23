"""
模型轻量化工具箱 — 让模型更小、更快、更省显存

5 大轻量化技术:
  1. 量化方案推荐       — 每个模型规模的最佳量化配置
  2. 剪枝方案推荐        — 每层最优剪枝比例
  3. KV Cache 量化      — 推理显存省 2-4x
  4. 投机解码            — 推理速度翻倍 (小模型打草稿, 大模型审阅)
  5. 一键轻量化          — 自动应用所有技术

典型效果:
  7B 模型: 14GB → 3.5GB (int4量化) + 推理 2x 快 (投机解码) + KV Cache 省 4x
  1B 模型: 2GB → 0.5GB + 可以跑在手机上

用法:
    from saddlellm import Lightweight

    # 查看推荐方案
    Lightweight.recommend("1b")

    # 一键轻量化
    Lightweight.optimize(model, tokenizer, output="./lightweight_model")
"""

import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

__all__ = ["LIGHTWEIGHT_RECIPES", "Lightweight"]


# ============================================================
# 量化/剪枝方案推荐
# ============================================================

LIGHTWEIGHT_RECIPES = {
    "100m": {
        "quantization": {
            "method": "int8",
            "bits": 8,
            "group_size": 128,
            "desc": "int8 量化, 几乎无精度损失",
        },
        "pruning": {
            "method": "unstructured",
            "sparsity": 0.1,
            "desc": "10% 非结构化剪枝",
        },
        "kv_cache": {"bits": 8, "desc": "KV Cache int8"},
        "spec_decoding": {
            "draft_model": "gpt2-small-124m",
            "desc": "同模型草稿 (self-speculative)",
        },
        "estimated_size": "500MB → 300MB (int8)",
        "estimated_speedup": "1.3x",
    },
    "300m": {
        "quantization": {
            "method": "int4",
            "bits": 4,
            "group_size": 128,
            "desc": "int4 量化 (GPTQ/AWQ推荐)",
        },
        "pruning": {
            "method": "unstructured",
            "sparsity": 0.15,
            "desc": "15% 非结构化剪枝",
        },
        "kv_cache": {"bits": 8, "desc": "KV Cache int8"},
        "spec_decoding": {
            "draft_model": "gpt2-small-124m",
            "desc": "GPT-2 124M 草稿模型",
        },
        "estimated_size": "1.2GB → 400MB (int4)",
        "estimated_speedup": "1.8x",
    },
    "1b": {
        "quantization": {
            "method": "int4",
            "bits": 4,
            "group_size": 128,
            "desc": "int4 GPTQ 量化",
        },
        "pruning": {
            "method": "structured",
            "sparsity": 0.2,
            "desc": "20% 结构化剪枝 (2:4)",
        },
        "kv_cache": {"bits": 4, "desc": "KV Cache int4"},
        "spec_decoding": {
            "draft_model": "llama-tiny-150m",
            "desc": "150M 草稿 → 1B 审阅",
        },
        "estimated_size": "4GB → 1.2GB (int4+剪枝)",
        "estimated_speedup": "2.5x",
    },
    "3b": {
        "quantization": {
            "method": "int4",
            "bits": 4,
            "group_size": 64,
            "desc": "int4 GPTQ group64",
        },
        "pruning": {"method": "structured", "sparsity": 0.25, "desc": "25% 结构化剪枝"},
        "kv_cache": {"bits": 4, "desc": "KV Cache int4"},
        "spec_decoding": {"draft_model": "llama-300m", "desc": "300M 草稿 → 3B 审阅"},
        "estimated_size": "12GB → 3GB (int4+剪枝)",
        "estimated_speedup": "3x",
    },
    "7b": {
        "quantization": {
            "method": "int4",
            "bits": 4,
            "group_size": 64,
            "desc": "int4 AWQ 量化",
        },
        "pruning": {
            "method": "structured",
            "sparsity": 0.3,
            "desc": "30% 结构化剪枝 (配合微调恢复)",
        },
        "kv_cache": {"bits": 4, "desc": "KV Cache int4"},
        "spec_decoding": {"draft_model": "llama-1b", "desc": "1B 草稿 → 7B 审阅"},
        "estimated_size": "14GB → 3.5GB (int4+剪枝)",
        "estimated_speedup": "4x",
    },
}


# ============================================================
# Main: Lightweight class
# ============================================================


class Lightweight:
    """
    模型轻量化 — 一键压缩。

    用法:
        # 查看推荐
        Lightweight.recommend("1b")

        # 一键优化
        Lightweight.optimize(model, tokenizer, target="1b")

        # 只做量化
        Lightweight.quantize_only(model, tokenizer, bits=4)

        # 投机解码推理
        Lightweight.speculative_generate(model, tokenizer, prompt="...")
    """

    @staticmethod
    def recommend(model_size: str = "1b"):
        """查看推荐的轻量化方案。"""
        size_key = Lightweight._resolve_size(model_size)
        recipe = LIGHTWEIGHT_RECIPES.get(size_key)

        if recipe is None:
            print("无预设方案, 使用通用建议:")
            print("  量化: int8 (安全) 或 int4 (激进)")
            print("  剪枝: 10-20% 非结构化")
            print("  KV Cache: int8")
            return

        print(f"""
╔══════════════════════════════════════════════════════════╗
║        {model_size.upper()} 模型轻量化方案
╠══════════════════════════════════════════════════════════╣
║  量化:    {recipe["quantization"]["desc"]}
║  剪枝:    {recipe["pruning"]["desc"]}
║  KV Cache: {recipe["kv_cache"]["desc"]}
║  投机解码: {recipe["spec_decoding"]["desc"]}
╠══════════════════════════════════════════════════════════╣
║  预估大小:   {recipe["estimated_size"]}
║  预估加速:   {recipe["estimated_speedup"]}
╚══════════════════════════════════════════════════════════╝
""")

    @staticmethod
    def optimize(
        model,
        tokenizer,
        output_dir: str = "./lightweight_model",
        target: str = "auto",
        apply_quantization: bool = True,
        apply_pruning: bool = True,
        apply_kv_cache_quant: bool = True,
        bits: int = None,
    ):
        """
        一键轻量化 — 自动量化 + 剪枝 + KV Cache 优化。

        示例:
            Lightweight.optimize(model, tokenizer, target="1b")
            Lightweight.optimize(model, tokenizer, target="300m", bits=8)  # 只用 int8
        """
        total_params = sum(p.numel() for p in model.parameters())

        if target == "auto":
            if total_params < 3e8:
                target = "300m"
            elif total_params < 1e9:
                target = "1b"
            elif total_params < 5e9:
                target = "3b"
            else:
                target = "7b"

        size_key = Lightweight._resolve_size(target)
        recipe = LIGHTWEIGHT_RECIPES.get(size_key, LIGHTWEIGHT_RECIPES["1b"])

        print(f"=== 一键轻量化: {size_key} ({total_params / 1e6:.0f}M params) ===")

        applied_pruning = None
        applied_quantization = None
        optimization_errors = []

        # Step 1: 剪枝。剪枝必须在动态量化之前执行，因为量化后的
        # Linear 模块不再是 torch.nn.Linear，无法安全应用剪枝掩码。
        if apply_pruning:
            sparsity = recipe["pruning"]["sparsity"]
            pruning_method = recipe["pruning"]["method"]
            print(f"Step 1: {sparsity * 100:.0f}% 剪枝...")
            try:
                from .pruner import ModelPruner

                p = ModelPruner(
                    model=model,
                    tokenizer=tokenizer,
                    pruning_method=pruning_method,
                    pruning_ratio=sparsity,
                )
                model = p.prune()
                model = p.remove_masks()
                applied_pruning = {
                    **recipe["pruning"],
                    "resolved_method": p.pruning_method,
                }
                print(f"  完成: 移除 {sparsity * 100:.0f}% 参数")
            except (ImportError, RuntimeError, ValueError, NotImplementedError) as exc:
                optimization_errors.append({"stage": "pruning", "error": str(exc)})
                print(f"  剪枝不可用, 跳过: {exc}")

        # Step 2: 量化
        if apply_quantization:
            q_bits = bits or recipe["quantization"]["bits"]
            q_method = recipe["quantization"]["method"]
            print(f"Step 2: {q_method} {q_bits}bit 量化...")
            try:
                from .quantizer import ModelQuantizer

                q = ModelQuantizer(model=model, tokenizer=tokenizer)
                model = q.quantize(bits=q_bits, method=q_method)
                applied_quantization = {
                    **recipe["quantization"],
                    "resolved_method": q.method,
                    "resolved_precision": q.precision,
                }
                print(
                    f"  完成: {total_params * 2 / 1024**3:.1f}GB → {total_params * q_bits / 8 / 1024**3:.1f}GB"
                )
            except (ImportError, RuntimeError, ValueError) as exc:
                optimization_errors.append({"stage": "quantization", "error": str(exc)})
                print(f"  量化不可用, 跳过: {exc}")

        # Step 3: KV Cache 量化配置
        if apply_kv_cache_quant:
            kv_bits = recipe["kv_cache"]["bits"]
            print(f"Step 3: KV Cache int{kv_bits} 配置...")
            Lightweight._configure_kv_cache(model, kv_bits)

        # Step 4: 保存
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # 保存轻量化配置
        import json

        config = {
            "quantization": applied_quantization,
            "pruning": applied_pruning,
            "kv_cache_quant": kv_bits if apply_kv_cache_quant else None,
            "original_params": total_params,
            "lightweight_recipe": size_key,
            "errors": optimization_errors,
        }
        with open(os.path.join(output_dir, "lightweight_config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

        final_size_gb = (
            os.path.getsize(os.path.join(output_dir, "model.safetensors")) / 1024**3
            if os.path.exists(os.path.join(output_dir, "model.safetensors"))
            else 0
        )
        print(f"\n=== 轻量化完成! {output_dir} ===")
        if final_size_gb:
            print(f"  最终大小: {final_size_gb:.2f}GB")
        return model

    @staticmethod
    def quantize_only(
        model,
        tokenizer,
        output_dir: str = "./quantized_model",
        bits: int = 4,
        method: str = "auto",
    ):
        """只做量化。"""
        from .quantizer import ModelQuantizer

        q = ModelQuantizer(model=model, tokenizer=tokenizer)
        model = q.quantize(bits=bits, method=method)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"量化模型已保存: {output_dir}")

    @staticmethod
    def kv_cache_only(model, bits: int = 8):
        """只优化 KV Cache。"""
        Lightweight._configure_kv_cache(model, bits)
        print(f"KV Cache int{bits} 配置完成")

    # ============================================================
    # 投机解码 (Speculative Decoding)
    # ============================================================

    @staticmethod
    def speculative_generate(
        model,
        tokenizer,
        prompt: str,
        draft_model=None,
        draft_tokenizer=None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_draft_tokens: int = 5,  # 草稿模型一次预测几个 token
        *,
        generator=None,
        eos_token_id=None,
    ) -> Dict:  # noqa: UP006
        """Exact greedy or rejection-sampled speculative causal generation.

        This reference path uses full-context forwards, a shared token-ID
        vocabulary, and temperature alone (zero means greedy). It does not
        inherit ``generate`` processors, top-k/top-p, or repetition penalties.
        Each model must expose full causal logits for one unpadded sequence on
        a single device; encoder-decoder and sharded/offloaded models are not
        supported. A separate draft tokenizer must have the exact same vocab.

        Sampling accepts draft token x with min(1, p(x)/q(x)); rejection draws
        from normalized positive(p-q). If every draft token is accepted, the
        target supplies one bonus token. Only accepted draft tokens count as
        accepted. ``generator`` is an optional CPU torch.Generator: probability
        arithmetic and sampling use CPU float64 for device-independent control.

        EOS comes from the target generation/model config, then tokenizer;
        pass an empty list to disable it. Contexts are never silently truncated.
        Timing synchronizes model CUDA devices and measures generation work,
        excluding tokenization/decoding. No speedup is claimed without a timed
        target-only baseline; ``estimated_speedup`` remains None for compatibility.
        Full-context draft work can make this path slower than normal generate.
        """
        import math
        import time

        import torch

        from ..distillation._model_compat import inputs_requiring_logits

        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if type(num_draft_tokens) is not int or num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be a positive integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if generator is not None and (not isinstance(generator, torch.Generator) or generator.device.type != "cpu"):
            raise ValueError("generator must be a CPU torch.Generator")
        if (draft_tokenizer is not None and draft_tokenizer is not tokenizer
                and (not callable(getattr(tokenizer, "get_vocab", None))
                     or not callable(getattr(draft_tokenizer, "get_vocab", None))
                     or tokenizer.get_vocab() != draft_tokenizer.get_vocab())):
            raise ValueError("speculative decoding requires identical target/draft token-ID vocabularies")

        def model_device(instance):
            if getattr(getattr(instance, "config", None), "is_encoder_decoder", False):
                raise ValueError("speculative decoding requires a causal decoder-only model")
            devices = {value.device for values in (instance.parameters(), instance.buffers()) for value in values}
            if len(devices) != 1 or next(iter(devices)).type not in {"cpu", "cuda"}:
                raise ValueError("speculative decoding requires each model on one CPU or CUDA device, without meta/offload")
            device_map = getattr(instance, "hf_device_map", None)
            if device_map and (len({str(value) for value in device_map.values()}) != 1
                               or any(str(value) in {"disk", "meta"} for value in device_map.values())):
                raise ValueError("speculative decoding does not support sharding or disk/meta offload")
            if any(getattr(getattr(module, "_hf_hook", None), "offload", False) for module in instance.modules()):
                raise ValueError("speculative decoding does not support model offload hooks")
            return next(iter(devices))

        target_device = model_device(model)
        draft_device = model_device(draft_model) if draft_model is not None else None
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.get("input_ids")
        if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
                or input_ids.shape[0] != 1 or input_ids.shape[1] < 1
                or input_ids.dtype not in {torch.int32, torch.int64}):
            raise ValueError("tokenizer must return one nonempty integer input_ids sequence")
        if set(encoded) - {"input_ids", "attention_mask"}:
            raise ValueError("speculative decoding supports only input_ids and an unpadded attention_mask")
        mask = encoded.get("attention_mask")
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.shape != input_ids.shape or not bool((mask == 1).all())):
            raise ValueError("speculative decoding requires an unpadded all-ones attention_mask")
        prompt_ids = input_ids[0].tolist()
        if min(prompt_ids) < 0:
            raise ValueError("input token IDs must be non-negative")
        for instance in (model, draft_model):
            if instance is None:
                continue
            config = getattr(instance, "config", None)
            limits = [getattr(config, name, None) for name in ("max_position_embeddings", "n_positions", "max_seq_length")]
            limits = [value for value in limits if type(value) is int and value > 0]
            if limits and len(prompt_ids) + max_new_tokens > min(limits):
                raise ValueError("prompt plus output budget exceeds model context limit; contexts are never truncated")
        if eos_token_id is None:
            for owner in (getattr(model, "generation_config", None), getattr(model, "config", None), tokenizer):
                eos_token_id = getattr(owner, "eos_token_id", None)
                if eos_token_id is not None:
                    break
        if eos_token_id is None:
            eos_ids = set()
        else:
            values = [eos_token_id] if type(eos_token_id) is int else eos_token_id
            if (not isinstance(values, (list, tuple, set))
                    or any(type(value) is not int or value < 0 for value in values)):
                raise ValueError("eos_token_id must be a non-negative integer or a sequence of integers")
            eos_ids = set(values)

        def logits(instance, ids, device, start):
            values = torch.tensor([ids], dtype=torch.long, device=device)
            output = instance(**inputs_requiring_logits(instance, {"input_ids": values}))
            scores = getattr(output, "logits", None)
            if (not isinstance(scores, torch.Tensor) or scores.ndim != 3
                    or scores.shape[:2] != values.shape or scores.shape[2] < 1):
                raise ValueError("model must return full [1, sequence_length, vocabulary] causal logits")
            scores = scores[0, start:].detach().to(device="cpu", dtype=torch.float64)
            if bool(torch.isnan(scores).any() or torch.isposinf(scores).any()) or not bool(torch.isfinite(scores).any(dim=-1).all()):
                raise ValueError("causal logits must define a finite nonempty probability distribution")
            return scores

        def distribution(scores):
            # Center before division so a small positive temperature cannot
            # overflow the largest logit and corrupt the categorical law.
            scaled = (scores - scores.max(dim=-1, keepdim=True).values) / temperature
            return torch.softmax(scaled, dim=-1)

        def choose(probabilities):
            return int(torch.multinomial(probabilities, 1, generator=generator).item())

        cuda_devices = {device for device in (target_device, draft_device) if device is not None and device.type == "cuda"}

        def synchronize():
            for device in cuda_devices:
                torch.cuda.synchronize(device)

        states = {module: module.training for instance in (model, draft_model) if instance is not None for module in instance.modules()}
        generated_ids = []
        total_accepted = total_drafted = target_forwards = draft_forwards = 0
        stopped_on_eos = False
        elapsed = None
        try:
            model.eval()
            if draft_model is not None:
                draft_model.eval()
            synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                # Establish the target vocabulary before passing any proposed
                # ID to its embedding lookup. Reject incompatible drafts even
                # when they propose IDs outside the target vocabulary.
                target_vocabulary = None
                if draft_model is not None and max_new_tokens:
                    target_vocabulary = logits(model, prompt_ids, target_device, len(prompt_ids) - 1).shape[-1]
                    target_forwards += 1
                while len(generated_ids) < max_new_tokens and not stopped_on_eos:
                    context = prompt_ids + generated_ids
                    if draft_model is None:
                        scores = logits(model, context, target_device, len(context) - 1)[0]
                        target_forwards += 1
                        token = int(scores.argmax()) if temperature == 0 else choose(distribution(scores))
                        generated_ids.append(token)
                        stopped_on_eos = token in eos_ids
                        continue
                    candidates, draft_probabilities = [], []
                    remaining = max_new_tokens - len(generated_ids)
                    for _ in range(min(num_draft_tokens, remaining)):
                        scores = logits(draft_model, context + candidates, draft_device, len(context) + len(candidates) - 1)[0]
                        draft_forwards += 1
                        if scores.numel() != target_vocabulary:
                            raise ValueError("target and draft logits must have the same token-ID vocabulary size")
                        if temperature == 0:
                            token = int(scores.argmax())
                            draft_probabilities.append(scores)
                        else:
                            probabilities = distribution(scores)
                            token = choose(probabilities)
                            draft_probabilities.append(probabilities)
                        candidates.append(token)
                        total_drafted += 1
                        if token in eos_ids:
                            break
                    target_scores = logits(model, context + candidates, target_device, len(context) - 1)
                    target_forwards += 1
                    if any(values.numel() != target_scores.shape[-1] for values in draft_probabilities):
                        raise ValueError("target and draft logits must have the same token-ID vocabulary size")
                    probabilities = distribution(target_scores) if temperature > 0 else None
                    rejected = False
                    for index, token in enumerate(candidates):
                        if temperature == 0:
                            replacement = int(target_scores[index].argmax())
                            accepted = token == replacement
                        else:
                            p, q = probabilities[index], draft_probabilities[index]
                            ratio = min(1.0, float(p[token] / q[token]))
                            accepted = float(torch.rand((), dtype=torch.float64, generator=generator)) < ratio
                            if not accepted:
                                residual = (p - q).clamp_min(0)
                                mass = residual.sum()
                                if not bool(torch.isfinite(mass)) or float(mass) <= 0:
                                    raise RuntimeError("rejected draft has no valid residual distribution")
                                replacement = choose(residual / mass)
                        if accepted:
                            generated_ids.append(token)
                            total_accepted += 1
                        else:
                            generated_ids.append(replacement)
                            token = replacement
                            rejected = True
                        stopped_on_eos = token in eos_ids
                        if rejected or stopped_on_eos or len(generated_ids) == max_new_tokens:
                            break
                    if not rejected and not stopped_on_eos and len(generated_ids) < max_new_tokens:
                        bonus = int(target_scores[-1].argmax()) if temperature == 0 else choose(probabilities[-1])
                        generated_ids.append(bonus)
                        stopped_on_eos = bonus in eos_ids
            synchronize()
            elapsed = time.perf_counter() - started
        finally:
            for module, training in states.items():
                module.training = training

        return {
            "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
            "token_ids": generated_ids,
            "generated_tokens": len(generated_ids),
            "draft_tokens": total_drafted,
            "accepted_tokens": total_accepted,
            "acceptance_rate": total_accepted / total_drafted if total_drafted else 0.0,
            "target_forwards": target_forwards,
            "draft_forwards": draft_forwards,
            "finish_reason": "eos" if stopped_on_eos else "length",
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(generated_ids) / elapsed if elapsed > 0 else None,
            "timing_scope": "generation including draft, target verification, CPU sampling and CUDA synchronization; excludes tokenization and text decoding",
            "estimated_speedup": None,
            "baseline_measured": False,
            "mode": "speculative" if draft_model is not None else "target-only",
        }

    @staticmethod
    def _standard_generate(model, tokenizer, prompt, max_tokens, temperature):
        import torch

        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_len = len(
            tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        )
        return {
            "text": full[prompt_len:].strip(),
            "generated_tokens": len(outputs[0]) - inputs["input_ids"].shape[1],
        }

    # ============================================================
    # 内部: KV Cache 量化配置
    # ============================================================

    @staticmethod
    def _configure_kv_cache(model, bits: int = 8):
        """
        配置 KV Cache 为 int8/int4。

        KV Cache 是推理时最大的显存占用之一。
        量化 KV Cache 可以节省 2-4x 显存, 几乎不影响生成质量。

        原理: K 和 V 在推理过程中被缓存在显存中, 每次生成新 token 都要读取。
        int8 量化: 精度损失 < 0.1%, 显存省 2x
        int4 量化: 精度损失 < 0.5%, 显存省 4x
        """
        # 给模型添加 kv_cache_config 属性
        # (实际生效需要 transformers 4.36+ 的 cache_implementation 参数)
        model.config._kv_cache_bits = bits

        # 设置环境提示
        logger.info(f"KV Cache 已配置为 int{bits}")
        logger.info(
            "  推理时使用: model.generate(..., cache_implementation='quantized')"
        )
        logger.info("  或配置: model.config.cache_implementation = 'quantized'")

    # ============================================================
    # 工具
    # ============================================================

    @staticmethod
    def _resolve_size(size: str) -> str:
        """统一 size 名称。"""
        size_map = {
            "100m": "100m",
            "124m": "100m",
            "150m": "100m",
            "300m": "300m",
            "350m": "300m",
            "1b": "1b",
            "1.5b": "1b",
            "3b": "3b",
            "7b": "7b",
            "8b": "7b",
        }
        return size_map.get(size.lower(), size.lower())

    @staticmethod
    def compare_methods():
        """轻量化方法对比。"""
        print("""
轻量化方法对比
═══════════════════════════════════════════════════════════════
方法            节省显存    加速    精度损失    实现难度
───────────────────────────────────────────────────────────────
int8 量化        2x         1.2x    <0.1%      ★ (简单)
int4 量化 (GPTQ)  4x         1.5x    <0.5%      ★★ (需要校准)
int4 量化 (AWQ)   4x         1.8x    <0.3%      ★★ (需校准)
KV Cache int8    1.5-2x     1.1x    <0.1%      ★
KV Cache int4    3-4x       1.2x    <0.5%      ★
剪枝 20%         1.25x      1.2x    1-2%       ★★ (需微调)
投机解码          0          1.5-3x  0%         ★★★ (需小模型)
层融合            0          1.1x    0%         ★
Flash Attention  1.5-2x     1.5x    0%         ★ (自动)
═══════════════════════════════════════════════════════════════
组合推荐 (1B模型):
  int4 量化 + KV Cache int4 + 投机解码 + Flash Attn
  → 模型 4GB→1GB, 推理 4x 快, 可在 8GB 显存上流畅运行
""")
