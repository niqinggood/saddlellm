"""Backend and parallelism planning for an LLM training factory.

This module captures the design patterns of modern training stacks without
depending on them at import time: plugin-like backend selection, explicit
parallel dimensions, and export helpers for SaddleLLM, Megatron-style args,
and Colossal-style plugin specs.
"""
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrainingBackendSpec:
    name: str
    family: str
    source_inspiration: str
    best_for: List[str]
    capabilities: List[str]
    required_packages: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ParallelismPlan:
    backend: str
    total_gpus: int
    gpu_memory_gb: float
    model_params: int
    seq_length: int
    data_parallel: int
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    context_parallel: int = 1
    expert_parallel: int = 1
    sequence_parallel: bool = False
    micro_batch_size: int = 1
    global_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    estimated_model_state_gb_per_gpu: float = 0.0
    estimated_activation_gb_per_gpu: float = 0.0
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class FactoryBackendPlanner:
    """Plan distributed training backends and parallelism dimensions."""

    BACKENDS: Dict[str, TrainingBackendSpec] = {
        "single": TrainingBackendSpec(
            name="single",
            family="torch",
            source_inspiration="SaddleLLM/Transformers",
            best_for=["debug", "tiny_pretrain", "small_sft"],
            capabilities=["single_gpu", "cpu_fallback"],
            required_packages=["torch", "transformers"],
        ),
        "torch_ddp": TrainingBackendSpec(
            name="torch_ddp",
            family="torch",
            source_inspiration="Colossal-AI TorchDDPPlugin",
            best_for=["small_pretrain", "sft"],
            capabilities=["data_parallel", "low_complexity"],
            required_packages=["torch", "accelerate"],
        ),
        "torch_fsdp": TrainingBackendSpec(
            name="torch_fsdp",
            family="torch",
            source_inspiration="Colossal-AI TorchFSDPPlugin",
            best_for=["sft", "continued_pretrain", "medium_pretrain"],
            capabilities=["parameter_sharding", "optimizer_sharding", "activation_checkpointing"],
            required_packages=["torch", "accelerate"],
        ),
        "deepspeed_zero2": TrainingBackendSpec(
            name="deepspeed_zero2",
            family="deepspeed",
            source_inspiration="DeepSpeed/Colossal-AI LowLevelZeroPlugin",
            best_for=["sft", "preference", "medium_pretrain"],
            capabilities=["optimizer_sharding", "gradient_sharding", "offload"],
            required_packages=["deepspeed", "accelerate"],
        ),
        "deepspeed_zero3": TrainingBackendSpec(
            name="deepspeed_zero3",
            family="deepspeed",
            source_inspiration="DeepSpeed ZeRO-3 / Colossal-AI Gemini-style sharding",
            best_for=["large_sft", "continued_pretrain", "large_pretrain"],
            capabilities=["parameter_sharding", "optimizer_sharding", "offload"],
            required_packages=["deepspeed", "accelerate"],
        ),
        "hybrid_parallel": TrainingBackendSpec(
            name="hybrid_parallel",
            family="hybrid",
            source_inspiration="Megatron-LM TP/PP/CP + Colossal-AI HybridParallelPlugin",
            best_for=["large_pretrain", "long_context_pretrain"],
            capabilities=["tensor_parallel", "pipeline_parallel", "context_parallel", "sequence_parallel"],
            required_packages=["megatron-core or colossalai"],
            notes=["Exported as a plan; SaddleLLM orchestrator currently consumes the data-parallel fallback."],
        ),
        "moe_hybrid_parallel": TrainingBackendSpec(
            name="moe_hybrid_parallel",
            family="hybrid_moe",
            source_inspiration="Megatron-LM expert parallel + Colossal-AI MoeHybridParallelPlugin",
            best_for=["moe_pretrain", "large_sparse_model"],
            capabilities=["expert_parallel", "tensor_parallel", "pipeline_parallel", "sequence_parallel"],
            required_packages=["megatron-core or colossalai"],
            notes=["Use for MoE architecture planning; execution needs a matching MoE runtime."],
        ),
    }

    @classmethod
    def list_backends(cls) -> Dict[str, Dict]:
        return {name: spec.to_dict() for name, spec in cls.BACKENDS.items()}

    @classmethod
    def recommend_parallelism(
        cls,
        model_params: int,
        num_gpus: int = 1,
        gpu_memory_gb: float = 24.0,
        seq_length: int = 2048,
        global_batch_size: int = 64,
        micro_batch_size: Optional[int] = None,
        is_moe: bool = False,
        prefer_backend: str = "auto",
        training_stage: str = "pretrain",
    ) -> ParallelismPlan:
        total_gpus = max(1, int(num_gpus))
        backend = cls._choose_backend(model_params, total_gpus, gpu_memory_gb, is_moe, prefer_backend, training_stage)

        tp = cls._choose_tensor_parallel(model_params, total_gpus, backend)
        remaining = max(1, total_gpus // tp)
        pp = cls._choose_pipeline_parallel(model_params, remaining, backend)
        remaining = max(1, remaining // pp)
        cp = cls._choose_context_parallel(seq_length, remaining, backend)
        remaining = max(1, remaining // cp)
        ep = cls._choose_expert_parallel(is_moe, remaining, backend)
        used = tp * pp * cp * ep
        dp = max(1, total_gpus // used)

        if micro_batch_size is None:
            micro_batch_size = cls._default_micro_batch(model_params, seq_length, gpu_memory_gb)
        grad_accum = max(1, math.ceil(global_batch_size / max(1, micro_batch_size * dp)))

        model_state_gb, activation_gb = cls._estimate_memory(
            model_params=model_params,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            data_parallel=dp,
            tensor_parallel=tp,
            pipeline_parallel=pp,
            expert_parallel=ep,
            context_parallel=cp,
            backend=backend,
        )
        warnings: List[str] = []
        if model_state_gb + activation_gb > gpu_memory_gb * 0.92:
            warnings.append(
                "Estimated memory is close to or above GPU capacity; lower micro batch, enable checkpointing, or use ZeRO/FSDP/offload."
            )
        if backend in {"hybrid_parallel", "moe_hybrid_parallel"}:
            warnings.append("Hybrid plan is exported for Megatron/Colossal-style runtimes; HF Trainer fallback ignores TP/PP/CP/EP.")
        if cp > 1 and seq_length % (cp * 2) != 0:
            warnings.append("Context parallelism usually requires seq_length divisible by 2 * context_parallel.")
        if total_gpus % used != 0:
            warnings.append("Parallel dimensions do not evenly cover all GPUs; extra GPUs will be unused in this plan.")

        notes = [
            f"Backend selected for stage={training_stage}.",
            "Use bf16 and activation checkpointing for pretraining unless hardware says otherwise.",
        ]
        return ParallelismPlan(
            backend=backend,
            total_gpus=total_gpus,
            gpu_memory_gb=gpu_memory_gb,
            model_params=int(model_params),
            seq_length=int(seq_length),
            data_parallel=dp,
            tensor_parallel=tp,
            pipeline_parallel=pp,
            context_parallel=cp,
            expert_parallel=ep,
            sequence_parallel=tp > 1,
            micro_batch_size=int(micro_batch_size),
            global_batch_size=int(global_batch_size),
            gradient_accumulation_steps=grad_accum,
            estimated_model_state_gb_per_gpu=round(model_state_gb, 2),
            estimated_activation_gb_per_gpu=round(activation_gb, 2),
            warnings=warnings,
            notes=notes,
        )

    @classmethod
    def recommend_stage(
        cls,
        stage: str,
        model_params: int,
        num_gpus: int,
        gpu_memory_gb: float,
        seq_length: int = 2048,
        is_moe: bool = False,
    ) -> Dict:
        plan = cls.recommend_parallelism(
            model_params=model_params,
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
            seq_length=seq_length,
            is_moe=is_moe,
            training_stage=stage,
        )
        method = {
            "pretrain": "full-parameter causal LM training",
            "continued_pretrain": "full-parameter domain adaptive pretraining",
            "sft": "LoRA/QLoRA for low cost; FSDP/ZeRO for full SFT",
            "preference": "DPO/ORPO/KTO with reference-model memory budget",
            "rl": "rollout + reward + policy update loop; start with small batch GRPO/PPO",
            "distill": "teacher-generated data plus supervised/student preference training",
        }.get(stage, "custom training stage")
        return {
            "stage": stage,
            "method": method,
            "plan": plan.to_dict(),
            "orchestrator_distributed": cls.to_training_orchestrator_distributed(plan),
            "megatron_args": cls.to_megatron_style_args(plan),
            "colossal_plugin": cls.to_colossal_plugin_spec(plan),
        }

    @staticmethod
    def to_training_orchestrator_distributed(plan: ParallelismPlan) -> Dict:
        if plan.backend in {"hybrid_parallel", "moe_hybrid_parallel"}:
            raise NotImplementedError(
                "Hybrid TP/PP/CP/EP plans require a runtime adapter; "
                "the TrainingOrchestrator data-parallel fallback is intentionally disabled"
            )
        strategy_map = {
            "single": "single",
            "torch_ddp": "ddp",
            "torch_fsdp": "fsdp",
            "deepspeed_zero2": "deepspeed_zero2",
            "deepspeed_zero3": "deepspeed_zero3",
        }
        return {
            "strategy": strategy_map.get(plan.backend, "single"),
            "num_gpus": plan.total_gpus,
            "gradient_accumulation_steps": plan.gradient_accumulation_steps,
            "bf16": True,
            "fp16": False,
        }

    @staticmethod
    def to_megatron_style_args(plan: ParallelismPlan) -> Dict:
        return {
            "tensor_model_parallel_size": plan.tensor_parallel,
            "pipeline_model_parallel_size": plan.pipeline_parallel,
            "context_parallel_size": plan.context_parallel,
            "expert_model_parallel_size": plan.expert_parallel,
            "sequence_parallel": plan.sequence_parallel,
            "micro_batch_size": plan.micro_batch_size,
            "global_batch_size": plan.global_batch_size,
            "seq_length": plan.seq_length,
        }

    @staticmethod
    def to_colossal_plugin_spec(plan: ParallelismPlan) -> Dict:
        if plan.backend == "torch_ddp":
            return {"plugin": "TorchDDPPlugin"}
        if plan.backend == "torch_fsdp":
            return {"plugin": "TorchFSDPPlugin"}
        if plan.backend == "deepspeed_zero2":
            return {"plugin": "LowLevelZeroPlugin", "stage": 2}
        if plan.backend == "deepspeed_zero3":
            return {"plugin": "GeminiPlugin"}
        if plan.backend == "moe_hybrid_parallel":
            return {
                "plugin": "MoeHybridParallelPlugin",
                "tp_size": plan.tensor_parallel,
                "pp_size": plan.pipeline_parallel,
                "ep_size": plan.expert_parallel,
                "enable_sequence_parallelism": plan.sequence_parallel,
            }
        if plan.backend == "hybrid_parallel":
            return {
                "plugin": "HybridParallelPlugin",
                "tp_size": plan.tensor_parallel,
                "pp_size": plan.pipeline_parallel,
                "enable_sequence_parallelism": plan.sequence_parallel,
            }
        return {"plugin": "none"}

    @classmethod
    def save_plan(cls, path: str, plan: ParallelismPlan) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "plan": plan.to_dict(),
            "orchestrator_distributed": cls.to_training_orchestrator_distributed(plan),
            "megatron_args": cls.to_megatron_style_args(plan),
            "colossal_plugin": cls.to_colossal_plugin_spec(plan),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def _choose_backend(
        model_params: int,
        total_gpus: int,
        gpu_memory_gb: float,
        is_moe: bool,
        prefer_backend: str,
        training_stage: str,
    ) -> str:
        if prefer_backend != "auto":
            if prefer_backend not in FactoryBackendPlanner.BACKENDS:
                raise ValueError(f"Unknown backend: {prefer_backend}")
            return prefer_backend
        if is_moe:
            return "moe_hybrid_parallel" if total_gpus >= 4 else "deepspeed_zero3"
        if total_gpus <= 1:
            return "single"
        params_b = model_params / 1e9
        if training_stage in {"sft", "preference", "distill"} and params_b <= 13:
            return "deepspeed_zero2" if total_gpus >= 4 else "torch_fsdp"
        if params_b <= 3 and gpu_memory_gb >= 24:
            return "torch_ddp"
        if params_b <= 13:
            return "torch_fsdp" if total_gpus <= 4 else "deepspeed_zero2"
        if params_b <= 30:
            return "deepspeed_zero3"
        return "hybrid_parallel" if total_gpus >= 8 else "deepspeed_zero3"

    @staticmethod
    def _choose_tensor_parallel(model_params: int, total_gpus: int, backend: str) -> int:
        if backend not in {"hybrid_parallel", "moe_hybrid_parallel"}:
            return 1
        params_b = model_params / 1e9
        target = 8 if params_b >= 70 else (4 if params_b >= 30 else 2)
        return FactoryBackendPlanner._largest_factor(total_gpus, target)

    @staticmethod
    def _choose_pipeline_parallel(model_params: int, remaining_gpus: int, backend: str) -> int:
        if backend not in {"hybrid_parallel", "moe_hybrid_parallel"}:
            return 1
        params_b = model_params / 1e9
        target = 4 if params_b >= 70 else (2 if params_b >= 30 else 1)
        return FactoryBackendPlanner._largest_factor(remaining_gpus, target)

    @staticmethod
    def _choose_context_parallel(seq_length: int, remaining_gpus: int, backend: str) -> int:
        if backend not in {"hybrid_parallel", "moe_hybrid_parallel"} or seq_length < 8192:
            return 1
        for candidate in (4, 2):
            if candidate <= remaining_gpus and remaining_gpus % candidate == 0 and seq_length % (candidate * 2) == 0:
                return candidate
        return 1

    @staticmethod
    def _choose_expert_parallel(is_moe: bool, remaining_gpus: int, backend: str) -> int:
        if not is_moe or backend != "moe_hybrid_parallel":
            return 1
        return FactoryBackendPlanner._largest_factor(remaining_gpus, min(8, remaining_gpus))

    @staticmethod
    def _default_micro_batch(model_params: int, seq_length: int, gpu_memory_gb: float) -> int:
        params_b = model_params / 1e9
        if params_b >= 13 or seq_length >= 4096 or gpu_memory_gb <= 24:
            return 1
        if params_b >= 3:
            return 2
        return 4

    @staticmethod
    def _estimate_memory(
        model_params: int,
        seq_length: int,
        micro_batch_size: int,
        data_parallel: int,
        tensor_parallel: int,
        pipeline_parallel: int,
        expert_parallel: int,
        context_parallel: int,
        backend: str,
    ) -> (float, float):
        model_bf16_gb = model_params * 2 / (1024**3)
        shard = max(1, tensor_parallel * pipeline_parallel * expert_parallel)
        dp = max(1, data_parallel)
        if backend in {"deepspeed_zero3", "torch_fsdp"}:
            model_state = model_bf16_gb * 8.0 / max(1, shard * dp)
        elif backend == "deepspeed_zero2":
            model_state = (model_bf16_gb * 2.0 / shard) + (model_bf16_gb * 4.0 / max(1, shard * dp))
        elif backend in {"hybrid_parallel", "moe_hybrid_parallel"}:
            model_state = model_bf16_gb * 8.0 / shard
        else:
            model_state = model_bf16_gb * 8.0 / shard
        activation = (model_params / 1e9) * (seq_length / 2048) * micro_batch_size * 0.35
        activation = activation / max(1, tensor_parallel * context_parallel)
        return model_state, activation

    @staticmethod
    def _largest_factor(value: int, max_factor: int) -> int:
        value = max(1, int(value))
        max_factor = max(1, min(int(max_factor), value))
        for factor in range(max_factor, 0, -1):
            if value % factor == 0:
                return factor
        return 1
