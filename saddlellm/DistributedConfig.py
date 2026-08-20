"""
分布式训练配置 - 统一 DDP / FSDP / DeepSpeed ZeRO 配置
"""
import json
import os
import importlib.util
from typing import Literal, Optional, Dict
from dataclasses import dataclass


@dataclass
class DistributedConfig:
    strategy: Literal["ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3", "single"] = "single"
    num_gpus: int = 1
    num_nodes: int = 1
    gradient_accumulation_steps: int = 1
    ddp_backend: str = "auto"

    # FSDP
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_offload_params: bool = False
    fsdp_auto_wrap_policy: str = "transformer_based"
    fsdp_cpu_ram_efficient_loading: bool = False
    fsdp_sync_module_states: bool = True
    fsdp_use_orig_params: bool = False

    # DeepSpeed
    zero_offload_to_cpu: bool = False
    zero_offload_to_nvme: bool = False
    deepspeed_config_path: Optional[str] = None
    zero_stage3_param_persistence_threshold: int = 100000
    zero_reduce_bucket_size: int = 500000000
    zero_allgather_bucket_size: int = 500000000

    # 通用
    bf16: bool = True
    fp16: bool = False
    tf32: bool = False

    # 通信
    nccl_timeout: int = 1800  # 30分钟

    def __post_init__(self):
        self.strategy = str(self.strategy).lower()
        if self.strategy not in {"single", "ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"}:
            raise ValueError(f"unsupported distributed strategy: {self.strategy!r}")
        if int(self.num_gpus) < 1 or int(self.num_nodes) < 1:
            raise ValueError("num_gpus and num_nodes must be >= 1")
        self.ddp_backend = str(self.ddp_backend).lower()
        if self.ddp_backend not in {"auto", "nccl", "gloo", "mpi", "ucc", "hccl"}:
            raise ValueError(f"unsupported DDP backend: {self.ddp_backend!r}")
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled")

    def to_training_args(self) -> Dict:
        """转换为 HuggingFace TrainingArguments 可用的参数"""
        args = {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "tf32": self.tf32,
            "dataloader_num_workers": 4,
        }

        if self.strategy == "ddp":
            args["ddp_find_unused_parameters"] = False
            backend = self.ddp_backend
            if backend == "auto":
                try:
                    import torch
                    backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
                except Exception:
                    backend = "gloo"
            args["ddp_backend"] = backend
        elif self.strategy == "fsdp":
            args["fsdp"] = str(self.fsdp_sharding_strategy).upper()
            args["fsdp_config"] = self._fsdp_config_dict()
        elif self.strategy in ("deepspeed_zero2", "deepspeed_zero3"):
            if importlib.util.find_spec("deepspeed") is None:
                raise RuntimeError("DeepSpeed strategy selected but deepspeed is not installed")
            args["deepspeed"] = self.deepspeed_config_path or self._deepspeed_config_dict()

        return args

    def to_deepspeed_config(self) -> Dict:
        """生成 DeepSpeed 完整配置"""
        zero_stage = 3 if self.strategy == "deepspeed_zero3" else 2

        config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "bf16": {"enabled": self.bf16},
            "fp16": {"enabled": self.fp16},
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.zero_offload_to_cpu else "none",
                },
                "offload_param": {
                    "device": "cpu" if self.zero_offload_to_cpu else "none",
                },
                "stage3_gather_16bit_weights_on_model_save": True,
                "stage3_param_persistence_threshold": self.zero_stage3_param_persistence_threshold,
                "reduce_bucket_size": self.zero_reduce_bucket_size,
                "allgather_bucket_size": self.zero_allgather_bucket_size,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": "auto",
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto",
                },
            },
        }

        if self.zero_offload_to_nvme:
            config["zero_optimization"]["offload_optimizer"]["device"] = "nvme"
            config["zero_optimization"]["offload_optimizer"]["nvme_path"] = "./nvme_offload"

        return config

    def save_deepspeed_config(self, path: str) -> str:
        """保存 DeepSpeed 配置到 JSON 文件"""
        config = self.to_deepspeed_config()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        return path

    def save_accelerate_config(self, path: str) -> str:
        """Save a minimal Accelerate config for the selected strategy."""
        mixed_precision = "bf16" if self.bf16 else ("fp16" if self.fp16 else "no")
        if self.strategy == "fsdp":
            distributed_type = "FSDP"
        elif self.strategy in ("deepspeed_zero2", "deepspeed_zero3"):
            distributed_type = "DEEPSPEED"
        elif self.strategy == "ddp":
            distributed_type = "MULTI_GPU"
        else:
            distributed_type = "NO"
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": distributed_type,
            "mixed_precision": mixed_precision,
            "num_machines": self.num_nodes,
            "num_processes": max(1, self.num_gpus * self.num_nodes),
            "use_cpu": self.num_gpus == 0,
        }
        if distributed_type == "DEEPSPEED":
            config["deepspeed_config"] = self.to_deepspeed_config()
        if distributed_type == "FSDP":
            config["fsdp_config"] = self._fsdp_config_dict()

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return path

    @classmethod
    def recommend_for_model(
        cls,
        params: int,
        gpu_memory_gb: float,
        num_gpus: int = 1,
        prefer_offload: bool = False,
    ) -> "DistributedConfig":
        """Choose a conservative distributed strategy from model size and GPU memory."""
        model_gb = params * 2 / (1024**3)
        total_gpu_gb = max(1, num_gpus) * gpu_memory_gb
        if num_gpus <= 1:
            strategy = "single"
        elif model_gb * 8 < total_gpu_gb:
            strategy = "ddp"
        elif num_gpus <= 4:
            strategy = "deepspeed_zero2"
        else:
            strategy = "deepspeed_zero3"
        return cls(
            strategy=strategy,
            num_gpus=num_gpus,
            zero_offload_to_cpu=prefer_offload or model_gb * 12 > total_gpu_gb,
            bf16=True,
            fp16=False,
        )

    @classmethod
    def save_template_bundle(
        cls,
        output_dir: str,
        params: int,
        gpu_memory_gb: float,
        num_gpus: int = 1,
    ) -> Dict[str, str]:
        """Create DeepSpeed and Accelerate templates for a planned pretraining run."""
        os.makedirs(output_dir, exist_ok=True)
        cfg = cls.recommend_for_model(params=params, gpu_memory_gb=gpu_memory_gb, num_gpus=num_gpus)
        paths = {
            "strategy": cfg.strategy,
            "accelerate": cfg.save_accelerate_config(os.path.join(output_dir, "accelerate_config.json")),
        }
        if cfg.strategy in ("deepspeed_zero2", "deepspeed_zero3"):
            paths["deepspeed"] = cfg.save_deepspeed_config(os.path.join(output_dir, "deepspeed_config.json"))
        return paths

    def _fsdp_config_dict(self) -> Dict:
        return {
            "fsdp_sharding_strategy": str(self.fsdp_sharding_strategy).upper(),
            "fsdp_offload_params": self.fsdp_offload_params,
            "fsdp_auto_wrap_policy": f"TRANSFORMER_BASED_WRAP" if self.fsdp_auto_wrap_policy == "transformer_based" else "SIZE_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": "SaddleDecoderLayer",
            "fsdp_cpu_ram_efficient_loading": self.fsdp_cpu_ram_efficient_loading,
            "fsdp_sync_module_states": self.fsdp_sync_module_states,
            "fsdp_use_orig_params": self.fsdp_use_orig_params,
            "version": 1,
        }

    def _deepspeed_config_dict(self) -> Dict:
        return self.to_deepspeed_config()

    @staticmethod
    def auto_detect_gpus() -> int:
        """自动检测可用 GPU 数量"""
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0

    @staticmethod
    def recommend_strategy(num_gpus: int) -> str:
        """根据 GPU 数量推荐策略"""
        if num_gpus <= 1:
            return "single"
        elif num_gpus <= 2:
            return "ddp"
        elif num_gpus <= 4:
            return "deepspeed_zero2"
        else:
            return "deepspeed_zero3"
