"""
模型架构注册表 - 从 100M 到 7B 的预定义模型配置
支持架构: Dense, MoE (Mixture of Experts), DeepSeek-V3 style
基于 Chinchilla 缩放法则的最优训练量估算

前沿模型架构对比:
  GPT-2 (2019):   Dense, MHA, GeLU
  Llama (2023):    Dense, GQA, SwiGLU, RoPE, RMSNorm
  Mixtral (2024):  Sparse MoE, GQA, SwiGLU, RoPE, RMSNorm
  DeepSeek-V3:     Sparse MoE + MLA + Aux-loss-free + Multi-Token-Pred
  Mamba (2023):    SSM (State Space Model), 线性复杂度 O(N)
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def _get_torch_dtype():
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


@dataclass
class ModelSpec:
    name: str
    architecture: (
        str  # "llama" | "gpt2" | "gpt-neox" | "mixtral" | "deepseek" | "mamba"
    )
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_kv_heads: int  # GQA, 0 = MHA
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 0

    # ---- MoE (Mixture of Experts) ----
    num_experts: int = 0  # MoE 专家总数, 0=非MoE (Dense)
    num_experts_per_tok: int = 0  # 每个 token 激活的专家数 (top-k)
    expert_intermediate_size: int = 0  # 每个专家的 FFN 中间层大小
    router_aux_loss_coef: float = 0.001  # 辅助负载均衡 loss 系数
    shared_expert: bool = False  # DeepSeek-V3: 共享专家

    # ---- MLA (Multi-head Latent Attention) ----
    kv_lora_rank: int = 0  # DeepSeek-V3: KV 压缩维度, 0=不用MLA
    q_lora_rank: int = 0  # DeepSeek-V3: Q 压缩维度

    # ---- 多 Token 预测 ----
    multi_token_prediction: bool = False  # DeepSeek-V3: MTP
    num_mtp_layers: int = 0

    estimated_params: int = 0
    estimated_active_params: int = 0  # MoE: 每个 token 激活的参数量
    chinchilla_tokens: int = 0
    flops_per_token: float = 0

    def __post_init__(self):
        if self.estimated_params == 0:
            self.estimated_params = self._estimate_params()
        if self.chinchilla_tokens == 0:
            self.chinchilla_tokens = self._chinchilla_optimal_tokens()
        if self.flops_per_token == 0:
            self.flops_per_token = self._compute_flops_per_token()
        if self.estimated_active_params == 0:
            self.estimated_active_params = self._estimate_active_params()

    def _estimate_params(self) -> int:
        d = self.hidden_size
        n = self.num_hidden_layers
        kv = self.num_kv_heads if self.num_kv_heads > 0 else self.num_attention_heads
        v = self.vocab_size
        h_q = self.num_attention_heads

        embed = v * d
        head_dim = d // h_q

        # Attention: Q, K, V, O projections
        if self.kv_lora_rank > 0:
            # MLA: Q/KV 通过低秩压缩, 参数量大幅减少
            attn_params = (
                d * self.kv_lora_rank  # KV 压缩投影
                + self.kv_lora_rank * (kv + h_q) * head_dim  # 解压投影
                + d * self.q_lora_rank  # Q 压缩
                + self.q_lora_rank * h_q * head_dim  # Q 解压
                + h_q * head_dim * d  # O 投影
            )
        else:
            # 标准 MHA/GQA
            attn_params = (
                d * h_q * head_dim  # Q
                + d * kv * head_dim * 2  # K, V
                + h_q * head_dim * d  # O
            )

        # FFN
        if self.num_experts > 0:
            # MoE FFN: 每个 expert 有 gate/up/down 三个投影
            exp_ff = self.expert_intermediate_size or self.intermediate_size
            per_expert = d * exp_ff * 2 + exp_ff * d  # gate+up + down
            shared_ff = 0
            if self.shared_expert:
                shared_ff = d * exp_ff * 2 + exp_ff * d
            ffn_params = self.num_experts * per_expert + shared_ff
            # Router (no params, just a linear gate)
            ffn_params += d * self.num_experts
        else:
            # Dense FFN: gate + up + down (SwiGLU)
            ffn_params = d * self.intermediate_size * 2 + self.intermediate_size * d

        norm_params = 2 * d * n  # RMSNorm x2 per layer
        final_norm = d
        lm_head = v * d if not self.tie_word_embeddings else 0

        return (
            embed + (ffn_params + attn_params) * n + norm_params + final_norm + lm_head
        )

    def _estimate_active_params(self) -> int:
        """MoE: 每个 token 实际激活的参数量"""
        if self.num_experts == 0:
            return self.estimated_params
        # Dense layers + activated experts
        d = self.hidden_size
        n = self.num_hidden_layers
        v = self.vocab_size
        h_q = self.num_attention_heads
        kv = self.num_kv_heads if self.num_kv_heads > 0 else self.num_attention_heads
        head_dim = d // h_q

        # Attention params per layer (dense, always active)
        if self.kv_lora_rank > 0:
            attn_per_layer = (
                d * self.kv_lora_rank
                + self.kv_lora_rank * (kv + h_q) * head_dim
                + d * self.q_lora_rank
                + self.q_lora_rank * h_q * head_dim
                + h_q * head_dim * d
            )
        else:
            attn_per_layer = (
                d * h_q * head_dim + d * kv * head_dim * 2 + h_q * head_dim * d
            )

        # MoE FFN: only top-k experts activated
        exp_ff = self.expert_intermediate_size or self.intermediate_size
        per_expert_active = d * exp_ff * 2 + exp_ff * d  # gate+up + down
        active_experts = self.num_experts_per_tok or 2
        router = d * self.num_experts
        shared = (d * exp_ff * 2 + exp_ff * d) if self.shared_expert else 0
        moe_ffn_per_layer = active_experts * per_expert_active + shared + router

        embed = v * d
        norm_total = 2 * d * n + d
        lm_head = v * d if not self.tie_word_embeddings else 0

        return embed + n * (attn_per_layer + moe_ffn_per_layer) + norm_total + lm_head

    def _chinchilla_optimal_tokens(self) -> int:
        # Chinchilla: tokens = 20 * active_params (for MoE, it's about active params)
        active = self.estimated_active_params or self.estimated_params
        return active * 20

    def _compute_flops_per_token(self) -> float:
        # Forward FLOPs ≈ 2 * active_params (for MoE, only active params matter)
        active = self.estimated_active_params or self.estimated_params
        return 2.0 * active

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
            "estimated_params": self.estimated_params,
            "estimated_active_params": self.estimated_active_params,
            "chinchilla_tokens": self.chinchilla_tokens,
        }
        if self.num_experts > 0:
            d.update(
                {
                    "num_experts": self.num_experts,
                    "num_experts_per_tok": self.num_experts_per_tok,
                }
            )
        if self.kv_lora_rank > 0:
            d.update(
                {"kv_lora_rank": self.kv_lora_rank, "q_lora_rank": self.q_lora_rank}
            )
        return d

    def human_params(self) -> str:
        total = self.estimated_params
        active = self.estimated_active_params
        if self.num_experts > 0 and active != total:
            t = f"{total / 1e9:.1f}B" if total >= 1e9 else f"{total / 1e6:.0f}M"
            a = f"{active / 1e9:.1f}B" if active >= 1e9 else f"{active / 1e6:.0f}M"
            return f"{t} (激活{a})"
        if total >= 1e9:
            return f"{total / 1e9:.1f}B"
        return f"{total / 1e6:.0f}M"

    def human_tokens(self) -> str:
        t = self.chinchilla_tokens
        if t >= 1e9:
            return f"{t / 1e9:.1f}B"
        return f"{t / 1e6:.0f}M"


# ---- 预定义的缩放路线图 ----

MODEL_SPECS: Dict[str, ModelSpec] = {
    # ===== Stage 1: 100M 级别 - 单卡快速验证 =====
    "gpt2-small-124m": ModelSpec(
        name="gpt2-small-124m",
        architecture="gpt2",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_kv_heads=0,
        intermediate_size=3072,
        vocab_size=32000,
        max_position_embeddings=2048,
    ),
    "llama-tiny-150m": ModelSpec(
        name="llama-tiny-150m",
        architecture="llama",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_kv_heads=4,
        intermediate_size=3072,
        vocab_size=32000,
        max_position_embeddings=2048,
        rope_theta=10000.0,
    ),
    # ===== Stage 2: 300M 级别 - 单卡 A100 完整训练 =====
    "llama-300m": ModelSpec(
        name="llama-300m",
        architecture="llama",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_kv_heads=8,
        intermediate_size=4096,
        vocab_size=32000,
        max_position_embeddings=4096,
        rope_theta=10000.0,
    ),
    "gpt-neox-350m": ModelSpec(
        name="gpt-neox-350m",
        architecture="gpt-neox",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_kv_heads=0,
        intermediate_size=4096,
        vocab_size=32000,
        max_position_embeddings=4096,
        rope_theta=10000.0,
    ),
    # ===== Stage 3: 1B 级别 - 多卡训练,开始有意义的能力涌现 =====
    "llama-1b": ModelSpec(
        name="llama-1b",
        architecture="llama",
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=8192,
        vocab_size=32000,
        max_position_embeddings=8192,
        rope_theta=10000.0,
    ),
    "llama-1.5b": ModelSpec(
        name="llama-1.5b",
        architecture="llama",
        hidden_size=2048,
        num_hidden_layers=28,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=8192,
        vocab_size=64000,
        max_position_embeddings=8192,
        rope_theta=10000.0,
    ),
    # ===== Stage 4: 3B 级别 - SFT/RLHF 就绪的基座模型 =====
    "qwen-tiny-160m": ModelSpec(
        name="qwen-tiny-160m",
        architecture="qwen",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_kv_heads=2,
        intermediate_size=2048,
        vocab_size=151936,
        max_position_embeddings=4096,
        rope_theta=1000000.0,
        norm_eps=1e-6,
        tie_word_embeddings=True,
    ),
    "qwen-300m": ModelSpec(
        name="qwen-300m",
        architecture="qwen",
        hidden_size=1024,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_kv_heads=4,
        intermediate_size=2816,
        vocab_size=151936,
        max_position_embeddings=8192,
        rope_theta=1000000.0,
        norm_eps=1e-6,
        tie_word_embeddings=True,
    ),
    "qwen-700m": ModelSpec(
        name="qwen-700m",
        architecture="qwen",
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_kv_heads=4,
        intermediate_size=4096,
        vocab_size=151936,
        max_position_embeddings=8192,
        rope_theta=1000000.0,
        norm_eps=1e-6,
        tie_word_embeddings=True,
    ),
    "qwen-1.5b": ModelSpec(
        name="qwen-1.5b",
        architecture="qwen",
        hidden_size=2048,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_kv_heads=4,
        intermediate_size=5504,
        vocab_size=151936,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        norm_eps=1e-6,
        tie_word_embeddings=True,
    ),
    "llama-3b": ModelSpec(
        name="llama-3b",
        architecture="llama",
        hidden_size=3200,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=12800,
        vocab_size=64000,
        max_position_embeddings=8192,
        rope_theta=10000.0,
    ),
    # ===== Stage 5: 7B 级别 - 对标开源 7B 模型 =====
    "llama-7b": ModelSpec(
        name="llama-7b",
        architecture="llama",
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=64000,
        max_position_embeddings=32768,
        rope_theta=500000.0,
    ),
    "llama-8b": ModelSpec(
        name="llama-8b",
        architecture="llama",
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=16384,
        vocab_size=128000,
        max_position_embeddings=32768,
        rope_theta=500000.0,
    ),
    # ============================================================
    # MoE (Mixture of Experts) - 稀疏激活,总参数大但每 token 激活少
    # 核心优势: 同样算力下,总知识容量远大于 Dense 模型
    # MoE 参数量 = Dense部分 + num_experts × expert_params
    # 每 token 激活 = Dense部分 + num_experts_per_tok × expert_params
    # ============================================================
    # --- 小型 MoE (可用 4 GPU 训练) ---
    "moe-1b-8e": ModelSpec(
        name="moe-1b-8e",
        architecture="mixtral",
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_kv_heads=8,
        intermediate_size=4096,
        vocab_size=32000,
        max_position_embeddings=4096,
        num_experts=8,
        num_experts_per_tok=2,
        expert_intermediate_size=4096,
        router_aux_loss_coef=0.01,
    ),
    # --- Mixtral 8x7B 级别 (对标开源最强 MoE) ---
    "mixtral-8x7b": ModelSpec(
        name="mixtral-8x7b",
        architecture="mixtral",
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        num_experts=8,
        num_experts_per_tok=2,  # Top-2 路由
        expert_intermediate_size=14336,
        router_aux_loss_coef=0.01,
    ),
    # --- DeepSeek-V3 风格 (MoE + MLA + 共享专家 + MTP) ---
    # 总参数 ~671B, 激活参数 ~37B per token
    "deepseek-v3-style": ModelSpec(
        name="deepseek-v3-style",
        architecture="deepseek",
        hidden_size=7168,
        num_hidden_layers=61,
        num_attention_heads=128,
        num_kv_heads=128,
        intermediate_size=18432,
        vocab_size=129280,
        max_position_embeddings=131072,
        rope_theta=10000000.0,
        # MoE
        num_experts=256,  # 256 个专家 (其中 1 个共享)
        num_experts_per_tok=8,  # Top-8 路由
        expert_intermediate_size=2048,  # 每个专家的 FFN 较小
        router_aux_loss_coef=0.0,  # 无辅助 loss 负载均衡 (aux-loss-free)
        shared_expert=True,
        # MLA (Multi-head Latent Attention)
        kv_lora_rank=512,  # KV 压缩到 512 维
        q_lora_rank=1536,  # Q 压缩到 1536 维
        # 多 Token 预测 (MTP)
        multi_token_prediction=True,
        num_mtp_layers=1,
    ),
}


class ModelRegistry:
    """模型架构注册表,提供创建/查询/估算功能"""

    @staticmethod
    def get(name: str) -> ModelSpec:
        """获取模型规格"""
        if name not in MODEL_SPECS:
            available = ", ".join(MODEL_SPECS.keys())
            raise KeyError(f"未知模型: {name}. 可用: {available}")
        return MODEL_SPECS[name]

    @staticmethod
    def list_all() -> List[ModelSpec]:
        """列出所有模型"""
        return list(MODEL_SPECS.values())

    @staticmethod
    def list_by_size(
        max_params: Optional[int] = None, min_params: Optional[int] = None
    ) -> List[ModelSpec]:
        """按参数量筛选"""
        specs = MODEL_SPECS.values()
        if min_params is not None:
            specs = [s for s in specs if s.estimated_params >= min_params]
        if max_params is not None:
            specs = [s for s in specs if s.estimated_params <= max_params]
        return sorted(specs, key=lambda s: s.estimated_params)

    @staticmethod
    def architecture_support(name_or_spec) -> Dict:
        """返回某个模型规格对应的架构支持矩阵。"""
        from .Architecture import ArchitectureRegistry

        spec = (
            ModelRegistry.get(name_or_spec)
            if isinstance(name_or_spec, str)
            else name_or_spec
        )
        return ArchitectureRegistry.from_model_spec(spec).to_dict()

    @staticmethod
    def validate_architecture(name_or_spec, capability: str = "pretrain") -> bool:
        """校验某个模型规格是否支持指定能力；不支持时抛出明确错误。"""
        from .Architecture import ArchitectureRegistry

        spec = (
            ModelRegistry.get(name_or_spec)
            if isinstance(name_or_spec, str)
            else name_or_spec
        )
        ArchitectureRegistry.require(spec.architecture, capability)
        return True

    @staticmethod
    def create_model(
        spec: ModelSpec, vocab_size_override: Optional[int] = None
    ) -> "PreTrainedModel":
        """
        从 ModelSpec 创建 HuggingFace 模型。
        支持 Llama, GPT-2, GPT-NeoX 架构。
        """
        from transformers import (
            AutoModelForCausalLM,
            LlamaConfig,
            GPT2Config,
            GPTNeoXConfig,
        )

        vocab_size = vocab_size_override or spec.vocab_size
        ModelRegistry.validate_architecture(spec, "pretrain")

        if spec.architecture == "llama":
            config = LlamaConfig(
                hidden_size=spec.hidden_size,
                num_hidden_layers=spec.num_hidden_layers,
                num_attention_heads=spec.num_attention_heads,
                num_key_value_heads=spec.num_kv_heads
                if spec.num_kv_heads > 0
                else spec.num_attention_heads,
                intermediate_size=spec.intermediate_size,
                vocab_size=vocab_size,
                max_position_embeddings=spec.max_position_embeddings,
                rope_theta=spec.rope_theta,
                rms_norm_eps=spec.norm_eps,
                tie_word_embeddings=spec.tie_word_embeddings,
                use_cache=False,
            )
            return AutoModelForCausalLM.from_config(
                config, torch_dtype=_get_torch_dtype()
            )

        elif spec.architecture == "gpt2":
            config = GPT2Config(
                n_embd=spec.hidden_size,
                n_layer=spec.num_hidden_layers,
                n_head=spec.num_attention_heads,
                n_inner=spec.intermediate_size,
                vocab_size=vocab_size,
                n_positions=spec.max_position_embeddings,
                use_cache=False,
            )
            return AutoModelForCausalLM.from_config(
                config, torch_dtype=_get_torch_dtype()
            )

        elif spec.architecture == "gpt-neox":
            config = GPTNeoXConfig(
                hidden_size=spec.hidden_size,
                num_hidden_layers=spec.num_hidden_layers,
                num_attention_heads=spec.num_attention_heads,
                intermediate_size=spec.intermediate_size,
                vocab_size=vocab_size,
                max_position_embeddings=spec.max_position_embeddings,
                rotary_emb_base=spec.rope_theta,
                use_cache=False,
            )
            return AutoModelForCausalLM.from_config(
                config, torch_dtype=_get_torch_dtype()
            )

        elif spec.architecture == "qwen":
            try:
                from transformers import Qwen2Config
            except ImportError as exc:
                raise ImportError(
                    "Qwen scratch pretraining requires transformers with Qwen2Config support. "
                    "Please upgrade transformers or use a Llama model_config."
                ) from exc

            config = Qwen2Config(
                hidden_size=spec.hidden_size,
                num_hidden_layers=spec.num_hidden_layers,
                num_attention_heads=spec.num_attention_heads,
                num_key_value_heads=spec.num_kv_heads
                if spec.num_kv_heads > 0
                else spec.num_attention_heads,
                intermediate_size=spec.intermediate_size,
                vocab_size=vocab_size,
                max_position_embeddings=spec.max_position_embeddings,
                rope_theta=spec.rope_theta,
                rms_norm_eps=spec.norm_eps,
                tie_word_embeddings=spec.tie_word_embeddings,
                use_sliding_window=spec.use_sliding_window,
                sliding_window=spec.sliding_window or None,
                use_cache=False,
            )
            return AutoModelForCausalLM.from_config(
                config, torch_dtype=_get_torch_dtype()
            )

        elif spec.architecture == "mixtral":
            # MoE via Mixtral config
            from transformers import MixtralConfig

            config = MixtralConfig(
                hidden_size=spec.hidden_size,
                num_hidden_layers=spec.num_hidden_layers,
                num_attention_heads=spec.num_attention_heads,
                num_key_value_heads=spec.num_kv_heads
                if spec.num_kv_heads > 0
                else spec.num_attention_heads,
                intermediate_size=spec.intermediate_size,
                vocab_size=vocab_size,
                max_position_embeddings=spec.max_position_embeddings,
                rope_theta=spec.rope_theta,
                rms_norm_eps=spec.norm_eps,
                num_local_experts=spec.num_experts or 8,
                num_experts_per_tok=spec.num_experts_per_tok or 2,
                expert_intermediate_size=spec.expert_intermediate_size
                or spec.intermediate_size,
                router_aux_loss_coef=spec.router_aux_loss_coef,
                use_cache=False,
            )
            return AutoModelForCausalLM.from_config(
                config, torch_dtype=_get_torch_dtype()
            )

        elif spec.architecture == "deepseek":
            # DeepSeek-V3 风格: MoE + MLA
            # MLA (Multi-head Latent Attention) 需要自定义建模
            # 当前回退到 Mixtral 配置 + 注释说明 MLA 参数
            logger.warning(
                "DeepSeek-V3 MLA 需要自定义 modeling 代码。"
                "当前使用 Mixtral MoE 配置创建模型，MLA 参数已记录。"
                "MLA 关键参数: kv_lora_rank=%d, q_lora_rank=%d",
                spec.kv_lora_rank,
                spec.q_lora_rank,
            )
            from transformers import MixtralConfig

            config = MixtralConfig(
                hidden_size=spec.hidden_size,
                num_hidden_layers=spec.num_hidden_layers,
                num_attention_heads=spec.num_attention_heads,
                num_key_value_heads=spec.num_kv_heads
                if spec.num_kv_heads > 0
                else spec.num_attention_heads,
                intermediate_size=spec.intermediate_size,
                vocab_size=vocab_size,
                max_position_embeddings=spec.max_position_embeddings,
                rope_theta=spec.rope_theta,
                rms_norm_eps=spec.norm_eps,
                num_local_experts=spec.num_experts or 256,
                num_experts_per_tok=spec.num_experts_per_tok or 8,
                expert_intermediate_size=spec.expert_intermediate_size
                or spec.intermediate_size,
                router_aux_loss_coef=spec.router_aux_loss_coef,
                use_cache=False,
            )
            return AutoModelForCausalLM.from_config(
                config, torch_dtype=_get_torch_dtype()
            )

        else:
            raise ValueError(f"不支持的架构: {spec.architecture}")

    @staticmethod
    def create_saddle_model(spec: ModelSpec, vocab_size_override: Optional[int] = None):
        """Create SaddleLLM's own modular PyTorch model from a ModelSpec."""
        from .SaddleModeling import SaddleForCausalLM, SaddleModelConfig

        cfg = SaddleModelConfig.from_model_spec(spec)
        if vocab_size_override is not None:
            cfg.vocab_size = vocab_size_override
        return SaddleForCausalLM(cfg)

    @staticmethod
    def estimate_memory(
        spec: ModelSpec,
        batch_size: int = 1,
        seq_length: Optional[int] = None,
        dtype: str = "bf16",
        optimizer: str = "adamw",
        use_gradient_checkpointing: bool = True,
    ) -> Dict[str, float]:
        """
        估算训练/推理所需显存(GB)。

        返回: {"model_gb": ..., "optimizer_gb": ..., "activations_gb": ..., "total_gb": ...}
        """
        seq_len = seq_length or spec.max_position_embeddings
        bytes_per_param = 2 if dtype in ("bf16", "fp16") else 4
        param_count = spec.estimated_params

        # 模型参数
        model_gb = param_count * bytes_per_param / (1024**3)

        # 优化器状态 (AdamW: param + momentum + variance = 3x)
        opt_mult = 3 if optimizer == "adamw" else 1
        optimizer_gb = param_count * 4 * opt_mult / (1024**3)

        # 激活值
        n_layers = spec.num_hidden_layers
        hidden = spec.hidden_size
        activations_per_layer = batch_size * seq_len * hidden * bytes_per_param
        if use_gradient_checkpointing:
            activations_per_layer *= 0.25  # rough estimate
        activations_gb = n_layers * activations_per_layer / (1024**3)

        total_gb = model_gb + optimizer_gb + activations_gb

        return {
            "model_gb": round(model_gb, 2),
            "optimizer_gb": round(optimizer_gb, 2),
            "activations_gb": round(activations_gb, 2),
            "total_gb": round(total_gb, 2),
            "estimated_params": param_count,
        }

    @staticmethod
    def compare_specs(specs: Optional[List[str]] = None) -> str:
        """生成模型对比表格 (含架构类型和 MoE 信息)"""
        if specs is None:
            specs = list(MODEL_SPECS.keys())
        models = [MODEL_SPECS[s] for s in specs if s in MODEL_SPECS]

        header = f"{'Name':<22} {'Arch':>10} {'Params':>16} {'Tokens':>10} {'L':>4} {'d':>6} {'Heads':>6}"
        lines = [header, "-" * 90]
        for m in sorted(models, key=lambda x: x.estimated_params):
            arch = m.architecture
            if m.num_experts > 0:
                arch += f"-{m.num_experts}e"
            lines.append(
                f"{m.name:<22} {arch:>10} {m.human_params():>16} "
                f"{m.human_tokens():>10} {m.num_hidden_layers:>4} "
                f"{m.hidden_size:>6} {m.num_attention_heads:>6}"
            )
        return "\n".join(lines)

    @staticmethod
    def compare_architectures() -> str:
        """详细对比架构差异 (Dense vs MoE vs DeepSeek)"""
        return """
大模型架构对比
================

                     GPT-2    Llama-3    Mixtral     DeepSeek-V3
──────────────────────────────────────────────────────────────────
参数量               1.5B      8B         46.7B       671B
激活参数/Token       1.5B      8B         12.9B       37B
Attention 类型       MHA       GQA        GQA         MLA(压缩KV)
FFN 类型             GeLU      SwiGLU     SwiGLU-MoE  SwiGLU-MoE
专家数               1 (Dense) 1 (Dense)  8           256+1(共享)
每Token激活专家      1         1          2           8
KV Cache 压缩        1x        1x         1x          ~10x (MLA)
位置编码             学习式     RoPE       RoPE        RoPE
归一化               LayerNorm  RMSNorm    RMSNorm     RMSNorm
多Token预测          无        无         无          有(MTP)
负载均衡             不需要     不需要     Aux Loss    无Aux Loss
──────────────────────────────────────────────────────────────────
核心创新              奠基之作   GQA+SwiGLU  稀疏MoE    MLA+无Aux+多Token
训练效率             基准       2-3x        5-8x       10-15x
推理效率             基准       1.5x        1x(激活少) 3-5x(MLA压缩)
"""

    @staticmethod
    def recommend_for_gpu(
        gpu_memory_gb: int, strategy: str = "full_finetune"
    ) -> List[ModelSpec]:
        """根据 GPU 显存推荐可行的模型"""
        overhead_gb = 4  # CUDA context etc
        available = gpu_memory_gb - overhead_gb

        if strategy == "full_finetune":
            suitable = []
            for spec in sorted(MODEL_SPECS.values(), key=lambda s: s.estimated_params):
                mem = ModelRegistry.estimate_memory(
                    spec, batch_size=1, optimizer="adamw"
                )
                if mem["total_gb"] <= available:
                    suitable.append(spec)
            return suitable

        elif strategy == "lora":
            # LoRA 只需要 ~10-20% 的优化器显存
            suitable = []
            for spec in sorted(MODEL_SPECS.values(), key=lambda s: s.estimated_params):
                mem = ModelRegistry.estimate_memory(
                    spec, batch_size=1, optimizer="adamw"
                )
                lora_mem = (
                    mem["model_gb"] + mem["optimizer_gb"] * 0.15 + mem["activations_gb"]
                )
                if lora_mem <= available:
                    suitable.append(spec)
            return suitable

        return []
