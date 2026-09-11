"""
OpenAI & DeepSeek & Anthropic 独门技术

═══ DeepSeek 独有 ═══
  1. Multi-Token Prediction (MTP)      — 一次预测多个未来 token, 提升样本效率 2-3x
  2. Aux-loss-free MoE 负载均衡         — 动态偏置调整, 不用辅助 loss, 收敛更快

═══ OpenAI 独有 ═══
  3. Iterative Alignment               — 多轮 DPO/RL, 每轮生成的比上一轮更好
  4. Test-time Compute Scaling         — o1/o3 风格, 推理时分配更多"思考"时间

═══ Anthropic 独有 ═══
  5. RLAIF (RL from AI Feedback)       — 用强模型当裁判, 代替人工反馈
  6. SLURP Pretraining                 — 预训练阶段就混合指令数据
  7. Long Context RoPE Scaling         — Claude 200K 长上下文的秘密
  8. Sparse Autoencoder (SAE)          — 理解模型内部, 发现可解释特征

═══ 共同 ═══
  FP8 混合精度训练                      — H100/H800 上速度翻倍
"""
import math
import copy
import logging
import random
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Technique 1: Multi-Token Prediction (DeepSeek-V3)
# ============================================================

class MultiTokenPredictionHead(nn.Module):
    """
    多 Token 预测头。

    DeepSeek-V3 核心创新: 不只用最后一个 token 的 hidden state 预测下一个 token,
    而是用每个位置的 hidden state 预测未来 2-4 个 token。

    原理:
      标准 LM: h_t → logits_t+1 (预测 1 个 token)
      MTP:     h_t → [logits_t+1, logits_t+2, logits_t+3, logits_t+4]

    为什么有效:
      - 更强的梯度信号: 每个位置有 4 个预测目标而非 1 个
      - 更好的 representations: 强制模型学习更长程的依赖
      - 训练效率提升 2-3x: 同样数据量学得更快
      - 推理时可复用: MTP heads 可用于投机解码

    论文: DeepSeek-V3, Section 2.2.2
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_extra_tokens: int = 3,  # 额外预测 3 个未来 token (共预测 4 个)
        shared_head: bool = True,    # 所有 head 共享权重 (DeepSeek 的做法)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_extra_tokens = num_extra_tokens
        self.total_heads = num_extra_tokens + 1  # 1 main + N extra

        # 共享的输出投影 (所有 head 用同一套权重)
        if shared_head:
            self.output_projections = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(self.total_heads)
            ])
        else:
            self.output_projections = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(self.total_heads)
            ])

        # 每个 head 有自己的 embedding 归一化层
        self.head_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(self.total_heads)
        ])

        # 额外的 transformer block (可选, DeepSeek V3 加了 1 层)
        self.extra_block = None  # 可选: nn.TransformerEncoderLayer

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            list of logits for [t+1, t+2, ..., t+N]
        """
        outputs = []
        h = hidden_states
        for i in range(self.total_heads):
            # 每个 head 独立的 norm + 投影
            h_norm = self.head_norms[i](h)
            if self.extra_block is not None and i > 0:
                h_norm = self.extra_block(h_norm)
            logits = self.output_projections[i](h_norm)
            outputs.append(logits)
        return outputs


class MultiTokenPredictionLoss:
    """
    MTP 训练 Loss。

    对每个 future token 计算 CrossEntropy, 加权求和。
    DeepSeek V3: 权重递减 [1.0, 0.3, 0.1, 0.05]

    用法:
        mtp_head = MultiTokenPredictionHead(...)
        mtp_loss = MultiTokenPredictionLoss(weights=[1.0, 0.3, 0.1])

        hidden = model(...)
        logits_list = mtp_head(hidden)  # [logits_t+1, logits_t+2, logits_t+3]

        loss = mtp_loss(logits_list, labels)
    """

    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights or [1.0, 0.3, 0.1, 0.05]

    def compute(
        self,
        logits_list: List[torch.Tensor],
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算 MTP loss。

        logits_list[i] 的形状: (batch, seq_len, vocab_size)
        labels 的形状: (batch, seq_len)

        对第 k 个 head, 目标 label 是 labels 右移 k 位。
        """
        total_loss = 0.0
        head_losses = {}

        for k, logits in enumerate(logits_list):
            if k >= len(self.weights):
                break

            # 对齐: logits[t] 预测 labels[t+k]
            # logits 截断前部分, labels 截断后部分
            seq_len = logits.shape[1]
            if k > 0:
                shifted_logits = logits[:, :seq_len - k, :]
                shifted_labels = labels[:, k:seq_len]
            else:
                shifted_logits = logits
                shifted_labels = labels

            # Flatten
            flat_logits = shifted_logits.reshape(-1, shifted_logits.shape[-1])
            flat_labels = shifted_labels.reshape(-1)

            head_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
            head_losses[f"mtp_loss_{k}"] = head_loss.item()
            total_loss += self.weights[k] * head_loss

        return total_loss, head_losses


def add_mtp_to_model(model: nn.Module, vocab_size: int = None, num_extra_tokens: int = 3):
    """
    给现有模型添加 MTP head。

    用法:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_config(config)
        model_with_mtp = add_mtp_to_model(model)
    """
    hidden_size = None
    for name, param in model.named_parameters():
        if "embed" in name or "wte" in name:
            hidden_size = param.shape[-1]
            break

    if hidden_size is None:
        hidden_size = getattr(model.config, "hidden_size", 768)

    if vocab_size is None:
        vocab_size = getattr(model.config, "vocab_size", 32000)

    # 添加 MTP head
    mtp_head = MultiTokenPredictionHead(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_extra_tokens=num_extra_tokens,
    )

    # 附加到模型
    model.mtp_head = mtp_head
    model.mtp_loss_fn = MultiTokenPredictionLoss()

    logger.info(f"MTP head 已添加: hidden={hidden_size}, vocab={vocab_size}, extra_tokens={num_extra_tokens}")
    return model


# ============================================================
# Technique 2: Aux-loss-free MoE 负载均衡 (DeepSeek-V3)
# ============================================================

class AuxLossFreeRouter(nn.Module):
    """
    无辅助 Loss 的 MoE 路由器 — DeepSeek-V3 首创。

    传统 MoE: 用 aux loss 强制每个 Expert 被均等使用
    问题: aux loss 和主 loss 竞争, 导致收敛变慢，大梯度干扰

    DeepSeek-V3 方法:
      1. 每个 Expert 维护一个 bias
      2. 如果 Expert 超载, bias -= 一个小值 (让它被选中的概率降低)
      3. 如果 Expert 空闲, bias += 一个小值 (让它被选中的概率升高)
      4. 主 loss 不受影响, bias 单独更新
    -> 渐进式负载均衡, 不与主目标冲突

    实现要点:
      - bias 只在训练时更新, 推理时冻结
      - bias 更新量很小 (gamma ≈ 0.001)
      - 每个 step 监测 expert load, 动态调整
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        bias_update_rate: float = 0.001,  # gamma, 越小越渐进
        target_load: float = 1.0,           # 目标: 每个 expert load = total / num_experts
    ):
        super().__init__()
        self.num_experts = num_experts
        self.bias_update_rate = bias_update_rate
        self.target_load = target_load

        # 每个 Expert 的可学习 bias
        self.register_buffer("expert_bias", torch.zeros(num_experts))

        # 负载统计
        self.register_buffer("_load_counter", torch.zeros(num_experts))
        self.register_buffer("_total_tokens", torch.zeros(1))

    def forward(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        在 router logits 上加上 bias, 引导均衡路由。

        Args:
            router_logits: (batch, seq_len, num_experts)

        Returns:
            adjusted_logits: (batch, seq_len, num_experts)
        """
        return router_logits + self.expert_bias.unsqueeze(0).unsqueeze(0)

    def update_bias(self, expert_indices: torch.Tensor):
        """
        根据本轮负载更新 bias。

        Args:
            expert_indices: (num_tokens, top_k) — 每个 token 被路由到的 Expert 索引
        """
        with torch.no_grad():
            # 统计每个 Expert 被选了多少次
            load = torch.bincount(
                expert_indices.flatten(),
                minlength=self.num_experts
            ).float()

            avg_load = load.mean() + 1e-8

            # 超载的 Expert: bias 降低
            # 空闲的 Expert: bias 升高
            delta = self.bias_update_rate * (load / avg_load - self.target_load)

            # 符号取反: load > avg → bias 降低
            self.expert_bias -= delta

            # 累积统计
            self._load_counter += load
            self._total_tokens += expert_indices.numel()

    def get_load_balance(self) -> float:
        """返回负载均衡指标 (0-1, 1=完美均衡)。"""
        if self._total_tokens.item() < 1:
            return 1.0
        probs = self._load_counter / self._total_tokens
        # normalized entropy
        if self.num_experts <= 1:
            return 1.0
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        max_entropy = math.log(self.num_experts)
        return (entropy / max_entropy).item()


class AuxLossFreeMoELayer(nn.Module):
    """
    使用 Aux-loss-free 路由的 MoE FFN 层。

    结构:
      input → Router → TopK Experts (with bias routing) → weighted sum → output
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        bias_update_rate: float = 0.001,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.aux_free_router = AuxLossFreeRouter(
            num_experts, hidden_size, bias_update_rate
        )

        # Experts (gate + up + down for each)
        self.gate_proj = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False)
            for _ in range(num_experts)
        ])
        self.up_proj = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False)
            for _ in range(num_experts)
        ])
        self.down_proj = nn.ModuleList([
            nn.Linear(intermediate_size, hidden_size, bias=False)
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor, update_bias: bool = True):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            update_bias: 是否更新负载均衡 bias (训练时 True, 推理时 False)
        """
        batch, seq_len, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)

        # Router scores → apply aux-loss-free bias → top-k experts
        router_logits = self.router(flat)  # (B*S, num_experts)
        adjusted_logits = self.aux_free_router(router_logits)

        # Top-k experts
        topk_weights, topk_indices = torch.topk(adjusted_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)  # normalize over top-k

        if self.training and update_bias:
            self.aux_free_router.update_bias(topk_indices)

        # Compute expert outputs efficiently
        output = torch.zeros_like(flat)
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            mask = (topk_indices == expert_idx).any(dim=-1)  # (B*S,)
            if not mask.any():
                continue

            token_indices = mask.nonzero().squeeze(-1)
            expert_input = flat[token_indices]

            # Expert FFN: SiLU(gate(x)) * up(x) -> down
            gate_out = F.silu(self.gate_proj[expert_idx](expert_input))
            up_out = self.up_proj[expert_idx](expert_input)
            expert_out = self.down_proj[expert_idx](gate_out * up_out)

            # Weighted contribution
            tok_weights = topk_weights[token_indices]
            tok_k_idx = (topk_indices[token_indices] == expert_idx).float()
            tok_weights = (tok_weights * tok_k_idx).sum(dim=-1)

            output[token_indices] += expert_out * tok_weights.unsqueeze(-1)

        return output.view(batch, seq_len, hidden)


# ============================================================
# Technique 3: Iterative Alignment (OpenAI)
# ============================================================

class IterativeAlignment:
    """
    迭代对齐 — OpenAI 的核心方法论。

    不是一次性对齐, 而是多轮迭代:
      Round 1: 用人类偏好训练 DPO/RLHF
              → 生成新版模型
      Round 2: 用 Round 1 的模型生成更高质量的数据
              → 从生成结果中筛选最佳
              → 再次训练 (DPO/RLHF)
      Round 3+: 重复, 每轮模型都比上一轮更好

    为什么有效:
      - 每轮生成的数据质量更高 → 训练目标更高
      - 模型不断从"更好的自己"学习 → 自我提升
      - 避免初始数据的分布偏差

    数据集关系:
      Round 0: 人工标注 → 训练 v0 → generate → filter → Round 1 训练数据
      Round 1: (v0 生成的好数据) → 训练 v1 → generate → filter → Round 2 训练数据
      ...

    用法:
        aligner = IterativeAlignment(model, ref_model, tokenizer)

        # 每轮迭代
        for round_idx in range(3):
            data = aligner.generate_and_filter(prompts)
            aligner.train_one_round(data, method="dpo")
            aligner.save(f"model_round_{round_idx+1}")
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        num_candidates_per_prompt: int = 8,
        keep_best_ratio: float = 0.25,
        quality_threshold: float = 0.6,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates_per_prompt
        self.keep_best_ratio = keep_best_ratio
        self.quality_threshold = quality_threshold
        self.device = next(model.parameters()).device

        self._round = 0
        self._history: List[Dict] = []

    @torch.no_grad()
    def generate_and_filter(
        self,
        prompts: List[str],
        reward_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        生成候选 → 打分 → 筛选, 产生下一轮训练数据。

        返回: [
          {"prompt": "...", "chosen": "更好的回复", "rejected": "更差的回复", "score": 0.85},
          ...
        ]
        """
        from .FrontierAlign import RejectionSampling, ProcessRewardModel

        rs = RejectionSampling(self.model, self.tokenizer)
        prm = ProcessRewardModel(self.model, self.tokenizer)
        data = []

        for prompt in prompts:
            # Step 1: 生成多样候选
            candidates = rs.generate_candidates(prompt)

            if len(candidates) < 2:
                continue

            # Step 2: 打分 (PRM + 规则)
            scored = []
            for cand in candidates:
                steps = prm.split_into_steps(cand)
                step_scores = prm.score_steps_by_rules(prompt, steps)
                avg_score = prm.aggregate_scores(step_scores, method="min")

                if reward_fn:
                    avg_score += reward_fn(prompt, cand)
                    avg_score /= 2

                scored.append({"response": cand, "score": avg_score})

            scored.sort(key=lambda x: x["score"], reverse=True)

            # Step 3: 选 chosen (最好) 和 rejected (最差)
            if scored[0]["score"] >= self.quality_threshold:
                data.append({
                    "prompt": prompt,
                    "chosen": scored[0]["response"],
                    "rejected": scored[-1]["response"],
                    "score": scored[0]["score"],
                    "margin": scored[0]["score"] - scored[-1]["score"],
                })

        # 只保留 top keep_best_ratio
        data.sort(key=lambda x: x["margin"], reverse=True)
        keep_n = max(10, int(len(data) * self.keep_best_ratio))
        data = data[:keep_n]

        logger.info(f"Round {self._round+1}: 生成 {len(prompts)} prompts → "
                     f"筛选出 {len(data)} 对训练数据 (margin={data[0]['margin'] if data else 0:.2f})")
        return data

    def train_one_round(
        self,
        preference_data: List[Dict],
        method: str = "dpo",
        learning_rate: float = 5e-6,
        epochs: int = 1,
        **kwargs,
    ):
        """
        用筛选出的偏好数据训练一轮。

        method:
          - "dpo": 直接用 preference pairs (推荐)
          - "grpo": 用 reward 函数 + 组内比较
          - "sft_on_chosen": 只用 chosen 做 SFT
        """
        self._round += 1
        logger.info(f"=== Iterative Alignment Round {self._round} ===")
        logger.info(f"  数据: {len(preference_data)} pairs, method={method}")

        if method == "dpo":
            self._train_dpo(preference_data, learning_rate, epochs, **kwargs)
        elif method == "grpo":
            self._train_grpo(preference_data, **kwargs)
        elif method == "sft_on_chosen":
            self._train_sft(preference_data, learning_rate, epochs, **kwargs)

        # 更新 ref model 为当前 model
        self.ref_model.load_state_dict(self.model.state_dict())

        metrics = {
            "round": self._round,
            "method": method,
            "num_pairs": len(preference_data),
            "avg_margin": sum(d.get("margin", 0) for d in preference_data) / max(1, len(preference_data)),
        }
        self._history.append(metrics)
        return metrics

    def _train_dpo(self, data, lr, epochs, beta=0.1):
        """简化版 DPO 训练。"""
        from transformers import get_scheduler

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=10,
                                   num_training_steps=len(data) * epochs)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self._batchify(data, batch_size=2):
                loss = self._dpo_loss(batch, beta)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            logger.info(f"  DPO epoch {epoch+1}/{epochs}: loss={total_loss/len(data):.4f}")

    def _dpo_loss(self, batch: List[Dict], beta: float) -> torch.Tensor:
        """计算 DPO loss。"""
        total_loss = 0.0
        for d in batch:
            prompt = d["prompt"]
            chosen = d["chosen"]
            rejected = d["rejected"]

            # Encode
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected

            chosen_tokens = self.tokenizer(chosen_text, return_tensors="pt",
                                           truncation=True, max_length=2048).to(self.device)
            rejected_tokens = self.tokenizer(rejected_text, return_tensors="pt",
                                             truncation=True, max_length=2048).to(self.device)

            # Log-probs from policy
            with torch.no_grad():
                ref_chosen = self.ref_model(**chosen_tokens).logits
                ref_rejected = self.ref_model(**rejected_tokens).logits

            policy_chosen = self.model(**chosen_tokens).logits
            policy_rejected = self.model(**rejected_tokens).logits

            # Sequence-level log-probs
            logp_chosen = self._seq_log_prob(policy_chosen, chosen_tokens["input_ids"])
            logp_rejected = self._seq_log_prob(policy_rejected, rejected_tokens["input_ids"])
            ref_logp_chosen = self._seq_log_prob(ref_chosen, chosen_tokens["input_ids"])
            ref_logp_rejected = self._seq_log_prob(ref_rejected, rejected_tokens["input_ids"])

            # DPO loss: -log(sigma(beta * (logp_chosen - logp_rejected - ref_logp_chosen + ref_logp_rejected)))
            log_ratio = (logp_chosen - logp_rejected) - (ref_logp_chosen - ref_logp_rejected)
            total_loss += -F.logsigmoid(beta * log_ratio)

        return total_loss / len(batch)

    def _seq_log_prob(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """序列 log-probability。"""
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1).sum()

    def _train_grpo(self, data, **kwargs):
        from .GRPOTrainer import GRPOTrainer, GRPOConfig
        trainer = GRPOTrainer(self.model, self.ref_model, self.tokenizer, GRPOConfig(**kwargs))
        prompts = [d["prompt"] for d in data]
        trainer.train(prompts, reward_fn=lambda p, r: 0.5, num_steps=len(prompts))

    def _train_sft(self, data, lr, epochs):
        from transformers import get_scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for batch in self._batchify(data, batch_size=4):
                inputs = self.tokenizer(
                    [d["prompt"] + d["chosen"] for d in batch],
                    return_tensors="pt", truncation=True, max_length=2048, padding=True,
                ).to(self.device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            logger.info(f"  SFT epoch {epoch+1}/{epochs} done")

    def _batchify(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Round {self._round} model saved to {path}")

    def get_history(self) -> List[Dict]:
        return self._history


# ============================================================
# Technique 4: Test-time Compute Scaling (OpenAI o1/o3)
# ============================================================

class TestTimeCompute:
    """
    推理时计算缩放 — OpenAI o1/o3 的核心方法。

    核心思想: 不在训练时用更多 GPU, 而是在推理时分配更多"思考时间"。
    简单问题秒回, 复杂问题深度思考。

    实现策略 (从简到难):
      1. Chain-of-Thought: 强制模型写推理步骤
      2. Self-Refinement: 生成 → 自我检查 → 修订
      3. Best-of-N: 生成 N 个 → 选最好的
      4. Beam Search over Thoughts: 对推理步骤做束搜索
      5. Tree of Thoughts: 分支探索多种推理路径

    OpenAI o1: 内部 CoT + 推理时搜索 + 验证器
    OpenAI o3: o1 + 程序化推理 + 迭代改进

    用法:
        ttc = TestTimeCompute(model, tokenizer)

        # Level 1: 标准 CoT (快)
        answer = ttc.solve("x² + 5x + 6 = 0", level=1)

        # Level 3: Best-of-N (中等)
        answer = ttc.solve("证明根号2是无理数", level=3)

        # Level 5: 多轮自我修订 (慢但最强)
        answer = ttc.solve("实现一个红黑树", level=5)
    """

    LEVEL_CONFIGS = {
        1: {"name": "Chain-of-Thought", "num_samples": 1, "refine_rounds": 0, "verify": False},
        2: {"name": "CoT + Self-Check", "num_samples": 1, "refine_rounds": 1, "verify": True},
        3: {"name": "Best-of-N", "num_samples": 5, "refine_rounds": 0, "verify": True},
        4: {"name": "Best-of-N + 修订", "num_samples": 5, "refine_rounds": 2, "verify": True},
        5: {"name": "多轮探索 + 验证 + 修订", "num_samples": 10, "refine_rounds": 3, "verify": True},
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def solve(self, problem: str, level: int = 3, custom_verify_fn: Callable = None) -> Dict:
        """
        推理时计算缩放求解。

        返回:
          {
            "answer": "...",
            "reasoning": "...",
            "level": 3,
            "method": "Best-of-N",
            "compute_used": {"num_samples": 5, "refine_rounds": 0},
            "candidates_considered": [...],
          }
        """
        config = self.LEVEL_CONFIGS.get(level, self.LEVEL_CONFIGS[3])
        logger.info(f"TestTimeCompute Lv{level} ({config['name']}): {problem[:50]}...")

        # Step 1: 生成 N 个候选 (带 CoT)
        candidates = []
        for _ in range(config["num_samples"]):
            cot_prompt = f"""{problem}

让我们一步一步思考。先分析问题的关键要素, 然后逐步推理, 最后给出最终答案。

推理过程:"""
            response = self._generate(cot_prompt, temperature=0.8 if config["num_samples"] > 1 else 0.3)
            candidates.append(self._parse_response(response))

        # Step 2: 如果启用验证, 让模型自我检查每条推理
        if config["verify"]:
            verified = []
            for cand in candidates:
                check_prompt = f"""请检查以下推理是否正确。

问题: {problem}

推理: {cand["reasoning"]}

最终答案: {cand["answer"]}

这个推理过程:
1. 每一步的逻辑是否正确?
2. 计算是否准确?
3. 最终答案是否合理?

请评分 (1-10) 并指出任何问题。分数:"""
                check_result = self._generate(check_prompt, temperature=0.3)
                score = self._extract_score(check_result)
                cand["verification_score"] = score
                cand["verification_notes"] = check_result
                verified.append(cand)
            candidates = verified

        # Step 3: 选最好 + 修订
        candidates.sort(key=lambda x: x.get("verification_score", 0), reverse=True)
        best = candidates[0]

        for round_idx in range(config["refine_rounds"]):
            if round_idx == 0 and best.get("verification_score", 10) >= 9:
                break  # 已经很好, 不需要修订

            refine_prompt = f"""请改进以下解答。

问题: {problem}

当前解答:
推理: {best["reasoning"]}
答案: {best["answer"]}

批评意见: {best.get("verification_notes", "需要更好的解释")}

请给出改进后的推理和答案:"""
            refined = self._generate(refine_prompt, temperature=0.5)
            refined_parsed = self._parse_response(refined)
            refined_parsed["refine_round"] = round_idx + 1

            best = refined_parsed

        return {
            "answer": best["answer"],
            "reasoning": best.get("reasoning", ""),
            "level": level,
            "method": config["name"],
            "compute_used": {"num_samples": config["num_samples"], "refine_rounds": config["refine_rounds"]},
            "verification_score": best.get("verification_score", "N/A"),
            "total_candidates": len(candidates),
        }

    def _parse_response(self, text: str) -> Dict:
        """从模型输出中分离推理和答案。"""
        answer_markers = ["答案", "最终答案", "answer", "conclusion", "综上所述", "因此"]
        reasoning = text
        answer = text

        for marker in answer_markers:
            if marker in text:
                idx = text.rfind(marker)
                reasoning = text[:idx].strip()
                answer = text[idx:].strip()
                break

        return {"reasoning": reasoning, "answer": answer}

    def _extract_score(self, text: str) -> int:
        """从文本中提取数字评分。"""
        import re
        match = re.search(r"(\d+)\s*[分/]", text)
        if match:
            return min(10, max(1, int(match.group(1))))
        # 寻找行首的分数
        match = re.search(r"^(?:分数|评分|score)[:：]?\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if match:
            return min(10, max(1, int(match.group(1))))
        return 7  # 默认中等

    def _generate(self, prompt: str, temperature: float = 0.3) -> str:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=2048).to(self.device)
        prompt_len = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=temperature,
            do_sample=temperature > 0.1,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full[prompt_len:].strip()


# ============================================================
# Technique 5: RLAIF — RL from AI Feedback (Anthropic)
# ============================================================

class RLAIF:
    """
    RLAIF — Anthropic 最大的方法论贡献。

    传统 RLHF: 人类标注偏好 → 训练 Reward Model → PPO
    RLAIF:     AI 标注偏好 → 训练 Reward Model → PPO

    核心创新: 用一个更强的"裁判模型"代替人工标注。
    裁判模型根据 Constitution (宪法规则) 判断哪个回复更好。
    → 完全不需要人工标注 → 可以无限扩展

    论文: "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)
    """

    def __init__(
        self,
        policy_model,        # 被训练的策略模型
        judge_model=None,    # 裁判模型 (可以是同一个或更强的模型)
        tokenizer=None,
        constitution: List[str] = None,
    ):
        self.policy_model = policy_model
        self.judge_model = judge_model or policy_model  # 默认用同一个
        self.tokenizer = tokenizer
        self.device = next(policy_model.parameters()).device

        # 默认宪法规则 (Anthropic 风格的 HHH 原则)
        self.constitution = constitution or [
            "请选择更有帮助的回复。好的回复应该准确、完整、直接解决用户的问题。",
            "请选择更诚实的回复。如果回复承认不确定性或提供准确信息, 优先选择。",
            "请选择更无害的回复。避免鼓励非法、危险或不道德的行为。",
            "请选择更清晰的回复。好的回复结构清晰、语言流畅、容易被理解。",
            "请选择更具体的回复。具体例子和详细说明比模糊概括更有价值。",
            "请选择更公平的回复。避免对任何群体的偏见或歧视, 对所有人群尊重。",
        ]

    def judge_pair(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> Dict:
        """
        AI 裁判判断两个回复哪个更好。

        返回: {"winner": "A"|"B"|"tie", "reasoning": "...", "confidence": 0.8}
        """
        # 随机选几条宪法规则
        rules = random.sample(self.constitution, min(3, len(self.constitution)))

        judge_prompt = f"""你是一个严格的 AI 回复质量评审专家。请根据以下原则判断两个回复哪个更好。

评审原则:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(rules))}

用户问题: {prompt}

回复 A:
{response_a[:1500]}

回复 B:
{response_b[:1500]}

请判断:
1. 哪个回复更好? (A / B / 平局)
2. 为什么? (简要说明)
3. 你的把握有多大? (1-10)

输出格式:
选择: A/B/平局
原因: ...
把握: X/10"""

        judgment = self._generate(judge_prompt, temperature=0.3)

        winner = "tie"
        if "A" in judgment[:10] or "回复 A" in judgment[:50]:
            winner = "A"
        elif "B" in judgment[:10] or "回复 B" in judgment[:50]:
            winner = "B"

        confidence = self._extract_confidence(judgment)

        return {"winner": winner, "reasoning": judgment, "confidence": confidence}

    def generate_preference_data(
        self,
        prompts: List[str],
        num_pairs_per_prompt: int = 4,
    ) -> List[Dict]:
        """
        自动生成偏好对训练数据 — 完全不需要人工。

        流程:
          1. Policy model 对每个 prompt 生成多个回复
          2. Judge model 根据 Constitution 成对比较
          3. 输出 (prompt, chosen, rejected) 三元组
        """
        from .FrontierAlign import RejectionSampling
        rs = RejectionSampling(self.policy_model, self.tokenizer)
        data = []

        for prompt in prompts:
            # 生成候选
            candidates = rs.generate_candidates(prompt)

            if len(candidates) < 2:
                continue

            # AI 裁判排序 (简化版: 用 CLS 式比较)
            scored = []
            for cand in candidates:
                # 简单评分: 长度 + 结构
                score = 5.0
                if len(cand) > 100: score += 0.5
                if len(cand) > 500: score += 0.5
                if any(m in cand for m in ["首先", "其次", "因为", "所以", "1.", "2."]):
                    score += 0.5
                scored.append({"response": cand, "score": score})

            scored.sort(key=lambda x: x["score"], reverse=True)

            # 成对比较: 最好的 vs 最差的
            chosen = scored[0]["response"]
            rejected = scored[-1]["response"]

            # 用 judge model 验证
            judgment = self.judge_pair(prompt, chosen, rejected)

            if judgment["winner"] in ("A", "tie"):
                data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "judge_confidence": judgment["confidence"],
                    "source": "rlaif",
                })

        logger.info(f"RLAIF: {len(prompts)} prompts → {len(data)} preference pairs")
        return data

    def _generate(self, prompt: str, temperature: float = 0.3) -> str:
        import torch
        self.judge_model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=2048).to(self.device)
        with torch.no_grad():
            outputs = self.judge_model.generate(
                **inputs, max_new_tokens=512, temperature=temperature,
                do_sample=temperature > 0.1, top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_len = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        return full[prompt_len:].strip()

    def _extract_confidence(self, text: str) -> float:
        import re
        match = re.search(r"(\d+)\s*[/分10]", text)
        if match:
            return min(1.0, int(match.group(1)) / 10.0)
        return 0.7


# ============================================================
# Technique 6: SLURP Pretraining (Anthropic)
# ============================================================

class SLURPMixer:
    """
    SLURP — Anthropic 的预训练数据创新。

    传统预训练: 只用纯文本 (维基、网页等)
    SLURP:      预训练阶段就混入少量高质量的指令/对话数据

    为什么有效:
      - 模型在早期就看到"对话格式" → 后期 SFT 更容易
      - 减少 SFT 时的"灾难性遗忘"
      - 预训练阶段学到的基础能力 + 对话能力同时发展

    论文: "SLURP: Spoken Language Understanding Resource Package" / Anthropic 内部实践

    数据配比:
      纯文本     85-95%  (主体)
      指令数据     3-8%   (SFT式一问一答)
      对话数据     2-5%   (多轮对话)
      代码          0-5%   (结构化思维)

    用法:
        mixer = SLURPMixer(pretrain_corpus, instruct_data, chat_data)
        mixed_dataset = mixer.mix(ratios={"text": 0.90, "instruct": 0.06, "chat": 0.03, "code": 0.01})
    """

    DEFAULT_RATIOS = {
        "text": 0.90,       # 纯文本 (Wikipedia, C4, 书籍)
        "instruct": 0.05,   # 指令数据 (instruction-output pairs)
        "chat": 0.03,       # 多轮对话
        "code": 0.02,       # 代码
    }

    def __init__(
        self,
        text_data,
        instruct_data=None,
        chat_data=None,
        code_data=None,
    ):
        self.text_data = text_data
        self.instruct_data = instruct_data
        self.chat_data = chat_data
        self.code_data = code_data

    def mix(
        self,
        ratios: Dict[str, float] = None,
        seed: int = 42,
    ):
        """按配比混合数据。"""
        from datasets import interleave_datasets, Dataset

        ratios = ratios or self.DEFAULT_RATIOS

        datasets = []
        probabilities = []

        # 纯文本 → tokenized as CLM
        if self.text_data is not None and ratios.get("text", 0) > 0:
            datasets.append(self.text_data)
            probabilities.append(ratios["text"])

        # 指令数据 → 格式化为 "问: X\n答: Y" 再 tokenize
        if self.instruct_data is not None and ratios.get("instruct", 0) > 0:
            formatted = self._format_instruct(self.instruct_data)
            datasets.append(formatted)
            probabilities.append(ratios["instruct"])

        # 对话数据 → 格式化为多轮对话
        if self.chat_data is not None and ratios.get("chat", 0) > 0:
            formatted = self._format_chat(self.chat_data)
            datasets.append(formatted)
            probabilities.append(ratios["chat"])

        # 代码
        if self.code_data is not None and ratios.get("code", 0) > 0:
            datasets.append(self.code_data)
            probabilities.append(ratios["code"])

        if not datasets:
            raise ValueError("至少需要一种数据")

        # 归一化
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        logger.info(f"SLURP mixing: {dict(zip(ratios.keys(), probabilities))}")

        mixed = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

        return mixed

    def _format_instruct(self, data):
        """把 instruction-output 对格式化为预训练文本。"""
        def _format(example):
            instr = example.get("instruction", example.get("prompt", ""))
            out = example.get("output", example.get("response", ""))
            example["text"] = f"问: {instr}\n\n答: {out}"
            return example
        return data.map(_format)

    def _format_chat(self, data):
        """把多轮对话格式化为预训练文本。"""
        def _format(example):
            messages = example.get("messages", [])
            text_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prefix = "用户" if role == "user" else "助手"
                text_parts.append(f"{prefix}: {content}")
            example["text"] = "\n".join(text_parts)
            return example
        return data.map(_format)


# ============================================================
# Technique 7: Long Context RoPE Scaling (Anthropic Claude)
# ============================================================

class RoPEScaling:
    """
    RoPE 位置编码扩展 — Claude 200K 长上下文的实现方法。

    Anthropic 让 Claude 支持超长上下文的核心技巧:
      1. 预训练时用基础 RoPE (theta=10000)
      2. 微调时逐步增大 RoPE theta 或使用 NTK-aware 插值
      3. 对注意力做窗口化 + 全局 token

    三种扩展方法:
      - NTK-aware: 增大 theta 同时缩放高频分量
      - YaRN: NTK + 温度调整 (推荐)
      - Linear: 简单线性插值

    用法:
        # 将 Llama 模型从 4K 扩展到 32K
        scaled_config = RoPEScaling.ntk_aware(model.config, target_length=32768)
    """

    @staticmethod
    def ntk_aware(
        config,               # HuggingFace config (LlamaConfig)
        target_length: int,
        original_length: Optional[int] = None,
        alpha: Optional[float] = None,
    ):
        """
        NTK-aware RoPE 扩展。

        原理: 增大 theta → 降低高频旋转速度 → 支持更长上下文
        theta_new = theta_old * (target_length / original_length) ^ (dim / (dim - 2))
        """
        original = original_length or config.max_position_embeddings
        if alpha is None:
            dim = config.hidden_size // config.num_attention_heads
            alpha = (target_length / original) ** (dim / (dim - 2))

        new_theta = config.rope_theta * alpha
        config.max_position_embeddings = target_length
        config.rope_theta = new_theta

        logger.info(f"NTK-aware RoPE: {original} → {target_length}, "
                     f"theta={config.rope_theta:.0f}")
        return config

    @staticmethod
    def yarn(
        config,
        target_length: int,
        original_length: Optional[int] = None,
        temperature: float = 1.0,  # YaRN 的温度参数, 越小越保守
    ):
        """
        YaRN (Yet another RoPE extensioN) — 目前最推荐的 RoPE 扩展方法。

        比 NTK-aware 更好的原因是增加了温度控制,避免注意力熵崩塌。
        """
        original = original_length or config.max_position_embeddings
        dim = config.hidden_size // config.num_attention_heads

        # NTK 部分
        scale = target_length / original
        alpha = scale ** (dim / (dim - 2))
        new_theta = config.rope_theta * alpha

        # YaRN 温度调整
        # 对低频分量降低"旋转速度", 使其不那么容易互相干扰
        if temperature != 1.0:
            new_theta *= temperature

        config.max_position_embeddings = target_length
        config.rope_theta = new_theta

        logger.info(f"YaRN RoPE: {original} → {target_length}, "
                     f"theta={config.rope_theta:.0f}, temp={temperature}")
        return config

    @staticmethod
    def linear(config, target_length: int, original_length: Optional[int] = None):
        """
        线性 RoPE 插值 (最简单, 但效果通常不如 NTK/YaRN)。
        """
        original = original_length or config.max_position_embeddings
        scale = target_length / original
        config.max_position_embeddings = target_length
        config.rope_theta *= scale
        return config


# ============================================================
# Technique 8: Sparse Autoencoder — 理解模型内部 (Anthropic)
# ============================================================

class SparseAutoencoder(nn.Module):
    """
    稀疏自编码器 (SAE) — Anthropic Mechanistic Interpretability 的核心工具。

    作用: 把模型的激活值分解为可解释的特征。

    为什么需要 SAE:
      - 模型的隐藏层激活值 (4096维) 是稠密的, 难以解释
      - SAE 将其展开为更大的稀疏空间 (如 32768 维), 大多数为 0
      - 每个非零维度对应一个"概念" → 可解释!

    Anthropic 的 "Golden Gate Claude" 实验:
      在 SAE 中发现一个专门识别"金门大桥"的神经元
      → 放大该特征 → Claude 在每句话中都会提到金门大桥

    论文: "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
          (Anthropic, 2023)

    用法:
        sae = SparseAutoencoder(input_dim=4096, hidden_dim=32768)
        sae.train_on_activations(activations_from_model)

        # 分析: 某个特征在什么时候激活?
        feature_acts = sae.encode(model_activation)
        top_features = feature_acts.topk(5)  # 最强的 5 个激活特征
    """

    def __init__(self, input_dim: int, hidden_dim: int, l1_coefficient: float = 0.001):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.l1_coefficient = l1_coefficient

        # 初始化: decoder 权重行归一化
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, input_dim) — 模型某层的激活值

        Returns:
            reconstructed: (batch, input_dim)
            features: (batch, hidden_dim) — 稀疏特征表示
        """
        # Encode → ReLU (非负) → 稀疏
        encoded = F.relu(self.encoder(x))
        features = encoded

        # Decode → 重建
        reconstructed = self.decoder(features)

        # Loss: MSE(重建) + L1(稀疏)
        mse_loss = F.mse_loss(reconstructed, x)
        l1_loss = self.l1_coefficient * features.abs().mean()

        return reconstructed, features, mse_loss + l1_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """将激活值编码为稀疏特征。"""
        return F.relu(self.encoder(x))

    @torch.no_grad()
    def get_top_features(self, x: torch.Tensor, top_k: int = 10) -> Dict:
        """找到激活最强的特征。"""
        features = self.encode(x)
        if features.dim() > 1:
            features = features.mean(dim=0)
        values, indices = torch.topk(features, top_k)
        return {
            "indices": indices.tolist(),
            "activations": values.tolist(),
            "sparsity": (features > 0).float().mean().item(),
        }

    def train_on_activations(
        self,
        activations: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
    ):
        """
        在模型的激活值上训练 SAE。

        activations: (num_samples, input_dim) — 从模型中收集的隐藏层输出
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n = activations.shape[0]

        for epoch in range(epochs):
            perm = torch.randperm(n)
            total_loss = 0.0

            for i in range(0, n, batch_size):
                batch = activations[perm[i:i+batch_size]]
                _, _, loss = self.forward(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logger.info(f"SAE epoch {epoch+1}/{epochs}: loss={total_loss/ (n/batch_size):.4f}")

        logger.info("SAE 训练完成")


def analyze_model_features(model, tokenizer, text: str, layer_idx: int = -1):
    """
    用 SAE 分析模型在处理特定文本时激活了哪些概念。

    Anthropic 风格的可解释性分析。

    用法:
        analysis = analyze_model_features(model, tokenizer, "金门大桥在旧金山")
        print(analysis["top_features"])
    """
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output.detach().cpu())

    # 注册 hook
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if layers:
        target_layer = layers[layer_idx]
        handle = target_layer.register_forward_hook(hook_fn)

    # 前向
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    if not activations:
        return {"error": "未捕获到激活值"}

    return {
        "text": text,
        "layer": layer_idx,
        "activation_shape": list(activations[0].shape),
        "mean_activation": activations[0].mean().item(),
        "max_activation": activations[0].max().item(),
        "sparsity": (activations[0] == 0).float().mean().item(),
    }
