"""
Dense 模型预训练器 — 实战级训练稳定性控制

Dense 架构训练要点 (与 MoE 的区别):
  1. 所有参数每个 token 都激活 → 训练更简单但显存线性增长
  2. 不需要负载均衡 → 收敛更稳定
  3. 每 token FLOPs 确定 → MFU 更准确
  4. 关键超参: LR warmup + cosine decay, batch size scaling, weight init
"""
import os
import time
import math
import json
import logging
import random
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ============================================================
# Weight Initialization (训练稳定性的第一关)
# ============================================================

def init_weights_llama_style(model: nn.Module, initializer_range: float = 0.02):
    """
    Llama 风格的初始化 —— 对训练 from-scratch 的稳定性至关重要。

    - Embedding: N(0, 1/sqrt(d))
    - Linear: N(0, 0.02)  (小方差避免早期梯度爆炸)
    - LayerNorm/RMSNorm: weight=1, bias=0
    - 特殊处理: 残差层的最后投影初始化为 0 (减少早期方差累积)
    """
    for name, param in model.named_parameters():
        if "embed" in name:
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(param.shape[1]))
            else:
                nn.init.normal_(param, mean=0.0, std=initializer_range)
        elif "norm" in name:
            if param.dim() >= 1:
                nn.init.constant_(param, 1.0)  # weight
        elif "lm_head" in name:
            nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(param.shape[1]))
        elif "o_proj" in name or "down_proj" in name:
            # 残差路径最后一层: 初始化为 0 减少早期方差累积
            # (这是 Llama 2/3 的关键 trick，防止训练初期梯度爆炸)
            nn.init.zeros_(param)
        elif "weight" in name and param.dim() >= 2:
            nn.init.normal_(param, mean=0.0, std=initializer_range)
        elif "bias" in name and param.dim() >= 1:
            nn.init.zeros_(param)

    logger.info(f"模型初始化完成 (Llama-style, range={initializer_range})")


def init_weights_small_embed(model: nn.Module, initializer_range: float = 0.02):
    """
    对小模型的深度缩放初始化。

    小模型 (< 500M params) 需要更小的初始化方差，
    因为它们更容易梯度爆炸 (层数少 → 残差累积快)。
    """
    for name, param in model.named_parameters():
        if "embed" in name:
            if param.dim() >= 2:
                # 更小的 embedding 初始化
                nn.init.normal_(param, mean=0.0, std=1.0 / param.shape[1])
        elif "norm" in name:
            if param.dim() >= 1:
                nn.init.constant_(param, 1.0)
        elif "o_proj" in name or "down_proj" in name:
            nn.init.zeros_(param)
        elif "weight" in name and param.dim() >= 2:
            nn.init.normal_(param, mean=0.0, std=initializer_range * 0.5)
        elif "bias" in name and param.dim() >= 1:
            nn.init.zeros_(param)

    logger.info(f"模型初始化完成 (Small-model scaled, range={initializer_range})")


def init_weights_deepseek_style(model: nn.Module):
    """
    DeepSeek 风格初始化。

    特点: 更大的 embedding 初始化 + RMSNorm 初始化为 0 而非 1
    (RMSNorm weight=0 可以让训练早期残差路径直接传递信息)
    """
    for name, param in model.named_parameters():
        if "embed" in name:
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(param.shape[1]))
        elif "norm" in name or "rms_norm" in name:
            if param.dim() >= 1:
                # DeepSeek V3: RMSNorm weight starts at 0
                nn.init.zeros_(param)
        elif "o_proj" in name or "down_proj" in name:
            nn.init.zeros_(param)
        elif "weight" in name and param.dim() >= 2:
            nn.init.normal_(param, mean=0.0, std=0.006)  # Smaller std for larger models
        elif "bias" in name and param.dim() >= 1:
            nn.init.zeros_(param)

    logger.info("模型初始化完成 (DeepSeek-style)")


# ============================================================
# DensePretrainer 配置
# ============================================================

@dataclass
class DensePretrainConfig:
    # 数据
    max_seq_length: int = 2048
    global_batch_tokens: int = 524288    # 全局每步 tokens (~512K)

    # 学习率 (最关键的超参)
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5     # cosine 退火的最小值
    warmup_steps: int = 2000
    lr_schedule: str = "cosine"          # cosine | linear | constant
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8

    # 梯度
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # 训练稳定性
    loss_spike_detection: bool = True
    loss_spike_threshold: float = 3.0    # loss > moving_avg * threshold 视为 spike
    loss_spike_action: str = "skip"      # skip | rollback | warn
    loss_moving_avg_window: int = 100    # 滑动窗口大小
    grad_norm_log_interval: int = 100    # 梯度范数日志间隔

    # 混合精度
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True

    # Flash Attention
    use_flash_attention: bool = True     # 自动检测

    # 训练长度
    max_steps: int = 100000
    save_every_steps: int = 5000
    eval_every_steps: int = 1000
    log_every_steps: int = 50

    # 验证
    val_max_steps: int = 100             # 每次验证多少步
    val_perplexity: bool = True

    # 输出
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    resume: bool = True
    seed: int = 42

    # 优化
    gradient_checkpointing: bool = True
    compile_model: bool = False           # torch.compile (PyTorch 2.0+)
    activation_checkpointing_layers: int = 0  # 每多少层 checkpoint 一次, 0=全部


class DensePretrainer:
    """
    Dense 模型实战预训练器。

    解决 HuggingFace Trainer 没有覆盖的问题:
    - Loss spike 自动检测与恢复
    - 梯度统计 + 逐层梯度范数
    - 训练稳定性监控
    - 自适应 MFU 追踪
    - 可复现的数据 shuffle

    用法:
        config = DensePretrainConfig(
            max_seq_length=2048,
            learning_rate=3e-4,
            max_steps=100000,
        )
        trainer = DensePretrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            config=config,
        )

        # 自动初始化权重
        trainer.initialize_model(style="llama")

        trainer.train()

    Dense 模型各规模训练参考:
    ═══════════════════════════════════════════════════════════
      规模      batch_tokens   LR       warmup    weight_decay
    ───────────────────────────────────────────────────────────
      100M       256K         3e-4      2000      0.1
      300M       512K         3e-4      2000      0.1
      1B         1M           2e-4      3000      0.1
      3B         2M           1.5e-4    3000      0.1
      7B         4M           1e-4      5000      0.05
    ═══════════════════════════════════════════════════════════
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset,
        config: DensePretrainConfig = None,
        val_dataset = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or DensePretrainConfig()

        self.device = next(model.parameters()).device
        self._step = 0
        self._total_tokens = 0
        self._start_time = None

        # Loss spike 检测
        self._loss_history: List[float] = []
        self._loss_moving_avg = float("inf")

        # 梯度统计
        self._grad_norm_history: List[float] = []

        # MFU 追踪
        self._mfu_tracker = None

        # 优化器
        self.optimizer = None
        self.scheduler = None

        # 检测 Flash Attention
        self._has_flash_attn = self._detect_flash_attention()

        # 设置随机种子
        self._set_seed(self.config.seed)

        # 设置 TF32
        if self.config.tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def initialize_model(self, style: str = "llama"):
        """
        初始化模型权重 (必须在移动到 GPU 之后调用)。

        为什么需要手动 init:
        - HuggingFace 的 from_config() 默认 init 方差过大
        - 不正确的 init 导致训练初期 loss 不收敛或 NaN
        - 残差路径的 o_proj/down_proj 需要零初始化
        """
        if style == "llama":
            init_weights_llama_style(self.model)
        elif style == "small":
            init_weights_small_embed(self.model)
        elif style == "deepseek":
            init_weights_deepseek_style(self.model)
        else:
            logger.warning(f"未知初始化风格: {style}, 使用默认 PyTorch init")

    def train(self):
        """主训练循环"""
        config = self.config

        logger.info("=" * 60)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"DensePretrainer 开始训练")
        logger.info(f"  参数量: {total_params/1e9:.2f}B")
        logger.info(f"  序列长度: {config.max_seq_length}")
        logger.info(f"  全局 batch: {config.global_batch_tokens:,} tokens")
        logger.info(f"  学习率: {config.learning_rate} (warmup={config.warmup_steps})")
        logger.info(f"  Flash Attention: {self._has_flash_attn}")
        logger.info("=" * 60)

        # 1. 配置优化器
        self._setup_optimizer()

        # 2. 配置 DataLoader
        train_loader = self._setup_dataloader()

        # 3. 配置 AMP
        scaler = None
        if config.fp16:
            scaler = torch.cuda.amp.GradScaler()

        # 4. 梯度检查点
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # 5. torch.compile (PyTorch 2.0+)
        if config.compile_model:
            try:
                self.model = torch.compile(self.model)
                logger.info("torch.compile 启用")
            except Exception as e:
                logger.warning(f"torch.compile 失败: {e}")

        # 6. MFU 追踪
        from .TrainingMonitor import TrainingMonitor
        self._mfu_tracker = TrainingMonitor(
            model_params=total_params,
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            seq_length=config.max_seq_length,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # 7. 恢复检查点
        if config.resume:
            self._load_checkpoint()

        # 8. 训练循环
        self.model.train()
        self._start_time = time.time()
        optimizer_step = 0

        train_iter = iter(train_loader)

        for step in range(config.max_steps):
            self._step = step

            # ---- 数据加载 ----
            try:
                batch = next(train_iter)
            except StopIteration:
                train_loader = self._setup_dataloader()
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Causal LM: labels = input_ids
            if "labels" not in batch:
                batch["labels"] = batch["input_ids"].clone()

            # ---- 前向 ----
            self._mfu_tracker.on_step_start()

            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else torch.float32,
            ):
                outputs = self.model(**batch)
                loss = outputs.loss / config.gradient_accumulation_steps

            # ---- 反向 ----
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # ---- Loss spike 检测 ----
            loss_val = loss.item() * config.gradient_accumulation_steps
            if config.loss_spike_detection and self._detect_loss_spike(loss_val):
                logger.warning(f"Loss spike detected at step {step}: {loss_val:.4f}")
                if config.loss_spike_action == "skip":
                    self.optimizer.zero_grad()
                    self._mfu_tracker.on_step_end(
                        batch["input_ids"].numel() * self.config.gradient_accumulation_steps
                    )
                    continue
                elif config.loss_spike_action == "rollback":
                    if self._load_checkpoint():
                        continue

            self._loss_history.append(loss_val)
            self._update_loss_moving_avg()

            # ---- 优化器步 ----
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.max_grad_norm
                )
                self._grad_norm_history.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

                # 更新
                if scaler is not None:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                optimizer_step += 1

            # ---- MFU 更新 ----
            batch_tokens = batch["input_ids"].numel()
            self._total_tokens += batch_tokens
            self._mfu_tracker.on_step_end(batch_tokens)

            # ---- 日志 ----
            if (step + 1) % config.log_every_steps == 0:
                mfu_metrics = self._mfu_tracker.get_latest_metrics()
                lr = self.scheduler.get_last_lr()[0]
                self._log_step(step, loss_val, grad_norm if (step + 1) % config.gradient_accumulation_steps == 0 else None, lr, mfu_metrics)

            # ---- 验证 ----
            if self.val_dataset and (step + 1) % config.eval_every_steps == 0:
                self._run_validation(step)

            # ---- 保存 ----
            if (step + 1) % config.save_every_steps == 0:
                self._save_checkpoint(step)

        # 训练结束
        self._save_checkpoint("final")
        if self._mfu_tracker:
            self._mfu_tracker.summary()

    # ============================================================
    # 内部方法
    # ============================================================

    def _setup_optimizer(self):
        """配置 AdamW + Cosine LR Schedule"""
        config = self.config

        # Weight decay 不应用于 bias, norm, embedding
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "norm", "embed", "rms_norm", "layernorm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )

        # Cosine LR scheduler with linear warmup
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            # Cosine decay from 1 to min_lr_ratio
            progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
            min_ratio = config.min_learning_rate / config.learning_rate
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"优化器: AdamW lr={config.learning_rate}, wd={config.weight_decay}")

    def _setup_dataloader(self):
        """设置可复现的 DataLoader"""
        config = self.config
        from torch.utils.data import DataLoader
        import platform

        num_workers = 0 if platform.system() == "Windows" else 4

        def collate(batch):
            if isinstance(batch[0], dict):
                import torch
                keys = batch[0].keys()
                return {k: torch.stack([torch.tensor(x[k]) for x in batch]) for k in keys}
            elif isinstance(batch[0], list):
                import torch
                return torch.tensor(batch)
            return batch

        return DataLoader(
            self.train_dataset,
            batch_size=min(config.global_batch_tokens // config.max_seq_length, 32),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False if num_workers == 0 else True,
            drop_last=True,
            collate_fn=collate,
        )

    def _detect_loss_spike(self, loss_value: float) -> bool:
        """检测 loss spike: loss > moving_avg * threshold"""
        if len(self._loss_history) < 10:
            return False
        if self._loss_moving_avg > 100:  # 未初始化
            return False
        return loss_value > self._loss_moving_avg * self.config.loss_spike_threshold

    def _update_loss_moving_avg(self):
        if not self._loss_history:
            return
        window = self._loss_history[-self.config.loss_moving_avg_window:]
        self._loss_moving_avg = sum(window) / len(window)

    def _detect_flash_attention(self) -> bool:
        """检测 Flash Attention 是否可用"""
        try:
            # 检查 flash_attn
            import importlib.util
            if importlib.util.find_spec("flash_attn"):
                from flash_attn import flash_attn_func
                logger.info("Flash Attention 2 可用")
                return True
        except Exception:
            pass

        try:
            # 检查 PyTorch 2.0+ 内置 SDPA
            if hasattr(F, "scaled_dot_product_attention"):
                logger.info("使用 PyTorch SDPA (torch 2.0+)")
                return True
        except Exception:
            pass

        logger.info("Flash Attention 不可用,使用标准 Attention")
        return False

    def _run_validation(self, step: int):
        """运行验证: 计算 perplexity"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(self.val_dataset):
                if i >= self.config.val_max_steps:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item() * batch["input_ids"].numel()
                total_tokens += batch["input_ids"].numel()

        val_ppl = math.exp(total_loss / max(1, total_tokens))
        train_loss = self._loss_moving_avg if self._loss_history else float("inf")

        logger.info(f"[Val step={step}] train_loss={train_loss:.4f}, val_ppl={val_ppl:.2f}")

        self.model.train()

    def _log_step(self, step: int, loss: float, grad_norm, lr: float, mfu: Dict):
        """单行日志"""
        parts = [
            f"step={step}",
            f"loss={loss:.4f}",
            f"lr={lr:.2e}",
            f"MFU={mfu.get('mfu', 0):.1f}%",
            f"tok/s={mfu.get('tokens_per_sec', 0):.0f}",
        ]
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.2f}")
        logger.info(" | ".join(parts))

    def _save_checkpoint(self, step):
        """保存检查点"""
        save_dir = os.path.join(self.config.checkpoint_dir, f"step-{step}")
        os.makedirs(save_dir, exist_ok=True)

        # 模型
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # 优化器状态
        checkpoint = {
            "step": step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_moving_avg": self._loss_moving_avg,
            "total_tokens": self._total_tokens,
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, os.path.join(save_dir, "training_state.pt"))

        logger.info(f"检查点已保存: {save_dir}")

    def _load_checkpoint(self, checkpoint_dir: str = None) -> bool:
        """恢复训练状态"""
        ckpt_dir = checkpoint_dir or self.config.checkpoint_dir
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if not os.path.exists(state_path):
            return False

        checkpoint = torch.load(state_path, map_location=self.device)
        self._step = checkpoint["step"]
        self._loss_moving_avg = checkpoint.get("loss_moving_avg", float("inf"))
        self._total_tokens = checkpoint.get("total_tokens", 0)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"已恢复训练: step={self._step}, tokens={self._total_tokens/1e9:.1f}B")
        return True

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ============================================================
    # LR Range Test (学习率探测)
    # ============================================================

    def lr_range_test(
        self,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        steps: int = 500,
        smooth_window: int = 20,
    ) -> Dict:
        """
        学习率范围测试 (Leslie Smith 方法)。

        用于找到最优初始 LR：
        - 从小 LR 开始逐步增大
        - Loss 下降最快的位置 ≈ 最优 LR
        - Loss 开始反弹的位置 ≈ 过大 LR

        用法:
            result = trainer.lr_range_test()
            print(f"建议 LR: {result['recommended_lr']}")
        """
        self.model.train()
        lrs = []
        losses = []

        lr_factor = (max_lr / min_lr) ** (1.0 / steps)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=min_lr)
        current_lr = min_lr

        for step in range(steps):
            try:
                batch = next(iter(self._setup_dataloader()))
            except StopIteration:
                batch = next(iter(self._setup_dataloader()))
            batch = {k: v.to(self.device) for k, v in batch.items()}

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            outputs = self.model(**batch)
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            lrs.append(current_lr)
            losses.append(loss.item())
            current_lr *= lr_factor

        # 平滑 loss 曲线
        smoothed = []
        for i in range(len(losses)):
            start = max(0, i - smooth_window // 2)
            end = min(len(losses), i + smooth_window // 2)
            smoothed.append(sum(losses[start:end]) / (end - start))

        # 找 loss 下降最快的点
        min_loss = min(smoothed)
        min_idx = smoothed.index(min_loss)
        recommended = lrs[min_idx] / 10  # 经典建议: 取最优点 / 10

        logger.info(f"LR Range Test 完成: min_loss={min_loss:.4f} @ lr={lrs[min_idx]:.2e}")
        logger.info(f"建议初始 LR: {recommended:.2e}")

        return {
            "lrs": lrs,
            "losses": smoothed,
            "min_loss_lr": lrs[min_idx],
            "recommended_lr": recommended,
        }
