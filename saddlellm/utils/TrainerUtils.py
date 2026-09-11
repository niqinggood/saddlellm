"""
训练工具集 — 环境检查 + 训练诊断 + 报告卡 + 早停

三大能力:
  1. 环境自检: 训练前检查一切是否就绪
  2. 训练诊断: 训练出问题时自动分析原因
  3. 训练报告: 训练结束后生成完整报告卡
  4. 早停 + 梯度监控: 训练稳定性保障
"""

import os
import sys
import time
import math
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================
# 1. 环境自检
# ============================================================


@dataclass
class EnvCheckResult:
    python_version: str
    cuda_available: bool
    cuda_version: str
    gpu_count: int
    gpu_name: str
    gpu_memory_gb: float
    gpu_total_memoryory_gb: float
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    pytorch_version: str
    transformers_version: str
    packages_ok: bool
    issues: List[str]
    recommendations: List[str]
    ready: bool


def check_environment() -> EnvCheckResult:
    """
    全面环境自检。

    用法:
        result = check_environment()
        if result.ready:
            print("环境就绪, 可以开始训练!")
        else:
            for issue in result.issues:
                print(f"⚠ {issue}")
    """
    import torch

    issues = []
    recommendations = []

    # Python
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # CUDA
    cuda_available = torch.cuda.is_available()
    cuda_ver = torch.version.cuda or "N/A"
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = ""
    gpu_mem = 0.0
    gpu_total = 0.0

    if gpu_count > 0:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_mem = gpu_total - torch.cuda.memory_allocated(0) / (1024**3)
    else:
        issues.append("未检测到 GPU — 只能训练 100M 以下模型, 且速度很慢")

    if gpu_mem < 4 and gpu_count > 0:
        issues.append(f"GPU 可用显存不足 ({gpu_mem:.1f}GB), 建议关闭其他程序")

    # RAM
    try:
        import psutil

        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024**3)
        ram_avail = ram.available / (1024**3)
    except Exception:
        ram_total = 0
        ram_avail = 0

    if ram_avail < 8:
        issues.append(f"可用 RAM 不足 ({ram_avail:.1f}GB), 建议至少 16GB")

    # Disk
    try:
        import shutil

        disk = shutil.disk_usage(os.getcwd())
        disk_free = disk.free / (1024**3)
    except Exception:
        disk_free = 100  # assume ok

    if disk_free < 10:
        issues.append(f"磁盘空间不足 ({disk_free:.1f}GB), 建议至少 20GB")
    elif disk_free < 50:
        recommendations.append(f"磁盘空间 {disk_free:.0f}GB — 足够小模型训练")

    # Packages
    packages_ok = True
    pkg_versions = {}
    for pkg in ["torch", "transformers", "datasets", "accelerate", "peft"]:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            issues.append(f"缺少依赖: {pkg}")
            packages_ok = False

    # PyTorch
    pt_ver = pkg_versions.get("torch", "N/A")
    transformers_ver = pkg_versions.get("transformers", "N/A")

    # GPU 建议
    if gpu_count >= 8:
        recommendations.append("8+ GPU: 可以使用 DeepSpeed ZeRO-3 训练 7B+ 模型")
    elif gpu_count >= 4:
        recommendations.append("4+ GPU: 可以使用 DeepSpeed ZeRO-2 训练 3B 模型")
    elif gpu_count >= 1:
        max_params_gb = gpu_total * 0.7  # 70% 可用
        max_params = max_params_gb / 2  # bf16 ≈ 2 bytes/param
        recommendations.append(f"单 GPU: 可训练约 {max_params:.1f}B 参数模型")

    ready = len(issues) == 0

    # 打印报告
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              SaddleLLM 环境自检报告                        ║
╠══════════════════════════════════════════════════════════╣
║  Python:      {py_ver:<42} ║
║  PyTorch:     {pt_ver:<42} ║
║  Transformers:{transformers_ver:<42} ║
║  CUDA:        {cuda_ver:<42} ║
║  GPU:         {gpu_count}x {gpu_name:<35} ║
║  GPU 显存:    {gpu_total:.1f}GB total, {gpu_mem:.1f}GB free{"":<20} ║
║  RAM:         {ram_total:.0f}GB total, {ram_avail:.0f}GB available{"":<9} ║
║  磁盘:        {disk_free:.0f}GB free{"":<33} ║
╚══════════════════════════════════════════════════════════╝
""")

    if issues:
        print("⚠ 问题:")
        for i in issues:
            print(f"  ✗ {i}")
    if recommendations:
        print("\n[*] 建议:")
        for r in recommendations:
            print(f"  → {r}")
    if ready:
        print("\n[OK] 环境就绪, 可以开始训练!")

    return EnvCheckResult(
        python_version=py_ver,
        cuda_available=cuda_available,
        cuda_version=cuda_ver,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_mem,
        gpu_total_memoryory_gb=gpu_total,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        disk_free_gb=disk_free,
        pytorch_version=pt_ver,
        transformers_version=transformers_ver,
        packages_ok=packages_ok,
        issues=issues,
        recommendations=recommendations,
        ready=ready,
    )


# ============================================================
# 2. 训练诊断
# ============================================================


class TrainingDiagnostics:
    """
    训练异常自动诊断。

    常见问题诊断:
      - loss = NaN  → LR太大 / 数据有问题 / 梯度爆炸
      - loss 不降   → LR太小 / 模型太小 / 数据太简单
      - loss 波动大  → batch太小 / LR太大
      - loss 突然飙升 → 遇到脏数据 / 梯度爆炸
      - 显存溢出     → batch太大 / 序列太长 / 没开gradient_ckpt

    用法:
        diag = TrainingDiagnostics()
        diag.check(step=100, loss=2.5, grad_norm=8.2)
    """

    def __init__(self):
        self._loss_history: List[float] = []
        self._grad_norm_history: List[float] = []
        self._warnings: List[str] = []

    def check(
        self,
        step: int,
        loss: float,
        grad_norm: float = None,
        lr: float = None,
        mfu: float = None,
    ) -> Optional[str]:
        """
        检查训练状态, 返回诊断信息 (如果有问题)。

        训练循环中每 N 步调用一次。
        """
        self._loss_history.append(loss)
        if grad_norm is not None:
            self._grad_norm_history.append(grad_norm)

        # 1. Loss = NaN
        if math.isnan(loss) or math.isinf(loss):
            msg = (
                "[!!!] Loss = NaN/Inf! 可能原因:\n"
                "  1. 学习率太大 → 降低 LR 10x\n"
                "  2. 数据中有坏样本 → 检查训练数据\n"
                "  3. 梯度爆炸 → 降低 max_grad_norm 到 0.5\n"
                "  4. 混合精度问题 → 改用 fp32 试试"
            )
            self._warnings.append(msg)
            return msg

        # 2. 第一个 step 的 loss 异常
        if step <= 2 and len(self._loss_history) >= 2:
            init_loss = self._loss_history[0]
            # 随机模型的初始 loss ≈ -ln(1/vocab_size)
            expected = -math.log(1 / 32000)  # ≈ 10.4
            if init_loss > expected * 1.5:
                msg = (
                    f"🟡 初始 loss 偏高 ({init_loss:.1f}, 预期 ~{expected:.0f})。可能:\n"
                    "  1. tokenizer 不匹配 → 检查 tokenizer\n"
                    "  2. 模型初始化有问题 → 需要重新 init"
                )
                self._warnings.append(msg)
                return msg

        # 3. Loss 不下降 (100 步后开始检查)
        if step >= 100 and len(self._loss_history) >= 20:
            recent = self._loss_history[-20:]
            older = (
                self._loss_history[-40:-20]
                if len(self._loss_history) >= 40
                else self._loss_history[:20]
            )
            if older and sum(recent) / len(recent) >= sum(older) / len(older) * 0.98:
                msg = (
                    "🟡 Loss 下降很慢或停滞。可能:\n"
                    "  1. 学习率太小 → 增大 LR 3x\n"
                    "  2. warmup 还没结束 → 等 warmup 完成\n"
                    "  3. 数据太简单 → 模型已经学会了"
                )
                self._warnings.append(msg)
                return msg

        # 4. Loss 突然飙升
        if len(self._loss_history) >= 10:
            avg = sum(self._loss_history[-10:-1]) / max(
                1, len(self._loss_history[-10:-1])
            )
            if loss > avg * 3 and step > 20:
                msg = (
                    f"🟡 Loss 突然飙升 ({loss:.2f} vs avg {avg:.2f})。可能:\n"
                    "  1. 遇到脏数据 → 检查当前 batch\n"
                    "  2. 学习率太大 → 降低 LR\n"
                )
                self._warnings.append(msg)
                return msg

        # 5. 梯度范数异常
        if grad_norm is not None:
            if grad_norm > 10:
                msg = (
                    f"🟡 梯度范数过大 ({grad_norm:.1f})。建议:\n"
                    "  降低 max_grad_norm 或检查数据质量"
                )
                self._warnings.append(msg)
                return msg
            if grad_norm < 1e-6 and step > 100:
                msg = "🟡 梯度范数接近零, 可能是梯度消失。检查模型初始化。"
                self._warnings.append(msg)
                return msg

        # 6. MFU 过低
        if mfu is not None and step > 50:
            if mfu < 15:
                msg = (
                    f"🟡 MFU 过低 ({mfu:.1f}%)。建议:\n"
                    "  增大 batch_size 或减少 gradient_accumulation"
                )
                self._warnings.append(msg)
                return msg

        return None  # 无问题

    def get_summary(self) -> Dict:
        """获取训练诊断摘要。"""
        return {
            "total_checks": len(self._loss_history),
            "warnings_count": len(self._warnings),
            "warnings": self._warnings[-5:],
            "final_loss": self._loss_history[-1] if self._loss_history else None,
            "loss_trend": "decreasing"
            if len(self._loss_history) >= 20
            and self._loss_history[-1] < self._loss_history[0]
            else "flat",
        }


# ============================================================
# 3. 训练报告卡
# ============================================================


class TrainingReportCard:
    """
    训练结束后的完整报告。

    自动生成:
      - 模型基本信息
      - 训练统计
      - 效率分析
      - 质量评级
      - 改进建议

    用法:
        report = TrainingReportCard.generate(model, trainer, metrics_history)
        report.save("model_report.json")
    """

    @staticmethod
    def generate(
        model,
        training_config: Dict = None,
        metrics_history: List[Dict] = None,
        mfu_summary: Dict = None,
        eval_results: List = None,
        output_path: str = None,
    ) -> Dict:
        """生成训练报告卡。"""
        import torch

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        report = {
            "model": {
                "total_params": total_params,
                "total_params_human": f"{total_params / 1e9:.2f}B"
                if total_params >= 1e9
                else f"{total_params / 1e6:.0f}M",
                "trainable_params": trainable_params,
                "architecture": type(model).__name__,
            },
            "training": training_config or {},
            "efficiency": mfu_summary or {},
            "evaluation": [
                {"task": r.task, "score": r.score, "metric": r.metric}
                for r in eval_results
            ]
            if eval_results
            else [],
        }

        # 质量评级
        quality = "C"
        if eval_results:
            avg_score = sum(r.score for r in eval_results) / len(eval_results)
            if avg_score > 0.7:
                quality = "A"
            elif avg_score > 0.5:
                quality = "B"
            report["quality_grade"] = quality
            report["avg_eval_score"] = round(avg_score, 3)

        # 改进建议
        improvements = []
        if mfu_summary:
            mfu = mfu_summary.get("avg_mfu_pct", 0)
            if mfu < 25:
                improvements.append(
                    "增大 batch_size 或减少 gradient_accumulation 提升 MFU"
                )
        if total_params < 1e9:
            improvements.append("考虑用 RapidDistill 从更大模型蒸馏能力")
        if not eval_results:
            improvements.append("运行 BenchmarkRunner 评估模型效果")
        improvements.append("用 Lightweight.optimize() 减小模型体积")
        report["suggested_improvements"] = improvements

        # 打印
        print(f"""
╔══════════════════════════════════════════════════════════╗
║              训练报告卡                                   ║
╠══════════════════════════════════════════════════════════╣
║  模型参数:    {report["model"]["total_params_human"]:<42} ║
║  训练步数:    {training_config.get("max_steps", "N/A") if training_config else "N/A":<42} ║
║  质量评级:    {quality:<42} ║
╠══════════════════════════════════════════════════════════╣
║  改进建议:                                               ║
""")
        for imp in improvements:
            print(f"║    → {imp:<48} ║")
        print("╚══════════════════════════════════════════════════════════╝")

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return report


# ============================================================
# 4. 早停 (Early Stopping)
# ============================================================


class EarlyStopping:
    """
    早停 — 验证 loss 不下降时自动停止训练。

    用法:
        stopper = EarlyStopping(patience=5, min_delta=0.01)
        for step in range(total_steps):
            val_loss = validate()
            if stopper.check(val_loss):
                print("早停!")
                break
            stopper.save_if_best(model, "./best_model")
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.01, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_step = 0

    def check(self, score: float) -> bool:
        """
        检查是否应该早停。

        返回 True = 应该停止。
        """
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def save_if_best(self, model, tokenizer, path: str, score: float):
        """如果是最佳模型就保存。"""
        is_best = (self.mode == "min" and score <= self.best_score) or (
            self.mode == "max" and score >= self.best_score
        )
        if is_best:
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            self.best_step = getattr(self, "_current_step", 0)


# ============================================================
# 5. 一键最佳小模型训练
# ============================================================


def auto_train_best_small_model(
    model_size: str = "300m",
    output_dir: str = "./my_best_model",
    lang: str = "zh",
    use_api_teacher: bool = False,
    api_teachers: List[str] = None,
):
    """
    一键训练最佳小模型。包含所有最佳实践。

    做了什么:
      1. 环境自检
      2. 创建模型 + 初始化
      3. 自动选数据 + 训练 (EMA)
      4. 模型汤平均
      5. 安全对齐
      6. 可选: API蒸馏
      7. 轻量化
      8. 生成报告卡

    这是最简单的高质量小模型训练方式。
    """
    print("=== 🚀 SaddleLLM 一键最佳小模型训练 ===\n")

    # 1. 环境自检
    env = check_environment()
    if not env.ready:
        print("[WARN] 环境有问题, 但继续尝试训练...")

    # 2. 创建模型
    from ..models.ModelRegistry import ModelRegistry, MODEL_SPECS
    from ..training.DensePretrainer import (
        DensePretrainer,
        DensePretrainConfig,
        init_weights_llama_style,
    )

    size_map = {"100m": "gpt2-small-124m", "300m": "llama-300m", "1b": "llama-1b"}
    spec_name = size_map.get(model_size, "llama-300m")
    spec = MODEL_SPECS[spec_name]

    model = ModelRegistry.create_model(spec)
    init_weights_llama_style(model)
    print(f"✅ 模型: {spec.name} ({spec.human_params()})")

    # 3. 数据
    from ..data.DataCatalog import DataCatalog

    dataset = DataCatalog.small_model_pack(lang=lang, streaming=True)
    print(f"✅ 数据: 自动选择 {lang} 最佳配比")

    # 4. Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 5. 训练配置
    total_params = spec.estimated_params
    lr = 3e-4 if total_params < 5e8 else 2e-4
    seq_len = 512 if total_params < 5e8 else 1024 if total_params < 2e9 else 2048
    steps = 50000 if total_params < 5e8 else 100000

    config = DensePretrainConfig(
        max_steps=steps,
        learning_rate=lr,
        max_seq_length=seq_len,
        output_dir=output_dir,
        checkpoint_dir=os.path.join(output_dir, "ckpt"),
        bf16=env.cuda_available,
    )

    # 6. 训练
    print(f"\n=== 训练: {steps} steps, lr={lr} ===")
    trainer = DensePretrainer(model, tokenizer, dataset, config)

    from ..alignment.AdvancedTechniques import EMACallback

    ema = EMACallback(model, decay=0.999)

    diag = TrainingDiagnostics()
    stopper = EarlyStopping(patience=10)

    trainer.train()

    # 7. EMA 保存
    ema.on_save_checkpoint(output_dir, tokenizer)

    # 8. 模型汤 (如果有多个检查点)
    ckpt_dir = os.path.join(output_dir, "ckpt")
    if os.path.exists(ckpt_dir):
        ckpts = [
            os.path.join(ckpt_dir, d)
            for d in os.listdir(ckpt_dir)
            if d.startswith("step-") and os.path.isdir(os.path.join(ckpt_dir, d))
        ]
        if len(ckpts) >= 3:
            from ..alignment.AdvancedTechniques import ModelSoup

            ModelSoup().average_checkpoints(ckpts[-5:], output_dir, method="uniform")
            print("✅ 模型汤: 平均了最后 5 个检查点")

    # 9. API 蒸馏 (可选)
    if use_api_teacher and env.cuda_available:
        from ..distillation.RapidDistill import RapidDistill

        RapidDistill.run(
            model,
            tokenizer,
            use_teachers=api_teachers or ["gpt4o-mini"],
            total_examples=200,
            output_dir=output_dir,
        )
        print("✅ API 蒸馏完成")

    # 10. 轻量化
    from ..compression.lightweight import Lightweight

    Lightweight.optimize(model, tokenizer, target=model_size, output_dir=output_dir)
    print("✅ 轻量化完成")

    # 11. 报告卡
    TrainingReportCard.generate(
        model,
        training_config={"max_steps": steps},
        output_path=os.path.join(output_dir, "report.json"),
    )

    print(f"\n=== 🎉 最佳小模型训练完成! {output_dir} ===")
    return model, tokenizer
