"""
训练效率监控器 - MFU (Model FLOPs Utilization) + 吞吐量追踪
MFU 是大模型训练的命门指标 —— 告诉你 GPU 有多少算力真正用在计算上

MFU 定义: actual_FLOPs_per_second / theoretical_peak_FLOPs_per_second

典型 MFU 值:
  - 单卡训练 (DDP):        35-50%
  - DeepSpeed ZeRO-2 (4卡): 40-55%
  - DeepSpeed ZeRO-3 (8卡): 30-45%
  - TP+PP 混合并行 (16+卡):  25-40%
  - 优秀: >50%, 及格: >35%, 低效: <25%
"""
import time
import logging
import math
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


# GPU 理论峰值 TFLOPS (bf16/fp16 Tensor Core)
GPU_PEAK_TFLOPS = {
    "RTX 3090": 71,
    "RTX 4090": 165,
    "A100 40GB": 312,
    "A100 80GB": 312,
    "H100": 990,
    "H800": 990,
    "A800": 312,
    "V100": 125,
    "T4": 65,
    "default": 100,
}


@dataclass
class MFUConfig:
    gpu_type: str = "default"
    theoretical_peak_tflops: float = 0  # 覆盖 GPU_PEAK_TFLOPS
    mfu_window_size: int = 100          # MFU 计算滑动窗口步数
    log_every_n_steps: int = 10
    enable_sm_efficiency: bool = True   # SM 效率 (通过 pynvml)
    enable_memory_bandwidth: bool = False  # 显存带宽利用率
    alert_mfu_threshold: float = 0.25   # MFU 低于此值告警


class TrainingMonitor:
    """
    训练效率实时监控,追踪 MFU / tokens-per-second / GPU 利用率。

    用法:
        monitor = TrainingMonitor(model_params=1.5e9)
        monitor.on_step_start()

        # ... 训练一步 ...

        monitor.on_step_end(tokens_processed=batch_tokens)
        if step % 10 == 0:
            monitor.log_status(step)

        # 训练结束
        monitor.summary()
    """

    def __init__(
        self,
        model_params: int,
        config: Optional[MFUConfig] = None,
        num_gpus: int = 1,
        seq_length: int = 2048,
        gradient_accumulation_steps: int = 1,
    ):
        self.config = config or MFUConfig()
        self.model_params = model_params
        self.num_gpus = num_gpus
        self.seq_length = seq_length
        self.grad_accum = gradient_accumulation_steps

        # 理论峰值
        peak = self.config.theoretical_peak_tflops
        if peak <= 0:
            peak = GPU_PEAK_TFLOPS.get(self.config.gpu_type, GPU_PEAK_TFLOPS["default"])
        self.theoretical_peak_tflops = peak * num_gpus  # TFLOPS

        # FLOPs per token (forward + backward ≈ 2 * forward * 3)
        # Forward: 2*P, Backward: ~4*P, total ≈ 6*P per token
        self.flops_per_token = 6 * model_params
        if gradient_accumulation_steps > 1:
            self.flops_per_token *= gradient_accumulation_steps

        # 状态
        self.step_start_time = None
        self.total_tokens = 0
        self.total_time = 0.0
        self.step_count = 0

        # 滑动窗口
        self._mfu_window: deque = deque(maxlen=self.config.mfu_window_size)
        self._tps_window: deque = deque(maxlen=self.config.mfu_window_size)
        self._step_times: deque = deque(maxlen=self.config.mfu_window_size)

        # GPU 监控
        self._gpu_utils: List[float] = []
        self._gpu_mems: List[float] = []

        logger.info(f"TrainingMonitor 初始化: {model_params/1e9:.2f}B params, "
                     f"peak={self.theoretical_peak_tflops:.0f} TFLOPS, "
                     f"flops/token={self.flops_per_token/1e9:.2f} GFLOPS")

    def on_step_start(self):
        self.step_start_time = time.time()

    def on_step_end(self, tokens_processed: int):
        """训练一步结束,记录指标"""
        if self.step_start_time is None:
            return

        elapsed = time.time() - self.step_start_time
        self.total_tokens += tokens_processed
        self.total_time += elapsed
        self.step_count += 1

        # Tokens per second
        tps = tokens_processed / max(elapsed, 1e-6)
        self._tps_window.append(tps)

        # MFU
        actual_flops_per_sec = self.flops_per_token * tokens_processed / max(elapsed, 1e-6)
        peak_flops_per_sec = self.theoretical_peak_tflops * 1e12
        mfu = actual_flops_per_sec / max(peak_flops_per_sec, 1e-6)
        self._mfu_window.append(mfu)

        self._step_times.append(elapsed)

        # 告警
        if mfu < self.config.alert_mfu_threshold and self.step_count > 20:
            avg_tps = self._avg(self._tps_window)
            logger.warning(
                f"MFU 过低: {mfu*100:.1f}% (阈值 {self.config.alert_mfu_threshold*100:.0f}%), "
                f"tokens/s={avg_tps:.0f}, 检查 batch_size 和 gradient_accumulation"
            )

        self.step_start_time = None

    def log_status(self, step: int) -> Dict:
        """记录当前状态,返回指标字典"""
        avg_mfu = self._avg(self._mfu_window)
        avg_tps = self._avg(self._tps_window)
        avg_step_time = self._avg(self._step_times)

        # GPU 硬件指标
        gpu_util = None
        gpu_mem = None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem = mem.used / mem.total * 100
        except Exception:
            pass

        metrics = {
            "step": step,
            "mfu_pct": round(avg_mfu * 100, 1),
            "tokens_per_sec": round(avg_tps, 0),
            "tokens_per_sec_per_gpu": round(avg_tps / max(1, self.num_gpus), 0),
            "step_time_ms": round(avg_step_time * 1000, 1),
            "total_tokens": self.total_tokens,
            "total_hours": round(self.total_time / 3600, 2),
        }
        if gpu_util is not None:
            metrics["gpu_util_pct"] = gpu_util
            metrics["gpu_mem_pct"] = round(gpu_mem, 1) if gpu_mem else 0
            metrics["sm_efficiency"] = round(avg_mfu * 100 / max(gpu_util, 1), 1)

        # 单行日志
        parts = [
            f"step={step}",
            f"MFU={metrics['mfu_pct']}%",
            f"tokens/s={metrics['tokens_per_sec']:.0f}",
            f"step={metrics['step_time_ms']:.0f}ms",
        ]
        if gpu_util is not None:
            parts.append(f"GPU={gpu_util}%")
        logger.info(" | ".join(parts))

        return metrics

    def summary(self) -> Dict:
        """训练结束,生成汇总报告"""
        avg_mfu = self._avg(self._mfu_window)
        avg_tps = self._avg(self._tps_window)
        peak_mfu = max(self._mfu_window) if self._mfu_window else 0
        peak_tps = max(self._tps_window) if self._tps_window else 0

        total_tokens_b = self.total_tokens / 1e9
        total_hours = self.total_time / 3600

        # 效率评级
        if avg_mfu > 0.50:
            grade = "A - 优秀 (接近硬件极限)"
        elif avg_mfu > 0.35:
            grade = "B - 良好 (正常水平)"
        elif avg_mfu > 0.20:
            grade = "C - 及格 (有优化空间)"
        else:
            grade = "D - 低效 (需要排查)"

        summary = {
            "total_tokens_billions": round(total_tokens_b, 2),
            "total_hours": round(total_hours, 2),
            "avg_mfu_pct": round(avg_mfu * 100, 1),
            "peak_mfu_pct": round(peak_mfu * 100, 1),
            "avg_tokens_per_sec": round(avg_tps, 0),
            "peak_tokens_per_sec": round(peak_tps, 0),
            "effective_tflops": round(avg_mfu * self.theoretical_peak_tflops, 1),
            "grade": grade,
            "model_params_billions": round(self.model_params / 1e9, 2),
            "num_gpus": self.num_gpus,
            "gpu_type": self.config.gpu_type,
        }

        report = f"""
{'='*60}
训练效率报告
{'='*60}
模型: {summary['model_params_billions']:.1f}B params | GPU: {summary['num_gpus']}x {self.config.gpu_type}
序列长度: {self.seq_length} | 梯度累积: {self.grad_accum}
────────────────────────────────────────────────────
总训练Token:  {summary['total_tokens_billions']:.1f}B
总耗时:       {summary['total_hours']:.1f}h
平均吞吐:     {summary['avg_tokens_per_sec']:.0f} tokens/s
平均MFU:      {summary['avg_mfu_pct']:.1f}%   (峰值 {summary['peak_mfu_pct']:.1f}%)
有效算力:     {summary['effective_tflops']:.1f} TFLOPS / {self.theoretical_peak_tflops:.0f} TFLOPS
────────────────────────────────────────────────────
效率评级:     {grade}
{'='*60}
"""
        logger.info(report)
        return summary

    def estimate_completion(self, total_steps: int, current_step: int) -> Dict:
        """估算训练完成时间"""
        avg_step_time = self._avg(self._step_times)
        if avg_step_time <= 0:
            return {"status": "not_enough_data"}

        remaining_steps = total_steps - current_step
        eta_seconds = remaining_steps * avg_step_time
        total_estimated = total_steps * avg_step_time

        return {
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_pct": round(current_step / total_steps * 100, 1),
            "avg_step_time_ms": round(avg_step_time * 1000, 1),
            "eta_hours": round(eta_seconds / 3600, 1),
            "total_estimated_hours": round(total_estimated / 3600, 1),
            "estimated_finish_tokens_b": round(
                total_steps * self._avg(self._tps_window) * avg_step_time / 1e9, 1
            ) if self._tps_window else 0,
        }

    def _avg(self, window: deque) -> float:
        if not window:
            return 0.0
        return sum(window) / len(window)

    def get_latest_metrics(self) -> Dict:
        """获取最新指标 (供外部回调使用)"""
        return {
            "mfu": round(self._avg(self._mfu_window) * 100, 1) if self._mfu_window else 0,
            "tokens_per_sec": round(self._avg(self._tps_window), 0) if self._tps_window else 0,
            "step_time_ms": round(self._avg(self._step_times) * 1000, 1) if self._step_times else 0,
        }
