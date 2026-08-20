"""
实验追踪器 - 统一 WandB / TensorBoard / 本地日志
"""
import os
import json
import time
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    backend: str = "tensorboard"       # tensorboard, wandb, local
    output_dir: str = "./outputs"
    experiment_name: str = "experiment"
    run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 10
    save_checkpoints: bool = True
    save_code_snapshot: bool = False


class ExperimentTracker:
    """
    统一实验追踪。

    用法:
        tracker = ExperimentTracker(TrackingConfig(backend="tensorboard"))
        tracker.start()
        tracker.log_metrics({"loss": 2.3, "lr": 1e-4}, step=100)
        tracker.log_artifact("checkpoint.pt")
        tracker.end()
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self._backend = None
        self._start_time = None
        self._metrics_history: List[Dict] = []
        self._run_dir = None

    def start(self):
        self._start_time = time.time()
        run_name = self.config.run_name or f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_dir = os.path.join(self.config.output_dir, self.config.experiment_name, run_name)
        os.makedirs(self._run_dir, exist_ok=True)

        if self.config.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self._backend = SummaryWriter(log_dir=os.path.join(self._run_dir, "tb_logs"))
        elif self.config.backend == "wandb":
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project or self.config.experiment_name,
                    entity=self.config.wandb_entity,
                    name=run_name,
                    dir=self._run_dir,
                )
                self._backend = wandb
            except ImportError:
                logger.warning("wandb 未安装, 回退到本地日志")
                self.config.backend = "local"
                self._backend = self._LocalLogger(self._run_dir)
        else:
            self._backend = self._LocalLogger(self._run_dir)

        # 保存配置
        config_path = os.path.join(self._run_dir, "tracking_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"实验追踪已启动: {self._run_dir}")
        return self._run_dir

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """记录指标"""
        metrics["step"] = step
        metrics["timestamp"] = time.time()
        self._metrics_history.append(metrics)

        if isinstance(self._backend, self._LocalLogger):
            self._backend.log(metrics, step)
        elif hasattr(self._backend, "add_scalar"):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._backend.add_scalar(k, v, step)
        elif hasattr(self._backend, "log"):
            self._backend.log(metrics, step=step)

    def log_hyperparams(self, params: Dict):
        """记录超参数"""
        if hasattr(self._backend, "add_text"):
            params_str = json.dumps(params, ensure_ascii=False, indent=2)
            self._backend.add_text("hyperparameters", params_str, 0)
        elif hasattr(self._backend, "config"):
            self._backend.config.update(params)

        hp_path = os.path.join(self._run_dir, "hyperparams.json")
        with open(hp_path, "w") as f:
            json.dump(params, f, ensure_ascii=False, indent=2, default=str)

    def log_artifact(self, local_path: str, name: Optional[str] = None):
        """记录产物 (模型文件、图表等)"""
        import shutil
        dest = os.path.join(self._run_dir, "artifacts", name or os.path.basename(local_path))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isfile(local_path):
            shutil.copy2(local_path, dest)
        elif os.path.isdir(local_path):
            shutil.copytree(local_path, dest, dirs_exist_ok=True)

    def log_model_summary(self, model):
        """记录模型参数量和结构"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        summary = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_millions": round(total_params / 1e6, 2),
            "trainable_params_millions": round(trainable_params / 1e6, 2),
        }
        self.log_hyperparams({"model_summary": summary})

    def log_gpu_stats(self):
        """记录 GPU 状态"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.log_metrics({
                    f"gpu_{i}_util": util.gpu,
                    f"gpu_{i}_mem_used_gb": mem.used / (1024**3),
                    f"gpu_{i}_mem_total_gb": mem.total / (1024**3),
                }, step=0)
        except Exception:
            pass

    def end(self):
        """结束追踪"""
        duration = time.time() - self._start_time if self._start_time else 0

        # 保存所有指标历史
        metrics_path = os.path.join(self._run_dir, "metrics_history.json")
        with open(metrics_path, "w") as f:
            json.dump(self._metrics_history[-1000:], f, ensure_ascii=False, indent=2)

        if hasattr(self._backend, "close"):
            self._backend.close()
        elif hasattr(self._backend, "finish"):
            self._backend.finish()

        logger.info(f"实验追踪结束, 耗时 {duration/3600:.1f}h, 指标保存到 {self._run_dir}")

    def get_metrics_df(self):
        """返回 pandas DataFrame"""
        try:
            import pandas as pd
            return pd.DataFrame(self._metrics_history)
        except ImportError:
            return self._metrics_history

    class _LocalLogger:
        """本地 JSON 日志后端"""
        def __init__(self, run_dir: str):
            self.run_dir = run_dir
            self.log_path = os.path.join(run_dir, "metrics.jsonl")
            self._file = open(self.log_path, "w", encoding="utf-8")

        def log(self, metrics: Dict, step: int):
            metrics["step"] = step
            self._file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            self._file.flush()

        def close(self):
            self._file.close()
