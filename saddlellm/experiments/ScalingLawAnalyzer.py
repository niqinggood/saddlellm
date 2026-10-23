"""
缩放法则分析器 - Chinchilla scaling law 拟合与预测
"""
import json
import math
import logging
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScalingRun:
    model_name: str
    params: int           # 模型参数量
    tokens: int           # 训练 token 数
    loss: float           # 最终 loss
    compute_flops: Optional[float] = None
    downstream_score: Optional[float] = None


@dataclass
class PretrainRunPlan:
    model_name: str
    params: int
    active_params: int
    target_tokens: int
    max_steps: int
    global_batch_size: int
    seq_length: int
    approx_flops: float
    memory_estimate_gb: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_config(
        self,
        corpus_sources: Sequence[Dict],
        output_dir: str,
        tokenizer_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        per_device_batch_size: int = 1,
        stability_monitor: bool = True,
    ) -> Dict:
        return {
            "project": "saddlellm-pretrain-grid",
            "experiment": self.model_name,
            "pretrain_mode": "scratch",
            "stages": ["pretrain", "eval"],
            "model": {
                "config": self.model_name,
                "tokenizer": tokenizer_path,
            },
            "data": {
                "sources": list(corpus_sources),
                "max_seq_length": self.seq_length,
                "pack_sequences": True,
            },
            "training": {
                "max_steps": self.max_steps,
                "global_batch_size": self.global_batch_size,
                "per_device_batch_size": per_device_batch_size,
                "learning_rate": learning_rate,
                "warmup_steps": max(10, self.max_steps // 100),
                "stability_monitor": stability_monitor,
            },
            "logging": {
                "output_dir": output_dir,
                "backend": "tensorboard",
            },
        }


class ScalingLawAnalyzer:
    """
    拟合 Chinchilla 缩放法则,根据小模型预测大模型能力。

    核心公式:
        L(N, D) = E + A / N^alpha + B / D^beta

    其中 N = 参数量, D = 训练 token 数, E = 不可约减 loss

    用法:
        analyzer = ScalingLawAnalyzer()
        analyzer.add_run("100m", params=1.24e8, tokens=2e9, loss=3.2)
        analyzer.add_run("300m", params=3.0e8, tokens=6e9, loss=2.8)
        analyzer.fit()
        predicted = analyzer.predict_loss(params=1e9, tokens=20e9)
        print(f"1B 模型预测 loss: {predicted:.3f}")
    """

    def __init__(self):
        self.runs: List[ScalingRun] = []
        self._fitted = False
        self._E = None  # irreducible loss
        self._A = None  # param coefficient
        self._B = None  # data coefficient
        self._alpha = 0.34  # Chinchilla param exponent
        self._beta = 0.28   # Chinchilla data exponent

    def add_run(self, model_name: str, params: int, tokens: int, loss: float,
                compute_flops: Optional[float] = None):
        """添加一次训练运行的数据点"""
        if compute_flops is None:
            compute_flops = 6 * params * tokens  # 近似
        self.runs.append(ScalingRun(
            model_name=model_name, params=params, tokens=tokens,
            loss=loss, compute_flops=compute_flops,
        ))
        self._fitted = False

    def add_runs_from_summaries(self, summary_paths: List[str]):
        """从训练摘要 JSON 文件中批量导入"""
        for path in summary_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pretrain = data.get("results", {}).get("pretrain", {})
                eval_results = data.get("results", {}).get("eval", {})
                if pretrain and eval_results:
                    self.add_run(
                        model_name=data.get("experiment", "unknown"),
                        params=pretrain.get("estimated_params", pretrain.get("total_params", 0)),
                        tokens=data.get("training", {}).get("max_steps", 0) * data.get("training", {}).get("global_batch_size", 0),
                        loss=eval_results.get("perplexity", 100),
                    )
            except Exception as e:
                logger.warning(f"无法加载 {path}: {e}")

    def fit(self, alpha: float = 0.34, beta: float = 0.28) -> Dict:
        """
        拟合缩放法则参数。
        使用最小二乘法拟合 E, A, B。
        """
        self._fitted = False
        if len(self.runs) < 3:
            raise ValueError("At least three independent runs are required to fit E, A and B")
        if any(not math.isfinite(value) or value <= 0 for value in (alpha, beta)):
            raise ValueError("alpha and beta must be finite and positive")
        if any(not math.isfinite(r.params) or not math.isfinite(r.tokens)
               or r.params <= 0 or r.tokens <= 0 or not math.isfinite(r.loss) for r in self.runs):
            raise ValueError("Run params/tokens must be positive and loss must be finite")

        self._alpha = alpha
        self._beta = beta

        # L(N, D) = E + A / N^alpha + B / D^beta
        # 这是一个线性问题: L = E + A * x1 + B * x2
        # 其中 x1 = 1/N^alpha, x2 = 1/D^beta

        x1 = [1.0 / (r.params ** alpha) for r in self.runs]
        x2 = [1.0 / (r.tokens ** beta) for r in self.runs]
        y = [r.loss for r in self.runs]

        # Scale the features before solving: raw N^-alpha and D^-beta are
        # small enough that a fixed determinant tolerance falsely declares
        # ordinary large-model experiments singular.
        import numpy as np

        scale1, scale2 = max(x1), max(x2)
        design = np.column_stack((np.ones(len(y)), np.array(x1) / scale1, np.array(x2) / scale2))
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 3:
            raise ValueError("Runs do not independently identify E, A and B; vary params and tokens")
        self._E = float(coefficients[0])
        self._A = float(coefficients[1] / scale1)
        self._B = float(coefficients[2] / scale2)

        self._fitted = True

        return {
            "E": self._E,
            "A": self._A,
            "B": self._B,
            "alpha": self._alpha,
            "beta": self._beta,
            "formula": f"L(N,D) = {self._E:.3f} + {self._A:.1f}/N^{alpha} + {self._B:.1f}/D^{beta}",
            "r2": self._compute_r2(),
        }

    def predict_loss(self, params: int, tokens: int) -> float:
        """预测给定模型大小和训练量的 loss"""
        if any(not math.isfinite(value) or value <= 0 for value in (params, tokens)):
            raise ValueError("params and tokens must be finite and positive")
        if not self._fitted:
            self.fit()
        return self._E + self._A / (params ** self._alpha) + self._B / (tokens ** self._beta)

    def predict_chinchilla_optimal(self, compute_budget_flops: float) -> Tuple[int, int]:
        """
        给定计算预算,返回 Chinchilla 最优的 (参数量, 训练 token 数)。
        Minimize the fitted L(N,D), with C = 6*N*D. The continuous
        optimum is rounded down to positive integers within the budget.
        """
        if not math.isfinite(compute_budget_flops) or compute_budget_flops < 6:
            raise ValueError("compute_budget_flops must be finite and at least 6 for N,D >= 1")
        if not self._fitted:
            self.fit()
        if any(value is None or not math.isfinite(value) or value <= 0
               for value in (self._A, self._B, self._alpha, self._beta)):
            raise ValueError("A, B, alpha and beta must be finite and positive for an interior optimum")
        # N* = (alpha*A/(beta*B))^(1/(alpha+beta))
        #      * (C/6)^(beta/(alpha+beta)). Evaluate in log space.
        token_param_budget = compute_budget_flops / 6
        log_budget = math.log(token_param_budget)
        log_n = (math.log(self._alpha) + math.log(self._A)
                 - math.log(self._beta) - math.log(self._B)
                 + self._beta * log_budget) / (self._alpha + self._beta)
        n_opt = max(1, min(int(token_param_budget), int(math.exp(min(log_budget, max(0.0, log_n))))))
        d_opt = max(1, int(token_param_budget / n_opt))
        return n_opt, d_opt

    def recommend_next_run(self, budget_flops: float) -> Dict:
        """推荐下一个模型规模"""
        n_opt, d_opt = self.predict_chinchilla_optimal(budget_flops)
        predicted_loss = self.predict_loss(n_opt, d_opt)
        return {
            "recommended_params": n_opt,
            "recommended_tokens": d_opt,
            "predicted_loss": predicted_loss,
            "params_human": f"{n_opt/1e9:.1f}B" if n_opt >= 1e9 else f"{n_opt/1e6:.0f}M",
            "tokens_human": f"{d_opt/1e9:.1f}B" if d_opt >= 1e9 else f"{d_opt/1e6:.0f}M",
        }

    def plan_pretrain_grid(
        self,
        model_names: Optional[Sequence[str]] = None,
        token_multipliers: Sequence[float] = (2.0, 5.0, 10.0, 20.0),
        global_batch_size: int = 512,
        seq_length: int = 2048,
        max_tokens: Optional[int] = None,
        max_params: Optional[int] = None,
        dtype: str = "bf16",
    ) -> List[PretrainRunPlan]:
        """Generate a low-cost scratch-pretraining experiment grid."""
        from ..models.ModelRegistry import MODEL_SPECS, ModelRegistry

        names = list(model_names or MODEL_SPECS.keys())
        plans: List[PretrainRunPlan] = []
        tokens_per_step = max(1, global_batch_size * seq_length)
        for name in names:
            if name not in MODEL_SPECS:
                continue
            spec = MODEL_SPECS[name]
            if max_params and spec.estimated_params > max_params:
                continue
            active_params = spec.estimated_active_params or spec.estimated_params
            for mult in token_multipliers:
                target_tokens = int(active_params * mult)
                if max_tokens:
                    target_tokens = min(target_tokens, max_tokens)
                max_steps = max(1, math.ceil(target_tokens / tokens_per_step))
                memory = None
                try:
                    memory = ModelRegistry.estimate_memory(
                        spec,
                        batch_size=1,
                        seq_length=seq_length,
                        dtype=dtype,
                    ).get("total_gb")
                except Exception:
                    memory = None
                plans.append(PretrainRunPlan(
                    model_name=name,
                    params=spec.estimated_params,
                    active_params=active_params,
                    target_tokens=target_tokens,
                    max_steps=max_steps,
                    global_batch_size=global_batch_size,
                    seq_length=seq_length,
                    approx_flops=6.0 * active_params * target_tokens,
                    memory_estimate_gb=memory,
                ))
        return sorted(plans, key=lambda p: (p.params, p.target_tokens))

    def export_pretrain_grid(self, path: str, plans: List[PretrainRunPlan]) -> str:
        """Save a generated pretraining grid as JSON."""
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in plans], f, indent=2, ensure_ascii=False)
        return path

    def plot_scaling_curve(self, save_path: Optional[str] = None):
        """绘制缩放曲线 (需要 matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib 未安装,无法绘图")
            return

        if not self._fitted:
            self.fit()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图: Loss vs Params
        ax = axes[0]
        params_actual = [r.params for r in self.runs]
        losses_actual = [r.loss for r in self.runs]
        ax.scatter(params_actual, losses_actual, color="blue", s=60, zorder=5, label="实际")

        # 拟合曲线
        p_range = np.logspace(math.log10(min(params_actual) * 0.5),
                              math.log10(max(params_actual) * 2), 100)
        fixed_tokens = int(np.median([r.tokens for r in self.runs]))
        l_pred = [self.predict_loss(int(p), fixed_tokens) for p in p_range]
        ax.plot(p_range, l_pred, "r--", label=f"拟合 (D={fixed_tokens/1e9:.1f}B tokens)")

        ax.set_xscale("log")
        ax.set_xlabel("参数量 (N)")
        ax.set_ylabel("Loss")
        ax.set_title("缩放法则: Loss vs 参数量")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 右图: Loss vs Tokens
        ax = axes[1]
        for run in self.runs:
            ax.scatter(run.tokens, run.loss, color="green", s=60, zorder=5)
            ax.annotate(run.model_name, (run.tokens, run.loss),
                        textcoords="offset points", xytext=(5, -10), fontsize=8)

        d_range = np.logspace(math.log10(min(r.tokens for r in self.runs) * 0.5),
                              math.log10(max(r.tokens for r in self.runs) * 2), 100)
        fixed_params = int(np.median(params_actual))
        l_pred_d = [self.predict_loss(fixed_params, int(d)) for d in d_range]
        ax.plot(d_range, l_pred_d, "orange", linestyle="--",
                label=f"拟合 (N={fixed_params/1e6:.0f}M params)")

        ax.set_xscale("log")
        ax.set_xlabel("训练 Token 数 (D)")
        ax.set_ylabel("Loss")
        ax.set_title("缩放法则: Loss vs 训练量")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"缩放曲线已保存到 {save_path}")
        else:
            plt.show()

    def generate_report(self) -> str:
        """生成缩放分析报告"""
        if not self._fitted:
            self.fit()

        lines = [
            "=" * 60,
            "缩放法则分析报告",
            "=" * 60,
            f"数据点数: {len(self.runs)}",
            f"拟合公式: L(N,D) = {self._E:.3f} + {self._A:.1f}/N^{self._alpha} + {self._B:.1f}/D^{self._beta}",
            f"不可约减 loss (E): {self._E:.3f}",
            "",
            "--- 已有运行 ---",
        ]

        for r in sorted(self.runs, key=lambda x: x.params):
            pred = self.predict_loss(r.params, r.tokens)
            lines.append(
                f"  {r.model_name:<20} params={r.params/1e6:>6.0f}M  "
                f"tokens={r.tokens/1e9:>5.1f}B  actual_loss={r.loss:.3f}  "
                f"predicted_loss={pred:.3f}  delta={r.loss-pred:+.3f}"
            )

        lines.append("")
        lines.append("--- 预测 ---")
        for params in [1e8, 3e8, 1e9, 3e9, 7e9]:
            optimal_tokens = int(params * 20)  # Chinchilla
            pred = self.predict_loss(int(params), optimal_tokens)
            name = f"{params/1e9:.1f}B" if params >= 1e9 else f"{params/1e6:.0f}M"
            lines.append(f"  {name:<20} optimal_tokens={optimal_tokens/1e9:>5.1f}B  predicted_loss={pred:.3f}")

        return "\n".join(lines)

    def _compute_r2(self) -> float:
        """计算拟合的 R²"""
        ss_res = 0.0
        ss_tot = 0.0
        y_mean = sum(r.loss for r in self.runs) / len(self.runs)
        for run in self.runs:
            pred = self.predict_loss(run.params, run.tokens)
            ss_res += (run.loss - pred) ** 2
            ss_tot += (run.loss - y_mean) ** 2
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0
