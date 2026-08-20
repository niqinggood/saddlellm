"""
课程学习调度器 - 分阶段训练策略
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    name: str
    start_step: int
    end_step: int
    max_seq_length: int = 2048
    learning_rate_mult: float = 1.0
    batch_size_mult: float = 1.0
    data_quality_threshold: float = 0.3
    domain_weights: Dict[str, float] = field(default_factory=dict)
    description: str = ""


class CurriculumScheduler:
    """
    课程学习调度器。
    分 4 个标准阶段:
    1. Warmup:   短序列(512), 高质量简单文本, 建立基础语言能力
    2. Main:     全序列(2048-4096), 混合领域数据, 主要知识获取
    3. Anneal:   降低学习率, 高质量精选数据, 长序列
    4. Cooldown: 极低学习率, 指令/对话数据, 提升 downstream 能力

    用法:
        scheduler = CurriculumScheduler.standard_curriculum(total_steps=100000)
        for step in range(total_steps):
            cfg = scheduler.get_current_config(step)
            # 根据 cfg 调整 data sampler 和 training args
    """

    def __init__(self, stages: List[StageConfig]):
        self.stages = sorted(stages, key=lambda s: s.start_step)
        self._validate_stages()
        self._current_stage_idx = 0

    @classmethod
    def standard_curriculum(
        cls,
        total_steps: int,
        max_seq_length: int = 2048,
        base_lr: float = 3e-4,
    ) -> "CurriculumScheduler":
        """创建标准 4 阶段课程表"""
        warmup_end = int(total_steps * 0.05)     # 前 5%
        main_end = int(total_steps * 0.85)        # 5%-85%
        anneal_end = int(total_steps * 0.95)      # 85%-95%
        # cooldown: 95%-100%

        stages = [
            StageConfig(
                name="warmup",
                start_step=0,
                end_step=warmup_end,
                max_seq_length=min(512, max_seq_length),
                learning_rate_mult=1.0,
                data_quality_threshold=0.6,
                domain_weights={"wikipedia": 0.5, "books": 0.3, "news": 0.2},
                description="短序列高质量数据预热",
            ),
            StageConfig(
                name="main",
                start_step=warmup_end,
                end_step=main_end,
                max_seq_length=max_seq_length,
                learning_rate_mult=1.0,
                data_quality_threshold=0.3,
                domain_weights={},  # uniform
                description="全序列混合数据主训练",
            ),
            StageConfig(
                name="anneal",
                start_step=main_end,
                end_step=anneal_end,
                max_seq_length=max_seq_length,
                learning_rate_mult=0.1,
                data_quality_threshold=0.5,
                domain_weights={"wikipedia": 0.3, "books": 0.3, "academic": 0.4},
                description="降低学习率,高质量数据退火",
            ),
            StageConfig(
                name="cooldown",
                start_step=anneal_end,
                end_step=total_steps,
                max_seq_length=max_seq_length,
                learning_rate_mult=0.01,
                data_quality_threshold=0.7,
                domain_weights={"instruction": 0.5, "conversation": 0.3, "qa": 0.2},
                description="极低学习率,指令数据冷却,提升下游能力",
            ),
        ]

        return cls(stages)

    def get_current_config(self, step: int) -> StageConfig:
        """获取当前步数对应的阶段配置"""
        for i, stage in enumerate(self.stages):
            if stage.start_step <= step < stage.end_step:
                self._current_stage_idx = i
                return stage
        # 超出范围,返回最后一个阶段
        return self.stages[-1]

    def is_stage_transition(self, step: int) -> bool:
        """检查当前步数是否是阶段切换点"""
        for stage in self.stages:
            if step == stage.start_step:
                return True
        return False

    def get_stage(self, step: int) -> Optional[StageConfig]:
        return self.get_current_config(step)

    @property
    def current_stage_name(self) -> str:
        return self.stages[self._current_stage_idx].name if self.stages else "unknown"

    def get_progress(self, step: int) -> Dict[str, Any]:
        """获取训练进度摘要"""
        total = self.stages[-1].end_step
        cfg = self.get_current_config(step)
        stage_progress = (step - cfg.start_step) / max(1, cfg.end_step - cfg.start_step)

        return {
            "step": step,
            "total_steps": total,
            "overall_progress": step / total,
            "current_stage": cfg.name,
            "stage_progress": stage_progress,
            "stage_description": cfg.description,
            "max_seq_length": cfg.max_seq_length,
            "lr_multiplier": cfg.learning_rate_mult,
        }

    def plot_curriculum(self, save_path: Optional[str] = None):
        """可视化课程表"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib 未安装,无法绘图")
            return

        total = self.stages[-1].end_step
        steps = np.arange(total)
        seq_lens = np.array([self.get_current_config(s).max_seq_length for s in steps])
        lr_mults = np.array([self.get_current_config(s).learning_rate_mult for s in steps])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax1.plot(steps, seq_lens)
        ax1.set_ylabel("Max Seq Length")
        ax1.set_title("课程学习调度")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, lr_mults, color="orange")
        ax2.set_ylabel("LR Multiplier")
        ax2.set_xlabel("Training Step")
        ax2.grid(True, alpha=0.3)

        # 标记阶段边界
        colors = ["green", "blue", "orange", "red"]
        for i, stage in enumerate(self.stages):
            for ax in [ax1, ax2]:
                ax.axvspan(stage.start_step, stage.end_step, alpha=0.1,
                           color=colors[i % len(colors)])
                ax.axvline(stage.start_step, color=colors[i % len(colors)],
                           linestyle="--", linewidth=1, alpha=0.5)
            mid = (stage.start_step + stage.end_step) // 2
            ax1.text(mid, ax1.get_ylim()[1] * 0.95, stage.name,
                     ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()

    def _validate_stages(self):
        """验证阶段配置没有空隙或重叠"""
        for i in range(len(self.stages) - 1):
            if self.stages[i].end_step != self.stages[i + 1].start_step:
                logger.warning(
                    f"阶段 '{self.stages[i].name}' 和 '{self.stages[i+1].name}' "
                    f"之间有空隙或重叠: end={self.stages[i].end_step}, "
                    f"next_start={self.stages[i+1].start_step}"
                )
