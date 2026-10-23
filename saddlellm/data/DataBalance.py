#!/usr/bin/env python3
import argparse
import json
import logging
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    path: str
    domain: str
    weight: float = 1.0
    current_ratio: float = 0.0


class BalancedPretrainer:
    """Disabled legacy sampler prototype; it never trained a model.

    The old implementation returned random losses and created empty checkpoint
    directories. Use saddlellm.training.NativeTrainer / the pretrain pipeline
    for real forward/backward, optimizer updates and resumable checkpoints.
    """

    _ERROR = (
        "BalancedPretrainer (balance_data) is an unsupported legacy prototype: "
        "it has no model, optimizer or real checkpoint implementation. "
        "Use saddlellm.training.NativeTrainer or the framework pretrain stage."
    )

    def __init__(self, args):
        # Fail before tokenizer downloads, dataset reads or output creation.
        raise NotImplementedError(self._ERROR)

    def train(self):
        raise NotImplementedError(self._ERROR)

    def _train_step(self, batch, domain: str) -> float:
        raise NotImplementedError(self._ERROR)

    def _save_checkpoint(self, epoch: int):
        raise NotImplementedError(self._ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="已停用的数据均衡训练原型；实际预训练请使用 NativeTrainer 或框架 pretrain 阶段")

    # 数据集参数
    parser.add_argument("--datasets", required=True, type=json.loads,
                        help="JSON格式的数据集配置列表")
    parser.add_argument("--tokenizer-path", required=True,
                        help="分词器路径")

    # 采样策略
    parser.add_argument("--sampling-strategy", default="dynamic",
                        choices=["dynamic", "curriculum", "hybrid"],
                        help="数据采样策略")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="权重预热轮次")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练总轮次")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="全局批大小")
    parser.add_argument("--steps-per-epoch", type=int, default=1000,
                        help="每轮训练步数")
    parser.add_argument("--learning-rate", type=float, default=6e-5,
                        help="学习率")

    # 日志和保存
    parser.add_argument("--output-dir", required=True,
                        help="输出目录")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="保存间隔(epoch)")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="日志间隔(step)")

    return parser.parse_args()


def main():
    args = parse_args()
    trainer = BalancedPretrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
