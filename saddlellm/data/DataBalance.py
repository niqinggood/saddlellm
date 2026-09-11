#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

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
    """智能数据均衡预训练器"""

    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        self.datasets = self._load_datasets()
        self.domain_weights = self._init_domain_weights()
        self.loss_history = defaultdict(list)

    def _load_datasets(self) -> List[DatasetConfig]:
        """加载所有数据集配置"""
        datasets = []
        for ds in self.args.datasets:
            if not Path(ds['path']).exists():
                raise ValueError(f"数据集路径不存在: {ds['path']}")

            datasets.append(DatasetConfig(
                path=ds['path'],
                domain=ds['domain'],
                weight=ds['weight']
            ))

        logger.info(f"加载了 {len(datasets)} 个数据集")
        return datasets

    def _init_domain_weights(self) -> Dict[str, float]:
        """初始化领域权重"""
        weights = {}
        total = sum(ds.weight for ds in self.datasets)
        for ds in self.datasets:
            weights[ds.domain] = ds.weight / total
        return weights

    def _adaptive_sampling(self, epoch: int) -> Dict[str, float]:
        """自适应采样策略核心算法"""
        # 初始阶段保持配置权重
        if epoch < self.args.warmup_epochs:
            return self.domain_weights

        # 计算各领域平均损失变化
        domain_loss_changes = {}
        for domain in self.loss_history:
            if len(self.loss_history[domain]) >= 2:
                changes = np.diff(self.loss_history[domain][-5:])  # 最近5个step的变化
                domain_loss_changes[domain] = np.mean(changes)

        # 动态调整权重
        adjusted_weights = {}
        total_loss_change = sum(abs(c) for c in domain_loss_changes.values())

        for domain, weight in self.domain_weights.items():
            if domain in domain_loss_changes:
                # 损失下降慢的领域增加权重
                loss_change = domain_loss_changes[domain]
                if total_loss_change > 0:
                    adjustment = (abs(loss_change) / total_loss_change) * 0.2  # 最大调整20%
                    if loss_change > 0:  # 损失在增加
                        adjustment *= 1.5  # 更大幅度增加权重
                    adjusted_weights[domain] = weight * (1 + adjustment)
                else:
                    adjusted_weights[domain] = weight
            else:
                adjusted_weights[domain] = weight

        # 归一化
        total = sum(adjusted_weights.values())
        return {d: w / total for d, w in adjusted_weights.items()}

    def _load_data_chunk(self, dataset: DatasetConfig, chunk_size: int = 10000):
        """加载数据块"""
        # 这里简化为按行读取，实际应根据数据集格式实现
        with open(dataset.path, 'r') as f:
            lines = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                lines.append(json.loads(line) if dataset.path.endswith('.jsonl') else line.strip())
            return lines

    def train(self):
        """执行训练"""
        logger.info("开始预训练...")

        for epoch in range(self.args.epochs):
            logger.info(f"开始第 {epoch + 1}/{self.args.epochs} 轮训练")

            # 1. 确定当前epoch的数据混合比例
            if self.args.sampling_strategy == 'dynamic':
                domain_ratios = self._adaptive_sampling(epoch)
            else:
                domain_ratios = self.domain_weights

            # 2. 为每个领域创建数据加载器
            dataloaders = {}
            for ds in self.datasets:
                data = self._load_data_chunk(ds)
                dataloaders[ds.domain] = DataLoader(
                    data,
                    batch_size=int(self.args.batch_size * domain_ratios[ds.domain]),
                    shuffle=True
                )

            # 3. 混合数据训练
            self._train_epoch(epoch, dataloaders, domain_ratios)

            # 4. 保存检查点
            if (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int, dataloaders: Dict, domain_ratios: Dict):
        """训练单个epoch"""
        # 这里简化训练逻辑，实际应实现完整训练流程
        for step in range(self.args.steps_per_epoch):
            # 动态调整batch数据来源
            batch_domains = np.random.choice(
                list(domain_ratios.keys()),
                size=self.args.batch_size,
                p=list(domain_ratios.values())
            )

            # 模拟训练步骤
            domain_losses = {}
            for domain in set(batch_domains):
                # 从对应领域的数据加载器获取batch
                batch = next(iter(dataloaders[domain]))
                # 这里应该是实际的模型训练逻辑
                loss = self._train_step(batch, domain)
                domain_losses[domain] = loss
                self.loss_history[domain].append(loss)

            # 定期打印进度
            if (step + 1) % self.args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch + 1} - Step {step + 1} | "
                    f"Losses: {domain_losses} | "
                    f"Mix Ratios: {domain_ratios}"
                )

    def _train_step(self, batch, domain: str) -> float:
        """模拟训练步骤"""
        # 实际实现中这里应该是真正的模型训练逻辑
        return np.random.uniform(0.1, 1.0)  # 返回模拟loss

    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型状态、优化器状态等
        logger.info(f"保存检查点到 {checkpoint_dir}")
        # 实际实现中这里应该保存模型和训练状态


def parse_args():
    parser = argparse.ArgumentParser(description="大模型预训练数据均衡处理器")

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
