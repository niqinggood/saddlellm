"""Single-device DAPO for task-supplied verifiable rewards.

Implements decoupled clipping, dynamic reward-variance sampling, token-weighted
loss and soft overlong reward shaping (arXiv:2503.14476). This is a native
research backend, not the distributed verl system used by the paper.
"""

# ruff: noqa: N999 -- Matches the public module naming convention.

import math
from dataclasses import dataclass

import torch

from .GRPOTrainer import GRPOConfig, GRPOTrainer


@dataclass
class DAPOConfig(GRPOConfig):
    epsilon_high: float = 0.28
    beta: float = 0.0
    dynamic_sampling: bool = True
    overlong_buffer: int = 16
    overlong_penalty: float = 1.0

    def validate(self):
        super().validate()
        if not self.epsilon <= self.epsilon_high < 1:
            raise ValueError("epsilon_high must be >= epsilon and < 1")
        if self.beta != 0 or not self.dynamic_sampling:
            raise ValueError("DAPO requires beta=0 and dynamic_sampling=True")
        if not 0 < self.overlong_buffer <= self.max_new_tokens:
            raise ValueError("overlong_buffer must be in [1, max_new_tokens]")
        if not math.isfinite(self.overlong_penalty) or self.overlong_penalty < 0:
            raise ValueError("overlong_penalty must be finite and >= 0")


class DAPOTrainer(GRPOTrainer):
    def __init__(self, model, ref_model, tokenizer, config=None):
        super().__init__(model, ref_model, tokenizer, config or DAPOConfig())

    def _clip_policy_ratio(self, ratio):
        return ratio.clamp(1 - self.config.epsilon, 1 + self.config.epsilon_high)

    def _aggregate_sequence_losses(self, losses, lengths):
        weights = losses[0].new_tensor(lengths)
        return (torch.stack(losses) * weights).sum() / weights.sum()

    def _shape_rollout_rewards(self, rewards, rollout_ids):
        threshold = self.config.max_new_tokens - self.config.overlong_buffer
        return [
            reward
            - self.config.overlong_penalty
            * min(1.0, max(0, len(ids) - threshold) / self.config.overlong_buffer)
            for reward, ids in zip(rewards, rollout_ids)
        ]
