"""Native clipped PPO with a learned token value head and terminal verifier.

Reuses SaddleLLM's rollout scorer. GAE, frozen rollout log probabilities,
multiple update epochs, clipped value loss and a fixed reference are explicit.
The critic uses detached policy hidden states and trains its own linear head.
"""

# ruff: noqa: N999 -- Matches the public module naming convention.

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .GRPOTrainer import GRPOConfig, GRPOTrainer


@dataclass
class VerifiablePPOConfig(GRPOConfig):
    ppo_epochs: int = 2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    value_clip: float = 0.2
    dynamic_sampling: bool = False

    def validate(self):
        super().validate()
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if not math.isfinite(self.value_coefficient) or self.value_coefficient <= 0:
            raise ValueError("value_coefficient must be finite and positive")
        if not math.isfinite(self.value_clip) or self.value_clip <= 0:
            raise ValueError("value_clip must be finite and positive")
        if self.dynamic_sampling or self.gradient_accumulation_steps != 1:
            raise ValueError(
                "Native PPO requires dynamic_sampling=False and gradient_accumulation_steps=1"
            )


def generalized_advantages(rewards, values, gamma=1.0, gae_lambda=0.95):
    """Terminal episodes: no bootstrap beyond the final sampled token."""
    advantages = torch.zeros_like(values)
    running = torch.zeros((), device=values.device, dtype=values.dtype)
    for t in reversed(range(len(values))):
        following = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * following - values[t]
        running = delta + gamma * gae_lambda * running
        advantages[t] = running
    return advantages, advantages + values


class VerifiablePPOTrainer(GRPOTrainer):
    def __init__(self, model, ref_model, tokenizer, config=None):
        super().__init__(model, ref_model, tokenizer, config or VerifiablePPOConfig())
        hidden = getattr(model.config, "hidden_size", None) or getattr(
            model.config, "n_embd", None
        )
        if not hidden:
            raise ValueError("PPO requires model.config.hidden_size or n_embd")
        self.value_head = torch.nn.Linear(hidden, 1).to(
            self.device, dtype=torch.float32
        )
        torch.nn.init.zeros_(self.value_head.weight)
        torch.nn.init.zeros_(self.value_head.bias)
        parameters = [p for p in model.parameters() if p.requires_grad] + list(
            self.value_head.parameters()
        )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _forward_values(self, ids, mask, width):
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )
        log_probs = self._completion_log_probs(output.logits, ids, width).squeeze(0)
        hidden = output.hidden_states[-1][:, width - 1 : -1, :].detach().float()
        values = self.value_head(hidden).squeeze(0).squeeze(-1)
        return log_probs, values

    def train(
        self,
        prompts,
        reward_fn,
        num_steps=1000,
        reward_model=None,
        eval_prompts=None,
        metrics_callback=None,
        rollout_callback=None,
    ):
        if not prompts or num_steps < 1:
            raise ValueError("PPO requires prompts and positive num_steps")
        for _ in range(num_steps):
            started = time.perf_counter()
            batch = self._sample_prompts(prompts, self.config.per_device_batch_size)
            expanded = [p for p in batch for _ in range(self.config.num_generations)]
            texts, tokens, old_logs, prefix, prefix_masks, width = (
                self._generate_responses_with_log_probs(expanded)
            )
            rewards = [
                float(reward_fn(p, t))
                + (float(reward_model(p, t)) if reward_model else 0.0)
                for p, t in zip(expanded, texts)
            ]
            if not all(math.isfinite(r) for r in rewards):
                raise ValueError("reward_fn returned a non-finite value")
            if rollout_callback:
                rollout_callback(
                    {
                        "step": self._step + 1,
                        "prompts": expanded,
                        "responses": texts,
                        "rewards": rewards,
                        "completion_tokens": [len(t) for t in tokens],
                    }
                )
            trajectories = []
            self.model.eval()  # Disable dropout in both rollout scoring and updates.
            for i, ids in enumerate(tokens):
                if not len(ids):
                    continue
                # Clone inference tensors before using them in differentiable passes.
                ids = torch.cat([prefix[i], ids]).unsqueeze(0).clone()
                mask = (
                    torch.cat([prefix_masks[i], torch.ones_like(tokens[i])])
                    .unsqueeze(0)
                    .clone()
                )
                with torch.no_grad():
                    _, values = self._forward_values(ids, mask, width)
                    token_rewards = torch.zeros_like(values)
                    if self.config.beta:
                        ref = self._reference_logits(ids, mask)
                        ref_logs = self._completion_log_probs(ref, ids, width).squeeze(
                            0
                        )
                        token_rewards -= self.config.beta * (old_logs[i] - ref_logs)
                    token_rewards[-1] += rewards[i]
                    advantages, returns = generalized_advantages(
                        token_rewards, values, self.config.gamma, self.config.gae_lambda
                    )
                trajectories.append(
                    (
                        ids,
                        mask,
                        old_logs[i].clone(),
                        values.clone(),
                        advantages.clone(),
                        returns.clone(),
                    )
                )
            if not trajectories:
                raise RuntimeError("PPO produced no trainable completion tokens")
            flat = torch.cat([t[4] for t in trajectories])
            mean, std = flat.mean(), flat.std(unbiased=False)
            epoch_losses, value_losses = [], []
            for _epoch in range(self.config.ppo_epochs):
                self.optimizer.zero_grad(set_to_none=True)
                for ids, mask, old, old_values, advantages, returns in trajectories:
                    logs, values = self._forward_values(ids, mask, width)
                    advantages = (advantages - mean) / (std + 1e-8)
                    ratio = (logs - old).clamp(-20, 20).exp()
                    actor_loss = -torch.minimum(
                        ratio * advantages, self._clip_policy_ratio(ratio) * advantages
                    ).mean()
                    clipped = old_values + (values - old_values).clamp(
                        -self.config.value_clip, self.config.value_clip
                    )
                    critic_loss = (
                        0.5
                        * torch.maximum(
                            (values - returns).square(), (clipped - returns).square()
                        ).mean()
                    )
                    loss = actor_loss + self.config.value_coefficient * critic_loss
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite PPO loss")
                    (loss / len(trajectories)).backward()
                    epoch_losses.append(float(loss.detach()))
                    value_losses.append(float(critic_loss.detach()))
                parameters = [
                    p for p in self.model.parameters() if p.requires_grad
                ] + list(self.value_head.parameters())
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        parameters, self.config.max_grad_norm
                    )
                )
                if not math.isfinite(norm):
                    raise FloatingPointError("Non-finite PPO gradient")
                self.optimizer.step()
                self._optimizer_step += 1
            self._step += 1
            metrics = {
                "step": self._step,
                "optimizer_step": self._optimizer_step,
                "loss": sum(epoch_losses) / len(epoch_losses),
                "value_loss": sum(value_losses) / len(value_losses),
                "reward_mean": sum(rewards) / len(rewards),
                "grad_norm": norm,
                "sampled_groups": len(batch),
                "accepted_groups": len(batch),
                "sampled_rollouts": len(texts),
                "duration_s": time.perf_counter() - started,
            }
            self._metrics_history.append(metrics)
            if metrics_callback:
                metrics_callback(metrics)
        return self._metrics_history

    def save(self, path):
        super().save(path)
        torch.save(self.value_head.state_dict(), Path(path) / "value_head.pt")
