"""A small, dependency-light Group Relative Policy Optimization trainer.

The native trainer is intended for smoke tests and small research pilots.  It
supports either a separate frozen reference model or a single PEFT model whose
adapter can be disabled to recover the reference policy.  The latter keeps a
Qwen3-0.6B LoRA pilot comfortably within a consumer GPU memory budget.

For every prompt the trainer samples ``G`` completions, normalizes rewards
within the group, and applies a token-level clipped policy objective plus a KL
penalty to the reference policy.  One policy update is made per rollout batch,
so the detached rollout-policy log probabilities are the PPO/GRPO ``old`` log
probabilities for that update.
"""

from __future__ import annotations

import logging
import math
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


RewardFn = Callable[[str, str], float]


@dataclass
class GRPOConfig:
    """Configuration for :class:`GRPOTrainer`.

    ``num_steps`` passed to :meth:`GRPOTrainer.train` counts rollout
    micro-steps.  An optimizer update is performed every
    ``gradient_accumulation_steps`` micro-steps.
    """

    # Rollout generation
    num_generations: int = 8
    max_prompt_length: int = 1024
    max_new_tokens: int = 512
    max_sequence_length: int = 2048
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.0

    # Optimization
    learning_rate: float = 1e-6
    beta: float = 0.04
    epsilon: float = 0.2
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0

    # Training
    num_epochs: int = 1  # Kept for API compatibility; train() is step based.
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # Prompt selection. ``shuffle_cycle`` visits every prompt once before
    # reshuffling, which avoids repeatedly drawing the same small subset.
    prompt_sampling_strategy: str = "random"  # random | shuffle_cycle

    # DAPO-style dynamic sampling.  When enabled, groups whose rewards have no
    # variance are discarded and replacement prompts are rolled out until the
    # requested batch is filled or the attempt budget is exhausted.
    dynamic_sampling: bool = False
    dynamic_sampling_max_attempts: int = 8
    dynamic_sampling_min_reward_std: float = 1e-8

    # KL estimate: k1 | k2 | k3. k3 is non-negative and low variance.
    kl_estimator: str = "k3"

    # Reward processing
    reward_normalize: bool = True
    reward_clip: float = 10.0

    # A value <= 0 keeps the reference policy fixed, which is standard RLVR.
    ref_model_update_interval: int = 0

    # Logging/reproducibility
    log_every_n_steps: int = 1
    bf16: bool = True
    seed: int = 42

    def validate(self) -> None:
        if self.num_generations < 2:
            raise ValueError("GRPO requires num_generations >= 2")
        if self.per_device_batch_size < 1:
            raise ValueError("per_device_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.prompt_sampling_strategy not in {"random", "shuffle_cycle"}:
            raise ValueError(
                "prompt_sampling_strategy must be one of: random, shuffle_cycle"
            )
        if self.dynamic_sampling_max_attempts < 1:
            raise ValueError("dynamic_sampling_max_attempts must be >= 1")
        if self.dynamic_sampling_min_reward_std < 0:
            raise ValueError("dynamic_sampling_min_reward_std must be >= 0")
        if self.max_new_tokens < 1 or self.max_prompt_length < 1:
            raise ValueError("prompt and completion lengths must be positive")
        if self.max_sequence_length <= self.max_prompt_length:
            raise ValueError("max_sequence_length must exceed max_prompt_length")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_sequence_length < self.max_prompt_length + self.max_new_tokens:
            raise ValueError(
                "max_sequence_length must cover max_prompt_length + max_new_tokens"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.beta < 0:
            raise ValueError("beta must be >= 0")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")
        if self.log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be >= 1")
        if self.kl_estimator not in {"k1", "k2", "k3"}:
            raise ValueError("kl_estimator must be one of: k1, k2, k3")


class GRPOTrainer:
    """Native GRPO trainer for small experiments.

    Args:
        model: Trainable causal language model.  A PEFT model is recommended.
        ref_model: Frozen reference model.  May be ``None`` when ``model`` is a
            PEFT model exposing ``disable_adapter()``; the adapter-disabled
            model is then used as the reference without a second model copy.
        tokenizer: Hugging Face-compatible tokenizer.
        config: GRPO configuration.
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config: Optional[GRPOConfig] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()
        self.config.validate()

        self._adapter_reference = ref_model is None and callable(
            getattr(model, "disable_adapter", None)
        )
        if ref_model is None and not self._adapter_reference and self.config.beta > 0:
            raise ValueError(
                "ref_model is required unless model is a PEFT model exposing "
                "disable_adapter(), or beta is set to 0"
            )

        if self.ref_model is not None:
            for parameter in self.ref_model.parameters():
                parameter.requires_grad = False
            self.ref_model.eval()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.device = next(self.model.parameters()).device
        self._step = 0
        self._optimizer_step = 0
        self._metrics_history: List[Dict[str, float]] = []
        self._rng = random.Random(self.config.seed)
        self._prompt_cycle_key: Tuple[str, ...] = ()
        self._prompt_cycle_indices: List[int] = []
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def train(
        self,
        prompts: List[str],
        reward_fn: RewardFn,
        num_steps: int = 1000,
        reward_model: Optional[RewardFn] = None,
        eval_prompts: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """Train on text prompts using a deterministic/verifiable reward.

        ``reward_fn`` receives the exact prompt string supplied to generation,
        which allows callers to keep a private ``prompt -> ground truth`` map
        without embedding answers in model-visible text.
        """

        if not prompts:
            raise ValueError("prompts must not be empty")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        logger.info(
            "Starting GRPO: steps=%s G=%s beta=%s adapter_reference=%s",
            num_steps,
            self.config.num_generations,
            self.config.beta,
            self._adapter_reference,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        accumulated_gradient_steps = 0

        for local_step in range(num_steps):
            losses: List[torch.Tensor] = []
            kl_values: List[float] = []
            sampled_rewards: List[float] = []
            accepted_rewards: List[float] = []
            sampled_responses: List[str] = []
            informative_groups = 0
            group_reward_stds: List[float] = []
            group_size = self.config.num_generations
            sampled_groups = 0
            accepted_groups = 0
            sampling_attempts = 0
            target_groups = min(self.config.per_device_batch_size, len(prompts))
            max_attempts = (
                self.config.dynamic_sampling_max_attempts
                if self.config.dynamic_sampling
                else 1
            )

            while accepted_groups < target_groups and sampling_attempts < max_attempts:
                candidate_count = target_groups - accepted_groups
                batch_prompts = self._sample_prompts(prompts, candidate_count)
                if not batch_prompts:
                    break
                sampling_attempts += 1
                all_prompts = [
                    prompt
                    for prompt in batch_prompts
                    for _ in range(group_size)
                ]
                (
                    responses,
                    rollout_ids,
                    rollout_log_probs,
                    padded_prompt_ids,
                    padded_prompt_masks,
                    prompt_width,
                ) = self._generate_responses_with_log_probs(all_prompts)
                rewards = []
                for prompt, response in zip(all_prompts, responses):
                    reward = float(reward_fn(prompt, response))
                    if reward_model is not None:
                        reward += float(reward_model(prompt, response))
                    if not math.isfinite(reward):
                        raise ValueError("reward_fn returned a non-finite value")
                    rewards.append(reward)
                sampled_rewards.extend(rewards)
                sampled_responses.extend(responses)

                for group_index, group_prompt in enumerate(batch_prompts):
                    start = group_index * group_size
                    group_responses = responses[start : start + group_size]
                    group_rewards = rewards[start : start + group_size]
                    reward_std = float(
                        torch.tensor(group_rewards).std(unbiased=False)
                    )
                    group_reward_stds.append(reward_std)
                    sampled_groups += 1
                    informative = (
                        reward_std > self.config.dynamic_sampling_min_reward_std
                    )
                    if informative:
                        informative_groups += 1
                    if self.config.dynamic_sampling and not informative:
                        continue

                    accepted_groups += 1
                    accepted_rewards.extend(group_rewards)
                    advantages = self._compute_group_advantages(group_rewards)
                    for offset, (response, advantage) in enumerate(
                        zip(group_responses, advantages)
                    ):
                        seq_index = start + offset
                        result = self._compute_grpo_loss(
                            prompt=group_prompt,
                            response=response,
                            advantage=advantage,
                            response_ids=rollout_ids[seq_index],
                            old_log_probs=rollout_log_probs[seq_index],
                            padded_prompt_ids=padded_prompt_ids[seq_index],
                            padded_prompt_mask=padded_prompt_masks[seq_index],
                            prompt_width=prompt_width,
                        )
                        if result is None:
                            continue
                        loss, kl = result
                        losses.append(loss)
                        kl_values.append(kl)

            if losses:
                total_loss = torch.stack(losses).mean()
                scaled_loss = total_loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                accumulated_gradient_steps += 1
            else:
                total_loss = torch.zeros((), device=self.device)

            is_last = local_step + 1 == num_steps
            should_update = (
                accumulated_gradient_steps >= self.config.gradient_accumulation_steps
                or (is_last and accumulated_gradient_steps > 0)
            )
            grad_norm = 0.0
            if should_update:
                self._correct_partial_accumulation(accumulated_gradient_steps)
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                grad_norm = float(grad_norm_t.detach().cpu())
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_gradient_steps = 0
                self._optimizer_step += 1
                interval = self.config.ref_model_update_interval
                if interval > 0 and self._optimizer_step % interval == 0:
                    self._update_ref_model()

            self._step += 1
            metrics = {
                "step": self._step,
                "optimizer_step": self._optimizer_step,
                "loss": float(total_loss.detach().cpu()),
                "kl_divergence": sum(kl_values) / max(1, len(kl_values)),
                "reward_mean": sum(sampled_rewards) / max(1, len(sampled_rewards)),
                "reward_std": float(
                    torch.tensor(sampled_rewards).std(unbiased=False)
                ) if sampled_rewards else 0.0,
                "accepted_reward_mean": sum(accepted_rewards)
                / max(1, len(accepted_rewards)),
                "group_reward_std_mean": sum(group_reward_stds)
                / max(1, len(group_reward_stds)),
                "informative_group_rate": informative_groups
                / max(1, sampled_groups),
                "sampled_groups": sampled_groups,
                "accepted_groups": accepted_groups,
                "discarded_groups": sampled_groups - accepted_groups,
                "sampling_attempts": sampling_attempts,
                "sampled_rollouts": len(sampled_rewards),
                "effective_rollouts": accepted_groups * group_size,
                "mean_completion_chars": sum(map(len, sampled_responses))
                / max(1, len(sampled_responses)),
                "grad_norm": grad_norm,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self._metrics_history.append(metrics)

            if self._step % self.config.log_every_n_steps == 0:
                logger.info(
                    "step=%d opt_step=%d loss=%.4f kl=%.4f reward=%.3f "
                    "informative=%.2f accepted=%d/%d",
                    self._step,
                    self._optimizer_step,
                    metrics["loss"],
                    metrics["kl_divergence"],
                    metrics["reward_mean"],
                    metrics["informative_group_rate"],
                    metrics["accepted_groups"],
                    metrics["sampled_groups"],
                )

            if (
                eval_prompts
                and self._step % (self.config.log_every_n_steps * 5) == 0
            ):
                logger.info("Eval: %s", self.evaluate(eval_prompts, reward_fn))

        logger.info("GRPO training complete")
        return self._metrics_history

    def _correct_partial_accumulation(self, accumulated_steps: int) -> None:
        """Turn a final partial sum of ``loss / target`` into its actual mean."""

        target = self.config.gradient_accumulation_steps
        if accumulated_steps < 1 or accumulated_steps >= target:
            return
        correction = target / accumulated_steps
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    parameter.grad.mul_(correction)

    def _generate_responses(self, prompts: Sequence[str]) -> List[str]:
        responses, *_ = self._generate_responses_with_log_probs(prompts)
        return responses

    def _generate_responses_with_log_probs(self, prompts: Sequence[str]):
        """Generate completions; return texts, token ids, and per-token
        log probabilities of the sampled tokens under the rollout policy.

        The log probabilities are scored with a no-grad forward pass over the
        sampled token stream (policy frozen at rollout time, eval mode) using
        the model's raw next-token distribution, without sampling warps
        (temperature/top_p).  This keeps both sides of the GRPO ratio
        (old log probs here, new log probs in the update) on the same
        distribution; ``outputs.scores`` is not used because in current
        transformers versions it already contains warped (top_p-masked)
        logits.
        """
        previous_training = self.model.training
        self.model.eval()
        previous_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_length,
                add_special_tokens=False,
            ).to(self.device)
            prompt_width = inputs["input_ids"].shape[1]
            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.temperature > 0,
                "repetition_penalty": self.config.repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.config.temperature > 0:
                generation_kwargs.update(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, return_dict_in_generate=True, **generation_kwargs
                )

            with torch.inference_mode():
                scored = self.model(
                    input_ids=outputs.sequences,
                    attention_mask=torch.cat(
                        [
                            inputs["attention_mask"],
                            torch.ones_like(outputs.sequences[:, prompt_width:]),
                        ],
                        dim=1,
                    ),
                    use_cache=False,
                ).logits
            completion_log_probs = F.log_softmax(
                scored[:, prompt_width - 1 : -1, :].float(), dim=-1
            ).gather(
                dim=-1,
                index=outputs.sequences[:, prompt_width:].unsqueeze(-1),
            ).squeeze(-1)

            pad_id = self.tokenizer.pad_token_id
            responses = []
            rollout_ids = []
            rollout_log_probs = []
            for seq_idx, generated in enumerate(outputs.sequences[:, prompt_width:]):
                # Left-padded batches pad already-finished sequences; those
                # positions were never sampled, so trim at the first pad.
                if pad_id is not None:
                    pad_positions = (generated == pad_id).nonzero()
                    if len(pad_positions):
                        generated = generated[: int(pad_positions[0])]
                responses.append(
                    self.tokenizer.decode(
                        generated, skip_special_tokens=True
                    ).strip()
                )
                rollout_ids.append(generated)
                rollout_log_probs.append(
                    completion_log_probs[seq_idx, : len(generated)]
                )
            # The padded prompt rows accompany the rollout log probabilities:
            # the loss forward must reuse them so both sides of the ratio see
            # the same absolute (RoPE) positions, otherwise left padding
            # shifts the completion tokens relative to an unpadded forward.
            return (
                responses,
                rollout_ids,
                rollout_log_probs,
                inputs["input_ids"],
                inputs["attention_mask"],
                prompt_width,
            )
        finally:
            self.tokenizer.padding_side = previous_padding_side
            self.model.train(previous_training)

    def _compute_group_advantages(self, rewards: Sequence[float]) -> List[float]:
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if self.config.reward_clip > 0:
            rewards_t = torch.clamp(
                rewards_t,
                -self.config.reward_clip,
                self.config.reward_clip,
            )
        centered = rewards_t - rewards_t.mean()
        if self.config.reward_normalize:
            std = rewards_t.std(unbiased=False)
            if std > 1e-8:
                centered = centered / (std + 1e-8)
        return centered.tolist()

    def _tokenize_trajectory(
        self, prompt: str, response: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_prompt_length,
        )["input_ids"]
        remaining = self.config.max_sequence_length - len(prompt_ids)
        if remaining <= 0:
            return None
        response_ids = self.tokenizer(
            response,
            add_special_tokens=False,
            truncation=True,
            max_length=remaining,
        )["input_ids"]
        if not response_ids:
            return None
        input_ids = torch.tensor(
            [prompt_ids + response_ids], dtype=torch.long, device=self.device
        )
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, len(prompt_ids)

    @staticmethod
    def _completion_log_probs(
        logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int
    ) -> torch.Tensor:
        completion_logits = logits[:, prompt_len - 1 : -1, :]
        completion_labels = input_ids[:, prompt_len:]
        log_probs = F.log_softmax(completion_logits.float(), dim=-1)
        return torch.gather(
            log_probs,
            dim=-1,
            index=completion_labels.unsqueeze(-1),
        ).squeeze(-1)

    def _reference_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.ref_model is not None:
            with torch.no_grad():
                return self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
        if self._adapter_reference:
            with torch.no_grad(), self.model.disable_adapter():
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
        return None

    def _compute_grpo_loss(
        self,
        prompt: str,
        response: str,
        advantage: float,
        response_ids: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        padded_prompt_ids: Optional[torch.Tensor] = None,
        padded_prompt_mask: Optional[torch.Tensor] = None,
        prompt_width: Optional[int] = None,
    ) -> Optional[Tuple[torch.Tensor, float]]:
        if response_ids is not None:
            # Reuse the rollout-time padded prompt so the forward sees the
            # same absolute positions as the rollout log probabilities; a
            # fresh unpadded tokenization would shift RoPE positions and
            # bias the ratio.
            remaining = self.config.max_sequence_length - prompt_width
            if remaining <= 0:
                return None
            response_ids = response_ids[:remaining]
            if old_log_probs is not None:
                old_log_probs = old_log_probs[:remaining].to(self.device)
            if response_ids.numel() == 0:
                return None
            input_ids = torch.cat([padded_prompt_ids, response_ids]).unsqueeze(0)
            attention_mask = torch.cat(
                [padded_prompt_mask, torch.ones_like(response_ids)]
            ).unsqueeze(0)
            prompt_len = prompt_width
        else:
            trajectory = self._tokenize_trajectory(prompt, response)
            if trajectory is None:
                return None
            input_ids, attention_mask, prompt_len = trajectory

        policy_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        policy_log_probs = self._completion_log_probs(
            policy_logits, input_ids, prompt_len
        )

        # The old log probabilities are those of the sampled tokens at rollout
        # time (rescored while the policy is frozen), not tensors from the
        # differentiable update forward pass.
        if old_log_probs is None:
            old_log_probs = policy_log_probs.detach()
        log_ratio_old = torch.clamp(policy_log_probs - old_log_probs, -20.0, 20.0)
        ratio = torch.exp(log_ratio_old)
        advantage_t = torch.as_tensor(
            advantage, dtype=policy_log_probs.dtype, device=self.device
        )
        unclipped = ratio * advantage_t
        clipped = torch.clamp(
            ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon
        ) * advantage_t
        policy_loss = -torch.minimum(unclipped, clipped).mean()

        reference_logits = self._reference_logits(input_ids, attention_mask)
        if reference_logits is None or self.config.beta == 0:
            kl = torch.zeros((), dtype=policy_loss.dtype, device=self.device)
        else:
            reference_log_probs = self._completion_log_probs(
                reference_logits, input_ids, prompt_len
            ).detach()
            # log(ref / policy), matching the common sampled KL estimators.
            ref_policy_log_ratio = torch.clamp(
                reference_log_probs - policy_log_probs, -20.0, 20.0
            )
            kl = self._estimate_kl(ref_policy_log_ratio).mean()

        loss = policy_loss + self.config.beta * kl
        return loss, float(kl.detach().cpu())

    def _estimate_kl(self, ref_policy_log_ratio: torch.Tensor) -> torch.Tensor:
        if self.config.kl_estimator == "k1":
            return -ref_policy_log_ratio
        if self.config.kl_estimator == "k2":
            return 0.5 * ref_policy_log_ratio.square()
        return torch.exp(ref_policy_log_ratio) - ref_policy_log_ratio - 1.0

    def _sample_prompts(self, prompts: Sequence[str], batch_size: int) -> List[str]:
        count = min(batch_size, len(prompts))
        if count < 1:
            return []
        if self.config.prompt_sampling_strategy == "random":
            indices = self._rng.sample(range(len(prompts)), count)
            return [prompts[index] for index in indices]

        key = tuple(prompts)
        if key != self._prompt_cycle_key:
            self._prompt_cycle_key = key
            self._prompt_cycle_indices = []
        indices: List[int] = []
        while len(indices) < count:
            if not self._prompt_cycle_indices:
                self._prompt_cycle_indices = list(range(len(prompts)))
                self._rng.shuffle(self._prompt_cycle_indices)
            take = min(count - len(indices), len(self._prompt_cycle_indices))
            indices.extend(self._prompt_cycle_indices[:take])
            del self._prompt_cycle_indices[:take]
        return [prompts[index] for index in indices]

    def _update_ref_model(self) -> None:
        if self.ref_model is None:
            logger.debug("Adapter-disabled reference is fixed; update skipped")
            return
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        logger.debug("Reference model updated")

    @torch.no_grad()
    def evaluate(self, eval_prompts: List[str], reward_fn: RewardFn) -> Dict[str, float]:
        if not eval_prompts:
            return {"eval_best_of_g_reward": 0.0, "eval_avg_reward": 0.0, "num_samples": 0}
        sample = eval_prompts[:20]
        best_rewards = []
        all_rewards = []
        for prompt in sample:
            responses = self._generate_responses(
                [prompt] * self.config.num_generations
            )
            rewards = [float(reward_fn(prompt, response)) for response in responses]
            best_rewards.append(max(rewards))
            all_rewards.extend(rewards)
        return {
            "eval_best_of_g_reward": sum(best_rewards) / len(best_rewards),
            "eval_avg_reward": sum(all_rewards) / len(all_rewards),
            "num_samples": len(sample),
        }

    @torch.no_grad()
    def profile_prompts(
        self,
        prompts: Sequence[str],
        reward_fn: RewardFn,
        max_prompts: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """Measure source-side group learnability before training.

        A prompt is informative when its sampled group has non-zero reward
        variance. This method uses only the source verifier and returns raw
        responses so any smoke-run curriculum remains auditable.
        """

        selected = list(prompts)
        if max_prompts is not None:
            selected = selected[:max_prompts]
        profiles: List[Dict[str, object]] = []
        for prompt in selected:
            responses = self._generate_responses(
                [prompt] * self.config.num_generations
            )
            rewards = [float(reward_fn(prompt, response)) for response in responses]
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            reward_std = float(reward_tensor.std(unbiased=False))
            profiles.append(
                {
                    "prompt": prompt,
                    "responses": responses,
                    "rewards": rewards,
                    "reward_mean": float(reward_tensor.mean()),
                    "reward_std": reward_std,
                    "informative": reward_std
                    > self.config.dynamic_sampling_min_reward_std,
                }
            )
        return profiles

    def save(self, path: str) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output)
        self.tokenizer.save_pretrained(output)
        logger.info("Model saved to %s", output)

    def get_metrics_df(self):
        try:
            import pandas as pd

            return pd.DataFrame(self._metrics_history)
        except ImportError:
            return self._metrics_history

    @staticmethod
    def reward_length(
        target_min: int = 100,
        target_max: int = 2000,
        penalty_rate: float = 0.001,
    ) -> RewardFn:
        def _fn(prompt: str, response: str) -> float:
            del prompt
            length = len(response)
            if target_min <= length <= target_max:
                return 1.0
            if length < target_min:
                return max(0.0, length / target_min) * 0.5
            return max(0.0, 1.0 - (length - target_max) * penalty_rate)

        return _fn

    @staticmethod
    def reward_format(
        required_patterns: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
    ) -> RewardFn:
        required = required_patterns or []
        forbidden = forbidden_patterns or []

        def _fn(prompt: str, response: str) -> float:
            del prompt
            score = 1.0
            for pattern in required:
                if not re.search(pattern, response):
                    score -= 0.3
            for pattern in forbidden:
                if re.search(pattern, response):
                    score -= 0.3
            return max(0.0, score)

        return _fn

    @staticmethod
    def reward_combined(
        *reward_fns: RewardFn,
        weights: Optional[List[float]] = None,
    ) -> RewardFn:
        active_weights = weights or [1.0] * len(reward_fns)
        if len(active_weights) != len(reward_fns):
            raise ValueError("weights and reward_fns must have the same length")

        def _fn(prompt: str, response: str) -> float:
            return sum(
                weight * reward_fn(prompt, response)
                for reward_fn, weight in zip(reward_fns, active_weights)
            )

        return _fn

    @staticmethod
    def reward_math_accuracy(
        extract_answer: Optional[Callable[[str], str]] = None,
    ) -> RewardFn:
        """Build a simple exact-answer reward for math-style prompts."""

        answer_pattern = re.compile(
            r"(?:answer|答案)(?:\s+is|是|为)?\s*[:：]?\s*([^\s,，。]+)",
            flags=re.IGNORECASE,
        )

        def _fn(prompt: str, response: str) -> float:
            if extract_answer is not None:
                try:
                    expected = extract_answer(prompt)
                    predicted = extract_answer(response)
                    return float(str(expected).strip() == str(predicted).strip())
                except Exception:
                    return 0.0
            expected_match = answer_pattern.search(prompt)
            predicted_match = answer_pattern.search(response)
            if not expected_match or not predicted_match:
                return 0.0
            return float(expected_match.group(1) == predicted_match.group(1))

        return _fn
