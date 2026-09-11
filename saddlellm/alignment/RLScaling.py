"""Slime-inspired RL scaling utilities for long-reasoning post-training.

The design borrows the useful abstraction from modern RL post-training systems:
separate rollout generation, reward evaluation, data-buffer filtering, and the
actual policy update.  It intentionally stays lightweight so it can run with the
existing SaddleLLM GRPO trainer, while leaving hooks for vLLM/SGLang/Megatron
style backends later.
"""
import json
import logging
import math
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


RewardFn = Callable[[str, str], float]
RolloutFn = Callable[[List[str], Dict], List[str]]


@dataclass
class RolloutSegment:
    """A semantic span inside an agent rollout for process-level credit."""

    role: str
    text: str
    start: int = 0
    end: int = 0
    reward: float = 0.0
    credit: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class RolloutSample:
    prompt: str
    response: str
    reward: float
    advantage: float = 0.0
    accepted: bool = True
    reason: str = ""
    process_reward: float = 0.0
    segments: List[RolloutSegment] = field(default_factory=list)
    token_credit: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)

    def to_sft_record(self) -> Dict[str, str]:
        return {
            "instruction": self.prompt,
            "output": self.response,
            "reward": self.reward,
        }


@dataclass
class RLScalingConfig:
    num_generations: int = 8
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    min_reward: float = 0.0
    min_response_chars: int = 16
    max_response_chars: int = 12000
    drop_low_variance_groups: bool = True
    min_group_reward_std: float = 1e-4
    keep_top_k_per_prompt: Optional[int] = None
    seed: int = 42
    use_triage_credit: bool = False
    triage_outcome_weight: float = 0.7
    triage_process_weight: float = 0.3


@dataclass
class TriageConfig:
    """Configuration for TRIAGE-style segment credit assignment."""

    outcome_weight: float = 0.7
    process_weight: float = 0.3
    role_weights: Dict[str, float] = field(default_factory=lambda: {
        "search": 0.4,
        "click": 0.2,
        "edit": 0.3,
        "navigate": 0.2,
        "tool": 0.3,
        "reason": 0.3,
        "answer": 0.6,
        "other": 0.0,
    })
    split_blank_lines: bool = True
    min_segment_chars: int = 4


class ActionRoleClassifier:
    """Heuristic role classifier for agent traces.

    The default implementation is intentionally lightweight. Production users
    can replace it with a model-based classifier while keeping the same
    RolloutSegment interface.
    """

    ROLE_PATTERNS = {
        "search": [
            r"\bsearch\b", r"\bquery\b", r"\bgoogle\b", r"\blook up\b",
            r"搜索", r"查询", r"检索",
        ],
        "click": [
            r"\bclick\b", r"\bopen\b", r"\bselect\b", r"\bfollow link\b",
            r"点击", r"打开", r"选择",
        ],
        "edit": [
            r"\bedit\b", r"\bpatch\b", r"\bmodify\b", r"\bwrite\b", r"\breplace\b",
            r"编辑", r"修改", r"写入", r"替换",
        ],
        "navigate": [
            r"\bnavigate\b", r"\bgo to\b", r"\bscroll\b", r"\bback\b", r"\bnext\b",
            r"跳转", r"滚动", r"返回", r"下一页",
        ],
        "tool": [
            r"\btool\b", r"\bcall\b", r"\bexecute\b", r"\brun\b", r"\bshell\b",
            r"调用", r"执行", r"运行",
        ],
        "reason": [
            r"\bbecause\b", r"\btherefore\b", r"\bplan\b", r"\banalyze\b", r"\bthink\b",
            r"因为", r"所以", r"分析", r"计划", r"推理",
        ],
        "answer": [
            r"\bfinal\b", r"\banswer\b", r"\bconclusion\b", r"\bresult\b",
            r"答案", r"结论", r"最终",
        ],
    }

    def __init__(self, config: Optional[TriageConfig] = None):
        self.config = config or TriageConfig()

    def segment(self, text: str) -> List[RolloutSegment]:
        raw_parts = self._split(text)
        segments: List[RolloutSegment] = []
        cursor = 0
        for part in raw_parts:
            start = text.find(part, cursor)
            if start < 0:
                start = cursor
            end = start + len(part)
            cursor = end
            role = self.classify(part)
            segments.append(RolloutSegment(role=role, text=part, start=start, end=end))
        return segments

    def classify(self, text: str) -> str:
        lowered = text.lower()
        best_role = "other"
        best_hits = 0
        for role, patterns in self.ROLE_PATTERNS.items():
            hits = sum(1 for pattern in patterns if re.search(pattern, lowered, re.I))
            if hits > best_hits:
                best_role = role
                best_hits = hits
        return best_role

    def _split(self, text: str) -> List[str]:
        if not text:
            return []
        if self.config.split_blank_lines:
            parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        else:
            parts = [p.strip() for p in text.splitlines() if p.strip()]
        if len(parts) <= 1:
            parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text) if p.strip()]
        return [p for p in parts if len(p) >= self.config.min_segment_chars] or [text.strip()]


class TriageCreditAssigner:
    """Assign role-aware process credit to rollout segments."""

    def __init__(
        self,
        config: Optional[TriageConfig] = None,
        role_reward_fn: Optional[Callable[[RolloutSegment, RolloutSample], float]] = None,
        classifier: Optional[ActionRoleClassifier] = None,
    ):
        self.config = config or TriageConfig()
        self.classifier = classifier or ActionRoleClassifier(self.config)
        self.role_reward_fn = role_reward_fn

    def assign(self, sample: RolloutSample) -> RolloutSample:
        if not sample.segments:
            sample.segments = self.classifier.segment(sample.response)
        if not sample.segments:
            sample.process_reward = 0.0
            sample.token_credit = []
            return sample

        credits = []
        for segment in sample.segments:
            role_prior = self.config.role_weights.get(segment.role, self.config.role_weights.get("other", 0.0))
            role_reward = self.role_reward_fn(segment, sample) if self.role_reward_fn else role_prior
            segment.reward = float(role_reward)
            segment.credit = (
                self.config.outcome_weight * float(sample.reward)
                + self.config.process_weight * float(role_reward)
            )
            credits.append(segment.credit)

        sample.process_reward = sum(credits) / max(1, len(credits))
        sample.token_credit = self._expand_token_credit(sample)
        sample.metadata["triage_roles"] = self.role_histogram(sample)
        sample.metadata["triage_process_reward"] = sample.process_reward
        return sample

    @staticmethod
    def role_histogram(sample: RolloutSample) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for segment in sample.segments:
            hist[segment.role] = hist.get(segment.role, 0) + 1
        return hist

    def _expand_token_credit(self, sample: RolloutSample) -> List[float]:
        token_credits: List[float] = []
        for segment in sample.segments:
            approx_tokens = max(1, len(re.findall(r"\S+", segment.text)))
            token_credits.extend([segment.credit] * approx_tokens)
        return token_credits


class TriageRolloutProcessor:
    """Batch helper for annotating rollouts with TRIAGE-style credit."""

    def __init__(self, assigner: Optional[TriageCreditAssigner] = None):
        self.assigner = assigner or TriageCreditAssigner()

    def process(self, samples: Iterable[RolloutSample]) -> List[RolloutSample]:
        return [self.assigner.assign(sample) for sample in samples]


class RewardManager:
    """Composable reward functions for reasoning RL."""

    def __init__(self, reward_fns: Sequence[RewardFn], weights: Optional[Sequence[float]] = None):
        if not reward_fns:
            raise ValueError("RewardManager requires at least one reward function")
        self.reward_fns = list(reward_fns)
        self.weights = list(weights or [1.0] * len(reward_fns))
        if len(self.reward_fns) != len(self.weights):
            raise ValueError("reward_fns and weights must have the same length")

    def __call__(self, prompt: str, response: str) -> float:
        total = 0.0
        weight_sum = 0.0
        for fn, weight in zip(self.reward_fns, self.weights):
            total += float(fn(prompt, response)) * weight
            weight_sum += abs(weight)
        return total / max(weight_sum, 1e-8)

    @staticmethod
    def length_reward(target_min: int = 200, target_max: int = 4000) -> RewardFn:
        def _reward(prompt: str, response: str) -> float:
            n = len(response)
            if target_min <= n <= target_max:
                return 1.0
            if n < target_min:
                return max(0.0, n / max(target_min, 1))
            return max(0.0, 1.0 - (n - target_max) / max(target_max, 1))
        return _reward

    @staticmethod
    def reasoning_format_reward() -> RewardFn:
        patterns = [
            r"(therefore|so|because|step|first|second|finally)",
            r"(因此|所以|因为|步骤|首先|其次|最后)",
        ]

        def _reward(prompt: str, response: str) -> float:
            if not response.strip():
                return 0.0
            score = 0.2
            if any(re.search(p, response, re.I) for p in patterns):
                score += 0.4
            if re.search(r"```|\\boxed|答案|answer", response, re.I):
                score += 0.2
            if not _has_bad_repetition(response):
                score += 0.2
            return min(1.0, score)
        return _reward

    @staticmethod
    def exact_answer_reward(answer_extractor: Callable[[str], str], target_lookup: Dict[str, str]) -> RewardFn:
        def _reward(prompt: str, response: str) -> float:
            expected = target_lookup.get(prompt)
            if expected is None:
                return 0.0
            predicted = answer_extractor(response)
            return 1.0 if str(predicted).strip() == str(expected).strip() else 0.0
        return _reward


class RolloutBuffer:
    """Stores and filters rollout samples before policy update."""

    def __init__(self):
        self.samples: List[RolloutSample] = []

    def add_many(self, samples: Iterable[RolloutSample]):
        self.samples.extend(samples)

    def accepted(self) -> List[RolloutSample]:
        return [s for s in self.samples if s.accepted]

    def by_prompt(self) -> Dict[str, List[RolloutSample]]:
        grouped: Dict[str, List[RolloutSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.prompt, []).append(sample)
        return grouped

    def save_jsonl(self, path: str, accepted_only: bool = True) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = self.accepted() if accepted_only else self.samples
        with open(path, "w", encoding="utf-8") as f:
            for sample in rows:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
        return path

    def to_sft_dataset(self) -> List[Dict[str, str]]:
        return [sample.to_sft_record() for sample in self.accepted()]


class RLScalingTrainer:
    """Rollout/reward/filter loop for long-reasoning RL scaling."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        reward_manager: Optional[RewardManager] = None,
        rollout_fn: Optional[RolloutFn] = None,
        config: Optional[RLScalingConfig] = None,
        triage_assigner: Optional[TriageCreditAssigner] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_manager = reward_manager
        self.rollout_fn = rollout_fn
        self.config = config or RLScalingConfig()
        self.triage_assigner = triage_assigner
        self.buffer = RolloutBuffer()
        random.seed(self.config.seed)

    def collect(self, prompts: Sequence[str], reward_fn: Optional[RewardFn] = None) -> RolloutBuffer:
        rewarder = reward_fn or self.reward_manager
        if rewarder is None:
            raise ValueError("collect() requires reward_fn or reward_manager")

        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * self.config.num_generations)

        responses = self._rollout(expanded_prompts)
        samples = [
            RolloutSample(prompt=p, response=r, reward=float(rewarder(p, r)))
            for p, r in zip(expanded_prompts, responses)
        ]
        if self.config.use_triage_credit:
            if self.triage_assigner is None:
                self.triage_assigner = TriageCreditAssigner(TriageConfig(
                    outcome_weight=self.config.triage_outcome_weight,
                    process_weight=self.config.triage_process_weight,
                ))
            for sample in samples:
                self.triage_assigner.assign(sample)
                sample.metadata["outcome_reward"] = sample.reward
                sample.reward = (
                    self.config.triage_outcome_weight * sample.reward
                    + self.config.triage_process_weight * sample.process_reward
                )
        self._annotate_group_advantages(samples)
        self._filter_samples(samples)
        self.buffer.add_many(samples)
        logger.info(
            "Collected %s rollouts, accepted %s",
            len(samples),
            sum(1 for sample in samples if sample.accepted),
        )
        return self.buffer

    def train_grpo(
        self,
        ref_model,
        reward_fn: Optional[RewardFn] = None,
        num_steps: int = 100,
        grpo_config=None,
    ):
        """Run existing SaddleLLM GRPO on accepted prompts."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("train_grpo requires model and tokenizer")
        accepted = self.buffer.accepted()
        if not accepted:
            raise ValueError("No accepted rollouts. Call collect() first or relax filters.")

        from .GRPOTrainer import GRPOConfig, GRPOTrainer

        config = grpo_config or GRPOConfig(
            num_generations=self.config.num_generations,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        prompts = sorted({sample.prompt for sample in accepted})
        rewarder = reward_fn or self.reward_manager
        if rewarder is None:
            raise ValueError("train_grpo requires reward_fn or reward_manager")

        trainer = GRPOTrainer(
            model=self.model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            config=config,
        )
        return trainer.train(prompts=prompts, reward_fn=rewarder, num_steps=num_steps)

    def export_sft_jsonl(self, path: str, top_k_per_prompt: Optional[int] = None) -> str:
        top_k = top_k_per_prompt or self.config.keep_top_k_per_prompt
        samples = self.buffer.accepted()
        if top_k:
            selected = []
            for group in self.buffer.by_prompt().values():
                selected.extend(
                    sorted([s for s in group if s.accepted], key=lambda s: s.reward, reverse=True)[:top_k]
                )
            samples = selected

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_sft_record(), ensure_ascii=False) + "\n")
        return path

    def _rollout(self, prompts: List[str]) -> List[str]:
        params = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.rollout_fn is not None:
            return self.rollout_fn(prompts, params)
        if self.model is None or self.tokenizer is None:
            raise ValueError("No rollout_fn provided and model/tokenizer are missing")

        import torch

        device = next(self.model.parameters()).device
        responses = []
        self.model.eval()
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(text[len(prompt):].strip() if text.startswith(prompt) else text)
        self.model.train()
        return responses

    def _annotate_group_advantages(self, samples: List[RolloutSample]):
        grouped: Dict[str, List[RolloutSample]] = {}
        for sample in samples:
            grouped.setdefault(sample.prompt, []).append(sample)

        for group in grouped.values():
            rewards = [s.reward for s in group]
            mean = sum(rewards) / max(1, len(rewards))
            variance = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards))
            std = math.sqrt(variance)
            for sample in group:
                sample.advantage = 0.0 if std < 1e-8 else (sample.reward - mean) / std
                sample.metadata["group_reward_mean"] = mean
                sample.metadata["group_reward_std"] = std

    def _filter_samples(self, samples: List[RolloutSample]):
        grouped: Dict[str, List[RolloutSample]] = {}
        for sample in samples:
            grouped.setdefault(sample.prompt, []).append(sample)

        for group in grouped.values():
            group_std = group[0].metadata.get("group_reward_std", 0.0)
            drop_group = self.config.drop_low_variance_groups and group_std < self.config.min_group_reward_std
            ranked = sorted(group, key=lambda s: s.reward, reverse=True)
            keep_set = set(id(s) for s in ranked[: self.config.keep_top_k_per_prompt]) if self.config.keep_top_k_per_prompt else None

            for sample in group:
                if drop_group:
                    sample.accepted = False
                    sample.reason = "low_group_reward_variance"
                elif sample.reward < self.config.min_reward:
                    sample.accepted = False
                    sample.reason = "reward_below_threshold"
                elif len(sample.response) < self.config.min_response_chars:
                    sample.accepted = False
                    sample.reason = "response_too_short"
                elif len(sample.response) > self.config.max_response_chars:
                    sample.accepted = False
                    sample.reason = "response_too_long"
                elif _has_bad_repetition(sample.response):
                    sample.accepted = False
                    sample.reason = "bad_repetition"
                elif keep_set is not None and id(sample) not in keep_set:
                    sample.accepted = False
                    sample.reason = "not_top_k"


def _has_bad_repetition(text: str) -> bool:
    if len(text) < 80:
        return False
    if re.search(r"(.{8,80})\1{3,}", text, re.S):
        return True
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 40:
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if trigrams and len(set(trigrams)) / len(trigrams) < 0.35:
            return True
    return False
