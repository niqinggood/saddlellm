"""On-policy multi-teacher distillation for post-training.

This module adds a MOPD-style control plane without forcing a specific heavy
trainer.  The important contract is that teachers supervise responses sampled
from the current student policy, not a static offline prompt/answer set.
"""
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .UniversalDistiller import TeacherInterface


StudentRolloutFn = Callable[[List[str], Dict], List[str]]
TeacherScoreFn = Callable[[str, str, str], float]


@dataclass
class OnPolicyTeacherSpec:
    name: str
    teacher: Optional[TeacherInterface] = None
    domain: str = "general"
    weight: float = 1.0
    instruction: str = ""


@dataclass
class TeacherPolicySignal:
    teacher_name: str
    domain: str
    prompt: str
    student_response: str
    teacher_response: str
    score: float = 0.0
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MOPDConfig:
    num_rollouts_per_prompt: int = 1
    max_new_tokens: int = 1024
    student_temperature: float = 0.9
    student_top_p: float = 0.95
    teacher_temperature: float = 0.3
    teacher_max_tokens: int = 1024
    aggregation: str = "best_score"  # best_score | weighted_round_robin
    output_dir: str = "./mopd"
    seed: int = 42
    include_student_response_in_prompt: bool = True


class MultiTeacherOnPolicyDistiller:
    """Collect MOPD-style on-policy teacher signals.

    A typical cycle is:
      prompts -> current student rollout -> each teacher improves/critiques the
      student response -> aggregate teacher behavior -> export SFT-style JSONL.
    """

    def __init__(
        self,
        teachers: Sequence[OnPolicyTeacherSpec],
        student_model=None,
        student_tokenizer=None,
        rollout_fn: Optional[StudentRolloutFn] = None,
        score_fn: Optional[TeacherScoreFn] = None,
        config: Optional[MOPDConfig] = None,
    ):
        if not teachers:
            raise ValueError("MultiTeacherOnPolicyDistiller requires at least one teacher")
        self.teachers = list(teachers)
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.rollout_fn = rollout_fn
        self.score_fn = score_fn
        self.config = config or MOPDConfig()
        self.records: List[Dict] = []
        random.seed(self.config.seed)

    def collect(self, prompts: Sequence[str]) -> List[Dict]:
        expanded = []
        for prompt in prompts:
            expanded.extend([prompt] * self.config.num_rollouts_per_prompt)
        student_responses = self._student_rollout(expanded)

        batch_records = []
        for prompt, student_response in zip(expanded, student_responses):
            signals = [
                self._teacher_signal(spec, prompt, student_response)
                for spec in self.teachers
            ]
            selected = self._aggregate(signals)
            record = {
                "prompt": prompt,
                "student_response": student_response,
                "teacher_signals": [signal.to_dict() for signal in signals],
                "selected_teacher": selected.teacher_name,
                "selected_domain": selected.domain,
                "instruction": prompt,
                "output": selected.teacher_response,
                "task": "mopd",
                "metadata": {
                    "on_policy": True,
                    "aggregation": self.config.aggregation,
                    "selected_score": selected.score,
                    "student_response_chars": len(student_response or ""),
                },
            }
            batch_records.append(record)
        self.records.extend(batch_records)
        return batch_records

    def save_jsonl(self, path: Optional[str] = None, records: Optional[Iterable[Dict]] = None) -> str:
        path = path or os.path.join(self.config.output_dir, "mopd_on_policy.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = list(records) if records is not None else self.records
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def planning_summary(self) -> Dict:
        return {
            "stage": "mopd",
            "status": "planned",
            "num_teachers": len(self.teachers),
            "teachers": [
                {"name": t.name, "domain": t.domain, "weight": t.weight}
                for t in self.teachers
            ],
            "config": asdict(self.config),
            "notes": [
                "Teachers supervise student-generated rollouts.",
                "Exported JSONL is SFT-compatible through instruction/output fields.",
                "Use score_fn for domain-specific teacher selection when available.",
            ],
        }

    def _student_rollout(self, prompts: List[str]) -> List[str]:
        params = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.student_temperature,
            "top_p": self.config.student_top_p,
        }
        if self.rollout_fn is not None:
            return self.rollout_fn(prompts, params)
        if self.student_model is None or self.student_tokenizer is None:
            return ["" for _ in prompts]

        import torch

        device = next(self.student_model.parameters()).device
        self.student_model.eval()
        responses = []
        for prompt in prompts:
            inputs = self.student_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                output = self.student_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.student_temperature,
                    top_p=self.config.student_top_p,
                    do_sample=self.config.student_temperature > 0,
                    pad_token_id=self.student_tokenizer.pad_token_id or self.student_tokenizer.eos_token_id,
                )
            text = self.student_tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(text[len(prompt):].strip() if text.startswith(prompt) else text)
        self.student_model.train()
        return responses

    def _teacher_signal(
        self,
        spec: OnPolicyTeacherSpec,
        prompt: str,
        student_response: str,
    ) -> TeacherPolicySignal:
        teacher_prompt = self._build_teacher_prompt(spec, prompt, student_response)
        if spec.teacher is None:
            teacher_response = student_response
        else:
            teacher_response = spec.teacher.generate(
                teacher_prompt,
                temperature=self.config.teacher_temperature,
                max_tokens=self.config.teacher_max_tokens,
            )
        score = self._score(prompt, student_response, teacher_response)
        return TeacherPolicySignal(
            teacher_name=spec.name,
            domain=spec.domain,
            prompt=prompt,
            student_response=student_response,
            teacher_response=teacher_response,
            score=score,
            weight=spec.weight,
        )

    def _build_teacher_prompt(self, spec: OnPolicyTeacherSpec, prompt: str, student_response: str) -> str:
        lead = spec.instruction or (
            "Improve the student's response for your domain. Return only the improved answer."
        )
        if not self.config.include_student_response_in_prompt:
            return f"{lead}\n\nPrompt:\n{prompt}"
        return f"{lead}\n\nPrompt:\n{prompt}\n\nStudent response:\n{student_response}"

    def _score(self, prompt: str, student_response: str, teacher_response: str) -> float:
        if self.score_fn is not None:
            return float(self.score_fn(prompt, student_response, teacher_response))
        if not teacher_response:
            return 0.0
        length_score = min(1.0, len(teacher_response) / 512.0)
        improvement = 0.2 if len(teacher_response) > len(student_response or "") else 0.0
        return min(1.0, 0.5 + length_score * 0.3 + improvement)

    def _aggregate(self, signals: Sequence[TeacherPolicySignal]) -> TeacherPolicySignal:
        if self.config.aggregation == "weighted_round_robin":
            total = sum(max(0.0, s.weight) for s in signals)
            if total <= 0:
                return random.choice(list(signals))
            pick = random.random() * total
            cursor = 0.0
            for signal in signals:
                cursor += max(0.0, signal.weight)
                if cursor >= pick:
                    return signal
            return signals[-1]
        return max(signals, key=lambda s: (s.score * max(s.weight, 1e-8), s.score))
