"""Minimal end-to-end RLVR pilot for the AASB public-policy datasets.

This module turns the research design into an executable experiment:

1. build leakage-controlled A/B train and benchmark partitions;
2. evaluate a frozen Qwen3 policy on Benchmark A and Benchmark B;
3. run GRPO with a deterministic Dataset-A verifier;
4. evaluate the updated policy with the same prompts and decoding settings;
5. persist predictions, metrics, transfer deltas, and reproducibility metadata.

The implementation is intentionally small.  It validates the experimental
framework and code path; a smoke run is not evidence for a scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .GRPOTrainer import GRPOConfig, GRPOTrainer


logger = logging.getLogger(__name__)


A_LABELS: Tuple[str, ...] = (
    "agree",
    "qualified_agree",
    "disagree",
    "qualified_disagree",
    "alternative_proposal",
    "mixed",
    "no_response",
)
B_LABELS: Tuple[str, ...] = tuple("ABCDEFGHI") + ("UNCLEAR",)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_DATASET_A_REVISION = "4b99edfbc61fa6e1025e09803beb4d01e4bcce39"
DEFAULT_DATASET_B_REVISION = "06a00245e747794f10c609fe84df5350856782b6"


@dataclass(frozen=True)
class PolicyExample:
    example_id: str
    case_key: str
    task: str
    question_id: str
    question: str
    organization: str
    response: str
    answer: Tuple[str, ...]
    candidate_themes: Mapping[str, str] = field(default_factory=dict)
    original_split: str = ""

    @property
    def primary_label(self) -> str:
        return self.answer[0] if self.answer else "INVALID"


@dataclass
class PublicPolicyPilotConfig:
    model_name: str = DEFAULT_MODEL
    model_revision: str = DEFAULT_MODEL_REVISION
    dataset_a: str = "qyy752457002/AASB_Climate_Dataset"
    dataset_a_revision: str = DEFAULT_DATASET_A_REVISION
    dataset_b: str = "qyy752457002/AASB_Topic_Modelling_Dataset"
    dataset_b_revision: str = DEFAULT_DATASET_B_REVISION
    output_dir: str = "outputs/public_policy_rlvr_qwen3_0_6b_smoke"

    # A deterministic joint group split prevents the same consultation case
    # from appearing in source training and target evaluation.
    split_seed: int = 42
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    train_samples: int = 24
    eval_samples_a: int = 8
    eval_samples_b: int = 8

    use_cot: bool = False
    enable_qwen_thinking: bool = False
    eval_k: int = 2
    eval_temperature: float = 0.7
    eval_top_p: float = 0.9
    eval_max_new_tokens: int = 48

    train_steps: int = 4
    num_generations: int = 4
    train_temperature: float = 1.0
    train_top_p: float = 0.95
    train_max_new_tokens: int = 48
    max_prompt_length: int = 1024
    max_sequence_length: int = 1152
    max_response_chars: int = 3000
    learning_rate: float = 5e-5
    beta: float = 0.02
    epsilon: float = 0.2
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    profile_source_prompts: bool = True
    train_on_informative_only: bool = True

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    seed: int = 42
    save_adapter: bool = True

    def validate(self) -> None:
        if self.model_name == DEFAULT_MODEL and self.model_revision == "main":
            logger.warning("The model revision is not pinned; reproducibility is weaker")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train + validation fractions must be < 1")
        for name in ("train_samples", "eval_samples_a", "eval_samples_b"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.eval_k < 1:
            raise ValueError("eval_k must be >= 1")
        if self.max_response_chars < 200:
            raise ValueError("max_response_chars must be >= 200")

    @classmethod
    def from_yaml(
        cls, path: Union[os.PathLike, str]
    ) -> "PublicPolicyPilotConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config


def _parse_problem(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError(f"problem must be JSON text or dict, got {type(value)!r}")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("problem JSON must contain an object")
    return parsed


def _normalize_question_id(problem: Mapping[str, Any]) -> str:
    explicit = str(problem.get("question_id", "")).strip().upper()
    if explicit:
        match = re.search(r"(\d+)", explicit)
        return f"Q{int(match.group(1))}" if match else explicit
    question = str(problem.get("question", ""))
    match = re.search(r"\[\s*Question\s+(\d+)\s*\]", question, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer question_id from: {question[:80]!r}")
    return f"Q{int(match.group(1))}"


def _normalize_organization(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _case_key(problem: Mapping[str, Any]) -> str:
    return f"{_normalize_question_id(problem)}::{_normalize_organization(problem.get('organization'))}"


def _stable_bucket(case_key: str, seed: int, buckets: int = 10_000) -> int:
    digest = hashlib.sha256(f"{seed}:{case_key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % buckets


def split_for_case(
    case_key: str,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> str:
    value = _stable_bucket(case_key, seed) / 10_000
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


class AASBDataModule:
    """Load, join, audit, and group-split the two AASB task views."""

    def __init__(self, config: PublicPolicyPilotConfig):
        self.config = config
        self.examples_a: List[PolicyExample] = []
        self.examples_b: List[PolicyExample] = []
        self.audit: Dict[str, Any] = {}

    def load(self) -> "AASBDataModule":
        from datasets import load_dataset

        dataset_a = load_dataset(
            self.config.dataset_a,
            revision=self.config.dataset_a_revision,
        )
        dataset_b = load_dataset(
            self.config.dataset_b,
            revision=self.config.dataset_b_revision,
        )

        b_records: Dict[str, Tuple[Dict[str, Any], str, Sequence[str]]] = {}
        official_a: Dict[str, set[str]] = defaultdict(set)
        official_b: Dict[str, set[str]] = defaultdict(set)
        for split_name, split in dataset_b.items():
            for row in split:
                problem = _parse_problem(row["problem"])
                key = _case_key(problem)
                answer = row["answer"]
                if isinstance(answer, str):
                    answer = [answer]
                b_records[key] = (problem, split_name, tuple(map(str, answer)))
                official_b[split_name].add(key)

        joined_a_with_response = 0
        for split_name, split in dataset_a.items():
            for row in split:
                problem = _parse_problem(row["problem"])
                key = _case_key(problem)
                official_a[split_name].add(key)
                b_match = b_records.get(key)
                if b_match:
                    response = str(b_match[0].get("response", "")).strip()
                    joined_a_with_response += 1
                else:
                    # The A repository does not contain raw responses.  The
                    # rationale is a fallback only for the few unmatched cases
                    # and is explicitly counted in the audit.
                    response = str(problem.get("key_rationale", "")).strip()
                question_id = _normalize_question_id(problem)
                organization = str(problem.get("organization", "")).strip()
                self.examples_a.append(
                    PolicyExample(
                        example_id=f"A:{key}",
                        case_key=key,
                        task="A",
                        question_id=question_id,
                        question=str(problem.get("question", "")).strip(),
                        organization=organization,
                        response=response,
                        answer=(str(row["answer"]),),
                        original_split=split_name,
                    )
                )

        for split_name, split in dataset_b.items():
            for row in split:
                problem = _parse_problem(row["problem"])
                key = _case_key(problem)
                raw_answer = row["answer"]
                if isinstance(raw_answer, str):
                    raw_answer = [raw_answer]
                answer = tuple(sorted(map(str, raw_answer), key=_b_label_sort_key))
                candidate_themes = {
                    str(label): str(description)
                    for label, description in dict(
                        problem.get("candidate_themes", {})
                    ).items()
                }
                self.examples_b.append(
                    PolicyExample(
                        example_id=f"B:{key}",
                        case_key=key,
                        task="B",
                        question_id=_normalize_question_id(problem),
                        question=str(problem.get("question", "")).strip(),
                        organization=str(problem.get("organization", "")).strip(),
                        response=str(problem.get("response", "")).strip(),
                        answer=answer,
                        candidate_themes=candidate_themes,
                        original_split=split_name,
                    )
                )

        derived_a = self._derived_keys(self.examples_a)
        derived_b = self._derived_keys(self.examples_b)
        self.audit = {
            "dataset_a_rows": len(self.examples_a),
            "dataset_b_rows": len(self.examples_b),
            "a_rows_joined_to_raw_b_response": joined_a_with_response,
            "a_rows_using_rationale_fallback": len(self.examples_a)
            - joined_a_with_response,
            "all_case_overlap": len(
                {example.case_key for example in self.examples_a}
                & {example.case_key for example in self.examples_b}
            ),
            "official_a_train_b_test_overlap": len(
                official_a["train"] & official_b["test"]
            ),
            "official_b_train_a_test_overlap": len(
                official_b["train"] & official_a["test"]
            ),
            "derived_a_train_b_test_overlap": len(
                derived_a["train"] & derived_b["test"]
            ),
            "derived_b_train_a_test_overlap": len(
                derived_b["train"] & derived_a["test"]
            ),
            "derived_split_counts_a": {
                split: sum(
                    self.split_of(example) == split for example in self.examples_a
                )
                for split in ("train", "validation", "test")
            },
            "derived_split_counts_b": {
                split: sum(
                    self.split_of(example) == split for example in self.examples_b
                )
                for split in ("train", "validation", "test")
            },
            "a_label_counts": dict(Counter(e.primary_label for e in self.examples_a)),
            "b_label_counts": dict(
                Counter(label for e in self.examples_b for label in e.answer)
            ),
        }
        if self.audit["derived_a_train_b_test_overlap"] != 0:
            raise AssertionError("joint group split leaked A train cases into B test")
        return self

    def split_of(self, example: PolicyExample) -> str:
        return split_for_case(
            example.case_key,
            self.config.split_seed,
            self.config.train_fraction,
            self.config.validation_fraction,
        )

    def _derived_keys(self, examples: Iterable[PolicyExample]) -> Dict[str, set[str]]:
        result: Dict[str, set[str]] = defaultdict(set)
        for example in examples:
            result[self.split_of(example)].add(example.case_key)
        return result

    def select(
        self,
        task: str,
        split: str,
        limit: int,
        balanced: bool = False,
    ) -> List[PolicyExample]:
        source = self.examples_a if task == "A" else self.examples_b
        candidates = [example for example in source if self.split_of(example) == split]
        candidates.sort(
            key=lambda example: _stable_bucket(
                example.example_id, self.config.seed, buckets=2**31 - 1
            )
        )
        if not balanced:
            return candidates[:limit]

        by_label: Dict[str, List[PolicyExample]] = defaultdict(list)
        for example in candidates:
            by_label[example.primary_label].append(example)
        selected: List[PolicyExample] = []
        labels = sorted(by_label)
        while len(selected) < limit and labels:
            next_labels = []
            for label in labels:
                if by_label[label] and len(selected) < limit:
                    selected.append(by_label[label].pop(0))
                if by_label[label]:
                    next_labels.append(label)
            labels = next_labels
        return selected


def _b_label_sort_key(label: str) -> Tuple[int, str]:
    return (1 if label == "UNCLEAR" else 0, label)


def _abbreviate_evidence(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    marker = "\n[... evidence truncated for the prompt budget ...]\n"
    available = max_chars - len(marker)
    head = int(available * 0.75)
    tail = available - head
    return text[:head] + marker + text[-tail:]


def build_user_prompt(
    example: PolicyExample,
    use_cot: bool = False,
    max_response_chars: Optional[int] = None,
) -> str:
    reasoning_instruction = (
        "Briefly analyze the evidence before giving the final answer. "
        if use_cot
        else ""
    )
    response = _abbreviate_evidence(example.response, max_response_chars)
    if example.task == "A":
        labels = ", ".join(A_LABELS)
        return (
            "Classify the stance expressed in this public-policy consultation response.\n\n"
            f"Question: {example.question}\n"
            f"Organization: {example.organization}\n"
            f"Response: {response}\n\n"
            f"Allowed labels: {labels}.\n"
            f"{reasoning_instruction}Use the literal tag name 'answer'. For "
            "example: <answer>agree</answer>. Replace only the text 'agree' "
            "with your chosen label; never emit <LABEL> or <LABELS>."
        )

    themes = "\n".join(
        f"{label}: {description}"
        for label, description in sorted(example.candidate_themes.items())
    )
    return (
        "Identify every candidate theme supported by this public-policy response.\n\n"
        f"Question: {example.question}\n"
        f"Organization: {example.organization}\n"
        f"Response: {response}\n\n"
        f"Candidate themes:\n{themes}\n\n"
        "Use UNCLEAR only when none of the candidate themes can be verified. "
        f"{reasoning_instruction}Use the literal tag name 'answer'. Examples: "
        "<answer>A,C</answer> or <answer>UNCLEAR</answer>. Replace only the "
        "text between the tags; never emit <LABEL> or <LABELS>."
    )


def render_chat_prompt(
    tokenizer,
    user_prompt: str,
    enable_thinking: bool = False,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful public-policy analyst. Follow the requested "
                "answer schema; do not invent labels."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer(response: str, task: str) -> Tuple[str, ...]:
    tagged = _ANSWER_TAG.search(response)
    content = tagged.group(1) if tagged else response
    if task == "A":
        raw = content.strip().casefold()
        normalized = re.sub(r"[\s-]+", "_", raw).strip(".,;:!?`'\"")
        for label in sorted(A_LABELS, key=len, reverse=True):
            if normalized == label:
                return (label,)
        # A conservative fallback supports a label plus punctuation but avoids
        # matching "agree" inside "disagree".
        for label in sorted(A_LABELS, key=len, reverse=True):
            label_pattern = re.escape(label).replace(r"\_", r"[\s_-]+")
            if re.search(rf"(?<![a-z]){label_pattern}(?![a-z])", raw):
                return (label,)
        return tuple()

    # Task B requires an explicit <answer> tag whose content is a comma- or
    # space-separated list of single theme letters, or the bare literal
    # UNCLEAR.  Free-text is not parsed: prose routinely contains single
    # letters (e.g. "I think ...") and was previously misparsed as labels.
    if not tagged:
        return tuple()
    items = re.split(r"[,\s]+", content.strip())
    if not items or any(not item for item in items):
        return tuple()
    letters = set()
    has_unclear = False
    for item in items:
        if item.upper() == "UNCLEAR":
            has_unclear = True
        elif re.fullmatch(r"[A-I]", item.upper()):
            letters.add(item.upper())
        else:
            return tuple()
    if has_unclear:
        return tuple() if letters else ("UNCLEAR",)
    return tuple(sorted(letters))


def build_verifiable_reward(
    prompt_examples: Mapping[str, PolicyExample],
    correctness_weight: float = 0.9,
    format_weight: float = 0.1,
):
    if not math_is_close(correctness_weight + format_weight, 1.0):
        raise ValueError("reward weights must sum to 1")

    def reward(prompt: str, response: str) -> float:
        example = prompt_examples[prompt]
        prediction = extract_answer(response, example.task)
        correct = float(prediction == example.answer)
        valid_format = float(_ANSWER_TAG.search(response) is not None and bool(prediction))
        return correctness_weight * correct + format_weight * valid_format

    return reward


def math_is_close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance


def _macro_f1_single(
    expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> float:
    scores = []
    for label in labels:
        tp = sum(e == label and p == label for e, p in zip(expected, predicted))
        fp = sum(e != label and p == label for e, p in zip(expected, predicted))
        fn = sum(e == label and p != label for e, p in zip(expected, predicted))
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return sum(scores) / len(scores)


def _set_f1(expected: Sequence[str], predicted: Sequence[str]) -> float:
    expected_set, predicted_set = set(expected), set(predicted)
    if not expected_set and not predicted_set:
        return 1.0
    denominator = len(expected_set) + len(predicted_set)
    return 2 * len(expected_set & predicted_set) / denominator if denominator else 0.0


def _jaccard(expected: Sequence[str], predicted: Sequence[str]) -> float:
    expected_set, predicted_set = set(expected), set(predicted)
    union = expected_set | predicted_set
    return len(expected_set & predicted_set) / len(union) if union else 1.0


def compute_metrics(
    examples: Sequence[PolicyExample],
    predictions: Sequence[Sequence[Tuple[str, ...]]],
    format_validity: Optional[Sequence[Sequence[bool]]] = None,
) -> Dict[str, float]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length")
    if not examples:
        return {"num_examples": 0}
    k = len(predictions[0])
    if k < 1 or any(len(group) != k for group in predictions):
        raise ValueError("each example must have the same positive number of predictions")
    if format_validity is not None:
        if len(format_validity) != len(examples) or any(
            len(group) != k for group in format_validity
        ):
            raise ValueError("format_validity must have the same [N, K] shape")

    exact_values = []
    pass_values = []
    invalid_values = []
    consistency_values = []
    first_predictions = []
    example_f1_values = []
    jaccard_values = []
    for example, group in zip(examples, predictions):
        exact_group = [float(prediction == example.answer) for prediction in group]
        exact_values.extend(exact_group)
        pass_values.append(float(any(exact_group)))
        invalid_values.extend(float(not prediction) for prediction in group)
        consistency_values.append(max(Counter(group).values()) / k)
        first_predictions.append(group[0][0] if group[0] else "INVALID")
        example_f1_values.extend(_set_f1(example.answer, pred) for pred in group)
        jaccard_values.extend(_jaccard(example.answer, pred) for pred in group)

    metrics = {
        "num_examples": len(examples),
        "k": k,
        "avg_at_k": sum(exact_values) / len(exact_values),
        "pass_at_k": sum(pass_values) / len(pass_values),
        "invalid_rate": sum(invalid_values) / len(invalid_values),
        "consistency_at_k": sum(consistency_values) / len(consistency_values),
    }
    if format_validity is not None:
        format_values = [float(value) for group in format_validity for value in group]
        metrics["format_valid_rate"] = sum(format_values) / len(format_values)
        metrics["schema_violation_rate"] = 1.0 - metrics["format_valid_rate"]
    if examples[0].task == "A":
        expected = [example.primary_label for example in examples]
        metrics.update(
            accuracy=sum(e == p for e, p in zip(expected, first_predictions))
            / len(expected),
            macro_f1=_macro_f1_single(expected, first_predictions, A_LABELS),
        )
    else:
        metrics.update(
            exact_match_at_1=sum(
                example.answer == group[0]
                for example, group in zip(examples, predictions)
            )
            / len(examples),
            example_f1_at_k=sum(example_f1_values) / len(example_f1_values),
            jaccard_at_k=sum(jaccard_values) / len(jaccard_values),
        )
    return metrics


class PolicyBenchmarkEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        config: PublicPolicyPilotConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def evaluate(
        self,
        examples: Sequence[PolicyExample],
        output_path: Optional[Union[os.PathLike, str]] = None,
    ) -> Dict[str, Any]:
        previous_training = self.model.training
        previous_padding_side = self.tokenizer.padding_side
        self.model.eval()
        self.tokenizer.padding_side = "left"
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        prediction_groups: List[List[Tuple[str, ...]]] = []
        format_groups: List[List[bool]] = []
        records = []
        try:
            for example in examples:
                user_prompt = build_user_prompt(
                    example,
                    use_cot=self.config.use_cot,
                    max_response_chars=self.config.max_response_chars,
                )
                prompt = render_chat_prompt(
                    self.tokenizer,
                    user_prompt,
                    enable_thinking=self.config.enable_qwen_thinking,
                )
                prompts = [prompt] * self.config.eval_k
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                    add_special_tokens=False,
                ).to(self.device)
                prompt_width = inputs["input_ids"].shape[1]
                generation_kwargs = {
                    "max_new_tokens": self.config.eval_max_new_tokens,
                    "do_sample": self.config.eval_k > 1,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                if self.config.eval_k > 1:
                    generation_kwargs.update(
                        temperature=self.config.eval_temperature,
                        top_p=self.config.eval_top_p,
                    )
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                responses = [
                    self.tokenizer.decode(
                        output[prompt_width:], skip_special_tokens=True
                    ).strip()
                    for output in outputs
                ]
                parsed = [extract_answer(response, example.task) for response in responses]
                format_valid = [
                    _ANSWER_TAG.search(response) is not None and bool(prediction)
                    for response, prediction in zip(responses, parsed)
                ]
                prediction_groups.append(parsed)
                format_groups.append(format_valid)
                records.append(
                    {
                        "example_id": example.example_id,
                        "case_key": example.case_key,
                        "task": example.task,
                        "expected": list(example.answer),
                        "user_prompt": user_prompt,
                        "response_truncated": len(example.response)
                        > self.config.max_response_chars,
                        "responses": responses,
                        "predictions": [list(value) for value in parsed],
                        "format_valid": format_valid,
                    }
                )
        finally:
            self.tokenizer.padding_side = previous_padding_side
            self.model.train(previous_training)

        result = {
            "metrics": compute_metrics(
                examples, prediction_groups, format_validity=format_groups
            ),
            "records": records,
        }
        if output_path is not None:
            _write_json(output_path, result)
        return result


def load_qwen3_lora(config: PublicPolicyPilotConfig):
    """Load Qwen3 and attach a small LoRA policy adapter."""

    from packaging.version import Version
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

    if Version(__version__) < Version("4.51.0"):
        raise RuntimeError(
            f"Qwen3 requires transformers>=4.51.0; found {__version__}. "
            "Install experiments/public_policy_rlvr/requirements.txt in an "
            "isolated environment."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The Qwen3 RLVR smoke run requires a CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = False
    if callable(getattr(model, "gradient_checkpointing_enable", None)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if callable(getattr(model, "enable_input_require_grads", None)):
        model.enable_input_require_grads()

    # PEFT probes every optional quantization backend while dispatching LoRA.
    # A broken, globally installed AutoAWQ must not make an ordinary BF16 model
    # unusable inside an otherwise isolated environment.
    if getattr(model, "quantization_method", None) is None:
        try:
            from peft.tuners.lora import awq as peft_awq

            peft_awq.is_auto_awq_available = lambda: False
        except ImportError:
            pass

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    if callable(getattr(model, "enable_input_require_grads", None)):
        model.enable_input_require_grads()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    model_metadata = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }
    return model, tokenizer, model_metadata


def prepare_pilot_data(config: PublicPolicyPilotConfig) -> Dict[str, Any]:
    data = AASBDataModule(config).load()
    train_a = data.select("A", "train", config.train_samples, balanced=True)
    train_b = data.select("B", "train", config.train_samples, balanced=False)
    benchmark_a = data.select("A", "test", config.eval_samples_a)
    benchmark_b = data.select("B", "test", config.eval_samples_b)
    return {
        "data_module": data,
        "train_a": train_a,
        "train_b": train_b,
        "benchmark_a": benchmark_a,
        "benchmark_b": benchmark_b,
    }


def _example_manifest(example: PolicyExample, split: str) -> Dict[str, Any]:
    return {
        "example_id": example.example_id,
        "case_key": example.case_key,
        "task": example.task,
        "derived_split": split,
        "original_split": example.original_split,
        "answer": list(example.answer),
    }


def dry_run_public_policy_pilot(config: PublicPolicyPilotConfig) -> Dict[str, Any]:
    """Validate data, split isolation, prompts, parsers, and rewards without a model."""

    config.validate()
    prepared = prepare_pilot_data(config)
    data: AASBDataModule = prepared["data_module"]
    train_a: List[PolicyExample] = prepared["train_a"]
    benchmark_a: List[PolicyExample] = prepared["benchmark_a"]
    benchmark_b: List[PolicyExample] = prepared["benchmark_b"]

    # A plain prompt map is enough to unit-test the private verifier; the model
    # run replaces these with tokenizer-rendered chat prompts.
    prompt_examples = {
        build_user_prompt(
            example,
            config.use_cot,
            max_response_chars=config.max_response_chars,
        ): example
        for example in train_a
    }
    reward = build_verifiable_reward(prompt_examples)
    reward_checks = []
    for prompt, example in prompt_examples.items():
        correct = f"<answer>{','.join(example.answer)}</answer>"
        wrong = "<answer>INVALID</answer>"
        reward_checks.append(
            {
                "example_id": example.example_id,
                "correct_reward": reward(prompt, correct),
                "wrong_reward": reward(prompt, wrong),
            }
        )
    if not all(item["correct_reward"] == 1.0 for item in reward_checks):
        raise AssertionError("correct-answer reward smoke check failed")

    manifest = {
        "config": asdict(config),
        "audit": data.audit,
        "selected": {
            "train_a": [
                _example_manifest(example, data.split_of(example)) for example in train_a
            ],
            "benchmark_a": [
                _example_manifest(example, data.split_of(example))
                for example in benchmark_a
            ],
            "benchmark_b": [
                _example_manifest(example, data.split_of(example))
                for example in benchmark_b
            ],
        },
        "reward_checks": reward_checks,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "dry_run_manifest.json", manifest)
    return manifest


def run_public_policy_pilot(config: PublicPolicyPilotConfig) -> Dict[str, Any]:
    """Execute the minimal Qwen3 baseline -> A-RLVR -> A/B evaluation loop."""

    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_pilot_data(config)
    data: AASBDataModule = prepared["data_module"]
    train_a: List[PolicyExample] = prepared["train_a"]
    benchmark_a: List[PolicyExample] = prepared["benchmark_a"]
    benchmark_b: List[PolicyExample] = prepared["benchmark_b"]

    model, tokenizer, model_metadata = load_qwen3_lora(config)
    evaluator = PolicyBenchmarkEvaluator(model, tokenizer, config)
    logger.info("Running pre-RLVR Benchmark A")
    pre_a = evaluator.evaluate(benchmark_a, output_dir / "pre_benchmark_a.json")
    logger.info("Running pre-RLVR Benchmark B")
    pre_b = evaluator.evaluate(benchmark_b, output_dir / "pre_benchmark_b.json")

    prompt_examples: Dict[str, PolicyExample] = {}
    for example in train_a:
        prompt = render_chat_prompt(
            tokenizer,
            build_user_prompt(
                example,
                config.use_cot,
                max_response_chars=config.max_response_chars,
            ),
            enable_thinking=config.enable_qwen_thinking,
        )
        prompt_examples[prompt] = example
    reward_fn = build_verifiable_reward(prompt_examples)
    grpo_config = GRPOConfig(
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_new_tokens=config.train_max_new_tokens,
        max_sequence_length=config.max_sequence_length,
        temperature=config.train_temperature,
        top_p=config.train_top_p,
        learning_rate=config.learning_rate,
        beta=config.beta,
        epsilon=config.epsilon,
        per_device_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ref_model_update_interval=0,
        log_every_n_steps=1,
        seed=config.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        config=grpo_config,
    )
    source_profiles: List[Dict[str, Any]] = []
    training_prompts = list(prompt_examples)
    if config.profile_source_prompts:
        source_profiles = trainer.profile_prompts(training_prompts, reward_fn)
        informative_prompts = [
            str(profile["prompt"])
            for profile in source_profiles
            if profile["informative"]
        ]
        if config.train_on_informative_only and informative_prompts:
            training_prompts = informative_prompts
        elif config.train_on_informative_only:
            logger.warning(
                "No informative source groups found; falling back to all prompts"
            )

        serializable_profiles = []
        for profile in source_profiles:
            prompt = str(profile["prompt"])
            example = prompt_examples[prompt]
            serializable_profiles.append(
                {
                    "example_id": example.example_id,
                    "case_key": example.case_key,
                    "label": list(example.answer),
                    "responses": profile["responses"],
                    "rewards": profile["rewards"],
                    "reward_mean": profile["reward_mean"],
                    "reward_std": profile["reward_std"],
                    "informative": profile["informative"],
                }
            )
        _write_json(output_dir / "source_profile.json", serializable_profiles)
    training_metrics = trainer.train(
        prompts=training_prompts,
        reward_fn=reward_fn,
        num_steps=config.train_steps,
    )
    _write_json(output_dir / "training_metrics.json", training_metrics)
    if config.save_adapter:
        trainer.save(str(output_dir / "adapter"))

    logger.info("Running post-RLVR Benchmark A")
    post_a = evaluator.evaluate(benchmark_a, output_dir / "post_benchmark_a.json")
    logger.info("Running post-RLVR Benchmark B")
    post_b = evaluator.evaluate(benchmark_b, output_dir / "post_benchmark_b.json")

    delta_a = _metric_delta(pre_a["metrics"], post_a["metrics"])
    delta_b = _metric_delta(pre_b["metrics"], post_b["metrics"])
    result = {
        "status": "completed",
        "scope": (
            "framework smoke test only; not confirmatory evidence of RLVR "
            "generalization"
        ),
        "config": asdict(config),
        "environment": environment_metadata(),
        "model": model_metadata,
        "data_audit": data.audit,
        "selected_counts": {
            "train_a": len(train_a),
            "benchmark_a": len(benchmark_a),
            "benchmark_b": len(benchmark_b),
            "truncated_train_a_responses": sum(
                len(example.response) > config.max_response_chars
                for example in train_a
            ),
            "truncated_benchmark_a_responses": sum(
                len(example.response) > config.max_response_chars
                for example in benchmark_a
            ),
            "truncated_benchmark_b_responses": sum(
                len(example.response) > config.max_response_chars
                for example in benchmark_b
            ),
        },
        "pre": {"benchmark_a": pre_a["metrics"], "benchmark_b": pre_b["metrics"]},
        "post": {"benchmark_a": post_a["metrics"], "benchmark_b": post_b["metrics"]},
        "delta": {"a_to_a": delta_a, "a_to_b": delta_b},
        "training": {
            "steps": training_metrics,
            "source_profile": {
                "enabled": config.profile_source_prompts,
                "num_profiled": len(source_profiles),
                "num_informative": sum(
                    bool(profile["informative"]) for profile in source_profiles
                ),
                "informative_rate": (
                    sum(bool(profile["informative"]) for profile in source_profiles)
                    / len(source_profiles)
                    if source_profiles
                    else None
                ),
                "train_on_informative_only": config.train_on_informative_only,
                "num_training_prompts": len(training_prompts),
            },
            "mean_informative_group_rate": sum(
                item["informative_group_rate"] for item in training_metrics
            )
            / max(1, len(training_metrics)),
        },
    }
    _write_json(output_dir / "result.json", result)
    return result


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, float]:
    result = {}
    for key in sorted(set(before) & set(after)):
        left, right = before[key], after[key]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if key not in {"num_examples", "k"}:
                result[key] = float(right - left)
    return result


def environment_metadata() -> Dict[str, Any]:
    import datasets
    import peft
    import transformers

    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "gpu": gpu,
    }


def _write_json(path: Union[os.PathLike, str], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


__all__ = [
    "A_LABELS",
    "B_LABELS",
    "AASBDataModule",
    "PolicyBenchmarkEvaluator",
    "PolicyExample",
    "PublicPolicyPilotConfig",
    "build_user_prompt",
    "build_verifiable_reward",
    "compute_metrics",
    "dry_run_public_policy_pilot",
    "extract_answer",
    "load_qwen3_lora",
    "render_chat_prompt",
    "run_public_policy_pilot",
    "split_for_case",
]
