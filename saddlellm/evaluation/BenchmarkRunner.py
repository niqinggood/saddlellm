"""
标准化评测框架 - 跨模型大小对比
"""
import os
import json
import logging
import time
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

BENCHMARK_TASKS = {
    "perplexity": "困惑度 (语言建模质量)",
    "mmlu": "MMLU 快速子集 (默认 7 科，可通过 mmlu_subjects 配置)",
    "hellaswag": "常识推理",
    "arc_easy": "ARC 简单科学推理",
    "arc_challenge": "ARC 困难科学推理",
    "winogrande": "代词消解",
    "piqa": "物理常识推理",
    "boolq": "布尔问答",
    "lambada": "长程依赖语言建模",
    "humaneval": "代码生成 (需要沙箱执行，本入口不支持)",
    "gsm8k": "小学数学推理",
}


@dataclass
class EvalResult:
    task: str
    score: Optional[float]
    metric: str
    num_samples: int
    duration_sec: float
    model_params: Optional[int] = None
    model_name: Optional[str] = None
    status: str = "ok"
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class BenchmarkRunner:
    """
    标准化 LLM 评测。

    用法:
        runner = BenchmarkRunner("path/to/model", tasks=["perplexity", "hellaswag"])
        results = runner.run()
        runner.save_results(results, "results.json")

        # 跨模型对比
        comparison = BenchmarkRunner.compare([
            "results/100m.json",
            "results/300m.json",
            "results/1b.json",
        ])
    """

    def __init__(
        self,
        model_path: Optional[str],
        tasks: Optional[List[str]] = None,
        max_samples: int = 1000,
        batch_size: int = 8,
        device: str = "auto",
        tokenizer_path: Optional[str] = None,
        max_new_tokens: int = 256,
        mmlu_subjects: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.tasks = ["perplexity"] if tasks is None else list(tasks)
        if not self.tasks:
            raise ValueError("At least one benchmark task is required")
        unknown = set(self.tasks) - set(BENCHMARK_TASKS)
        if unknown:
            raise ValueError(f"Unknown benchmark tasks: {sorted(unknown)}")
        if "humaneval" in self.tasks:
            raise NotImplementedError("HumanEval requires sandboxed code execution; BenchmarkRunner does not implement it")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (max_samples, batch_size, max_new_tokens)):
            raise ValueError("max_samples, batch_size and max_new_tokens must be positive integers")
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.device = device
        self.tokenizer_path = tokenizer_path or model_path
        self.max_new_tokens = max_new_tokens
        self.mmlu_subjects = list(mmlu_subjects) if mmlu_subjects is not None else [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "college_chemistry", "computer_security", "high_school_mathematics",
        ]
        if not self.mmlu_subjects or len(set(self.mmlu_subjects)) != len(self.mmlu_subjects):
            raise ValueError("mmlu_subjects must contain unique subject names")
        self._task_details = {}

        self._model = None
        self._tokenizer = None

    def run(self) -> List[EvalResult]:
        """Run tasks and retain explicit failed/partial results rather than dropping them."""
        self._load_model()
        results = []

        for task in self.tasks:
            logger.info(f"评测任务: {task}")
            t0 = time.time()
            self._task_details = {"num_samples": 0}
            score = None
            metric = "perplexity" if task == "perplexity" else "accuracy"
            status, error = "ok", None

            try:
                if task == "perplexity":
                    score = self._eval_perplexity()
                    metric = "perplexity"
                elif task == "hellaswag":
                    score = self._eval_hellaswag()
                    metric = "accuracy"
                elif task == "mmlu":
                    score = self._eval_mmlu()
                    metric = "accuracy"
                elif task == "arc_easy":
                    score = self._eval_arc("ARC-Easy")
                    metric = "accuracy"
                elif task == "arc_challenge":
                    score = self._eval_arc("ARC-Challenge")
                    metric = "accuracy"
                elif task == "winogrande":
                    score = self._eval_winogrande()
                    metric = "accuracy"
                elif task == "piqa":
                    score = self._eval_piqa()
                    metric = "accuracy"
                elif task == "boolq":
                    score = self._eval_boolq()
                    metric = "accuracy"
                elif task == "lambada":
                    score = self._eval_lambada()
                    metric = "accuracy"
                elif task == "gsm8k":
                    score = self._eval_gsm8k()
                    metric = "accuracy"
                if score is None or not math.isfinite(score):
                    raise ValueError("Evaluation did not produce a finite score")
                if self._task_details.get("failures"):
                    status = "partial"
            except Exception as e:
                logger.warning(f"  {task} 评测失败: {e}")
                status, error, score = "failed", f"{type(e).__name__}: {e}", None

            details = dict(self._task_details)
            result = EvalResult(
                task=task, score=score, metric=metric,
                num_samples=details.pop("num_samples", 0), duration_sec=time.time() - t0,
                model_params=sum(p.numel() for p in self._model.parameters()),
                model_name=os.path.basename(self.model_path) if self.model_path else type(self._model).__name__,
                status=status, error=error,
                metadata=details,
            )
            results.append(result)

        return results

    # ---- 各任务实现 ----

    def _eval_perplexity(self) -> float:
        import torch
        from datasets import load_dataset

        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
        except Exception:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        total_loss = 0.0
        total_tokens = 0
        total_samples = 0
        self._task_details.update(dataset="wikitext", config="wikitext-2-raw-v1", split="test",
                                  max_length=512, scoring="token-weighted causal language-model NLL")
        self._model.eval()

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i >= self.max_samples:
                    break
                text = example.get("text", "")
                if not text or len(text.strip()) < 10:
                    continue
                inputs = self._tokenizer(text, return_tensors="pt",
                                         truncation=True, max_length=512)
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                outputs = self._model(**inputs, labels=inputs["input_ids"])
                tokens = int(inputs["attention_mask"][:, 1:].sum().item()) if "attention_mask" in inputs else inputs["input_ids"].numel() - 1
                if tokens <= 0:
                    continue
                total_loss += outputs.loss.item() * tokens
                total_tokens += tokens
                total_samples += 1
                self._task_details.update(num_samples=total_samples, num_scored_tokens=total_tokens)

        if total_tokens == 0:
            raise ValueError("No scorable language-model tokens")
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()

    def _eval_hellaswag(self) -> float:
        return self._eval_hf_multiple_choice("Rowan/hellaswag", split="validation")

    def _eval_mmlu(self) -> float:
        from datasets import load_dataset

        correct, total, failures, subjects = 0, 0, {}, {}
        for subject in self.mmlu_subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test", streaming=True)
                self._eval_hf_multiple_choice_ds(ds)
                stats = dict(self._last_mc_stats)
                subjects[subject] = stats
                correct += stats["correct"]
                total += stats["num_samples"]
            except Exception as exc:
                failures[subject] = f"{type(exc).__name__}: {exc}"
        self._task_details.update(
            dataset="cais/mmlu", split="test", scope="configured_subject_subset",
            requested_subjects=self.mmlu_subjects, subjects=subjects, failures=failures,
            aggregation="micro_accuracy", max_samples_per_subject=self.max_samples,
            num_samples=total, correct=correct,
            attempted=sum(stats["attempted"] for stats in subjects.values()),
            skipped=sum(stats["skipped"] for stats in subjects.values()),
        )
        if total == 0:
            raise ValueError(f"No MMLU subjects could be scored: {failures}")
        return correct / total

    def _eval_arc(self, config: str) -> float:
        return self._eval_hf_multiple_choice("ai2_arc", config)

    def _eval_winogrande(self) -> float:
        return self._eval_hf_multiple_choice("winogrande", "winogrande_xl", split="validation")

    def _eval_piqa(self) -> float:
        return self._eval_hf_multiple_choice("piqa", split="validation")

    def _eval_boolq(self) -> float:
        return self._eval_hf_multiple_choice("boolq", split="validation")

    def _eval_lambada(self) -> float:
        import torch
        from datasets import load_dataset

        dataset = load_dataset("lambada", split="test", streaming=True)
        correct, total, skipped = 0, 0, 0
        self._task_details.update(dataset="lambada", split="test", scoring="last-word greedy exact match")
        self._model.eval()
        for i, example in enumerate(dataset):
            if i >= self.max_samples:
                break
            words = example.get("text", "").rsplit(None, 1)
            if len(words) != 2:
                skipped += 1
                continue
            context, target = words
            # Generate only as many tokens as the known held-out word needs.
            target_ids = self._tokenizer(" " + target, add_special_tokens=False)["input_ids"]
            inputs = self._tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self._model.generate(**inputs, do_sample=False, max_new_tokens=max(1, len(target_ids)))
            generated = self._tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            correct += int(generated == target)
            total += 1
            self._task_details.update(num_samples=total, skipped=skipped)
        self._task_details.update(num_samples=total, skipped=skipped)
        if total == 0:
            raise ValueError("No scorable LAMBADA examples")
        return correct / total

    def _eval_gsm8k(self) -> float:
        import torch
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
        correct, total, skipped, unparsed = 0, 0, 0, 0
        self._task_details.update(dataset="openai/gsm8k", config="main", split="test",
                                  scoring="zero-shot greedy numeric exact match",
                                  answer_extraction="final ####, boxed, or explicit final-answer line",
                                  max_new_tokens=self.max_new_tokens)
        self._model.eval()
        for i, example in enumerate(dataset):
            if i >= self.max_samples:
                break
            reference = self._extract_gsm8k_answer(example.get("answer", ""))
            question = example.get("question")
            if not isinstance(question, str) or not question.strip() or reference is None:
                skipped += 1
                continue
            prompt = f"Question: {question}\nSolve the problem. End with '#### <numeric answer>'.\nAnswer:"
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            completion = self._tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            prediction = self._extract_gsm8k_answer(completion)
            unparsed += int(prediction is None)
            correct += int(prediction is not None and prediction == reference)
            total += 1
            self._task_details.update(num_samples=total, skipped=skipped, unparsed_predictions=unparsed, correct=correct)
        self._task_details.update(num_samples=total, skipped=skipped, unparsed_predictions=unparsed, correct=correct)
        if total == 0:
            raise ValueError("No valid GSM8K examples")
        return correct / total

    @staticmethod
    def _extract_gsm8k_answer(text: str):
        """Require a final answer rather than accepting an incidental reasoning number."""
        if not isinstance(text, str):
            return None
        if "####" in text:
            value = text.rsplit("####", 1)[1].strip()
        else:
            boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
            value = boxed[-1].strip() if boxed else text.strip().splitlines()[-1] if text.strip() else ""
            value = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*", "", value, flags=re.I)
        value = value.strip().strip("$ ")
        number = r"[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        match = re.fullmatch(rf"({number})(?:\s*/\s*({number}))?\s*[.!]?", value)
        if not match:
            return None
        try:
            result = Fraction(Decimal(match[1].replace(",", "")))
            if match[2] is not None:
                result /= Fraction(Decimal(match[2].replace(",", "")))
            return result
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    def _eval_hf_multiple_choice(self, dataset_name: str, config: Optional[str] = None, split: str = "test") -> float:
        """通用多选题评测"""
        from datasets import load_dataset

        self._task_details.update(dataset=dataset_name, config=config, split=split)
        dataset = load_dataset(dataset_name, config, split=split, streaming=True)
        return self._eval_hf_multiple_choice_ds(dataset)

    @staticmethod
    def _multiple_choice_example(example):
        """Normalize the actual schemas used by the supported benchmark datasets."""
        if "sol1" in example:
            context, choices, label = example.get("goal", ""), [example["sol1"], example["sol2"]], example.get("label")
        elif "passage" in example and isinstance(example.get("answer"), bool):
            context = f"{example['passage']}\nQuestion: {example.get('question', '')}\nAnswer:"
            choices, label = ["no", "yes"], int(example["answer"])
        elif "option1" in example:
            context, choices, label = example.get("sentence", ""), [example["option1"], example["option2"]], int(example["answer"]) - 1
            if "_" in context:
                choices = [context.replace("_", value) for value in choices]
                context = "Complete the sentence:"
        else:
            context = example.get("ctx", example.get("context", example.get("question", "")))
            choices = example.get("endings", example.get("choices", []))
            label = example.get("label", example.get("answer", example.get("answerKey")))
        choice_labels = None
        if isinstance(choices, dict):
            choice_labels = choices.get("label")
            choices = choices.get("text", [])
        if not isinstance(context, str) or not context.strip() or not isinstance(choices, (list, tuple)) or len(choices) < 2 or label is None:
            raise ValueError("Malformed multiple-choice example")
        choices = [value.get("text", "") if isinstance(value, dict) else value for value in choices]
        if not all(isinstance(value, str) and value for value in choices):
            raise ValueError("Choices must contain nonempty text")
        if isinstance(label, str):
            if choice_labels and label in choice_labels:
                label = choice_labels.index(label)
            elif label.isdigit():
                label = int(label)
            elif len(label) == 1 and "A" <= label <= "Z":
                label = ord(label) - ord("A")
            else:
                raise ValueError(f"Unknown answer label: {label!r}")
        if not isinstance(label, int) or not 0 <= label < len(choices):
            raise ValueError("Answer label is out of range")
        return context, choices, label

    def _choice_loss(self, context: str, ending: str) -> float:
        """Mean continuation NLL; context tokens do not contribute to choice scores."""
        import torch

        prefix = self._tokenizer(context, add_special_tokens=False)["input_ids"]
        tokens = self._tokenizer(f"{context} {ending}", add_special_tokens=False)["input_ids"]
        shared = 0
        while shared < min(len(prefix), len(tokens)) and prefix[shared] == tokens[shared]:
            shared += 1
        input_ids = torch.tensor([tokens], device=self._model.device)
        labels = input_ids.clone()
        labels[:, :shared] = -100
        if not bool((labels[:, 1:] != -100).any()):
            raise ValueError("Choice has no scorable continuation tokens")
        with torch.inference_mode():
            loss = float(self._model(input_ids=input_ids, labels=labels).loss.item())
        if not math.isfinite(loss):
            raise ValueError("Nonfinite choice loss")
        return loss

    def _eval_hf_multiple_choice_ds(self, dataset) -> float:
        correct, total, skipped, attempted = 0, 0, 0, 0
        self._model.eval()
        for example in dataset:
            if attempted >= self.max_samples:
                break
            attempted += 1
            try:
                context, choices, label = self._multiple_choice_example(example)
            except (KeyError, ValueError, TypeError):
                skipped += 1
                continue
            losses = [self._choice_loss(context, choice) for choice in choices]
            correct += int(losses.index(min(losses)) == label)
            total += 1
            self._task_details["num_samples"] = total
        self._last_mc_stats = dict(correct=correct, num_samples=total, attempted=attempted, skipped=skipped)
        self._task_details.update(self._last_mc_stats, scoring="mean continuation NLL", max_samples=self.max_samples)
        if total == 0:
            raise ValueError(f"No scorable multiple-choice examples (attempted={attempted}, skipped={skipped})")
        return correct / total

    def _load_model(self):
        if self._model is not None:
            return
        if self.model_path is None:
            raise ValueError("model_path is required unless an in-memory model and tokenizer are supplied")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            self._model = self._model.cpu()

    def save_results(self, results: List[EvalResult], path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def load_results(cls, path: str) -> List[EvalResult]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvalResult(**d) for d in data]

    @staticmethod
    def compare(result_paths: List[str], output_path: Optional[str] = None) -> str:
        """比较多个模型的结果,生成对比表格"""
        all_results = []
        for rp in result_paths:
            if os.path.exists(rp):
                with open(rp, "r", encoding="utf-8") as f:
                    all_results.append(json.load(f))

        if not all_results:
            return "无可对比的结果"

        # 收集所有任务
        all_tasks = set()
        for results in all_results:
            for r in results:
                all_tasks.add(r["task"])
        all_tasks = sorted(all_tasks)

        # 建立表格
        lines = []
        header = f"{'Task':<20}"
        for i, results in enumerate(all_results):
            name = results[0].get("model_name", f"Model-{i}") if results else f"Model-{i}"
            params = (results[0].get("model_params") or 0) if results else 0
            header += f" {name}({params/1e6:.0f}M)"
        lines.append(header)
        lines.append("-" * len(header))

        for task in all_tasks:
            line = f"{task:<20}"
            for results in all_results:
                task_results = [r for r in results if r["task"] == task]
                if task_results and task_results[0].get("score") is not None:
                    suffix = "*" if task_results[0].get("status") == "partial" else ""
                    line += f" {task_results[0]['score']:>14.4f}{suffix}"
                elif task_results:
                    line += f" {'FAILED':>15}"
                else:
                    line += f" {'N/A':>15}"
            lines.append(line)

        table = "\n".join(lines)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(table)
        return table
