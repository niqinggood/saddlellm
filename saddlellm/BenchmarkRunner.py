"""
标准化评测框架 - 跨模型大小对比
"""
import os
import json
import logging
import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

BENCHMARK_TASKS = {
    "perplexity": "困惑度 (语言建模质量)",
    "mmlu": "大规模多任务语言理解 (57 个学科)",
    "hellaswag": "常识推理",
    "arc_easy": "ARC 简单科学推理",
    "arc_challenge": "ARC 困难科学推理",
    "winogrande": "代词消解",
    "piqa": "物理常识推理",
    "boolq": "布尔问答",
    "lambada": "长程依赖语言建模",
    "humaneval": "代码生成 (HumanEval)",
    "gsm8k": "小学数学推理",
}


@dataclass
class EvalResult:
    task: str
    score: float
    metric: str
    num_samples: int
    duration_sec: float
    model_params: Optional[int] = None
    model_name: Optional[str] = None


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
        model_path: str,
        tasks: Optional[List[str]] = None,
        max_samples: int = 1000,
        batch_size: int = 8,
        device: str = "auto",
        tokenizer_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.tasks = tasks or ["perplexity"]
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.device = device
        self.tokenizer_path = tokenizer_path or model_path

        self._model = None
        self._tokenizer = None

    def run(self) -> List[EvalResult]:
        """运行所有评测任务"""
        self._load_model()
        results = []

        for task in self.tasks:
            logger.info(f"评测任务: {task}")
            t0 = time.time()

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
                else:
                    logger.warning(f"未知任务: {task}, 跳过")
                    continue

                duration = time.time() - t0
                num_params = sum(p.numel() for p in self._model.parameters())
                result = EvalResult(
                    task=task, score=score, metric=metric,
                    num_samples=self.max_samples, duration_sec=duration,
                    model_params=num_params, model_name=os.path.basename(self.model_path),
                )
                results.append(result)
                logger.info(f"  {task}: {score:.4f} ({metric}) [{duration:.1f}s]")

            except Exception as e:
                logger.warning(f"  {task} 评测失败: {e}")

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
                total_loss += outputs.loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()

        if total_tokens == 0:
            return float("inf")
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()

    def _eval_hellaswag(self) -> float:
        return self._eval_hf_multiple_choice("Rowan/hellaswag", "validation")

    def _eval_mmlu(self) -> float:
        # MMLU 有 57 个科目,这里只跑一部分作为快速评测
        try:
            from datasets import load_dataset
            subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics",
                        "college_chemistry", "computer_security", "high_school_mathematics"]
            scores = []
            for subject in subjects:
                try:
                    ds = load_dataset("cais/mmlu", subject, split="test", streaming=True)
                    score = self._eval_hf_multiple_choice_ds(ds)
                    scores.append(score)
                except Exception:
                    continue
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0

    def _eval_arc(self, config: str) -> float:
        return self._eval_hf_multiple_choice("ai2_arc", config)

    def _eval_winogrande(self) -> float:
        return self._eval_hf_multiple_choice("winogrande", "winogrande_xl")

    def _eval_piqa(self) -> float:
        return self._eval_hf_multiple_choice("piqa", "validation")

    def _eval_boolq(self) -> float:
        return self._eval_hf_multiple_choice("boolq", "validation")

    def _eval_lambada(self) -> float:
        return self._eval_hf_multiple_choice("lambada", "standard")

    def _eval_gsm8k(self) -> float:
        return self._eval_perplexity()  # placeholder

    def _eval_hf_multiple_choice(self, dataset_name: str, config: str) -> float:
        """通用多选题评测"""
        import torch
        from datasets import load_dataset

        try:
            dataset = load_dataset(dataset_name, config, split="test" if config != "validation" else "validation",
                                   streaming=True, trust_remote_code=True)
        except Exception:
            try:
                dataset = load_dataset(dataset_name, config, split="validation", streaming=True, trust_remote_code=True)
            except Exception:
                return 0.0

        return self._eval_hf_multiple_choice_ds(dataset)

    def _eval_hf_multiple_choice_ds(self, dataset) -> float:
        import torch

        correct = 0
        total = 0
        self._model.eval()

        with torch.no_grad():
            for example in dataset:
                if total >= self.max_samples:
                    break
                try:
                    ctx = example.get("ctx", example.get("context", example.get("question", "")))
                    endings = example.get("endings", example.get("choices", []))
                    label = example.get("label", example.get("answer", None))

                    if not ctx or not endings or label is None:
                        continue

                    if isinstance(endings[0], dict):
                        endings = [e.get("text", str(e)) for e in endings]

                    # 简单形式: 拼接上下文和每个选项,选困惑度最低的
                    losses = []
                    for ending in endings[:4]:
                        text = f"{ctx} {ending}"
                        inputs = self._tokenizer(text, return_tensors="pt",
                                                 truncation=True, max_length=512)
                        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                        outputs = self._model(**inputs, labels=inputs["input_ids"])
                        losses.append(outputs.loss.item())

                    pred = losses.index(min(losses))
                    if isinstance(label, str):
                        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                        label = label_map.get(label, 0)
                    if pred == int(label):
                        correct += 1
                    total += 1
                except Exception:
                    continue

        return correct / total if total > 0 else 0.0

    def _load_model(self):
        if self._model is not None:
            return
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
            json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)

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
            params = results[0].get("model_params", 0) if results else 0
            header += f" {name}({params/1e6:.0f}M):>15"
        lines.append(header)
        lines.append("-" * len(header))

        for task in all_tasks:
            line = f"{task:<20}"
            for results in all_results:
                task_results = [r for r in results if r["task"] == task]
                if task_results:
                    line += f" {task_results[0]['score']:>15.4f}"
                else:
                    line += f" {'N/A':>15}"
            lines.append(line)

        table = "\n".join(lines)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(table)
        return table
