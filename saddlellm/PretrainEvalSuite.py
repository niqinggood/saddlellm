"""Pretraining evaluation suite management.

This layer keeps heldout perplexity, domain QA/generation checks, benchmark
contamination references, and run-to-run comparisons in one artifact.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class EvalTaskSpec:
    name: str
    dataset: str
    metrics: List[str] = field(default_factory=lambda: ["perplexity"])
    dataset_config: Optional[str] = None
    split: str = "validation"
    task_type: str = "generation"
    text_column: Optional[str] = None
    target_column: Optional[str] = None
    max_samples: int = 1000
    max_length: int = 512
    max_new_tokens: int = 128
    contamination_refs: List[str] = field(default_factory=list)
    contamination_threshold: float = 0.35
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvalSuiteResult:
    suite_name: str
    model_path: str
    results: Dict[str, Dict]
    contamination: Dict[str, Dict] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "suite_name": self.suite_name,
            "model_path": self.model_path,
            "results": self.results,
            "contamination": self.contamination,
            "summary": self.summary,
        }


class PretrainEvalSuite:
    """Run and compare a fixed set of pretraining evaluations."""

    DOMAIN_DEFAULTS = {
        "medical": [
            ("heldout_general_ppl", "wikitext", ["perplexity"], "wikitext-2-raw-v1"),
            ("clinical_heldout_ppl", "./eval/medical_heldout.jsonl", ["perplexity"], None),
            ("clinical_qa", "./eval/medical_qa.jsonl", ["rouge", "bleu"], None),
        ],
        "biology": [
            ("heldout_general_ppl", "wikitext", ["perplexity"], "wikitext-2-raw-v1"),
            ("biology_heldout_ppl", "./eval/biology_heldout.jsonl", ["perplexity"], None),
            ("biology_qa", "./eval/biology_qa.jsonl", ["rouge", "bleu"], None),
        ],
        "research": [
            ("heldout_general_ppl", "wikitext", ["perplexity"], "wikitext-2-raw-v1"),
            ("paper_heldout_ppl", "./eval/research_heldout.jsonl", ["perplexity"], None),
            ("citation_summary", "./eval/research_qa.jsonl", ["rouge", "bleu"], None),
        ],
        "risk": [
            ("heldout_general_ppl", "wikitext", ["perplexity"], "wikitext-2-raw-v1"),
            ("policy_heldout_ppl", "./eval/risk_heldout.jsonl", ["perplexity"], None),
            ("risk_qa", "./eval/risk_qa.jsonl", ["rouge", "bleu"], None),
        ],
        "semiconductor": [
            ("heldout_general_ppl", "wikitext", ["perplexity"], "wikitext-2-raw-v1"),
            ("process_heldout_ppl", "./eval/semiconductor_heldout.jsonl", ["perplexity"], None),
            ("process_qa", "./eval/semiconductor_qa.jsonl", ["rouge", "bleu"], None),
        ],
    }

    def __init__(self, name: str, tasks: Sequence[EvalTaskSpec]):
        self.name = name
        self.tasks = list(tasks)

    @classmethod
    def for_domain(cls, domain: str, max_samples: int = 500) -> "PretrainEvalSuite":
        key = (domain or "research").lower()
        raw_tasks = cls.DOMAIN_DEFAULTS.get(key, cls.DOMAIN_DEFAULTS["research"])
        tasks = [
            EvalTaskSpec(
                name=name,
                dataset=dataset,
                metrics=metrics,
                dataset_config=dataset_config,
                split="test" if dataset == "wikitext" else "validation",
                target_column="answer" if any(m in metrics for m in ("rouge", "bleu")) else None,
                max_samples=max_samples,
                notes=f"default {key} pretraining eval",
            )
            for name, dataset, metrics, dataset_config in raw_tasks
        ]
        return cls(name=f"{key}-pretrain-eval", tasks=tasks)

    @classmethod
    def from_dict(cls, data: Dict) -> "PretrainEvalSuite":
        return cls(
            name=data.get("name", "pretrain-eval"),
            tasks=[EvalTaskSpec(**task) for task in data.get("tasks", [])],
        )

    @classmethod
    def from_json(cls, path: str) -> "PretrainEvalSuite":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict:
        return {"name": self.name, "tasks": [t.to_dict() for t in self.tasks]}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def run(
        self,
        model_path: str,
        output_dir: str,
        tokenizer_path: Optional[str] = None,
        batch_size: int = 4,
        cpu: bool = False,
        skip_missing_local: bool = True,
    ) -> EvalSuiteResult:
        from .DataStrategy import ContaminationDetector
        from .LLModelEvalute import Evaluator

        os.makedirs(output_dir, exist_ok=True)
        evaluator = Evaluator()
        results: Dict[str, Dict] = {}
        contamination: Dict[str, Dict] = {}

        for task in self.tasks:
            if skip_missing_local and self._is_missing_local(task.dataset):
                results[task.name] = {"skipped": True, "reason": f"missing local dataset: {task.dataset}"}
                continue
            task_output = os.path.join(output_dir, task.name)
            result = evaluator.evaluate(
                model_path=model_path,
                dataset=task.dataset,
                dataset_config=task.dataset_config,
                metrics=task.metrics,
                max_samples=task.max_samples,
                task_type=task.task_type,
                split=task.split,
                text_column=task.text_column,
                target_column=task.target_column,
                max_length=task.max_length,
                max_new_tokens=task.max_new_tokens,
                batch_size=batch_size,
                cpu=cpu,
                output_dir=task_output,
            )
            result["evaluated_model_path"] = model_path
            if tokenizer_path:
                result["tokenizer_path"] = tokenizer_path
            results[task.name] = result

            if task.contamination_refs and not self._is_missing_local(task.dataset):
                detector = ContaminationDetector(threshold=task.contamination_threshold).add_references(
                    task.contamination_refs
                )
                contamination[task.name] = detector.scan_files(
                    task.dataset,
                    text_column=task.text_column or "text",
                    max_samples=task.max_samples,
                ).to_dict()

        suite_result = EvalSuiteResult(
            suite_name=self.name,
            model_path=model_path,
            results=results,
            contamination=contamination,
            summary=self.summarize(results),
        )
        with open(os.path.join(output_dir, "suite_results.json"), "w", encoding="utf-8") as f:
            json.dump(suite_result.to_dict(), f, ensure_ascii=False, indent=2)
        return suite_result

    def summarize(self, results: Dict[str, Dict]) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for task_name, result in results.items():
            if result.get("skipped") or result.get("error"):
                continue
            for metric_name, metric in result.get("metrics", {}).items():
                if isinstance(metric, dict) and "score" in metric:
                    summary[f"{task_name}.{metric_name}"] = metric["score"]
        return summary

    @staticmethod
    def compare(result_paths: Sequence[str], output_path: Optional[str] = None) -> Dict:
        runs = []
        for path in result_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            runs.append({"path": path, **data})
        metrics = sorted({k for run in runs for k in run.get("summary", {})})
        rows = []
        for metric in metrics:
            values = [run.get("summary", {}).get(metric) for run in runs]
            baseline = next((v for v in values if isinstance(v, (int, float))), None)
            deltas = [
                (v - baseline) if isinstance(v, (int, float)) and baseline is not None else None
                for v in values
            ]
            rows.append({"metric": metric, "values": values, "deltas_from_first": deltas})
        comparison = {"runs": [run["path"] for run in runs], "metrics": rows}
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
        return comparison

    def _is_missing_local(self, dataset: str) -> bool:
        return bool(dataset) and dataset.startswith((".", "/", "\\")) and not os.path.exists(dataset)
