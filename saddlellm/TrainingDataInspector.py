"""Fast local data preflight for LLM post-training stages."""
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class TrainingDataInspection:
    path: str
    task: str = "sft"
    exists: bool = False
    local_file: bool = False
    sampled_records: int = 0
    usable_records: int = 0
    detected_schemas: Dict[str, int] = field(default_factory=dict)
    avg_prompt_chars: float = 0.0
    avg_completion_chars: float = 0.0
    duplicate_records: int = 0
    identical_preference_pairs: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def usable_ratio(self) -> float:
        return self.usable_records / self.sampled_records if self.sampled_records else 0.0

    @property
    def ready(self) -> bool:
        return not self.errors

    @property
    def severity(self) -> str:
        if self.errors:
            return "error"
        if self.warnings:
            return "warning"
        return "ok"

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["usable_ratio"] = self.usable_ratio
        data["ready"] = self.ready
        data["severity"] = self.severity
        return data


class TrainingDataInspector:
    """Inspect a small sample before expensive model loading starts."""

    @classmethod
    def inspect_file(cls, path: str, task: str = "sft", max_records: int = 256) -> TrainingDataInspection:
        report = TrainingDataInspection(path=path, task=task)
        if not path:
            report.errors.append("No data path provided.")
            report.recommendations.append("Set the stage data_path or provide an upstream stage that materializes data.")
            return report
        report.exists = os.path.exists(path)
        report.local_file = os.path.isfile(path)
        if not report.exists:
            report.warnings.append("Path does not exist locally; assuming it may be a remote dataset id.")
            report.recommendations.append("For remote datasets, run a small local export through inspect_training_data before full training.")
            return report
        if not report.local_file:
            report.warnings.append("Path exists but is not a file; skipping fast local inspection.")
            report.recommendations.append("Point data_path at a JSONL/JSON/CSV file for strict preflight validation.")
            return report

        from .PostTrainingData import PostTrainingDataAdapter

        try:
            prompt_chars = 0
            completion_chars = 0
            seen_signatures = set()
            for record in cls._iter_records(path, max_records=max_records):
                report.sampled_records += 1
                schema = PostTrainingDataAdapter.detect_schema(record)
                report.detected_schemas[schema] = report.detected_schemas.get(schema, 0) + 1
                normalized = PostTrainingDataAdapter.normalize_record(record, task=task)
                if normalized is not None:
                    report.usable_records += 1
                    signature = cls._record_signature(normalized)
                    if signature in seen_signatures:
                        report.duplicate_records += 1
                    seen_signatures.add(signature)
                    p_len, c_len = cls._lengths(normalized)
                    prompt_chars += p_len
                    completion_chars += c_len
                    if cls._has_identical_preference_pair(normalized):
                        report.identical_preference_pairs += 1
        except Exception as exc:
            report.errors.append(f"Inspection failed: {exc}")
            report.recommendations.append("Verify the file extension, encoding, and JSON/CSV structure.")
        if report.usable_records:
            report.avg_prompt_chars = round(prompt_chars / report.usable_records, 2)
            report.avg_completion_chars = round(completion_chars / report.usable_records, 2)
        if report.sampled_records == 0 and not report.errors:
            report.errors.append("No records were sampled from the local file.")
            report.recommendations.append("Check that the file is non-empty and uses JSONL/JSON/CSV format.")
        if report.sampled_records and report.usable_records == 0:
            report.errors.append("No sampled records matched the expected training schema.")
            report.recommendations.append(cls._schema_recommendation(task))
        elif report.sampled_records and report.usable_ratio < 0.8:
            report.warnings.append(f"Only {report.usable_ratio:.0%} of sampled records matched the expected schema.")
            report.recommendations.append("Normalize the dataset before training to avoid silent sample drops.")
        if report.duplicate_records:
            report.warnings.append(f"Detected {report.duplicate_records} duplicate usable records in the inspected sample.")
            report.recommendations.append("Deduplicate post-training data before long runs.")
        if report.identical_preference_pairs:
            report.errors.append(f"Detected {report.identical_preference_pairs} preference rows where chosen and rejected are identical.")
            report.recommendations.append("Remove or fix preference rows with identical chosen/rejected completions.")
        return report

    @staticmethod
    def _schema_recommendation(task: str) -> str:
        task = (task or "sft").lower()
        if task in {"preference", "dpo", "orpo"}:
            return "Expected preference rows with prompt plus chosen/rejected completions."
        if task == "kto":
            return "Expected KTO rows with prompt, completion, and label/score/accepted."
        if task in {"rl", "rlhf", "grpo"}:
            return "Expected RL rows with prompt/query/instruction fields."
        return "Expected SFT rows with messages, text, or instruction plus output/response/answer."

    @staticmethod
    def _record_signature(record: Dict) -> str:
        return json.dumps(record, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _lengths(record: Dict) -> Tuple[int, int]:
        if record.get("prompt") is not None:
            prompt_len = len(str(record.get("prompt", "")))
            if record.get("chosen") is not None:
                completion_len = max(len(str(record.get("chosen", ""))), len(str(record.get("rejected", ""))))
            else:
                completion_len = len(str(record.get("completion", record.get("reference_answer", ""))))
            return prompt_len, completion_len
        if record.get("text") is not None:
            return 0, len(str(record.get("text", "")))
        messages = record.get("messages", [])
        prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") != "assistant")
        completion = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "assistant")
        return len(prompt), len(completion)

    @staticmethod
    def _has_identical_preference_pair(record: Dict) -> bool:
        return record.get("chosen") is not None and str(record.get("chosen", "")) == str(record.get("rejected", ""))

    @staticmethod
    def _iter_records(path: str, max_records: int) -> Iterable[Dict]:
        ext = os.path.splitext(path)[1].lower()
        count = 0
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
                    count += 1
                    if count >= max_records:
                        return
        elif ext == ".json":
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ("data", "train", "instances", "examples"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                else:
                    data = [data]
            for item in data[:max_records] if isinstance(data, list) else []:
                if isinstance(item, dict):
                    yield item
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    yield dict(row)
                    count += 1
                    if count >= max_records:
                        return
        else:
            raise ValueError(f"Unsupported local inspection file type: {path}")


def inspect_training_data(path: str, task: str = "sft", max_records: int = 256) -> Dict:
    return TrainingDataInspector.inspect_file(path=path, task=task, max_records=max_records).to_dict()
