"""Post-training dataset normalization.

This module is the shared data adapter layer for SFT, preference training, and
rollout-derived alignment data.  It accepts common schemas used by
LLaMA-Factory, TRL/Open-Instruct style datasets, ShareGPT exports, Alpaca
records, and simple prompt/completion files, then emits a small canonical
schema that SaddleLLM trainers can consume consistently.
"""
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "instruction": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
}


@dataclass
class NormalizationReport:
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    task: str = "sft"
    input_format: str = "auto"
    total_records: int = 0
    kept_records: int = 0
    dropped_records: int = 0
    detected_schemas: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class PostTrainingDataAdapter:
    """Normalize common post-training record formats."""

    @classmethod
    def normalize_record(
        cls,
        record: Dict,
        task: str = "sft",
        source_format: str = "auto",
        keep_metadata: bool = True,
    ) -> Optional[Dict]:
        task = (task or "sft").lower()
        if task in {"preference", "dpo", "orpo"}:
            return cls.normalize_preference_record(record, source_format, keep_metadata)
        if task == "kto":
            return cls.normalize_kto_record(record, source_format, keep_metadata)
        if task in {"rl", "rlhf", "grpo"}:
            return cls.normalize_rl_record(record, source_format, keep_metadata)
        return cls.normalize_sft_record(record, source_format, keep_metadata)

    @classmethod
    def normalize_sft_record(
        cls,
        record: Dict,
        source_format: str = "auto",
        keep_metadata: bool = True,
    ) -> Optional[Dict]:
        messages = cls._extract_messages(record)
        if messages:
            if not cls._has_assistant(messages):
                return None
            normalized = {"messages": messages}
        elif record.get("text"):
            normalized = {"text": str(record["text"])}
        else:
            instruction = cls._first(record, ["instruction", "question", "prompt", "query", "input"])
            extra_input = str(record.get("input", "") or "") if record.get("instruction") else ""
            output = cls._first(record, ["output", "response", "answer", "completion", "target"])
            if not instruction or not output:
                return None
            user_content = f"{instruction}\n{extra_input}".strip() if extra_input else str(instruction)
            normalized = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": str(output)},
                ],
                "instruction": str(instruction),
                "input": extra_input,
                "output": str(output),
            }
        return cls._attach_metadata(normalized, record, "sft", source_format, keep_metadata)

    @classmethod
    def normalize_preference_record(
        cls,
        record: Dict,
        source_format: str = "auto",
        keep_metadata: bool = True,
    ) -> Optional[Dict]:
        prompt = cls._prompt_from_record(record)
        chosen = cls._completion_text(cls._first_value(record, ["chosen", "preferred", "accept", "positive", "winner"]))
        rejected = cls._completion_text(cls._first_value(record, ["rejected", "reject", "negative", "loser"]))
        if not prompt or chosen is None or rejected is None:
            return None
        normalized = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        return cls._attach_metadata(normalized, record, "preference", source_format, keep_metadata)

    @classmethod
    def normalize_kto_record(
        cls,
        record: Dict,
        source_format: str = "auto",
        keep_metadata: bool = True,
    ) -> Optional[Dict]:
        prompt = cls._prompt_from_record(record)
        completion = cls._completion_text(cls._first_value(record, ["completion", "response", "answer", "output", "chosen", "rejected"]))
        if not prompt or completion is None:
            return None
        label = record.get("label", record.get("score", record.get("accepted")))
        if isinstance(label, str):
            label = label.lower() in {"1", "true", "yes", "chosen", "positive", "accepted"}
        elif label is None:
            label = bool(record.get("chosen")) and not bool(record.get("rejected"))
        normalized = {
            "prompt": prompt,
            "completion": completion,
            "label": bool(label),
        }
        return cls._attach_metadata(normalized, record, "kto", source_format, keep_metadata)

    @classmethod
    def normalize_rl_record(
        cls,
        record: Dict,
        source_format: str = "auto",
        keep_metadata: bool = True,
    ) -> Optional[Dict]:
        prompt = cls._prompt_from_record(record)
        if not prompt:
            return None
        normalized = {
            "prompt": prompt,
            "reference_answer": cls._first(record, ["answer", "reference", "target", "output"]) or "",
            "reward_model": record.get("reward_model", record.get("verifier", "")),
            "metadata": dict(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {},
        }
        return cls._attach_metadata(normalized, record, "rl", source_format, keep_metadata)

    @classmethod
    def normalize_records(
        cls,
        records: Iterable[Dict],
        task: str = "sft",
        source_format: str = "auto",
        keep_metadata: bool = True,
    ) -> (List[Dict], NormalizationReport):
        normalized: List[Dict] = []
        report = NormalizationReport(task=task, input_format=source_format)
        for record in records:
            report.total_records += 1
            detected = cls.detect_schema(record)
            report.detected_schemas[detected] = report.detected_schemas.get(detected, 0) + 1
            item = cls.normalize_record(record, task=task, source_format=source_format, keep_metadata=keep_metadata)
            if item is None:
                report.dropped_records += 1
                continue
            normalized.append(item)
            report.kept_records += 1
        if report.kept_records == 0:
            report.warnings.append("No records survived normalization.")
        return normalized, report

    @classmethod
    def normalize_file(
        cls,
        input_path: str,
        output_path: str,
        task: str = "sft",
        source_format: str = "auto",
        keep_metadata: bool = True,
    ) -> NormalizationReport:
        records = cls.load_records(input_path)
        normalized, report = cls.normalize_records(records, task=task, source_format=source_format, keep_metadata=keep_metadata)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in normalized:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        report.input_path = input_path
        report.output_path = output_path
        report_path = output_path + ".report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        return report

    @staticmethod
    def load_records(path: str) -> List[Dict]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".jsonl":
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            for key in ("data", "train", "instances", "examples"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        if ext == ".csv":
            with open(path, "r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        if ext in {".txt", ".md"}:
            with open(path, "r", encoding="utf-8") as f:
                return [{"text": f.read()}]
        raise ValueError(f"Unsupported post-training data file: {path}")

    @staticmethod
    def detect_schema(record: Dict) -> str:
        if "messages" in record:
            return "messages"
        if "conversations" in record:
            return "sharegpt"
        if "chosen" in record and "rejected" in record:
            return "preference_pair"
        if "instruction" in record and "output" in record:
            return "alpaca"
        if "prompt" in record and "completion" in record:
            return "prompt_completion"
        if "text" in record:
            return "text"
        return "unknown"

    @classmethod
    def _extract_messages(cls, record: Dict) -> List[Dict[str, str]]:
        raw = record.get("messages") or record.get("conversations") or record.get("conversation")
        if not raw:
            return []
        messages = []
        for message in raw:
            if not isinstance(message, dict):
                continue
            role = message.get("role", message.get("from", message.get("speaker", "user")))
            content = message.get("content", message.get("value", message.get("text", "")))
            role = ROLE_ALIASES.get(str(role).lower(), str(role).lower())
            content = str(content or "").strip()
            if role and content:
                messages.append({"role": role, "content": content})
        return messages

    @classmethod
    def _prompt_from_record(cls, record: Dict) -> str:
        prompt = cls._first(record, ["prompt", "instruction", "question", "query", "input"])
        if prompt:
            return str(prompt)
        messages = cls._extract_messages(record)
        if not messages:
            return ""
        prompt_messages = []
        for message in messages:
            if message["role"] == "assistant":
                break
            prompt_messages.append(f"{message['role']}: {message['content']}")
        return "\n".join(prompt_messages).strip()

    @classmethod
    def _completion_text(cls, value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            messages = cls._extract_messages({"messages": value})
            assistant = [m["content"] for m in messages if m["role"] == "assistant"]
            return assistant[-1] if assistant else "\n".join(m["content"] for m in messages)
        if isinstance(value, dict):
            if "content" in value or "text" in value or "value" in value:
                return str(value.get("content", value.get("text", value.get("value", ""))))
            messages = cls._extract_messages({"messages": [value]})
            return messages[0]["content"] if messages else json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _first(record: Dict, keys: Sequence[str]) -> str:
        value = PostTrainingDataAdapter._first_value(record, keys)
        return "" if value is None else str(value)

    @staticmethod
    def _first_value(record: Dict, keys: Sequence[str]):
        for key in keys:
            if key in record and record[key] not in (None, ""):
                return record[key]
        return None

    @staticmethod
    def _has_assistant(messages: Sequence[Dict[str, str]]) -> bool:
        return any(message.get("role") == "assistant" for message in messages)

    @staticmethod
    def _attach_metadata(normalized: Dict, record: Dict, task: str, source_format: str, keep_metadata: bool) -> Dict:
        normalized["task"] = task
        normalized["source_format"] = source_format if source_format != "auto" else PostTrainingDataAdapter.detect_schema(record)
        if keep_metadata:
            metadata = dict(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {}
            for key in ("id", "source", "dataset", "category", "domain", "license"):
                if key in record:
                    metadata[key] = record[key]
            if metadata:
                normalized["metadata"] = metadata
        return normalized


def normalize_post_training_record(record: Dict, task: str = "sft", source_format: str = "auto") -> Optional[Dict]:
    return PostTrainingDataAdapter.normalize_record(record, task=task, source_format=source_format)


def normalize_post_training_file(
    input_path: str,
    output_path: str,
    task: str = "sft",
    source_format: str = "auto",
) -> Dict:
    return PostTrainingDataAdapter.normalize_file(
        input_path=input_path,
        output_path=output_path,
        task=task,
        source_format=source_format,
    ).to_dict()

