"""Fast local data preflight for LLM post-training stages."""
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from itertools import combinations
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
    prompt_prefix_samples: int = 0
    unique_prompt_prefixes: int = 0
    dominant_prompt_prefix_ratio: float = 0.0
    completion_shape_samples: int = 0
    unique_completion_shapes: int = 0
    dominant_completion_shape_ratio: float = 0.0
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


@dataclass
class AutoPostTrainingDecision:
    """Explainable selection of one stable post-training objective."""

    path: str
    selected_stage: str
    sampled_records: int
    task_counts: Dict[str, int]
    confidence: float
    reasons: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    inspection: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


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
            prompt_prefixes: Counter[str] = Counter()
            completion_shapes: Counter[str] = Counter()
            for record in cls._iter_records(path, max_records=max_records):
                report.sampled_records += 1
                schema = PostTrainingDataAdapter.detect_schema(record)
                report.detected_schemas[schema] = report.detected_schemas.get(schema, 0) + 1
                try:
                    normalized = PostTrainingDataAdapter.normalize_record(record, task=task)
                except ValueError as exc:
                    raise ValueError(f"Record {report.sampled_records}: {exc}") from exc
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
                    if (task or "sft").lower() == "sft":
                        prompt_prefix = cls._prompt_prefix_signature(normalized)
                        if prompt_prefix:
                            prompt_prefixes[prompt_prefix] += 1
                        completion_shape = cls._completion_shape_signature(normalized)
                        if completion_shape:
                            completion_shapes[completion_shape] += 1
        except Exception as exc:
            report.errors.append(f"Inspection failed: {exc}")
            report.recommendations.append("Verify the file extension, encoding, and JSON/CSV structure.")
        if report.usable_records:
            report.avg_prompt_chars = round(prompt_chars / report.usable_records, 2)
            report.avg_completion_chars = round(completion_chars / report.usable_records, 2)
        if prompt_prefixes:
            report.prompt_prefix_samples = sum(prompt_prefixes.values())
            report.unique_prompt_prefixes = len(prompt_prefixes)
            report.dominant_prompt_prefix_ratio = round(
                max(prompt_prefixes.values()) / report.prompt_prefix_samples, 4
            )
        if completion_shapes:
            report.completion_shape_samples = sum(completion_shapes.values())
            report.unique_completion_shapes = len(completion_shapes)
            report.dominant_completion_shape_ratio = round(
                max(completion_shapes.values()) / report.completion_shape_samples, 4
            )
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
        if (
            (task or "sft").lower() == "sft"
            and report.prompt_prefix_samples >= 20
            and report.dominant_prompt_prefix_ratio >= 0.8
        ):
            report.warnings.append(
                "Prompt-template concentration is "
                f"{report.dominant_prompt_prefix_ratio:.0%} "
                f"({report.unique_prompt_prefixes} unique leading template(s) across "
                f"{report.prompt_prefix_samples} inspected SFT rows); this can cause template overfitting."
            )
            report.recommendations.append(
                "Add natural, paraphrased, and no-wrapper prompt variants; keep variants of the same source example in the same data split."
            )
            if report.dominant_completion_shape_ratio >= 0.9:
                report.warnings.append(
                    "Prompt and completion formats are both highly concentrated "
                    f"(completion shape {report.dominant_completion_shape_ratio:.0%}); "
                    "the model may learn formatting cues instead of the task."
                )
                report.recommendations.append(
                    "Vary response forms where the task permits, while preserving an explicit output contract for structured tasks."
                )
        return report

    @classmethod
    def select_post_training(
        cls, path: str, max_records: int = 256
    ) -> AutoPostTrainingDecision:
        """Choose SFT, DPO, or KTO from an unambiguous local dataset.

        Auto selection deliberately fails closed for mixed schemas.  Combining
        unrelated objectives or silently dropping records is much more damaging
        than asking the caller to split the data and provide an explicit order.
        """

        if not path or not os.path.isfile(path):
            raise ValueError(
                "Auto post-training requires a local JSONL, JSON, or CSV file "
                "so its schema can be inspected before training."
            )

        task_counts: Counter[str] = Counter()
        kto_labels = set()
        sampled_records = 0
        for record in cls._iter_records(path, max_records=max_records):
            sampled_records += 1
            task = cls._auto_post_training_task(record)
            task_counts[task] += 1
            if task == "kto":
                try:
                    kto_labels.add(cls._coerce_binary_label(record))
                except ValueError as exc:
                    raise ValueError(f"Record {sampled_records}: {exc}") from exc

        if sampled_records == 0:
            raise ValueError("Auto post-training could not sample any data records.")
        unknown = task_counts.get("unknown", 0)
        recognized = [
            task
            for task in ("sft", "dpo", "kto")
            if task_counts.get(task, 0)
        ]
        if unknown or len(recognized) != 1:
            counts = ", ".join(
                f"{task}={count}" for task, count in sorted(task_counts.items())
            )
            raise ValueError(
                "Auto post-training requires one unambiguous data contract; "
                f"sampled task counts: {counts}. Split mixed data into separate "
                "files and specify an ordered --stage command."
            )

        selected = recognized[0]
        if selected == "kto" and kto_labels != {False, True}:
            raise ValueError(
                "Auto detected KTO data, but the inspected sample does not contain "
                "both desirable (true) and undesirable (false) labels."
            )

        report = cls.inspect_file(path, task=selected, max_records=max_records)
        if not report.ready:
            raise ValueError(
                f"Auto selected {selected.upper()}, but data inspection failed: "
                + "; ".join(report.errors)
            )

        if selected == "dpo":
            reason = (
                "Every sampled row provides a prompt with chosen and rejected "
                "answers, so paired preference optimization (DPO) is appropriate."
            )
        elif selected == "kto":
            reason = (
                "Every sampled row provides one completion with an explicit binary "
                "desirability label, and both label classes are present."
            )
        else:
            reason = (
                "Every sampled row provides a supervised target using messages, "
                "instruction/output, text, or prompt/completion format."
            )
        return AutoPostTrainingDecision(
            path=path,
            selected_stage=selected,
            sampled_records=sampled_records,
            task_counts=dict(task_counts),
            confidence=1.0,
            reasons=[reason],
            guardrails=[
                "PPO is never selected automatically.",
                "Mixed or unknown schemas are rejected instead of silently dropped.",
                "The selected stage still runs the normal data inspection and validation.",
            ],
            inspection=report.to_dict(),
        )

    @staticmethod
    def _auto_post_training_task(record: Dict) -> str:
        from .PostTrainingData import PostTrainingDataAdapter

        if "chosen" in record and "rejected" in record:
            return "dpo"
        if "prompt" in record:
            explicit_label = "completion" in record and any(
                key in record for key in ("label", "score", "accepted")
            )
            one_preference = ("chosen" in record) != ("rejected" in record)
            if explicit_label or one_preference:
                return "kto"
        schema = PostTrainingDataAdapter.detect_schema(record)
        if schema in {"messages", "sharegpt", "alpaca", "prompt_completion", "text"}:
            return "sft"
        return "unknown"

    @staticmethod
    def _coerce_binary_label(record: Dict) -> bool:
        from .PostTrainingData import parse_kto_label

        return parse_kto_label(record)

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

    @classmethod
    def _prompt_prefix_signature(cls, record: Dict) -> str:
        text = cls._prompt_text(record).replace("\r\n", "\n").strip()
        if not text:
            return ""
        paragraphs = re.split(r"\n\s*\n", text, maxsplit=1)
        if len(paragraphs) > 1 and paragraphs[0].strip():
            prefix = paragraphs[0]
        else:
            prefix = text
            for marker in ("题目：", "题目:", "问题：", "问题:", "Question:", "Input:", "输入："):
                index = prefix.find(marker)
                if index > 0:
                    prefix = prefix[:index]
                    break
            else:
                lines = [line.strip() for line in prefix.splitlines() if line.strip()]
                if len(lines) > 1:
                    prefix = lines[0]
        return re.sub(r"\s+", " ", prefix).strip().casefold()[:512]

    @classmethod
    def _completion_shape_signature(cls, record: Dict) -> str:
        text = cls._completion_text(record).replace("\r\n", "\n").strip()
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        shapes = []
        for line in lines[:8]:
            if re.fullmatch(r"[A-Za-z0-9]{1,3}", line):
                shapes.append("short-token")
                continue
            label = re.match(r"^([^：:]{1,24})[：:]", line)
            if label:
                normalized_label = re.sub(r"\s+", " ", label.group(1)).strip().casefold()
                shapes.append(f"label:{normalized_label}")
                continue
            if re.match(r"^(?:[-*•]|\d+[.)、])\s*", line):
                shapes.append("list-item")
                continue
            shapes.append("text")
        return f"lines:{len(lines)}|" + "|".join(shapes)

    @staticmethod
    def _prompt_text(record: Dict) -> str:
        if record.get("prompt") is not None:
            return str(record.get("prompt", ""))
        messages = record.get("messages") or []
        if messages:
            user_messages = [
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "user"
            ]
            if user_messages:
                return "\n".join(user_messages)
            return "\n".join(
                str(message.get("content", ""))
                for message in messages
                if message.get("role") != "assistant"
            )
        return ""

    @staticmethod
    def _completion_text(record: Dict) -> str:
        if record.get("completion") is not None:
            return str(record.get("completion", ""))
        messages = record.get("messages") or []
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "assistant"
        )

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
                    if max_records > 0 and count >= max_records:
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
            selected = (
                data[:max_records]
                if isinstance(data, list) and max_records > 0
                else data
                if isinstance(data, list)
                else []
            )
            for item in selected:
                if isinstance(item, dict):
                    yield item
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    yield dict(row)
                    count += 1
                    if max_records > 0 and count >= max_records:
                        return
        else:
            raise ValueError(f"Unsupported local inspection file type: {path}")


def inspect_training_data(path: str, task: str = "sft", max_records: int = 256) -> Dict:
    return TrainingDataInspector.inspect_file(path=path, task=task, max_records=max_records).to_dict()


DEFAULT_GROUP_FIELDS = (
    "group_id",
    "case_id",
    "conversation_id",
    "patient_id",
    "document_id",
    "source_record_id",
)
# Prefer the unwrapped question when a dataset keeps both ``question`` and a
# templated ``instruction``.  This avoids treating the common instruction
# wrapper as evidence that otherwise different cases are near duplicates.
DEFAULT_CONTENT_FIELDS = ("question", "instruction", "prompt", "text")


def _nested_value(record: Dict, path: str):
    value = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _normalized_identity_text(value) -> str:
    text = unicodedata.normalize("NFKC", str("" if value is None else value)).casefold()
    return "".join(character for character in text if character.isalnum())


def _record_content(record: Dict, content_fields: Tuple[str, ...]) -> str:
    for field_name in content_fields:
        value = _nested_value(record, field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = record.get("messages")
    if isinstance(messages, list):
        prompt_messages = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") != "assistant"
        ]
        if any(item.strip() for item in prompt_messages):
            return "\n".join(prompt_messages).strip()
    return ""


def _record_groups(record: Dict, group_fields: Tuple[str, ...]) -> set[str]:
    groups: set[str] = set()
    for field_name in group_fields:
        value = _nested_value(record, field_name)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                continue
            normalized = _normalized_identity_text(item)
            if normalized:
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
                groups.add(f"{field_name}:sha256:{digest}")
    return groups


def _content_simhash(text: str) -> int:
    width = 3 if len(text) >= 12 else 1
    features = Counter(
        text[index : index + width]
        for index in range(max(1, len(text) - width + 1))
    )
    vector = [0] * 64
    for feature, weight in features.items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += weight if value & (1 << bit) else -weight
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def _near_duplicate_pairs(
    left: List[Dict],
    right: List[Dict],
    threshold: float,
    max_examples: int,
    include_content_previews: bool,
) -> Tuple[int, List[Dict]]:
    """Find high-similarity cross-split text using SimHash candidate bands."""

    right_bands: Dict[Tuple[int, int], List[int]] = {}
    for right_index, item in enumerate(right):
        fingerprint = item["simhash"]
        for band in range(8):
            key = (band, (fingerprint >> (band * 8)) & 0xFF)
            right_bands.setdefault(key, []).append(right_index)

    matches = 0
    examples: List[Dict] = []
    seen: set[Tuple[int, int]] = set()
    for left_index, left_item in enumerate(left):
        candidates: set[int] = set()
        fingerprint = left_item["simhash"]
        for band in range(8):
            key = (band, (fingerprint >> (band * 8)) & 0xFF)
            candidates.update(right_bands.get(key, ()))
        for right_index in candidates:
            pair = (left_index, right_index)
            if pair in seen:
                continue
            seen.add(pair)
            right_item = right[right_index]
            if left_item["signature"] == right_item["signature"]:
                continue
            left_text = left_item["normalized_content"]
            right_text = right_item["normalized_content"]
            if min(len(left_text), len(right_text)) < 20:
                continue
            length_similarity = 2 * min(len(left_text), len(right_text)) / (
                len(left_text) + len(right_text)
            )
            if length_similarity < threshold:
                continue
            if (fingerprint ^ right_item["simhash"]).bit_count() > 7:
                continue
            similarity = SequenceMatcher(None, left_text, right_text).ratio()
            if similarity < threshold:
                continue
            matches += 1
            if len(examples) < max_examples:
                example = {
                    "left_index": left_item["index"],
                    "right_index": right_item["index"],
                    "similarity": round(similarity, 4),
                    "left_sha256": left_item["signature"],
                    "right_sha256": right_item["signature"],
                }
                if include_content_previews:
                    example["left_preview"] = left_item["content"][:160]
                    example["right_preview"] = right_item["content"][:160]
                examples.append(example)
    return matches, examples


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_training_splits(
    train_path: str,
    validation_path: Optional[str] = None,
    test_path: Optional[str] = None,
    *,
    group_fields: Optional[Iterable[str]] = None,
    content_fields: Optional[Iterable[str]] = None,
    near_duplicate_threshold: float = 0.96,
    strict: bool = False,
    max_examples: int = 20,
    include_content_previews: bool = False,
) -> Dict:
    """Audit group, exact-text, and likely near-duplicate leakage across splits.

    Group and exact-content overlaps always block.  Approximate near-duplicates
    and missing group ids are warnings unless ``strict`` is enabled, because
    fixed prompt wrappers can otherwise create false positives.  Domain data
    owns the group ids; the core only interprets the declared fields.
    """

    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    if max_examples < 0:
        raise ValueError("max_examples cannot be negative")
    resolved_group_fields = tuple(
        item.strip()
        for item in (group_fields or DEFAULT_GROUP_FIELDS)
        if item and item.strip()
    )
    resolved_content_fields = tuple(
        item.strip()
        for item in (content_fields or DEFAULT_CONTENT_FIELDS)
        if item and item.strip()
    )
    if not resolved_content_fields:
        raise ValueError("at least one content field is required")

    supplied = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }
    paths = {name: os.path.abspath(path) for name, path in supplied.items() if path}
    if len(paths) < 2:
        raise ValueError("split audit requires at least two data files")

    split_records: Dict[str, List[Dict]] = {}
    split_summaries: Dict[str, Dict] = {}
    warnings: List[str] = []
    issues: List[str] = []
    for split_name, path in paths.items():
        if not os.path.isfile(path):
            raise ValueError(f"{split_name} data file does not exist: {path}")
        prepared: List[Dict] = []
        group_records = 0
        missing_content = 0
        for index, record in enumerate(
            TrainingDataInspector._iter_records(path, max_records=0)
        ):
            content = _record_content(record, resolved_content_fields)
            normalized_content = _normalized_identity_text(content)
            if not normalized_content:
                missing_content += 1
                continue
            groups = _record_groups(record, resolved_group_fields)
            group_records += bool(groups)
            prepared.append(
                {
                    "index": index,
                    "content": content,
                    "normalized_content": normalized_content,
                    "signature": hashlib.sha256(
                        normalized_content.encode("utf-8")
                    ).hexdigest(),
                    "groups": groups,
                    "simhash": _content_simhash(normalized_content),
                }
            )
        if not prepared:
            issues.append(f"{split_name} has no records with auditable prompt content")
        missing_groups = len(prepared) - group_records
        split_records[split_name] = prepared
        split_summaries[split_name] = {
            "path": path,
            "sha256": _sha256_file(path),
            "records": len(prepared),
            "records_with_group": group_records,
            "records_missing_group": max(0, missing_groups),
            "records_missing_content": missing_content,
            "group_coverage": round(group_records / len(prepared), 4)
            if prepared
            else 0.0,
        }
        if missing_groups:
            message = (
                f"{split_name} has {missing_groups}/{len(prepared)} records without "
                "a configured group id; text leakage is still checked"
            )
            (issues if strict else warnings).append(message)

    pair_results: Dict[str, Dict] = {}
    cross_split_overlap: Dict[str, int] = {}
    for left_name, right_name in combinations(paths, 2):
        left = split_records[left_name]
        right = split_records[right_name]
        pair_name = f"{left_name}_{right_name}"
        left_groups = {group for item in left for group in item["groups"]}
        right_groups = {group for item in right for group in item["groups"]}
        shared_groups = sorted(left_groups & right_groups)
        left_signatures = {item["signature"] for item in left}
        right_signatures = {item["signature"] for item in right}
        shared_signatures = left_signatures & right_signatures
        exact_examples = []
        if shared_signatures:
            right_by_signature = {item["signature"]: item for item in right}
            for item in left:
                if item["signature"] in shared_signatures:
                    match = right_by_signature[item["signature"]]
                    example = {
                        "left_index": item["index"],
                        "right_index": match["index"],
                        "content_sha256": item["signature"],
                    }
                    if include_content_previews:
                        example["preview"] = item["content"][:160]
                    exact_examples.append(example)
                    if len(exact_examples) >= max_examples:
                        break
        near_count, near_examples = _near_duplicate_pairs(
            left,
            right,
            near_duplicate_threshold,
            max_examples,
            include_content_previews,
        )
        blocking = len(shared_groups) + len(shared_signatures)
        if strict:
            blocking += near_count
        cross_split_overlap[pair_name] = blocking
        pair_results[pair_name] = {
            "group_overlap": len(shared_groups),
            "group_examples": shared_groups[:max_examples],
            "exact_content_overlap": len(shared_signatures),
            "exact_content_examples": exact_examples,
            "near_duplicate_overlap": near_count,
            "near_duplicate_examples": near_examples,
            "near_duplicate_threshold": near_duplicate_threshold,
            "blocking_overlap": blocking,
        }
        if shared_groups:
            issues.append(
                f"{pair_name} shares {len(shared_groups)} configured group id(s)"
            )
        if shared_signatures:
            issues.append(
                f"{pair_name} shares {len(shared_signatures)} exact normalized prompt(s)"
            )
        if near_count:
            message = (
                f"{pair_name} contains {near_count} likely near-duplicate prompt pair(s) "
                f"at similarity >= {near_duplicate_threshold:.2f}"
            )
            (issues if strict else warnings).append(message)

    return {
        "schema_version": 1,
        "kind": "training_split_leakage",
        "passed": not issues and not any(cross_split_overlap.values()),
        "status": "passed" if not issues and not any(cross_split_overlap.values()) else "failed",
        "strict": strict,
        "group_fields": list(resolved_group_fields),
        "content_fields": list(resolved_content_fields),
        "content_previews_included": include_content_previews,
        "splits": split_summaries,
        "pairs": pair_results,
        "cross_split_overlap": cross_split_overlap,
        "issues": issues,
        "warnings": warnings,
        "limitations": [
            "Near-duplicate matching is an approximate high-similarity screen.",
            "Group identifiers are hashed in reports; raw content previews are omitted unless explicitly enabled.",
            "A clean report does not prove that copyrighted, private, or benchmark-contaminated data is absent.",
        ],
    }
