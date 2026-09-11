"""Vision-language-action utilities for embodied post-training.

The VLA layer normalizes robot trajectories into a multimodal SFT-compatible
schema and provides simple action tokenization helpers.  It deliberately avoids
binding SaddleLLM to a specific robot stack, simulator, or control frequency.
"""
import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .MultimodalData import MultimodalAsset


@dataclass
class VLAActionSpace:
    action_dim: int = 7
    action_type: str = "continuous"  # continuous | discrete | text
    bins: int = 256
    min_value: float = -1.0
    max_value: float = 1.0
    include_gripper: bool = True
    action_token_prefix: str = "<act_"
    action_token_suffix: str = ">"
    control_hz: float = 10.0
    coordinate_frame: str = "end_effector_delta"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VLAAction:
    values: List[float] = field(default_factory=list)
    text: str = ""
    gripper: Optional[float] = None
    coordinate_frame: str = "end_effector_delta"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VLASample:
    instruction: str
    action: VLAAction
    images: List[MultimodalAsset] = field(default_factory=list)
    proprio: Dict = field(default_factory=dict)
    messages: List[Dict[str, str]] = field(default_factory=list)
    task: str = "vla_sft"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "action": self.action.to_dict(),
            "images": [image.to_dict() for image in self.images],
            "proprio": self.proprio,
            "messages": self.messages,
            "task": self.task,
            "metadata": self.metadata,
        }


@dataclass
class VLANormalizationReport:
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    total_records: int = 0
    kept_records: int = 0
    dropped_records: int = 0
    detected_schemas: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VLACollatorConfig:
    max_seq_length: int = 2048
    train_on_prompt: bool = False
    pad_to_multiple_of: Optional[int] = 8
    image_token: str = "<image>"
    assistant_role: str = "assistant"
    user_role: str = "user"

    def to_dict(self) -> Dict:
        return asdict(self)


class VLAActionTokenizer:
    """Convert continuous robot actions into stable text tokens."""

    def __init__(self, action_space: Optional[VLAActionSpace] = None):
        self.action_space = action_space or VLAActionSpace()

    def encode(self, action: Union[VLAAction, Sequence[float], str]) -> str:
        if isinstance(action, str):
            return action
        if not isinstance(action, VLAAction):
            action = VLAAction(values=[float(v) for v in action])
        if action.text:
            return action.text
        values = list(action.values)
        if action.gripper is not None and self.action_space.include_gripper:
            values.append(float(action.gripper))
        tokens = [self._token_for_value(value) for value in values]
        return " ".join(tokens)

    def decode(self, text: str) -> VLAAction:
        token_re = re.escape(self.action_space.action_token_prefix) + r"(\d+)" + re.escape(self.action_space.action_token_suffix)
        bins = [int(match) for match in re.findall(token_re, text)]
        values = [self._value_for_bin(bin_id) for bin_id in bins]
        gripper = None
        if self.action_space.include_gripper and len(values) > self.action_space.action_dim:
            gripper = values[-1]
            values = values[:-1]
        return VLAAction(values=values, gripper=gripper, coordinate_frame=self.action_space.coordinate_frame)

    def action_vocab(self) -> List[str]:
        return [
            f"{self.action_space.action_token_prefix}{idx}{self.action_space.action_token_suffix}"
            for idx in range(self.action_space.bins)
        ]

    def special_tokens(self, include_image_token: bool = True, image_token: str = "<image>") -> List[str]:
        tokens = self.action_vocab()
        if include_image_token and image_token not in tokens:
            tokens = [image_token] + tokens
        return tokens

    def register_with_tokenizer(
        self,
        tokenizer: Any,
        include_image_token: bool = True,
        image_token: str = "<image>",
        resize_model: Optional[Any] = None,
    ) -> Dict:
        tokens = self.special_tokens(include_image_token=include_image_token, image_token=image_token)
        if hasattr(tokenizer, "add_special_tokens"):
            added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        elif hasattr(tokenizer, "add_tokens"):
            added = tokenizer.add_tokens(tokens, special_tokens=True)
        else:
            raise TypeError("tokenizer must provide add_special_tokens() or add_tokens()")
        if resize_model is not None and hasattr(resize_model, "resize_token_embeddings"):
            resize_model.resize_token_embeddings(len(tokenizer))
        token_ids = {}
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in tokens}
        return {
            "added_tokens": int(added or 0),
            "num_action_tokens": len(self.action_vocab()),
            "num_special_tokens": len(tokens),
            "image_token": image_token if include_image_token else None,
            "token_ids": token_ids,
        }

    def _token_for_value(self, value: float) -> str:
        lo = self.action_space.min_value
        hi = self.action_space.max_value
        clipped = min(max(float(value), lo), hi)
        ratio = (clipped - lo) / max(hi - lo, 1e-8)
        bin_id = int(round(ratio * (self.action_space.bins - 1)))
        return f"{self.action_space.action_token_prefix}{bin_id}{self.action_space.action_token_suffix}"

    def _value_for_bin(self, bin_id: int) -> float:
        bin_id = min(max(int(bin_id), 0), self.action_space.bins - 1)
        ratio = bin_id / max(self.action_space.bins - 1, 1)
        return self.action_space.min_value + ratio * (self.action_space.max_value - self.action_space.min_value)


class VLADataAdapter:
    """Normalize Open-X/RT-style and custom robot trajectory records."""

    IMAGE_TOKEN = "<image>"

    @classmethod
    def normalize_record(
        cls,
        record: Dict,
        image_root: Optional[str] = None,
        action_tokenizer: Optional[VLAActionTokenizer] = None,
    ) -> Optional[VLASample]:
        tokenizer = action_tokenizer or VLAActionTokenizer()
        instruction = cls._first(record, ["instruction", "prompt", "task", "goal", "language_instruction", "query"])
        action = cls._extract_action(record)
        if not instruction or action is None:
            return None
        images = cls._extract_images(record, image_root=image_root)
        if not images:
            return None
        proprio = cls._extract_proprio(record)
        action_text = tokenizer.encode(action)
        user_content = instruction
        if cls.IMAGE_TOKEN not in user_content:
            user_content = f"{cls.IMAGE_TOKEN}\n{user_content}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": action_text},
        ]
        metadata = dict(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {}
        for key in ("id", "episode_id", "step_id", "source", "dataset", "robot", "scene", "license"):
            if key in record:
                metadata[key] = record[key]
        return VLASample(
            instruction=str(instruction),
            action=action,
            images=images,
            proprio=proprio,
            messages=messages,
            metadata=metadata,
        )

    @classmethod
    def normalize_records(
        cls,
        records: Iterable[Dict],
        image_root: Optional[str] = None,
        action_space: Optional[VLAActionSpace] = None,
    ) -> (List[VLASample], VLANormalizationReport):
        tokenizer = VLAActionTokenizer(action_space)
        samples: List[VLASample] = []
        report = VLANormalizationReport()
        for record in records:
            expanded_records = cls._expand_trajectory_record(record)
            for item in expanded_records:
                report.total_records += 1
                schema = cls.detect_schema(item)
                report.detected_schemas[schema] = report.detected_schemas.get(schema, 0) + 1
                sample = cls.normalize_record(item, image_root=image_root, action_tokenizer=tokenizer)
                if sample is None:
                    report.dropped_records += 1
                    continue
                samples.append(sample)
                report.kept_records += 1
        if report.kept_records == 0:
            report.warnings.append("No VLA records survived normalization.")
        return samples, report

    @classmethod
    def normalize_file(
        cls,
        input_path: str,
        output_path: str,
        image_root: Optional[str] = None,
        action_space: Optional[VLAActionSpace] = None,
    ) -> VLANormalizationReport:
        records = cls.load_records(input_path)
        samples, report = cls.normalize_records(
            records,
            image_root=image_root or os.path.dirname(input_path),
            action_space=action_space,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        report.input_path = input_path
        report.output_path = output_path
        with open(output_path + ".report.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        return report

    @staticmethod
    def load_records(path: str) -> List[Dict]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".jsonl":
            rows = []
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if ext == ".json":
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            for key in ("data", "episodes", "trajectories", "steps", "annotations", "examples"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        if ext == ".csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        raise ValueError(f"Unsupported VLA data file: {path}")

    @staticmethod
    def detect_schema(record: Dict) -> str:
        if "language_instruction" in record and "action" in record:
            return "open_x_step"
        if "trajectory" in record or "steps" in record:
            return "trajectory"
        if "images" in record and ("actions" in record or "action" in record):
            return "vla_messages"
        if any(key in record for key in ("image", "image_path")) and ("actions" in record or "action" in record):
            return "vla_step"
        if "observation" in record and "action" in record:
            return "robot_observation_action"
        return "unknown"

    @classmethod
    def _expand_trajectory_record(cls, record: Dict) -> List[Dict]:
        steps = record.get("steps") or record.get("trajectory")
        if not isinstance(steps, list):
            return [record]
        base = {
            key: value
            for key, value in record.items()
            if key not in {"steps", "trajectory"}
        }
        expanded = []
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            item = dict(base)
            item.update(step)
            item.setdefault("episode_id", record.get("episode_id", record.get("id")))
            item.setdefault("step_id", step.get("step_id", idx))
            for key in ("instruction", "prompt", "task", "goal", "language_instruction"):
                if key not in item and key in base:
                    item[key] = base[key]
            expanded.append(item)
        return expanded or [record]

    @classmethod
    def _extract_action(cls, record: Dict) -> Optional[VLAAction]:
        raw = record.get("action", record.get("actions", record.get("target_action")))
        if raw is None:
            return None
        if isinstance(raw, str):
            return VLAAction(text=raw)
        if isinstance(raw, dict):
            values = raw.get("values", raw.get("world_vector", raw.get("delta", raw.get("joints", []))))
            if isinstance(values, str):
                try:
                    values = json.loads(values)
                except Exception:
                    values = []
            gripper = raw.get("gripper", raw.get("gripper_closedness_action"))
            frame = raw.get("coordinate_frame", raw.get("frame", "end_effector_delta"))
            metadata = {k: v for k, v in raw.items() if k not in {"values", "world_vector", "delta", "joints", "gripper", "gripper_closedness_action", "coordinate_frame", "frame"}}
            return VLAAction(
                values=[float(v) for v in values] if isinstance(values, list) else [],
                gripper=float(gripper) if gripper not in (None, "") else None,
                coordinate_frame=frame,
                metadata=metadata,
            )
        if isinstance(raw, list):
            return VLAAction(values=[float(v) for v in raw])
        return None

    @classmethod
    def _extract_images(cls, record: Dict, image_root: Optional[str]) -> List[MultimodalAsset]:
        observation = record.get("observation", {}) if isinstance(record.get("observation"), dict) else {}
        raw_images = (
            record.get("images")
            or record.get("image")
            or record.get("image_path")
            or observation.get("images")
            or observation.get("image")
            or observation.get("image_path")
        )
        if raw_images is None:
            return []
        if isinstance(raw_images, (str, dict)):
            raw_images = [raw_images]
        images: List[MultimodalAsset] = []
        for item in raw_images:
            metadata = {}
            if isinstance(item, dict):
                path = item.get("path", item.get("image", item.get("file_name", item.get("url", ""))))
                metadata = {k: v for k, v in item.items() if k not in {"path", "image", "file_name", "url"}}
            else:
                path = str(item)
            if not path:
                continue
            if image_root and not os.path.isabs(path) and not path.startswith(("http://", "https://")):
                path = os.path.abspath(os.path.join(image_root, path))
            images.append(MultimodalAsset(path=path, metadata=metadata))
        return images

    @staticmethod
    def _extract_proprio(record: Dict) -> Dict:
        observation = record.get("observation", {}) if isinstance(record.get("observation"), dict) else {}
        raw = record.get("proprio", record.get("state", record.get("robot_state", observation.get("proprio", observation.get("state", {})))))
        return dict(raw) if isinstance(raw, dict) else {"values": raw} if raw not in (None, "") else {}

    @staticmethod
    def _first(record: Dict, keys: Sequence[str]) -> str:
        for key in keys:
            value = record.get(key)
            if value not in (None, ""):
                return str(value)
        observation = record.get("observation", {}) if isinstance(record.get("observation"), dict) else {}
        for key in keys:
            value = observation.get(key)
            if value not in (None, ""):
                return str(value)
        return ""


class VLADataset:
    """Lightweight JSONL dataset for normalized VLA samples."""

    def __init__(self, samples: Sequence[Dict]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    @classmethod
    def from_file(cls, path: str) -> "VLADataset":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return cls(rows)


class VLADataCollator:
    """Tokenize normalized VLA chat samples for causal-LM behavior cloning.

    Images, proprioception, and decoded action metadata are carried through the
    batch for later multimodal collators.  The text labels default to the final
    assistant action only.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[VLACollatorConfig] = None,
        action_tokenizer: Optional[VLAActionTokenizer] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or VLACollatorConfig()
        self.action_tokenizer = action_tokenizer or VLAActionTokenizer()

    def __call__(self, features: Sequence[Dict]) -> Dict:
        import torch

        encoded_rows = [self._encode_feature(feature) for feature in features]
        max_len = max((len(row[0]) for row in encoded_rows), default=0)
        if self.config.pad_to_multiple_of and max_len:
            multiple = self.config.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple
        pad_id = self._pad_token_id()
        input_ids, attention_mask, labels = [], [], []
        for ids, prefix_len in encoded_rows:
            row_labels = list(ids) if self.config.train_on_prompt else [-100] * min(prefix_len, len(ids)) + list(ids[prefix_len:])
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(row_labels + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images": [feature.get("images", []) for feature in features],
            "proprio": [feature.get("proprio", {}) for feature in features],
            "actions": [feature.get("action", {}) for feature in features],
            "metadata": [feature.get("metadata", {}) for feature in features],
        }

    def _encode_feature(self, feature: Dict) -> Tuple[List[int], int]:
        messages = feature.get("messages") or self._messages_from_feature(feature)
        prefix, full = self._format_messages(messages)
        prefix_ids = self._tokenize(prefix)
        full_ids = self._tokenize(full)
        if self.config.max_seq_length and len(full_ids) > self.config.max_seq_length:
            overflow = len(full_ids) - self.config.max_seq_length
            full_ids = full_ids[overflow:]
            prefix_len = max(len(prefix_ids) - overflow, 0)
        else:
            prefix_len = len(prefix_ids)
        return full_ids, prefix_len

    def _format_messages(self, messages: Sequence[Dict[str, str]]) -> Tuple[str, str]:
        assistant_role = self.config.assistant_role
        assistant_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == assistant_role:
                assistant_idx = idx
                break
        if assistant_idx is None:
            text = "\n".join(self._format_line(message) for message in messages)
            return text, text
        prefix_messages = messages[:assistant_idx]
        answer = str(messages[assistant_idx].get("content", "")).strip()
        prefix = "".join(self._format_line(message) for message in prefix_messages)
        prefix += f"{assistant_role}: "
        return prefix, f"{prefix}{answer}\n"

    @staticmethod
    def _format_line(message: Dict[str, str]) -> str:
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        return f"{role}: {content}\n"

    def _messages_from_feature(self, feature: Dict) -> List[Dict[str, str]]:
        instruction = str(feature.get("instruction", "")).strip()
        action = feature.get("action", {})
        action_text = action.get("text") if isinstance(action, dict) else ""
        if not action_text and isinstance(action, dict):
            values = action.get("values", [])
            action_text = self.action_tokenizer.encode(values)
        if self.config.image_token and self.config.image_token not in instruction:
            instruction = f"{self.config.image_token}\n{instruction}".strip()
        return [
            {"role": self.config.user_role, "content": instruction},
            {"role": self.config.assistant_role, "content": str(action_text or "").strip()},
        ]

    def _tokenize(self, text: str) -> List[int]:
        output = self.tokenizer(text, add_special_tokens=False)
        ids = output["input_ids"] if isinstance(output, dict) else output.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(item) for item in ids]

    def _pad_token_id(self) -> int:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        return int(eos_id) if eos_id is not None else 0


class VLATrainingPlanner:
    """Create planning artifacts for VLA SFT and future RL fine-tuning."""

    @staticmethod
    def create_plan(
        data_path: Optional[str] = None,
        image_root: Optional[str] = None,
        action_space: Optional[VLAActionSpace] = None,
        output_dir: str = "./vla_plan",
    ) -> Dict:
        action_space = action_space or VLAActionSpace()
        tokenizer = VLAActionTokenizer(action_space)
        collator_config = VLACollatorConfig().to_dict()
        return {
            "stage": "vla_sft",
            "status": "planned",
            "data_path": data_path,
            "image_root": image_root,
            "output_dir": output_dir,
            "action_space": action_space.to_dict(),
            "action_vocab_size": len(tokenizer.action_vocab()),
            "action_special_tokens": tokenizer.special_tokens(),
            "collator_config": collator_config,
            "recommended_stages": ["vla_sft", "offline_preference", "sim_rollout_rl", "safety_eval"],
            "notes": [
                "Normalize robot steps to messages + images + action tokens first.",
                "Start with supervised behavior cloning before online robot or simulator RL.",
                "Register action_special_tokens with the tokenizer and resize model embeddings before training.",
                "Keep action coordinate frames and control frequency in metadata for deployment.",
            ],
        }


def normalize_vla_file(
    input_path: str,
    output_path: str,
    image_root: Optional[str] = None,
    action_space: Optional[VLAActionSpace] = None,
) -> Dict:
    return VLADataAdapter.normalize_file(
        input_path=input_path,
        output_path=output_path,
        image_root=image_root,
        action_space=action_space,
    ).to_dict()


def register_vla_action_tokens(
    tokenizer: Any,
    action_space: Optional[VLAActionSpace] = None,
    include_image_token: bool = True,
    image_token: str = "<image>",
    resize_model: Optional[Any] = None,
) -> Dict:
    return VLAActionTokenizer(action_space).register_with_tokenizer(
        tokenizer=tokenizer,
        include_image_token=include_image_token,
        image_token=image_token,
        resize_model=resize_model,
    )
