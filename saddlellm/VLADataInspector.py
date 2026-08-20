"""Fast local preflight for vision-language-action training data."""
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .VLA import VLAAction, VLAActionSpace, VLADataAdapter, VLAActionTokenizer


@dataclass
class VLADataInspection:
    path: str
    image_root: Optional[str] = None
    exists: bool = False
    local_file: bool = False
    sampled_records: int = 0
    expanded_records: int = 0
    usable_records: int = 0
    detected_schemas: Dict[str, int] = field(default_factory=dict)
    missing_instruction: int = 0
    missing_action: int = 0
    missing_images: int = 0
    missing_local_images: int = 0
    action_dim_mismatches: int = 0
    out_of_range_actions: int = 0
    episode_step_warnings: int = 0
    avg_images_per_record: float = 0.0
    avg_action_values: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def usable_ratio(self) -> float:
        return self.usable_records / self.expanded_records if self.expanded_records else 0.0

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


class VLADataInspector:
    """Inspect robot trajectory rows before model or image loading starts."""

    @classmethod
    def inspect_file(
        cls,
        path: str,
        image_root: Optional[str] = None,
        action_space: Optional[VLAActionSpace] = None,
        max_records: int = 256,
    ) -> VLADataInspection:
        report = VLADataInspection(path=path, image_root=image_root)
        action_space = action_space or VLAActionSpace()
        tokenizer = VLAActionTokenizer(action_space)
        if not path:
            report.errors.append("No VLA data path provided.")
            report.recommendations.append("Set vla.data_path to a JSONL/JSON/CSV robot trajectory file.")
            return report

        report.exists = os.path.exists(path)
        report.local_file = os.path.isfile(path)
        if not report.exists:
            report.warnings.append("Path does not exist locally; assuming it may be a remote robot dataset id.")
            report.recommendations.append("Export a small local VLA shard and run inspect-vla before full training.")
            return report
        if not report.local_file:
            report.warnings.append("Path exists but is not a file; skipping strict VLA inspection.")
            report.recommendations.append("Point vla.data_path at a JSONL/JSON/CSV file for strict validation.")
            return report

        resolved_image_root = image_root or os.path.dirname(os.path.abspath(path))
        image_count = 0
        action_value_count = 0
        episodes: Dict[str, List[int]] = {}

        try:
            for record in cls._sample_records(VLADataAdapter.load_records(path), max_records=max_records):
                report.sampled_records += 1
                for item in VLADataAdapter._expand_trajectory_record(record):
                    if report.expanded_records >= max_records:
                        break
                    report.expanded_records += 1
                    schema = VLADataAdapter.detect_schema(item)
                    report.detected_schemas[schema] = report.detected_schemas.get(schema, 0) + 1

                    instruction = VLADataAdapter._first(
                        item,
                        ["instruction", "prompt", "task", "goal", "language_instruction", "query"],
                    )
                    action = VLADataAdapter._extract_action(item)
                    images = VLADataAdapter._extract_images(item, image_root=resolved_image_root)

                    if not instruction:
                        report.missing_instruction += 1
                    if action is None:
                        report.missing_action += 1
                    if not images:
                        report.missing_images += 1

                    if action is not None:
                        dim_ok, value_count, out_of_range = cls._inspect_action(action, action_space)
                        action_value_count += value_count
                        if not dim_ok:
                            report.action_dim_mismatches += 1
                        if out_of_range:
                            report.out_of_range_actions += 1

                    missing_images = cls._missing_local_images(images)
                    report.missing_local_images += len(missing_images)
                    image_count += len(images)
                    cls._record_episode_step(item, episodes)

                    sample = VLADataAdapter.normalize_record(
                        item,
                        image_root=resolved_image_root,
                        action_tokenizer=tokenizer,
                    )
                    if sample is not None:
                        report.usable_records += 1
        except Exception as exc:
            report.errors.append(f"VLA inspection failed: {exc}")
            report.recommendations.append("Verify file extension, UTF-8 encoding, JSON/CSV structure, and action/image fields.")

        if report.expanded_records:
            report.avg_images_per_record = round(image_count / report.expanded_records, 2)
            report.avg_action_values = round(action_value_count / report.expanded_records, 2)
            report.episode_step_warnings = cls._count_step_warnings(episodes)

        cls._finalize(report)
        return report

    @staticmethod
    def _sample_records(records: Iterable[Dict], max_records: int) -> Iterable[Dict]:
        for idx, record in enumerate(records):
            if idx >= max_records:
                return
            if isinstance(record, dict):
                yield record

    @staticmethod
    def _inspect_action(action: VLAAction, action_space: VLAActionSpace) -> Tuple[bool, int, bool]:
        if action.text:
            return True, 0, False
        values = list(action.values)
        if action.gripper is not None and action_space.include_gripper:
            values.append(float(action.gripper))
        expected_lengths = {action_space.action_dim}
        if action_space.include_gripper:
            expected_lengths.add(action_space.action_dim + 1)
        dim_ok = len(values) in expected_lengths
        out_of_range = any(value < action_space.min_value or value > action_space.max_value for value in values)
        return dim_ok, len(values), out_of_range

    @staticmethod
    def _missing_local_images(images) -> List[str]:
        missing = []
        for image in images:
            path = getattr(image, "path", "") or ""
            if not path or path.startswith(("http://", "https://", "s3://", "gs://", "data:")):
                continue
            if not os.path.exists(path):
                missing.append(path)
        return missing

    @staticmethod
    def _record_episode_step(record: Dict, episodes: Dict[str, List[int]]) -> None:
        episode = record.get("episode_id", record.get("episode", record.get("trajectory_id")))
        step = record.get("step_id", record.get("step", record.get("timestep")))
        if episode in (None, "") or step in (None, ""):
            return
        try:
            episodes.setdefault(str(episode), []).append(int(step))
        except Exception:
            return

    @staticmethod
    def _count_step_warnings(episodes: Dict[str, List[int]]) -> int:
        warnings = 0
        for steps in episodes.values():
            if len(steps) < 2:
                continue
            ordered = sorted(set(steps))
            expected = list(range(ordered[0], ordered[-1] + 1))
            if ordered != expected:
                warnings += 1
        return warnings

    @staticmethod
    def _finalize(report: VLADataInspection) -> None:
        if report.sampled_records == 0 and not report.errors:
            report.errors.append("No records were sampled from the VLA data file.")
            report.recommendations.append("Check that the file is non-empty and uses JSONL/JSON/CSV format.")
        if report.expanded_records and report.usable_records == 0:
            report.errors.append("No sampled VLA records had instruction, action, and image fields.")
            report.recommendations.append("Expected rows with instruction/task/goal, action/actions, and image/images/image_path.")
        elif report.expanded_records and report.usable_ratio < 0.8:
            report.warnings.append(f"Only {report.usable_ratio:.0%} of expanded VLA records normalized successfully.")
            report.recommendations.append("Normalize or filter incomplete robot steps before training.")
        if report.action_dim_mismatches:
            report.errors.append(f"Detected {report.action_dim_mismatches} VLA rows with action dimension mismatches.")
            report.recommendations.append("Align action vectors with vla.action_space.action_dim and gripper settings.")
        if report.missing_local_images:
            report.errors.append(f"Detected {report.missing_local_images} missing local image files.")
            report.recommendations.append("Fix image_root or image paths before VLA SFT.")
        if report.out_of_range_actions:
            report.warnings.append(f"Detected {report.out_of_range_actions} rows with actions outside configured min/max range.")
            report.recommendations.append("Actions will be clipped during tokenization; verify action normalization.")
        if report.episode_step_warnings:
            report.warnings.append(f"Detected {report.episode_step_warnings} episodes with non-contiguous sampled step ids.")
            report.recommendations.append("Check whether missing robot steps are intentional filtering or data loss.")


def inspect_vla_data(
    path: str,
    image_root: Optional[str] = None,
    action_space: Optional[VLAActionSpace] = None,
    max_records: int = 256,
) -> Dict:
    return VLADataInspector.inspect_file(
        path=path,
        image_root=image_root,
        action_space=action_space,
        max_records=max_records,
    ).to_dict()
