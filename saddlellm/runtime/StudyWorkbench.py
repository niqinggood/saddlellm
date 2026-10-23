"""Read-only adapters for persisted public-policy experiment evidence.

This module does not import experiment projects, run commands, load a model,
or choose runs by modification time. Artifact IDs are an explicit allowlist.
"""

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_METHODS = {
    "base": "原始模型",
    "sft": "SFT",
    "dpo": "DPO",
    "rl": "GRPO / RLVR",
    "sft_rl": "SFT + GRPO",
}
_SECRET_KEYS = {
    "password",
    "passwd",
    "token",
    "access_token",
    "api_key",
    "secret",
    "authorization",
    "credentials",
    "environment",
    "environment_overrides",
    "env",
}
_ARTIFACTS = {
    "beginner-word": (
        "analysis/SaddleLLM_Qwen3_公共政策后训练实验报告_入门版.docx",
        "公共政策实验报告（入门版）",
        "report",
    ),
    "technical-word": (
        "analysis/SaddleLLM_Qwen3_公共政策后训练实验报告.docx",
        "公共政策实验报告（技术版）",
        "report",
    ),
    "report-html": ("analysis/REPORT.html", "实验分析 HTML", "report"),
    "reproduction-zip": (
        "analysis/SaddleLLM_公共政策实验复现工具.zip",
        "实验复现工具包",
        "reproduction",
    ),
    "reproduction-guide": (
        "analysis/word_rewrite/REPRODUCE.md",
        "复现步骤",
        "reproduction",
    ),
    "comparison-csv": ("comparison.csv", "实验对比 CSV", "analysis"),
    "comparison-xlsx": ("comparison.xlsx", "实验对比 Excel", "analysis"),
    "comparison-json": ("comparison.json", "原始指标 JSON", "analysis"),
    "verification-json": (
        "analysis/verification.json",
        "保存证据核验结果",
        "verification",
    ),
}


def _public(value: Any) -> Any:
    """Exclude credentials and process environments, including nested values."""
    if isinstance(value, dict):
        return {
            str(k): _public(v)
            for k, v in value.items()
            if str(k).casefold() not in _SECRET_KEYS
            and not re.search(
                r"(?:^|_)(?:password|passwd|secret|token|api_key)$",
                str(k),
                re.IGNORECASE,
            )
        }
    if isinstance(value, list):
        return [_public(v) for v in value]
    return value


def _argv(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    hide_next = False
    for part in value:
        text = str(part)
        if hide_next:
            result.append("[REDACTED]")
            hide_next = False
        elif re.match(
            r"^--?(password|passwd|token|api[-_]key|secret)$", text, re.IGNORECASE
        ):
            result.append(text)
            hide_next = True
        elif re.search(
            r"(?:password|passwd|token|api[-_]key|secret)=", text, re.IGNORECASE
        ):
            result.append(text.split("=", 1)[0] + "=[REDACTED]")
        else:
            result.append(text)
    return result


def _display(argv: list[str]) -> str:
    return (
        "& " + " ".join("'" + part.replace("'", "''") + "'" for part in argv)
        if argv
        else ""
    )


def _case_key(problem: dict[str, Any]) -> str | None:
    explicit = str(problem.get("question_id", "")).strip().upper()
    match = (
        re.search(r"(\d+)", explicit)
        if explicit
        else re.search(
            r"\[\s*Question\s+(\d+)\s*\]",
            str(problem.get("question", "")),
            re.IGNORECASE,
        )
    )
    question = f"Q{int(match.group(1))}" if match else explicit
    if not question:
        return None
    organization = re.sub(
        r"\s+", " ", str(problem.get("organization", "")).strip()
    ).casefold()
    return f"{question}::{organization}"


class StudyWorkbench:
    """Present recorded studies beneath one explicitly selected project root."""

    def __init__(self, project: str):
        self.project = Path(project).expanduser().resolve()
        if not self.project.is_dir():
            raise FileNotFoundError(f"Project directory does not exist: {self.project}")
        self._raw_cache: dict[
            tuple[Any, ...], tuple[list[dict[str, Any]], list[str]]
        ] = {}

    @staticmethod
    def _contained(root: Path, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError("Evidence path is outside the selected directory")
        return resolved

    def _study(self, study_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", study_id):
            raise KeyError("Unknown study")
        root = self._contained(
            self.project, self.project / "outputs" / "studies" / study_id
        )
        if not root.is_dir() or not (root / "experiment_summary.json").is_file():
            raise KeyError("Unknown study")
        return root

    def _json(self, root: Path, relative: str, default: Any = None) -> Any:
        path = self._contained(root, root / relative)
        if not path.is_file():
            return {} if default is None else default
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {} if default is None else default

    def _jsonl(self, root: Path, relative: str) -> list[dict[str, Any]]:
        path = self._contained(root, root / relative)
        if not path.is_file():
            return []
        records = []
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
        return records

    def _manifest(self, root: Path) -> dict[str, Any]:
        config = self._json(root, "resolved_config.json")
        seed = config.get("data", {}).get("split_seed", 42)
        if isinstance(seed, int):
            value = self._json(root, f"data_manifest_seed_{seed}.json")
            if value:
                return value
        # Only inspect the study's top level, never its source snapshots.
        paths = sorted(root.glob("data_manifest_seed_*.json"))
        return self._json(root, paths[0].name) if len(paths) == 1 else {}

    def _run_ids(self, root: Path) -> list[str]:
        summary = self._json(root, "experiment_summary.json")
        candidates = summary.get("run_ids", summary.get("completed_run_ids", []))
        return list(
            dict.fromkeys(
                name
                for name in candidates
                if isinstance(name, str)
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            )
        )

    def _baseline_id(self, root: Path, seed: int | None = None) -> str | None:
        for run_id in self._run_ids(root):
            result = self._json(root, f"runs/{run_id}/run_result.json")
            if result.get("method") == "base" and (
                seed is None or result.get("seed") == seed
            ):
                return run_id
        return None

    def list_studies(self) -> dict[str, Any]:
        parent = self._contained(self.project, self.project / "outputs" / "studies")
        studies = []
        if parent.is_dir():
            for child in sorted(parent.iterdir(), key=lambda item: item.name):
                if not child.is_dir():
                    continue
                try:
                    root = self._study(child.name)
                except (KeyError, ValueError):
                    continue
                summary = self._json(root, "experiment_summary.json")
                studies.append(
                    {
                        "id": child.name,
                        "name": self._study_name(child.name),
                        "status": summary.get("status", "unknown"),
                        "run_count": len(self._run_ids(root)),
                        "finished_at": summary.get("finished_at"),
                        "read_only": True,
                    }
                )
        return {"studies": studies, "project": self.project.name, "read_only": True}

    @staticmethod
    def _study_name(study_id: str) -> str:
        if study_id == "qwen3_06b_direct_seed42":
            return "Qwen3-0.6B 公共政策后训练实验"
        if "preflight" in study_id:
            return "Qwen3-0.6B 小样本流程预检"
        return study_id

    def _commands(self, root: Path) -> list[dict[str, Any]]:
        evidence = self._json(root, "analysis/word_rewrite/command_evidence.json")
        commands = []
        for step in evidence.get("steps", []):
            argv = _argv(step.get("command"))
            level = str(step.get("command_evidence_level", ""))
            kind = (
                "actual" if "实际" in level and "未保存" not in level else "equivalent"
            )
            if "原始准备argv未记录" in level:
                kind = "recorded-step"
            commands.append(
                {
                    "id": f"step-{step.get('step', len(commands) + 1)}",
                    "title": step.get("title", ""),
                    "command": argv,
                    "display": _display(argv),
                    "kind": kind,
                    "evidence_level": level,
                    "result": _public(step.get("result", {})),
                    "explanation": step.get("plain_language", ""),
                    "evidence": [
                        str(item).replace(str(root), ".")
                        for item in step.get("evidence", [])
                    ],
                    "anomalies": _public(step.get("anomalies", [])),
                }
            )
        if not commands:
            argv = _argv(self._json(root, "launch_command.json", []))
            if argv:
                commands.append(
                    {
                        "id": "launch",
                        "title": "保存的实验启动命令",
                        "command": argv,
                        "display": _display(argv),
                        "kind": "actual",
                        "evidence_level": "launch_command.json 保存的原始 argv",
                        "result": _public(self._json(root, "run_exit.json")),
                        "explanation": "只展示历史命令，网页不会执行。",
                        "evidence": ["launch_command.json"],
                    }
                )
        return commands

    def _checks(self, root: Path) -> list[dict[str, Any]]:
        checks = []
        doctor = self._json(root, "doctor.json")
        if doctor:
            checks.append(
                {
                    "id": "project-doctor",
                    "name": "正式训练前项目 doctor",
                    "status": doctor.get("status", "unknown"),
                    "timing": "正式 all 命令内部、训练开始前",
                    "scope": "项目依赖、GRPO API、CUDA 与数据路径；不代表模型质量或数据全量审计",
                    "checks": _public(doctor.get("checks", [])),
                    "errors": doctor.get("errors", []),
                    "warnings": doctor.get("warnings", []),
                    "summary": f"保存了 {len(doctor.get('checks', []))} 项检查",
                }
            )
        doctor = self._json(root, "analysis/word_rewrite/top_level_doctor_result.json")
        command = self._json(
            root, "analysis/word_rewrite/top_level_doctor_command.json"
        )
        if doctor:
            rows = [
                {
                    "name": name,
                    "value": doctor.get("packages", {}).get(name),
                    "ok": result == "ok",
                }
                for name, result in doctor.get("import_checks", {}).items()
            ]
            rows.append(
                {
                    "name": "device",
                    "value": {
                        k: doctor.get(k)
                        for k in (
                            "gpu_name",
                            "gpu_total_memory_gb",
                            "gpu_memory_gb",
                            "ram_available_gb",
                            "disk_free_gb",
                        )
                    },
                    "ok": doctor.get("ready") is True,
                }
            )
            checks.append(
                {
                    "id": "framework-doctor",
                    "name": "报告修订时框架 doctor",
                    "status": "ok" if doctor.get("ready") else "not_ready",
                    "timing": command.get(
                        "timing_label", "报告修订时追加，非训练启动时检查"
                    ),
                    "recorded_at": command.get("started_utc"),
                    "scope": doctor.get(
                        "ready_scope", "dependency_device_preflight_not_model_capacity"
                    ),
                    "checks": rows,
                    "errors": doctor.get("issues", []),
                    "warnings": doctor.get("warnings", []),
                    "summary": "历史依赖与设备快照；可用显存不是预留容量",
                    "exit_code": command.get("exit_code"),
                }
            )
        dry = self._json(root, "dry_run_manifest.json")
        if dry:
            checks.append(
                {
                    "id": "dry-run",
                    "name": "训练前数据与奖励 dry-run",
                    "status": dry.get("status", "unknown"),
                    "timing": "正式 all 命令内部、训练开始前",
                    "scope": "保存的样本选择与奖励检查",
                    "checks": [],
                    "errors": [],
                    "warnings": [],
                    "summary": "只显示已保存的 dry-run 证据",
                    "selected_counts": dry.get("selected_counts", {}),
                    "reward_checks": _public(dry.get("reward_checks", {})),
                }
            )
        return checks

    def _datasets(self, root: Path) -> list[dict[str, Any]]:
        manifest = self._manifest(root)
        audit = manifest.get("audit", {})
        profile = self._json(root, "analysis/data_profile.json")
        readable = self._json(root, "analysis/word_rewrite/data_evidence.json")
        selected = manifest.get("selected", {})
        datasets = []
        for task, name, description, storage in [
            (
                "A",
                "AASB_Climate_Dataset",
                "根据机构咨询回应判断七种立场，单标签任务。A 原始数据没有回应正文；本轮按问题号和机构联接 B 的 response。",
                "Parquet",
            ),
            (
                "B",
                "aasb_topic_modelling",
                "根据机构咨询回应选择当前问题的主题，可有多个标签。A–I 的含义由该问题的 candidate_themes 定义。",
                "JSONL",
            ),
        ]:
            schema = readable.get("schemas", {}).get(task, {})
            official = {}
            for file, count in schema.get("file_record_counts", {}).items():
                split = (
                    "test" if "test" in Path(file.replace("\\", "/")).name else "train"
                )
                official[split] = official.get(split, 0) + count
            files = [
                {
                    "name": Path(str(item.get("path", "")).replace("\\", "/")).name,
                    "split": item.get("split"),
                    "bytes": item.get("bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in audit.get("dataset_sources", {})
                .get(task, {})
                .get("files", [])
            ]
            suffix = task.lower()
            datasets.append(
                {
                    "id": task,
                    "name": name,
                    "task": "立场分类" if task == "A" else "主题多标签分类",
                    "format": schema.get("storage", storage),
                    "raw_records": audit.get(f"dataset_{suffix}_rows"),
                    "official_splits": official,
                    "derived_splits": audit.get(f"derived_split_counts_{suffix}", {}),
                    "selected": {
                        "train": len(selected.get(f"train_{suffix}", [])),
                        "test": len(selected.get(f"benchmark_{suffix}", [])),
                    },
                    "label_counts": audit.get(f"{suffix}_label_counts", {}),
                    "description": description,
                    "field_usage": readable.get("field_usage_zh", {}).get(task, {}),
                    "labels": readable.get("a_label_gloss_zh", {})
                    if task == "A"
                    else readable.get(
                        "b_label_gloss_zh", "主题字母以每道题的选项定义为准"
                    ),
                    "schema": schema.get("outer_field_types", {}),
                    "problem_schema": schema.get("parsed_problem_field_types", {}),
                    "files": files,
                    "profile": _public(profile.get("tasks", {}).get(task, {})),
                    "selected_profile": {
                        key: _public(profile.get("selected", {}).get(key, {}))
                        for key in (f"train_{suffix}", f"benchmark_{suffix}")
                    },
                    "split_explanation": "train/test 页面展示本次实际选中的派生训练/测试样本；original_train/original_test 展示文件原始划分。联合案例划分与原文件划分不同。",
                }
            )
        return datasets

    def _artifacts(self, root: Path) -> list[dict[str, Any]]:
        items = []
        for artifact_id, (relative, name, kind) in _ARTIFACTS.items():
            path = self._contained(root, root / relative)
            if path.is_file():
                items.append(
                    {
                        "id": artifact_id,
                        "name": name,
                        "format": path.suffix.lstrip("."),
                        "kind": kind,
                        "bytes": path.stat().st_size,
                    }
                )
        return items

    def study_detail(self, study_id: str) -> dict[str, Any]:
        root = self._study(study_id)
        config = self._json(root, "resolved_config.json")
        summary = self._json(root, "experiment_summary.json")
        analysis = self._json(root, "analysis/analysis.json")
        model = config.get("model", {})
        runs = [self.run_detail(study_id, name) for name in self._run_ids(root)]
        seeds = list(
            dict.fromkeys(
                run.get("seed") for run in runs if run.get("seed") is not None
            )
        )
        counts = {
            task: sorted(
                {
                    run["metrics"][task]["post"]["num_examples"]
                    for run in runs
                    if isinstance(run["metrics"][task]["post"].get("num_examples"), int)
                }
            )
            for task in ("A", "B")
        }
        k = config.get("evaluation", {}).get("k")
        scope = f"保存的训练种子：{seeds or '未记录'}；评测案例数 A={counts['A'] or '未记录'}、B={counts['B'] or '未记录'}。"
        if len(seeds) == 1:
            scope += "单一训练种子不能证明多种子稳定性。"
        caveats = [
            "这里回放已完成实验的真实保存记录，当前页面不会启动训练。",
            scope,
            "A/B 共享咨询案例，跨任务迁移不能视作独立跨领域验证。",
            "训练方法的 loss 含义不同，不能跨方法以 loss 大小排名。",
            f"逐例对比使用首个回答；配置 k={k if k is not None else '未记录'}，多次采样的 pass@k 不能当作单次准确率。",
            "没有保存的实时 GPU 曲线或逐步验证 loss 会留空，不从最终指标生成曲线。",
            "SFT+GRPO 复用同来源的 SFT adapter，其 SFT 历史是共享阶段，耗时需区分共享训练成本。",
        ]
        if "preflight" in study_id or any(
            n < 30 for values in counts.values() for n in values
        ):
            caveats.insert(
                1, "这是小样本流程预检，用于验证训练能否跑通；不用于训练方法效果排名。"
            )
        return {
            "id": study_id,
            "name": self._study_name(study_id),
            "project": self.project.name,
            "status": summary.get("status", "unknown"),
            "finished_at": summary.get("finished_at"),
            "read_only": True,
            "model": {
                "name": "Qwen3-0.6B"
                if "Qwen3-0.6B" in str(model.get("name", ""))
                else model.get("name"),
                "revision": model.get("revision"),
                "lora_rank": model.get("lora_rank"),
                "lora_alpha": model.get("lora_alpha"),
                "lora_dropout": model.get("lora_dropout"),
            },
            "config": _public(
                {
                    key: config[key]
                    for key in (
                        "model",
                        "data",
                        "evaluation",
                        "rl",
                        "reward",
                        "sft",
                        "preference",
                        "matrix",
                    )
                    if key in config
                }
            ),
            "datasets": self._datasets(root),
            "runs": runs,
            "checks": self._checks(root),
            "commands": self._commands(root),
            "issues": _public(self._json(root, "issue_log.json", [])),
            "artifacts": self._artifacts(root),
            "caveats": caveats,
            "data_context": self._json(
                root, "analysis/word_rewrite/data_evidence.json"
            ).get("data_context_zh", ""),
            "data_audit": _public(self._manifest(root).get("audit", {})),
            "analysis": {
                "selection": _public(analysis.get("selection", {})),
                "regressions": _public(analysis.get("regressions", [])),
                "quality_regressions": _public(analysis.get("quality_regressions", [])),
            },
            "sources": {
                "summary": "experiment_summary.json",
                "config": "resolved_config.json",
                "comparison": "comparison.json",
                "data_manifest": "data_manifest_seed_<seed>.json",
                "issues": "issue_log.json",
            },
        }

    def run_detail(self, study_id: str, run_id: str) -> dict[str, Any]:
        root = self._study(study_id)
        if run_id not in self._run_ids(root):
            raise KeyError("Unknown run")
        result = self._json(root, f"runs/{run_id}/run_result.json")
        method = str(result.get("method", "unknown"))
        source = result.get("source", "-")
        history = []
        training = result.get("training") or {}
        for stage, value in training.items():
            if not isinstance(value, dict):
                continue
            points = value.get("steps", value.get("log_history", []))
            evidence = f"runs/{run_id}/run_result.json#training.{stage}"
            if stage == "dpo" and not points:
                state = self._json(
                    root, f"runs/{run_id}/dpo/adapter/trainer_state.json"
                )
                points = state.get("log_history", [])
                evidence = f"runs/{run_id}/dpo/adapter/trainer_state.json"
            clean_points = [
                {
                    key: number
                    for key, number in point.items()
                    if isinstance(number, (int, float, bool)) or number is None
                }
                for point in points
                if isinstance(point, dict)
            ]
            history.append(
                {
                    "stage": stage,
                    "name": _METHODS.get(stage, stage),
                    "points": clean_points,
                    "source": evidence,
                    "shared": method == "sft_rl" and stage == "sft",
                    "validation_history": [],
                    "diagnostics": _public(value.get("diagnostics", {})),
                    "summary": _public(value.get("train_metrics", {})),
                    "explanation": "保存的训练记录；未保存逐步验证曲线。",
                }
            )
        analysis = self._json(root, "analysis/analysis.json")
        metrics = {}
        for task, key, primary, label in [
            ("A", "benchmark_a", "accuracy", "首个回答准确率"),
            ("B", "benchmark_b", "exact_match_at_1", "首个回答标签集合完全匹配率"),
        ]:
            row = next(
                (
                    item
                    for item in analysis.get("rows", [])
                    if item.get("run_id") == run_id and item.get("target") == task
                ),
                {},
            )
            metrics[task] = {
                "pre": result.get("pre", {}).get(key, {}),
                "post": result.get("post", {}).get(key, {}),
                "delta": result.get("delta", {}).get(key, {}),
                "primary_metric": primary,
                "primary_label": label,
                "unit": "ratio",
                "paired": row.get("paired", {}),
                "organization_paired": row.get("organization_paired", {}),
                "breakdown": row.get("breakdown", {}),
            }
        run_issues = [
            item
            for item in self._json(root, "issue_log.json", [])
            if isinstance(item, dict)
            and (item.get("run_id") == run_id or run_id in item.get("run_ids", []))
        ]
        return {
            "id": run_id,
            "name": _METHODS.get(method, method)
            + (f" · 来源 {source}" if source not in (None, "-", "Base") else ""),
            "method": method,
            "source": source,
            "status": result.get("status", "missing_evidence"),
            "seed": result.get("seed"),
            "duration_s": result.get("duration_s"),
            "peak_cuda_allocated_gib": result.get("peak_cuda_allocated_gib"),
            "finished_at": result.get("finished_at"),
            "config_fingerprint": result.get("config_fingerprint"),
            "metrics": _public(metrics),
            "history": history,
            "commands": [
                {
                    "id": "study-launch",
                    "title": "该运行由整组实验 all 命令启动",
                    "kind": "recorded-step",
                    "command": _argv(self._json(root, "launch_command.json", [])),
                    "display": _display(
                        _argv(self._json(root, "launch_command.json", []))
                    ),
                    "explanation": "没有单独 run argv；保留整组实验的实际启动命令。",
                }
            ],
            "issues": _public(run_issues),
            "training_summary": {
                stage: {
                    key: _public(value.get(key))
                    for key in (
                        "status",
                        "global_step",
                        "num_pairs",
                        "num_source_examples",
                        "dataset",
                        "diagnostics",
                        "negative_source",
                        "engine",
                    )
                    if key in value
                }
                for stage, value in training.items()
                if isinstance(value, dict)
            },
            "source_file": f"runs/{run_id}/run_result.json",
        }

    def _raw_files(self, root: Path, source: str) -> list[tuple[Path, str, str | None]]:
        files = (
            self._manifest(root)
            .get("audit", {})
            .get("dataset_sources", {})
            .get(source, {})
            .get("files", [])
        )
        allowed = [
            self._contained(
                root, root / "source_snapshot" / self.project.name / "data"
            ),
            self._contained(self.project, self.project / "data"),
        ]
        result = []
        for item in files:
            raw = re.sub(r"[\\/]+", "/", str(item.get("path", "")))
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            # Relocate a frozen snapshot from another computer without trusting
            # the manifest's absolute location as an arbitrary file capability.
            if not path.is_file():
                marker = "/source_snapshot/"
                if marker in raw:
                    path = (
                        root / "source_snapshot" / raw.split(marker, 1)[1]
                    ).resolve()
            if (
                path.suffix.lower() not in {".parquet", ".jsonl"}
                or not any(path.is_relative_to(base) for base in allowed)
                or not path.is_file()
            ):
                continue
            result.append((path, str(item.get("split", "unknown")), item.get("sha256")))
        return result

    def _raw(self, root: Path, source: str) -> tuple[list[dict[str, Any]], list[str]]:
        files = self._raw_files(root, source)
        cache_key = (
            str(root),
            source,
            tuple(
                (str(path), path.stat().st_mtime_ns, path.stat().st_size, sha)
                for path, _, sha in files
            ),
        )
        if cache_key in self._raw_cache:
            return self._raw_cache[cache_key]
        records: list[dict[str, Any]] = []
        notes = []
        for path, split, expected_sha in files:
            if expected_sha:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_sha:
                    notes.append(
                        f"{path.name} 当前 SHA256 与训练清单不一致，未把该文件的原始记录混入历史实验。"
                    )
                    continue
            if path.suffix.lower() == ".parquet":
                try:
                    from pyarrow import parquet
                except ImportError:
                    notes.append(
                        "读取 A 原始 Parquet 需要安装可选依赖 pyarrow；仍可查看保存的模型提示与评测。"
                    )
                    continue
                values = parquet.read_table(
                    path, columns=["problem", "answer"]
                ).to_pylist()
            else:
                values = []
                with path.open(encoding="utf-8-sig") as stream:
                    for line in stream:
                        if line.strip():
                            values.append(json.loads(line))
            for index, raw in enumerate(values):
                problem = raw.get("problem", {})
                if isinstance(problem, str):
                    try:
                        problem = json.loads(problem)
                    except json.JSONDecodeError:
                        problem = {}
                if not isinstance(problem, dict):
                    problem = {}
                case = _case_key(problem)
                target = raw.get("answer", [])
                if isinstance(target, str):
                    target = [target]
                records.append(
                    {
                        "id": f"{source}:{case}"
                        if case
                        else f"{source}:{split}:{index}",
                        "raw": _public(raw),
                        "parsed_problem": _public(problem),
                        "target": target,
                        "case_key": case,
                        "original_split": split,
                        "source_file": path.name,
                        "row_index": index,
                        "source": source,
                    }
                )
        if not records and not notes:
            notes.append("没有可读取的已冻结原始数据文件；不会从模型结果反造原始记录。")
        self._raw_cache[cache_key] = (records, notes)
        return records, notes

    @staticmethod
    def _page(
        rows: list[dict[str, Any]], offset: int, limit: int, query: str
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError(
                "offset must be nonnegative; limit must be between 1 and 100"
            )
        if query:
            needle = query.casefold()
            rows = [
                row
                for row in rows
                if needle in json.dumps(row, ensure_ascii=False).casefold()
            ]
        return {
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "rows": rows[offset : offset + limit],
        }

    def data_samples(
        self,
        study_id: str,
        source: str = "A",
        split: str = "train",
        offset: int = 0,
        limit: int = 10,
        query: str = "",
    ) -> dict[str, Any]:
        root = self._study(study_id)
        if source not in {"A", "B"} or split not in {
            "train",
            "test",
            "validation",
            "original_train",
            "original_test",
        }:
            raise ValueError("Unknown source or split")
        raw, notes = self._raw(root, source)
        lookup = {record["id"]: record for record in raw}
        selected = self._manifest(root).get("selected", {})
        suffix = source.lower()
        prompts: dict[str, dict[str, Any]] = {}
        pair_source = None
        for run_id in self._run_ids(root):
            run = self._json(root, f"runs/{run_id}/run_result.json")
            if run.get("method") == "dpo" and run.get("source") == source:
                pair_source = f"runs/{run_id}/dpo/preference_pairs.jsonl"
                break
        pairs = self._jsonl(root, pair_source) if pair_source else []
        for pair in pairs:
            prompts[str(pair.get("example_id"))] = {
                "input": pair.get("prompt"),
                "chosen": pair.get("chosen"),
                "rejected": pair.get("rejected"),
                "input_source": pair_source,
                "notes": [
                    "这是保存的 DPO 完整 chat-template 提示；其 Direct user 内容也用于同来源的训练。",
                    "DPO rejected 为依据金标构造的合法错标签，不是模型采样的错误答案。",
                ],
            }
        baseline_id = self._baseline_id(root)
        for item in (
            self._json(root, f"runs/{baseline_id}/final_benchmark_{suffix}.json").get(
                "records", []
            )
            if baseline_id
            else []
        ):
            prompts[str(item.get("example_id"))] = {
                "input": item.get("user_prompt"),
                "input_source": f"runs/{baseline_id}/final_benchmark_{suffix}.json",
                "notes": [
                    "这是评测时保存的真实 user_prompt，不包含 system/chat-template。"
                ],
            }
        if split.startswith("original_"):
            chosen = [
                dict(record)
                for record in raw
                if record["original_split"] == split.removeprefix("original_")
            ]
        elif split == "validation":
            chosen = []
            notes = [
                *notes,
                "清单保存了派生 validation 的总数，没有保存本次选用的验证记录；本实验未使用独立验证集挑选训练 checkpoint。",
            ]
        else:
            selection_key = (
                f"train_{suffix}" if split == "train" else f"benchmark_{suffix}"
            )
            chosen = []
            for item in selected.get(selection_key, []):
                record_id = str(item.get("example_id", ""))
                row = dict(
                    lookup.get(
                        record_id,
                        {
                            "id": record_id,
                            "raw": None,
                            "parsed_problem": {},
                            "source": source,
                            "original_split": item.get("original_split"),
                            "case_key": item.get("case_key"),
                            "target": item.get("answer", []),
                        },
                    )
                )
                if row["raw"] is None:
                    row["notes"] = [
                        "原始记录不可用；标签与 ID 来自保存的样本选择清单。"
                    ]
                chosen.append(row)
        for row in chosen:
            row["split"] = split
            prompt = prompts.get(row["id"], {})
            row.update(prompt)
            row.setdefault("input", None)
            row.setdefault("notes", [])
            if not prompt:
                row["notes"] = [
                    *row["notes"],
                    "该记录没有保存的实际模型提示，input 留空。",
                ]
        result = self._page(chosen, offset, limit, query)
        result.update(
            {
                "source": source,
                "split": split,
                "notes": notes,
                "explanation": "train/test 为本次实际选中的派生样本；original_train/original_test 为原文件划分。原始 problem 是内嵌 JSON 字符串，parsed_problem 展示解析后的字段。",
            }
        )
        return result

    @staticmethod
    def _correctness(record: dict[str, Any]) -> bool | None:
        gold = record.get("expected")
        draws = record.get("predictions")
        if (
            not isinstance(gold, list)
            or not gold
            or not all(isinstance(label, str) for label in gold)
            or not isinstance(draws, list)
            or not draws
        ):
            return None
        prediction = draws[0]
        if prediction is None:
            # A saved failed parse is a wrong answer; an absent draw is unknown.
            return False
        if not isinstance(prediction, list) or not all(
            isinstance(label, str) for label in prediction
        ):
            return None
        return set(prediction) == set(gold)

    def predictions(
        self,
        study_id: str,
        run_id: str,
        task: str = "A",
        outcome: str = "all",
        offset: int = 0,
        limit: int = 10,
        query: str = "",
    ) -> dict[str, Any]:
        root = self._study(study_id)
        if run_id not in self._run_ids(root):
            raise KeyError("Unknown run")
        if task not in {"A", "B"} or outcome not in {
            "all",
            "improved",
            "regressed",
            "both_correct",
            "both_wrong",
            "unpaired",
        }:
            raise ValueError("Unknown task or outcome")
        suffix = task.lower()
        metadata = self._json(root, f"runs/{run_id}/run_result.json")
        baseline_id = self._baseline_id(root, metadata.get("seed"))
        baseline_records = (
            self._json(root, f"runs/{baseline_id}/final_benchmark_{suffix}.json").get(
                "records", []
            )
            if baseline_id
            else []
        )
        baseline = {row.get("example_id"): row for row in baseline_records}
        current = self._json(root, f"runs/{run_id}/final_benchmark_{suffix}.json").get(
            "records", []
        )
        rows = []
        for record in current:
            previous = baseline.get(record.get("example_id"))
            expected = record.get("expected", [])
            draws = record.get("predictions", [])
            responses = record.get("responses", [])
            pred = draws[0] if draws else None
            correct = self._correctness(record)
            paired = (
                previous is not None
                and correct is not None
                and self._correctness(previous) is not None
                and bool(record.get("example_id"))
                and isinstance(record.get("user_prompt"), str)
                and bool(record.get("user_prompt"))
                and set(previous.get("expected", [])) == set(expected)
                and previous.get("user_prompt") == record.get("user_prompt")
            )
            before_draws = previous.get("predictions", []) if paired else []
            before_pred = before_draws[0] if before_draws else None
            before_correct = self._correctness(previous) if paired else None
            if before_correct is None:
                group = "unpaired"
            else:
                group = (
                    "both_correct"
                    if correct and before_correct
                    else "improved"
                    if correct
                    else "regressed"
                    if before_correct
                    else "both_wrong"
                )
            if outcome != "all" and outcome != group:
                continue
            previous_responses = previous.get("responses", []) if paired else []
            rows.append(
                {
                    "id": record.get("example_id"),
                    "case_key": record.get("case_key"),
                    "task": task,
                    "organization": record.get("organization"),
                    "question_id": record.get("question_id"),
                    "input": record.get("user_prompt"),
                    "target": expected,
                    "baseline_output": previous_responses[0]
                    if previous_responses
                    else None,
                    "output": responses[0] if responses else None,
                    "baseline_prediction": before_pred,
                    "prediction": pred,
                    "baseline_correct": before_correct,
                    "correct": correct,
                    "outcome": group,
                    "responses": responses,
                    "predictions": draws,
                    "format_valid": record.get("format_valid", []),
                    "response_truncated": record.get("response_truncated"),
                    "prompt_token_truncated": record.get("prompt_token_truncated"),
                    "source_file": f"runs/{run_id}/final_benchmark_{suffix}.json",
                    "baseline_source_file": f"runs/{baseline_id}/final_benchmark_{suffix}.json"
                    if paired
                    else None,
                }
            )
        result = self._page(rows, offset, limit, query)
        observed_k = sorted({len(record.get("predictions", [])) for record in current})
        result.update(
            {
                "task": task,
                "outcome": outcome,
                "run_id": run_id,
                "baseline_id": baseline_id,
                "draw_index": 0,
                "observed_k": observed_k,
                "explanation": f"使用保存的第一个回答（@1）按标签集合相等分组。保存的每题采样数：{observed_k or '未记录'}；标签正确与 format_valid 分别记录，pass@k 不是这里的正确率。基线只按同训练种子、相同 ID、金标与真实输入配对。",
            }
        )
        return result

    def artifact_path(self, study_id: str, artifact_id: str) -> Path:
        root = self._study(study_id)
        if artifact_id not in _ARTIFACTS:
            raise KeyError("Unknown artifact")
        path = self._contained(root, root / _ARTIFACTS[artifact_id][0])
        if not path.is_file():
            raise KeyError("Artifact not available")
        return path
