"""Create a small, runnable SaddleLLM post-training starter workspace."""
# ruff: noqa: N999
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

SUPPORTED_PREFERENCE_METHODS = ("dpo", "kto")
DEFAULT_QUICKSTART_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


_SFT_EXAMPLES = (
    {
        "instruction": "用一句话解释什么是机器学习。",
        "output": "机器学习是让计算机从数据中总结规律，并据此进行预测或决策的方法。",
    },
    {
        "instruction": "把“易用性很重要”翻译成英文。",
        "output": "Usability is very important.",
    },
)

_DPO_EXAMPLES = (
    {
        "prompt": "用户问：如何开始学习 Python？",
        "chosen": "先掌握变量、条件和循环，再通过一个小项目边做边学。",
        "rejected": "随便看看就行，不需要练习。",
    },
    {
        "prompt": "用户问：程序报错时应该怎么办？",
        "chosen": "先阅读错误信息和堆栈，再缩小问题范围并构造最小复现。",
        "rejected": "忽略错误并反复重启程序。",
    },
)

_KTO_EXAMPLES = (
    {
        "prompt": "请给出一个清晰的排错建议。",
        "completion": "先记录错误信息，再逐步排除最近的改动。",
        "label": True,
    },
    {
        "prompt": "请给出一个清晰的排错建议。",
        "completion": "不要看日志，直接删除所有文件。",
        "label": False,
    },
    {
        "prompt": "如何写出易维护的函数？",
        "completion": "让函数职责单一，并使用能表达意图的名称。",
        "label": True,
    },
    {
        "prompt": "如何写出易维护的函数？",
        "completion": "把所有逻辑都塞进一个函数。",
        "label": False,
    },
)


@dataclass(frozen=True)
class QuickstartResult:
    """Files and next commands produced by :func:`create_quickstart`."""

    workspace: str
    config: str
    stages: list[str]
    model: str
    data_files: dict[str, str]
    created_files: list[str]
    next_steps: list[str]
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def create_quickstart(
    output_dir: str = "./saddlellm-quickstart",
    *,
    stages: str | Iterable[str] = "sft",
    preference_method: str = "dpo",
    model: str = DEFAULT_QUICKSTART_MODEL,
    sft_data: str | None = None,
    preference_data: str | None = None,
    qlora: bool = False,
    overwrite: bool = False,
) -> QuickstartResult:
    """Create a safe starter project without downloading a model or training.

    Existing generated targets are never replaced unless ``overwrite`` is true.
    User-provided datasets are only referenced and are never modified.
    """
    stages = _normalize_stages(stages)
    preference_method = str(preference_method).strip().lower()
    if preference_method not in SUPPORTED_PREFERENCE_METHODS:
        choices = ", ".join(SUPPORTED_PREFERENCE_METHODS)
        raise ValueError(
            f"Unsupported preference method {preference_method!r}. Choose one of: {choices}."
        )
    if not str(model).strip():
        raise ValueError("model must be a non-empty local path or Hugging Face model id.")

    workspace = Path(output_dir).expanduser().resolve()
    if workspace.exists() and not workspace.is_dir():
        raise NotADirectoryError(f"Quickstart output is not a directory: {workspace}")

    supplied_sft = _existing_file(sft_data, "SFT data") if sft_data else None
    needs_preference = "preference" in stages
    supplied_preference = (
        _existing_file(preference_data, "preference data") if preference_data else None
    )
    if preference_data and not needs_preference:
        raise ValueError("--preference-data requires the preference stage.")

    config_path = workspace / "config.yaml"
    readme_path = workspace / "README.md"
    sft_path = supplied_sft or workspace / "data" / "sft.jsonl"
    preference_path: Path | None = None
    if needs_preference:
        filename = f"{preference_method}.jsonl"
        preference_path = supplied_preference or workspace / "data" / filename

    generated_targets = [config_path, readme_path]
    if supplied_sft is None:
        generated_targets.append(sft_path)
    if needs_preference and supplied_preference is None and preference_path is not None:
        generated_targets.append(preference_path)
    conflicts = [path for path in generated_targets if path.exists()]
    if conflicts and not overwrite:
        rendered = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"Quickstart files already exist: {rendered}. "
            "Choose another directory or pass --force to replace only these files."
        )

    data_config = {"sft": _path_for_config(sft_path, workspace)}
    if preference_path is not None:
        data_config["preference"] = _path_for_config(preference_path, workspace)

    config = {
        "project": "saddlellm",
        "experiment": workspace.name or "quickstart",
        "model": {
            "name_or_path": str(model).strip(),
            "tokenizer": str(model).strip(),
            "backend": "hf",
        },
        "stages": stages,
        "training": {"max_steps": 10},
        "distributed": {
            "strategy": "single",
            "num_gpus": 1,
            "bf16": False,
            "fp16": False,
        },
        "logging": {"output_dir": "outputs/quickstart", "backend": "local"},
        "eval": {"enabled": False},
        "sft": {
            "enabled": "sft" in stages,
            "data_path": data_config["sft"],
            "use_lora": True,
            "use_qlora": bool(qlora),
            "epochs": 1,
            "learning_rate": 2e-4,
            "per_device_batch_size": 1,
            "max_seq_length": 256,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "warmup_steps": 0,
            "logging_steps": 1,
            "trust_remote_code": False,
        },
    }
    if needs_preference:
        config["preference"] = {
            "enabled": True,
            "method": preference_method,
            "data_path": data_config["preference"],
            "use_lora": True,
            "use_qlora": bool(qlora),
            "epochs": 1,
            "learning_rate": 5e-6,
            "per_device_batch_size": 2 if preference_method == "kto" else 1,
            "max_length": 256,
            "max_prompt_length": 128,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "warmup_steps": 0,
            "logging_steps": 1,
            "trust_remote_code": False,
        }

    workspace.mkdir(parents=True, exist_ok=True)
    if supplied_sft is None:
        _write_jsonl(sft_path, _SFT_EXAMPLES)
    if needs_preference and supplied_preference is None and preference_path is not None:
        examples = _DPO_EXAMPLES if preference_method == "dpo" else _KTO_EXAMPLES
        _write_jsonl(preference_path, examples)
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    inspect_commands = [f'saddle-llm inspect-data "{data_config["sft"]}" --task sft']
    if preference_path is not None:
        task = preference_method
        inspect_commands.append(
            f'saddle-llm inspect-data "{data_config["preference"]}" --task {task}'
        )
    relative_steps = [
        *inspect_commands,
        'saddle-llm validate-config "config.yaml"',
        'saddle-llm train "config.yaml" --dry-run',
        'saddle-llm train "config.yaml"',
    ]
    readme_path.write_text(
        _render_readme(stages, model, relative_steps),
        encoding="utf-8",
    )

    next_steps = [f'cd "{workspace}"', *relative_steps]
    data_files = {"sft": str(sft_path)}
    if preference_path is not None:
        data_files["preference"] = str(preference_path)
    return QuickstartResult(
        workspace=str(workspace),
        config=str(config_path),
        stages=list(stages),
        model=str(model).strip(),
        data_files=data_files,
        created_files=[str(path) for path in generated_targets],
        next_steps=next_steps,
        note=(
            "示例数据仅用于验证流程；真实训练前请替换为经过清洗、去重和划分的数据。"
        ),
    )


def _normalize_stages(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        stages = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        stages = [str(item).strip().lower() for item in value if str(item).strip()]
    if stages not in (["sft"], ["sft", "preference"]):
        raise ValueError("Quickstart stages must be `sft` or `sft,preference`.")
    return stages


def _existing_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _path_for_config(path: Path, workspace: Path) -> str:
    try:
        return Path(os.path.relpath(path, workspace)).as_posix()
    except ValueError:
        return str(path)


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    path.write_text(text, encoding="utf-8")


def _render_readme(stages: list[str], model: str, steps: list[str]) -> str:
    commands = "\n".join(steps)
    return f"""# SaddleLLM 快速开始

这个目录由 `saddle-llm quickstart` 创建。

- 阶段：`{", ".join(stages)}`
- 基础模型：`{model}`
- 默认只运行 10 步，使用 LoRA，不启用 QLoRA 和自动评测。

先在当前目录依次执行：

```powershell
{commands}
```

最后一条命令会下载模型并开始训练。示例数据只用于确认配置可运行；真实训练前请替换数据，并根据硬件调整 `config.yaml`。
"""
