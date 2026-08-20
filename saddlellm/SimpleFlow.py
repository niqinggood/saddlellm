"""Compile SaddleLLM's minimal ``flow`` YAML into orchestrator config.

The short form is intentionally small and opinionated.  It covers the stable
post-training path while preserving the canonical TrainingOrchestrator schema
as the execution contract.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List, Mapping


class SimpleFlowCompiler:
    """Compiler for configs such as ``flow: sft+dpo``."""

    SUPPORTED_STAGES = {"sft", "dpo", "kto", "eval", "export"}
    PLANNED_STAGES = {"grpo", "serve"}

    @classmethod
    def is_simple_flow(cls, raw: Mapping[str, Any]) -> bool:
        return "flow" in raw

    @classmethod
    def compile(cls, raw: Mapping[str, Any], base_dir: str | None = None) -> Dict[str, Any]:
        raw = copy.deepcopy(dict(raw))
        base_dir = base_dir or os.getcwd()
        flow_tokens = cls._parse_flow(raw.get("flow"), raw)
        cls._validate_flow(flow_tokens)

        model_raw = raw.get("model")
        if isinstance(model_raw, str):
            model_name = model_raw.strip()
            model_config: Dict[str, Any] = {}
        elif isinstance(model_raw, Mapping):
            model_config = dict(model_raw)
            model_name = str(
                model_config.get("name_or_path")
                or model_config.get("name")
                or model_config.get("path")
                or ""
            ).strip()
        else:
            model_config = {}
            model_name = ""
        if not model_name:
            raise ValueError("Simple flow requires `model: <name-or-path>`.")

        data = cls._mapping(raw.get("data"), "data")
        train = cls._mapping(raw.get("train"), "train")
        advanced = cls._mapping(raw.get("advanced"), "advanced")
        evaluation = cls._mapping(raw.get("evaluation"), "evaluation")
        export_settings = cls._mapping(raw.get("export"), "export")

        output = str(raw.get("output") or f"outputs/{raw.get('name', 'posttrain-run')}")
        max_length = int(
            advanced.get("max_sequence_length", train.get("max_sequence_length", 2048))
        )
        max_steps = int(train.get("max_steps", -1))
        batch_size = int(train.get("batch_size", 1))
        grad_accum = int(advanced.get("gradient_accumulation_steps", train.get("gradient_accumulation_steps", 4)))
        qlora = bool(train.get("qlora", True))
        lora = bool(train.get("lora", qlora)) or qlora
        trust_remote_code = bool(advanced.get("trust_remote_code", False))
        local_files_only = bool(advanced.get("local_files_only", False))
        gradient_checkpointing = bool(advanced.get("gradient_checkpointing", True))
        resume_from_checkpoint = advanced.get(
            "resume_from_checkpoint",
            advanced.get("resume", False),
        )

        execution_stages: List[str] = []
        if "sft" in flow_tokens:
            execution_stages.append("sft")
        preference_method = next((stage for stage in ("dpo", "kto") if stage in flow_tokens), None)
        if preference_method:
            execution_stages.append("preference")

        evaluation_enabled = bool(evaluation.get("enabled", True))
        evaluation_gate = cls._mapping(evaluation.get("gate"), "evaluation.gate")
        gate_enabled = bool(evaluation_gate.get("enabled", False))
        if (
            "eval" in flow_tokens
            or (execution_stages and evaluation_enabled)
            or ("export" in flow_tokens and gate_enabled)
        ):
            execution_stages.append("eval")
        if "export" in flow_tokens:
            execution_stages.append("export")

        if not execution_stages:
            raise ValueError("Simple flow did not resolve to an executable stage.")

        compiled: Dict[str, Any] = {
            "project": str(raw.get("project", "saddlellm")),
            "experiment": str(raw.get("name", "posttrain-run")),
            "seed": int(raw.get("seed", 42)),
            "model": {
                "name_or_path": model_name,
                "tokenizer": model_config.get("tokenizer", model_name),
                "backend": model_config.get("backend", "hf"),
                "config": model_config.get("config", "qwen-tiny-160m"),
            },
            "stages": execution_stages,
            "data": {
                "sources": [],
                "max_seq_length": max_length,
                "pack_sequences": False,
            },
            "training": {
                "max_steps": max_steps,
                "per_device_batch_size": batch_size,
                "global_batch_size": int(train.get("global_batch_size", batch_size * grad_accum)),
                "learning_rate": float(train.get("learning_rate", 2e-4)),
                "warmup_steps": int(train.get("warmup_steps", 0)),
                "gradient_checkpointing": gradient_checkpointing,
                "save_every_steps": int(train.get("save_steps", 200)),
                "eval_every_steps": int(train.get("eval_steps", 0)),
                "log_every_steps": int(train.get("logging_steps", 10)),
                "dry_run": bool(train.get("dry_run", False)),
                "preflight_only": bool(train.get("preflight_only", False)),
                "resume_from_checkpoint": resume_from_checkpoint,
            },
            "distributed": {
                "strategy": str(advanced.get("strategy", "single")),
                "num_gpus": int(advanced.get("num_gpus", 1)),
                "num_nodes": int(advanced.get("num_nodes", 1)),
                "ddp_backend": str(advanced.get("ddp_backend", "auto")),
                "gradient_accumulation_steps": grad_accum,
                "bf16": bool(advanced.get("bf16", True)),
                "fp16": bool(advanced.get("fp16", False)),
            },
            "logging": {
                "output_dir": output,
                "backend": str(advanced.get("logging_backend", "local")),
            },
            "eval": cls._compile_evaluation(evaluation, evaluation_enabled),
            "export": {
                "enabled": "export" in flow_tokens,
                "output_dir": str(export_settings.get("output_dir") or os.path.join(output, "release")),
                "format": str(export_settings.get("format", "hf")),
                "model_path": export_settings.get("model_path"),
                "merge_lora": bool(export_settings.get("merge_lora", True)),
                "safe_serialization": bool(export_settings.get("safe_serialization", True)),
                "require_gate": bool(export_settings.get("require_gate", gate_enabled)),
                "trust_remote_code": bool(export_settings.get("trust_remote_code", trust_remote_code)),
                "local_files_only": bool(export_settings.get("local_files_only", local_files_only)),
                "device": str(export_settings.get("device", "auto")),
                "dtype": str(export_settings.get("dtype", "auto")),
                "hash_weights": bool(export_settings.get("hash_weights", False)),
                "overwrite": bool(export_settings.get("overwrite", False)),
            },
            "simple_flow": {
                "source": str(raw.get("flow")),
                "resolved": flow_tokens,
                "base_dir": os.path.abspath(base_dir),
            },
        }

        if "sft" in flow_tokens:
            sft_path = cls._required_data_path(data, "sft")
            compiled["sft"] = {
                "enabled": True,
                "data_path": sft_path,
                "use_lora": lora,
                "use_qlora": qlora,
                "epochs": int(train.get("epochs", 1)),
                "learning_rate": float(train.get("learning_rate", 2e-4)),
                "per_device_batch_size": batch_size,
                "max_seq_length": max_length,
                "gradient_accumulation_steps": grad_accum,
                "warmup_steps": int(train.get("warmup_steps", 0)),
                "save_steps": int(train.get("save_steps", 200)),
                "eval_steps": int(train.get("eval_steps", 0)),
                "logging_steps": int(train.get("logging_steps", 10)),
                "validation_split": float(train.get("validation_split", 0.0)),
                "response_template": train.get("response_template"),
                "local_files_only": local_files_only,
                "trust_remote_code": trust_remote_code,
                "gradient_checkpointing": gradient_checkpointing,
                "optim": train.get("optim"),
            }

        if preference_method:
            pref = cls._mapping(raw.get(preference_method), preference_method)
            pref_path = cls._required_data_path(data, "preference")
            pref_batch = int(pref.get("batch_size", max(2, batch_size) if preference_method == "kto" else batch_size))
            compiled["preference"] = {
                "enabled": True,
                "method": preference_method,
                "data_path": pref_path,
                "beta": float(pref.get("beta", 0.1)),
                "learning_rate": float(pref.get("learning_rate", 5e-6)),
                "epochs": int(pref.get("epochs", train.get("epochs", 1))),
                "per_device_batch_size": pref_batch,
                "max_length": int(pref.get("max_length", max_length)),
                "max_prompt_length": int(pref.get("max_prompt_length", max(1, max_length // 2))),
                "use_lora": bool(pref.get("lora", lora)),
                "use_qlora": bool(pref.get("qlora", qlora)),
                "gradient_accumulation_steps": int(pref.get("gradient_accumulation_steps", grad_accum)),
                "warmup_steps": int(pref.get("warmup_steps", train.get("warmup_steps", 0))),
                "save_steps": int(pref.get("save_steps", train.get("save_steps", 200))),
                "logging_steps": int(pref.get("logging_steps", train.get("logging_steps", 10))),
                "validation_split": float(pref.get("validation_split", 0.0)),
                "eval_steps": int(pref.get("eval_steps", 0)),
                "local_files_only": bool(pref.get("local_files_only", local_files_only)),
                "trust_remote_code": bool(pref.get("trust_remote_code", trust_remote_code)),
                "gradient_checkpointing": bool(pref.get("gradient_checkpointing", gradient_checkpointing)),
                "report_to": str(pref.get("report_to", "none")),
            }
        return compiled

    @classmethod
    def _parse_flow(cls, flow: Any, raw: Mapping[str, Any]) -> List[str]:
        if not isinstance(flow, str) or not flow.strip():
            raise ValueError("`flow` must be a non-empty string such as `sft+dpo`.")
        normalized = flow.lower().replace(" ", "")
        if normalized == "full":
            alignment = str(raw.get("alignment", "dpo")).lower()
            normalized = f"sft+{alignment}+eval+export"
        tokens = [token for token in normalized.split("+") if token]
        if not tokens:
            raise ValueError("`flow` did not contain any stages.")
        return list(dict.fromkeys(tokens))

    @classmethod
    def _validate_flow(cls, tokens: Iterable[str]) -> None:
        tokens = list(tokens)
        planned = [token for token in tokens if token in cls.PLANNED_STAGES]
        if planned:
            raise NotImplementedError(
                "The following simple-flow stages are documented as planned but are not executable yet: "
                + ", ".join(planned)
            )
        unknown = [token for token in tokens if token not in cls.SUPPORTED_STAGES]
        if unknown:
            raise ValueError(f"Unknown simple-flow stages: {unknown}")
        if "dpo" in tokens and "kto" in tokens:
            raise ValueError("Choose one preference method per simple flow: dpo or kto.")

    @staticmethod
    def _mapping(value: Any, name: str) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"`{name}` must be a mapping/object.")
        return dict(value)

    @staticmethod
    def _required_data_path(data: Mapping[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Simple flow requires `data.{key}`.")
        return value

    @staticmethod
    def _compile_evaluation(evaluation: Mapping[str, Any], enabled: bool) -> Dict[str, Any]:
        suite = evaluation.get("suite", "standard")
        if isinstance(suite, str):
            tasks = ["perplexity"] if suite == "standard" else [suite]
        elif isinstance(suite, list):
            tasks = [str(task) for task in suite]
        else:
            raise ValueError("`evaluation.suite` must be a string or list.")
        gate = SimpleFlowCompiler._mapping(evaluation.get("gate"), "evaluation.gate")
        rules = gate.get("rules", {})
        if not isinstance(rules, Mapping):
            raise ValueError("`evaluation.gate.rules` must be a mapping/object.")
        return {
            "enabled": enabled,
            "tasks": tasks,
            "dataset": str(evaluation.get("dataset", "wikitext")),
            "dataset_config": str(evaluation.get("dataset_config", "wikitext-2-raw-v1")),
            "max_samples": int(evaluation.get("max_samples", 1000)),
            "fail_on_error": bool(evaluation.get("fail_on_error", True)),
            "gate": {
                "enabled": bool(gate.get("enabled", False)),
                "rules": copy.deepcopy(dict(rules)),
                "require_all": bool(gate.get("require_all", True)),
                "fail_on_rejection": bool(gate.get("fail_on_rejection", True)),
                "output_path": gate.get("output_path"),
            },
        }
