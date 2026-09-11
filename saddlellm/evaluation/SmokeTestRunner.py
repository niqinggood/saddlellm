"""End-to-end smoke tests for the local SaddleLLM training stack."""
import base64
import contextlib
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SmokeCommandResult:
    name: str
    command: List[str]
    returncode: int
    expected_returncode: int
    seconds: float
    passed: bool
    stdout_tail: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SmokeTestResult:
    ready: bool
    work_dir: str
    trained: bool
    total_seconds: float
    commands: List[SmokeCommandResult] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["commands"] = [item.to_dict() for item in self.commands]
        return data


class SmokeTestRunner:
    """Create tiny fixtures and run a complete local smoke suite."""

    def __init__(
        self,
        work_dir: str = "build/smoke_e2e_auto",
        run_training: bool = True,
        clean: bool = True,
        run_doctor: bool = True,
        timeout_seconds: int = 180,
        use_subprocess: bool = True,
    ):
        self.work_dir = Path(work_dir)
        self.run_training = run_training
        self.clean = clean
        self.run_doctor = run_doctor
        self.timeout_seconds = timeout_seconds
        self.use_subprocess = use_subprocess

    def run(self) -> SmokeTestResult:
        start = time.time()
        if self.clean and self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self._prepare_fixtures()

        commands: List[SmokeCommandResult] = []
        if self.run_doctor:
            commands.append(self._run_command("doctor", ["doctor"], expected_returncode=0))

        commands.extend([
            self._run_command("compileall", ["-m", "compileall", "saddle_llm"], module=False),
            self._run_command("inspect-sft", ["inspect-data", str(self._data_path("sft.jsonl")), "--task", "sft", "--max-records", "20"]),
            self._run_command("inspect-dpo", ["inspect-data", str(self._data_path("preference.jsonl")), "--task", "dpo", "--max-records", "20"]),
            self._run_command(
                "inspect-vla",
                [
                    "inspect-vla",
                    str(self._data_path("vla.jsonl")),
                    "--image-root",
                    str(self._image_dir()),
                    "--action-dim",
                    "7",
                    "--max-records",
                    "20",
                ],
            ),
            self._run_command(
                "inspect-vla-bad-dim",
                [
                    "inspect-vla",
                    str(self._data_path("vla.jsonl")),
                    "--image-root",
                    str(self._image_dir()),
                    "--action-dim",
                    "6",
                    "--no-gripper",
                    "--max-records",
                    "20",
                ],
                expected_returncode=1,
            ),
        ])

        for config_name in ["sft", "vla", "dpo", "orpo", "kto"]:
            commands.append(self._run_command(f"validate-{config_name}", ["validate-config", str(self._config_path(config_name))]))
        commands.append(
            self._run_command(
                "validate-kto-bad-batch",
                ["validate-config", str(self._config_path("kto_bad_batch"))],
                expected_returncode=1,
            )
        )

        if self.run_training:
            for config_name in ["sft", "vla", "dpo", "orpo", "kto"]:
                commands.append(self._run_command(f"train-{config_name}", ["train", str(self._config_path(config_name))]))
            commands.append(
                self._run_command(
                    "train-kto-bad-batch",
                    ["train", str(self._config_path("kto_bad_batch"))],
                    expected_returncode=1,
                )
            )
            commands.append(self._check_training_artifacts())

        issues = [item.name for item in commands if not item.passed]
        artifacts = {
            "tiny_model": str(self._model_dir()),
            "data_dir": str(self._data_dir()),
            "configs_dir": str(self.work_dir),
            "summary": str(self.work_dir / "smoke_summary.json"),
        }
        result = SmokeTestResult(
            ready=not issues,
            work_dir=str(self.work_dir),
            trained=self.run_training,
            total_seconds=round(time.time() - start, 3),
            commands=commands,
            artifacts=artifacts,
            issues=issues,
        )
        with open(self.work_dir / "smoke_summary.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        return result

    def _run_command(self, name: str, args: List[str], expected_returncode: int = 0, module: bool = True) -> SmokeCommandResult:
        command = [sys.executable]
        if module:
            command.extend(["-m", "saddle_llm.cli"])
        command.extend(args)
        start = time.time()
        if not self.use_subprocess:
            return self._run_in_process(name, command, args, expected_returncode, module, start)
        completed = subprocess.run(
            command,
            cwd=os.getcwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_seconds,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        return SmokeCommandResult(
            name=name,
            command=command,
            returncode=completed.returncode,
            expected_returncode=expected_returncode,
            seconds=round(time.time() - start, 3),
            passed=completed.returncode == expected_returncode,
            stdout_tail=lines[-12:],
        )

    def _run_in_process(
        self,
        name: str,
        command: List[str],
        args: List[str],
        expected_returncode: int,
        module: bool,
        start: float,
    ) -> SmokeCommandResult:
        buffer = io.StringIO()
        returncode = 0
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                if module:
                    from ..cli import main

                    returncode = int(main(args) or 0)
                else:
                    import compileall

                    returncode = 0 if compileall.compile_dir("saddle_llm", quiet=1) else 1
            except SystemExit as exc:
                returncode = int(exc.code or 0) if isinstance(exc.code, int) else 1
            except Exception as exc:
                returncode = 1
                print(f"{type(exc).__name__}: {exc}")
            finally:
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        return SmokeCommandResult(
            name=name,
            command=command + ["# in-process"],
            returncode=returncode,
            expected_returncode=expected_returncode,
            seconds=round(time.time() - start, 3),
            passed=returncode == expected_returncode,
            stdout_tail=lines[-12:],
        )

    def _prepare_fixtures(self) -> None:
        self._data_dir().mkdir(parents=True, exist_ok=True)
        self._image_dir().mkdir(parents=True, exist_ok=True)
        self._write_tiny_model()
        self._write_data()
        self._write_configs()

    def _write_tiny_model(self) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "<eos>": 2,
            "<image>": 3,
            "What": 4,
            "is": 5,
            "2": 6,
            "+": 7,
            "?": 8,
            "4": 9,
            "5": 10,
            "Say": 11,
            "hello": 12,
            "Hello": 13,
            "Goodbye": 14,
            "pick": 15,
            "cube": 16,
            "move": 17,
            "left": 18,
            "assistant": 19,
            "user": 20,
            ":": 21,
            "\n": 22,
        }
        tokenizer_impl = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_impl.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_impl,
            unk_token="<unk>",
            pad_token="<pad>",
            eos_token="<eos>",
        )
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = GPT2LMHeadModel(config)
        self._model_dir().mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self._model_dir())
        tokenizer.save_pretrained(self._model_dir())

    def _write_data(self) -> None:
        self._write_jsonl(
            self._data_path("sft.jsonl"),
            [
                {"instruction": "What is 2 + 2?", "output": "4"},
                {"instruction": "Say hello", "output": "Hello"},
            ],
        )
        self._write_jsonl(
            self._data_path("preference.jsonl"),
            [
                {"prompt": "What is 2 + 2?", "chosen": "4", "rejected": "5"},
                {"prompt": "Say hello", "chosen": "Hello", "rejected": "Goodbye"},
            ],
        )
        self._write_jsonl(
            self._data_path("kto.jsonl"),
            [
                {"prompt": "What is 2 + 2?", "completion": "4", "label": True},
                {"prompt": "What is 2 + 2?", "completion": "5", "label": False},
            ],
        )
        self._write_jsonl(
            self._data_path("vla.jsonl"),
            [
                {
                    "instruction": "pick cube",
                    "image": "frame0.png",
                    "action": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "episode_id": "e1",
                    "step_id": 0,
                },
                {
                    "instruction": "move left",
                    "image": "frame0.png",
                    "action": [0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "episode_id": "e1",
                    "step_id": 1,
                },
            ],
        )
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        with open(self._image_dir() / "frame0.png", "wb") as f:
            f.write(png)

    def _write_configs(self) -> None:
        import yaml

        configs = {
            "sft": self._base_config("tiny-sft-smoke", "orchestrator_sft_out", ["sft"]),
            "vla": self._base_config("tiny-vla-smoke", "orchestrator_vla_out", ["vla_sft"]),
            "dpo": self._preference_config("tiny-dpo-smoke", "orchestrator_dpo_out", "dpo", "preference.jsonl", 1),
            "orpo": self._preference_config("tiny-orpo-smoke", "orchestrator_orpo_out", "orpo", "preference.jsonl", 1),
            "kto": self._preference_config("tiny-kto-smoke", "orchestrator_kto_out", "kto", "kto.jsonl", 2),
            "kto_bad_batch": self._preference_config(
                "tiny-kto-bad-batch",
                "orchestrator_kto_bad_batch_out",
                "kto",
                "kto.jsonl",
                1,
            ),
        }
        configs["sft"]["sft"] = {
            "enabled": True,
            "data_path": str(self._data_path("sft.jsonl")),
            "use_lora": False,
            "use_qlora": False,
            "epochs": 1,
            "learning_rate": 5e-4,
            "per_device_batch_size": 1,
            "max_seq_length": 32,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 0,
            "save_steps": 10,
            "eval_steps": 0,
            "response_template": None,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        configs["vla"]["vla"] = {
            "enabled": True,
            "data_path": str(self._data_path("vla.jsonl")),
            "image_root": str(self._image_dir()),
            "train": True,
            "use_lora": False,
            "use_qlora": False,
            "learning_rate": 5e-4,
            "epochs": 1,
            "max_steps": 1,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 64,
            "action_space": {
                "action_dim": 7,
                "min_value": -1.0,
                "max_value": 1.0,
                "include_gripper": True,
            },
        }
        for name, config in configs.items():
            with open(self._config_path(name), "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    def _base_config(self, experiment: str, output_name: str, stages: List[str]) -> Dict:
        return {
            "project": "smoke",
            "experiment": experiment,
            "model": {
                "name_or_path": str(self._model_dir()),
                "tokenizer": str(self._model_dir()),
                "backend": "hf",
            },
            "stages": stages,
            "training": {"max_steps": 1, "preflight_only": False, "dry_run": False},
            "distributed": {"strategy": "single", "num_gpus": 1, "bf16": False, "fp16": False},
            "logging": {"output_dir": str(self.work_dir / output_name), "backend": "local"},
        }

    def _preference_config(
        self,
        experiment: str,
        output_name: str,
        method: str,
        data_name: str,
        batch_size: int,
    ) -> Dict:
        config = self._base_config(experiment, output_name, ["preference"])
        config["preference"] = {
            "enabled": True,
            "method": method,
            "data_path": str(self._data_path(data_name)),
            "beta": 0.1,
            "learning_rate": 1e-5,
            "epochs": 1,
            "per_device_batch_size": batch_size,
            "max_length": 32,
            "max_prompt_length": 16,
            "use_lora": False,
            "use_qlora": False,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 0,
            "save_steps": 10,
            "report_to": "none",
        }
        return config

    @staticmethod
    def _write_jsonl(path: Path, rows: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")

    def _check_training_artifacts(self) -> SmokeCommandResult:
        start = time.time()
        expected = {
            "sft": self.work_dir / "orchestrator_sft_out" / "summary.json",
            "vla": self.work_dir / "orchestrator_vla_out" / "summary.json",
            "dpo": self.work_dir / "orchestrator_dpo_out" / "summary.json",
            "orpo": self.work_dir / "orchestrator_orpo_out" / "summary.json",
            "kto": self.work_dir / "orchestrator_kto_out" / "summary.json",
        }
        missing = [f"{name}: {path}" for name, path in expected.items() if not path.exists()]
        details = []
        for name, path in expected.items():
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                stages = summary.get("stages_completed", [])
                details.append(f"{name}: stages={stages}")
            except Exception as exc:
                missing.append(f"{name}: unreadable summary: {exc}")
        return SmokeCommandResult(
            name="check-training-artifacts",
            command=["internal", "check-training-artifacts"],
            returncode=1 if missing else 0,
            expected_returncode=0,
            seconds=round(time.time() - start, 3),
            passed=not missing,
            stdout_tail=(missing or details)[-12:],
        )

    def _model_dir(self) -> Path:
        return self.work_dir / "tiny_model"

    def _data_dir(self) -> Path:
        return self.work_dir / "data"

    def _image_dir(self) -> Path:
        return self._data_dir() / "images"

    def _data_path(self, name: str) -> Path:
        return self._data_dir() / name

    def _config_path(self, name: str) -> Path:
        return self.work_dir / f"{name}_config.yaml"


def run_smoke_tests(
    work_dir: str = "build/smoke_e2e_auto",
    run_training: bool = True,
    clean: bool = True,
    run_doctor: bool = True,
    timeout_seconds: int = 180,
    use_subprocess: bool = True,
) -> Dict:
    return SmokeTestRunner(
        work_dir=work_dir,
        run_training=run_training,
        clean=clean,
        run_doctor=run_doctor,
        timeout_seconds=timeout_seconds,
        use_subprocess=use_subprocess,
    ).run().to_dict()
