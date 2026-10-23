"""Managed torchrun launch for the optional Megatron Bridge runtime."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from ..factory.BackendAdapters import BackendAdapterRegistry, LaunchPlan
from ..runtime.BackendConfig import MegatronBackendConfig


def write_json(path, value):
    BackendAdapterRegistry._atomic_write(
        str(path),
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def merged_overrides(common, specific):
    result = copy.deepcopy(common)
    for key, value in specific.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged_overrides(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def prepare_sft_jsonl(source, destination):
    """Adapt explicit prompt/completion or instruction/output JSONL to Bridge."""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"SFT JSONL does not exist: {source}")
    count = 0
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with (
            source.open(encoding="utf-8-sig") as reader,
            temporary.open("w", encoding="utf-8") as writer,
        ):
            for index, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"SFT JSONL row {index} must be an object")
                prompt = row.get("prompt", row.get("instruction"))
                completion = row.get("completion", row.get("output"))
                if (
                    not isinstance(prompt, str)
                    or not isinstance(completion, str)
                    or not completion
                ):
                    raise ValueError(
                        f"SFT row {index} requires prompt/completion "
                        "or instruction/output strings"
                    )
                extra = row.get("input", "")
                if extra:
                    if not isinstance(extra, str):
                        raise ValueError(f"SFT row {index} input must be a string")
                    prompt += "\n" + extra
                writer.write(
                    json.dumps(
                        {"input": prompt, "output": completion}, ensure_ascii=False
                    )
                    + "\n"
                )
                count += 1
        if not count:
            raise ValueError("SFT JSONL contains no training rows")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


class MegatronTrainingBackend:
    def __init__(self, config: MegatronBackendConfig):
        config.validate()
        self.config = config

    def make_job(
        self,
        stage,
        output_dir,
        *,
        pretrain_mode="scratch",
        model_path=None,
        sft_data_path=None,
        validation_data_path=None,
    ):
        if stage not in {"pretrain", "sft"}:
            raise ValueError("Megatron runtime supports pretrain/sft only")
        recipe = self.config.recipe_for(stage)
        if not recipe:
            raise ValueError(f"Megatron recipe is required for {stage}")
        output = Path(output_dir).resolve()
        overrides = merged_overrides(
            self.config.overrides, self.config.stage_overrides.get(stage, {})
        )
        if "save" in overrides.get("checkpoint", {}):
            raise ValueError(
                "checkpoint.save is owned by SaddleLLM; set logging.output_dir instead"
            )
        if stage == "pretrain" and pretrain_mode == "scratch":
            checkpoint = overrides.get("checkpoint", {})
            if checkpoint.get("pretrained_checkpoint"):
                raise ValueError(
                    "scratch pretraining cannot initialize pretrained_checkpoint"
                )
        run_id = self.config.run_id or uuid.uuid4().hex
        return {
            "run_id": run_id,
            "stage": stage,
            "recipe": recipe,
            "overrides": overrides,
            "pretrain_mode": pretrain_mode,
            "model_path": model_path,
            "checkpoint_path": str(output / "checkpoints"),
            "hf_path": str(output / "hf_model")
            if self.config.hf_export_source
            else None,
            "hf_export_source": self.config.hf_export_source,
            "sft_data_path": sft_data_path,
            "validation_data_path": validation_data_path,
            "result_path": str(output / f"job-{run_id}.result.json"),
        }

    def launch_plan(self, job_path):
        config = self.config
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node",
            str(config.nproc_per_node),
        ]
        if config.nnodes == 1:
            command.append("--standalone")
        else:
            command.extend(
                [
                    "--nnodes",
                    str(config.nnodes),
                    "--node-rank",
                    str(config.node_rank),
                    "--master-addr",
                    config.master_addr,
                    "--master-port",
                    str(config.master_port),
                ]
            )
        command.extend(
            ["-m", "saddlellm.training.MegatronWorker", str(Path(job_path).resolve())]
        )
        return LaunchPlan(
            "megatron",
            command,
            launcher="torchrun",
            topology={
                "num_gpus_per_node": config.nproc_per_node,
                "num_nodes": config.nnodes,
                "world_size": config.nproc_per_node * config.nnodes,
            },
            notes=[
                "Executes Megatron Bridge recipes; never falls back to Trainer.",
                "Multi-node jobs require a shared output/data filesystem.",
            ],
        )

    def _check_runtime(self):
        if sys.platform != "linux":
            raise RuntimeError(
                "Megatron training requires a supported Linux CUDA runtime; "
                "dry-run works on Windows"
            )
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError(
                "Megatron owns torchrun; do not nest it "
                "inside an existing distributed worker"
            )
        try:
            available = importlib.util.find_spec("megatron.bridge") is not None
        except ModuleNotFoundError:
            available = False
        if not available:
            raise ImportError(
                "Megatron Bridge is missing; "
                "use an official NeMo/Megatron Bridge GPU environment"
            )
        import torch

        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() < self.config.nproc_per_node
        ):
            raise RuntimeError(
                "Megatron requires enough visible CUDA GPUs for nproc_per_node"
            )

    def _execute(self, plan):
        # Use an argument list, never a shell. stdout goes to stderr so CLI JSON
        # stdout remains machine readable. Kill the process group on cancellation.
        process = subprocess.Popen(
            plan.command,
            start_new_session=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        try:
            returncode = process.wait(timeout=self.config.timeout_seconds)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, plan.command)

    def run(self, stage, output_dir, *, dry_run=False, **job_kwargs):
        job = self.make_job(stage, output_dir, **job_kwargs)
        output = Path(output_dir).resolve()
        if not dry_run:
            self._check_runtime()
            if Path(job["result_path"]).exists():
                raise FileExistsError(
                    "Megatron run_id already completed; choose a new unique run_id"
                )
            for target in (job["checkpoint_path"], job["hf_path"]):
                if target and Path(target).is_dir() and any(Path(target).iterdir()):
                    resume = job["overrides"].get("checkpoint", {}).get("load")
                    if target == job["hf_path"] or not resume:
                        raise FileExistsError(
                            "Megatron output exists; choose a new run directory "
                            "or explicit native resume"
                        )
        output.mkdir(parents=True, exist_ok=True)
        job_path = output / f"job-{job['run_id']}.json"
        write_json(job_path, job)
        plan = self.launch_plan(job_path)
        plan_path = output / f"launch_plan.node-{self.config.node_rank}.json"
        BackendAdapterRegistry.save_launch_plan(plan, str(plan_path))
        if dry_run:
            return {
                "status": "planned",
                "backend": "megatron",
                "plan_path": str(plan_path),
                "job_path": str(job_path),
            }
        self._execute(plan)
        result_path = Path(job["result_path"])
        if not result_path.is_file():
            raise RuntimeError("Megatron exited without a completed result manifest")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("run_id") != job["run_id"] or result.get("status") != "completed":
            raise RuntimeError("Megatron result manifest is stale or incomplete")
        checkpoint = Path(job["checkpoint_path"])
        if (
            result.get("checkpoint_path") != str(checkpoint)
            or not (checkpoint / "latest_checkpointed_iteration.txt").is_file()
        ):
            raise RuntimeError("Megatron did not produce a verified native checkpoint")
        if job["hf_path"]:
            hf = Path(job["hf_path"])
            if (
                result.get("model_path") != str(hf)
                or not (hf / "config.json").is_file()
            ):
                raise RuntimeError("Megatron HF export is incomplete")
            if not any(hf.glob("*.safetensors")) and not any(
                hf.glob("pytorch_model*.bin")
            ):
                raise RuntimeError("Megatron HF export has no model weights")
        return {
            **result,
            "backend": "megatron",
            "plan_path": str(plan_path),
            "job_path": str(job_path),
        }
