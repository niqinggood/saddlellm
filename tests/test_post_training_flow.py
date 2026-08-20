from __future__ import annotations

import pytest

from saddlellm.PostTrainingCompatibility import (
    post_training_runtime_report,
    supported_kwargs,
)
from saddlellm.SimpleFlow import SimpleFlowCompiler
from saddlellm.TrainingPlanEstimator import TrainingPlanEstimator


def test_simple_sft_dpo_flow_compiles_to_canonical_orchestrator_config():
    compiled = SimpleFlowCompiler.compile(
        {
            "name": "tiny-domain",
            "model": "local/tiny-model",
            "flow": "sft+dpo",
            "data": {
                "sft": "data/sft.jsonl",
                "preference": "data/preference.jsonl",
            },
            "train": {
                "qlora": False,
                "lora": False,
                "epochs": 2,
                "batch_size": 2,
                "max_steps": 3,
                "max_sequence_length": 128,
            },
            "advanced": {"resume": "outputs/tiny-domain/checkpoint-2"},
            "dpo": {"beta": 0.2},
            "evaluation": {"enabled": True, "suite": ["perplexity"]},
            "output": "outputs/tiny-domain",
        }
    )

    assert compiled["stages"] == ["sft", "preference", "eval"]
    assert compiled["model"]["name_or_path"] == "local/tiny-model"
    assert compiled["training"]["max_steps"] == 3
    assert compiled["sft"]["data_path"] == "data/sft.jsonl"
    assert compiled["sft"]["use_lora"] is False
    assert compiled["preference"]["method"] == "dpo"
    assert compiled["preference"]["data_path"] == "data/preference.jsonl"
    assert compiled["preference"]["beta"] == pytest.approx(0.2)
    assert compiled["training"]["resume_from_checkpoint"] == "outputs/tiny-domain/checkpoint-2"


def test_epoch_schedule_does_not_claim_zero_planned_tokens():
    estimate = TrainingPlanEstimator.estimate(
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        num_gpus=1,
        max_seq_length=128,
        max_steps=-1,
        epochs=2,
    )
    assert estimate.schedule_mode == "epochs"
    assert estimate.planned_tokens is None


def test_simple_kto_flow_gets_a_valid_default_batch_size():
    compiled = SimpleFlowCompiler.compile(
        {
            "model": "local/tiny-model",
            "flow": "sft+kto",
            "data": {"sft": "sft.jsonl", "preference": "kto.jsonl"},
            "evaluation": {"enabled": False},
            "output": "outputs/kto",
        }
    )
    assert compiled["stages"] == ["sft", "preference"]
    assert compiled["preference"]["method"] == "kto"
    assert compiled["preference"]["per_device_batch_size"] == 2


def test_simple_flow_rejects_missing_stage_data_and_planned_stages():
    with pytest.raises(ValueError, match="data.preference"):
        SimpleFlowCompiler.compile(
            {
                "model": "local/tiny-model",
                "flow": "sft+dpo",
                "data": {"sft": "sft.jsonl"},
            }
        )
    with pytest.raises(NotImplementedError, match="grpo"):
        SimpleFlowCompiler.compile(
            {
                "model": "local/tiny-model",
                "flow": "sft+grpo",
                "data": {"sft": "sft.jsonl", "rl": "rl.jsonl"},
            }
        )


def test_supported_kwargs_preserves_values_for_deprecation_wrappers():
    def wrapper(*args, **kwargs):
        return args, kwargs

    values = {"model": object(), "processing_class": object(), "unused": None}
    filtered = supported_kwargs(wrapper, values)
    assert set(filtered) == {"model", "processing_class"}


def test_sft_uses_trl_sft_config_instead_of_plain_training_arguments(tmp_path):
    from saddlellm.PeftSFTTrainer import (
        SFTTrainConfig,
        _make_training_args,
        _maybe_quantization_config,
    )

    args = _make_training_args(
        SFTTrainConfig(
            model_path="unused",
            dataset_path="unused",
            output_path=str(tmp_path),
            max_steps=1,
            max_seq_length=64,
            gradient_checkpointing=False,
        ),
        device="cpu",
    )
    assert args.__class__.__name__ == "SFTConfig"
    assert args.max_steps == 1
    assert getattr(args, "max_length", getattr(args, "max_seq_length", None)) == 64

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _maybe_quantization_config(
            SFTTrainConfig(
                model_path="unused",
                dataset_path="unused",
                output_path=str(tmp_path),
                use_lora=True,
                use_qlora=True,
            ),
            device="cpu",
        )


@pytest.mark.skipif(
    not post_training_runtime_report()["compatible"],
    reason="requires requirements-posttrain.txt",
)
def test_pinned_runtime_completes_real_post_training_smoke(tmp_path):
    from saddlellm.SmokeTestRunner import run_smoke_tests
    from saddlellm.TrainingOrchestrator import TrainingOrchestrator

    result = run_smoke_tests(
        work_dir=str(tmp_path / "posttrain-smoke"),
        run_training=True,
        clean=True,
        run_doctor=False,
        timeout_seconds=180,
        use_subprocess=False,
    )
    assert result["ready"], result["issues"]
    assert not result["issues"]

    data_dir = result["artifacts"]["data_dir"]
    compiled = SimpleFlowCompiler.compile(
        {
            "name": "simple-flow-e2e",
            "model": result["artifacts"]["tiny_model"],
            "flow": "sft+dpo+eval+export",
            "data": {
                "sft": f"{data_dir}/sft.jsonl",
                "preference": f"{data_dir}/preference.jsonl",
            },
            "train": {
                "lora": False,
                "qlora": False,
                "max_steps": 1,
                "batch_size": 1,
                "logging_steps": 1,
            },
            "dpo": {"learning_rate": 1e-5},
            "advanced": {
                "bf16": False,
                "fp16": False,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
                "local_files_only": True,
                "max_sequence_length": 32,
            },
            "evaluation": {
                "enabled": True,
                "suite": ["perplexity"],
                "dataset": f"{data_dir}/sft.jsonl",
                "max_samples": 2,
                "gate": {
                    "enabled": True,
                    "rules": {"perplexity": {"max": 1000}},
                },
            },
            "export": {
                "output_dir": str(tmp_path / "simple-flow-release"),
                "require_gate": True,
                "local_files_only": True,
            },
            "output": str(tmp_path / "simple-flow-output"),
        }
    )
    stages = TrainingOrchestrator.from_dict(compiled).run()
    assert stages["sft"]["status"] == "completed"
    assert stages["preference"]["status"] == "completed"
    assert stages["eval"]["status"] == "completed"
    assert "perplexity" in stages["eval"]["metrics"]
    assert stages["eval"]["release_gate"]["accepted"] is True
    assert stages["export"]["status"] == "completed"
    assert stages["export"]["source_type"] == "local_checkpoint"
    assert (tmp_path / "simple-flow-release" / "release_manifest.json").is_file()
