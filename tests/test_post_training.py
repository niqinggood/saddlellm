from __future__ import annotations

import pytest

from saddlellm.training.PostTrainingCompatibility import (
    post_training_runtime_report,
    supported_kwargs,
)
from saddlellm.training.TrainingPlanEstimator import TrainingPlanEstimator
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def test_plain_multistage_config_is_parsed_without_conversion():
    config = {
        "model": {"name_or_path": "local/tiny-model"},
        "stages": ["sft", "preference", "eval"],
        "training": {
            "max_steps": 3,
            "resume_from_checkpoint": "outputs/tiny-domain/checkpoint-2",
        },
        "sft": {
            "enabled": True,
            "data_path": "data/sft.jsonl",
            "use_lora": False,
        },
        "preference": {
            "enabled": True,
            "method": "dpo",
            "data_path": "data/preference.jsonl",
            "beta": 0.2,
        },
        "eval": {"enabled": True},
    }

    parsed = TrainingOrchestrator._parse_config(config)
    assert parsed.stages == ["sft", "preference", "eval"]
    assert parsed.model_name_or_path == "local/tiny-model"
    assert parsed.training.max_steps == 3
    assert parsed.sft.data_path == "data/sft.jsonl"
    assert parsed.sft.use_lora is False
    assert parsed.preference.method == "dpo"
    assert parsed.preference.data_path == "data/preference.jsonl"
    assert parsed.preference.beta == pytest.approx(0.2)


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


def test_preference_method_is_explicit_in_plain_config():
    parsed = TrainingOrchestrator._parse_config(
        {
            "stages": ["preference"],
            "preference": {"enabled": True, "method": "kto", "data_path": "kto.jsonl"},
        }
    )
    assert parsed.stages == ["preference"]
    assert parsed.preference.method == "kto"


@pytest.mark.parametrize("old_field", ["flow", "stage", "apiVersion"])
def test_old_config_formats_are_rejected(old_field):
    with pytest.raises(ValueError, match="Unsupported old config fields"):
        TrainingOrchestrator._parse_config(
            {old_field: "sft", "stages": ["sft"]}
        )


def test_supported_kwargs_preserves_values_for_deprecation_wrappers():
    def wrapper(*args, **kwargs):
        return args, kwargs

    values = {"model": object(), "processing_class": object(), "unused": None}
    filtered = supported_kwargs(wrapper, values)
    assert set(filtered) == {"model", "processing_class"}


def test_sft_uses_trl_sft_config_instead_of_plain_training_arguments(tmp_path):
    from saddlellm.training.PeftSFTTrainer import (
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
    from saddlellm.evaluation.SmokeTestRunner import run_smoke_tests
    from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator

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
    config = {
        "model": {
            "name_or_path": result["artifacts"]["tiny_model"],
            "tokenizer": result["artifacts"]["tiny_model"],
            "backend": "hf",
        },
        "stages": ["sft", "preference", "eval", "export"],
        "training": {"max_steps": 1, "warmup_steps": 0},
        "distributed": {
            "strategy": "single",
            "bf16": False,
            "fp16": False,
            "gradient_accumulation_steps": 1,
        },
        "logging": {"output_dir": str(tmp_path / "config-output"), "backend": "local"},
        "sft": {
            "enabled": True,
            "data_path": f"{data_dir}/sft.jsonl",
            "use_lora": False,
            "use_qlora": False,
            "per_device_batch_size": 1,
            "max_seq_length": 32,
            "logging_steps": 1,
            "gradient_checkpointing": False,
            "local_files_only": True,
        },
        "preference": {
            "enabled": True,
            "method": "dpo",
            "data_path": f"{data_dir}/preference.jsonl",
            "learning_rate": 1e-5,
            "use_lora": False,
            "use_qlora": False,
            "per_device_batch_size": 1,
            "max_length": 32,
            "max_prompt_length": 16,
            "logging_steps": 1,
            "gradient_checkpointing": False,
            "local_files_only": True,
        },
        "eval": {
            "enabled": True,
            "tasks": ["perplexity"],
            "dataset": f"{data_dir}/sft.jsonl",
            "max_samples": 2,
            "gate": {
                "enabled": True,
                "rules": {"perplexity": {"max": 1000}},
            },
        },
        "export": {
            "enabled": True,
            "output_dir": str(tmp_path / "config-release"),
            "require_gate": True,
            "local_files_only": True,
        },
    }
    stages = TrainingOrchestrator.from_dict(config).run()
    assert stages["sft"]["status"] == "completed"
    assert stages["preference"]["status"] == "completed"
    assert stages["eval"]["status"] == "completed"
    assert "perplexity" in stages["eval"]["metrics"]
    assert stages["eval"]["release_gate"]["accepted"] is True
    assert stages["export"]["status"] == "completed"
    assert stages["export"]["source_type"] == "local_checkpoint"
    assert (tmp_path / "config-release" / "release_manifest.json").is_file()
