from __future__ import annotations

import json

import pytest

from saddlellm.training.PostTrainingCompatibility import (
    latest_train_metrics,
    post_training_runtime_report,
    resolve_post_training_precision,
    resolve_trainer_resume_checkpoint,
    supported_kwargs,
    validate_resume_training_args,
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


def test_post_training_precision_is_explicit_and_device_safe(tmp_path):
    import torch

    from saddlellm.training.PeftSFTTrainer import SFTTrainConfig, _make_training_args
    from saddlellm.training.Preference import (
        PreferenceTrainConfig,
        _make_training_args as make_preference_args,
    )

    actual, dtype, bf16, fp16 = resolve_post_training_precision("fp32", "cuda")
    assert (actual, dtype, bf16, fp16) == ("fp32", torch.float32, False, False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        resolve_post_training_precision("fp16", "cpu")

    sft_args = _make_training_args(
        SFTTrainConfig(
            model_path="unused",
            dataset_path="unused",
            output_path=str(tmp_path / "sft"),
            precision="fp32",
        ),
        device="cuda",
    )
    assert sft_args.bf16 is False
    assert sft_args.fp16 is False

    preference_args = make_preference_args(
        PreferenceTrainConfig(
            model_path="unused",
            dataset_path="unused",
            output_path=str(tmp_path / "preference"),
            precision="fp32",
        )
    )
    assert preference_args.bf16 is False
    assert preference_args.fp16 is False


def test_resume_checkpoint_is_resolved_and_latest_loss_stays_visible(tmp_path):
    checkpoint_1 = tmp_path / "checkpoint-1"
    checkpoint_2 = tmp_path / "checkpoint-2"
    checkpoint_1.mkdir()
    checkpoint_2.mkdir()

    assert resolve_trainer_resume_checkpoint(True, str(tmp_path)) == str(
        checkpoint_2.resolve()
    )
    assert resolve_trainer_resume_checkpoint(str(checkpoint_1), "unused") == str(
        checkpoint_1.resolve()
    )
    assert latest_train_metrics(
        [
            {"loss": 3.2, "step": 1},
            {"eval_loss": 3.0, "step": 1},
            {"loss": 2.9, "learning_rate": 1e-5, "step": 2},
            {"train_loss": 1.45, "step": 2},
        ]
    ) == {"loss": 2.9, "learning_rate": 1e-5, "step": 2}

    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        resolve_trainer_resume_checkpoint(True, str(tmp_path / "missing"))


def test_exact_resume_rejects_changed_train_batch_size(tmp_path):
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"train_batch_size": 2}),
        encoding="utf-8",
    )

    class Args:
        train_batch_size = 1

    with pytest.raises(ValueError, match="train_batch_size mismatch"):
        validate_resume_training_args(str(checkpoint), Args())

    Args.train_batch_size = 2
    validate_resume_training_args(str(checkpoint), Args())


def test_orchestrator_maps_explicit_distributed_precision_to_post_training():
    parsed = TrainingOrchestrator._parse_config(
        {
            "stages": ["sft", "preference"],
            "distributed": {"bf16": False, "fp16": False},
            "sft": {"enabled": True, "data_path": "sft.jsonl"},
            "preference": {
                "enabled": True,
                "method": "dpo",
                "data_path": "preference.jsonl",
            },
        }
    )

    assert parsed.sft.precision == "fp32"
    assert parsed.preference.precision == "fp32"


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


def test_preference_config_keeps_lora_target_modules():
    parsed = TrainingOrchestrator._parse_config(
        {
            "stages": ["preference"],
            "preference": {
                "enabled": True,
                "method": "dpo",
                "data_path": "dpo.jsonl",
                "lora": {"target_modules": ["c_attn"]},
            },
        }
    )
    assert parsed.preference.lora_target_modules == ["c_attn"]


def test_post_training_config_accepts_explicit_validation_and_all_linear():
    parsed = TrainingOrchestrator._parse_config(
        {
            "stages": ["sft", "preference"],
            "sft": {
                "enabled": True,
                "data_path": "sft.jsonl",
                "validation_data_path": "sft-validation.jsonl",
                "lora": {"target_modules": "all-linear"},
            },
            "preference": {
                "enabled": True,
                "method": "dpo",
                "data_path": "dpo.jsonl",
                "validation_data_path": "dpo-validation.jsonl",
                "lora": {"target_modules": "all-linear"},
            },
        }
    )

    assert parsed.sft.validation_data_path == "sft-validation.jsonl"
    assert parsed.sft.lora_target_modules == "all-linear"
    assert parsed.preference.validation_data_path == "dpo-validation.jsonl"
    assert parsed.preference.lora_target_modules == "all-linear"


def test_preference_detects_upstream_peft_adapter_base_model(tmp_path):
    from saddlellm.training.Preference import _adapter_base_model

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "local/base-model"}),
        encoding="utf-8",
    )

    assert _adapter_base_model(str(adapter)) == "local/base-model"
    assert _adapter_base_model(str(tmp_path / "full-model")) is None


def test_preference_rejects_adapter_without_base_model(tmp_path):
    from saddlellm.training.Preference import _adapter_base_model

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="base_model_name_or_path"):
        _adapter_base_model(str(adapter))


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


def test_sft_native_completion_format_separates_prompt_and_answer():
    from saddlellm.training.PeftSFTTrainer import _format_sft_batch

    class TinyChatTokenizer:
        @staticmethod
        def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            rendered = "".join(
                f"<{message['role']}>{message['content']}" for message in messages
            )
            if add_generation_prompt:
                rendered += "<assistant>"
            return rendered

    formatted = _format_sft_batch(
        {"instruction": ["What is hypertension?"], "output": ["Persistently elevated blood pressure."]},
        TinyChatTokenizer(),
        completion_only=True,
    )
    assert formatted == {
        "prompt": ["<user>What is hypertension?<assistant>"],
        "completion": ["Persistently elevated blood pressure."],
    }


def test_sft_completion_format_trains_every_assistant_turn():
    from saddlellm.training.PeftSFTTrainer import _format_sft_batch

    class TinyChatTokenizer:
        @staticmethod
        def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            rendered = "".join(
                f"<{message['role']}>{message['content']}" for message in messages
            )
            if add_generation_prompt:
                rendered += "<assistant>"
            return rendered

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "second answer"},
    ]
    formatted = _format_sft_batch(
        {"messages": [messages]},
        TinyChatTokenizer(),
        completion_only=True,
    )

    assert formatted["completion"] == ["first answer", "second answer"]
    assert formatted["prompt"] == [
        "<user>first question<assistant>",
        "<user>first question<assistant>first answer<user>follow-up<assistant>",
    ]


def test_sft_training_args_enable_native_completion_loss(tmp_path):
    from saddlellm.training.PeftSFTTrainer import SFTTrainConfig, _make_training_args

    args = _make_training_args(
        SFTTrainConfig(
            model_path="unused",
            dataset_path="unused",
            output_path=str(tmp_path),
        ),
        device="cpu",
        native_completion_only=True,
    )
    if hasattr(args, "completion_only_loss"):
        assert args.completion_only_loss is True


def test_post_training_selects_best_checkpoint_with_explicit_validation(tmp_path):
    from saddlellm.training.PeftSFTTrainer import SFTTrainConfig, _make_training_args
    from saddlellm.training.Preference import (
        PreferenceTrainConfig,
        _make_training_args as make_preference_args,
    )

    sft_args = _make_training_args(
        SFTTrainConfig(
            model_path="unused",
            dataset_path="train.jsonl",
            validation_dataset_path="validation.jsonl",
            output_path=str(tmp_path / "sft"),
            eval_steps=0,
        ),
        device="cpu",
        native_completion_only=True,
    )
    assert sft_args.load_best_model_at_end is True
    assert sft_args.eval_strategy.value == "epoch"
    assert sft_args.save_strategy.value == "epoch"
    assert sft_args.metric_for_best_model == "eval_loss"
    assert sft_args.greater_is_better is False

    preference_args = make_preference_args(
        PreferenceTrainConfig(
            model_path="unused",
            dataset_path="train.jsonl",
            validation_dataset_path="validation.jsonl",
            output_path=str(tmp_path / "preference"),
            eval_steps=0,
        )
    )
    assert preference_args.load_best_model_at_end is True
    assert preference_args.eval_strategy.value == "epoch"
    assert preference_args.save_strategy.value == "epoch"
    assert preference_args.metric_for_best_model == "eval_loss"


def test_best_checkpoint_step_schedule_requires_matching_cadence(tmp_path):
    from saddlellm.training.PeftSFTTrainer import SFTTrainConfig, _make_training_args

    with pytest.raises(ValueError, match="multiple of eval_steps"):
        _make_training_args(
            SFTTrainConfig(
                model_path="unused",
                dataset_path="train.jsonl",
                validation_dataset_path="validation.jsonl",
                output_path=str(tmp_path),
                eval_steps=3,
                save_steps=5,
            ),
            device="cpu",
        )


def test_native_sft_builds_one_masked_sample_per_assistant_turn():
    from saddlellm.training.NativePostTraining import _sft_features

    class TinyTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        @staticmethod
        def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            rendered = "".join(
                f"<{message['role']}>{message['content']}" for message in messages
            )
            if add_generation_prompt:
                rendered += "<assistant>"
            return rendered

        @staticmethod
        def __call__(text, **kwargs):
            ids = [ord(char) % 127 + 1 for char in text]
            return {"input_ids": ids[: kwargs["max_length"]]}

    features = _sft_features(
        {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        },
        TinyTokenizer(),
        128,
    )

    assert len(features) == 2
    assert all(any(label != -100 for label in item["labels"]) for item in features)


def test_sft_summary_uses_latest_json_safe_eval_metrics():
    from saddlellm.training.PeftSFTTrainer import _latest_eval_metrics

    metrics = _latest_eval_metrics(
        [
            {"loss": 2.0, "step": 1},
            {"eval_loss": 1.9, "eval_runtime": 2.5, "epoch": 0.5, "step": 1},
            {"loss": 1.8, "step": 2},
            {"eval_loss": 1.7, "eval_runtime": 2.2, "epoch": 1.0, "step": 2},
        ]
    )

    assert metrics == {
        "eval_loss": 1.7,
        "eval_runtime": 2.2,
        "epoch": 1.0,
        "step": 2,
    }


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
