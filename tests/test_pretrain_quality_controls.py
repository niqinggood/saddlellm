from types import SimpleNamespace

import pytest

from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def _pretrain_config():
    return {
        "stages": ["pretrain"],
        "model": {"backend": "hf", "name_or_path": "unused-model"},
        "pretrain_mode": "continue",
        "data": {
            "sources": [
                {"type": "local", "path": "unused.jsonl", "streaming": False}
            ],
            "quality_min_score": 0.73,
        },
        "training": {
            "per_device_batch_size": 2,
            "global_batch_size": 8,
            "learning_rate": 3e-4,
            "min_lr": 3e-5,
            "eval_every_steps": 100,
            "save_every_steps": 500,
        },
        "distributed": {
            "strategy": "single",
            "num_gpus": 1,
            "num_nodes": 1,
            "bf16": False,
            "fp16": False,
        },
        "eval": {"enabled": False},
    }


def test_pretrain_quality_defaults_and_threshold_are_parsed():
    parsed = TrainingOrchestrator._parse_config(_pretrain_config())

    assert parsed.data.append_eos is True
    assert parsed.data.validation_split == 0.01
    assert parsed.data.quality_min_score == 0.73
    assert parsed.training.load_best_model_at_end is True
    assert parsed.distributed.gradient_accumulation_steps is None
    assert parsed.training.loss_chunk_size == 1024
    assert parsed.training.dataloader_pin_memory is True
    assert parsed.training.dataloader_persistent_workers is True
    assert parsed.training.dataloader_prefetch_factor == 2


def test_pretrain_optimizer_and_gradient_checkpointing_are_not_ignored():
    config = _pretrain_config()
    config["training"].update(
        {"optimizer": "adamw_torch_fused", "gradient_checkpointing": False}
    )

    parsed = TrainingOrchestrator._parse_config(config)

    assert parsed.training.optimizer == "adamw_torch_fused"
    assert parsed.training.gradient_checkpointing is False


def test_pretrain_loss_chunk_and_dataloader_tuning_are_configurable():
    config = _pretrain_config()
    config["training"].update(
        {
            "loss_chunk_size": 256,
            "dataloader_num_workers": 6,
            "dataloader_pin_memory": False,
            "dataloader_persistent_workers": False,
            "dataloader_prefetch_factor": 4,
        }
    )

    parsed = TrainingOrchestrator._parse_config(config)

    assert parsed.training.loss_chunk_size == 256
    assert parsed.training.dataloader_num_workers == 6
    assert parsed.training.dataloader_pin_memory is False
    assert parsed.training.dataloader_persistent_workers is False
    assert parsed.training.dataloader_prefetch_factor == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss_chunk_size", -1),
        ("loss_chunk_size", True),
        ("dataloader_num_workers", -1),
        ("dataloader_prefetch_factor", 0),
    ],
)
def test_pretrain_validator_rejects_invalid_efficiency_controls(field, value):
    config = _pretrain_config()
    config["training"][field] = value

    result = TrainingConfigValidator.validate(config, inspect_data=False)

    assert result.valid is False
    assert any(field in issue for issue in result.issues)


def test_pretrain_local_files_only_is_parsed_from_model_section():
    config = _pretrain_config()
    config["model"]["local_files_only"] = True

    parsed = TrainingOrchestrator._parse_config(config)

    assert parsed.model_local_files_only is True


def test_validator_reports_inherited_architecture_for_continuation():
    result = TrainingConfigValidator.validate(
        _pretrain_config(), inspect_data=False
    )

    assert result.architecture_analysis == {
        "mode": "continue",
        "source": "checkpoint",
        "model_name_or_path": "unused-model",
        "structure_inherited": True,
    }


def test_validator_expands_and_analyzes_scratch_architecture_preset():
    config = _pretrain_config()
    config["pretrain_mode"] = "scratch"
    config["model"] = {"backend": "saddle", "config": "qwen-tiny-160m"}

    result = TrainingConfigValidator.validate(config, inspect_data=False)

    assert result.valid is True
    assert result.architecture_analysis["mode"] == "scratch"
    assert result.architecture_analysis["source"] == "model.config preset"
    assert result.architecture_analysis["name"] == "qwen-tiny-160m"
    assert result.architecture_analysis["total_params"] > 0
    assert result.architecture_analysis["architecture_axes"]["attention"] == "gqa"


class _Tokenizer:
    bos_token_id = 6
    eos_token_id = 7
    pad_token_id = 7

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_pretrain_tokenizer_contract_reports_compatible_vocabularies():
    model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=12,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )

    contract = TrainingOrchestrator.validate_pretrain_tokenizer_contract(
        model, _Tokenizer(10), append_eos=True
    )

    assert contract["model_vocab_size"] == 12
    assert contract["tokenizer_vocab_size"] == 10
    assert contract["unused_embedding_rows"] == 2
    assert contract["bos_token_id"] == 6
    assert model.config.bos_token_id == 6
    assert model.config.eos_token_id == 7
    assert model.config.pad_token_id == 7


def test_pretrain_tokenizer_contract_rejects_missing_eos_and_oversized_vocab():
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=8))
    tokenizer = _Tokenizer(9)
    tokenizer.eos_token_id = None

    with pytest.raises(ValueError, match="eos_token_id"):
        TrainingOrchestrator.validate_pretrain_tokenizer_contract(
            model, tokenizer, append_eos=True
        )

    tokenizer.eos_token_id = 7
    with pytest.raises(ValueError, match="exceeds model"):
        TrainingOrchestrator.validate_pretrain_tokenizer_contract(
            model, tokenizer, append_eos=True
        )


def test_pretrain_validator_rejects_inconsistent_explicit_global_batch():
    config = _pretrain_config()
    config["distributed"]["gradient_accumulation_steps"] = 1

    result = TrainingConfigValidator.validate(config, inspect_data=False)

    assert result.valid is False
    assert any("global_batch_size" in issue for issue in result.issues)


def test_pretrain_validator_accepts_automatic_gradient_accumulation():
    result = TrainingConfigValidator.validate(
        _pretrain_config(), inspect_data=False
    )

    assert result.valid is True
    assert not result.issues


def test_deepspeed_efficiency_controls_are_parsed_and_validated():
    config = _pretrain_config()
    config["distributed"].update(
        {
            "tf32": True,
            "zero_overlap_comm": False,
            "zero_contiguous_gradients": False,
            "zero_reduce_scatter": False,
            "zero_allgather_partitions": False,
            "zero_stage3_prefetch_bucket_size": 123,
            "zero_reduce_bucket_size": 456,
            "zero_allgather_bucket_size": 789,
            "zero_offload_pin_memory": False,
        }
    )

    parsed = TrainingOrchestrator._parse_config(config)

    assert parsed.distributed.tf32 is True
    assert parsed.distributed.zero_overlap_comm is False
    assert parsed.distributed.zero_contiguous_gradients is False
    assert parsed.distributed.zero_reduce_scatter is False
    assert parsed.distributed.zero_allgather_partitions is False
    assert parsed.distributed.zero_stage3_prefetch_bucket_size == 123
    assert parsed.distributed.zero_reduce_bucket_size == 456
    assert parsed.distributed.zero_allgather_bucket_size == 789
    assert parsed.distributed.zero_offload_pin_memory is False


def test_validator_rejects_conflicting_or_invalid_zero_offload_controls():
    config = _pretrain_config()
    config["distributed"].update(
        {
            "zero_offload_to_cpu": True,
            "zero_offload_to_nvme": True,
            "zero_reduce_bucket_size": -1,
        }
    )

    result = TrainingConfigValidator.validate(config, inspect_data=False)

    assert result.valid is False
    assert any("mutually exclusive" in issue for issue in result.issues)
    assert any("zero_reduce_bucket_size" in issue for issue in result.issues)


def test_pretrain_validator_requires_compatible_eval_and_save_intervals():
    config = _pretrain_config()
    config["training"]["save_every_steps"] = 150

    result = TrainingConfigValidator.validate(config, inspect_data=False)

    assert result.valid is False
    assert any("multiple" in issue for issue in result.issues)
