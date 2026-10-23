"""Replay selection must remain deterministic and preserve held-out data."""
from __future__ import annotations

import json
import math

import pytest

from saddlellm.data.ReplayData import (
    filter_replay_records,
    mix_replay_samples,
    replay_source_fingerprint,
    sft_source_groups,
    validate_replay_config,
)


def test_replay_is_disabled_by_default_and_validated_before_data_loading():
    assert validate_replay_config(None) is False
    assert validate_replay_config("general.jsonl", 0.2) is True
    assert validate_replay_config("history.jsonl", replay_count=3) is True


@pytest.mark.parametrize("path,ratio,count", [
    (None, 0.2, None), ("data", 0, None), ("", 0.2, None),
    ("data", 1, None), ("data", -0.1, None), ("data", math.nan, None),
    ("data", math.inf, None), ("data", True, None), ("data", "0.2", None),
    ("data", 0.2, 1), ("data", 0, 0), ("data", 0, True), ("data", 0, 1.5),
])
def test_replay_rejects_invalid_and_ambiguous_controls(path, ratio, count):
    with pytest.raises(ValueError):
        validate_replay_config(path, ratio, count)


def test_source_exclusion_links_conversation_groups_and_cross_schema_prompts():
    held_out = [{"instruction": " held   out ", "output": "answer", "case_id": "case-a"}]
    replay = [
        {"messages": [{"role": "user", "content": "held out"},
                      {"role": "assistant", "content": "another answer"}],
         "conversation_id": "linked"},
        {"instruction": "another variant", "output": "variant", "conversation_id": "linked"},
        {"instruction": "case variant", "output": "variant", "metadata": {"case_id": "case-a"}},
        {"instruction": "general knowledge", "output": "retained", "source": "same-dataset"},
        {"invalid": "record"},
    ]
    eligible, report = filter_replay_records(replay, held_out)
    assert eligible == [replay[3]]
    assert report["excluded_validation_records"] == 3
    assert report["invalid_records"] == 1


def test_sft_groups_connect_prompt_and_content_variants_transitively():
    records = [
        {"instruction": "same question", "output": "answer one", "case_id": "left"},
        {"messages": [{"role": "user", "content": "same   question"},
                      {"role": "assistant", "content": "answer two"}], "case_id": "right"},
        {"instruction": "another question", "output": "answer three", "case_id": "right"},
        {"text": " general knowledge ", "source_id": "whole-corpus"},
        {"text": "general   knowledge", "source_id": "whole-corpus"},
        {"text": "independent knowledge", "source_id": "whole-corpus"},
    ]
    assert sft_source_groups(records) == [0, 0, 0, 3, 3, 5]


def test_ratio_means_final_training_mixture_and_is_reproducible():
    primary = [{"text": f"task {i}"} for i in range(7)]
    replay = [{"text": f"general {i}"} for i in range(20)]
    before = list(primary)
    mixed, report = mix_replay_samples(primary, replay, replay_ratio=0.3, seed=7)
    again, again_report = mix_replay_samples(primary, replay, replay_ratio=0.3, seed=7)
    assert mixed == again and report == again_report
    assert report["primary_samples"] == 7
    assert report["replay_samples"] == 3
    assert report["actual_ratio"] == pytest.approx(0.3)
    assert primary == before
    assert mix_replay_samples(primary, replay, replay_ratio=0.3, seed=8)[0] != mixed


def test_rounding_and_count_are_explicit_and_no_oversampling_is_hidden():
    train = [{"text": "task"}]
    replay = [{"text": f"general {i}"} for i in range(3)]
    _, report = mix_replay_samples(train, replay, replay_ratio=0.1)
    assert report["requested_ratio"] == 0.1
    assert report["actual_ratio"] == 0.5
    with pytest.raises(ValueError, match="requested 4 distinct samples but only 3"):
        mix_replay_samples(train, replay, replay_count=4)


def test_final_sample_checks_exclude_tokenized_validation_and_duplicates():
    task = {"input_ids": [1, 2], "labels": [-100, 2]}
    held_out = {"input_ids": [1, 3], "labels": [-100, 3]}
    retained = {"input_ids": [1, 4], "labels": [-100, 4]}
    mixed, report = mix_replay_samples(
        [task], [task, held_out, retained, retained], replay_count=1,
        validation_samples=[held_out],
    )
    assert held_out not in mixed
    assert task in mixed and retained in mixed
    assert report["excluded_validation_samples"] == 1
    assert report["excluded_duplicate_samples"] == 2
    assert replay_source_fingerprint([retained]) != replay_source_fingerprint([task])


def test_held_out_inputs_remain_excluded_when_loss_mask_schema_differs():
    train = [{"input_ids": [1, 2], "labels": [-100, 2]}]
    held_out = {"input_ids": [1, 3], "labels": [-100, 3]}
    replay = [{"input_ids": [1, 3], "labels": [1, 3]},
              {"input_ids": [1, 4], "labels": [-100, 4]}]
    mixed, report = mix_replay_samples(train, replay, replay_count=1, validation_samples=[held_out])
    assert all(item["input_ids"] != held_out["input_ids"] for item in mixed)
    assert report["excluded_validation_samples"] == 1


def _tiny_checkpoint(tmp_path, backend):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(13)
    path = tmp_path / "base"
    vocabulary = {word: i for i, word in enumerate([
        "<pad>", "<bos>", "<eos>", "<unk>", "task", "general", "answer",
        "validation", "zero", "one", "two", "three", "four", "five", "user", "assistant", ":",
    ])}
    backend_tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer, pad_token="<pad>", bos_token="<bos>",
        eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)
    if backend == "saddle":
        from saddlellm.models.ModelBlueprint import (
            AttentionBlueprint,
            FFNBlueprint,
            ModelBlueprint,
        )

        model = ModelBlueprint(
            name="tiny-replay", hidden_size=16, vocab_size=len(vocabulary), num_layers=1,
            max_position_embeddings=32,
            attention=AttentionBlueprint(kind="gqa", backend="eager", num_heads=2, num_kv_heads=1),
            ffn=FFNBlueprint(kind="swiglu", intermediate_size=32),
        ).build_model()
        model.config.loss_chunk_size = 4
    else:
        from transformers import GPT2Config, GPT2LMHeadModel

        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(vocabulary), n_positions=32, n_ctx=32, n_embd=16, n_layer=1, n_head=2,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        ))
    model.save_pretrained(path)
    return path


@pytest.mark.parametrize("mode", ["text_truncation", "completion_truncation", "text_eos"])
@pytest.mark.parametrize("eligible_sample", [False, True])
def test_hf_replay_exclusion_uses_actual_trl_prepared_tokens(
    tmp_path, monkeypatch, mode, eligible_sample,
):
    import torch

    from saddlellm.training import PeftSFTTrainer as sft

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base = _tiny_checkpoint(tmp_path, "hf")
    if mode == "completion_truncation":
        training = {"instruction": "task one", "output": "answer one"}
        validation = {"instruction": "validation unknown_A", "output": "answer zero"}
        colliding = {"instruction": "validation unknown_B", "output": "answer four"}
        independent = {"instruction": "general five", "output": "answer five"}
        max_length = 7
        response_template = "### Answer:"
    else:
        training = {"text": "task one answer one"}
        independent = {"text": "general five answer five"}
        response_template = None
        if mode == "text_truncation":
            validation = {"text": "general three answer zero"}
            colliding = {"text": "general three answer four"}
            max_length = 2
        else:
            validation = {"text": "general three"}
            colliding = {"text": "general three<eos>"}
            max_length = 24
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    train_path.write_text(json.dumps(training) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    replay_rows = [colliding, independent] if eligible_sample else [colliding]
    replay_path.write_text("\n".join(json.dumps(item) for item in replay_rows), encoding="utf-8")
    captured = {}
    original_builder = sft._build_sft_trainer

    class TrainingReached(Exception):
        pass

    def builder(**kwargs):
        trainer = original_builder(**kwargs)
        captured["trainer"] = trainer

        def inspect_before_training(**unused):
            captured["train"] = list(trainer.train_dataset)
            captured["validation"] = list(trainer.eval_dataset)
            raise TrainingReached

        trainer.train = inspect_before_training
        return trainer

    monkeypatch.setattr(sft, "_build_sft_trainer", builder)
    config = sft.SFTTrainConfig(
        model_path=str(base), dataset_path=str(train_path), output_path=str(tmp_path / "output"),
        validation_dataset_path=str(validation_path), replay_data_path=str(replay_path), replay_count=1,
        max_seq_length=max_length, response_template=response_template,
        use_lora=False, use_qlora=False, gradient_checkpointing=False, load_best_model_at_end=False,
        max_steps=1, warmup_steps=0, local_files_only=True,
    )
    if eligible_sample:
        with pytest.raises(TrainingReached):
            sft.train_sft(config)
        assert len(captured["train"]) == 2
        train_inputs = {tuple(item["input_ids"]) for item in captured["train"]}
        validation_inputs = {tuple(item["input_ids"]) for item in captured["validation"]}
        assert train_inputs.isdisjoint(validation_inputs)
        assert all("_saddle_replay_origin" not in item for item in captured["train"])
    else:
        with pytest.raises(ValueError, match="requested 1 distinct samples but only 0"):
            sft.train_sft(config)
        assert "train" not in captured


def test_native_sft_automatic_split_receives_prompt_identity_groups(tmp_path, monkeypatch):
    from saddlellm.training import NativePostTraining as native

    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps(item) for item in [
        {"instruction": "shared question", "output": "answer one"},
        {"instruction": "shared question", "output": "answer two"},
        {"instruction": "independent question", "output": "answer three"},
    ]), encoding="utf-8")

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class SplitReached(Exception):
        pass

    def inspect_split(features, fraction, seed, *, groups):
        assert groups == [0, 0, 2]
        raise SplitReached

    monkeypatch.setattr(native, "_activate_local_device", lambda: None)
    monkeypatch.setattr(native.SaddleModelAdapter, "load_tokenizer", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(native, "_sft_features", lambda *a: [{"input_ids": [1, 2], "labels": [-100, 2]}])
    monkeypatch.setattr(native, "_split_features", inspect_split)
    with pytest.raises(SplitReached):
        native.train_native_sft(native.NativeSFTConfig(
            model_path="unused", dataset_path=str(source), output_path=str(tmp_path / "out"),
            validation_split=0.5,
        ))


@pytest.mark.parametrize("backend", ["saddle", "hf"])
@pytest.mark.parametrize("retention_weight", [0.0, 0.2])
def test_real_cpu_sft_with_replay_and_retention_preserves_validation(
    tmp_path, monkeypatch, backend, retention_weight,
):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base = _tiny_checkpoint(tmp_path, backend)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    replay = tmp_path / "replay.jsonl"
    train.write_text(json.dumps({"text": "task one answer one"}) + "\n" +
                     json.dumps({"text": "task two answer two"}) + "\n", encoding="utf-8")
    validation.write_text(json.dumps({"text": "validation answer zero"}) + "\n", encoding="utf-8")
    replay.write_text("\n".join(json.dumps(item) for item in [
        {"text": "validation answer zero"}, {"text": "general three answer three"},
        {"text": "general four answer four"},
    ]), encoding="utf-8")
    output = tmp_path / "trained"
    controls = {
        "model_path": str(base), "dataset_path": str(train), "output_path": str(output),
        "validation_dataset_path": str(validation), "replay_data_path": str(replay), "replay_count": 1,
        "max_steps": 1, "batch_size": 1, "warmup_steps": 0, "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False, "load_best_model_at_end": False, "save_steps": 100,
        "retention_kl_weight": retention_weight, "retention_temperature": 2.0,
        "reference_model_path": str(tmp_path / "must-not-load") if not retention_weight else None,
    }
    if backend == "saddle":
        from saddlellm.training.NativePostTraining import (
            NativeSFTConfig,
            train_native_sft,
        )

        result = train_native_sft(NativeSFTConfig(**controls, max_length=24))
    else:
        from saddlellm.training.PeftSFTTrainer import train_model

        result = train_model(**controls, use_lora=False, use_qlora=False,
                             response_template=None, max_seq_length=24, local_files_only=True)
    assert result["global_step"] == 1
    assert result["train_samples"] == 3
    assert result["eval_samples"] == 1
    assert result["replay"]["excluded_validation_records"] == 1
    assert result["replay"]["replay_samples"] == 1
    assert json.loads((output / "replay_manifest.json").read_text()) == result["replay"]
    if retention_weight:
        assert result["retention"]["enabled"] is True
        assert result["retention"]["loss_calls"] == 1
        assert math.isfinite(result["retention"]["mean_kl"])
        assert json.loads((output / "retention_manifest.json").read_text()) == result["retention"]
    else:
        assert result["retention"] == {"enabled": False}
        assert not (output / "retention_manifest.json").exists()
