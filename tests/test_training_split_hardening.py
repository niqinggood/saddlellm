"""Runtime checks must block leakage before any optimizer update."""

import json

import pytest

from saddlellm.data.TrainingSplitAudit import audit_record_splits


def test_equivalent_chat_and_prompt_completion_cannot_cross_holdout(tmp_path):
    train = [{"prompt": "answer this", "completion": "first"}]
    validation = [{"messages": [{"role": "user", "content": "answer  this"},
                                {"role": "assistant", "content": "different"}]}]
    path = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="leakage"):
        audit_record_splits(train, validation, report_path=path)
    report = json.loads(path.read_text())
    assert not report["passed"] and report["source_overlap"] > 0
    assert "answer this" not in path.read_text()


def test_explicit_domain_groups_are_audited_even_when_prompts_differ():
    with pytest.raises(ValueError, match="leakage"):
        audit_record_splits([{"text": "one", "metadata": {"case_id": "case7"}}],
                            [{"text": "two", "case_id": "case7"}])


@pytest.mark.parametrize("field", ["patient_id", "document_id", "source_record_id"])
def test_group_policy_is_shared_by_split_replay_and_audit(field):
    from saddlellm.data.GroupSplit import source_record_groups
    from saddlellm.data.ReplayData import filter_replay_records, sft_source_groups

    heldout = {"text": "held out context", field: "ID-42"}
    replay = {"text": "different text", "metadata": {field: "id42"}}
    unrelated = {"text": "third context", field: "ID-43"}
    groups = source_record_groups([heldout, replay, unrelated])
    assert groups[0] == groups[1] != groups[2]
    groups = sft_source_groups([heldout, replay, unrelated])
    assert groups[0] == groups[1] != groups[2]
    selected, report = filter_replay_records([replay, unrelated], [heldout])
    assert selected == [unrelated] and report["excluded_validation_records"] == 1


def test_tokenization_truncation_collision_ignores_padding():
    with pytest.raises(ValueError, match="prepared token"):
        audit_record_splits([], [],
            train_features=[{"input_ids": [4, 8, 0], "attention_mask": [1, 1, 0]}],
            validation_features=[{"input_ids": [4, 8]}])


def test_zero_group_id_is_real_and_boolean_is_not_a_group():
    from saddlellm.data.GroupSplit import source_record_groups

    assert source_record_groups([{"case_id": 0}, {"case_id": "0"}]) == [0, 0]
    assert source_record_groups([{"case_id": False}, {"case_id": False}]) == [0, 1]


def test_audit_report_cannot_overwrite_source(tmp_path):
    source = tmp_path / "records.json"
    source.write_text("original")
    with pytest.raises(ValueError, match="overwrite"):
        audit_record_splits([], [], report_path=source, protected_paths=[source])
    assert source.read_text() == "original"


def test_trl_separate_preference_answers_do_not_create_false_leakage():
    report = audit_record_splits([], [], task="dpo",
        train_features=[{"prompt_input_ids": [1, 2], "chosen_input_ids": [5], "rejected_input_ids": [6]}],
        validation_features=[{"prompt_input_ids": [3, 4], "chosen_input_ids": [5], "rejected_input_ids": [6]}])
    assert report["passed"]


def test_same_prepared_prompt_with_different_answers_is_not_new_prompt_validation():
    with pytest.raises(ValueError, match="prepared token"):
        audit_record_splits([], [], task="dpo",
            train_features=[{"chosen_input_ids": [1, 2, 5], "chosen_labels": [-100, -100, 5]}],
            validation_features=[{"prompt_input_ids": [1, 2], "chosen_input_ids": [6]}])


@pytest.mark.parametrize("mask", ["completion_mask", "assistant_masks"])
def test_trl_pre_collation_masks_detect_truncated_prompt_collision(mask):
    with pytest.raises(ValueError, match="prepared token"):
        audit_record_splits([], [],
            train_features=[{"input_ids": [9, 8, 1], mask: [0, 0, 1]}],
            validation_features=[{"input_ids": [9, 8, 2], mask: [0, 0, 1]}])


@pytest.mark.parametrize("task,prefix", [("kto", "completion_"), ("orpo", "chosen_")])
def test_full_preference_inputs_do_not_get_prompt_prepended_twice(task, prefix):
    with pytest.raises(ValueError, match="prepared token"):
        audit_record_splits([], [], task=task,
            train_features=[{"prompt_input_ids": [1], prefix + "input_ids": [1, 2, 3],
                             prefix + "labels": [-100, 2, 3]}],
            validation_features=[{"prompt_input_ids": [1, 2], prefix + "input_ids": [1, 2, 3],
                                  prefix + "labels": [-100, -100, 3]}])


def test_preference_split_preserves_prompts_and_case_metadata(monkeypatch, tmp_path):
    from datasets import Dataset

    from saddlellm.data import TrainingSplitAudit
    from saddlellm.models import TokenizerLoader
    from saddlellm.training import Preference as preference

    records = [{"prompt": "same", "chosen": "a", "rejected": "b", "case_id": "A"},
               {"prompt": "same", "chosen": "c", "rejected": "d", "case_id": "B"},
               {"prompt": "linked", "chosen": "a", "rejected": "b", "case_id": "B"},
               {"prompt": "held out", "chosen": "a", "rejected": "b", "case_id": "C"}]
    monkeypatch.setattr(preference, "_device", lambda: "cpu")
    monkeypatch.setattr(preference, "_load_dataset", lambda _: Dataset.from_list(records))
    monkeypatch.setattr(TokenizerLoader, "load_tokenizer_compatible", lambda *a, **k: type("T", (), {"pad_token": "pad"})())
    class ReachedAudit(Exception):
        pass
    def capture(train, validation, **kwargs):
        train, validation = list(train), list(validation)
        assert {r["prompt"] for r in train}.isdisjoint({r["prompt"] for r in validation})
        assert {r["case_id"] for r in train}.isdisjoint({r["case_id"] for r in validation})
        assert sorted([len(train), len(validation)]) == [1, 3]
        raise ReachedAudit
    monkeypatch.setattr(TrainingSplitAudit, "audit_record_splits", capture)
    config = preference.PreferenceTrainConfig("unused", "records", str(tmp_path), validation_split=.5,
                                               use_lora=False, use_qlora=False)
    with pytest.raises(ReachedAudit):
        preference.train_preference(config)


def test_native_explicit_leakage_blocks_model_load(monkeypatch, tmp_path):
    from saddlellm.training import NativePostTraining as native

    train, validation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    for path, answer in ((train, "one"), (validation, "two")):
        path.write_text(json.dumps({"prompt": "same context", "completion": answer}) + "\n")
    monkeypatch.setattr(native, "_activate_local_device", lambda: None)
    monkeypatch.setattr(native.SaddleModelAdapter, "load_tokenizer", lambda *a, **k:
                        type("T", (), {"pad_token_id": 0, "eos_token_id": 1})())
    monkeypatch.setattr(native, "_sft_features", lambda *a: [{"input_ids": [1, 2], "labels": [-100, 2]}])
    def forbidden(*a, **k):
        pytest.fail("leaky data reached model loading")
    monkeypatch.setattr(native.SaddleModelAdapter, "load_model", forbidden)
    with pytest.raises(ValueError, match="leakage"):
        native.train_native_sft(native.NativeSFTConfig("unused", str(train), str(tmp_path / "out"),
                                                       validation_dataset_path=str(validation)))
