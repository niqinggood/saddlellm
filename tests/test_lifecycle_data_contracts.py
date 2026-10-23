"""Data-contract regressions with fake trainers; no model training is executed."""

import json
import tempfile
from types import SimpleNamespace

import pytest

from saddlellm.data.PostTrainingData import PostTrainingDataAdapter
from saddlellm.data.TrainingDataInspector import TrainingDataInspector


def kto_record(label):
    return {"prompt": "A question", "completion": "An answer", "label": label}


@pytest.mark.parametrize("label", [" true ", " YES ", " accepted "])
def test_kto_label_whitespace_agrees_between_inspector_and_normalizer(label):
    row = kto_record(label)
    assert TrainingDataInspector._coerce_binary_label(row) is True
    assert PostTrainingDataAdapter.normalize_record(row, task="kto")["label"] is True


@pytest.mark.parametrize("label", [None, "unknown", "", 2, -1, float("nan"), {}, []])
def test_invalid_kto_labels_report_the_record_instead_of_becoming_false(label):
    with pytest.raises(ValueError, match=r"[Rr](?:ecord|ow) 2.*(?:label|Label)"):
        PostTrainingDataAdapter.normalize_records([kto_record(True), kto_record(label)], task="kto")


def test_completion_only_kto_cannot_invent_a_missing_negative_label():
    with pytest.raises(ValueError, match="label"):
        PostTrainingDataAdapter.normalize_record(
            {"prompt": "A question", "completion": "An answer"}, task="kto"
        )


@pytest.mark.parametrize("label, expected", [
    (True, True), (False, False), (0, False), (1, True), (0.0, False), (1.0, True),
    ("TRUE", True), (" false ", False), (" No ", False), ("negative", False),
    ("rejected", False), (" POSITIVE ", True),
])
def test_valid_binary_kto_aliases_keep_their_meaning(label, expected):
    row = kto_record(label)
    assert TrainingDataInspector._coerce_binary_label(row) is expected
    assert PostTrainingDataAdapter.normalize_record(row, task="kto")["label"] is expected


@pytest.mark.parametrize("field, expected", [("chosen", True), ("rejected", False)])
def test_legacy_one_sided_kto_records_have_unambiguous_labels(field, expected):
    row = {"prompt": "A question", field: "A completion"}
    assert TrainingDataInspector._auto_post_training_task(row) == "kto"
    assert TrainingDataInspector._coerce_binary_label(row) is expected
    assert PostTrainingDataAdapter.normalize_record(row, task="kto")["label"] is expected


def test_kto_score_auto_selection_and_invalid_row_preflight_are_consistent(tmp_path):
    path = tmp_path / "scored.jsonl"
    rows = [{"prompt": "One", "completion": "A", "score": 1},
            {"prompt": "Two", "completion": "B", "score": 0}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert TrainingDataInspector.select_post_training(str(path)).selected_stage == "kto"
    rows[1]["score"] = "unknown"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = TrainingDataInspector.inspect_file(str(path), task="kto")
    assert not report.ready
    assert any("Record 2" in error and "label" in error for error in report.errors)
    with pytest.raises(ValueError, match="Record 2.*label"):
        TrainingDataInspector.select_post_training(str(path))


def test_explicit_group_ids_survive_normalization_without_treating_source_as_a_group():
    from saddlellm.data.GroupSplit import source_record_groups

    rows = [sft_record("A"), sft_record("A"), sft_record("B")]
    normalized = [PostTrainingDataAdapter.normalize_record(row, task="sft") for row in rows]
    assert source_record_groups(normalized) == [0, 0, 2]
    assert source_record_groups([{"source_id": "corpus"}, {"source_id": "corpus"}]) == [0, 1]
    assert source_record_groups([{"conversation_id": "C"}, {"metadata": {"conversation_id": "C"}}]) == [0, 0]


class SplitCaptured(Exception):
    pass


class ChatTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"

    def apply_chat_template(self, messages, *, add_generation_prompt=False, **kwargs):
        rendered = "".join(f"{item['role']}: {item['content']}\n" for item in messages)
        return rendered + ("assistant: " if add_generation_prompt else "")


def sft_record(case, turn_count=2):
    messages = []
    for turn in range(turn_count):
        messages.extend([
            {"role": "user", "content": f"case={case}; question={turn}"},
            {"role": "assistant", "content": f"case={case}; answer={turn}"},
        ])
    return {"messages": messages, "case_id": case, "source_id": "whole-corpus"}


def capture_hf_sft(monkeypatch, rows, *, seed=42, validation_rows=None):
    from datasets import Dataset

    from saddlellm.training import PeftSFTTrainer as sft

    class FakeSFTConfig:
        def __init__(self, completion_only_loss=None):
            pass

    model = SimpleNamespace()
    model.to = lambda *_: model
    captured = {}
    monkeypatch.setattr(sft, "_device", lambda: "cpu")
    monkeypatch.setattr("saddlellm.models.TokenizerLoader.load_tokenizer_compatible", lambda *a, **k: ChatTokenizer())
    monkeypatch.setattr(sft, "SFTConfig", FakeSFTConfig)
    monkeypatch.setattr(sft, "DataCollatorForCompletionOnlyLM", None)
    monkeypatch.setattr(sft, "_load_sft_dataset", lambda path: Dataset.from_list(validation_rows if path == "validation" else rows))
    monkeypatch.setattr(sft.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(sft, "_make_training_args", lambda *a, **k: object())

    def builder(**kwargs):
        captured.update(kwargs)
        raise SplitCaptured

    monkeypatch.setattr(sft, "_build_sft_trainer", builder)
    with tempfile.TemporaryDirectory(prefix="saddle-hf-split-") as output_path:
        config = sft.SFTTrainConfig(
            model_path="never-loaded", dataset_path="training", output_path=output_path,
            use_lora=False, use_qlora=False, gradient_checkpointing=False,
            validation_split=0.5 if validation_rows is None else 0.0,
            validation_dataset_path="validation" if validation_rows is not None else None,
            seed=seed,
        )
        with pytest.raises(SplitCaptured):
            sft.train_sft(config)
    return captured["train_dataset"], captured["eval_dataset"]


def dataset_cases(dataset):
    return {row["completion"].split("case=", 1)[1].split(";", 1)[0] for row in dataset}


@pytest.mark.parametrize("seed", [1, 42, 2026])
def test_hf_sft_keeps_multi_turn_conversations_and_case_variants_together(monkeypatch, seed):
    rows = [sft_record(case) for case in ("A", "A", "B", "B", "C", "D")]
    train, validation = capture_hf_sft(monkeypatch, rows, seed=seed)
    assert dataset_cases(train).isdisjoint(dataset_cases(validation))
    assert dataset_cases(train) | dataset_cases(validation) == {"A", "B", "C", "D"}
    assert len(train) + len(validation) == 12
    again = capture_hf_sft(monkeypatch, rows, seed=seed)
    assert train.to_list() == again[0].to_list()
    assert validation.to_list() == again[1].to_list()


def test_hf_sft_many_turns_from_one_source_cannot_fake_an_independent_holdout(monkeypatch):
    with pytest.raises(ValueError, match="(?:group|source record|conversation)"):
        capture_hf_sft(monkeypatch, [sft_record("A", turn_count=8)])


def test_hf_sft_without_declared_ids_still_holds_out_whole_source_conversations(monkeypatch):
    rows = [sft_record(case) for case in ("A", "B", "C", "D")]
    for row in rows:
        row.pop("case_id")
    train, validation = capture_hf_sft(monkeypatch, rows)
    assert dataset_cases(train).isdisjoint(dataset_cases(validation))
    assert len(train) == len(validation) == 4


def test_hf_explicit_validation_remains_the_users_own_assignment(monkeypatch):
    train, validation = capture_hf_sft(
        monkeypatch, [sft_record("training", 2)],
        validation_rows=[sft_record("heldout", 3)],
    )
    assert len(train) == 2
    assert len(validation) == 3


def setup_native_dpo(monkeypatch, tmp_path, rows):
    from saddlellm.training import NativePostTraining as native

    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(native, "_activate_local_device", lambda: None)
    monkeypatch.setattr(native.SaddleModelAdapter, "load_tokenizer", lambda *a, **k: ChatTokenizer())
    monkeypatch.setattr(native, "_preference_features", lambda record, *a: None if record.get("invalid") else {"marker": record["id"]})
    monkeypatch.setattr(native.SaddleModelAdapter, "load_model", lambda *a, **k: (_ for _ in ()).throw(SplitCaptured()))
    return native, native.NativeDPOConfig(
        model_path="never-loaded", dataset_path=str(path),
        output_path=str(tmp_path / "never-written"), validation_split=0.5,
    )


def preference_row(index, prompt, case, **extra):
    return {"id": index, "prompt": prompt, "case_id": case,
            "chosen": "Good", "rejected": "Bad", "source_id": "whole-corpus", **extra}


def test_native_dpo_groups_same_prompt_and_cases_after_filtering(monkeypatch, tmp_path):
    rows = [
        preference_row(0, "Same question", "A"),
        preference_row(1, "Other", "A", invalid=True),
        preference_row(2, " Same   question ", "B"),
        preference_row(3, "Different wording", "B"),
        preference_row(4, "Independent C", "C"),
        preference_row(5, "Independent D", "D"),
    ]
    native, config = setup_native_dpo(monkeypatch, tmp_path, rows)
    original = native._split_features

    def grouped_split(features, fraction, seed, *, groups=None):
        assert groups is not None
        assert len(groups) == len(features) == 5
        assert groups[0] == groups[1] == groups[2]
        assert len(set(groups)) == 3
        train, validation = original(features, fraction, seed, groups=groups)
        side = {row["marker"]: "train" for row in train}
        side.update({row["marker"]: "validation" for row in validation})
        assert side[0] == side[2] == side[3]
        assert 1 not in side
        return train, validation

    monkeypatch.setattr(native, "_split_features", grouped_split)
    with pytest.raises(SplitCaptured):
        native.train_native_dpo(config)


def test_native_dpo_one_prompt_is_not_an_independent_holdout(monkeypatch, tmp_path):
    rows = [preference_row(index, "Only one prompt", str(index)) for index in range(4)]
    native, config = setup_native_dpo(monkeypatch, tmp_path, rows)
    with pytest.raises(ValueError, match="(?:group|source record|prompt)"):
        native.train_native_dpo(config)


def test_native_dpo_explicit_validation_is_not_regrouped(monkeypatch, tmp_path):
    rows = [preference_row(index, "Same prompt", "Same case") for index in range(2)]
    native, config = setup_native_dpo(monkeypatch, tmp_path, rows)
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text(
        json.dumps(preference_row(2, "Held-out prompt", "Held-out case")) + "\n",
        encoding="utf-8",
    )
    config.validation_split = 0.0
    config.validation_dataset_path = str(validation_path)

    def unexpected_split(*args, **kwargs):
        pytest.fail("explicit validation assignment was regrouped")

    monkeypatch.setattr(native, "_split_features", unexpected_split)
    with pytest.raises(SplitCaptured):
        native.train_native_dpo(config)


def test_native_sft_declared_cases_keep_related_source_records_together(monkeypatch, tmp_path):
    from saddlellm.training import NativePostTraining as native

    path = tmp_path / "conversation-variants.jsonl"
    rows = [{"id": 0, "case_id": "A"}, {"id": 1, "case_id": "A"}, {"id": 2, "case_id": "B"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(native, "_activate_local_device", lambda: None)
    monkeypatch.setattr(native.SaddleModelAdapter, "load_tokenizer", lambda *a, **k: ChatTokenizer())
    monkeypatch.setattr(native, "_sft_features", lambda record, *a: [{"marker": record["id"]}])

    def capture(features, fraction, seed, *, groups):
        assert groups == [0, 0, 2]
        raise SplitCaptured

    monkeypatch.setattr(native, "_split_features", capture)
    with pytest.raises(SplitCaptured):
        native.train_native_sft(native.NativeSFTConfig(
            model_path="never-loaded", dataset_path=str(path),
            output_path=str(tmp_path / "never-written"), validation_split=0.5,
        ))
