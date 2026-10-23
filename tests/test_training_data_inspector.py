from __future__ import annotations

import json

from saddlellm.data.TrainingDataInspector import (
    audit_training_splits,
    inspect_training_data,
)
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_inspector_warns_about_concentrated_sft_templates(tmp_path):
    path = tmp_path / "templated.jsonl"
    rows = [
        {
            "instruction": (
                "请回答下面的护理学单项选择题。\n\n"
                f"题目：第{i}题是什么？\n选项：\nA. 甲\nB. 乙"
            ),
            "output": f"{'A' if i % 2 else 'B'}\n正确选项：{'甲' if i % 2 else '乙'}",
        }
        for i in range(24)
    ]
    _write_jsonl(path, rows)

    result = inspect_training_data(str(path), task="sft", max_records=100)

    assert result["ready"] is True
    assert result["severity"] == "warning"
    assert result["unique_prompt_prefixes"] == 1
    assert result["dominant_prompt_prefix_ratio"] == 1.0
    assert result["unique_completion_shapes"] == 1
    assert result["dominant_completion_shape_ratio"] == 1.0
    assert any("template overfitting" in warning for warning in result["warnings"])
    assert any("same data split" in recommendation for recommendation in result["recommendations"])


def test_inspector_does_not_warn_for_diverse_natural_prompts(tmp_path):
    path = tmp_path / "diverse.jsonl"
    rows = [
        {
            "instruction": f"患者场景{i}下应当如何进行护理评估？",
            "output": f"先核对患者身份与风险因素，再完成第{i}项评估。",
        }
        for i in range(24)
    ]
    _write_jsonl(path, rows)

    result = inspect_training_data(str(path), task="sft", max_records=100)

    assert result["ready"] is True
    assert result["severity"] == "ok"
    assert result["unique_prompt_prefixes"] == 24
    assert result["dominant_prompt_prefix_ratio"] < 0.8
    assert not any("template overfitting" in warning for warning in result["warnings"])


def test_inspector_ignores_shared_system_message_when_measuring_prompt_diversity(tmp_path):
    path = tmp_path / "conversational.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "你是一名严谨的护理教学助手。"},
                {"role": "user", "content": f"患者场景{i}下首先应评估什么？"},
                {"role": "assistant", "content": f"首先核对第{i}项生命体征与风险因素。"},
            ]
        }
        for i in range(24)
    ]
    _write_jsonl(path, rows)

    result = inspect_training_data(str(path), task="sft", max_records=100)

    assert result["ready"] is True
    assert result["severity"] == "ok"
    assert result["unique_prompt_prefixes"] == 24
    assert not any("template overfitting" in warning for warning in result["warnings"])


def test_split_audit_blocks_group_and_exact_prompt_overlap(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(
        train,
        [
            {"case_id": "case-shared", "question": "患者血压骤降时先做什么？", "output": "评估"},
            {"case_id": "case-exact-left", "question": "给药前如何核对？", "output": "核对"},
        ],
    )
    _write_jsonl(
        validation,
        [
            {"case_id": "case-shared", "question": "同一病例的另一种问法是什么？", "output": "评估"},
            {"case_id": "case-exact-right", "question": "给药前如何核对？", "output": "核对"},
        ],
    )

    result = audit_training_splits(str(train), str(validation))

    assert result["passed"] is False
    pair = result["pairs"]["train_validation"]
    assert pair["group_overlap"] == 1
    assert pair["exact_content_overlap"] == 1
    assert pair["blocking_overlap"] == 2
    assert "preview" not in pair["exact_content_examples"][0]
    assert pair["group_examples"][0].startswith("case_id:sha256:")


def test_split_audit_warns_for_near_duplicates_unless_strict(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    common = "患者出现呼吸困难、口唇发绀和血氧下降，护士应立即如何评估并处理"
    _write_jsonl(train, [{"case_id": "left", "question": common + "甲", "output": "处理"}])
    _write_jsonl(validation, [{"case_id": "right", "question": common + "乙", "output": "处理"}])

    advisory = audit_training_splits(str(train), str(validation))
    strict = audit_training_splits(str(train), str(validation), strict=True)

    assert advisory["passed"] is True
    assert advisory["pairs"]["train_validation"]["near_duplicate_overlap"] == 1
    assert any("near-duplicate" in warning for warning in advisory["warnings"])
    assert strict["passed"] is False
    assert strict["pairs"]["train_validation"]["blocking_overlap"] == 1


def test_training_preflight_automatically_blocks_validation_leakage(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    row = {"case_id": "same-case", "instruction": "同一问题", "output": "同一回答"}
    _write_jsonl(train, [row])
    _write_jsonl(validation, [row])

    report = TrainingConfigValidator.validate(
        {
            "stages": ["sft"],
            "model": {"name_or_path": "unused-model"},
            "sft": {
                "enabled": True,
                "data_path": str(train),
                "validation_data_path": str(validation),
            },
            "distributed": {"bf16": False, "fp16": False},
        }
    )

    assert report.valid is False
    assert report.data_inspections["sft_split_leakage"]["passed"] is False
    assert any("shares 1 configured group" in issue for issue in report.issues)
