from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from saddlellm.evaluation.EvaluationComparison import (
    build_evaluation_comparison,
    comparison_markdown,
    exact_two_sided_binomial_p,
    paired_metrics,
)
from saddlellm.evaluation.MultipleChoiceEvaluator import (
    MultipleChoiceEvaluationRequest,
    _load_model,
    _prompt_robustness,
    _render_prompt,
    _select_runtime,
    _validate_records,
    expand_prompt_variants,
    load_evaluation_records,
    load_prompt_variant_templates,
)
from saddlellm.evaluation.ReleaseGate import release_gate_accepted
from saddlellm.models.ModelLoader import load_causal_lm, resolve_model_reference


def _evaluation(predictions, answers, *, data_hash="same"):
    details = []
    for index, (prediction, answer) in enumerate(zip(predictions, answers)):
        details.append(
            {
                "record_id": f"id-{index}",
                "source_index": index,
                "question": f"question-{index}",
                "answer": answer,
                "prediction": prediction,
                "correct": prediction == answer,
            }
        )
    return {
        "model": "model",
        "adapter": None,
        "data_sha256": data_hash,
        "details": details,
        "generations": [],
    }


def _rubric_evaluation(outcomes, scores, critical, *, data_hash="rubric"):
    return {
        "task_type": "rubric",
        "model": "model",
        "adapter": None,
        "data_sha256": data_hash,
        "details": [
            {
                "record_id": f"rubric-{index}",
                "source_index": index,
                "instruction": f"prompt-{index}",
                "rubric_sha256": f"hash-{index}",
                "passed": passed,
                "score": score,
                "critical_failures": ["critical"] if failed_critical else [],
            }
            for index, (passed, score, failed_critical) in enumerate(
                zip(outcomes, scores, critical)
            )
        ],
    }


def _completion_evaluation(nlls, *, data_hash="completion"):
    import math

    mean_nll = sum(nlls) / len(nlls)
    return {
        "task_type": "completion",
        "model": "model",
        "adapter": None,
        "data_sha256": data_hash,
        "samples": len(nlls),
        "completion_tokens": len(nlls) * 2,
        "completion_nll": mean_nll,
        "completion_perplexity": math.exp(mean_nll),
        "details": [
            {
                "record_id": f"completion-{index}",
                "source_index": index,
                "completion_tokens": 2,
                "completion_nll": nll,
            }
            for index, nll in enumerate(nlls)
        ],
        "generations": [],
    }


def test_rubric_comparison_uses_pass_outcome_and_critical_gate() -> None:
    base = _rubric_evaluation(
        [False] * 6,
        [0.4] * 6,
        [True] * 6,
    )
    candidate = _rubric_evaluation(
        [True] * 6,
        [1.0] * 6,
        [False] * 6,
    )

    result = build_evaluation_comparison(
        base_validation=base,
        candidate_validation=candidate,
    )

    assert result["sets"]["validation"]["rubric_pass"]["candidate_accuracy"] == 1.0
    assert result["sets"]["validation"]["rubric_safety"][
        "candidate_critical_failure_count"
    ] == 0
    assert result["selection_eligible"] is True


def test_rubric_comparison_blocks_candidate_critical_failure() -> None:
    base = _rubric_evaluation([False] * 6, [0.4] * 6, [False] * 6)
    candidate = _rubric_evaluation(
        [True] * 6,
        [1.0] * 6,
        [True, False, False, False, False, False],
    )

    result = build_evaluation_comparison(
        base_validation=base,
        candidate_validation=candidate,
    )

    assert result["selection_eligible"] is False


def test_completion_comparison_uses_lower_ppl_and_paired_sign_test() -> None:
    base = _completion_evaluation([2.0] * 6)
    candidate = _completion_evaluation([1.0] * 6)

    result = build_evaluation_comparison(
        base_validation=base,
        candidate_validation=candidate,
        minimum_gain_percentage_points=10.0,
    )
    metrics = result["sets"]["validation"]["completion_perplexity"]

    assert metrics["candidate_completion_perplexity"] < metrics[
        "base_completion_perplexity"
    ]
    assert metrics["relative_perplexity_improvement_percent"] > 60.0
    assert metrics["records_improved"] == 6
    assert metrics["records_regressed"] == 0
    assert metrics["paired_exact_two_sided_sign_p"] == pytest.approx(0.03125)
    assert result["selection_eligible"] is True
    assert result["decision"] == "validation_passed"
    assert result["accepted"] is False
    markdown = comparison_markdown(result)
    assert "Completion perplexity" in markdown
    assert "Sign-test p" in markdown


def test_completion_comparison_rejects_misaligned_rows() -> None:
    base = _completion_evaluation([2.0, 2.0])
    candidate = _completion_evaluation([1.0, 1.0])
    candidate["details"][1]["record_id"] = "different"

    with pytest.raises(ValueError, match="not aligned"):
        build_evaluation_comparison(
            base_validation=base,
            candidate_validation=candidate,
        )


def test_completion_comparison_requires_matching_tokenization() -> None:
    base = _completion_evaluation([2.0, 2.0])
    candidate = _completion_evaluation([1.0, 1.0])
    candidate["details"][1]["completion_tokens"] = 3
    candidate["completion_tokens"] = 5

    with pytest.raises(ValueError, match="token counts differ at row 1"):
        build_evaluation_comparison(
            base_validation=base,
            candidate_validation=candidate,
        )


def test_evaluation_package_exposes_generic_public_api() -> None:
    from saddlellm import evaluate_multiple_choice
    from saddlellm.evaluation import (
        MultipleChoiceEvaluationRequest,
        build_evaluation_comparison,
    )

    assert callable(evaluate_multiple_choice)
    assert callable(build_evaluation_comparison)
    assert MultipleChoiceEvaluationRequest.__name__ == "MultipleChoiceEvaluationRequest"


def test_prompt_rendering_falls_back_for_tokenizer_without_chat_template() -> None:
    class PlainTokenizer:
        @staticmethod
        def apply_chat_template(*args, **kwargs):
            raise ValueError(
                "Cannot use chat template because tokenizer.chat_template is not set"
            )

    assert _render_prompt(PlainTokenizer(), "question", enable_thinking=False) == "question"


def test_load_evaluation_records_supports_jsonl_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"id":"1","question":"q1","answer":"A"}\n'
        '{"id":"2","question":"q2","answer":"B"}\n',
        encoding="utf-8",
    )

    rows = load_evaluation_records(str(path), limit=1)

    assert rows == [{"id": "1", "question": "q1", "answer": "A"}]


def test_prompt_robustness_reports_flips_and_worst_variant() -> None:
    details = [
        {
            "case_id": "case-1",
            "variant": "standard",
            "answer": "A",
            "prediction": "A",
            "correct": True,
        },
        {
            "case_id": "case-1",
            "variant": "bare",
            "answer": "A",
            "prediction": "B",
            "correct": False,
        },
        {
            "case_id": "case-2",
            "variant": "standard",
            "answer": "B",
            "prediction": "B",
            "correct": True,
        },
        {
            "case_id": "case-2",
            "variant": "bare",
            "answer": "B",
            "prediction": "B",
            "correct": True,
        },
    ]

    result = _prompt_robustness(details)

    assert result["cases"] == 2
    assert result["prediction_flip_rate"] == 0.5
    assert result["all_variants_correct_rate"] == 0.5
    assert result["by_variant"]["standard"]["accuracy"] == 1.0
    assert result["by_variant"]["bare"]["accuracy"] == 0.5
    assert result["worst_variant_accuracy"] == 0.5


def test_prompt_variant_records_require_consistent_answer() -> None:
    request = MultipleChoiceEvaluationRequest(
        model="model",
        data="data.jsonl",
        case_field="case_id",
        variant_field="variant",
    )
    rows = [
        {"case_id": "one", "variant": "a", "question": "q", "answer": "A"},
        {"case_id": "one", "variant": "b", "question": "q", "answer": "B"},
    ]

    with pytest.raises(ValueError, match="must share one answer"):
        _validate_records(rows, request)


def test_declarative_prompt_variants_expand_domain_owned_templates(tmp_path) -> None:
    path = tmp_path / "variants.json"
    path.write_text(
        '{"variants":['
        '{"name":"standard","template":"{instruction}"},'
        '{"name":"bare","template":"{question}\\n{options}"}'
        ']}',
        encoding="utf-8",
    )
    variants = load_prompt_variant_templates(str(path))
    rows = expand_prompt_variants(
        [
            {
                "id": "case-1",
                "instruction": "full prompt",
                "question": "question",
                "options": "A. one\nB. two",
                "answer": "A",
            }
        ],
        variants,
        id_field="id",
        prompt_field="instruction",
    )

    assert [row["id"] for row in rows] == [
        "case-1::standard",
        "case-1::bare",
    ]
    assert rows[0]["instruction"] == "full prompt"
    assert rows[1]["instruction"] == "question\nA. one\nB. two"
    assert {row["_saddle_case_id"] for row in rows} == {"case-1"}


def test_prompt_variant_expansion_formats_mapping_options_readably() -> None:
    rows = expand_prompt_variants(
        [{"id": "one", "question": "q", "options": {"A": "one", "B": "two"}}],
        [
            {"name": "one", "template": "{question}\n{options}"},
            {"name": "two", "template": "Choose:\n{options}"},
        ],
        id_field="id",
        prompt_field="instruction",
    )

    assert rows[0]["instruction"] == "q\nA. one\nB. two"


def test_resolve_model_reference_makes_cached_repo_unambiguously_local() -> None:
    cached = r"C:\cache\models--example--model\snapshots\abc"
    with patch("huggingface_hub.snapshot_download", return_value=cached) as download:
        assert resolve_model_reference(
            "example/model", local_files_only=True
        ) == cached
    download.assert_called_once_with(repo_id="example/model", local_files_only=True)


def test_model_loader_forwards_current_dtype_keyword(tmp_path: Path, monkeypatch) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    captured = {}
    sentinel = object()

    def fake_from_pretrained(path, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)

    loaded = load_causal_lm(tmp_path, dtype="float32")

    assert loaded is sentinel
    assert captured["dtype"] is torch.float32
    assert "torch_dtype" not in captured


def test_packaged_quantized_model_uses_canonical_loader(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    from saddlellm.compression.quantizer import ModelQuantizer

    path = tmp_path / "quantized"
    path.mkdir()
    (path / "quantization_manifest.json").write_text(
        json.dumps({"method": "dynamic", "precision": "int8"}),
        encoding="utf-8",
    )
    sentinel = object()
    captured = {}

    def fake_load(model_path, **kwargs):
        captured.update({"model_path": model_path, **kwargs})
        return type("LoadedQuantizer", (), {"model": sentinel})()

    monkeypatch.setattr(ModelQuantizer, "load", fake_load)

    assert load_causal_lm(path, device="cpu", dtype="float32") is sentinel
    assert captured["model_path"] == str(path.resolve())
    assert captured["device_map"] == "cpu"


def test_packaged_quantized_evaluation_auto_selects_cpu(tmp_path: Path) -> None:
    import json
    import torch

    path = tmp_path / "quantized"
    path.mkdir()
    (path / "quantization_manifest.json").write_text(
        json.dumps({"method": "dynamic", "precision": "int8"}),
        encoding="utf-8",
    )

    device, dtype = _select_runtime(
        MultipleChoiceEvaluationRequest(model=str(path), data="data.jsonl")
    )

    assert device == "cpu"
    assert dtype is torch.float32


def test_packaged_quantized_model_rejects_cuda(tmp_path: Path) -> None:
    import json

    path = tmp_path / "quantized"
    path.mkdir()
    (path / "quantization_manifest.json").write_text(
        json.dumps({"method": "dynamic", "precision": "int8"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require device='cpu'"):
        load_causal_lm(path, device="cuda", dtype="float32")


def test_adapter_evaluation_explains_incompatible_post_training_runtime(
    monkeypatch,
) -> None:
    import peft

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

    monkeypatch.setattr(
        "saddlellm.models.ModelLoader.load_model_and_tokenizer",
        lambda *args, **kwargs: (object(), Tokenizer()),
    )
    monkeypatch.setattr(
        "saddlellm.training.PostTrainingCompatibility.stabilize_peft_optional_backends",
        list,
    )
    monkeypatch.setattr(
        "saddlellm.training.PostTrainingCompatibility.post_training_runtime_report",
        lambda: {"issues": ["peft==0.7.1 is outside the tested range"]},
    )
    monkeypatch.setattr(
        "saddlellm.training.PostTrainingCompatibility.format_post_training_runtime_error",
        lambda feature, error: f"{feature}: install the tested stack ({error})",
    )
    monkeypatch.setattr(
        peft.PeftModel,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("unknown config field")),
    )

    with pytest.raises(RuntimeError, match="PEFT adapter evaluation: install the tested stack"):
        _load_model(
            MultipleChoiceEvaluationRequest(
                model="model",
                data="data.jsonl",
                adapter="adapter",
                device="cpu",
                dtype="float32",
            )
        )


def test_paired_metrics_counts_regressions_and_improvements() -> None:
    base = _evaluation(["A", "B", "C"], ["A", "A", "C"])
    candidate = _evaluation(["A", "A", "B"], ["A", "A", "C"])

    metrics = paired_metrics(base, candidate)

    assert metrics["base_wrong_candidate_right"] == 1
    assert metrics["base_right_candidate_wrong"] == 1
    assert metrics["net_gain"] == 0
    assert exact_two_sided_binomial_p(1, 2) == 1.0


def test_comparison_preserves_prompt_robustness_evidence() -> None:
    base = _evaluation(["A", "B", "B", "B"], ["A", "A", "B", "B"])
    candidate = _evaluation(["A", "A", "B", "B"], ["A", "A", "B", "B"])
    for result, flip_rate, all_correct, worst, case_flags in (
        (base, 0.5, 0.5, 0.5, [False, True]),
        (candidate, 0.0, 1.0, 1.0, [True, True]),
    ):
        result["prompt_variant_templates"] = {"sha256": "same-templates"}
        result["prompt_robustness"] = {
            "prediction_flip_rate": flip_rate,
            "all_variants_correct_rate": all_correct,
            "worst_variant_accuracy": worst,
            "by_variant": {
                "bare": {"accuracy": worst},
                "standard": {"accuracy": 1.0},
            },
            "case_details": [
                {"case_id": f"case-{index}", "all_variants_correct": flag}
                for index, flag in enumerate(case_flags)
            ],
        }

    result = build_evaluation_comparison(
        base_validation=base,
        candidate_validation=candidate,
        maximum_p_exclusive=1.1,
    )
    robustness = result["sets"]["validation"]["prompt_robustness"]

    assert robustness["all_variants_correct_gain_percentage_points"] == 50.0
    assert robustness["prediction_flip_reduction_percentage_points"] == 50.0
    assert robustness["worst_variant_accuracy_gain_percentage_points"] == 50.0
    assert robustness["all_variants_base_wrong_candidate_right"] == 1
    assert result["selection_eligible"] is True
    assert result["release_evidence_complete"] is False
    assert result["decision"] == "validation_passed"
    assert result["accepted"] is False
    assert not release_gate_accepted(result)


def test_comparison_requires_every_set_and_zero_manifest_overlap() -> None:
    base = _evaluation(["B", "B", "B", "B"], ["A", "A", "A", "A"])
    candidate = _evaluation(["A", "A", "A", "A"], ["A", "A", "A", "A"])
    result = build_evaluation_comparison(
        base_validation=base,
        candidate_validation=candidate,
        base_test=base,
        candidate_test=candidate,
        minimum_gain_percentage_points=5,
        maximum_p_exclusive=0.2,
        manifest={
            "exact_question_overlap_after_exclusion": {
                "train_validation": 0,
                "train_test": 0,
            }
        },
    )

    assert result["decision"] == "promoted"
    assert result["accepted"] is True
    assert result["criteria"]["set_gate"] == {
        "validation": True,
        "test": True,
    }


def test_release_gate_acceptance_reader_supports_both_result_schemas() -> None:
    assert release_gate_accepted({"accepted": True})
    assert release_gate_accepted({"decision": "promoted"})
    assert release_gate_accepted({"gate": {"status": "accepted"}})
    assert not release_gate_accepted({"decision": "not_promoted"})
    assert not release_gate_accepted({"enabled": False, "accepted": True})
    assert not release_gate_accepted({"criteria": {"set_gate": {"test": True}}})


def test_comparison_rejects_misaligned_data_hashes() -> None:
    with pytest.raises(ValueError, match="hashes differ"):
        build_evaluation_comparison(
            base_validation=_evaluation(["A"], ["A"], data_hash="base"),
            candidate_validation=_evaluation(["A"], ["A"], data_hash="candidate"),
        )
