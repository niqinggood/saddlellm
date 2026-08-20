import pytest

from saddlellm.GRPOTrainer import GRPOConfig
from saddlellm.PublicPolicyRLVR import (
    A_LABELS,
    PolicyExample,
    build_user_prompt,
    build_verifiable_reward,
    compute_metrics,
    extract_answer,
    split_for_case,
)


def example_a(answer="qualified_agree"):
    return PolicyExample(
        example_id="A:Q1::org",
        case_key="Q1::org",
        task="A",
        question_id="Q1",
        question="Do you agree?",
        organization="Org",
        response="We support the proposal, subject to implementation guidance.",
        answer=(answer,),
    )


def example_b(answer=("A", "C")):
    return PolicyExample(
        example_id="B:Q2::org",
        case_key="Q2::org",
        task="B",
        question_id="Q2",
        question="Which disclosures are useful?",
        organization="Org",
        response="Governance and reporting alignment are useful.",
        answer=answer,
        candidate_themes={"A": "Governance", "B": "Calculation", "C": "Alignment"},
    )


def test_answer_extractors_are_strict_and_canonical():
    assert extract_answer("<answer>qualified_agree</answer>", "A") == (
        "qualified_agree",
    )
    assert extract_answer("The answer is disagree.", "A") == ("disagree",)
    assert extract_answer("<answer>C, A, A</answer>", "B") == ("A", "C")
    assert extract_answer("<answer>UNCLEAR, A</answer>", "B") == ()


def test_verifiable_reward_keeps_ground_truth_out_of_prompt():
    example = example_a()
    prompt = build_user_prompt(example)
    assert "qualified_agree" in prompt  # It is one allowed label, not a hidden target.
    reward = build_verifiable_reward({prompt: example})
    assert reward(prompt, "<answer>qualified_agree</answer>") == pytest.approx(1.0)
    assert reward(prompt, "<answer>agree</answer>") == pytest.approx(0.1)
    assert reward(prompt, "qualified_agree") == pytest.approx(0.9)
    assert reward(prompt, "not a label") == pytest.approx(0.0)


def test_group_split_is_deterministic_and_case_level():
    first = split_for_case("Q1::org", 42, 0.8, 0.1)
    second = split_for_case("Q1::org", 42, 0.8, 0.1)
    assert first == second
    assert first in {"train", "validation", "test"}


def test_prompt_truncation_preserves_both_evidence_ends_and_instruction():
    example = example_a()
    long_response = "start " + ("middle " * 200) + "finish"
    example = PolicyExample(**{**example.__dict__, "response": long_response})
    prompt = build_user_prompt(example, max_response_chars=300)
    assert "start" in prompt
    assert "finish" in prompt
    assert "evidence truncated" in prompt
    assert "<answer>agree</answer>" in prompt
    assert "never emit <LABEL>" in prompt


def test_metrics_distinguish_pass_and_average():
    examples = [example_a("agree"), example_a("disagree")]
    predictions = [
        [("agree",), ("disagree",)],
        [("agree",), ("disagree",)],
    ]
    metrics = compute_metrics(
        examples,
        predictions,
        format_validity=[[True, False], [True, True]],
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["avg_at_k"] == pytest.approx(0.5)
    assert metrics["pass_at_k"] == pytest.approx(1.0)
    assert metrics["invalid_rate"] == 0.0
    assert metrics["format_valid_rate"] == pytest.approx(0.75)
    assert metrics["schema_violation_rate"] == pytest.approx(0.25)


def test_grpo_config_rejects_single_generation():
    with pytest.raises(ValueError, match="num_generations"):
        GRPOConfig(num_generations=1).validate()
