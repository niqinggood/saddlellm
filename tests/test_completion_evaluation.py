from pathlib import Path

import pytest

from saddlellm.evaluation.CompletionEvaluator import (
    CompletionEvaluationRequest,
    _render_pair,
    completion_markdown,
    evaluate_completions,
)


def test_render_pair_uses_only_assistant_suffix_for_completion() -> None:
    class ChatTokenizer:
        eos_token = "<eos>"

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            rendered = "".join(
                f"<{message['role']}>{message['content']}" for message in messages
            )
            if kwargs.get("add_generation_prompt"):
                rendered += "<assistant>"
            return rendered

    prompt, completion = _render_pair(ChatTokenizer(), "question", "answer", False)

    assert prompt == "<user>question<assistant>"
    assert completion == "answer"


def test_completion_evaluator_rejects_missing_reference_before_model_load(
    tmp_path: Path,
) -> None:
    data = tmp_path / "eval.jsonl"
    data.write_text('{"instruction":"question"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'output'"):
        evaluate_completions(
            CompletionEvaluationRequest(model="unused", data=str(data))
        )


def test_completion_markdown_explains_ppl_limit() -> None:
    report = completion_markdown(
        {
            "model": "base",
            "adapter": None,
            "data": "eval.jsonl",
            "samples": 2,
            "completion_tokens": 10,
            "completion_nll": 1.0,
            "completion_perplexity": 2.7182818,
            "generation_samples": 1,
            "elapsed_seconds": 0.5,
            "generations": [
                {
                    "record_id": "case-1",
                    "instruction": "What happened?",
                    "reference": "Reference answer",
                    "generated": "Generated answer",
                }
            ],
        }
    )

    assert "Completion PPL: 2.718282" in report
    assert "does not establish factual correctness" in report
    assert "Sample 1 — record `case-1`" in report
    assert "Generated answer" in report
    assert "Reference answer" in report
