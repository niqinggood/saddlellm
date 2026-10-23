"""Deterministic regression coverage for legacy public correctness failures."""

import json
import math
import sys
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from saddlellm.data.DataBalance import BalancedPretrainer
from saddlellm.data.pipeline import DataPipeline, PipelineConfig
from saddlellm.evaluation.BenchmarkRunner import BenchmarkRunner
from saddlellm.experiments.ScalingLawAnalyzer import ScalingLawAnalyzer


class CharacterTokenizer:
    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(char) for char in text]
        if return_tensors == "pt":
            ids = torch.tensor([ids])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        return "".join(chr(int(token)) for token in ids)


class GenerationModel(torch.nn.Module):
    def __init__(self, completions=()):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.completions = iter(completions)
        self.calls = []

    @property
    def device(self):
        return self.weight.device

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        tail = torch.tensor([[ord(char) for char in next(self.completions)]])
        return torch.cat((input_ids, tail), dim=1)


def runner_for(task, completions=(), **kwargs):
    runner = BenchmarkRunner("tiny-local", tasks=[task], **kwargs)
    runner._model = GenerationModel(completions)
    runner._tokenizer = CharacterTokenizer()
    return runner


def fake_datasets(monkeypatch, loader):
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=loader))


@pytest.mark.parametrize("task,error", [("made-up", ValueError), ("humaneval", NotImplementedError)])
def test_unknown_and_unimplemented_tasks_fail_before_model_loading(task, error):
    with pytest.raises(error):
        BenchmarkRunner("must-not-download", tasks=[task])


@pytest.mark.parametrize("kwargs", [{"max_samples": 0}, {"max_samples": -1}, {"max_new_tokens": 0}, {"batch_size": True}])
def test_evaluation_rejects_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        BenchmarkRunner("unused", **kwargs)


def test_gsm8k_generates_and_scores_numeric_answers_not_perplexity(monkeypatch):
    rows = [
        {"question": "Compute 1,200", "answer": "reasoning\n#### 1200"},
        {"question": "Compute half", "answer": "reasoning\n#### 0.5"},
        {"question": "Compute -2.5", "answer": "reasoning\n#### -2.5"},
        {"question": "Malformed reference", "answer": "no final answer"},
    ]
    loads = []
    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return rows
    fake_datasets(monkeypatch, load)
    runner = runner_for("gsm8k", ["Work\n#### 1,200", "\\boxed{1/2}", "I considered -2.5 and 8."])
    monkeypatch.setattr(runner, "_eval_perplexity", lambda: pytest.fail("GSM8K must not use perplexity"))

    result = runner.run()[0]

    assert result.metric == "accuracy"
    assert result.score == pytest.approx(2 / 3)
    assert result.num_samples == 3
    assert result.metadata["unparsed_predictions"] == 1
    assert result.metadata["skipped"] == 1
    assert all(call["do_sample"] is False for call in runner._model.calls)
    assert loads == [(("openai/gsm8k", "main"), {"split": "test", "streaming": True})]


@pytest.mark.parametrize("text,expected", [
    ("#### -2.50", Fraction(-5, 2)),
    ("The final answer is 42.", Fraction(42)),
    ("#### 1/0", None),
    ("The question contains 42 numbers", None),
    ("I tried 42 and 7.", None),
])
def test_gsm8k_final_answer_extraction(text, expected):
    assert BenchmarkRunner._extract_gsm8k_answer(text) == expected


def test_evaluation_failure_is_saved_instead_of_silently_dropped(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise OSError("offline fixture")
    fake_datasets(monkeypatch, broken)
    runner = runner_for("gsm8k")
    result = runner.run()[0]
    assert result.status == "failed" and result.score is None
    assert result.num_samples == 0
    assert "offline fixture" in result.error
    path = tmp_path / "results.json"
    runner.save_results([result], str(path))
    assert json.loads(path.read_text(encoding="utf-8"))[0]["score"] is None
    assert runner.load_results(str(path))[0] == result
    assert "FAILED" in runner.compare([str(path)])


def test_in_memory_model_does_not_require_a_model_path(monkeypatch):
    fake_datasets(monkeypatch, lambda *args, **kwargs: [{"question": "Q", "answer": "#### 42"}])
    runner = runner_for("gsm8k", ["#### 42"])
    runner.model_path = None
    result = runner.run()[0]
    assert result.score == 1 and result.model_name == "GenerationModel"


def test_mmlu_reports_subset_failures_and_actual_micro_denominator(monkeypatch):
    right = {"question": "Q", "choices": ["correct", "wrong"], "answer": 0}
    wrong = {**right, "answer": 1}
    def load(name, subject, **kwargs):
        if subject == "unavailable":
            raise OSError("subject unavailable")
        return [right] if subject == "one" else [wrong, wrong]
    fake_datasets(monkeypatch, load)
    runner = runner_for("mmlu", max_samples=10, mmlu_subjects=["one", "two", "unavailable"])
    monkeypatch.setattr(runner, "_choice_loss", lambda context, choice: 0 if choice == "correct" else 1)

    result = runner.run()[0]

    assert result.status == "partial"
    assert result.score == pytest.approx(1 / 3)
    assert result.num_samples == 3
    assert result.metadata["aggregation"] == "micro_accuracy"
    assert result.metadata["subjects"]["one"]["num_samples"] == 1
    assert "unavailable" in result.metadata["failures"]
    assert result.metadata["requested_subjects"] == ["one", "two", "unavailable"]


@pytest.mark.parametrize("row,expected", [
    ({"ctx": "Q", "endings": ["a", "b"], "label": "1"}, 1),
    ({"question": "Q", "choices": {"label": ["1", "2"], "text": ["a", "b"]}, "answerKey": "2"}, 1),
    ({"goal": "Q", "sol1": "a", "sol2": "b", "label": 1}, 1),
    ({"passage": "P", "question": "Q", "answer": True}, 1),
    ({"sentence": "The _ runs", "option1": "a", "option2": "b", "answer": "2"}, 1),
])
def test_supported_multiple_choice_schemas(row, expected):
    _, choices, label = BenchmarkRunner._multiple_choice_example(row)
    assert label == expected and len(choices) == 2


def test_choice_loss_masks_context_and_preserves_all_choices(monkeypatch):
    runner = runner_for("mmlu")
    observed = []
    def forward(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(loss=torch.tensor(0.75))
    monkeypatch.setattr(runner._model, "forward", forward)
    assert runner._choice_loss("long question", "answer") == 0.75
    labels = observed[0]["labels"]
    assert (labels[0, :len("long question")] == -100).all()
    assert (labels[0, len("long question"):] != -100).all()

    scored = []
    monkeypatch.setattr(runner, "_choice_loss", lambda context, choice: scored.append(choice) or (0 if choice == "e" else 1))
    score = runner._eval_hf_multiple_choice_ds([{"question": "Q", "choices": list("abcde"), "answer": 4}])
    assert scored == list("abcde") and score == 1


def test_all_invalid_examples_fail_instead_of_reporting_zero_accuracy(monkeypatch):
    fake_datasets(monkeypatch, lambda *args, **kwargs: [{"unrelated": "data"}])
    result = runner_for("hellaswag").run()[0]
    assert result.status == "failed" and result.score is None
    assert result.num_samples == 0 and result.metadata["skipped"] == 1


def test_perplexity_uses_actual_shifted_token_counts_and_actual_samples(monkeypatch):
    rows = [{"text": ""}, {"text": "a" * 10}, {"text": "b" * 20}]
    fake_datasets(monkeypatch, lambda *args, **kwargs: rows)
    runner = runner_for("perplexity", max_samples=50)
    def forward(input_ids, **kwargs):
        return SimpleNamespace(loss=torch.tensor(1.0 if input_ids.shape[1] == 10 else 3.0))
    monkeypatch.setattr(runner._model, "forward", forward)
    result = runner.run()[0]
    assert result.num_samples == 2
    assert result.metadata["num_scored_tokens"] == 28
    assert result.score == pytest.approx(math.exp((9 * 1 + 19 * 3) / 28))


def test_lambada_scores_generated_held_out_word(monkeypatch):
    fake_datasets(monkeypatch, lambda *args, **kwargs: [{"text": "The answer is blue"}, {"text": "The answer is red"}])
    result = runner_for("lambada", [" blue", "blue"]).run()[0]
    assert result.score == 0.5 and result.num_samples == 2
    assert result.metadata["scoring"] == "last-word greedy exact match"


class MaterializedRows:
    def __init__(self, texts):
        self.rows = [{"text": text} for text in texts]
    @property
    def num_rows(self):
        return len(self.rows)
    def filter(self, predicate):
        self.rows = [row for row in self.rows if predicate(row)]
        return self


def test_exact_dedup_hashes_full_normalized_text_not_a_shared_prefix():
    first = "a" * 2500 + " first ending"
    second = "a" * 2500 + " second ending"
    pipeline = DataPipeline(PipelineConfig(dedup_method="exact"))
    pipeline._dataset = MaterializedRows([first, second, "\n" + first.upper() + "\t", "Café", "Cafe\u0301"])
    pipeline.deduplicate()
    assert [row["text"] for row in pipeline._dataset.rows] == [first, second, "Café"]
    assert pipeline.stats["dedup_effective_method"] == "exact"
    assert pipeline.stats["deduplicated_samples"] == 3


@pytest.mark.parametrize("method,module", [("simhash", "simhash"), ("minhash", "datasketch")])
def test_missing_approximate_dedup_dependencies_report_exact_fallback(monkeypatch, method, module):
    monkeypatch.setitem(sys.modules, module, None)
    pipeline = DataPipeline(PipelineConfig(dedup_method=method))
    pipeline._dataset = MaterializedRows(["same text", "same text", "different text"])
    pipeline.deduplicate()
    assert pipeline._dataset.num_rows == 2
    assert pipeline.stats["dedup_requested_method"] == method
    assert pipeline.stats["dedup_effective_method"] == "exact"
    assert module in pipeline.stats["dedup_fallback_reason"]


def test_legacy_balancing_cannot_simulate_training_or_empty_checkpoints(tmp_path):
    with pytest.raises(NotImplementedError, match="unsupported legacy prototype"):
        BalancedPretrainer(SimpleNamespace(tokenizer_path="must-not-download", output_dir=str(tmp_path)))
    bypassed = object.__new__(BalancedPretrainer)
    for method, args in [(bypassed.train, ()), (bypassed._train_step, ({}, "domain")), (bypassed._save_checkpoint, (0,))]:
        with pytest.raises(NotImplementedError, match="no model"):
            method(*args)
    assert list(tmp_path.iterdir()) == []


def fitted_scaling(A=100, B=200, alpha=0.34, beta=0.28):
    analyzer = ScalingLawAnalyzer()
    analyzer._fitted, analyzer._E = True, 1.5
    analyzer._A, analyzer._B, analyzer._alpha, analyzer._beta = A, B, alpha, beta
    return analyzer


def test_chinchilla_optimum_uses_all_fitted_coefficients_and_respects_budget():
    analyzer = fitted_scaling()
    budget = 6e18
    N, D = analyzer.predict_chinchilla_optimal(budget)
    expected = (0.34 * 100 / (0.28 * 200)) ** (1 / 0.62) * (budget / 6) ** (0.28 / 0.62)
    assert N == int(expected)
    assert D == int(budget / (6 * N))
    assert 6 * N * D <= budget
    assert analyzer.predict_loss(N, D) < analyzer.predict_loss(N * 2, int(D / 2))
    analyzer._A *= 10
    larger_N, smaller_D = analyzer.predict_chinchilla_optimal(budget)
    assert larger_N > N and smaller_D < D


@pytest.mark.parametrize("budget", [0, -1, 5.99, float("inf"), float("nan")])
def test_chinchilla_rejects_invalid_compute_budget(budget):
    with pytest.raises(ValueError, match="compute_budget"):
        fitted_scaling().predict_chinchilla_optimal(budget)


@pytest.mark.parametrize("key,value", [("_A", 0), ("_B", -1), ("_alpha", float("nan")), ("_beta", 0)])
def test_chinchilla_rejects_nonpositive_or_nonfinite_fit(key, value):
    analyzer = fitted_scaling()
    setattr(analyzer, key, value)
    with pytest.raises(ValueError, match="A, B, alpha and beta"):
        analyzer.predict_chinchilla_optimal(1e18)


def test_scaling_fit_recovers_coefficients_on_large_realistic_scales():
    analyzer = ScalingLawAnalyzer()
    for params, tokens in [(1e8, 1e9), (1e8, 4e9), (4e8, 1e9), (4e8, 4e9)]:
        loss = 1.5 + 100 / params ** 0.34 + 200 / tokens ** 0.28
        analyzer.add_run("synthetic", params=params, tokens=tokens, loss=loss)
    fit = analyzer.fit()
    assert [fit["E"], fit["A"], fit["B"]] == pytest.approx([1.5, 100, 200])
    assert all(value >= 1 for value in analyzer.predict_chinchilla_optimal(6e18))


def test_scaling_fit_rejects_unidentifiable_runs():
    analyzer = ScalingLawAnalyzer()
    for _ in range(3):
        analyzer.add_run("duplicate", params=1e8, tokens=1e9, loss=2)
    with pytest.raises(ValueError, match="independently identify"):
        analyzer.fit()
