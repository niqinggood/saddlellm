"""Recovery and QAT must audit the effective tokenized supervision contexts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from saddlellm.compression.QuantizationAwareTraining import QATConfig, run_qat
from saddlellm.compression.RecoveryTrainer import RecoveryConfig, run_recovery


class LowerTokenizer:
    """Real deterministic normalization makes distinct source strings collide."""

    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        vocab = {"user:": 1, "upper": 3, "assistant:": 4, "one": 5, "two": 6, "lower": 7}
        return {"input_ids": [vocab[word] for word in text.lower().split()][:kwargs.get("max_length", 512)]}

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        text = " ".join(message["role"] + ": " + message["content"] for message in messages)
        return text + (" assistant:" if add_generation_prompt else "")

    def save_pretrained(self, directory):
        (Path(directory) / "tokenizer_stub.json").write_text("{}", encoding="utf-8")


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)
        self.projection = torch.nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask=None, return_logits=False):
        assert return_logits is True
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))

    def save_pretrained(self, directory, state_dict=None):
        torch.save(self.state_dict() if state_dict is None else state_dict, Path(directory) / "weights.pt")


def _chat(prompt, answer):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def _write(path, record):
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return str(path)


def _run(route, tmp_path, model, train, validation):
    train_data = _write(tmp_path / "train.jsonl", train)
    validation_data = _write(tmp_path / "validation.jsonl", validation)
    output = str(tmp_path / "model-output")
    options = {"train_data": train_data, "validation_data": validation_data, "validation_split": 0,
               "max_steps": 1, "learning_rate": .01, "device": "cpu"}
    if route == "recovery":
        return run_recovery(RecoveryConfig("unused-source", output_dir=output, **options),
                            student_model=model, tokenizer=LowerTokenizer())
    return run_qat("unused-source", output, QATConfig(enabled=True, **options),
                   model=model, tokenizer=LowerTokenizer()).report


@pytest.mark.parametrize("route", ["recovery", "qat"])
@pytest.mark.parametrize("collision", ["padding_label", "normalized_chat_prompt"])
def test_effective_prompt_leakage_rejected_before_model_mutation_or_output(tmp_path, route, collision):
    if collision == "padding_label":
        # The pad label is permitted input and becomes -100 in _features.
        # Before that masking, label[0] hides the later response boundary.
        train = {"input_ids": [0, 1, 4, 5, 2], "attention_mask": [0, 1, 1, 1, 1],
                 "labels": [0, -100, -100, 5, 2]}
        validation = {"input_ids": [0, 1, 4, 6, 2], "attention_mask": [0, 1, 1, 1, 1],
                      "labels": [0, -100, -100, 6, 2]}
    else:
        # The case-sensitive source identities differ, but the actual model
        # sees the same prompt tokens with different supervised completions.
        train, validation = _chat("UPPER", "one"), _chat("upper", "two")
    model = TinyCausalLM()
    original_state = copy.deepcopy(model.state_dict())
    original_parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    try:
        with pytest.raises(ValueError, match="prepared token identity"):
            _run(route, tmp_path, model, train, validation)
    finally:
        # Guard the failure boundary as well as the error text: a rejected
        # dataset must not already have trained or installed fake-quant layers.
        assert type(model.projection) is torch.nn.Linear
        assert {name: id(parameter) for name, parameter in model.named_parameters()} == original_parameters
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
        assert not (tmp_path / "model-output").exists()


@pytest.mark.parametrize("route", ["recovery", "qat"])
def test_distinct_effective_prompts_train_and_report_the_shared_split_audit(tmp_path, route):
    model = TinyCausalLM()
    initial_embedding = model.embedding.weight.detach().clone()
    report = _run(route, tmp_path, model, _chat("UPPER", "one"), _chat("lower", "two"))
    assert report["actual_steps"] == 1
    assert not torch.equal(model.embedding.weight, initial_embedding)
    audit = report["data"]["split_audit"]
    assert audit["kind"] == "runtime_training_split_audit"
    assert audit["passed"] is True
    assert audit["source_overlap"] == 0 and audit["prepared_overlap"] == 0
    assert audit["prepared_sequences_checked"] == [1, 1]
    assert audit["train_records"] == 1 and audit["validation_records"] == 1
    report_name = "recovery_report.json" if route == "recovery" else "qat_report.json"
    saved = json.loads((tmp_path / "model-output" / report_name).read_text(encoding="utf-8"))
    assert saved["data"]["split_audit"] == audit
