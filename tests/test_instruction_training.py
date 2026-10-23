"""Real saved tokenizer checks; no fabricated model output or weight downloads."""
import json
import os
from pathlib import Path

import pytest

from saddlellm.agents.instruction_training import format_training_record, formatted_token_batch, partition_evidence, training_format
from saddlellm.agents.RealWorldOperators import _data_config, _record_text


@pytest.fixture
def tokenizer():
    path = Path(os.environ.get("SADDLE_LLM_TEST_TOKENIZER", r"C:\Users\niqin\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"))
    if not path.is_dir():
        pytest.skip("requires an existing local saved tokenizer; downloads are disabled")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    return tokenizer


def test_response_only_labels_use_real_token_offsets_and_exclude_prompt_padding(tokenizer):
    record = {"messages": [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Name a metal."},
                           {"role": "assistant", "content": "Copper."}, {"role": "user", "content": "Name another."},
                           {"role": "assistant", "content": "Iron."}]}
    formatted = format_training_record(record, tokenizer, True)
    target = tokenizer.decode([token for token in formatted["labels"] if token != -100])
    assert "Copper." in target and "Iron." in target
    assert "Name" not in target and "briefly" not in target and "assistant" not in target
    batch = formatted_token_batch(tokenizer, [formatted], len(formatted["input_ids"]) + 5, "cpu")
    assert batch["labels"][0, -5:].tolist() == [-100] * 5
    assert batch["attention_mask"][0, -5:].tolist() == [0] * 5
    assert (batch["labels"][0] != -100).sum() < batch["attention_mask"][0].sum()


def test_whole_chat_loss_and_instruction_mapping_keep_saved_model_format(tokenizer):
    record = {"instruction": "Summarize the material", "input": "Copper is conductive.", "output": "Copper conducts electricity."}
    formatted = format_training_record(record, tokenizer, False)
    assert "<|user|>" in formatted["text"] and "Instruction:" not in formatted["text"]
    assert formatted["labels"] == formatted["input_ids"]
    assert "Copper is conductive." in formatted["text"]


def test_missing_template_and_response_truncation_fail_explicitly(tokenizer):
    record = {"instruction": "A fairly long user question needs an answer", "output": "A clear answer."}
    formatted = format_training_record(record, tokenizer, True)
    with pytest.raises(ValueError, match="truncates every"):
        formatted_token_batch(tokenizer, [formatted], 2, "cpu")
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="no saved chat template"):
        format_training_record(record, tokenizer, True)


@pytest.mark.parametrize("record", [{"text": "plain corpus"}, {"messages": [{"role": "user", "content": "unanswered"}]},
                                   {"messages": [{"role": "assistant", "content": "bad order"}]}, {"instruction": "x", "output": ""}])
def test_invalid_chat_training_records_are_not_generic_english_fallback(tokenizer, record):
    with pytest.raises(ValueError):
        format_training_record(record, tokenizer, True)


def test_raw_defaults_remain_compatible_and_response_only_is_explicit():
    assert training_format({}) == ("raw", False)
    assert _record_text({"instruction": "Question", "output": "Answer"}, "text") == "Instruction:\nQuestion\n\nResponse:\nAnswer"
    with pytest.raises(ValueError, match="requires"):
        training_format({"responseOnlyLoss": True})


def test_identical_content_is_detected_despite_new_file_name_or_whitespace():
    evidence = partition_evidence(["Copper  conducts\n electricity"], ["Copper conducts electricity"])
    assert evidence["overlapUniqueSamples"] == 1
    assert evidence["trainRecordHashes"] == evidence["evalRecordHashes"]


def test_unread_tail_differences_do_not_hide_identical_actual_token_inputs(tokenizer):
    evidence = partition_evidence(["Copper conducts electricity. First unused tail."], ["Copper conducts electricity. Different unused tail."], tokenizer, 4)
    assert evidence["overlapUniqueSamples"] == 0
    assert evidence["overlapTokenSequences"] == 1


def test_one_source_internal_split_has_content_disjointness_proof(tokenizer, tmp_path):
    records = [{"instruction": f"Mechanical format test {index}", "output": f"Response number {index}."} for index in range(10)]
    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in records), "utf-8")
    train, evaluation, lineage = _data_config({"datasetPath": str(source), "trainingTextFormat": "chat_template", "responseOnlyLoss": True}, tmp_path, tokenizer)
    evidence = lineage["dataPartitionEvidence"]
    assert len(train) == 8 and len(evaluation) == 2 and evidence["overlapUniqueSamples"] == 0
    assert lineage["datasetPath"] == lineage["evalDatasetPath"]
    assert all(record["responseOnlyLoss"] for record in train + evaluation)
