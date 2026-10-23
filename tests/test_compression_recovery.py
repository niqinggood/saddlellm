"""Recovery must optimize a real checkpoint and preserve pruning on reload."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from saddlellm.compression.RecoveryTrainer import (
    MASKS_FILENAME,
    RecoveryConfig,
    _features,
    _split_records,
    recovery_loss,
    run_recovery,
    validate_recovery_config,
)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    unk_token_id = 3

    def get_vocab(self):
        return {str(index): index for index in range(8)}

    def __call__(self, text, **kwargs):
        words = {"hello": 4, "world": 5, "user:": 6, "assistant:": 7}
        return {"input_ids": [words.get(word, 3) for word in text.split()][:kwargs.get("max_length", 512)]}

    def save_pretrained(self, path):
        (Path(path) / "tokenizer_stub.json").write_text("{}", encoding="utf-8")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)
        self.projection = torch.nn.Linear(8, 8, bias=False)
        self.calls = []

    def forward(self, input_ids, attention_mask=None, return_logits=False):
        # Native models can require an explicit request for logits.
        assert return_logits is True
        self.calls.append((self.training, torch.is_grad_enabled()))
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))

    def save_pretrained(self, path, state_dict=None):
        from pathlib import Path

        torch.save(self.state_dict() if state_dict is None else state_dict, Path(path) / "weights.pt")


def _jsonl(path, records):
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    return str(path)


def _config(tmp_path, **kwargs):
    data = _jsonl(tmp_path / "train.jsonl", [{"input_ids": [1, 4, 5, 2]}, {"input_ids": [1, 5, 4, 2]}])
    return RecoveryConfig("existing-checkpoint", data, str(tmp_path / "recovered"),
                          validation_split=0, max_steps=3, device="cpu", **kwargs)


def test_shifted_ce_and_kd_ignore_prompt_padding_and_last_logits():
    torch.manual_seed(17)
    student = torch.randn(2, 5, 8, requires_grad=True)
    teacher = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.tensor([[-100, -100, 4, 2, -100], [-100, 5, -100, -100, -100]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    loss, metrics = recovery_loss(student, labels, attention_mask=attention, teacher_logits=teacher,
                                  temperature=2.5, distill_weight=.7)
    selection = torch.stack([student[0, 1], student[0, 2], student[1, 0]])
    reference = torch.stack([teacher[0, 1], teacher[0, 2], teacher[1, 0]]).detach()
    ce = F.cross_entropy(selection, torch.tensor([4, 2, 5]))
    kd = F.kl_div(F.log_softmax(selection / 2.5, -1), F.softmax(reference / 2.5, -1), reduction="batchmean") * 2.5 ** 2
    torch.testing.assert_close(loss, .3 * ce + .7 * kd)
    torch.testing.assert_close(metrics["ce_loss"], ce)
    torch.testing.assert_close(metrics["kd_loss"], kd)
    assert metrics["tokens"] == 3
    loss.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[0, 0]) == 0  # prompt-only prediction
    assert torch.count_nonzero(student.grad[:, -1]) == 0
    assert torch.count_nonzero(student.grad[1, 1:]) == 0


def test_padding_token_equal_to_eos_does_not_mask_real_eos():
    logits = torch.randn(1, 3, 8, requires_grad=True)
    labels = torch.tensor([[4, 2, -100]])
    _, metrics = recovery_loss(logits, labels, attention_mask=torch.tensor([[1, 1, 0]]))
    assert metrics["tokens"] == 1
    torch.testing.assert_close(metrics["ce_loss"], F.cross_entropy(logits[0, :1], torch.tensor([2])))


def test_loss_rejects_empty_targets_and_vocab_mismatch():
    with pytest.raises(ValueError, match="no supervised"):
        recovery_loss(torch.randn(1, 3, 8), torch.full((1, 3), -100))
    with pytest.raises(ValueError, match="vocabulary"):
        recovery_loss(torch.randn(1, 3, 8), torch.tensor([[1, 4, 2]]), teacher_logits=torch.randn(1, 3, 9))


@pytest.mark.parametrize("problem", ["missing_file", "unsupported_file", "incomplete_parameters"])
def test_declared_exact_pruning_masks_cannot_fall_back_to_inferred_zeros(tmp_path, problem):
    config = _config(tmp_path)
    source = tmp_path / "pruned"
    source.mkdir()
    manifest = {"masks_file": MASKS_FILENAME, "masked_parameters": ["projection.weight"]}
    if problem == "unsupported_file":
        manifest["masks_file"] = "other_masks.pt"
    elif problem == "incomplete_parameters":
        manifest["masked_parameters"].append("embedding.weight")
        torch.save({"projection.weight": torch.ones(8, 8, dtype=torch.bool)}, source / MASKS_FILENAME)
    (source / "pruning_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    model = TinyModel()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises((ValueError, FileNotFoundError), match="masks_file|missing masks|parameter list"):
        run_recovery(replace(config, student_checkpoint=str(source)), student_model=model, tokenizer=Tokenizer())
    assert not Path(config.output_dir).exists()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("route", ["recovery", "qat"])
def test_inconsistent_exact_mask_rejected_before_erasing_source_weights(tmp_path, route):
    from saddlellm.compression.QuantizationAwareTraining import QATConfig, run_qat

    config = _config(tmp_path)
    source = tmp_path / "pruned"
    source.mkdir()
    mask = torch.ones(8, 8, dtype=torch.bool)
    mask[0, 0] = False
    torch.save({"projection.weight": mask}, source / MASKS_FILENAME)
    (source / "pruning_manifest.json").write_text(json.dumps({
        "masks_file": MASKS_FILENAME, "masked_parameters": ["projection.weight"],
    }), encoding="utf-8")
    model = TinyModel()
    with torch.no_grad():
        model.projection.weight.fill_(1)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="disagrees with materialized weights"):
        if route == "recovery":
            run_recovery(replace(config, student_checkpoint=str(source)), student_model=model, tokenizer=Tokenizer())
        else:
            qat = QATConfig(enabled=True, train_data=config.train_data, validation_split=0, max_steps=1)
            run_qat(source, config.output_dir, qat, model=model, tokenizer=Tokenizer())
    assert type(model.projection) is torch.nn.Linear
    assert not Path(config.output_dir).exists()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_real_steps_update_existing_student_and_freeze_teacher(tmp_path):
    config = _config(tmp_path)
    student, teacher, tokenizer = TinyModel(), TinyModel(), Tokenizer()
    teacher.train()
    teacher.projection.weight.requires_grad_(False)
    teacher_flags = [parameter.requires_grad for parameter in teacher.parameters()]
    teacher_before = copy.deepcopy(teacher.state_dict())
    before = student.projection.weight.detach().clone()
    report = run_recovery(config, student_model=student, teacher_model=teacher,
                          tokenizer=tokenizer, teacher_tokenizer=tokenizer)
    assert report["actual_steps"] == 3
    assert report["effective_distill_weight"] == .5
    assert report["teacher_used"] and not report["qat"]
    assert not torch.equal(student.projection.weight, before)
    assert all(not training and not gradients for training, gradients in teacher.calls)
    assert teacher.training
    assert [parameter.requires_grad for parameter in teacher.parameters()] == teacher_flags
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert json.loads((tmp_path / "recovered" / "recovery_report.json").read_text())["actual_steps"] == 3


def test_teacher_absent_is_explicit_ce_only_recovery(tmp_path):
    report = run_recovery(_config(tmp_path), student_model=TinyModel(), tokenizer=Tokenizer())
    assert report["effective_distill_weight"] == 0
    assert report["teacher_used"] is False
    assert all(step["kd_loss"] == 0 for step in report["loss_history"])
    assert report["validation_before"] is None


def test_teacher_vocab_mapping_and_shared_model_are_rejected(tmp_path):
    config = _config(tmp_path)
    bad_tokenizer = Tokenizer()
    bad_tokenizer.get_vocab = lambda: {"changed": 0}
    with pytest.raises(ValueError, match="token-ID vocabularies"):
        run_recovery(config, student_model=TinyModel(), teacher_model=TinyModel(),
                     tokenizer=Tokenizer(), teacher_tokenizer=bad_tokenizer)
    shared = TinyModel()
    with pytest.raises(ValueError, match="independent"):
        run_recovery(config, student_model=shared, teacher_model=shared,
                     tokenizer=Tokenizer(), teacher_tokenizer=Tokenizer())
    teacher = TinyModel()
    teacher.embedding.weight = torch.nn.Parameter(shared.embedding.weight.detach())
    with pytest.raises(ValueError, match="separate parameter storage"):
        run_recovery(config, student_model=shared, teacher_model=teacher,
                     tokenizer=Tokenizer(), teacher_tokenizer=Tokenizer())


def test_source_split_keeps_duplicate_conversations_together(tmp_path):
    config = _config(tmp_path)
    conversation = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"},
                                  {"role": "user", "content": "world"}, {"role": "assistant", "content": "hello"}]}
    other = {"text": "hello world hello"}
    _jsonl(tmp_path / "train.jsonl", [dict(conversation, id=1), dict(conversation, id=2), other])
    train, validation, info = _split_records(replace(config, validation_split=.5))
    assert len(train) + len(validation) == 3
    assert info["mode"] == "source_record_split"
    assert (len(train), len(validation)) in {(1, 2), (2, 1)}
    assert not ({json.dumps(item, sort_keys=True) for item in train} & {json.dumps(item, sort_keys=True) for item in validation})
    features = _features([conversation], Tokenizer(), 32)
    assert len(features) == 2
    assert all(feature["labels"][0] == -100 for feature in features)


def test_source_split_links_prompt_and_explicit_group_transitively(tmp_path):
    config = _config(tmp_path)
    records = [{"prompt": "hello", "completion": "world", "conversation_id": "one"},
               {"prompt": "hello", "completion": "hello", "conversation_id": "two"},
               {"prompt": "world", "completion": "hello", "conversation_id": "two"},
               {"text": "independent text"}]
    _jsonl(tmp_path / "train.jsonl", records)
    train, validation, _ = _split_records(replace(config, validation_split=.5))
    assert sorted([len(train), len(validation)]) == [1, 3]
    heldout = _jsonl(tmp_path / "validation.jsonl", [{"prompt": "hello", "completion": "changed answer"}])
    with pytest.raises(ValueError, match="prompts or conversation groups"):
        _split_records(replace(config, validation_data=heldout))


@pytest.mark.parametrize("record", [{"input_ids": "123"}, {"input_ids": [[1, 2]]},
                                    {"input_ids": [1, True]}, {"input_ids": [1, 2], "labels": [[-100], [2]]},
                                    {"input_ids": [1, 2], "attention_mask": [1]}])
def test_malformed_tokenized_source_rejected_before_model_loading(tmp_path, record):
    config = _config(tmp_path)
    _jsonl(tmp_path / "train.jsonl", [record])
    with pytest.raises(ValueError, match="input_ids|labels|attention_mask"):
        _split_records(config)


def test_explicit_validation_catches_normalized_and_truncated_leakage(tmp_path):
    config = _config(tmp_path)
    validation = _jsonl(tmp_path / "validation.jsonl", [{"id": "different", "input_ids": [1, 4, 5, 2]}])
    with pytest.raises(ValueError, match="overlapping source"):
        run_recovery(replace(config, validation_data=validation), student_model=TinyModel(), tokenizer=Tokenizer())
    _jsonl(tmp_path / "validation.jsonl", [{"input_ids": [0, 1, 4, 5, 2, 0], "attention_mask": [0, 1, 1, 1, 1, 0]}])
    with pytest.raises(ValueError, match="overlapping source"):
        _split_records(replace(config, validation_data=validation))
    _jsonl(tmp_path / "train.jsonl", [{"input_ids": [1, 4, 5, 2, 4]}])
    _jsonl(tmp_path / "validation.jsonl", [{"input_ids": [1, 4, 5, 2, 5]}])
    with pytest.raises(ValueError, match="prepared token identity"):
        run_recovery(replace(config, validation_data=validation, max_seq_length=4), student_model=TinyModel(), tokenizer=Tokenizer())


def test_live_pruning_masks_survive_optimizer_and_serialization(tmp_path):
    from torch.nn.utils import prune

    config = _config(tmp_path, weight_decay=.3, learning_rate=.2)
    model = TinyModel()
    mask = torch.ones_like(model.projection.weight)
    mask.view(-1)[::2] = 0
    prune.custom_from_mask(model.projection, "weight", mask)
    report = run_recovery(config, student_model=model, tokenizer=Tokenizer())
    saved = torch.load(tmp_path / "recovered" / "weights.pt", weights_only=True)
    assert "projection.weight_orig" not in saved
    assert "projection.weight_mask" not in saved
    assert torch.count_nonzero(saved["projection.weight"][mask == 0]) == 0
    assert report["sparsity_after"]["pruned_weights_nonzero"] == 0
    exact = torch.load(tmp_path / "recovered" / MASKS_FILENAME, weights_only=True)
    assert torch.equal(exact["projection.weight"], mask.bool())
    restored = TinyModel()
    restored.load_state_dict(saved)
    config2 = replace(config, student_checkpoint=str(tmp_path / "recovered"), output_dir=str(tmp_path / "recovered-again"))
    second = run_recovery(config2, student_model=restored, tokenizer=Tokenizer())
    assert second["pruning_masks"]["source"] == "saved_exact_masks"
    assert second["sparsity_after"]["preserved_pruned_parameters"] == int((mask == 0).sum())
    assert torch.count_nonzero(restored.projection.weight[mask == 0]) == 0


def test_legacy_materialized_pruning_infers_zeros_and_saves_exact_masks(tmp_path):
    config = _config(tmp_path)
    checkpoint = tmp_path / "pruned"
    checkpoint.mkdir()
    (checkpoint / "pruning_manifest.json").write_text(json.dumps({"target_modules": ["projection"], "masks_materialized": True}))
    model = TinyModel()
    with torch.no_grad():
        model.projection.weight[0] = 0
    report = run_recovery(replace(config, student_checkpoint=str(checkpoint)), student_model=model, tokenizer=Tokenizer())
    assert report["pruning_masks"]["inferred_from_materialized_zeros"] is True
    assert report["sparsity_after"]["preserved_pruned_parameters"] == 8
    assert torch.count_nonzero(model.projection.weight[0]) == 0


def test_empty_saved_masks_cannot_silently_disable_pruning_preservation(tmp_path):
    config = _config(tmp_path)
    checkpoint = tmp_path / "pruned"
    checkpoint.mkdir()
    torch.save({}, checkpoint / MASKS_FILENAME)
    with pytest.raises(ValueError, match="nonempty dictionary"):
        run_recovery(replace(config, student_checkpoint=str(checkpoint)), student_model=TinyModel(), tokenizer=Tokenizer())


@pytest.mark.parametrize("field,value", [("max_steps", 0), ("batch_size", True), ("validation_split", float("nan")),
                                         ("distill_weight", 1.1), ("temperature", 0), ("learning_rate", float("inf"))])
def test_config_rejects_invalid_controls(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        validate_recovery_config(replace(_config(tmp_path), **{field: value}))


def test_dry_run_and_packed_quantized_artifacts(tmp_path):
    config = _config(tmp_path)
    assert validate_recovery_config(config)["model_loaded"] is False
    assert not (tmp_path / "recovered").exists()
    packed = tmp_path / "packed"
    packed.mkdir()
    (packed / "quantization_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="not QAT"):
        validate_recovery_config(replace(config, student_checkpoint=str(packed)))
    quantized = TinyModel()
    quantized.is_loaded_in_4bit = True
    with pytest.raises(ValueError, match="floating-point checkpoint"):
        run_recovery(config, student_model=quantized, tokenizer=Tokenizer())


@pytest.mark.parametrize("source", ["student_checkpoint", "teacher_checkpoint"])
@pytest.mark.parametrize("nested", [False, True])
def test_output_cannot_overwrite_or_nest_inside_source_checkpoint(tmp_path, source, nested):
    config = _config(tmp_path)
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    output = checkpoint / "recovered" if nested else checkpoint
    with pytest.raises(ValueError, match=f"outside {source}"):
        validate_recovery_config(replace(config, **{source: str(checkpoint)}, output_dir=str(output)))
    assert not (checkpoint / "recovered").exists()


def _save_tokenizer(path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = tokenizers.Tokenizer(WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                                             "hello": 4, "world": 5, "tiny": 6, "model": 7}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.save_pretrained(path)
    return tokenizer


@pytest.mark.parametrize("architecture", ["native", "hf"])
def test_load_train_save_reload_real_checkpoint_on_cpu(tmp_path, architecture):
    from saddlellm.models.ModelLoader import load_causal_lm

    torch.manual_seed(71)
    checkpoint = tmp_path / "student"
    if architecture == "native":
        from saddlellm.models.ModelBlueprint import (
            AttentionBlueprint,
            FFNBlueprint,
            ModelBlueprint,
        )

        model = ModelBlueprint(name="recovery-test", hidden_size=16, vocab_size=8, num_layers=1,
            max_position_embeddings=32, attention=AttentionBlueprint(kind="gqa", backend="eager", num_heads=4, num_kv_heads=2),
            ffn=FFNBlueprint(kind="swiglu", intermediate_size=32)).build_model()
    else:
        from transformers import GPT2Config, GPT2LMHeadModel

        model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=32, n_ctx=32, n_embd=8, n_layer=1, n_head=1,
                                         bos_token_id=1, eos_token_id=2, pad_token_id=0))
    model.save_pretrained(checkpoint)
    _save_tokenizer(checkpoint)
    original = copy.deepcopy(model.state_dict())
    data = _jsonl(tmp_path / "train.jsonl", [{"text": "hello world tiny"}, {"text": "tiny model hello"}])
    validation = _jsonl(tmp_path / "validation.jsonl", [{"text": "world model tiny"}])
    config = RecoveryConfig(str(checkpoint), data, str(tmp_path / "recovered"), teacher_checkpoint=str(checkpoint),
                            validation_data=validation, validation_split=0, max_steps=2, batch_size=2,
                            device="cpu", local_files_only=True, learning_rate=.01)
    report = run_recovery(config)
    restored = load_causal_lm(tmp_path / "recovered", device="cpu", local_files_only=True)
    assert report["actual_steps"] == 2 and report["teacher_used"]
    assert report["validation_before"]["supervised_tokens"] == 2
    assert report["validation_after"]["supervised_tokens"] == 2
    assert any(not torch.equal(value, original[name]) for name, value in restored.state_dict().items())
    reloaded_source = load_causal_lm(checkpoint, device="cpu", local_files_only=True)
    assert all(torch.equal(value, original[name]) for name, value in reloaded_source.state_dict().items())
