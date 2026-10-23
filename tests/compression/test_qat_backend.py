"""Exercise matching QAT/converted numerics and real CPU training artifacts."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from saddlellm.compression.QuantizationAwareTraining import (
    BACKEND,
    QATConfig,
    convert_qat_model,
    load_qat,
    prepare_qat_model,
    run_qat,
    validate_qat_config,
)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    unk_token_id = 3

    def get_vocab(self):
        return {str(index): index for index in range(8)}

    def save_pretrained(self, path):
        (Path(path) / "tokenizer_stub.json").write_text("{}", encoding="utf-8")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)
        self.projection = torch.nn.Linear(8, 8, bias=True)

    def forward(self, input_ids, attention_mask=None, return_logits=False):
        assert return_logits
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def _data(path, sequences):
    path.write_text("\n".join(json.dumps({"input_ids": ids}) for ids in sequences) + "\n", encoding="utf-8")
    return str(path)


def _config(tmp_path, **kwargs):
    train_data = _data(tmp_path / "train.jsonl", [[1, 4, 5, 2], [1, 5, 4, 2]])
    return QATConfig(enabled=True, train_data=train_data, validation_split=0,
                     max_steps=2, learning_rate=.01, device="cpu", **kwargs)


def test_disabled_config_and_run_do_not_import_training_stack():
    code = """
import sys
from saddlellm.compression.QuantizationAwareTraining import QATConfig, run_qat, validate_qat_config
model = object()
plan = validate_qat_config(QATConfig())
result = run_qat(None, None, QATConfig(), model=model)
assert result.model is model and result.report['actual_steps'] == 0
assert not plan['model_loaded']
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_profile_validation_and_enabled_data_validation(tmp_path):
    with pytest.raises(ValueError, match="backend"):
        validate_qat_config(QATConfig(backend="pretend-int4"))
    with pytest.raises(TypeError, match="boolean"):
        validate_qat_config(QATConfig(enabled="true"))
    config = _config(tmp_path)
    plan = validate_qat_config(config, "original", tmp_path / "qat")
    assert plan["backend"] == BACKEND and plan["activation_precision"] == "float32"
    with pytest.raises(ValueError, match="max_steps"):
        validate_qat_config(replace(config, max_steps=0), "original", tmp_path / "qat")


def test_fake_quant_gradients_and_conversion_match_exactly():
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3))
    original_parameter = model[0].weight
    inputs = torch.randn(2, 5)
    float_output = model(inputs).detach()
    targets = prepare_qat_model(model)
    assert len(targets) == 2 and model[0].weight is original_parameter
    with torch.no_grad():
        model[0].weight[0].zero_()  # zero rows must be finite and stay zero
    output = model(inputs)
    output.sum().backward()
    assert torch.isfinite(original_parameter.grad).all()
    assert original_parameter.grad.count_nonzero()
    expected = output.detach()
    convert_qat_model(model)
    assert model[0].qweight.dtype == torch.int8
    assert model[0].qweight[0].count_nonzero() == 0
    assert model[0].qweight.min() >= -127
    assert not any(name.endswith("weight") for name, _ in model.named_parameters())
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    assert not torch.equal(expected, float_output)


def test_real_qat_train_convert_save_load_with_heldout_and_masks(tmp_path):
    config = _config(tmp_path)
    validation = _data(tmp_path / "validation.jsonl", [[1, 6, 7, 2]])
    config = replace(config, validation_data=validation)
    source = tmp_path / "source"
    source.mkdir()
    model, tokenizer = TinyModel(), Tokenizer()
    with torch.no_grad():
        model.projection.weight[0, :4].zero_()
    mask = model.projection.weight.detach().ne(0)
    torch.save({"projection.weight": mask}, source / "pruning_masks.pt")
    float_before = model.projection.weight.detach().clone()
    output = tmp_path / "qat"
    result = run_qat(source, output, config, model=model, tokenizer=tokenizer)
    report = result.report
    assert result.model is model and report["actual_steps"] == 2
    assert not result.model.training
    assert report["qat"] and report["converted_modules"] == 1
    assert report["validation_float_before"]["supervised_tokens"] == 3
    assert report["validation_fake_quant_after"] == report["validation_converted"]
    assert report["sparsity_after"]["pruned_weights_nonzero"] == 0
    assert model.projection.qweight[0, :4].count_nonzero() == 0
    assert not torch.equal(model.projection.weight, float_before)
    loaded = load_qat(output, model=TinyModel())
    inputs = torch.tensor([[1, 4, 5, 2]])
    torch.testing.assert_close(model(inputs, return_logits=True).logits,
                               loaded(inputs, return_logits=True).logits, rtol=0, atol=0)
    manifest = json.loads((output / "quantization_manifest.json").read_text(encoding="utf-8"))
    assert not manifest["accelerated_int8_kernels"]
    assert not manifest["self_contained"]  # custom TinyModel requires architecture injection
    state = torch.load(output / "qat_model.pt", weights_only=True)
    assert state["projection.qweight"].dtype == torch.int8
    assert "projection.weight" not in state
    assert json.loads((output / "qat_report.json").read_text(encoding="utf-8"))["qat"]


def test_live_pruning_and_teacher_preserved(tmp_path):
    from torch.nn.utils import prune

    model, teacher, tokenizer = TinyModel(), TinyModel(), Tokenizer()
    prune.l1_unstructured(model.projection, "weight", amount=.5)
    mask = model.projection.weight_mask.detach().bool().clone()
    teacher.train()
    before = copy.deepcopy(teacher.state_dict())
    result = run_qat("original", tmp_path / "qat", _config(tmp_path), model=model, tokenizer=tokenizer,
                     teacher_model=teacher, teacher_tokenizer=tokenizer)
    assert result.report["teacher_used"]
    assert teacher.training
    assert model.projection.qweight[~mask].count_nonzero() == 0
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_hf_gpt2_conv1d_self_contained_roundtrip(tmp_path):
    transformers = pytest.importorskip("transformers")

    config = transformers.GPT2Config(vocab_size=8, n_positions=8, n_embd=8, n_layer=1, n_head=2,
                                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    model = transformers.GPT2LMHeadModel(config)
    result = run_qat("unused-source", tmp_path / "qat", _config(tmp_path), model=model, tokenizer=Tokenizer())
    assert result.report["self_contained"]
    assert any(target["transposed"] for target in result.report["modules"])
    assert result.report["converted_modules"] == 5
    inputs = torch.tensor([[1, 4, 5]])
    model.eval()
    expected = model(inputs).logits
    reloaded = load_qat(tmp_path / "qat")
    torch.testing.assert_close(reloaded(inputs).logits, expected, rtol=0, atol=0)
    generated = reloaded.generate(inputs, max_new_tokens=2, do_sample=False, pad_token_id=0, eos_token_id=2)
    assert inputs.shape[1] < generated.shape[1] <= inputs.shape[1] + 2


@pytest.mark.parametrize("fused", [False, True])
def test_native_self_contained_roundtrip_with_padded_vocabulary(tmp_path, fused):
    from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

    config = SaddleModelConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                intermediate_size=16, max_position_embeddings=16,
                                fused_qkv=fused, fused_gate_up=fused)
    model = SaddleForCausalLM(config)
    result = run_qat("unused-source", tmp_path / "qat", _config(tmp_path), model=model, tokenizer=Tokenizer())
    assert result.report["self_contained"]
    inputs = torch.tensor([[1, 4, 5]])
    model.eval()
    expected = model(input_ids=inputs, return_logits=True).logits
    reloaded = load_qat(tmp_path / "qat")
    torch.testing.assert_close(reloaded(input_ids=inputs, return_logits=True).logits, expected, rtol=0, atol=0)
    generated = reloaded.generate(inputs, max_new_tokens=2, do_sample=False, pad_token_id=0, eos_token_id=2)
    assert inputs.shape[1] < generated.shape[1] <= inputs.shape[1] + 2


def test_reject_data_leakage_and_unsupported_models_before_output(tmp_path):
    config = _config(tmp_path)
    config = replace(config, validation_data=config.train_data)
    with pytest.raises(ValueError, match="overlapping"):
        run_qat("unused", tmp_path / "out", config, model=TinyModel(), tokenizer=Tokenizer())
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="no supported"):
        prepare_qat_model(torch.nn.Sequential(torch.nn.Embedding(8, 8)))


def test_load_rejects_backend_and_state_path_mutation(tmp_path):
    run_qat("unused", tmp_path / "out", _config(tmp_path), model=TinyModel(), tokenizer=Tokenizer())
    path = tmp_path / "out" / "quantization_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["state_file"] = "../other.pt"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="state file"):
        load_qat(path.parent, model=TinyModel())
