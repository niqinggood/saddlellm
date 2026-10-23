"""Configuration switches, public dispatch and quantized loading contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from saddlellm.compression import (
    QATConfig,
    QuantizationConfig,
    apply_quantization,
    plan_quantization,
)


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return str(path)


def test_defaults_are_a_strict_identity_noop(tmp_path):
    model, tokenizer = object(), object()
    result = apply_quantization(model, tokenizer, output_dir=tmp_path / "unused")
    assert result.model is model and result.tokenizer is tokenizer
    assert result.report["status"] == "skipped"
    assert not result.report["enabled"]
    assert not (tmp_path / "unused").exists()


def test_disabled_checkpoint_is_not_loaded_or_resolved():
    path = "missing/remote-model"
    result = apply_quantization(path, config={"ptq": {"enabled": False}, "qat": {"enabled": False}})
    assert result.model is path
    assert result.output_dir is None


def test_disabled_public_import_does_not_import_tensor_or_ptq_dependencies():
    script = """
import sys
from saddlellm import apply_quantization, QuantizationConfig
model = object()
result = apply_quantization(model, config=QuantizationConfig())
assert result.model is model
assert not any(name in sys.modules for name in ('torch','transformers','llmcompressor','compressed_tensors'))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=45, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("config,match", [
    ({"ptq": {"enabled": True}, "qat": {"enabled": True}}, "either ptq or qat"),
    ({"ptq": {"enabled": "false"}}, "booleans"),
    ({"enabled": True}, "Unknown quantization"),
    ({"qat": {"enable": True}}, "Unknown qat"),
    ({"ptq": True}, "mapping"),
])
def test_invalid_or_ambiguous_switches_fail_before_loading(tmp_path, config, match):
    with pytest.raises((ValueError, TypeError), match=match):
        apply_quantization("missing-model", config=config, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("method", ["awq", "gptq"])
def test_ptq_configuration_dispatch_and_no_caller_mutation(tmp_path, monkeypatch, method):
    import saddlellm.compression.CalibratedPTQ as backend

    model, tokenizer = object(), object()
    seen = {}
    settings = {"ptq": {"enabled": True, "method": method, "calibration_data": ["representative text"], "group_size": 8}}
    original = json.dumps(settings, sort_keys=True)

    def run(source, output, config, **kwargs):
        seen.update(source=source, output=output, config=config, **kwargs)
        return backend.PTQResult(model, tokenizer, {"method": config.method, "format": "compressed-tensors"}, output)

    monkeypatch.setattr(backend, "run_calibrated_ptq", run)
    result = apply_quantization(model, tokenizer, config=settings, output_dir=tmp_path / "out")
    assert seen["model"] is model and seen["tokenizer"] is tokenizer
    assert seen["config"].method == method
    assert result.model is model and result.report["route"] == "ptq"
    assert json.dumps(settings, sort_keys=True) == original


def test_typed_qat_switch_and_dry_run_do_not_train(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationAwareTraining as backend

    data = write_rows(tmp_path / "train.jsonl", [{"text": "one two"}, {"text": "three four"}])
    config = QuantizationConfig(qat=QATConfig(enabled=True, train_data=data, max_steps=2))
    monkeypatch.setattr(backend, "run_qat", lambda *a, **k: pytest.fail("dry-run trained"))
    result = apply_quantization("missing-model", config=config, output_dir=tmp_path / "out", dry_run=True)
    assert result.report["route"] == "qat" and result.report["status"] == "planned"
    assert result.report["model_loaded"] is False
    assert not (tmp_path / "out").exists()


def test_inactive_qat_paths_do_not_affect_the_enabled_ptq_route(tmp_path):
    destination = tmp_path / "out"
    config = {"ptq": {"enabled": True, "calibration_data": ["one two"]},
              "qat": {"enabled": False, "teacher_checkpoint": str(destination), "train_data": str(destination / "unused.jsonl")}}
    plan = plan_quantization("base", config=config, output_dir=destination)
    assert plan["route"] == "ptq" and not destination.exists()


def test_qat_dispatch_preserves_requested_profile(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationAwareTraining as backend

    data = write_rows(tmp_path / "train.jsonl", [{"text": "one two"}, {"text": "three four"}])
    seen = {}
    model, tokenizer = object(), object()

    def run(source, output, config, **kwargs):
        seen.update(source=source, config=config, **kwargs)
        return backend.QATResult(model, tokenizer, {"backend": config.backend, "actual_steps": config.max_steps}, output)

    monkeypatch.setattr(backend, "run_qat", run)
    result = apply_quantization(model, tokenizer, output_dir=tmp_path / "out", config={"qat": {"enabled": True, "train_data": data, "max_steps": 3}})
    assert seen["model"] is model and seen["config"].max_steps == 3
    assert result.report["route"] == "qat" and result.report["actual_steps"] == 3


def test_output_never_overwrites_model_or_calibration(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    calibration = write_rows(tmp_path / "calibration.jsonl", [{"text": "one two"}])
    config = {"ptq": {"enabled": True, "calibration_path": calibration}}
    for output in (source, source / "child", Path(calibration)):
        with pytest.raises(ValueError, match="output_dir"):
            plan_quantization(source, config=config, output_dir=output)


@pytest.mark.parametrize("manifest", [
    {"method": "qat", "backend": "unknown"},
    {"method": "awq", "backend": "llm-compressor", "format": "awq"},
    {"method": "gptq", "format": "compressed-tensors"},
])
def test_unknown_package_formats_are_not_treated_as_dynamic_int8(tmp_path, manifest):
    from saddlellm.models.ModelLoader import load_causal_lm

    (tmp_path / "quantization_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported packaged"):
        load_causal_lm(tmp_path, device="cpu")


def test_qat_package_dispatches_its_own_loader(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationAwareTraining as backend
    from saddlellm.models.ModelLoader import load_causal_lm

    (tmp_path / "quantization_manifest.json").write_text(json.dumps({"method": "qat", "backend": "portable-int8-weight-only"}), encoding="utf-8")
    sentinel = object()
    monkeypatch.setattr(backend, "load_qat", lambda *a, **k: sentinel)
    assert load_causal_lm(tmp_path, device="cpu", dtype="float32") is sentinel
    with pytest.raises(ValueError, match="float32"):
        load_causal_lm(tmp_path, dtype="float16")


def test_compressed_tensors_loading_is_explicitly_decompressed(tmp_path, monkeypatch):
    import saddlellm.compression.CalibratedPTQ as backend
    from saddlellm.models.ModelLoader import load_causal_lm

    (tmp_path / "quantization_manifest.json").write_text(json.dumps({"method": "gptq", "backend": "llm-compressor", "format": "compressed-tensors"}), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "compressed-tensors", "quantization_status": "compressed", "format": "pack-quantized"}}), encoding="utf-8")
    seen = {}

    class Model:
        def to(self, device):
            return self

    def load(path, **kwargs):
        seen.update(kwargs)
        return Model()

    monkeypatch.setattr(backend, "load_calibrated_ptq", load)
    model = load_causal_lm(tmp_path, device="cpu", local_files_only=True)
    assert seen["device"] == "cpu"
    assert seen["local_files_only"] is True
    assert model._saddle_quantization_load_mode == "compressed-tensors-decompressed"


def test_new_ptq_cli_reuses_the_function_configuration(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationPipeline as pipeline
    from saddlellm import cli

    seen = {}

    def apply(model, **kwargs):
        seen.update(model=model, **kwargs)
        return pipeline.QuantizationResult(model, None, {"status": "planned", "enabled": True})

    monkeypatch.setattr(pipeline, "apply_quantization", apply)
    assert cli.main(["quantize", "base", str(tmp_path / "out"), "--method", "gptq", "--calibration-data", "calibration.jsonl", "--group-size", "64", "--dry-run"]) == 0
    assert seen["config"]["ptq"]["method"] == "gptq"
    assert seen["config"]["ptq"]["group_size"] == 64
    assert seen["dry_run"] is True


def test_cli_config_both_disabled_does_not_create_output(tmp_path, capsys):
    from saddlellm import cli

    source = tmp_path / "config.json"
    source.write_text('{"ptq":{"enabled":false},"qat":{"enabled":false}}', encoding="utf-8")
    output = tmp_path / "out"
    assert cli.main(["quantize", "nonexistent-model", str(output), "--config", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "skipped"
    assert not output.exists()


@pytest.mark.parametrize("architecture", ["hf", "native"])
def test_public_qat_function_reload_serve_and_regression(tmp_path, architecture):
    """The function result and its artifact share evaluation/serving numerics."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from saddlellm.evaluation.RegressionSuite import run_regression_suite
    from saddlellm.models.ModelLoader import load_causal_lm
    from saddlellm.runtime.InferenceServer import (
        InferenceServerSettings,
        load_inference_model,
    )

    source = tmp_path / "source"
    vocabulary = {word: index for index, word in enumerate(["<pad>", "<bos>", "<eos>", "<unk>", "one", "two", "three", "four"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.save_pretrained(source)
    torch.manual_seed(5)
    if architecture == "hf":
        model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=32, n_embd=8, n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2, pad_token_id=0))
    else:
        from saddlellm.models.ModelBlueprint import (
            AttentionBlueprint,
            FFNBlueprint,
            ModelBlueprint,
        )

        model = ModelBlueprint(name="function-qat", hidden_size=8, vocab_size=8, num_layers=1,
            max_position_embeddings=32, attention=AttentionBlueprint(num_heads=2, num_kv_heads=2),
            ffn=FFNBlueprint(intermediate_size=16)).build_model()
    model.save_pretrained(source)
    train = write_rows(tmp_path / "train.jsonl", [{"text": "one two three"}, {"text": "two three four"}])
    heldout = write_rows(tmp_path / "heldout.jsonl", [{"instruction": "four", "output": "one"}])
    output = tmp_path / "qat"
    result = apply_quantization(str(source), output_dir=output, config={"qat": {
        "enabled": True, "train_data": train, "validation_split": 0, "max_steps": 1,
        "max_seq_length": 16, "device": "cpu", "local_files_only": True,
    }})
    assert result.report["actual_steps"] == 1 and result.report["route"] == "qat"
    loaded = load_causal_lm(output, device="cpu", local_files_only=True)
    served, _ = load_inference_model(InferenceServerSettings(
        model_path=str(output), backend="local", device="cpu", local_files_only=True,
    ))
    with pytest.raises(ValueError, match="requires dtype='float32'"):
        load_inference_model(InferenceServerSettings(
            model_path=str(output), backend="local", device="cpu", dtype="float16",
            local_files_only=True,
        ))
    batch = {"input_ids": torch.tensor([[4, 5, 6]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    with torch.no_grad():
        expected = result.model(**batch).logits
        torch.testing.assert_close(loaded(**batch).logits, expected)
        torch.testing.assert_close(served(**batch).logits, expected)
    report = run_regression_suite(str(source), str(output), {"tasks": [{
        "name": "quality", "task": "completion", "data": heldout,
        "options": {"device": "cpu", "local_files_only": True, "max_length": 16},
        "metrics": {"completion_nll": {"max_drop": .5, "higher_is_better": False}},
    }]})
    assert report["accepted"], report


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_qat_evaluation_auto_dtype_preserves_fp32(tmp_path, device):
    import torch

    from saddlellm.evaluation.MultipleChoiceEvaluator import (
        MultipleChoiceEvaluationRequest,
        _select_runtime,
    )

    (tmp_path / "quantization_manifest.json").write_text(json.dumps({
        "method": "qat", "backend": "portable-int8-weight-only",
    }), encoding="utf-8")
    actual_device, dtype = _select_runtime(MultipleChoiceEvaluationRequest(
        model=str(tmp_path), data="unused.jsonl", device=device,
    ))
    assert actual_device == ("cpu" if device == "auto" else device)
    assert dtype is torch.float32
