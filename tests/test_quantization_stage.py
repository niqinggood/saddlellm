"""Optional compression participates in checkpoint selection and resume safety."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from saddlellm.compression.QuantizationPipeline import (
    QuantizationConfig,
    QuantizationResult,
)
from saddlellm.compression.QuantizationStage import (
    QuantizationStageConfig,
    latest_language_model_path,
    validate_quantization_stage,
)
from saddlellm.framework import normalized_run_fingerprint
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator


def _config(tmp_path, *, stages=None, enabled=True):
    return {
        "model": {"name_or_path": "unloaded-model"},
        "stages": stages or ["quantize"],
        "quantization": {"ptq": {
            "enabled": enabled, "method": "gptq", "group_size": 8,
            "calibration_data": ["one two three"], "device": "cpu", "local_files_only": True,
        }},
        "distributed": {"bf16": False, "fp16": False},
        "logging": {"backend": "local", "output_dir": str(tmp_path / "run")},
        "eval": {"enabled": False},
    }


def test_stage_defaults_are_typed_disabled_and_keep_upstream_selection(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationPipeline as function

    raw = _config(tmp_path, enabled=False)
    raw.pop("quantization")
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        assert isinstance(runner.config.quantization, QuantizationStageConfig)
        assert not runner.config.quantization.ptq.enabled
        assert not runner.config.quantization.qat.enabled
        monkeypatch.setattr(function, "apply_quantization", lambda *a, **k: pytest.fail("disabled conversion"))
        runner._stage_results["sft"] = {"model_path": "trained-model"}
        runner._run_quantization_stage()
        assert runner._stage_results["quantize"]["status"] == "skipped"
        assert "model_path" not in runner._stage_results["quantize"]
        assert latest_language_model_path(runner) == "trained-model"
        assert not (tmp_path / "run" / "quantized").exists()
    finally:
        runner._close_logging_handlers()


@pytest.mark.parametrize("method", ["awq", "gptq"])
def test_stage_delegates_function_config_and_prefers_latest_executed_checkpoint(tmp_path, monkeypatch, method):
    import saddlellm.compression.QuantizationPipeline as function

    raw = _config(tmp_path)
    raw["quantization"]["ptq"]["method"] = method
    runner = TrainingOrchestrator.from_dict(raw)
    seen = {}

    def convert(source, **kwargs):
        seen.update(source=source, **kwargs)
        return QuantizationResult(None, None, {"status": "completed", "route": "ptq"})

    monkeypatch.setattr(function, "apply_quantization", convert)
    try:
        runner.config.stages = ["sft", "rlhf", "preference", "quantize"]
        runner._stage_results.update({
            "sft": {"model_path": "sft-model"}, "rlhf": {"model_path": "rlhf-model"},
            "preference": {"model_path": "latest-model"},
        })
        runner._model, runner._tokenizer = object(), object()
        runner._run_quantization_stage()
        assert seen["source"] == "latest-model"
        assert type(seen["config"]) is QuantizationConfig
        assert seen["config"].ptq.method == method
        assert seen["output_dir"] == str(tmp_path / "run" / "quantized")
        assert seen["dry_run"] is False
        assert runner._model is None and runner._tokenizer is None
        assert latest_language_model_path(runner) == str((tmp_path / "run" / "quantized").resolve())
    finally:
        runner._close_logging_handlers()


def test_explicit_quantization_checkpoint_overrides_upstream(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationPipeline as function

    raw = _config(tmp_path)
    raw["quantization"]["model_path"] = "explicit-source"
    raw["quantization"]["output_dir"] = str(tmp_path / "chosen-output")
    seen = []
    monkeypatch.setattr(function, "apply_quantization", lambda source, **kwargs: (
        seen.append(source) or QuantizationResult(None, None, {"status": "completed"})
    ))
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        runner._stage_results["sft"] = {"model_path": "upstream-source"}
        runner._run_quantization_stage()
        assert seen == ["explicit-source"]
        assert runner._stage_results["quantize"]["model_path"] == str((tmp_path / "chosen-output").resolve())
    finally:
        runner._close_logging_handlers()


@pytest.mark.parametrize("mode", ["dry_run", "preflight_only"])
def test_planned_quantization_eval_and_export_never_load_model(tmp_path, monkeypatch, mode):
    import saddlellm.compression.CalibratedPTQ as backend
    from saddlellm.evaluation.LLModelEvalute import Evaluator

    raw = _config(tmp_path, stages=["quantize", "eval", "export"])
    raw["training"] = {mode: True}
    raw["eval"] = {"enabled": True}
    raw["export"] = {"enabled": True}
    monkeypatch.setattr(backend, "run_calibrated_ptq", lambda *a, **k: pytest.fail("planning loaded backend"))
    monkeypatch.setattr(Evaluator, "evaluate", lambda *a, **k: pytest.fail("planning evaluated model"))
    results = TrainingOrchestrator.from_dict(raw).run()
    output = str((tmp_path / "run" / "quantized").resolve())
    assert results["quantize"]["status"] == results["eval"]["status"] == results["export"]["status"] == "planned"
    assert results["quantize"]["model_loaded"] is False
    assert results["eval"]["model"] == results["export"]["model_path"] == output
    assert results["export"]["merge_lora"] is False
    assert not Path(output).exists()
    assert not (tmp_path / "run" / "release").exists()


@pytest.mark.parametrize("stages,match", [
    (["eval"], "requires a quantize stage"),
    (["quantize", "sft"], "after sft"),
    (["quantize", "preference"], "after preference"),
    (["quantize", "rlhf"], "after rlhf"),
    (["quantize", "pretrain"], "after pretrain"),
    (["eval", "quantize"], "before eval"),
    (["export", "quantize"], "before export"),
])
def test_enabled_stage_enforces_training_and_release_order(tmp_path, stages, match):
    config = TrainingOrchestrator._parse_config(_config(tmp_path, stages=stages))
    with pytest.raises(ValueError, match=match):
        validate_quantization_stage(config)


@pytest.mark.parametrize("changes,match", [
    ({"ptq": {"enabled": "false"}}, "booleans"),
    ({"ptq": {"enabled": True}, "qat": {"enabled": True}}, "either ptq or qat"),
    ({"merge_lora": "false"}, "must be boolean"),
    ({"model_path": " "}, "nonempty path"),
])
def test_invalid_config_fails_static_validation(tmp_path, changes, match):
    raw = _config(tmp_path)
    raw["quantization"].update(changes)
    report = TrainingConfigValidator.validate(raw, inspect_data=False)
    assert not report.valid
    assert any(match in issue for issue in report.issues), report.issues


def test_eval_adapter_cannot_overlay_quantized_checkpoint(tmp_path):
    raw = _config(tmp_path, stages=["quantize", "eval"])
    raw["eval"].update(enabled=True, adapter="adapter")
    with pytest.raises(ValueError, match="eval.adapter"):
        TrainingOrchestrator.from_dict(raw)


def test_quantization_static_validator_rejects_distributed_stage(tmp_path):
    raw = _config(tmp_path)
    raw["distributed"].update(strategy="ddp", num_gpus=2)
    report = TrainingConfigValidator.validate(raw, inspect_data=False)
    assert not report.valid
    assert any("quantize" in issue and "ddp" in issue for issue in report.issues), report.issues


def test_explicit_dag_requires_dependency_for_quantization_handoff(tmp_path):
    raw = _config(tmp_path, stages=["quantize", "eval"])
    raw["pipeline"] = {"dependencies": {"quantize": [], "eval": []}}
    with pytest.raises(ValueError, match="eval depend on quantize"):
        TrainingOrchestrator.from_dict(raw)


def test_explicit_dag_accepts_transitive_handoffs(tmp_path):
    raw = _config(tmp_path, stages=["export", "eval", "quantize"])
    raw["pipeline"] = {"dependencies": {"quantize": [], "eval": ["quantize"], "export": ["eval"]}}
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        assert runner.config.stages == ["quantize", "eval", "export"]
    finally:
        runner._close_logging_handlers()


def test_unused_quantization_config_preserves_legacy_resume_fingerprint(tmp_path):
    raw = _config(tmp_path, stages=["eval"], enabled=False)
    baseline = asdict(TrainingOrchestrator._parse_config(raw))
    baseline.pop("quantization")
    raw["quantization"].update(output_dir="unused-output", model_path="unused-model")
    raw["quantization"]["ptq"]["method"] = "awq"
    assert normalized_run_fingerprint(TrainingOrchestrator._parse_config(raw)) == normalized_run_fingerprint(baseline)
    raw["stages"] = ["quantize"]
    first = normalized_run_fingerprint(TrainingOrchestrator._parse_config(raw))
    raw["quantization"]["ptq"]["enabled"] = True
    assert first != normalized_run_fingerprint(TrainingOrchestrator._parse_config(raw))


def test_completed_quantization_resumes_without_overwriting_output(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationPipeline as function

    raw = _config(tmp_path)
    calls = []

    def convert(source, *, output_dir, **kwargs):
        calls.append(source)
        destination = Path(output_dir)
        destination.mkdir()
        (destination / "already-quantized.json").write_text("{}", encoding="utf-8")
        return QuantizationResult(None, None, {"status": "completed"})

    monkeypatch.setattr(function, "apply_quantization", convert)
    first = TrainingOrchestrator.from_dict(raw).run()
    raw["pipeline"] = {"resume": True}
    resumed = TrainingOrchestrator.from_dict(raw).run()
    assert calls == ["unloaded-model"]
    assert resumed["quantize"] == first["quantize"]
    assert Path(resumed["quantize"]["model_path"], "already-quantized.json").is_file()


def test_adapter_is_merged_to_separate_float_checkpoint_before_conversion(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationPipeline as function
    from saddlellm.runtime.ModelExporter import ModelExporter

    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    raw = _config(tmp_path)
    raw["quantization"]["model_path"] = str(source)
    events = []
    def merge(request):
        events.append(("merge", request))
        Path(request.output_dir).mkdir()

    monkeypatch.setattr(ModelExporter, "export", merge)

    def convert(source, **kwargs):
        events.append(("convert", source))
        return QuantizationResult(None, None, {"status": "completed"})

    monkeypatch.setattr(function, "apply_quantization", convert)
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        runner._run_quantization_stage()
        prepared = str((tmp_path / "run" / "quantized-float").resolve())
        assert [event[0] for event in events] == ["merge", "convert"]
        assert events[0][1].merge_lora is True
        assert events[0][1].dtype == "float32"
        assert events[0][1].output_dir == events[1][1] == prepared
        assert runner._stage_results["quantize"]["source_model"] == str(source)
    finally:
        runner._close_logging_handlers()
    raw["quantization"]["merge_lora"] = False
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        with pytest.raises(ValueError, match="merge_lora"):
            runner._run_quantization_stage()
        assert len(events) == 2
    finally:
        runner._close_logging_handlers()


def _adapter_preparation_fixture(tmp_path, monkeypatch):
    from saddlellm.runtime.ModelExporter import ModelExporter

    base, source = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    source.mkdir()
    (base / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"base-inference-weights")
    (source / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": str(base),
    }), encoding="utf-8")
    (source / "adapter_model.safetensors").write_bytes(b"adapter-inference-weights")
    exports = []

    def merge(request):
        exports.append(request)
        merged = Path(request.output_dir)
        merged.mkdir()
        (merged / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
        (merged / "model.safetensors").write_bytes(b"merged-inference-weights")

    monkeypatch.setattr(ModelExporter, "export", merge)
    raw = _config(tmp_path)
    raw["quantization"]["model_path"] = str(source)
    return raw, base, source, exports


def test_failed_conversion_retries_using_verified_adapter_preparation(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationPipeline as function

    raw, _base, _adapter, exports = _adapter_preparation_fixture(tmp_path, monkeypatch)
    conversions = []

    def convert(source, **kwargs):
        conversions.append(source)
        if len(conversions) == 1:
            raise RuntimeError("calibration interrupted")
        return QuantizationResult(None, None, {"status": "completed"})

    monkeypatch.setattr(function, "apply_quantization", convert)
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        with pytest.raises(RuntimeError, match="calibration interrupted"):
            runner._run_quantization_stage()
        runner._run_quantization_stage()
        assert len(exports) == 1
        assert conversions == [str((tmp_path / "run" / "quantized-float").resolve())] * 2
        assert runner._stage_results["quantize"]["status"] == "completed"
    finally:
        runner._close_logging_handlers()


def test_adapter_preparation_never_writes_into_its_base_checkpoint(tmp_path, monkeypatch):
    import saddlellm.compression.QuantizationPipeline as function

    raw, base, _adapter, exports = _adapter_preparation_fixture(tmp_path, monkeypatch)
    before = (base / "model.safetensors").read_bytes()
    output = base / "quantized"
    raw["quantization"]["output_dir"] = str(output)
    monkeypatch.setattr(function, "apply_quantization", lambda *a, **k: pytest.fail("conversion touched base checkpoint"))
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        with pytest.raises(ValueError, match="adapter's base checkpoint"):
            runner._run_quantization_stage()
        assert not exports
        assert not output.exists() and not (base / "quantized-float").exists()
        assert (base / "model.safetensors").read_bytes() == before
    finally:
        runner._close_logging_handlers()


@pytest.mark.parametrize("tamper", ["base", "adapter", "merged", "record_missing", "record_invalid"])
def test_adapter_retry_rejects_changed_or_unverified_preparation(tmp_path, monkeypatch, tamper):
    import saddlellm.compression.QuantizationPipeline as function

    raw, base, adapter, exports = _adapter_preparation_fixture(tmp_path, monkeypatch)
    conversions = []

    def convert(source, **kwargs):
        conversions.append(source)
        raise RuntimeError("calibration interrupted")

    monkeypatch.setattr(function, "apply_quantization", convert)
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        with pytest.raises(RuntimeError, match="calibration interrupted"):
            runner._run_quantization_stage()
        merged = tmp_path / "run" / "quantized-float"
        marker = merged / "quantization_preparation.json"
        if tamper == "record_missing":
            marker.unlink()
        elif tamper == "record_invalid":
            marker.write_text("{}", encoding="utf-8")
        else:
            target = {"base": base / "model.safetensors", "adapter": adapter / "adapter_model.safetensors",
                      "merged": merged / "model.safetensors"}[tamper]
            target.write_bytes(b"different-inference-weights")
        with pytest.raises(ValueError, match="preparation record|contents changed"):
            runner._run_quantization_stage()
        assert len(exports) == len(conversions) == 1
    finally:
        runner._close_logging_handlers()


@pytest.mark.parametrize("existing_package", [False, True])
def test_export_defaults_to_no_adapter_merge_for_quantized_path_alias(tmp_path, monkeypatch, existing_package):
    source = tmp_path / "quantized"
    source.mkdir()
    if existing_package:
        (source / "quantization_manifest.json").write_text(json.dumps({
            "method": "qat", "backend": "portable-int8-weight-only",
        }), encoding="utf-8")
    raw = _config(tmp_path, stages=["export"], enabled=False)
    raw["training"] = {"dry_run": True}
    raw["export"] = {"enabled": True, "model_path": "./quantized"}
    monkeypatch.chdir(tmp_path)
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        if not existing_package:
            runner._stage_results["quantize"] = {"status": "completed", "model_path": str(source)}
        runner._run_export_stage()
        assert runner._stage_results["export"]["merge_lora"] is False
    finally:
        runner._close_logging_handlers()


def test_legacy_qat_evaluator_uses_fp32_even_when_cuda_selected(tmp_path, monkeypatch):
    import torch

    import saddlellm.models.ModelLoader as loading
    from saddlellm.evaluation.LLModelEvalute import Evaluator

    (tmp_path / "quantization_manifest.json").write_text(json.dumps({
        "method": "qat", "backend": "portable-int8-weight-only",
    }), encoding="utf-8")
    seen = {}
    model = SimpleNamespace(eval=lambda: None)
    tokenizer = SimpleNamespace(pad_token="<pad>")

    def load(path, **kwargs):
        seen.update(kwargs)
        return model, tokenizer

    monkeypatch.setattr(loading, "load_model_and_tokenizer", load)
    assert Evaluator()._load_model(str(tmp_path), "generation", "cuda", False) == (model, tokenizer)
    assert seen["dtype"] is torch.float32
    assert seen["device"] == "cuda"


def _tiny_checkpoint(tmp_path, architecture):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    source = tmp_path / "source"
    vocabulary = {word: index for index, word in enumerate([
        "<pad>", "<bos>", "<eos>", "<unk>", "one", "two", "three", "four",
    ])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
        unk_token="<unk>", model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(source)
    torch.manual_seed(5)
    if architecture == "hf":
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=8, n_positions=32, n_embd=8, n_layer=1, n_head=1,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        ))
    elif architecture == "native":
        from saddlellm.models.ModelBlueprint import (
            AttentionBlueprint,
            FFNBlueprint,
            ModelBlueprint,
        )

        model = ModelBlueprint(
            name="stage-qat", hidden_size=8, vocab_size=8, num_layers=1, max_position_embeddings=32,
            attention=AttentionBlueprint(num_heads=2, num_kv_heads=2), ffn=FFNBlueprint(intermediate_size=16),
        ).build_model()
    else:
        from transformers import LlamaConfig, LlamaForCausalLM

        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False,
        ))
    model.save_pretrained(source)
    return source, tokenizer


@pytest.mark.parametrize("architecture", ["hf", "native"])
def test_real_cpu_qat_stage_regression_export_reload_and_resume(tmp_path, architecture):
    import torch

    from saddlellm.models.ModelLoader import load_causal_lm

    source, tokenizer = _tiny_checkpoint(tmp_path, architecture)
    train = tmp_path / "train.jsonl"
    train.write_text('{"text":"one two three"}\n{"text":"two three four"}\n', encoding="utf-8")
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text('{"instruction":"four","output":"one"}\n', encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"tasks": [{
        "name": "heldout", "task": "completion", "data": str(heldout),
        "options": {"device": "cpu", "local_files_only": True, "max_length": 16},
        "metrics": {"completion_nll": {"max_drop": .5, "higher_is_better": False}},
    }]}), encoding="utf-8")
    raw = _config(tmp_path, stages=["quantize", "eval", "export"])
    raw["model"]["name_or_path"] = str(source)
    raw["quantization"] = {"qat": {
        "enabled": True, "train_data": str(train), "validation_split": 0,
        "max_steps": 1, "max_seq_length": 16, "device": "cpu", "local_files_only": True,
    }}
    raw["eval"] = {"enabled": True, "regression_suite": str(suite)}
    raw["export"] = {
        "enabled": True, "require_gate": True, "local_files_only": True,
        "format": "saddle" if architecture == "native" else "hf",
    }
    results = TrainingOrchestrator.from_dict(raw).run()
    assert results["quantize"]["actual_steps"] == 1
    assert results["eval"]["accepted"] is True
    assert results["export"]["status"] == "completed"
    quantized = Path(results["quantize"]["model_path"])
    release = Path(results["export"]["output_dir"])
    assert results["eval"]["model"] == str(quantized)
    assert (release / "quantization_manifest.json").read_bytes() == (quantized / "quantization_manifest.json").read_bytes()
    model = load_causal_lm(quantized, device="cpu", local_files_only=True)
    exported = load_causal_lm(release, device="cpu", local_files_only=True)
    batch = tokenizer("one two three", return_tensors="pt")
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).logits, exported(**batch).logits)
    assert any(buffer.dtype is torch.int8 for buffer in exported.buffers())
    state = json.loads((tmp_path / "run" / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state["stages"]["quantize"]["status"] == "completed"
    raw["pipeline"] = {"resume": True}
    resumed = TrainingOrchestrator.from_dict(raw).run()
    assert resumed["quantize"] == json.loads(json.dumps(results["quantize"]))
    assert resumed["eval"] == json.loads(json.dumps(results["eval"]))


def test_real_lora_adapter_merge_qat_stage_export_and_reload(tmp_path):
    import torch
    from safetensors.torch import load_file

    from saddlellm.evaluation.ModelIdentity import (
        compute_model_identity,
        model_identities_match,
    )
    from saddlellm.models.ModelLoader import load_causal_lm
    from saddlellm.training.PostTrainingCompatibility import (
        post_training_runtime_report,
        stabilize_peft_optional_backends,
    )

    if not post_training_runtime_report()["compatible"]:
        pytest.skip("requires requirements-posttrain.txt")
    stabilize_peft_optional_backends()
    from peft import LoraConfig, get_peft_model

    base, tokenizer = _tiny_checkpoint(tmp_path, "hf")
    model = get_peft_model(
        load_causal_lm(base, device="cpu", dtype="float32", local_files_only=True),
        LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["c_attn"]),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    batch = tokenizer("one two three", return_tensors="pt")
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    model.train()
    model(**batch, labels=batch["input_ids"]).loss.backward()
    optimizer.step()
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, trainable))
    model.eval()
    with torch.no_grad():
        expected_merged_logits = model(**batch).logits
    adapter = tmp_path / "trained-adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    train = tmp_path / "train.jsonl"
    train.write_text('{"text":"one two three"}\n{"text":"two three four"}\n', encoding="utf-8")
    raw = _config(tmp_path, stages=["quantize", "export"])
    raw["model"]["name_or_path"] = str(base)
    raw["quantization"] = {
        "model_path": str(adapter),
        "qat": {
            "enabled": True, "train_data": str(train), "validation_split": 0,
            "max_steps": 1, "max_seq_length": 16, "device": "cpu", "local_files_only": True,
        },
    }
    raw["export"] = {"enabled": True, "local_files_only": True}
    results = TrainingOrchestrator.from_dict(raw).run()
    assert results["quantize"]["actual_steps"] == 1
    prepared = Path(results["quantize"]["prepared_source"])
    assert prepared == (tmp_path / "run" / "quantized-float").resolve()
    assert not (prepared / "adapter_config.json").exists()
    preparation = json.loads((prepared / "quantization_preparation.json").read_text(encoding="utf-8"))
    assert preparation["dtype"] == "float32"
    assert model_identities_match(preparation["source_model_identity"], compute_model_identity(adapter))
    assert model_identities_match(preparation["model_identity"], compute_model_identity(prepared))
    float_weights = load_file(str(prepared / "model.safetensors"))
    assert float_weights
    assert all(tensor.dtype is torch.float32 for tensor in float_weights.values() if tensor.is_floating_point())
    merged = load_causal_lm(prepared, device="cpu", dtype="float32", local_files_only=True)
    quantized = load_causal_lm(results["quantize"]["model_path"], device="cpu", local_files_only=True)
    exported = load_causal_lm(results["export"]["output_dir"], device="cpu", local_files_only=True)
    with torch.no_grad():
        torch.testing.assert_close(merged(**batch).logits, expected_merged_logits)
        torch.testing.assert_close(quantized(**batch).logits, exported(**batch).logits)
    assert any(buffer.dtype is torch.int8 for buffer in exported.buffers())


@pytest.mark.skipif(importlib.util.find_spec("llmcompressor") is None, reason="optional real PTQ backend is not installed")
def test_real_cpu_gptq_stage_export_preserves_packed_weights_and_reload(tmp_path):
    import torch
    from safetensors.torch import load_file

    from saddlellm.models.ModelLoader import load_causal_lm

    source, tokenizer = _tiny_checkpoint(tmp_path, "llama")
    raw = _config(tmp_path, stages=["quantize", "export"])
    raw["model"]["name_or_path"] = str(source)
    raw["quantization"]["ptq"].update(bits=4, max_seq_length=8, max_samples=1)
    raw["export"] = {"enabled": True, "local_files_only": True}
    results = TrainingOrchestrator.from_dict(raw).run()
    assert results["quantize"]["status"] == results["export"]["status"] == "completed"
    quantized = Path(results["quantize"]["model_path"])
    release = Path(results["export"]["output_dir"])
    assert (release / "model.safetensors").read_bytes() == (quantized / "model.safetensors").read_bytes()
    packed = load_file(str(release / "model.safetensors"))
    assert any(key.endswith(".weight_packed") and tensor.dtype is torch.int32 for key, tensor in packed.items())
    original = load_causal_lm(quantized, device="cpu", local_files_only=True)
    loaded = load_causal_lm(release, device="cpu", local_files_only=True)
    batch = tokenizer("one two three", return_tensors="pt")
    with torch.no_grad():
        torch.testing.assert_close(original(**batch).logits, loaded(**batch).logits)
