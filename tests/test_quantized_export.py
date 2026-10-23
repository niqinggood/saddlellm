"""Quantized releases retain their evaluated storage and numeric contract."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from saddlellm.evaluation.ModelIdentity import compute_model_identity
from saddlellm.runtime.ModelExporter import ModelExporter, ModelExportRequest


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _package(path, method="qat", *, native=False):
    path.mkdir()
    _json(path / ("saddle_config.json" if native else "config.json"), {"model_type": "gpt2"})
    _json(path / "tokenizer.json", {"test": "unchanged-tokenizer"})
    _json(path / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
    if method in {"awq", "gptq"}:
        manifest = {"method": method, "backend": "llm-compressor", "format": "compressed-tensors", "self_contained": True}
        _json(path / "config.json", {
            "model_type": "llama",
            "quantization_config": {
                "quant_method": "compressed-tensors", "quantization_status": "compressed",
                "format": "pack-quantized", "config_groups": {"group_0": {"targets": ["Linear"]}},
            },
        })
        (path / "model.safetensors").write_bytes(b"packed-integer-tensors")
    elif method == "qat":
        manifest = {
            "schema_version": 1, "method": "qat", "backend": "portable-int8-weight-only",
            "precision": "int8", "activation_precision": "float32",
            "weight_scheme": "symmetric_per_output_channel", "self_contained": True,
            "state_file": "qat_model.pt", "modules": [{"name": "projection"}],
        }
        (path / "qat_model.pt").write_bytes(b"int8-weights-fp32-scales")
    else:
        manifest = {"method": "dynamic", "precision": "int8", "state_file": "quantized_model.pt"}
        (path / "quantized_model.pt").write_bytes(b"legacy-dynamic-int8")
    _json(path / "quantization_manifest.json", manifest)
    return path


def _gate(source):
    return {"accepted": True, "candidate_model_identity": compute_model_identity(source)}


@pytest.mark.parametrize("method,native", [("awq", False), ("gptq", False), ("qat", False), ("qat", True)])
def test_packaged_export_preserves_bytes_and_gate_identity(tmp_path, monkeypatch, method, native):
    source = _package(tmp_path / "source", method, native=native)
    expected = {path.name: path.read_bytes() for path in source.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("packed release export must never materialize or merge a model")

    monkeypatch.setattr(ModelExporter, "_materialize_model", forbidden)
    monkeypatch.setattr(ModelExporter, "_merge_adapter", forbidden)
    result = ModelExporter.export(
        ModelExportRequest(
            str(source), str(tmp_path / "release"), require_gate=True,
            format="saddle" if native else "hf", merge_lora=not native,
        ),
        gate=_gate(source),
    )
    destination = Path(result.model_path)
    for name, data in expected.items():
        assert (destination / name).read_bytes() == data
    release = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert release["source_type"] == "quantized_checkpoint"
    assert release["quantization"]["packed_weights_preserved"]
    assert release["gate"]["accepted"]
    assert release["gate_binding"]["status"] == "matched"
    assert release["source_model_identity"]["fingerprint"] == release["model_identity"]["fingerprint"]
    assert "load_causal_lm" in (destination / "SADDLELLM_RELEASE.md").read_text(encoding="utf-8")


def test_legacy_dynamic_export_keeps_local_copy_contract(tmp_path, monkeypatch):
    source = _package(tmp_path / "source", "dynamic")
    # Legacy export accepts a local dynamic state without interpreting it.
    (source / "tokenizer.json").unlink()
    (source / "tokenizer_config.json").unlink()
    monkeypatch.setattr(ModelExporter, "_materialize_model", lambda *args: pytest.fail("must copy legacy dynamic weights"))
    result = ModelExporter.export(ModelExportRequest(str(source), str(tmp_path / "release")))
    assert result.source_type == "local_checkpoint"
    assert (Path(result.model_path) / "quantized_model.pt").read_bytes() == b"legacy-dynamic-int8"


def test_indexed_cuda_materialization_keeps_exact_device(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    request = ModelExportRequest("source", "release", device="cuda:1", dtype="float32")
    ModelExporter._validate_request(request)
    kwargs, device = ModelExporter._load_kwargs(request)
    assert device == "cuda:1"
    assert kwargs["device_map"] == {"": "cuda:1"}
    assert kwargs["dtype"] == torch.float32


def test_indexed_cuda_materialization_rejects_unavailable_device(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="outside the available"):
        ModelExporter._load_kwargs(ModelExportRequest("source", "release", device="cuda:1"))


@pytest.mark.parametrize("method,overrides,match", [
    ("awq", {"format": "saddle", "merge_lora": False}, "requires saddle_config"),
    ("qat", {"dtype": "float16"}, "device/dtype conversion"),
    ("qat", {"device": "cuda"}, "device/dtype conversion"),
    ("gptq", {"dtype": "float32"}, "device/dtype conversion"),
    ("awq", {"device": "cpu"}, "device/dtype conversion"),
])
def test_unsupported_conversion_preserves_existing_destination(tmp_path, method, overrides, match):
    source = _package(tmp_path / "source", method)
    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        ModelExporter.export(ModelExportRequest(str(source), str(destination), overwrite=True, **overrides))
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("change,match", [
    ("adapter", "does not support PEFT"), ("state", "state file"),
    ("nonselfcontained", "self-contained supported"), ("tokenizer", "tokenizer files"),
])
def test_incomplete_qat_fails_before_destination_creation(tmp_path, change, match):
    source = _package(tmp_path / "source")
    if change == "adapter":
        _json(source / "adapter_config.json", {})
    elif change == "state":
        (source / "qat_model.pt").unlink()
    elif change == "tokenizer":
        (source / "tokenizer.json").unlink()
    else:
        manifest = json.loads((source / "quantization_manifest.json").read_text(encoding="utf-8"))
        manifest["self_contained"] = False
        _json(source / "quantization_manifest.json", manifest)
    destination = tmp_path / "release"
    with pytest.raises(ValueError, match=match):
        ModelExporter.export(ModelExportRequest(str(source), str(destination)))
    assert not destination.exists()


def test_ptq_missing_declared_shard_fails_before_overwrite(tmp_path):
    source = _package(tmp_path / "source", "gptq")
    _json(source / "model.safetensors.index.json", {"weight_map": {"layer.weight_packed": "missing.safetensors"}})
    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="missing a declared"):
        ModelExporter.export(ModelExportRequest(str(source), str(destination), overwrite=True))
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_ptq_dense_configuration_cannot_claim_packed_export(tmp_path):
    source = _package(tmp_path / "source", "awq")
    _json(source / "config.json", {"model_type": "llama"})
    with pytest.raises(ValueError, match="compressed integer"):
        ModelExporter.export(ModelExportRequest(str(source), str(tmp_path / "release")))
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("overrides,match", [({"format": "hf", "merge_lora": False}, "cannot be exported"), ({"format": "saddle"}, "merge_lora")])
def test_native_qat_requires_native_format_without_adapter_merge(tmp_path, overrides, match):
    source = _package(tmp_path / "source", native=True)
    with pytest.raises(ValueError, match=match):
        ModelExporter.export(ModelExportRequest(str(source), str(tmp_path / "release"), **overrides))
    assert not (tmp_path / "release").exists()


def test_gate_rejects_changed_quantization_metadata_without_overwriting(tmp_path):
    source = _package(tmp_path / "source", "gptq")
    evidence = _gate(source)
    manifest = json.loads((source / "quantization_manifest.json").read_text(encoding="utf-8"))
    manifest["bits"] = 8
    _json(source / "quantization_manifest.json", manifest)
    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        ModelExporter.export(
            ModelExportRequest(str(source), str(destination), overwrite=True, require_gate=True), gate=evidence,
        )
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("method,native", [("awq", False), ("gptq", False), ("qat", False), ("qat", True)])
@pytest.mark.parametrize("changed", ["source", "copied-state"])
def test_quantized_export_rejects_mid_export_state_changes(
    tmp_path, monkeypatch, method, native, changed,
):
    source = _package(tmp_path / "source", method, native=native)
    evidence = _gate(source)
    destination = tmp_path / "release"
    destination.mkdir()
    (destination / "keep.txt").write_bytes(b"previous-release")
    state_name = "qat_model.pt" if method == "qat" else "model.safetensors"
    original_copy = ModelExporter._copy_local_release

    def copy(current_source, staged):
        original_copy(current_source, staged)
        changed_root = current_source if changed == "source" else staged
        (changed_root / state_name).write_bytes(b"changed-integer-state")

    monkeypatch.setattr(ModelExporter, "_copy_local_release", copy)
    with pytest.raises(ValueError, match="content|weights"):
        ModelExporter.export(
            ModelExportRequest(str(source), str(destination), require_gate=True, overwrite=True,
                               format="saddle" if native else "hf", merge_lora=not native),
            gate=evidence,
        )
    assert list(destination.iterdir()) == [destination / "keep.txt"]
    assert (destination / "keep.txt").read_bytes() == b"previous-release"
    assert not list(tmp_path.glob(".release.export-*"))


@pytest.mark.parametrize("native", [False, True])
def test_real_qat_export_reloads_with_matching_logits(tmp_path, native):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from saddlellm.compression.QuantizationAwareTraining import QATConfig, run_qat
    from saddlellm.models.ModelLoader import load_causal_lm

    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "red": 4, "blue": 5, "green": 6, "white": 7}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    if native:
        from saddlellm.models.ModelBlueprint import (
            AttentionBlueprint,
            FFNBlueprint,
            ModelBlueprint,
        )

        model = ModelBlueprint(
            name="qat-export", hidden_size=16, vocab_size=8, num_layers=1, max_position_embeddings=32,
            attention=AttentionBlueprint(kind="gqa", backend="eager", num_heads=4, num_kv_heads=2),
            ffn=FFNBlueprint(kind="swiglu", intermediate_size=32),
        ).build_model()
    else:
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=8, n_embd=16, n_layer=1, n_head=2, n_positions=32,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        ))
    train = tmp_path / "train.jsonl"
    train.write_text('{"input_ids":[1,4,5,2]}\n{"input_ids":[1,5,4,2]}\n', encoding="utf-8")
    source = tmp_path / "qat"
    trained = run_qat(
        "in-memory", source,
        QATConfig(enabled=True, train_data=str(train), validation_split=0, max_steps=1, device="cpu"),
        model=model, tokenizer=tokenizer,
    )
    result = ModelExporter.export(
        ModelExportRequest(str(source), str(tmp_path / "release"), format="saddle" if native else "hf", merge_lora=not native),
    )
    loaded = load_causal_lm(result.model_path, device="cpu", local_files_only=True)
    inputs = torch.tensor([[1, 4, 5]])
    with torch.no_grad():
        before = trained.model(input_ids=inputs, **({"return_logits": True} if native else {})).logits
        after = loaded(input_ids=inputs, **({"return_logits": True} if native else {})).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    state = torch.load(Path(result.model_path) / "qat_model.pt", weights_only=True)
    assert any(value.dtype == torch.int8 for value in state.values())


@pytest.mark.skipif(importlib.util.find_spec("llmcompressor") is None, reason="optional real PTQ backend is not installed")
@pytest.mark.parametrize("method", ["awq", "gptq"])
def test_real_ptq_export_preserves_integer_storage_and_reloads(tmp_path, method):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from safetensors.torch import load_file

    from saddlellm.compression.CalibratedPTQ import PTQConfig, run_calibrated_ptq
    from saddlellm.models.ModelLoader import load_model_and_tokenizer

    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[UNK]": 0, "[PAD]": 1, "one": 2, "two": 3, "three": 4, "four": 5}, unk_token="[UNK]",
    ))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        model_input_names=["input_ids", "attention_mask"],
    )
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        pad_token_id=1, tie_word_embeddings=False,
    ))
    source = tmp_path / "ptq"
    packed = run_calibrated_ptq(
        None, source, PTQConfig(
            enabled=True, method=method, bits=4, group_size=8, device="cpu", local_files_only=True,
            max_seq_length=8, calibration_data=["one two three four one two", "four three two one four three"],
        ), model=model, tokenizer=tokenizer,
    )
    result = ModelExporter.export(
        ModelExportRequest(str(source), str(tmp_path / "release"), require_gate=True), gate=_gate(source),
    )
    released, tokenizer = load_model_and_tokenizer(result.model_path, device="cpu", local_files_only=True)
    inputs = tokenizer("one two three", return_tensors="pt")
    with torch.no_grad():
        torch.testing.assert_close(released(**inputs).logits, packed.model(**inputs).logits, rtol=0, atol=0)
    before = load_file(str(source / "model.safetensors"))
    after = load_file(str(Path(result.model_path) / "model.safetensors"))
    assert any(value.dtype in {torch.int8, torch.int32} for value in after.values())
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
