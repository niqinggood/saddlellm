"""Portable QAT artifact -> explicit W8A16 CUDA execution and reload parity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from saddlellm.compression.QuantizationAwareTraining import QATConfig, load_qat, run_qat
from saddlellm.models.ModelLoader import load_causal_lm


class Tokenizer:
    pad_token_id, bos_token_id, eos_token_id, unk_token_id = 0, 1, 2, 3

    def get_vocab(self):
        return {str(index): index for index in range(19)}

    def save_pretrained(self, path):
        (Path(path) / "tokenizer_stub.json").write_text("{}", encoding="utf-8")


def artifact_hashes(path):
    return {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in path.iterdir() if item.is_file()}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    from transformers import GPT2Config, GPT2LMHeadModel

    from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

    root = tmp_path_factory.mktemp("qat-cuda")
    train = root / "train.jsonl"
    train.write_text('\n'.join(json.dumps({"input_ids": row}) for row in [[1, 4, 6, 2], [1, 5, 7, 2]]) + '\n', encoding="utf-8")
    torch.manual_seed(941)
    models = {
        "native": SaddleForCausalLM(SaddleModelConfig(vocab_size=19, hidden_size=16, intermediate_size=32,
                                                      num_hidden_layers=1, num_attention_heads=2,
                                                      num_key_value_heads=1, max_position_embeddings=32,
                                                      attention_backend="eager")),
        "hf": GPT2LMHeadModel(GPT2Config(vocab_size=19, n_embd=16, n_layer=1, n_head=2, n_positions=32,
                                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                                        eos_token_id=2, pad_token_id=0)),
    }
    paths = {}
    for name, model in models.items():
        output = root / name
        result = run_qat("unused-source", output,
                         QATConfig(enabled=True, train_data=str(train), validation_split=0,
                                   max_steps=1, batch_size=1, max_seq_length=8, device="cpu"),
                         model=model, tokenizer=Tokenizer())
        assert result.report["actual_steps"] == 1
        paths[name] = output
    return paths


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float16), ("auto", torch.float16),
                                         ("cuda", None), ("cuda", torch.float32)])
def test_cuda_loading_requires_explicit_compatible_device_and_dtype_before_io(device, dtype):
    with pytest.raises(ValueError):
        load_qat("does-not-exist", device=device, dtype=dtype, inference_backend="cuda-w8a16")


def test_unknown_or_inapplicable_backend_rejected_without_hub_access():
    with pytest.raises(ValueError, match="inference_backend"):
        load_qat("does-not-exist", inference_backend="imaginary")
    with pytest.raises(ValueError, match="portable QAT artifact"):
        load_causal_lm("not-a-local-model", device="cuda", dtype="float16", quantized_backend="cuda-w8a16")


def test_unavailable_kernel_fails_before_artifact_io_without_fallback(monkeypatch):
    import saddlellm.models.QATW8A16 as backend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend, "cuda_int8_supported", lambda *args, **kwargs: (False, "injected unavailable kernel"))
    with pytest.raises(RuntimeError, match="no fallback"):
        load_qat("does-not-exist", device="cuda", dtype="float16", inference_backend="cuda-w8a16")


def test_default_loading_retains_fp32_reference_and_does_not_require_cuda(artifacts):
    model = load_causal_lm(artifacts["native"], device="cpu", local_files_only=True)
    assert not hasattr(model, "_saddle_qat_inference")
    with torch.inference_mode():
        assert model(torch.tensor([[1, 4, 5]])).logits.dtype == torch.float32
    with pytest.raises(ValueError, match="float32"):
        load_qat(artifacts["native"], dtype=torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA execution required")
@pytest.mark.parametrize("architecture", ["native", "hf"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_integer_artifact_reload_uses_kernels_without_requantizing(artifacts, architecture, dtype):
    from saddlellm.models.QATW8A16 import QATW8A16Linear, qat_cuda_status

    path = artifacts[architecture]
    before = artifact_hashes(path)
    state = torch.load(path / "qat_model.pt", weights_only=True)
    model = load_causal_lm(path, device="cuda", dtype=dtype, local_files_only=True,
                           quantized_backend="cuda-w8a16")
    for name, layer in model.named_modules():
        if isinstance(layer, QATW8A16Linear):
            assert torch.equal(layer.qweight.cpu(), state[name + ".qweight"])
            assert torch.equal(layer.scales.cpu(), state[name + ".scales"])
            assert layer.scales.dtype == torch.float32
            assert layer.weight_t.dtype == torch.int8
    ids = torch.tensor([[1, 4, 6]], device="cuda")
    reference = load_qat(path, device="cuda")
    with torch.inference_mode():
        expected = reference(ids).logits
        actual = model(ids).logits
        torch.testing.assert_close(actual.float(), expected, rtol=0.06, atol=0.012)
        generated = model.generate(ids, max_new_tokens=3, do_sample=False, eos_token_id=[], pad_token_id=0)
    assert actual.dtype == dtype
    assert generated.shape == (1, 6)
    status = qat_cuda_status(model)
    assert status["kernel_calls"] > 0
    assert status["cuda_linears"] == status["linear_count"] > 0
    assert status["weight_materializations"] == 0
    assert status["codes_requantized"] is False
    reloaded = load_qat(path, device="cuda", dtype=dtype, inference_backend="cuda-w8a16")
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids).logits, actual, rtol=0, atol=0)
    assert artifact_hashes(path) == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA execution required")
def test_root_training_labels_and_float_export_are_explicitly_blocked(artifacts):
    model = load_qat(artifacts["native"], device="cuda", dtype="float16", inference_backend="cuda-w8a16")
    ids = torch.tensor([[1, 4]], device="cuda")
    with pytest.raises(ValueError, match="inference-only"):
        model.train()
    assert not model.training
    with pytest.raises(ValueError, match="inference_mode"):
        model(ids)
    with torch.inference_mode(), pytest.raises(ValueError, match="training labels"):
        model(input_ids=ids, labels=ids)
    with pytest.raises(ValueError, match="original portable"):
        model.save_pretrained("never-written")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA execution required")
def test_weight_property_preserves_head_shape_dtype_and_counts_materialization(artifacts):
    from saddlellm.models.QATW8A16 import QATW8A16Linear, qat_cuda_status

    model = load_qat(artifacts["hf"], device="cuda", dtype=torch.bfloat16, inference_backend="cuda-w8a16")
    head = model.get_output_embeddings()
    assert isinstance(head, QATW8A16Linear)
    before = qat_cuda_status(model)
    with torch.inference_mode():
        weight = head.weight
        expected = (head.qweight.float() * head.scales[:, None]).bfloat16()
    torch.testing.assert_close(weight, expected, rtol=0, atol=0)
    assert weight.shape == (19, 16) and weight.is_cuda and weight.dtype == torch.bfloat16
    assert qat_cuda_status(model)["weight_materializations"] == before["weight_materializations"] + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA execution required")
@pytest.mark.parametrize("failure", ["cpu", "float32", "autocast", "gradients"])
def test_layer_rejects_unsupported_execution_without_dense_fallback(artifacts, failure):
    from saddlellm.models.QATW8A16 import QATW8A16Linear

    model = load_qat(artifacts["native"], device="cuda", dtype="float16", inference_backend="cuda-w8a16")
    layer = next(module for module in model.modules() if isinstance(module, QATW8A16Linear))
    values = torch.ones(1, layer.in_features, device="cuda", dtype=torch.float16)
    if failure == "cpu":
        values = values.cpu()
    elif failure == "float32":
        values = values.float()
    if failure == "gradients":
        with pytest.raises(ValueError, match="inference_mode"):
            layer(values)
    else:
        with torch.inference_mode(), torch.autocast("cuda", enabled=failure == "autocast"), pytest.raises(ValueError):
            layer(values)
    assert layer.kernel_calls == 0
    assert layer.weight_materializations == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA execution required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 3, 8, 33])
def test_existing_w8a16_kernel_matches_same_integer_codes_with_odd_shapes(dtype, rows):
    from saddlellm.compression.QuantizationAwareTraining import _module_types
    from saddlellm.models.QATW8A16 import QATW8A16Linear

    torch.manual_seed(81)
    _, packed_type = _module_types()
    packed = packed_type(35, 53, True)
    packed.qweight.copy_(torch.randint(-127, 128, (53, 35), dtype=torch.int8))
    packed.qweight[0].zero_()
    packed.scales.copy_(torch.logspace(-4, 0, 53))
    packed.bias.copy_(torch.randn(53))
    layer = QATW8A16Linear(packed).to(device="cuda", dtype=dtype)
    values = torch.randn(rows, 35, device="cuda", dtype=dtype)
    with torch.inference_mode():
        expected = torch.nn.functional.linear(values.float(),
                                                layer.qweight.float() * layer.scales[:, None], layer.bias).to(dtype)
        actual = layer(values)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual[:, 0], layer.bias[0].to(dtype).expand(rows), rtol=0, atol=0)
    assert torch.equal(layer.qweight.cpu(), packed.qweight)
    assert torch.equal(layer.scales.cpu(), packed.scales)
    assert layer.scales.dtype == torch.float32
    assert layer.kernel_calls == 1 and layer.weight_materializations == 0
