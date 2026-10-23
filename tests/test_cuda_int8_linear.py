"""Operator/contract checks, not small-model performance proxies for 8B."""

import copy

import pytest
import torch
from torch import nn

from saddlellm.models.CUDAInt8 import (
    CUDAInt8Linear,
    cuda_int8_status,
    prepare_cuda_int8,
)
from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def source_linear(in_features=35, out_features=49, bias=True):
    torch.manual_seed(17)
    return nn.Linear(in_features, out_features, bias=bias).eval().requires_grad_(False)


def integer_reference(layer, values):
    flat = values.cpu().float().reshape(-1, layer.in_features)
    scales = flat.abs().amax(-1, keepdim=True) / 127
    scales[scales == 0] = 1
    activations = (flat / scales).round().clamp(-127, 127).to(torch.int32)
    accum = (
        activations
        @ layer.weight_t[: layer.in_features, : layer.out_features].cpu().int()
    )
    result = accum.float() * scales * layer.weight_scales[: layer.out_features].cpu()
    if layer.bias is not None:
        result += layer.bias.cpu()
    return result.to(values.dtype).reshape(*values.shape[:-1], layer.out_features)


def test_storage_quantization_zero_rows_and_fp32_scale_preservation():
    source = source_linear()
    with torch.no_grad():
        source.weight[0].zero_()
    original = source.weight.clone()
    layer = CUDAInt8Linear(source)
    assert layer.weight_t.dtype == torch.int8
    assert layer.weight_t.shape == (48, 64)
    assert layer.weight_t.is_contiguous()
    assert layer.weight_scales[0] == 1
    assert not layer.weight_t[:, 0].any()
    assert not layer.weight_t[35:].any()
    assert not layer.weight_t[:, 49:].any()
    assert not any(p.requires_grad for p in layer.parameters())
    assert torch.equal(source.weight, original)
    scales = layer.weight_scales.clone()
    bias = layer.bias.clone()
    layer.half().bfloat16()
    assert layer.weight_scales.dtype == layer.bias.dtype == torch.float32
    assert torch.equal(layer.weight_scales, scales)
    assert torch.equal(layer.bias, bias)


@pytest.mark.parametrize("failure", ["training", "grad", "nan", "hook", "subclass"])
def test_rejects_unsafe_source(failure):
    source = source_linear()
    if failure == "training":
        source.train()
    elif failure == "grad":
        source.weight.requires_grad_(True)
    elif failure == "nan":
        with torch.no_grad():
            source.weight[0, 0] = float("nan")
    elif failure == "hook":
        source.register_forward_hook(lambda *args: None)
    else:

        class CustomLinear(nn.Linear):
            pass

        source = CustomLinear(35, 49).eval().requires_grad_(False)
    with pytest.raises(ValueError):
        CUDAInt8Linear(source)


def test_rejects_cpu_forward_grad_and_training():
    layer = CUDAInt8Linear(source_linear())
    with pytest.raises(ValueError, match="inference_mode"):
        layer(torch.ones(1, 35))
    with torch.inference_mode(), pytest.raises(ValueError, match="CUDA FP16/BF16"):
        layer(torch.ones(1, 35))
    with pytest.raises(ValueError, match="inference-only"):
        layer.train()
    assert not layer.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(35,), (1, 35), (2, 3, 35), (17, 35), (33, 35)])
def test_cuda_matches_exact_integer_reference(dtype, shape):
    layer = CUDAInt8Linear(source_linear(), activation_bits=8).cuda()
    torch.manual_seed(27)
    values = torch.randn(shape, device="cuda", dtype=dtype)
    with torch.inference_mode():
        actual = layer(values)
        expected = integer_reference(layer, values)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert layer.kernel_calls == 1
    assert actual.dtype == dtype
    assert actual.shape == (*shape[:-1], 49)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 4, 5, 17, 33, 129])
def test_weight_only_kernels_preserve_float_activations(dtype, rows):
    layer = CUDAInt8Linear(source_linear(), activation_bits=16).cuda()
    torch.manual_seed(47)
    values = torch.randn((rows, 35), device="cuda", dtype=dtype)
    values[:, 0] *= 100  # Explicit activation outlier, NOT clipped/quantized.
    with torch.inference_mode():
        actual = layer(values)
        raw = torch.nn.functional.linear(
            values.float(), layer.weight_t[:35, :49].t().float()
        )
        expected = (raw * layer.weight_scales[:49] + layer.bias).to(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.008 if dtype == torch.bfloat16 else 0.001,
        rtol=0.008 if dtype == torch.bfloat16 else 0.001,
    )
    assert actual.dtype == dtype
    assert layer.activation_bits == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features", "expected"),
    [
        (8, 4096, 6144, (16, 128)),
        (8, 4096, 24576, (16, 128)),
        (128, 4096, 24576, (32, 64)),
        (128, 4096, 6144, (32, 64)),
        (8, 2048, 6144, (32, 64)),
        (8, 12288, 4096, (32, 64)),
    ],
)
def test_weight_only_tile_selection(rows, in_features, out_features, expected):
    from saddlellm.models.Int8WeightKernels import _matmul_tile

    assert _matmul_tile(rows, in_features, out_features) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features"),
    [(8, 4096, 6144), (8, 4096, 24576), (128, 4096, 24576)],
)
def test_weight_only_tuned_tile_matches_original_kernel(
    dtype, rows, in_features, out_features
):
    import triton

    from saddlellm.models.Int8WeightKernels import _matmul, weight_only_linear

    torch.manual_seed(211)
    inputs = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    weights = torch.randint(
        -127, 128, (in_features, out_features), device="cuda", dtype=torch.int8
    )
    scales = torch.rand(out_features, device="cuda", dtype=torch.float32)
    baseline = torch.empty(rows, out_features, device="cuda", dtype=dtype)
    with torch.inference_mode():
        _matmul[(triton.cdiv(rows, 32), triton.cdiv(out_features, 64))](
            inputs, weights, scales, scales, baseline,
            rows, in_features, out_features, out_features, False,
            32, 64, 64, num_warps=4, num_stages=3,
        )
        tuned = weight_only_linear(inputs, weights, scales, None, out_features)
    torch.testing.assert_close(tuned, baseline, atol=0, rtol=0)


def native_model():
    config = SaddleModelConfig(
        vocab_size=97,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        qk_norm=True,
        rms_norm_weight_in_fp32=False,
        rope_compute_fp32=True,
        attention_backend="eager",
    )
    torch.manual_seed(33)
    return SaddleForCausalLM(config).half().eval().requires_grad_(False)


def test_model_preparation_keeps_float_head_and_defaults():
    model = native_model()
    output = model.lm_head.weight
    embeddings = model.embed_tokens.weight
    optimize_native_inference(model)
    report = prepare_cuda_int8(model)
    assert report["linear_count"] == 8
    assert report["lossy"] is True
    assert model.lm_head.weight is output
    assert model.embed_tokens.weight is embeddings
    assert model.lm_head.weight.dtype == torch.float16
    assert cuda_int8_status(model)["cuda_linears"] == 0
    with pytest.raises(ValueError, match="already"):
        prepare_cuda_int8(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_and_unpacked_same_quantization_generate_same_tokens():
    unpacked = native_model()
    packed = copy.deepcopy(unpacked)
    optimize_native_inference(packed)
    prepare_cuda_int8(unpacked)
    prepare_cuda_int8(packed)
    unpacked.cuda()
    packed.cuda()
    tokens = torch.tensor([[4, 6, 8, 10]], device="cuda")
    with torch.inference_mode():
        ref, own = unpacked(tokens).logits, packed(tokens).logits
        torch.testing.assert_close(own, ref, atol=0, rtol=0)
        assert torch.equal(
            unpacked.generate(
                tokens, do_sample=False, max_new_tokens=6, eos_token_id=[]
            ),
            packed.generate(tokens, do_sample=False, max_new_tokens=6, eos_token_id=[]),
        )
    assert cuda_int8_status(packed)["kernel_calls"] > 0


def test_hooked_model_is_not_partially_rewritten():
    model = native_model()
    first = model.layers[0].self_attn.q_proj
    model.layers[1].self_attn.q_proj.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match="hooked"):
        prepare_cuda_int8(model)
    assert model.layers[0].self_attn.q_proj is first
    assert cuda_int8_status(model)["linear_count"] == 0


def test_quantized_native_rejects_training_and_false_float_export(tmp_path):
    model = native_model()
    prepare_cuda_int8(model)
    with pytest.raises(ValueError, match="inference-only"):
        model.train()
    assert not model.training
    with pytest.raises(ValueError, match="cannot be exported"):
        model.save_pretrained(tmp_path / "unsafe_export")
    assert not (tmp_path / "unsafe_export").exists()
    model.eval()


def test_numeric_failure_marks_partial_model_unusable(tmp_path):
    model = native_model()
    with torch.no_grad():
        model.layers[1].self_attn.q_proj.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prepare_cuda_int8(model)
    assert model._saddle_cuda_int8["status"] == "failed_discard_model"
    with pytest.raises(ValueError, match="cannot be exported"):
        model.save_pretrained(tmp_path / "unsafe_export")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantization": "4bit"},
        {"fuse_projections": "yes"},
        {"quantization": "int8", "device": "cpu"},
        {"quantization": "int8", "device": "cuda", "dtype": "float32"},
        {"gpu_reserve_bytes": True},
        {"gpu_reserve_bytes": -1},
    ],
)
def test_loader_rejects_invalid_options_before_checkpoint_io(kwargs):
    from saddlellm.models.ModelLoader import load_native_causal_lm

    with pytest.raises(ValueError):
        load_native_causal_lm(
            "nonexistent-offline-checkpoint", local_files_only=True, **kwargs
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantization": "int8"},
        {"backend": "native", "device": "cpu", "quantization": "int8"},
        {
            "backend": "native",
            "device": "cuda",
            "dtype": "float32",
            "quantization": "int8",
        },
        {"backend": "native", "quantization": "8bit"},
        {"backend": "native", "dtype": "float32", "quantization": "int8"},
        {"backend": "vllm", "quantization": "int8"},
    ],
)
def test_service_rejects_wrong_int8_modes(kwargs):
    from saddlellm.runtime.InferenceServer import InferenceServerSettings

    with pytest.raises(ValueError):
        InferenceServerSettings(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_prefix_cache_uses_and_invalidates_scale_fingerprint():
    from saddlellm.models.NativePrefixCache import NativePrefixCache

    model = native_model()
    optimize_native_inference(model)
    prepare_cuda_int8(model)
    model.cuda()
    cache = NativePrefixCache(max_bytes=1024**2, min_prefix_tokens=2)
    tokens = torch.tensor([[4, 6, 8, 10]], device="cuda")
    kwargs = dict(
        do_sample=False, max_new_tokens=4, eos_token_id=[], prefix_cache=cache
    )
    with torch.inference_mode():
        first = model.generate(tokens, **kwargs)
        second = model.generate(tokens, **kwargs)
    assert torch.equal(first, second)
    assert cache.snapshot()["hits"] == 1
    invalidations = cache.snapshot()["invalidations"]
    with torch.no_grad():
        model.layers[0].self_attn.qkv_packed_proj.weight_scales.mul_(1.01)
    with torch.inference_mode():
        model.generate(tokens, **kwargs)
    assert cache.snapshot()["invalidations"] > invalidations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fuse", [False, True])
def test_public_int8_loader_keeps_checkpoint_files_and_hf_out_of_execution(
    tmp_path, monkeypatch, fuse
):
    import hashlib

    from transformers import Qwen3Config, Qwen3ForCausalLM

    from saddlellm.models.ModelLoader import load_native_causal_lm

    config = Qwen3Config(
        vocab_size=97,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=96,
        pad_token_id=0,
    )
    source = Qwen3ForCausalLM(config).half().eval()
    source.save_pretrained(tmp_path)
    checkpoint = tmp_path / "model.safetensors"
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    del source
    monkeypatch.setattr(
        Qwen3ForCausalLM,
        "from_pretrained",
        lambda *a, **k: pytest.fail("HF loader called"),
    )
    model = load_native_causal_lm(
        tmp_path,
        device="cuda",
        dtype="float16",
        quantization="int8",
        local_files_only=True,
        fuse_projections=fuse,
    )
    assert type(model) is SaddleForCausalLM
    assert model.config.attention_backend == "eager"
    status = cuda_int8_status(model)
    assert status["linear_count"] == status["cuda_linears"] == (8 if fuse else 14)
    assert status["status"] == "ready"
    assert model.lm_head.weight.dtype == torch.float16
    assert (
        model.layers[0].self_attn.qkv_packed_proj is not None
        if fuse
        else model.layers[0].self_attn.qkv_packed_proj is None
    )
    with torch.inference_mode():
        result = model.generate(
            torch.tensor([[4, 6, 8, 10]], device="cuda"),
            do_sample=False,
            max_new_tokens=6,
            eos_token_id=[],
        )
    assert result.shape == (1, 10)
    assert cuda_int8_status(model)["kernel_calls"] > 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash


def test_cli_int8_options_are_forwarded_without_backend_switch(monkeypatch):
    import uvicorn

    import saddlellm.runtime.InferenceServer as server
    from saddlellm import cli

    seen = []
    monkeypatch.setattr(
        server,
        "create_inference_app",
        lambda settings: seen.append(settings) or object(),
    )
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    assert (
        cli.main(
            [
                "serve",
                "offline-qwen3",
                "--backend",
                "native",
                "--device",
                "cuda",
                "--quantization",
                "int8",
                "--no-fuse-projections",
            ]
        )
        == 0
    )
    assert seen[0].backend == "native"
    assert seen[0].quantization == "int8"
    assert seen[0].fuse_projections is False
    assert seen[0].cpu_prepack_linear is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_public_server_reports_actual_int8_and_serves(monkeypatch):
    from fastapi.testclient import TestClient
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from saddlellm.runtime.InferenceServer import (
        InferenceServerSettings,
        create_inference_app,
    )

    model = native_model()
    optimize_native_inference(model)
    prepare_cuda_int8(model)
    model.cuda()
    core = Tokenizer(models.WordLevel({f"x{i}": i for i in range(97)}, unk_token="x0"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="x0", pad_token="x0", eos_token="x96"
    )
    app = create_inference_app(
        InferenceServerSettings(
            backend="native",
            max_input_tokens=64,
            default_max_new_tokens=4,
            max_new_tokens=8,
        ),
        model=model,
        tokenizer=tokenizer,
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["quantization"] == "saddle-w8a16"
        assert health["native_int8"]["cuda_linears"] == 8
        assert health["execution_backend"] == "native"
        response = client.post(
            "/v1/completions",
            json={"prompt": "x4 x6 x8 x10", "max_tokens": 4, "temperature": 0},
        )
        assert response.status_code == 200
        assert response.json()["usage"]["completion_tokens"] > 0
        assert client.get("/health").json()["native_int8"]["kernel_calls"] > 0
