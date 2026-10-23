"""Same-weight contracts for native dense architecture adapters."""

import json

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.ModelLoader import load_native_causal_lm
from saddlellm.models.NativeArchitectures import native_architecture_config
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.InferenceServer import InferenceServerSettings, create_inference_app


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_checkpoint(path, family, *, tied=False, sharded=False):
    torch.manual_seed(482)
    config_cls, model_cls = {
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "llama": (LlamaConfig, LlamaForCausalLM),
    }[family]
    config = config_cls(
        vocab_size=97, hidden_size=48, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=96,
        pad_token_id=0, tie_word_embeddings=tied,
    )
    config._attn_implementation = "eager"
    hf = model_cls(config).eval()
    with torch.no_grad():
        for name, value in hf.named_parameters():
            if name.endswith(".bias"):
                value.copy_(torch.linspace(-0.07, 0.09, value.numel()))
            elif "norm.weight" in name:
                value.copy_(torch.linspace(0.8, 1.2, value.numel()))
    hf.save_pretrained(path, max_shard_size="4KB" if sharded else "1GB")
    core = Tokenizer(models.WordLevel({f"x{i}": i for i in range(97)}, unk_token="x0"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="x0", pad_token="x0", bos_token="x1", eos_token="x96",
    )
    tokenizer.chat_template = "{{bos_token}} {% for message in messages %}{{message['content']}} {% endfor %}"
    tokenizer.save_pretrained(path)
    return hf


@pytest.mark.parametrize("family", ["qwen2", "llama"])
@pytest.mark.parametrize("tied,sharded", [(False, False), (True, True)])
@pytest.mark.parametrize("backend", ["eager", "sdpa", "sdpa_reference"])
def test_same_weights_prefill_cached_decode_packing_and_reload(tmp_path, family, tied, sharded, backend):
    hf = make_checkpoint(tmp_path / "hf", family, tied=tied, sharded=sharded)
    hf.config._attn_implementation = "sdpa" if backend == "sdpa_reference" else backend
    native = load_native_causal_lm(tmp_path / "hf", local_files_only=True, attention_backend=backend)
    assert type(native) is SaddleForCausalLM
    assert native._saddle_native_source["model_type"] == family
    assert native.config.qkv_bias == (family == "qwen2")
    assert native.config.qk_norm is False
    for name, value in native.state_dict().items():
        key = name if name.startswith("lm_head.") else "model." + name
        assert torch.equal(value, hf.state_dict()[key]), key
    tokens = torch.tensor([[1, 5, 7, 9], [1, 4, 6, 8]])
    with torch.inference_mode():
        ref_cache = own_cache = None
        current = tokens
        for _ in range(5):
            ref = hf(current, past_key_values=ref_cache, use_cache=True)
            own = native(current, past_key_values=own_cache, use_cache=True)
            torch.testing.assert_close(own.logits, ref.logits, atol=3e-6, rtol=3e-5)
            current = ref.logits[:, -1].argmax(-1, keepdim=True)
            ref_cache, own_cache = ref.past_key_values, own.past_key_values
        expected = hf.generate(tokens, max_new_tokens=8, do_sample=False, eos_token_id=[])
        assert torch.equal(native.generate(tokens, max_new_tokens=8, do_sample=False, eos_token_id=[]), expected)
        report = optimize_native_inference(native)
        assert report.packed_qkv == (0 if family == "qwen2" else 2)
        assert report.packed_gate_up == 2
        assert torch.equal(native.generate(tokens, max_new_tokens=8, do_sample=False, eos_token_id=[]), expected)
    native.save_pretrained(tmp_path / "native")
    restored = load_native_causal_lm(tmp_path / "native", local_files_only=True)
    assert restored.config.qkv_bias == native.config.qkv_bias
    with torch.inference_mode():
        torch.testing.assert_close(restored(tokens).logits, native(tokens).logits)


@pytest.mark.parametrize("family", ["qwen2", "llama"])
def test_native_service_never_uses_hf_model_and_supports_streaming(tmp_path, monkeypatch, family):
    make_checkpoint(tmp_path, family)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("HF fallback"))
    app = create_inference_app(InferenceServerSettings(
        model_path=str(tmp_path), backend="native", device="cpu", dtype="float32",
        local_files_only=True, web_ui=False, max_input_tokens=32,
        max_new_tokens=8, default_max_new_tokens=4,
    ))
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["execution_backend"] == "native"
        assert health["native_source"]["model_type"] == family
        request = {"messages": [{"role": "user", "content": "x4 x5 x6"}], "temperature": 0, "max_tokens": 4}
        plain = client.post("/v1/chat/completions", json=request)
        assert plain.status_code == 200
        streamed = client.post("/v1/chat/completions", json={**request, "stream": True, "stream_options": {"include_usage": True}})
        assert streamed.status_code == 200
        chunks = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        answer = "".join(choice.get("delta", {}).get("content", "") for chunk in chunks for choice in chunk.get("choices", []))
        assert answer == plain.json()["choices"][0]["message"]["content"]
        assert next(chunk["usage"] for chunk in chunks if chunk.get("usage")) == plain.json()["usage"]


@pytest.mark.parametrize("family", ["qwen2", "llama"])
@pytest.mark.parametrize("change", [
    {"rope_scaling": {"rope_type": "yarn", "factor": 4}},
    {"quantization_config": {"quant_method": "fp8"}},
    {"use_sliding_window": True}, {"mlp_bias": True},
    {"architectures": ["OtherForCausalLM"]}, {"use_mrope": True},
])
def test_unsupported_variants_are_rejected(tmp_path, family, change):
    make_checkpoint(tmp_path, family)
    raw = json.loads((tmp_path / "config.json").read_text())
    raw.update(change)
    with pytest.raises(ValueError):
        native_architecture_config(raw)


def test_missing_qwen2_bias_is_not_silently_zeroed(tmp_path):
    make_checkpoint(tmp_path, "qwen2")
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    weights.pop("model.layers.0.self_attn.q_proj.bias")
    save_file(weights, path)
    with pytest.raises(ValueError, match="missing Qwen2.*q_proj.bias"):
        load_native_causal_lm(tmp_path, local_files_only=True)


@pytest.mark.parametrize("family", ["deepseek_v3", "kimi_k2", "minimax", "minimax_m2"])
def test_unimplemented_full_models_are_not_misidentified_as_dense(family):
    with pytest.raises(ValueError, match="unsupported model_type"):
        native_architecture_config({"model_type": family})


def test_bias_config_never_creates_unsupported_packed_checkpoint():
    assert SaddleModelConfig().qkv_bias is False
    with pytest.raises(ValueError, match="unfused"):
        SaddleModelConfig(qkv_bias=True, fused_qkv=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("family", ["qwen2", "llama"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_low_precision_same_prefix_logits(tmp_path, family, dtype):
    hf = make_checkpoint(tmp_path, family).to(device="cuda", dtype=dtype)
    native = load_native_causal_lm(tmp_path, device="cuda", dtype=dtype, local_files_only=True, attention_backend="eager")
    current = torch.tensor([[1, 5, 7, 9]], device="cuda")
    ref_cache = own_cache = None
    with torch.inference_mode():
        for _ in range(5):
            ref = hf(current, past_key_values=ref_cache, use_cache=True)
            own = native(current, past_key_values=own_cache, use_cache=True)
            torch.testing.assert_close(own.logits, ref.logits, atol=0.004, rtol=0.04)
            current = ref.logits[:, -1].argmax(-1, keepdim=True)
            ref_cache, own_cache = ref.past_key_values, own.past_key_values


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("family", ["qwen2", "llama"])
def test_bf16_without_flash_defaults_to_reference_but_honors_override(tmp_path, monkeypatch, family):
    make_checkpoint(tmp_path, family)
    monkeypatch.setattr("saddlellm.models.OperatorBackends._flash_attention_available", lambda: False)
    native = load_native_causal_lm(tmp_path, device="cuda", dtype="bfloat16", local_files_only=True)
    assert native.config.attention_backend == "sdpa_reference"
    assert all(layer.config.attention_backend == "sdpa_reference" for layer in native.layers)
    explicit = load_native_causal_lm(tmp_path, device="cuda", dtype="bfloat16", local_files_only=True, attention_backend="sdpa")
    assert explicit.config.attention_backend == "sdpa"
