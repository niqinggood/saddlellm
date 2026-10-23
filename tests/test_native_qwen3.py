"""Offline, same-weight checks for the real Saddle Qwen3 execution path."""

import json

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.ModelLoader import load_native_causal_lm
from saddlellm.models.NativeQwen3 import qwen3_native_config
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.InferenceServer import (
    InferenceServerSettings,
    create_inference_app,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_checkpoint(directory, *, sharded=False, tied=False, dtype=torch.float32):
    torch.manual_seed(173)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=96,
        pad_token_id=0,
        tie_word_embeddings=tied,
    )
    config._attn_implementation = "sdpa"
    hf = Qwen3ForCausalLM(config).eval()
    with torch.no_grad():
        for module in hf.modules():
            if isinstance(module, Qwen3RMSNorm):
                module.weight.copy_(torch.linspace(0.8, 1.2, module.weight.numel()))
    hf = hf.to(dtype=dtype)
    hf.save_pretrained(directory, max_shard_size="4KB" if sharded else "1GB")
    vocabulary = {f"x{index}": index for index in range(config.vocab_size)}
    tokenizer_core = Tokenizer(models.WordLevel(vocabulary, unk_token="x0"))
    tokenizer_core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core,
        unk_token="x0",
        pad_token="x0",
        bos_token="x1",
        eos_token="x96",
    )
    tokenizer.chat_template = (
        "{{bos_token}} {% for message in messages %}{{message['content']}} {% endfor %}"
    )
    tokenizer.save_pretrained(directory)
    return hf, tokenizer


@pytest.fixture
def checkpoint(tmp_path):
    hf, tokenizer = make_checkpoint(tmp_path)
    return tmp_path, hf, tokenizer


@pytest.mark.parametrize("sharded,tied", [(False, False), (True, False), (True, True)])
def test_loader_strict_same_weights_without_hf_construction(
    tmp_path, sharded, tied, monkeypatch
):
    hf, _ = make_checkpoint(tmp_path, sharded=sharded, tied=tied)
    from transformers import AutoModelForCausalLM

    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: pytest.fail("HF loader called"),
    )
    native = load_native_causal_lm(tmp_path, device="cpu", local_files_only=True)
    assert type(native) is SaddleForCausalLM
    assert not native.training
    assert not any(
        parameter.requires_grad or parameter.is_meta
        for parameter in native.parameters()
    )
    assert not any(buffer.is_meta for buffer in native.buffers())
    for name, value in native.state_dict().items():
        hf_name = name if name.startswith("lm_head.") else "model." + name
        assert torch.equal(value, hf.state_dict()[hf_name]), name
    assert native.config.rms_norm_weight_in_fp32 is False
    assert native.config.rope_compute_fp32 is True
    assert native._saddle_native_source["generation"] == "native_step"
    if tied:
        assert native.lm_head.weight is native.embed_tokens.weight


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_cpu_prefill_cached_decode_and_greedy_match(checkpoint, backend):
    path, hf, _ = checkpoint
    native = load_native_causal_lm(path, local_files_only=True)
    hf.config._attn_implementation = backend
    native.config.attention_backend = backend
    for layer in native.layers:
        layer.config.attention_backend = backend
    tokens = torch.tensor([[3, 17, 22, 41], [9, 28, 55, 61]])
    with torch.inference_mode():
        ref = hf(tokens, use_cache=True)
        own = native(tokens, use_cache=True)
        torch.testing.assert_close(own.logits, ref.logits, atol=2e-6, rtol=2e-5)
        next_tokens = torch.tensor([[11], [19]])
        ref_next = hf(next_tokens, past_key_values=ref.past_key_values, use_cache=True)
        own_next = native(
            next_tokens, past_key_values=own.past_key_values, use_cache=True
        )
        torch.testing.assert_close(
            own_next.logits, ref_next.logits, atol=2e-6, rtol=2e-5
        )
        ref_ids = hf.generate(
            tokens, do_sample=False, max_new_tokens=8, eos_token_id=None
        )
        own_ids = native.generate(
            tokens, do_sample=False, max_new_tokens=8, eos_token_id=[]
        )
        assert torch.equal(own_ids, ref_ids)
        optimize_native_inference(native)
        assert torch.equal(
            native.generate(tokens, do_sample=False, max_new_tokens=8, eos_token_id=[]),
            ref_ids,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_low_precision_and_rope_constants(checkpoint, dtype, backend):
    path, hf, _ = checkpoint
    native = load_native_causal_lm(
        path, device="cuda", dtype=dtype, local_files_only=True
    )
    # Real HF low-precision loading preserves its nonpersistent FP32 RoPE.
    # A blanket .to(dtype) on a previously initialized FP32 model does not.
    hf = Qwen3ForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=backend,
    ).to(device="cuda")
    for layer in native.layers:
        layer.config.attention_backend = backend
    tokens = torch.tensor([[3, 17, 22, 41, 66]], device="cuda")
    with torch.inference_mode():
        reference = hf(tokens).logits
        result = native(tokens).logits
        torch.testing.assert_close(result, reference, atol=0.004, rtol=0.04)
        ref_ids = hf.generate(
            tokens, do_sample=False, max_new_tokens=8, eos_token_id=None
        )
        own_ids = native.generate(
            tokens, do_sample=False, max_new_tokens=8, eos_token_id=[]
        )
        if backend == "eager":
            # The mathematical reference must match; SDPA can dispatch different
            # kernels, and a BF16 near-tie in this fixture DOES change argmax.
            assert torch.equal(own_ids, ref_ids)
        else:
            assert own_ids.shape == ref_ids.shape
        # Teacher forcing checks EACH cached step on the same prefix, rather
        # than comparing unrelated logits after an argmax divergence.
        hf_cache = native_cache = None
        step_tokens = tokens
        for _ in range(8):
            ref = hf(
                step_tokens, past_key_values=hf_cache, use_cache=True, logits_to_keep=1
            )
            own = native(
                step_tokens,
                past_key_values=native_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            torch.testing.assert_close(own.logits, ref.logits, atol=0.004, rtol=0.04)
            hf_cache, native_cache = ref.past_key_values, own.past_key_values
            step_tokens = ref.logits[:, -1].argmax(-1, keepdim=True)
        optimize_native_inference(native)
        packed_ids = native.generate(
            tokens, do_sample=False, max_new_tokens=8, eos_token_id=[]
        )
        if backend == "eager":
            assert torch.equal(packed_ids, ref_ids)
        else:
            assert packed_ids.shape == ref_ids.shape
    # Later .to() conversions must not quantize fixed inverse frequencies.
    native.to(dtype=torch.float32).to(dtype=dtype)
    rotary = native.layers[0].self_attn.rotary
    assert rotary.inv_freq.dtype == torch.float32
    expected = 1.0 / (
        rotary.base ** (torch.arange(0, 16, 2, device="cuda").float() / 16)
    )
    assert torch.equal(rotary.inv_freq, expected)


def test_rope_matches_reference_at_long_positions_and_autocast(checkpoint):
    path, hf, _ = checkpoint
    native = load_native_causal_lm(path, local_files_only=True).to(dtype=torch.bfloat16)
    positions = torch.arange(4090, 4100).unsqueeze(0)
    hf_rotary = hf.model.rotary_emb
    x = torch.empty(1, 10, 48, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = hf_rotary(x, positions)
        actual = native.layers[0].self_attn.rotary(
            10, offset=4090, dtype=torch.bfloat16
        )
    assert torch.equal(actual[0], expected[0][0])
    assert torch.equal(actual[1], expected[1][0])


def test_left_padded_generation_matches_reference_and_unpadded_rows(checkpoint):
    path, hf, _ = checkpoint
    native = load_native_causal_lm(path, local_files_only=True)
    optimize_native_inference(native)
    tokens = torch.tensor([[0, 0, 4, 5, 6], [8, 9, 10, 11, 12]])
    mask = (tokens != 0).long()
    reference = hf.generate(
        tokens,
        attention_mask=mask,
        do_sample=False,
        max_new_tokens=6,
        eos_token_id=None,
    )
    own = native.generate(
        tokens, attention_mask=mask, do_sample=False, max_new_tokens=6, eos_token_id=[]
    )
    assert torch.equal(own, reference)
    individual = native.generate(
        tokens[:1, 2:], do_sample=False, max_new_tokens=6, eos_token_id=[]
    )
    assert torch.equal(own[:1, 5:], individual[:, 3:])


@pytest.mark.parametrize(
    "change,error",
    [
        ({"model_type": "qwen3_moe"}, "dense"),
        ({"architectures": ["Qwen3Model"]}, "ForCausalLM"),
        ({"quantization_config": {"quant_method": "awq"}}, "quantized"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"attention_bias": True}, "attention_bias"),
        ({"use_sliding_window": True}, "sliding"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 4}}, "RoPE"),
        ({"rope_parameters": {"rope_type": "default"}}, "RoPE"),
        ({"layer_types": ["sliding_attention", "full_attention"]}, "full_attention"),
        ({"partial_rotary_factor": 0.5}, "rotary"),
        ({"attention_dropout": 0.1}, "dropout"),
        ({"pretraining_tp": 2}, "pretraining_tp"),
        ({"head_dim": 15}, "even"),
        ({"num_key_value_heads": 3}, "ratio"),
        ({"num_hidden_layers": True}, "positive integer"),
        ({"rms_norm_eps": float("nan")}, "finite"),
        ({"rope_theta": -1}, "positive"),
        ({"eos_token_id": [None]}, "token IDs"),
    ],
)
def test_unsupported_config_is_rejected_before_loading(checkpoint, change, error):
    path, _, _ = checkpoint
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    raw.update(change)
    with pytest.raises(ValueError, match=error):
        qwen3_native_config(raw)


@pytest.mark.parametrize("eos,pad", [(None, 0), ([], 0), (96, 96), ([95, 96], 95)])
def test_missing_pad_handles_disabled_or_multiple_eos(checkpoint, eos, pad):
    path, _, _ = checkpoint
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    raw.update(pad_token_id=None, eos_token_id=eos)
    config = qwen3_native_config(raw)
    assert config.pad_token_id == pad
    assert config.eos_token_id == eos


@pytest.mark.parametrize(
    "change,error",
    [
        ("missing", "missing Qwen3"),
        ("extra", "unexpected"),
        ("shape", "shape mismatch"),
        ("integer", "tensors only"),
    ],
)
def test_corrupt_tensor_contract_fails_closed(checkpoint, change, error):
    path, _, _ = checkpoint
    weights = load_file(str(path / "model.safetensors"))
    key = "model.layers.0.self_attn.q_norm.weight"
    if change == "missing":
        weights.pop(key)
    elif change == "extra":
        weights["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(64)
    elif change == "shape":
        weights[key] = weights[key][:-1]
    else:
        weights[key] = weights[key].long()
    save_file(weights, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match=error):
        load_native_causal_lm(path, local_files_only=True)


def test_shard_index_filename_and_missing_shards(tmp_path):
    make_checkpoint(tmp_path, sharded=True)
    path = tmp_path / "model.safetensors.index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    first = next(iter(index["weight_map"]))
    for value, error in [
        ("../model.safetensors", ValueError),
        ("missing.safetensors", FileNotFoundError),
    ]:
        index["weight_map"][first] = value
        path.write_text(json.dumps(index), encoding="utf-8")
        with pytest.raises(error):
            load_native_causal_lm(tmp_path, local_files_only=True)


def test_dtype_auto_special_tokens_and_tied_saddle_roundtrip(tmp_path):
    make_checkpoint(tmp_path, tied=True, dtype=torch.bfloat16)
    generation_path = tmp_path / "generation_config.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["eos_token_id"] = [95, 96]
    generation_path.write_text(json.dumps(generation), encoding="utf-8")
    model = load_native_causal_lm(tmp_path, local_files_only=True)
    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.config.eos_token_id == [95, 96]
    output = tmp_path / "native-export"
    model.save_pretrained(str(output))
    reloaded = load_native_causal_lm(output, device="cpu", dtype="bfloat16")
    assert reloaded.config.rms_norm_weight_in_fp32 is False
    assert reloaded.config.rope_compute_fp32 is True
    assert reloaded.lm_head.weight is reloaded.embed_tokens.weight
    tokens = torch.tensor([[4, 5, 6]])
    assert torch.equal(
        model.generate(tokens, do_sample=False, max_new_tokens=4),
        reloaded.generate(tokens, do_sample=False, max_new_tokens=4),
    )


def test_qwen3_bf16_serving_keeps_gate_up_packing_without_qkv(checkpoint):
    path, _, _ = checkpoint
    app = create_inference_app(
        InferenceServerSettings(
            model_path=str(path),
            backend="native",
            device="cpu",
            dtype="bfloat16",
            local_files_only=True,
            web_ui=False,
        )
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
    report = health["native_projection_packing"]
    assert health["native_source"]["model_type"] == "qwen3"
    assert report["packed_qkv"] == 0
    assert report["packed_gate_up"] == 2
    assert len(report["skipped"]) == 2
    assert all("numerical fidelity" in reason for reason in report["skipped"].values())


@pytest.mark.parametrize("cpu_prepack", [False, True])
def test_native_server_loads_real_model_and_handles_streaming(
    checkpoint, monkeypatch, cpu_prepack
):
    path, _, _ = checkpoint
    from transformers import AutoModelForCausalLM

    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: pytest.fail("HF fallback"),
    )
    app = create_inference_app(
        InferenceServerSettings(
            model_path=str(path),
            backend="native",
            device="cpu",
            dtype="float32",
            local_files_only=True,
            max_input_tokens=64,
            max_new_tokens=8,
            default_max_new_tokens=8,
            prefix_cache_mb=1,
            native_attention_backend="eager",
            cpu_prepack_linear=cpu_prepack,
        )
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["backend"] == health["execution_backend"] == "native"
        assert health["streaming_backend"] == "native_step"
        assert health["native_attention_backend"] == "eager"
        assert health["native_source"]["model_forward"] == "saddle"
        assert health["native_projection_packing"]["packed_qkv"] == 2
        assert health["cpu_prepack"]["active_caches"] == (9 if cpu_prepack else 0)
        prompt = " ".join(f"x{index}" for index in range(3, 19))
        first = client.post(
            "/v1/completions",
            json={"prompt": prompt, "temperature": 0, "max_tokens": 4},
        )
        assert first.status_code == 200
        assert first.json()["usage"]["completion_tokens"] == 4
        second = client.post(
            "/v1/completions",
            json={"prompt": prompt, "temperature": 0, "max_tokens": 4},
        )
        assert second.json()["choices"] == first.json()["choices"]
        assert client.get("/health").json()["prefix_cache"]["hits"] == 1
        streamed = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "x4 x5 x6"}],
                "temperature": 0,
                "max_tokens": 4,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert streamed.status_code == 200
        assert "data: [DONE]" in streamed.text
        chunks = [
            json.loads(row[6:])
            for row in streamed.text.splitlines()
            if row.startswith("data: {")
        ]
        assert any(
            (chunk.get("usage") or {}).get("completion_tokens") == 4 for chunk in chunks
        )
        if cpu_prepack:
            assert client.get("/health").json()["cpu_prepack"]["decode_calls"] > 0


def test_native_never_accepts_hf_model_or_quantization(checkpoint):
    _, hf, tokenizer = checkpoint
    with pytest.raises(ValueError, match="no HF fallback"):
        create_inference_app(
            InferenceServerSettings(backend="native"), model=hf, tokenizer=tokenizer
        )
    for kwargs in [{"quantization": "4bit"}, {"trust_remote_code": True}]:
        with pytest.raises(ValueError):
            InferenceServerSettings(backend="native", **kwargs)
    with pytest.raises(ValueError, match="requires backend=native"):
        InferenceServerSettings(native_attention_backend="eager")
    with pytest.raises(ValueError, match="loading option"):
        create_inference_app(
            InferenceServerSettings(backend="native", native_attention_backend="eager"),
            model=hf,
            tokenizer=tokenizer,
        )


def test_public_loader_is_lazy_and_old_config_defaults_unchanged():
    from saddlellm import load_native_causal_lm as public_loader

    assert public_loader is load_native_causal_lm
    defaults = SaddleModelConfig()
    assert defaults.rms_norm_weight_in_fp32 is True
    assert defaults.rope_compute_fp32 is False


def test_remote_metadata_is_validated_first_and_revision_is_pinned(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / ("a" * 40)
    make_checkpoint(snapshot)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    model = load_native_causal_lm("fixture/qwen3", local_files_only=True)
    assert type(model) is SaddleForCausalLM
    assert calls[0]["allow_patterns"] == ["config.json", "generation_config.json"]
    assert calls[1]["revision"] == "a" * 40
    assert all(call["local_files_only"] is True for call in calls)
    config_path = snapshot / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["model_type"] = "qwen3_moe"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    calls.clear()
    with pytest.raises(ValueError, match="dense"):
        load_native_causal_lm("fixture/qwen3", local_files_only=True)
    assert len(calls) == 1  # No large-weight download for unsupported models.


def test_duplicate_json_keys_and_mismatched_tied_weights(checkpoint):
    path, _, _ = checkpoint
    config_path = path / "config.json"
    original = config_path.read_text(encoding="utf-8")
    config_path.write_text(original[:-2] + ',"model_type":"qwen3"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_native_causal_lm(path, local_files_only=True)
    raw = json.loads(original)
    raw["tie_word_embeddings"] = True
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unequal"):
        load_native_causal_lm(path, local_files_only=True)


def test_cuda_memory_guard_and_unsupported_dtype_fail_before_transfer(
    checkpoint, monkeypatch
):
    path, _, _ = checkpoint
    with pytest.raises(ValueError, match="dtype"):
        load_native_causal_lm(path, dtype="float64", local_files_only=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (0, 49140 * 1024 * 1024))
    with pytest.raises(MemoryError, match="no implicit CPU offload"):
        load_native_causal_lm(path, device="cuda", local_files_only=True)


@pytest.mark.parametrize("cpu_prepack", [False, True])
def test_same_weight_benchmark_records_counts_and_does_not_overwrite(
    checkpoint, tmp_path, cpu_prepack
):
    from tools.benchmark_qwen3_native import main

    # Real Qwen3 checkpoints default to sampling. HF >= 4.50 can override
    # GenerationConfig(do_sample=False), which equals its global default.
    generation_path = checkpoint[0] / "generation_config.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation.update(
        do_sample=True,
        temperature=1.5,
        top_p=0.7,
        top_k=7,
        transformers_version="4.57.6",
    )
    generation_path.write_text(json.dumps(generation), encoding="utf-8")
    report_path = tmp_path / "benchmark.json"
    args = [
        "--model",
        str(checkpoint[0]),
        "--device",
        "cpu",
        "--prompt-tokens",
        "4",
        "--decode-tokens",
        "4",
        "--correctness-steps",
        "3",
        "--warmup",
        "1",
        "--repeats",
        "2",
        "--output",
        str(report_path),
    ]
    if cpu_prepack:
        args.extend(["--cpu-prepack", "--include-native-baseline"])
    assert main(args) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["all_generated_tokens_equal"] is True
    assert report["logit_comparison"]["all_within_tolerance"] is True
    assert report["conditions"]["decode_tokens"] == 4
    assert report["generation_policy"]["do_sample"] is False
    assert report["generation_policy"]["hf_use_model_defaults"] is False
    if cpu_prepack:
        assert report["cpu_prepack"]["active_caches"] == 9
        assert "native_without_prepack" in report["results"]
        assert report["native_baseline_logit_comparison"]["all_within_tolerance"]
    assert all(
        row["generated_tokens_per_call"] == 4 for row in report["results"].values()
    )
    assert len(report["checkpoint_sha256"]["model.safetensors"]) == 64
    before = report_path.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert report_path.read_bytes() == before


@pytest.mark.parametrize("attention", ["eager", "sdpa_reference"])
def test_cli_native_backend_is_forwarded(monkeypatch, checkpoint, attention):
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
                str(checkpoint[0]),
                "--backend",
                "native",
                "--device",
                "cpu",
                "--native-attention-backend",
                attention,
            ]
        )
        == 0
    )
    assert seen[0].backend == "native"
    assert seen[0].native_attention_backend == attention
