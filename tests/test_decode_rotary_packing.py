"""Decode-only rotary packing must preserve elementwise rounding exactly."""

import pytest
import torch

from saddlellm.models.SaddleModeling import apply_rotary_pos_emb


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,heads,kv_heads,seq_len,positions", [(1, 16, 8, 1, False), (3, 12, 2, 1, True), (2, 4, 4, 5, True)])
def test_rotary_exact(device, dtype, batch, heads, kv_heads, seq_len, positions):
    torch.manual_seed(74)
    # Transposed, noncontiguous views match model projections.
    q = torch.randn(batch, seq_len, heads, 64, device=device, dtype=dtype).transpose(1, 2)
    k = torch.randn(batch, seq_len, kv_heads, 64, device=device, dtype=dtype).transpose(1, 2)
    shape = (batch, seq_len, 64) if positions else (seq_len, 64)
    angles = torch.randn(shape, device=device).to(dtype)
    with torch.inference_mode():
        reference = apply_rotary_pos_emb(q, k, angles.cos(), angles.sin())
        candidate = apply_rotary_pos_emb(q, k, angles.cos(), angles.sin(), pack_decode=True)
    for old, new in zip(reference, candidate):
        assert torch.equal(old, new)


def test_gradients_keep_reference_path():
    q = torch.randn(1, 4, 1, 8, requires_grad=True)
    k = torch.randn(1, 2, 1, 8, requires_grad=True)
    cos, sin = torch.randn(1, 8), torch.randn(1, 8)
    ref = apply_rotary_pos_emb(q, k, cos, sin)
    own = apply_rotary_pos_emb(q, k, cos, sin, pack_decode=True)
    expected = torch.autograd.grad(sum(x.sum() for x in ref), (q, k))
    actual = torch.autograd.grad(sum(x.sum() for x in own), (q, k))
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_optimizer_reports_decode_rotary_and_can_disable():
    from saddlellm.models.InferenceOptimizer import optimize_native_inference
    from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig

    model = SaddleForCausalLM(SaddleModelConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    )).eval().cuda()
    report = optimize_native_inference(model, pack_qkv=False, pack_decode_rotary=True)
    assert report.decode_rotary_packed == 2
    assert all(layer.self_attn._pack_decode_rotary for layer in model.layers)
    report = optimize_native_inference(model, pack_qkv=False, pack_decode_rotary=False)
    assert report.decode_rotary_packed == 0
    assert not any(layer.self_attn._pack_decode_rotary for layer in model.layers)
    with pytest.raises(ValueError, match="pack_decode_rotary"):
        optimize_native_inference(model, pack_decode_rotary="yes")
