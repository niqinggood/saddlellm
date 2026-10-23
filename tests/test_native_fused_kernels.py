"""Numerical contracts for opt-in CUDA decode fusion and safe fallbacks."""
import copy
import importlib.util

import pytest
import torch

from saddlellm.models.InferenceOptimizer import optimize_native_inference
from saddlellm.models.NativeFusedKernels import can_fuse_norm, can_fuse_rotary, load_decode_kernels
from saddlellm.models.SaddleModeling import (
    SaddleRMSNorm, SaddleForCausalLM, SaddleModelConfig, apply_rotary_pos_emb,
)

cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec('triton') is None,
    reason='CUDA and optional Triton are required',
)

def small_model():
    return SaddleForCausalLM(SaddleModelConfig(
        vocab_size=257, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        qk_norm=True, rms_norm_weight_in_fp32=False, attention_backend='sdpa',
        max_position_embeddings=128,
    )).eval()


@cuda
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('weight_in_fp32', [True, False])
@pytest.mark.parametrize('shape', [(1,1,2560),(4,1,2560),(2,32,1,128),(3,1,37)])
def test_norm_matches_eager_rounding(dtype, weight_in_fp32, shape):
    torch.manual_seed(43)
    norm = SaddleRMSNorm(shape[-1], weight_in_fp32=weight_in_fp32).cuda().to(dtype).eval()
    fused, _ = load_decode_kernels()
    with torch.inference_mode():
        norm.weight.uniform_(-2,2)
        # Include values large enough that BF16/FP16 variance is unsafe.
        for scale in (0.001, 1, 1000):
            x = torch.randn(shape, device='cuda', dtype=dtype) * scale
            reference = norm(x)
            actual = fused(x, norm.weight, norm.eps, weight_in_fp32)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('batch,qheads,kheads,dim', [(1,32,8,128),(4,8,2,32),(2,3,3,40)])
def test_rotary_preserves_both_product_rounds_and_strides(dtype,batch,qheads,kheads,dim):
    torch.manual_seed(44)
    _, fused = load_decode_kernels()
    # Last-token views exercise nontrivial batch/head strides and offsets.
    q = torch.randn(batch,qheads,5,dim,device='cuda',dtype=dtype)[:,:,4:]
    k = torch.randn(batch,kheads,7,dim,device='cuda',dtype=dtype)[:,:,6:]
    cos = torch.randn(9,dim,device='cuda',dtype=dtype)[8:]
    sin = torch.randn(9,dim,device='cuda',dtype=dtype)[8:]
    with torch.inference_mode():
        assert can_fuse_rotary(q,k,cos,sin)
        expected=apply_rotary_pos_emb(q,k,cos,sin)
        actual=fused(q,k,cos,sin)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_cpu_and_invalid_options_leave_model_unchanged():
    model=small_model()
    params=[id(p) for p in model.parameters()]
    for value,error in [('yes','boolean'),(True,'CUDA')]:
        with pytest.raises(ValueError,match=error):
            optimize_native_inference(model,fused_decode_kernels=value)
        assert [id(p) for p in model.parameters()]==params
    with torch.inference_mode():
        assert not can_fuse_norm(torch.ones(1,1,128),torch.ones(128))


@cuda
def test_model_sequences_scores_weights_hooks_and_training():
    torch.manual_seed(45)
    model=small_model().cuda().bfloat16()
    optimize_native_inference(model,pack_qkv=False)
    reference=copy.deepcopy(model)
    params={key:id(p) for key,p in model.named_parameters()}
    state={key:p.clone() for key,p in model.state_dict().items()}
    report=optimize_native_inference(model,pack_qkv=False,fused_decode_kernels=True)
    assert report.fused_decode_norms==9 and report.fused_decode_rotary==2
    assert params=={key:id(p) for key,p in model.named_parameters()}
    called=[]
    hook=model.norm.register_forward_hook(lambda *args:called.append(True))
    for batch in (1,4):
        ids=torch.randint(10,250,(batch,9),device='cuda')
        a=model.generate(ids,max_new_tokens=24,do_sample=False,eos_token_id=[],output_scores=True,return_dict_in_generate=True)
        b=reference.generate(ids,max_new_tokens=24,do_sample=False,eos_token_id=[],output_scores=True,return_dict_in_generate=True)
        assert torch.equal(a.sequences,b.sequences)
        for left,right in zip(a.scores,b.scores):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert called
    hook.remove()
    assert all(torch.equal(state[key],value) for key,value in model.state_dict().items())
    # Fusion must be bypassed in gradient/training mode.
    def forbidden(*args): raise AssertionError('fused inference kernel reached in training')
    model.norm._fused_decode=forbidden
    model.train()
    ids=torch.tensor([[4,5,6]],device='cuda')
    model(ids,labels=ids).loss.backward()
    assert model.norm.weight.grad is not None


@cuda
def test_fallback_for_prefill_autocast_noncontiguous_and_grad():
    x=torch.randn(2,1,128,device='cuda',dtype=torch.bfloat16)
    weight=torch.ones(128,device='cuda',dtype=x.dtype)
    assert not can_fuse_norm(x,weight)
    with torch.inference_mode():
        assert can_fuse_norm(x,weight)
        assert not can_fuse_norm(x.expand(2,4,128),weight)
        assert not can_fuse_norm(x[:,:,::2],weight[::2])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            assert not can_fuse_norm(x,weight)


@cuda
def test_fused_bf16_cohorts_match_independent_eager_sessions():
    from contextlib import ExitStack
    from saddlellm.models.NativeBatchDecode import NativeBatchDecodeRunner
    from saddlellm.models.NativeFusedKernels import warmup_decode_kernels

    torch.manual_seed(46)
    model=small_model().cuda().bfloat16()
    optimize_native_inference(model,pack_qkv=False)
    ids=[torch.randint(10,250,(1,7),device='cuda') for _ in range(4)]
    references=[model.generate(x,max_new_tokens=12,do_sample=False,eos_token_id=[])
                for x in ids]
    optimize_native_inference(model,pack_qkv=False,fused_decode_kernels=True)
    warmup_decode_kernels(model,max_batch_size=4)
    with ExitStack() as stack:
        sessions=[stack.enter_context(model.create_generation_session(x,max_new_tokens=12,do_sample=False,eos_token_id=[])) for x in ids]
        runner=stack.enter_context(NativeBatchDecodeRunner(model,max_batch_size=4))
        for step in range(12):
            runner.step(sessions[step%4:]+sessions[:step%4])
        assert runner.snapshot()['rowwise_linear_forwards']>0
        assert all(torch.equal(s.sequences(),r) for s,r in zip(sessions,references))


@pytest.mark.parametrize('values', [dict(native_fused_kernels=1),dict(backend='vllm'),dict(device='cpu'),dict(quantization='int8')])
def test_serving_rejects_incompatible_fusion(values):
    from saddlellm.runtime.InferenceServer import InferenceServerSettings
    with pytest.raises(ValueError,match='native_fused_kernels'):
        InferenceServerSettings(**(dict(backend='native',device='cuda',native_fused_kernels=True)|values))


def test_cli_selects_fused_kernels(monkeypatch):
    import uvicorn
    from saddlellm.cli import main
    from saddlellm.runtime import InferenceServer

    settings=[]
    monkeypatch.setattr(InferenceServer,'create_inference_app',lambda value:settings.append(value))
    monkeypatch.setattr(uvicorn,'run',lambda *args,**kwargs:None)
    assert main(['serve','model','--backend','native','--device','cuda','--native-fused-kernels','--port','0'])==0
    assert settings[0].native_fused_kernels is True
