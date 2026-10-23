"""Chunked prompt correctness, bounded scheduling, cancellation and SSE integration."""
import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.NativeStreamBatcher import NativeStreamBatcher
from saddlellm.runtime.NativeTokenScheduler import NativeTokenScheduler


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(backend='sdpa',**kwargs):
    torch.manual_seed(283)
    return SaddleForCausalLM(SaddleModelConfig(vocab_size=41,hidden_size=32,
        intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=2,max_position_embeddings=128,attention_backend=backend,**kwargs)).eval()


def session(m, ids, **kwargs):
    return m.create_generation_session(torch.tensor([ids]),max_new_tokens=4,do_sample=False,
        eos_token_id=[],output_scores=True,**kwargs)


def batcher(m, **kwargs):
    return NativeStreamBatcher(m,max_batch_size=2,max_resident=4,max_batch_tokens=512,
        wait_ms=0,prefill_chunk_size=4,max_step_tokens=5,**kwargs)


@pytest.mark.parametrize('backend',['eager','sdpa','sdpa_reference'])
@pytest.mark.parametrize('chunk',[1,3,7,64])
def test_chunked_logits_tokens_budget_and_skipped_intermediate_head(backend,chunk):
    m=model(backend,qk_norm=True);ids=list(range(4,21))
    with session(m,ids) as ref:
        while not ref.done:ref.step()
        expected=ref.sequences();scores=ref.scores
    calls=[];head=[]
    hook=m.register_forward_pre_hook(lambda _,a,kw:calls.append(kw['input_ids'].shape[1]),with_kwargs=True)
    head_hook=m.lm_head.register_forward_hook(lambda *_:head.append(True))
    try:
        with session(m,ids) as s:
            computed=0
            while s.prefill_remaining_tokens:
                step=s.prefill_chunk(chunk);computed+=step.computed_tokens
                assert 0<step.computed_tokens<=chunk
                assert computed+step.remaining_tokens==len(ids)
                assert s.generated_tokens==0
                assert all(c.length==computed for c in s.past_key_values)
            assert len(head)==1
            count=len(calls)
            assert s.prefill_chunk(chunk).computed_tokens==0
            s.step();assert len(calls)==count
            while not s.done:s.step()
            assert s.prefill_computed_tokens==len(ids)
            assert torch.equal(expected,s.sequences())
            for a,b in zip(scores,s.scores):torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    finally:hook.remove();head_hook.remove()


@pytest.mark.parametrize('resume',['step','prefill'])
def test_partial_prompt_can_resume_through_original_api(resume):
    m=model();ids=list(range(4,19))
    reference=m.generate(torch.tensor([ids]),max_new_tokens=4,do_sample=False,eos_token_id=[])
    with session(m,ids) as s:
        s.prefill_chunk(3)
        if resume=='prefill':s.prefill()
        while not s.done:s.step()
        assert torch.equal(reference,s.sequences())
        assert s.prefill_computed_tokens==len(ids)


@pytest.mark.parametrize('invalid',[0,-1,True,1.5,None])
def test_invalid_budget_does_not_mutate_session(invalid):
    with session(model(),[4,5,6]) as s:
        with pytest.raises(ValueError,match='budget'):s.prefill_chunk(invalid)
        assert all(c.length==0 for c in s.past_key_values)
        assert not s._failed


def test_unsupported_sessions_are_explicit_and_keep_normal_path():
    for kwargs in [dict(use_cache=False),dict(cache_implementation='dynamic'),dict(attention_mask=torch.tensor([[0,1,1]]))]:
        with session(model(),[4,5,6],**kwargs) as s:
            with pytest.raises(NotImplementedError):s.prefill_chunk(2)
            assert not s._failed
            s.step()


def test_scheduler_decode_priority_and_computation_budget():
    schedule=NativeTokenScheduler(max_sequences=4,max_step_tokens=8,prefill_chunk_size=6)
    sessions=[SimpleNamespace(prefill_remaining_tokens=100,generated_tokens=0),
              SimpleNamespace(prefill_remaining_tokens=0,generated_tokens=2),
              SimpleNamespace(prefill_remaining_tokens=0,generated_tokens=3),
              SimpleNamespace(prefill_remaining_tokens=9,generated_tokens=0)]
    plan=schedule.plan(sessions)
    assert [(p.index,p.stage,p.tokens) for p in plan]==[(1,'decode',1),(2,'decode',1),(0,'prefill',6)]
    assert sum(p.tokens for p in plan)==8


def test_live_decode_finishes_before_long_prefill_continues():
    m=model();long=session(m,[4]*21);short=session(m,[7]*3);short.step()
    runner=batcher(m);entered=threading.Event();release=threading.Event();trace=[]
    def observe(_,args,kwargs):
        size=kwargs['input_ids'].shape[1];trace.append(size)
        if len(trace)==1:
            entered.set();assert release.wait(5)
    hook=m.register_forward_pre_hook(observe,with_kwargs=True)
    try:
        runner.register(long);runner.register(short)
        with ThreadPoolExecutor(2) as pool:
            lf=pool.submit(runner.step,long,threading.Event())
            assert entered.wait(5)
            sf=pool.submit(runner.step,short,threading.Event())
            deadline=time.monotonic()+5
            while runner.snapshot()['pending']<1 and time.monotonic()<deadline:time.sleep(.001)
            assert runner.snapshot()['pending']>=1
            release.set()
            assert sf.result(5).generated_tokens==2
            assert lf.result(5).generated_tokens==1
        assert trace[:2]==[4,1]
        assert max(trace)<=4
        status=runner.snapshot()['scheduler']
        assert status['prefill_tokens']==21 and status['prefill_chunks']==6
        assert status['peak_step_tokens']<=5
    finally:
        release.set();hook.remove()
        runner.unregister(long);runner.unregister(short)
        long.close();short.close();runner.close()


def test_cancellation_after_a_chunk_stops_prompt_work_and_releases_references():
    m=model();s=session(m,[4]*30);cancel=threading.Event();runner=batcher(m)
    reference=weakref.ref(s)
    hook=m.register_forward_hook(lambda *_:cancel.set())
    try:
        runner.register(s)
        assert runner.step(s,cancel) is None
        assert s.generated_tokens==0 and s.prefill_computed_tokens==4
        runner.unregister(s);s.close();del s
        deadline=time.monotonic()+2
        while reference() is not None and time.monotonic()<deadline:gc.collect();time.sleep(.005)
        assert reference() is None
        snap=runner.snapshot()
        assert snap['pending']==snap['inflight']==snap['registered']==0
        assert snap['cancelled_steps']==1
    finally:hook.remove();runner.close()


def test_failed_chunk_cannot_resume_and_worker_serves_next_request():
    m=model();runner=batcher(m);bad=session(m,[4]*19)
    def fail(*_):raise RuntimeError('injected chunk failure')
    hook=m.layers[1].register_forward_pre_hook(fail)
    try:
        runner.register(bad)
        with pytest.raises(RuntimeError,match='injected'):runner.step(bad,threading.Event())
        with pytest.raises(RuntimeError,match='failed'):bad.step()
        runner.unregister(bad);bad.close();hook.remove()
        with session(m,[7]*3) as good:
            runner.register(good)
            assert runner.step(good,threading.Event()).generated_tokens==1
            runner.unregister(good)
    finally:hook.remove();runner.close()


@pytest.mark.parametrize('endpoint',['completions','chat/completions'])
@pytest.mark.parametrize('streaming',[False,True])
@pytest.mark.parametrize('temperature',[0,0.7])
def test_all_http_generation_routes_use_one_token_scheduler(endpoint,streaming,temperature):
    import asyncio
    import json
    import httpx
    from saddlellm.runtime.InferenceServer import InferenceServerSettings,create_inference_app

    class Tokenizer:
        pad_token_id=0
        eos_token_id=2
        def __call__(self,text,**kwargs):
            ids=torch.tensor([[4,5,6,7,8,9,10,11,12]])
            return dict(input_ids=ids,attention_mask=torch.ones_like(ids))
        def decode(self,ids,**kwargs):return ' '.join(str(int(x)) for x in ids)
        def apply_chat_template(self,*args,**kwargs):return 'prompt'
    m=model(eos_token_id=[])
    options=InferenceServerSettings(backend='native',native_stream_batching=True,
        max_concurrency=2,batch_size=2,batch_token_budget=256,max_input_tokens=32,
        max_new_tokens=4,default_max_new_tokens=4,native_prefill_chunk_size=3,
        native_max_step_tokens=4,web_ui=False)
    app=create_inference_app(options,model=m,tokenizer=Tokenizer())
    assert app.state.batcher is None
    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                body=dict(temperature=temperature,max_tokens=4,stream=streaming)
                body.update(dict(prompt='prompt',echo=True) if endpoint=='completions' else dict(messages=[dict(role='user',content='prompt')]))
                if streaming:body['stream_options']={'include_usage':True}
                response=await asyncio.wait_for(client.post('/v1/'+endpoint,json=body),10)
                assert response.status_code==200
                if streaming:
                    parts=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
                    assert all('error' not in part for part in parts)
                    usage=[part['usage'] for part in parts if part.get('usage')][-1]
                else:usage=response.json()['usage']
                assert usage==dict(prompt_tokens=9,completion_tokens=4,total_tokens=13)
                health=(await client.get('/health')).json()
                assert health['stream_scheduling']=='native_token_budget'
                status=health['native_stream_batching']
                assert status['scheduler']['prefill_chunks']==3
                assert status['scheduler']['prefill_tokens']==9
                assert status['scheduler']['peak_step_tokens']<=4
                assert status['pending']==status['inflight']==status['registered']==0
    asyncio.run(run())


@pytest.mark.parametrize('values',[{'native_prefill_chunk_size':True},{'native_max_step_tokens':1},
    {'prefix_cache_mb':16},{'native_stream_batching':False},{'native_max_step_tokens':-2}])
def test_settings_reject_incompatible_scheduler_options(values):
    from saddlellm.runtime.InferenceServer import InferenceServerSettings
    with pytest.raises(ValueError):
        InferenceServerSettings(**(dict(backend='native',native_stream_batching=True,max_concurrency=2,
            batch_size=2,native_prefill_chunk_size=3)|values))


def test_cli_scheduler_options(monkeypatch):
    import uvicorn
    from saddlellm.cli import main
    from saddlellm.runtime import InferenceServer
    configured=[]
    monkeypatch.setattr(InferenceServer,'create_inference_app',lambda settings:configured.append(settings))
    monkeypatch.setattr(uvicorn,'run',lambda *a,**kw:None)
    assert main(['serve','model','--backend','native','--native-stream-batching',
        '--max-concurrency','4','--batch-size','4','--native-prefill-chunk-size','128',
        '--native-max-step-tokens','256','--port','0'])==0
    assert configured[0].native_prefill_chunk_size==128
    assert configured[0].native_max_step_tokens==256


def test_prefill_only_forward_is_inference_only():
    m=model();ids=torch.tensor([[4,5,6]])
    with pytest.raises(ValueError,match='prefill-only'):
        m(ids,use_cache=True,_prefill_only=True)
    with torch.inference_mode():
        with pytest.raises(ValueError,match='prefill-only'):
            m(ids,use_cache=False,_prefill_only=True)
        output=m(ids,use_cache=True,_prefill_only=True)
        assert output.logits is None and output.past_key_values is not None
