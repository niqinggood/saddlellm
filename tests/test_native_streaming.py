import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
import torch

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.runtime.InferenceServer import (
    CompletionRequest,
    InferenceServerSettings,
    LocalGenerationEngine,
    create_inference_app,
)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(self, ids, **kwargs):
        return " ".join(str(item) for item in ids if item not in {0, 2})


def engine(tokenizer=None):
    torch.manual_seed(173)
    instance = SaddleForCausalLM(
        SaddleModelConfig(
            vocab_size=13,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            eos_token_id=[],
        )
    ).eval()
    return LocalGenerationEngine(
        instance,
        tokenizer or Tokenizer(),
        InferenceServerSettings(
            default_max_new_tokens=4,
            max_new_tokens=8,
        ),
    )


def stream(instance, state=None, **kwargs):
    return instance._stream_generate(
        "prompt",
        max_tokens=4,
        temperature=0,
        top_p=1,
        repetition_penalty=1,
        stop=kwargs.pop("stop", None),
        stream_state=state,
        **kwargs,
    )


def capture_sessions(monkeypatch, instance):
    created = []
    original = instance.model.create_generation_session

    def create(*args, **kwargs):
        session = original(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(instance.model, "create_generation_session", create)
    return created


def test_native_stream_needs_no_background_thread_and_matches_complete(monkeypatch):
    instance = engine()
    reference = instance.complete(
        CompletionRequest(prompt="prompt", temperature=0, top_p=1, max_tokens=4)
    )
    created = capture_sessions(monkeypatch, instance)
    monkeypatch.setattr(
        threading, "Thread", lambda *a, **kw: pytest.fail("unexpected model thread")
    )
    state = {}
    pieces = list(stream(instance, state))
    assert "".join(pieces) == reference["choices"][0]["text"]
    assert state["completion_tokens"] == 4
    assert state["finish_reason"] == "length"
    assert state["worker_done"].is_set()
    assert created[0].closed and created[0].past_key_values is None


def test_stop_split_across_tokens_is_trimmed_and_stops_actual_generation(monkeypatch):
    class SplitTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return "".join(
                {
                    0: "",
                    2: "",
                    4: "prompt<STOP>",
                    5: "",
                    8: "hello ",
                    9: "<ST",
                    10: "OP>",
                    11: " ignored",
                }[item]
                for item in ids
            )

    instance = engine(SplitTokenizer())
    created = capture_sessions(monkeypatch, instance)

    def planned(input_ids, **kwargs):
        scores = torch.full((1, 1, 13), -100.0)
        scores[0, 0, [8, 9, 10, 11][input_ids.shape[1] - 2]] = 10
        return SimpleNamespace(logits=scores, past_key_values=None)

    monkeypatch.setattr(instance.model, "forward", planned)
    state = {}
    assert "".join(stream(instance, state, stop="<STOP>")) == "hello "
    assert state["completion_tokens"] == 3
    assert state["finish_reason"] == "stop"
    assert created[0].generated_tokens == 3 and created[0].closed


def test_pre_cancelled_stream_does_not_start_model(monkeypatch):
    instance = engine()
    event = threading.Event()
    event.set()
    monkeypatch.setattr(
        instance.model,
        "create_generation_session",
        lambda **kw: pytest.fail("cancelled request started generation"),
    )
    state = {"cancel_event": event}
    assert list(stream(instance, state)) == []
    assert state["completion_tokens"] == 0
    assert state["finish_reason"] == "stop"
    assert state["worker_done"].is_set()


def test_closing_partially_consumed_stream_releases_cache_without_more_steps(
    monkeypatch,
):
    instance = engine()
    created = capture_sessions(monkeypatch, instance)
    state = {}
    iterator = stream(instance, state)
    assert next(iterator)
    count = created[0].generated_tokens
    iterator.close()
    assert created[0].generated_tokens == count
    assert created[0].closed and created[0].buffers is None
    assert created[0].past_key_values is None
    assert state["worker_done"].is_set()
    assert state["completion_tokens"] == count


def test_failed_native_stream_closes_its_request_state(monkeypatch):
    instance = engine()
    created = capture_sessions(monkeypatch, instance)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(instance.model, "forward", fail)
    state = {}
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        list(stream(instance, state))
    assert created[0].closed and created[0].past_key_values is None
    assert state["worker_done"].is_set()


def test_eos_at_exact_token_budget_is_stop_for_complete_and_stream(monkeypatch):
    instance = engine()
    instance.model.config.eos_token_id = 2

    def planned(input_ids, **kwargs):
        scores = torch.full((1, 1, 13), -100.0)
        scores[0, 0, 9 if input_ids.shape[1] == 2 else 2] = 10
        return SimpleNamespace(logits=scores, past_key_values=None)

    monkeypatch.setattr(instance.model, "forward", planned)
    completion = instance.complete(
        CompletionRequest(prompt="prompt", temperature=0, max_tokens=2)
    )
    state = {}
    text = "".join(
        instance._stream_generate(
            "prompt",
            max_tokens=2,
            temperature=0,
            top_p=1,
            repetition_penalty=1,
            stop=None,
            stream_state=state,
        )
    )
    assert completion["choices"][0]["finish_reason"] == state["finish_reason"] == "stop"
    assert completion["usage"]["completion_tokens"] == state["completion_tokens"] == 2
    assert completion["choices"][0]["text"] == text == "9"


def test_generate_override_keeps_existing_worker_stream_path(monkeypatch):
    instance = engine()
    original = instance.model.generate
    calls = []

    def custom(**kwargs):
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(instance.model, "generate", custom)
    assert not instance.supports_step_generation
    state = {}
    assert "".join(stream(instance, state))
    assert calls == [1]
    assert state["worker_done"].is_set()


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
def test_asgi_disconnect_waits_for_prefill_before_releasing_model_permit(monkeypatch, endpoint):
    async def run():
        instance = engine()
        app = create_inference_app(
            instance.settings, model=instance.model, tokenizer=instance.tokenizer
        )
        created = capture_sessions(monkeypatch, instance)
        original = instance.model.forward
        entered, release, followup_entered = (threading.Event() for _ in range(3))
        disconnected = asyncio.Event()
        calls = []

        def blocked(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                assert release.wait(5)
            else:
                followup_entered.set()
            return original(*args, **kwargs)

        monkeypatch.setattr(instance.model, "forward", blocked)
        body = json.dumps(
            {
                **({"messages": [{"role": "user", "content": "hello"}]} if endpoint == "chat/completions"
                   else {"prompt": "hello"}),
                "temperature": 0,
                "max_tokens": 4,
                "stream": True,
            }
        ).encode()
        first = True
        sent = []

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": f"/v1/{endpoint}",
            "raw_path": f"/v1/{endpoint}".encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
        }
        http_task = asyncio.create_task(app(scope, receive, send))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            disconnected.set()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                followup = asyncio.create_task(
                    client.post(
                        "/v1/completions",
                        json={
                            "prompt": "hello",
                            "temperature": 0,
                            "max_tokens": 1,
                        },
                    )
                )
                await asyncio.sleep(0.03)
                assert not followup_entered.is_set()
                assert not http_task.done()
                release.set()
                await asyncio.wait_for(http_task, 5)
                assert (await followup).status_code == 200
        finally:
            release.set()
            await asyncio.wait_for(http_task, 5)
        assert created[0].closed and created[0].buffers is None
        assert created[0].generated_tokens == 1
        assert followup_entered.is_set()
        assert sent[0]["type"] == "http.response.start"

    asyncio.run(run())
