"""Official SDK/protocol tests; fixture outputs do not measure 8B quality/speed."""

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest
import torch

from saddlellm.runtime.InferenceServer import (
    InferenceServerSettings,
    create_inference_app,
)
from saddlellm.runtime.VLLMServing import VLLMRuntime, create_vllm_inference_app


class Tokenizer:
    pad_token_id = 0
    eos_token_id = None

    def apply_chat_template(self, messages, **options):
        self.messages, self.options = messages, options
        return " ".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return [3 + index for index, _ in enumerate(text.split())]

    def __call__(self, text, return_tensors, truncation, max_length):
        tokens = self.encode(text)
        if truncation:
            tokens = tokens[:max_length]
        return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones(1, len(tokens), dtype=torch.long)}

    def decode(self, ids, **kwargs):
        return " ".join(f"t{int(value)}" for value in ids)


class Model:
    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=128, eos_token_id=None)

    def __init__(self):
        self.calls = []
        self.fail = False

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("PRIVATE_MODEL_EXCEPTION")
        tokens = torch.arange(10, 10 + kwargs["max_new_tokens"]).unsqueeze(0)
        if kwargs.get("streamer"):
            streamer = kwargs["streamer"]
            streamer.put(input_ids)
            for token in tokens[0]:
                streamer.put(token.reshape(1))
            streamer.end()
        return torch.cat([input_ids, tokens], dim=-1)


class AsyncEngine:
    model_config = SimpleNamespace(max_model_len=128, dtype="bfloat16", quantization=None)

    def __init__(self):
        self.calls = []
        self.fail = False

    async def check_health(self):
        pass

    def shutdown(self, timeout):
        pass

    async def abort(self, request_id):
        pass

    async def generate(self, prompt, params, request_id):
        self.calls.append(params)
        if self.fail:
            raise RuntimeError("PRIVATE_MODEL_EXCEPTION")
        for count in range(1, params["max_tokens"] + 1):
            yield SimpleNamespace(finished=count == params["max_tokens"], outputs=[SimpleNamespace(
                text="a" * count, token_ids=list(range(count)),
                finish_reason="length" if count == params["max_tokens"] else None,
            )])


def setup(backend, **options):
    settings = InferenceServerSettings(backend=backend, model_name="served-model", model_path="offline",
                                       default_max_new_tokens=4, max_new_tokens=8, web_ui=False,
                                       api_key="dummy-local-test-key", **options)
    tokenizer = Tokenizer()
    if backend == "vllm":
        model = AsyncEngine()
        runtime = VLLMRuntime(settings, engine=model, tokenizer=tokenizer,
                              sampling_factory=lambda **kwargs: kwargs)
        app = create_vllm_inference_app(settings, runtime=runtime)
    else:
        model = Model()
        app = create_inference_app(settings, model=model, tokenizer=tokenizer)
    return app, model, tokenizer


def sdk_for(app, key="dummy-local-test-key"):
    # No OpenAI network or paid API calls: this transport only reaches the in-process app.
    return openai.AsyncOpenAI(api_key=key, base_url="http://loopback/v1", max_retries=0,
                              http_client=httpx.AsyncClient(transport=httpx.ASGITransport(
                                  app=app, raise_app_exceptions=False)))


@pytest.mark.parametrize("backend", ["local", "vllm"])
def test_sdk_models_chat_alias_text_parts_developer_and_stream_usage(backend):
    async def run():
        app, model, tokenizer = setup(backend)
        async with app.router.lifespan_context(app), sdk_for(app) as sdk:
            assert (await sdk.models.list()).data[0].id == "served-model"
            messages = [{"role": "developer", "content": "Be concise"},
                        {"role": "user", "content": [{"type": "text", "text": "hello"}]}]
            result = await sdk.chat.completions.create(model="served-model", messages=messages,
                                                       max_completion_tokens=2, temperature=0)
            assert result.usage.completion_tokens == 2
            assert result.choices[0].finish_reason == "length"
            assert model.calls[0]["max_tokens" if backend == "vllm" else "max_new_tokens"] == 2
            assert tokenizer.messages[0]["role"] == "system"
            assert tokenizer.messages[1]["content"] == "hello"
            assert "enable_thinking" not in tokenizer.options
            stream = await sdk.chat.completions.create(
                model="served-model", messages=messages, temperature=0,
                max_completion_tokens=2, stream=True, stream_options={"include_usage": True},
            )
            chunks = [chunk async for chunk in stream]
            assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 2
            assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
            assert {chunk.id for chunk in chunks} == {chunks[0].id}
            response = await sdk.completions.create(model="served-model", prompt="hello", max_tokens=2, temperature=0)
            assert response.usage.completion_tokens == 2 and response.choices[0].text

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["local", "vllm"])
def test_sdk_errors_model_names_unsupported_tools_and_secret_redaction(backend):
    async def run():
        app, model, _ = setup(backend)
        async with app.router.lifespan_context(app), sdk_for(app) as sdk:
            with pytest.raises(openai.NotFoundError) as failure:
                await sdk.completions.create(model="wrong-model", prompt="hello", max_tokens=2)
            assert failure.value.body["type"] == "invalid_request_error"
            with pytest.raises(openai.UnprocessableEntityError) as failure:
                await sdk.chat.completions.create(model="served-model", messages=[{"role": "user", "content": "PRIVATE_PROMPT"}],
                                                  tools=[{"type": "function", "function": {"name": "PRIVATE_TOOL"}}])
            assert "PRIVATE_PROMPT" not in str(failure.value.body)
            assert "PRIVATE_TOOL" not in str(failure.value.body)
            assert failure.value.body["param"] == "tools" and not model.calls
            model.fail = True
            with pytest.raises(openai.InternalServerError) as failure:
                await sdk.completions.create(model="served-model", prompt="hello", max_tokens=2)
            assert "PRIVATE_MODEL_EXCEPTION" not in str(failure.value.body)
        async with sdk_for(app, "wrong-test-key") as sdk:
            with pytest.raises(openai.AuthenticationError):
                await sdk.models.list()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["local", "vllm"])
@pytest.mark.parametrize("thinking", [None, False, True])
def test_template_thinking_policy_is_explicit_and_not_a_hidden_speed_switch(backend, thinking):
    async def run():
        app, _, tokenizer = setup(backend, enable_thinking=thinking)
        async with app.router.lifespan_context(app), sdk_for(app) as sdk:
            await sdk.chat.completions.create(model="served-model", messages=[{"role": "user", "content": "hello"}], max_tokens=2)
            if thinking is None:
                assert "enable_thinking" not in tokenizer.options
            else:
                assert tokenizer.options["enable_thinking"] is thinking

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_local_rejects_overlong_prompt_before_generation_or_sse_headers(stream):
    async def run():
        app, model, _ = setup("local", max_input_tokens=3)
        async with app.router.lifespan_context(app), sdk_for(app) as sdk:
            with pytest.raises(openai.BadRequestError):
                await sdk.completions.create(model="served-model", prompt="one two three four", max_tokens=2, stream=stream)
            assert not model.calls

    asyncio.run(run())


def test_legacy_prompt_truncation_requires_an_explicit_setting():
    async def run():
        app, _, _ = setup("local", max_input_tokens=3, truncate_input=True)
        async with app.router.lifespan_context(app), sdk_for(app) as sdk:
            result = await sdk.completions.create(model="served-model", prompt="one two three four", max_tokens=2)
            assert result.usage.prompt_tokens == 3

    asyncio.run(run())


@pytest.mark.parametrize("fields", [
    {"max_tokens": 2, "max_completion_tokens": 3},
    {"max_completion_tokens": True}, {"max_completion_tokens": 1.5},
    {"response_format": {"type": "json_object"}}, {"seed": 1},
])
def test_unsupported_or_ambiguous_parameters_are_not_silently_ignored(fields):
    async def run():
        app, model, _ = setup("local")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer dummy-local-test-key"},
                                         json={"messages": [{"role": "user", "content": "PRIVATE"}], **fields})
            assert response.status_code == 422 and "error" in response.json()
            assert "PRIVATE" not in response.text and not model.calls

    asyncio.run(run())


def test_local_admission_bounds_waiters_but_keeps_health_available():
    async def run():
        app, _, _ = setup("local", max_concurrency=1, max_pending_requests=1)
        entered, release = asyncio.Event(), asyncio.Event()

        async def held(scope, receive, send):
            entered.set()
            await release.wait()

        from saddlellm.runtime.InferenceAdmission import (
            LocalAdmissionMiddleware,
            LocalHTTPAdmission,
        )

        admission = LocalHTTPAdmission(1)
        middleware = LocalAdmissionMiddleware(held, admission)
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}
        task = asyncio.create_task(middleware(scope, None, None))
        await entered.wait()
        messages = []

        async def send(message):
            messages.append(message)

        await middleware(scope, None, send)
        assert messages[0]["status"] == 503 and admission.rejected == 1
        assert admission.active == 1
        release.set()
        await task
        assert admission.active == 0
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/health")).json()["admission"]["limit"] == 2

    asyncio.run(run())
