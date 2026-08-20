import json

import pytest

import saddle_llm.AgentModelProvider as provider_module
from saddle_llm.AgentModelProvider import ModelProviderError, endpoint


def test_endpoint_builds_openai_compatible_chat_path():
    assert endpoint("https://api.example.com/v1") == "https://api.example.com/v1/chat/completions"
    assert endpoint("https://api.example.com/v1/chat/completions") == (
        "https://api.example.com/v1/chat/completions"
    )
    assert endpoint("http://127.0.0.1:11434/v1") == (
        "http://127.0.0.1:11434/v1/chat/completions"
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.com/v1",
        "https://user:password@api.example.com/v1",
        "https://api.example.com/v1?api_key=secret",
        "https://api.example.com/v1#secret",
        "https://169.254.169.254/latest/meta-data",
        "https://127.0.0.2/v1",
        "file:///tmp/provider",
    ],
)
def test_endpoint_rejects_credentials_plaintext_remote_and_reserved_networks(base_url):
    with pytest.raises(ModelProviderError):
        endpoint(base_url)


def test_private_provider_requires_explicit_opt_in_and_tls():
    with pytest.raises(ModelProviderError):
        endpoint("https://10.0.0.5/v1")
    assert endpoint("https://10.0.0.5/v1", allow_private_network=True) == (
        "https://10.0.0.5/v1/chat/completions"
    )
    with pytest.raises(ModelProviderError):
        endpoint("http://10.0.0.5/v1", allow_private_network=True)


def test_minimax_request_splits_reasoning_out_of_display_content(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "<think>private reasoning</think>Final answer",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def open_request(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("MINIMAX_API_KEY", "test-only-key")
    monkeypatch.setattr(provider_module, "_validate_provider_host", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(provider_module, "_open_provider_request", open_request)

    message = provider_module.chat_completion(
        {
            "baseUrl": "https://api.minimaxi.com/v1",
            "model": "MiniMax-M3",
            "credentialEnv": "MINIMAX_API_KEY",
            "timeoutSeconds": 9,
        },
        {"model": "MiniMax-M3", "temperature": 0.1, "maxTokens": 128},
        [{"role": "user", "content": "test"}],
    )

    assert captured["payload"]["reasoning_split"] is True
    assert captured["timeout"] == 9
    assert message["content"] == "Final answer"
    assert "private reasoning" not in message["content"]
    assert message["reasoning_details"][0]["text"] == "private reasoning"
    assert message["_finish_reason"] == "stop"
