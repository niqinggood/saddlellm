import http.client
import json
import errno
import ssl
import urllib.error

import pytest

import saddlellm.agents.AgentModelProvider as provider_module
from saddlellm.agents.AgentModelProvider import ModelProviderError, endpoint


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


def test_transient_remote_disconnect_is_retried_once(monkeypatch):
    attempts = []

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
                            "message": {"role": "assistant", "content": "Recovered"},
                        }
                    ]
                }
            ).encode("utf-8")

    def open_request(_request, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise http.client.RemoteDisconnected("remote closed without response")
        return Response()

    sleeps = []
    monkeypatch.setenv("MINIMAX_API_KEY", "test-only-key")
    monkeypatch.setattr(provider_module, "_validate_provider_host", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(provider_module, "_open_provider_request", open_request)
    monkeypatch.setattr(provider_module.time, "sleep", sleeps.append)

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

    assert attempts == [9, 9]
    assert sleeps == [provider_module.PROVIDER_RETRY_DELAY_SECONDS]
    assert message["content"] == "Recovered"


def test_transient_retry_can_be_disabled(monkeypatch):
    attempts = []

    def open_request(_request, timeout):
        del timeout
        attempts.append(1)
        raise ConnectionResetError("connection reset")

    monkeypatch.setenv("MINIMAX_API_KEY", "test-only-key")
    monkeypatch.setattr(provider_module, "_validate_provider_host", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(provider_module, "_open_provider_request", open_request)

    with pytest.raises(ModelProviderError, match="connection reset"):
        provider_module.chat_completion(
            {
                "baseUrl": "https://api.minimaxi.com/v1",
                "model": "MiniMax-M3",
                "credentialEnv": "MINIMAX_API_KEY",
                "maxRetries": 0,
            },
            {"model": "MiniMax-M3"},
            [{"role": "user", "content": "test"}],
        )

    assert attempts == [1]


def test_wrapped_tls_enoent_has_a_bounded_retry_and_preserves_unknown_attempt_usage(monkeypatch):
    attempts = []
    class Response:
        def __enter__(self):return self
        def __exit__(self,*_):return False
        def read(self,_):
            return json.dumps({'choices':[{'message':{'content':'Recovered'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':10,'completion_tokens':4,'total_tokens':14,'unexpected_secret':'must not propagate'}}).encode()
    def open_request(*_,**__):
        attempts.append(1)
        if len(attempts)==1:raise urllib.error.URLError(FileNotFoundError(errno.ENOENT,'TLS handshake failed'))
        return Response()
    monkeypatch.setattr(provider_module,'_open_provider_request',open_request)
    monkeypatch.setattr(provider_module.time,'sleep',lambda _:None)
    message=provider_module.chat_completion({'baseUrl':'http://127.0.0.1/v1','model':'fixture'}, {}, [])
    assert len(attempts)==2
    assert message['_provider_attempts']==[
        {'attempt':1,'status':'transport_failed','usage':None},
        {'attempt':2,'status':'response_received','usage':{'prompt_tokens':10,'completion_tokens':4,'total_tokens':14}}]
    assert 'must not propagate' not in json.dumps(message)


@pytest.mark.parametrize('error,expected_calls',[
    (urllib.error.URLError(FileNotFoundError(errno.ENOENT,'handshake failed')),2),
    (urllib.error.URLError(ssl.SSLCertVerificationError('invalid certificate')),1),
    (urllib.error.HTTPError('http://127.0.0.1/v1/chat/completions',401,'unauthorized',{},None),1),
], ids=['wrapped-handshake', 'certificate-verification', 'authentication'])
def test_retry_exhaustion_and_nonretryable_tls_or_auth_failures_keep_attempt_records(monkeypatch,error,expected_calls):
    attempts=[]
    def open_request(*_,**__):attempts.append(1);raise error
    monkeypatch.setattr(provider_module,'_open_provider_request',open_request)
    monkeypatch.setattr(provider_module.time,'sleep',lambda _:None)
    with pytest.raises(ModelProviderError) as failure:
        provider_module.chat_completion({'baseUrl':'http://127.0.0.1/v1','model':'fixture'}, {}, [])
    assert len(attempts)==len(failure.value.provider_attempts)==expected_calls
    assert all(row['usage'] is None for row in failure.value.provider_attempts)
