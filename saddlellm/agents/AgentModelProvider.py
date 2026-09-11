"""Model-provider adapter used by ``saddle_ml.agent``.

Agent orchestration deliberately lives in :mod:`saddle_ml.agent`.  This module
only translates a model request to the OpenAI-compatible chat-completions
protocol and applies network/credential safety checks.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Mapping, Sequence


MAX_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_PROVIDER_MAX_RETRIES = 1
MAX_PROVIDER_MAX_RETRIES = 2
PROVIDER_RETRY_DELAY_SECONDS = 0.25
_PRIVATE_PROVIDER_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)


class ModelProviderError(RuntimeError):
    """Raised when a model provider request is invalid or fails safely."""


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def _separate_inline_reasoning(message: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep reasoning available for model continuity but out of user content."""

    cleaned = dict(message)
    content = str(cleaned.get("content") or "")
    extracted = [match.strip() for match in _THINK_BLOCK.findall(content) if match.strip()]
    content = _THINK_BLOCK.sub("", content)
    open_tag = re.search(r"<think>", content, re.IGNORECASE)
    if open_tag:
        unfinished = content[open_tag.end():].strip()
        if unfinished:
            extracted.append(unfinished)
        content = content[:open_tag.start()]
    cleaned["content"] = content.strip()
    if extracted and not cleaned.get("reasoning_details"):
        cleaned["reasoning_details"] = [
            {"type": "reasoning.text", "text": "\n\n".join(extracted)}
        ]
    return cleaned


def _private_provider_address(address: Any) -> bool:
    return any(address in network for network in _PRIVATE_PROVIDER_NETWORKS)


def _validate_provider_host(
    hostname: str,
    *,
    resolve: bool = False,
    allow_private: bool = False,
) -> None:
    hostname = str(hostname or "").strip().lower()
    if not hostname:
        raise ModelProviderError("provider.baseUrl must include a hostname")
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if literal.is_link_local or literal.is_loopback or literal.is_multicast or literal.is_unspecified:
            raise ModelProviderError("provider.baseUrl cannot target link-local or reserved networks")
        if not literal.is_global and not (allow_private and _private_provider_address(literal)):
            raise ModelProviderError("provider.baseUrl cannot target private or reserved networks")
        return
    if not resolve:
        return
    try:
        addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ModelProviderError("provider hostname could not be resolved") from exc
    resolved_addresses = {item[4][0] for item in addresses if item and item[4]}
    if not resolved_addresses:
        raise ModelProviderError("provider hostname did not resolve to an address")
    for value in resolved_addresses:
        try:
            resolved = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ModelProviderError("provider hostname resolved to an invalid address") from exc
        if resolved.is_link_local or resolved.is_loopback or resolved.is_multicast or resolved.is_unspecified:
            raise ModelProviderError("provider hostname resolved to a blocked address")
        if not resolved.is_global and not (allow_private and _private_provider_address(resolved)):
            raise ModelProviderError("provider hostname resolves to a private or reserved network")


def endpoint(base_url: str, allow_private_network: bool = False) -> str:
    """Return a validated chat-completions endpoint."""

    if not str(base_url or "").strip():
        raise ModelProviderError("provider.baseUrl is required for openai_compatible")
    parsed = urllib.parse.urlparse(str(base_url).strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ModelProviderError(
            "provider.baseUrl must be an http(s) URL without embedded credentials, query, or fragment"
        )
    _validate_provider_host(
        parsed.hostname,
        resolve=False,
        allow_private=bool(allow_private_network),
    )
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ModelProviderError("unencrypted HTTP providers are allowed only on loopback")
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401, ANN001
        return None


def _open_provider_request(request: urllib.request.Request, timeout: int):
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def _provider_max_retries(provider: Mapping[str, Any]) -> int:
    try:
        retries = int(provider.get("maxRetries", DEFAULT_PROVIDER_MAX_RETRIES))
    except (TypeError, ValueError):
        retries = DEFAULT_PROVIDER_MAX_RETRIES
    return max(0, min(retries, MAX_PROVIDER_MAX_RETRIES))


def _transient_provider_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.URLError):
        error = error.reason
    return isinstance(error, (ConnectionError, TimeoutError))


def chat_completion(
    provider: Mapping[str, Any],
    agent: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Call one OpenAI-compatible model and return its assistant message."""

    allow_private = bool(provider.get("allowPrivateNetwork", False))
    url = endpoint(str(provider.get("baseUrl") or ""), allow_private_network=allow_private)
    model = str(agent.get("model") or provider.get("model") or "").strip()
    if not model:
        raise ModelProviderError("provider.model (or agent.model) is required")
    credential_env = str(provider.get("credentialEnv") or "").strip()
    parsed = urllib.parse.urlparse(url)
    _validate_provider_host(parsed.hostname or "", resolve=True, allow_private=allow_private)
    api_key = os.environ.get(credential_env, "") if credential_env else ""
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise ModelProviderError("remote provider requires provider.credentialEnv")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": float(agent.get("temperature", 0.2)),
        "max_tokens": int(agent.get("maxTokens", 512)),
    }
    hostname = (parsed.hostname or "").lower()
    if model.lower().startswith("minimax-") or hostname.endswith("minimaxi.com") or hostname.endswith("minimax.io"):
        # MiniMax's compatible API otherwise embeds <think> in content.  The
        # split form keeps reasoning for tool-call continuity without exposing
        # it in node status, reports, or downstream business data.
        payload["reasoning_split"] = True
    if tools:
        payload["tools"] = [dict(tool) for tool in tools]
        payload["tool_choice"] = "auto"
    headers = {"Content-Type": "application/json", "User-Agent": "Saddle-Agent/2.0"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout = int(provider.get("timeoutSeconds", 60))
    max_retries = _provider_max_retries(provider)
    for attempt in range(max_retries + 1):
        try:
            with _open_provider_request(request, timeout=timeout) as response:
                body = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
                if len(body) > MAX_PROVIDER_RESPONSE_BYTES:
                    raise ModelProviderError("provider response exceeded the 8 MiB safety limit")
            break
        except urllib.error.HTTPError as exc:
            if 300 <= int(exc.code) < 400:
                raise ModelProviderError("provider redirect was blocked") from exc
            raise ModelProviderError("provider HTTP %d" % exc.code) from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            if attempt < max_retries and _transient_provider_error(exc):
                time.sleep(PROVIDER_RETRY_DELAY_SECONDS * (attempt + 1))
                continue
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            raise ModelProviderError("provider request failed: %s" % reason) from exc
    try:
        result = json.loads(body.decode("utf-8"))
        choice = result["choices"][0]
        message = choice["message"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ModelProviderError("provider response has no valid choices[0].message") from exc
    if not isinstance(message, Mapping):
        raise ModelProviderError("provider response assistant message is invalid")
    normalized = _separate_inline_reasoning(message)
    finish_reason = str(choice.get("finish_reason") or "").strip().lower()
    if finish_reason:
        # Response metadata is consumed by the Agent runtime but intentionally
        # excluded from the assistant message sent back to the provider.
        normalized["_finish_reason"] = finish_reason
    return normalized


__all__ = [
    "DEFAULT_PROVIDER_MAX_RETRIES",
    "MAX_PROVIDER_RESPONSE_BYTES",
    "ModelProviderError",
    "chat_completion",
    "endpoint",
]
