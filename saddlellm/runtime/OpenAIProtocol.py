"""Shared text-only OpenAI API contract and sanitized error responses."""

from __future__ import annotations

import logging

API_CAPABILITIES = {
    "protocol": "openai-compatible-text-subset",
    "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions"],
    "streaming": True, "stream_usage": True, "choices": 1,
    "max_completion_tokens": True, "text_content_parts": True,
    "tools": False, "multimodal": False, "responses_api": False,
    "unsupported_parameters": "rejected",
}


def install_openai_error_handlers(app):
    """Keep HTTP status/headers but never echo validation input or tracebacks."""
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException

    logger = logging.getLogger(__name__)

    def error(message, status, *, param=None, headers=None):
        kind = ("authentication_error" if status == 401 else
                "permission_error" if status == 403 else
                "rate_limit_error" if status == 429 else
                "server_error" if status >= 500 else "invalid_request_error")
        return JSONResponse({"error": {"message": message, "type": kind,
                                       "param": param, "code": str(status)}},
                            status_code=status, headers=headers)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        # Existing routes only supply controlled public HTTPException messages.
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return error(message, exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        problems = exc.errors()
        first = problems[0] if problems else {}
        location = first.get("loc", ())
        param = ".".join(str(item) for item in location if item != "body")[:160] or None
        # Do not copy error.input, ctx, or exception text: they can contain prompts/secrets.
        message = ("Unsupported request parameter." if first.get("type") == "extra_forbidden"
                   else "Invalid request parameter; check the supported text API schema.")
        return error(message, 422, param=param)

    @app.exception_handler(Exception)
    async def server_error(request, exc):
        logger.exception("Inference API request failed")
        return error("Model generation failed.", 500)
