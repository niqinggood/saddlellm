"""Bound local HTTP admission, including queued and streaming requests.

This is backpressure, not continuous GPU batching. State belongs to one ASGI
event loop/process; model permits remain owned by the generation engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalHTTPAdmission:
    limit: int
    active: int = 0
    admitted: int = 0
    rejected: int = 0

    def snapshot(self):
        return {"owner": "local_http", "limit": self.limit,
                "active": self.active, "admitted": self.admitted, "rejected": self.rejected}


class LocalAdmissionMiddleware:
    """Pure ASGI: do not release a stream slot when only headers were sent."""

    def __init__(self, app, admission):
        self.app, self.admission = app, admission

    async def __call__(self, scope, receive, send):
        protected = (scope["type"] == "http" and scope.get("method") == "POST"
                     and scope.get("path") in {"/v1/completions", "/v1/chat/completions"})
        if not protected:
            return await self.app(scope, receive, send)
        state = self.admission
        # No await between check and reservation; one event loop owns state.
        if state.active >= state.limit:
            from starlette.responses import JSONResponse

            state.rejected += 1
            response = JSONResponse({"error": {"message": "Inference request queue is full; retry later.",
                                               "type": "server_error", "param": None,
                                               "code": "queue_full"}}, status_code=503,
                                    headers={"Retry-After": "1"})
            return await response(scope, receive, send)
        state.active += 1
        state.admitted += 1
        try:
            return await self.app(scope, receive, send)
        finally:
            state.active -= 1
