"""Shared chat-page assets and durable-history routes for local and vLLM serving."""
from __future__ import annotations

import asyncio
import ipaddress
from pathlib import Path
from typing import Literal

from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from .ChatHistory import ChatHistoryStore
from .WebUI import chat_page


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=200_000)


class HistoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[HistoryMessage] = Field(min_length=2, max_length=200)


def install_chat_web_routes(app, settings, model_name: str, authorize) -> None:
    """Install the same page/history behavior for either inference backend."""
    assets = Path(__file__).parent / "assets"

    async def history_access(request: Request) -> None:
        if settings.api_key:
            return
        def loopback(address: str) -> bool:
            if address.lower() == "localhost":
                return True
            try:
                return ipaddress.ip_address(address).is_loopback
            except ValueError:
                return False

        client_host = request.client.host if request.client else ""
        # Check both the socket peer and requested host. A local reverse proxy
        # must not expose unauthenticated history under a public hostname.
        if not loopback(client_host) or not loopback(request.url.hostname or ""):
            raise HTTPException(
                status_code=403,
                detail="Chat history requires a loopback connection or an API key.",
            )

    history_dependencies = [Depends(authorize), Depends(history_access)]

    @app.get("/", response_class=HTMLResponse)
    @app.get("/saddle", response_class=HTMLResponse)
    async def web_chat():
        return chat_page(
            settings.ui_title,
            default_max_tokens=settings.default_max_new_tokens,
            max_tokens=settings.max_new_tokens,
            history_enabled=settings.chat_history,
        )

    @app.get("/api/saddle/catalog")
    async def saddle_introduction(response: Response, lang: Literal["zh", "en"] = "zh"):
        from .SaddleGuide import saddle_catalog

        response.headers["Cache-Control"] = "no-cache"
        return saddle_catalog(lang)

    @app.get("/assets/{name}", include_in_schema=False)
    async def web_asset(name: str):
        media_type = {
            "chat.css": "text/css",
            "chat.js": "application/javascript",
            "saddle-intro.css": "text/css",
            "saddle-intro.js": "application/javascript",
            "marked.min.js": "application/javascript",
        }.get(name)
        if media_type is None:
            raise HTTPException(status_code=404, detail="Asset not found.")
        return FileResponse(assets / name, media_type=media_type,
                            headers={"Cache-Control": "no-cache"})

    @app.get("/assets/katex/{name}", include_in_schema=False)
    async def katex_asset(name: str):
        if name not in {"katex.min.css", "katex.min.js"}:
            raise HTTPException(status_code=404, detail="Asset not found.")
        media_type = "text/css" if name.endswith(".css") else "application/javascript"
        return FileResponse(assets / "katex" / name, media_type=media_type,
                            headers={"Cache-Control": "no-cache"})

    @app.get("/assets/katex/fonts/{name}", include_in_schema=False)
    async def katex_font(name: str):
        if (Path(name).name != name or "\\" in name
                or not name.startswith("KaTeX_") or not name.endswith(".woff2")):
            raise HTTPException(status_code=404, detail="Asset not found.")
        font = assets / "katex" / "fonts" / name
        if not font.is_file():
            raise HTTPException(status_code=404, detail="Asset not found.")
        return FileResponse(font, media_type="font/woff2",
                            headers={"Cache-Control": "public, max-age=86400"})

    if not settings.chat_history:
        return
    store = ChatHistoryStore(Path.cwd())

    @app.get("/api/chat/sessions", dependencies=history_dependencies)
    async def list_sessions(response: Response):
        response.headers["Cache-Control"] = "no-store"
        return {"sessions": await asyncio.to_thread(store.list_sessions, model_name)}

    @app.get("/api/chat/sessions/{session_id}", dependencies=history_dependencies)
    async def get_session(session_id: str, response: Response):
        response.headers["Cache-Control"] = "no-store"
        try:
            session = await asyncio.to_thread(store.get_session, session_id, model_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        return session

    @app.put("/api/chat/sessions/{session_id}", dependencies=history_dependencies)
    async def save_session(session_id: str, payload: HistoryWrite, response: Response):
        response.headers["Cache-Control"] = "no-store"
        try:
            return await asyncio.to_thread(
                store.save_session, session_id, model_name,
                [message.model_dump() for message in payload.messages],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/chat/sessions/{session_id}", dependencies=history_dependencies)
    async def delete_session(session_id: str):
        try:
            removed = await asyncio.to_thread(store.delete_session, session_id, model_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not removed:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
