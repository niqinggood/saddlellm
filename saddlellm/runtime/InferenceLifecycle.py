"""Own asynchronous request tasks until their underlying work has stopped."""
from __future__ import annotations

import asyncio


async def drain_task(task):
    """Finish cleanup despite repeated task/AnyIO cancellation; consume errors.

    Only for cleanup: callers must preserve their original exception/result.
    Cancelling an asyncio wrapper does not terminate a running model thread.
    """
    import anyio

    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            task.result()
        except (Exception, asyncio.CancelledError):
            pass


async def run_until_disconnect(receive, operation, cancel_event):
    """Watch ASGI disconnect after the body is parsed; no timer/polling loop.

    Returns None for a disconnected client. The caller owns the response for
    that now-closed transport. Never use alongside another receive consumer
    (e.g. StreamingResponse); this helper is for non-streaming routes only.
    """
    async def watch():
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return

    watcher = asyncio.create_task(watch())
    work = asyncio.create_task(operation)
    try:
        done, _ = await asyncio.wait((watcher, work), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        # Surface receive failures rather than silently losing the watcher.
        await watcher
        cancel_event.set()
        work.cancel()
        await drain_task(work)
        return None
    except BaseException:
        cancel_event.set()
        work.cancel()
        await drain_task(work)
        raise
    finally:
        watcher.cancel()
        await drain_task(watcher)
