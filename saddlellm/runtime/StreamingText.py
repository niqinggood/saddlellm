"""Bounded text look-behind for stop strings that span streaming chunks."""

from __future__ import annotations


class StreamingTextBuffer:
    def __init__(self, stop=None):
        stops = [stop] if isinstance(stop, str) else list(stop or [])
        self.stops = tuple(item for item in stops if item)
        self.hold_back = max((len(item) for item in self.stops), default=1) - 1
        self.pending = ""
        self.stopped = False
        self.closed = False

    def push(self, piece: str) -> str:
        if self.stopped or self.closed:
            return ""
        text = self.pending + piece
        positions = [
            position for item in self.stops if (position := text.find(item)) >= 0
        ]
        if positions:
            self.stopped = True
            self.pending = ""
            return text[: min(positions)]
        end = max(0, len(text) - self.hold_back)
        self.pending = text[end:]
        return text[:end]

    def finish(self) -> str:
        self.closed = True
        text = "" if self.stopped else self.pending
        self.pending = ""
        return text
