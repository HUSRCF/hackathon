"""Deterministic backend for CPU-only tests and demos."""

from __future__ import annotations

import time
from collections.abc import Iterable

from ..models import ChatRequest, ChatResponse, StreamTiming
from .base import BackendError


class MockBackend:
    name = "mock"

    def __init__(self, responses: Iterable[ChatResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        last = request.messages[-1].content or ""
        return ChatResponse(content=f"mock: {last}", finish_reason="stop", model=request.model)

    def stream_complete(self, request: ChatRequest) -> tuple[ChatResponse, StreamTiming]:
        start = time.perf_counter()
        response = self.complete(request)
        end = time.perf_counter()
        return response, StreamTiming(end - start, end - start)

    def list_models(self) -> tuple[str, ...]:
        return ("mock-model",)

    def health(self) -> dict:
        return {"status": "ok", "model": "mock-model"}

    def assert_exhausted(self) -> None:
        if self._responses:
            raise BackendError(f"{len(self._responses)} mock responses were not consumed")
