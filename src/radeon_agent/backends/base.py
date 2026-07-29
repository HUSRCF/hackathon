"""LLM backend contract kept independent from the Agent implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import ChatRequest, ChatResponse, StreamTiming


class BackendError(RuntimeError):
    pass


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def health(self) -> dict:
        """Return a small, non-secret health snapshot."""

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Run one non-streaming completion."""

    def stream_complete(self, request: ChatRequest) -> tuple[ChatResponse, StreamTiming]:
        """Run one streaming completion and report client-side timing."""

    def list_models(self) -> tuple[str, ...]:
        """List models advertised by the endpoint, if supported."""
