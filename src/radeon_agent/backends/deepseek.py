"""DeepSeek V4 functional bootstrap through its OpenAI-compatible endpoint."""

from __future__ import annotations

from dataclasses import replace

from ..models import ChatRequest, ChatResponse, StreamTiming
from .openai_compatible import OpenAICompatibleBackend


class DeepSeekBackend(OpenAICompatibleBackend):
    """Use V4 in non-thinking mode so multi-turn tool calls remain portable."""

    name = "deepseek"

    def __init__(
        self,
        base_url: str = "https://api.deepseek.com",
        *,
        api_key: str,
        timeout_seconds: float = 300.0,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek requires DEEPSEEK_API_KEY or RADEON_AGENT_API_KEY")
        super().__init__(
            base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            user_agent="radeon-agent-lab/0.1 (deepseek-bootstrap)",
        )

    @staticmethod
    def _non_thinking(request: ChatRequest) -> ChatRequest:
        extra = {**request.extra, "thinking": {"type": "disabled"}}
        return replace(request, extra=extra)

    def complete(self, request: ChatRequest) -> ChatResponse:
        return super().complete(self._non_thinking(request))

    def stream_complete(self, request: ChatRequest) -> tuple[ChatResponse, StreamTiming]:
        return super().stream_complete(self._non_thinking(request))
