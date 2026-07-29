"""Transport-neutral request and response models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(
                    self.arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        return payload


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int = 512
    tools: tuple[dict[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_openai(self, *, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai() for message in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if self.tools:
            payload["tools"] = list(self.tools)
            payload["tool_choice"] = "auto"
        payload.update(self.extra)
        return payload


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class StreamTiming:
    total_seconds: float
    time_to_first_token_seconds: float | None

