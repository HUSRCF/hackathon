"""Bounded Agent loop with explicit tools and no arbitrary code execution."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .backends.base import LLMBackend
from .models import ChatRequest, ChatResponse, Message, Usage
from .tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """You are a private local assistant running on an AMD Radeon GPU.
Use only the provided tools. Never invent a tool result. Ask before irreversible or external
actions. Store a memory only when it is reusable and appropriate to retain. Answer concisely
and make uncertainty explicit."""


class AgentLimitError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AgentResult:
    answer: str
    messages: tuple[Message, ...]
    model_calls: int
    tool_calls: int
    elapsed_seconds: float
    usage: Usage


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


class Agent:
    def __init__(
        self,
        backend: LLMBackend,
        *,
        model: str,
        tools: ToolRegistry | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 6,
        timeout_seconds: float = 300.0,
        max_tokens: int = 2048,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self.backend = backend
        self.model = model
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def run(self, user_input: str) -> AgentResult:
        if not user_input.strip():
            raise ValueError("user input cannot be empty")
        start = time.monotonic()
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input),
        ]
        tool_call_count = 0
        total_usage = Usage()

        for model_calls in range(1, self.max_steps + 1):
            if time.monotonic() - start > self.timeout_seconds:
                raise AgentLimitError("agent timeout exceeded")
            response: ChatResponse = self.backend.complete(
                ChatRequest(
                    model=self.model,
                    messages=tuple(messages),
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    tools=self.tools.schemas(),
                )
            )
            total_usage = Usage(
                prompt_tokens=_sum_optional(
                    total_usage.prompt_tokens, response.usage.prompt_tokens
                ),
                completion_tokens=_sum_optional(
                    total_usage.completion_tokens, response.usage.completion_tokens
                ),
                total_tokens=_sum_optional(total_usage.total_tokens, response.usage.total_tokens),
            )
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                return AgentResult(
                    answer=response.content,
                    messages=tuple(messages),
                    model_calls=model_calls,
                    tool_calls=tool_call_count,
                    elapsed_seconds=time.monotonic() - start,
                    usage=total_usage,
                )

            for call in response.tool_calls:
                tool_call_count += 1
                result = self.tools.execute(call.name, call.arguments)
                messages.append(
                    Message(
                        role="tool",
                        content=result.as_message_content(),
                        name=call.name,
                        tool_call_id=call.id,
                    )
                )

        raise AgentLimitError(
            f"agent reached max_steps={self.max_steps} before producing a final answer"
        )
