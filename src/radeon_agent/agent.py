"""Bounded Agent loop with explicit tools and no arbitrary code execution."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .backends.base import LLMBackend
from .models import ChatRequest, Message, StreamTiming, Usage
from .tools import ToolRegistry, ToolResult

ToolSchemaSelector = Callable[
    [tuple[Message, ...], tuple[str, ...]],
    Iterable[str],
]

DEFAULT_SYSTEM_PROMPT = """You are a private local assistant running on an AMD Radeon GPU.
Use only the provided tools. Never invent a tool result. Ask before irreversible or external
actions. Store a memory only when it is reusable and appropriate to retain. Answer concisely
and make uncertainty explicit."""


class AgentLimitError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        tool_call_trace: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        super().__init__(message)
        self.tool_call_trace = tool_call_trace


@dataclass(frozen=True, slots=True)
class AgentResult:
    answer: str
    messages: tuple[Message, ...]
    model_calls: int
    tool_calls: int
    elapsed_seconds: float
    usage: Usage
    model_timings: tuple[StreamTiming, ...] = ()
    model_usages: tuple[Usage, ...] = ()
    tool_results: tuple[ToolExecution, ...] = ()
    exposed_tool_names: tuple[tuple[str, ...], ...] = ()
    exposed_tool_schema_bytes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolExecution:
    name: str
    ok: bool


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
        stream: bool = False,
        request_extra: dict[str, Any] | None = None,
        tool_schema_selector: ToolSchemaSelector | None = None,
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
        self.stream = stream
        self.request_extra = dict(request_extra or {})
        self.tool_schema_selector = tool_schema_selector

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
        model_timings: list[StreamTiming] = []
        model_usages: list[Usage] = []
        tool_call_trace: list[tuple[str, ...]] = []
        tool_results: list[ToolExecution] = []
        exposed_tool_names: list[tuple[str, ...]] = []
        exposed_tool_schema_bytes: list[int] = []

        for model_calls in range(1, self.max_steps + 1):
            if time.monotonic() - start > self.timeout_seconds:
                raise AgentLimitError(
                    "agent timeout exceeded",
                    tool_call_trace=tuple(tool_call_trace),
                )
            available_tools = self.tools.names()
            requested_tools = (
                tuple(
                    dict.fromkeys(
                        self.tool_schema_selector(tuple(messages), available_tools)
                    )
                )
                if self.tool_schema_selector is not None
                else available_tools
            )
            unknown_tools = sorted(set(requested_tools) - set(available_tools))
            if unknown_tools:
                raise ValueError(
                    "tool schema selector returned unknown tools: "
                    + ", ".join(unknown_tools)
                )
            selected_tools = tuple(
                name for name in available_tools if name in requested_tools
            )
            schemas = self.tools.schemas(selected_tools)
            exposed_tool_names.append(selected_tools)
            exposed_tool_schema_bytes.append(
                len(
                    json.dumps(
                        schemas,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            )
            request = ChatRequest(
                model=self.model,
                messages=tuple(messages),
                temperature=0.0,
                max_tokens=self.max_tokens,
                tools=schemas,
                extra=self.request_extra,
            )
            if self.stream:
                response, timing = self.backend.stream_complete(request)
                model_timings.append(timing)
            else:
                response = self.backend.complete(request)
            model_usages.append(response.usage)
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
            tool_call_trace.append(tuple(call.name for call in response.tool_calls))
            if not response.tool_calls:
                return AgentResult(
                    answer=response.content,
                    messages=tuple(messages),
                    model_calls=model_calls,
                    tool_calls=tool_call_count,
                    elapsed_seconds=time.monotonic() - start,
                    usage=total_usage,
                    model_timings=tuple(model_timings),
                    model_usages=tuple(model_usages),
                    tool_results=tuple(tool_results),
                    exposed_tool_names=tuple(exposed_tool_names),
                    exposed_tool_schema_bytes=tuple(exposed_tool_schema_bytes),
                )

            for call in response.tool_calls:
                tool_call_count += 1
                if call.name not in selected_tools:
                    result = ToolResult(
                        name=call.name,
                        ok=False,
                        error=(
                            f"tool {call.name!r} was not exposed for this model turn"
                        ),
                    )
                else:
                    result = self.tools.execute(call.name, call.arguments)
                tool_results.append(ToolExecution(name=call.name, ok=result.ok))
                messages.append(
                    Message(
                        role="tool",
                        content=result.as_message_content(),
                        name=call.name,
                        tool_call_id=call.id,
                    )
                )

        rendered_trace = " > ".join(
            ",".join(names) if names else "<final>"
            for names in tool_call_trace
        )
        raise AgentLimitError(
            f"agent reached max_steps={self.max_steps} before producing a final "
            f"answer; tool-call trace: {rendered_trace}",
            tool_call_trace=tuple(tool_call_trace),
        )
