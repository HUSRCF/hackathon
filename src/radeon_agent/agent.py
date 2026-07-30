"""Bounded Agent loop with explicit tools and no arbitrary code execution."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .backends.base import LLMBackend
from .models import ChatRequest, Message, StreamTiming, ToolCall, Usage
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


class AgentApprovalPendingError(RuntimeError):
    def __init__(self, pending: AgentPendingResult) -> None:
        super().__init__(
            f"agent session {pending.session_id!r} is waiting for approval "
            f"{pending.pending_tool.approval_id!r}"
        )
        self.pending = pending


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


@dataclass(frozen=True, slots=True)
class PendingToolExecution:
    approval_id: str
    name: str
    call_id: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentPendingResult:
    status: str
    session_id: str
    pending_tool: PendingToolExecution
    messages: tuple[Message, ...]
    model_calls: int
    tool_calls: int
    active_elapsed_seconds: float
    usage: Usage
    model_timings: tuple[StreamTiming, ...] = ()
    model_usages: tuple[Usage, ...] = ()
    tool_results: tuple[ToolExecution, ...] = ()
    exposed_tool_names: tuple[tuple[str, ...], ...] = ()
    exposed_tool_schema_bytes: tuple[int, ...] = ()


@dataclass(slots=True)
class _AgentRunState:
    session_id: str
    messages: list[Message]
    model_calls: int = 0
    tool_call_count: int = 0
    active_elapsed_seconds: float = 0.0
    total_usage: Usage = field(default_factory=Usage)
    model_timings: list[StreamTiming] = field(default_factory=list)
    model_usages: list[Usage] = field(default_factory=list)
    tool_call_trace: list[tuple[str, ...]] = field(default_factory=list)
    tool_results: list[ToolExecution] = field(default_factory=list)
    exposed_tool_names: list[tuple[str, ...]] = field(default_factory=list)
    exposed_tool_schema_bytes: list[int] = field(default_factory=list)
    current_tool_calls: tuple[ToolCall, ...] = ()
    current_tool_index: int = 0
    current_selected_tools: tuple[str, ...] = ()
    counted_call_ids: set[str] = field(default_factory=set)
    pending_tool: PendingToolExecution | None = None


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
        self._paused_sessions: dict[str, _AgentRunState] = {}

    def run(self, user_input: str) -> AgentResult:
        """Run to completion for callers that do not use non-blocking approvals."""

        result = self.start(user_input)
        if isinstance(result, AgentPendingResult):
            raise AgentApprovalPendingError(result)
        return result

    def start(self, user_input: str) -> AgentResult | AgentPendingResult:
        if not user_input.strip():
            raise ValueError("user input cannot be empty")
        state = _AgentRunState(
            session_id=uuid4().hex,
            messages=[
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=user_input),
            ],
        )
        return self._drive(state)

    def resume(
        self,
        session_id: str,
        *,
        approval_id: str,
    ) -> AgentResult | AgentPendingResult:
        try:
            state = self._paused_sessions.pop(session_id)
        except KeyError as exc:
            raise KeyError(f"unknown or inactive agent session: {session_id}") from exc
        pending = state.pending_tool
        if pending is None or pending.approval_id != approval_id:
            self._paused_sessions[session_id] = state
            raise ValueError("approval_id does not match the paused tool invocation")
        state.pending_tool = None
        return self._drive(state)

    def pending_session_ids(self) -> tuple[str, ...]:
        return tuple(self._paused_sessions)

    def pending_approval_id(self, session_id: str) -> str:
        try:
            state = self._paused_sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown or inactive agent session: {session_id}") from exc
        if state.pending_tool is None:
            raise RuntimeError("agent session has no pending tool")
        return state.pending_tool.approval_id

    def _active_elapsed(self, state: _AgentRunState, segment_start: float) -> float:
        return state.active_elapsed_seconds + (time.monotonic() - segment_start)

    def _check_timeout(
        self,
        state: _AgentRunState,
        segment_start: float,
    ) -> None:
        if self._active_elapsed(state, segment_start) > self.timeout_seconds:
            raise AgentLimitError(
                "agent active-compute timeout exceeded",
                tool_call_trace=tuple(state.tool_call_trace),
            )

    def _pause(
        self,
        state: _AgentRunState,
        pending: PendingToolExecution,
        segment_start: float,
    ) -> AgentPendingResult:
        state.pending_tool = pending
        elapsed = self._active_elapsed(state, segment_start)
        self._paused_sessions[state.session_id] = state
        return AgentPendingResult(
            status="WAITING_APPROVAL",
            session_id=state.session_id,
            pending_tool=pending,
            messages=tuple(state.messages),
            model_calls=state.model_calls,
            tool_calls=state.tool_call_count,
            active_elapsed_seconds=elapsed,
            usage=state.total_usage,
            model_timings=tuple(state.model_timings),
            model_usages=tuple(state.model_usages),
            tool_results=tuple(state.tool_results),
            exposed_tool_names=tuple(state.exposed_tool_names),
            exposed_tool_schema_bytes=tuple(state.exposed_tool_schema_bytes),
        )

    def _complete(
        self,
        state: _AgentRunState,
        *,
        answer: str,
        segment_start: float,
    ) -> AgentResult:
        self._paused_sessions.pop(state.session_id, None)
        return AgentResult(
            answer=answer,
            messages=tuple(state.messages),
            model_calls=state.model_calls,
            tool_calls=state.tool_call_count,
            elapsed_seconds=self._active_elapsed(state, segment_start),
            usage=state.total_usage,
            model_timings=tuple(state.model_timings),
            model_usages=tuple(state.model_usages),
            tool_results=tuple(state.tool_results),
            exposed_tool_names=tuple(state.exposed_tool_names),
            exposed_tool_schema_bytes=tuple(state.exposed_tool_schema_bytes),
        )

    def _drive(
        self,
        state: _AgentRunState,
    ) -> AgentResult | AgentPendingResult:
        segment_start = time.monotonic()
        try:
            while True:
                self._check_timeout(state, segment_start)
                if state.current_tool_index < len(state.current_tool_calls):
                    call = state.current_tool_calls[state.current_tool_index]
                    if call.id not in state.counted_call_ids:
                        state.counted_call_ids.add(call.id)
                        state.tool_call_count += 1
                    if call.name not in state.current_selected_tools:
                        result = ToolResult(
                            name=call.name,
                            ok=False,
                            error=(
                                f"tool {call.name!r} was not exposed for this model turn"
                            ),
                        )
                    else:
                        result = self.tools.execute(call.name, call.arguments)
                    if result.pending is not None:
                        approval_id = result.pending.get("approval_id")
                        if not isinstance(approval_id, str) or not approval_id:
                            raise ValueError(
                                "pending tool result omitted a valid approval_id"
                            )
                        return self._pause(
                            state,
                            PendingToolExecution(
                                approval_id=approval_id,
                                name=call.name,
                                call_id=call.id,
                                payload=dict(result.pending),
                            ),
                            segment_start,
                        )
                    state.tool_results.append(
                        ToolExecution(name=call.name, ok=result.ok)
                    )
                    state.messages.append(
                        Message(
                            role="tool",
                            content=result.as_message_content(),
                            name=call.name,
                            tool_call_id=call.id,
                        )
                    )
                    state.current_tool_index += 1
                    continue

                state.current_tool_calls = ()
                state.current_tool_index = 0
                state.current_selected_tools = ()
                if state.model_calls >= self.max_steps:
                    rendered_trace = " > ".join(
                        ",".join(names) if names else "<final>"
                        for names in state.tool_call_trace
                    )
                    raise AgentLimitError(
                        f"agent reached max_steps={self.max_steps} before producing "
                        f"a final answer; tool-call trace: {rendered_trace}",
                        tool_call_trace=tuple(state.tool_call_trace),
                    )

                available_tools = self.tools.names()
                requested_tools = (
                    tuple(
                        dict.fromkeys(
                            self.tool_schema_selector(
                                tuple(state.messages),
                                available_tools,
                            )
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
                state.exposed_tool_names.append(selected_tools)
                state.exposed_tool_schema_bytes.append(
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
                    messages=tuple(state.messages),
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    tools=schemas,
                    extra=self.request_extra,
                )
                if self.stream:
                    response, timing = self.backend.stream_complete(request)
                    state.model_timings.append(timing)
                else:
                    response = self.backend.complete(request)
                state.model_calls += 1
                state.model_usages.append(response.usage)
                state.total_usage = Usage(
                    prompt_tokens=_sum_optional(
                        state.total_usage.prompt_tokens,
                        response.usage.prompt_tokens,
                    ),
                    completion_tokens=_sum_optional(
                        state.total_usage.completion_tokens,
                        response.usage.completion_tokens,
                    ),
                    total_tokens=_sum_optional(
                        state.total_usage.total_tokens,
                        response.usage.total_tokens,
                    ),
                )
                state.messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                state.tool_call_trace.append(
                    tuple(call.name for call in response.tool_calls)
                )
                if not response.tool_calls:
                    return self._complete(
                        state,
                        answer=response.content,
                        segment_start=segment_start,
                    )
                state.current_tool_calls = response.tool_calls
                state.current_tool_index = 0
                state.current_selected_tools = selected_tools
        finally:
            state.active_elapsed_seconds += time.monotonic() - segment_start
