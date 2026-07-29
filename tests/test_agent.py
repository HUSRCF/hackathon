from __future__ import annotations

import json

import pytest

from radeon_agent.agent import Agent, AgentLimitError
from radeon_agent.backends import MockBackend
from radeon_agent.models import ChatResponse, ToolCall, Usage
from radeon_agent.tools import SideEffect, ToolRegistry, ToolSpec


def _add_tool() -> ToolSpec:
    return ToolSpec(
        name="add",
        description="Add two integers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        handler=lambda arguments: arguments["a"] + arguments["b"],
    )


def test_agent_runs_tool_then_returns_answer() -> None:
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3}),),
                usage=Usage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
            ),
            ChatResponse(
                content="结果是 5。",
                usage=Usage(prompt_tokens=15, completion_tokens=5, total_tokens=20),
            ),
        ]
    )
    agent = Agent(backend, model="test", tools=ToolRegistry([_add_tool()]))

    result = agent.run("2+3 等于多少？")

    assert result.answer == "结果是 5。"
    assert result.model_calls == 2
    assert result.tool_calls == 1
    assert result.usage.total_tokens == 34
    tool_message = backend.requests[1].messages[-1]
    assert tool_message.role == "tool"
    assert json.loads(tool_message.content or "{}")["value"] == 5


def test_agent_enforces_max_steps() -> None:
    call = ChatResponse(
        content="",
        tool_calls=(ToolCall(id="again", name="add", arguments={"a": 1, "b": 1}),),
    )
    backend = MockBackend([call, call])
    agent = Agent(
        backend,
        model="test",
        tools=ToolRegistry([_add_tool()]),
        max_steps=2,
    )

    with pytest.raises(AgentLimitError, match="max_steps=2"):
        agent.run("keep calling")


def test_side_effect_policy_returns_tool_error_to_model() -> None:
    tool = ToolSpec(
        name="write_note",
        description="Write a note.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda arguments: arguments["text"],
        side_effect=SideEffect.LOCAL_WRITE,
    )
    registry = ToolRegistry([tool], max_side_effect=SideEffect.NONE)

    result = registry.execute("write_note", {"text": "secret"})

    assert not result.ok
    assert "policy allows NONE" in (result.error or "")

