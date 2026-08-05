from __future__ import annotations

import json

import pytest

from radeon_agent.agent import Agent, AgentLimitError, AgentPendingResult
from radeon_agent.backends import MockBackend
from radeon_agent.models import ChatResponse, ToolCall, Usage
from radeon_agent.tools import SideEffect, ToolPendingError, ToolRegistry, ToolSpec


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


def test_agent_counts_reused_tool_call_ids_on_distinct_model_turns() -> None:
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(id="reused", name="add", arguments={"a": 1, "b": 1}),
                ),
            ),
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(id="reused", name="add", arguments={"a": 2, "b": 2}),
                ),
            ),
            ChatResponse(content="完成。"),
        ]
    )
    agent = Agent(backend, model="test", tools=ToolRegistry([_add_tool()]))

    result = agent.run("依次计算两次。")

    assert result.tool_calls == 2
    assert len(result.tool_results) == 2


def test_agent_blocks_identical_failed_tool_retry_when_policy_disallows_it() -> None:
    executions = 0

    def fail(_arguments):
        nonlocal executions
        executions += 1
        raise ValueError("fixture failure")

    tool = ToolSpec(
        name="fail_once",
        description="Always fail for this test.",
        parameters={"type": "object", "properties": {}},
        handler=fail,
    )
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(ToolCall(id="first", name="fail_once", arguments={}),),
            ),
            ChatResponse(
                content="",
                tool_calls=(ToolCall(id="second", name="fail_once", arguments={}),),
            ),
            ChatResponse(content="已停止。"),
        ]
    )
    agent = Agent(
        backend,
        model="test",
        tools=ToolRegistry([tool]),
        allow_failed_tool_retries=False,
    )

    result = agent.run("失败后停止。")

    assert executions == 1
    assert result.tool_calls == 2
    assert [item.automatic_retry_blocked for item in result.tool_results] == [
        False,
        True,
    ]
    blocked_message = json.loads(backend.requests[2].messages[-1].content or "{}")
    assert "automatic retry blocked" in blocked_message["error"]


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


def test_agent_exposes_only_selected_tools_and_records_schema_size() -> None:
    backend = MockBackend([ChatResponse(content="done")])
    other = ToolSpec(
        name="other",
        description="An unrelated tool.",
        parameters={"type": "object", "properties": {}},
        handler=lambda _arguments: None,
    )
    agent = Agent(
        backend,
        model="test",
        tools=ToolRegistry([_add_tool(), other]),
        tool_schema_selector=lambda _messages, _available: ("add",),
    )

    result = agent.run("add")

    assert [schema["function"]["name"] for schema in backend.requests[0].tools] == [
        "add"
    ]
    assert result.exposed_tool_names == (("add",),)
    assert result.exposed_tool_schema_bytes[0] > 0


def test_agent_rejects_registered_tool_not_exposed_to_model() -> None:
    executed = False

    def execute_other(_arguments: dict) -> None:
        nonlocal executed
        executed = True

    other = ToolSpec(
        name="other",
        description="An unrelated tool.",
        parameters={"type": "object", "properties": {}},
        handler=execute_other,
    )
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(ToolCall(id="hidden", name="other", arguments={}),),
            ),
            ChatResponse(content="blocked"),
        ]
    )
    agent = Agent(
        backend,
        model="test",
        tools=ToolRegistry([_add_tool(), other]),
        tool_schema_selector=lambda _messages, _available: ("add",),
    )

    result = agent.run("try hidden")

    assert executed is False
    assert len(result.tool_results) == 1
    assert result.tool_results[0].name == "other"
    assert result.tool_results[0].ok is False
    message = json.loads(backend.requests[1].messages[-1].content or "{}")
    assert "not exposed" in message["error"]


def test_tool_result_view_keeps_full_host_value_but_compacts_model_message() -> None:
    tool = ToolSpec(
        name="lookup",
        description="Return a large result.",
        parameters={"type": "object", "properties": {}},
        handler=lambda _arguments: {"full": "value", "artifact_id": "sha256:abc"},
        result_view=lambda value: {"artifact_id": value["artifact_id"]},
    )

    result = ToolRegistry([tool]).execute("lookup", {})

    assert result.value == {"full": "value", "artifact_id": "sha256:abc"}
    assert json.loads(result.as_message_content())["value"] == {
        "artifact_id": "sha256:abc"
    }


def test_tool_result_view_failure_does_not_relabel_completed_side_effect() -> None:
    tool = ToolSpec(
        name="write",
        description="Complete a write before projection.",
        parameters={"type": "object", "properties": {}},
        handler=lambda _arguments: {"written": True},
        side_effect=SideEffect.LOCAL_WRITE,
        result_view=lambda _value: 1 / 0,
    )

    result = ToolRegistry(
        [tool], max_side_effect=SideEffect.LOCAL_WRITE
    ).execute("write", {})

    assert result.ok is True
    assert json.loads(result.as_message_content())["value"] == {"written": True}


def test_agent_pauses_without_sending_a_false_tool_failure_then_resumes() -> None:
    approved = False

    def approval_tool(_arguments: dict) -> str:
        if not approved:
            raise ToolPendingError(
                {
                    "approval_id": "approval-1",
                    "status": "WAITING_APPROVAL",
                }
            )
        return "executed"

    tool = ToolSpec(
        name="permissioned",
        description="A permissioned fixture.",
        parameters={"type": "object", "properties": {}},
        handler=approval_tool,
    )
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(id="permissioned-1", name="permissioned", arguments={}),
                ),
            ),
            ChatResponse(content="completed"),
        ]
    )
    agent = Agent(backend, model="test", tools=ToolRegistry([tool]))

    pending = agent.start("run the permissioned tool")

    assert isinstance(pending, AgentPendingResult)
    assert pending.status == "WAITING_APPROVAL"
    assert pending.tool_calls == 1
    assert len(backend.requests) == 1
    assert not any(message.role == "tool" for message in pending.messages)

    approved = True
    result = agent.resume(
        pending.session_id,
        approval_id=pending.pending_tool.approval_id,
    )

    assert not isinstance(result, AgentPendingResult)
    assert result.answer == "completed"
    assert result.tool_calls == 1
    assert len(backend.requests) == 2
    tool_message = backend.requests[1].messages[-1]
    assert json.loads(tool_message.content or "{}")["value"] == "executed"


def test_agent_rejects_wrong_resume_id_without_losing_paused_session() -> None:
    tool = ToolSpec(
        name="permissioned",
        description="A permissioned fixture.",
        parameters={"type": "object", "properties": {}},
        handler=lambda _arguments: (_ for _ in ()).throw(
            ToolPendingError({"approval_id": "approval-1"})
        ),
    )
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(id="permissioned-1", name="permissioned", arguments={}),
                ),
            )
        ]
    )
    agent = Agent(backend, model="test", tools=ToolRegistry([tool]))
    pending = agent.start("pause")
    assert isinstance(pending, AgentPendingResult)

    with pytest.raises(ValueError, match="does not match"):
        agent.resume(pending.session_id, approval_id="wrong")

    assert agent.pending_session_ids() == (pending.session_id,)


def test_approval_wait_does_not_consume_active_compute_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved = False
    clock = [10.0]
    monkeypatch.setattr("radeon_agent.agent.time.monotonic", lambda: clock[0])

    def approval_tool(_arguments: dict) -> str:
        if not approved:
            raise ToolPendingError({"approval_id": "approval-1"})
        return "executed"

    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(id="permissioned-1", name="permissioned", arguments={}),
                ),
            ),
            ChatResponse(content="completed"),
        ]
    )
    agent = Agent(
        backend,
        model="test",
        tools=ToolRegistry(
            [
                ToolSpec(
                    name="permissioned",
                    description="A permissioned fixture.",
                    parameters={"type": "object", "properties": {}},
                    handler=approval_tool,
                )
            ]
        ),
        timeout_seconds=1.0,
    )
    pending = agent.start("pause")
    assert isinstance(pending, AgentPendingResult)

    clock[0] += 3600.0
    approved = True
    result = agent.resume(
        pending.session_id,
        approval_id=pending.pending_tool.approval_id,
    )

    assert not isinstance(result, AgentPendingResult)
    assert result.answer == "completed"
    assert result.elapsed_seconds == 0.0
