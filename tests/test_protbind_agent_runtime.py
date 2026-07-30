from __future__ import annotations

import json

import pytest

from protbind_agent.agent_runtime import (
    ProtBindAgentPendingResult,
    ProtBindAgentRuntime,
    require_loopback_hipfire_url,
)
from protbind_agent.agent_tools import ProtBindAgentTools
from protbind_agent.mcp_server import ProtBindMCPService
from radeon_agent.backends import MockBackend
from radeon_agent.models import ChatResponse, ToolCall


def test_builtin_agent_accepts_only_exact_loopback_hipfire_url() -> None:
    require_loopback_hipfire_url("http://127.0.0.1:11435/v1")
    require_loopback_hipfire_url("http://[::1]:11435/v1/")

    for value in (
        "https://127.0.0.1:11435/v1",
        "http://localhost.example:11435/v1",
        "http://127.0.0.1:11435/other",
        "http://127.0.0.1:11435/v1?token=secret",
    ):
        with pytest.raises(PermissionError, match="exact loopback"):
            require_loopback_hipfire_url(value)


def test_builtin_tools_are_a_fixed_in_process_allowlist(tmp_path) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    tools = ProtBindAgentTools(service)

    names = {spec.name for spec in tools.specs()}

    assert names == {
        "doctor",
        "case_status",
        "case_report",
        "case_dossier",
        "case_pose_view",
        "artifact_metadata",
        "knowledge_model_status",
        "memory_search",
        "fetch_public_data",
        "case_create",
        "case_advance",
        "case_attach_support",
        "library_plan_import",
        "library_apply_import",
        "knowledge_import",
        "knowledge_search",
        "library_rag_sync",
        "memory_write",
    }
    assert not names & {"bash", "shell", "mcp_call", "read_file", "write_file"}


def test_permissioned_tool_requires_fresh_host_confirmation(tmp_path) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    decisions = iter((False, True))
    tools = ProtBindAgentTools(service, confirmation=lambda _preview: next(decisions))
    registry = tools.registry()

    denied = registry.execute(
        "case_create",
        {"case_path": "missing.json", "index_path": "missing.sqlite"},
    )
    allowed = registry.execute(
        "case_create",
        {"case_path": "missing.json", "index_path": "missing.sqlite"},
    )

    assert denied.ok is False
    assert "declined" in (denied.error or "")
    assert allowed.ok is False
    assert "does not exist" in (allowed.error or "")
    assert [event.confirmed for event in tools.audit_events] == [False, True]
    assert len(tools.shadow_plans) == 2
    assert tools.shadow_plans[0].plan_id == tools.shadow_plans[1].plan_id
    assert tools.audit_events[0].shadow_plan_id == tools.shadow_plans[0].plan_id
    assert tools.shadow_plans[0].snapshot.revalidation_required is True


def test_agent_calls_bounded_tool_and_returns_timeline(tmp_path) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(ToolCall(id="doctor-1", name="doctor", arguments={}),),
            ),
            ChatResponse(content="本地能力已检查。"),
        ]
    )
    runtime = ProtBindAgentRuntime(
        service,
        backend,
        model="fixture",
        confirmation=lambda _preview: False,
    )

    result = runtime.run("请检查本地能力。")

    assert result.answer == "本地能力已检查。"
    assert result.tool_calls == 1
    assert result.tool_results == ({"name": "doctor", "ok": True},)
    assert result.validated_artifact_citations == ()
    assert result.citation_warnings == ()
    assert result.tool_timeline[0]["tool"] == "doctor"
    assert result.tool_timeline[0]["ok"] is True
    assert result.exposed_tool_names[0] == ("doctor", "knowledge_model_status")
    assert result.tool_routes[0]["mode"] == "routed"
    tool_message = backend.requests[1].messages[-1]
    value = json.loads(tool_message.content or "{}")
    assert value["value"]["offline_default"] is True


def test_agent_appends_only_a_citation_returned_by_a_tool(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    citation = f"sha256:{'a' * 64}"
    monkeypatch.setattr(service, "doctor", lambda: {"artifact_id": citation})
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(ToolCall(id="doctor-1", name="doctor", arguments={}),),
            ),
            ChatResponse(content=f"检查完成。sha256:{'b' * 64}"),
        ]
    )
    runtime = ProtBindAgentRuntime(
        service,
        backend,
        model="fixture",
        confirmation=lambda _preview: False,
    )

    result = runtime.run("请检查并引用工具证据。")

    assert result.answer.endswith(citation)
    assert f"sha256:{'b' * 64}" not in result.answer
    assert result.validated_artifact_citations == (citation,)
    assert result.citation_warnings


def test_formal_workload_routes_three_tools_and_keeps_cited_compact_results(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    gate_digest = "a" * 64
    knowledge_digest = "b" * 64
    memory_digest = "c" * 64
    monkeypatch.setattr(
        service,
        "case_status",
        lambda run_id: {
            "gate": {
                "phase": "PREFLIGHT",
                "run_id": run_id,
                "stage": None,
                "decision": "COMPLETE",
                "manifest_sha256": "d" * 64,
                "policy_sha256": "e" * 64,
                "checks": [],
                "required_actions": [],
                "continuation_token": None,
                "automatic_retry": False,
            },
            "gate_receipt": {
                "sha256": gate_digest,
                "media_type": "application/json",
                "size_bytes": 1,
                "producer": "test",
            },
            "run": {"state": "REPORTED", "next_stage": None},
        },
    )
    monkeypatch.setattr(
        service,
        "knowledge_search",
        lambda **_arguments: {
            "query": "boundary",
            "scope": "evidence",
            "answer_mode": "retrieval-only",
            "evidence": [
                {
                    "id": "doc:1",
                    "text": "scientific boundary",
                    "metadata": {
                        "artifact_id": f"sha256:{knowledge_digest}",
                        "page": 2,
                    },
                }
            ],
        },
    )

    def memory_write(_self, run_id, preference=None):
        return {
            "written": True,
            "artifact": {
                "sha256": memory_digest,
                "media_type": "application/json",
                "size_bytes": 1,
                "producer": "test",
            },
            "run_id": run_id,
            "preference": preference,
            "scientific_state_changed": False,
        }

    monkeypatch.setattr(
        "protbind_agent.agent_tools.ExperienceStore.write",
        memory_write,
    )
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="status",
                        name="case_status",
                        arguments={"run_id": "run-1"},
                    ),
                    ToolCall(
                        id="knowledge",
                        name="knowledge_search",
                        arguments={"query": "boundary", "scope": "evidence"},
                    ),
                    ToolCall(
                        id="memory",
                        name="memory_write",
                        arguments={"run_id": "run-1", "preference": "offline"},
                    ),
                ),
            ),
            ChatResponse(content=f"完成。sha256:{memory_digest}"),
        ]
    )
    runtime = ProtBindAgentRuntime(
        service,
        backend,
        model="fixture",
        confirmation=lambda _preview: True,
    )

    result = runtime.run(
        "依次调用 case_status、knowledge_search 和 memory_write，然后总结。"
    )

    assert result.exposed_tool_names[0] == (
        "case_status",
        "knowledge_search",
        "memory_write",
    )
    assert result.tool_calls == 3
    assert all(item["ok"] for item in result.tool_results)
    assert result.validated_artifact_citations == (f"sha256:{memory_digest}",)
    assert len(result.shadow_plans) == 2
    status_message = json.loads(backend.requests[1].messages[-3].content or "{}")
    assert status_message["value"]["gate_receipt"]["artifact_id"] == (
        f"sha256:{gate_digest}"
    )


def test_runtime_returns_waiting_approval_then_resumes_without_an_extra_model_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    monkeypatch.setattr(
        service,
        "case_create",
        lambda **_arguments: {"run_id": "run-1", "state": "CREATED"},
    )
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="create-1",
                        name="case_create",
                        arguments={
                            "case_path": "case.json",
                            "index_path": "index.sqlite",
                        },
                    ),
                ),
            ),
            ChatResponse(content="案例已创建。"),
        ]
    )
    runtime = ProtBindAgentRuntime(service, backend, model="fixture")

    pending = runtime.start("创建案例")

    assert isinstance(pending, ProtBindAgentPendingResult)
    assert pending.status == "WAITING_APPROVAL"
    assert pending.approval["shadow_plan"]["status"] == "WAITING_APPROVAL"
    assert len(backend.requests) == 1
    assert pending.tool_timeline[-1]["status"] == "WAITING_APPROVAL"

    result = runtime.resume(
        pending.session_id,
        pending.approval_id,
        approved=True,
    )

    assert not isinstance(result, ProtBindAgentPendingResult)
    assert result.answer == "案例已创建。"
    assert len(backend.requests) == 2
    assert [event["status"] for event in result.tool_timeline] == [
        "WAITING_APPROVAL",
        "EXECUTED",
    ]
    assert result.approvals[-1]["status"] == "EXECUTED"
    assert len(result.shadow_plans) == 1


def test_runtime_decline_resumes_with_a_real_permission_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    called = False

    def create(**_arguments):
        nonlocal called
        called = True

    monkeypatch.setattr(service, "case_create", create)
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="create-1",
                        name="case_create",
                        arguments={
                            "case_path": "case.json",
                            "index_path": "index.sqlite",
                        },
                    ),
                ),
            ),
            ChatResponse(content="用户拒绝，未创建案例。"),
        ]
    )
    runtime = ProtBindAgentRuntime(service, backend, model="fixture")
    pending = runtime.start("创建案例")
    assert isinstance(pending, ProtBindAgentPendingResult)

    result = runtime.resume(
        pending.session_id,
        pending.approval_id,
        approved=False,
    )

    assert not isinstance(result, ProtBindAgentPendingResult)
    assert called is False
    assert result.approvals[-1]["status"] == "DECLINED"
    tool_message = json.loads(backend.requests[1].messages[-1].content or "{}")
    assert tool_message["ok"] is False
    assert "declined" in tool_message["error"]


def test_parallel_permissioned_calls_pause_once_per_fresh_approval(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    created: list[str] = []

    def create(case_path: str, **_arguments):
        created.append(case_path)
        return {"case_path": case_path}

    monkeypatch.setattr(service, "case_create", create)
    backend = MockBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="create-1",
                        name="case_create",
                        arguments={
                            "case_path": "one.json",
                            "index_path": "index.sqlite",
                        },
                    ),
                    ToolCall(
                        id="create-2",
                        name="case_create",
                        arguments={
                            "case_path": "two.json",
                            "index_path": "index.sqlite",
                        },
                    ),
                ),
            ),
            ChatResponse(content="两个已批准调用均完成。"),
        ]
    )
    runtime = ProtBindAgentRuntime(service, backend, model="fixture")
    first = runtime.start("依次调用两次 case_create")
    assert isinstance(first, ProtBindAgentPendingResult)

    second = runtime.resume(first.session_id, first.approval_id, approved=True)

    assert isinstance(second, ProtBindAgentPendingResult)
    assert second.approval_id != first.approval_id
    assert created == ["one.json"]
    assert len(backend.requests) == 1

    final = runtime.resume(second.session_id, second.approval_id, approved=True)

    assert not isinstance(final, ProtBindAgentPendingResult)
    assert created == ["one.json", "two.json"]
    assert [request["status"] for request in final.approvals] == [
        "EXECUTED",
        "EXECUTED",
    ]
    assert len(final.shadow_plans) == 2
