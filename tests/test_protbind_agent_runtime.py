from __future__ import annotations

import json

import pytest

from protbind_agent.agent_runtime import (
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
