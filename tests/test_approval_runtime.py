from __future__ import annotations

import json
import threading

import pytest

from protbind_agent.agent_tools import ActionPreview, ProtBindAgentTools
from protbind_agent.approval_runtime import ApprovalCoordinator
from protbind_agent.mcp_server import ProtBindMCPService
from radeon_agent.tools import ToolPendingError


def _preview(arguments_sha256: str = "a" * 64) -> ActionPreview:
    return ActionPreview(
        tool="case_advance",
        arguments_sha256=arguments_sha256,
        reads=("manifest",),
        writes=("one stage",),
        network="none",
        scientific_state_change=True,
        expected_next_state="one accepted stage",
        recovery="fresh status required",
        manifest_sha256="b" * 64,
        policy_sha256="c" * 64,
    )


def test_approval_is_fresh_redacted_and_consumed_once() -> None:
    coordinator = ApprovalCoordinator()

    with pytest.raises(ToolPendingError) as first_error:
        coordinator(_preview())
    first = first_error.value.pending
    serialized = json.dumps(first, ensure_ascii=False)

    assert first["status"] == "WAITING_APPROVAL"
    assert first["approval_id"] not in first["shadow_plan"]["plan_id"]
    assert "continuation_token" not in serialized

    with coordinator.resume_scope(first["approval_id"], approved=True):
        assert coordinator(_preview()) is True
        coordinator.complete_current(ok=True, error_type=None)

    assert coordinator.get(first["approval_id"])["status"] == "EXECUTED"
    with pytest.raises(ValueError, match="already consumed"):
        coordinator.decide(first["approval_id"], approved=True)

    with pytest.raises(ToolPendingError) as second_error:
        coordinator(_preview())
    second = second_error.value.pending
    assert second["approval_id"] != first["approval_id"]
    assert second["shadow_plan"]["plan_id"] == first["shadow_plan"]["plan_id"]


def test_idle_tasks_are_cancelled_on_decision() -> None:
    entered = threading.Event()

    def blocking_task(_task, _plan, cancel_event):
        entered.set()
        cancel_event.wait(timeout=2)
        return None

    coordinator = ApprovalCoordinator(idle_task_function=blocking_task)
    with pytest.raises(ToolPendingError) as error:
        coordinator(_preview())
    approval_id = error.value.pending["approval_id"]
    assert entered.wait(timeout=1)

    decided = coordinator.decide(approval_id, approved=False)

    assert decided["status"] == "DECLINED"
    assert decided["idle_tasks_running"] is False
    assert decided["cancellation_latency_seconds"] is not None
    assert decided["idle_tasks"]
    assert all(item["status"] == "CANCELLED" for item in decided["idle_tasks"])


def test_changed_binding_marks_approval_stale() -> None:
    coordinator = ApprovalCoordinator()
    with pytest.raises(ToolPendingError) as error:
        coordinator(_preview("a" * 64))
    approval_id = error.value.pending["approval_id"]

    with (
        coordinator.resume_scope(approval_id, approved=True),
        pytest.raises(Exception, match="stale"),
    ):
        coordinator(_preview("d" * 64))

    assert coordinator.get(approval_id)["status"] == "STALE"


def test_case_advance_rechecks_fresh_gate_after_approval(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    current_token = ["a" * 64]
    advance_called = False

    def status(run_id: str):
        return {
            "gate": {
                "run_id": run_id,
                "continuation_token": current_token[0],
                "manifest_sha256": "b" * 64,
                "policy_sha256": "c" * 64,
            }
        }

    def advance(_run_id: str, _continuation_token: str):
        nonlocal advance_called
        advance_called = True
        return {"acceptance": {"decision": "ACCEPTED"}}

    monkeypatch.setattr(service, "case_status", status)
    monkeypatch.setattr(service, "case_advance", advance)
    coordinator = ApprovalCoordinator()
    tools = ProtBindAgentTools(service, confirmation=coordinator)
    registry = tools.registry()
    initial = registry.execute("case_status", {"run_id": "run-1"})
    assert initial.ok

    pending = registry.execute(
        "case_advance",
        {"run_id": "run-1", "continuation_token": current_token[0]},
    )
    assert pending.pending is not None
    approval_id = pending.pending["approval_id"]
    current_token[0] = "d" * 64

    with coordinator.resume_scope(approval_id, approved=True):
        stale = registry.execute(
            "case_advance",
            {"run_id": "run-1", "continuation_token": "a" * 64},
        )

    assert stale.ok is False
    assert "fresh case_status" in (stale.error or "")
    assert advance_called is False
    assert coordinator.get(approval_id)["status"] == "STALE"
