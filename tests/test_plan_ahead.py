from __future__ import annotations

from protbind_agent.agent_tools import ActionPreview
from protbind_agent.plan_ahead import (
    build_shadow_plan,
    digest_arguments,
    shadow_plan_is_current,
)


def _preview(arguments_sha256: str) -> ActionPreview:
    return ActionPreview(
        tool="case_advance",
        arguments_sha256=arguments_sha256,
        reads=("current manifest",),
        writes=("one stage record",),
        network="none",
        scientific_state_change=True,
        expected_next_state="one accepted stage",
        recovery="request a fresh gate",
    )


def test_shadow_plan_is_deterministic_and_contains_no_raw_arguments() -> None:
    arguments = {
        "run_id": "private-run-name",
        "continuation_token": "f" * 64,
    }
    digest = digest_arguments(arguments)

    first = build_shadow_plan(_preview(digest))
    second = build_shadow_plan(_preview(digest))

    assert first == second
    assert first.status == "WAITING_APPROVAL"
    assert first.protocol_revision == "2"
    assert first.plan_id == second.plan_id
    rendered = str(first.to_dict())
    assert "private-run-name" not in rendered
    assert "f" * 64 not in rendered
    assert "continuation-token-use" in first.forbidden_before_approval
    assert "compile-one-stage-postflight-checklist" in first.safe_idle_tasks


def test_shadow_plan_rejects_changed_arguments_or_snapshot() -> None:
    digest = digest_arguments({"run_id": "run-1"})
    plan = build_shadow_plan(
        _preview(digest),
        manifest_sha256="a" * 64,
        policy_sha256="b" * 64,
    )

    assert shadow_plan_is_current(
        plan,
        arguments_sha256=digest,
        manifest_sha256="a" * 64,
        policy_sha256="b" * 64,
    )
    assert not shadow_plan_is_current(
        plan,
        arguments_sha256=digest_arguments({"run_id": "run-2"}),
        manifest_sha256="a" * 64,
        policy_sha256="b" * 64,
    )
    assert not shadow_plan_is_current(
        plan,
        arguments_sha256=digest,
        manifest_sha256="c" * 64,
        policy_sha256="b" * 64,
    )
