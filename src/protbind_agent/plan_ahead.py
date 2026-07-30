"""Pure, side-effect-free ShadowPlan compiler.

The compiler receives an already constructed host action preview. It performs no
filesystem, network, model, database, or workflow access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

SHADOW_PLAN_SCHEMA_VERSION = "1.0"


class ActionPreviewLike(Protocol):
    tool: str
    arguments_sha256: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    network: str
    scientific_state_change: bool
    expected_next_state: str
    recovery: str


@dataclass(frozen=True, slots=True)
class ShadowPlanSnapshot:
    manifest_sha256: str | None
    policy_sha256: str | None
    revalidation_required: bool = True


@dataclass(frozen=True, slots=True)
class ShadowPlan:
    schema_version: str
    kind: str
    status: str
    plan_id: str
    tool: str
    arguments_sha256: str
    snapshot: ShadowPlanSnapshot
    safe_idle_tasks: tuple[str, ...]
    forbidden_before_approval: tuple[str, ...]
    branches: dict[str, str]
    scientific_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def digest_arguments(arguments: dict[str, Any]) -> str:
    """Hash tool arguments without copying their raw values into a ShadowPlan."""

    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_shadow_plan(
    preview: ActionPreviewLike,
    *,
    manifest_sha256: str | None = None,
    policy_sha256: str | None = None,
) -> ShadowPlan:
    """Compile conditional idle work; never execute or authorize it."""

    safe_tasks = [
        "render-action-preview",
        "compile-conditional-branches",
        "prepare-cancellable-report-skeleton",
    ]
    if preview.network != "none":
        safe_tasks.append("render-exact-network-disclosure")
    if preview.writes:
        safe_tasks.append("render-declared-write-set")
    if preview.scientific_state_change:
        safe_tasks.append("compile-one-stage-postflight-checklist")

    snapshot = ShadowPlanSnapshot(
        manifest_sha256=manifest_sha256,
        policy_sha256=policy_sha256,
    )
    payload = {
        "schema_version": SHADOW_PLAN_SCHEMA_VERSION,
        "kind": "protbind.shadow-plan",
        "status": "WAITING_APPROVAL",
        "tool": preview.tool,
        "arguments_sha256": preview.arguments_sha256,
        "snapshot": asdict(snapshot),
        "safe_idle_tasks": safe_tasks,
        "forbidden_before_approval": [
            "private-data-read",
            "network-access",
            "scientific-state-write",
            "continuation-token-use",
            "memory-write",
        ],
        "branches": {
            "approved": (
                "cancel idle work, revalidate arguments and snapshot/policy bindings, "
                "then execute only the confirmed tool"
            ),
            "declined": (
                "discard the ephemeral plan without tool, state, network, or memory work"
            ),
            "stale": "discard the plan and request a fresh gate and confirmation",
            "tool_error": "record an explicit control failure and stop",
        },
        "scientific_semantics": (
            "Conditional preparation only; this plan is not a tool result, "
            "scientific evidence, authorization, or continuation token."
        ),
    }
    return ShadowPlan(
        schema_version=SHADOW_PLAN_SCHEMA_VERSION,
        kind="protbind.shadow-plan",
        status="WAITING_APPROVAL",
        plan_id=_plan_id(payload),
        tool=preview.tool,
        arguments_sha256=preview.arguments_sha256,
        snapshot=snapshot,
        safe_idle_tasks=tuple(safe_tasks),
        forbidden_before_approval=tuple(payload["forbidden_before_approval"]),
        branches=dict(payload["branches"]),
        scientific_semantics=str(payload["scientific_semantics"]),
    )


def shadow_plan_is_current(
    plan: ShadowPlan,
    *,
    arguments_sha256: str,
    manifest_sha256: str | None = None,
    policy_sha256: str | None = None,
) -> bool:
    """Check immutable bindings before a caller considers adopting a plan."""

    if plan.arguments_sha256 != arguments_sha256:
        return False
    if (
        plan.snapshot.manifest_sha256 is not None
        and plan.snapshot.manifest_sha256 != manifest_sha256
    ):
        return False
    return not (
        plan.snapshot.policy_sha256 is not None
        and plan.snapshot.policy_sha256 != policy_sha256
    )
