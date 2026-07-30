"""Non-blocking approval control and cancellable deterministic idle work."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from radeon_agent.tools import ToolPendingError, ToolPermissionError

from .plan_ahead import ShadowPlan, build_shadow_plan, shadow_plan_is_current

APPROVAL_PROTOCOL_REVISION = "2"
APPROVAL_SCHEMA_VERSION = "1.0"


class ApprovalPreview(Protocol):
    tool: str
    arguments_sha256: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    network: str
    scientific_state_change: bool
    expected_next_state: str
    recovery: str
    manifest_sha256: str | None
    policy_sha256: str | None


IdleTaskFunction = Callable[[str, ShadowPlan, threading.Event], Any]


@dataclass(frozen=True, slots=True)
class IdleTaskReceipt:
    task: str
    status: str
    output_sha256: str | None
    output_bytes: int
    duration_seconds: float
    error_type: str | None = None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _default_idle_task(
    task: str,
    plan: ShadowPlan,
    cancel_event: threading.Event,
) -> Any:
    """Render only data already present in the redacted ShadowPlan."""

    if cancel_event.is_set():
        return None
    common = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "protocol_revision": APPROVAL_PROTOCOL_REVISION,
        "plan_id": plan.plan_id,
        "tool": plan.tool,
        "arguments_sha256": plan.arguments_sha256,
    }
    if task == "render-action-preview":
        return {
            **common,
            "status": "WAITING_APPROVAL",
            "scientific_semantics": plan.scientific_semantics,
        }
    if task == "compile-conditional-branches":
        return {**common, "branches": plan.branches}
    if task == "prepare-cancellable-report-skeleton":
        return {
            **common,
            "sections": [
                "approval-decision",
                "tool-control-receipt",
                "scientific-result-pending",
            ],
        }
    if task == "render-exact-network-disclosure":
        return {
            **common,
            "network": "declared in the host action preview; no request was sent",
        }
    if task == "render-declared-write-set":
        return {
            **common,
            "writes": "declared in the host action preview; no write was performed",
        }
    if task == "compile-one-stage-postflight-checklist":
        return {
            **common,
            "checks": [
                "fresh-continuation-token",
                "manifest-and-policy-binding",
                "one-main-stage-only",
                "accepted-postflight-required",
            ],
        }
    raise ValueError(f"unknown deterministic idle task: {task}")


class CancellableIdleTaskRunner:
    """Run CPU-only ShadowPlan projections without reading external state."""

    def __init__(
        self,
        plan: ShadowPlan,
        *,
        task_function: IdleTaskFunction = _default_idle_task,
    ) -> None:
        self.plan = plan
        self.task_function = task_function
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._receipts: list[IdleTaskReceipt] = []
        self._thread = threading.Thread(
            target=self._run,
            name=f"protbind-idle-{plan.plan_id[:12]}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _append(self, receipt: IdleTaskReceipt) -> None:
        with self._lock:
            self._receipts.append(receipt)

    def _run(self) -> None:
        for task in self.plan.safe_idle_tasks:
            started = time.monotonic()
            if self._cancel.is_set():
                self._append(
                    IdleTaskReceipt(
                        task=task,
                        status="CANCELLED",
                        output_sha256=None,
                        output_bytes=0,
                        duration_seconds=time.monotonic() - started,
                    )
                )
                continue
            try:
                output = self.task_function(task, self.plan, self._cancel)
                if self._cancel.is_set() or output is None:
                    receipt = IdleTaskReceipt(
                        task=task,
                        status="CANCELLED",
                        output_sha256=None,
                        output_bytes=0,
                        duration_seconds=time.monotonic() - started,
                    )
                else:
                    encoded = _canonical_bytes(output)
                    receipt = IdleTaskReceipt(
                        task=task,
                        status="COMPLETED",
                        output_sha256=hashlib.sha256(encoded).hexdigest(),
                        output_bytes=len(encoded),
                        duration_seconds=time.monotonic() - started,
                    )
            except Exception as exc:
                receipt = IdleTaskReceipt(
                    task=task,
                    status="FAILED",
                    output_sha256=None,
                    output_bytes=0,
                    duration_seconds=time.monotonic() - started,
                    error_type=type(exc).__name__,
                )
            self._append(receipt)

    def cancel(self, *, join_timeout_seconds: float = 1.0) -> float:
        started = time.monotonic()
        self._cancel.set()
        self._thread.join(timeout=join_timeout_seconds)
        return time.monotonic() - started

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def receipts(self) -> tuple[IdleTaskReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)


@dataclass(slots=True)
class ApprovalRequest:
    approval_id: str
    action: dict[str, Any]
    plan: ShadowPlan
    status: str = "WAITING_APPROVAL"
    approved: bool | None = None
    created_at_monotonic: float = field(default_factory=time.monotonic)
    decided_at_monotonic: float | None = None
    cancellation_latency_seconds: float | None = None
    error_type: str | None = None
    idle_runner: CancellableIdleTaskRunner | None = field(
        default=None,
        repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        now = self.decided_at_monotonic or time.monotonic()
        return {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "protocol_revision": APPROVAL_PROTOCOL_REVISION,
            "approval_id": self.approval_id,
            "status": self.status,
            "approved": self.approved,
            "wait_seconds": max(0.0, now - self.created_at_monotonic),
            "cancellation_latency_seconds": self.cancellation_latency_seconds,
            "error_type": self.error_type,
            "action": dict(self.action),
            "shadow_plan": self.plan.to_dict(),
            "idle_tasks": (
                [asdict(receipt) for receipt in self.idle_runner.receipts()]
                if self.idle_runner is not None
                else []
            ),
            "idle_tasks_running": bool(
                self.idle_runner is not None and self.idle_runner.alive
            ),
        }


class ApprovalCoordinator:
    """Pause a tool call, then consume one explicit decision during resume."""

    def __init__(
        self,
        *,
        idle_task_function: IdleTaskFunction = _default_idle_task,
    ) -> None:
        self.idle_task_function = idle_task_function
        self._lock = threading.RLock()
        self._requests: dict[str, ApprovalRequest] = {}
        self._active_approval_id: ContextVar[str | None] = ContextVar(
            f"protbind_approval_{id(self)}",
            default=None,
        )
        self._active_consumed: ContextVar[bool] = ContextVar(
            f"protbind_approval_consumed_{id(self)}",
            default=False,
        )

    @property
    def active_approval_id(self) -> str | None:
        if self._active_consumed.get():
            return None
        return self._active_approval_id.get()

    def _resumed_approval_id(self) -> str | None:
        return self._active_approval_id.get()

    def __call__(self, preview: ApprovalPreview) -> bool:
        active_id = self.active_approval_id
        if active_id is None:
            plan = build_shadow_plan(preview)
            request = ApprovalRequest(
                approval_id=uuid.uuid4().hex,
                action={
                    "tool": preview.tool,
                    "arguments_sha256": preview.arguments_sha256,
                    "reads": list(preview.reads),
                    "writes": list(preview.writes),
                    "network": preview.network,
                    "scientific_state_change": preview.scientific_state_change,
                    "expected_next_state": preview.expected_next_state,
                    "recovery": preview.recovery,
                    "manifest_sha256": preview.manifest_sha256,
                    "policy_sha256": preview.policy_sha256,
                },
                plan=plan,
            )
            runner = CancellableIdleTaskRunner(
                plan,
                task_function=self.idle_task_function,
            )
            request.idle_runner = runner
            with self._lock:
                self._requests[request.approval_id] = request
            runner.start()
            raise ToolPendingError(request.to_dict())

        self._active_consumed.set(True)
        with self._lock:
            request = self._request(active_id)
            if request.status not in {"APPROVED", "DECLINED"}:
                raise ToolPermissionError(
                    f"approval {active_id!r} cannot dispatch from {request.status}"
                )
            current = shadow_plan_is_current(
                request.plan,
                arguments_sha256=preview.arguments_sha256,
                manifest_sha256=preview.manifest_sha256,
                policy_sha256=preview.policy_sha256,
            )
            if request.plan.tool != preview.tool or not current:
                request.status = "STALE"
                request.error_type = "ApprovalBindingMismatch"
                raise ToolPermissionError(
                    "approval is stale or bound to a different tool invocation"
                )
            if not request.approved:
                return False
            request.status = "DISPATCHING"
            return True

    def _request(self, approval_id: str) -> ApprovalRequest:
        try:
            return self._requests[approval_id]
        except KeyError as exc:
            raise KeyError(f"unknown approval_id: {approval_id}") from exc

    def get(self, approval_id: str) -> dict[str, Any]:
        with self._lock:
            return self._request(approval_id).to_dict()

    def requests(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(request.to_dict() for request in self._requests.values())

    def current_status(self) -> str | None:
        active_id = self._resumed_approval_id()
        if active_id is None:
            return None
        with self._lock:
            return self._request(active_id).status

    def decide(self, approval_id: str, *, approved: bool) -> dict[str, Any]:
        with self._lock:
            request = self._request(approval_id)
            if request.status != "WAITING_APPROVAL":
                raise ValueError(
                    f"approval decision already consumed from {request.status}"
                )
            request.approved = bool(approved)
            request.status = "APPROVED" if approved else "DECLINED"
            request.decided_at_monotonic = time.monotonic()
            runner = request.idle_runner
        if runner is not None:
            latency = runner.cancel()
            with self._lock:
                request.cancellation_latency_seconds = latency
        return request.to_dict()

    @contextmanager
    def resume_scope(
        self,
        approval_id: str,
        *,
        approved: bool,
    ) -> Iterator[ApprovalRequest]:
        self.decide(approval_id, approved=approved)
        with self._lock:
            request = self._request(approval_id)
        token = self._active_approval_id.set(approval_id)
        consumed_token = self._active_consumed.set(False)
        try:
            yield request
        finally:
            self._active_consumed.reset(consumed_token)
            self._active_approval_id.reset(token)

    def mark_stale_current(self, *, error_type: str) -> None:
        active_id = self._resumed_approval_id()
        if active_id is None:
            return
        with self._lock:
            request = self._request(active_id)
            request.status = "STALE"
            request.error_type = error_type

    def complete_current(self, *, ok: bool, error_type: str | None) -> None:
        active_id = self._resumed_approval_id()
        if active_id is None:
            return
        with self._lock:
            request = self._request(active_id)
            if request.status in {"DECLINED", "STALE"}:
                if error_type is not None:
                    request.error_type = request.error_type or error_type
                return
            request.status = "EXECUTED" if ok else "FAILED"
            request.error_type = error_type
