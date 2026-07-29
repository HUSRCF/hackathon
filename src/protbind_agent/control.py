"""One-stage-at-a-time closed-loop control for interactive ProtBind clients."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import canonical_json_bytes, sha256_bytes
from .manifest import RunManifest, RunState
from .models import ArtifactRef, ResearchMode
from .privacy import redact_text
from .workflow import ProtBindWorkflow

CONTROL_SCHEMA_VERSION = "1.0"

_CONTROL_POLICY = {
    "schema_version": CONTROL_SCHEMA_VERSION,
    "execution_granularity": "exactly-one-main-stage",
    "preflight_required": True,
    "postflight_manifest_reaudit_required": True,
    "continuation_token_binds": [
        "run_id",
        "manifest_sha256",
        "next_stage",
        "policy_sha256",
    ],
    "automatic_retry": False,
    "terminal_decisions": ["UNSUPPORTED", "FAILED", "COMPLETE"],
}
CONTROL_POLICY_SHA256 = sha256_bytes(canonical_json_bytes(_CONTROL_POLICY))


class GateDecision(StrEnum):
    READY = "READY"
    ACCEPTED = "ACCEPTED"
    NEEDS_ACTION = "NEEDS_ACTION"
    RETRYABLE = "RETRYABLE"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    status: CheckStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": redact_text(self.detail),
        }


def _manifest_sha256(manifest: RunManifest) -> str:
    return sha256_bytes(canonical_json_bytes(manifest.to_dict()))


def _continuation_token(
    manifest: RunManifest,
    stage: RunState,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": CONTROL_SCHEMA_VERSION,
                "run_id": manifest.run_id,
                "manifest_sha256": _manifest_sha256(manifest),
                "next_stage": stage.value,
                "policy_sha256": CONTROL_POLICY_SHA256,
            }
        )
    )


def _failure_action(code: str) -> str:
    actions = {
        "CAPABILITY_UNAVAILABLE": (
            "Install or configure the named local capability, run doctor, then request "
            "a fresh gate."
        ),
        "INPUT_NOT_PREPARED": (
            "Attach the exact missing support artifact or configure the named worker, "
            "then request a fresh gate."
        ),
        "RECEPTOR_UNAVAILABLE": (
            "Attach a content-addressed receptor or a verified ESMFold-v1 receipt."
        ),
        "LIGAND_QUERY_UNAVAILABLE": (
            "Provide reference SMILES or a ligand pharmacophore artifact."
        ),
        "POCKET_QUERY_UNAVAILABLE": (
            "Provide a pocket pharmacophore or a receptor plus a bounded pocket hypothesis."
        ),
        "NO_SELECTABLE_CANDIDATES": (
            "Inspect explicit quick-Vina failures; revise inputs in a new protocol run "
            "or attach an independently frozen selection batch."
        ),
        "WORKER_CRASH": (
            "Inspect the redacted worker failure, verify the pinned runtime and resource "
            "lease, then explicitly retry this stage."
        ),
    }
    return actions.get(
        code,
        "Resolve the recorded recoverable failure and explicitly request a fresh stage gate.",
    )


def _is_unsupported(code: str, message: str) -> bool:
    value = f"{code} {message}".upper()
    return code.upper().startswith("UNSUPPORTED") or any(
        marker in value
        for marker in (
            "COVALENT LIGAND",
            "POLYMER LIGAND",
            "METAL CENTER",
            "NONSTANDARD CHEMISTRY",
        )
    )


class StageControlStore:
    """Atomic path-safe receipt index kept beside the scientific manifest."""

    def __init__(self, workflow: ProtBindWorkflow) -> None:
        self.workflow = workflow

    def _run_dir(self, run_id: str) -> Path:
        return self.workflow.manifests.path_for(run_id).parent

    def _ledger_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "control.json"

    @contextmanager
    def lock(self, run_id: str) -> Iterator[None]:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "control.lock"
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def record(self, run_id: str, receipt: ArtifactRef) -> None:
        path = self._ledger_path(run_id)
        value: dict[str, Any]
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != CONTROL_SCHEMA_VERSION
                or value.get("run_id") != run_id
                or value.get("policy_sha256") != CONTROL_POLICY_SHA256
                or not isinstance(value.get("receipts"), list)
            ):
                raise ValueError("stage-control ledger is invalid")
        else:
            value = {
                "schema_version": CONTROL_SCHEMA_VERSION,
                "run_id": run_id,
                "policy_sha256": CONTROL_POLICY_SHA256,
                "receipts": [],
            }
        references = value["receipts"]
        assert isinstance(references, list)
        if not any(
            isinstance(item, dict) and item.get("sha256") == receipt.sha256
            for item in references
        ):
            references.append(receipt.to_dict())
        data = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            + b"\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self, run_id: str) -> dict[str, Any]:
        path = self._ledger_path(run_id)
        if not path.is_file():
            return {
                "schema_version": CONTROL_SCHEMA_VERSION,
                "run_id": run_id,
                "policy_sha256": CONTROL_POLICY_SHA256,
                "receipts": [],
            }
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != CONTROL_SCHEMA_VERSION
            or value.get("run_id") != run_id
            or value.get("policy_sha256") != CONTROL_POLICY_SHA256
            or not isinstance(value.get("receipts"), list)
        ):
            raise ValueError("stage-control ledger is invalid")
        for item in value["receipts"]:
            try:
                reference = ArtifactRef.from_dict(item)
            except (TypeError, ValueError) as exc:
                raise ValueError("stage-control ledger contains an invalid receipt") from exc
            self.workflow.artifacts.resolve(reference)
            if reference.producer != "protbind.stage-gate-receipt":
                raise ValueError("stage-control ledger contains a non-receipt artifact")
        return value


class StageGateController:
    """Advance a run only after a fresh gate and accept exactly one stage."""

    def __init__(self, workflow: ProtBindWorkflow) -> None:
        self.workflow = workflow
        self.store = StageControlStore(workflow)

    @staticmethod
    def _summary(manifest: RunManifest) -> dict[str, Any]:
        failure = manifest.failures[-1] if manifest.failures else None
        return {
            "run_id": manifest.run_id,
            "case_id": manifest.case_id,
            "state": manifest.state.value,
            "last_completed_stage": manifest.last_completed_stage.value,
            "next_stage": (
                manifest.next_stage.value if manifest.next_stage is not None else None
            ),
            "cofold_status": manifest.cofold_status.value,
            "latest_failure": (
                {
                    "stage": failure.stage.value,
                    "code": failure.code,
                    "message": redact_text(failure.message),
                    "recoverable": failure.recoverable,
                }
                if failure is not None
                else None
            ),
        }

    def _requirements(
        self,
        manifest: RunManifest,
        stage: RunState,
    ) -> tuple[list[GateCheck], list[str]]:
        checks: list[GateCheck] = []
        actions: list[str] = []
        case = self.workflow.load_case(manifest)

        def require(name: str, condition: bool, failure: str, action: str) -> None:
            checks.append(
                GateCheck(
                    name=name,
                    status=CheckStatus.PASS if condition else CheckStatus.FAIL,
                    detail="requirement satisfied" if condition else failure,
                )
            )
            if not condition:
                actions.append(action)

        if stage is RunState.RECEPTOR_READY:
            receptor_available = any(
                item is not None
                for item in (
                    case.target.structure,
                    manifest.artifacts.get("support_esmfold_structure"),
                    manifest.artifacts.get("support_receptor_structure"),
                )
            )
            require(
                "receptor-available",
                receptor_available,
                "no content-addressed receptor is available",
                "Attach receptor_structure or a verified esmfold_receipt.",
            )
        elif stage is RunState.SCREENED:
            if case.mode in {ResearchMode.BOTH, ResearchMode.LIGAND_ONLY}:
                ligand_ready = case.ligand is not None and (
                    case.ligand.smiles is not None
                    or case.ligand.pharmacophore is not None
                )
                require(
                    "ligand-query-available",
                    ligand_ready,
                    "ligand branch lacks SMILES and pharmacophore",
                    "Provide reference SMILES or a ligand pharmacophore.",
                )
            if case.mode in {ResearchMode.BOTH, ResearchMode.POCKET_ONLY}:
                receptor_ready = manifest.artifacts.get("receptor_ready_structure")
                pocket_ready = case.pocket is not None and (
                    case.pocket.pharmacophore is not None
                    or receptor_ready is not None
                )
                require(
                    "pocket-query-available",
                    pocket_ready,
                    "pocket branch lacks a pharmacophore or receptor",
                    "Provide a pocket pharmacophore or bounded pocket hypothesis.",
                )
        elif stage is RunState.SELECTED:
            manual = (
                manifest.artifacts.get("support_selection_batch") is not None
                or manifest.artifacts.get("support_openfold_batch") is not None
            )
            automatic = self.workflow.config.workers.get(RunState.SELECTED) is not None
            require(
                "selection-path-configured",
                manual or automatic,
                "neither a frozen selection batch nor quick-Vina worker is configured",
                "Attach selection_batch or configure the SELECTED quick-Vina worker.",
            )
            if automatic and not manual:
                require(
                    "vina-environment-lock",
                    manifest.artifacts.get("support_vina_environment_lock") is not None,
                    "automatic selection lacks support_vina_environment_lock",
                    "Attach the exact Vina/Meeko environment lock.",
                )
        elif stage in {RunState.DOCKED, RunState.VALIDATED}:
            require(
                f"{stage.value.lower()}-worker-configured",
                self.workflow.config.workers.get(stage) is not None,
                f"no {stage.value} worker is configured",
                f"Configure the pinned {stage.value} worker before continuing.",
            )

        if not checks:
            checks.append(
                GateCheck(
                    name="stage-prerequisites",
                    status=CheckStatus.PASS,
                    detail="no additional stage-specific prerequisite is missing",
                )
            )
        return checks, actions

    def _persist_receipt(self, payload: dict[str, Any]) -> ArtifactRef:
        receipt = self.workflow.artifacts.put_json(
            payload,
            producer="protbind.stage-gate-receipt",
            producer_version=__version__,
        )
        self.store.record(str(payload["run_id"]), receipt)
        return receipt

    def inspect(self, run_id: str) -> dict[str, Any]:
        manifest = self.workflow.manifests.load(run_id)
        checks: list[GateCheck] = []
        actions: list[str] = []
        stage = manifest.next_stage
        decision: GateDecision
        token: str | None = None

        try:
            self.workflow.audit_manifest(manifest)
            checks.append(
                GateCheck(
                    "manifest-and-artifact-audit",
                    CheckStatus.PASS,
                    "all completed stage bindings and artifact hashes revalidated",
                )
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            checks.append(
                GateCheck(
                    "manifest-and-artifact-audit",
                    CheckStatus.FAIL,
                    redact_text(str(exc)),
                )
            )
            decision = GateDecision.FAILED
            actions.append("Create a new run or restore the exact bound code/input artifacts.")
        else:
            if manifest.state is RunState.FAILED:
                failure = manifest.failures[-1]
                decision = (
                    GateDecision.UNSUPPORTED
                    if _is_unsupported(failure.code, failure.message)
                    else GateDecision.FAILED
                )
                checks.append(
                    GateCheck(
                        "terminal-failure",
                        CheckStatus.FAIL,
                        f"{failure.code}: {failure.message}",
                    )
                )
                actions.append("Create a new run with corrected inputs or protocol.")
            elif stage is None:
                decision = GateDecision.COMPLETE
                checks.append(
                    GateCheck(
                        "workflow-complete",
                        CheckStatus.PASS,
                        "all main stages have accepted records",
                    )
                )
            elif manifest.state is RunState.DEGRADED and _is_unsupported(
                manifest.failures[-1].code,
                manifest.failures[-1].message,
            ):
                failure = manifest.failures[-1]
                decision = GateDecision.UNSUPPORTED
                checks.append(
                    GateCheck(
                        "unsupported-system",
                        CheckStatus.FAIL,
                        f"{failure.code}: {failure.message}",
                    )
                )
                actions.append(
                    "Keep the system unsupported or start a separately revised protocol."
                )
            else:
                stage_checks, stage_actions = self._requirements(manifest, stage)
                checks.extend(stage_checks)
                actions.extend(stage_actions)
                requirements_pass = all(
                    item.status is not CheckStatus.FAIL for item in stage_checks
                )
                if not requirements_pass:
                    decision = GateDecision.NEEDS_ACTION
                elif manifest.state is RunState.DEGRADED:
                    failure = manifest.failures[-1]
                    if failure.recoverable:
                        decision = GateDecision.RETRYABLE
                        actions.append(_failure_action(failure.code))
                        token = _continuation_token(manifest, stage)
                    else:
                        decision = GateDecision.FAILED
                else:
                    decision = GateDecision.READY
                    token = _continuation_token(manifest, stage)

        payload = {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "kind": "protbind.stage-gate",
            "phase": "PREFLIGHT",
            "run_id": manifest.run_id,
            "stage": stage.value if stage is not None else None,
            "decision": decision.value,
            "manifest_sha256": _manifest_sha256(manifest),
            "manifest_updated_at": manifest.updated_at,
            "policy_sha256": CONTROL_POLICY_SHA256,
            "checks": [item.to_dict() for item in checks],
            "required_actions": list(dict.fromkeys(actions)),
            "continuation_token": token,
            "automatic_retry": False,
        }
        reference = self._persist_receipt(payload)
        return {
            "gate": payload,
            "gate_receipt": reference.to_dict(),
            "run": self._summary(manifest),
        }

    def advance(self, run_id: str, continuation_token: str) -> dict[str, Any]:
        if not isinstance(continuation_token, str) or len(continuation_token) != 64:
            raise ValueError("continuation_token must be a SHA-256 token")
        with self.store.lock(run_id):
            preflight = self.inspect(run_id)
            gate = preflight["gate"]
            if gate["decision"] not in {
                GateDecision.READY.value,
                GateDecision.RETRYABLE.value,
            }:
                raise ValueError(
                    f"stage cannot advance while gate decision is {gate['decision']}"
                )
            if gate["continuation_token"] != continuation_token:
                raise ValueError(
                    "stale or mismatched continuation token; request a fresh case_status"
                )
            stage = RunState(str(gate["stage"]))
            before = self.workflow.manifests.load(run_id)
            before_sha256 = _manifest_sha256(before)
            result = self.workflow.run(before, stop_after=stage)
            result = self.workflow.manifests.load(result.run_id)

            checks: list[GateCheck] = [
                GateCheck(
                    "preflight-token",
                    CheckStatus.PASS,
                    "fresh token matched the exact pre-stage manifest and policy",
                )
            ]
            actions: list[str] = []
            if result.state is stage and result.last_completed_stage is stage:
                try:
                    self.workflow.audit_manifest(result)
                    record = result.stage_records[stage.value]
                    for output in record.outputs:
                        self.workflow.artifacts.resolve(output)
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    decision = GateDecision.FAILED
                    checks.append(
                        GateCheck(
                            "postflight-audit",
                            CheckStatus.FAIL,
                            redact_text(str(exc)),
                        )
                    )
                    actions.append(
                        "Do not continue; restore the exact stage artifacts or create a new run."
                    )
                else:
                    decision = GateDecision.ACCEPTED
                    checks.extend(
                        (
                            GateCheck(
                                "stage-transition",
                                CheckStatus.PASS,
                                f"exactly one main stage advanced to {stage.value}",
                            ),
                            GateCheck(
                                "postflight-audit",
                                CheckStatus.PASS,
                                "stage record, cache binding, outputs and configuration "
                                "revalidated",
                            ),
                        )
                    )
                    if record.warnings:
                        checks.append(
                            GateCheck(
                                "stage-warnings",
                                CheckStatus.WARN,
                                "; ".join(record.warnings),
                            )
                        )
            else:
                failure = result.failures[-1] if result.failures else None
                if failure is None:
                    decision = GateDecision.FAILED
                    detail = "stage returned without completion or a failure record"
                    actions.append("Do not continue; inspect the run manifest.")
                elif _is_unsupported(failure.code, failure.message):
                    decision = GateDecision.UNSUPPORTED
                    detail = f"{failure.code}: {failure.message}"
                    actions.append(
                        "Keep this system unsupported or start a separately revised protocol."
                    )
                elif failure.recoverable:
                    stage_checks, stage_actions = self._requirements(result, stage)
                    if any(item.status is CheckStatus.FAIL for item in stage_checks):
                        decision = GateDecision.NEEDS_ACTION
                    else:
                        decision = GateDecision.RETRYABLE
                    detail = f"{failure.code}: {failure.message}"
                    actions.extend(stage_actions)
                    actions.append(_failure_action(failure.code))
                else:
                    decision = GateDecision.FAILED
                    detail = f"{failure.code}: {failure.message}"
                    actions.append("Create a new run with corrected inputs or protocol.")
                checks.append(
                    GateCheck(
                        "stage-transition",
                        CheckStatus.FAIL,
                        detail,
                    )
                )

            payload = {
                "schema_version": CONTROL_SCHEMA_VERSION,
                "kind": "protbind.stage-acceptance",
                "phase": "POSTFLIGHT",
                "run_id": result.run_id,
                "stage": stage.value,
                "decision": decision.value,
                "pre_manifest_sha256": before_sha256,
                "manifest_sha256": _manifest_sha256(result),
                "manifest_updated_at": result.updated_at,
                "policy_sha256": CONTROL_POLICY_SHA256,
                "preflight_receipt": preflight["gate_receipt"],
                "checks": [item.to_dict() for item in checks],
                "required_actions": list(dict.fromkeys(actions)),
                "automatic_retry": False,
            }
            acceptance = self._persist_receipt(payload)
            next_gate = self.inspect(run_id)
            return {
                "acceptance": payload,
                "acceptance_receipt": acceptance.to_dict(),
                "next_gate": next_gate,
                "run": self._summary(result),
            }
