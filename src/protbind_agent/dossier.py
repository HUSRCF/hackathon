"""Deterministic run-completion dossiers for humans and bounded Agents."""

from __future__ import annotations

import html
import json
from typing import Any

from . import __version__
from .artifacts import ArtifactStore, canonical_json_bytes
from .manifest import STAGE_ORDER, CofoldStatus, RunManifest, RunState
from .models import ArtifactRef
from .privacy import redact_text

DOSSIER_SCHEMA_VERSION = "1.0"
_MAIN_STAGES = STAGE_ORDER[1:]


def _artifact_summary(reference: ArtifactRef) -> dict[str, Any]:
    return {
        "artifact_id": reference.artifact_id,
        "media_type": reference.media_type,
        "size_bytes": reference.size_bytes,
        "producer": reference.producer,
        "producer_version": reference.producer_version,
        "source": reference.source,
        "license": reference.license,
    }


def _control_receipts(
    manifest: RunManifest,
    artifacts: ArtifactStore,
    control_history: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if control_history is None:
        return []
    references = control_history.get("receipts")
    if not isinstance(references, list):
        raise ValueError("control history has no receipts array")

    receipts: list[dict[str, Any]] = []
    for raw_reference in references:
        if not isinstance(raw_reference, dict):
            raise ValueError("control receipt reference must be an object")
        reference = ArtifactRef.from_dict(raw_reference)
        if reference.producer != "protbind.stage-gate-receipt":
            raise ValueError("control history contains a non-gate receipt")
        value = artifacts.read_json(reference)
        if not isinstance(value, dict) or value.get("run_id") != manifest.run_id:
            raise ValueError("control receipt is not bound to this run")
        if value.get("kind") not in {
            "protbind.stage-gate",
            "protbind.stage-acceptance",
        }:
            raise ValueError("control receipt kind is invalid")
        checks = value.get("checks", [])
        actions = value.get("required_actions", [])
        if not isinstance(checks, list) or not isinstance(actions, list):
            raise ValueError("control receipt checks/actions are invalid")
        receipts.append(
            {
                "artifact_id": reference.artifact_id,
                "kind": value["kind"],
                "phase": value.get("phase"),
                "stage": value.get("stage"),
                "decision": value.get("decision"),
                "manifest_sha256": value.get("manifest_sha256"),
                "checks": checks,
                "required_actions": [redact_text(str(item)) for item in actions],
                "automatic_retry": value.get("automatic_retry", False),
                # Continuation tokens are deliberately not copied into reports.
                "continuation_token_disclosed": False,
            }
        )
    return receipts


def _acceptance_by_stage(receipts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        stage = receipt.get("stage")
        if receipt.get("phase") == "POSTFLIGHT" and isinstance(stage, str):
            result[stage] = receipt
    return result


def _stage_rows(
    manifest: RunManifest,
    acceptances: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_failure = manifest.failures[-1] if manifest.failures else None
    rows: list[dict[str, Any]] = []
    for position, stage in enumerate(_MAIN_STAGES, start=1):
        record = manifest.stage_records.get(stage.value)
        acceptance = acceptances.get(stage.value)
        if record is not None:
            status = (
                "COMPLETED_ACCEPTED"
                if acceptance is not None and acceptance.get("decision") == "ACCEPTED"
                else "COMPLETED_UNRECEIPTED"
            )
        elif latest_failure is not None and latest_failure.stage is stage:
            status = "BLOCKED_RETRYABLE" if latest_failure.recoverable else "FAILED"
        elif manifest.next_stage is stage:
            status = "NEXT"
        else:
            status = "PENDING"
        rows.append(
            {
                "position": position,
                "stage": stage.value,
                "status": status,
                "completed": record is not None,
                "accepted": bool(
                    acceptance is not None and acceptance.get("decision") == "ACCEPTED"
                ),
                "duration_seconds": record.duration_seconds if record is not None else None,
                "completed_at": record.completed_at if record is not None else None,
                "input_sha256": record.input_hash if record is not None else None,
                "config_sha256": record.config_hash if record is not None else None,
                "cache_key_sha256": record.cache_key if record is not None else None,
                "outputs": (
                    [_artifact_summary(item) for item in record.outputs]
                    if record is not None
                    else []
                ),
                "warnings": (
                    [redact_text(item) for item in record.warnings]
                    if record is not None
                    else []
                ),
                "acceptance_receipt": (
                    acceptance.get("artifact_id") if acceptance is not None else None
                ),
            }
        )
    return rows


def build_run_dossier(
    manifest: RunManifest,
    artifacts: ArtifactStore,
    *,
    control_history: dict[str, Any] | None = None,
    pose_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a path-free snapshot without inventing completion or evidence."""

    receipts = _control_receipts(manifest, artifacts, control_history)
    stages = _stage_rows(manifest, _acceptance_by_stage(receipts))
    completed_count = sum(bool(item["completed"]) for item in stages)
    accepted_count = sum(bool(item["accepted"]) for item in stages)
    total_duration = sum(
        float(item["duration_seconds"])
        for item in stages
        if item["duration_seconds"] is not None
    )
    failures = [
        {
            **failure.to_dict(),
            "message": redact_text(failure.message),
        }
        for failure in manifest.failures
    ]
    optional_cofold: dict[str, Any] = {
        "status": manifest.cofold_status.value,
        "completed": manifest.cofold_status is CofoldStatus.COMPLETED,
        "record": (
            {
                "duration_seconds": manifest.cofold_record.duration_seconds,
                "completed_at": manifest.cofold_record.completed_at,
                "outputs": [
                    _artifact_summary(item) for item in manifest.cofold_record.outputs
                ],
                "warnings": [
                    redact_text(item) for item in manifest.cofold_record.warnings
                ],
            }
            if manifest.cofold_record is not None
            else None
        ),
        "failure": (
            {
                **manifest.cofold_failure.to_dict(),
                "message": redact_text(manifest.cofold_failure.message),
            }
            if manifest.cofold_failure is not None
            else None
        ),
    }
    return {
        "schema_version": DOSSIER_SCHEMA_VERSION,
        "kind": "protbind.run-dossier",
        "run": {
            "run_id": manifest.run_id,
            "case_id": manifest.case_id,
            "manifest_schema_version": manifest.schema_version,
            "state": manifest.state.value,
            "last_completed_stage": manifest.last_completed_stage.value,
            "next_stage": (
                manifest.next_stage.value if manifest.next_stage is not None else None
            ),
            "created_at": manifest.created_at,
            "updated_at": manifest.updated_at,
        },
        "completion": {
            "completed_stage_count": completed_count,
            "accepted_stage_count": accepted_count,
            "total_main_stage_count": len(stages),
            "core_completion_fraction": completed_count / len(stages),
            "closed_loop_acceptance_fraction": accepted_count / len(stages),
            "core_workflow_complete": completed_count == len(stages),
            "closed_loop_complete": accepted_count == len(stages),
            "total_recorded_stage_duration_seconds": total_duration,
        },
        "stages": stages,
        "optional_cofold": optional_cofold,
        "failures": failures,
        "control": {
            "policy_sha256": (
                control_history.get("policy_sha256")
                if control_history is not None
                else None
            ),
            "receipt_count": len(receipts),
            "receipts": receipts,
            "continuation_tokens_disclosed": False,
        },
        "artifacts": {
            "case": _artifact_summary(manifest.case_artifact),
            "inputs": {
                name: _artifact_summary(reference)
                for name, reference in sorted(manifest.input_artifacts.items())
            },
            "named_outputs": {
                name: _artifact_summary(reference)
                for name, reference in sorted(manifest.artifacts.items())
            },
        },
        "poses": pose_summary
        or {
            "available": False,
            "reason": "No DOCKED pose scene was available at this snapshot.",
        },
        "interpretation_limits": [
            "Stage completion means the declared computation and artifact audit completed; "
            "it does not establish experimental binding.",
            "Only ACCEPTED postflight receipts count as closed-loop stage acceptance.",
            "Vina values are tool scores, not experimental binding free energies.",
            "Pose scenes and screenshots are visual QA aids, not independent scientific evidence.",
        ],
    }


def dossier_markdown(dossier: dict[str, Any]) -> str:
    run = dossier["run"]
    completion = dossier["completion"]
    lines = [
        "# ProtBind run completion dossier",
        "",
        f"- Run: `{run['run_id']}`",
        f"- Case: `{run['case_id']}`",
        f"- State: `{run['state']}`",
        f"- Last completed stage: `{run['last_completed_stage']}`",
        f"- Next stage: `{run['next_stage'] or 'none'}`",
        "- Main stages completed: "
        f"`{completion['completed_stage_count']}/{completion['total_main_stage_count']}`",
        "- Closed-loop stages accepted: "
        f"`{completion['accepted_stage_count']}/{completion['total_main_stage_count']}`",
        "- Recorded main-stage time: "
        f"`{completion['total_recorded_stage_duration_seconds']:.3f} s`",
        "",
        "## Stage completion",
        "",
        "| # | Stage | Status | Time (s) | Outputs | Warnings | Acceptance |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for stage in dossier["stages"]:
        duration = (
            f"{stage['duration_seconds']:.3f}"
            if stage["duration_seconds"] is not None
            else "—"
        )
        lines.append(
            f"| {stage['position']} | `{stage['stage']}` | `{stage['status']}` | "
            f"{duration} | {len(stage['outputs'])} | {len(stage['warnings'])} | "
            f"`{stage['acceptance_receipt'] or 'none'}` |"
        )
        for warning in stage["warnings"]:
            lines.append(f"  - `{stage['stage']}` warning: {warning}")
    lines.extend(
        [
            "",
            "## Optional complex prediction",
            "",
            f"- Status: `{dossier['optional_cofold']['status']}`",
            "",
            "## Failures and required attention",
            "",
        ]
    )
    if dossier["failures"]:
        for failure in dossier["failures"]:
            lines.append(
                f"- `{failure['stage']}` / `{failure['code']}` / "
                f"recoverable=`{failure['recoverable']}` — {failure['message']}"
            )
    else:
        lines.append("No failure record is present.")
    lines.extend(["", "## Pose inspection", ""])
    poses = dossier["poses"]
    if not poses.get("available"):
        lines.append(str(poses.get("reason", "No pose scene is available.")))
    else:
        lines.append(
            f"- Docked candidates: `{poses['candidate_count']}`; "
            f"geometry summaries: `{poses['geometry_summary_count']}`."
        )
        for candidate in poses["candidates"]:
            validation = candidate.get("validation", {})
            geometry = candidate.get("geometry", {})
            lines.append(
                f"- `{candidate['candidate_id']}` / `{candidate['molecule_id']}` — "
                f"Vina tool score `{candidate['vina_score']}`; "
                f"PB-valid `{validation.get('posebusters_valid')}`; "
                f"minimum heavy-atom distance "
                f"`{geometry.get('minimum_heavy_atom_distance_angstrom')}` Å; "
                f"scene `{candidate['scene_artifact_id']}`."
            )
    lines.extend(["", "## Control receipts", ""])
    receipts = dossier["control"]["receipts"]
    if receipts:
        for receipt in receipts:
            lines.append(
                f"- `{receipt['phase']}` `{receipt['stage']}` → "
                f"`{receipt['decision']}` — `{receipt['artifact_id']}`"
            )
    else:
        lines.append(
            "No stage-control receipts were indexed; completed stages are not represented "
            "as closed-loop accepted."
        )
    lines.extend(["", "## Artifact inventory", ""])
    lines.append(f"- Case artifact: `{dossier['artifacts']['case']['artifact_id']}`")
    for group in ("inputs", "named_outputs"):
        lines.append(f"- {group}:")
        entries = dossier["artifacts"][group]
        if not entries:
            lines.append("  - none")
        for name, reference in entries.items():
            lines.append(
                f"  - `{name}` → `{reference['artifact_id']}` "
                f"({reference['media_type']}, {reference['producer']})"
            )
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in dossier["interpretation_limits"])
    return "\n".join(lines) + "\n"


def dossier_html(dossier: dict[str, Any]) -> str:
    markdown = dossier_markdown(dossier)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>ProtBind run dossier</title>"
        "<style>body{font:15px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:1rem;"
        "border-radius:.5rem}</style></head><body><pre>"
        + html.escape(markdown)
        + "</pre></body></html>"
    )


def persist_run_dossier(
    dossier: dict[str, Any],
    artifacts: ArtifactStore,
) -> dict[str, ArtifactRef]:
    """Persist three content-addressed views without mutating the scientific manifest."""

    return {
        "json": artifacts.put_bytes(
            canonical_json_bytes(dossier),
            media_type="application/json",
            producer="protbind.run-dossier",
            producer_version=__version__,
        ),
        "markdown": artifacts.put_bytes(
            dossier_markdown(dossier).encode("utf-8"),
            media_type="text/markdown",
            producer="protbind.run-dossier",
            producer_version=__version__,
        ),
        "html": artifacts.put_bytes(
            dossier_html(dossier).encode("utf-8"),
            media_type="text/html",
            producer="protbind.run-dossier",
            producer_version=__version__,
        ),
    }


def dossier_content(dossier: dict[str, Any], format: str) -> str:
    if format == "json":
        return json.dumps(dossier, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if format == "markdown":
        return dossier_markdown(dossier)
    if format == "html":
        return dossier_html(dossier)
    raise ValueError("dossier format must be json, markdown, or html")
