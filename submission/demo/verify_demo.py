#!/usr/bin/env python3
"""Fail-closed acceptance checks for the public submission demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(payload + b"\n")


def _load_artifact(workspace: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    digest = artifact.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("manifest contains an invalid artifact SHA-256")
    path = workspace / "objects" / digest[:2] / digest[2:]
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise RuntimeError("content-addressed backend receipt failed SHA-256 verification")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("backend receipt is not a JSON object")
    return value


def verify_preflight(doctor_path: Path, output: Path) -> None:
    doctor = _load(doctor_path)
    hardware = doctor.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("doctor receipt omitted hardware")
    architectures = hardware.get("architectures")
    if not isinstance(architectures, list) or not any(
        isinstance(item, str) and item.startswith("gfx") for item in architectures
    ):
        raise RuntimeError("doctor did not detect an AMD GPU architecture")
    if doctor.get("hsa_override_active") is not False:
        raise RuntimeError("HSA_OVERRIDE_GFX_VERSION must not be active")
    capabilities = {
        item.get("name"): item
        for item in doctor.get("capabilities", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if not bool(capabilities.get("hipcc", {}).get("available")):
        raise RuntimeError("hipcc is unavailable")
    receipt = {
        "schema_version": "1.0",
        "kind": "protbind.submission-demo-host-preflight",
        "accepted": True,
        "architectures": architectures,
        "competition_roles": hardware.get("competition_roles", []),
        "hsa_override_active": False,
        "hipcc_available": True,
        "doctor_sha256": hashlib.sha256(doctor_path.read_bytes()).hexdigest(),
    }
    _write(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def verify_parity(cpu_path: Path, hip_path: Path, output: Path) -> None:
    cpu = _load(cpu_path)
    hip = _load(hip_path)
    checks = {
        "index_sha256_equal": cpu.get("index", {}).get("index_sha256")
        == hip.get("index", {}).get("index_sha256"),
        "query_sha256_equal": cpu.get("query_sha256") == hip.get("query_sha256"),
        "result_count_equal": cpu.get("result_count") == hip.get("result_count"),
        "complete_ranked_ids_equal": cpu.get("ranked_molecule_ids_sha256")
        == hip.get("ranked_molecule_ids_sha256"),
        "cpu_backend_exact": cpu.get("backend") == "cpu-reference",
        "hip_backend_exact": hip.get("backend")
        == "hip-prefilter+cpu-exact-ranking",
        "hip_evidence_eligible": hip.get("eligible_as_hip_performance_evidence")
        is True,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError("CPU/HIP parity gate failed: " + ", ".join(failed))
    receipt = {
        "schema_version": "1.0",
        "kind": "protbind.submission-demo-cpu-hip-parity",
        "accepted": True,
        "checks": checks,
        "result_count": cpu["result_count"],
        "ranked_molecule_ids_sha256": cpu["ranked_molecule_ids_sha256"],
        "cpu_process_p50_seconds": cpu["duration_seconds"]["p50"],
        "hip_process_p50_seconds": hip["duration_seconds"]["p50"],
        "hip_kernel_p50_seconds": hip["kernel_seconds"]["p50"],
        "scientific_boundary": (
            "Synthetic geometric-screening protocol smoke; not binding, affinity, "
            "or end-to-end acceleration evidence."
        ),
    }
    _write(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def verify_case(
    case_run_path: Path,
    manifest_path: Path,
    workspace: Path,
    output: Path,
) -> None:
    run = _load(case_run_path)
    manifest = _load(manifest_path)
    screened = manifest.get("stage_records", {}).get("SCREENED", {})
    screened_outputs = screened.get("outputs", [])
    backend_artifact = manifest.get("artifacts", {}).get("screen_backend_ligand")
    if not isinstance(backend_artifact, dict):
        raise ValueError("manifest omitted the ligand backend receipt")
    backend = _load_artifact(workspace, backend_artifact)
    checks = {
        "schema_2": run.get("schema_version") == "2.0",
        "case_id_exact": run.get("case_id") == "submission-demo-synthetic",
        "screened": run.get("state") == "SCREENED",
        "last_completed_screened": run.get("last_completed_stage") == "SCREENED",
        "no_failures": run.get("failures") == [],
        "manifest_matches_summary": manifest.get("run_id") == run.get("run_id")
        and manifest.get("state") == run.get("state"),
        "screening_artifact_present": len(screened_outputs) == 1
        and screened_outputs[0].get("producer") == "protbind.tripharm.hip-assisted",
        "backend_receipt_present": isinstance(
            run.get("artifacts", {}).get("screen_backend_ligand"), str
        ),
        "hip_backend_committed": backend.get("requested_backend") == "hip"
        and backend.get("committed_backend") == "hip"
        and backend.get("fallback") is False,
        "ranked_ids_exact": backend.get("ranked_molecule_ids_exact") is True
        and backend.get("cpu_ranked_ids_sha256")
        == backend.get("hip_ranked_ids_sha256"),
        "gfx_architecture_recorded": isinstance(backend.get("kernel"), dict)
        and str(backend["kernel"].get("architecture", "")).startswith("gfx"),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError("synthetic case acceptance failed: " + ", ".join(failed))
    receipt = {
        "schema_version": "1.0",
        "kind": "protbind.submission-demo-case-acceptance",
        "accepted": True,
        "checks": checks,
        "run_id": run["run_id"],
        "state": run["state"],
        "screening_artifact": "sha256:" + screened_outputs[0]["sha256"],
        "backend_receipt": run["artifacts"]["screen_backend_ligand"],
        "backend": backend["backend"],
        "architecture": backend["kernel"]["architecture"],
        "scientific_boundary": (
            "Synthetic protocol fixture only; it makes no biological or compound claim."
        ),
    }
    _write(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--doctor", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    parity = commands.add_parser("parity")
    parity.add_argument("--cpu", type=Path, required=True)
    parity.add_argument("--hip", type=Path, required=True)
    parity.add_argument("--output", type=Path, required=True)

    case = commands.add_parser("case")
    case.add_argument("--case-run", type=Path, required=True)
    case.add_argument("--manifest", type=Path, required=True)
    case.add_argument("--workspace", type=Path, required=True)
    case.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        verify_preflight(args.doctor, args.output)
    elif args.command == "parity":
        verify_parity(args.cpu, args.hip, args.output)
    elif args.command == "case":
        verify_case(args.case_run, args.manifest, args.workspace, args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
