#!/usr/bin/env python3
"""CPU-only AutoDock Vina profile used solely for selection pruning.

This adapter deliberately delegates receptor/ligand preparation, chemical identity
checks, command execution, and score extraction to ``vina_worker``.  Its separate
protocol prevents a low-cost ranking run from being mistaken for the evidence-grade
DOCKED stage.
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

WORKER_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = WORKER_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(WORKER_ROOT))

import vina_worker as vina  # noqa: E402

from protbind_agent.artifacts import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from protbind_agent.chemistry import smiles_formal_charge  # noqa: E402
from protbind_agent.models import ArtifactRef  # noqa: E402
from protbind_agent.selection import (  # noqa: E402
    DOCKING_BOX_COORDINATE_FRAME,
    QUICK_VINA_BATCH_KIND,
    QUICK_VINA_INPUT_KIND,
    QUICK_VINA_PURPOSE,
    validate_docking_box_receipt,
)
from protbind_agent.worker_protocol import (  # noqa: E402
    WorkerProvenance,
    WorkerRequest,
    WorkerResponse,
)
from protbind_agent.worker_sdk import WorkerFailure, serve_worker  # noqa: E402

ENGINE = "vina-quick"
PROFILE_VERSION = "1.3"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_LOCAL_FAILURE_CODES = frozenset({"UNSUPPORTED_CHEMISTRY"})


def base_model_revision(parameters: dict[str, Any]) -> str:
    versions = {
        name: parameters.get(name)
        for name in ("rdkit_version", "gemmi_version", "numpy_version", "scipy_version")
    }
    if any(not isinstance(value, str) or not value.strip() for value in versions.values()):
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "quick Vina requires exact RDKit/Gemmi/NumPy/SciPy version pins",
            recoverable=False,
        )
    return (
        f"autodock-vina-{vina.VINA_VERSION}+meeko-{vina.MEEKO_VERSION}+"
        f"rdkit-{versions['rdkit_version']}+gemmi-{versions['gemmi_version']}+"
        f"numpy-{versions['numpy_version']}+scipy-{versions['scipy_version']}"
    )


def quick_model_revision(parameters: dict[str, Any]) -> str:
    return f"selection-quick-vina-{PROFILE_VERSION}+{base_model_revision(parameters)}"


def composite_code_sha256(
    environment_lock_sha256: str,
    runtime_assets_sha256: str,
) -> str:
    """Bind the thin profile, shared Vina implementation, and frozen runtime."""

    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "profile": {
                    "name": ENGINE,
                    "version": PROFILE_VERSION,
                    "purpose": QUICK_VINA_PURPOSE,
                    "cpu": 1,
                    "maximum_exhaustiveness": 16,
                    "maximum_num_modes": 3,
                },
                "adapter_sha256": sha256_file(Path(__file__)),
                "shared_vina_adapter_sha256": sha256_file(Path(vina.__file__)),
                "protbind_runtime_sha256": vina.protbind_runtime_sha256(),
                "environment_lock_sha256": environment_lock_sha256,
                "runtime_assets_sha256": runtime_assets_sha256,
                "vina_version": vina.VINA_VERSION,
                "meeko_version": vina.MEEKO_VERSION,
            }
        )
    )


def _reference(value: Any, name: str) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerFailure(
            "INVALID_INPUT",
            f"{name} is not a valid artifact reference",
            recoverable=False,
        ) from exc


def _vector(value: Any, name: str, *, positive: bool = False) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            not isinstance(item, int | float)
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise WorkerFailure(
            "INVALID_INPUT",
            f"{name} must contain three finite numbers",
            recoverable=False,
        )
    result = [float(item) for item in value]
    if positive and any(item <= 0 for item in result):
        raise WorkerFailure(
            "INVALID_INPUT", f"{name} values must be positive", recoverable=False
        )
    return result


def _profile(parameters: dict[str, Any]) -> dict[str, Any]:
    cpu = parameters.get("cpu", 1)
    exhaustiveness = parameters.get("exhaustiveness", 8)
    num_modes = parameters.get("num_modes", 1)
    if cpu != 1 or isinstance(cpu, bool):
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "selection quick Vina is pinned to one CPU thread",
            recoverable=False,
        )
    if (
        not isinstance(exhaustiveness, int)
        or isinstance(exhaustiveness, bool)
        or not 1 <= exhaustiveness <= 16
    ):
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "quick Vina exhaustiveness must be an integer in [1, 16]",
            recoverable=False,
        )
    if (
        not isinstance(num_modes, int)
        or isinstance(num_modes, bool)
        or not 1 <= num_modes <= 3
    ):
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "quick Vina num_modes must be an integer in [1, 3]",
            recoverable=False,
        )
    if parameters.get("scoring", "vina") != "vina":
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "selection quick Vina is pinned to the Vina scoring function",
            recoverable=False,
        )
    return {
        "cpu": 1,
        "exhaustiveness": exhaustiveness,
        "num_modes": num_modes,
        "scoring": "vina",
    }


def _expected_request_id(value: dict[str, Any], request: dict[str, Any]) -> str:
    digest = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "purpose": QUICK_VINA_PURPOSE,
                "screening_sha256": value["screening_sha256"],
                "library_index_sha256": value["library_index_sha256"],
                "receptor_sha256": _reference(value["receptor"], "receptor").sha256,
                "molecule_id": request["molecule_id"],
                "microstate_id": request["microstate_id"],
                "canonical_isomeric_smiles": request["canonical_isomeric_smiles"],
                "heavy_element_counts": request["heavy_element_counts"],
                "formal_charge": request["formal_charge"],
                "box_center": request["box_center"],
                "box_size": request["box_size"],
                "box_source": request["box_source"],
                "coordinate_frame": request["coordinate_frame"],
                "docking_box_receipt_sha256": request[
                    "docking_box_receipt_sha256"
                ],
            }
        )
    )
    return f"quick-{digest[:24]}"


def _read_input(request: WorkerRequest, store: Any) -> dict[str, Any]:
    value = store.read_json(request.input)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "1.0"
        or value.get("kind") != QUICK_VINA_INPUT_KIND
        or value.get("purpose") != QUICK_VINA_PURPOSE
    ):
        raise WorkerFailure(
            "INVALID_INPUT",
            "vina-quick requires a protbind.quick-vina-input v1.0 artifact",
            recoverable=False,
        )
    for name in (
        "selection_preparation_sha256",
        "screening_sha256",
        "library_index_sha256",
    ):
        if not isinstance(value.get(name), str) or _SHA256.fullmatch(value[name]) is None:
            raise WorkerFailure(
                "INVALID_INPUT", f"{name} must be a SHA-256 commitment", recoverable=False
            )
    case_id = value.get("case_id")
    if not isinstance(case_id, str) or _SAFE_ID.fullmatch(case_id) is None:
        raise WorkerFailure("INVALID_INPUT", "case_id is not path-safe", recoverable=False)
    receptor = _reference(value.get("receptor"), "quick Vina receptor")
    environment_lock = _reference(
        value.get("environment_lock"), "quick Vina environment lock"
    )
    box_receipt = _reference(
        value.get("docking_box_receipt"), "quick Vina docking box receipt"
    )
    store.resolve(receptor)
    store.resolve(environment_lock)
    try:
        receipt_value = validate_docking_box_receipt(
            store, box_receipt, receptor=receptor
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise WorkerFailure(
            "INVALID_INPUT",
            f"quick Vina docking box receipt is invalid: {exc}",
            recoverable=False,
        ) from exc
    if value.get("site_derivation_evidence") != receipt_value.get(
        "site_derivation_evidence"
    ):
        raise WorkerFailure(
            "INVALID_INPUT",
            "quick Vina site evidence differs from its docking box receipt",
            recoverable=False,
        )
    requests = value.get("requests")
    if (
        not isinstance(requests, list)
        or not 1 <= len(requests) <= 256
        or value.get("request_count") != len(requests)
    ):
        raise WorkerFailure(
            "INVALID_INPUT",
            "quick Vina input must contain exactly 1..256 counted requests",
            recoverable=False,
        )
    observed_ids: set[str] = set()
    observed_keys: set[tuple[str, str]] = set()
    parsed: list[dict[str, Any]] = []
    for raw in requests:
        if not isinstance(raw, dict):
            raise WorkerFailure(
                "INVALID_INPUT", "quick Vina request must be an object", recoverable=False
            )
        item = dict(raw)
        ids = tuple(item.get(name) for name in ("request_id", "molecule_id", "microstate_id"))
        if any(not isinstance(entry, str) or _SAFE_ID.fullmatch(entry) is None for entry in ids):
            raise WorkerFailure(
                "INVALID_INPUT", "quick Vina request IDs are not path-safe", recoverable=False
            )
        smiles = item.get("canonical_isomeric_smiles")
        counts = item.get("heavy_element_counts")
        formal_charge = item.get("formal_charge")
        box_source = item.get("box_source")
        if (
            not isinstance(smiles, str)
            or not smiles.strip()
            or not isinstance(counts, dict)
            or not counts
            or any(
                not isinstance(element, str)
                or element != element.upper()
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
                for element, count in counts.items()
            )
            or not isinstance(formal_charge, int)
            or isinstance(formal_charge, bool)
            or not isinstance(box_source, str)
            or not box_source.strip()
            or item.get("purpose") != QUICK_VINA_PURPOSE
        ):
            raise WorkerFailure(
                "INVALID_INPUT",
                "quick Vina request lacks frozen chemistry, box source, or purpose",
                recoverable=False,
            )
        if formal_charge != smiles_formal_charge(smiles):
            raise WorkerFailure(
                "INVALID_INPUT",
                "quick Vina request formal charge differs from its SMILES",
                recoverable=False,
            )
        item["box_center"] = _vector(item.get("box_center"), "box_center")
        item["box_size"] = _vector(item.get("box_size"), "box_size", positive=True)
        if (
            item["box_center"] != receipt_value["center"]
            or item["box_size"] != receipt_value["size"]
            or box_source != receipt_value["source_kind"]
            or item.get("coordinate_frame") != DOCKING_BOX_COORDINATE_FRAME
            or item.get("docking_box_receipt_sha256") != box_receipt.sha256
        ):
            raise WorkerFailure(
                "INVALID_INPUT",
                "quick Vina request box differs from its coordinate-frame receipt",
                recoverable=False,
            )
        request_id, molecule_id, microstate_id = ids
        key = (molecule_id, microstate_id)
        if request_id in observed_ids or key in observed_keys:
            raise WorkerFailure(
                "INVALID_INPUT", "quick Vina request identities repeat", recoverable=False
            )
        if request_id != _expected_request_id(value, item):
            raise WorkerFailure(
                "INVALID_INPUT",
                "quick Vina request_id differs from its input commitments",
                recoverable=False,
            )
        observed_ids.add(request_id)
        observed_keys.add(key)
        parsed.append(item)
    return {
        "value": value,
        "receptor": receptor,
        "environment_lock": environment_lock,
        "box_receipt": box_receipt,
        "box_receipt_value": receipt_value,
        "requests": parsed,
    }


def _artifact(value: Any) -> ArtifactRef | None:
    if not isinstance(value, dict):
        return None
    required = {"sha256", "media_type", "size_bytes", "producer"}
    allowed = required | {"producer_version", "source", "license"}
    if not required.issubset(value) or not set(value).issubset(allowed):
        return None
    try:
        return ArtifactRef.from_dict(value)
    except (KeyError, TypeError, ValueError):
        return None


def _output_closure(
    store: Any,
    primary: ArtifactRef,
    *,
    allowed_inputs: set[ArtifactRef],
) -> tuple[ArtifactRef, ...]:
    """Return every non-input ArtifactRef reachable from the primary output."""

    ordered = [primary]
    observed = {primary}
    queued = [primary]
    while queued:
        reference = queued.pop(0)
        store.resolve(reference)
        if reference.media_type != "application/json":
            continue
        value = store.read_json(reference)

        def visit(item: Any) -> None:
            nested = _artifact(item)
            if nested is not None:
                store.resolve(nested)
                if nested in allowed_inputs or nested in observed:
                    return
                observed.add(nested)
                ordered.append(nested)
                if nested.media_type == "application/json":
                    queued.append(nested)
                return
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
    return tuple(ordered)


def _handler(request: WorkerRequest, store: Any) -> WorkerResponse:
    started = time.perf_counter()
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise WorkerFailure(
            "OFFLINE_POLICY_VIOLATION",
            "HSA_OVERRIDE_GFX_VERSION is forbidden",
            recoverable=False,
        )
    if any(os.environ.get(name) for name in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    )):
        raise WorkerFailure(
            "RESOURCE_POLICY_VIOLATION",
            "selection quick Vina is CPU-only and rejects visible GPU masks",
            recoverable=False,
        )
    if request.seed == 0 or request.seed > 2**31 - 1:
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "deterministic quick Vina requires a seed in [1, 2147483647]",
            recoverable=False,
        )
    profile = _profile(request.parameters)
    parsed = _read_input(request, store)
    runtime_assets_sha256 = request.provenance.weight_sha256
    expected_revision = quick_model_revision(request.parameters)
    expected_code = composite_code_sha256(
        parsed["environment_lock"].sha256, runtime_assets_sha256
    )
    if request.provenance.model_revision != expected_revision:
        raise WorkerFailure(
            "PROVENANCE_MISMATCH",
            "quick Vina profile revision differs from the pinned toolchain",
            recoverable=False,
        )
    if request.provenance.code_sha256 != expected_code:
        raise WorkerFailure(
            "CODE_HASH_MISMATCH",
            "quick Vina composite code identity differs from provenance",
            recoverable=False,
        )

    candidates = [
        {
            "candidate_id": item["request_id"],
            "request_id": item["request_id"],
            "molecule_id": item["molecule_id"],
            "microstate_id": item["microstate_id"],
            "canonical_isomeric_smiles": item["canonical_isomeric_smiles"],
            "heavy_element_counts": item["heavy_element_counts"],
            "formal_charge": item["formal_charge"],
            "receptor": parsed["receptor"].to_dict(),
            "box_center": item["box_center"],
            "box_size": item["box_size"],
            "box_source": item["box_source"],
            "coordinate_frame": item["coordinate_frame"],
            "docking_box_receipt": parsed["box_receipt"].to_dict(),
        }
        for item in parsed["requests"]
    ]
    internal_selection = store.put_json(
        {
            "schema_version": "2.0",
            "kind": "protbind.selection-bundle",
            "purpose": QUICK_VINA_PURPOSE,
            "receptor": parsed["receptor"].to_dict(),
            "docking_box_receipt": parsed["box_receipt"].to_dict(),
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        producer="vina-quick.internal-selection",
        producer_version=PROFILE_VERSION,
        source=request.input.artifact_id,
    )
    internal_envelope = store.put_json(
        {
            "schema_version": "2.0",
            "kind": "protbind.stage-input",
            "stage": "DOCKED",
            "case_id": parsed["value"]["case_id"],
            "input_artifacts": {},
            "supporting_artifacts": {
                "support_vina_environment_lock": parsed["environment_lock"].to_dict()
            },
            "previous": {
                "stage": "SELECTED",
                "scientific_outputs": [internal_selection.to_dict()],
                "receipt": None,
            },
        },
        producer="vina-quick.internal-stage-input",
        producer_version=PROFILE_VERSION,
        source=request.input.artifact_id,
    )
    inner_provenance = WorkerProvenance(
        model_revision=base_model_revision(request.parameters),
        weight_sha256=runtime_assets_sha256,
        code_sha256=vina.composite_code_sha256(
            parsed["environment_lock"].sha256, runtime_assets_sha256
        ),
    )
    inner_request = WorkerRequest(
        job_id=f"{request.job_id}-base-vina",
        engine=vina.ENGINE,
        input=internal_envelope,
        parameters=request.parameters,
        seed=request.seed,
        provenance=inner_provenance,
    )
    inner_response = vina._handler(inner_request, store)
    if (
        inner_response.job_id != inner_request.job_id
        or inner_response.engine != inner_request.engine
        or inner_response.provenance != inner_request.provenance
    ):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "shared Vina handler returned a mismatched response identity",
            recoverable=False,
        )
    if inner_response.error is not None:
        raise WorkerFailure(
            inner_response.error.code,
            inner_response.error.message,
            recoverable=inner_response.error.recoverable,
        )
    if not inner_response.outputs:
        raise WorkerFailure(
            "TOOL_EXECUTION_FAILED",
            "shared Vina handler returned no verified docking bundle",
            recoverable=True,
        )
    inner_bundle = inner_response.outputs[0]
    bundle_value = store.read_json(inner_bundle)
    if not isinstance(bundle_value, dict) or bundle_value.get("kind") != (
        "protbind.docking-bundle"
    ):
        raise WorkerFailure(
            "OUTPUT_INVALID", "shared Vina output is not a docking bundle", recoverable=False
        )
    successes = bundle_value.get("candidates")
    failures = bundle_value.get("failures")
    if not isinstance(successes, list) or not isinstance(failures, list):
        raise WorkerFailure(
            "OUTPUT_INVALID", "shared Vina bundle lacks explicit coverage", recoverable=False
        )
    success_by_id = {
        str(item.get("parent_candidate_id")): item
        for item in successes
        if isinstance(item, dict)
    }
    failure_by_id = {
        str(item.get("candidate_id")): item
        for item in failures
        if isinstance(item, dict)
    }
    if (
        len(success_by_id) != len(successes)
        or len(failure_by_id) != len(failures)
        or set(success_by_id) & set(failure_by_id)
        or (set(success_by_id) | set(failure_by_id))
        != {str(item["request_id"]) for item in parsed["requests"]}
    ):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "shared Vina result identities are ambiguous or incomplete",
            recoverable=False,
        )

    evaluations: list[dict[str, Any]] = []
    for item in parsed["requests"]:
        request_id = item["request_id"]
        failure = failure_by_id.get(request_id)
        if failure is not None:
            error = failure.get("error")
            if not isinstance(error, dict):
                raise WorkerFailure(
                    "OUTPUT_INVALID", "shared Vina failure lacks an error", recoverable=False
                )
            code = error.get("code")
            reason = error.get("message")
            recoverable = error.get("recoverable")
            if (
                not isinstance(code, str)
                or not code.strip()
                or not isinstance(reason, str)
                or not reason.strip()
                or not isinstance(recoverable, bool)
            ):
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "shared Vina failure has an invalid error receipt",
                    recoverable=False,
                )
            if code not in _CANDIDATE_LOCAL_FAILURE_CODES:
                raise WorkerFailure(
                    code,
                    f"quick Vina infrastructure failed while evaluating {request_id}: "
                    f"{reason}",
                    recoverable=recoverable,
                )
            if recoverable:
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "candidate-local unsupported chemistry cannot be recoverable",
                    recoverable=False,
                )
            evaluations.append(
                {
                    "request_id": request_id,
                    "molecule_id": item["molecule_id"],
                    "microstate_id": item["microstate_id"],
                    "status": "failed",
                    "code": code,
                    "reason": reason,
                    "recoverable": False,
                    "seed": request.seed,
                    "box_center": item["box_center"],
                    "box_size": item["box_size"],
                    "box_source": item["box_source"],
                    "coordinate_frame": item["coordinate_frame"],
                    "docking_box_receipt_sha256": parsed["box_receipt"].sha256,
                }
            )
            continue
        success = success_by_id.get(request_id)
        if success is None:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "shared Vina did not exactly cover every quick request",
                recoverable=False,
            )
        pose = _reference(success.get("pose"), "quick Vina pose")
        inner_evidence = _reference(success.get("evidence"), "inner Vina evidence")
        score = success.get("vina_score")
        semantics = success.get("vina_score_semantics")
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or semantics != vina.SCORE_SEMANTICS
            or success.get("box_center") != item["box_center"]
            or success.get("box_size") != item["box_size"]
        ):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "shared Vina success changed score semantics or frozen box",
                recoverable=False,
            )
        evidence = store.put_json(
            {
                "schema_version": "1.0",
                "kind": "protbind.tool-evidence",
                "tool": "vina",
                "tool_version": vina.VINA_VERSION,
                "purpose": QUICK_VINA_PURPOSE,
                "request_id": request_id,
                "molecule_id": item["molecule_id"],
                "microstate_id": item["microstate_id"],
                "seed": request.seed,
                "metrics": {
                    "score": float(score),
                    "score_semantics": semantics,
                    "box_center": item["box_center"],
                    "box_size": item["box_size"],
                    "box_source": item["box_source"],
                    "coordinate_frame": item["coordinate_frame"],
                    "docking_box_receipt_sha256": parsed["box_receipt"].sha256,
                    "purpose": QUICK_VINA_PURPOSE,
                    "formal_charge": item["formal_charge"],
                    **profile,
                },
                "inputs": {
                    "quick_vina_input": request.input.to_dict(),
                    "receptor": parsed["receptor"].to_dict(),
                    "docking_box_receipt": parsed["box_receipt"].to_dict(),
                    "pose": pose.to_dict(),
                    "inner_vina_evidence": inner_evidence.to_dict(),
                },
            },
            producer="vina-quick.evidence",
            producer_version=PROFILE_VERSION,
            source=request.input.artifact_id,
        )
        evaluations.append(
            {
                "request_id": request_id,
                "molecule_id": item["molecule_id"],
                "microstate_id": item["microstate_id"],
                "status": "completed",
                "score": float(score),
                "score_semantics": semantics,
                "pose": pose.to_dict(),
                "evidence": evidence.to_dict(),
                "seed": request.seed,
                "box_center": item["box_center"],
                "box_size": item["box_size"],
                "box_source": item["box_source"],
                "coordinate_frame": item["coordinate_frame"],
                "docking_box_receipt_sha256": parsed["box_receipt"].sha256,
            }
        )
    if len(evaluations) != len(parsed["requests"]):
        raise WorkerFailure(
            "OUTPUT_INVALID", "quick Vina coverage count is incomplete", recoverable=False
        )
    success_count = sum(item["status"] == "completed" for item in evaluations)
    failure_count = len(evaluations) - success_count
    inner_metadata = _reference(bundle_value.get("run_metadata"), "inner run metadata")
    runtime_metadata = store.put_json(
        {
            "schema_version": "1.0",
            "purpose": QUICK_VINA_PURPOSE,
            "environment_lock": parsed["environment_lock"].to_dict(),
            "docking_box_receipt": parsed["box_receipt"].to_dict(),
            "inner_vina_run_metadata": inner_metadata.to_dict(),
            "execution": {
                "device": "cpu",
                "cpu_threads": 1,
                "purpose": QUICK_VINA_PURPOSE,
                "seed": request.seed,
                "scoring": "vina",
                "exhaustiveness": profile["exhaustiveness"],
                "num_modes": profile["num_modes"],
                "input_candidate_count": len(evaluations),
                "successful_candidate_count": success_count,
                "failed_candidate_count": failure_count,
                "box_source": parsed["box_receipt_value"]["source_kind"],
                "coordinate_frame": parsed["box_receipt_value"][
                    "coordinate_frame"
                ],
                "docking_box_receipt_sha256": parsed["box_receipt"].sha256,
                "site_derivation_verified": parsed["box_receipt_value"][
                    "validation"
                ]["site_derivation_verified"],
                "site_scientific_interpretation": parsed["box_receipt_value"][
                    "validation"
                ]["scientific_interpretation"],
            },
        },
        producer="vina-quick.run-metadata",
        producer_version=PROFILE_VERSION,
        source=request.input.artifact_id,
    )
    batch = store.put_json(
        {
            "schema_version": "1.0",
            "kind": QUICK_VINA_BATCH_KIND,
            "purpose": QUICK_VINA_PURPOSE,
            "input": request.input.to_dict(),
            "selection_preparation_sha256": parsed["value"][
                "selection_preparation_sha256"
            ],
            "receptor": parsed["receptor"].to_dict(),
            "docking_box_receipt": parsed["box_receipt"].to_dict(),
            "request_count": len(evaluations),
            "success_count": success_count,
            "failure_count": failure_count,
            "evaluations": evaluations,
            "run_metadata": runtime_metadata.to_dict(),
            "inner_docking_bundle": inner_bundle.to_dict(),
        },
        producer="vina-quick.evaluation-batch",
        producer_version=PROFILE_VERSION,
        source=request.input.artifact_id,
    )
    outputs = _output_closure(
        store,
        batch,
        allowed_inputs={
            request.input,
            parsed["receptor"],
            parsed["environment_lock"],
            parsed["box_receipt"],
        },
    )
    return WorkerResponse(
        job_id=request.job_id,
        engine=request.engine,
        outputs=outputs,
        provenance=request.provenance,
        timings_seconds={
            **inner_response.timings_seconds,
            "quick_profile_total": time.perf_counter() - started,
        },
        peak_vram_bytes=None,
        warnings=(
            *inner_response.warnings,
            "quick Vina is selection-pruning evidence only; DOCKED must rerun Vina",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(serve_worker(ENGINE, _handler))
