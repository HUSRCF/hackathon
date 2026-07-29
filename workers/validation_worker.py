#!/usr/bin/env python3
"""Offline, provenance-bound validation worker for docked poses.

PoseBusters validation of the Vina pose is mandatory.  A reference pose,
cofold prediction, sPyRMSD, ProLIF, and OpenMM are independent optional
evidence groups; failure of a cofolder must never reject a valid docked pose.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protbind_agent.artifacts import (  # noqa: E402
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from protbind_agent.interaction_fingerprint import compare_prolif_paths  # noqa: E402
from protbind_agent.models import ArtifactRef  # noqa: E402
from protbind_agent.worker_protocol import WorkerRequest, WorkerResponse  # noqa: E402
from protbind_agent.worker_sdk import WorkerFailure, serve_worker  # noqa: E402

ENGINE = "protbind-validation"
TOOLCHAIN_KIND = "protbind.validation-toolchain-manifest"
INPUT_BATCH_KIND = "protbind.validation-input-batch"
POSEBUSTERS_CONFIG = "dock"
POSEBUSTERS_CONFIGS = ("dock", "redock")
MAX_CANDIDATES = 16
MAX_PACKAGE_FILES = 20_000
MAX_PACKAGE_BYTES = 2 * 1024**3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_TOOLS = {
    "posebusters": ("posebusters", "posebusters"),
    "spyrmsd": ("spyrmsd", "spyrmsd"),
    "prolif": ("prolif", "prolif"),
    "openmm": ("openmm", "openmm"),
}
_LIGAND_SUFFIXES = {
    "chemical/x-mdl-sdfile": ".sdf",
    "chemical/x-sdf": ".sdf",
    "chemical/x-mdl-molfile": ".mol",
    "chemical/x-mol2": ".mol2",
    "chemical/x-pdb": ".pdb",
}
_RECEPTOR_SUFFIXES = {"chemical/x-pdb": ".pdb"}
_SYSTEM_SUFFIXES = {
    "application/xml": ".xml",
    "text/xml": ".xml",
    "application/vnd.openmm.system+xml": ".xml",
}
def protbind_runtime_sha256() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src" / "protbind_agent"
    sources = [
        (str(path.relative_to(repository_root)), sha256_file(path))
        for path in sorted(source_root.rglob("*.py"))
    ]
    if not sources:
        raise RuntimeError("ProtBind runtime source manifest is empty")
    return sha256_bytes(canonical_json_bytes(sources))


def composite_code_sha256() -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "adapter_sha256": sha256_file(Path(__file__)),
                "protbind_runtime_sha256": protbind_runtime_sha256(),
            }
        )
    )


def package_tree_sha256(root: Path) -> tuple[str, int, int]:
    """Hash an imported package tree, including native binaries and data assets."""

    root = root.resolve()
    entries: list[tuple[str, str]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        entries.append((path.relative_to(root).as_posix(), sha256_file(path)))
        total_bytes += path.stat().st_size
        if len(entries) > MAX_PACKAGE_FILES or total_bytes > MAX_PACKAGE_BYTES:
            raise WorkerFailure(
                "TOOLCHAIN_INVALID",
                "a validation package exceeds the attestation size limit",
                recoverable=False,
            )
    if not entries:
        raise WorkerFailure(
            "TOOLCHAIN_INVALID",
            "a validation package has no attestable files",
            recoverable=False,
        )
    return sha256_bytes(canonical_json_bytes(entries)), len(entries), total_bytes


def _reference(value: Any, name: str) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", f"{name} is not an ArtifactRef", recoverable=True
        ) from exc


def _json_object(store: ArtifactStore, reference: ArtifactRef, name: str) -> dict[str, Any]:
    try:
        value = store.read_json(reference)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", f"{name} is not valid JSON", recoverable=True
        ) from exc
    if not isinstance(value, dict):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", f"{name} must be a JSON object", recoverable=True
        )
    return value


def _stage_inputs(
    request: WorkerRequest, store: ArtifactStore
) -> tuple[
    dict[str, Any],
    ArtifactRef,
    dict[str, Any],
    ArtifactRef,
    dict[str, Any],
    dict[str, Any],
]:
    envelope = _json_object(store, request.input, "validation stage envelope")
    if (
        envelope.get("schema_version") not in {"1.0", "2.0"}
        or envelope.get("kind") != "protbind.stage-input"
        or envelope.get("stage") != "VALIDATED"
    ):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "validation requires a protbind.stage-input v1/v2 VALIDATED envelope",
            recoverable=True,
        )
    previous = envelope.get("previous")
    if not isinstance(previous, dict) or previous.get("stage") != "DOCKED":
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "validation stage input must be bound to DOCKED",
            recoverable=True,
        )
    scientific_outputs = previous.get("scientific_outputs")
    if not isinstance(scientific_outputs, list) or not scientific_outputs:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", "DOCKED has no scientific output", recoverable=True
        )
    docking_reference = _reference(scientific_outputs[0], "DOCKED bundle")
    docking = _json_object(store, docking_reference, "DOCKED bundle")
    if (
        docking.get("schema_version") not in {"1.0", "2.0"}
        or docking.get("kind") != "protbind.docking-bundle"
    ):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "the previous output is not a protbind.docking-bundle v1/v2 artifact",
            recoverable=True,
        )
    supporting = envelope.get("supporting_artifacts")
    if not isinstance(supporting, dict):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", "supporting_artifacts must be an object", recoverable=True
        )
    toolchain_reference = _reference(
        supporting.get("support_validation_toolchain"),
        "support_validation_toolchain",
    )
    batch_reference = _reference(
        supporting.get("support_validation_batch"), "support_validation_batch"
    )
    toolchain = _json_object(store, toolchain_reference, "validation toolchain")
    batch = _json_object(store, batch_reference, "validation input batch")
    case_id = envelope.get("case_id")
    if case_id is None and envelope.get("schema_version") == "1.0":
        # Legacy envelopes carried the whole case artifact.  Keep them
        # readable without using case contents to authorize a reference pose.
        case_reference = _reference(envelope.get("case"), "research case")
        legacy_case = _json_object(store, case_reference, "research case")
        case_id = legacy_case.get("case_id", "legacy-case")
    if not isinstance(case_id, str) or not case_id.strip():
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "validation stage input requires a non-empty case_id",
            recoverable=True,
        )
    return (
        docking,
        docking_reference,
        toolchain,
        toolchain_reference,
        batch,
        {"case_id": case_id, "supporting": supporting},
    )


def _validate_parameters(parameters: dict[str, Any]) -> tuple[int, float]:
    allowed = {"openmm_max_iterations", "openmm_max_displacement_angstrom"}
    unknown = set(parameters) - allowed
    if unknown:
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "unsupported validation parameters: " + ", ".join(sorted(unknown)),
            recoverable=False,
        )
    iterations = parameters.get("openmm_max_iterations", 200)
    displacement = parameters.get("openmm_max_displacement_angstrom", 1.5)
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or not 1 <= iterations <= 1000
    ):
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "openmm_max_iterations must be an integer in [1, 1000]",
            recoverable=False,
        )
    if (
        not isinstance(displacement, int | float)
        or isinstance(displacement, bool)
        or not math.isfinite(float(displacement))
        or not 0 < float(displacement) <= 10
    ):
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "openmm_max_displacement_angstrom must be finite and in (0, 10]",
            recoverable=False,
        )
    return iterations, float(displacement)


def _validate_toolchain(
    request: WorkerRequest,
    store: ArtifactStore,
    reference: ArtifactRef,
    value: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    if value.get("schema_version") != "1.0" or value.get("kind") != TOOLCHAIN_KIND:
        raise WorkerFailure(
            "TOOLCHAIN_INVALID",
            f"validation toolchain must satisfy {TOOLCHAIN_KIND} v1.0",
            recoverable=False,
        )
    if request.provenance.weight_sha256 != reference.sha256 or (
        request.provenance.model_revision != f"validation-toolchain:{reference.sha256}"
    ):
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            "validation toolchain artifact does not match request provenance",
            recoverable=False,
        )
    if request.provenance.code_sha256 != composite_code_sha256():
        raise WorkerFailure(
            "CODE_HASH_MISMATCH",
            "validation adapter/runtime code does not match request provenance",
            recoverable=False,
        )
    fixture = value.get("test_fixture", False)
    if not isinstance(fixture, bool):
        raise WorkerFailure(
            "TOOLCHAIN_INVALID", "test_fixture must be boolean", recoverable=False
        )
    test_runtime = os.environ.get("PROTBIND_TEST_RUNTIME") == "1"
    if fixture != test_runtime:
        raise WorkerFailure(
            "TOOLCHAIN_INVALID",
            "fixture toolchains require the direct protocol-test runtime, and production "
            "toolchains forbid it",
            recoverable=False,
        )
    configured_pb = value.get("posebusters_configs", value.get("posebusters_config"))
    if configured_pb == POSEBUSTERS_CONFIG:
        configured_pb = [POSEBUSTERS_CONFIG]
    if (
        not isinstance(configured_pb, list)
        or not configured_pb
        or any(item not in POSEBUSTERS_CONFIGS for item in configured_pb)
        or len(configured_pb) != len(set(configured_pb))
    ):
        raise WorkerFailure(
            "TOOLCHAIN_INVALID",
            "PoseBusters configurations must be a unique subset of dock/redock",
            recoverable=False,
        )
    tools = value.get("tools")
    if not isinstance(tools, dict) or "posebusters" not in tools:
        raise WorkerFailure(
            "TOOLCHAIN_INVALID",
            "the toolchain must pin PoseBusters",
            recoverable=False,
        )
    if set(tools) - set(_TOOLS):
        raise WorkerFailure(
            "TOOLCHAIN_INVALID", "the toolchain names an unsupported tool", recoverable=False
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, entry in tools.items():
        if not isinstance(entry, dict) or set(entry) != {
            "version",
            "package_source_sha256",
        }:
            raise WorkerFailure(
                "TOOLCHAIN_INVALID",
                f"{name} pin requires only version and package_source_sha256",
                recoverable=False,
            )
        version = entry.get("version")
        source_sha = entry.get("package_source_sha256")
        if not isinstance(version, str) or not version.strip() or not isinstance(
            source_sha, str
        ) or not _SHA256.fullmatch(source_sha):
            raise WorkerFailure(
                "TOOLCHAIN_INVALID", f"{name} has an invalid runtime pin", recoverable=False
            )
        normalized[name] = {
            "version": version,
            "package_source_sha256": source_sha,
        }
    assets = value.get("assets", [])
    if not isinstance(assets, list) or len(assets) > 64:
        raise WorkerFailure(
            "TOOLCHAIN_INVALID",
            "toolchain assets must be an array of at most 64 refs",
            recoverable=False,
        )
    for index, asset in enumerate(assets):
        store.resolve(_reference(asset, f"toolchain asset {index}"))
    return normalized, fixture


def _attest_tool(name: str, pin: dict[str, Any]) -> dict[str, Any]:
    distribution_name, module_name = _TOOLS[name]
    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE",
            f"pinned {name} is not installed",
            recoverable=True,
        ) from exc
    if distribution.version != pin["version"]:
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            f"installed {name} version differs from the pinned toolchain",
            recoverable=False,
        )
    spec = importlib.util.find_spec(module_name)
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if len(locations) != 1:
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            f"installed {name} module root is unavailable or ambiguous",
            recoverable=False,
        )
    package_root = Path(locations[0]).resolve()
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not package_root.is_relative_to(distribution_root):
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            f"imported {name} is shadowed outside its pinned distribution",
            recoverable=False,
        )
    source_sha, file_count, total_bytes = package_tree_sha256(package_root)
    if source_sha != pin["package_source_sha256"]:
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            f"installed {name} files differ from the pinned source manifest",
            recoverable=False,
        )
    return {
        "distribution": distribution_name,
        "module": module_name,
        "version": distribution.version,
        "package_source_sha256": source_sha,
        "package_file_count": file_count,
        "package_size_bytes": total_bytes,
    }


def _optional_attestation(
    name: str, pins: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    if name not in pins:
        return None, f"{name}: not pinned in the validation toolchain"
    # Once an optional tool is explicitly pinned, an absent or mismatched
    # runtime is an integrity failure, not ordinary capability absence.
    return _attest_tool(name, pins[name]), None


def _same_reference(left: ArtifactRef, right: ArtifactRef) -> bool:
    return left == right


def _validate_batch(
    store: ArtifactStore,
    batch: dict[str, Any],
    docking: dict[str, Any],
    docking_reference: ArtifactRef,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if batch.get("schema_version") not in {"1.0", "2.0"} or batch.get("kind") != INPUT_BATCH_KIND:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            f"validation input batch must satisfy {INPUT_BATCH_KIND} v1/v2",
            recoverable=True,
        )
    frozen_docking = _reference(batch.get("docking_bundle"), "batch docking_bundle")
    if not _same_reference(frozen_docking, docking_reference):
        raise WorkerFailure(
            "LINEAGE_MISMATCH",
            "validation batch is not bound to the exact DOCKED bundle",
            recoverable=False,
        )
    upstream = docking.get("candidates")
    prepared = batch.get("candidates")
    if (
        not isinstance(upstream, list)
        or not isinstance(prepared, list)
        or not 1 <= len(upstream) <= MAX_CANDIDATES
        or len(prepared) != len(upstream)
    ):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "validation requires matching non-empty candidate arrays of at most sixteen",
            recoverable=True,
        )
    prepared_by_id: dict[str, dict[str, Any]] = {}
    for value in prepared:
        if not isinstance(value, dict) or not isinstance(value.get("candidate_id"), str):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "prepared validation candidates require candidate_id",
                recoverable=True,
            )
        candidate_id = value["candidate_id"]
        if candidate_id in prepared_by_id:
            raise WorkerFailure(
                "LINEAGE_MISMATCH", "duplicate validation candidate_id", recoverable=False
            )
        prepared_by_id[candidate_id] = value
    resolved: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for upstream_value in upstream:
        if not isinstance(upstream_value, dict):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", "DOCKED candidate is not an object", recoverable=True
            )
        candidate_id = upstream_value.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in prepared_by_id:
            raise WorkerFailure(
                "LINEAGE_MISMATCH",
                "validation batch does not cover the exact DOCKED candidates",
                recoverable=False,
            )
        value = prepared_by_id[candidate_id]
        for field in ("molecule_id", "microstate_id"):
            if value.get(field) != upstream_value.get(field):
                raise WorkerFailure(
                    "LINEAGE_MISMATCH",
                    f"validation candidate has a mismatched {field}",
                    recoverable=False,
                )
        docked_pose = _reference(value.get("docked_pose"), "prepared docked_pose")
        expected_docked = _reference(upstream_value.get("pose"), "DOCKED pose")
        store.resolve(docked_pose)
        if not _same_reference(docked_pose, expected_docked):
            raise WorkerFailure(
                "LINEAGE_MISMATCH",
                "validation candidate is not bound to its docked pose",
                recoverable=False,
            )
        expected_cofold_value = upstream_value.get("cofold_structure")
        prepared_cofold_value = value.get("cofold_pose")
        if (expected_cofold_value is None) != (prepared_cofold_value is None):
            raise WorkerFailure(
                "LINEAGE_MISMATCH",
                "optional cofold pose presence differs between docking and validation",
                recoverable=False,
            )
        if expected_cofold_value is not None:
            cofold_pose = _reference(prepared_cofold_value, "prepared cofold_pose")
            expected_cofold = _reference(
                expected_cofold_value, "DOCKED cofold_structure"
            )
            store.resolve(cofold_pose)
            if not _same_reference(cofold_pose, expected_cofold):
                raise WorkerFailure(
                    "LINEAGE_MISMATCH",
                    "validation candidate is not bound to its optional cofold pose",
                    recoverable=False,
                )
        posebusters = value.get("posebusters")
        if not isinstance(posebusters, dict):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "each candidate requires prepared PoseBusters inputs",
                recoverable=True,
            )
        required_pb = {"docked_ligand", "docked_receptor"}
        optional_pb = {
            "cofold_ligand",
            "cofold_receptor",
            "reference_ligand",
        }
        if not required_pb.issubset(posebusters) or set(posebusters) - (
            required_pb | optional_pb
        ):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "PoseBusters inputs require docked ligand/receptor and only optional "
                "cofold/reference inputs",
                recoverable=True,
            )
        if ("cofold_ligand" in posebusters) != ("cofold_receptor" in posebusters):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "optional cofold PoseBusters inputs must be supplied as a pair",
                recoverable=True,
            )
        for name in posebusters:
            store.resolve(_reference(posebusters[name], f"PoseBusters {name}"))
        resolved.append((upstream_value, value))
    if set(prepared_by_id) != {
        str(value.get("candidate_id")) for value in upstream if isinstance(value, dict)
    }:
        raise WorkerFailure(
            "LINEAGE_MISMATCH",
            "validation batch contains candidates absent from DOCKED",
            recoverable=False,
        )
    return resolved


def _materialize(
    store: ArtifactStore,
    value: Any,
    name: str,
    directory: Path,
    suffixes: dict[str, str],
    *,
    max_bytes: int,
) -> tuple[ArtifactRef, Path]:
    reference = _reference(value, name)
    if reference.media_type not in suffixes:
        raise ValueError(f"{name} has unsupported media type {reference.media_type}")
    if reference.size_bytes <= 0 or reference.size_bytes > max_bytes:
        raise ValueError(f"{name} exceeds its bounded artifact size")
    source = store.resolve(reference)
    destination = directory / f"{name.replace('_', '-')}{suffixes[reference.media_type]}"
    destination.write_bytes(source.read_bytes())
    return reference, destination


def _boolean_report(
    frame: Any,
    name: str,
    *,
    exclude_reference_rmsd: bool = False,
) -> tuple[bool, dict[str, bool]]:
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
        raise ValueError(f"PoseBusters returned an invalid {name} report shape")
    try:
        items = list(frame.iloc[0].items())
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(f"PoseBusters returned an unreadable {name} report") from exc
    checks: dict[str, bool] = {}
    for column, result in items:
        if isinstance(column, tuple):
            check = "::".join(str(part) for part in column if str(part))
        else:
            check = str(column)
        if not check or check in checks or not (
            isinstance(result, bool) or type(result).__name__ == "bool_"
        ):
            raise ValueError(f"PoseBusters {name} report is not a unique boolean matrix")
        checks[check] = bool(result)
    validity_checks = {
        check: result
        for check, result in checks.items()
        if not (exclude_reference_rmsd and "rmsd" in check.lower())
    }
    if not validity_checks:
        raise ValueError(f"PoseBusters {name} report has no geometry/chemistry checks")
    return all(validity_checks.values()), checks


def _run_posebusters(
    store: ArtifactStore,
    prepared: dict[str, Any],
    directory: Path,
    *,
    reference_pose: ArtifactRef | None,
) -> tuple[dict[str, Any], dict[str, ArtifactRef]]:
    pb = prepared["posebusters"]
    operational: dict[str, ArtifactRef] = {}
    paths: dict[str, Path] = {}
    try:
        ligand_names = ["docked_ligand"]
        if "cofold_ligand" in pb:
            ligand_names.append("cofold_ligand")
        for name in ligand_names:
            operational[name], paths[name] = _materialize(
                store,
                pb[name],
                name,
                directory,
                _LIGAND_SUFFIXES,
                max_bytes=10 * 1024**2,
            )
        receptor_names = ["docked_receptor"]
        if "cofold_receptor" in pb:
            receptor_names.append("cofold_receptor")
        for name in receptor_names:
            operational[name], paths[name] = _materialize(
                store,
                pb[name],
                name,
                directory,
                _RECEPTOR_SUFFIXES,
                max_bytes=50 * 1024**2,
            )
        config = "redock" if reference_pose is not None else "dock"
        if reference_pose is not None:
            supplied_reference = pb.get("reference_ligand")
            if supplied_reference is not None and not _same_reference(
                _reference(supplied_reference, "PoseBusters reference_ligand"),
                reference_pose,
            ):
                raise ValueError("PoseBusters reference differs from authorized pose")
            operational["reference_ligand"], paths["reference_ligand"] = _materialize(
                store,
                reference_pose.to_dict(),
                "reference_ligand",
                directory,
                _LIGAND_SUFFIXES,
                max_bytes=10 * 1024**2,
            )
        module = importlib.import_module("posebusters")
        buster = module.PoseBusters(
            config=config,
            top_n=1,
            max_workers=0,
            chunk_size=1,
        )
        docked_kwargs: dict[str, Any] = {
            "mol_pred": str(paths["docked_ligand"]),
            "mol_cond": str(paths["docked_receptor"]),
            "full_report": False,
        }
        if reference_pose is not None:
            docked_kwargs["mol_true"] = str(paths["reference_ligand"])
        docked_frame = buster.bust(**docked_kwargs)
        docked_valid, docked_checks = _boolean_report(
            docked_frame,
            "docked",
            exclude_reference_rmsd=reference_pose is not None,
        )
        cofold_valid: bool | None = None
        cofold_checks: dict[str, bool] | None = None
        if "cofold_ligand" in paths:
            cofold_frame = buster.bust(
                mol_pred=str(paths["cofold_ligand"]),
                mol_cond=str(paths["cofold_receptor"]),
                full_report=False,
            )
            cofold_valid, cofold_checks = _boolean_report(cofold_frame, "cofold")
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise WorkerFailure(
            "POSEBUSTERS_FAILED",
            f"PoseBusters could not produce a boolean dock report: {type(exc).__name__}",
            recoverable=True,
        ) from exc
    return (
        {
            "valid": docked_valid,
            "docked_valid": docked_valid,
            "cofold_valid": cofold_valid,
            "docked_checks": docked_checks,
            "cofold_checks": cofold_checks,
            "config": config,
        },
        operational,
    )


def _allowed_reference_hashes(context: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    supporting = context["supporting"]
    support = supporting.get("support_reference_pose")
    if support is not None:
        allowed.add(_reference(support, "support_reference_pose").sha256)
    return allowed


def _authorized_reference_pose(
    store: ArtifactStore,
    prepared: dict[str, Any],
    context: dict[str, Any],
) -> ArtifactRef | None:
    value = prepared.get("reference_pose")
    if value is None:
        return None
    reference = _reference(value, "reference_pose")
    store.resolve(reference)
    if reference.sha256 not in _allowed_reference_hashes(context):
        raise WorkerFailure(
            "LINEAGE_MISMATCH",
            "reference pose requires an explicit support_reference_pose artifact",
            recoverable=False,
        )
    return reference


def _receipt_attests(
    store: ArtifactStore,
    value: Any,
    *,
    kind: str,
    expected: dict[str, ArtifactRef],
) -> bool:
    """Return true only for a non-fixture, exact-reference preparation receipt."""

    if value is None:
        return False
    try:
        reference = _reference(value, f"{kind} receipt")
        receipt = _json_object(store, reference, f"{kind} receipt")
    except WorkerFailure:
        return False
    if (
        receipt.get("schema_version") != "2.0"
        or receipt.get("kind") != kind
        or receipt.get("test_fixture") is not False
    ):
        return False
    checks = receipt.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(not isinstance(result, bool) for result in checks.values())
        or not all(checks.values())
    ):
        return False
    for name, expected_reference in expected.items():
        try:
            actual = _reference(receipt.get(name), f"{kind}.{name}")
            store.resolve(actual)
        except (WorkerFailure, OSError):
            return False
        if not _same_reference(actual, expected_reference):
            return False
    return True


def _preparation_attested(
    store: ArtifactStore,
    docking: dict[str, Any],
    upstream: dict[str, Any],
) -> bool:
    pose_fields = {
        "input_ligand": upstream.get("ligand_sdf"),
        "pose_sdf": upstream.get("pose_sdf", upstream.get("pose")),
        "pose_pdbqt": upstream.get("pose_pdbqt"),
        "all_modes_sdf": upstream.get("all_modes_sdf"),
        "all_modes_pdbqt": upstream.get(
            "all_modes_pdbqt", upstream.get("all_modes")
        ),
    }
    receptor_fields = {
        "receptor": docking.get("receptor"),
        "receptor_preparation_input": docking.get("receptor_preparation_input"),
        "prepared_receptor": docking.get("prepared_receptor"),
    }
    if any(value is None for value in (*pose_fields.values(), *receptor_fields.values())):
        return False
    try:
        pose_expected = {
            name: _reference(value, f"DOCKED {name}")
            for name, value in pose_fields.items()
        }
        receptor_expected = {
            name: _reference(value, f"DOCKED {name}")
            for name, value in receptor_fields.items()
        }
    except WorkerFailure:
        return False
    return _receipt_attests(
        store,
        upstream.get("pose_extraction_receipt"),
        kind="protbind.pose-extraction-receipt",
        expected=pose_expected,
    ) and _receipt_attests(
        store,
        docking.get("receptor_preparation_receipt"),
        kind="protbind.receptor-preparation-receipt",
        expected=receptor_expected,
    )


def _run_spyrmsd(
    store: ArtifactStore,
    upstream: dict[str, Any],
    prepared: dict[str, Any],
    context: dict[str, Any],
    directory: Path,
) -> tuple[float | None, dict[str, ArtifactRef] | None, str | None]:
    reference_value = prepared.get("reference_pose")
    inputs = prepared.get("spyrmsd")
    if reference_value is None or inputs is None:
        return None, None, "spyrmsd: no authorized reference pose and prepared comparison"
    if not isinstance(inputs, dict) or set(inputs) != {
        "reference_ligand",
        "predicted_ligand",
    }:
        return None, None, "spyrmsd: prepared inputs are incomplete"
    reference_pose = _reference(reference_value, "reference_pose")
    if reference_pose.sha256 not in _allowed_reference_hashes(context):
        raise WorkerFailure(
            "LINEAGE_MISMATCH",
            "sPyRMSD reference pose is not authorized by the case/support inputs",
            recoverable=False,
        )
    try:
        reference_ligand, reference_path = _materialize(
            store,
            inputs["reference_ligand"],
            "spyrmsd_reference",
            directory,
            _LIGAND_SUFFIXES,
            max_bytes=10 * 1024**2,
        )
        predicted_ligand, predicted_path = _materialize(
            store,
            inputs["predicted_ligand"],
            "spyrmsd_prediction",
            directory,
            _LIGAND_SUFFIXES,
            max_bytes=10 * 1024**2,
        )
        io_module = importlib.import_module("spyrmsd.io")
        rmsd_module = importlib.import_module("spyrmsd.rmsd")
        loader = getattr(io_module, "loadmol", None) or getattr(io_module, "load", None)
        if not callable(loader):
            raise AttributeError("sPyRMSD molecule loader is unavailable")
        reference_molecule = loader(str(reference_path))
        predicted_molecule = loader(str(predicted_path))
        values = rmsd_module.rmsdwrapper(
            reference_molecule,
            predicted_molecule,
            symmetry=True,
            center=False,
            minimize=False,
            strip=True,
        )
        value = values[0] if isinstance(values, list | tuple) else values
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("sPyRMSD did not return one finite non-negative RMSD")
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return None, None, f"spyrmsd: comparison failed ({type(exc).__name__})"
    return (
        float(value),
        {
            "reference_pose": reference_pose,
            "predicted_pose": _reference(upstream["pose"], "DOCKED pose"),
            "reference_ligand": reference_ligand,
            "predicted_ligand": predicted_ligand,
        },
        None,
    )


def _run_prolif(
    store: ArtifactStore,
    prepared: dict[str, Any],
    directory: Path,
) -> tuple[float | None, dict[str, ArtifactRef] | None, dict[str, Any] | None, str | None]:
    inputs = prepared.get("prolif")
    required = {"docked_ligand", "docked_receptor"}
    allowed = required | {
        "cofold_ligand",
        "cofold_receptor",
        "reference_ligand",
        "reference_receptor",
    }
    if (
        not isinstance(inputs, dict)
        or not required.issubset(inputs)
        or set(inputs) - allowed
    ):
        return None, None, None, "prolif: prepared docked inputs are unavailable"
    for prefix in ("cofold", "reference"):
        if (f"{prefix}_ligand" in inputs) != (f"{prefix}_receptor" in inputs):
            return None, None, None, f"prolif: {prefix} inputs must be supplied as a pair"
    try:
        refs: dict[str, ArtifactRef] = {}
        paths: dict[str, Path] = {}
        ligand_names = [name for name in inputs if name.endswith("_ligand")]
        receptor_names = [name for name in inputs if name.endswith("_receptor")]
        for name in ligand_names:
            refs[name], paths[name] = _materialize(
                store,
                inputs[name],
                f"prolif_{name}",
                directory,
                _LIGAND_SUFFIXES,
                max_bytes=10 * 1024**2,
            )
        for name in receptor_names:
            refs[name], paths[name] = _materialize(
                store,
                inputs[name],
                f"prolif_{name}",
                directory,
                _RECEPTOR_SUFFIXES,
                max_bytes=50 * 1024**2,
            )
        comparison_name: str | None = None
        for prefix in ("reference", "cofold"):
            if f"{prefix}_ligand" not in paths:
                continue
            comparison_name = prefix
            break
        metrics = compare_prolif_paths(
            docked_ligand_path=paths["docked_ligand"],
            docked_receptor_path=paths["docked_receptor"],
            comparison_ligand_path=(
                paths[f"{comparison_name}_ligand"] if comparison_name is not None else None
            ),
            comparison_receptor_path=(
                paths[f"{comparison_name}_receptor"] if comparison_name is not None else None
            ),
            comparison_name=comparison_name,
        )
        similarity = metrics["ifp_similarity"]
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return None, None, None, f"prolif: fingerprint comparison failed ({type(exc).__name__})"
    return similarity, refs, metrics, None


def _run_openmm(
    store: ArtifactStore,
    prepared: dict[str, Any],
    directory: Path,
    *,
    max_iterations: int,
    max_displacement_angstrom: float,
) -> tuple[
    bool | None,
    bool | None,
    dict[str, ArtifactRef] | None,
    dict[str, Any] | None,
    str | None,
]:
    inputs = prepared.get("openmm")
    if not isinstance(inputs, dict):
        return None, None, None, None, "openmm: a prepared System and coordinates were not supplied"
    if set(inputs) != {"system", "coordinates", "platform"}:
        return None, None, None, None, "openmm: prepared inputs are incomplete"
    platform_name = inputs.get("platform")
    if platform_name not in {"CPU", "HIP"}:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "OpenMM platform must be explicitly CPU or HIP",
            recoverable=True,
        )
    # Resolve both declared artifacts before attempting to deserialize either
    # one so even an explicit parameterization failure remains bound to the
    # complete frozen input pair.
    refs: dict[str, ArtifactRef] = {
        "system": _reference(inputs["system"], "openmm system"),
        "coordinates": _reference(inputs["coordinates"], "openmm coordinates"),
    }
    for reference in refs.values():
        store.resolve(reference)
    try:
        _, system_path = _materialize(
            store,
            refs["system"].to_dict(),
            "openmm_system",
            directory,
            _SYSTEM_SUFFIXES,
            max_bytes=100 * 1024**2,
        )
        _, coordinate_path = _materialize(
            store,
            refs["coordinates"].to_dict(),
            "openmm_coordinates",
            directory,
            _RECEPTOR_SUFFIXES,
            max_bytes=100 * 1024**2,
        )
        import openmm
        from openmm import app, unit

        system = openmm.XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))
        coordinates = app.PDBFile(str(coordinate_path))
        if system.getNumParticles() != len(coordinates.positions):
            raise ValueError("OpenMM System particle count differs from coordinates")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        metrics = {"system_deserialized": False, "failure": type(exc).__name__}
        return (
            False,
            None,
            refs,
            metrics,
            f"hard:openmm System/coordinate validation failed ({type(exc).__name__})",
        )
    available = {
        openmm.Platform.getPlatform(index).getName()
        for index in range(openmm.Platform.getNumPlatforms())
    }
    if platform_name not in available:
        return (
            True,
            None,
            refs,
            {
                "system_deserialized": True,
                "particle_count_matches": True,
                "requested_platform": platform_name,
            },
            f"openmm: requested {platform_name} platform is unavailable",
        )
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = None
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        platform_properties = (
            {"DeviceIndex": "0", "Precision": "mixed"}
            if platform_name == "HIP"
            else {"Threads": "1"}
        )
        context = openmm.Context(system, integrator, platform, platform_properties)
        context.setPositions(coordinates.positions)
        initial_state = context.getState(getEnergy=True, getPositions=True)
        initial_energy = float(
            initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        initial_positions = initial_state.getPositions(asNumpy=True).value_in_unit(
            unit.angstrom
        )
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=max_iterations)
        final_state = context.getState(getEnergy=True, getPositions=True)
        final_energy = float(
            final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        final_positions = final_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        displacement_squared = ((final_positions - initial_positions) ** 2).sum(axis=1)
        max_displacement = float(displacement_squared.max() ** 0.5)
        finite_result = (
            math.isfinite(initial_energy)
            and math.isfinite(final_energy)
            and math.isfinite(max_displacement)
        )
        geometry_valid = finite_result and max_displacement <= max_displacement_angstrom
        metrics: dict[str, Any] = {
            "system_deserialized": True,
            "particle_count_matches": True,
            "minimization_completed": True,
            "geometry_gate_passed": geometry_valid,
            "platform": platform_name,
            "platform_properties": platform_properties,
            "max_iterations": max_iterations,
            "geometry_displacement_threshold_angstrom": max_displacement_angstrom,
            "semantic_scope": (
                "local minimization geometry gate; not molecular-dynamics "
                "stability and not proof of parameterization provenance"
            ),
        }
        if finite_result:
            metrics.update(
                {
                    "initial_energy_kj_mol": initial_energy,
                    "final_energy_kj_mol": final_energy,
                    "energy_change_kj_mol": final_energy - initial_energy,
                    "max_displacement_angstrom": max_displacement,
                }
            )
        else:
            metrics["failure"] = "non-finite energy or displacement"
        reason = (
            "openmm: serialized System lacks an independently attested parameterization "
            "receipt; local minimization is not a stability simulation"
            if geometry_valid
            else "hard:openmm local-minimization geometry gate failed"
        )
        return True, geometry_valid, refs, metrics, reason
    except Exception as exc:  # OpenMM plugins expose backend-specific exception classes.
        return (
            True,
            False,
            refs,
            {
                "system_deserialized": True,
                "particle_count_matches": True,
                "minimization_completed": False,
                "geometry_gate_passed": False,
                "platform": platform_name,
                "failure": type(exc).__name__,
            },
            f"hard:openmm minimization failed ({type(exc).__name__})",
        )
    finally:
        if context is not None:
            del context
        del integrator


def _evidence(
    store: ArtifactStore,
    *,
    tool: str,
    molecule_id: str,
    candidate_id: str,
    metrics: dict[str, Any],
    inputs: dict[str, ArtifactRef],
    runtime: dict[str, Any],
    fixture: bool,
) -> ArtifactRef:
    producer = f"test-fixture-{tool}" if fixture else f"protbind.{tool}"
    return store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.tool-evidence",
            "tool": tool,
            "molecule_id": molecule_id,
            "candidate_id": candidate_id,
            "metrics": metrics,
            "inputs": {name: reference.to_dict() for name, reference in inputs.items()},
            "runtime": runtime,
            "test_fixture": fixture,
        },
        producer=producer,
        producer_version=str(runtime["version"]),
    )


def _handler(request: WorkerRequest, store: ArtifactStore) -> WorkerResponse:
    started = time.perf_counter()
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise WorkerFailure(
            "ARCHITECTURE_SPOOFING", "HSA_OVERRIDE_GFX_VERSION is forbidden", recoverable=False
        )
    max_iterations, max_displacement = _validate_parameters(request.parameters)
    (
        docking,
        docking_reference,
        toolchain,
        toolchain_reference,
        batch,
        context,
    ) = _stage_inputs(request, store)
    pins, fixture = _validate_toolchain(
        request, store, toolchain_reference, toolchain
    )
    posebusters_runtime = _attest_tool("posebusters", pins["posebusters"])
    optional_runtimes: dict[str, dict[str, Any] | None] = {}
    optional_runtime_reasons: dict[str, str | None] = {}
    for name in ("spyrmsd", "prolif", "openmm"):
        runtime, reason = _optional_attestation(name, pins)
        optional_runtimes[name] = runtime
        optional_runtime_reasons[name] = reason
    candidates = _validate_batch(store, batch, docking, docking_reference)
    configured_pb = toolchain.get(
        "posebusters_configs", toolchain.get("posebusters_config")
    )
    configured_pb_configs = (
        {configured_pb} if isinstance(configured_pb, str) else set(configured_pb or [])
    )
    if any(prepared.get("reference_pose") is not None for _, prepared in candidates) and (
        "redock" not in configured_pb_configs
    ):
        raise WorkerFailure(
            "TOOLCHAIN_INVALID",
            "reference validation requires the redock PoseBusters configuration pin",
            recoverable=False,
        )
    hip_requested = any(
        isinstance(prepared.get("openmm"), dict)
        and prepared["openmm"].get("platform") == "HIP"
        for _, prepared in candidates
    )
    if hip_requested:
        visible_device = os.environ.get("HIP_VISIBLE_DEVICES", "")
        conflicting_masks = {
            name
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
            )
            if os.environ.get(name)
        }
        if (
            not visible_device.isascii()
            or not visible_device.isdecimal()
            or (len(visible_device) > 1 and visible_device.startswith("0"))
            or conflicting_masks
        ):
            raise WorkerFailure(
                "RESOURCE_POLICY_VIOLATION",
                "OpenMM HIP requires one canonical HIP_VISIBLE_DEVICES index and no aliases",
                recoverable=False,
            )
    output_candidates: list[dict[str, Any]] = []
    evidence_references: list[ArtifactRef] = []
    warnings: list[str] = []
    if fixture:
        warnings.append("protocol fixture runtime; outputs are not scientific evidence")
    for position, (upstream, prepared) in enumerate(candidates):
        molecule_id = str(upstream["molecule_id"])
        candidate_id = str(upstream["candidate_id"])
        candidate_directory = Path(tempfile.mkdtemp(prefix=f"validation-{position}-"))
        preparation_attested = _preparation_attested(store, docking, upstream)
        unsupported: list[str] = []
        if not preparation_attested:
            unsupported.append(
                "validation-preparation: exact pose/receptor derivation receipts are "
                "missing, incomplete, or fixture-only; grade is capped at HYPOTHESIS_ONLY"
            )
        candidate_evidence: list[ArtifactRef] = []
        try:
            reference_pose = _authorized_reference_pose(store, prepared, context)
            has_reference_pose = reference_pose is not None
            pb_metrics, pb_operational = _run_posebusters(
                store,
                prepared,
                candidate_directory,
                reference_pose=reference_pose,
            )
            pb_inputs = {
                "docked_pose": _reference(upstream["pose"], "DOCKED pose"),
                **pb_operational,
            }
            if upstream.get("cofold_structure") is not None:
                pb_inputs["cofold_pose"] = _reference(
                    upstream["cofold_structure"], "DOCKED cofold_structure"
                )
            pb_evidence = _evidence(
                store,
                tool="posebusters",
                molecule_id=molecule_id,
                candidate_id=candidate_id,
                metrics=pb_metrics,
                inputs=pb_inputs,
                runtime=posebusters_runtime,
                fixture=fixture,
            )
            candidate_evidence.append(pb_evidence)

            symmetry_rmsd: float | None = None
            spyrmsd_runtime = optional_runtimes["spyrmsd"]
            if spyrmsd_runtime is None:
                unsupported.append(optional_runtime_reasons["spyrmsd"] or "spyrmsd unavailable")
            else:
                symmetry_rmsd, spyrmsd_inputs, reason = _run_spyrmsd(
                    store,
                    upstream,
                    prepared,
                    context,
                    candidate_directory,
                )
                if reason is not None:
                    unsupported.append(reason)
                elif symmetry_rmsd is not None and spyrmsd_inputs is not None:
                    if reference_pose != spyrmsd_inputs["reference_pose"]:
                        raise WorkerFailure(
                            "LINEAGE_MISMATCH",
                            "sPyRMSD used a different reference pose",
                            recoverable=False,
                        )
                    candidate_evidence.append(
                        _evidence(
                            store,
                            tool="spyrmsd",
                            molecule_id=molecule_id,
                            candidate_id=candidate_id,
                            metrics={"symmetry_rmsd_angstrom": symmetry_rmsd},
                            inputs=spyrmsd_inputs,
                            runtime=spyrmsd_runtime,
                            fixture=fixture,
                        )
                    )

            ifp_similarity: float | None = None
            prolif_metrics: dict[str, Any] | None = None
            prolif_runtime = optional_runtimes["prolif"]
            if prolif_runtime is None:
                unsupported.append(optional_runtime_reasons["prolif"] or "prolif unavailable")
            else:
                ifp_similarity, prolif_operational, prolif_metrics, reason = _run_prolif(
                    store, prepared, candidate_directory
                )
                if reason is not None:
                    unsupported.append(reason)
                elif prolif_operational is not None and prolif_metrics is not None:
                    prolif_inputs = {
                        "pose": _reference(upstream["pose"], "DOCKED pose"),
                        **prolif_operational,
                    }
                    if reference_pose is not None:
                        prolif_inputs["reference_pose"] = reference_pose
                    elif upstream.get("cofold_structure") is not None:
                        prolif_inputs["cofold_pose"] = _reference(
                            upstream["cofold_structure"], "DOCKED cofold_structure"
                        )
                    candidate_evidence.append(
                        _evidence(
                            store,
                            tool="prolif",
                            molecule_id=molecule_id,
                            candidate_id=candidate_id,
                            metrics=prolif_metrics,
                            inputs=prolif_inputs,
                            runtime=prolif_runtime,
                            fixture=fixture,
                        )
                    )

            openmm_system_loadable: bool | None = None
            openmm_geometry_valid: bool | None = None
            openmm_runtime = optional_runtimes["openmm"]
            if openmm_runtime is None:
                unsupported.append(optional_runtime_reasons["openmm"] or "openmm unavailable")
            else:
                (
                    openmm_system_loadable,
                    openmm_geometry_valid,
                    openmm_operational,
                    openmm_metrics,
                    reason,
                ) = _run_openmm(
                    store,
                    prepared,
                    candidate_directory,
                    max_iterations=max_iterations,
                    max_displacement_angstrom=max_displacement,
                )
                if reason is not None:
                    unsupported.append(reason)
                if openmm_system_loadable is not None and openmm_metrics is not None:
                    candidate_evidence.append(
                        _evidence(
                            store,
                            tool="openmm",
                            molecule_id=molecule_id,
                            candidate_id=candidate_id,
                            metrics=openmm_metrics,
                            inputs={
                                "pose": _reference(upstream["pose"], "DOCKED pose"),
                                **(openmm_operational or {}),
                            },
                            runtime=openmm_runtime,
                            fixture=fixture,
                        )
                    )

            decision = "Protocol fixture only; not scientific evidence. " if fixture else ""
            if not pb_metrics["valid"]:
                decision += (
                    "Rejected because the docked Vina pose failed the pinned "
                    f"PoseBusters {pb_metrics['config']} checks."
                )
            elif openmm_system_loadable is False or openmm_geometry_valid is False:
                decision += "Rejected by the explicit OpenMM local-minimization geometry gate."
            else:
                decision += "The docked Vina pose passed PoseBusters checks. "
                if preparation_attested:
                    decision += (
                        "Pose and receptor preparation lineage is attested; optional "
                        "reference/cofold metrics remain non-binding evidence."
                    )
                else:
                    decision += (
                        "Preparation lineage is not fully attested, so the grade remains "
                        "HYPOTHESIS_ONLY."
                    )
            bundle: dict[str, Any] = {
                "preparation_attested": preparation_attested,
                "posebusters_valid": pb_metrics["valid"],
                "vina_pose_valid": pb_metrics["docked_valid"],
                "cofold_pose_valid": pb_metrics["cofold_valid"],
                "unsupported_reasons": unsupported,
                "evidence": [item.to_dict() for item in candidate_evidence],
            }
            if symmetry_rmsd is not None:
                bundle["symmetry_rmsd_angstrom"] = symmetry_rmsd
            if prolif_metrics is not None:
                counts = prolif_metrics["counts"]
                bundle.update(
                    {
                        "ifp_similarity": prolif_metrics["ifp_similarity"],
                        "ifp_reference_recovery": prolif_metrics[
                            "reference_interaction_recovery"
                        ],
                        "ifp_predicted_precision": prolif_metrics[
                            "predicted_interaction_precision"
                        ],
                        "ifp_docked_label_count": counts["docked"],
                        "ifp_comparison_label_count": counts["comparison"],
                        "ifp_intersection_count": counts["intersection"],
                        "ifp_union_count": counts["union"],
                    }
                )
            candidate_output: dict[str, Any] = {
                "candidate_id": candidate_id,
                "molecule_id": molecule_id,
                "microstate_id": upstream["microstate_id"],
                "docked_pose": upstream["pose"],
                "engine": "test-fixture-validation" if fixture else ENGINE,
                "seed": request.seed,
                "has_reference_pose": has_reference_pose,
                "decision_reason": decision,
                "bundle": bundle,
            }
            if upstream.get("cofold_structure") is not None:
                candidate_output["cofold_pose"] = upstream["cofold_structure"]
            if has_reference_pose and reference_pose is not None:
                candidate_output["reference_pose"] = reference_pose.to_dict()
            output_candidates.append(candidate_output)
            evidence_references.extend(candidate_evidence)
        finally:
            # Temporary files contain staged scientific inputs.  Remove individual
            # files eagerly; the empty directory is local to the isolated worker.
            for path in candidate_directory.iterdir():
                if path.is_file():
                    path.unlink()
            candidate_directory.rmdir()
    producer = "test-fixture-validation-worker" if fixture else ENGINE
    primary = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.validation-bundle",
            "candidates": output_candidates,
            "toolchain": toolchain_reference.to_dict(),
            "test_fixture": fixture,
        },
        producer=producer,
        producer_version="1.0",
    )
    return WorkerResponse(
        job_id=request.job_id,
        engine=request.engine,
        outputs=(primary, *evidence_references),
        provenance=request.provenance,
        timings_seconds={"validation_total": time.perf_counter() - started},
        peak_vram_bytes=None,
        warnings=tuple(warnings),
    )


if __name__ == "__main__":
    raise SystemExit(serve_worker(ENGINE, _handler))
