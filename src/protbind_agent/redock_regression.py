"""Fail-closed aggregation for hash-bound redocking regression manifests.

This module does not select a holdout or silently repair scientific inputs. It
verifies an existing holdout-selection receipt, preserves every frozen case in
the denominator, and evaluates artifact files declared by a hash-bound
redocking result. Optional ligand hydrogen preparation requires an explicitly
supplied, independent artifact store and emits a verifiable receipt.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes, sha256_file
from .interaction_fingerprint import (
    compare_prolif_paths,
    prepare_prolif_ligand,
    prepare_prolif_receptor,
)

TARGET_CASE_COUNT = 10
_SHA256_LENGTH = 64


class RegressionIntegrityError(ValueError):
    """A manifest, path, or content hash cannot be trusted."""


class RegressionDesign(StrEnum):
    PILOT_RETROSPECTIVE = "PILOT_RETROSPECTIVE"
    FROZEN_HOLDOUT = "FROZEN_HOLDOUT"


@dataclass(frozen=True, slots=True)
class RedockRegressionConfig:
    target_case_count: int = TARGET_CASE_COUNT
    rmsd_threshold_angstrom: float = 2.0
    recorded_rmsd_tolerance_angstrom: float = 1e-5

    def __post_init__(self) -> None:
        if self.target_case_count != TARGET_CASE_COUNT:
            raise ValueError(f"redock regression target_case_count is fixed at {TARGET_CASE_COUNT}")
        if self.rmsd_threshold_angstrom != 2.0:
            raise ValueError("redock regression RMSD threshold is fixed at 2.0 angstrom")
        for name, value in (
            ("rmsd_threshold_angstrom", self.rmsd_threshold_angstrom),
            (
                "recorded_rmsd_tolerance_angstrom",
                self.recorded_rmsd_tolerance_angstrom,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.1",
            "target_case_count": self.target_case_count,
            "rmsd_threshold_angstrom": float(self.rmsd_threshold_angstrom),
            "recorded_rmsd_tolerance_angstrom": float(self.recorded_rmsd_tolerance_angstrom),
            "top1_success": "PoseBusters-valid AND recomputed symmetry RMSD <= threshold",
            "top5_success": (
                "any of the first five records from the hash-bound Vina SDF is "
                "independently PoseBusters-valid with recomputed symmetry RMSD <= threshold"
            ),
            "rate_denominator": "all cases frozen in the regression manifest",
        }


RMSDEvaluator = Callable[[Path, Path], float]
PoseBustersEvaluator = Callable[[Path, Path, Path], dict[str, Any]]
IFPEvaluator = Callable[..., dict[str, Any]]


def _distribution_identity(name: str) -> dict[str, Any]:
    try:
        distribution = importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError:
        return {"status": "missing", "version": None, "record_sha256": None}
    record = distribution.read_text("RECORD")
    return {
        "status": "installed",
        "version": distribution.version,
        "record_sha256": (
            sha256_bytes(record.encode("utf-8")) if record is not None else None
        ),
    }


def _evaluator_identity(value: Callable[..., Any] | None, default_name: str) -> dict[str, str]:
    if value is None:
        return {
            "mode": "REAL_DEFAULT",
            "implementation": default_name,
        }
    module = getattr(value, "__module__", type(value).__module__)
    qualified_name = getattr(value, "__qualname__", type(value).__qualname__)
    return {
        "mode": "INJECTED",
        "implementation": f"{module}.{qualified_name}",
    }


def _runtime_identity(
    *,
    evaluator_bindings: dict[str, dict[str, str]],
) -> dict[str, Any]:
    source_files = {
        "protbind_agent/redock_regression.py": sha256_file(Path(__file__)),
        "protbind_agent/interaction_fingerprint.py": sha256_file(
            Path(__file__).with_name("interaction_fingerprint.py")
        ),
    }
    core = {
        "schema_version": "1.1",
        "evaluator_mode": (
            "INJECTED_TEST_EVALUATORS"
            if any(value["mode"] == "INJECTED" for value in evaluator_bindings.values())
            else "REAL_TOOLS"
        ),
        "evaluator_bindings": evaluator_bindings,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_name": Path(sys.executable).name,
        "distributions": {
            name: _distribution_identity(name)
            for name in ("rdkit", "spyrmsd", "posebusters", "prolif", "numpy", "pandas")
        },
        "source_files": source_files,
    }
    return {**core, "runtime_sha256": sha256_bytes(canonical_json_bytes(core))}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise RegressionIntegrityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RegressionIntegrityError(f"{name} must be a non-empty POSIX repo-relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RegressionIntegrityError(f"{name} must not be absolute or contain traversal")
    return path


def _resolve_relative(base: Path, value: Any, name: str) -> tuple[PurePosixPath, Path]:
    relative = _relative_path(value, name)
    root = base.resolve()
    resolved = (root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RegressionIntegrityError(f"{name} escapes its declared root") from exc
    return relative, resolved


def _read_hash_bound_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RegressionIntegrityError(f"{name} does not exist")
    observed = sha256_file(path)
    if observed != _require_sha256(expected_sha256, f"{name}.sha256"):
        raise RegressionIntegrityError(f"{name} SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegressionIntegrityError(f"{name} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise RegressionIntegrityError(f"{name} must contain a JSON object")
    return value


def _validate_internal_hash(
    value: dict[str, Any],
    *,
    hash_key: str,
    name: str,
) -> str:
    declared = _require_sha256(value.get(hash_key), f"{name}.{hash_key}")
    body = {key: item for key, item in value.items() if key != hash_key}
    observed = sha256_bytes(canonical_json_bytes(body))
    if observed != declared:
        raise RegressionIntegrityError(f"{name} internal {hash_key} mismatch")
    return declared


def _validate_holdout_manifest(
    repo_root: Path,
    pointer: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(pointer, dict) or set(pointer) != {
        "path",
        "sha256",
        "selection_hash",
    }:
        raise RegressionIntegrityError("frozen holdout pointer is invalid")
    relative, path = _resolve_relative(repo_root, pointer["path"], "holdout.path")
    holdout = _read_hash_bound_json(path, pointer["sha256"], "holdout manifest")
    holdout_schema = holdout.get("schema_version")
    if holdout_schema not in {"1.0", "1.1"}:
        raise RegressionIntegrityError("holdout manifest schema is unsupported")
    selection_hash = _validate_internal_hash(
        holdout,
        hash_key="selection_hash",
        name="holdout manifest",
    )
    if selection_hash != _require_sha256(pointer["selection_hash"], "holdout.selection_hash"):
        raise RegressionIntegrityError("holdout selection hash binding mismatch")
    if holdout.get("requested_count") != TARGET_CASE_COUNT:
        raise RegressionIntegrityError(
            f"frozen holdout must request exactly {TARGET_CASE_COUNT} cases"
        )
    selected = holdout.get("selected")
    if not isinstance(selected, list) or len(selected) != TARGET_CASE_COUNT:
        raise RegressionIntegrityError(
            f"frozen holdout must contain exactly {TARGET_CASE_COUNT} selected cases"
        )
    if holdout_schema == "1.1":
        for source_name in ("dataset_source", "candidate_list"):
            source = holdout.get(source_name)
            if not isinstance(source, dict):
                raise RegressionIntegrityError(
                    f"holdout manifest 1.1 requires {source_name} provenance"
                )
            _require_sha256(
                source.get("sha256"), f"holdout.{source_name}.sha256"
            )
        policy = holdout.get("eligibility_policy")
        if not isinstance(policy, dict):
            raise RegressionIntegrityError(
                "holdout manifest 1.1 requires eligibility_policy provenance"
            )
        if not isinstance(policy.get("version"), str) or not policy["version"]:
            raise RegressionIntegrityError("holdout eligibility policy version is missing")
        _require_sha256(
            policy.get("source_sha256"),
            "holdout.eligibility_policy.source_sha256",
        )
        radius = policy.get("pocket_radius_angstrom")
        if (
            isinstance(radius, bool)
            or not isinstance(radius, int | float)
            or not math.isfinite(float(radius))
            or float(radius) <= 0
        ):
            raise RegressionIntegrityError(
                "holdout eligibility pocket radius must be finite and positive"
            )
        for policy_name in ("heterogen_policy", "stereo_policy"):
            if not isinstance(policy.get(policy_name), str) or not policy[policy_name]:
                raise RegressionIntegrityError(
                    f"holdout eligibility {policy_name} is missing"
                )
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in selected:
        if not isinstance(candidate, dict):
            raise RegressionIntegrityError("holdout selected case must be an object")
        case_id = candidate.get("complex_id")
        if not isinstance(case_id, str) or not case_id or case_id in by_id:
            raise RegressionIntegrityError("holdout selected case IDs must be unique strings")
        for artifact_name in ("receptor", "native_ligand"):
            artifact = candidate.get(artifact_name)
            if not isinstance(artifact, dict):
                raise RegressionIntegrityError(f"holdout {case_id} has no {artifact_name} artifact")
            _require_sha256(artifact.get("sha256"), f"holdout.{case_id}.{artifact_name}.sha256")
        if holdout_schema == "1.1":
            unsupported_flags = {
                "contains_metal": candidate.get("contains_metal"),
                "requires_cofactor": candidate.get("requires_cofactor"),
                "pocket_altloc_ambiguous": candidate.get("pocket_altloc_ambiguous"),
                "missing_pocket_heavy_atoms": candidate.get(
                    "missing_pocket_heavy_atoms"
                ),
                "contains_nonstandard_protein_residue": candidate.get(
                    "contains_nonstandard_protein_residue"
                ),
                "missing_backbone_atoms": candidate.get("missing_backbone_atoms"),
                "ligand_unspecified_stereo": candidate.get(
                    "ligand_unspecified_stereo"
                ),
            }
            if any(value is not False for value in unsupported_flags.values()):
                declared = ", ".join(
                    name for name, value in unsupported_flags.items() if value is not False
                )
                raise RegressionIntegrityError(
                    f"holdout selected case {case_id} declares unsupported flags: {declared}"
                )
            if candidate.get("receptor_model_count") != 1:
                raise RegressionIntegrityError(
                    f"holdout selected case {case_id} receptor_model_count is not one"
                )
        by_id[case_id] = candidate
    return (
        {
            "path": relative.as_posix(),
            "sha256": pointer["sha256"],
            "selection_hash": selection_hash,
            "selected_case_ids": list(by_id),
        },
        by_id,
    )


def _validate_regression_manifest(
    repo_root: Path,
    manifest_path: Path | str,
) -> tuple[dict[str, Any], PurePosixPath, str]:
    relative, path = _resolve_relative(repo_root, str(manifest_path), "manifest_path")
    if not path.is_file():
        raise RegressionIntegrityError("regression manifest does not exist")
    raw_sha256 = sha256_file(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegressionIntegrityError("regression manifest is not readable JSON") from exc
    if not isinstance(manifest, dict):
        raise RegressionIntegrityError("regression manifest must be a JSON object")
    if set(manifest) != {
        "schema_version",
        "evaluation_design",
        "target_case_count",
        "holdout",
        "cases",
        "manifest_sha256",
    }:
        raise RegressionIntegrityError("regression manifest has missing or unknown fields")
    if manifest["schema_version"] != "1.0":
        raise RegressionIntegrityError("regression manifest schema is unsupported")
    try:
        design = RegressionDesign(manifest["evaluation_design"])
    except (TypeError, ValueError) as exc:
        raise RegressionIntegrityError("regression evaluation_design is invalid") from exc
    if manifest["target_case_count"] != TARGET_CASE_COUNT:
        raise RegressionIntegrityError(f"regression target_case_count must be {TARGET_CASE_COUNT}")
    _validate_internal_hash(
        manifest,
        hash_key="manifest_sha256",
        name="regression manifest",
    )
    cases = manifest["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= TARGET_CASE_COUNT:
        raise RegressionIntegrityError(
            f"regression manifest must freeze between 1 and {TARGET_CASE_COUNT} cases"
        )
    case_ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "result"}:
            raise RegressionIntegrityError("regression case entry is invalid")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise RegressionIntegrityError("regression case IDs must be unique strings")
        case_ids.append(case_id)
        pointer = case["result"]
        if pointer is None:
            continue
        if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
            raise RegressionIntegrityError(f"result pointer for {case_id} is invalid")
        _relative_path(pointer["path"], f"cases.{case_id}.result.path")
        _require_sha256(pointer["sha256"], f"cases.{case_id}.result.sha256")
    if design is RegressionDesign.FROZEN_HOLDOUT:
        if len(cases) != TARGET_CASE_COUNT:
            raise RegressionIntegrityError(
                f"FROZEN_HOLDOUT must freeze exactly {TARGET_CASE_COUNT} cases"
            )
        if manifest["holdout"] is None:
            raise RegressionIntegrityError("FROZEN_HOLDOUT requires a holdout manifest")
    elif manifest["holdout"] is not None:
        raise RegressionIntegrityError(
            "PILOT_RETROSPECTIVE must not claim a frozen holdout binding"
        )
    return manifest, relative, raw_sha256


def _artifact_file(
    result_directory: Path,
    record: Any,
    *,
    name: str,
    expected_scope: str,
    default_scope_from_directory: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict):
        raise ValueError(f"{name} artifact record is missing")
    if default_scope_from_directory:
        declared_scope = record.get("access_scope", "DOCKING_VISIBLE")
    else:
        declared_scope = record.get("access_scope")
    if declared_scope != expected_scope:
        raise RegressionIntegrityError(f"{name} artifact scope mismatch")
    relative, path = _resolve_relative(result_directory, record.get("file"), f"{name}.file")
    if expected_scope == "DOCKING_VISIBLE" and relative.parts[0] != "artifacts":
        raise RegressionIntegrityError(f"{name} must be under artifacts/")
    if expected_scope == "VALIDATION_ONLY" and relative.parts[0] != "validation-only":
        raise RegressionIntegrityError(f"{name} must be under validation-only/")
    declared_sha = _require_sha256(record.get("sha256"), f"{name}.sha256")
    if not path.is_file() or sha256_file(path) != declared_sha:
        raise RegressionIntegrityError(f"{name} file SHA-256 mismatch")
    return path, {
        "path": relative.as_posix(),
        "sha256": declared_sha,
        "access_scope": expected_scope,
    }


def _top1_binding(result_directory: Path, top1: dict[str, Any]) -> dict[str, Any]:
    """Validate the declared top-one binding without reading its split-mode file.

    Scientific recomputation uses records derived from the bound multi-record
    Vina SDF.  The per-mode path remains descriptive lineage only and cannot
    become an unbound second source of coordinates.
    """

    pose = top1.get("pose_artifact")
    if not isinstance(pose, dict):
        raise ValueError("top1 pose artifact is missing")
    if pose.get("access_scope", "DOCKING_VISIBLE") != "DOCKING_VISIBLE":
        raise RegressionIntegrityError("top1 pose artifact scope mismatch")
    relative, path = _resolve_relative(result_directory, top1.get("file"), "top1.file")
    del path
    if relative.parts[0] != "artifacts":
        raise RegressionIntegrityError("top1 must be under artifacts/")
    return {
        "path": relative.as_posix(),
        "sha256": _require_sha256(pose.get("sha256"), "top1.sha256"),
        "access_scope": "DOCKING_VISIBLE",
    }


def _finite(value: Any, name: str, *, non_negative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or (non_negative and float(value) < 0)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _probability(value: Any, name: str) -> float:
    result = _finite(value, name, non_negative=True)
    if result > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _default_spyrmsd(reference_path: Path, predicted_path: Path) -> float:
    from rdkit import Chem
    from spyrmsd.molecule import Molecule
    from spyrmsd.rmsd import rmsdwrapper

    def load(path: Path) -> Any:
        molecules = [
            molecule
            for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
            if molecule is not None
        ]
        if len(molecules) != 1:
            raise ValueError("sPyRMSD input must contain exactly one readable molecule")
        return molecules[0]

    reference = Molecule.from_rdkit(load(reference_path))
    predicted = Molecule.from_rdkit(load(predicted_path))
    values = rmsdwrapper(
        reference,
        predicted,
        symmetry=True,
        center=False,
        minimize=False,
        strip=True,
    )
    value = values[0] if isinstance(values, list | tuple) else values
    return _finite(value, "recomputed symmetry RMSD", non_negative=True)


def _flatten_posebusters_row(frame: Any) -> dict[str, Any]:
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] != 1:
        raise ValueError("PoseBusters full report must contain exactly one row")
    try:
        items = list(frame.iloc[0].items())
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("PoseBusters full report is unreadable") from exc
    values: dict[str, Any] = {}
    for column, value in items:
        components = (
            [str(part) for part in column if str(part)]
            if isinstance(column, tuple)
            else [str(column)]
        )
        for component in components:
            if component in values and values[component] != value:
                raise ValueError(f"PoseBusters column {component} is ambiguous")
            values[component] = value
    return values


def _boolean_posebusters_report(frame: Any) -> tuple[bool, dict[str, bool]]:
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
        raise ValueError("PoseBusters did not return one boolean result row")
    checks: dict[str, bool] = {}
    try:
        items = frame.iloc[0].items()
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("PoseBusters boolean report is unreadable") from exc
    for column, value in items:
        name = (
            "::".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
        )
        if not name or name in checks or not (
            isinstance(value, bool) or type(value).__name__ == "bool_"
        ):
            raise ValueError("PoseBusters report is not a unique boolean matrix")
        checks[name] = bool(value)
    geometry_checks = {
        check: result for check, result in checks.items() if "rmsd" not in check.lower()
    }
    if not geometry_checks:
        raise ValueError("PoseBusters report has no geometry/chemistry checks")
    return all(geometry_checks.values()), checks


def _default_posebusters(
    predicted_path: Path,
    native_path: Path,
    receptor_path: Path,
) -> dict[str, Any]:
    from posebusters import PoseBusters

    validator = PoseBusters(
        config="redock",
        top_n=1,
        max_workers=0,
        chunk_size=1,
    )
    frame = validator.bust(
        mol_pred=str(predicted_path),
        mol_true=str(native_path),
        mol_cond=str(receptor_path),
        full_report=True,
    )
    boolean_frame = validator.bust(
        mol_pred=str(predicted_path),
        mol_true=str(native_path),
        mol_cond=str(receptor_path),
        full_report=False,
    )
    posebusters_valid, posebusters_checks = _boolean_posebusters_report(boolean_frame)
    row = _flatten_posebusters_row(frame)
    count = row.get("num_pairwise_clashes_protein")
    if type(count) is not int and type(count).__name__ not in {"int32", "int64"}:
        raise ValueError("PoseBusters protein clash count is not an integer")
    clash = row.get("most_extreme_clash_protein")
    if not isinstance(clash, bool) and type(clash).__name__ not in {"bool", "bool_"}:
        raise ValueError("PoseBusters protein clash indicator is not boolean")
    return {
        "posebusters_valid": posebusters_valid,
        "posebusters_checks": posebusters_checks,
        "energy_ratio": row.get("energy_ratio"),
        "mol_pred_energy": row.get("mol_pred_energy"),
        "ensemble_avg_energy": row.get("ensemble_avg_energy"),
        "protein_ligand_pairwise_clash_count": int(count),
        "protein_ligand_clash_detected": bool(clash),
        "protein_ligand_volume_overlap": row.get("volume_overlap_protein"),
        "protein_ligand_minimum_distance_angstrom": row.get("smallest_distance_protein"),
    }


def _validate_posebusters_metrics(value: Any) -> dict[str, Any]:
    expected = {
        "posebusters_valid",
        "posebusters_checks",
        "energy_ratio",
        "mol_pred_energy",
        "ensemble_avg_energy",
        "protein_ligand_pairwise_clash_count",
        "protein_ligand_clash_detected",
        "protein_ligand_volume_overlap",
        "protein_ligand_minimum_distance_angstrom",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("PoseBusters evaluator returned incomplete full-report metrics")
    valid = value["posebusters_valid"]
    checks = value["posebusters_checks"]
    if type(valid) is not bool:
        raise ValueError("PoseBusters validity must be boolean")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(
            not isinstance(name, str) or not name or type(check) is not bool
            for name, check in checks.items()
        )
    ):
        raise ValueError("PoseBusters validity checks must be a non-empty boolean mapping")
    geometry_checks = {
        name: check for name, check in checks.items() if "rmsd" not in name.lower()
    }
    if not geometry_checks or valid is not all(geometry_checks.values()):
        raise ValueError("PoseBusters validity differs from its geometry/chemistry checks")
    count = value["protein_ligand_pairwise_clash_count"]
    if type(count) is not int or count < 0:
        raise ValueError("PoseBusters clash count must be a non-negative integer")
    detected = value["protein_ligand_clash_detected"]
    if type(detected) is not bool:
        raise ValueError("PoseBusters clash indicator must be boolean")
    if detected is not (count > 0):
        raise ValueError("PoseBusters clash count and indicator are inconsistent")
    return {
        "posebusters_valid": valid,
        "posebusters_checks": dict(sorted(checks.items())),
        "energy_ratio": _finite(
            value["energy_ratio"], "PoseBusters energy_ratio", non_negative=True
        ),
        "mol_pred_energy": _finite(value["mol_pred_energy"], "PoseBusters mol_pred_energy"),
        "ensemble_avg_energy": _finite(
            value["ensemble_avg_energy"], "PoseBusters ensemble_avg_energy"
        ),
        "protein_ligand_pairwise_clash_count": count,
        "protein_ligand_clash_detected": detected,
        "protein_ligand_volume_overlap": _probability(
            value["protein_ligand_volume_overlap"],
            "PoseBusters protein-ligand volume overlap",
        ),
        "protein_ligand_minimum_distance_angstrom": _finite(
            value["protein_ligand_minimum_distance_angstrom"],
            "PoseBusters protein-ligand minimum distance",
            non_negative=True,
        ),
        "semantics": (
            "energy_ratio and internal ligand energies are PoseBusters conformational "
            "diagnostics; they are not binding energies or binding free energies."
        ),
    }


def _validate_ifp_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("ProLIF evaluator returned no metrics")
    metrics = {
        "jaccard": _probability(value.get("ifp_similarity"), "ProLIF Jaccard"),
        "reference_interaction_recovery": _probability(
            value.get("reference_interaction_recovery"),
            "ProLIF reference interaction recovery",
        ),
        "predicted_interaction_precision": _probability(
            value.get("predicted_interaction_precision"),
            "ProLIF predicted interaction precision",
        ),
    }
    counts = value.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("ProLIF interaction counts are missing")
    for name in ("docked", "comparison", "intersection", "union"):
        count = counts.get(name)
        if type(count) is not int or count < 0:
            raise ValueError(f"ProLIF {name} count must be a non-negative integer")
    docked = counts["docked"]
    comparison = counts["comparison"]
    intersection = counts["intersection"]
    union = counts["union"]
    if intersection > min(docked, comparison) or union != (
        docked + comparison - intersection
    ):
        raise ValueError("ProLIF interaction counts are internally inconsistent")
    expected_scores = {
        "jaccard": intersection / union if union else None,
        "reference_interaction_recovery": (
            intersection / comparison if comparison else None
        ),
        "predicted_interaction_precision": (
            intersection / docked if docked else None
        ),
    }
    for name, expected in expected_scores.items():
        observed = metrics[name]
        if expected is None or not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"ProLIF {name} differs from its interaction counts")
    metrics["counts"] = {
        name: counts[name] for name in ("docked", "comparison", "intersection", "union")
    }
    metrics["semantics"] = "IFP agreement is structural evidence only; it is not affinity evidence."
    return metrics


def _import_ligand_for_prolif(
    store: ArtifactStore,
    path: Path,
    record: dict[str, Any],
    *,
    artifact_scope: str,
) -> tuple[Any, dict[str, Any]]:
    media_type = record.get("media_type")
    producer = record.get("producer")
    if not isinstance(media_type, str) or not media_type:
        raise ValueError("ProLIF source ligand media_type is missing")
    if not isinstance(producer, str) or not producer:
        raise ValueError("ProLIF source ligand producer is missing")
    imported = store.put_bytes(
        path.read_bytes(),
        media_type=media_type,
        producer=producer,
        producer_version=str(record.get("producer_version") or "unknown"),
        source=record.get("source"),
        license=record.get("license"),
    )
    if imported.sha256 != _require_sha256(record.get("sha256"), "ProLIF source SHA-256"):
        raise RegressionIntegrityError("ProLIF source import changed the ligand SHA-256")
    prepared = prepare_prolif_ligand(
        store,
        imported,
        artifact_scope=artifact_scope,
    )
    receipt = store.read_json(prepared.receipt)
    if (
        not isinstance(receipt, dict)
        or receipt.get("kind") != "protbind.prolif-ligand-preparation-receipt"
        or receipt.get("artifact_scope") != artifact_scope
        or receipt.get("input_ligand", {}).get("sha256") != imported.sha256
        or receipt.get("prepared_ligand", {}).get("sha256") != prepared.prepared_ligand.sha256
        or receipt.get("hydrogens_added") != prepared.hydrogens_added
        or receipt.get("heavy_atom_identity_preserved") is not True
        or receipt.get("heavy_atom_max_coordinate_delta_angstrom")
        != prepared.heavy_atom_max_coordinate_delta_angstrom
    ):
        raise RegressionIntegrityError("ProLIF ligand preparation receipt is inconsistent")
    return prepared, {
        "artifact_scope": artifact_scope,
        "input_ligand": imported.to_dict(),
        "prepared_ligand": prepared.prepared_ligand.to_dict(),
        "preparation_receipt": prepared.receipt.to_dict(),
        "hydrogens_added": prepared.hydrogens_added,
        "heavy_atom_max_coordinate_delta_angstrom": (
            prepared.heavy_atom_max_coordinate_delta_angstrom
        ),
        "receipt_summary": {
            "method": receipt.get("method"),
            "rdkit_version": receipt.get("rdkit_version"),
            "input_sha256": imported.sha256,
            "output_sha256": prepared.prepared_ligand.sha256,
            "input_explicit_hydrogen_count": receipt.get("input_explicit_hydrogen_count"),
            "output_explicit_hydrogen_count": receipt.get("output_explicit_hydrogen_count"),
            "heavy_atom_identity_preserved": True,
            "coordinate_tolerance_angstrom": receipt.get("coordinate_tolerance_angstrom"),
        },
    }


def _import_receptor_for_prolif(
    store: ArtifactStore,
    path: Path,
    record: dict[str, Any],
    *,
    docked_ligand: Any,
    comparison_ligand: Any,
) -> tuple[Any, dict[str, Any]]:
    media_type = record.get("media_type")
    producer = record.get("producer")
    if media_type != "chemical/x-pdb" or not isinstance(producer, str) or not producer:
        raise ValueError("ProLIF source receptor metadata is invalid")
    imported = store.put_bytes(
        path.read_bytes(),
        media_type=media_type,
        producer=producer,
        producer_version=str(record.get("producer_version") or "unknown"),
        source=record.get("source"),
        license=record.get("license"),
    )
    if imported.sha256 != _require_sha256(record.get("sha256"), "ProLIF receptor SHA-256"):
        raise RegressionIntegrityError("ProLIF source import changed the receptor SHA-256")
    prepared = prepare_prolif_receptor(
        store,
        imported,
        docked_ligand,
        comparison_ligand,
    )
    receipt = store.read_json(prepared.receipt)
    if (
        not isinstance(receipt, dict)
        or receipt.get("kind") != "protbind.prolif-receptor-preparation-receipt"
        or receipt.get("input_receptor", {}).get("sha256") != imported.sha256
        or receipt.get("prepared_receptor", {}).get("sha256")
        != prepared.prepared_receptor.sha256
        or receipt.get("atom_identity_preserved") is not True
        or receipt.get("selected_residue_count") != prepared.selected_residue_count
        or receipt.get("selected_atom_count") != prepared.selected_atom_count
        or receipt.get("coordinate_max_delta_angstrom")
        != prepared.coordinate_max_delta_angstrom
    ):
        raise RegressionIntegrityError("ProLIF receptor preparation receipt is inconsistent")
    return prepared, {
        "input_receptor": imported.to_dict(),
        "prepared_receptor": prepared.prepared_receptor.to_dict(),
        "preparation_receipt": prepared.receipt.to_dict(),
        "method": receipt.get("method"),
        "cutoff_angstrom": receipt.get("cutoff_angstrom"),
        "selected_residue_count": prepared.selected_residue_count,
        "selected_atom_count": prepared.selected_atom_count,
        "explicit_hydrogen_count": receipt.get("explicit_hydrogen_count"),
        "atom_identity_preserved": True,
        "coordinate_max_delta_angstrom": prepared.coordinate_max_delta_angstrom,
        "coordinate_tolerance_angstrom": receipt.get(
            "coordinate_tolerance_angstrom"
        ),
    }


def _sanitize_failure(exc: Exception, repo_root: Path) -> dict[str, str]:
    message = f"{type(exc).__name__}: {exc}".replace(str(repo_root.resolve()), "<repo>")
    return {
        "code": type(exc).__name__.upper(),
        "message": message,
    }


def _validated_redock_failure(value: Any, repo_root: Path) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"stage", "code", "message"}:
        raise RegressionIntegrityError("failed redock result has invalid failure metadata")
    output: dict[str, str] = {}
    for name in ("stage", "code", "message"):
        item = value[name]
        if (
            not isinstance(item, str)
            or not item
            or any(ord(character) < 32 for character in item)
        ):
            raise RegressionIntegrityError(
                f"failed redock result failure.{name} is invalid"
            )
        output[name] = item.replace(str(repo_root.resolve()), "<repo>")
    return output


def _materialize_bound_vina_modes(
    poses_path: Path,
    destination: Path,
    *,
    declared_pose_count: Any,
) -> list[tuple[Path, str]]:
    """Split a committed multi-record Vina SDF into ephemeral ranked records."""

    from rdkit import Chem

    if type(declared_pose_count) is not int or declared_pose_count < 1:
        raise ValueError("completed redock result has an invalid pose_count")
    records = list(Chem.SDMolSupplier(str(poses_path), removeHs=False))
    if len(records) != declared_pose_count or any(record is None for record in records):
        raise ValueError("bound Vina SDF records do not match pose_count")
    materialized: list[tuple[Path, str]] = []
    for rank, molecule in enumerate(records[:5], start=1):
        path = destination / f"bound-vina-mode-{rank:02d}.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(molecule)
        writer.close()
        if not path.is_file():
            raise ValueError("RDKit did not materialize a bound Vina pose")
        materialized.append((path, sha256_file(path)))
    return materialized


def _validate_optional_source_modes(
    source_modes: Any,
    mode_audits: list[dict[str, Any]],
    config: RedockRegressionConfig,
) -> None:
    """Cross-check newer source summaries without using their split-mode files."""

    if source_modes is None:
        return
    if not isinstance(source_modes, list) or len(source_modes) != len(mode_audits):
        raise ValueError("source top5_modes does not match the bound Vina pose count")
    for expected_mode, (source, audit) in enumerate(
        zip(source_modes, mode_audits, strict=True), start=1
    ):
        if not isinstance(source, dict) or source.get("mode") != expected_mode:
            raise ValueError("source top5_modes ranking is invalid")
        artifact = source.get("pose_artifact")
        if (
            not isinstance(artifact, dict)
            or artifact.get("sha256") != audit["derived_pose_sha256"]
            or type(source.get("posebusters_valid")) is not bool
            or source["posebusters_valid"] is not audit["posebusters_valid"]
            or type(source.get("recovered")) is not bool
            or source["recovered"] is not audit["success"]
        ):
            raise ValueError("source top5_modes differs from independent recomputation")
        recorded = _finite(
            source.get("symmetry_rmsd_angstrom"),
            f"source mode {expected_mode} symmetry RMSD",
            non_negative=True,
        )
        if (
            abs(recorded - audit["recomputed_symmetry_rmsd_angstrom"])
            > config.recorded_rmsd_tolerance_angstrom
        ):
            raise ValueError("source top5_modes RMSD differs from independent recomputation")


def _validate_source_pose_summary(
    result: dict[str, Any],
    top1: dict[str, Any],
    mode_audits: list[dict[str, Any]],
    config: RedockRegressionConfig,
) -> tuple[bool, bool]:
    if not mode_audits:
        raise ValueError("no bound Vina poses were evaluated")
    first = mode_audits[0]
    recorded_rmsd = _finite(
        top1.get("symmetry_rmsd_angstrom"),
        "recorded top1 symmetry RMSD",
        non_negative=True,
    )
    if (
        abs(recorded_rmsd - first["recomputed_symmetry_rmsd_angstrom"])
        > config.recorded_rmsd_tolerance_angstrom
    ):
        raise ValueError("recomputed top1 symmetry RMSD differs from the frozen result")
    if (
        top1.get("mode") != 1
        or type(top1.get("posebusters_valid")) is not bool
        or top1["posebusters_valid"] is not first["posebusters_valid"]
        or type(top1.get("recovered")) is not bool
        or top1["recovered"] is not first["success"]
        or type(result.get("top1_recovered")) is not bool
        or result["top1_recovered"] is not first["success"]
    ):
        raise ValueError("source top1 recovery differs from independent recomputation")

    top5_success = any(mode["success"] for mode in mode_audits)
    source_top5 = result.get("top5_recovered")
    oracle = result.get("top5_oracle")
    if type(source_top5) is not bool or not isinstance(oracle, dict):
        raise ValueError("source top5 recovery summary is missing")
    best = min(mode_audits, key=lambda mode: mode["recomputed_symmetry_rmsd_angstrom"])
    recovered = [mode for mode in mode_audits if mode["success"]]
    oracle_rmsd = _finite(
        oracle.get("best_symmetry_rmsd_angstrom"),
        "source top5 oracle symmetry RMSD",
        non_negative=True,
    )
    expected_first = recovered[0]["mode"] if recovered else None
    if (
        oracle.get("evaluated_modes") != len(mode_audits)
        or oracle.get("best_mode") != best["mode"]
        or abs(oracle_rmsd - best["recomputed_symmetry_rmsd_angstrom"])
        > config.recorded_rmsd_tolerance_angstrom
        or oracle.get("any_pb_valid_and_rmsd_le_2") is not top5_success
        or oracle.get("first_recovered_mode") != expected_first
        or source_top5 is not top5_success
    ):
        raise ValueError("source top5 oracle differs from independent recomputation")
    _validate_optional_source_modes(result.get("top5_modes"), mode_audits, config)
    return first["success"], top5_success


def _evaluate_completed(
    *,
    repo_root: Path,
    result_path: Path,
    result: dict[str, Any],
    selected_candidate: dict[str, Any] | None,
    config: RedockRegressionConfig,
    rmsd_evaluator: RMSDEvaluator,
    posebusters_evaluator: PoseBustersEvaluator,
    ifp_evaluator: IFPEvaluator,
    prolif_artifact_store: ArtifactStore | None,
) -> dict[str, Any]:
    top1 = result.get("top1")
    artifacts = result.get("artifacts")
    if not isinstance(top1, dict) or not isinstance(artifacts, dict):
        raise ValueError("completed redock result lacks top1/artifact records")
    result_directory = result_path.parent
    top1_binding = _top1_binding(result_directory, top1)
    native_path, native_binding = _artifact_file(
        result_directory,
        artifacts.get("native_reference"),
        name="native_reference",
        expected_scope="VALIDATION_ONLY",
    )
    receptor_path, receptor_binding = _artifact_file(
        result_directory,
        artifacts.get("prepared_receptor"),
        name="prepared_receptor",
        expected_scope="DOCKING_VISIBLE",
    )
    poses_path, poses_binding = _artifact_file(
        result_directory,
        artifacts.get("vina_poses_sdf"),
        name="vina_poses_sdf",
        expected_scope="DOCKING_VISIBLE",
    )
    if selected_candidate is not None:
        receptor_input_path, receptor_input_binding = _artifact_file(
            result_directory,
            artifacts.get("receptor_input"),
            name="receptor_input",
            expected_scope="DOCKING_VISIBLE",
        )
        del receptor_input_path
        if (
            native_binding["sha256"] != selected_candidate["native_ligand"]["sha256"]
            or receptor_input_binding["sha256"] != selected_candidate["receptor"]["sha256"]
        ):
            raise RegressionIntegrityError(
                "completed result inputs do not match the frozen holdout candidate"
            )
    if prolif_artifact_store is not None:
        store_root = prolif_artifact_store.root.resolve()
        resolved_result_directory = result_directory.resolve()
        if (
            store_root == resolved_result_directory
            or store_root in resolved_result_directory.parents
            or resolved_result_directory in store_root.parents
        ):
            raise RegressionIntegrityError(
                "ProLIF derivation store must not overlap a source redock directory"
            )

    with tempfile.TemporaryDirectory(prefix="protbind-bound-vina-") as temporary:
        directory = Path(temporary)
        materialized = _materialize_bound_vina_modes(
            poses_path,
            directory,
            declared_pose_count=result.get("pose_count"),
        )
        if materialized[0][1] != top1_binding["sha256"]:
            raise RegressionIntegrityError(
                "top1 artifact does not match the first bound Vina SDF record"
            )
        mode_audits: list[dict[str, Any]] = []
        posebusters_reports: list[dict[str, Any]] = []
        for mode, (mode_path, mode_sha256) in enumerate(materialized, start=1):
            rmsd = _finite(
                rmsd_evaluator(native_path, mode_path),
                f"recomputed mode {mode} symmetry RMSD",
                non_negative=True,
            )
            report = _validate_posebusters_metrics(
                posebusters_evaluator(mode_path, native_path, receptor_path)
            )
            success = (
                report["posebusters_valid"]
                and rmsd <= config.rmsd_threshold_angstrom
            )
            gate_checks = {
                name: passed
                for name, passed in report["posebusters_checks"].items()
                if "rmsd" not in name.lower()
            }
            non_gate_rmsd_checks = {
                name: passed
                for name, passed in report["posebusters_checks"].items()
                if "rmsd" in name.lower()
            }
            mode_audits.append(
                {
                    "mode": mode,
                    "parent_vina_poses_sha256": poses_binding["sha256"],
                    "derived_pose_sha256": mode_sha256,
                    "posebusters_valid": report["posebusters_valid"],
                    "posebusters_gate_check_count": len(gate_checks),
                    "posebusters_gate_failed_checks": [
                        name for name, passed in gate_checks.items() if not passed
                    ],
                    "posebusters_non_gate_rmsd_checks": non_gate_rmsd_checks,
                    "recomputed_symmetry_rmsd_angstrom": rmsd,
                    "success": success,
                }
            )
            posebusters_reports.append(report)

        top1_success, top5_success = _validate_source_pose_summary(
            result, top1, mode_audits, config
        )
        top1_path = materialized[0][0]
        recorded_rmsd = _finite(
            top1.get("symmetry_rmsd_angstrom"),
            "recorded top1 symmetry RMSD",
            non_negative=True,
        )
        recomputed_rmsd = mode_audits[0]["recomputed_symmetry_rmsd_angstrom"]
        difference = abs(recorded_rmsd - recomputed_rmsd)

        # compare_prolif_paths deliberately rejects missing explicit hydrogens.
        # The opt-in path writes only to a non-overlapping derivation store.
        prolif_preparation: dict[str, Any] | None = None
        if prolif_artifact_store is None:
            ifp_value = ifp_evaluator(
                docked_ligand_path=top1_path,
                docked_receptor_path=receptor_path,
                comparison_ligand_path=native_path,
                comparison_receptor_path=receptor_path,
                comparison_name="native_reference",
            )
        else:
            top1_prepared, top1_preparation = _import_ligand_for_prolif(
                prolif_artifact_store,
                top1_path,
                top1["pose_artifact"],
                artifact_scope="DOCKING_VISIBLE",
            )
            native_prepared, native_preparation = _import_ligand_for_prolif(
                prolif_artifact_store,
                native_path,
                artifacts["native_reference"],
                artifact_scope="VALIDATION_ONLY",
            )
            receptor_prepared, receptor_preparation = _import_receptor_for_prolif(
                prolif_artifact_store,
                receptor_path,
                artifacts["prepared_receptor"],
                docked_ligand=top1_prepared.prepared_ligand,
                comparison_ligand=native_prepared.prepared_ligand,
            )
            prolif_preparation = {
                "mode": "RECEIPTED_LIGAND_ADDHS_AND_RECEPTOR_POCKET_CROP",
                "docked_ligand": top1_preparation,
                "native_reference": native_preparation,
                "receptor": receptor_preparation,
            }
            docked_prepared_path = directory / "docked-prepared.sdf"
            native_prepared_path = directory / "native-prepared.sdf"
            receptor_prepared_path = directory / "receptor-pocket-prepared.pdb"
            docked_prepared_path.write_bytes(
                prolif_artifact_store.read_bytes(top1_prepared.prepared_ligand)
            )
            native_prepared_path.write_bytes(
                prolif_artifact_store.read_bytes(native_prepared.prepared_ligand)
            )
            receptor_prepared_path.write_bytes(
                prolif_artifact_store.read_bytes(receptor_prepared.prepared_receptor)
            )
            ifp_value = ifp_evaluator(
                docked_ligand_path=docked_prepared_path,
                docked_receptor_path=receptor_prepared_path,
                comparison_ligand_path=native_prepared_path,
                comparison_receptor_path=receptor_prepared_path,
                comparison_name="native_reference",
            )
        ifp = _validate_ifp_metrics(ifp_value)

    return {
        "status": "METRICS_COMPLETED",
        "artifact_bindings": {
            "top1": top1_binding,
            "vina_poses_sdf": poses_binding,
            "native_reference": native_binding,
            "prepared_receptor": receptor_binding,
        },
        "pose_recovery": {
            "top1_posebusters_valid": mode_audits[0]["posebusters_valid"],
            "top1_recorded_symmetry_rmsd_angstrom": recorded_rmsd,
            "top1_recomputed_symmetry_rmsd_angstrom": recomputed_rmsd,
            "top1_rmsd_absolute_difference_angstrom": difference,
            "top1_success": top1_success,
            "top5_oracle_success": top5_success,
            "top5_success": top5_success,
            "top5_mode_audits": mode_audits,
            "rmsd_threshold_angstrom": config.rmsd_threshold_angstrom,
        },
        "posebusters_full_report": posebusters_reports[0],
        "interaction_fingerprint": ifp,
        "prolif_ligand_preparation": prolif_preparation,
        "failure": None,
    }


def _metric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _aggregate_metric_summaries(cases: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [case for case in cases if case["status"] == "METRICS_COMPLETED"]

    def values(*keys: str) -> list[float]:
        output: list[float] = []
        for case in completed:
            value: Any = case["metrics"]
            for key in keys:
                value = value[key]
            output.append(float(value))
        return output

    return {
        "top1_symmetry_rmsd_angstrom": _metric_summary(
            values("pose_recovery", "top1_recomputed_symmetry_rmsd_angstrom")
        ),
        "posebusters_energy_ratio": _metric_summary(
            values("posebusters_full_report", "energy_ratio")
        ),
        "posebusters_mol_pred_energy": _metric_summary(
            values("posebusters_full_report", "mol_pred_energy")
        ),
        "posebusters_ensemble_avg_energy": _metric_summary(
            values("posebusters_full_report", "ensemble_avg_energy")
        ),
        "protein_ligand_pairwise_clash_count": _metric_summary(
            values(
                "posebusters_full_report",
                "protein_ligand_pairwise_clash_count",
            )
        ),
        "protein_ligand_volume_overlap": _metric_summary(
            values("posebusters_full_report", "protein_ligand_volume_overlap")
        ),
        "protein_ligand_minimum_distance_angstrom": _metric_summary(
            values(
                "posebusters_full_report",
                "protein_ligand_minimum_distance_angstrom",
            )
        ),
        "ifp_jaccard": _metric_summary(values("interaction_fingerprint", "jaccard")),
        "ifp_reference_interaction_recovery": _metric_summary(
            values(
                "interaction_fingerprint",
                "reference_interaction_recovery",
            )
        ),
        "ifp_predicted_interaction_precision": _metric_summary(
            values(
                "interaction_fingerprint",
                "predicted_interaction_precision",
            )
        ),
        "energy_semantics": (
            "PoseBusters energy_ratio and internal ligand energy columns are "
            "conformational diagnostics, not binding energies or free energies."
        ),
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def build_redock_regression(
    repo_root: Path,
    manifest_path: Path | str,
    *,
    config: RedockRegressionConfig | None = None,
    rmsd_evaluator: RMSDEvaluator | None = None,
    posebusters_evaluator: PoseBustersEvaluator | None = None,
    ifp_evaluator: IFPEvaluator | None = None,
    prolif_artifact_store: ArtifactStore | None = None,
) -> dict[str, Any]:
    """Build a deterministic result without mutating source redocking runs.

    Supplying ``prolif_artifact_store`` explicitly authorizes only receipted
    ligand-hydrogen preparation into that independent store.
    """

    config = config or RedockRegressionConfig()
    repo_root = repo_root.resolve()
    manifest, manifest_relative, manifest_file_sha256 = _validate_regression_manifest(
        repo_root, manifest_path
    )
    design = RegressionDesign(manifest["evaluation_design"])
    holdout_binding: dict[str, Any] | None = None
    selected_by_id: dict[str, dict[str, Any]] = {}
    if design is RegressionDesign.FROZEN_HOLDOUT:
        holdout_binding, selected_by_id = _validate_holdout_manifest(repo_root, manifest["holdout"])
        manifest_ids = [case["case_id"] for case in manifest["cases"]]
        if manifest_ids != list(selected_by_id):
            raise RegressionIntegrityError(
                "regression cases do not exactly match frozen holdout selection order"
            )

    rmsd = rmsd_evaluator or _default_spyrmsd
    posebusters = posebusters_evaluator or _default_posebusters
    ifp = ifp_evaluator or compare_prolif_paths
    case_results: list[dict[str, Any]] = []
    attempted = 0
    completed = 0
    failed = 0
    metric_failed = 0
    top1_successes = 0
    top5_successes = 0
    for case in manifest["cases"]:
        case_id = case["case_id"]
        pointer = case["result"]
        if pointer is None:
            case_results.append(
                {
                    "case_id": case_id,
                    "attempted": False,
                    "redock_status": "NOT_ATTEMPTED",
                    "status": "NOT_ATTEMPTED",
                    "metrics": None,
                    "failure": None,
                }
            )
            continue
        attempted += 1
        result_relative, result_path = _resolve_relative(
            repo_root,
            pointer["path"],
            f"cases.{case_id}.result.path",
        )
        result = _read_hash_bound_json(
            result_path,
            pointer["sha256"],
            f"redock result {case_id}",
        )
        if result.get("schema_version") not in {"1.1", "1.2"}:
            raise RegressionIntegrityError(f"{case_id} redock result schema is unsupported")
        if result.get("benchmark") != "redock":
            raise RegressionIntegrityError(f"{case_id} result is not a redock benchmark")
        redock_status = result.get("status")
        if redock_status == "FAILED":
            failed += 1
            case_results.append(
                {
                    "case_id": case_id,
                    "attempted": True,
                    "result": {
                        "path": result_relative.as_posix(),
                        "sha256": pointer["sha256"],
                    },
                    "redock_status": "FAILED",
                    "status": "REDOCK_FAILED",
                    "metrics": None,
                    "failure": _validated_redock_failure(result.get("failure"), repo_root),
                }
            )
            continue
        if redock_status != "COMPLETED":
            raise RegressionIntegrityError(f"{case_id} redock status must be COMPLETED or FAILED")
        if result.get("failure") is not None:
            raise RegressionIntegrityError(
                f"{case_id} completed redock result must not contain failure metadata"
            )
        completed += 1
        try:
            metrics = _evaluate_completed(
                repo_root=repo_root,
                result_path=result_path,
                result=result,
                selected_candidate=selected_by_id.get(case_id),
                config=config,
                rmsd_evaluator=rmsd,
                posebusters_evaluator=posebusters,
                ifp_evaluator=ifp,
                prolif_artifact_store=prolif_artifact_store,
            )
        except RegressionIntegrityError:
            raise
        except Exception as exc:
            metric_failed += 1
            case_results.append(
                {
                    "case_id": case_id,
                    "attempted": True,
                    "result": {
                        "path": result_relative.as_posix(),
                        "sha256": pointer["sha256"],
                    },
                    "redock_status": "COMPLETED",
                    "status": "METRIC_FAILED",
                    "metrics": None,
                    "failure": _sanitize_failure(exc, repo_root),
                }
            )
            continue
        if metrics["pose_recovery"]["top1_success"]:
            top1_successes += 1
        if metrics["pose_recovery"]["top5_success"]:
            top5_successes += 1
        case_results.append(
            {
                "case_id": case_id,
                "attempted": True,
                "result": {
                    "path": result_relative.as_posix(),
                    "sha256": pointer["sha256"],
                },
                "redock_status": "COMPLETED",
                "status": "METRICS_COMPLETED",
                "metrics": {
                    key: value for key, value in metrics.items() if key not in {"status", "failure"}
                },
                "failure": None,
            }
        )

    manifest_case_count = len(manifest["cases"])
    evaluator_bindings = {
        "symmetry_rmsd": _evaluator_identity(
            rmsd_evaluator,
            "protbind_agent.redock_regression._default_spyrmsd",
        ),
        "posebusters": _evaluator_identity(
            posebusters_evaluator,
            "protbind_agent.redock_regression._default_posebusters",
        ),
        "interaction_fingerprint": _evaluator_identity(
            ifp_evaluator,
            "protbind_agent.interaction_fingerprint.compare_prolif_paths",
        ),
    }
    runtime = _runtime_identity(evaluator_bindings=evaluator_bindings)
    mechanical_evaluation_complete = (
        design is RegressionDesign.FROZEN_HOLDOUT
        and holdout_binding is not None
        and manifest_case_count == TARGET_CASE_COUNT
        and attempted == TARGET_CASE_COUNT
        and completed == TARGET_CASE_COUNT
        and failed == 0
        and metric_failed == 0
    )
    gate_complete = (
        mechanical_evaluation_complete and runtime["evaluator_mode"] == "REAL_TOOLS"
    )
    config_payload = {
        **config.to_dict(),
        "prolif_ligand_preparation_mode": (
            "STRICT_EXISTING_EXPLICIT_HYDROGENS"
            if prolif_artifact_store is None
            else "RECEIPTED_LIGAND_ADDHS_AND_RECEPTOR_POCKET_CROP"
        ),
    }
    gate_blockers: list[str] = []
    if design is not RegressionDesign.FROZEN_HOLDOUT:
        gate_blockers.append("evaluation_design_is_not_frozen_holdout")
    if holdout_binding is None:
        gate_blockers.append("holdout_selection_is_not_hash_bound")
    if manifest_case_count != TARGET_CASE_COUNT:
        gate_blockers.append("manifest_does_not_contain_exactly_ten_cases")
    if attempted != TARGET_CASE_COUNT:
        gate_blockers.append("not_all_cases_attempted")
    if completed != TARGET_CASE_COUNT or failed:
        gate_blockers.append("not_all_redock_runs_completed")
    if metric_failed:
        gate_blockers.append("not_all_required_metrics_completed")
    if runtime["evaluator_mode"] != "REAL_TOOLS":
        gate_blockers.append("injected_evaluator_not_scientific_gate_evidence")
    core = {
        "schema_version": "1.1",
        "analysis": "PROTBIND_REDOCK_REGRESSION",
        "evaluation_design": design.value,
        "target_case_count": TARGET_CASE_COUNT,
        "mechanical_evaluation_complete": mechanical_evaluation_complete,
        "gate_complete": gate_complete,
        "gate_blockers": gate_blockers,
        "gate_requirements": {
            "evaluation_design": RegressionDesign.FROZEN_HOLDOUT.value,
            "exact_frozen_case_count": TARGET_CASE_COUNT,
            "holdout_selection_hash_bound": True,
            "all_cases_attempted": True,
            "all_redock_results_completed": True,
            "all_required_metrics_completed": True,
            "evaluator_mode": "REAL_TOOLS",
            "pilot_can_complete_gate": False,
        },
        "manifest": {
            "path": manifest_relative.as_posix(),
            "file_sha256": manifest_file_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "holdout": holdout_binding,
        "config": config_payload,
        "config_sha256": sha256_bytes(canonical_json_bytes(config_payload)),
        "runtime": runtime,
        "denominators": {
            "frozen": manifest_case_count,
            "attempted": attempted,
            "completed": completed,
            "failed": failed,
            "metric_failed": metric_failed,
            "not_attempted": manifest_case_count - attempted,
            "metrics_completed": completed - metric_failed,
        },
        "denominator_semantics": {
            "frozen": (
                "all cases committed by this regression manifest; for a retrospective "
                "pilot this is not a prospective holdout claim"
            ),
            "completed": "source redock status COMPLETED, before regression metric checks",
            "metrics_completed": "redock completed and every required regression metric succeeded",
            "pose_recovery_rate_denominator": (
                "all manifest cases, including failures and unattempted cases"
            ),
        },
        "pose_recovery_rates": {
            "top1": _rate(top1_successes, manifest_case_count),
            "top5": _rate(top5_successes, manifest_case_count),
        },
        "pose_recovery_rate_semantics": {
            "top1": "highest-ranked Vina record; independent PB-valid AND symmetry RMSD <= 2 A",
            "top5": "retrospective oracle over the first five Vina records, never a top-one claim",
        },
        "metric_summaries": _aggregate_metric_summaries(case_results),
        "cases": case_results,
        "scientific_boundaries": [
            "PILOT_RETROSPECTIVE results never complete the frozen-holdout gate.",
            "FAILED and unattempted redocks remain in the frozen denominator.",
            (
                "Injected evaluator callbacks can complete mechanical tests but can "
                "never complete the scientific gate."
            ),
            (
                "Missing explicit hydrogens cause ProLIF metric failure in strict mode."
                if prolif_artifact_store is None
                else "Any added ligand hydrogens use RDKit AddHs(addCoords=True) with "
                "hash-bound artifacts and unchanged-heavy-atom verification; the receptor "
                "is a receipted 8 A whole-residue union around both ligand poses with "
                "unchanged atom identities and coordinates."
            ),
            "PoseBusters energy_ratio and internal ligand energies are not binding energies "
            "or binding free energies.",
            "Top-5 is an oracle pose-recovery metric, not a prospective top-1 claim.",
            (
                "Content hashes detect changes relative to the supplied holdout manifest, "
                "but do not prove that case selection predated outcome inspection; a trusted "
                "external timestamp or signature is required for a prospective "
                "preregistration claim."
            ),
            (
                "Undefined ProLIF denominators fail required IFP metrics; they are "
                "never converted to perfect or zero agreement."
            ),
        ],
    }
    return {
        **core,
        "regression_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def persist_redock_regression(result: dict[str, Any], output: Path) -> None:
    """Atomically persist one already-built regression dictionary."""

    if not isinstance(result, dict) or not _is_sha256(result.get("regression_sha256")):
        raise ValueError("redock regression result is missing its hash")
    core = {key: value for key, value in result.items() if key != "regression_sha256"}
    if sha256_bytes(canonical_json_bytes(core)) != result["regression_sha256"]:
        raise ValueError("redock regression result hash mismatch")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
