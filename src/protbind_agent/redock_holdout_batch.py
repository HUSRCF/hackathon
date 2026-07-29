"""Result-blind, resumable execution of a frozen ten-case redocking holdout.

The holdout must already have been selected without docking results.  This
module binds that selection to exact input artifacts, code, executables, and
redocking parameters before the first case starts.  Existing case results are
only resumed after the same bindings are revalidated; an incompatible partial
run fails closed instead of being overwritten or silently substituted.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes, sha256_file
from .models import ArtifactRef
from .redock_benchmark import (
    RedockBenchmarkConfig,
    _protbind_code_receipt,
    run_redock_benchmark,
)

TARGET_CASE_COUNT = 10
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PROTOCOL_REVISION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SUPPORTED_REDOCK_SCHEMAS = frozenset({"1.1", "1.2"})
_UNSUPPORTED_FLAGS = (
    "contains_metal",
    "requires_cofactor",
    "pocket_altloc_ambiguous",
    "missing_pocket_heavy_atoms",
    "contains_nonstandard_protein_residue",
    "missing_backbone_atoms",
    "ligand_unspecified_stereo",
)


class RedockHoldoutBatchError(ValueError):
    """A frozen batch binding is missing, inconsistent, or unsafe to resume."""


@dataclass(frozen=True, slots=True)
class RedockHoldoutBatchConfig:
    redock: RedockBenchmarkConfig
    max_parallel_cases: int = 2
    protocol_revision: str | None = None

    def __post_init__(self) -> None:
        if type(self.max_parallel_cases) is not int or not 1 <= self.max_parallel_cases <= 2:
            raise ValueError("fixed holdout max_parallel_cases must be one or two")
        for name in (
            "vina",
            "mk_prepare_receptor",
            "mk_prepare_ligand",
            "mk_export",
        ):
            if getattr(self.redock, name) is None:
                raise ValueError(f"fixed holdout requires an explicit {name} executable")
        if self.protocol_revision is not None and (
            type(self.protocol_revision) is not str
            or _PROTOCOL_REVISION.fullmatch(self.protocol_revision) is None
        ):
            raise ValueError(
                "fixed holdout protocol_revision must be a lowercase identifier"
            )
        if self.redock.conservative_receptor_repair and self.protocol_revision is None:
            raise ValueError(
                "conservative repair holdout requires an explicit protocol_revision"
            )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _internal_hash(value: dict[str, Any], key: str, name: str) -> str:
    declared = value.get(key)
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None:
        raise RedockHoldoutBatchError(f"{name} has no valid {key}")
    body = {field: item for field, item in value.items() if field != key}
    if sha256_bytes(canonical_json_bytes(body)) != declared:
        raise RedockHoldoutBatchError(f"{name} {key} does not match its content")
    return declared


def _relative_to_repo(repo_root: Path, path: Path, name: str) -> PurePosixPath:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RedockHoldoutBatchError(f"{name} must be inside the repository root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RedockHoldoutBatchError(f"{name} is not a safe repo-relative path")
    return PurePosixPath(relative.as_posix())


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RedockHoldoutBatchError(f"{name} does not exist")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RedockHoldoutBatchError(f"{name} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise RedockHoldoutBatchError(f"{name} must be a JSON object")
    return value


def _artifact(value: Any, name: str) -> ArtifactRef:
    if not isinstance(value, dict):
        raise RedockHoldoutBatchError(f"{name} is not an artifact reference")
    try:
        return ArtifactRef.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise RedockHoldoutBatchError(f"{name} is not a valid artifact reference") from exc


def _load_holdout(
    repo_root: Path,
    holdout_path: Path,
    artifact_store: ArtifactStore,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    holdout_relative = _relative_to_repo(repo_root, holdout_path, "holdout manifest")
    holdout = _read_json(holdout_path, "holdout manifest")
    if holdout.get("schema_version") != "1.1":
        raise RedockHoldoutBatchError("fixed holdout requires schema 1.1")
    selection_hash = _internal_hash(holdout, "selection_hash", "holdout manifest")
    if holdout.get("requested_count") != TARGET_CASE_COUNT:
        raise RedockHoldoutBatchError("fixed holdout must request exactly ten cases")
    selected = holdout.get("selected")
    if not isinstance(selected, list) or len(selected) != TARGET_CASE_COUNT:
        raise RedockHoldoutBatchError("fixed holdout must select exactly ten cases")
    policy = holdout.get("eligibility_policy")
    if not isinstance(policy, dict) or policy.get("selection_reads_docking_results") is not False:
        raise RedockHoldoutBatchError("holdout policy must declare result-blind selection")

    case_ids: list[str] = []
    case_records: list[dict[str, Any]] = []
    case_slugs: set[str] = set()
    for index, candidate in enumerate(selected, start=1):
        if not isinstance(candidate, dict):
            raise RedockHoldoutBatchError("selected holdout entries must be objects")
        case_id = candidate.get("complex_id")
        if (
            not isinstance(case_id, str)
            or _CASE_ID.fullmatch(case_id) is None
            or case_id in case_ids
        ):
            raise RedockHoldoutBatchError("selected holdout case IDs are invalid or duplicated")
        case_slug = case_id.lower()
        if case_slug in case_slugs:
            raise RedockHoldoutBatchError("selected holdout case paths would collide")
        case_ids.append(case_id)
        case_slugs.add(case_slug)
        if any(candidate.get(flag) is not False for flag in _UNSUPPORTED_FLAGS):
            raise RedockHoldoutBatchError(f"{case_id} declares an unsupported scientific flag")
        if candidate.get("receptor_model_count") != 1:
            raise RedockHoldoutBatchError(f"{case_id} does not contain exactly one receptor model")
        receptor = _artifact(candidate.get("receptor"), f"{case_id}.receptor")
        ligand = _artifact(candidate.get("native_ligand"), f"{case_id}.native_ligand")
        receptor_path = artifact_store.resolve(receptor)
        ligand_path = artifact_store.resolve(ligand)
        if receptor.media_type != "chemical/x-pdb":
            raise RedockHoldoutBatchError(f"{case_id} receptor media type is unsupported")
        if ligand.media_type != "chemical/x-mdl-sdfile":
            raise RedockHoldoutBatchError(f"{case_id} native ligand media type is unsupported")
        case_records.append(
            {
                "ordinal": index,
                "case_id": case_id,
                "case_slug": case_slug,
                "license": candidate.get("license"),
                "receptor": receptor,
                "native_ligand": ligand,
                "receptor_path": receptor_path,
                "native_ligand_path": ligand_path,
            }
        )
    return (
        {
            "path": holdout_relative.as_posix(),
            "sha256": sha256_file(holdout_path),
            "selection_hash": selection_hash,
        },
        case_records,
    )


def _explicit_tool(value: str | Path | None, name: str) -> Path:
    if value is None:
        raise RedockHoldoutBatchError(f"fixed holdout requires explicit {name}")
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RedockHoldoutBatchError(f"fixed holdout {name} is not executable")
    return path


def _tool_bindings(config: RedockBenchmarkConfig) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for field, display in (
        ("vina", "vina"),
        ("mk_prepare_receptor", "mk_prepare_receptor.py"),
        ("mk_prepare_ligand", "mk_prepare_ligand.py"),
        ("mk_export", "mk_export.py"),
    ):
        path = _explicit_tool(getattr(config, field), display)
        bindings[field] = {
            "executable": path.name,
            "sha256": sha256_file(path),
        }
    python = Path(sys.executable).resolve()
    bindings["python"] = {
        "executable": python.name,
        "sha256": sha256_file(python),
    }
    return bindings


def _redock_config_payload(config: RedockBenchmarkConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "padding_angstrom": float(config.padding_angstrom),
        "exhaustiveness": config.exhaustiveness,
        "num_modes": config.num_modes,
        "energy_range": float(config.energy_range),
        "cpu": config.cpu,
        "timeout_seconds": float(config.timeout_seconds),
        "vina_scoring": "vina",
        "conservative_receptor_repair": config.conservative_receptor_repair,
        "repair_protected_radius_angstrom": float(
            config.repair_protected_radius_angstrom
        ),
        "restrained_sidechain_optimization": (
            config.restrained_sidechain_optimization
        ),
        "sidechain_optimization_iteration_limits": list(
            config.sidechain_optimization_iteration_limits
        ),
        "screening_calibration": None,
    }


def _expected_case_config(
    base: RedockBenchmarkConfig,
    record: dict[str, Any],
) -> RedockBenchmarkConfig:
    receptor: ArtifactRef = record["receptor"]
    ligand: ArtifactRef = record["native_ligand"]
    return replace(
        base,
        receptor_source=receptor.source,
        native_ligand_source=ligand.source,
        input_license=record["license"],
        calibration_target_id=None,
    )


def _result_config_payload(config: RedockBenchmarkConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "padding_angstrom": config.padding_angstrom,
        "exhaustiveness": config.exhaustiveness,
        "num_modes": config.num_modes,
        "energy_range": config.energy_range,
        "cpu": config.cpu,
        "vina_scoring": "vina",
        "receptor_source": config.receptor_source,
        "native_ligand_source": config.native_ligand_source,
        "input_license": config.input_license,
        "conservative_receptor_repair": config.conservative_receptor_repair,
        "repair_protected_radius_angstrom": (
            config.repair_protected_radius_angstrom
        ),
        "restrained_sidechain_optimization": (
            config.restrained_sidechain_optimization
        ),
        "sidechain_optimization_iteration_limits": list(
            config.sidechain_optimization_iteration_limits
        ),
        "screening_calibration": None,
    }


def _validate_result(
    result_path: Path,
    record: dict[str, Any],
    case_config: RedockBenchmarkConfig,
    *,
    expected_code_sha256: str,
    tool_bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = _read_json(result_path, f"{record['case_id']} redock result")
    if result.get("schema_version") not in _SUPPORTED_REDOCK_SCHEMAS:
        raise RedockHoldoutBatchError(f"{record['case_id']} redock schema is unsupported")
    if result.get("benchmark") != "redock" or result.get("status") not in {
        "COMPLETED",
        "FAILED",
    }:
        raise RedockHoldoutBatchError(f"{record['case_id']} has no terminal redock result")
    if result.get("code_sha256") != expected_code_sha256:
        raise RedockHoldoutBatchError(f"{record['case_id']} code binding does not match the plan")
    expected_config = _result_config_payload(case_config)
    if result.get("config") != expected_config:
        raise RedockHoldoutBatchError(f"{record['case_id']} config does not match the plan")
    config_sha = result.get("config_sha256")
    if config_sha is not None and config_sha != sha256_bytes(canonical_json_bytes(expected_config)):
        raise RedockHoldoutBatchError(f"{record['case_id']} config hash is inconsistent")

    inputs = result.get("input_commitments")
    receptor: ArtifactRef = record["receptor"]
    ligand: ArtifactRef = record["native_ligand"]
    expected_inputs = {
        "receptor": {
            "sha256": receptor.sha256,
            "size_bytes": receptor.size_bytes,
            "media_type": receptor.media_type,
            "source": receptor.source,
            "license": record["license"],
        },
        "native_ligand": {
            "sha256": ligand.sha256,
            "size_bytes": ligand.size_bytes,
            "media_type": ligand.media_type,
            "source": ligand.source,
            "license": record["license"],
        },
    }
    if inputs != expected_inputs:
        raise RedockHoldoutBatchError(
            f"{record['case_id']} input commitments do not match the frozen holdout"
        )

    observed_tools = result.get("toolchain")
    if observed_tools is not None:
        for name, binding in tool_bindings.items():
            observed = observed_tools.get(name) if isinstance(observed_tools, dict) else None
            if (
                not isinstance(observed, dict)
                or observed.get("executable") != binding["executable"]
                or observed.get("sha256") != binding["sha256"]
            ):
                raise RedockHoldoutBatchError(
                    f"{record['case_id']} {name} binding does not match the plan"
                )
    elif result["status"] == "COMPLETED":
        raise RedockHoldoutBatchError(f"{record['case_id']} completed without a toolchain receipt")
    return result


def _write_or_validate_plan(path: Path, plan: dict[str, Any]) -> None:
    if path.exists():
        observed = _read_json(path, "existing frozen run plan")
        if observed != plan:
            raise RedockHoldoutBatchError(
                "existing run-plan.json differs from the requested frozen batch"
            )
        return
    _atomic_json(path, plan)


def _progress_payload(
    plan_sha256: str,
    case_records: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    for record in case_records:
        case_id = record["case_id"]
        completed = results.get(case_id)
        cases.append(
            {
                "case_id": case_id,
                "state": "TERMINAL" if completed is not None else "PENDING",
                "result": completed,
            }
        )
    core = {
        "schema_version": "1.0",
        "kind": "PROTBIND_FROZEN_REDOCK_PROGRESS",
        "run_plan_sha256": plan_sha256,
        "terminal_count": len(results),
        "target_case_count": TARGET_CASE_COUNT,
        "cases": cases,
    }
    return {**core, "progress_sha256": sha256_bytes(canonical_json_bytes(core))}


def run_frozen_redock_holdout(
    repo_root: Path,
    holdout_path: Path,
    holdout_artifact_store: ArtifactStore,
    output: Path,
    *,
    config: RedockHoldoutBatchConfig,
    runner: Callable[..., dict[str, Any]] = run_redock_benchmark,
) -> dict[str, Any]:
    """Execute or safely resume all ten cases and emit a regression manifest."""

    repo_root = repo_root.resolve()
    holdout_path = holdout_path.resolve()
    output = output.resolve()
    output_relative = _relative_to_repo(repo_root, output, "batch output")
    artifact_store_relative = _relative_to_repo(
        repo_root, holdout_artifact_store.root, "holdout artifact store"
    )
    holdout_binding, case_records = _load_holdout(
        repo_root, holdout_path, holdout_artifact_store
    )
    tool_bindings = _tool_bindings(config.redock)
    code_receipt = _protbind_code_receipt()
    plan_cases = [
        {
            "ordinal": record["ordinal"],
            "case_id": record["case_id"],
            "receptor": record["receptor"].to_dict(),
            "native_ligand": record["native_ligand"].to_dict(),
            "result_path": (
                output_relative / "cases" / record["case_slug"] / "result.json"
            ).as_posix(),
        }
        for record in case_records
    ]
    plan_core = {
        "schema_version": "1.0",
        "kind": "PROTBIND_FROZEN_REDOCK_RUN_PLAN",
        "holdout": holdout_binding,
        "holdout_artifact_store": artifact_store_relative.as_posix(),
        "target_case_count": TARGET_CASE_COUNT,
        "execution": {
            "max_parallel_cases": config.max_parallel_cases,
            "case_substitution_allowed": False,
            "existing_results_require_full_binding_validation": True,
        },
        "protocol_revision": config.protocol_revision,
        "redock_config": _redock_config_payload(config.redock),
        "tool_bindings": tool_bindings,
        "code": code_receipt,
        "cases": plan_cases,
    }
    plan = {
        **plan_core,
        "run_plan_sha256": sha256_bytes(canonical_json_bytes(plan_core)),
    }
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "run-plan.json"
    _write_or_validate_plan(plan_path, plan)

    results: dict[str, dict[str, Any]] = {}

    def run_case(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        case_directory = output / "cases" / record["case_slug"]
        result_path = case_directory / "result.json"
        case_config = _expected_case_config(config.redock, record)
        resumed = result_path.is_file()
        if resumed:
            result = _validate_result(
                result_path,
                record,
                case_config,
                expected_code_sha256=code_receipt["manifest_sha256"],
                tool_bindings=tool_bindings,
            )
        else:
            if case_directory.exists():
                raise RedockHoldoutBatchError(
                    f"partial directory for {record['case_id']} has no result.json"
                )
            result = runner(
                record["receptor_path"],
                record["native_ligand_path"],
                case_directory,
                config=case_config,
            )
            result = _validate_result(
                result_path,
                record,
                case_config,
                expected_code_sha256=code_receipt["manifest_sha256"],
                tool_bindings=tool_bindings,
            )
        return result, resumed

    with ThreadPoolExecutor(max_workers=config.max_parallel_cases) as executor:
        pending = {executor.submit(run_case, record): record for record in case_records}
        for future in as_completed(pending):
            record = pending[future]
            result, resumed = future.result()
            result_path = output / "cases" / record["case_slug"] / "result.json"
            summary = {
                "path": _relative_to_repo(
                    repo_root, result_path, f"{record['case_id']} result"
                ).as_posix(),
                "sha256": sha256_file(result_path),
                "status": result["status"],
                "scientific_status": result.get("scientific_status"),
                "resumed": resumed,
            }
            if result["status"] == "COMPLETED":
                summary.update(
                    {
                        "top1_recovered": result.get("top1_recovered"),
                        "top5_recovered": result.get("top5_recovered"),
                    }
                )
            else:
                summary["failure"] = result.get("failure")
            results[record["case_id"]] = summary
            _atomic_json(
                output / "progress.json",
                _progress_payload(plan["run_plan_sha256"], case_records, results),
            )

    cases = [
        {
            "case_id": record["case_id"],
            "result": {
                "path": results[record["case_id"]]["path"],
                "sha256": results[record["case_id"]]["sha256"],
            },
        }
        for record in case_records
    ]
    manifest_core = {
        "schema_version": "1.0",
        "evaluation_design": "FROZEN_HOLDOUT",
        "target_case_count": TARGET_CASE_COUNT,
        "holdout": holdout_binding,
        "cases": cases,
    }
    regression_manifest = {
        **manifest_core,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest_core)),
    }
    manifest_path = output / "regression-manifest.json"
    if manifest_path.exists() and _read_json(
        manifest_path, "existing regression manifest"
    ) != regression_manifest:
        raise RedockHoldoutBatchError("existing regression manifest differs from batch results")
    _atomic_json(manifest_path, regression_manifest)

    ordered_results = [results[record["case_id"]] for record in case_records]
    summary_core = {
        "schema_version": "1.0",
        "kind": "PROTBIND_FROZEN_REDOCK_BATCH_RESULT",
        "run_plan": {
            "path": _relative_to_repo(repo_root, plan_path, "run plan").as_posix(),
            "sha256": sha256_file(plan_path),
            "run_plan_sha256": plan["run_plan_sha256"],
        },
        "regression_manifest": {
            "path": _relative_to_repo(
                repo_root, manifest_path, "regression manifest"
            ).as_posix(),
            "sha256": sha256_file(manifest_path),
            "manifest_sha256": regression_manifest["manifest_sha256"],
        },
        "target_case_count": TARGET_CASE_COUNT,
        "protocol_revision": config.protocol_revision,
        "terminal_count": len(ordered_results),
        "completed_count": sum(item["status"] == "COMPLETED" for item in ordered_results),
        "failed_count": sum(item["status"] == "FAILED" for item in ordered_results),
        "top1_recovered_count": sum(
            item.get("top1_recovered") is True for item in ordered_results
        ),
        "top5_recovered_count": sum(
            item.get("top5_recovered") is True for item in ordered_results
        ),
        "cases": [
            {"case_id": record["case_id"], **results[record["case_id"]]}
            for record in case_records
        ],
        "score_semantics": (
            "Vina values are docking model scores, not experimental affinities or binding "
            "free energies. Pose recovery is a known-site structural calibration only."
        ),
    }
    summary = {
        **summary_core,
        "batch_result_sha256": sha256_bytes(canonical_json_bytes(summary_core)),
    }
    _atomic_json(output / "batch-result.json", summary)
    return summary
