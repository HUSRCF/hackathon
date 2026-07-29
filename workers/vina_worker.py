#!/usr/bin/env python3
"""Deterministic offline Meeko/AutoDock Vina adapter for the DOCKED stage."""

from __future__ import annotations

import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protbind_agent.artifacts import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from protbind_agent.chemistry import smiles_formal_charge  # noqa: E402
from protbind_agent.models import ArtifactRef  # noqa: E402
from protbind_agent.worker_protocol import (  # noqa: E402
    WorkerRequest,
    WorkerResponse,
)
from protbind_agent.worker_sdk import WorkerFailure, serve_worker  # noqa: E402

ENGINE = "vina"
VINA_VERSION = "1.2.7"
MEEKO_VERSION = "0.7.1"
SCORE_SEMANTICS = "AutoDock Vina tool score only; not an experimental binding free energy"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_VERSION = re.compile(r"(?i)(?:autodock\s+)?vina[^0-9]*(\d+\.\d+\.\d+)")
_RESULT = re.compile(
    r"^REMARK\s+VINA\s+RESULT:\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s*$"
)
_MEEKO_SMILES = "REMARK SMILES "
_MEEKO_SMILES_INDEX = "REMARK SMILES IDX"
_MEEKO_H_PARENT = "REMARK H PARENT"
_MEEKO_INPUT_INDEX = "REMARK INDEX MAP"
_ALLOWED_ORGANIC_ATOMIC_NUMBERS = {5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
_PDBQT_ELEMENT_BY_TYPE = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "NS": "N",
    "O": "O",
    "OA": "O",
    "OS": "O",
    "F": "F",
    "P": "P",
    "S": "S",
    "SA": "S",
    "CL": "CL",
    "BR": "BR",
    "I": "I",
    "SI": "SI",
    "B": "B",
}
_STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def protbind_runtime_sha256() -> str:
    """Hash the host protocol implementation imported by this adapter."""

    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src" / "protbind_agent"
    sources = [
        (str(path.relative_to(repository_root)), sha256_file(path))
        for path in sorted(source_root.rglob("*.py"))
    ]
    if not sources:
        raise RuntimeError("ProtBind runtime source manifest is empty")
    return sha256_bytes(canonical_json_bytes(sources))


def composite_code_sha256(environment_lock_sha256: str, runtime_assets_sha256: str) -> str:
    """Bind adapter/host code, the frozen environment, and executable assets."""

    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "adapter_sha256": sha256_file(Path(__file__)),
                "protbind_runtime_sha256": protbind_runtime_sha256(),
                "environment_lock_sha256": environment_lock_sha256,
                "runtime_assets_sha256": runtime_assets_sha256,
                "vina_version": VINA_VERSION,
                "meeko_version": MEEKO_VERSION,
            }
        )
    )


def _reference(value: Any, name: str) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            f"{name} is not a valid artifact reference",
            recoverable=False,
        ) from exc


def _require_executable(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE",
            f"an explicit absolute {name} path is required",
            recoverable=True,
        )
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE",
            f"the configured {name} is not an executable local file",
            recoverable=True,
        )
    return path.resolve()


def _finite_vector(value: Any, name: str, *, positive: bool = False) -> list[float]:
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
            "INPUT_NOT_PREPARED",
            f"{name} must contain three finite numbers",
            recoverable=False,
        )
    result = [float(item) for item in value]
    if positive and any(item <= 0 for item in result):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            f"{name} values must be positive",
            recoverable=False,
        )
    return result


def _int_parameter(
    parameters: dict[str, Any], name: str, default: int, minimum: int, maximum: int
) -> int:
    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            f"{name} must be an integer in [{minimum}, {maximum}]",
            recoverable=False,
        )
    return value


def _float_parameter(
    parameters: dict[str, Any],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = parameters.get(name, default)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            f"{name} must be finite and in [{minimum}, {maximum}]",
            recoverable=False,
        )
    return float(value)


def _package_attestation(package: str, expected_version: str) -> dict[str, Any]:
    """Hash every non-cache installed package file from the imported package root."""

    try:
        distribution_version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError as exc:
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE",
            f"the pinned {package} distribution is not installed",
            recoverable=True,
        ) from exc
    if distribution_version != expected_version:
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            f"installed {package} version differs from the configured exact version",
            recoverable=False,
        )
    spec = importlib_util.find_spec(package)
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if len(locations) > 1:
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            f"the imported {package} package root is ambiguous",
            recoverable=False,
        )
    entries: list[tuple[str, str]] = []
    if len(locations) == 1:
        root = Path(locations[0]).resolve()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            entries.append((f"{package}/{path.relative_to(root).as_posix()}", sha256_file(path)))
    elif spec is not None and spec.origin is not None:
        module_path = Path(spec.origin).resolve()
        if module_path.is_file():
            entries.append((f"{package}/{module_path.name}", sha256_file(module_path)))
    if not entries:
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            f"the installed {package} package contains no attestable files",
            recoverable=False,
        )
    return {
        "version": distribution_version,
        "file_count": len(entries),
        "source_sha256": sha256_bytes(canonical_json_bytes(entries)),
    }


def _run_version(executable: Path, timeout: float) -> str:
    try:
        completed = subprocess.run(
            (str(executable), "--version"),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=_child_environment(1),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE",
            "the configured Vina executable could not report its version",
            recoverable=True,
        ) from exc
    version_text = "\n".join((completed.stdout, completed.stderr))
    match = _VERSION.search(version_text)
    if completed.returncode != 0 or match is None:
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            "the configured Vina executable has no parseable version",
            recoverable=False,
        )
    return match.group(1)


def runtime_asset_attestation(parameters: dict[str, Any]) -> dict[str, Any]:
    """Measure the exact local binaries and package code used by the worker."""

    vina = _require_executable(parameters.get("vina_executable"), "Vina executable")
    receptor = _require_executable(
        parameters.get("meeko_prepare_receptor_executable"),
        "Meeko receptor-preparation executable",
    )
    ligand = _require_executable(
        parameters.get("meeko_prepare_ligand_executable"),
        "Meeko ligand-preparation executable",
    )
    expected_vina = parameters.get("vina_version", VINA_VERSION)
    expected_meeko = parameters.get("meeko_version", MEEKO_VERSION)
    expected_rdkit = parameters.get("rdkit_version")
    expected_gemmi = parameters.get("gemmi_version")
    expected_numpy = parameters.get("numpy_version")
    expected_scipy = parameters.get("scipy_version")
    if expected_vina != VINA_VERSION or expected_meeko != MEEKO_VERSION:
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            f"this adapter is pinned to Vina {VINA_VERSION} and Meeko {MEEKO_VERSION}",
            recoverable=False,
        )
    if not isinstance(expected_rdkit, str) or not expected_rdkit.strip():
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "rdkit_version must pin the exact installed RDKit release",
            recoverable=False,
        )
    if not isinstance(expected_gemmi, str) or not expected_gemmi.strip():
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "gemmi_version must pin the exact installed Gemmi release",
            recoverable=False,
        )
    for package, expected in (("numpy", expected_numpy), ("scipy", expected_scipy)):
        if not isinstance(expected, str) or not expected.strip():
            raise WorkerFailure(
                "INVALID_PARAMETERS",
                f"{package}_version must pin the exact installed release",
                recoverable=False,
            )
    timeout = _float_parameter(parameters, "version_timeout_seconds", 15.0, 1.0, 60.0)
    measured_vina = _run_version(vina, timeout)
    if measured_vina != expected_vina:
        raise WorkerFailure(
            "TOOLCHAIN_MISMATCH",
            "Vina executable version differs from the pinned release",
            recoverable=False,
        )
    test_runtime = os.environ.get("PROTBIND_TEST_RUNTIME") == "1"
    if test_runtime:
        meeko_attestation = {
            "version": expected_meeko,
            "file_count": 0,
            "source_sha256": "fixture-runtime",
        }
        rdkit_attestation = {
            "version": expected_rdkit,
            "file_count": 0,
            "source_sha256": "fixture-runtime",
        }
        gemmi_attestation = {
            "version": expected_gemmi,
            "file_count": 0,
            "source_sha256": "fixture-runtime",
        }
        numpy_attestation = {
            "version": expected_numpy,
            "file_count": 0,
            "source_sha256": "fixture-runtime",
        }
        scipy_attestation = {
            "version": expected_scipy,
            "file_count": 0,
            "source_sha256": "fixture-runtime",
        }
    else:
        meeko_attestation = _package_attestation("meeko", expected_meeko)
        rdkit_attestation = _package_attestation("rdkit", expected_rdkit)
        gemmi_attestation = _package_attestation("gemmi", expected_gemmi)
        numpy_attestation = _package_attestation("numpy", expected_numpy)
        scipy_attestation = _package_attestation("scipy", expected_scipy)
    value = {
        "schema_version": "1.0",
        "python": {
            "version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "executable_name": Path(sys.executable).name,
            "executable_sha256": sha256_file(Path(sys.executable).resolve()),
        },
        "vina": {
            "version": measured_vina,
            "executable_name": vina.name,
            "executable_sha256": sha256_file(vina),
        },
        "meeko": {
            **meeko_attestation,
            "prepare_receptor_name": receptor.name,
            "prepare_receptor_sha256": sha256_file(receptor),
            "prepare_ligand_name": ligand.name,
            "prepare_ligand_sha256": sha256_file(ligand),
        },
        "rdkit": rdkit_attestation,
        "gemmi": gemmi_attestation,
        "numpy": numpy_attestation,
        "scipy": scipy_attestation,
        "official_runtime": False,
        "trust_level": (
            "test-fixture"
            if test_runtime
            else "hash-attested-local-without-reviewed-upstream-allowlist"
        ),
    }
    value["runtime_assets_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


def _read_stage_input(request: WorkerRequest, store: Any) -> dict[str, Any]:
    envelope = store.read_json(request.input)
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") not in {"1.0", "2.0"}
        or envelope.get("kind") != "protbind.stage-input"
        or envelope.get("stage") != "DOCKED"
    ):
        raise WorkerFailure(
            "INVALID_INPUT",
            "Vina requires a protbind.stage-input v1.0/v2.0 DOCKED envelope",
            recoverable=False,
        )
    previous = envelope.get("previous")
    if not isinstance(previous, dict) or previous.get("stage") not in {
        "SELECTED",
        "COFOLDED",
    }:
        raise WorkerFailure(
            "INVALID_INPUT",
            "Vina input must immediately follow SELECTED or legacy COFOLDED",
            recoverable=False,
        )
    scientific = previous.get("scientific_outputs")
    if not isinstance(scientific, list) or not scientific:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            f"{previous.get('stage')} scientific outputs are missing",
            recoverable=False,
        )
    upstream_reference = _reference(scientific[0], f"{previous.get('stage')} primary output")
    upstream = store.read_json(upstream_reference)
    legacy_cofold = previous.get("stage") == "COFOLDED"
    candidates = upstream.get("candidates") if isinstance(upstream, dict) else None
    if isinstance(upstream, dict) and candidates is None:
        candidates = upstream.get("selected_candidates")
    expected_kind = "protbind.cofold-bundle" if legacy_cofold else "protbind.selection-bundle"
    if (
        not isinstance(upstream, dict)
        or upstream.get("schema_version") not in {"1.0", "2.0"}
        or upstream.get("kind") != expected_kind
        or not isinstance(candidates, list)
        or not candidates
    ):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            f"{previous.get('stage')} primary output is not a non-empty {expected_kind}",
            recoverable=False,
        )
    supporting = envelope.get("supporting_artifacts")
    if not isinstance(supporting, dict):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "DOCKED input has no supporting artifacts",
            recoverable=False,
        )
    lock_reference = _reference(
        supporting.get("support_vina_environment_lock"),
        "support_vina_environment_lock",
    )
    store.resolve(lock_reference)
    batch_reference: ArtifactRef | None = None
    batch: dict[str, Any] | None = None
    batch_value = supporting.get("support_selection_batch")
    batch_name = "support_selection_batch"
    if batch_value is None:
        batch_value = supporting.get("support_openfold_batch")
        batch_name = "support_openfold_batch"
    if batch_value is not None:
        batch_reference = _reference(batch_value, batch_name)
        batch_value_parsed = store.read_json(batch_reference)
        if not isinstance(batch_value_parsed, dict):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", f"{batch_name} is not a JSON object", recoverable=False
            )
        batch = batch_value_parsed
    if legacy_cofold and (
        batch is None
        or batch.get("schema_version") != "1.0"
        or batch.get("kind") != "protbind.cofold-input-batch"
        or not isinstance(batch.get("cofold_candidates"), list)
    ):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "legacy COFOLDED docking requires support_openfold_batch",
            recoverable=False,
        )
    receptor_value = upstream.get("receptor")
    if receptor_value is None and batch is not None:
        receptor_value = batch.get("receptor")
    receptor = _reference(receptor_value, f"{previous.get('stage')} receptor")
    store.resolve(receptor)
    return {
        "envelope": envelope,
        "upstream_stage": previous.get("stage"),
        "upstream_reference": upstream_reference,
        "upstream": upstream,
        "upstream_candidates": candidates,
        "cofold_reference": upstream_reference if legacy_cofold else None,
        "cofold": upstream if legacy_cofold else None,
        "selection_reference": upstream_reference if not legacy_cofold else None,
        "selection": upstream if not legacy_cofold else None,
        "batch_reference": batch_reference,
        "batch": batch,
        "lock_reference": lock_reference,
        "receptor": receptor,
    }


def _legacy_candidate_inputs(parsed: dict[str, Any], store: Any) -> list[dict[str, Any]]:
    cofold_candidates = parsed["cofold"]["candidates"]
    batch = parsed["batch"]
    batch_candidates = batch["cofold_candidates"]
    microstates = batch.get("microstates")
    quick_vina = batch.get("quick_vina")
    if (
        not isinstance(microstates, list)
        or not isinstance(quick_vina, dict)
        or not isinstance(quick_vina.get("evaluated"), list)
    ):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "cofold batch lacks microstate or frozen quick-Vina evidence",
            recoverable=False,
        )
    microstate_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in microstates:
        if not isinstance(item, dict):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", "microstate entry is not an object", recoverable=False
            )
        key = (str(item.get("molecule_id")), str(item.get("microstate_id")))
        if key in microstate_by_key:
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", "microstate identities are not unique", recoverable=False
            )
        microstate_by_key[key] = item
    quick_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in quick_vina["evaluated"]:
        if not isinstance(item, dict):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", "quick-Vina entry is not an object", recoverable=False
            )
        key = (str(item.get("molecule_id")), str(item.get("microstate_id")))
        if key in quick_by_key:
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "selected quick-Vina microstate is not unique",
                recoverable=False,
            )
        quick_by_key[key] = item
    batch_by_molecule = {
        str(item.get("molecule_id")): item for item in batch_candidates if isinstance(item, dict)
    }
    if len(batch_by_molecule) != len(batch_candidates):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "cofold batch candidates have duplicate/invalid molecule IDs",
            recoverable=False,
        )
    results: list[dict[str, Any]] = []
    observed: set[str] = set()
    for position, candidate in enumerate(cofold_candidates):
        if not isinstance(candidate, dict):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", "cofold candidate is not an object", recoverable=False
            )
        molecule_id = candidate.get("molecule_id")
        microstate_id = candidate.get("microstate_id")
        candidate_id = candidate.get("candidate_id")
        if any(
            not isinstance(value, str) or not value or not _SAFE_ID.fullmatch(value)
            for value in (molecule_id, microstate_id, candidate_id)
        ):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "cofold candidate IDs must be non-empty path-safe strings",
                recoverable=False,
            )
        if molecule_id in observed:
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", "cofold molecule IDs cannot repeat", recoverable=False
            )
        observed.add(molecule_id)
        expected = batch_by_molecule.get(molecule_id)
        if (
            expected is None
            or position >= len(batch_candidates)
            or batch_candidates[position] is not expected
            or expected.get("candidate_id") != candidate_id
            or expected.get("microstate_id") != microstate_id
        ):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "cofold output does not preserve the frozen top-8 candidate order",
                recoverable=False,
            )
        key = (molecule_id, microstate_id)
        microstate = microstate_by_key.get(key)
        quick = quick_by_key.get(key)
        if microstate is None or quick is None:
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "cofold candidate is not bound to its microstate and quick-Vina box",
                recoverable=False,
            )
        smiles = microstate.get("canonical_isomeric_smiles")
        element_counts = microstate.get("heavy_element_counts")
        if (
            not isinstance(smiles, str)
            or not smiles
            or not isinstance(element_counts, dict)
            or not element_counts
            or any(
                not isinstance(name, str)
                or name != name.upper()
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
                for name, count in element_counts.items()
            )
        ):
            raise WorkerFailure(
                "UNSUPPORTED_CHEMISTRY",
                "selected microstate lacks canonical SMILES or heavy-element identity",
                recoverable=False,
            )
        structure = _reference(candidate.get("structure"), "cofold candidate structure")
        store.resolve(structure)
        results.append(
            {
                "candidate": candidate,
                "batch_candidate": expected,
                "microstate": microstate,
                "smiles": smiles,
                "heavy_element_counts": element_counts,
                "structure": structure,
                "box_center": _finite_vector(quick.get("box_center"), "box_center"),
                "box_size": _finite_vector(quick.get("box_size"), "box_size", positive=True),
            }
        )
    if len(results) != len(batch_candidates):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "Vina requires the complete frozen cofold candidate set",
            recoverable=False,
        )
    return results


def _selection_candidate_inputs(parsed: dict[str, Any], store: Any) -> list[dict[str, Any]]:
    selection_candidates = parsed["upstream_candidates"]
    batch = parsed.get("batch")
    batch_microstates = batch.get("microstates", []) if isinstance(batch, dict) else []
    batch_quick = batch.get("quick_vina", {}) if isinstance(batch, dict) else {}
    quick_evaluated = batch_quick.get("evaluated", []) if isinstance(batch_quick, dict) else []
    microstate_by_key = {
        (str(item.get("molecule_id")), str(item.get("microstate_id"))): item
        for item in batch_microstates
        if isinstance(item, dict)
    }
    quick_by_key = {
        (str(item.get("molecule_id")), str(item.get("microstate_id"))): item
        for item in quick_evaluated
        if isinstance(item, dict)
    }
    if len(microstate_by_key) != len(batch_microstates) or len(quick_by_key) != len(
        quick_evaluated
    ):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "selection support batch has duplicate or invalid microstate identities",
            recoverable=False,
        )
    results: list[dict[str, Any]] = []
    observed_candidates: set[str] = set()
    observed_microstates: set[tuple[str, str]] = set()
    for candidate in selection_candidates:
        if not isinstance(candidate, dict):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", "selection candidate is not an object", recoverable=False
            )
        candidate_id = candidate.get("candidate_id")
        molecule_id = candidate.get("molecule_id")
        microstate_id = candidate.get("microstate_id")
        if any(
            not isinstance(value, str) or not value or not _SAFE_ID.fullmatch(value)
            for value in (candidate_id, molecule_id, microstate_id)
        ):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "selection candidate IDs must be non-empty path-safe strings",
                recoverable=False,
            )
        key = (molecule_id, microstate_id)
        if candidate_id in observed_candidates or key in observed_microstates:
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "selection candidate and microstate identities must be unique",
                recoverable=False,
            )
        observed_candidates.add(candidate_id)
        observed_microstates.add(key)
        nested_microstate = candidate.get("microstate")
        if not isinstance(nested_microstate, dict):
            nested_microstate = {}
        frozen_microstate = microstate_by_key.get(key, {})
        smiles = candidate.get("canonical_isomeric_smiles")
        if smiles is None:
            smiles = nested_microstate.get("canonical_isomeric_smiles")
        if smiles is None:
            smiles = frozen_microstate.get("canonical_isomeric_smiles")
        element_counts = candidate.get("heavy_element_counts")
        if element_counts is None:
            element_counts = nested_microstate.get("heavy_element_counts")
        if element_counts is None:
            element_counts = frozen_microstate.get("heavy_element_counts")
        if (
            not isinstance(smiles, str)
            or not smiles
            or not isinstance(element_counts, dict)
            or not element_counts
            or any(
                not isinstance(name, str)
                or name != name.upper()
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
                for name, count in element_counts.items()
            )
        ):
            raise WorkerFailure(
                "UNSUPPORTED_CHEMISTRY",
                "selection candidate lacks canonical SMILES or heavy-element identity",
                recoverable=False,
            )
        formal_charge = candidate.get("formal_charge")
        if formal_charge is None:
            formal_charge = nested_microstate.get("formal_charge")
        if formal_charge is None:
            formal_charge = frozen_microstate.get("formal_charge")
        if formal_charge is not None and (
            not isinstance(formal_charge, int)
            or isinstance(formal_charge, bool)
            or formal_charge != smiles_formal_charge(smiles)
        ):
            raise WorkerFailure(
                "UNSUPPORTED_CHEMISTRY",
                "selection candidate formal charge differs from its SMILES",
                recoverable=False,
            )
        quick = quick_by_key.get(key, {})
        box = candidate.get("docking_box")
        if not isinstance(box, dict):
            box = {}
        center = candidate.get("box_center", box.get("center", quick.get("box_center")))
        size = candidate.get("box_size", box.get("size", quick.get("box_size")))
        structure_value = candidate.get("cofold_structure", candidate.get("structure"))
        structure = None
        if structure_value is not None:
            structure = _reference(structure_value, "optional selection cofold structure")
            store.resolve(structure)
        candidate_receptor_value = candidate.get("receptor")
        if (
            candidate_receptor_value is not None
            and _reference(candidate_receptor_value, "selection candidate receptor")
            != parsed["receptor"]
        ):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "selection candidate receptor differs from the frozen bundle receptor",
                recoverable=False,
            )
        results.append(
            {
                "candidate": candidate,
                "batch_candidate": None,
                "microstate": nested_microstate or frozen_microstate,
                "smiles": smiles,
                "heavy_element_counts": dict(sorted(element_counts.items())),
                "formal_charge": formal_charge,
                "structure": structure,
                "box_center": _finite_vector(center, "box_center"),
                "box_size": _finite_vector(size, "box_size", positive=True),
            }
        )
    return results


def _candidate_inputs(parsed: dict[str, Any], store: Any) -> list[dict[str, Any]]:
    if parsed["upstream_stage"] == "COFOLDED":
        return _legacy_candidate_inputs(parsed, store)
    return _selection_candidate_inputs(parsed, store)


def _child_environment(cpu: int) -> dict[str, str]:
    allowed = ("PATH", "LD_LIBRARY_PATH", "HOME", "TMPDIR")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.update(
        {
            "PROTBIND_NETWORK_POLICY": "deny",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": str(cpu),
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        }
    )
    return environment


def _run_command(argv: tuple[str, ...], *, timeout: float, code: str, cpu: int) -> float:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=_child_environment(cpu),
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkerFailure(
            "TOOL_TIMEOUT", f"{code} exceeded its configured timeout", recoverable=True
        ) from exc
    except OSError as exc:
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE", f"{code} could not be executed", recoverable=True
        ) from exc
    if completed.returncode != 0:
        raise WorkerFailure(
            "TOOL_EXECUTION_FAILED",
            f"{code} exited unsuccessfully; no docking score was accepted",
            recoverable=True,
        )
    return time.perf_counter() - started


def _validate_receptor_input(path: Path) -> bytes | None:
    if os.environ.get("PROTBIND_TEST_RUNTIME") == "1":
        if not path.read_bytes():
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", "fixture receptor is empty", recoverable=False
            )
        return None
    try:
        import gemmi
    except ImportError as exc:
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE",
            "Gemmi is required to validate the receptor before Meeko preparation",
            recoverable=True,
        ) from exc
    try:
        structure = gemmi.read_structure(str(path))
    except Exception as exc:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", "receptor structure is not parseable", recoverable=False
        ) from exc
    models = list(structure)
    if len(models) != 1:
        raise WorkerFailure(
            "UNSUPPORTED_TARGET",
            "v1 receptor preparation requires exactly one structural model",
            recoverable=False,
        )
    residue_count = 0
    for chain in models[0]:
        for residue in chain:
            if residue.name.upper() not in _STANDARD_AMINO_ACIDS:
                raise WorkerFailure(
                    "UNSUPPORTED_CHEMISTRY",
                    "receptor contains water, metal, ligand, or non-standard residue; "
                    "supply a conservatively prepared protein-only receptor",
                    recoverable=False,
                )
            residue_count += 1
            for atom in residue:
                position = atom.pos
                if not all(
                    math.isfinite(float(value)) for value in (position.x, position.y, position.z)
                ):
                    raise WorkerFailure(
                        "INPUT_NOT_PREPARED",
                        "receptor contains non-finite coordinates",
                        recoverable=False,
                    )
    if not residue_count:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", "receptor has no standard protein residues", recoverable=False
        )
    try:
        pdb = structure.make_pdb_string().encode()
    except Exception as exc:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "receptor cannot be represented as a deterministic protein-only PDB",
            recoverable=False,
        ) from exc
    if not pdb:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", "normalized receptor PDB is empty", recoverable=False
        )
    return pdb


def _fixture_sdf(name: str, elements: dict[str, int]) -> bytes:
    atoms = [element.title() for element, count in sorted(elements.items()) for _ in range(count)]
    lines = [
        name,
        "ProtBind test fixture; not scientific chemistry",
        "",
        f"{len(atoms):>3}  0  0  0  0  0  0  0  0  0999 V2000",
    ]
    lines.extend(
        f"{index * 0.1:10.4f}{0.0:10.4f}{0.0:10.4f} {element:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
        for index, element in enumerate(atoms)
    )
    lines.extend(("M  END", "$$$$", ""))
    return "\n".join(lines).encode()


def _ligand_sdf(smiles: str, elements: dict[str, int], seed: int, name: str, path: Path) -> None:
    if os.environ.get("PROTBIND_TEST_RUNTIME") == "1":
        path.write_bytes(_fixture_sdf(name, elements))
        return
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE",
            "RDKit is required to generate deterministic ligand conformers",
            recoverable=True,
        ) from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise WorkerFailure(
            "UNSUPPORTED_CHEMISTRY",
            "ligand SMILES is invalid or contains multiple disconnected components",
            recoverable=False,
        )
    observed: dict[str, int] = {}
    for atom in molecule.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if atomic_number not in _ALLOWED_ORGANIC_ATOMIC_NUMBERS:
            raise WorkerFailure(
                "UNSUPPORTED_CHEMISTRY",
                "ligand contains a metal or unsupported non-organic element",
                recoverable=False,
            )
        symbol = atom.GetSymbol().upper()
        observed[symbol] = observed.get(symbol, 0) + 1
    if observed != dict(sorted(elements.items())) or sum(observed.values()) > 100:
        raise WorkerFailure(
            "UNSUPPORTED_CHEMISTRY",
            "ligand SMILES does not preserve the frozen heavy-element identity",
            recoverable=False,
        )
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.useRandomCoords = False
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise WorkerFailure(
            "UNSUPPORTED_CHEMISTRY",
            "RDKit ETKDGv3 could not generate a ligand conformer",
            recoverable=False,
        )
    molecule.SetProp("_Name", name)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(molecule)
    finally:
        writer.close()
    if not path.is_file() or path.stat().st_size == 0:
        raise WorkerFailure("OUTPUT_INVALID", "RDKit emitted no ligand SDF", recoverable=False)


def _remark_pairs(text: str, prefix: str, *, required: bool) -> tuple[tuple[int, int], ...]:
    values: list[int] = []
    for line in text.splitlines():
        if line == prefix or line.startswith(prefix + " "):
            fields = line[len(prefix) :].split()
            try:
                values.extend(int(field) for field in fields)
            except ValueError as exc:
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    f"Meeko {prefix.removeprefix('REMARK ')} metadata is malformed",
                    recoverable=False,
                ) from exc
    if required and not values:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            f"Meeko {prefix.removeprefix('REMARK ')} metadata is missing",
            recoverable=False,
        )
    if len(values) % 2:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            f"Meeko {prefix.removeprefix('REMARK ')} metadata has an odd index count",
            recoverable=False,
        )
    pairs = tuple(zip(values[::2], values[1::2], strict=True))
    if any(left < 1 or right < 1 for left, right in pairs):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            f"Meeko {prefix.removeprefix('REMARK ')} indices must be positive",
            recoverable=False,
        )
    return pairs


def _pdbqt_atom_records(text: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        fields = line.split()
        if len(line) < 54 or len(fields) < 3:
            raise WorkerFailure(
                "OUTPUT_INVALID", "PDBQT atom record is truncated", recoverable=False
            )
        try:
            serial = int(line[6:11])
            coordinates = tuple(
                float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54))
            )
        except ValueError as exc:
            raise WorkerFailure(
                "OUTPUT_INVALID", "PDBQT atom identity/coordinates are invalid", recoverable=False
            ) from exc
        if serial in records or not all(math.isfinite(value) for value in coordinates):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "PDBQT atom serials must be unique and coordinates finite",
                recoverable=False,
            )
        records[serial] = {
            "coordinates": coordinates,
            "atom_type": fields[-1].upper(),
        }
    if not records:
        raise WorkerFailure("OUTPUT_INVALID", "PDBQT contains no atom records", recoverable=False)
    return records


def _meeko_identity_metadata(text: str) -> dict[str, Any]:
    """Parse and validate Meeko's lossless chemical-identity remarks.

    PDBQT itself does not retain bond orders, formal charges, carbon-bound
    hydrogens, or stereochemistry.  Meeko's SMILES/SMILES-IDX/H-PARENT records
    are consequently part of the scientific result, not optional comments.
    """

    smiles_values = [
        line[len(_MEEKO_SMILES) :].strip()
        for line in text.splitlines()
        if line.startswith(_MEEKO_SMILES) and not line.startswith(_MEEKO_SMILES_INDEX)
    ]
    if (
        len(smiles_values) != 1
        or not smiles_values[0]
        or any(character.isspace() for character in smiles_values[0])
    ):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko PDBQT must contain exactly one unambiguous SMILES identity record",
            recoverable=False,
        )
    smiles_index = _remark_pairs(text, _MEEKO_SMILES_INDEX, required=True)
    hydrogen_parent = _remark_pairs(text, _MEEKO_H_PARENT, required=False)
    input_index = _remark_pairs(text, _MEEKO_INPUT_INDEX, required=True)
    atoms = _pdbqt_atom_records(text)

    smiles_keys = [left for left, _ in smiles_index]
    smiles_serials = [right for _, right in smiles_index]
    h_parents = [left for left, _ in hydrogen_parent]
    hydrogen_serials = [right for _, right in hydrogen_parent]
    input_keys = [left for left, _ in input_index]
    input_serials = [right for _, right in input_index]
    if (
        len(smiles_keys) != len(set(smiles_keys))
        or len(smiles_serials) != len(set(smiles_serials))
        or len(input_keys) != len(set(input_keys))
        or len(input_serials) != len(set(input_serials))
        or len(hydrogen_serials) != len(set(hydrogen_serials))
        or set(smiles_serials) & set(hydrogen_serials)
    ):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko identity/index records are not bijective",
            recoverable=False,
        )
    if sorted(smiles_keys) != list(range(1, len(smiles_keys) + 1)) or any(
        parent not in set(smiles_keys) for parent in h_parents
    ):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko SMILES atom indices are incomplete or H-PARENT is unbound",
            recoverable=False,
        )
    mapped_serials = set(smiles_serials) | set(hydrogen_serials)
    chemical_serials = {
        serial
        for serial, atom in atoms.items()
        if not re.fullmatch(r"G[0-3]", str(atom["atom_type"]))
    }
    if mapped_serials != chemical_serials or set(input_serials) != chemical_serials:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko identity/index records do not cover every chemical PDBQT atom",
            recoverable=False,
        )
    if any(atoms[serial]["atom_type"] not in {"H", "HD", "HS"} for serial in hydrogen_serials):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko H-PARENT metadata points to a non-hydrogen atom",
            recoverable=False,
        )
    value = {
        "smiles": smiles_values[0],
        "smiles_index": [list(pair) for pair in smiles_index],
        "hydrogen_parent": [list(pair) for pair in hydrogen_parent],
        "input_index": [list(pair) for pair in input_index],
    }
    value["sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


def _chemical_identity(molecule: Any) -> dict[str, Any]:
    """Return an atom-order-independent RDKit chemical identity attestation."""

    from rdkit import Chem

    try:
        normalized = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
        Chem.SanitizeMol(normalized)
        Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
    except Exception as exc:
        raise WorkerFailure(
            "OUTPUT_INVALID", "ligand chemical identity cannot be sanitized", recoverable=False
        ) from exc
    atom_features: list[list[Any]] = []
    chiral_labels: list[str] = []
    for atom in normalized.GetAtoms():
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        atom_features.append(
            [
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                atom.GetIsAromatic(),
                atom.GetTotalNumHs(includeNeighbors=True),
                str(atom.GetChiralTag()),
                cip,
            ]
        )
        if cip:
            chiral_labels.append(cip)
    bond_features = sorted(
        [
            str(bond.GetBondType()),
            bond.GetIsAromatic(),
            str(bond.GetStereo()),
        ]
        for bond in normalized.GetBonds()
    )
    value = {
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            normalized, canonical=True, isomericSmiles=True
        ),
        "formal_charge": Chem.GetFormalCharge(normalized),
        "heavy_atom_count": normalized.GetNumHeavyAtoms(),
        "total_hydrogen_count": sum(
            atom.GetTotalNumHs(includeNeighbors=True) for atom in normalized.GetAtoms()
        ),
        "atom_features": sorted(atom_features),
        "bond_features": bond_features,
        "cip_labels": sorted(chiral_labels),
    }
    value["sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


def _read_sdf_molecules(data: bytes, description: str) -> list[Any]:
    try:
        from rdkit import Chem

        supplier = Chem.ForwardSDMolSupplier(
            io.BytesIO(data), sanitize=True, removeHs=False, strictParsing=True
        )
        molecules = list(supplier)
    except Exception as exc:
        raise WorkerFailure(
            "OUTPUT_INVALID", f"{description} is not a parseable SDF", recoverable=False
        ) from exc
    if not molecules or any(molecule is None for molecule in molecules):
        raise WorkerFailure(
            "OUTPUT_INVALID", f"{description} contains an invalid molecule", recoverable=False
        )
    return molecules


def _write_single_sdf(molecule: Any) -> bytes:
    try:
        from rdkit import Chem

        buffer = io.StringIO()
        writer = Chem.SDWriter(buffer)
        writer.write(molecule)
        writer.close()
        value = buffer.getvalue().encode()
    except Exception as exc:
        raise WorkerFailure(
            "OUTPUT_INVALID", "top Vina pose could not be serialized as SDF", recoverable=False
        ) from exc
    if not value:
        raise WorkerFailure("OUTPUT_INVALID", "top Vina pose SDF is empty", recoverable=False)
    return value


def _validate_mode_coordinate_mapping(
    molecule: Any, model_text: str, metadata: dict[str, Any]
) -> None:
    atoms = _pdbqt_atom_records(model_text)
    conformer = molecule.GetConformer()
    for smiles_index, serial in metadata["smiles_index"]:
        position = conformer.GetAtomPosition(smiles_index - 1)
        expected = atoms[serial]["coordinates"]
        if any(
            abs(observed - target) > 0.0011
            for observed, target in zip(position, expected, strict=True)
        ):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "docked SDF coordinates are not bound to the Meeko SMILES atom map",
                recoverable=False,
            )


def _extract_docked_sdf(
    *,
    input_sdf: bytes,
    vina_pdbqt: bytes,
    model_texts: list[str],
    scores: list[float],
    prepared_metadata: dict[str, Any] | None,
    fixture_name: str,
    fixture_elements: dict[str, int],
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Reconstruct all docked modes with Meeko and attest chemical identity."""

    if os.environ.get("PROTBIND_TEST_RUNTIME") == "1":
        top = _fixture_sdf(f"{fixture_name}-docked-top", fixture_elements)
        all_modes = _fixture_sdf(f"{fixture_name}-docked-all-modes", fixture_elements)
        return (
            top,
            all_modes,
            {
                "identity_method": "test-fixture-only",
                "mode_count": len(scores),
                "input_identity_sha256": "test-fixture",
                "output_identity_sha256": "test-fixture",
                "meeko_metadata_sha256": "test-fixture",
            },
        )
    if prepared_metadata is None:
        raise WorkerFailure(
            "OUTPUT_INVALID", "prepared ligand has no Meeko identity metadata", recoverable=False
        )
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate

        pdbqt_molecule = PDBQTMolecule(vina_pdbqt.decode("utf-8"))
        all_modes_text, failures = RDKitMolCreate.write_sd_string(pdbqt_molecule)
    except Exception as exc:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko could not reconstruct chemically complete docked poses",
            recoverable=False,
        ) from exc
    if failures or not all_modes_text:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko failed to reconstruct one or more docked poses",
            recoverable=False,
        )
    all_modes = all_modes_text.encode()
    input_molecules = _read_sdf_molecules(input_sdf, "prepared-input ligand")
    output_molecules = _read_sdf_molecules(all_modes, "Meeko docked-pose output")
    if (
        len(input_molecules) != 1
        or len(output_molecules) != len(scores)
        or len(model_texts) != len(scores)
    ):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "docked SDF/PDBQT mode counts do not match the Vina score count",
            recoverable=False,
        )
    input_identity = _chemical_identity(input_molecules[0])
    try:
        from rdkit import Chem

        metadata_molecule = Chem.MolFromSmiles(prepared_metadata["smiles"])
    except Exception as exc:
        raise WorkerFailure(
            "OUTPUT_INVALID", "Meeko SMILES identity cannot be parsed", recoverable=False
        ) from exc
    if metadata_molecule is None:
        raise WorkerFailure(
            "OUTPUT_INVALID", "Meeko SMILES identity cannot be parsed", recoverable=False
        )
    metadata_identity = _chemical_identity(metadata_molecule)
    if metadata_identity != input_identity:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko ligand preparation changed formal charge, bonds, hydrogens, or stereochemistry",
            recoverable=False,
        )
    output_identities: list[dict[str, Any]] = []
    for mode_index, (molecule, model_text, score) in enumerate(
        zip(output_molecules, model_texts, scores, strict=True), start=1
    ):
        identity = _chemical_identity(molecule)
        if identity != input_identity:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                f"Meeko docked SDF mode {mode_index} changed ligand chemical identity",
                recoverable=False,
            )
        _validate_mode_coordinate_mapping(molecule, model_text, prepared_metadata)
        if not molecule.HasProp("meeko"):
            raise WorkerFailure(
                "OUTPUT_INVALID", "docked SDF lacks Meeko score provenance", recoverable=False
            )
        try:
            properties = json.loads(molecule.GetProp("meeko"))
            exported_score = properties["free_energy"]
            if isinstance(exported_score, bool) or not math.isclose(
                float(exported_score), score, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "docked SDF score provenance differs from Vina PDBQT",
                recoverable=False,
            ) from exc
        output_identities.append(identity)
    return (
        _write_single_sdf(output_molecules[0]),
        all_modes,
        {
            "identity_method": "meeko-smiles-index-rdkit-graph-v1",
            "mode_count": len(output_molecules),
            "input_identity_sha256": input_identity["sha256"],
            "output_identity_sha256": output_identities[0]["sha256"],
            "meeko_metadata_sha256": prepared_metadata["sha256"],
            "canonical_isomeric_smiles": input_identity["canonical_isomeric_smiles"],
            "formal_charge": input_identity["formal_charge"],
            "heavy_atom_count": input_identity["heavy_atom_count"],
            "total_hydrogen_count": input_identity["total_hydrogen_count"],
        },
    )


def _atom_signature(line: str) -> tuple[str, str, str]:
    fields = line.split()
    if len(line) < 54 or len(fields) < 3:
        raise WorkerFailure("OUTPUT_INVALID", "PDBQT atom record is truncated", recoverable=False)
    try:
        coordinates = tuple(float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))
    except ValueError as exc:
        raise WorkerFailure(
            "OUTPUT_INVALID", "PDBQT atom coordinates are invalid", recoverable=False
        ) from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise WorkerFailure(
            "OUTPUT_INVALID", "PDBQT contains non-finite coordinates", recoverable=False
        )
    return line[12:16].strip(), line[17:20].strip(), fields[-1]


def _pdbqt_signatures(text: str) -> list[tuple[str, str, str]]:
    signatures = [
        _atom_signature(line) for line in text.splitlines() if line.startswith(("ATOM  ", "HETATM"))
    ]
    if not signatures:
        raise WorkerFailure("OUTPUT_INVALID", "PDBQT contains no atom records", recoverable=False)
    return signatures


def _pdbqt_invariants(text: str) -> list[tuple[str, ...]]:
    """Canonicalize every non-coordinate identity/torsion field in a ligand PDBQT."""

    records: list[tuple[str, ...]] = []
    topology_records = {"ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF"}
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            _atom_signature(line)
            records.append(
                (
                    line[:6].strip(),
                    line[6:11].strip(),
                    line[12:16].strip(),
                    line[16:17].strip(),
                    line[17:20].strip(),
                    line[21:22].strip(),
                    line[22:26].strip(),
                    line[26:27].strip(),
                    *line[54:].split(),
                )
            )
            continue
        fields = line.split()
        if fields and fields[0] in topology_records:
            records.append(tuple(fields))
    if not records:
        raise WorkerFailure(
            "OUTPUT_INVALID", "PDBQT invariant record set is empty", recoverable=False
        )
    return records


def _validate_prepared_pdbqt(path: Path, *, ligand: bool) -> list[tuple[str, str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise WorkerFailure(
            "OUTPUT_INVALID", "Meeko emitted no valid UTF-8 PDBQT", recoverable=False
        ) from exc
    signatures = _pdbqt_signatures(text)
    if ligand and not all(marker in text for marker in ("ROOT", "ENDROOT", "TORSDOF")):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko ligand PDBQT lacks its torsion-tree records",
            recoverable=False,
        )
    return signatures


def _pdbqt_heavy_elements(
    signatures: list[tuple[str, str, str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, atom_type in signatures:
        normalized = atom_type.upper()
        if normalized in {"H", "HD", "HS"}:
            continue
        if re.fullmatch(r"CG[0-3]", normalized):
            element = "C"
        elif re.fullmatch(r"G[0-3]", normalized):
            # Meeko macrocycle glue pseudoatoms carry no chemical element.
            continue
        else:
            element = _PDBQT_ELEMENT_BY_TYPE.get(normalized)
        if element is None:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "Meeko ligand PDBQT contains an unsupported atom type",
                recoverable=False,
            )
        counts[element] = counts.get(element, 0) + 1
    return dict(sorted(counts.items()))


def _receptor_heavy_identities(text: str, *, pdbqt: bool) -> Counter[tuple[str, ...]]:
    identities: Counter[tuple[str, ...]] = Counter()
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 54:
            raise WorkerFailure(
                "OUTPUT_INVALID", "receptor atom record is truncated", recoverable=False
            )
        atom_name = line[12:16].strip()
        if pdbqt:
            fields = line.split()
            if not fields:
                raise WorkerFailure(
                    "OUTPUT_INVALID", "receptor PDBQT atom type is missing", recoverable=False
                )
            if fields[-1].upper() in {"H", "HD"}:
                continue
        else:
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not element:
                letters = "".join(character for character in atom_name if character.isalpha())
                element = letters[:1].upper()
            if element == "H":
                continue
        identities[
            (
                atom_name,
                line[17:20].strip(),
                line[21:22].strip(),
                line[22:26].strip(),
                line[26:27].strip(),
            )
        ] += 1
    if not identities:
        raise WorkerFailure(
            "OUTPUT_INVALID", "receptor has no heavy-atom identities", recoverable=False
        )
    return identities


def _validate_receptor_binding(source_pdb: Path, prepared_pdbqt: Path) -> None:
    if os.environ.get("PROTBIND_TEST_RUNTIME") == "1":
        return
    try:
        source = source_pdb.read_text(encoding="utf-8")
        prepared = prepared_pdbqt.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise WorkerFailure(
            "OUTPUT_INVALID", "receptor identity files are unreadable", recoverable=False
        ) from exc
    if _receptor_heavy_identities(source, pdbqt=False) != _receptor_heavy_identities(
        prepared, pdbqt=True
    ):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko receptor PDBQT does not preserve protein heavy-atom/residue identity",
            recoverable=False,
        )


def _receptor_identity_sha256(text: str, *, pdbqt: bool) -> str:
    identities = _receptor_heavy_identities(text, pdbqt=pdbqt)
    value = [
        [*identity, count]
        for identity, count in sorted(identities.items(), key=lambda item: item[0])
    ]
    return sha256_bytes(canonical_json_bytes(value))


def _parse_vina_output(
    path: Path,
    expected_signatures: list[tuple[str, str, str]],
    expected_invariants: list[tuple[str, ...]],
    expected_metadata: dict[str, Any] | None,
    maximum_modes: int,
) -> tuple[list[float], bytes, list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise WorkerFailure(
            "OUTPUT_INVALID", "Vina emitted no valid UTF-8 PDBQT", recoverable=False
        ) from exc
    lines = text.splitlines()
    models: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("MODEL"):
            if current is not None:
                raise WorkerFailure(
                    "OUTPUT_INVALID", "Vina output has nested MODEL records", recoverable=False
                )
            current = [line]
        elif line.startswith("ENDMDL"):
            if current is None:
                raise WorkerFailure(
                    "OUTPUT_INVALID", "Vina output has unmatched ENDMDL", recoverable=False
                )
            current.append(line)
            models.append(current)
            current = None
        elif current is not None:
            current.append(line)
    if current is not None or not 1 <= len(models) <= maximum_modes:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Vina output has an invalid number of complete modes",
            recoverable=False,
        )
    scores: list[float] = []
    for model in models:
        result_lines = [line for line in model if line.startswith("REMARK VINA RESULT:")]
        if len(result_lines) != 1:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "every Vina mode must contain exactly one tool-emitted score",
                recoverable=False,
            )
        match = _RESULT.fullmatch(result_lines[0])
        if match is None:
            raise WorkerFailure(
                "OUTPUT_INVALID", "Vina score record is malformed", recoverable=False
            )
        values = tuple(float(match.group(index)) for index in range(1, 4))
        if not all(math.isfinite(value) for value in values):
            raise WorkerFailure(
                "OUTPUT_INVALID", "Vina score record is non-finite", recoverable=False
            )
        score = values[0]
        signatures = _pdbqt_signatures("\n".join(model))
        if signatures != expected_signatures:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "Vina pose atom identities differ from the Meeko ligand input",
                recoverable=False,
            )
        if _pdbqt_invariants("\n".join(model)) != expected_invariants:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "Vina pose changed ligand serial/residue/charge/torsion identity",
                recoverable=False,
            )
        if (
            expected_metadata is not None
            and _meeko_identity_metadata("\n".join(model)) != expected_metadata
        ):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "Vina pose changed Meeko SMILES/index chemical-identity metadata",
                recoverable=False,
            )
        scores.append(score)
    if scores != sorted(scores):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Vina modes are not ordered by ascending tool score",
            recoverable=False,
        )
    first_pose = ("\n".join(models[0]) + "\n").encode()
    model_texts = ["\n".join(model) + "\n" for model in models]
    return scores, first_pose, model_texts


def _deduplicate(references: list[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    observed: set[str] = set()
    return tuple(
        reference
        for reference in references
        if not (reference.sha256 in observed or observed.add(reference.sha256))
    )


def _dock_candidate(
    *,
    request: WorkerRequest,
    store: Any,
    parsed: dict[str, Any],
    item: dict[str, Any],
    rank: int,
    directory: Path,
    vina_executable: Path,
    ligand_executable: Path,
    receptor_pdbqt: Path,
    receptor_pdbqt_ref: ArtifactRef,
    receptor_preparation_input_ref: ArtifactRef,
    receptor_preparation_receipt_ref: ArtifactRef,
    runtime_engine: str,
    runtime_assets_sha256: str,
    cpu: int,
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    command_timeout: float,
) -> tuple[dict[str, Any], list[ArtifactRef], float, float]:
    """Dock one frozen candidate, returning only tool-derived score evidence."""

    candidate = item["candidate"]
    receptor = parsed["receptor"]
    candidate_digest = sha256_bytes(candidate["candidate_id"].encode())[:8]
    safe_stem = f"candidate-{rank:04d}-{candidate_digest}"
    sdf_path = directory / f"{safe_stem}.sdf"
    ligand_pdbqt = directory / f"{safe_stem}.pdbqt"
    vina_output = directory / f"{safe_stem}-vina.pdbqt"
    ligand_started = time.perf_counter()
    _ligand_sdf(
        item["smiles"],
        item["heavy_element_counts"],
        request.seed,
        safe_stem,
        sdf_path,
    )
    _run_command(
        (
            sys.executable,
            str(ligand_executable),
            "-i",
            str(sdf_path),
            "-o",
            str(ligand_pdbqt),
            "--add_index_map",
        ),
        timeout=command_timeout,
        code="Meeko ligand preparation",
        cpu=cpu,
    )
    ligand_signatures = _validate_prepared_pdbqt(ligand_pdbqt, ligand=True)
    ligand_pdbqt_text = ligand_pdbqt.read_text(encoding="utf-8")
    ligand_invariants = _pdbqt_invariants(ligand_pdbqt_text)
    ligand_metadata = (
        None
        if os.environ.get("PROTBIND_TEST_RUNTIME") == "1"
        else _meeko_identity_metadata(ligand_pdbqt_text)
    )
    if _pdbqt_heavy_elements(ligand_signatures) != item["heavy_element_counts"]:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "Meeko ligand PDBQT does not preserve the frozen heavy-element identity",
            recoverable=False,
        )
    ligand_prep_seconds = time.perf_counter() - ligand_started
    center = item["box_center"]
    size = item["box_size"]
    vina_argv = (
        str(vina_executable),
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--scoring",
        "vina",
        "--cpu",
        str(cpu),
        "--seed",
        str(request.seed),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--energy_range",
        str(energy_range),
        "--out",
        str(vina_output),
    )
    docking_seconds = _run_command(
        vina_argv,
        timeout=command_timeout,
        code="AutoDock Vina",
        cpu=cpu,
    )
    scores, first_pose, model_texts = _parse_vina_output(
        vina_output,
        ligand_signatures,
        ligand_invariants,
        ligand_metadata,
        num_modes,
    )
    top_pose_sdf, all_modes_sdf, extraction_details = _extract_docked_sdf(
        input_sdf=sdf_path.read_bytes(),
        vina_pdbqt=vina_output.read_bytes(),
        model_texts=model_texts,
        scores=scores,
        prepared_metadata=ligand_metadata,
        fixture_name=safe_stem,
        fixture_elements=item["heavy_element_counts"],
    )
    frozen_formal_charge = item.get("formal_charge")
    if (
        frozen_formal_charge is not None
        and os.environ.get("PROTBIND_TEST_RUNTIME") != "1"
        and extraction_details.get("formal_charge") != frozen_formal_charge
    ):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "docked ligand formal charge differs from the frozen microstate",
            recoverable=False,
        )
    sdf_ref = store.put_bytes(
        sdf_path.read_bytes(),
        media_type="chemical/x-mdl-sdfile",
        producer=f"{runtime_engine}.rdkit-etkdgv3",
        producer_version=str(request.parameters["rdkit_version"]),
        source=request.input.artifact_id,
    )
    ligand_ref = store.put_bytes(
        ligand_pdbqt.read_bytes(),
        media_type="chemical/x-pdbqt",
        producer=f"{runtime_engine}.meeko-ligand",
        producer_version=MEEKO_VERSION,
        source=request.input.artifact_id,
    )
    all_modes_pdbqt_ref = store.put_bytes(
        vina_output.read_bytes(),
        media_type="chemical/x-pdbqt",
        producer=f"{runtime_engine}.all-modes",
        producer_version=VINA_VERSION,
        source=request.input.artifact_id,
    )
    pose_pdbqt_ref = store.put_bytes(
        first_pose,
        media_type="chemical/x-pdbqt",
        producer=runtime_engine,
        producer_version=VINA_VERSION,
        source=request.input.artifact_id,
    )
    pose_sdf_ref = store.put_bytes(
        top_pose_sdf,
        media_type="chemical/x-mdl-sdfile",
        producer=f"{runtime_engine}.meeko-docked-top-mode",
        producer_version=MEEKO_VERSION,
        source=request.input.artifact_id,
    )
    all_modes_sdf_ref = store.put_bytes(
        all_modes_sdf,
        media_type="chemical/x-mdl-sdfile",
        producer=f"{runtime_engine}.meeko-docked-all-modes",
        producer_version=MEEKO_VERSION,
        source=request.input.artifact_id,
    )
    pose_extraction_receipt = store.put_json(
        {
            "schema_version": "2.0",
            "kind": "protbind.pose-extraction-receipt",
            "test_fixture": os.environ.get("PROTBIND_TEST_RUNTIME") == "1",
            "input_ligand": sdf_ref.to_dict(),
            "prepared_ligand": ligand_ref.to_dict(),
            "pose_sdf": pose_sdf_ref.to_dict(),
            "pose_pdbqt": pose_pdbqt_ref.to_dict(),
            "all_modes_sdf": all_modes_sdf_ref.to_dict(),
            "all_modes_pdbqt": all_modes_pdbqt_ref.to_dict(),
            "checks": {
                "meeko_metadata_complete": True,
                "meeko_metadata_unchanged": True,
                "chemical_identity_preserved": True,
                "element_isotope_preserved": True,
                "formal_charge_preserved": True,
                "bond_order_aromaticity_preserved": True,
                "stereochemistry_preserved": True,
                "hydrogen_count_preserved": True,
                "atom_mapping_bijective": True,
                "all_modes_reconstructed": True,
                "mode_count_matches_scores": True,
                "coordinates_bound_to_atom_map": True,
            },
            "details": extraction_details,
        },
        producer=f"{runtime_engine}.pose-extraction-receipt",
        producer_version=MEEKO_VERSION,
        source=request.input.artifact_id,
    )
    evidence_inputs = {
        "stage_envelope": request.input.to_dict(),
        "receptor": receptor.to_dict(),
        "prepared_receptor": receptor_pdbqt_ref.to_dict(),
        "receptor_preparation_input": receptor_preparation_input_ref.to_dict(),
        "receptor_preparation_receipt": receptor_preparation_receipt_ref.to_dict(),
        "ligand_sdf": sdf_ref.to_dict(),
        "prepared_ligand": ligand_ref.to_dict(),
        "pose": pose_sdf_ref.to_dict(),
        "pose_sdf": pose_sdf_ref.to_dict(),
        "pose_pdbqt": pose_pdbqt_ref.to_dict(),
        "all_modes": all_modes_pdbqt_ref.to_dict(),
        "all_modes_sdf": all_modes_sdf_ref.to_dict(),
        "all_modes_pdbqt": all_modes_pdbqt_ref.to_dict(),
        "pose_extraction_receipt": pose_extraction_receipt.to_dict(),
    }
    if parsed["upstream_stage"] == "SELECTED":
        evidence_inputs["selection_bundle"] = parsed["selection_reference"].to_dict()
    else:
        evidence_inputs["cofold_bundle"] = parsed["cofold_reference"].to_dict()
        evidence_inputs["cofold_structure"] = item["structure"].to_dict()
    evidence = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.tool-evidence",
            "tool": "vina",
            "tool_version": VINA_VERSION,
            "candidate_id": f"vina-{candidate['candidate_id']}",
            "parent_candidate_id": candidate["candidate_id"],
            "molecule_id": candidate["molecule_id"],
            "microstate_id": candidate["microstate_id"],
            "seed": request.seed,
            "metrics": {
                "score": scores[0],
                "score_semantics": SCORE_SEMANTICS,
                "mode_scores": scores,
                "mode_count": len(scores),
                "box_center": center,
                "box_size": size,
                "scoring": "vina",
                "cpu": cpu,
                "exhaustiveness": exhaustiveness,
                "num_modes": num_modes,
                "energy_range": energy_range,
                "heavy_element_counts": item["heavy_element_counts"],
                **(
                    {"formal_charge": frozen_formal_charge}
                    if frozen_formal_charge is not None
                    else {}
                ),
                "pose_extraction_method": extraction_details["identity_method"],
                "pose_identity_sha256": extraction_details["output_identity_sha256"],
            },
            "inputs": evidence_inputs,
            "runtime_assets_sha256": runtime_assets_sha256,
            "timings_seconds": {
                "ligand_preparation": ligand_prep_seconds,
                "vina_command": docking_seconds,
            },
        },
        producer=f"{runtime_engine}.evidence",
        producer_version=VINA_VERSION,
        source=request.input.artifact_id,
    )
    references = [
        sdf_ref,
        ligand_ref,
        all_modes_pdbqt_ref,
        pose_pdbqt_ref,
        pose_sdf_ref,
        all_modes_sdf_ref,
        pose_extraction_receipt,
        evidence,
    ]
    candidate_output = {
        "candidate_id": f"vina-{candidate['candidate_id']}",
        "molecule_id": candidate["molecule_id"],
        "parent_candidate_id": candidate["candidate_id"],
        "microstate_id": candidate["microstate_id"],
        "engine": runtime_engine,
        "seed": request.seed,
        "pose": pose_sdf_ref.to_dict(),
        "pose_sdf": pose_sdf_ref.to_dict(),
        "pose_pdbqt": pose_pdbqt_ref.to_dict(),
        "vina_score": scores[0],
        "vina_score_semantics": SCORE_SEMANTICS,
        "box_center": center,
        "box_size": size,
        "heavy_element_counts": item["heavy_element_counts"],
        **(
            {"formal_charge": frozen_formal_charge}
            if frozen_formal_charge is not None
            else {}
        ),
        "receptor": receptor.to_dict(),
        "prepared_receptor": receptor_pdbqt_ref.to_dict(),
        "receptor_preparation_receipt": receptor_preparation_receipt_ref.to_dict(),
        "prepared_ligand": ligand_ref.to_dict(),
        "all_modes": all_modes_pdbqt_ref.to_dict(),
        "all_modes_sdf": all_modes_sdf_ref.to_dict(),
        "all_modes_pdbqt": all_modes_pdbqt_ref.to_dict(),
        "pose_extraction_receipt": pose_extraction_receipt.to_dict(),
        "evidence": evidence.to_dict(),
    }
    if item["structure"] is not None:
        candidate_output["cofold_structure"] = item["structure"].to_dict()
    return (
        candidate_output,
        references,
        ligand_prep_seconds,
        docking_seconds,
    )


def _handler(request: WorkerRequest, store: Any) -> WorkerResponse:
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise WorkerFailure(
            "OFFLINE_POLICY_VIOLATION",
            "HSA_OVERRIDE_GFX_VERSION is forbidden",
            recoverable=False,
        )
    if request.seed == 0 or request.seed > 2**31 - 1:
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "deterministic Vina/ETKDG execution requires a seed in [1, 2147483647]",
            recoverable=False,
        )
    parsed = _read_stage_input(request, store)
    candidates = _candidate_inputs(parsed, store)
    cpu = _int_parameter(request.parameters, "cpu", 1, 1, 32)
    exhaustiveness = _int_parameter(request.parameters, "exhaustiveness", 32, 1, 256)
    num_modes = _int_parameter(request.parameters, "num_modes", 9, 1, 20)
    energy_range = _float_parameter(request.parameters, "energy_range", 3.0, 0.1, 100.0)
    command_timeout = _float_parameter(
        request.parameters, "command_timeout_seconds", 1800.0, 1.0, 86_400.0
    )
    if request.parameters.get("scoring", "vina") != "vina":
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "v1 evidence docking is pinned to the Vina scoring function",
            recoverable=False,
        )
    vina_executable = _require_executable(
        request.parameters.get("vina_executable"), "Vina executable"
    )
    receptor_executable = _require_executable(
        request.parameters.get("meeko_prepare_receptor_executable"),
        "Meeko receptor-preparation executable",
    )
    ligand_executable = _require_executable(
        request.parameters.get("meeko_prepare_ligand_executable"),
        "Meeko ligand-preparation executable",
    )
    attestation_started = time.perf_counter()
    runtime_attestation = runtime_asset_attestation(request.parameters)
    runtime_assets_sha256 = str(runtime_attestation["runtime_assets_sha256"])
    if request.provenance.weight_sha256 != runtime_assets_sha256:
        raise WorkerFailure(
            "ASSET_HASH_MISMATCH",
            "Vina/Meeko/RDKit runtime assets differ from request provenance",
            recoverable=False,
        )
    expected_revision = (
        f"autodock-vina-{VINA_VERSION}+meeko-{MEEKO_VERSION}+"
        f"rdkit-{request.parameters['rdkit_version']}+"
        f"gemmi-{request.parameters['gemmi_version']}+"
        f"numpy-{request.parameters['numpy_version']}+"
        f"scipy-{request.parameters['scipy_version']}"
    )
    if request.provenance.model_revision != expected_revision:
        raise WorkerFailure(
            "PROVENANCE_MISMATCH",
            "Vina toolchain revision differs from the exact configured versions",
            recoverable=False,
        )
    expected_code = composite_code_sha256(parsed["lock_reference"].sha256, runtime_assets_sha256)
    if request.provenance.code_sha256 != expected_code:
        raise WorkerFailure(
            "CODE_HASH_MISMATCH",
            "Vina adapter composite code identity does not match provenance",
            recoverable=False,
        )
    attestation_seconds = time.perf_counter() - attestation_started
    runtime_engine = (
        "test-fixture-vina"
        if os.environ.get("PROTBIND_TEST_RUNTIME") == "1"
        else "attested-local-autodock-vina"
    )
    warnings: list[str] = []
    if runtime_engine == "test-fixture-vina":
        warnings.append("fixture Vina/Meeko output; not scientific evidence")
    if any(any(size > 30.0 for size in item["box_size"]) for item in candidates):
        warnings.append(
            "one or more docking box dimensions exceed 30 A; interpret search coverage cautiously"
        )
    outputs: list[ArtifactRef] = []
    candidate_outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    receptor_prep_seconds = 0.0
    ligand_prep_seconds = 0.0
    docking_seconds = 0.0
    with tempfile.TemporaryDirectory(prefix="protbind-vina-") as temporary:
        directory = Path(temporary)
        receptor = parsed["receptor"]
        receptor_suffix = (
            ".cif"
            if receptor.media_type
            in {
                "chemical/x-mmcif",
                "chemical/mmcif",
            }
            else ".pdb"
        )
        if receptor_suffix == ".pdb" and receptor.media_type != "chemical/x-pdb":
            raise WorkerFailure(
                "INPUT_NOT_PREPARED",
                "Vina receptor must be a PDB or mmCIF artifact",
                recoverable=False,
            )
        receptor_input = directory / f"receptor{receptor_suffix}"
        receptor_input.write_bytes(store.read_bytes(receptor))
        normalized_receptor = _validate_receptor_input(receptor_input)
        if normalized_receptor is not None:
            receptor_input = directory / "receptor-normalized.pdb"
            receptor_input.write_bytes(normalized_receptor)
            receptor_suffix = ".pdb"
        receptor_preparation_input_ref = store.put_bytes(
            receptor_input.read_bytes(),
            media_type=("chemical/x-pdb" if receptor_suffix == ".pdb" else "chemical/x-mmcif"),
            producer=f"{runtime_engine}.receptor-preparation-input",
            producer_version=str(request.parameters["gemmi_version"]),
            source=receptor.artifact_id,
        )
        receptor_pdbqt = directory / "receptor.pdbqt"
        if receptor_suffix == ".pdb":
            receptor_argv = (
                sys.executable,
                str(receptor_executable),
                "--read_pdb",
                str(receptor_input),
                "--output_basename",
                str(directory / "receptor"),
                "--write_pdbqt",
                str(receptor_pdbqt),
            )
        else:
            receptor_argv = (
                sys.executable,
                str(receptor_executable),
                "-i",
                str(receptor_input),
                "-o",
                str(directory / "receptor"),
                "-p",
                str(receptor_pdbqt),
            )
        receptor_prep_seconds = _run_command(
            receptor_argv,
            timeout=command_timeout,
            code="Meeko receptor preparation",
            cpu=cpu,
        )
        _validate_prepared_pdbqt(receptor_pdbqt, ligand=False)
        _validate_receptor_binding(receptor_input, receptor_pdbqt)
        receptor_pdbqt_ref = store.put_bytes(
            receptor_pdbqt.read_bytes(),
            media_type="chemical/x-pdbqt",
            producer=f"{runtime_engine}.meeko-receptor",
            producer_version=MEEKO_VERSION,
            source=receptor.artifact_id,
        )
        if os.environ.get("PROTBIND_TEST_RUNTIME") == "1":
            receptor_input_identity = "test-fixture"
            receptor_pdbqt_identity = "test-fixture"
        else:
            receptor_input_identity = _receptor_identity_sha256(
                receptor_input.read_text(encoding="utf-8"), pdbqt=False
            )
            receptor_pdbqt_identity = _receptor_identity_sha256(
                receptor_pdbqt.read_text(encoding="utf-8"), pdbqt=True
            )
            if receptor_input_identity != receptor_pdbqt_identity:
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "receptor preparation identity digest changed after validation",
                    recoverable=False,
                )
        receptor_preparation_receipt_ref = store.put_json(
            {
                "schema_version": "2.0",
                "kind": "protbind.receptor-preparation-receipt",
                "test_fixture": os.environ.get("PROTBIND_TEST_RUNTIME") == "1",
                "receptor": receptor.to_dict(),
                "receptor_preparation_input": receptor_preparation_input_ref.to_dict(),
                "prepared_receptor": receptor_pdbqt_ref.to_dict(),
                "checks": {
                    "single_model_standard_protein": True,
                    "normalized_receptor_nonempty": True,
                    "prepared_pdbqt_nonempty": True,
                    "finite_coordinates": True,
                    "heavy_atom_residue_identity_preserved": True,
                },
                "details": {
                    "input_heavy_atom_residue_identity_sha256": receptor_input_identity,
                    "prepared_heavy_atom_residue_identity_sha256": receptor_pdbqt_identity,
                    "preparation_tool": "meeko",
                    "preparation_tool_version": MEEKO_VERSION,
                },
            },
            producer=f"{runtime_engine}.receptor-preparation-receipt",
            producer_version=MEEKO_VERSION,
            source=request.input.artifact_id,
        )
        outputs.extend(
            (
                receptor_preparation_input_ref,
                receptor_pdbqt_ref,
                receptor_preparation_receipt_ref,
            )
        )
        for rank, item in enumerate(candidates, start=1):
            candidate = item["candidate"]
            try:
                docked, references, ligand_seconds, vina_seconds = _dock_candidate(
                    request=request,
                    store=store,
                    parsed=parsed,
                    item=item,
                    rank=rank,
                    directory=directory,
                    vina_executable=vina_executable,
                    ligand_executable=ligand_executable,
                    receptor_pdbqt=receptor_pdbqt,
                    receptor_pdbqt_ref=receptor_pdbqt_ref,
                    receptor_preparation_input_ref=receptor_preparation_input_ref,
                    receptor_preparation_receipt_ref=receptor_preparation_receipt_ref,
                    runtime_engine=runtime_engine,
                    runtime_assets_sha256=runtime_assets_sha256,
                    cpu=cpu,
                    exhaustiveness=exhaustiveness,
                    num_modes=num_modes,
                    energy_range=energy_range,
                    command_timeout=command_timeout,
                )
            except WorkerFailure as exc:
                if exc.code not in {
                    "UNSUPPORTED_CHEMISTRY",
                    "TOOL_EXECUTION_FAILED",
                    "TOOL_TIMEOUT",
                    "OUTPUT_INVALID",
                }:
                    raise
                failure = {
                    "candidate_id": candidate["candidate_id"],
                    "parent_candidate_id": candidate["candidate_id"],
                    "molecule_id": candidate["molecule_id"],
                    "microstate_id": candidate["microstate_id"],
                    "engine": runtime_engine,
                    "receptor": receptor.to_dict(),
                    "seed": request.seed,
                    "box_center": item["box_center"],
                    "box_size": item["box_size"],
                    "heavy_element_counts": item["heavy_element_counts"],
                    "stage": "ligand_preparation_or_vina",
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "recoverable": exc.recoverable,
                    },
                }
                if item["structure"] is not None:
                    failure["cofold_structure"] = item["structure"].to_dict()
                failures.append(failure)
                continue
            candidate_outputs.append(docked)
            outputs.extend(references)
            ligand_prep_seconds += ligand_seconds
            docking_seconds += vina_seconds
    upstream_candidate_ids = [item["candidate"]["candidate_id"] for item in candidates]
    covered_candidate_ids = [
        *(item["parent_candidate_id"] for item in candidate_outputs),
        *(item["candidate_id"] for item in failures),
    ]
    if sorted(covered_candidate_ids) != sorted(upstream_candidate_ids) or len(
        covered_candidate_ids
    ) != len(set(covered_candidate_ids)):
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "docking successes and failures do not exactly cover the upstream candidates",
            recoverable=False,
        )
    if failures:
        warnings.append(
            f"{len(failures)} candidate(s) failed explicitly; no Vina score was fabricated"
        )
    runtime_metadata = store.put_json(
        {
            "schema_version": "1.0",
            "toolchain": runtime_attestation,
            "environment_lock": parsed["lock_reference"].to_dict(),
            "execution": {
                "device": "cpu",
                "cpu_threads": cpu,
                "seed": request.seed,
                "scoring": "vina",
                "exhaustiveness": exhaustiveness,
                "num_modes": num_modes,
                "energy_range": energy_range,
                "input_candidate_count": len(candidates),
                "successful_candidate_count": len(candidate_outputs),
                "failed_candidate_count": len(failures),
            },
        },
        producer=f"{runtime_engine}.run-metadata",
        producer_version=VINA_VERSION,
        source=request.input.artifact_id,
    )
    outputs.append(runtime_metadata)
    bundle_value = {
        "schema_version": ("2.0" if parsed["upstream_stage"] == "SELECTED" else "1.0"),
        "kind": "protbind.docking-bundle",
        "score_semantics": SCORE_SEMANTICS,
        "receptor": parsed["receptor"].to_dict(),
        "receptor_preparation_input": receptor_preparation_input_ref.to_dict(),
        "prepared_receptor": receptor_pdbqt_ref.to_dict(),
        "receptor_preparation_receipt": receptor_preparation_receipt_ref.to_dict(),
        "upstream_candidate_ids": upstream_candidate_ids,
        "candidate_count": len(candidate_outputs),
        "failure_count": len(failures),
        "candidates": candidate_outputs,
        "failures": failures,
        "run_metadata": runtime_metadata.to_dict(),
    }
    if parsed["upstream_stage"] == "SELECTED":
        bundle_value["upstream_selection_bundle"] = parsed["selection_reference"].to_dict()
    else:
        bundle_value["upstream_cofold_bundle"] = parsed["cofold_reference"].to_dict()
    bundle = store.put_json(
        bundle_value,
        producer=f"{runtime_engine}.bundle",
        producer_version=VINA_VERSION,
        source=request.input.artifact_id,
    )
    return WorkerResponse(
        job_id=request.job_id,
        engine=request.engine,
        outputs=(bundle, *_deduplicate(outputs)),
        provenance=request.provenance,
        timings_seconds={
            "runtime_attestation": attestation_seconds,
            "receptor_preparation": receptor_prep_seconds,
            "ligand_preparation": ligand_prep_seconds,
            "vina_command": docking_seconds,
        },
        peak_vram_bytes=None,
        warnings=tuple(warnings),
    )


if __name__ == "__main__":
    raise SystemExit(serve_worker(ENGINE, _handler))
