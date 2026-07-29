"""One-complex, offline redocking calibration with sealed native coordinates."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import __version__
from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes, sha256_file
from .preparation import (
    ReceptorPreparationUnsupportedError,
    conservative_heavy_atom_repair,
    restrained_sidechain_geometry_optimize,
)
from .redock_calibration import (
    KnownSiteCalibrationConfig,
    build_known_site_calibration_receipt,
    validate_known_site_calibration_receipt,
)
from .redocking import (
    build_redocking_case,
    native_derived_box,
    prepare_redocking_ligand,
    seal_validation_reference,
    symmetry_rmsd,
)
from .structure import (
    inspect_box_atom_overlap,
    inspect_declared_connections,
    inspect_structure,
)

_EXPECTED_VINA_VERSION = "1.2.7"
_EXPECTED_MEEKO_VERSION = "0.7.1"
_EXPECTED_POSEBUSTERS_VERSION = "0.6.5"
_EXPECTED_SPYRMSD_VERSION = "0.9.0"
_EXPECTED_RDKIT_VERSION = "2025.9.3"
_EXPECTED_GEMMI_VERSION = "0.7.5"
_EXPECTED_NUMPY_VERSION = "2.2.6"

_VINA_SCORE = re.compile(
    r"^REMARK\s+VINA\s+RESULT:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
    re.MULTILINE,
)
_DISCLAIMERS = (
    "Redocking is a pose-recovery calibration on a known holo site; it is not blind docking.",
    "The native ligand chemical identity and native-derived box are authorized inputs, but "
    "native ligand coordinates are withheld from Meeko and Vina until docking is committed.",
    "Vina scores are model scores, not experimental binding free energies or affinity values.",
    "Top-5 pose recovery is an oracle metric: it uses the best reference RMSD among the five "
    "highest-ranked Vina modes and is not a prospective top-1 claim.",
    "Pose recovery does not establish biological activity or experimental binding.",
)


class RedockBenchmarkError(RuntimeError):
    def __init__(self, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code


@dataclass(frozen=True, slots=True)
class RedockBenchmarkConfig:
    seed: int = 20260721
    padding_angstrom: float = 5.0
    exhaustiveness: int = 32
    num_modes: int = 9
    energy_range: float = 3.0
    cpu: int = 1
    timeout_seconds: float = 1800.0
    vina: str | Path | None = None
    mk_prepare_receptor: str | Path | None = None
    mk_prepare_ligand: str | Path | None = None
    mk_export: str | Path | None = None
    receptor_source: str | None = None
    native_ligand_source: str | None = None
    input_license: str | None = None
    conservative_receptor_repair: bool = False
    repair_protected_radius_angstrom: float = 6.0
    restrained_sidechain_optimization: bool = False
    sidechain_optimization_iteration_limits: tuple[int, ...] = (250, 1000, 5000)
    calibration_target_id: str | None = None
    calibration_required_rank: str = "top1"
    calibration_rmsd_threshold_angstrom: float = 2.0

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("redocking seed must fit in Vina's signed 32-bit range")
        if not math.isfinite(self.padding_angstrom) or self.padding_angstrom <= 0:
            raise ValueError("redocking padding must be finite and positive")
        if self.exhaustiveness < 1:
            raise ValueError("Vina exhaustiveness must be positive")
        if not 1 <= self.num_modes <= 100:
            raise ValueError("Vina num_modes must be in [1, 100]")
        if not math.isfinite(self.energy_range) or self.energy_range <= 0:
            raise ValueError("Vina energy_range must be finite and positive")
        if self.cpu < 1:
            raise ValueError("Vina cpu count must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("tool timeout must be finite and positive")
        for name, value in (
            ("receptor_source", self.receptor_source),
            ("native_ligand_source", self.native_ligand_source),
        ):
            if value is None:
                continue
            if value != value.strip() or len(value) > 512 or any(
                character in value for character in "\r\n"
            ):
                raise ValueError(f"{name} must be one stripped provenance identifier")
            if not value.startswith(("https://", "doi:", "dataset:", "pdb:", "rcsb:")):
                raise ValueError(
                    f"{name} must use https://, doi:, dataset:, pdb:, or rcsb: provenance"
                )
            if value.startswith("https://"):
                parsed = urlsplit(value)
                if parsed.username or parsed.password or parsed.query or parsed.fragment:
                    raise ValueError(
                        f"{name} HTTPS provenance must not contain credentials, query, or fragment"
                    )
        if self.input_license is not None and (
            self.input_license != self.input_license.strip()
            or len(self.input_license) > 256
            or any(character in self.input_license for character in "\r\n")
        ):
            raise ValueError("input_license must be one stripped identifier")
        if type(self.conservative_receptor_repair) is not bool:
            raise ValueError("conservative_receptor_repair must be boolean")
        if type(self.restrained_sidechain_optimization) is not bool:
            raise ValueError("restrained_sidechain_optimization must be boolean")
        if self.restrained_sidechain_optimization and not self.conservative_receptor_repair:
            raise ValueError(
                "restrained side-chain optimization requires conservative receptor repair"
            )
        if (
            type(self.sidechain_optimization_iteration_limits) is not tuple
            or not 1 <= len(self.sidechain_optimization_iteration_limits) <= 8
            or any(
                type(value) is not int or not 1 <= value <= 100_000
                for value in self.sidechain_optimization_iteration_limits
            )
            or tuple(sorted(set(self.sidechain_optimization_iteration_limits)))
            != self.sidechain_optimization_iteration_limits
        ):
            raise ValueError(
                "side-chain optimization iteration limits must be 1-8 strictly "
                "increasing positive integers no greater than 100000"
            )
        if (
            isinstance(self.repair_protected_radius_angstrom, bool)
            or not isinstance(self.repair_protected_radius_angstrom, int | float)
            or not math.isfinite(float(self.repair_protected_radius_angstrom))
            or float(self.repair_protected_radius_angstrom) <= 0
        ):
            raise ValueError("repair protected radius must be finite and positive")
        if self.calibration_target_id is not None:
            KnownSiteCalibrationConfig(
                target_id=self.calibration_target_id,
                required_rank=self.calibration_required_rank,
                rmsd_threshold_angstrom=self.calibration_rmsd_threshold_angstrom,
            )
        elif self.calibration_required_rank not in {"top1", "top5"}:
            raise ValueError("calibration required rank must be top1 or top5")
        if (
            isinstance(self.calibration_rmsd_threshold_angstrom, bool)
            or not isinstance(self.calibration_rmsd_threshold_angstrom, int | float)
            or not math.isfinite(self.calibration_rmsd_threshold_angstrom)
            or self.calibration_rmsd_threshold_angstrom <= 0
        ):
            raise ValueError("calibration RMSD threshold must be finite and positive")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_tool(value: str | Path | None, default: str, *, bundled: bool = False) -> Path:
    if value is None and bundled:
        candidate = _project_root() / "tools" / "bin" / default
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    requested = str(value) if value is not None else default
    if "/" in requested or "\\" in requested:
        candidate = Path(requested).expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RedockBenchmarkError(
                "preflight", "TOOL_UNAVAILABLE", f"{default} is not an executable file"
            )
        return candidate
    discovered = shutil.which(requested)
    if discovered is None:
        raise RedockBenchmarkError(
            "preflight", "TOOL_UNAVAILABLE", f"{requested} is not available on PATH"
        )
    return Path(discovered).resolve()


def _atomic_json(path: Path, value: Any) -> None:
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


def _validate_input_paths(receptor: Path, native_ligand: Path) -> None:
    # Content-addressed ArtifactStore object names intentionally have no suffix.
    # Keep rejecting an explicitly contradictory suffix; extensionless inputs are
    # parsed by the strict Gemmi/RDKit gates immediately after this availability check.
    if receptor.suffix and receptor.suffix.lower() != ".pdb":
        raise RedockBenchmarkError(
            "preflight", "RECEPTOR_FORMAT", "redocking benchmark receptor must be a PDB file"
        )
    if native_ligand.suffix and native_ligand.suffix.lower() not in {".sdf", ".mol"}:
        raise RedockBenchmarkError(
            "preflight", "LIGAND_FORMAT", "native ligand must be an SDF or MOL file"
        )
    for name, path in (("receptor", receptor), ("native ligand", native_ligand)):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RedockBenchmarkError(
                "preflight", "INPUT_UNAVAILABLE", f"{name} input is missing or empty"
            )


def _validate_protein_only_receptor(path: Path) -> None:
    inspection = inspect_structure(path, max_chains=2, max_residues=700)
    if inspection.metal_elements:
        raise RedockBenchmarkError(
            "receptor_validation",
            "METAL_SYSTEM",
            "v1 redocking does not support receptor metals: "
            + ", ".join(inspection.metal_elements),
        )
    if inspection.nonstandard_residues:
        raise RedockBenchmarkError(
            "receptor_validation",
            "NONSTANDARD_RECEPTOR",
            "receptor contains nonstandard protein residues: "
            + ", ".join(inspection.nonstandard_residues),
        )
    if inspection.alternate_location_atoms:
        raise RedockBenchmarkError(
            "receptor_validation",
            "UNRESOLVED_ALTLOC",
            "protein-only benchmark receptor must have alternate locations resolved",
        )
    if inspection.missing_backbone_residues:
        raise RedockBenchmarkError(
            "receptor_validation",
            "MISSING_BACKBONE",
            "receptor has residues missing N/CA/C: "
            + ", ".join(inspection.missing_backbone_residues),
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RedockBenchmarkError(
            "receptor_validation", "RECEPTOR_ENCODING", "receptor PDB is not UTF-8 text"
        ) from exc
    heterogens = sorted(
        {
            line[17:20].strip().upper()
            for line in lines
            if line.startswith("HETATM")
        }
    )
    if heterogens:
        raise RedockBenchmarkError(
            "receptor_validation",
            "RECEPTOR_NOT_PROTEIN_ONLY",
            "benchmark receptor contains HETATM components; prepare a dry protein-only PDB "
            "explicitly: " + ", ".join(heterogens),
        )
    connection = inspect_declared_connections(path)
    if connection.covalent_detected:
        raise RedockBenchmarkError(
            "receptor_validation",
            "COVALENT_SYSTEM",
            "receptor declares an unsupported covalent connection",
        )


def _canonicalize_pdb_atom_record_order(source: Path, destination: Path) -> dict[str, Any]:
    # Reorder only intact ATOM records by serial and return a byte-level receipt.
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    atom_indices = [index for index, line in enumerate(lines) if line.startswith(b"ATOM  ")]
    if not atom_indices:
        raise RedockBenchmarkError(
            "receptor_validation", "NO_PROTEIN_ATOMS", "receptor contains no ATOM records"
        )
    serials: list[int] = []
    for index in atom_indices:
        try:
            serials.append(int(lines[index][6:11]))
        except (ValueError, IndexError) as exc:
            raise RedockBenchmarkError(
                "receptor_validation",
                "INVALID_ATOM_SERIAL",
                "receptor has an unparseable PDB atom serial",
            ) from exc
    if len(set(serials)) != len(serials):
        raise RedockBenchmarkError(
            "receptor_validation",
            "DUPLICATE_ATOM_SERIAL",
            "receptor atom serials must be unique before safe record reordering",
        )
    changed = serials != sorted(serials)
    atom_lines = [lines[index] for index in atom_indices]
    if changed and any(
        line.startswith((b"ANISOU", b"MODEL ", b"ENDMDL")) for line in lines
    ):
        raise RedockBenchmarkError(
            "receptor_validation",
            "UNSAFE_RECORD_REORDER",
            "out-of-order atoms with ANISOU or multi-model records require explicit repair",
        )
    if changed:
        ordered = [line for _, line in sorted(zip(serials, atom_lines, strict=True))]
        rebuilt = list(lines)
        for index, atom_line in zip(atom_indices, ordered, strict=True):
            rebuilt[index] = atom_line
        output_bytes = b"".join(rebuilt)
    else:
        output_bytes = raw
    destination.write_bytes(output_bytes)
    output_atom_lines = [
        line for line in output_bytes.splitlines(keepends=True) if line.startswith(b"ATOM  ")
    ]
    if sorted(atom_lines) != sorted(output_atom_lines):
        raise RedockBenchmarkError(
            "receptor_validation",
            "ATOM_RECORD_INTEGRITY_FAILURE",
            "PDB record-order normalization changed the ATOM record multiset",
        )
    return {
        "schema_version": "1.0",
        "operation": "stable_atom_serial_record_order",
        "record_order_changed": changed,
        "atom_record_count": len(atom_lines),
        "atom_records_added": 0,
        "atom_records_removed": 0,
        "atom_record_bytes_changed": False,
        "coordinates_changed": False,
        "input_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "atom_record_multiset_sha256": sha256_bytes(b"".join(sorted(atom_lines))),
    }


def _box_overlaps_receptor(
    receptor: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> bool:
    return inspect_box_atom_overlap(
        receptor,
        center=center,
        size=size,
    ).protein_atom_overlap


def _ligand_heavy_atom_points(path: Path) -> tuple[tuple[float, float, float], ...]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RedockBenchmarkError(
            "receptor_repair",
            "RDKIT_UNAVAILABLE",
            "RDKit is required to protect the native-ligand pocket during receptor repair",
        ) from exc
    molecules = [
        molecule
        for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
        if molecule is not None
    ]
    if len(molecules) != 1 or molecules[0].GetNumConformers() != 1:
        raise RedockBenchmarkError(
            "receptor_repair",
            "INVALID_NATIVE_REFERENCE",
            "native reference must contain exactly one readable coordinate conformer",
        )
    molecule = molecules[0]
    conformer = molecule.GetConformer()
    points = tuple(
        tuple(float(value) for value in conformer.GetAtomPosition(atom.GetIdx()))
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    )
    if not points or any(not all(math.isfinite(value) for value in point) for point in points):
        raise RedockBenchmarkError(
            "receptor_repair",
            "INVALID_NATIVE_REFERENCE",
            "native reference has no finite heavy-atom coordinates",
        )
    return points


def _execute(
    *,
    stage: str,
    executable: Path,
    arguments: list[str],
    cwd: Path,
    logs: Path,
    timeout_seconds: float,
    allow_nonzero: bool = False,
    log_name: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RedockBenchmarkError(
            stage, "TOOL_TIMEOUT", f"{executable.name} exceeded the configured timeout"
        ) from exc
    except OSError as exc:
        raise RedockBenchmarkError(
            stage, "TOOL_EXECUTION_FAILED", f"{executable.name} could not be executed"
        ) from exc
    duration = time.perf_counter() - started
    log_stem = log_name or stage
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", log_stem) is None:
        raise ValueError("tool log_name must be a safe lowercase identifier")
    log_path = logs / f"{log_stem}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    record = {
        "stage": stage,
        "argv": [executable.name, *arguments],
        "returncode": completed.returncode,
        "duration_seconds": duration,
        "log": f"logs/{log_path.name}",
    }
    if completed.returncode != 0 and not allow_nonzero:
        raise RedockBenchmarkError(
            stage,
            "TOOL_NONZERO_EXIT",
            f"{executable.name} exited with code {completed.returncode}; see {record['log']}",
        )
    return record


def _retryable_rdkit_receptor_failure(log_text: str) -> bool:
    markers = (
        "AtomValenceException",
        "Explicit valence for atom",
        "Chem.SanitizeMol",
        "[RDKit] ERROR",
    )
    return any(marker in log_text for marker in markers)


def _retryable_sidechain_geometry_failure(code: str) -> bool:
    return code in {
        "SIDECHAIN_OPTIMIZATION_FAILED",
        "INVALID_OPTIMIZED_BOND_LENGTH",
        "OPTIMIZED_SIDECHAIN_STERIC_CLASH",
        "OPTIMIZATION_CHANGED_CHIRALITY",
    }


def _require_output(path: Path, stage: str, *, contains: bytes | None = None) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RedockBenchmarkError(
            stage, "MISSING_TOOL_OUTPUT", f"{path.name} was not produced"
        )
    if contains is not None and contains not in path.read_bytes():
        raise RedockBenchmarkError(
            stage, "INVALID_TOOL_OUTPUT", f"{path.name} lacks required structural records"
        )


def _split_sdf(path: Path, destination: Path, limit: int) -> tuple[Path, ...]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RedockBenchmarkError(
            "pose_export", "RDKIT_UNAVAILABLE", "RDKit is required to read exported poses"
        ) from exc
    records = list(
        Chem.ForwardSDMolSupplier(
            str(path), removeHs=False, sanitize=True, strictParsing=True
        )
    )
    if not records or any(molecule is None for molecule in records):
        raise RedockBenchmarkError(
            "pose_export", "UNREADABLE_EXPORTED_POSE", "Meeko exported an unreadable SDF record"
        )
    if len(records) > limit:
        raise RedockBenchmarkError(
            "pose_export", "TOO_MANY_EXPORTED_POSES", "Meeko exported more poses than requested"
        )
    paths: list[Path] = []
    for mode, molecule in enumerate(records, start=1):
        mode_path = destination / f"vina-mode-{mode:02d}.sdf"
        writer = Chem.SDWriter(str(mode_path))
        writer.write(molecule)
        writer.close()
        paths.append(mode_path)
    return tuple(paths)


def _boolean_posebusters_report(frame: Any) -> tuple[bool, dict[str, bool]]:
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
        raise ValueError("PoseBusters did not return one boolean result row")
    checks: dict[str, bool] = {}
    for column, value in frame.iloc[0].items():
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


def _posebusters_mode_checks(
    mode_path: Path,
    native_path: Path,
    receptor_path: Path,
) -> tuple[bool, dict[str, bool]]:
    try:
        from posebusters import PoseBusters

        frame = PoseBusters(
            config="redock", top_n=1, max_workers=0, chunk_size=1
        ).bust(
            mol_pred=str(mode_path),
            mol_true=str(native_path),
            mol_cond=str(receptor_path),
            full_report=False,
        )
        return _boolean_posebusters_report(frame)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RedockBenchmarkError(
            "validation",
            "POSEBUSTERS_FAILED",
            f"PoseBusters redock validation failed ({type(exc).__name__})",
        ) from exc


def _installed_version(distribution: str, expected: str) -> str:
    try:
        observed = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RedockBenchmarkError(
            "preflight",
            "TOOL_UNAVAILABLE",
            f"required distribution {distribution} is not installed",
        ) from exc
    if observed != expected:
        raise RedockBenchmarkError(
            "preflight",
            "TOOL_VERSION_MISMATCH",
            f"{distribution} {observed} does not match the pinned {expected}",
        )
    return observed


def _vina_version(path: Path, timeout_seconds: float) -> str:
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=min(timeout_seconds, 30.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RedockBenchmarkError(
            "preflight", "TOOL_UNAVAILABLE", "Vina version probe failed"
        ) from exc
    match = re.search(r"(?:AutoDock Vina\s+)?v?(\d+\.\d+\.\d+)", completed.stdout)
    if completed.returncode != 0 or match is None:
        raise RedockBenchmarkError(
            "preflight", "TOOL_VERSION_MISMATCH", "Vina did not report a parseable version"
        )
    observed = match.group(1)
    if observed != _EXPECTED_VINA_VERSION:
        raise RedockBenchmarkError(
            "preflight",
            "TOOL_VERSION_MISMATCH",
            f"Vina {observed} does not match the pinned {_EXPECTED_VINA_VERSION}",
        )
    return observed


def _tool_receipt(path: Path, version: str) -> dict[str, str]:
    return {
        "executable": path.name,
        "version": version,
        "sha256": sha256_file(path),
    }


def _protbind_code_receipt() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    entries = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(package_root.rglob("*.py"))
        if path.is_file()
    ]
    return {
        "schema_version": "1.0",
        "root": "src/protbind_agent",
        "selection": "all regular **/*.py files",
        "file_count": len(entries),
        "manifest_sha256": sha256_bytes(canonical_json_bytes(entries)),
        "entries": entries,
    }


def run_redock_benchmark(
    receptor: Path,
    native_ligand: Path,
    output: Path,
    *,
    config: RedockBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run one offline redocking experiment and always persist an explicit result."""

    config = config or RedockBenchmarkConfig()
    receptor = receptor.resolve()
    native_ligand = native_ligand.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = output / "artifacts"
    validation_files = output / "validation-only"
    logs = output / "logs"
    files.mkdir()
    validation_files.mkdir()
    logs.mkdir()
    store = ArtifactStore(output / "store")
    code_receipt = _protbind_code_receipt()
    result: dict[str, Any] = {
        "schema_version": "1.2",
        "benchmark": "redock",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "RUNNING",
        "scientific_status": "UNKNOWN",
        "code": code_receipt,
        "code_sha256": code_receipt["manifest_sha256"],
        "config": {
            "seed": config.seed,
            "padding_angstrom": config.padding_angstrom,
            "exhaustiveness": config.exhaustiveness,
            "num_modes": config.num_modes,
            "energy_range": config.energy_range,
            "cpu": config.cpu,
            "vina_scoring": "vina",
            "receptor_source": config.receptor_source
            or f"local-import:{receptor.name}",
            "native_ligand_source": config.native_ligand_source
            or f"local-import:{native_ligand.name}",
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
            "screening_calibration": (
                KnownSiteCalibrationConfig(
                    target_id=config.calibration_target_id,
                    required_rank=config.calibration_required_rank,
                    rmsd_threshold_angstrom=config.calibration_rmsd_threshold_angstrom,
                ).to_dict()
                if config.calibration_target_id is not None
                else None
            ),
        },
        "commands": [],
        "artifacts": {},
        "failure": None,
        "disclaimers": list(_DISCLAIMERS),
    }
    stage = "preflight"
    try:
        _validate_input_paths(receptor, native_ligand)
        result["input_commitments"] = {
            "receptor": {
                "sha256": sha256_file(receptor),
                "size_bytes": receptor.stat().st_size,
                "media_type": "chemical/x-pdb",
                "source": result["config"]["receptor_source"],
                "license": config.input_license,
            },
            "native_ligand": {
                "sha256": sha256_file(native_ligand),
                "size_bytes": native_ligand.stat().st_size,
                "media_type": "chemical/x-mdl-sdfile",
                "source": result["config"]["native_ligand_source"],
                "license": config.input_license,
            },
        }
        _validate_protein_only_receptor(receptor)
        vina = _resolve_tool(config.vina, "vina", bundled=True)
        mk_receptor = _resolve_tool(config.mk_prepare_receptor, "mk_prepare_receptor.py")
        mk_ligand = _resolve_tool(config.mk_prepare_ligand, "mk_prepare_ligand.py")
        mk_export = _resolve_tool(config.mk_export, "mk_export.py")
        vina_version = _vina_version(vina, config.timeout_seconds)
        meeko_version = _installed_version("meeko", _EXPECTED_MEEKO_VERSION)
        posebusters_version = _installed_version(
            "posebusters", _EXPECTED_POSEBUSTERS_VERSION
        )
        spyrmsd_version = _installed_version("spyrmsd", _EXPECTED_SPYRMSD_VERSION)
        rdkit_version = _installed_version("rdkit", _EXPECTED_RDKIT_VERSION)
        gemmi_version = _installed_version("gemmi", _EXPECTED_GEMMI_VERSION)
        numpy_version = _installed_version("numpy", _EXPECTED_NUMPY_VERSION)
        result["toolchain"] = {
            "python": _tool_receipt(
                Path(sys.executable).resolve(), sys.version.split()[0]
            ),
            "vina": _tool_receipt(vina, vina_version),
            "mk_prepare_receptor": _tool_receipt(mk_receptor, meeko_version),
            "mk_prepare_ligand": _tool_receipt(mk_ligand, meeko_version),
            "mk_export": _tool_receipt(mk_export, meeko_version),
            "posebusters": {"version": posebusters_version},
            "spyrmsd": {"version": spyrmsd_version},
            "rdkit": {"version": rdkit_version},
            "gemmi": {"version": gemmi_version},
            "numpy": {"version": numpy_version},
            "attestation_scope": (
                "Exact package versions and executable SHA-256 values from the local AIAA "
                "runtime; Python package source trees are not re-attested by this command."
            ),
        }
        result["config_sha256"] = sha256_bytes(canonical_json_bytes(result["config"]))
        result["toolchain_sha256"] = sha256_bytes(
            canonical_json_bytes(result["toolchain"])
        )
        result["run_identity_sha256"] = sha256_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "1.0",
                    "config_sha256": result["config_sha256"],
                    "toolchain_sha256": result["toolchain_sha256"],
                    "code_sha256": result["code_sha256"],
                }
            )
        )

        receptor_copy = files / "receptor-input.pdb"
        native_copy = validation_files / "native-reference.sdf"
        receptor_copy.write_bytes(receptor.read_bytes())
        native_copy.write_bytes(native_ligand.read_bytes())
        receptor_ref = store.import_file(
            receptor_copy,
            media_type="chemical/x-pdb",
            producer="protbind.redocking.local-input",
            source=result["config"]["receptor_source"],
            license=config.input_license,
        )
        native_ref = store.import_file(
            native_copy,
            media_type="chemical/x-mdl-sdfile",
            producer="protbind.redocking.validation-reference",
            source=result["config"]["native_ligand_source"],
            license=config.input_license,
        )
        result["artifacts"]["receptor_input"] = {
            **receptor_ref.to_dict(),
            "access_scope": "DOCKING_VISIBLE",
            "file": "artifacts/receptor-input.pdb",
        }
        result["artifacts"]["native_reference"] = {
            **native_ref.to_dict(),
            "access_scope": "VALIDATION_ONLY",
            "file": "validation-only/native-reference.sdf",
        }
        receptor_meeko_input = files / "receptor-meeko-input.pdb"
        order_receipt_payload = _canonicalize_pdb_atom_record_order(
            receptor_copy, receptor_meeko_input
        )
        receptor_meeko_input_ref = store.import_file(
            receptor_meeko_input,
            media_type="chemical/x-pdb",
            producer="protbind.redocking.pdb-record-order",
            producer_version=__version__,
            source=receptor_ref.artifact_id,
            license=receptor_ref.license,
        )
        order_receipt_payload.update(
            {
                "source_receptor": receptor_ref.artifact_id,
                "output_receptor": receptor_meeko_input_ref.artifact_id,
            }
        )
        order_receipt_ref = store.put_json(
            order_receipt_payload,
            producer="protbind.redocking.pdb-record-order-receipt",
            producer_version=__version__,
            source=receptor_ref.artifact_id,
            license=receptor_ref.license,
        )
        result["artifacts"].update(
            {
                "receptor_meeko_input": {
                    **receptor_meeko_input_ref.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                    "file": "artifacts/receptor-meeko-input.pdb",
                },
                "receptor_record_order_receipt": {
                    **order_receipt_ref.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                    "record_order_changed": order_receipt_payload[
                        "record_order_changed"
                    ],
                },
            }
        )

        stage = "ligand_initialization"
        box = native_derived_box(store, native_ref, padding_angstrom=config.padding_angstrom)
        ligand = prepare_redocking_ligand(store, native_ref, seed=config.seed)
        sealed = seal_validation_reference("redock", native_ref, ligand.identity)
        ligand_input = files / "ligand-independent-etkdg.sdf"
        ligand_input.write_bytes(store.read_bytes(ligand.ligand_3d))
        result["artifacts"]["ligand_independent_start"] = {
            **ligand.ligand_3d.to_dict(),
            "access_scope": "DOCKING_VISIBLE",
            "file": "artifacts/ligand-independent-etkdg.sdf",
        }
        result["artifacts"]["ligand_identity"] = {
            **ligand.identity.to_dict(),
            "access_scope": "DOCKING_VISIBLE",
        }
        result["artifacts"]["ligand_preparation_receipt"] = {
            **ligand.receipt.to_dict(),
            "access_scope": "DOCKING_VISIBLE",
        }
        result["artifacts"]["native_box_receipt"] = {
            **box.receipt.to_dict(),
            "access_scope": "DOCKING_VISIBLE",
        }
        receptor_docking_input = receptor_meeko_input
        receptor_docking_ref = receptor_meeko_input_ref
        receptor_repair = None
        if config.conservative_receptor_repair:
            stage = "receptor_repair"
            try:
                receptor_repair = conservative_heavy_atom_repair(
                    store,
                    receptor_meeko_input_ref,
                    protected_points=_ligand_heavy_atom_points(native_copy),
                    protected_radius=config.repair_protected_radius_angstrom,
                )
            except ReceptorPreparationUnsupportedError as exc:
                details = "; ".join(exc.details)
                message = str(exc) + (f" ({details})" if details else "")
                raise RedockBenchmarkError(stage, exc.code, message) from exc
            receptor_docking_ref = receptor_repair.structure
            receptor_docking_input = files / "receptor-heavy-atom-repaired.pdb"
            receptor_docking_input.write_bytes(store.read_bytes(receptor_docking_ref))
            _validate_protein_only_receptor(receptor_docking_input)
            result["receptor_repair"] = {
                "mode": "PDBFIXER_HEAVY_ATOM_ONLY",
                "added_heavy_atom_count": receptor_repair.added_heavy_atom_count,
                "removed_hydrogen_count": receptor_repair.removed_hydrogen_count,
                "missing_residue_count": receptor_repair.missing_residue_count,
                "missing_residues_rebuilt": False,
                "hydrogens_added": False,
                "original_heavy_atom_max_coordinate_delta_angstrom": (
                    receptor_repair.original_heavy_atom_max_coordinate_delta_angstrom
                ),
                "warnings": list(receptor_repair.warnings),
            }
            result["artifacts"].update(
                {
                    "receptor_heavy_atom_repaired": {
                        **receptor_docking_ref.to_dict(),
                        "access_scope": "DOCKING_VISIBLE",
                        "file": "artifacts/receptor-heavy-atom-repaired.pdb",
                    },
                    "receptor_repair_receipt": {
                        **receptor_repair.receipt.to_dict(),
                        "access_scope": "DOCKING_VISIBLE",
                    },
                }
            )
        if not _box_overlaps_receptor(receptor_docking_input, box.center, box.size):
            raise RedockBenchmarkError(
                stage,
                "BOX_RECEPTOR_FRAME_MISMATCH",
                "native-derived box contains no receptor atoms; inputs may use different frames",
            )

        stage = "receptor_preparation"
        receptor_pdbqt = files / "receptor.pdbqt"
        receptor_json = files / "receptor.json"
        receptor_prepared = files / "receptor-prepared.pdb"
        accepted_optimization = None
        optimization_attempts: list[dict[str, Any]] = []
        optimization_required = bool(
            config.restrained_sidechain_optimization
            and receptor_repair is not None
            and receptor_repair.added_heavy_atom_count > 0
        )
        if optimization_required:
            for attempt_index, iteration_limit in enumerate(
                config.sidechain_optimization_iteration_limits, start=1
            ):
                attempt_record: dict[str, Any] = {
                    "attempt": attempt_index,
                    "iteration_limit": iteration_limit,
                    "status": "RUNNING",
                }
                try:
                    optimized = restrained_sidechain_geometry_optimize(
                        store,
                        receptor_meeko_input_ref,
                        receptor_repair.structure,
                        iteration_limit=iteration_limit,
                    )
                except ReceptorPreparationUnsupportedError as exc:
                    attempt_record.update(
                        {
                            "status": "GEOMETRY_REJECTED",
                            "failure": {
                                "code": exc.code,
                                "message": str(exc),
                                "details": list(exc.details),
                            },
                        }
                    )
                    optimization_attempts.append(attempt_record)
                    if _retryable_sidechain_geometry_failure(
                        exc.code
                    ) and attempt_index < len(
                        config.sidechain_optimization_iteration_limits
                    ):
                        continue
                    details = "; ".join(exc.details)
                    message = str(exc) + (f" ({details})" if details else "")
                    raise RedockBenchmarkError(
                        "sidechain_geometry_optimization", exc.code, message
                    ) from exc

                optimized_path = (
                    files / f"receptor-sidechain-optimized-attempt-{attempt_index}.pdb"
                )
                optimized_path.write_bytes(store.read_bytes(optimized.structure))
                _validate_protein_only_receptor(optimized_path)
                attempt_stem = f"receptor-attempt-{attempt_index}"
                attempt_pdbqt = files / f"{attempt_stem}.pdbqt"
                attempt_json = files / f"{attempt_stem}.json"
                attempt_prepared = files / f"{attempt_stem}-prepared.pdb"
                attempt_args = [
                    "--read_pdb",
                    optimized_path.name,
                    "-o",
                    attempt_stem,
                    "-p",
                    attempt_pdbqt.name,
                    "-j",
                    attempt_json.name,
                    "--write_pdb",
                    attempt_prepared.name,
                    "--box_center",
                    *(f"{value:.8f}" for value in box.center),
                    "--box_size",
                    *(f"{value:.8f}" for value in box.size),
                ]
                command = _execute(
                    stage=stage,
                    executable=mk_receptor,
                    arguments=list(attempt_args),
                    cwd=files,
                    logs=logs,
                    timeout_seconds=config.timeout_seconds,
                    allow_nonzero=True,
                    log_name=f"receptor_preparation_attempt_{attempt_index}",
                )
                result["commands"].append(command)
                attempt_record.update(
                    {
                        "optimized_receptor": optimized.structure.to_dict(),
                        "optimization_receipt": optimized.receipt.to_dict(),
                        "meeko_returncode": command["returncode"],
                        "meeko_log": command["log"],
                    }
                )
                if command["returncode"] != 0:
                    log_text = (output / command["log"]).read_text(
                        encoding="utf-8", errors="replace"
                    )
                    retryable = _retryable_rdkit_receptor_failure(log_text)
                    attempt_record.update(
                        {
                            "status": "MEEKO_REJECTED",
                            "retryable_rdkit_chemistry_failure": retryable,
                        }
                    )
                    optimization_attempts.append(attempt_record)
                    if retryable and attempt_index < len(
                        config.sidechain_optimization_iteration_limits
                    ):
                        continue
                    code = (
                        "RDKIT_RECEPTOR_GRAPH_INVALID_AFTER_RETRIES"
                        if retryable
                        else "TOOL_NONZERO_EXIT"
                    )
                    raise RedockBenchmarkError(
                        stage,
                        code,
                        f"{mk_receptor.name} rejected constrained side-chain attempt "
                        f"{attempt_index}; see {command['log']}",
                    )

                _require_output(attempt_pdbqt, stage, contains=b"ATOM")
                _require_output(attempt_json, stage)
                _require_output(attempt_prepared, stage, contains=b"ATOM")
                shutil.copyfile(attempt_pdbqt, receptor_pdbqt)
                shutil.copyfile(attempt_json, receptor_json)
                shutil.copyfile(attempt_prepared, receptor_prepared)
                receptor_docking_input = optimized_path
                receptor_docking_ref = optimized.structure
                accepted_optimization = optimized
                attempt_record["status"] = "ACCEPTED"
                attempt_record["retryable_rdkit_chemistry_failure"] = False
                optimization_attempts.append(attempt_record)
                result["artifacts"].update(
                    {
                        "receptor_sidechain_optimized": {
                            **optimized.structure.to_dict(),
                            "access_scope": "DOCKING_VISIBLE",
                            "file": str(optimized_path.relative_to(output)),
                        },
                        "receptor_sidechain_optimization_receipt": {
                            **optimized.receipt.to_dict(),
                            "access_scope": "DOCKING_VISIBLE",
                        },
                    }
                )
                break
            result["sidechain_optimization"] = {
                "enabled": True,
                "required": True,
                "status": "ACCEPTED",
                "iteration_limits": list(
                    config.sidechain_optimization_iteration_limits
                ),
                "accepted_iteration_limit": accepted_optimization.iteration_limit,
                "attempts": optimization_attempts,
                "original_heavy_atoms_fixed": True,
                "meeko_rdkit_validation_required": True,
            }
        else:
            if config.restrained_sidechain_optimization:
                result["sidechain_optimization"] = {
                    "enabled": True,
                    "required": False,
                    "status": "NOT_REQUIRED",
                    "reason": "conservative repair added no receptor heavy atoms",
                    "iteration_limits": list(
                        config.sidechain_optimization_iteration_limits
                    ),
                    "attempts": [],
                }
            receptor_args = [
                "--read_pdb",
                receptor_docking_input.name,
                "-o",
                "receptor",
                "-p",
                receptor_pdbqt.name,
                "-j",
                receptor_json.name,
                "--write_pdb",
                receptor_prepared.name,
                "--box_center",
                *(f"{value:.8f}" for value in box.center),
                "--box_size",
                *(f"{value:.8f}" for value in box.size),
            ]
            result["commands"].append(
                _execute(
                    stage=stage,
                    executable=mk_receptor,
                    arguments=list(receptor_args),
                    cwd=files,
                    logs=logs,
                    timeout_seconds=config.timeout_seconds,
                )
            )
        _require_output(receptor_pdbqt, stage, contains=b"ATOM")
        _require_output(receptor_json, stage)
        _require_output(receptor_prepared, stage, contains=b"ATOM")
        prepared_receptor_ref = store.import_file(
            receptor_prepared,
            media_type="chemical/x-pdb",
            producer="meeko.mk_prepare_receptor",
            producer_version=meeko_version,
            source=receptor_docking_ref.artifact_id,
            license=receptor_ref.license,
        )
        receptor_pdbqt_ref = store.import_file(
            receptor_pdbqt,
            media_type="chemical/x-pdbqt",
            producer="meeko.mk_prepare_receptor",
            producer_version=meeko_version,
            source=receptor_docking_ref.artifact_id,
            license=receptor_ref.license,
        )
        receptor_json_ref = store.import_file(
            receptor_json,
            media_type="application/json",
            producer="meeko.mk_prepare_receptor",
            producer_version=meeko_version,
            source=receptor_docking_ref.artifact_id,
            license=receptor_ref.license,
        )
        receptor_receipt = store.put_json(
            {
                "schema_version": "1.0",
                "source_receptor": receptor_ref.artifact_id,
                "meeko_input_receptor": receptor_meeko_input_ref.artifact_id,
                "heavy_atom_repair_enabled": config.conservative_receptor_repair,
                "heavy_atom_repaired_receptor": (
                    receptor_repair.structure.artifact_id
                    if receptor_repair is not None
                    else receptor_meeko_input_ref.artifact_id
                ),
                "heavy_atom_repair_receipt": (
                    receptor_repair.receipt.artifact_id
                    if receptor_repair is not None
                    else None
                ),
                "restrained_sidechain_optimization_enabled": (
                    config.restrained_sidechain_optimization
                ),
                "accepted_sidechain_optimized_receptor": (
                    accepted_optimization.structure.artifact_id
                    if accepted_optimization is not None
                    else None
                ),
                "accepted_sidechain_optimization_receipt": (
                    accepted_optimization.receipt.artifact_id
                    if accepted_optimization is not None
                    else None
                ),
                "sidechain_optimization_attempts": optimization_attempts,
                "final_meeko_input_receptor": receptor_docking_ref.artifact_id,
                "record_order_receipt": order_receipt_ref.artifact_id,
                "prepared_receptor": prepared_receptor_ref.artifact_id,
                "receptor_pdbqt": receptor_pdbqt_ref.artifact_id,
                "meeko_json": receptor_json_ref.artifact_id,
                "box_receipt": box.receipt.artifact_id,
                "protein_only_input_required": True,
                "allow_bad_residues": False,
                "possible_cofactors_silently_removed": False,
            },
            producer="protbind.redocking.meeko-receptor-receipt",
            producer_version=__version__,
            source=receptor_ref.artifact_id,
            license=receptor_ref.license,
        )
        result["artifacts"].update(
            {
                "prepared_receptor": {
                    **prepared_receptor_ref.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                    "file": "artifacts/receptor-prepared.pdb",
                },
                "receptor_meeko_json": {
                    **receptor_json_ref.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                    "file": "artifacts/receptor.json",
                },
                "receptor_preparation_receipt": {
                    **receptor_receipt.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                },
            }
        )

        stage = "ligand_preparation"
        ligand_pdbqt = files / "ligand.pdbqt"
        result["commands"].append(
            _execute(
                stage=stage,
                executable=mk_ligand,
                arguments=[
                    "-i",
                    ligand_input.name,
                    "-o",
                    ligand_pdbqt.name,
                    "--add_index_map",
                ],
                cwd=files,
                logs=logs,
                timeout_seconds=config.timeout_seconds,
            )
        )
        _require_output(ligand_pdbqt, stage, contains=b"ROOT")
        ligand_pdbqt_ref = store.import_file(
            ligand_pdbqt,
            media_type="chemical/x-pdbqt",
            producer="meeko.mk_prepare_ligand",
            producer_version=meeko_version,
            source=ligand.identity.artifact_id,
            license=ligand.ligand_3d.license,
        )
        case = build_redocking_case(
            case_id="redock",
            receptor=prepared_receptor_ref,
            receptor_preparation_receipt=receptor_receipt,
            ligand=ligand,
            box=box,
            sealed_reference=sealed,
            seed=config.seed,
        )
        case_payload = case.to_docking_dict()
        case_bytes = canonical_json_bytes(case_payload)
        if native_ref.sha256.encode() in case_bytes:
            raise RedockBenchmarkError(
                stage, "REFERENCE_LEAK", "docking-visible case exposes native artifact identity"
            )
        case_path = files / "docking-visible-case.json"
        case_path.write_bytes(case_bytes)
        case_ref = store.import_file(
            case_path,
            media_type="application/json",
            producer="protbind.redocking.docking-visible-case",
            license=config.input_license,
        )

        stage = "vina"
        poses_pdbqt = files / "vina-poses.pdbqt"
        vina_args = [
            "--receptor",
            receptor_pdbqt.name,
            "--ligand",
            ligand_pdbqt.name,
            "--center_x",
            f"{box.center[0]:.8f}",
            "--center_y",
            f"{box.center[1]:.8f}",
            "--center_z",
            f"{box.center[2]:.8f}",
            "--size_x",
            f"{box.size[0]:.8f}",
            "--size_y",
            f"{box.size[1]:.8f}",
            "--size_z",
            f"{box.size[2]:.8f}",
            "--scoring",
            "vina",
            "--exhaustiveness",
            str(config.exhaustiveness),
            "--num_modes",
            str(config.num_modes),
            "--energy_range",
            str(config.energy_range),
            "--cpu",
            str(config.cpu),
            "--seed",
            str(config.seed),
            "--out",
            poses_pdbqt.name,
        ]
        result["commands"].append(
            _execute(
                stage=stage,
                executable=vina,
                arguments=vina_args,
                cwd=files,
                logs=logs,
                timeout_seconds=config.timeout_seconds,
            )
        )
        _require_output(poses_pdbqt, stage, contains=b"VINA RESULT")
        scores = [
            {
                "mode": mode,
                "vina_score": float(match[0]),
                "rmsd_lb_from_best": float(match[1]),
                "rmsd_ub_from_best": float(match[2]),
            }
            for mode, match in enumerate(_VINA_SCORE.findall(poses_pdbqt.read_text()), start=1)
        ]
        if not scores or any(
            not all(math.isfinite(value) for value in item.values() if isinstance(value, float))
            for item in scores
        ):
            raise RedockBenchmarkError(
                stage, "INVALID_VINA_SCORES", "Vina output has no finite mode scores"
            )

        stage = "pose_export"
        poses_sdf = files / "vina-poses.sdf"
        result["commands"].append(
            _execute(
                stage=stage,
                executable=mk_export,
                arguments=[poses_pdbqt.name, "-s", poses_sdf.name],
                cwd=files,
                logs=logs,
                timeout_seconds=config.timeout_seconds,
            )
        )
        _require_output(poses_sdf, stage, contains=b"$$$$")
        mode_paths = _split_sdf(poses_sdf, files, config.num_modes)
        if len(mode_paths) != len(scores):
            raise RedockBenchmarkError(
                stage,
                "POSE_SCORE_COUNT_MISMATCH",
                "Meeko SDF mode count differs from Vina score count",
            )
        poses_pdbqt_ref = store.import_file(
            poses_pdbqt,
            media_type="chemical/x-pdbqt",
            producer="autodock-vina",
            producer_version=vina_version,
            source=case_ref.artifact_id,
            license=config.input_license,
        )
        poses_sdf_ref = store.import_file(
            poses_sdf,
            media_type="chemical/x-mdl-sdfile",
            producer="meeko.mk_export",
            producer_version=meeko_version,
            source=poses_pdbqt_ref.artifact_id,
            license=config.input_license,
        )
        mode_refs = tuple(
            store.import_file(
                path,
                media_type="chemical/x-mdl-sdfile",
                producer="protbind.redocking.mode-split",
                producer_version=__version__,
                source=poses_sdf_ref.artifact_id,
                license=config.input_license,
            )
            for path in mode_paths
        )

        stage = "validation"
        released = sealed.release(case, committed_docking_pose=poses_sdf_ref)
        release_receipt = store.put_json(
            released.to_validation_dict(),
            producer="protbind.redocking.validation-release",
            producer_version=__version__,
            source=poses_sdf_ref.artifact_id,
            license=config.input_license,
        )
        evaluated: list[dict[str, Any]] = []
        for index, (mode_path, mode_ref) in enumerate(
            zip(mode_paths[:5], mode_refs[:5], strict=True), start=1
        ):
            pb_valid, pb_checks = _posebusters_mode_checks(
                mode_path, native_copy, receptor_prepared
            )
            rmsd = symmetry_rmsd(store, native_ref, mode_ref).value_angstrom
            evaluated.append(
                {
                    **scores[index - 1],
                    "posebusters_valid": pb_valid,
                    "posebusters_checks": pb_checks,
                    "symmetry_rmsd_angstrom": rmsd,
                    "recovered": pb_valid and rmsd <= 2.0,
                    "pose_artifact": mode_ref.to_dict(),
                    "file": f"artifacts/{mode_path.name}",
                }
            )
        top1 = evaluated[0]
        oracle_best = min(evaluated, key=lambda item: item["symmetry_rmsd_angstrom"])
        recovered_modes = [item for item in evaluated if item["recovered"]]
        top5_recovered = bool(recovered_modes)
        scientific_status = (
            "REDOCKING_RECOVERED_TOP1"
            if top1["recovered"]
            else (
                "REDOCKING_RECOVERED_TOP5_ONLY"
                if top5_recovered
                else "NOT_RECOVERED_TOP5"
            )
        )
        result.update(
            {
                "status": "COMPLETED",
                "scientific_status": scientific_status,
                "evidence_grade": (
                    "REDOCKING_RECOVERED" if top5_recovered else "HYPOTHESIS_ONLY"
                ),
                "top1_recovered": bool(top1["recovered"]),
                "top5_recovered": top5_recovered,
                "pose_count": len(mode_refs),
                "top1": top1,
                "top5_modes": evaluated,
                "top5_oracle": {
                    "evaluated_modes": len(evaluated),
                    "best_mode": oracle_best["mode"],
                    "best_symmetry_rmsd_angstrom": oracle_best[
                        "symmetry_rmsd_angstrom"
                    ],
                    "any_pb_valid_and_rmsd_le_2": top5_recovered,
                    "first_recovered_mode": (
                        recovered_modes[0]["mode"] if recovered_modes else None
                    ),
                },
                "metrics_definition": {
                    "success": "PoseBusters redock valid AND same-frame symmetry RMSD <= 2.0 A",
                    "top1": "highest-ranked Vina mode",
                    "top5": "oracle best among up to five highest-ranked Vina modes",
                },
            }
        )
        result["artifacts"].update(
            {
                "docking_visible_case": {
                    **case_ref.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                    "file": "artifacts/docking-visible-case.json",
                },
                "receptor_pdbqt": {
                    **receptor_pdbqt_ref.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                    "file": "artifacts/receptor.pdbqt",
                },
                "ligand_pdbqt": {
                    **ligand_pdbqt_ref.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                    "file": "artifacts/ligand.pdbqt",
                },
                "vina_poses_pdbqt": {
                    **poses_pdbqt_ref.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                    "file": "artifacts/vina-poses.pdbqt",
                },
                "vina_poses_sdf": {
                    **poses_sdf_ref.to_dict(),
                    "access_scope": "DOCKING_VISIBLE",
                    "file": "artifacts/vina-poses.sdf",
                },
                "validation_release": {
                    **release_receipt.to_dict(),
                    "access_scope": "VALIDATION_ONLY",
                },
            }
        )
        if config.calibration_target_id is not None:
            stage = "calibration_receipt"
            calibration_config = KnownSiteCalibrationConfig(
                target_id=config.calibration_target_id,
                required_rank=config.calibration_required_rank,
                rmsd_threshold_angstrom=config.calibration_rmsd_threshold_angstrom,
            )
            calibration_source_ref = store.put_json(
                result,
                producer="protbind.redocking.calibration-source",
                producer_version=__version__,
                source=f"redock-run:{result['run_identity_sha256']}",
                license=config.input_license,
            )
            calibration_source_path = validation_files / "calibration-source-result.json"
            calibration_source_path.write_bytes(store.read_bytes(calibration_source_ref))
            calibration = build_known_site_calibration_receipt(
                result,
                calibration_config,
                source_result=calibration_source_ref,
            )
            validate_known_site_calibration_receipt(
                calibration,
                expected_target_id=config.calibration_target_id,
                expected_prepared_receptor_sha256=result["artifacts"][
                    "prepared_receptor"
                ]["sha256"],
                require_pass=False,
                store=store,
            )
            calibration_ref = store.put_json(
                calibration,
                producer="protbind.redocking.known-site-calibration",
                producer_version=__version__,
                source=f"redock-run:{result['run_identity_sha256']}",
                license=config.input_license,
            )
            calibration_path = files / "known-site-calibration.json"
            calibration_path.write_bytes(store.read_bytes(calibration_ref))
            result["screening_calibration"] = calibration
            result["artifacts"]["calibration_source_result"] = {
                **calibration_source_ref.to_dict(),
                "access_scope": "VALIDATION_AND_CALIBRATION_ONLY",
                "file": "validation-only/calibration-source-result.json",
            }
            result["artifacts"]["screening_calibration_receipt"] = {
                **calibration_ref.to_dict(),
                "access_scope": "DOCKING_VISIBLE",
                "file": "artifacts/known-site-calibration.json",
            }
    except Exception as exc:
        if isinstance(exc, RedockBenchmarkError):
            failure_stage = exc.stage
            code = exc.code
            message = str(exc)
        else:
            failure_stage = stage
            code = type(exc).__name__.upper()
            message = f"{type(exc).__name__}: {exc}"
        for private_path in (str(receptor), str(native_ligand), str(output)):
            message = message.replace(private_path, Path(private_path).name)
        result.update(
            {
                "status": "FAILED",
                "scientific_status": "UNKNOWN",
                "failure": {
                    "stage": failure_stage,
                    "code": code,
                    "message": message,
                },
            }
        )
    _atomic_json(output / "result.json", result)
    return result
