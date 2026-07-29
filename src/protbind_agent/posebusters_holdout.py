"""Offline, leakage-resistant PoseBusters holdout procurement.

The official archive contains both the 428-case pre-review pool and the protein,
ligand, and generated-conformer files used by the PoseBusters paper.  The 308-case
no-crystal-contact subset is maintained separately by PoseBench.  This module
requires both inputs to be content-pinned, audits every one of the 308 cases
against ProtBind's declared v1 boundary, and only then hash-sorts eligible IDs.

No docking result or score is read during selection.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import (
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .models import ArtifactRef
from .preparation import _STANDARD_HEAVY_ATOMS, _WATER_NAMES
from .redocking import (
    HoldoutSelectionManifest,
    RedockingBenchmarkCandidate,
    persist_holdout_manifest,
    select_holdout_manifest,
)
from .structure import inspect_structure

POSEBUSTERS_ZENODO_RECORD = "8278563"
POSEBUSTERS_ARCHIVE_MD5 = "f004ac7c4e68317b5348497d2bb6bee6"
POSEBUSTERS_ARCHIVE_SHA256 = (
    "495a8f432ee5612c0dfa3cc582829f112bfca3c29dddc2db2c3a8dc7609e721c"
)
POSEBUSTERS_ARCHIVE_SIZE = 53_660_397
POSEBENCH_COMMIT = "c5d728d2a31ddb0a27512be75ea2d44e391e6529"
POSEBUSTERS_308_IDS_SHA256 = (
    "a69a7b6b9a5a52531933078ef983e6c069e3a987a1d7a733bd7d72cbe1793de6"
)
POSEBUSTERS_308_COUNT = 308
DEFAULT_NAMESPACE = "protbind-posebusters-308-redock-v1-20260725"
DEFAULT_POCKET_RADIUS_ANGSTROM = 6.0

_CASE_ID = re.compile(r"^[0-9A-Z]{4}_[0-9A-Z]{1,3}$")
_ALLOWED_LIGAND_ATOMIC_NUMBERS = frozenset({1, 6, 7, 8, 9, 15, 16, 17})
_METAL_ATOMIC_NUMBERS = frozenset(
    {
        3,
        4,
        11,
        12,
        13,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        55,
        56,
        78,
        79,
        80,
        82,
    }
)
_ARCHIVE_PREFIX = "posebusters_benchmark_set"
_MAX_RECEPTOR_BYTES = 16 * 1024 * 1024
_MAX_LIGAND_BYTES = 2 * 1024 * 1024
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024


class PoseBustersHoldoutError(ValueError):
    """The public source or one of its declared cases failed deterministic audit."""


@dataclass(frozen=True, slots=True)
class PoseBustersFreezeResult:
    manifest: HoldoutSelectionManifest
    manifest_artifact: ArtifactRef
    archive_sha256: str
    candidate_list_sha256: str
    candidate_count: int
    eligible_count: int
    exclusion_reason_counts: dict[str, int]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "dataset": "PoseBusters Benchmark 308",
            "dataset_version": self.manifest.dataset_version,
            "dataset_license": self.manifest.dataset_license,
            "candidate_count": self.candidate_count,
            "eligible_count": self.eligible_count,
            "selected_count": len(self.manifest.selected),
            "selected_case_ids": [
                candidate.complex_id for candidate in self.manifest.selected
            ],
            "selection_hash": self.manifest.selection_hash,
            "manifest_artifact_id": self.manifest_artifact.artifact_id,
            "archive_sha256": self.archive_sha256,
            "candidate_list_sha256": self.candidate_list_sha256,
            "exclusion_reason_counts": dict(sorted(self.exclusion_reason_counts.items())),
            "docking_results_inspected_during_selection": False,
        }


@dataclass(frozen=True, slots=True)
class _LigandAudit:
    record_count: int
    heavy_atom_count: int
    ordinary_nonpolymer: bool
    contains_metal: bool
    unspecified_stereo: bool
    heavy_atom_coordinates: tuple[tuple[float, float, float], ...]


def _md5_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _artifact_ref(
    data: bytes,
    *,
    media_type: str,
    source: str,
) -> ArtifactRef:
    return ArtifactRef(
        sha256=sha256_bytes(data),
        media_type=media_type,
        size_bytes=len(data),
        producer="posebusters.zenodo-import",
        producer_version=POSEBUSTERS_ZENODO_RECORD,
        source=source,
        license="CC-BY-4.0",
    )


def _source_complex_bytes(
    case_id: str,
    receptor: ArtifactRef,
    native_ligand: ArtifactRef,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "1.0",
            "dataset": "PoseBusters Benchmark",
            "case_id": case_id,
            "archive_sha256": POSEBUSTERS_ARCHIVE_SHA256,
            "receptor_sha256": receptor.sha256,
            "native_ligand_sha256": native_ligand.sha256,
            "source_record": f"https://zenodo.org/records/{POSEBUSTERS_ZENODO_RECORD}",
            "license": "CC-BY-4.0",
        }
    )


def _source_complex_ref(
    case_id: str,
    receptor: ArtifactRef,
    native_ligand: ArtifactRef,
) -> ArtifactRef:
    data = _source_complex_bytes(case_id, receptor, native_ligand)
    return _artifact_ref(
        data,
        media_type="application/vnd.protbind.dataset-complex+json",
        source=f"dataset:posebusters-benchmark/{case_id}",
    )


def _validate_member_names(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    total_uncompressed = 0
    for info in archive.infolist():
        name = info.filename
        pure = PurePosixPath(name)
        if (
            not name
            or name.startswith(("/", "\\"))
            or "\\" in name
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise PoseBustersHoldoutError(f"unsafe archive member path: {name!r}")
        if name in members:
            raise PoseBustersHoldoutError(f"duplicate archive member path: {name}")
        members[name] = info
        total_uncompressed += info.file_size
    if total_uncompressed > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise PoseBustersHoldoutError("archive exceeds the uncompressed-size safety limit")
    return members


def _read_member(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    name: str,
    *,
    maximum_bytes: int,
) -> bytes:
    info = members.get(name)
    if info is None or info.is_dir():
        raise PoseBustersHoldoutError(f"required archive member is missing: {name}")
    if info.file_size < 1 or info.file_size > maximum_bytes:
        raise PoseBustersHoldoutError(
            f"archive member size is outside the safety limit: {name}"
        )
    data = archive.read(info)
    if len(data) != info.file_size:
        raise PoseBustersHoldoutError(f"archive member length mismatch: {name}")
    return data


def _parse_candidate_ids(data: bytes, *, expected_count: int) -> tuple[str, ...]:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise PoseBustersHoldoutError("candidate list must be ASCII") from exc
    identifiers = tuple(line.strip().upper() for line in text.splitlines() if line.strip())
    if len(identifiers) != expected_count:
        raise PoseBustersHoldoutError(
            f"candidate list has {len(identifiers)} IDs; expected {expected_count}"
        )
    if len(set(identifiers)) != len(identifiers):
        raise PoseBustersHoldoutError("candidate list contains duplicate IDs")
    invalid = tuple(value for value in identifiers if _CASE_ID.fullmatch(value) is None)
    if invalid:
        raise PoseBustersHoldoutError(
            "candidate list contains invalid PDB_CCD IDs: " + ", ".join(invalid[:5])
        )
    return identifiers


def _inspect_ligand(data: bytes) -> _LigandAudit:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise PoseBustersHoldoutError(
            "RDKit is required to audit PoseBusters ligands"
        ) from exc
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(data), removeHs=False, sanitize=True, strictParsing=True
    )
    records = list(supplier)
    valid = [molecule for molecule in records if molecule is not None]
    if len(records) != 1 or len(valid) != 1:
        return _LigandAudit(
            record_count=len(records),
            heavy_atom_count=0,
            ordinary_nonpolymer=False,
            contains_metal=False,
            unspecified_stereo=False,
            heavy_atom_coordinates=(),
        )
    molecule = valid[0]
    parent = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
    Chem.AssignStereochemistry(parent, cleanIt=True, force=True)
    atomic_numbers = tuple(atom.GetAtomicNum() for atom in parent.GetAtoms())
    contains_metal = any(value in _METAL_ATOMIC_NUMBERS for value in atomic_numbers)
    heavy_atom_count = parent.GetNumHeavyAtoms()
    unspecified_stereo = any(
        str(stereo.specified) == "Unspecified"
        for stereo in Chem.FindPotentialStereo(parent)
    )
    fragments = Chem.GetMolFrags(parent)
    ordinary = (
        len(fragments) == 1
        and 1 <= heavy_atom_count <= 100
        and all(value in _ALLOWED_LIGAND_ATOMIC_NUMBERS for value in atomic_numbers)
        and not contains_metal
        and molecule.GetNumConformers() == 1
    )
    coordinates: list[tuple[float, float, float]] = []
    if molecule.GetNumConformers() == 1:
        conformer = molecule.GetConformer()
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            point = conformer.GetAtomPosition(atom.GetIdx())
            values = (float(point.x), float(point.y), float(point.z))
            if any(not math.isfinite(value) for value in values):
                ordinary = False
                coordinates = []
                break
            coordinates.append(values)
    return _LigandAudit(
        record_count=1,
        heavy_atom_count=heavy_atom_count,
        ordinary_nonpolymer=ordinary,
        contains_metal=contains_metal,
        unspecified_stereo=unspecified_stereo,
        heavy_atom_coordinates=tuple(coordinates),
    )


def _minimum_site_distance(residue: Any, points: tuple[tuple[float, float, float], ...]) -> float:
    minimum = math.inf
    for atom in residue:
        if atom.element.atomic_number <= 1:
            continue
        atom_point = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
        for point in points:
            minimum = min(minimum, math.dist(atom_point, point))
    return minimum


def _pocket_quality_flags(
    structure: Any,
    ligand_points: tuple[tuple[float, float, float], ...],
    *,
    pocket_radius_angstrom: float,
) -> tuple[bool, bool]:
    if not ligand_points:
        return False, False
    missing_pocket_heavy_atoms = False
    ambiguous_pocket_altloc = False
    for chain in structure[0]:
        for residue in chain:
            residue_name = residue.name.strip().upper()
            if residue_name not in _STANDARD_HEAVY_ATOMS:
                continue
            if _minimum_site_distance(residue, ligand_points) > pocket_radius_angstrom:
                continue
            actual = {
                atom.name.strip().upper()
                for atom in residue
                if atom.element.atomic_number > 1
            }
            if _STANDARD_HEAVY_ATOMS[residue_name] - actual:
                missing_pocket_heavy_atoms = True
            groups: dict[str, list[Any]] = {}
            for atom in residue:
                groups.setdefault(atom.name.strip().upper(), []).append(atom)
            for atoms in groups.values():
                if len(atoms) < 2:
                    continue
                best = max(float(atom.occ) for atom in atoms)
                if sum(math.isclose(float(atom.occ), best, abs_tol=1e-6) for atom in atoms) > 1:
                    ambiguous_pocket_altloc = True
    return missing_pocket_heavy_atoms, ambiguous_pocket_altloc


def _audit_candidate(
    case_id: str,
    receptor_data: bytes,
    ligand_data: bytes,
    receptor_path: Path,
    *,
    pocket_radius_angstrom: float,
) -> RedockingBenchmarkCandidate:
    ligand = _inspect_ligand(ligand_data)
    receptor_path.write_bytes(receptor_data)
    inspection = inspect_structure(receptor_path, max_chains=None, max_residues=None)
    try:
        import gemmi
    except ImportError as exc:
        raise PoseBustersHoldoutError(
            "Gemmi is required to audit PoseBusters receptors"
        ) from exc
    try:
        structure = gemmi.read_pdb_string(receptor_data.decode("utf-8"))
    except (RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise PoseBustersHoldoutError(f"Gemmi could not parse receptor {case_id}") from exc
    if not structure:
        raise PoseBustersHoldoutError(f"receptor {case_id} contains no coordinate model")
    missing_pocket, ambiguous_altloc = _pocket_quality_flags(
        structure,
        ligand.heavy_atom_coordinates,
        pocket_radius_angstrom=pocket_radius_angstrom,
    )
    heterogens = {
        line[17:20].strip().upper()
        for line in receptor_data.decode("utf-8").splitlines()
        if line.startswith("HETATM")
        and line[17:20].strip().upper() not in _WATER_NAMES
    }
    source = f"dataset:posebusters-benchmark/{case_id}"
    receptor = _artifact_ref(
        receptor_data,
        media_type="chemical/x-pdb",
        source=source + "/protein.pdb",
    )
    native_ligand = _artifact_ref(
        ligand_data,
        media_type="chemical/x-mdl-sdfile",
        source=source + "/ligand.sdf",
    )
    _, ligand_instance = case_id.split("_", 1)
    return RedockingBenchmarkCandidate(
        complex_id=case_id,
        ligand_instance_id=ligand_instance,
        source_complex=_source_complex_ref(case_id, receptor, native_ligand),
        receptor=receptor,
        native_ligand=native_ligand,
        license="CC-BY-4.0",
        protein_chain_count=inspection.chain_count,
        protein_residue_count=inspection.residue_count,
        ligand_count=ligand.record_count,
        ligand_heavy_atom_count=ligand.heavy_atom_count,
        is_non_covalent=True,
        ordinary_nonpolymer_ligand=ligand.ordinary_nonpolymer,
        contains_metal=bool(inspection.metal_elements) or ligand.contains_metal,
        requires_cofactor=bool(heterogens),
        pocket_altloc_ambiguous=ambiguous_altloc,
        missing_pocket_heavy_atoms=missing_pocket,
        receptor_model_count=len(structure),
        contains_nonstandard_protein_residue=bool(inspection.nonstandard_residues),
        missing_backbone_atoms=bool(inspection.missing_backbone_residues),
        ligand_unspecified_stereo=ligand.unspecified_stereo,
    )


def _member_names(case_id: str) -> tuple[str, str]:
    base = f"{_ARCHIVE_PREFIX}/{case_id}/{case_id}"
    return f"{base}_protein.pdb", f"{base}_ligand.sdf"


def _materialize_selected(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    store: ArtifactStore,
    selected: tuple[RedockingBenchmarkCandidate, ...],
) -> None:
    for candidate in selected:
        receptor_name, ligand_name = _member_names(candidate.complex_id)
        receptor_data = _read_member(
            archive, members, receptor_name, maximum_bytes=_MAX_RECEPTOR_BYTES
        )
        ligand_data = _read_member(
            archive, members, ligand_name, maximum_bytes=_MAX_LIGAND_BYTES
        )
        receptor = store.put_bytes(
            receptor_data,
            media_type=candidate.receptor.media_type,
            producer=candidate.receptor.producer,
            producer_version=candidate.receptor.producer_version,
            source=candidate.receptor.source,
            license=candidate.receptor.license,
        )
        native_ligand = store.put_bytes(
            ligand_data,
            media_type=candidate.native_ligand.media_type,
            producer=candidate.native_ligand.producer,
            producer_version=candidate.native_ligand.producer_version,
            source=candidate.native_ligand.source,
            license=candidate.native_ligand.license,
        )
        complex_data = _source_complex_bytes(
            candidate.complex_id, receptor, native_ligand
        )
        source_complex = store.put_bytes(
            complex_data,
            media_type=candidate.source_complex.media_type,
            producer=candidate.source_complex.producer,
            producer_version=candidate.source_complex.producer_version,
            source=candidate.source_complex.source,
            license=candidate.source_complex.license,
        )
        if (
            receptor != candidate.receptor
            or native_ligand != candidate.native_ligand
            or source_complex != candidate.source_complex
        ):
            raise RuntimeError("materialized holdout artifact identity mismatch")


def freeze_posebusters_holdout(
    archive_path: Path,
    candidate_list_path: Path,
    store: ArtifactStore,
    *,
    count: int = 10,
    namespace: str = DEFAULT_NAMESPACE,
    pocket_radius_angstrom: float = DEFAULT_POCKET_RADIUS_ANGSTROM,
    expected_archive_md5: str = POSEBUSTERS_ARCHIVE_MD5,
    expected_archive_sha256: str = POSEBUSTERS_ARCHIVE_SHA256,
    expected_candidate_list_sha256: str = POSEBUSTERS_308_IDS_SHA256,
    expected_candidate_count: int = POSEBUSTERS_308_COUNT,
) -> PoseBustersFreezeResult:
    """Audit the pinned public pool and freeze a result-blind hash-sorted holdout."""

    if not archive_path.is_file() or not candidate_list_path.is_file():
        raise FileNotFoundError("PoseBusters archive and candidate list must both exist")
    if count < 1:
        raise ValueError("holdout count must be positive")
    if (
        isinstance(pocket_radius_angstrom, bool)
        or not isinstance(pocket_radius_angstrom, int | float)
        or not math.isfinite(float(pocket_radius_angstrom))
        or pocket_radius_angstrom <= 0
    ):
        raise ValueError("pocket radius must be finite and positive")
    _validate_digest(expected_archive_sha256, "expected_archive_sha256")
    _validate_digest(
        expected_candidate_list_sha256, "expected_candidate_list_sha256"
    )
    if re.fullmatch(r"[0-9a-f]{32}", expected_archive_md5) is None:
        raise ValueError("expected_archive_md5 must be a lowercase MD5 digest")
    if archive_path.stat().st_size != POSEBUSTERS_ARCHIVE_SIZE and (
        expected_archive_sha256 == POSEBUSTERS_ARCHIVE_SHA256
    ):
        raise PoseBustersHoldoutError("official archive size does not match its frozen record")
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != expected_archive_sha256:
        raise PoseBustersHoldoutError("PoseBusters archive SHA-256 mismatch")
    if _md5_file(archive_path) != expected_archive_md5:
        raise PoseBustersHoldoutError("PoseBusters archive publisher MD5 mismatch")
    candidate_list_data = candidate_list_path.read_bytes()
    candidate_list_sha256 = sha256_bytes(candidate_list_data)
    if candidate_list_sha256 != expected_candidate_list_sha256:
        raise PoseBustersHoldoutError("PoseBusters 308 candidate-list SHA-256 mismatch")
    candidate_ids = _parse_candidate_ids(
        candidate_list_data, expected_count=expected_candidate_count
    )

    dataset_source = ArtifactRef(
        sha256=archive_sha256,
        media_type="application/zip",
        size_bytes=archive_path.stat().st_size,
        producer="zenodo",
        producer_version=POSEBUSTERS_ZENODO_RECORD,
        source=f"https://zenodo.org/records/{POSEBUSTERS_ZENODO_RECORD}",
        license="CC-BY-4.0",
    )
    candidate_list = store.put_bytes(
        candidate_list_data,
        media_type="text/plain",
        producer="posebench",
        producer_version=POSEBENCH_COMMIT,
        source=(
            "https://raw.githubusercontent.com/BioinfoMachineLearning/PoseBench/"
            f"{POSEBENCH_COMMIT}/data/posebusters_pdb_ccd_ids.txt"
        ),
        license="MIT",
    )
    policy = {
        "version": "posebusters-holdout-1.0",
        "source_sha256": sha256_file(Path(__file__)),
        "pocket_radius_angstrom": float(pocket_radius_angstrom),
        "heterogen_policy": (
            "exclude any non-water HETATM component; no silent cofactor deletion"
        ),
        "stereo_policy": "exclude any RDKit potential stereo element marked Unspecified",
        "missing_atom_policy": (
            "exclude standard residues missing expected heavy atoms within the ligand-atom "
            "pocket radius; outside-pocket defects remain eligible and may fail downstream"
        ),
        "selection_reads_docking_results": False,
        "explicit_case_exclusions": [],
    }
    candidates: list[RedockingBenchmarkCandidate] = []
    try:
        with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(
            prefix="protbind-posebusters-audit-"
        ) as temporary_directory:
            members = _validate_member_names(archive)
            receptor_path = Path(temporary_directory) / "candidate.pdb"
            for case_id in candidate_ids:
                receptor_name, ligand_name = _member_names(case_id)
                receptor_data = _read_member(
                    archive,
                    members,
                    receptor_name,
                    maximum_bytes=_MAX_RECEPTOR_BYTES,
                )
                ligand_data = _read_member(
                    archive,
                    members,
                    ligand_name,
                    maximum_bytes=_MAX_LIGAND_BYTES,
                )
                candidates.append(
                    _audit_candidate(
                        case_id,
                        receptor_data,
                        ligand_data,
                        receptor_path,
                        pocket_radius_angstrom=float(pocket_radius_angstrom),
                    )
                )
            manifest = select_holdout_manifest(
                candidates,
                dataset_name="PoseBusters Benchmark 308",
                dataset_version=(
                    f"zenodo-{POSEBUSTERS_ZENODO_RECORD}+posebench-{POSEBENCH_COMMIT}"
                ),
                dataset_license="CC-BY-4.0",
                count=count,
                namespace=namespace,
                excluded_complex_ids=(),
                dataset_source=dataset_source,
                candidate_list=candidate_list,
                eligibility_policy=policy,
            )
            _materialize_selected(archive, members, store, manifest.selected)
    except zipfile.BadZipFile as exc:
        raise PoseBustersHoldoutError("PoseBusters archive is not a valid ZIP") from exc

    manifest_artifact = persist_holdout_manifest(store, manifest)
    reason_counts = Counter(
        reason for exclusion in manifest.exclusions for reason in exclusion.reasons
    )
    eligible_count = sum(not candidate.exclusion_reasons() for candidate in candidates)
    return PoseBustersFreezeResult(
        manifest=manifest,
        manifest_artifact=manifest_artifact,
        archive_sha256=archive_sha256,
        candidate_list_sha256=candidate_list_sha256,
        candidate_count=len(candidates),
        eligible_count=eligible_count,
        exclusion_reason_counts=dict(reason_counts),
    )


def write_holdout_manifest(
    path: Path,
    manifest: HoldoutSelectionManifest,
    *,
    overwrite: bool = False,
) -> None:
    """Persist a human-readable hash-bound manifest without partial replacement."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"holdout manifest already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
