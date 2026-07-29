"""Deterministic primitives for leakage-resistant redocking benchmarks.

Redocking is a calibration experiment, not ordinary prospective docking.  The
native ligand's chemical identity and a native-derived search box are legitimate
docking inputs, while its coordinates are validation-only.  This module encodes
that boundary by returning separate docking-visible and sealed-reference objects.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from . import __version__
from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes
from .models import ArtifactRef

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METAL_ATOMIC_NUMBERS = frozenset(
    {
        3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
        78, 79, 80, 82,
    }
)


class RedockingCapabilityError(RuntimeError):
    """A required scientific library is unavailable."""


class RedockingUnsupportedError(ValueError):
    """A molecule or benchmark entry falls outside the v1 support boundary."""


class ReferenceAccessError(RuntimeError):
    """Validation coordinates were requested without a committed docking output."""


class ArtifactAccessScope(StrEnum):
    DOCKING_VISIBLE = "DOCKING_VISIBLE"
    VALIDATION_ONLY = "VALIDATION_ONLY"


@dataclass(frozen=True, slots=True)
class NativeDerivedBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    padding_angstrom: float
    heavy_atom_count: int
    native_pose_commitment: str
    receipt: ArtifactRef
    definition: str = "redock-known-site"

    def __post_init__(self) -> None:
        _validate_vector(self.center, "box center", positive=False)
        _validate_vector(self.size, "box size", positive=True)
        if not math.isfinite(self.padding_angstrom) or self.padding_angstrom <= 0:
            raise ValueError("box padding must be finite and positive")
        if self.heavy_atom_count < 1:
            raise ValueError("box receipt requires at least one heavy atom")
        if not _SHA256.fullmatch(self.native_pose_commitment):
            raise ValueError("native pose commitment must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "center": list(self.center),
            "size": list(self.size),
            "padding_angstrom": self.padding_angstrom,
            "heavy_atom_count": self.heavy_atom_count,
            "native_pose_commitment": self.native_pose_commitment,
            "receipt": self.receipt.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RedockingLigandPreparation:
    ligand_3d: ArtifactRef
    identity: ArtifactRef
    receipt: ArtifactRef
    canonical_isomeric_smiles: str
    formal_charge: int
    heavy_atom_count: int
    seed: int
    rdkit_seed: int
    minimization_method: str
    minimization_converged: bool
    native_pose_commitment: str

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.native_pose_commitment):
            raise ValueError("ligand preparation native pose commitment is invalid")


@dataclass(frozen=True, slots=True)
class RedockingCase:
    """Only inputs that a docking worker is permitted to read."""

    case_id: str
    receptor: ArtifactRef
    receptor_preparation_receipt: ArtifactRef
    ligand: ArtifactRef
    ligand_identity: ArtifactRef
    ligand_preparation_receipt: ArtifactRef
    box: NativeDerivedBox
    seed: int
    reference_commitment: str

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.case_id):
            raise ValueError("redocking case_id must be a safe identifier")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("redocking seed must fit in an unsigned 32-bit integer")
        if not _SHA256.fullmatch(self.reference_commitment):
            raise ValueError("reference commitment must be a lowercase SHA-256 digest")

    def to_docking_dict(self) -> dict[str, Any]:
        """Serialize without ever including the validation-only native pose."""

        return {
            "schema_version": "1.0",
            "artifact_scope": ArtifactAccessScope.DOCKING_VISIBLE.value,
            "case_id": self.case_id,
            "receptor": self.receptor.to_dict(),
            "receptor_preparation_receipt": self.receptor_preparation_receipt.to_dict(),
            "ligand": self.ligand.to_dict(),
            "ligand_identity": self.ligand_identity.to_dict(),
            "ligand_preparation_receipt": self.ligand_preparation_receipt.to_dict(),
            "box_center": list(self.box.center),
            "box_size": list(self.box.size),
            "box_receipt": self.box.receipt.to_dict(),
            "seed": self.seed,
            "reference_commitment": self.reference_commitment,
        }


@dataclass(frozen=True, slots=True)
class SealedValidationReference:
    """Native coordinates kept outside the docking-visible case payload."""

    case_id: str
    native_pose: ArtifactRef
    native_identity: ArtifactRef
    commitment: str
    artifact_scope: ArtifactAccessScope = ArtifactAccessScope.VALIDATION_ONLY

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.case_id):
            raise ValueError("sealed reference case_id must be a safe identifier")
        expected = reference_commitment(
            self.case_id, self.native_pose, self.native_identity
        )
        if self.commitment != expected:
            raise ValueError("sealed reference commitment does not match its artifacts")
        if self.artifact_scope is not ArtifactAccessScope.VALIDATION_ONLY:
            raise ValueError("native pose must have VALIDATION_ONLY scope")

    def release(
        self,
        case: RedockingCase,
        *,
        committed_docking_pose: ArtifactRef,
    ) -> ReleasedValidationReference:
        if case.case_id != self.case_id or case.reference_commitment != self.commitment:
            raise ReferenceAccessError("sealed reference does not belong to this redocking case")
        if committed_docking_pose.sha256 == self.native_pose.sha256:
            raise ReferenceAccessError("docking output cannot be the native reference artifact")
        return ReleasedValidationReference(
            case_id=self.case_id,
            native_pose=self.native_pose,
            native_identity=self.native_identity,
            committed_docking_pose=committed_docking_pose,
            commitment=self.commitment,
        )


@dataclass(frozen=True, slots=True)
class ReleasedValidationReference:
    case_id: str
    native_pose: ArtifactRef
    native_identity: ArtifactRef
    committed_docking_pose: ArtifactRef
    commitment: str
    artifact_scope: ArtifactAccessScope = ArtifactAccessScope.VALIDATION_ONLY

    def to_validation_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "artifact_scope": self.artifact_scope.value,
            "case_id": self.case_id,
            "native_pose": self.native_pose.to_dict(),
            "native_identity": self.native_identity.to_dict(),
            "committed_docking_pose": self.committed_docking_pose.to_dict(),
            "reference_commitment": self.commitment,
        }


@dataclass(frozen=True, slots=True)
class SymmetryRMSDResult:
    value_angstrom: float
    implementation: str = "spyrmsd"
    symmetry_corrected: bool = True
    centered: bool = False
    minimized: bool = False
    hydrogens_stripped: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.value_angstrom) or self.value_angstrom < 0:
            raise ValueError("symmetry RMSD must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RedockingBenchmarkCandidate:
    complex_id: str
    ligand_instance_id: str
    source_complex: ArtifactRef
    receptor: ArtifactRef
    native_ligand: ArtifactRef
    license: str
    protein_chain_count: int
    protein_residue_count: int
    ligand_count: int
    ligand_heavy_atom_count: int
    is_non_covalent: bool = True
    ordinary_nonpolymer_ligand: bool = True
    contains_metal: bool = False
    requires_cofactor: bool = False
    pocket_altloc_ambiguous: bool = False
    missing_pocket_heavy_atoms: bool = False
    receptor_model_count: int = 1
    contains_nonstandard_protein_residue: bool = False
    missing_backbone_atoms: bool = False
    ligand_unspecified_stereo: bool = False

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.complex_id):
            raise ValueError("benchmark complex_id must be a safe identifier")
        if not self.ligand_instance_id.strip():
            raise ValueError("benchmark ligand_instance_id cannot be empty")
        if not self.license.strip():
            raise ValueError("benchmark candidate requires an explicit license")
        if self.receptor_model_count < 1:
            raise ValueError("benchmark candidate receptor_model_count must be positive")

    def exclusion_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not 1 <= self.protein_chain_count <= 2:
            reasons.append("protein_chain_count_outside_1_2")
        if not 1 <= self.protein_residue_count <= 700:
            reasons.append("protein_residue_count_outside_1_700")
        if self.ligand_count != 1:
            reasons.append("ligand_count_not_one")
        if not 1 <= self.ligand_heavy_atom_count <= 100:
            reasons.append("ligand_heavy_atom_count_outside_1_100")
        if not self.is_non_covalent:
            reasons.append("covalent_ligand")
        if not self.ordinary_nonpolymer_ligand:
            reasons.append("nonordinary_or_polymer_ligand")
        if self.contains_metal:
            reasons.append("metal_system")
        if self.requires_cofactor:
            reasons.append("required_cofactor")
        if self.pocket_altloc_ambiguous:
            reasons.append("ambiguous_pocket_altloc")
        if self.missing_pocket_heavy_atoms:
            reasons.append("missing_pocket_heavy_atoms")
        if self.receptor_model_count != 1:
            reasons.append("receptor_model_count_not_one")
        if self.contains_nonstandard_protein_residue:
            reasons.append("nonstandard_protein_residue")
        if self.missing_backbone_atoms:
            reasons.append("missing_backbone_atoms")
        if self.ligand_unspecified_stereo:
            reasons.append("ligand_unspecified_stereo")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "complex_id": self.complex_id,
            "ligand_instance_id": self.ligand_instance_id,
            "source_complex": self.source_complex.to_dict(),
            "receptor": self.receptor.to_dict(),
            "native_ligand": self.native_ligand.to_dict(),
            "license": self.license,
            "protein_chain_count": self.protein_chain_count,
            "protein_residue_count": self.protein_residue_count,
            "ligand_count": self.ligand_count,
            "ligand_heavy_atom_count": self.ligand_heavy_atom_count,
            "is_non_covalent": self.is_non_covalent,
            "ordinary_nonpolymer_ligand": self.ordinary_nonpolymer_ligand,
            "contains_metal": self.contains_metal,
            "requires_cofactor": self.requires_cofactor,
            "pocket_altloc_ambiguous": self.pocket_altloc_ambiguous,
            "missing_pocket_heavy_atoms": self.missing_pocket_heavy_atoms,
            "receptor_model_count": self.receptor_model_count,
            "contains_nonstandard_protein_residue": (
                self.contains_nonstandard_protein_residue
            ),
            "missing_backbone_atoms": self.missing_backbone_atoms,
            "ligand_unspecified_stereo": self.ligand_unspecified_stereo,
        }


@dataclass(frozen=True, slots=True)
class HoldoutExclusion:
    complex_id: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"complex_id": self.complex_id, "reasons": list(self.reasons)}


@dataclass(frozen=True, slots=True)
class HoldoutSelectionManifest:
    dataset_name: str
    dataset_version: str
    dataset_license: str
    namespace: str
    requested_count: int
    selected: tuple[RedockingBenchmarkCandidate, ...]
    exclusions: tuple[HoldoutExclusion, ...]
    selection_hash: str
    dataset_source: ArtifactRef | None = None
    candidate_list: ArtifactRef | None = None
    eligibility_policy: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.selected) != self.requested_count:
            raise ValueError("holdout manifest does not contain the requested count")
        if not _SHA256.fullmatch(self.selection_hash):
            raise ValueError("holdout selection_hash must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.1",
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_license": self.dataset_license,
            "dataset_source": (
                self.dataset_source.to_dict() if self.dataset_source is not None else None
            ),
            "candidate_list": (
                self.candidate_list.to_dict() if self.candidate_list is not None else None
            ),
            "eligibility_policy": self.eligibility_policy,
            "namespace": self.namespace,
            "requested_count": self.requested_count,
            "selection_rule": (
                "declared eligibility filter, then sha256(namespace + ':' + complex_id), "
                "then complex_id"
            ),
            "selected": [candidate.to_dict() for candidate in self.selected],
            "exclusions": [exclusion.to_dict() for exclusion in self.exclusions],
            "selection_hash": self.selection_hash,
        }


def _rdkit() -> tuple[Any, Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors
    except ImportError as exc:
        raise RedockingCapabilityError(
            "RDKit is required for redocking ligand preparation"
        ) from exc
    return Chem, AllChem, rdMolDescriptors


def _load_single_ligand(store: ArtifactStore, artifact: ArtifactRef) -> Any:
    Chem, _, _ = _rdkit()
    data = store.read_bytes(artifact)
    molecules: list[Any] = []
    if b"$$$$" in data:
        supplier = Chem.ForwardSDMolSupplier(
            io.BytesIO(data), removeHs=False, sanitize=True, strictParsing=True
        )
        records = list(supplier)
        if len(records) == 1 and records[0] is not None:
            molecules = [records[0]]
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("ligand artifact is not a readable MDL structure") from exc
        molecule = Chem.MolFromMolBlock(
            text, removeHs=False, sanitize=True, strictParsing=True
        )
        if molecule is not None:
            molecules = [molecule]
    if len(molecules) != 1:
        raise ValueError("ligand artifact must contain exactly one readable molecule")
    molecule = molecules[0]
    if molecule.GetNumConformers() != 1:
        raise ValueError("ligand artifact must contain exactly one coordinate conformer")
    conformer = molecule.GetConformer()
    for index in range(molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(index)
        if not all(math.isfinite(float(value)) for value in (point.x, point.y, point.z)):
            raise ValueError("ligand artifact contains non-finite coordinates")
    return molecule


def _validate_vector(
    value: tuple[float, float, float], name: str, *, positive: bool
) -> None:
    if len(value) != 3 or any(
        isinstance(item, bool)
        or not isinstance(item, int | float)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"{name} must contain three finite numbers")
    if positive and any(float(item) <= 0 for item in value):
        raise ValueError(f"{name} values must be positive")


def _native_pose_commitment(native_pose: ArtifactRef) -> str:
    """Bind derived inputs to a pose without exposing its resolvable artifact hash."""

    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "purpose": "redocking-native-pose-source",
                "native_pose_sha256": native_pose.sha256,
            }
        )
    )


def native_derived_box(
    store: ArtifactStore,
    native_pose: ArtifactRef,
    *,
    padding_angstrom: float = 5.0,
) -> NativeDerivedBox:
    """Create an axis-aligned known-site box around native heavy-atom coordinates."""

    if not math.isfinite(padding_angstrom) or padding_angstrom <= 0:
        raise ValueError("box padding must be finite and positive")
    molecule = _load_single_ligand(store, native_pose)
    conformer = molecule.GetConformer()
    coordinates = [
        tuple(float(value) for value in conformer.GetAtomPosition(atom.GetIdx()))
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not coordinates:
        raise RedockingUnsupportedError("native ligand has no heavy atoms")
    minima = tuple(min(point[axis] for point in coordinates) for axis in range(3))
    maxima = tuple(max(point[axis] for point in coordinates) for axis in range(3))
    center = tuple((minima[axis] + maxima[axis]) / 2.0 for axis in range(3))
    size = tuple(
        maxima[axis] - minima[axis] + 2.0 * padding_angstrom for axis in range(3)
    )
    commitment = _native_pose_commitment(native_pose)
    payload = {
        "schema_version": "1.0",
        "definition": "redock-known-site",
        "algorithm": "axis-aligned native heavy-atom bounds plus symmetric padding",
        "native_pose_commitment": commitment,
        "center": list(center),
        "size": list(size),
        "padding_angstrom": padding_angstrom,
        "heavy_atom_count": len(coordinates),
        "native_coordinates_exposed_to_docking": False,
        "native_coordinates_used_for_box_derivation": True,
    }
    receipt = store.put_json(
        payload,
        producer="protbind.redocking.native-box-receipt",
        producer_version=__version__,
        source="validation-only-derived:native-pose-commitment",
        license=native_pose.license,
    )
    return NativeDerivedBox(
        center=center,
        size=size,
        padding_angstrom=padding_angstrom,
        heavy_atom_count=len(coordinates),
        native_pose_commitment=commitment,
        receipt=receipt,
    )


def _molecule_identity(molecule: Any) -> dict[str, Any]:
    Chem, _, rdMolDescriptors = _rdkit()
    copy = Chem.Mol(molecule)
    Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
    atoms = []
    for atom in copy.GetAtoms():
        atoms.append(
            {
                "index": atom.GetIdx(),
                "atomic_number": atom.GetAtomicNum(),
                "isotope": atom.GetIsotope(),
                "formal_charge": atom.GetFormalCharge(),
                "aromatic": atom.GetIsAromatic(),
                "chiral_tag": str(atom.GetChiralTag()),
                "cip": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
                "atom_map_number": atom.GetAtomMapNum(),
            }
        )
    bonds = [
        {
            "begin": min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            "end": max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            "bond_type": str(bond.GetBondType()),
            "aromatic": bond.GetIsAromatic(),
            "stereo": str(bond.GetStereo()),
            "stereo_atoms": list(bond.GetStereoAtoms()),
        }
        for bond in copy.GetBonds()
    ]
    bonds.sort(key=lambda item: (item["begin"], item["end"]))
    return {
        "schema_version": "1.0",
        "canonical_isomeric_smiles": Chem.MolToSmiles(copy, isomericSmiles=True),
        "formula": rdMolDescriptors.CalcMolFormula(copy),
        "formal_charge": sum(atom.GetFormalCharge() for atom in copy.GetAtoms()),
        "heavy_atom_count": sum(atom.GetAtomicNum() > 1 for atom in copy.GetAtoms()),
        "atoms": atoms,
        "bonds": bonds,
        "contains_coordinates": False,
    }


def _sdf_bytes(molecule: Any) -> bytes:
    Chem, _, _ = _rdkit()
    handle = io.StringIO()
    writer = Chem.SDWriter(handle)
    writer.SetKekulize(True)
    writer.write(molecule, confId=0)
    writer.flush()
    writer.close()
    return handle.getvalue().encode("utf-8")


def prepare_redocking_ligand(
    store: ArtifactStore,
    native_pose: ArtifactRef,
    *,
    seed: int = 20260721,
    reject_unspecified_stereo: bool = True,
) -> RedockingLigandPreparation:
    """Discard native coordinates and generate one deterministic docking conformer."""

    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("redocking seed must fit in an unsigned 32-bit integer")
    Chem, AllChem, _ = _rdkit()
    native = _load_single_ligand(store, native_pose)
    parent = Chem.RemoveHs(Chem.Mol(native), sanitize=True)
    Chem.AssignStereochemistry(parent, cleanIt=True, force=True)
    if len(Chem.GetMolFrags(parent)) != 1:
        raise RedockingUnsupportedError("redocking ligand must contain exactly one component")
    heavy_atom_count = parent.GetNumHeavyAtoms()
    if not 1 <= heavy_atom_count <= 100:
        raise RedockingUnsupportedError("redocking ligand must contain 1-100 heavy atoms")
    metals = sorted(
        {
            atom.GetSymbol()
            for atom in parent.GetAtoms()
            if atom.GetAtomicNum() in _METAL_ATOMIC_NUMBERS
        }
    )
    if metals:
        raise RedockingUnsupportedError(
            "redocking ligand contains unsupported metal elements: " + ", ".join(metals)
        )
    unspecified_stereo = tuple(
        f"{stereo.type}:{stereo.centeredOn}"
        for stereo in Chem.FindPotentialStereo(parent)
        if str(stereo.specified) == "Unspecified"
    )
    if reject_unspecified_stereo and unspecified_stereo:
        raise RedockingUnsupportedError(
            "redocking ligand has unspecified potentially critical stereochemistry: "
            + ", ".join(unspecified_stereo)
        )
    identity_payload = _molecule_identity(parent)
    canonical_smiles = str(identity_payload["canonical_isomeric_smiles"])
    formal_charge = int(identity_payload["formal_charge"])
    identity = store.put_json(
        identity_payload,
        producer="protbind.redocking.coordinate-free-ligand-identity",
        producer_version=__version__,
        license=native_pose.license,
    )

    fresh = Chem.AddHs(Chem.Mol(parent))
    fresh.RemoveAllConformers()
    rdkit_seed = int(seed % (2**31 - 1)) or 1
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = rdkit_seed
    parameters.numThreads = 1
    parameters.useRandomCoords = False
    parameters.enforceChirality = True
    status = int(AllChem.EmbedMolecule(fresh, parameters))
    if status != 0 or fresh.GetNumConformers() != 1:
        raise RedockingUnsupportedError("ETKDGv3 could not generate a ligand conformer")

    minimization_method: str
    optimization_status: int
    if AllChem.MMFFHasAllMoleculeParams(fresh):
        minimization_method = "MMFF94s"
        optimization_status = int(
            AllChem.MMFFOptimizeMolecule(
                fresh, mmffVariant="MMFF94s", maxIters=1000, confId=0
            )
        )
    elif AllChem.UFFHasAllMoleculeParams(fresh):
        minimization_method = "UFF"
        optimization_status = int(AllChem.UFFOptimizeMolecule(fresh, maxIters=1000, confId=0))
    else:
        raise RedockingUnsupportedError(
            "ligand cannot be parameterized by either MMFF94s or UFF"
        )
    if optimization_status < 0:
        raise RedockingUnsupportedError(
            f"{minimization_method} ligand minimization failed"
        )
    minimization_converged = optimization_status == 0
    fresh.SetProp("PROTBIND_CANONICAL_ISOMERIC_SMILES", canonical_smiles)
    fresh.SetProp("PROTBIND_COORDINATE_SOURCE", "ETKDGv3")
    fresh.SetProp("PROTBIND_SEED", str(seed))
    fresh.SetProp("PROTBIND_MINIMIZATION", minimization_method)
    output = store.put_bytes(
        _sdf_bytes(fresh),
        media_type="chemical/x-mdl-sdfile",
        producer="protbind.redocking.etkdg-ligand",
        producer_version=__version__,
        source=f"coordinate-free-identity:{identity.sha256}",
        license=native_pose.license,
    )
    roundtrip = _load_single_ligand(store, output)
    roundtrip_parent = Chem.RemoveHs(roundtrip, sanitize=True)
    roundtrip_identity = _molecule_identity(roundtrip_parent)
    if roundtrip_identity != identity_payload:
        raise RuntimeError("prepared ligand SDF did not preserve chemical identity")
    receipt_payload = {
        "schema_version": "1.0",
        "native_pose_commitment": _native_pose_commitment(native_pose),
        "coordinate_free_identity_artifact_id": identity.artifact_id,
        "prepared_ligand_artifact_id": output.artifact_id,
        "native_coordinates_discarded": True,
        "native_coordinates_copied": False,
        "conformer_generation": "ETKDGv3",
        "seed": seed,
        "rdkit_seed": rdkit_seed,
        "num_threads": 1,
        "minimization_method": minimization_method,
        "minimization_status": optimization_status,
        "minimization_converged": minimization_converged,
        "canonical_isomeric_smiles": canonical_smiles,
        "formal_charge": formal_charge,
        "heavy_atom_count": heavy_atom_count,
        "unspecified_stereo": list(unspecified_stereo),
        "identity_roundtrip_valid": True,
    }
    receipt = store.put_json(
        receipt_payload,
        producer="protbind.redocking.ligand-preparation-receipt",
        producer_version=__version__,
        source="validation-only-derived:native-pose-commitment",
        license=native_pose.license,
    )
    return RedockingLigandPreparation(
        ligand_3d=output,
        identity=identity,
        receipt=receipt,
        canonical_isomeric_smiles=canonical_smiles,
        formal_charge=formal_charge,
        heavy_atom_count=heavy_atom_count,
        seed=seed,
        rdkit_seed=rdkit_seed,
        minimization_method=minimization_method,
        minimization_converged=minimization_converged,
        native_pose_commitment=_native_pose_commitment(native_pose),
    )


def reference_commitment(
    case_id: str, native_pose: ArtifactRef, native_identity: ArtifactRef
) -> str:
    if not _SAFE_ID.fullmatch(case_id):
        raise ValueError("redocking case_id must be a safe identifier")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "case_id": case_id,
                "native_pose_sha256": native_pose.sha256,
                "native_identity_sha256": native_identity.sha256,
            }
        )
    )


def seal_validation_reference(
    case_id: str,
    native_pose: ArtifactRef,
    native_identity: ArtifactRef,
) -> SealedValidationReference:
    return SealedValidationReference(
        case_id=case_id,
        native_pose=native_pose,
        native_identity=native_identity,
        commitment=reference_commitment(case_id, native_pose, native_identity),
    )


def build_redocking_case(
    *,
    case_id: str,
    receptor: ArtifactRef,
    receptor_preparation_receipt: ArtifactRef,
    ligand: RedockingLigandPreparation,
    box: NativeDerivedBox,
    sealed_reference: SealedValidationReference,
    seed: int,
) -> RedockingCase:
    if sealed_reference.case_id != case_id:
        raise ValueError("sealed reference case_id does not match redocking case")
    if box.native_pose_commitment != _native_pose_commitment(sealed_reference.native_pose):
        raise ValueError("box and sealed reference derive from different native poses")
    if ligand.identity.sha256 != sealed_reference.native_identity.sha256:
        raise ValueError("ligand preparation and sealed reference use different identities")
    if ligand.native_pose_commitment != _native_pose_commitment(
        sealed_reference.native_pose
    ):
        raise ValueError("ligand preparation and sealed reference derive from different poses")
    if ligand.seed != seed:
        raise ValueError("ligand preparation seed does not match redocking case seed")
    return RedockingCase(
        case_id=case_id,
        receptor=receptor,
        receptor_preparation_receipt=receptor_preparation_receipt,
        ligand=ligand.ligand_3d,
        ligand_identity=ligand.identity,
        ligand_preparation_receipt=ligand.receipt,
        box=box,
        seed=seed,
        reference_commitment=sealed_reference.commitment,
    )


def symmetry_rmsd(
    store: ArtifactStore,
    reference_pose: ArtifactRef,
    predicted_pose: ArtifactRef,
) -> SymmetryRMSDResult:
    """Compute same-frame symmetry-aware RMSD without centering or fitting."""

    Chem, _, _ = _rdkit()
    reference = _load_single_ligand(store, reference_pose)
    predicted = _load_single_ligand(store, predicted_pose)
    reference_identity = _molecule_identity(Chem.RemoveHs(reference, sanitize=True))
    predicted_identity = _molecule_identity(Chem.RemoveHs(predicted, sanitize=True))
    identity_fields = ("canonical_isomeric_smiles", "formal_charge", "heavy_atom_count")
    if any(reference_identity[key] != predicted_identity[key] for key in identity_fields):
        raise ValueError("reference and predicted poses do not encode the same ligand identity")
    try:
        from spyrmsd.molecule import Molecule
        from spyrmsd.rmsd import rmsdwrapper
    except ImportError as exc:
        raise RedockingCapabilityError(
            "sPyRMSD is required for symmetry-aware redocking RMSD"
        ) from exc
    reference_molecule = Molecule.from_rdkit(Chem.Mol(reference))
    predicted_molecule = Molecule.from_rdkit(Chem.Mol(predicted))
    try:
        values = rmsdwrapper(
            reference_molecule,
            predicted_molecule,
            symmetry=True,
            center=False,
            minimize=False,
            strip=True,
        )
        value = values[0] if isinstance(values, list | tuple) else values
        result = float(value)
    except (IndexError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("sPyRMSD could not compare the ligand graphs") from exc
    return SymmetryRMSDResult(value_angstrom=result)


def select_holdout_manifest(
    candidates: Iterable[RedockingBenchmarkCandidate],
    *,
    dataset_name: str,
    dataset_version: str,
    dataset_license: str,
    count: int = 10,
    namespace: str = "protbind-redock-v1",
    excluded_complex_ids: Iterable[str] = (),
    dataset_source: ArtifactRef | None = None,
    candidate_list: ArtifactRef | None = None,
    eligibility_policy: dict[str, Any] | None = None,
) -> HoldoutSelectionManifest:
    """Filter and hash-sort a benchmark suite without inspecting model results."""

    if not dataset_name.strip() or not dataset_version.strip() or not dataset_license.strip():
        raise ValueError("dataset name, version, and license are required")
    if count < 1:
        raise ValueError("holdout count must be positive")
    if not namespace.strip():
        raise ValueError("holdout namespace cannot be empty")
    values = tuple(candidates)
    identifiers = [candidate.complex_id for candidate in values]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("holdout candidates contain duplicate complex IDs")
    explicitly_excluded = frozenset(excluded_complex_ids)
    eligible: list[RedockingBenchmarkCandidate] = []
    exclusions: list[HoldoutExclusion] = []
    for candidate in values:
        reasons = list(candidate.exclusion_reasons())
        if candidate.complex_id in explicitly_excluded:
            reasons.append("development_or_explicitly_excluded_case")
        if reasons:
            exclusions.append(
                HoldoutExclusion(candidate.complex_id, tuple(sorted(set(reasons))))
            )
        else:
            eligible.append(candidate)
    eligible.sort(
        key=lambda candidate: (
            sha256_bytes(f"{namespace}:{candidate.complex_id}".encode()),
            candidate.complex_id,
        )
    )
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} eligible redocking cases remain; {count} requested"
        )
    selected = tuple(eligible[:count])
    for candidate in eligible[count:]:
        exclusions.append(
            HoldoutExclusion(
                candidate.complex_id,
                ("not_in_first_n_after_hash_sort",),
            )
        )
    exclusions.sort(key=lambda item: item.complex_id)
    body = {
        "schema_version": "1.1",
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_license": dataset_license,
        "dataset_source": (
            dataset_source.to_dict() if dataset_source is not None else None
        ),
        "candidate_list": (
            candidate_list.to_dict() if candidate_list is not None else None
        ),
        "eligibility_policy": eligibility_policy,
        "namespace": namespace,
        "requested_count": count,
        "selection_rule": (
            "declared eligibility filter, then sha256(namespace + ':' + complex_id), "
            "then complex_id"
        ),
        "selected": [candidate.to_dict() for candidate in selected],
        "exclusions": [exclusion.to_dict() for exclusion in exclusions],
    }
    selection_hash = sha256_bytes(canonical_json_bytes(body))
    return HoldoutSelectionManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_license=dataset_license,
        namespace=namespace,
        requested_count=count,
        selected=selected,
        exclusions=tuple(exclusions),
        selection_hash=selection_hash,
        dataset_source=dataset_source,
        candidate_list=candidate_list,
        eligibility_policy=eligibility_policy,
    )


def persist_holdout_manifest(
    store: ArtifactStore, manifest: HoldoutSelectionManifest
) -> ArtifactRef:
    """Freeze selected IDs, files, licenses, policy, and selection hash as an artifact."""

    return store.put_json(
        manifest.to_dict(),
        producer="protbind.redocking.holdout-manifest",
        producer_version=__version__,
        license=manifest.dataset_license,
    )
