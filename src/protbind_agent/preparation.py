"""Fail-closed receptor preparation for ordinary non-covalent systems.

The helpers in this module deliberately separate two operations:

* :func:`extract_redocking_receptor` performs deterministic structural gating and
  produces a protein-only receptor.  It never repairs chemistry.
* :func:`conservative_repair` lets PDBFixer add missing atoms and hydrogens, but
  explicitly disables missing-residue (loop) reconstruction.

Keeping the gate before repair prevents a missing pocket atom, metal, covalent
ligand, or possible cofactor from being silently converted into an apparently
supported receptor.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any

from . import __version__
from .artifacts import ArtifactStore
from .models import ArtifactRef
from .structure import inspect_declared_connections


class PreparationCapabilityError(RuntimeError):
    """A required deterministic preparation dependency is unavailable."""


class ReceptorPreparationUnsupportedError(ValueError):
    """The receptor is outside the explicitly supported v1 chemistry boundary."""

    def __init__(self, code: str, message: str, *, details: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


_STANDARD_HEAVY_ATOMS: dict[str, frozenset[str]] = {
    "ALA": frozenset({"N", "CA", "C", "O", "CB"}),
    "ARG": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}),
    "ASN": frozenset({"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"}),
    "ASP": frozenset({"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"}),
    "CYS": frozenset({"N", "CA", "C", "O", "CB", "SG"}),
    "GLN": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"}),
    "GLU": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"}),
    "GLY": frozenset({"N", "CA", "C", "O"}),
    "HIS": frozenset({"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"}),
    "ILE": frozenset({"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"}),
    "LEU": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"}),
    "LYS": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"}),
    "MET": frozenset({"N", "CA", "C", "O", "CB", "CG", "SD", "CE"}),
    "PHE": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "PRO": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD"}),
    "SER": frozenset({"N", "CA", "C", "O", "CB", "OG"}),
    "THR": frozenset({"N", "CA", "C", "O", "CB", "OG1", "CG2"}),
    "TRP": frozenset(
        {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}
    ),
    "TYR": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}),
    "VAL": frozenset({"N", "CA", "C", "O", "CB", "CG1", "CG2"}),
}
_WATER_NAMES = frozenset({"HOH", "WAT", "DOD"})
_METAL_ELEMENTS = frozenset(
    {
        "LI", "BE", "NA", "MG", "AL", "K", "CA", "SC", "TI", "V", "CR", "MN",
        "FE", "CO", "NI", "CU", "ZN", "GA", "RB", "SR", "Y", "ZR", "NB", "MO",
        "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "CS", "BA", "PT", "AU",
        "HG", "PB",
    }
)


@dataclass(frozen=True, slots=True)
class ResidueSelector:
    """Unambiguous residue instance in the deposited coordinate model."""

    chain_id: str
    residue_name: str
    sequence_number: int
    insertion_code: str = ""

    def __post_init__(self) -> None:
        chain = self.chain_id.strip()
        name = self.residue_name.strip().upper()
        insertion = self.insertion_code.strip()
        if not chain or len(chain) > 16:
            raise ValueError("residue selector chain_id must contain 1-16 characters")
        if not name or len(name) > 8:
            raise ValueError("residue selector residue_name must contain 1-8 characters")
        if len(insertion) > 1:
            raise ValueError("residue selector insertion_code must contain at most one character")
        object.__setattr__(self, "chain_id", chain)
        object.__setattr__(self, "residue_name", name)
        object.__setattr__(self, "insertion_code", insertion)

    def matches(self, chain: Any, residue: Any) -> bool:
        insertion = str(residue.seqid.icode).strip("\x00 ")
        return (
            chain.name == self.chain_id
            and residue.name.upper() == self.residue_name
            and int(residue.seqid.num) == self.sequence_number
            and insertion == self.insertion_code
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "residue_name": self.residue_name,
            "sequence_number": self.sequence_number,
            "insertion_code": self.insertion_code,
        }


@dataclass(frozen=True, slots=True)
class RemovedComponent:
    residue_id: str
    residue_name: str
    category: str
    atom_count: int
    minimum_site_distance_angstrom: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "residue_id": self.residue_id,
            "residue_name": self.residue_name,
            "category": self.category,
            "atom_count": self.atom_count,
            "minimum_site_distance_angstrom": self.minimum_site_distance_angstrom,
        }


@dataclass(frozen=True, slots=True)
class ReceptorExtractionResult:
    structure: ArtifactRef
    receipt: ArtifactRef
    native_ligand: ResidueSelector
    removed_components: tuple[RemovedComponent, ...]
    alternate_location_atoms: int
    alternate_conformers_removed: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConservativeRepairResult:
    structure: ArtifactRef
    receipt: ArtifactRef
    missing_residue_count: int
    added_heavy_atom_count: int
    warnings: tuple[str, ...]
    removed_hydrogen_count: int = 0
    original_heavy_atom_max_coordinate_delta_angstrom: float = 0.0


@dataclass(frozen=True, slots=True)
class RestrainedSidechainOptimizationResult:
    structure: ArtifactRef
    receipt: ArtifactRef
    iteration_limit: int
    fixed_original_heavy_atom_count: int
    mobile_added_heavy_atom_count: int
    original_heavy_atom_max_coordinate_delta_angstrom: float
    minimum_nonbonded_distance_ratio: float
    chirality_center_count: int
    initial_energy_kj_mol: float
    final_energy_kj_mol: float


@dataclass(frozen=True, slots=True)
class ReceptorPreparationResult:
    structure: ArtifactRef
    receipt: ArtifactRef
    extraction: ReceptorExtractionResult
    repair: ConservativeRepairResult


def _gemmi() -> Any:
    try:
        import gemmi
    except ImportError as exc:
        raise PreparationCapabilityError(
            "Gemmi is required for deterministic receptor extraction"
        ) from exc
    return gemmi


def _parse_structure(data: bytes) -> Any:
    gemmi = _gemmi()
    try:
        text = data.decode("utf-8")
        if text.lstrip().lower().startswith("data_"):
            structure = gemmi.make_structure_from_block(
                gemmi.cif.read_string(text).sole_block()
            )
        else:
            structure = gemmi.read_pdb_string(text)
    except (RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Gemmi could not parse receptor coordinates") from exc
    if len(structure) == 0:
        raise ValueError("receptor contains no coordinate model")
    return structure


def _position(atom: Any) -> tuple[float, float, float]:
    return (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))


def _distance_to_site(atom: Any, center: tuple[float, float, float]) -> float:
    return math.dist(_position(atom), center)


def _residue_id(chain: Any, residue: Any) -> str:
    insertion = str(residue.seqid.icode).strip("\x00 ")
    return f"{chain.name}:{residue.name.upper()}:{residue.seqid.num}{insertion}"


def _validate_site(center: tuple[float, float, float], radius: float) -> None:
    if len(center) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value)
        for value in center
    ):
        raise ValueError("site center must contain three finite numbers")
    if isinstance(radius, bool) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("pocket radius must be a finite positive number")


def _resolve_altlocs(
    structure: Any,
    *,
    site_center: tuple[float, float, float],
    pocket_radius: float,
) -> tuple[int, int, tuple[str, ...]]:
    observed = 0
    removed = 0
    warnings: list[str] = []
    for chain in structure[0]:
        for residue in chain:
            groups: dict[str, list[Any]] = {}
            for atom in residue:
                groups.setdefault(atom.name.strip().upper(), []).append(atom)
                if str(atom.altloc).strip("\x00 "):
                    observed += 1
            for atom_name, atoms in groups.items():
                if len(atoms) == 1:
                    if str(atoms[0].altloc).strip("\x00 "):
                        atoms[0].altloc = "\x00"
                    continue
                best_occupancy = max(float(atom.occ) for atom in atoms)
                best = [
                    atom
                    for atom in atoms
                    if math.isclose(float(atom.occ), best_occupancy, abs_tol=1e-6)
                ]
                residue_in_pocket = any(
                    _distance_to_site(atom, site_center) <= pocket_radius for atom in atoms
                )
                if len(best) > 1 and residue_in_pocket:
                    raise ReceptorPreparationUnsupportedError(
                        "AMBIGUOUS_POCKET_ALTLOC",
                        "equal-occupancy alternate conformers occur inside the docking pocket",
                        details=(_residue_id(chain, residue) + f":{atom_name}",),
                    )
                best.sort(
                    key=lambda atom: (
                        str(atom.altloc).strip("\x00 ") or " ",
                        _position(atom),
                    )
                )
                chosen = best[0]
                if len(best) > 1:
                    warnings.append(
                        f"outside-pocket altloc tie resolved lexically at "
                        f"{_residue_id(chain, residue)}:{atom_name}"
                    )
                for index in reversed(range(len(residue))):
                    candidate = residue[index]
                    if candidate.name.strip().upper() == atom_name and candidate is not chosen:
                        del residue[index]
                        removed += 1
                chosen.altloc = "\x00"
    return observed, removed, tuple(warnings)


def extract_redocking_receptor(
    store: ArtifactStore,
    structure: ArtifactRef,
    *,
    native_ligand: ResidueSelector,
    site_center: tuple[float, float, float],
    pocket_radius: float = 6.0,
) -> ReceptorExtractionResult:
    """Extract one dry protein receptor while refusing unsupported pocket chemistry.

    The native ligand is removed only by its full residue instance.  Waters are
    removed under an explicit dry-receptor policy.  Every other non-protein
    component is treated as a possible required cofactor and rejected rather than
    silently discarded.
    """

    _validate_site(site_center, pocket_radius)
    source_path = store.resolve(structure)
    connection_check = inspect_declared_connections(source_path)
    if connection_check.covalent_detected:
        raise ReceptorPreparationUnsupportedError(
            "COVALENT_LIGAND",
            "receptor declares a covalent protein/ligand connection",
            details=(connection_check.status,),
        )
    parsed = _parse_structure(source_path.read_bytes())
    if len(parsed) != 1:
        raise ReceptorPreparationUnsupportedError(
            "MULTI_MODEL_RECEPTOR",
            "redocking receptor must contain exactly one coordinate model",
        )
    working = parsed.clone()
    alternate_atoms, alternate_removed, altloc_warnings = _resolve_altlocs(
        working,
        site_center=site_center,
        pocket_radius=pocket_radius,
    )
    model = working[0]
    removed_components: list[RemovedComponent] = []
    unsupported_components: list[str] = []
    metal_elements: set[str] = set()
    ligand_matches = 0
    ligand_heavy_atoms = 0
    protein_residues = 0
    protein_chains: set[str] = set()
    warnings = list(altloc_warnings)

    for chain in model:
        for residue_index in reversed(range(len(chain))):
            residue = chain[residue_index]
            name = residue.name.upper()
            atoms = tuple(residue)
            elements = {
                atom.element.name.upper()
                for atom in atoms
                if atom.element.name.upper() not in {"H", "D"}
            }
            metals = elements & _METAL_ELEMENTS
            metal_elements.update(metals)
            distances = tuple(_distance_to_site(atom, site_center) for atom in atoms)
            minimum_distance = min(distances) if distances else None
            identifier = _residue_id(chain, residue)
            if native_ligand.matches(chain, residue):
                ligand_matches += 1
                ligand_heavy_atoms = sum(
                    atom.element.name.upper() not in {"H", "D"} for atom in atoms
                )
                if metals:
                    unsupported_components.append(identifier + " (metal-containing ligand)")
                    continue
                removed_components.append(
                    RemovedComponent(
                        residue_id=identifier,
                        residue_name=name,
                        category="native_ligand",
                        atom_count=len(atoms),
                        minimum_site_distance_angstrom=minimum_distance,
                    )
                )
                del chain[residue_index]
            elif name in _STANDARD_HEAVY_ATOMS:
                protein_residues += 1
                protein_chains.add(chain.name)
            elif name in _WATER_NAMES:
                removed_components.append(
                    RemovedComponent(
                        residue_id=identifier,
                        residue_name=name,
                        category="water",
                        atom_count=len(atoms),
                        minimum_site_distance_angstrom=minimum_distance,
                    )
                )
                del chain[residue_index]
            else:
                qualifier = "metal" if metals else "possible cofactor/nonstandard component"
                unsupported_components.append(f"{identifier} ({qualifier})")

    if ligand_matches != 1:
        raise ReceptorPreparationUnsupportedError(
            "NATIVE_LIGAND_INSTANCE_MISMATCH",
            "native ligand selector must match exactly one residue",
            details=(f"matches={ligand_matches}",),
        )
    if not 1 <= ligand_heavy_atoms <= 100:
        raise ReceptorPreparationUnsupportedError(
            "UNSUPPORTED_LIGAND_SIZE",
            "native ligand must contain 1-100 heavy atoms",
            details=(f"heavy_atoms={ligand_heavy_atoms}",),
        )
    if metal_elements:
        raise ReceptorPreparationUnsupportedError(
            "METAL_SYSTEM",
            "v1 redocking does not support receptors or ligands containing metals",
            details=tuple(sorted(metal_elements)),
        )
    if unsupported_components:
        raise ReceptorPreparationUnsupportedError(
            "POSSIBLE_REQUIRED_COFACTOR",
            "non-water heterogens are not silently removed from a redocking receptor",
            details=tuple(sorted(unsupported_components)),
        )
    if not 1 <= len(protein_chains) <= 2 or protein_residues > 700:
        raise ReceptorPreparationUnsupportedError(
            "RECEPTOR_SIZE_LIMIT",
            "v1 receptor must contain 1-2 protein chains and at most 700 residues",
            details=(
                f"protein_chains={len(protein_chains)}",
                f"protein_residues={protein_residues}",
            ),
        )
    if any(len(chain_id) != 1 for chain_id in protein_chains):
        raise ReceptorPreparationUnsupportedError(
            "PDB_CHAIN_ID_UNREPRESENTABLE",
            "PDBFixer preparation currently requires one-character chain IDs",
            details=tuple(sorted(protein_chains)),
        )

    pocket_missing: list[str] = []
    outside_missing: list[str] = []
    for chain in model:
        for residue in chain:
            name = residue.name.upper()
            if name not in _STANDARD_HEAVY_ATOMS:
                continue
            actual = {
                atom.name.strip().upper()
                for atom in residue
                if atom.element.name.upper() not in {"H", "D"}
            }
            missing = tuple(sorted(_STANDARD_HEAVY_ATOMS[name] - actual))
            if not missing:
                continue
            distance = min(
                (_distance_to_site(atom, site_center) for atom in residue),
                default=math.inf,
            )
            description = f"{_residue_id(chain, residue)} missing {','.join(missing)}"
            if distance <= pocket_radius:
                pocket_missing.append(description)
            else:
                outside_missing.append(description)
    if pocket_missing:
        raise ReceptorPreparationUnsupportedError(
            "MISSING_POCKET_HEAVY_ATOMS",
            "one or more pocket residues have missing heavy atoms",
            details=tuple(pocket_missing),
        )
    if outside_missing:
        warnings.append(
            f"{len(outside_missing)} outside-pocket residues have missing heavy atoms; "
            "PDBFixer may repair them"
        )

    for chain_index in reversed(range(len(model))):
        if len(model[chain_index]) == 0:
            del model[chain_index]
    working.connections.clear()
    working.name = working.name or "protbind_redocking_receptor"
    protein_only = store.put_bytes(
        working.make_pdb_string().encode("utf-8"),
        media_type="chemical/x-pdb",
        producer="protbind.receptor.extract",
        producer_version=__version__,
        source=structure.artifact_id,
        license=structure.license,
    )
    receipt_payload = {
        "schema_version": "1.0",
        "source_artifact_id": structure.artifact_id,
        "protein_only_artifact_id": protein_only.artifact_id,
        "native_ligand": native_ligand.to_dict(),
        "site": {
            "center": [float(value) for value in site_center],
            "pocket_radius_angstrom": float(pocket_radius),
            "definition": "redock-known-site",
        },
        "connection_check": connection_check.to_dict(),
        "dry_receptor_policy": True,
        "alternate_location_atoms": alternate_atoms,
        "alternate_conformers_removed": alternate_removed,
        "altloc_selection": "highest occupancy; outside-pocket ties lexical; pocket ties rejected",
        "removed_components": [item.to_dict() for item in removed_components],
        "possible_cofactors_silently_removed": False,
        "metals_present": False,
        "pocket_missing_heavy_atoms": [],
        "outside_pocket_missing_heavy_atoms": outside_missing,
        "protein_chain_count": len(protein_chains),
        "protein_residue_count": protein_residues,
        "warnings": warnings,
    }
    receipt = store.put_json(
        receipt_payload,
        producer="protbind.receptor.extraction-receipt",
        producer_version=__version__,
        source=structure.artifact_id,
    )
    return ReceptorExtractionResult(
        structure=protein_only,
        receipt=receipt,
        native_ligand=native_ligand,
        removed_components=tuple(removed_components),
        alternate_location_atoms=alternate_atoms,
        alternate_conformers_removed=alternate_removed,
        warnings=tuple(warnings),
    )


def conservative_repair(
    store: ArtifactStore,
    structure: ArtifactRef,
    *,
    ph: float = 7.4,
) -> ConservativeRepairResult:
    """Add missing atoms/hydrogens while recording and refusing missing-loop filling."""

    if not 0 < ph < 14:
        raise ValueError("pH must be between 0 and 14")
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise PreparationCapabilityError(
            "PDBFixer and OpenMM are required for conservative receptor repair"
        ) from exc
    input_path = store.resolve(structure)
    fixer = PDBFixer(filename=str(input_path))
    fixer.findMissingResidues()
    missing_residues = tuple(
        {
            "chain_index": int(chain_index),
            "insertion_index": int(insertion_index),
            "residue_names": tuple(str(name) for name in names),
        }
        for (chain_index, insertion_index), names in sorted(fixer.missingResidues.items())
    )
    missing_residue_count = sum(len(item["residue_names"]) for item in missing_residues)
    warnings: list[str] = []
    if missing_residue_count:
        warnings.append(
            f"{missing_residue_count} missing residues were detected and deliberately not rebuilt"
        )
    # PDBFixer would otherwise build residue spans. ProtBind v1 forbids silent loop reconstruction.
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    added_heavy_atom_count = sum(len(atoms) for atoms in fixer.missingAtoms.values())
    added_heavy_atom_count += sum(len(atoms) for atoms in fixer.missingTerminals.values())
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    handle = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    output = store.put_bytes(
        handle.getvalue().encode(),
        media_type="chemical/x-pdb",
        producer="protbind.pdbfixer.conservative",
        producer_version=__version__,
        source=structure.artifact_id,
        license=structure.license,
    )
    receipt = store.put_json(
        {
            "schema_version": "1.1",
            "input_artifact_id": structure.artifact_id,
            "output_artifact_id": output.artifact_id,
            "ph": ph,
            "missing_residue_count": missing_residue_count,
            "missing_residues": missing_residues,
            "missing_residues_rebuilt": False,
            "added_heavy_atom_count": added_heavy_atom_count,
            "hydrogens_added": True,
            "heterogens_removed": False,
            "warnings": warnings,
        },
        producer="protbind.pdbfixer.receipt",
        producer_version=__version__,
        source=structure.artifact_id,
    )
    return ConservativeRepairResult(
        structure=output,
        receipt=receipt,
        missing_residue_count=missing_residue_count,
        added_heavy_atom_count=added_heavy_atom_count,
        warnings=tuple(warnings),
    )


def _heavy_atom_inventory(
    structure: Any,
) -> tuple[
    dict[tuple[str, int, str, str, str], tuple[float, float, float]],
    frozenset[tuple[str, int, str, str]],
]:
    if len(structure) != 1:
        raise ReceptorPreparationUnsupportedError(
            "MULTI_MODEL_RECEPTOR",
            "conservative repair requires exactly one receptor model",
        )
    atoms: dict[tuple[str, int, str, str, str], tuple[float, float, float]] = {}
    residues: set[tuple[str, int, str, str]] = set()
    for chain in structure[0]:
        for residue in chain:
            insertion = str(residue.seqid.icode).strip("\x00 ")
            residue_key = (
                chain.name,
                int(residue.seqid.num),
                insertion,
                residue.name.strip().upper(),
            )
            residues.add(residue_key)
            for atom in residue:
                if atom.element.atomic_number <= 1:
                    continue
                atom_key = (*residue_key, atom.name.strip().upper())
                if atom_key in atoms:
                    raise ReceptorPreparationUnsupportedError(
                        "AMBIGUOUS_ATOM_IDENTITY",
                        "conservative repair requires unique heavy-atom identities",
                        details=(str(atom_key),),
                    )
                atoms[atom_key] = _position(atom)
    return atoms, frozenset(residues)


def _missing_heavy_atom_partition(
    structure: Any,
    protected_points: tuple[tuple[float, float, float], ...],
    protected_radius: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not protected_points or any(
        len(point) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in point
        )
        for point in protected_points
    ):
        raise ValueError("protected_points must contain finite three-dimensional points")
    if (
        isinstance(protected_radius, bool)
        or not isinstance(protected_radius, int | float)
        or not math.isfinite(float(protected_radius))
        or float(protected_radius) <= 0
    ):
        raise ValueError("protected_radius must be finite and positive")
    pocket: list[str] = []
    outside: list[str] = []
    for chain in structure[0]:
        for residue in chain:
            name = residue.name.strip().upper()
            expected = _STANDARD_HEAVY_ATOMS.get(name)
            if expected is None:
                continue
            actual = {
                atom.name.strip().upper()
                for atom in residue
                if atom.element.atomic_number > 1
            }
            missing = tuple(sorted(expected - actual))
            if not missing:
                continue
            minimum = min(
                (
                    math.dist(_position(atom), point)
                    for atom in residue
                    if atom.element.atomic_number > 1
                    for point in protected_points
                ),
                default=math.inf,
            )
            description = f"{_residue_id(chain, residue)} missing {','.join(missing)}"
            (pocket if minimum <= float(protected_radius) else outside).append(
                description
            )
    return tuple(sorted(pocket)), tuple(sorted(outside))


def conservative_heavy_atom_repair(
    store: ArtifactStore,
    structure: ArtifactRef,
    *,
    protected_points: tuple[tuple[float, float, float], ...],
    protected_radius: float = 6.0,
    coordinate_tolerance_angstrom: float = 0.002,
) -> ConservativeRepairResult:
    """Repair only outside-pocket standard-residue heavy atoms.

    Existing hydrogens are removed when repair is required, and PDBFixer is not
    allowed to rebuild missing residues or add hydrogens.  Meeko remains the
    single protonation/charge authority for the downstream docking receptor.
    Every original heavy atom must survive with the same residue/atom identity
    and coordinates within the declared serialization tolerance.
    """

    if (
        isinstance(coordinate_tolerance_angstrom, bool)
        or not isinstance(coordinate_tolerance_angstrom, int | float)
        or not math.isfinite(float(coordinate_tolerance_angstrom))
        or float(coordinate_tolerance_angstrom) <= 0
    ):
        raise ValueError("coordinate tolerance must be finite and positive")
    source_structure = _parse_structure(store.read_bytes(structure))
    before_atoms, before_residues = _heavy_atom_inventory(source_structure)
    pocket_missing, outside_missing = _missing_heavy_atom_partition(
        source_structure,
        protected_points,
        protected_radius,
    )
    if pocket_missing:
        raise ReceptorPreparationUnsupportedError(
            "MISSING_POCKET_HEAVY_ATOMS",
            "pocket heavy atoms cannot be reconstructed for redocking calibration",
            details=pocket_missing,
        )

    if not outside_missing:
        receipt = store.put_json(
            {
                "schema_version": "1.0",
                "method": "PDBFixer heavy-atom-only repair",
                "input_artifact_id": structure.artifact_id,
                "output_artifact_id": structure.artifact_id,
                "repair_required": False,
                "protected_radius_angstrom": float(protected_radius),
                "protected_point_count": len(protected_points),
                "pocket_missing_heavy_atoms": [],
                "outside_pocket_missing_heavy_atoms": [],
                "missing_residue_count": 0,
                "missing_residues_rebuilt": False,
                "added_heavy_atom_count": 0,
                "hydrogens_added": False,
                "removed_hydrogen_count": 0,
                "original_heavy_atom_identity_preserved": True,
                "original_heavy_atom_max_coordinate_delta_angstrom": 0.0,
                "coordinate_tolerance_angstrom": float(
                    coordinate_tolerance_angstrom
                ),
                "warnings": [],
            },
            producer="protbind.pdbfixer.heavy-atom-only-receipt",
            producer_version=__version__,
            source=structure.artifact_id,
            license=structure.license,
        )
        return ConservativeRepairResult(
            structure=structure,
            receipt=receipt,
            missing_residue_count=0,
            added_heavy_atom_count=0,
            warnings=(),
        )

    try:
        from openmm.app import Modeller, PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise PreparationCapabilityError(
            "PDBFixer and OpenMM are required for conservative heavy-atom repair"
        ) from exc

    input_path = store.resolve(structure)
    fixer = PDBFixer(filename=str(input_path))
    fixer.findMissingResidues()
    missing_residues = tuple(
        {
            "chain_index": int(chain_index),
            "insertion_index": int(insertion_index),
            "residue_names": tuple(str(name) for name in names),
        }
        for (chain_index, insertion_index), names in sorted(fixer.missingResidues.items())
    )
    missing_residue_count = sum(len(item["residue_names"]) for item in missing_residues)
    fixer.missingResidues = {}
    modeller = Modeller(fixer.topology, fixer.positions)
    hydrogens = [
        atom
        for atom in modeller.topology.atoms()
        if atom.element is not None and atom.element.atomic_number == 1
    ]
    if hydrogens:
        modeller.delete(hydrogens)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions
    fixer.findMissingAtoms()
    reported_added_heavy_atoms = sum(len(atoms) for atoms in fixer.missingAtoms.values())
    reported_added_heavy_atoms += sum(
        len(atoms) for atoms in fixer.missingTerminals.values()
    )
    fixer.addMissingAtoms()
    handle = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    output_bytes = handle.getvalue().encode()
    output_structure = _parse_structure(output_bytes)
    after_atoms, after_residues = _heavy_atom_inventory(output_structure)
    if after_residues != before_residues:
        raise ReceptorPreparationUnsupportedError(
            "REPAIR_CHANGED_RESIDUE_IDENTITY",
            "heavy-atom repair changed the receptor residue set",
        )
    missing_original = tuple(sorted(set(before_atoms) - set(after_atoms)))
    if missing_original:
        raise ReceptorPreparationUnsupportedError(
            "REPAIR_REMOVED_ORIGINAL_HEAVY_ATOMS",
            "heavy-atom repair removed original receptor atoms",
            details=tuple(str(value) for value in missing_original[:20]),
        )
    maximum_delta = max(
        (math.dist(before_atoms[key], after_atoms[key]) for key in before_atoms),
        default=0.0,
    )
    if maximum_delta > float(coordinate_tolerance_angstrom):
        raise ReceptorPreparationUnsupportedError(
            "REPAIR_MOVED_ORIGINAL_HEAVY_ATOMS",
            "heavy-atom repair moved an original receptor atom beyond tolerance",
            details=(f"maximum_delta_angstrom={maximum_delta:.6f}",),
        )
    post_pocket_missing, post_outside_missing = _missing_heavy_atom_partition(
        output_structure,
        protected_points,
        protected_radius,
    )
    if post_pocket_missing or post_outside_missing:
        raise ReceptorPreparationUnsupportedError(
            "REPAIR_LEFT_MISSING_HEAVY_ATOMS",
            "PDBFixer did not produce a heavy-atom-complete standard receptor",
            details=tuple((*post_pocket_missing, *post_outside_missing)),
        )
    output_hydrogen_count = sum(
        atom.element.atomic_number == 1
        for chain in output_structure[0]
        for residue in chain
        for atom in residue
    )
    if output_hydrogen_count:
        raise ReceptorPreparationUnsupportedError(
            "REPAIR_ADDED_OR_RETAINED_HYDROGENS",
            "heavy-atom-only repair must leave protonation to Meeko",
            details=(f"output_hydrogen_count={output_hydrogen_count}",),
        )
    observed_added_heavy_atoms = len(set(after_atoms) - set(before_atoms))
    if observed_added_heavy_atoms != reported_added_heavy_atoms:
        raise ReceptorPreparationUnsupportedError(
            "REPAIR_ATOM_COUNT_MISMATCH",
            "PDBFixer reported and observed added-heavy-atom counts differ",
            details=(
                f"reported={reported_added_heavy_atoms}",
                f"observed={observed_added_heavy_atoms}",
            ),
        )
    warnings: list[str] = [
        f"{len(outside_missing)} outside-pocket residues had missing heavy atoms repaired"
    ]
    if missing_residue_count:
        warnings.append(
            f"{missing_residue_count} missing residues were detected and deliberately not rebuilt"
        )
    output = store.put_bytes(
        output_bytes,
        media_type="chemical/x-pdb",
        producer="protbind.pdbfixer.heavy-atom-only",
        producer_version=__version__,
        source=structure.artifact_id,
        license=structure.license,
    )
    receipt = store.put_json(
        {
            "schema_version": "1.0",
            "method": "PDBFixer heavy-atom-only repair",
            "pdbfixer_version": importlib_metadata.version("pdbfixer"),
            "openmm_version": importlib_metadata.version("openmm"),
            "input_artifact_id": structure.artifact_id,
            "output_artifact_id": output.artifact_id,
            "repair_required": True,
            "protected_radius_angstrom": float(protected_radius),
            "protected_point_count": len(protected_points),
            "pocket_missing_heavy_atoms": [],
            "outside_pocket_missing_heavy_atoms": outside_missing,
            "missing_residue_count": missing_residue_count,
            "missing_residues": missing_residues,
            "missing_residues_rebuilt": False,
            "added_heavy_atom_count": observed_added_heavy_atoms,
            "hydrogens_added": False,
            "removed_hydrogen_count": len(hydrogens),
            "original_heavy_atom_identity_preserved": True,
            "original_heavy_atom_max_coordinate_delta_angstrom": maximum_delta,
            "coordinate_tolerance_angstrom": float(coordinate_tolerance_angstrom),
            "post_repair_standard_residue_missing_heavy_atoms": [],
            "warnings": warnings,
        },
        producer="protbind.pdbfixer.heavy-atom-only-receipt",
        producer_version=__version__,
        source=structure.artifact_id,
        license=structure.license,
    )
    return ConservativeRepairResult(
        structure=output,
        receipt=receipt,
        missing_residue_count=missing_residue_count,
        added_heavy_atom_count=observed_added_heavy_atoms,
        warnings=tuple(warnings),
        removed_hydrogen_count=len(hydrogens),
        original_heavy_atom_max_coordinate_delta_angstrom=maximum_delta,
    )


_VDW_RADIUS_ANGSTROM = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
}


def _openmm_atom_key(atom: Any) -> tuple[str, int, str, str, str]:
    residue = atom.residue
    try:
        sequence_number = int(residue.id)
    except (TypeError, ValueError) as exc:
        raise ReceptorPreparationUnsupportedError(
            "UNREPRESENTABLE_OPTIMIZATION_ATOM_IDENTITY",
            "OpenMM could not preserve an integer PDB residue number",
            details=(f"chain={residue.chain.id}", f"residue={residue.id}"),
        ) from exc
    return (
        residue.chain.id,
        sequence_number,
        str(residue.insertionCode).strip("\x00 "),
        residue.name.strip().upper(),
        atom.name.strip().upper(),
    )


def _heavy_atom_elements(
    structure: Any,
) -> dict[tuple[str, int, str, str, str], str]:
    elements: dict[tuple[str, int, str, str, str], str] = {}
    for chain in structure[0]:
        for residue in chain:
            insertion = str(residue.seqid.icode).strip("\x00 ")
            residue_key = (
                chain.name,
                int(residue.seqid.num),
                insertion,
                residue.name.strip().upper(),
            )
            for atom in residue:
                if atom.element.atomic_number <= 1:
                    continue
                elements[(*residue_key, atom.name.strip().upper())] = (
                    atom.element.name.upper()
                )
    return elements


def _determinant(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> float:
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


def _chirality_signatures(
    atoms: dict[tuple[str, int, str, str, str], tuple[float, float, float]],
) -> dict[tuple[str, int, str, str, str], float]:
    residues: dict[
        tuple[str, int, str, str],
        dict[str, tuple[float, float, float]],
    ] = {}
    for (*residue_key, atom_name), position in atoms.items():
        residues.setdefault(tuple(residue_key), {})[atom_name] = position
    signatures: dict[tuple[str, int, str, str, str], float] = {}
    for residue_key, positions in residues.items():
        residue_name = residue_key[3]
        definitions: list[tuple[str, tuple[str, str, str]]] = []
        if residue_name != "GLY":
            definitions.append(("CA", ("N", "C", "CB")))
        if residue_name == "ILE":
            definitions.append(("CB", ("CA", "CG1", "CG2")))
        elif residue_name == "THR":
            definitions.append(("CB", ("CA", "OG1", "CG2")))
        for center_name, neighbor_names in definitions:
            required = (center_name, *neighbor_names)
            if any(name not in positions for name in required):
                continue
            center = positions[center_name]
            vectors = tuple(
                tuple(positions[name][axis] - center[axis] for axis in range(3))
                for name in neighbor_names
            )
            signatures[(*residue_key, center_name)] = _determinant(*vectors)
    return signatures


def restrained_sidechain_geometry_optimize(
    store: ArtifactStore,
    original_structure: ArtifactRef,
    repaired_structure: ArtifactRef,
    *,
    iteration_limit: int,
    coordinate_tolerance_angstrom: float = 0.002,
    nonbonded_distance_ratio_threshold: float = 0.60,
    nonbonded_cutoff_angstrom: float = 10.0,
    minimization_tolerance_kj_mol_nm: float = 10.0,
) -> RestrainedSidechainOptimizationResult:
    """Relax only PDBFixer-added side-chain atoms with original heavy atoms fixed.

    Temporary hydrogens are added for ff14SB parameterization and removed from the
    output.  The resulting geometry is accepted only if atom identity, original
    coordinates, bonded distances, nonbonded distances, and heavy-atom chirality
    are preserved.  This is a preparation repair, not a physical stability or
    binding-energy calculation.
    """

    if type(iteration_limit) is not int or iteration_limit < 1:
        raise ValueError("side-chain optimization iteration_limit must be positive")
    for name, value in (
        ("coordinate_tolerance_angstrom", coordinate_tolerance_angstrom),
        ("nonbonded_distance_ratio_threshold", nonbonded_distance_ratio_threshold),
        ("nonbonded_cutoff_angstrom", nonbonded_cutoff_angstrom),
        ("minimization_tolerance_kj_mol_nm", minimization_tolerance_kj_mol_nm),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < float(nonbonded_distance_ratio_threshold) < 1:
        raise ValueError("nonbonded distance ratio threshold must be less than one")

    original = _parse_structure(store.read_bytes(original_structure))
    repaired = _parse_structure(store.read_bytes(repaired_structure))
    original_atoms, original_residues = _heavy_atom_inventory(original)
    repaired_atoms, repaired_residues = _heavy_atom_inventory(repaired)
    if original_residues != repaired_residues:
        raise ReceptorPreparationUnsupportedError(
            "OPTIMIZATION_RESIDUE_IDENTITY_MISMATCH",
            "restrained optimization requires the same original and repaired residues",
        )
    if not set(original_atoms) <= set(repaired_atoms):
        raise ReceptorPreparationUnsupportedError(
            "OPTIMIZATION_MISSING_ORIGINAL_HEAVY_ATOMS",
            "repaired receptor is missing one or more original heavy atoms",
        )
    added_keys = frozenset(set(repaired_atoms) - set(original_atoms))
    if not added_keys:
        raise ReceptorPreparationUnsupportedError(
            "OPTIMIZATION_HAS_NO_ADDED_HEAVY_ATOMS",
            "restrained side-chain optimization requires repaired heavy atoms",
        )

    try:
        import openmm
        from openmm import LocalEnergyMinimizer, Platform, VerletIntegrator, unit
        from openmm.app import CutoffNonPeriodic, ForceField, Modeller, PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise PreparationCapabilityError(
            "OpenMM and PDBFixer are required for restrained side-chain optimization"
        ) from exc

    try:
        fixer = PDBFixer(filename=str(store.resolve(repaired_structure)))
        fixer.addMissingHydrogens(7.0)
        transient_hydrogen_count = sum(
            atom.element is not None and atom.element.atomic_number == 1
            for atom in fixer.topology.atoms()
        )
        forcefield = ForceField("amber14-all.xml")
        system = forcefield.createSystem(
            fixer.topology,
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=float(nonbonded_cutoff_angstrom) * unit.angstrom,
            constraints=None,
            rigidWater=True,
        )
    except Exception as exc:
        raise ReceptorPreparationUnsupportedError(
            "SIDECHAIN_FORCEFIELD_PARAMETERIZATION_FAILED",
            "OpenMM ff14SB could not parameterize the repaired standard receptor",
            details=(type(exc).__name__, str(exc)[:256]),
        ) from exc
    topology_atoms = list(fixer.topology.atoms())
    topology_keys: dict[int, tuple[str, int, str, str, str]] = {}
    fixed_count = 0
    mobile_added_count = 0
    for atom in topology_atoms:
        if atom.element is None or atom.element.atomic_number <= 1:
            continue
        atom_key = _openmm_atom_key(atom)
        topology_keys[atom.index] = atom_key
        if atom_key in original_atoms:
            system.setParticleMass(atom.index, 0 * unit.dalton)
            fixed_count += 1
        elif atom_key in added_keys:
            mobile_added_count += 1
        else:
            raise ReceptorPreparationUnsupportedError(
                "OPTIMIZATION_UNEXPECTED_HEAVY_ATOM",
                "hydrogen parameterization introduced an unexpected heavy atom",
                details=(str(atom_key),),
            )
    if fixed_count != len(original_atoms) or mobile_added_count != len(added_keys):
        raise ReceptorPreparationUnsupportedError(
            "OPTIMIZATION_ATOM_IDENTITY_MISMATCH",
            "OpenMM topology does not match the original/repaired heavy-atom inventories",
            details=(
                f"fixed={fixed_count}/{len(original_atoms)}",
                f"mobile_added={mobile_added_count}/{len(added_keys)}",
            ),
        )

    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    try:
        platform = Platform.getPlatformByName("CPU")
        context = openmm.Context(system, integrator, platform, {"Threads": "1"})
    except Exception as exc:
        raise PreparationCapabilityError(
            "OpenMM CPU platform is required for deterministic side-chain optimization"
        ) from exc
    context.setPositions(fixer.positions)
    initial_state = context.getState(getEnergy=True)
    initial_energy = float(
        initial_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    )
    try:
        LocalEnergyMinimizer.minimize(
            context,
            tolerance=(
                float(minimization_tolerance_kj_mol_nm)
                * unit.kilojoules_per_mole
                / unit.nanometer
            ),
            maxIterations=iteration_limit,
        )
    except Exception as exc:
        raise ReceptorPreparationUnsupportedError(
            "SIDECHAIN_OPTIMIZATION_FAILED",
            "OpenMM could not minimize the repaired side-chain geometry",
            details=(type(exc).__name__, str(exc)[:256]),
        ) from exc
    final_state = context.getState(getPositions=True, getEnergy=True)
    final_energy = float(
        final_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    )
    if not math.isfinite(initial_energy) or not math.isfinite(final_energy):
        raise ReceptorPreparationUnsupportedError(
            "NONFINITE_SIDECHAIN_OPTIMIZATION_ENERGY",
            "side-chain optimization produced a non-finite force-field diagnostic",
        )

    modeller = Modeller(fixer.topology, final_state.getPositions())
    hydrogens = [
        atom
        for atom in modeller.topology.atoms()
        if atom.element is not None and atom.element.atomic_number == 1
    ]
    modeller.delete(hydrogens)
    handle = io.StringIO()
    PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    output_bytes = handle.getvalue().encode()
    output_structure = _parse_structure(output_bytes)
    output_atoms, output_residues = _heavy_atom_inventory(output_structure)
    if output_residues != repaired_residues or set(output_atoms) != set(repaired_atoms):
        raise ReceptorPreparationUnsupportedError(
            "OPTIMIZATION_CHANGED_ATOM_IDENTITY",
            "side-chain optimization changed receptor residue or heavy-atom identity",
        )
    maximum_delta = max(
        (math.dist(original_atoms[key], output_atoms[key]) for key in original_atoms),
        default=0.0,
    )
    if maximum_delta > float(coordinate_tolerance_angstrom):
        raise ReceptorPreparationUnsupportedError(
            "OPTIMIZATION_MOVED_ORIGINAL_HEAVY_ATOMS",
            "side-chain optimization moved an original receptor heavy atom",
            details=(f"maximum_delta_angstrom={maximum_delta:.6f}",),
        )

    direct_bonds: set[
        frozenset[tuple[str, int, str, str, str]]
    ] = set()
    for first, second in fixer.topology.bonds():
        first_key = topology_keys.get(first.index)
        second_key = topology_keys.get(second.index)
        if first_key is not None and second_key is not None:
            direct_bonds.add(frozenset((first_key, second_key)))
    bonded_violations: list[str] = []
    for pair in direct_bonds:
        if not pair & added_keys:
            continue
        first_key, second_key = tuple(pair)
        distance = math.dist(output_atoms[first_key], output_atoms[second_key])
        if not 0.90 <= distance <= 2.20:
            bonded_violations.append(
                f"{first_key}<->{second_key} distance={distance:.4f}"
            )
    if bonded_violations:
        raise ReceptorPreparationUnsupportedError(
            "INVALID_OPTIMIZED_BOND_LENGTH",
            "optimized added atoms have implausible covalent bond lengths",
            details=tuple(sorted(bonded_violations)[:20]),
        )

    elements = _heavy_atom_elements(output_structure)
    minimum_ratio = math.inf
    minimum_pair: tuple[
        tuple[str, int, str, str, str],
        tuple[str, int, str, str, str],
    ] | None = None
    nonbonded_violations: list[str] = []
    ordered_keys = sorted(output_atoms)
    for first_index, first_key in enumerate(ordered_keys):
        for second_key in ordered_keys[first_index + 1 :]:
            if first_key not in added_keys and second_key not in added_keys:
                continue
            pair = frozenset((first_key, second_key))
            if pair in direct_bonds:
                continue
            first_radius = _VDW_RADIUS_ANGSTROM.get(elements[first_key])
            second_radius = _VDW_RADIUS_ANGSTROM.get(elements[second_key])
            if first_radius is None or second_radius is None:
                raise ReceptorPreparationUnsupportedError(
                    "UNSUPPORTED_OPTIMIZATION_ELEMENT",
                    "side-chain geometry gate lacks a van der Waals radius",
                    details=(elements[first_key], elements[second_key]),
                )
            distance = math.dist(output_atoms[first_key], output_atoms[second_key])
            ratio = distance / (first_radius + second_radius)
            if ratio < minimum_ratio:
                minimum_ratio = ratio
                minimum_pair = (first_key, second_key)
            if ratio < float(nonbonded_distance_ratio_threshold):
                nonbonded_violations.append(
                    f"{first_key}<->{second_key} ratio={ratio:.4f} distance={distance:.4f}"
                )
    if nonbonded_violations:
        raise ReceptorPreparationUnsupportedError(
            "OPTIMIZED_SIDECHAIN_STERIC_CLASH",
            "optimized added atoms retain an unacceptable heavy-atom overlap",
            details=tuple(sorted(nonbonded_violations)[:20]),
        )
    if not math.isfinite(minimum_ratio) or minimum_pair is None:
        raise ReceptorPreparationUnsupportedError(
            "MISSING_OPTIMIZATION_GEOMETRY_METRIC",
            "side-chain optimization produced no nonbonded heavy-atom metric",
        )

    reference_chirality = _chirality_signatures(repaired_atoms)
    output_chirality = _chirality_signatures(output_atoms)
    if set(reference_chirality) != set(output_chirality):
        raise ReceptorPreparationUnsupportedError(
            "OPTIMIZATION_CHIRALITY_IDENTITY_MISMATCH",
            "side-chain optimization changed the set of checkable chiral centers",
        )
    flipped: list[str] = []
    collapsed: list[str] = []
    for center, before_value in reference_chirality.items():
        after_value = output_chirality[center]
        if abs(before_value) < 0.10 or abs(after_value) < 0.10:
            collapsed.append(
                f"{center} before={before_value:.6f} after={after_value:.6f}"
            )
        elif before_value * after_value <= 0:
            flipped.append(
                f"{center} before={before_value:.6f} after={after_value:.6f}"
            )
    if collapsed or flipped:
        raise ReceptorPreparationUnsupportedError(
            "OPTIMIZATION_CHANGED_CHIRALITY",
            "side-chain optimization inverted or collapsed a heavy-atom chiral center",
            details=tuple((*sorted(flipped), *sorted(collapsed))[:20]),
        )

    output = store.put_bytes(
        output_bytes,
        media_type="chemical/x-pdb",
        producer="protbind.openmm.restrained-sidechain-geometry",
        producer_version=__version__,
        source=repaired_structure.artifact_id,
        license=repaired_structure.license,
    )
    receipt = store.put_json(
        {
            "schema_version": "1.0",
            "method": "OpenMM ff14SB cutoff restrained side-chain minimization",
            "input_original_receptor": original_structure.artifact_id,
            "input_repaired_receptor": repaired_structure.artifact_id,
            "output_receptor": output.artifact_id,
            "openmm_version": importlib_metadata.version("openmm"),
            "pdbfixer_version": importlib_metadata.version("pdbfixer"),
            "force_field": "amber14-all.xml",
            "nonbonded_method": "CutoffNonPeriodic",
            "nonbonded_cutoff_angstrom": float(nonbonded_cutoff_angstrom),
            "platform": "CPU",
            "platform_threads": 1,
            "iteration_limit": iteration_limit,
            "minimization_tolerance_kj_mol_nm": float(
                minimization_tolerance_kj_mol_nm
            ),
            "initial_energy_kj_mol": initial_energy,
            "final_energy_kj_mol": final_energy,
            "energy_semantics": (
                "Preparation-only force-field diagnostic; not binding energy, free energy, "
                "or a physical stability claim."
            ),
            "fixed_original_heavy_atom_count": fixed_count,
            "mobile_added_heavy_atom_count": mobile_added_count,
            "transient_hydrogen_count": transient_hydrogen_count,
            "output_hydrogen_count": 0,
            "original_heavy_atom_identity_preserved": True,
            "original_heavy_atom_max_coordinate_delta_angstrom": maximum_delta,
            "coordinate_tolerance_angstrom": float(coordinate_tolerance_angstrom),
            "bonded_added_atom_geometry_valid": True,
            "nonbonded_distance_ratio_threshold": float(
                nonbonded_distance_ratio_threshold
            ),
            "minimum_nonbonded_distance_ratio": minimum_ratio,
            "minimum_nonbonded_pair": [str(value) for value in minimum_pair],
            "chirality_reference": "PDBFixer heavy-atom repair output",
            "chirality_center_count": len(reference_chirality),
            "chirality_signs_preserved": True,
            "meeko_rdkit_validation_required_downstream": True,
        },
        producer="protbind.openmm.restrained-sidechain-geometry-receipt",
        producer_version=__version__,
        source=repaired_structure.artifact_id,
        license=repaired_structure.license,
    )
    return RestrainedSidechainOptimizationResult(
        structure=output,
        receipt=receipt,
        iteration_limit=iteration_limit,
        fixed_original_heavy_atom_count=fixed_count,
        mobile_added_heavy_atom_count=mobile_added_count,
        original_heavy_atom_max_coordinate_delta_angstrom=maximum_delta,
        minimum_nonbonded_distance_ratio=minimum_ratio,
        chirality_center_count=len(reference_chirality),
        initial_energy_kj_mol=initial_energy,
        final_energy_kj_mol=final_energy,
    )


def prepare_redocking_receptor(
    store: ArtifactStore,
    structure: ArtifactRef,
    *,
    native_ligand: ResidueSelector,
    site_center: tuple[float, float, float],
    pocket_radius: float = 6.0,
    ph: float = 7.4,
) -> ReceptorPreparationResult:
    """Run the strict extraction gate followed by conservative PDBFixer repair."""

    extraction = extract_redocking_receptor(
        store,
        structure,
        native_ligand=native_ligand,
        site_center=site_center,
        pocket_radius=pocket_radius,
    )
    repair = conservative_repair(store, extraction.structure, ph=ph)
    warnings = tuple((*extraction.warnings, *repair.warnings))
    receipt = store.put_json(
        {
            "schema_version": "1.0",
            "source_artifact_id": structure.artifact_id,
            "prepared_receptor_artifact_id": repair.structure.artifact_id,
            "extraction_receipt_artifact_id": extraction.receipt.artifact_id,
            "repair_receipt_artifact_id": repair.receipt.artifact_id,
            "native_ligand": native_ligand.to_dict(),
            "site_center": list(site_center),
            "pocket_radius_angstrom": pocket_radius,
            "ph": ph,
            "preparation_attested": True,
            "missing_residue_spans_rebuilt": False,
            "possible_cofactors_silently_removed": False,
            "warnings": warnings,
        },
        producer="protbind.receptor.preparation-receipt",
        producer_version=__version__,
        source=structure.artifact_id,
    )
    return ReceptorPreparationResult(
        structure=repair.structure,
        receipt=receipt,
        extraction=extraction,
        repair=repair,
    )
