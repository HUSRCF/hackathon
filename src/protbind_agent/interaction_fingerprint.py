"""Deterministic ProLIF interaction-fingerprint metrics.

The module keeps the interaction vocabulary and comparison semantics shared by
the validation worker and scientific redocking regressions.  Labels describe a
protein residue and interaction type; they are not affinity measurements.
"""

from __future__ import annotations

import importlib
import io
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import ArtifactStore
from .models import ArtifactRef

PROLIF_INTERACTIONS = (
    "Hydrophobic",
    "HBDonor",
    "HBAcceptor",
    "PiStacking",
    "Anionic",
    "Cationic",
    "CationPi",
    "PiCation",
)


@dataclass(frozen=True, slots=True)
class ProLIFLigandPreparation:
    """Hash-bound ligand prepared only for interaction-fingerprint analysis."""

    prepared_ligand: ArtifactRef
    receipt: ArtifactRef
    hydrogens_added: int
    heavy_atom_max_coordinate_delta_angstrom: float


@dataclass(frozen=True, slots=True)
class ProLIFReceptorPreparation:
    """Receipted binding-site subset for local interaction fingerprints."""

    prepared_receptor: ArtifactRef
    receipt: ArtifactRef
    selected_residue_count: int
    selected_atom_count: int
    coordinate_max_delta_angstrom: float


def _single_sdf_molecule(data: bytes) -> Any:
    from rdkit import Chem

    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(data),
        removeHs=False,
        sanitize=True,
        strictParsing=True,
    )
    records = list(supplier)
    if len(records) != 1 or records[0] is None:
        raise ValueError("ProLIF ligand preparation requires exactly one readable SDF record")
    molecule = records[0]
    if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise ValueError("ProLIF ligand preparation requires one 3D conformer")
    return molecule


def prepare_prolif_ligand(
    store: ArtifactStore,
    ligand: ArtifactRef,
    *,
    artifact_scope: str,
) -> ProLIFLigandPreparation:
    """Add explicit H without moving heavy atoms and emit a preparation receipt.

    Existing explicit-H inputs are reused byte-for-byte.  Hydrogen-free inputs
    are converted with RDKit ``AddHs(addCoords=True)``; atom identity and every
    heavy-atom coordinate are checked before the derived artifact is accepted.
    """

    if ligand.media_type not in {"chemical/x-mdl-sdfile", "chemical/x-sdf"}:
        raise ValueError("ProLIF ligand preparation currently requires SDF input")
    if artifact_scope not in {"DOCKING_VISIBLE", "VALIDATION_ONLY"}:
        raise ValueError("ProLIF ligand artifact_scope is invalid")
    from rdkit import Chem, rdBase

    original = _single_sdf_molecule(store.read_bytes(ligand))
    original_hydrogens = sum(
        atom.GetAtomicNum() == 1 for atom in original.GetAtoms()
    )
    heavy_atoms = [atom for atom in original.GetAtoms() if atom.GetAtomicNum() != 1]
    identity = [
        (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            int(atom.GetChiralTag()),
        )
        for atom in heavy_atoms
    ]
    conformer = original.GetConformer()
    coordinates = [tuple(conformer.GetAtomPosition(atom.GetIdx())) for atom in heavy_atoms]

    if original_hydrogens:
        prepared = ligand
        hydrogens_added = 0
        maximum_delta = 0.0
        method = "reuse-existing-explicit-hydrogens"
    else:
        molecule = Chem.AddHs(Chem.Mol(original), addCoords=True)
        prepared_heavy_atoms = [
            atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
        ]
        prepared_identity = [
            (
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                int(atom.GetChiralTag()),
            )
            for atom in prepared_heavy_atoms
        ]
        if prepared_identity != identity:
            raise ValueError("RDKit AddHs changed heavy-atom identity")
        prepared_conformer = molecule.GetConformer()
        deltas = [
            sum(
                (
                    prepared_conformer.GetAtomPosition(atom.GetIdx())[axis]
                    - coordinates[index][axis]
                )
                ** 2
                for axis in range(3)
            )
            ** 0.5
            for index, atom in enumerate(prepared_heavy_atoms)
        ]
        maximum_delta = max(deltas, default=0.0)
        if maximum_delta > 1e-6:
            raise ValueError("RDKit AddHs moved a heavy atom")
        hydrogens_added = sum(
            atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()
        )
        if hydrogens_added < 1:
            raise ValueError("RDKit AddHs did not produce explicit hydrogens")
        data = (Chem.MolToMolBlock(molecule) + "\n$$$$\n").encode("utf-8")
        prepared = store.put_bytes(
            data,
            media_type="chemical/x-mdl-sdfile",
            producer="protbind.prolif-ligand-preparation",
            producer_version=__version__,
            source=ligand.artifact_id,
            license=ligand.license,
        )
        method = "RDKit AddHs(addCoords=True)"

    prepared_molecule = _single_sdf_molecule(store.read_bytes(prepared))
    output_hydrogens = sum(
        atom.GetAtomicNum() == 1 for atom in prepared_molecule.GetAtoms()
    )
    receipt = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.prolif-ligand-preparation-receipt",
            "artifact_scope": artifact_scope,
            "input_ligand": ligand.to_dict(),
            "prepared_ligand": prepared.to_dict(),
            "method": method,
            "rdkit_version": rdBase.rdkitVersion,
            "input_heavy_atom_count": len(heavy_atoms),
            "input_explicit_hydrogen_count": original_hydrogens,
            "output_explicit_hydrogen_count": output_hydrogens,
            "hydrogens_added": hydrogens_added,
            "heavy_atom_identity_preserved": True,
            "heavy_atom_max_coordinate_delta_angstrom": maximum_delta,
            "coordinate_tolerance_angstrom": 1e-6,
            "scientific_scope": "interaction-fingerprint-analysis-only",
        },
        producer="protbind.prolif-ligand-preparation-receipt",
        producer_version=__version__,
        source=ligand.artifact_id,
        license=ligand.license,
    )
    return ProLIFLigandPreparation(
        prepared_ligand=prepared,
        receipt=receipt,
        hydrogens_added=hydrogens_added,
        heavy_atom_max_coordinate_delta_angstrom=maximum_delta,
    )


def _artifact_ligand_heavy_points(
    store: ArtifactStore,
    ligand: ArtifactRef,
) -> tuple[tuple[float, float, float], ...]:
    molecule = _single_sdf_molecule(store.read_bytes(ligand))
    conformer = molecule.GetConformer()
    points = tuple(
        tuple(float(value) for value in conformer.GetAtomPosition(atom.GetIdx()))
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    )
    if not points or any(not all(math.isfinite(value) for value in point) for point in points):
        raise ValueError("ProLIF pocket selection requires finite ligand heavy atoms")
    return points


def _pdb_atom_inventory(
    structure: Any,
) -> dict[tuple[str, int, str, str, str], tuple[float, float, float]]:
    atoms: dict[tuple[str, int, str, str, str], tuple[float, float, float]] = {}
    for chain in structure[0]:
        for residue in chain:
            insertion = str(residue.seqid.icode).strip("\x00 ")
            for atom in residue:
                key = (
                    chain.name,
                    int(residue.seqid.num),
                    insertion,
                    residue.name.strip().upper(),
                    atom.name.strip().upper(),
                )
                if key in atoms:
                    raise ValueError("ProLIF receptor crop requires unique atom identities")
                atoms[key] = (
                    float(atom.pos.x),
                    float(atom.pos.y),
                    float(atom.pos.z),
                )
    return atoms


def prepare_prolif_receptor(
    store: ArtifactStore,
    receptor: ArtifactRef,
    docked_ligand: ArtifactRef,
    comparison_ligand: ArtifactRef,
    *,
    cutoff_angstrom: float = 8.0,
    coordinate_tolerance_angstrom: float = 0.002,
) -> ProLIFReceptorPreparation:
    """Select the residue union near either pose and emit a verifiable PDB.

    Interaction types in :data:`PROLIF_INTERACTIONS` are local.  Cropping keeps
    unrelated distant receptor defects from poisoning RDKit bond perception,
    while the union around docked and comparison ligands prevents either pose
    from choosing the other's evidence set.  Whole residues (including their
    explicit hydrogens) are retained without coordinate modification.
    """

    for name, value in (
        ("cutoff_angstrom", cutoff_angstrom),
        ("coordinate_tolerance_angstrom", coordinate_tolerance_angstrom),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if receptor.media_type != "chemical/x-pdb":
        raise ValueError("ProLIF receptor preparation requires a PDB artifact")
    try:
        import gemmi
    except ImportError as exc:
        raise ValueError("Gemmi is required for ProLIF receptor cropping") from exc
    try:
        source_structure = gemmi.read_pdb_string(store.read_bytes(receptor).decode("utf-8"))
    except (RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Gemmi could not parse the prepared ProLIF receptor") from exc
    if len(source_structure) != 1:
        raise ValueError("ProLIF receptor crop requires exactly one coordinate model")
    source_inventory = _pdb_atom_inventory(source_structure)
    docked_points = _artifact_ligand_heavy_points(store, docked_ligand)
    comparison_points = _artifact_ligand_heavy_points(store, comparison_ligand)
    selection_points = (*docked_points, *comparison_points)
    working = source_structure.clone()
    selected_residue_ids: list[str] = []
    for chain_index in reversed(range(len(working[0]))):
        chain = working[0][chain_index]
        for residue_index in reversed(range(len(chain))):
            residue = chain[residue_index]
            minimum = min(
                (
                    math.dist(
                        (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)),
                        point,
                    )
                    for atom in residue
                    if atom.element.atomic_number > 1
                    for point in selection_points
                ),
                default=math.inf,
            )
            if minimum > float(cutoff_angstrom):
                del chain[residue_index]
                continue
            insertion = str(residue.seqid.icode).strip("\x00 ")
            selected_residue_ids.append(
                f"{chain.name}:{residue.name.strip().upper()}:{residue.seqid.num}{insertion}"
            )
        if len(chain) == 0:
            del working[0][chain_index]
    if not selected_residue_ids:
        raise ValueError("ProLIF receptor crop selected no binding-site residues")
    working.connections.clear()
    output_bytes = working.make_pdb_string().encode("utf-8")
    try:
        roundtrip = gemmi.read_pdb_string(output_bytes.decode("utf-8"))
    except (RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Gemmi could not round-trip the ProLIF receptor crop") from exc
    output_inventory = _pdb_atom_inventory(roundtrip)
    if not output_inventory or any(key not in source_inventory for key in output_inventory):
        raise ValueError("ProLIF receptor crop changed atom identity")
    maximum_delta = max(
        (
            math.dist(source_inventory[key], output_inventory[key])
            for key in output_inventory
        ),
        default=0.0,
    )
    if maximum_delta > float(coordinate_tolerance_angstrom):
        raise ValueError("ProLIF receptor crop moved an atom beyond tolerance")
    explicit_hydrogens = sum(
        atom.element.atomic_number == 1
        for chain in roundtrip[0]
        for residue in chain
        for atom in residue
    )
    if explicit_hydrogens < 1:
        raise ValueError("ProLIF receptor crop requires existing explicit hydrogens")
    output = store.put_bytes(
        output_bytes,
        media_type="chemical/x-pdb",
        producer="protbind.prolif-receptor-pocket-crop",
        producer_version=__version__,
        source=receptor.artifact_id,
        license=receptor.license,
    )
    selected_residue_ids.sort()
    receipt = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.prolif-receptor-preparation-receipt",
            "input_receptor": receptor.to_dict(),
            "docked_ligand": docked_ligand.to_dict(),
            "comparison_ligand": comparison_ligand.to_dict(),
            "prepared_receptor": output.to_dict(),
            "method": "whole-residue union around docked and comparison heavy atoms",
            "cutoff_angstrom": float(cutoff_angstrom),
            "selected_residue_count": len(selected_residue_ids),
            "selected_residue_ids": selected_residue_ids,
            "selected_atom_count": len(output_inventory),
            "explicit_hydrogen_count": explicit_hydrogens,
            "atom_identity_preserved": True,
            "coordinate_max_delta_angstrom": maximum_delta,
            "coordinate_tolerance_angstrom": float(coordinate_tolerance_angstrom),
            "scientific_scope": "interaction-fingerprint-analysis-only",
        },
        producer="protbind.prolif-receptor-preparation-receipt",
        producer_version=__version__,
        source=receptor.artifact_id,
        license=receptor.license,
    )
    return ProLIFReceptorPreparation(
        prepared_receptor=output,
        receipt=receipt,
        selected_residue_count=len(selected_residue_ids),
        selected_atom_count=len(output_inventory),
        coordinate_max_delta_angstrom=maximum_delta,
    )


def _validated_labels(values: Iterable[str], name: str) -> set[str]:
    labels = set(values)
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError(f"{name} labels must be non-empty strings")
    return labels


def interaction_fingerprint_metrics(
    docked_labels: Iterable[str],
    comparison_labels: Iterable[str] | None,
    *,
    comparison_name: str | None,
) -> dict[str, Any]:
    """Summarize Jaccard, reference recovery, and prediction precision.

    Empty denominators are reported as ``None`` instead of being converted to a
    perfect score.  This prevents an interaction-free pose/reference pair from
    becoming positive consensus evidence.
    """

    docked = _validated_labels(docked_labels, "docked")
    if comparison_labels is None:
        if comparison_name is not None:
            raise ValueError("comparison_name requires comparison labels")
        comparison: set[str] | None = None
    else:
        if not isinstance(comparison_name, str) or not comparison_name:
            raise ValueError("comparison labels require a comparison_name")
        comparison = _validated_labels(comparison_labels, "comparison")

    intersection_count: int | None = None
    union_count: int | None = None
    similarity: float | None = None
    reference_recovery: float | None = None
    predicted_precision: float | None = None
    if comparison is not None:
        intersection_count = len(docked & comparison)
        union_count = len(docked | comparison)
        if union_count:
            similarity = intersection_count / union_count
        if comparison:
            reference_recovery = intersection_count / len(comparison)
        if docked:
            predicted_precision = intersection_count / len(docked)

    return {
        "ifp_similarity": similarity,
        "similarity": "Jaccard/Tanimoto over residue-and-interaction labels",
        "reference_interaction_recovery": reference_recovery,
        "predicted_interaction_precision": predicted_precision,
        "interactions": list(PROLIF_INTERACTIONS),
        "docked_labels": sorted(docked),
        "comparison": comparison_name,
        "comparison_labels": sorted(comparison) if comparison is not None else None,
        "counts": {
            "docked": len(docked),
            "comparison": len(comparison) if comparison is not None else None,
            "intersection": intersection_count,
            "union": union_count,
        },
        "semantics": (
            "Interaction-label agreement is structural evidence only; it is not "
            "an experimental affinity or binding-free-energy estimate."
        ),
    }


def load_rdkit_molecule(path: Path, *, receptor: bool) -> Any:
    """Load one prepared molecule and require explicit hydrogens."""

    from rdkit import Chem

    if receptor:
        molecule = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    elif path.suffix.lower() == ".sdf":
        molecules = [
            molecule
            for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
            if molecule is not None
        ]
        if len(molecules) != 1:
            raise ValueError("ligand input must contain exactly one readable molecule")
        molecule = molecules[0]
    elif path.suffix.lower() == ".mol":
        molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    elif path.suffix.lower() == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    elif path.suffix.lower() == ".pdb":
        molecule = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    else:
        molecule = None
    if molecule is None:
        raise ValueError("RDKit could not parse a prepared ProLIF molecule")
    if not any(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()):
        raise ValueError("ProLIF inputs require explicit hydrogens")
    return molecule


def fingerprint_labels(fingerprint: Any, ligand: Any, receptor: Any) -> set[str]:
    """Return stable ``protein-residue|interaction`` labels from ProLIF."""

    result = fingerprint.generate(ligand, receptor, metadata=False)
    if not isinstance(result, dict):
        raise ValueError("ProLIF returned a non-dictionary fingerprint")
    interaction_names = tuple(fingerprint.interactions)
    labels: set[str] = set()
    for residue_pair, vector in result.items():
        if not isinstance(residue_pair, tuple) or len(residue_pair) != 2:
            raise ValueError("ProLIF returned an invalid residue pair")
        bits = list(vector)
        if len(bits) != len(interaction_names):
            raise ValueError("ProLIF interaction vector length changed")
        protein_residue = str(residue_pair[1])
        for interaction, present in zip(interaction_names, bits, strict=True):
            if bool(present):
                labels.add(f"{protein_residue}|{interaction}")
    return labels


def compare_prolif_paths(
    *,
    docked_ligand_path: Path,
    docked_receptor_path: Path,
    comparison_ligand_path: Path | None = None,
    comparison_receptor_path: Path | None = None,
    comparison_name: str | None = None,
    prolif_module: Any | None = None,
) -> dict[str, Any]:
    """Compute shared ProLIF metrics from prepared, hydrogenated structures."""

    if (comparison_ligand_path is None) != (comparison_receptor_path is None):
        raise ValueError("comparison ligand and receptor must be supplied as a pair")
    if comparison_ligand_path is None and comparison_name is not None:
        raise ValueError("comparison_name requires comparison structures")
    if comparison_ligand_path is not None and (
        not isinstance(comparison_name, str) or not comparison_name
    ):
        raise ValueError("comparison structures require a comparison_name")

    prolif = prolif_module or importlib.import_module("prolif")
    docked_ligand = prolif.Molecule.from_rdkit(
        load_rdkit_molecule(docked_ligand_path, receptor=False)
    )
    docked_receptor = prolif.Molecule.from_rdkit(
        load_rdkit_molecule(docked_receptor_path, receptor=True)
    )
    fingerprint = prolif.Fingerprint(list(PROLIF_INTERACTIONS), count=False)
    docked_labels = fingerprint_labels(fingerprint, docked_ligand, docked_receptor)

    comparison_labels: set[str] | None = None
    if comparison_ligand_path is not None and comparison_receptor_path is not None:
        comparison_ligand = prolif.Molecule.from_rdkit(
            load_rdkit_molecule(comparison_ligand_path, receptor=False)
        )
        comparison_receptor = prolif.Molecule.from_rdkit(
            load_rdkit_molecule(comparison_receptor_path, receptor=True)
        )
        comparison_labels = fingerprint_labels(
            fingerprint, comparison_ligand, comparison_receptor
        )
    return interaction_fingerprint_metrics(
        docked_labels,
        comparison_labels,
        comparison_name=comparison_name,
    )
