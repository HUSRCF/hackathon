"""Optional RDKit ingestion for SDF/SMILES/CSV/Parquet libraries.

Imports stay local so the core protocol and precomputed-feature workflow remain
usable in a minimal environment.  Missing RDKit is a hard, explicit capability
error rather than a silent chemistry downgrade.
"""

from __future__ import annotations

import csv
import functools
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tripharm import FeatureConformer, FeaturePoint, FeatureType, IndexedMolecule

_FAMILY_MAP = {
    "Donor": FeatureType.DONOR,
    "Acceptor": FeatureType.ACCEPTOR,
    "Aromatic": FeatureType.AROMATIC,
    "Hydrophobe": FeatureType.HYDROPHOBE,
    "LumpedHydrophobe": FeatureType.HYDROPHOBE,
    "PosIonizable": FeatureType.POSITIVE,
    "NegIonizable": FeatureType.NEGATIVE,
}
_METALS = frozenset(
    {3, 4, 11, 12, 13}
    | set(range(19, 33))
    | set(range(37, 52))
    | set(range(55, 85))
    | set(range(87, 119))
)


class ChemistryCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MicrostateRecord:
    microstate_id: str
    canonical_isomeric_smiles: str
    formal_charge: int
    parent_standardized_smiles: str
    enumeration_method: str = "Dimorphite-DL 2.0.2 + RDKit tautomer enumeration"
    uncertainty: str = "heuristic protonation/tautomer state; not an experimental assignment"


@dataclass(frozen=True, slots=True)
class LigandInspection:
    heavy_atom_count: int
    metal_elements: tuple[str, ...]
    unassigned_stereocenters: int
    standardized_isomeric_smiles: str
    formal_charge: int


def _rdkit() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from rdkit import Chem, RDConfig
        from rdkit.Chem import AllChem, ChemicalFeatures
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:
        raise ChemistryCapabilityError(
            "RDKit is required for chemical library ingestion; install the core "
            "science environment or provide precomputed feature JSONL"
        ) from exc
    return Chem, RDConfig, AllChem, ChemicalFeatures, rdMolStandardize


@functools.lru_cache(maxsize=1)
def _feature_factory() -> Any:
    _, rd_config, _, chemical_features, _ = _rdkit()
    return chemical_features.BuildFeatureFactory(
        str(Path(rd_config.RDDataDir) / "BaseFeatures.fdef")
    )


def _raw_molecules(path: Path) -> Iterator[tuple[str, str | None, Any]]:
    chem, _, _, _, _ = _rdkit()
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".sd"}:
        supplier = chem.SDMolSupplier(str(path), removeHs=False)
        for index, molecule in enumerate(supplier, start=1):
            if molecule is None:
                raise ValueError(f"RDKit could not parse SDF record {index}")
            name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
            yield name or f"{path.stem}-{index:06d}", None, molecule
        return
    if suffix in {".smi", ".smiles", ".txt"}:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                smiles = fields[0]
                molecule = chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise ValueError(f"invalid SMILES at {path.name}:{line_number}")
                molecule_id = fields[1] if len(fields) > 1 else f"{path.stem}-{line_number:06d}"
                yield molecule_id, smiles, molecule
        return
    rows: list[dict[str, Any]]
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ChemistryCapabilityError(
                "PyArrow is required for Parquet library ingestion"
            ) from exc
        rows = parquet.read_table(path).to_pylist()
    else:
        raise ValueError(
            "unsupported chemical input; use .sdf, .smi/.smiles, .csv, .parquet, "
            "or precomputed .jsonl"
        )
    for index, row in enumerate(rows, start=1):
        lowered = {str(key).lower(): value for key, value in row.items()}
        smiles_value = lowered.get("smiles") or lowered.get("canonical_smiles")
        if not smiles_value:
            raise ValueError(f"missing SMILES in table row {index}")
        smiles = str(smiles_value)
        molecule = chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid SMILES in table row {index}")
        molecule_id = str(
            lowered.get("molecule_id")
            or lowered.get("id")
            or f"{path.stem}-{index:06d}"
        )
        yield molecule_id, smiles, molecule


def _standardize(
    molecule: Any,
    *,
    reject_metals: bool = True,
    enforce_size: bool = True,
    neutralize: bool = False,
) -> Any:
    chem, _, _, _, standardize = _rdkit()
    cleaned = standardize.Cleanup(chem.Mol(molecule))
    parent = standardize.FragmentParent(cleaned)
    # Charge-bearing pharmacophore features and docking microstates must not be
    # erased by library standardization.  A neutral representation may be
    # requested explicitly for a scaffold-only operation, but is never allowed
    # to overwrite the indexed parent.
    if neutralize:
        parent = standardize.Uncharger().uncharge(parent)
    chem.AssignStereochemistry(parent, cleanIt=True, force=True)
    if enforce_size and parent.GetNumHeavyAtoms() > 100:
        raise ValueError("v1 supports ligands with at most 100 heavy atoms")
    metals = sorted(
        {atom.GetSymbol() for atom in parent.GetAtoms() if atom.GetAtomicNum() in _METALS}
    )
    if metals and reject_metals:
        raise ValueError(f"metal-containing ligand is unsupported in v1: {', '.join(metals)}")
    return parent


def _reference_structure(path: Path, media_type: str | None) -> Any:
    chem, _, _, _, _ = _rdkit()
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".sd"} or media_type == "chemical/x-mdl-sdfile":
        molecules = [
            molecule
            for molecule in chem.SDMolSupplier(str(path), removeHs=False)
            if molecule is not None
        ]
        if len(molecules) != 1:
            raise ValueError("reference ligand SDF must contain exactly one valid molecule")
        return molecules[0]
    if suffix in {".pdb", ".ent"} or media_type == "chemical/x-pdb":
        molecule = chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    elif suffix == ".mol2":
        molecule = chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    elif suffix in {".mol", ".mdl"}:
        molecule = chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    else:
        raise ValueError("reference ligand structure must be SDF, MOL, MOL2, or PDB")
    if molecule is None:
        raise ValueError("RDKit could not parse the reference ligand structure")
    return molecule


def inspect_reference_ligand(
    *,
    smiles: str | None,
    structure_path: Path | None,
    structure_media_type: str | None = None,
) -> LigandInspection:
    """Compute v1 chemistry gates from molecular input rather than trusting flags."""

    chem, _, _, _, _ = _rdkit()
    molecules: list[Any] = []
    if smiles:
        molecule = chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("invalid reference-ligand SMILES")
        molecules.append(molecule)
    if structure_path is not None:
        molecules.append(_reference_structure(structure_path, structure_media_type))
    if not molecules:
        raise ValueError("ligand inspection requires SMILES or a molecular structure")
    standardized = [
        _standardize(molecule, reject_metals=False, enforce_size=False)
        for molecule in molecules
    ]
    identities = [
        chem.MolToSmiles(molecule, isomericSmiles=True) for molecule in standardized
    ]
    if len(set(identities)) != 1:
        raise ValueError("ligand SMILES and structure standardize to different identities")
    molecule = standardized[0]
    chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    centers = chem.FindMolChiralCenters(
        molecule,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )
    unassigned = sum(label == "?" for _, label in centers)
    metals = tuple(
        sorted(
            {
                atom.GetSymbol()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() in _METALS
            }
        )
    )
    return LigandInspection(
        heavy_atom_count=int(molecule.GetNumHeavyAtoms()),
        metal_elements=metals,
        unassigned_stereocenters=unassigned,
        standardized_isomeric_smiles=identities[0],
        formal_charge=sum(int(atom.GetFormalCharge()) for atom in molecule.GetAtoms()),
    )


def canonical_microstate_parent_identity(smiles: str) -> str:
    """Collapse charge/tautomer variants while preserving heavy-atom connectivity/stereo."""

    chem, _, _, _, standardize = _rdkit()
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid microstate SMILES")
    cleaned = standardize.Cleanup(molecule)
    parent = standardize.FragmentParent(cleaned)
    parent = standardize.ChargeParent(parent)
    parent = standardize.TautomerParent(parent)
    chem.AssignStereochemistry(parent, cleanIt=True, force=True)
    return str(chem.MolToSmiles(parent, isomericSmiles=True))


def bemis_murcko_scaffold_smiles(smiles: str) -> str:
    """Return a Murcko scaffold or a deterministic acyclic element-graph key."""

    chem, _, _, _, _ = _rdkit()
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - part of a normal RDKit install
        raise ChemistryCapabilityError(
            "RDKit MurckoScaffold is required for diversity verification"
        ) from exc
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid parent SMILES for scaffold calculation")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    value = str(chem.MolToSmiles(scaffold, isomericSmiles=True))
    if value:
        return value
    try:
        from rdkit.Chem import rdMolHash
    except ImportError as exc:  # pragma: no cover - part of a normal RDKit install
        raise ChemistryCapabilityError(
            "RDKit rdMolHash is required for acyclic diversity keys"
        ) from exc
    graph = str(
        rdMolHash.MolHash(molecule, rdMolHash.HashFunction.ElementGraph)
    )
    if not graph:
        raise ValueError("RDKit emitted an empty acyclic diversity graph")
    return f"ACYCLIC:{graph}"


def heavy_element_counts(smiles: str) -> dict[str, int]:
    """Return the exact non-hydrogen element multiset encoded by a SMILES string."""

    chem, _, _, _, _ = _rdkit()
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid ligand SMILES")
    return dict(
        sorted(
            Counter(
                atom.GetSymbol().upper()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() > 1
            ).items()
        )
    )


def smiles_formal_charge(smiles: str) -> int:
    """Return the molecular formal charge encoded by one canonical microstate."""

    chem, _, _, _, _ = _rdkit()
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid ligand SMILES")
    return sum(int(atom.GetFormalCharge()) for atom in molecule.GetAtoms())


def enumerate_microstates(
    smiles: str,
    *,
    ph_min: float = 6.4,
    ph_max: float = 8.4,
    max_states: int = 4,
) -> tuple[MicrostateRecord, ...]:
    """Enumerate a bounded, deterministic prospective docking state set."""

    if not ph_min <= ph_max:
        raise ValueError("microstate pH range must be ordered")
    if not 1 <= max_states <= 4:
        raise ValueError("microstate max_states must be in [1, 4]")
    chem, _, _, _, standardize = _rdkit()
    parent = chem.MolFromSmiles(smiles)
    if parent is None:
        raise ValueError("invalid parent SMILES for microstate enumeration")
    chem.AssignStereochemistry(parent, cleanIt=True, force=True)
    centers = chem.FindMolChiralCenters(
        parent,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )
    if any(label == "?" for _, label in centers):
        raise ValueError("unassigned tetrahedral stereochemistry cannot enter docking")
    try:
        potential = chem.FindPotentialStereo(parent)
    except AttributeError:  # pragma: no cover - supported RDKit versions provide it
        potential = ()
    if any(
        "Bond_Double" in str(getattr(item, "type", ""))
        and "Unspecified" in str(getattr(item, "specified", ""))
        for item in potential
    ):
        raise ValueError("unassigned double-bond stereochemistry cannot enter docking")
    parent_smiles = str(chem.MolToSmiles(parent, isomericSmiles=True))
    parent_identity = canonical_microstate_parent_identity(parent_smiles)
    try:
        from dimorphite_dl import protonate_smiles
    except ImportError as exc:
        raise ChemistryCapabilityError(
            "Dimorphite-DL 2.0.2 is required for prospective docking microstates"
        ) from exc
    protonated = protonate_smiles(
        parent_smiles,
        ph_min=float(ph_min),
        ph_max=float(ph_max),
        precision=1.0,
        max_variants=16,
    )
    if not isinstance(protonated, list) or not protonated:
        raise ValueError("Dimorphite-DL produced no protonation state")
    tautomer_enumerator = standardize.TautomerEnumerator()
    canonical: set[str] = set()
    for protonated_smiles in protonated:
        molecule = chem.MolFromSmiles(str(protonated_smiles))
        if molecule is None:
            continue
        for tautomer in tautomer_enumerator.Enumerate(molecule):
            value = str(chem.MolToSmiles(tautomer, isomericSmiles=True))
            if canonical_microstate_parent_identity(value) == parent_identity:
                canonical.add(value)
    if not canonical:
        raise ValueError("microstate enumeration did not preserve the molecular parent")
    selected = sorted(canonical)[:max_states]
    records: list[MicrostateRecord] = []
    for index, value in enumerate(selected, start=1):
        molecule = chem.MolFromSmiles(value)
        if molecule is None:  # pragma: no cover - value was generated by RDKit
            raise AssertionError("RDKit failed to parse its own canonical SMILES")
        records.append(
            MicrostateRecord(
                microstate_id=f"state-{index:02d}",
                canonical_isomeric_smiles=value,
                formal_charge=sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
                parent_standardized_smiles=parent_smiles,
            )
        )
    return tuple(records)


def _features_for_conformer(molecule: Any, conformer_id: int) -> tuple[FeaturePoint, ...]:
    features: list[FeaturePoint] = []
    seen: set[tuple[FeatureType, tuple[int, ...]]] = set()
    for feature in _feature_factory().GetFeaturesForMol(molecule, confId=conformer_id):
        feature_type = _FAMILY_MAP.get(feature.GetFamily())
        if feature_type is None:
            continue
        atom_indices = tuple(sorted(int(index) for index in feature.GetAtomIds()))
        key = (feature_type, atom_indices)
        if key in seen:
            continue
        seen.add(key)
        position = feature.GetPos()
        features.append(
            FeaturePoint(
                feature_type=feature_type,
                position=(float(position.x), float(position.y), float(position.z)),
                atom_indices=atom_indices,
            )
        )
    features.sort(key=lambda item: (item.feature_type.value, item.atom_indices, item.position))
    if len(features) < 3:
        raise ValueError("molecule yields fewer than three supported pharmacophore features")
    return tuple(features)


def molecule_to_index_record(
    molecule_id: str,
    molecule: Any,
    *,
    original_smiles: str | None,
    source: str,
    seed: int,
    max_conformers: int = 4,
) -> IndexedMolecule:
    chem, _, all_chem, _, _ = _rdkit()
    parent = _standardize(molecule)
    original = original_smiles or chem.MolToSmiles(molecule, isomericSmiles=True)
    standardized = chem.MolToSmiles(parent, isomericSmiles=True)
    with_hydrogens = chem.AddHs(parent)
    parameters = all_chem.ETKDGv3()
    parameters.randomSeed = int(seed % (2**31 - 1))
    parameters.numThreads = 1
    parameters.useRandomCoords = False
    conformer_ids = tuple(
        int(value)
        for value in all_chem.EmbedMultipleConfs(
            with_hydrogens, numConfs=max_conformers, params=parameters
        )
    )
    if not conformer_ids:
        raise ValueError(f"ETKDGv3 failed for molecule {molecule_id}")
    conformers = tuple(
        FeatureConformer(
            conformer_id=rank,
            features=_features_for_conformer(with_hydrogens, conformer_id),
        )
        for rank, conformer_id in enumerate(conformer_ids)
    )
    return IndexedMolecule(
        molecule_id=molecule_id,
        original_smiles=original,
        standardized_smiles=standardized,
        conformers=conformers,
        source=source,
    )


def load_chemical_library(
    path: Path,
    *,
    seed: int = 20260721,
    max_conformers: int = 4,
) -> Iterator[IndexedMolecule]:
    """Yield standardized parents without overwriting the original identifiers/SMILES."""

    for molecule_id, smiles, molecule in _raw_molecules(path):
        yield molecule_to_index_record(
            molecule_id,
            molecule,
            original_smiles=smiles,
            source=f"local-import:{path.name}",
            seed=seed,
            max_conformers=max_conformers,
        )


def smiles_pharmacophore(
    smiles: str, *, seed: int = 20260721, max_points: int = 12
) -> tuple[FeaturePoint, ...]:
    chem, _, _, _, _ = _rdkit()
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid reference-ligand SMILES")
    record = molecule_to_index_record(
        "query",
        molecule,
        original_smiles=smiles,
        source="case-input",
        seed=seed,
        max_conformers=1,
    )
    features = record.conformers[0].features
    # Charge and directional features carry more information than generic
    # hydrophobes.  The tie-break remains deterministic.
    priority = {
        FeatureType.POSITIVE: 0,
        FeatureType.NEGATIVE: 0,
        FeatureType.DONOR: 1,
        FeatureType.ACCEPTOR: 1,
        FeatureType.AROMATIC: 2,
        FeatureType.HYDROPHOBE: 3,
    }
    return tuple(
        sorted(features, key=lambda item: (priority[item.feature_type], item.atom_indices))[
            :max_points
        ]
    )
