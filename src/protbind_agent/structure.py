"""Gemmi-based structural limits and heuristic pocket pharmacophore generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tripharm import FeaturePoint, FeatureType

_STANDARD_AMINO_ACIDS = frozenset(
    {
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
)
_ONE_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
_METAL_ELEMENTS = frozenset(
    {
        "LI",
        "BE",
        "NA",
        "MG",
        "AL",
        "K",
        "CA",
        "SC",
        "TI",
        "V",
        "CR",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "GA",
        "RB",
        "SR",
        "Y",
        "ZR",
        "NB",
        "MO",
        "TC",
        "RU",
        "RH",
        "PD",
        "AG",
        "CD",
        "IN",
        "SN",
        "CS",
        "BA",
        "PT",
        "AU",
        "HG",
        "PB",
    }
)
_AROMATIC_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
}
_CHARGED_ATOMS = {
    ("ASP", "OD1"): FeatureType.POSITIVE,
    ("ASP", "OD2"): FeatureType.POSITIVE,
    ("GLU", "OE1"): FeatureType.POSITIVE,
    ("GLU", "OE2"): FeatureType.POSITIVE,
    ("LYS", "NZ"): FeatureType.NEGATIVE,
    ("ARG", "NE"): FeatureType.NEGATIVE,
    ("ARG", "NH1"): FeatureType.NEGATIVE,
    ("ARG", "NH2"): FeatureType.NEGATIVE,
}
_DONOR_ATOMS = {
    ("SER", "OG"),
    ("THR", "OG1"),
    ("TYR", "OH"),
    ("CYS", "SG"),
    ("ASN", "ND2"),
    ("GLN", "NE2"),
    ("TRP", "NE1"),
    ("HIS", "ND1"),
    ("HIS", "NE2"),
}
_ACCEPTOR_ATOMS = {
    ("SER", "OG"),
    ("THR", "OG1"),
    ("TYR", "OH"),
    ("CYS", "SG"),
    ("ASN", "OD1"),
    ("GLN", "OE1"),
    ("HIS", "ND1"),
    ("HIS", "NE2"),
}
_HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PRO"})
_POCKET_FEATURE_TYPE_ORDER = (
    FeatureType.DONOR,
    FeatureType.ACCEPTOR,
    FeatureType.AROMATIC,
    FeatureType.HYDROPHOBE,
    FeatureType.POSITIVE,
    FeatureType.NEGATIVE,
)


class StructureCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StructureInspection:
    chain_count: int
    residue_count: int
    chain_ids: tuple[str, ...]
    sequences: tuple[str, ...]
    metal_elements: tuple[str, ...]
    nonstandard_residues: tuple[str, ...]
    alternate_location_atoms: int
    missing_backbone_residues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConnectionInspection:
    """Declared protein/ligand connection evidence, not inferred bond topology."""

    coordinate_format: str
    scope: str
    status: str
    declared_covalent_connections: int
    protein_ligand_conect_edges: int
    notes: tuple[str, ...] = ()

    @property
    def covalent_detected(self) -> bool:
        return (
            self.declared_covalent_connections > 0
            or self.protein_ligand_conect_edges > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_format": self.coordinate_format,
            "scope": self.scope,
            "status": self.status,
            "declared_covalent_connections": self.declared_covalent_connections,
            "protein_ligand_conect_edges": self.protein_ligand_conect_edges,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ComplexInspection:
    protein_chain_ids: tuple[str, ...]
    protein_sequences: tuple[str, ...]
    ligand_chain_id: str
    ligand_residue_count: int
    ligand_heavy_element_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class BoxAtomOverlapInspection:
    """Deterministic receptor-atom counts for one axis-aligned Cartesian box.

    Atom overlap is only a coordinate-frame plausibility check.  It does not
    establish that the box is a biological binding site.
    """

    coordinate_format: str
    model_index: int
    receptor_heavy_atom_count: int
    protein_heavy_atom_count: int
    receptor_heavy_atom_count_inside_box: int
    protein_heavy_atom_count_inside_box: int
    nearest_receptor_heavy_atom_distance_to_center_angstrom: float | None
    nearest_protein_heavy_atom_distance_to_center_angstrom: float | None

    @property
    def receptor_atom_overlap(self) -> bool:
        return self.receptor_heavy_atom_count_inside_box > 0

    @property
    def protein_atom_overlap(self) -> bool:
        return self.protein_heavy_atom_count_inside_box > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_format": self.coordinate_format,
            "model_index": self.model_index,
            "receptor_heavy_atom_count": self.receptor_heavy_atom_count,
            "protein_heavy_atom_count": self.protein_heavy_atom_count,
            "receptor_heavy_atom_count_inside_box": (
                self.receptor_heavy_atom_count_inside_box
            ),
            "protein_heavy_atom_count_inside_box": (
                self.protein_heavy_atom_count_inside_box
            ),
            "nearest_receptor_heavy_atom_distance_to_center_angstrom": (
                self.nearest_receptor_heavy_atom_distance_to_center_angstrom
            ),
            "nearest_protein_heavy_atom_distance_to_center_angstrom": (
                self.nearest_protein_heavy_atom_distance_to_center_angstrom
            ),
            "receptor_atom_overlap": self.receptor_atom_overlap,
            "protein_atom_overlap": self.protein_atom_overlap,
            "interpretation": "coordinate-frame-plausibility-only",
            "biological_site_validity_inferred": False,
        }


def _gemmi() -> Any:
    try:
        import gemmi
    except ImportError as exc:
        raise StructureCapabilityError("Gemmi is required for PDB/mmCIF inspection") from exc
    return gemmi


def _structure(path: Path) -> Any:
    gemmi = _gemmi()
    try:
        text = path.read_text(encoding="utf-8")
        if text.lstrip().lower().startswith("data_"):
            document = gemmi.cif.read_string(text)
            structure = gemmi.make_structure_from_block(document.sole_block())
        else:
            structure = gemmi.read_pdb_string(text)
    except Exception as exc:
        raise ValueError(f"Gemmi could not parse structure {path.name}: {exc}") from exc
    if len(structure) == 0:
        raise ValueError("structure contains no models")
    return structure


def inspect_box_atom_overlap(
    path: Path,
    *,
    center: tuple[float, float, float] | list[float],
    size: tuple[float, float, float] | list[float],
) -> BoxAtomOverlapInspection:
    """Count first-model heavy atoms inside a box using exact receptor coordinates."""

    if (
        len(center) != 3
        or len(size) != 3
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in (*center, *size)
        )
        or any(float(value) <= 0 for value in size)
    ):
        raise ValueError("box center/size must contain finite 3D coordinates and positive size")
    normalized_center = tuple(float(value) for value in center)
    half_size = tuple(float(value) / 2.0 for value in size)
    text = path.read_text(encoding="utf-8")
    coordinate_format = "mmcif" if text.lstrip().lower().startswith("data_") else "pdb"
    structure = _structure(path)
    model = structure[0]
    receptor_count = 0
    protein_count = 0
    receptor_inside = 0
    protein_inside = 0
    nearest_receptor: float | None = None
    nearest_protein: float | None = None
    for chain in model:
        for residue in chain:
            is_protein = residue.name.strip().upper() in _STANDARD_AMINO_ACIDS
            for atom in residue:
                if atom.element.atomic_number <= 1:
                    continue
                point = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                if any(not math.isfinite(value) for value in point):
                    raise ValueError("receptor contains non-finite atom coordinates")
                distance = math.dist(point, normalized_center)
                receptor_count += 1
                nearest_receptor = (
                    distance
                    if nearest_receptor is None
                    else min(nearest_receptor, distance)
                )
                inside = all(
                    abs(point[axis] - normalized_center[axis]) <= half_size[axis]
                    for axis in range(3)
                )
                if inside:
                    receptor_inside += 1
                if is_protein:
                    protein_count += 1
                    nearest_protein = (
                        distance
                        if nearest_protein is None
                        else min(nearest_protein, distance)
                    )
                    if inside:
                        protein_inside += 1
    return BoxAtomOverlapInspection(
        coordinate_format=coordinate_format,
        model_index=0,
        receptor_heavy_atom_count=receptor_count,
        protein_heavy_atom_count=protein_count,
        receptor_heavy_atom_count_inside_box=receptor_inside,
        protein_heavy_atom_count_inside_box=protein_inside,
        nearest_receptor_heavy_atom_distance_to_center_angstrom=(
            round(nearest_receptor, 6) if nearest_receptor is not None else None
        ),
        nearest_protein_heavy_atom_distance_to_center_angstrom=(
            round(nearest_protein, 6) if nearest_protein is not None else None
        ),
    )


def inspect_structure(
    path: Path,
    *,
    max_chains: int | None = 2,
    max_residues: int | None = 700,
) -> StructureInspection:
    structure = _structure(path)
    protein_chains: list[str] = []
    protein_sequences: list[str] = []
    residue_count = 0
    metals: set[str] = set()
    nonstandard: set[str] = set()
    missing_backbone: list[str] = []
    altloc = 0
    for chain in structure[0]:
        sequence: list[str] = []
        for residue in chain:
            name = residue.name.upper()
            if name in _STANDARD_AMINO_ACIDS:
                residue_count += 1
                sequence.append(_ONE_LETTER[name])
                atom_names = {atom.name.strip().upper() for atom in residue}
                if not {"N", "CA", "C"}.issubset(atom_names):
                    missing_backbone.append(_residue_id(chain.name, residue))
            elif residue.het_flag == "A":
                nonstandard.add(name)
            for atom in residue:
                if any(not math.isfinite(value) for value in _position(atom)):
                    raise ValueError("structure contains non-finite coordinates")
                element = atom.element.name.upper()
                if element in _METAL_ELEMENTS:
                    metals.add(element)
                if str(atom.altloc).strip("\x00 "):
                    altloc += 1
        if sequence:
            protein_chains.append(chain.name)
            protein_sequences.append("".join(sequence))
    if not protein_chains:
        raise ValueError("structure contains no standard protein residues")
    if max_chains is not None and len(protein_chains) > max_chains:
        raise ValueError(f"structure supports at most {max_chains} protein chains")
    if max_residues is not None and residue_count > max_residues:
        raise ValueError(f"structure supports at most {max_residues} protein residues")
    return StructureInspection(
        chain_count=len(protein_chains),
        residue_count=residue_count,
        chain_ids=tuple(protein_chains),
        sequences=tuple(protein_sequences),
        metal_elements=tuple(sorted(metals)),
        nonstandard_residues=tuple(sorted(nonstandard)),
        alternate_location_atoms=altloc,
        missing_backbone_residues=tuple(missing_backbone),
    )


def inspect_declared_connections(path: Path) -> ConnectionInspection:
    """Inspect explicit covalent records without claiming complete bond knowledge."""

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read structure connection records: {path.name}") from exc
    if data.lstrip().lower().startswith(b"data_"):
        return _inspect_mmcif_connections(data)
    return _inspect_pdb_connections(path, data)


def _inspect_mmcif_connections(data: bytes) -> ConnectionInspection:
    gemmi = _gemmi()
    try:
        block = gemmi.cif.read_string(data.decode("utf-8")).sole_block()
        connection_types = tuple(
            gemmi.cif.as_string(str(value)).strip().lower()
            for value in block.find_values("_struct_conn.conn_type_id")
        )
    except (RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("mmCIF covalent-connection metadata is unreadable") from exc
    covalent_count = sum(value.startswith("covale") for value in connection_types)
    return ConnectionInspection(
        coordinate_format="mmcif",
        scope="declared _struct_conn records across all entities",
        status=(
            "covalent_detected"
            if covalent_count
            else "no_declared_covalent_connection"
        ),
        declared_covalent_connections=covalent_count,
        protein_ligand_conect_edges=0,
        notes=("disulfide records are not classified as covalent ligands",),
    )


def _inspect_pdb_connections(path: Path, data: bytes) -> ConnectionInspection:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("PDB connection records are not UTF-8 text") from exc
    structure = _structure(path)
    declared_covalent = sum(
        "covale" in str(connection.type).lower()
        for connection in structure.connections
    )
    atom_classes: dict[int, str] = {}
    conect_lines: list[str] = []
    for line in text.splitlines():
        record = line[:6]
        if record in {"ATOM  ", "HETATM"}:
            try:
                serial = int(line[6:11])
            except (ValueError, IndexError):
                continue
            residue_name = line[17:20].strip().upper()
            if record == "ATOM  " and residue_name in _STANDARD_AMINO_ACIDS:
                atom_classes[serial] = "protein"
            elif record == "HETATM" and residue_name not in {
                "HOH",
                "WAT",
                "DOD",
                *_STANDARD_AMINO_ACIDS,
            }:
                atom_classes[serial] = "ligand"
            else:
                atom_classes[serial] = "other"
        elif record == "CONECT":
            conect_lines.append(line)
    edges: set[tuple[int, int]] = set()
    unresolved = False
    for line in conect_lines:
        serials: list[int] = []
        for offset in range(6, len(line), 5):
            field = line[offset : offset + 5].strip()
            if not field:
                continue
            try:
                serials.append(int(field))
            except ValueError:
                unresolved = True
        if len(serials) < 2:
            unresolved = True
            continue
        source = serials[0]
        for destination in serials[1:]:
            if source not in atom_classes or destination not in atom_classes:
                unresolved = True
                continue
            if {atom_classes[source], atom_classes[destination]} == {
                "protein",
                "ligand",
            }:
                edges.add(tuple(sorted((source, destination))))
    detected = declared_covalent > 0 or bool(edges)
    status = "covalent_detected" if detected else (
        "unknown" if unresolved else "partial_no_declared_crosslink"
    )
    notes = [
        "PDB LINK/SSBOND plus protein-ligand CONECT were inspected; absence is partial evidence"
    ]
    if unresolved:
        notes.append("one or more CONECT serial references could not be classified")
    return ConnectionInspection(
        coordinate_format="pdb",
        scope="declared LINK plus standard-protein/nonstandard-ligand CONECT edges",
        status=status,
        declared_covalent_connections=declared_covalent,
        protein_ligand_conect_edges=len(edges),
        notes=tuple(notes),
    )


def select_protein_chains(path: Path, chain_ids: tuple[str, ...]) -> bytes:
    """Return a deterministic first-model mmCIF containing only selected proteins."""

    if not chain_ids or len(chain_ids) > 2 or len(set(chain_ids)) != len(chain_ids):
        raise ValueError("chain selection requires one or two unique chain IDs")
    structure = _structure(path).clone()
    while len(structure) > 1:
        del structure[1]
    model = structure[0]
    available = {chain.name for chain in model}
    missing = tuple(chain for chain in chain_ids if chain not in available)
    if missing:
        raise ValueError("selected protein chains are absent: " + ", ".join(missing))
    selected = {chain.name: chain.clone() for chain in model if chain.name in chain_ids}
    for chain in selected.values():
        for index in reversed(range(len(chain))):
            if chain[index].name.upper() not in _STANDARD_AMINO_ACIDS:
                del chain[index]
    for chain in tuple(chain.name for chain in model):
        model.remove_chain(chain)
    for chain_id in chain_ids:
        model.add_chain(selected[chain_id], unique_name=False)
    structure.remove_empty_chains()
    if len(model) != len(chain_ids):
        raise ValueError("chain selection did not produce the requested protein chains")
    structure.name = structure.name or "protbind_selected_receptor"
    document = structure.make_mmcif_document()
    return document.as_string().encode("utf-8")


def _residue_id(chain_name: str, residue: Any) -> str:
    insertion = str(residue.seqid.icode).strip("\x00 ")
    return f"{chain_name}:{residue.seqid.num}{insertion}"


def _position(atom: Any) -> tuple[float, float, float]:
    return (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))


def inspect_predicted_complex(
    path: Path,
    *,
    expected_sequences: tuple[str, ...],
    ligand_chain_id: str = "Z",
) -> ComplexInspection:
    """Validate finite protein/ligand coordinates and deterministic chain identities."""

    structure = _structure(path)
    protein: dict[str, str] = {}
    ligand_counts: dict[str, int] = {}
    ligand_residues = 0
    unexpected_molecular_chains: set[str] = set()
    for chain in structure[0]:
        sequence: list[str] = []
        for residue in chain:
            name = residue.name.upper()
            if name in _STANDARD_AMINO_ACIDS:
                sequence.append(_ONE_LETTER[name])
                atom_names = {atom.name.strip().upper() for atom in residue}
                if not {"N", "CA", "C"}.issubset(atom_names):
                    raise ValueError(
                        "predicted complex contains a protein residue missing N/CA/C"
                    )
            elif chain.name == ligand_chain_id and name not in {"HOH", "WAT", "DOD"}:
                ligand_residues += 1
                for atom in residue:
                    coordinates = _position(atom)
                    if any(not math.isfinite(value) for value in coordinates):
                        raise ValueError("predicted complex contains non-finite coordinates")
                    element = atom.element.name.upper()
                    if element not in {"H", "D"}:
                        ligand_counts[element] = ligand_counts.get(element, 0) + 1
            elif name not in {"HOH", "WAT", "DOD"} and any(True for _ in residue):
                unexpected_molecular_chains.add(chain.name)
            for atom in residue:
                if any(not math.isfinite(value) for value in _position(atom)):
                    raise ValueError("predicted complex contains non-finite coordinates")
        if sequence:
            protein[chain.name] = "".join(sequence)
    expected_ids = tuple(chr(ord("A") + index) for index in range(len(expected_sequences)))
    if tuple(protein) != expected_ids or tuple(protein.values()) != expected_sequences:
        raise ValueError("predicted complex protein chains/sequences differ from the request")
    if ligand_residues != 1 or not ligand_counts:
        raise ValueError("predicted complex must contain one non-empty ligand residue on chain Z")
    if unexpected_molecular_chains:
        raise ValueError("predicted complex contains unexpected non-water molecular chains")
    return ComplexInspection(
        protein_chain_ids=tuple(protein),
        protein_sequences=tuple(protein.values()),
        ligand_chain_id=ligand_chain_id,
        ligand_residue_count=ligand_residues,
        ligand_heavy_element_counts=dict(sorted(ligand_counts.items())),
    )


def _toward_center(
    position: tuple[float, float, float], center: tuple[float, float, float], distance: float = 2.8
) -> tuple[float, float, float]:
    vector = tuple(center[index] - position[index] for index in range(3))
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-8:
        return position
    step = min(distance, norm * 0.5)
    return tuple(position[index] + step * vector[index] / norm for index in range(3))


def _cluster(points: list[FeaturePoint], threshold: float) -> list[FeaturePoint]:
    clusters: list[list[FeaturePoint]] = []
    for point in sorted(points, key=lambda item: (item.feature_type.value, item.position)):
        destination = None
        for cluster in clusters:
            if cluster[0].feature_type is point.feature_type:
                centroid = tuple(
                    sum(item.position[index] for item in cluster) / len(cluster)
                    for index in range(3)
                )
                if math.dist(point.position, centroid) <= threshold:
                    destination = cluster
                    break
        if destination is None:
            clusters.append([point])
        else:
            destination.append(point)
    return [
        FeaturePoint(
            feature_type=cluster[0].feature_type,
            position=tuple(
                sum(item.position[index] for item in cluster) / len(cluster)
                for index in range(3)
            ),
            atom_indices=tuple(
                sorted({index for item in cluster for index in item.atom_indices})
            ),
        )
        for cluster in clusters
    ]


def _select_diverse_pocket_points(
    points: list[FeaturePoint],
    *,
    center: tuple[float, float, float],
    max_points: int,
) -> tuple[FeaturePoint, ...]:
    """Apply a deterministic per-type cap before high-information triangle selection.

    Pocket atoms can contain many charged side chains.  A global
    charge-first truncation makes the downstream top-64 triangle set chemically
    unreachable for otherwise plausible ligands.  Round-robin selection keeps
    every available feature family represented before taking another point of
    the same family.  The hard per-type cap prevents sparse families from
    donating all of their unused quota back to one dominant family.

    Points remain nearest-pocket-center first within each family.  TriPharm's
    ``select_query_triangles`` subsequently selects the high-information
    triangles from this chemically diverse, bounded point set.
    """

    if max_points < 1:
        raise ValueError("max_points must be positive")
    grouped: dict[FeatureType, list[FeaturePoint]] = {
        feature_type: [] for feature_type in _POCKET_FEATURE_TYPE_ORDER
    }
    for point in points:
        grouped[point.feature_type].append(point)
    active_types = tuple(
        feature_type
        for feature_type in _POCKET_FEATURE_TYPE_ORDER
        if grouped[feature_type]
    )
    if not active_types:
        return ()
    per_type_cap = math.ceil(max_points / len(active_types))
    for feature_type in active_types:
        grouped[feature_type].sort(
            key=lambda item: (
                math.dist(item.position, center),
                item.position,
                item.atom_indices,
            )
        )
    selected: list[FeaturePoint] = []
    for rank in range(per_type_cap):
        for feature_type in active_types:
            candidates = grouped[feature_type]
            if rank < len(candidates):
                selected.append(candidates[rank])
                if len(selected) == max_points:
                    return tuple(selected)
    return tuple(selected)


def pocket_pharmacophore(
    path: Path,
    *,
    residues: tuple[str, ...] = (),
    center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] | None = None,
    cluster_angstrom: float = 1.5,
    max_points: int = 12,
) -> tuple[FeaturePoint, ...]:
    """Generate complementary cavity points; this is explicitly heuristic."""

    if not residues and center is None:
        raise ValueError("pocket generation requires residue IDs or center/box")
    if (center is None) != (box_size is None):
        raise ValueError("center and box_size must be provided together")
    structure = _structure(path)
    requested = set(residues)
    selected: list[tuple[str, Any, Any, int]] = []
    serial = 0
    for chain in structure[0]:
        for residue in chain:
            residue_id = _residue_id(chain.name, residue)
            if residue.name.upper() not in _STANDARD_AMINO_ACIDS:
                continue
            for atom in residue:
                serial += 1
                position = _position(atom)
                in_residues = residue_id in requested
                in_box = False
                if center is not None and box_size is not None:
                    in_box = all(
                        abs(position[index] - center[index]) <= box_size[index] / 2
                        for index in range(3)
                    )
                if in_residues or in_box:
                    selected.append((residue_id, residue, atom, serial))
    if requested - {item[0] for item in selected}:
        missing = ", ".join(sorted(requested - {item[0] for item in selected}))
        raise ValueError(f"requested pocket residues were not found: {missing}")
    if not selected:
        raise ValueError("no protein atoms selected for pocket hypothesis")
    if center is None:
        center = tuple(
            sum(_position(item[2])[index] for item in selected) / len(selected)
            for index in range(3)
        )
    points: list[FeaturePoint] = []
    selected_by_residue: dict[str, list[tuple[Any, Any, int]]] = {}
    for residue_id, residue, atom, atom_serial in selected:
        selected_by_residue.setdefault(residue_id, []).append((residue, atom, atom_serial))
        residue_name = residue.name.upper()
        atom_name = atom.name.strip().upper()
        feature_type = _CHARGED_ATOMS.get((residue_name, atom_name))
        if feature_type is None and (residue_name, atom_name) in _DONOR_ATOMS:
            feature_type = FeatureType.ACCEPTOR
        if feature_type is None and (residue_name, atom_name) in _ACCEPTOR_ATOMS:
            feature_type = FeatureType.DONOR
        if feature_type is None and atom_name == "O":
            feature_type = FeatureType.DONOR
        if feature_type is None and atom_name == "N":
            feature_type = FeatureType.ACCEPTOR
        if (
            feature_type is None
            and residue_name in _HYDROPHOBIC_RESIDUES
            and atom.element.name.upper() == "C"
            and atom_name not in {"C", "CA"}
        ):
            feature_type = FeatureType.HYDROPHOBE
        if feature_type is not None:
            points.append(
                FeaturePoint(
                    feature_type=feature_type,
                    position=_toward_center(_position(atom), center),
                    atom_indices=(atom_serial,),
                )
            )
    # Ring centroids must be derived after the complete residue has been selected;
    # emitting one while streaming atoms would depend on PDB atom order and could
    # use only the first three ring atoms.
    for entries in selected_by_residue.values():
        residue_name = entries[0][0].name.upper()
        names = _AROMATIC_ATOMS.get(residue_name)
        if names is None:
            continue
        atoms = {
            selected_atom.name.strip().upper(): (selected_atom, selected_serial)
            for _, selected_atom, selected_serial in entries
        }
        ring_atoms = [atoms[name] for name in names if name in atoms]
        if len(ring_atoms) < 3:
            continue
        ring_center = tuple(
            sum(_position(item[0])[index] for item in ring_atoms) / len(ring_atoms)
            for index in range(3)
        )
        points.append(
            FeaturePoint(
                feature_type=FeatureType.AROMATIC,
                position=_toward_center(ring_center, center),
                atom_indices=tuple(item[1] for item in ring_atoms),
            )
        )
    result = _select_diverse_pocket_points(
        _cluster(points, cluster_angstrom),
        center=center,
        max_points=max_points,
    )
    if len(result) < 3:
        raise ValueError("pocket heuristic produced fewer than three feature points")
    return result
