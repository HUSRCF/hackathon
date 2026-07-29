"""Artifact-bound docking scenes and deterministic geometry summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from . import __version__
from .artifacts import ArtifactStore
from .manifest import RunManifest, RunState
from .models import ArtifactRef, ValidationBundle
from .validation import classify_evidence

POSE_SCENE_SCHEMA_VERSION = "1.0"
_VIEWER_FORMATS = {
    "chemical/x-pdb": "pdb",
    "chemical/x-mmcif": "cif",
    "chemical/x-mdl-sdfile": "sdf",
    "chemical/x-pdbqt": "pdbqt",
}


@dataclass(frozen=True, slots=True)
class _HeavyAtom:
    element: str
    position: tuple[float, float, float]
    residue: str | None = None


def viewer_format(reference: ArtifactRef) -> str:
    try:
        return _VIEWER_FORMATS[reference.media_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported local viewer media type: {reference.media_type}"
        ) from exc


def _reference(value: Any, label: str) -> ArtifactRef:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an ArtifactRef")
    return ArtifactRef.from_dict(value)


def _validation_bundle(value: Any) -> ValidationBundle:
    if not isinstance(value, dict):
        raise ValueError("validation bundle must be an object")
    payload = dict(value)
    payload["unsupported_reasons"] = tuple(payload.get("unsupported_reasons", ()))
    payload["evidence"] = tuple(
        ArtifactRef.from_dict(item) for item in payload.get("evidence", ())
    )
    return ValidationBundle(**payload)


def _receptor_atoms(data: bytes, format: str) -> list[_HeavyAtom]:
    try:
        import gemmi
    except ImportError as exc:
        raise RuntimeError("Gemmi is required for pose geometry summaries") from exc
    text = data.decode("utf-8")
    if format == "cif":
        structure = gemmi.make_structure_from_block(
            gemmi.cif.read_string(text).sole_block()
        )
    elif format == "pdb":
        structure = gemmi.read_pdb_string(text)
    else:
        raise ValueError("receptor geometry requires PDB or mmCIF")
    result: list[_HeavyAtom] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_name = (
                    f"{chain.name}:{residue.name}{residue.seqid.num}"
                    f"{residue.seqid.icode.strip()}"
                )
                for atom in residue:
                    element = atom.element.name.upper()
                    if element in {"H", "D", "X"}:
                        continue
                    position = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                    if not all(math.isfinite(item) for item in position):
                        raise ValueError("receptor contains non-finite coordinates")
                    result.append(_HeavyAtom(element, position, residue_name))
    if not result:
        raise ValueError("receptor has no heavy atoms")
    return result


def _pdbqt_ligand_atoms(data: bytes) -> list[_HeavyAtom]:
    result: list[_HeavyAtom] = []
    for raw_line in data.decode("utf-8").splitlines():
        if not raw_line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            position = (
                float(raw_line[30:38]),
                float(raw_line[38:46]),
                float(raw_line[46:54]),
            )
        except ValueError as exc:
            raise ValueError("PDBQT ligand contains invalid coordinates") from exc
        atom_type = raw_line[77:].strip().split(maxsplit=1)[0]
        element = "".join(character for character in atom_type if character.isalpha())
        element = (
            element[:2] if element[:2].title() in {"Cl", "Br"} else element[:1]
        ).upper()
        if element not in {"H", "D"}:
            result.append(_HeavyAtom(element or "X", position))
    if not result:
        raise ValueError("PDBQT ligand has no heavy atoms")
    return result


def _ligand_atoms(data: bytes, format: str) -> list[_HeavyAtom]:
    if format == "pdbqt":
        return _pdbqt_ligand_atoms(data)
    if format != "sdf":
        raise ValueError("ligand geometry requires canonical SDF or PDBQT")
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for pose geometry summaries") from exc
    supplier = Chem.ForwardSDMolSupplier(
        BytesIO(data),
        sanitize=False,
        removeHs=False,
        strictParsing=True,
    )
    molecule = next((item for item in supplier if item is not None), None)
    if molecule is None or molecule.GetNumConformers() != 1:
        raise ValueError("SDF pose must contain one readable coordinate conformer")
    conformer = molecule.GetConformer()
    result: list[_HeavyAtom] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        position = (float(point.x), float(point.y), float(point.z))
        if not all(math.isfinite(item) for item in position):
            raise ValueError("ligand contains non-finite coordinates")
        result.append(_HeavyAtom(atom.GetSymbol().upper(), position))
    if not result:
        raise ValueError("ligand has no heavy atoms")
    return result


def _geometry_summary(
    receptor: ArtifactRef,
    pose: ArtifactRef,
    store: ArtifactStore,
    *,
    box_center: list[float],
    box_size: list[float],
) -> dict[str, Any]:
    try:
        receptor_atoms = _receptor_atoms(
            store.read_bytes(receptor), viewer_format(receptor)
        )
        ligand_atoms = _ligand_atoms(store.read_bytes(pose), viewer_format(pose))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "heuristic": True,
            "scientific_gate": False,
        }

    minimum = math.inf
    sub_2a_pairs = 0
    pocket_residues: set[str] = set()
    for ligand_atom in ligand_atoms:
        for receptor_atom in receptor_atoms:
            distance = math.dist(ligand_atom.position, receptor_atom.position)
            minimum = min(minimum, distance)
            if distance < 2.0:
                sub_2a_pairs += 1
            if distance <= 5.0 and receptor_atom.residue is not None:
                pocket_residues.add(receptor_atom.residue)
    inside_box = all(
        abs(atom.position[axis] - float(box_center[axis]))
        <= float(box_size[axis]) / 2.0 + 1e-6
        for atom in ligand_atoms
        for axis in range(3)
    )
    return {
        "available": True,
        "heuristic": True,
        "scientific_gate": False,
        "receptor_heavy_atom_count": len(receptor_atoms),
        "ligand_heavy_atom_count": len(ligand_atoms),
        "minimum_heavy_atom_distance_angstrom": round(minimum, 4),
        "sub_2_angstrom_pair_count": sub_2a_pairs,
        "pocket_residue_count_within_5_angstrom": len(pocket_residues),
        "pocket_residues_within_5_angstrom": sorted(pocket_residues),
        "all_ligand_heavy_atoms_inside_declared_box": inside_box,
        "interpretation": (
            "Visual-QA geometry only. Sub-2A pairs are counts at a declared display "
            "threshold, not a PoseBusters result or an inferred binding claim."
        ),
    }


def _validation_map(
    manifest: RunManifest,
    store: ArtifactStore,
) -> dict[str, dict[str, Any]]:
    record = manifest.stage_records.get(RunState.VALIDATED.value)
    if record is None:
        return {}
    value = store.read_json(record.outputs[0])
    candidates = value.get("candidates") if isinstance(value, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("validation artifact has no candidates array")
    result: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("validation candidate must be an object")
        candidate_id = str(item["candidate_id"])
        bundle = _validation_bundle(item["bundle"])
        result[candidate_id] = {
            "evidence_grade": classify_evidence(
                bundle,
                has_reference_pose=bool(item.get("has_reference_pose", False)),
            ).value,
            "posebusters_valid": bundle.posebusters_valid,
            "symmetry_rmsd_angstrom": bundle.symmetry_rmsd_angstrom,
            "ifp_similarity": bundle.ifp_similarity,
            "ifp_reference_recovery": bundle.ifp_reference_recovery,
            "ifp_predicted_precision": bundle.ifp_predicted_precision,
            "openmm_parameterized": bundle.openmm_parameterized,
            "openmm_stable": bundle.openmm_stable,
            "unsupported_reasons": list(bundle.unsupported_reasons),
            "evidence_artifact_ids": [item.artifact_id for item in bundle.evidence],
            "decision_reason": str(item.get("decision_reason", "")),
        }
    return result


def build_pose_scene_summary(
    manifest: RunManifest,
    store: ArtifactStore,
    *,
    include_geometry: bool = True,
) -> dict[str, Any]:
    """Return coordinate-free scene metadata and content-addressed QA receipts."""

    record = manifest.stage_records.get(RunState.DOCKED.value)
    if record is None:
        return {
            "schema_version": POSE_SCENE_SCHEMA_VERSION,
            "kind": "protbind.pose-scene-summary",
            "available": False,
            "reason": "DOCKED has not completed.",
            "candidate_count": 0,
            "geometry_summary_count": 0,
            "candidates": [],
        }
    bundle = store.read_json(record.outputs[0])
    if not isinstance(bundle, dict) or not isinstance(bundle.get("candidates"), list):
        raise ValueError("DOCKED output has no candidate array")
    receptor_value = bundle.get("receptor")
    default_receptor = (
        _reference(receptor_value, "DOCKED receptor")
        if isinstance(receptor_value, dict)
        else manifest.artifacts.get("receptor_ready_structure")
        or manifest.artifacts.get("support_receptor_structure")
    )
    if default_receptor is None:
        raise ValueError("DOCKED output has no viewable receptor reference")
    store.resolve(default_receptor)
    receptor_format = viewer_format(default_receptor)
    validations = _validation_map(manifest, store)
    candidates: list[dict[str, Any]] = []
    for raw_candidate in bundle["candidates"]:
        if not isinstance(raw_candidate, dict):
            raise ValueError("DOCKED candidate must be an object")
        candidate_id = str(raw_candidate["candidate_id"])
        pose = _reference(raw_candidate["pose"], "DOCKED pose")
        store.resolve(pose)
        pose_format = viewer_format(pose)
        center = raw_candidate.get("box_center")
        size = raw_candidate.get("box_size")
        if (
            not isinstance(center, list)
            or not isinstance(size, list)
            or len(center) != 3
            or len(size) != 3
        ):
            raise ValueError("DOCKED candidate has no finite three-dimensional box")
        geometry = (
            _geometry_summary(
                default_receptor,
                pose,
                store,
                box_center=center,
                box_size=size,
            )
            if include_geometry
            else {"available": False, "reason": "geometry summary not requested"}
        )
        scene = {
            "schema_version": POSE_SCENE_SCHEMA_VERSION,
            "kind": "protbind.pose-scene",
            "run_id": manifest.run_id,
            "candidate_id": candidate_id,
            "molecule_id": str(raw_candidate["molecule_id"]),
            "microstate_id": raw_candidate.get("microstate_id"),
            "engine": str(raw_candidate.get("engine", "unknown")),
            "receptor": default_receptor.to_dict(),
            "receptor_format": receptor_format,
            "pose": pose.to_dict(),
            "pose_format": pose_format,
            "box_center": [float(item) for item in center],
            "box_size": [float(item) for item in size],
            "vina_score": raw_candidate.get("vina_score"),
            "vina_score_semantics": raw_candidate.get("vina_score_semantics"),
            "validation": validations.get(
                candidate_id,
                {
                    "evidence_grade": None,
                    "posebusters_valid": None,
                    "reason": "VALIDATED evidence is not available for this pose.",
                },
            ),
            "geometry": geometry,
            "visualization_semantics": (
                "Local visual QA only; this scene is not evidence of experimental binding."
            ),
            "coordinates_disclosed_to_agent": False,
        }
        scene_artifact = store.put_json(
            scene,
            producer="protbind.pose-scene",
            producer_version=__version__,
        )
        scene["scene_artifact_id"] = scene_artifact.artifact_id
        scene["local_view_path"] = (
            f"/runs/{manifest.run_id}/poses/{candidate_id}"
        )
        candidates.append(scene)
    return {
        "schema_version": POSE_SCENE_SCHEMA_VERSION,
        "kind": "protbind.pose-scene-summary",
        "available": bool(candidates),
        "reason": None if candidates else "DOCKED completed without a successful pose.",
        "candidate_count": len(candidates),
        "geometry_summary_count": sum(
            bool(item["geometry"].get("available")) for item in candidates
        ),
        "candidates": candidates,
        "coordinates_disclosed_to_agent": False,
        "interpretation": (
            "Agent-visible values come from docked/validation artifacts and deterministic "
            "Gemmi/RDKit geometry. Coordinate bytes remain local to the loopback viewer."
        ),
    }
