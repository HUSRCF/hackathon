"""Dataset adapters and metrics for independent pharmacophore-screen baselines."""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from .chemistry import _rdkit, _standardize, molecule_to_existing_conformer_record
from .screening_protocol import verify_validation_authorization
from .tripharm import (
    FeatureConformer,
    FeaturePoint,
    FeatureType,
    IndexedMolecule,
    query_index,
    read_index_metadata,
    read_query,
    select_query_triangles,
)
from .tripharm_hip import query_index_batch_hip, query_index_hip

_PHARMER_TYPES = {
    "Aromatic": FeatureType.AROMATIC,
    "HydrogenDonor": FeatureType.DONOR,
    "HydrogenAcceptor": FeatureType.ACCEPTOR,
    "Hydrophobic": FeatureType.HYDROPHOBE,
    "PositiveIon": FeatureType.POSITIVE,
    "NegativeIon": FeatureType.NEGATIVE,
}
_PHARMER_RMSD = re.compile(r">\s*<rmsd>\s*\r?\n([^\r\n]+)", re.IGNORECASE)
_TRIPHARM_TO_PHARMER = {
    FeatureType.DONOR: ("HydrogenDonor", 0.5),
    FeatureType.ACCEPTOR: ("HydrogenAcceptor", 0.5),
    FeatureType.AROMATIC: ("Aromatic", 1.1),
    FeatureType.HYDROPHOBE: ("Hydrophobic", 1.0),
    FeatureType.POSITIVE: ("PositiveIon", 0.75),
    FeatureType.NEGATIVE: ("NegativeIon", 0.75),
}


def prepare_labeled_sdf(
    active_sdf: Path,
    inactive_sdf: Path,
    output_sdf: Path,
    labels_output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create one coordinate-preserving SDF with collision-free record IDs."""

    if not overwrite and (output_sdf.exists() or labels_output.exists()):
        raise FileExistsError("prepared SDF or label manifest already exists")
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - exercised by capability checks
        raise RuntimeError("RDKit is required to prepare a screening benchmark") from exc

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    labels_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_sdf = output_sdf.with_suffix(output_sdf.suffix + ".tmp")
    labels: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    writer = Chem.SDWriter(str(temporary_sdf))
    try:
        for label, source in (("active", active_sdf), ("inactive", inactive_sdf)):
            supplier = Chem.SDMolSupplier(str(source), removeHs=False)
            for record_index, molecule in enumerate(supplier, start=1):
                record_id = f"{label}-{record_index:06d}"
                if molecule is None:
                    failures.append(
                        {
                            "record_id": record_id,
                            "label": label,
                            "source_record_index": record_index,
                            "reason": "invalid SDF record",
                        }
                    )
                    continue
                original_name = (
                    molecule.GetProp("_Name").strip()
                    if molecule.HasProp("_Name")
                    else ""
                )
                group_id = original_name or f"record-{record_index:06d}"
                molecule.SetProp("_Name", record_id)
                molecule.SetProp("ProtBindLabel", label)
                molecule.SetProp("ProtBindGroup", group_id)
                writer.write(molecule)
                labels.append(
                    {
                        "record_id": record_id,
                        "group_commitment": sha256_bytes(group_id.encode("utf-8")),
                        "label": label,
                        "source_file": source.name,
                        "source_record_index": record_index,
                    }
                )
    finally:
        writer.close()
    temporary_sdf.replace(output_sdf)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.labeled-screening-library",
        "semantics": (
            "record-level experimental labels; multiple protonation/stereo records may "
            "share one redacted source group"
        ),
        "inputs": {
            "active_sdf_sha256": sha256_file(active_sdf),
            "inactive_sdf_sha256": sha256_file(inactive_sdf),
        },
        "prepared_sdf_sha256": sha256_file(output_sdf),
        "counts": {
            "active": sum(item["label"] == "active" for item in labels),
            "inactive": sum(item["label"] == "inactive" for item in labels),
            "failed": len(failures),
        },
        "records": labels,
        "failures": failures,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    labels_output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_smiles_conformer_sdf(
    active_smiles: Path,
    inactive_smiles: Path,
    output_sdf: Path,
    labels_output: Path,
    *,
    seed: int = 20260721,
    max_conformers: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate one deterministic shared conformer ensemble from labeled SMILES."""

    if not 1 <= max_conformers <= 4:
        raise ValueError("max_conformers must be in [1, 4]")
    if not overwrite and (output_sdf.exists() or labels_output.exists()):
        raise FileExistsError("prepared SDF or label manifest already exists")
    chem, _, all_chem, _, _ = _rdkit()
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    labels_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_sdf = output_sdf.with_suffix(output_sdf.suffix + ".tmp")
    writer = chem.SDWriter(str(temporary_sdf))
    labels: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for label, source in (("active", active_smiles), ("inactive", inactive_smiles)):
            for record_index, raw_line in enumerate(
                source.read_text(encoding="utf-8").splitlines(), start=1
            ):
                fields = raw_line.split()
                if not fields:
                    continue
                smiles = fields[0]
                source_id = fields[1] if len(fields) > 1 else f"line-{record_index}"
                record_id = f"{label}-{record_index:06d}"
                molecule = chem.MolFromSmiles(smiles)
                if molecule is None:
                    failures.append(
                        {
                            "record_id": record_id,
                            "label": label,
                            "source_record_index": record_index,
                            "reason": "invalid SMILES",
                        }
                    )
                    continue
                try:
                    parent = _standardize(molecule)
                    standardized_smiles = chem.MolToSmiles(parent, isomericSmiles=True)
                    embedded = chem.AddHs(parent)
                    parameters = all_chem.ETKDGv3()
                    parameters.randomSeed = int(seed % (2**31 - 1))
                    parameters.numThreads = 1
                    parameters.useRandomCoords = False
                    conformer_ids = tuple(
                        int(value)
                        for value in all_chem.EmbedMultipleConfs(
                            embedded,
                            numConfs=max_conformers,
                            params=parameters,
                        )
                    )
                    if not conformer_ids:
                        raise ValueError("ETKDGv3 produced no conformer")
                    for conformer_rank, conformer_id in enumerate(conformer_ids):
                        embedded.SetProp("_Name", f"{record_id}--c{conformer_rank:02d}")
                        embedded.SetProp("ProtBindParent", record_id)
                        embedded.SetProp("ProtBindLabel", label)
                        embedded.SetProp(
                            "ProtBindStandardizedSmiles", standardized_smiles
                        )
                        writer.write(embedded, confId=conformer_id)
                except ValueError as exc:
                    failures.append(
                        {
                            "record_id": record_id,
                            "label": label,
                            "source_record_index": record_index,
                            "reason": str(exc),
                        }
                    )
                    continue
                labels.append(
                    {
                        "record_id": record_id,
                        "source_id_commitment": sha256_bytes(source_id.encode("utf-8")),
                        "label": label,
                        "source_file": source.name,
                        "source_record_index": record_index,
                        "conformer_count": len(conformer_ids),
                    }
                )
    finally:
        writer.close()
    temporary_sdf.replace(output_sdf)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.labeled-screening-conformer-library",
        "semantics": "record-level experimental labels with shared deterministic ETKDGv3",
        "inputs": {
            "active_smiles_sha256": sha256_file(active_smiles),
            "inactive_smiles_sha256": sha256_file(inactive_smiles),
        },
        "seed": seed,
        "max_conformers": max_conformers,
        "prepared_sdf_sha256": sha256_file(output_sdf),
        "counts": {
            "active": sum(item["label"] == "active" for item in labels),
            "inactive": sum(item["label"] == "inactive" for item in labels),
            "failed": len(failures),
            "conformers": sum(int(item["conformer_count"]) for item in labels),
        },
        "records": labels,
        "failures": failures,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    labels_output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_precomputed_sdf(
    path: Path,
    *,
    failures: list[dict[str, Any]] | None = None,
    strict: bool = True,
) -> Iterator[IndexedMolecule]:
    """Yield TriPharm records from the exact 3D coordinates in a prepared SDF.

    Benchmark callers may set ``strict=False`` only when they retain ``failures``
    in the result receipt and keep failed records in the evaluation denominator.
    """

    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required to read a screening benchmark") from exc
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for record_index, molecule in enumerate(supplier, start=1):
        if molecule is None:
            raise ValueError(f"RDKit could not parse prepared SDF record {record_index}")
        molecule_id = molecule.GetProp("_Name").strip()
        if not molecule_id:
            raise ValueError(f"prepared SDF record {record_index} has no identifier")
        try:
            yield molecule_to_existing_conformer_record(
                molecule_id,
                molecule,
                original_smiles=None,
                source=f"precomputed-3d:{path.name}",
            )
        except ValueError as exc:
            if strict:
                raise
            if failures is None:
                raise ValueError(
                    "non-strict SDF loading requires a failure receipt list"
                ) from exc
            failures.append(
                {
                    "record_id": molecule_id,
                    "source_record_index": record_index,
                    "reason": str(exc),
                }
            )


def load_grouped_precomputed_sdf(
    path: Path,
    *,
    failures: list[dict[str, Any]] | None = None,
    strict: bool = True,
) -> Iterator[IndexedMolecule]:
    """Group sequential equal-name SDF records into one TriPharm molecule."""

    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required to read a screening benchmark") from exc
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    current_id: str | None = None
    current: list[IndexedMolecule] = []

    def finish() -> IndexedMolecule | None:
        if not current:
            return None
        first = current[0]
        if any(item.standardized_smiles != first.standardized_smiles for item in current[1:]):
            raise ValueError(f"conformers disagree on standardized identity: {first.molecule_id}")
        return IndexedMolecule(
            molecule_id=first.molecule_id,
            original_smiles=first.original_smiles,
            standardized_smiles=first.standardized_smiles,
            conformers=tuple(
                FeatureConformer(conformer_id=index, features=item.conformers[0].features)
                for index, item in enumerate(current)
            ),
            source=first.source,
        )

    for record_index, molecule in enumerate(supplier, start=1):
        if molecule is None:
            raise ValueError(f"RDKit could not parse prepared SDF record {record_index}")
        conformer_name = molecule.GetProp("_Name").strip()
        molecule_id = (
            molecule.GetProp("ProtBindParent").strip()
            if molecule.HasProp("ProtBindParent")
            else conformer_name
        )
        if not conformer_name or not molecule_id:
            raise ValueError(f"prepared SDF record {record_index} has no identifier")
        if current_id is not None and molecule_id != current_id:
            completed = finish()
            if completed is not None:
                yield completed
            current = []
        current_id = molecule_id
        try:
            item = molecule_to_existing_conformer_record(
                molecule_id,
                molecule,
                original_smiles=None,
                source=f"shared-precomputed-3d:{path.name}",
            )
            if molecule.HasProp("ProtBindStandardizedSmiles"):
                stable_smiles = molecule.GetProp("ProtBindStandardizedSmiles")
                item = IndexedMolecule(
                    molecule_id=item.molecule_id,
                    original_smiles=stable_smiles,
                    standardized_smiles=stable_smiles,
                    conformers=item.conformers,
                    source=item.source,
                )
            current.append(item)
        except ValueError as exc:
            current = []
            current_id = None
            if strict:
                raise
            if failures is None:
                raise ValueError(
                    "non-strict grouped SDF loading requires a failure receipt list"
                ) from exc
            failures.append(
                {
                    "record_id": molecule_id,
                    "source_record_index": record_index,
                    "reason": str(exc),
                }
            )
    completed = finish()
    if completed is not None:
        yield completed


def _point_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.dist(
        (float(left["x"]), float(left["y"]), float(left["z"])),
        (float(right["x"]), float(right["y"]), float(right["z"])),
    )


def _query_subset_key(
    indexed_points: Sequence[tuple[int, dict[str, Any]]],
) -> tuple[int, float, float, tuple[int, ...]]:
    points = [point for _, point in indexed_points]
    distances = [
        _point_distance(left, right)
        for left, right in itertools.combinations(points, 2)
    ]
    return (
        len({str(point["name"]) for point in points}),
        min(distances),
        sum(distances),
        tuple(-index for index, _ in indexed_points),
    )


def freeze_pharmer_query_subset(
    source_query: Path,
    pharmer_query_output: Path,
    tripharm_query_output: Path,
    receipt_output: Path,
    *,
    max_points: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze a label-blind, geometrically diverse query shared by both engines."""

    outputs = (pharmer_query_output, tripharm_query_output, receipt_output)
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("one or more frozen query outputs already exist")
    payload = json.loads(source_query.read_text(encoding="utf-8"))
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError("Pharmer query does not contain a points array")
    eligible = [
        (index, point)
        for index, point in enumerate(points)
        if point.get("enabled") is True and point.get("name") in _PHARMER_TYPES
    ]
    if len(eligible) < 3:
        raise ValueError("Pharmer query contains fewer than three supported points")
    subset_size = min(max_points, len(eligible))
    if subset_size < 3:
        raise ValueError("max_points must be at least three")
    selected = max(itertools.combinations(eligible, subset_size), key=_query_subset_key)
    selected_indices = {index for index, _ in selected}

    frozen = json.loads(json.dumps(payload))
    for index, point in enumerate(frozen["points"]):
        point["enabled"] = index in selected_indices
    features = [
        FeaturePoint(
            feature_type=_PHARMER_TYPES[str(point["name"])],
            position=(float(point["x"]), float(point["y"]), float(point["z"])),
        ).to_dict()
        for _, point in selected
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    pharmer_query_output.write_text(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    tripharm_query_output.write_text(
        json.dumps({"features": features}, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.shared-pharmacophore-query",
        "selection": (
            "label-blind exhaustive subset; maximize feature-type diversity, then "
            "minimum pair distance, then total pair distance, then source order"
        ),
        "source_query_sha256": sha256_file(source_query),
        "max_points": max_points,
        "selected_source_indices": sorted(selected_indices),
        "selected_features": features,
        "pharmer_query_sha256": sha256_file(pharmer_query_output),
        "tripharm_query_sha256": sha256_file(tripharm_query_output),
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    receipt_output.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def freeze_pharmer_triangle_panel(
    source_query: Path,
    panel_directory: Path,
    tripharm_query_output: Path,
    receipt_output: Path,
    *,
    max_triangles: int = 64,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze the same high-information triangle panel for Pharmer and TriPharm."""

    if max_triangles < 1:
        raise ValueError("max_triangles must be positive")
    if not overwrite and (
        panel_directory.exists()
        or tripharm_query_output.exists()
        or receipt_output.exists()
    ):
        raise FileExistsError("triangle panel output already exists")
    payload = json.loads(source_query.read_text(encoding="utf-8"))
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError("Pharmer query does not contain a points array")
    eligible = [
        (source_index, point)
        for source_index, point in enumerate(points)
        if point.get("enabled") is True and point.get("name") in _PHARMER_TYPES
    ]
    features = tuple(
        FeaturePoint(
            feature_type=_PHARMER_TYPES[str(point["name"])],
            position=(float(point["x"]), float(point["y"]), float(point["z"])),
        )
        for _, point in eligible
    )
    if len(features) < 3:
        raise ValueError("Pharmer query contains fewer than three supported points")
    triangles = select_query_triangles(features, max_triangles=max_triangles)
    if not triangles:
        raise ValueError("Pharmer query contains no non-degenerate triangles")

    if panel_directory.exists():
        existing = list(panel_directory.iterdir())
        if existing:
            raise FileExistsError("refusing to replace a non-empty triangle panel directory")
    panel_directory.mkdir(parents=True, exist_ok=True)
    tripharm_query_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    panel: list[dict[str, Any]] = []
    for rank, triangle in enumerate(triangles):
        eligible_indices = triangle.feature_indices
        source_indices = tuple(eligible[index][0] for index in eligible_indices)
        frozen = json.loads(json.dumps(payload))
        enabled = set(source_indices)
        for source_index, point in enumerate(frozen["points"]):
            point["enabled"] = source_index in enabled
        output = panel_directory / f"triangle-{rank:03d}.json"
        output.write_text(
            json.dumps(frozen, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        panel.append(
            {
                "rank": rank,
                "eligible_feature_indices": list(eligible_indices),
                "source_point_indices": list(source_indices),
                "type_key": triangle.type_key,
                "query_sha256": sha256_file(output),
                "file": output.name,
            }
        )
    tripharm_query_output.write_text(
        json.dumps(
            {"features": [feature.to_dict() for feature in features]},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.shared-pharmacophore-triangle-panel",
        "selection": (
            "label-blind TriPharm information ordering: feature-type diversity, "
            "triangle area, perimeter, then source indices"
        ),
        "source_query_sha256": sha256_file(source_query),
        "feature_count": len(features),
        "max_triangles": max_triangles,
        "triangle_count": len(panel),
        "tripharm_query_sha256": sha256_file(tripharm_query_output),
        "panel": panel,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    receipt_output.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def export_training_ensemble_pharmer_panel(
    *,
    selection_receipt: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Export selected train-only TriPharm queries as Pharmer triangle queries."""

    if output_dir.exists():
        raise FileExistsError("Pharmer ensemble panel output already exists")
    selection = json.loads(selection_receipt.read_text(encoding="utf-8"))
    selected = selection.get("selected_query_sha256")
    if not isinstance(selected, dict) or not selected:
        raise ValueError("selection receipt contains no selected query commitments")
    output_dir.mkdir(parents=True)
    panel: list[dict[str, Any]] = []
    for query_rank, name in enumerate(sorted(selected)):
        source = candidate_dir / name
        if sha256_file(source) != selected[name]:
            raise ValueError("selected query hash does not match the candidate bank")
        features = read_query(source)
        triangles = select_query_triangles(features, max_triangles=64)
        for triangle_rank, triangle in enumerate(triangles):
            points = []
            for feature_index in triangle.feature_indices:
                feature = features[feature_index]
                pharmer_type, radius = _TRIPHARM_TO_PHARMER[feature.feature_type]
                points.append(
                    {
                        "enabled": True,
                        "name": pharmer_type,
                        "radius": radius,
                        "size": max(1, len(feature.atom_indices)),
                        "x": feature.position[0],
                        "y": feature.position[1],
                        "z": feature.position[2],
                    }
                )
            output = output_dir / f"q{query_rank:02d}-t{triangle_rank:03d}.json"
            output.write_bytes(canonical_json_bytes({"points": points}) + b"\n")
            panel.append(
                {
                    "query_rank": query_rank,
                    "triangle_rank": triangle_rank,
                    "source_query": name,
                    "source_feature_indices": list(triangle.feature_indices),
                    "file": output.name,
                    "sha256": sha256_file(output),
                }
            )
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.train-only-pharmer-triangle-ensemble",
        "selection_receipt_sha256": sha256_file(selection_receipt),
        "feature_mapping": {
            key.value: value[0] for key, value in _TRIPHARM_TO_PHARMER.items()
        },
        "semantics": (
            "same train-derived point coordinates and triangle decomposition; "
            "Pharmer and RDKit library feature perception remain independent"
        ),
        "triangle_query_count": len(panel),
        "panel": panel,
        "validation_inputs_read": [],
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    (output_dir / "panel-receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def ranked_screen_metrics(
    ranked_ids: Sequence[str],
    labels: dict[str, bool],
    *,
    fractions: Iterable[float] = (0.01, 0.05),
) -> dict[str, Any]:
    """Compute deterministic record-level retrieval metrics without hidden negatives."""

    if not labels or not any(labels.values()) or all(labels.values()):
        raise ValueError("labels must contain at least one active and one inactive")
    unknown = [identifier for identifier in ranked_ids if identifier not in labels]
    if unknown:
        raise ValueError(f"ranked IDs contain unknown records: {unknown[:3]}")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked IDs must be unique")
    active_total = sum(labels.values())
    library_size = len(labels)
    active_retrieved = sum(labels[identifier] for identifier in ranked_ids)
    prevalence = active_total / library_size
    cutoffs: dict[str, Any] = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("metric fractions must be in (0, 1]")
        cutoff = max(1, math.ceil(library_size * fraction))
        selected = ranked_ids[:cutoff]
        true_positives = sum(labels[identifier] for identifier in selected)
        complete = len(ranked_ids) >= cutoff
        item: dict[str, Any] = {
            "cutoff": cutoff,
            "retrieved": len(selected),
            "true_positives": true_positives,
            "status": "COMPLETE" if complete else "INCOMPLETE",
        }
        if complete:
            item["recall"] = true_positives / active_total
            item["enrichment_factor"] = (true_positives / cutoff) / prevalence
        cutoffs[f"{fraction:.6f}"] = item
    return {
        "library_size": library_size,
        "active_total": active_total,
        "ranked_count": len(ranked_ids),
        "active_retrieved_total": active_retrieved,
        "hit_set": {
            "precision": active_retrieved / len(ranked_ids) if ranked_ids else 0.0,
            "recall": active_retrieved / active_total,
            "enrichment_factor": (
                (active_retrieved / len(ranked_ids)) / prevalence if ranked_ids else 0.0
            ),
        },
        "cutoffs": cutoffs,
    }


def _score_groups(
    scores: dict[str, float], labels: dict[str, bool]
) -> list[tuple[float, int, int]]:
    if set(scores) != set(labels):
        missing = sorted(set(labels) - set(scores))
        extra = sorted(set(scores) - set(labels))
        raise ValueError(
            f"complete scores must match labels exactly; missing={missing[:3]}, extra={extra[:3]}"
        )
    if not labels or not any(labels.values()) or all(labels.values()):
        raise ValueError("labels must contain at least one active and one inactive")
    grouped: dict[float, list[int]] = {}
    for identifier, score in scores.items():
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("screen scores must be finite")
        counts = grouped.setdefault(value, [0, 0])
        counts[0 if labels[identifier] else 1] += 1
    return [
        (score, counts[0], counts[1])
        for score, counts in sorted(grouped.items(), reverse=True)
    ]


def _tie_aware_enrichment(
    groups: Sequence[tuple[float, int, int]],
    *,
    active_total: int,
    library_size: int,
    fraction: float,
) -> dict[str, Any]:
    if not 0 < fraction <= 1:
        raise ValueError("metric fractions must be in (0, 1]")
    cutoff = max(1, math.ceil(library_size * fraction))
    selected = 0
    active_before = 0
    expected = 0.0
    lower = 0
    upper = 0
    boundary_score: float | None = None
    boundary_tied = False
    for score, active_count, inactive_count in groups:
        size = active_count + inactive_count
        remaining = cutoff - selected
        if remaining <= 0:
            break
        if size <= remaining:
            selected += size
            active_before += active_count
            expected += active_count
            lower += active_count
            upper += active_count
            continue
        boundary_score = score
        boundary_tied = True
        expected += active_count * (remaining / size)
        lower += max(0, remaining - inactive_count)
        upper += min(remaining, active_count)
        selected += remaining
        break
    prevalence = active_total / library_size
    return {
        "fraction": fraction,
        "cutoff": cutoff,
        "expected_true_positives": expected,
        "true_positive_bounds": [lower, upper],
        "expected_recall": expected / active_total,
        "expected_enrichment_factor": (expected / cutoff) / prevalence,
        "enrichment_factor_bounds": [
            (lower / cutoff) / prevalence,
            (upper / cutoff) / prevalence,
        ],
        "boundary_score": boundary_score,
        "boundary_is_tied": boundary_tied,
    }


def _tie_aware_bedroc(
    groups: Sequence[tuple[float, int, int]],
    *,
    active_total: int,
    library_size: int,
    alpha: float,
) -> float:
    if alpha <= 0:
        raise ValueError("BEDROC alpha must be positive")
    rank = 1
    observed_sum = 0.0
    for _, active_count, inactive_count in groups:
        size = active_count + inactive_count
        weights = [math.exp(-alpha * item / library_size) for item in range(rank, rank + size)]
        observed_sum += active_count * (sum(weights) / size)
        rank += size

    def positional_sum(start: int, count: int) -> float:
        return sum(
            math.exp(-alpha * item / library_size)
            for item in range(start, start + count)
        )

    maximum = positional_sum(1, active_total)
    minimum = positional_sum(library_size - active_total + 1, active_total)
    if math.isclose(maximum, minimum):
        return 0.5
    return (observed_sum - minimum) / (maximum - minimum)


def complete_score_metrics(
    scores: dict[str, float],
    labels: dict[str, bool],
    *,
    fractions: Iterable[float] = (0.01, 0.05),
    bedroc_alpha: float = 20.0,
) -> dict[str, Any]:
    """Compute complete-population, tie-invariant screening metrics."""

    groups = _score_groups(scores, labels)
    active_total = sum(labels.values())
    library_size = len(labels)
    inactive_total = library_size - active_total
    prevalence = active_total / library_size

    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    concordant = 0.0
    negatives_below = inactive_total
    for _, active_count, inactive_count in groups:
        negatives_below -= inactive_count
        true_positives += active_count
        false_positives += inactive_count
        recall = true_positives / active_total
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        concordant += active_count * negatives_below
        concordant += 0.5 * active_count * inactive_count

    return {
        "library_size": library_size,
        "active_total": active_total,
        "inactive_total": inactive_total,
        "prevalence": prevalence,
        "score_group_count": len(groups),
        "zero_score_count": sum(1 for value in scores.values() if value == 0.0),
        "average_precision": average_precision,
        "average_precision_lift": average_precision / prevalence,
        "roc_auc": concordant / (active_total * inactive_total),
        "bedroc": {
            "alpha": bedroc_alpha,
            "value": _tie_aware_bedroc(
                groups,
                active_total=active_total,
                library_size=library_size,
                alpha=bedroc_alpha,
            ),
        },
        "cutoffs": {
            f"{fraction:.6f}": _tie_aware_enrichment(
                groups,
                active_total=active_total,
                library_size=library_size,
                fraction=fraction,
            )
            for fraction in fractions
        },
    }


def bootstrap_complete_score_metrics(
    scores: dict[str, float],
    labels: dict[str, bool],
    *,
    replicates: int = 1_000,
    seed: int = 20_260_802,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Stratified bootstrap intervals for complete-score screening metrics."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be in (0, 1)")
    _score_groups(scores, labels)
    active_scores = [scores[key] for key, active in labels.items() if active]
    inactive_scores = [scores[key] for key, active in labels.items() if not active]
    generator = random.Random(seed)
    samples: dict[str, list[float]] = {
        "average_precision": [],
        "average_precision_lift": [],
        "roc_auc": [],
        "bedroc_alpha20": [],
        "ef1": [],
    }
    for replicate in range(replicates):
        sampled_scores: dict[str, float] = {}
        sampled_labels: dict[str, bool] = {}
        for index in range(len(active_scores)):
            key = f"a{replicate}-{index}"
            sampled_scores[key] = generator.choice(active_scores)
            sampled_labels[key] = True
        for index in range(len(inactive_scores)):
            key = f"i{replicate}-{index}"
            sampled_scores[key] = generator.choice(inactive_scores)
            sampled_labels[key] = False
        metrics = complete_score_metrics(sampled_scores, sampled_labels)
        samples["average_precision"].append(metrics["average_precision"])
        samples["average_precision_lift"].append(metrics["average_precision_lift"])
        samples["roc_auc"].append(metrics["roc_auc"])
        samples["bedroc_alpha20"].append(metrics["bedroc"]["value"])
        samples["ef1"].append(metrics["cutoffs"]["0.010000"]["expected_enrichment_factor"])

    tail = (1.0 - confidence) / 2.0

    def quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        location = probability * (len(ordered) - 1)
        lower = math.floor(location)
        upper = math.ceil(location)
        if lower == upper:
            return ordered[lower]
        weight = location - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "method": "target-stratified bootstrap with replacement",
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "intervals": {
            name: [quantile(values, tail), quantile(values, 1.0 - tail)]
            for name, values in samples.items()
        },
    }


def _pharmer_panel_ranking(paths: Sequence[Path]) -> tuple[list[str], dict[str, Any]]:
    triangles: dict[str, set[str]] = {}
    best_rmsd: dict[str, float] = {}
    conformer_hit_count = 0
    for path in sorted(paths, key=lambda item: item.name):
        for record in path.read_text(encoding="utf-8", errors="replace").split("$$$$"):
            lines = record.lstrip("\r\n").splitlines()
            if not lines or not lines[0].strip():
                continue
            conformer_id = lines[0].strip()
            parent_id = re.sub(r"--c\d+$", "", conformer_id)
            match = _PHARMER_RMSD.search(record)
            rmsd = float(match.group(1)) if match else math.inf
            triangles.setdefault(parent_id, set()).add(path.name)
            best_rmsd[parent_id] = min(best_rmsd.get(parent_id, math.inf), rmsd)
            conformer_hit_count += 1
    ranking = sorted(
        triangles,
        key=lambda identifier: (
            -len(triangles[identifier]),
            best_rmsd[identifier],
            identifier,
        ),
    )
    return ranking, {
        "query_count": len(paths),
        "conformer_hit_count": conformer_hit_count,
        "parent_hit_count": len(ranking),
        "ranking": "descending triangle-query coverage, best RMSD, parent ID",
    }


def _pharmer_panel_complete_scores(
    paths: Sequence[Path], labels: dict[str, bool]
) -> dict[str, float]:
    triangles: dict[str, set[str]] = {}
    best_rmsd: dict[str, float] = {}
    for path in sorted(paths, key=lambda item: item.name):
        for record in path.read_text(encoding="utf-8", errors="replace").split("$$$$"):
            lines = record.lstrip("\r\n").splitlines()
            if not lines or not lines[0].strip():
                continue
            parent_id = re.sub(r"--c\d+$", "", lines[0].strip())
            if parent_id not in labels:
                raise ValueError(f"Pharmer result contains unknown parent ID: {parent_id}")
            match = _PHARMER_RMSD.search(record)
            rmsd = float(match.group(1)) if match else math.inf
            triangles.setdefault(parent_id, set()).add(path.name)
            best_rmsd[parent_id] = min(best_rmsd.get(parent_id, math.inf), rmsd)
    scores = dict.fromkeys(labels, 0.0)
    for parent_id, matched_queries in triangles.items():
        rmsd_term = 0.0 if math.isinf(best_rmsd[parent_id]) else 1.0 / (
            1.0 + best_rmsd[parent_id]
        )
        scores[parent_id] = len(matched_queries) + rmsd_term
    return scores


def build_frozen_ensemble_three_way_receipt(
    *,
    dataset_name: str,
    target: str,
    labels_path: Path,
    index_path: Path,
    selection_receipt: Path,
    candidate_dir: Path,
    pharmer_hit_paths: Sequence[Path],
    protocol_path: Path,
    authorization_path: Path,
    output: Path,
    hip_executable: Path,
    hip_static_cache_dir: Path,
    pharmer_provenance: dict[str, Any],
    bootstrap_replicates: int = 1_000,
    bootstrap_seed: int = 20260802,
) -> dict[str, Any]:
    """Evaluate one frozen train-only ensemble on one authorized validation split."""

    if output.exists():
        raise FileExistsError("prospective ensemble receipt already exists")
    authorization = verify_validation_authorization(
        protocol_path=protocol_path,
        target=target,
        receipt_path=authorization_path,
    )
    label_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    unsigned_labels = {
        key: value for key, value in label_payload.items() if key != "manifest_sha256"
    }
    if label_payload.get("manifest_sha256") != sha256_bytes(
        canonical_json_bytes(unsigned_labels)
    ):
        raise ValueError("validation label manifest hash is invalid")
    committed_target = authorization["protocol"]["prospective_targets"][target]
    if (
        label_payload.get("inputs", {}).get("active_smiles_sha256")
        != committed_target["active_validation"]["sha256"]
        or label_payload.get("inputs", {}).get("inactive_smiles_sha256")
        != committed_target["inactive_validation"]["sha256"]
    ):
        raise ValueError("validation library does not match frozen split hashes")
    _, index_stats = read_index_metadata(index_path)
    if index_stats.input_sha256 != label_payload.get("prepared_sdf_sha256"):
        raise ValueError("validation index does not match the prepared SDF commitment")
    labels = {
        str(item["record_id"]): item["label"] == "active"
        for item in label_payload["records"]
    }
    for failure in label_payload.get("failures", []):
        record_id = failure.get("record_id")
        label = failure.get("label")
        if isinstance(record_id, str) and label in {"active", "inactive"}:
            labels[record_id] = label == "active"
    selection = json.loads(selection_receipt.read_text(encoding="utf-8"))
    selected = selection.get("selected_query_sha256")
    chosen = selection.get("chosen")
    if not isinstance(selected, dict) or not selected or not isinstance(chosen, dict):
        raise ValueError("selection receipt is missing its frozen ensemble")
    query_paths = [candidate_dir / name for name in sorted(selected)]
    for path in query_paths:
        if sha256_file(path) != selected[path.name]:
            raise ValueError("selected query hash does not match frozen receipt")
    tolerance = float(chosen["tolerance_angstrom"])
    queries = tuple(read_query(path) for path in query_paths)

    cpu_started = time.perf_counter()
    cpu_batches = tuple(
        query_index(
            index_path,
            query,
            top_k=len(labels),
            tolerance_angstrom=tolerance,
        )
        for query in queries
    )
    cpu_seconds = time.perf_counter() - cpu_started
    cpu_scores = dict.fromkeys(labels, 0.0)
    for hits in cpu_batches:
        for hit in hits:
            if hit.molecule_id not in cpu_scores:
                raise ValueError("TriPharm CPU result contains an unknown validation ID")
            cpu_scores[hit.molecule_id] = max(
                cpu_scores[hit.molecule_id], hit.geometric_match_score
            )

    hip_result = query_index_batch_hip(
        index_path,
        queries,
        executable=hip_executable,
        static_cache_dir=hip_static_cache_dir,
        tolerance_angstrom=tolerance,
        top_k=len(labels),
        cpu_reference_ids=tuple(
            tuple(hit.molecule_id for hit in hits) for hits in cpu_batches
        ),
    )
    hip_scores = dict.fromkeys(labels, 0.0)
    for hits in hip_result.hits:
        for hit in hits:
            if hit.molecule_id not in hip_scores:
                raise ValueError("TriPharm HIP result contains an unknown validation ID")
            hip_scores[hit.molecule_id] = max(
                hip_scores[hit.molecule_id], hit.geometric_match_score
            )
    cpu_score_sha256 = sha256_bytes(canonical_json_bytes(cpu_scores))
    hip_score_sha256 = sha256_bytes(canonical_json_bytes(hip_scores))
    if cpu_score_sha256 != hip_score_sha256:
        raise ValueError("TriPharm CPU/HIP complete ensemble score parity failed")

    pharmer_scores = _pharmer_panel_complete_scores(pharmer_hit_paths, labels)
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.prospective-frozen-ensemble-three-way-screen",
        "dataset": {"name": dataset_name, "target": target, "split": "validation"},
        "claim_boundary": (
            "one-shot retrospective retrieval on untouched experimental labels; "
            "not affinity prediction or wet-lab prospective validation"
        ),
        "protocol_sha256": authorization["protocol"]["protocol_sha256"],
        "authorization_sha256": authorization["authorization"][
            "authorization_sha256"
        ],
        "inputs": {
            "labels_sha256": sha256_file(labels_path),
            "index_sha256": sha256_file(index_path),
            "selection_receipt_sha256": sha256_file(selection_receipt),
            "query_sha256": {path.name: sha256_file(path) for path in query_paths},
            "pharmer_hit_sha256": {
                path.name: sha256_file(path)
                for path in sorted(pharmer_hit_paths, key=lambda item: item.name)
            },
        },
        "population": {
            "records": len(labels),
            "actives": sum(labels.values()),
            "inactives": len(labels) - sum(labels.values()),
            "preprocessing_failures": len(label_payload.get("failures", [])),
            "failures_retained_as_zero": True,
        },
        "ensemble": {
            "size": len(queries),
            "tolerance_angstrom": tolerance,
            "aggregation": "maximum geometric match score across queries",
        },
        "pharmer_cpu": {
            "provenance": pharmer_provenance,
            "score": "matched triangle-query count plus reciprocal (1 + best RMSD)",
            "metrics": complete_score_metrics(pharmer_scores, labels),
            "bootstrap": bootstrap_complete_score_metrics(
                pharmer_scores,
                labels,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
        },
        "tripharm_cpu": {
            "wall_seconds": cpu_seconds,
            "complete_score_sha256": cpu_score_sha256,
            "metrics": complete_score_metrics(cpu_scores, labels),
            "bootstrap": bootstrap_complete_score_metrics(
                cpu_scores,
                labels,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
        },
        "tripharm_hip": {
            "complete_score_sha256": hip_score_sha256,
            "exact_to_cpu": True,
            "metrics": complete_score_metrics(hip_scores, labels),
            "receipt": hip_result.receipt,
        },
        "feature_semantics_warning": (
            "Pharmer SMARTS and RDKit BaseFeatures are independent feature perception "
            "implementations; application-lane disagreement is not a HIP correctness test"
        ),
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def build_three_way_screen_receipt(
    *,
    dataset_name: str,
    dataset_split: str,
    labels_path: Path,
    index_path: Path,
    query_path: Path,
    pharmer_hit_paths: Sequence[Path],
    output: Path,
    hip_executable: Path | None = None,
    pharmer_provenance: dict[str, Any] | None = None,
    top_k: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate Pharmer, TriPharm CPU, and optional HIP on one frozen library."""

    if output.exists() and not overwrite:
        raise FileExistsError("three-way screening receipt already exists")
    label_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = {
        str(item["record_id"]): item["label"] == "active"
        for item in label_payload["records"]
    }
    for item in label_payload.get("failures", []):
        record_id = item.get("record_id")
        label = item.get("label")
        if isinstance(record_id, str) and label in {"active", "inactive"}:
            labels[record_id] = label == "active"
    limit = top_k or len(labels)
    query = read_query(query_path)
    cpu_started = time.perf_counter()
    cpu_hits = query_index(index_path, query, top_k=limit)
    cpu_seconds = time.perf_counter() - cpu_started
    cpu_ids = [hit.molecule_id for hit in cpu_hits]
    pharmer_ids, pharmer_summary = _pharmer_panel_ranking(pharmer_hit_paths)
    unknown = sorted((set(cpu_ids) | set(pharmer_ids)) - set(labels))
    if unknown:
        raise ValueError(f"screen results contain unknown parent IDs: {unknown[:3]}")

    hip: dict[str, Any]
    if hip_executable is None:
        hip = {"status": "NOT_RUN"}
    else:
        hip_started = time.perf_counter()
        hip_result = query_index_hip(
            index_path,
            query,
            executable=hip_executable,
            top_k=limit,
        )
        hip_seconds = time.perf_counter() - hip_started
        hip_ids = [hit.molecule_id for hit in hip_result.hits]
        hip = {
            "status": "COMPLETED",
            "wall_seconds": hip_seconds,
            "ranked_ids_exact_to_cpu": hip_ids == cpu_ids,
            "metrics": ranked_screen_metrics(hip_ids, labels),
            "receipt": hip_result.receipt,
        }
    union = set(cpu_ids) | set(pharmer_ids)
    intersection = set(cpu_ids) & set(pharmer_ids)
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.three-way-pharmacophore-screen",
        "dataset": {"name": dataset_name, "split": dataset_split},
        "claim_boundary": (
            "retrospective experimental-label retrieval; not binding affinity, "
            "prospective hit validation, or an end-to-end GPU speedup claim"
        ),
        "inputs": {
            "labels_sha256": sha256_file(labels_path),
            "index_sha256": sha256_file(index_path),
            "query_sha256": sha256_file(query_path),
            "pharmer_hit_sha256": {
                path.name: sha256_file(path)
                for path in sorted(pharmer_hit_paths, key=lambda item: item.name)
            },
        },
        "pharmer_cpu": {
            "provenance": pharmer_provenance or {"status": "NOT_SUPPLIED"},
            **pharmer_summary,
            "metrics": ranked_screen_metrics(pharmer_ids, labels),
        },
        "tripharm_cpu": {
            "wall_seconds": cpu_seconds,
            "metrics": ranked_screen_metrics(cpu_ids, labels),
        },
        "tripharm_hip": hip,
        "cross_engine": {
            "intersection_count": len(intersection),
            "union_count": len(union),
            "hit_set_jaccard": len(intersection) / len(union) if union else 1.0,
            "feature_semantics_warning": (
                "Pharmer SMARTS and RDKit BaseFeatures are independent feature "
                "perception implementations; hit-set disagreement is not solely a "
                "search-kernel error measure"
            ),
        },
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt
