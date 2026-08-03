"""Deterministic train-only pharmacophore query candidate selection."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from .chemistry import ChemistryCapabilityError, smiles_pharmacophore
from .screening_benchmark import complete_score_metrics
from .tripharm import query_index, read_query


def _score_query_worker(
    task: tuple[str, str, int, float],
) -> tuple[float, str, dict[str, float]]:
    """Run one read-only query in an isolated process/SQLite connection."""

    index_path, query_path, top_k, tolerance = task
    hits = query_index(
        Path(index_path),
        read_query(Path(query_path)),
        top_k=top_k,
        tolerance_angstrom=tolerance,
    )
    return (
        tolerance,
        Path(query_path).name,
        {hit.molecule_id: hit.geometric_match_score for hit in hits},
    )


def _rdkit_fingerprints() -> tuple[Any, Any, Any]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise ChemistryCapabilityError(
            "RDKit is required for train-only query medoid selection"
        ) from exc
    return Chem, DataStructs, rdFingerprintGenerator


def prepare_inner_selection_split(
    active_train: Path,
    inactive_train: Path,
    output_dir: Path,
    *,
    inactive_limit: int = 10_000,
) -> dict[str, Any]:
    """Create a hash-fixed fit/selection split without validation inputs."""

    if inactive_limit < 1:
        raise ValueError("inactive selection limit must be positive")
    if output_dir.exists():
        raise FileExistsError("inner selection split output already exists")

    def records(path: Path) -> list[tuple[str, str, str]]:
        loaded = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = raw_line.split()
            if not fields or fields[0].startswith("#"):
                continue
            source_id = fields[1] if len(fields) > 1 else f"line-{line_number:06d}"
            commitment = sha256_bytes(source_id.encode("utf-8"))
            loaded.append((commitment, fields[0], source_id))
        return loaded

    active = records(active_train)
    inactive = records(inactive_train)
    active_selection = [item for item in active if int(item[0][:8], 16) % 10 < 2]
    active_fit = [item for item in active if item not in active_selection]
    if not active_fit or not active_selection:
        raise ValueError("hash split must leave active records in fit and selection")
    inactive_selection = sorted(inactive)[:inactive_limit]
    output_dir.mkdir(parents=True)

    def write(name: str, selected: list[tuple[str, str, str]]) -> Path:
        path = output_dir / name
        path.write_text(
            "".join(f"{smiles} {source_id}\n" for _, smiles, source_id in selected),
            encoding="utf-8",
        )
        return path

    paths = {
        "active_fit": write("active_fit.smi", active_fit),
        "active_selection": write("active_selection.smi", active_selection),
        "inactive_selection": write("inactive_selection.smi", inactive_selection),
    }
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.train-inner-selection-split",
        "active_train_sha256": sha256_file(active_train),
        "inactive_train_sha256": sha256_file(inactive_train),
        "partition": "first 32 commitment bits modulo 10; 0-1 selection, 2-9 fit",
        "inactive_selection": "lowest SHA-256 source-ID commitments",
        "inactive_limit": inactive_limit,
        "counts": {
            "active_fit": len(active_fit),
            "active_selection": len(active_selection),
            "inactive_selection": len(inactive_selection),
        },
        "outputs_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "validation_inputs_read": [],
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    (output_dir / "split-receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def freeze_training_query_candidates(
    active_train: Path,
    output_dir: Path,
    *,
    max_queries: int = 16,
    max_points: int = 8,
    conformer_seed: int = 20_260_802,
) -> dict[str, Any]:
    """Select diverse active-training candidates without reading validation artifacts."""

    if not 1 <= max_queries <= 64:
        raise ValueError("max_queries must be in [1, 64]")
    if not 3 <= max_points <= 12:
        raise ValueError("max_points must be in [3, 12]")
    if output_dir.exists():
        raise FileExistsError("query candidate output already exists")
    chem, data_structures, fingerprint_generator = _rdkit_fingerprints()
    generator = fingerprint_generator.GetMorganGenerator(radius=2, fpSize=2048)
    records: dict[str, tuple[str, str, Any]] = {}
    failures: list[dict[str, str]] = []
    with active_train.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            fields = raw_line.split()
            if not fields or fields[0].startswith("#"):
                continue
            smiles = fields[0]
            source_id = fields[1] if len(fields) > 1 else f"line-{line_number:06d}"
            molecule = chem.MolFromSmiles(smiles)
            if molecule is None:
                failures.append({"source_id": source_id, "reason": "invalid_smiles"})
                continue
            canonical = str(chem.MolToSmiles(molecule, isomericSmiles=True))
            identity = sha256_bytes(canonical.encode("utf-8"))
            candidate = (source_id, canonical, generator.GetFingerprint(molecule))
            existing = records.get(identity)
            if existing is None or source_id < existing[0]:
                records[identity] = candidate
    if not records:
        raise ValueError("active training split contains no valid molecules")
    pool = [
        (identity, source_id, canonical, fingerprint)
        for identity, (source_id, canonical, fingerprint) in records.items()
    ]
    pool.sort(key=lambda item: (item[0], item[1]))
    selected: list[tuple[str, str, str, Any]] = []
    remaining = list(pool)
    while remaining and len(selected) < max_queries:
        if not selected:
            chosen_index = 0
        else:
            ranked: list[tuple[float, str, str, int]] = []
            selected_fingerprints = [item[3] for item in selected]
            for index, (identity, source_id, _, fingerprint) in enumerate(remaining):
                maximum_similarity = max(
                    data_structures.TanimotoSimilarity(fingerprint, chosen)
                    for chosen in selected_fingerprints
                )
                ranked.append((maximum_similarity, identity, source_id, index))
            chosen_index = min(ranked)[3]
        identity, source_id, canonical, fingerprint = remaining.pop(chosen_index)
        try:
            features = smiles_pharmacophore(
                canonical,
                seed=conformer_seed,
                max_points=max_points,
            )
            if len(features) < 3:
                raise ValueError("fewer_than_three_features")
        except (ValueError, ChemistryCapabilityError) as exc:
            failures.append(
                {
                    "source_id": source_id,
                    "identity_sha256": identity,
                    "reason": type(exc).__name__,
                }
            )
            continue
        selected.append((identity, source_id, canonical, fingerprint))
        output_dir.mkdir(parents=True, exist_ok=True)
        query_path = output_dir / f"query-{len(selected) - 1:02d}.json"
        query_path.write_bytes(
            canonical_json_bytes(
                {
                    "features": [feature.to_dict() for feature in features],
                    "provenance": {
                        "role": "active_train",
                        "source_id_sha256": sha256_bytes(source_id.encode("utf-8")),
                        "canonical_smiles_sha256": identity,
                        "selection": "deterministic Morgan farthest-first medoid candidate",
                    },
                }
            )
            + b"\n"
        )
    if not selected:
        raise ValueError("no training active produced a valid pharmacophore query")
    queries = sorted(output_dir.glob("query-*.json"))
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.train-only-query-candidate-bank",
        "active_train_sha256": sha256_file(active_train),
        "input_record_count": sum(
            1 for line in active_train.read_text().splitlines() if line.strip()
        ),
        "unique_parent_count": len(records),
        "candidate_count": len(queries),
        "max_points": max_points,
        "conformer_seed": conformer_seed,
        "selection": "Morgan radius-2 2048-bit deterministic farthest-first",
        "query_sha256": {path.name: sha256_file(path) for path in queries},
        "failure_count": len(failures),
        "failures": failures,
        "validation_inputs_read": [],
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    (output_dir / "candidate-bank-receipt.json").write_bytes(
        canonical_json_bytes(receipt) + b"\n"
    )
    return receipt


def select_training_query_ensemble(
    *,
    index_path: Path,
    labels_path: Path,
    candidate_dir: Path,
    output: Path,
    ensemble_sizes: tuple[int, ...] = (4, 8, 16),
    tolerances: tuple[float, ...] = (0.75, 1.0, 1.25),
    max_workers: int = 1,
) -> dict[str, Any]:
    """Select the frozen ensemble using only an inner training selection split."""

    if output.exists():
        raise FileExistsError("query ensemble selection receipt already exists")
    if not ensemble_sizes or not tolerances:
        raise ValueError("ensemble sizes and tolerances cannot be empty")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if any(size < 1 for size in ensemble_sizes) or any(value <= 0 for value in tolerances):
        raise ValueError("ensemble sizes and tolerances must be positive")
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = {
        str(item["record_id"]): item["label"] == "active"
        for item in payload["records"]
    }
    for failure in payload.get("failures", []):
        record_id = failure.get("record_id")
        if not isinstance(record_id, str):
            continue
        label = failure.get("label")
        if label not in {"active", "inactive"}:
            label = record_id.split("-", 1)[0]
        if label in {"active", "inactive"}:
            labels[record_id] = label == "active"
    candidates = sorted(candidate_dir.glob("query-*.json"))
    if len(candidates) < max(ensemble_sizes):
        raise ValueError("candidate bank is smaller than the requested ensemble grid")

    selected_candidates = candidates[: max(ensemble_sizes)]
    tasks = [
        (str(index_path), str(path), len(labels), tolerance)
        for tolerance in sorted(tolerances)
        for path in selected_candidates
    ]
    if max_workers == 1:
        scored = [_score_query_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            scored = list(executor.map(_score_query_worker, tasks, chunksize=1))
    candidate_index = {path.name: index for index, path in enumerate(selected_candidates)}
    per_query = {
        (tolerance, candidate_index[name]): scores
        for tolerance, name, scores in scored
    }

    grid: list[dict[str, Any]] = []
    for size in sorted(ensemble_sizes):
        for tolerance in sorted(tolerances):
            scores = dict.fromkeys(labels, 0.0)
            for index in range(size):
                for identifier, score in per_query[(tolerance, index)].items():
                    if identifier not in scores:
                        raise ValueError("query result contains an unknown selection record")
                    scores[identifier] = max(scores[identifier], score)
            metrics = complete_score_metrics(scores, labels)
            grid.append(
                {
                    "ensemble_size": size,
                    "tolerance_angstrom": tolerance,
                    "metrics": metrics,
                    "complete_score_sha256": sha256_bytes(
                        canonical_json_bytes(scores)
                    ),
                }
            )
    chosen = min(
        grid,
        key=lambda item: (
            -item["metrics"]["average_precision"],
            -item["metrics"]["cutoffs"]["0.010000"][
                "expected_enrichment_factor"
            ],
            item["ensemble_size"],
            item["tolerance_angstrom"],
        ),
    )
    selected_paths = candidates[: int(chosen["ensemble_size"])]
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.train-only-query-ensemble-selection",
        "selection_rule": (
            "maximum inner-selection AP, then EF1, then smaller ensemble, "
            "then lower tolerance"
        ),
        "index_sha256": sha256_file(index_path),
        "labels_sha256": sha256_file(labels_path),
        "candidate_bank_receipt_sha256": sha256_file(
            candidate_dir / "candidate-bank-receipt.json"
        ),
        "grid": grid,
        "chosen": chosen,
        "selected_query_sha256": {
            path.name: sha256_file(path) for path in selected_paths
        },
        "execution": {
            "max_workers": max_workers,
            "worker_isolation": "process; independent read-only SQLite connection",
        },
        "validation_inputs_read": [],
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt
