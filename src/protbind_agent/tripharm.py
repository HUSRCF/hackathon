"""Deterministic three-point pharmacophore CPU reference and SQLite index.

The score is deliberately named a geometric match score.  It is not a binding
affinity, docking score, or experimental measurement.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import sqlite3
import statistics
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file

_SAFE_MOLECULE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_COHERENT_MAPPING_BEAM_WIDTH = 128


class FeatureType(StrEnum):
    DONOR = "Donor"
    ACCEPTOR = "Acceptor"
    AROMATIC = "Aromatic"
    HYDROPHOBE = "Hydrophobe"
    POSITIVE = "Positive"
    NEGATIVE = "Negative"


@dataclass(frozen=True, slots=True)
class FeaturePoint:
    feature_type: FeatureType
    position: tuple[float, float, float]
    atom_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.feature_type, FeatureType):
            object.__setattr__(self, "feature_type", FeatureType(self.feature_type))
        if len(self.position) != 3 or any(not math.isfinite(value) for value in self.position):
            raise ValueError("feature position must contain three finite coordinates")
        if any(index < 0 for index in self.atom_indices):
            raise ValueError("feature atom indices must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.feature_type.value,
            "position": list(self.position),
            "atom_indices": list(self.atom_indices),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FeaturePoint:
        return cls(
            feature_type=FeatureType(value.get("type", value.get("feature_type"))),
            position=tuple(float(item) for item in value["position"]),
            atom_indices=tuple(int(item) for item in value.get("atom_indices", ())),
        )


@dataclass(frozen=True, slots=True)
class FeatureConformer:
    conformer_id: int
    features: tuple[FeaturePoint, ...]

    def __post_init__(self) -> None:
        if self.conformer_id < 0:
            raise ValueError("conformer_id must be non-negative")
        if len(self.features) < 3:
            raise ValueError("a pharmacophore conformer needs at least three features")


@dataclass(frozen=True, slots=True)
class IndexedMolecule:
    molecule_id: str
    original_smiles: str
    standardized_smiles: str
    conformers: tuple[FeatureConformer, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        if not _SAFE_MOLECULE_ID.fullmatch(self.molecule_id):
            raise ValueError("molecule_id must be a safe 1-128 character identifier")
        if len(self.conformers) > 4:
            raise ValueError("TriPharm v1 indexes at most four conformers per molecule")
        if not self.conformers:
            raise ValueError("molecule requires at least one feature conformer")


@dataclass(frozen=True, slots=True)
class TriPharmConfig:
    bin_width_angstrom: float = 0.5
    tolerance_angstrom: float = 1.0
    max_conformers: int = 4
    max_query_points: int = 12
    max_query_triangles: int = 64
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported TriPharm schema: {self.schema_version}")
        if self.bin_width_angstrom <= 0 or self.tolerance_angstrom <= 0:
            raise ValueError("distance bin width and tolerance must be positive")
        if not 1 <= self.max_conformers <= 4:
            raise ValueError("max_conformers must be in [1, 4]")
        if self.max_query_points < 3:
            raise ValueError("max_query_points must be >= 3")
        if self.max_query_triangles < 1:
            raise ValueError("max_query_triangles must be >= 1")

    @property
    def config_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True, slots=True)
class Triangle:
    feature_indices: tuple[int, int, int]
    type_key: str
    sorted_distances: tuple[float, float, float]
    information_score: tuple[int, float, float]


@dataclass(frozen=True, slots=True)
class TriangleMatch:
    query_feature_indices: tuple[int, int, int]
    candidate_feature_indices: tuple[int, int, int]
    normalized_distance_error: float
    overlay: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class TriPharmHit:
    molecule_id: str
    conformer_id: int
    query_coverage: float
    median_normalized_distance_error: float
    matches: tuple[TriangleMatch, ...]
    original_smiles: str
    standardized_smiles: str

    @property
    def geometric_match_score(self) -> float:
        return self.query_coverage / (1.0 + self.median_normalized_distance_error)


@dataclass(frozen=True, slots=True)
class IndexStats:
    molecule_count: int
    conformer_count: int
    triangle_count: int
    input_sha256: str | None
    config_hash: str
    chemistry_verified: bool


def _distance(left: FeaturePoint, right: FeaturePoint) -> float:
    return math.dist(left.position, right.position)


def _triangle_area(points: Sequence[FeaturePoint]) -> float:
    a = tuple(points[1].position[index] - points[0].position[index] for index in range(3))
    b = tuple(points[2].position[index] - points[0].position[index] for index in range(3))
    cross = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
    return 0.5 * math.sqrt(sum(value * value for value in cross))


def enumerate_triangles(features: Sequence[FeaturePoint]) -> list[Triangle]:
    triangles: list[Triangle] = []
    for indices in itertools.combinations(range(len(features)), 3):
        points = [features[index] for index in indices]
        distances = tuple(
            sorted(
                (
                    _distance(points[0], points[1]),
                    _distance(points[0], points[2]),
                    _distance(points[1], points[2]),
                )
            )
        )
        if distances[0] <= 1e-8:
            continue
        type_key = "|".join(sorted(point.feature_type.value for point in points))
        information = (
            len({point.feature_type for point in points}),
            _triangle_area(points),
            sum(distances),
        )
        triangles.append(
            Triangle(
                feature_indices=indices,
                type_key=type_key,
                sorted_distances=distances,
                information_score=information,
            )
        )
    return triangles


def select_query_triangles(
    features: Sequence[FeaturePoint], *, max_triangles: int
) -> list[Triangle]:
    triangles = enumerate_triangles(features)
    triangles.sort(
        key=lambda item: (
            -item.information_score[0],
            -item.information_score[1],
            -item.information_score[2],
            item.feature_indices,
        )
    )
    return triangles[:max_triangles]


def _features_json(features: Sequence[FeaturePoint]) -> str:
    return canonical_json_bytes([feature.to_dict() for feature in features]).decode("utf-8")


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE molecules (
            molecule_id TEXT PRIMARY KEY,
            original_smiles TEXT NOT NULL,
            standardized_smiles TEXT NOT NULL,
            source TEXT
        );
        CREATE TABLE conformers (
            molecule_id TEXT NOT NULL,
            conformer_id INTEGER NOT NULL,
            features_json TEXT NOT NULL,
            PRIMARY KEY (molecule_id, conformer_id),
            FOREIGN KEY (molecule_id) REFERENCES molecules(molecule_id)
        );
        CREATE TABLE triangles (
            molecule_id TEXT NOT NULL,
            conformer_id INTEGER NOT NULL,
            feature_i INTEGER NOT NULL,
            feature_j INTEGER NOT NULL,
            feature_k INTEGER NOT NULL,
            type_key TEXT NOT NULL,
            bin_0 INTEGER NOT NULL,
            bin_1 INTEGER NOT NULL,
            bin_2 INTEGER NOT NULL,
            distance_0 REAL NOT NULL,
            distance_1 REAL NOT NULL,
            distance_2 REAL NOT NULL,
            FOREIGN KEY (molecule_id, conformer_id)
                REFERENCES conformers(molecule_id, conformer_id)
        );
        CREATE INDEX triangle_lookup
            ON triangles(type_key, bin_0, bin_1, bin_2);
        CREATE INDEX triangle_molecule
            ON triangles(molecule_id, conformer_id);
        """
    )


def build_index(
    molecules: Iterable[IndexedMolecule],
    output: Path,
    *,
    config: TriPharmConfig | None = None,
    input_sha256: str | None = None,
    chemistry_verified: bool = False,
    overwrite: bool = False,
) -> IndexStats:
    """Build an index atomically; an existing index is never silently replaced."""

    config = config or TriPharmConfig()
    if output.exists() and not overwrite:
        raise FileExistsError(f"index already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    molecule_count = conformer_count = triangle_count = 0
    try:
        connection = sqlite3.connect(temporary)
        try:
            _create_schema(connection)
            seen: set[str] = set()
            for molecule in molecules:
                if molecule.molecule_id in seen:
                    raise ValueError(f"duplicate molecule_id: {molecule.molecule_id}")
                seen.add(molecule.molecule_id)
                if len(molecule.conformers) > config.max_conformers:
                    raise ValueError(
                        f"{molecule.molecule_id} has more than {config.max_conformers} conformers"
                    )
                connection.execute(
                    "INSERT INTO molecules VALUES (?, ?, ?, ?)",
                    (
                        molecule.molecule_id,
                        molecule.original_smiles,
                        molecule.standardized_smiles,
                        molecule.source,
                    ),
                )
                molecule_count += 1
                for conformer in sorted(molecule.conformers, key=lambda item: item.conformer_id):
                    connection.execute(
                        "INSERT INTO conformers VALUES (?, ?, ?)",
                        (
                            molecule.molecule_id,
                            conformer.conformer_id,
                            _features_json(conformer.features),
                        ),
                    )
                    conformer_count += 1
                    for triangle in enumerate_triangles(conformer.features):
                        bins = tuple(
                            int(distance / config.bin_width_angstrom)
                            for distance in triangle.sorted_distances
                        )
                        connection.execute(
                            "INSERT INTO triangles VALUES "
                            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                molecule.molecule_id,
                                conformer.conformer_id,
                                *triangle.feature_indices,
                                triangle.type_key,
                                *bins,
                                *triangle.sorted_distances,
                            ),
                        )
                        triangle_count += 1
            metadata = {
                "schema_version": config.schema_version,
                "config": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
                "config_hash": config.config_hash,
                "input_sha256": input_sha256 or "",
                "molecule_count": str(molecule_count),
                "conformer_count": str(conformer_count),
                "triangle_count": str(triangle_count),
                "chemistry_verified": "true" if chemistry_verified else "false",
            }
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return IndexStats(
        molecule_count=molecule_count,
        conformer_count=conformer_count,
        triangle_count=triangle_count,
        input_sha256=input_sha256,
        config_hash=config.config_hash,
        chemistry_verified=chemistry_verified,
    )


def _parse_feature_conformer(value: dict[str, Any]) -> FeatureConformer:
    return FeatureConformer(
        conformer_id=int(value.get("conformer_id", value.get("id", 0))),
        features=tuple(FeaturePoint.from_dict(item) for item in value["features"]),
    )


def parse_indexed_molecule(value: dict[str, Any]) -> IndexedMolecule:
    original = value.get("original_smiles", value.get("smiles", ""))
    standardized = value.get("standardized_smiles", original)
    return IndexedMolecule(
        molecule_id=str(value["molecule_id"]),
        original_smiles=str(original),
        standardized_smiles=str(standardized),
        conformers=tuple(_parse_feature_conformer(item) for item in value["conformers"]),
        source=value.get("source"),
    )


def load_feature_jsonl(path: Path) -> Iterator[IndexedMolecule]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("record is not an object")
                yield parse_indexed_molecule(value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid feature record at {path.name}:{line_number}: {exc}"
                ) from exc


def build_jsonl_index(
    input_path: Path,
    output: Path,
    *,
    config: TriPharmConfig | None = None,
    overwrite: bool = False,
) -> IndexStats:
    return build_index(
        load_feature_jsonl(input_path),
        output,
        config=config or TriPharmConfig(),
        input_sha256=sha256_file(input_path),
        chemistry_verified=False,
        overwrite=overwrite,
    )


def read_query(path: Path) -> tuple[FeaturePoint, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("features")
    if not isinstance(value, list):
        raise ValueError("pharmacophore query must be a feature array or {features: [...]} object")
    return tuple(FeaturePoint.from_dict(item) for item in value)


def read_index_metadata(path: Path) -> tuple[TriPharmConfig, IndexStats]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    except sqlite3.Error as exc:
        raise ValueError(f"not a valid TriPharm index: {path.name}") from exc
    finally:
        connection.close()
    config = TriPharmConfig(**json.loads(metadata["config"]))
    if metadata.get("config_hash") != config.config_hash:
        raise ValueError("TriPharm index config hash mismatch")
    return config, IndexStats(
        molecule_count=int(metadata["molecule_count"]),
        conformer_count=int(metadata["conformer_count"]),
        triangle_count=int(metadata["triangle_count"]),
        input_sha256=metadata.get("input_sha256") or None,
        config_hash=metadata["config_hash"],
        chemistry_verified=metadata.get("chemistry_verified") == "true",
    )


def _load_features(
    connection: sqlite3.Connection,
    cache: dict[tuple[str, int], tuple[FeaturePoint, ...]],
    molecule_id: str,
    conformer_id: int,
) -> tuple[FeaturePoint, ...]:
    key = (molecule_id, conformer_id)
    if key not in cache:
        row = connection.execute(
            "SELECT features_json FROM conformers WHERE molecule_id = ? AND conformer_id = ?",
            key,
        ).fetchone()
        if row is None:
            raise ValueError(f"index references missing conformer: {molecule_id}/{conformer_id}")
        cache[key] = tuple(FeaturePoint.from_dict(item) for item in json.loads(row[0]))
    return cache[key]


def _best_correspondence(
    query_points: Sequence[FeaturePoint], candidate_points: Sequence[FeaturePoint], tolerance: float
) -> tuple[tuple[int, int, int], float] | None:
    edge_pairs = ((0, 1), (0, 2), (1, 2))
    best: tuple[float, float, tuple[int, int, int]] | None = None
    for permutation in itertools.permutations(range(3)):
        if any(
            query_points[index].feature_type
            is not candidate_points[permutation[index]].feature_type
            for index in range(3)
        ):
            continue
        errors = tuple(
            abs(
                _distance(query_points[left], query_points[right])
                - _distance(
                    candidate_points[permutation[left]], candidate_points[permutation[right]]
                )
            )
            / tolerance
            for left, right in edge_pairs
        )
        if any(error > 1.0 + 1e-12 for error in errors):
            continue
        candidate = (statistics.median(errors), statistics.fmean(errors), permutation)
        if best is None or candidate < best:
            best = candidate
    return (best[2], best[0]) if best is not None else None


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(value * vector[index] for index, value in enumerate(row)) for row in matrix]


def rigid_transform(
    source: Sequence[tuple[float, float, float]],
    target: Sequence[tuple[float, float, float]],
) -> tuple[tuple[float, float, float, float], ...]:
    """Return a Horn/Kabsch-like 4x4 transform mapping source onto target."""

    if len(source) != len(target) or len(source) < 3:
        raise ValueError("rigid transform requires equal point sets with at least three points")
    source_center = tuple(statistics.fmean(point[i] for point in source) for i in range(3))
    target_center = tuple(statistics.fmean(point[i] for point in target) for i in range(3))
    centered_source = [tuple(point[i] - source_center[i] for i in range(3)) for point in source]
    centered_target = [tuple(point[i] - target_center[i] for i in range(3)) for point in target]
    covariance = [[0.0] * 3 for _ in range(3)]
    for left, right in zip(centered_source, centered_target, strict=True):
        for i in range(3):
            for j in range(3):
                covariance[i][j] += left[i] * right[j]
    sxx, sxy, sxz = covariance[0]
    syx, syy, syz = covariance[1]
    szx, szy, szz = covariance[2]
    horn = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    shift = max(sum(abs(value) for value in row) for row in horn) + 1.0
    shifted = [
        [
            value + (shift if row_index == column_index else 0.0)
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(horn)
    ]
    quaternion = [1.0, 0.0, 0.0, 0.0]
    for _ in range(80):
        candidate = _mat_vec(shifted, quaternion)
        norm = math.sqrt(sum(value * value for value in candidate))
        if norm <= 1e-15:
            break
        candidate = [value / norm for value in candidate]
        if sum((candidate[i] - quaternion[i]) ** 2 for i in range(4)) <= 1e-28:
            quaternion = candidate
            break
        quaternion = candidate
    w, x, y, z = quaternion
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    rotated_center = tuple(
        sum(rotation[row][column] * source_center[column] for column in range(3))
        for row in range(3)
    )
    translation = tuple(target_center[i] - rotated_center[i] for i in range(3))
    return tuple(
        tuple(rotation[row][column] for column in range(3)) + (translation[row],)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def _merge_coherent_mapping(
    mapping: tuple[tuple[int, int], ...],
    match: TriangleMatch,
    query_features: Sequence[FeaturePoint],
    candidate_features: Sequence[FeaturePoint],
    tolerance: float,
) -> tuple[tuple[int, int], ...] | None:
    """Merge a triangle correspondence into one injective, coherent mapping."""

    forward = dict(mapping)
    reverse = {candidate_index: query_index for query_index, candidate_index in mapping}
    for query_index, candidate_index in zip(
        match.query_feature_indices,
        match.candidate_feature_indices,
        strict=True,
    ):
        existing_candidate = forward.get(query_index)
        if existing_candidate is not None and existing_candidate != candidate_index:
            return None
        existing_query = reverse.get(candidate_index)
        if existing_query is not None and existing_query != query_index:
            return None
        if (
            query_features[query_index].feature_type
            is not candidate_features[candidate_index].feature_type
        ):
            return None
        forward[query_index] = candidate_index
        reverse[candidate_index] = query_index

    # Triangle-local tolerances are insufficient when several disconnected or
    # overlapping triangles are combined.  Requiring every mapped pair to obey
    # the same distance tolerance makes the eventual global overlay coherent.
    bindings = tuple(sorted(forward.items()))
    for (left_query, left_candidate), (right_query, right_candidate) in itertools.combinations(
        bindings, 2
    ):
        query_distance = _distance(
            query_features[left_query], query_features[right_query]
        )
        candidate_distance = _distance(
            candidate_features[left_candidate], candidate_features[right_candidate]
        )
        if abs(query_distance - candidate_distance) > tolerance + 1e-12:
            return None
    return bindings


def _matches_for_mapping(
    mapping: tuple[tuple[int, int], ...], matches: Sequence[TriangleMatch]
) -> tuple[TriangleMatch, ...]:
    forward = dict(mapping)
    return tuple(
        match
        for match in matches
        if all(
            forward.get(query_index) == candidate_index
            for query_index, candidate_index in zip(
                match.query_feature_indices,
                match.candidate_feature_indices,
                strict=True,
            )
        )
    )


def _mapping_sort_key(
    mapping: tuple[tuple[int, int], ...], matches: Sequence[TriangleMatch]
) -> tuple[Any, ...]:
    selected = _matches_for_mapping(mapping, matches)
    errors = tuple(match.normalized_distance_error for match in selected)
    return (
        -len(mapping),
        statistics.median(errors) if errors else math.inf,
        -len(selected),
        statistics.fmean(errors) if errors else math.inf,
        mapping,
    )


def _select_coherent_matches(
    matches: Sequence[TriangleMatch],
    query_features: Sequence[FeaturePoint],
    candidate_features: Sequence[FeaturePoint],
    tolerance: float,
) -> tuple[TriangleMatch, ...]:
    """Select triangle matches explained by one bounded, deterministic mapping.

    The fast path is linear for the usual case where all best per-triangle
    correspondences agree.  Conflicting cases use a fixed-width beam, bounded
    by the configured maximum of 64 query triangles and 12 query features.
    """

    ordered = tuple(
        sorted(
            matches,
            key=lambda match: (
                match.query_feature_indices,
                match.normalized_distance_error,
                match.candidate_feature_indices,
            ),
        )
    )
    if not ordered:
        return ()

    mapping: tuple[tuple[int, int], ...] = ()
    for match in ordered:
        merged = _merge_coherent_mapping(
            mapping, match, query_features, candidate_features, tolerance
        )
        if merged is None:
            break
        mapping = merged
    else:
        best_mapping = mapping
        selected = ordered
        return _with_shared_overlay(
            best_mapping, selected, query_features, candidate_features
        )

    # Preserve both skipping and accepting a conflicting match.  A singleton
    # is injected at every step so a good mapping that starts late in the
    # deterministic order cannot be excluded merely because the empty state
    # was pruned.  Deduplication by the full mapping removes path duplicates.
    states: set[tuple[tuple[int, int], ...]] = {()}
    considered: list[TriangleMatch] = []
    for match in ordered:
        considered.append(match)
        expanded = set(states)
        singleton = _merge_coherent_mapping(
            (), match, query_features, candidate_features, tolerance
        )
        if singleton is not None:
            expanded.add(singleton)
        for state in states:
            merged = _merge_coherent_mapping(
                state, match, query_features, candidate_features, tolerance
            )
            if merged is not None:
                expanded.add(merged)
        states = set(
            sorted(
                expanded,
                key=lambda state: _mapping_sort_key(state, considered),
            )[:_COHERENT_MAPPING_BEAM_WIDTH]
        )

    nonempty_states = [state for state in states if state]
    if not nonempty_states:
        return ()
    best_mapping = min(
        nonempty_states,
        key=lambda state: _mapping_sort_key(state, ordered),
    )
    selected = _matches_for_mapping(best_mapping, ordered)
    return _with_shared_overlay(
        best_mapping, selected, query_features, candidate_features
    )


def _with_shared_overlay(
    mapping: tuple[tuple[int, int], ...],
    matches: Sequence[TriangleMatch],
    query_features: Sequence[FeaturePoint],
    candidate_features: Sequence[FeaturePoint],
) -> tuple[TriangleMatch, ...]:
    overlay = rigid_transform(
        [candidate_features[candidate_index].position for _, candidate_index in mapping],
        [query_features[query_index].position for query_index, _ in mapping],
    )
    return tuple(
        TriangleMatch(
            query_feature_indices=match.query_feature_indices,
            candidate_feature_indices=match.candidate_feature_indices,
            normalized_distance_error=match.normalized_distance_error,
            overlay=overlay,
        )
        for match in matches
    )


def query_index(
    index_path: Path,
    query_features: Sequence[FeaturePoint],
    *,
    top_k: int = 512,
    tolerance_angstrom: float | None = None,
    candidate_molecule_ids: set[str] | frozenset[str] | None = None,
) -> list[TriPharmHit]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    config, _ = read_index_metadata(index_path)
    tolerance = tolerance_angstrom or config.tolerance_angstrom
    if tolerance <= 0:
        raise ValueError("tolerance_angstrom must be positive")
    if not 3 <= len(query_features) <= config.max_query_points:
        raise ValueError(
            f"query must contain between 3 and {config.max_query_points} feature points"
        )
    query_triangles = select_query_triangles(
        query_features, max_triangles=config.max_query_triangles
    )
    if not query_triangles:
        return []
    radius = math.ceil(tolerance / config.bin_width_angstrom) + 1
    connection = sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)
    feature_cache: dict[tuple[str, int], tuple[FeaturePoint, ...]] = {}
    # One best match per query triangle and conformer prevents duplicate rows from
    # biasing the median error.
    candidates: dict[tuple[str, int], dict[tuple[int, int, int], TriangleMatch]] = {}
    try:
        for query_triangle in query_triangles:
            query_points = [query_features[index] for index in query_triangle.feature_indices]
            query_bins = tuple(
                int(distance / config.bin_width_angstrom)
                for distance in query_triangle.sorted_distances
            )
            rows = connection.execute(
                "SELECT molecule_id, conformer_id, feature_i, feature_j, feature_k "
                "FROM triangles WHERE type_key = ? "
                "AND bin_0 BETWEEN ? AND ? AND bin_1 BETWEEN ? AND ? "
                "AND bin_2 BETWEEN ? AND ?",
                (
                    query_triangle.type_key,
                    query_bins[0] - radius,
                    query_bins[0] + radius,
                    query_bins[1] - radius,
                    query_bins[1] + radius,
                    query_bins[2] - radius,
                    query_bins[2] + radius,
                ),
            )
            for molecule_id, conformer_id, feature_i, feature_j, feature_k in rows:
                if (
                    candidate_molecule_ids is not None
                    and molecule_id not in candidate_molecule_ids
                ):
                    continue
                all_candidate_features = _load_features(
                    connection, feature_cache, molecule_id, conformer_id
                )
                stored_indices = (feature_i, feature_j, feature_k)
                candidate_points = [all_candidate_features[index] for index in stored_indices]
                correspondence = _best_correspondence(query_points, candidate_points, tolerance)
                if correspondence is None:
                    continue
                permutation, error = correspondence
                mapped_indices = tuple(stored_indices[permutation[index]] for index in range(3))
                mapped_points = [all_candidate_features[index] for index in mapped_indices]
                match = TriangleMatch(
                    query_feature_indices=query_triangle.feature_indices,
                    candidate_feature_indices=mapped_indices,
                    normalized_distance_error=error,
                    overlay=rigid_transform(
                        [point.position for point in mapped_points],
                        [point.position for point in query_points],
                    ),
                )
                key = (molecule_id, conformer_id)
                existing = candidates.setdefault(key, {}).get(query_triangle.feature_indices)
                if existing is None or (
                    match.normalized_distance_error,
                    match.candidate_feature_indices,
                ) < (
                    existing.normalized_distance_error,
                    existing.candidate_feature_indices,
                ):
                    candidates[key][query_triangle.feature_indices] = match

        molecule_rows = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                "SELECT molecule_id, original_smiles, standardized_smiles FROM molecules"
            )
        }
    finally:
        connection.close()

    conformer_hits: list[TriPharmHit] = []
    for (molecule_id, conformer_id), matches_by_query in candidates.items():
        candidate_features = feature_cache[(molecule_id, conformer_id)]
        matches = _select_coherent_matches(
            tuple(matches_by_query.values()),
            query_features,
            candidate_features,
            tolerance,
        )
        if not matches:
            continue
        covered = {index for match in matches for index in match.query_feature_indices}
        original, standardized = molecule_rows[molecule_id]
        conformer_hits.append(
            TriPharmHit(
                molecule_id=molecule_id,
                conformer_id=conformer_id,
                query_coverage=len(covered) / len(query_features),
                median_normalized_distance_error=statistics.median(
                    match.normalized_distance_error for match in matches
                ),
                matches=matches,
                original_smiles=original,
                standardized_smiles=standardized,
            )
        )
    # Keep the best conformer, using the published deterministic ordering.
    conformer_hits.sort(
        key=lambda hit: (
            hit.molecule_id,
            -hit.query_coverage,
            hit.median_normalized_distance_error,
            hit.conformer_id,
        )
    )
    best_by_molecule: dict[str, TriPharmHit] = {}
    for hit in conformer_hits:
        best_by_molecule.setdefault(hit.molecule_id, hit)
    hits = list(best_by_molecule.values())
    hits.sort(
        key=lambda hit: (
            -hit.query_coverage,
            hit.median_normalized_distance_error,
            hit.molecule_id,
        )
    )
    return hits[:top_k]


def index_identity(path: Path) -> dict[str, Any]:
    config, stats = read_index_metadata(path)
    return {
        "index_sha256": sha256_file(path),
        "config_hash": config.config_hash,
        "input_sha256": stats.input_sha256,
        "molecule_count": stats.molecule_count,
        "conformer_count": stats.conformer_count,
        "triangle_count": stats.triangle_count,
        "chemistry_verified": stats.chemistry_verified,
    }
