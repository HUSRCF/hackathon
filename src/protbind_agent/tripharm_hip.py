"""HIP production prefilter with mandatory CPU exact-ranking parity."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from .tripharm import (
    FeaturePoint,
    FeatureType,
    TriPharmHit,
    query_index,
    read_index_metadata,
    select_query_triangles,
)

_INPUT_MAGIC = b"TPHIPQ1\0"
_OUTPUT_MAGIC = b"TPHIPO1\0"
_HEADER = struct.Struct("<8sIIIIf")
_OUTPUT_HEADER = struct.Struct("<8sIII")
_TRIANGLE = struct.Struct("<I3Bx3f")
_TYPE_ID = {value: index for index, value in enumerate(FeatureType)}
_REQUEST_CACHE_SCHEMA = "tripharm-hip-query-cache-1.0"
_STATIC_INDEX_CACHE_SCHEMA = "tripharm-hip-static-index-cache-1.0"
_STATIC_INDEX_MAGIC = b"TPHIPIDX1\0\0\0"
_BATCH_QUERY_MAGIC = b"TPHIPBAT1\0\0\0"
_BATCH_OUTPUT_MAGIC = b"TPHIPBO1\0\0\0\0"
_STATIC_INDEX_HEADER = struct.Struct("<12sIII")
_BATCH_QUERY_HEADER = struct.Struct("<12sIIIf")
_BATCH_OUTPUT_HEADER = struct.Struct("<12sIII")


class TriPharmHIPError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HIPQueryResult:
    hits: tuple[TriPharmHit, ...]
    receipt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class HIPBatchQueryResult:
    hits: tuple[tuple[TriPharmHit, ...], ...]
    receipt: dict[str, Any]


def _distance(left: FeaturePoint, right: FeaturePoint) -> float:
    return math.dist(left.position, right.position)


def _packed_triangle(
    molecule_index: int,
    points: tuple[FeaturePoint, FeaturePoint, FeaturePoint],
) -> bytes:
    return _TRIANGLE.pack(
        molecule_index,
        *(_TYPE_ID[point.feature_type] for point in points),
        _distance(points[0], points[1]),
        _distance(points[0], points[2]),
        _distance(points[1], points[2]),
    )


def _export_query(
    index_path: Path,
    features: tuple[FeaturePoint, ...],
    destination: Path,
) -> tuple[list[str], int, int, float]:
    config, stats = read_index_metadata(index_path)
    query_triangles = select_query_triangles(
        features,
        max_triangles=config.max_query_triangles,
    )
    if not query_triangles:
        raise TriPharmHIPError("HIP query has no non-degenerate query triangles")
    if len(query_triangles) > 64:
        raise TriPharmHIPError("HIP query supports at most 64 query triangles")
    if stats.triangle_count > 50_000_000:
        raise TriPharmHIPError("HIP production export exceeds 50 million triangles")
    connection = sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)
    try:
        molecule_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT molecule_id FROM molecules ORDER BY molecule_id"
            )
        ]
        molecule_index = {
            molecule_id: index for index, molecule_id in enumerate(molecule_ids)
        }
        cached_key: tuple[str, int] | None = None
        cached_features: tuple[FeaturePoint, ...] = ()

        def conformer(molecule_id: str, conformer_id: int) -> tuple[FeaturePoint, ...]:
            nonlocal cached_key, cached_features
            key = (molecule_id, conformer_id)
            if key != cached_key:
                row = connection.execute(
                    "SELECT features_json FROM conformers "
                    "WHERE molecule_id = ? AND conformer_id = ?",
                    key,
                ).fetchone()
                if row is None:
                    raise TriPharmHIPError("index triangle references a missing conformer")
                cached_features = tuple(
                    FeaturePoint.from_dict(item) for item in json.loads(row[0])
                )
                cached_key = key
            return cached_features

        with destination.open("wb") as output:
            output.write(
                _HEADER.pack(
                    _INPUT_MAGIC,
                    1,
                    stats.triangle_count,
                    len(molecule_ids),
                    len(query_triangles),
                    config.tolerance_angstrom,
                )
            )
            written = 0
            for row in connection.execute(
                "SELECT molecule_id, conformer_id, feature_i, feature_j, feature_k "
                "FROM triangles ORDER BY rowid"
            ):
                molecule_id, conformer_id, left, center, right = row
                points = conformer(str(molecule_id), int(conformer_id))
                output.write(
                    _packed_triangle(
                        molecule_index[str(molecule_id)],
                        (points[int(left)], points[int(center)], points[int(right)]),
                    )
                )
                written += 1
            if written != stats.triangle_count:
                raise TriPharmHIPError("index triangle count changed during HIP export")
            for triangle in query_triangles:
                points = tuple(features[index] for index in triangle.feature_indices)
                output.write(_packed_triangle(0, points))
    finally:
        connection.close()
    return (
        molecule_ids,
        stats.triangle_count,
        len(query_triangles),
        config.tolerance_angstrom,
    )


def _read_prefilter(
    path: Path,
    *,
    molecule_ids: list[str],
    query_count: int,
) -> frozenset[str]:
    data = path.read_bytes()
    if len(data) < _OUTPUT_HEADER.size:
        raise TriPharmHIPError("HIP output is truncated")
    magic, schema, molecule_count, returned_queries = _OUTPUT_HEADER.unpack_from(data)
    if (
        magic != _OUTPUT_MAGIC
        or schema != 1
        or molecule_count != len(molecule_ids)
        or returned_queries != query_count
    ):
        raise TriPharmHIPError("HIP output identity does not match the submitted query")
    mask_size = molecule_count * 8
    error_size = molecule_count * query_count * 4
    expected = _OUTPUT_HEADER.size + mask_size + error_size
    if len(data) != expected:
        raise TriPharmHIPError("HIP output size does not match its declared dimensions")
    masks = struct.unpack_from(f"<{molecule_count}Q", data, _OUTPUT_HEADER.size)
    return frozenset(
        molecule_ids[index] for index, mask in enumerate(masks) if mask != 0
    )


def _request_identity(
    index_path: Path,
    features: tuple[FeaturePoint, ...],
) -> tuple[list[str], int, int, float]:
    config, stats = read_index_metadata(index_path)
    query_count = len(
        select_query_triangles(features, max_triangles=config.max_query_triangles)
    )
    if not 1 <= query_count <= 64:
        raise TriPharmHIPError("HIP query must contain between 1 and 64 triangles")
    connection = sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)
    try:
        molecule_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT molecule_id FROM molecules ORDER BY molecule_id"
            )
        ]
    finally:
        connection.close()
    return molecule_ids, stats.triangle_count, query_count, config.tolerance_angstrom


def _validate_cached_request(
    request: Path,
    metadata_path: Path,
    *,
    triangle_count: int,
    molecule_count: int,
    query_count: int,
    tolerance: float,
) -> str:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TriPharmHIPError("HIP request cache metadata is invalid") from exc
    observed_sha256 = sha256_file(request)
    if metadata.get("request_sha256") != observed_sha256:
        raise TriPharmHIPError("HIP request cache hash mismatch")
    expected_size = _HEADER.size + (triangle_count + query_count) * _TRIANGLE.size
    if request.stat().st_size != expected_size:
        raise TriPharmHIPError("HIP request cache size mismatch")
    with request.open("rb") as source:
        header = source.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise TriPharmHIPError("HIP request cache header is truncated")
    magic, schema, triangles, molecules, queries, cached_tolerance = _HEADER.unpack(header)
    if (
        magic != _INPUT_MAGIC
        or schema != 1
        or triangles != triangle_count
        or molecules != molecule_count
        or queries != query_count
        or not math.isclose(cached_tolerance, tolerance, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise TriPharmHIPError("HIP request cache identity mismatch")
    return observed_sha256


def _prepare_cached_request(
    index_path: Path,
    features: tuple[FeaturePoint, ...],
    cache_dir: Path,
) -> tuple[Path, list[str], int, int, float, bool, str, str]:
    index_sha256 = sha256_file(index_path)
    query_sha256 = sha256_bytes(
        canonical_json_bytes([feature.to_dict() for feature in features])
    )
    cache_key = sha256_bytes(
        canonical_json_bytes(
            {
                "schema": _REQUEST_CACHE_SCHEMA,
                "index_sha256": index_sha256,
                "query_sha256": query_sha256,
            }
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    request = cache_dir / f"{cache_key}.tphipq"
    metadata = cache_dir / f"{cache_key}.json"
    molecule_ids, triangle_count, query_count, tolerance = _request_identity(
        index_path, features
    )
    if request.exists() or metadata.exists():
        if not request.is_file() or not metadata.is_file():
            raise TriPharmHIPError("HIP request cache entry is incomplete")
        request_sha256 = _validate_cached_request(
            request,
            metadata,
            triangle_count=triangle_count,
            molecule_count=len(molecule_ids),
            query_count=query_count,
            tolerance=tolerance,
        )
        return (
            request,
            molecule_ids,
            triangle_count,
            query_count,
            tolerance,
            True,
            cache_key,
            request_sha256,
        )

    lock = cache_dir / f"{cache_key}.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise TriPharmHIPError("HIP request cache entry is being created") from exc
    os.close(descriptor)
    temporary = cache_dir / f".{cache_key}.{os.getpid()}.partial"
    temporary_metadata = cache_dir / f".{cache_key}.{os.getpid()}.json.partial"
    try:
        exported = _export_query(index_path, features, temporary)
        if exported != (molecule_ids, triangle_count, query_count, tolerance):
            raise TriPharmHIPError("HIP request export identity changed while caching")
        request_sha256 = sha256_file(temporary)
        temporary_metadata.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "1.0",
                    "kind": _REQUEST_CACHE_SCHEMA,
                    "cache_key": cache_key,
                    "index_sha256": index_sha256,
                    "query_sha256": query_sha256,
                    "request_sha256": request_sha256,
                }
            )
            + b"\n"
        )
        os.replace(temporary, request)
        os.replace(temporary_metadata, metadata)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    return (
        request,
        molecule_ids,
        triangle_count,
        query_count,
        tolerance,
        False,
        cache_key,
        request_sha256,
    )


def _static_index_identity(index_path: Path) -> tuple[list[str], int, str]:
    config, stats = read_index_metadata(index_path)
    if stats.triangle_count > 50_000_000:
        raise TriPharmHIPError("HIP static export exceeds 50 million triangles")
    connection = sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)
    try:
        molecule_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT molecule_id FROM molecules ORDER BY molecule_id"
            )
        ]
    finally:
        connection.close()
    return molecule_ids, stats.triangle_count, config.config_hash


def _export_static_index(index_path: Path, destination: Path) -> tuple[list[str], int, str]:
    molecule_ids, triangle_count, config_hash = _static_index_identity(index_path)
    molecule_index = {
        molecule_id: index for index, molecule_id in enumerate(molecule_ids)
    }
    connection = sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)
    try:
        cached_key: tuple[str, int] | None = None
        cached_features: tuple[FeaturePoint, ...] = ()

        def conformer(molecule_id: str, conformer_id: int) -> tuple[FeaturePoint, ...]:
            nonlocal cached_key, cached_features
            key = (molecule_id, conformer_id)
            if key != cached_key:
                row = connection.execute(
                    "SELECT features_json FROM conformers "
                    "WHERE molecule_id = ? AND conformer_id = ?",
                    key,
                ).fetchone()
                if row is None:
                    raise TriPharmHIPError(
                        "index triangle references a missing conformer"
                    )
                cached_features = tuple(
                    FeaturePoint.from_dict(item) for item in json.loads(row[0])
                )
                cached_key = key
            return cached_features

        with destination.open("wb") as output:
            output.write(
                _STATIC_INDEX_HEADER.pack(
                    _STATIC_INDEX_MAGIC,
                    1,
                    triangle_count,
                    len(molecule_ids),
                )
            )
            written = 0
            for row in connection.execute(
                "SELECT molecule_id, conformer_id, feature_i, feature_j, feature_k "
                "FROM triangles ORDER BY rowid"
            ):
                molecule_id, conformer_id, left, center, right = row
                points = conformer(str(molecule_id), int(conformer_id))
                output.write(
                    _packed_triangle(
                        molecule_index[str(molecule_id)],
                        (points[int(left)], points[int(center)], points[int(right)]),
                    )
                )
                written += 1
            if written != triangle_count:
                raise TriPharmHIPError(
                    "index triangle count changed during static HIP export"
                )
    finally:
        connection.close()
    return molecule_ids, triangle_count, config_hash


def _validate_static_index_cache(
    static_path: Path,
    metadata_path: Path,
    *,
    triangle_count: int,
    molecule_count: int,
    config_hash: str,
) -> tuple[str, list[str]]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TriPharmHIPError("HIP static cache metadata is invalid") from exc
    observed_sha256 = sha256_file(static_path)
    if metadata.get("static_index_sha256") != observed_sha256:
        raise TriPharmHIPError("HIP static cache hash mismatch")
    if metadata.get("config_hash") != config_hash:
        raise TriPharmHIPError("HIP static cache config mismatch")
    molecule_ids = metadata.get("molecule_ids")
    if (
        not isinstance(molecule_ids, list)
        or len(molecule_ids) != molecule_count
        or any(not isinstance(item, str) for item in molecule_ids)
        or len(set(molecule_ids)) != molecule_count
    ):
        raise TriPharmHIPError("HIP static cache molecule table is invalid")
    expected_size = _STATIC_INDEX_HEADER.size + triangle_count * _TRIANGLE.size
    if static_path.stat().st_size != expected_size:
        raise TriPharmHIPError("HIP static cache size mismatch")
    with static_path.open("rb") as source:
        header = source.read(_STATIC_INDEX_HEADER.size)
    magic, schema, triangles, molecules = _STATIC_INDEX_HEADER.unpack(header)
    if (
        magic != _STATIC_INDEX_MAGIC
        or schema != 1
        or triangles != triangle_count
        or molecules != molecule_count
    ):
        raise TriPharmHIPError("HIP static cache header mismatch")
    return observed_sha256, molecule_ids


def prepare_static_index_cache(
    index_path: Path,
    cache_dir: Path,
) -> tuple[Path, list[str], int, bool, str, str]:
    """Create or validate a content-addressed, query-independent TPHIPIDX1 file."""

    index_sha256 = sha256_file(index_path)
    molecule_ids, triangle_count, config_hash = _static_index_identity(index_path)
    exporter_sha256 = sha256_file(Path(__file__))
    cache_key = sha256_bytes(
        canonical_json_bytes(
            {
                "schema": _STATIC_INDEX_CACHE_SCHEMA,
                "index_sha256": index_sha256,
                "config_hash": config_hash,
                "exporter_sha256": exporter_sha256,
            }
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    static_path = cache_dir / f"{cache_key}.tphipidx"
    metadata_path = cache_dir / f"{cache_key}.json"
    if static_path.exists() or metadata_path.exists():
        if not static_path.is_file() or not metadata_path.is_file():
            raise TriPharmHIPError("HIP static cache entry is incomplete")
        static_sha256, cached_ids = _validate_static_index_cache(
            static_path,
            metadata_path,
            triangle_count=triangle_count,
            molecule_count=len(molecule_ids),
            config_hash=config_hash,
        )
        if cached_ids != molecule_ids:
            raise TriPharmHIPError("HIP static cache molecule identity mismatch")
        return static_path, molecule_ids, triangle_count, True, cache_key, static_sha256

    lock = cache_dir / f"{cache_key}.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise TriPharmHIPError("HIP static cache entry is being created") from exc
    os.close(descriptor)
    temporary = cache_dir / f".{cache_key}.{os.getpid()}.partial"
    temporary_metadata = cache_dir / f".{cache_key}.{os.getpid()}.json.partial"
    try:
        exported_ids, exported_triangles, exported_config = _export_static_index(
            index_path, temporary
        )
        if (exported_ids, exported_triangles, exported_config) != (
            molecule_ids,
            triangle_count,
            config_hash,
        ):
            raise TriPharmHIPError("static index identity changed while exporting")
        static_sha256 = sha256_file(temporary)
        temporary_metadata.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "1.0",
                    "kind": _STATIC_INDEX_CACHE_SCHEMA,
                    "cache_key": cache_key,
                    "source_index_sha256": index_sha256,
                    "config_hash": config_hash,
                    "exporter_sha256": exporter_sha256,
                    "triangle_count": triangle_count,
                    "molecule_ids": molecule_ids,
                    "static_index_sha256": static_sha256,
                }
            )
            + b"\n"
        )
        os.replace(temporary, static_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    return static_path, molecule_ids, triangle_count, False, cache_key, static_sha256


def _write_batch_queries(
    query_features: Sequence[tuple[FeaturePoint, ...]],
    *,
    max_query_triangles: int,
    tolerance_angstrom: float,
    destination: Path,
) -> list[int]:
    triangle_sets = [
        select_query_triangles(features, max_triangles=max_query_triangles)
        for features in query_features
    ]
    if any(not triangles or len(triangles) > 64 for triangles in triangle_sets):
        raise TriPharmHIPError("each HIP batch query must have 1..64 triangles")
    offsets = [0]
    for triangles in triangle_sets:
        offsets.append(offsets[-1] + len(triangles))
    with destination.open("wb") as output:
        output.write(
            _BATCH_QUERY_HEADER.pack(
                _BATCH_QUERY_MAGIC,
                1,
                len(query_features),
                offsets[-1],
                tolerance_angstrom,
            )
        )
        output.write(struct.pack(f"<{len(offsets)}I", *offsets))
        for features, triangles in zip(query_features, triangle_sets, strict=True):
            for triangle in triangles:
                points = tuple(features[index] for index in triangle.feature_indices)
                output.write(_packed_triangle(0, points))
    return [len(triangles) for triangles in triangle_sets]


def _read_batch_prefilter(
    path: Path,
    *,
    molecule_ids: list[str],
    batch_count: int,
) -> tuple[frozenset[str], ...]:
    data = path.read_bytes()
    if len(data) < _BATCH_OUTPUT_HEADER.size:
        raise TriPharmHIPError("HIP batch output is truncated")
    magic, schema, molecule_count, returned_batches = _BATCH_OUTPUT_HEADER.unpack_from(data)
    expected_size = _BATCH_OUTPUT_HEADER.size + molecule_count * batch_count * 4
    if (
        magic != _BATCH_OUTPUT_MAGIC
        or schema != 1
        or molecule_count != len(molecule_ids)
        or returned_batches != batch_count
        or len(data) != expected_size
    ):
        raise TriPharmHIPError("HIP batch output identity mismatch")
    flags = struct.unpack_from(
        f"<{molecule_count * batch_count}I", data, _BATCH_OUTPUT_HEADER.size
    )
    return tuple(
        frozenset(
            molecule_ids[index]
            for index in range(molecule_count)
            if flags[batch * molecule_count + index] != 0
        )
        for batch in range(batch_count)
    )


def query_index_hip(
    index_path: Path,
    query_features: tuple[FeaturePoint, ...],
    *,
    executable: Path,
    top_k: int = 512,
    timeout_seconds: int = 600,
    request_cache_dir: Path | None = None,
    cpu_reference_ids: Sequence[str] | None = None,
) -> HIPQueryResult:
    if os.getenv("HSA_OVERRIDE_GFX_VERSION"):
        raise TriPharmHIPError("HSA_OVERRIDE_GFX_VERSION is forbidden")
    resolved = executable.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise TriPharmHIPError("configured HIP production query executable is unavailable")
    if cpu_reference_ids is None:
        cpu_started = time.perf_counter()
        cpu_hits = query_index(index_path, query_features, top_k=top_k)
        cpu_seconds = time.perf_counter() - cpu_started
        cpu_ids = [hit.molecule_id for hit in cpu_hits]
        cpu_reference_scope = "measured_inline"
    else:
        cpu_seconds = 0.0
        cpu_ids = list(cpu_reference_ids)
        if len(cpu_ids) != len(set(cpu_ids)) or len(cpu_ids) > top_k:
            raise ValueError("precomputed CPU reference IDs must be unique and fit top_k")
        cpu_reference_scope = "external_precomputed"
    with tempfile.TemporaryDirectory(prefix="protbind-tripharm-hip-") as temporary:
        response = Path(temporary) / "result.bin"
        export_started = time.perf_counter()
        if request_cache_dir is None:
            request = Path(temporary) / "query.bin"
            molecule_ids, triangle_count, query_count, tolerance = _export_query(
                index_path,
                query_features,
                request,
            )
            cache_hit = False
            cache_key = None
            request_sha256 = sha256_file(request)
        else:
            (
                request,
                molecule_ids,
                triangle_count,
                query_count,
                tolerance,
                cache_hit,
                cache_key,
                request_sha256,
            ) = _prepare_cached_request(index_path, query_features, request_cache_dir)
        request_prepare_seconds = time.perf_counter() - export_started
        environment = {
            key: os.environ[key]
            for key in (
                "PATH",
                "LD_LIBRARY_PATH",
                "ROCM_PATH",
                "ROCM_HOME",
                "HIP_PATH",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
            )
            if os.environ.get(key)
        }
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        try:
            process = subprocess.run(
                [str(resolved), "--input", str(request), "--output", str(response)],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TriPharmHIPError(f"HIP production query failed: {type(exc).__name__}") from exc
        if process.returncode != 0:
            detail = process.stderr.strip().splitlines()
            reason = detail[-1][:300] if detail else "non-zero exit"
            raise TriPharmHIPError(f"HIP production query failed: {reason}")
        try:
            kernel_receipt = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise TriPharmHIPError("HIP production query emitted invalid JSON") from exc
        candidates = _read_prefilter(
            response,
            molecule_ids=molecule_ids,
            query_count=query_count,
        )
    exact_started = time.perf_counter()
    hip_hits = query_index(
        index_path,
        query_features,
        top_k=top_k,
        candidate_molecule_ids=candidates,
    )
    exact_seconds = time.perf_counter() - exact_started
    hip_ids = [hit.molecule_id for hit in hip_hits]
    parity = cpu_ids == hip_ids
    receipt = {
        "schema_version": "1.0",
        "kind": "protbind.tripharm-hip-query-receipt",
        "backend": "hip-prefilter+cpu-exact-ranking",
        "score_semantics": "geometric pharmacophore match; not binding affinity",
        "executable_sha256": sha256_file(resolved),
        "index_sha256": sha256_file(index_path),
        "query_features_sha256": sha256_bytes(
            canonical_json_bytes([feature.to_dict() for feature in query_features])
        ),
        "top_k": top_k,
        "triangle_count": triangle_count,
        "query_triangle_count": query_count,
        "tolerance_angstrom": tolerance,
        "prefilter_molecule_count": len(candidates),
        "cpu_reference_seconds": cpu_seconds,
        "cpu_reference_scope": cpu_reference_scope,
        "request_prepare_seconds": request_prepare_seconds,
        "input_export_seconds": request_prepare_seconds if not cache_hit else 0.0,
        "request_cache_hit": cache_hit,
        "request_cache_key": cache_key,
        "request_sha256": request_sha256,
        "cpu_exact_finalize_seconds": exact_seconds,
        "kernel": kernel_receipt,
        "ranked_molecule_ids_exact": parity,
        "cpu_ranked_ids_sha256": sha256_bytes(canonical_json_bytes(cpu_ids)),
        "hip_ranked_ids_sha256": sha256_bytes(canonical_json_bytes(hip_ids)),
        "committed_backend": "hip" if parity else "none",
    }
    if not parity:
        raise TriPharmHIPError(
            "HIP/CPU top-k molecule_id order parity failed: "
            + json.dumps(receipt, sort_keys=True)
        )
    return HIPQueryResult(hits=tuple(hip_hits), receipt=receipt)


def query_index_batch_hip(
    index_path: Path,
    query_features: Sequence[tuple[FeaturePoint, ...]],
    *,
    executable: Path,
    static_cache_dir: Path,
    tolerance_angstrom: float,
    top_k: int = 512,
    timeout_seconds: int = 600,
    cpu_reference_ids: Sequence[Sequence[str]] | None = None,
) -> HIPBatchQueryResult:
    """Run an ensemble with one query-independent static upload and exact CPU ranking."""

    if os.getenv("HSA_OVERRIDE_GFX_VERSION"):
        raise TriPharmHIPError("HSA_OVERRIDE_GFX_VERSION is forbidden")
    if not query_features:
        raise ValueError("HIP batch requires at least one query")
    if tolerance_angstrom <= 0:
        raise ValueError("HIP batch tolerance must be positive")
    resolved = executable.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise TriPharmHIPError("configured HIP batch executable is unavailable")
    config, _ = read_index_metadata(index_path)

    if cpu_reference_ids is None:
        cpu_started = time.perf_counter()
        cpu_hits = tuple(
            tuple(
                query_index(
                    index_path,
                    features,
                    top_k=top_k,
                    tolerance_angstrom=tolerance_angstrom,
                )
            )
            for features in query_features
        )
        cpu_seconds = time.perf_counter() - cpu_started
        cpu_ids = tuple(tuple(hit.molecule_id for hit in hits) for hits in cpu_hits)
        cpu_reference_scope = "measured_inline"
    else:
        cpu_ids = tuple(tuple(items) for items in cpu_reference_ids)
        if len(cpu_ids) != len(query_features):
            raise ValueError("CPU batch reference count does not match query count")
        if any(len(ids) != len(set(ids)) or len(ids) > top_k for ids in cpu_ids):
            raise ValueError("CPU batch reference IDs must be unique and fit top_k")
        cpu_seconds = 0.0
        cpu_reference_scope = "external_precomputed"

    static_started = time.perf_counter()
    (
        static_path,
        molecule_ids,
        triangle_count,
        cache_hit,
        cache_key,
        static_sha256,
    ) = prepare_static_index_cache(index_path, static_cache_dir)
    static_prepare_seconds = time.perf_counter() - static_started
    with tempfile.TemporaryDirectory(prefix="protbind-tripharm-hip-batch-") as temporary:
        query_path = Path(temporary) / "queries.tphipbat"
        response_path = Path(temporary) / "result.tphipbo"
        query_triangle_counts = _write_batch_queries(
            query_features,
            max_query_triangles=config.max_query_triangles,
            tolerance_angstrom=tolerance_angstrom,
            destination=query_path,
        )
        environment = {
            key: os.environ[key]
            for key in (
                "PATH",
                "LD_LIBRARY_PATH",
                "ROCM_PATH",
                "ROCM_HOME",
                "HIP_PATH",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
            )
            if os.environ.get(key)
        }
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        process_started = time.perf_counter()
        try:
            process = subprocess.run(
                [
                    str(resolved),
                    "--index",
                    str(static_path),
                    "--queries",
                    str(query_path),
                    "--output",
                    str(response_path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TriPharmHIPError(
                f"HIP batch query failed: {type(exc).__name__}"
            ) from exc
        process_seconds = time.perf_counter() - process_started
        if process.returncode != 0:
            detail = process.stderr.strip().splitlines()
            reason = detail[-1][:300] if detail else "non-zero exit"
            raise TriPharmHIPError(f"HIP batch query failed: {reason}")
        try:
            kernel_receipt = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise TriPharmHIPError("HIP batch query emitted invalid JSON") from exc
        candidates = _read_batch_prefilter(
            response_path,
            molecule_ids=molecule_ids,
            batch_count=len(query_features),
        )
        query_payload_sha256 = sha256_file(query_path)

    exact_started = time.perf_counter()
    hip_hits = tuple(
        tuple(
            query_index(
                index_path,
                features,
                top_k=top_k,
                candidate_molecule_ids=candidate_ids,
                tolerance_angstrom=tolerance_angstrom,
            )
        )
        for features, candidate_ids in zip(query_features, candidates, strict=True)
    )
    exact_seconds = time.perf_counter() - exact_started
    hip_ids = tuple(tuple(hit.molecule_id for hit in hits) for hits in hip_hits)
    parity = cpu_ids == hip_ids
    def score_payload(
        batches: Sequence[Sequence[TriPharmHit]],
    ) -> list[list[list[str | float]]]:
        return [
            [[hit.molecule_id, hit.geometric_match_score] for hit in hits]
            for hits in batches
        ]
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.tripharm-hip-static-batch-query-receipt",
        "backend": "hip-static-resident-batch-prefilter+cpu-exact-ranking",
        "score_semantics": "geometric pharmacophore match; not binding affinity",
        "executable_sha256": sha256_file(resolved),
        "index_sha256": sha256_file(index_path),
        "static_index_sha256": static_sha256,
        "static_cache_key": cache_key,
        "static_cache_hit": cache_hit,
        "static_prepare_seconds": static_prepare_seconds,
        "static_export_seconds": 0.0 if cache_hit else static_prepare_seconds,
        "query_payload_sha256": query_payload_sha256,
        "query_features_sha256": [
            sha256_bytes(canonical_json_bytes([item.to_dict() for item in features]))
            for features in query_features
        ],
        "query_triangle_counts": query_triangle_counts,
        "tolerance_angstrom": tolerance_angstrom,
        "top_k": top_k,
        "triangle_count": triangle_count,
        "prefilter_molecule_counts": [len(items) for items in candidates],
        "cpu_reference_seconds": cpu_seconds,
        "cpu_reference_scope": cpu_reference_scope,
        "batch_process_seconds": process_seconds,
        "cpu_exact_finalize_seconds": exact_seconds,
        "kernel": kernel_receipt,
        "ranked_molecule_ids_exact": parity,
        "cpu_ranked_ids_sha256": sha256_bytes(canonical_json_bytes(cpu_ids)),
        "hip_ranked_ids_sha256": sha256_bytes(canonical_json_bytes(hip_ids)),
        "hip_ranked_scores_sha256": sha256_bytes(
            canonical_json_bytes(score_payload(hip_hits))
        ),
        "committed_backend": "hip" if parity else "none",
    }
    if cpu_reference_ids is None:
        receipt["cpu_ranked_scores_sha256"] = sha256_bytes(
            canonical_json_bytes(score_payload(cpu_hits))
        )
        if receipt["cpu_ranked_scores_sha256"] != receipt["hip_ranked_scores_sha256"]:
            parity = False
            receipt["ranked_molecule_ids_exact"] = False
            receipt["committed_backend"] = "none"
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    if not parity:
        raise TriPharmHIPError(
            "HIP/CPU batch ranked-score parity failed: "
            + json.dumps(receipt, sort_keys=True)
        )
    return HIPBatchQueryResult(hits=hip_hits, receipt=receipt)
