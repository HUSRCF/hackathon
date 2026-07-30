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


class TriPharmHIPError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HIPQueryResult:
    hits: tuple[TriPharmHit, ...]
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


def query_index_hip(
    index_path: Path,
    query_features: tuple[FeaturePoint, ...],
    *,
    executable: Path,
    top_k: int = 512,
    timeout_seconds: int = 600,
) -> HIPQueryResult:
    if os.getenv("HSA_OVERRIDE_GFX_VERSION"):
        raise TriPharmHIPError("HSA_OVERRIDE_GFX_VERSION is forbidden")
    resolved = executable.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise TriPharmHIPError("configured HIP production query executable is unavailable")
    cpu_started = time.perf_counter()
    cpu_hits = query_index(index_path, query_features, top_k=top_k)
    cpu_seconds = time.perf_counter() - cpu_started
    with tempfile.TemporaryDirectory(prefix="protbind-tripharm-hip-") as temporary:
        request = Path(temporary) / "query.bin"
        response = Path(temporary) / "result.bin"
        export_started = time.perf_counter()
        molecule_ids, triangle_count, query_count, tolerance = _export_query(
            index_path,
            query_features,
            request,
        )
        export_seconds = time.perf_counter() - export_started
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
    cpu_ids = [hit.molecule_id for hit in cpu_hits]
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
        "input_export_seconds": export_seconds,
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
