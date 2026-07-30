"""Warm CPU TriPharm benchmark with hash-bound inputs and honest semantics."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from radeon_agent.hardware import probe_hardware

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from .tripharm import index_identity, query_index, read_query
from .tripharm_hip import query_index_hip


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * percentile
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def benchmark_cpu(
    index_path: Path,
    query_path: Path,
    *,
    repetitions: int = 5,
    warmup_runs: int = 1,
    top_k: int = 512,
) -> dict[str, Any]:
    if repetitions < 1 or warmup_runs < 0:
        raise ValueError("repetitions must be >= 1 and warmup_runs must be >= 0")
    features = read_query(query_path)
    for _ in range(warmup_runs):
        query_index(index_path, features, top_k=top_k)
    durations: list[float] = []
    result_ids: list[list[str]] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        hits = query_index(index_path, features, top_k=top_k)
        durations.append(time.perf_counter() - started)
        result_ids.append([hit.molecule_id for hit in hits])
    if any(result != result_ids[0] for result in result_ids[1:]):
        raise RuntimeError("CPU reference returned non-deterministic ranked sets")
    hardware = probe_hardware().to_dict()
    if hardware.get("hsa_override_active"):
        raise ValueError(
            "HSA_OVERRIDE_GFX_VERSION is active; benchmark evidence is invalid"
        )
    return {
        "schema_version": "1.0",
        "backend": "cpu-reference",
        "eligible_as_hip_performance_evidence": False,
        "index": index_identity(index_path),
        "query_sha256": sha256_file(query_path),
        "top_k": top_k,
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "duration_seconds": {
            "samples": durations,
            "p50": statistics.median(durations),
            "p95": _percentile(durations, 0.95),
        },
        "ranked_molecule_ids_sha256": sha256_bytes(canonical_json_bytes(result_ids[0])),
        "result_count": len(result_ids[0]),
        "hardware": hardware,
    }


def benchmark_hip(
    index_path: Path,
    query_path: Path,
    executable: Path,
    *,
    repetitions: int = 5,
    warmup_runs: int = 1,
    top_k: int = 512,
) -> dict[str, Any]:
    if repetitions < 1 or warmup_runs < 0:
        raise ValueError("repetitions must be >= 1 and warmup_runs must be >= 0")
    features = read_query(query_path)
    for _ in range(warmup_runs):
        query_index_hip(
            index_path,
            features,
            executable=executable,
            top_k=top_k,
        )
    durations: list[float] = []
    kernel_durations: list[float] = []
    result_ids: list[list[str]] = []
    receipts: list[dict[str, Any]] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = query_index_hip(
            index_path,
            features,
            executable=executable,
            top_k=top_k,
        )
        durations.append(time.perf_counter() - started)
        receipts.append(result.receipt)
        kernel_durations.append(float(result.receipt["kernel"]["kernel_seconds"]))
        result_ids.append([hit.molecule_id for hit in result.hits])
    if any(result != result_ids[0] for result in result_ids[1:]):
        raise RuntimeError("HIP-assisted query returned non-deterministic ranked sets")
    hardware = probe_hardware().to_dict()
    if hardware.get("hsa_override_active"):
        raise ValueError(
            "HSA_OVERRIDE_GFX_VERSION is active; benchmark evidence is invalid"
        )
    architectures = {
        str(receipt.get("kernel", {}).get("architecture")) for receipt in receipts
    }
    parity = all(receipt.get("ranked_molecule_ids_exact") is True for receipt in receipts)
    return {
        "schema_version": "1.0",
        "backend": "hip-prefilter+cpu-exact-ranking",
        "eligible_as_hip_performance_evidence": parity
        and architectures == {"gfx1100"},
        "score_semantics": "geometric pharmacophore match; not binding affinity",
        "index": index_identity(index_path),
        "query_sha256": sha256_file(query_path),
        "hip_executable_sha256": sha256_file(executable),
        "top_k": top_k,
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "duration_seconds": {
            "samples": durations,
            "p50": statistics.median(durations),
            "p95": _percentile(durations, 0.95),
        },
        "kernel_seconds": {
            "samples": kernel_durations,
            "p50": statistics.median(kernel_durations),
            "p95": _percentile(kernel_durations, 0.95),
        },
        "architectures": sorted(architectures),
        "ranked_molecule_ids_exact": parity,
        "ranked_molecule_ids_sha256": sha256_bytes(
            canonical_json_bytes(result_ids[0])
        ),
        "result_count": len(result_ids[0]),
        "hardware": hardware,
        "receipts": receipts,
    }


def save_benchmark(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
