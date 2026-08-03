#!/usr/bin/env python3
"""Repeat the static-index HIP batch lane without the CPU exact finalizer."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from protbind_agent.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from protbind_agent.tripharm import read_query
from protbind_agent.tripharm_hip import _write_batch_queries


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[rank]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--static-index", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    selection = json.loads(args.selection_receipt.read_text(encoding="utf-8"))
    selected = selection["selected_query_sha256"]
    paths = [args.query_dir / name for name in sorted(selected)]
    for path in paths:
        if sha256_file(path) != selected[path.name]:
            raise ValueError(f"query hash mismatch: {path.name}")
    queries = tuple(read_query(path) for path in paths)
    tolerance = float(selection["chosen"]["tolerance_angstrom"])
    environment = {
        key: os.environ[key]
        for key in ("PATH", "LD_LIBRARY_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        if os.environ.get(key)
    }
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="protbind-static-kernel-bench-") as temporary:
        root = Path(temporary)
        payload = root / "queries.tphipbat"
        _write_batch_queries(
            queries,
            max_query_triangles=64,
            tolerance_angstrom=tolerance,
            destination=payload,
        )
        for repetition in range(args.warmups + args.repetitions):
            response = root / f"response-{repetition}.tphipbo"
            started = time.perf_counter()
            process = subprocess.run(
                [
                    str(args.executable.resolve()),
                    "--index",
                    str(args.static_index.resolve()),
                    "--queries",
                    str(payload),
                    "--output",
                    str(response),
                ],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            wall = time.perf_counter() - started
            if process.returncode != 0:
                raise RuntimeError(process.stderr[-500:])
            kernel = json.loads(process.stdout)
            if repetition >= args.warmups:
                records.append({"wall_seconds": wall, "kernel": kernel})
        payload_hash = sha256_file(payload)
    walls = [item["wall_seconds"] for item in records]
    kernels = [item["kernel"]["kernel_seconds"] for item in records]
    h2d = [item["kernel"]["host_to_device_seconds"] for item in records]
    core = {
        "schema_version": "1.0",
        "kind": "protbind.tripharm-static-kernel-performance",
        "claim_boundary": (
            "static load, process startup, transfers and HIP prefilter only; "
            "excludes CPU exact finalizer and is not end-to-end application "
            "speedup evidence"
        ),
        "target": args.target,
        "gpu_selector": os.environ.get("HIP_VISIBLE_DEVICES"),
        "static_index_sha256": sha256_file(args.static_index),
        "static_index_bytes": args.static_index.stat().st_size,
        "selection_receipt_sha256": sha256_file(args.selection_receipt),
        "executable_sha256": sha256_file(args.executable),
        "query_payload_sha256": payload_hash,
        "batch_queries": len(queries),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "summary": {
            "process_wall_p50_seconds": statistics.median(walls),
            "process_wall_p95_seconds": percentile(walls, 0.95),
            "kernel_p50_seconds": statistics.median(kernels),
            "kernel_p95_seconds": percentile(kernels, 0.95),
            "h2d_p50_seconds": statistics.median(h2d),
            "batches_per_second_p50": 1.0 / statistics.median(walls),
            "queries_per_second_p50": len(queries) / statistics.median(walls),
        },
        "measurements": records,
    }
    result = {**core, "receipt_sha256": sha256_bytes(canonical_json_bytes(core))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result) + b"\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "receipt_sha256": result["receipt_sha256"],
                "summary": result["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
