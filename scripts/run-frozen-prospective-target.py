#!/usr/bin/env python3
"""Thin logged launcher for an already frozen prospective target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protbind_agent.artifacts import sha256_file
from protbind_agent.screening_benchmark import (
    build_frozen_ensemble_three_way_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=("ALDH1", "MAPK1", "MTORC1"))
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    target = args.target
    scratch = args.scratch_root / target
    queries = Path("configs/pharmacophore-prospective-v2/queries") / target
    provenance = json.loads(
        (args.result_root / "pharmer-provenance.json").read_text(encoding="utf-8")
    )
    execution_manifest = args.result_root / f"{target}-execution-manifest-v2.json"
    provenance["target_execution_manifest_sha256"] = sha256_file(execution_manifest)
    output = args.result_root / f"lit-pcba-{target.lower()}-prospective-v1.json"
    result = build_frozen_ensemble_three_way_receipt(
        dataset_name="LIT-PCBA AVE-unbiased",
        target=target,
        labels_path=scratch / "validation-labels.json",
        index_path=scratch / "validation.sqlite",
        selection_receipt=queries / "selection-receipt.json",
        candidate_dir=queries,
        pharmer_hit_paths=tuple(sorted((scratch / "pharmer-panel-hits").glob("q*.sdf"))),
        protocol_path=args.result_root / "prospective-protocol-v1.json",
        authorization_path=args.result_root / "authorizations" / f"{target}.json",
        output=output,
        hip_executable=Path("build/tripharm_hip/tripharm_hip_batch_query"),
        hip_static_cache_dir=scratch / "hip-static-cache-v1",
        pharmer_provenance=provenance,
        bootstrap_replicates=1_000,
        bootstrap_seed=20260802,
    )
    print(
        json.dumps(
            {
                "target": target,
                "pharmer": result["pharmer_cpu"]["metrics"],
                "cpu": result["tripharm_cpu"]["metrics"],
                "hip_exact": result["tripharm_hip"]["exact_to_cpu"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
