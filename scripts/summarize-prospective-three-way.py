#!/usr/bin/env python3
"""Build a self-verifying aggregate from frozen prospective screen receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    receipts.sort(key=lambda item: item["dataset"]["target"])
    if [item["dataset"]["target"] for item in receipts] != [
        "ALDH1",
        "MAPK1",
        "MTORC1",
    ]:
        raise ValueError("aggregate requires exactly ALDH1, MAPK1, and MTORC1")
    protocol_hashes = {item["protocol_sha256"] for item in receipts}
    if len(protocol_hashes) != 1:
        raise ValueError("prospective receipts do not share one protocol hash")

    targets = []
    for path, item in zip(sorted(args.input, key=lambda p: p.name), receipts, strict=True):
        cpu = item["tripharm_cpu"]
        hip = item["tripharm_hip"]
        pharmer = item["pharmer_cpu"]
        if not hip["exact_to_cpu"] or cpu["complete_score_sha256"] != hip["complete_score_sha256"]:
            raise ValueError(f"CPU/HIP parity failed for {item['dataset']['target']}")
        targets.append(
            {
                "target": item["dataset"]["target"],
                "receipt_file": path.name,
                "receipt_file_sha256": sha256(path.read_bytes()),
                "receipt_sha256": item["receipt_sha256"],
                "records": item["population"]["records"],
                "actives": item["population"]["actives"],
                "failures_retained_as_zero": item["population"]["failures_retained_as_zero"],
                "tripharm": {
                    "average_precision": cpu["metrics"]["average_precision"],
                    "average_precision_lift": cpu["metrics"]["average_precision_lift"],
                    "ef1": cpu["metrics"]["cutoffs"]["0.010000"]["expected_enrichment_factor"],
                    "top1_true_positives": cpu["metrics"]["cutoffs"]["0.010000"][
                        "expected_true_positives"
                    ],
                    "roc_auc": cpu["metrics"]["roc_auc"],
                },
                "pharmer": {
                    "average_precision": pharmer["metrics"]["average_precision"],
                    "average_precision_lift": pharmer["metrics"]["average_precision_lift"],
                    "ef1": pharmer["metrics"]["cutoffs"]["0.010000"]["expected_enrichment_factor"],
                    "top1_true_positives": pharmer["metrics"]["cutoffs"]["0.010000"][
                        "expected_true_positives"
                    ],
                    "roc_auc": pharmer["metrics"]["roc_auc"],
                },
                "cpu_hip_exact": True,
                "cpu_seconds": cpu["wall_seconds"],
                "hip_kernel_seconds": hip["receipt"]["kernel"]["kernel_seconds"],
                "hip_cpu_exact_finalize_seconds": hip["receipt"]["cpu_exact_finalize_seconds"],
            }
        )

    tri_ap = [item["tripharm"]["average_precision"] for item in targets]
    pharmer_ap = [item["pharmer"]["average_precision"] for item in targets]
    tri_ef1 = [item["tripharm"]["ef1"] for item in targets]
    exploratory_count = sum(
        item["tripharm"]["average_precision_lift"] > 1.0 and item["tripharm"]["ef1"] > 1.0
        for item in targets
    )
    top1_positive_count = sum(item["tripharm"]["top1_true_positives"] >= 1 for item in targets)
    median_tri_ap = statistics.median(tri_ap)
    median_pharmer_ap = statistics.median(pharmer_ap)
    median_tri_ef1 = statistics.median(tri_ef1)
    core = {
        "schema_version": "1.0",
        "kind": "protbind.prospective-three-target-aggregate",
        "protocol_sha256": next(iter(protocol_hashes)),
        "claim_boundary": (
            "one-shot retrospective retrieval on untouched experimental labels; "
            "not affinity prediction, wet-lab validation, or end-to-end acceleration"
        ),
        "targets": targets,
        "aggregate": {
            "target_count": 3,
            "median_tripharm_average_precision": median_tri_ap,
            "median_pharmer_average_precision": median_pharmer_ap,
            "median_tripharm_ef1": median_tri_ef1,
            "targets_with_tripharm_ap_lift_and_ef1_above_one": exploratory_count,
            "targets_with_top1_active": top1_positive_count,
        },
        "gates": {
            "gate_complete": {
                "pass": all(
                    item["failures_retained_as_zero"] and item["cpu_hip_exact"] for item in targets
                ),
                "criterion": "all three full denominators, metrics, intervals, and CPU/HIP parity",
            },
            "exploratory_positive": {
                "pass": exploratory_count >= 2,
                "criterion": "AP lift > 1 and EF1 > 1 on at least two targets",
            },
            "competition_strength": {
                "pass": median_tri_ef1 >= 2.0
                and top1_positive_count >= 2
                and median_tri_ap >= median_pharmer_ap,
                "criterion": "median EF1 >= 2, top1 active on >=2 targets, median AP >= Pharmer",
                "components": {
                    "median_ef1_at_least_2": median_tri_ef1 >= 2.0,
                    "top1_active_on_at_least_2_targets": top1_positive_count >= 2,
                    "median_ap_not_below_pharmer": median_tri_ap >= median_pharmer_ap,
                },
            },
        },
    }
    result = {**core, "aggregate_sha256": sha256(canonical(core))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result) + b"\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "aggregate_sha256": result["aggregate_sha256"],
                "gates": result["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
