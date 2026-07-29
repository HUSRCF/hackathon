#!/usr/bin/env python3
"""Protocol-only quick-selection fixture; never scientific evidence."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _put(root: Path, value: bytes, media_type: str, producer: str) -> dict[str, Any]:
    digest = hashlib.sha256(value).hexdigest()
    directory = root / "objects" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / digest[2:]).write_bytes(value)
    return {
        "sha256": digest,
        "media_type": media_type,
        "size_bytes": len(value),
        "producer": producer,
        "producer_version": "fixture-1",
        "source": None,
        "license": None,
    }


def _put_json(root: Path, value: Any, producer: str) -> dict[str, Any]:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return _put(root, data, "application/json", producer)


def _read_json(root: Path, reference: dict[str, Any]) -> Any:
    path = root / "objects" / reference["sha256"][:2] / reference["sha256"][2:]
    return json.loads(path.read_text(encoding="utf-8"))


request = json.loads(sys.stdin.readline())
root = Path(os.environ["PROTBIND_ARTIFACT_ROOT"])
quick_input = _read_json(root, request["input"])
box_receipt = _read_json(root, quick_input["docking_box_receipt"])
marker_value = request["parameters"].get("fixture_marker")
previous_count = 0
if marker_value:
    marker = Path(marker_value)
    previous_count = int(marker.read_text(encoding="utf-8")) if marker.exists() else 0
    marker.write_text(str(previous_count + 1), encoding="utf-8")

if (
    request["parameters"].get("fixture_transient_fail_once", False) is True
    and previous_count == 0
):
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "job_id": request["job_id"],
                "engine": request["engine"],
                "outputs": [],
                "provenance": None,
                "timings_seconds": {},
                "peak_vram_bytes": None,
                "warnings": [],
                "error": {
                    "code": "TOOL_TIMEOUT",
                    "message": "fixture transient quick-Vina timeout",
                    "recoverable": True,
                },
            },
            separators=(",", ":"),
        )
    )
    raise SystemExit(0)

semantics = "AutoDock Vina tool score only; not an experimental binding free energy"
evaluations = []
referenced_outputs = []
all_fail = request["parameters"].get("fixture_all_fail", False) is True
for rank, item in enumerate(quick_input["requests"]):
    common = {
        "request_id": item["request_id"],
        "molecule_id": item["molecule_id"],
        "microstate_id": item["microstate_id"],
        "seed": request["seed"],
        "box_center": item["box_center"],
        "box_size": item["box_size"],
        "box_source": item["box_source"],
        "coordinate_frame": item["coordinate_frame"],
        "docking_box_receipt_sha256": item["docking_box_receipt_sha256"],
    }
    if all_fail:
        evaluations.append(
            {
                **common,
                "status": "failed",
                "code": "UNSUPPORTED_CHEMISTRY",
                "reason": "fixture requested deterministic unsupported chemistry",
                "recoverable": False,
            }
        )
        continue
    score = -7.0 + rank
    pose = _put(
        root,
        f"fixture-pose-{item['request_id']}".encode(),
        "chemical/x-mdl-sdfile",
        "fixture-vina-pose",
    )
    inner_evidence = _put_json(
        root,
        {
            "schema_version": "1.0",
            "kind": "protbind.tool-evidence",
            "tool": "vina",
            "tool_version": "1.2.7",
            "candidate_id": f"vina-{item['request_id']}",
            "parent_candidate_id": item["request_id"],
            "molecule_id": item["molecule_id"],
            "microstate_id": item["microstate_id"],
            "seed": request["seed"],
            "metrics": {
                "score": score,
                "box_center": item["box_center"],
                "box_size": item["box_size"],
                "box_source": item["box_source"],
                "coordinate_frame": item["coordinate_frame"],
                "docking_box_receipt_sha256": item[
                    "docking_box_receipt_sha256"
                ],
                "score_semantics": semantics,
                "scoring": "vina",
                "cpu": 1,
                "exhaustiveness": 8,
                "num_modes": 1,
                "heavy_element_counts": item["heavy_element_counts"],
                "formal_charge": item["formal_charge"],
            },
            "inputs": {
                "receptor": quick_input["receptor"],
                "docking_box_receipt": quick_input["docking_box_receipt"],
                "pose": pose,
                "pose_sdf": pose,
            },
        },
        "fixture-vina-inner-evidence",
    )
    outer_evidence = _put_json(
        root,
        {
            "schema_version": "1.0",
            "kind": "protbind.tool-evidence",
            "tool": "vina",
            "purpose": "selection-pruning-only",
            "request_id": item["request_id"],
            "molecule_id": item["molecule_id"],
            "microstate_id": item["microstate_id"],
            "seed": request["seed"],
            "metrics": {
                "score": score,
                "score_semantics": semantics,
                "box_center": item["box_center"],
                "box_size": item["box_size"],
                "box_source": item["box_source"],
                "coordinate_frame": item["coordinate_frame"],
                "docking_box_receipt_sha256": item[
                    "docking_box_receipt_sha256"
                ],
                "purpose": "selection-pruning-only",
                "scoring": "vina",
                "cpu": 1,
                "exhaustiveness": 8,
                "num_modes": 1,
                "formal_charge": item["formal_charge"],
            },
            "inputs": {
                "quick_vina_input": request["input"],
                "receptor": quick_input["receptor"],
                "docking_box_receipt": quick_input["docking_box_receipt"],
                "pose": pose,
                "inner_vina_evidence": inner_evidence,
            },
        },
        "fixture-vina-quick-evidence",
    )
    evaluations.append(
        {
            **common,
            "status": "completed",
            "score": score,
            "score_semantics": semantics,
            "pose": pose,
            "evidence": outer_evidence,
        }
    )
    referenced_outputs.extend((pose, outer_evidence, inner_evidence))

success_count = sum(item["status"] == "completed" for item in evaluations)
failure_count = len(evaluations) - success_count
inner_metadata = _put_json(
    root,
    {
        "schema_version": "1.0",
        "environment_lock": quick_input["environment_lock"],
        "docking_box_receipt": quick_input["docking_box_receipt"],
        "execution": {
            "device": "cpu",
            "cpu_threads": 1,
            "seed": request["seed"],
            "scoring": "vina",
            "exhaustiveness": 8,
            "num_modes": 1,
            "input_candidate_count": len(evaluations),
            "successful_candidate_count": success_count,
            "failed_candidate_count": failure_count,
            "box_source": quick_input["requests"][0]["box_source"],
            "coordinate_frame": quick_input["requests"][0]["coordinate_frame"],
            "docking_box_receipt_sha256": quick_input["docking_box_receipt"][
                "sha256"
            ],
            "site_derivation_verified": box_receipt["validation"][
                "site_derivation_verified"
            ],
            "site_scientific_interpretation": box_receipt["validation"][
                "scientific_interpretation"
            ],
        },
    },
    "fixture-vina-inner-run-metadata",
)
inner_bundle = _put_json(
    root,
    {
        "schema_version": "2.0",
        "kind": "protbind.docking-bundle",
        "receptor": quick_input["receptor"],
        "docking_box_receipt": quick_input["docking_box_receipt"],
        "upstream_candidate_ids": [item["request_id"] for item in quick_input["requests"]],
        "candidate_count": success_count,
        "failure_count": failure_count,
        "candidates": [],
        "failures": [],
        "run_metadata": inner_metadata,
    },
    "fixture-vina-inner-bundle",
)
metadata = _put_json(
    root,
    {
        "schema_version": "1.0",
        "environment_lock": quick_input["environment_lock"],
        "docking_box_receipt": quick_input["docking_box_receipt"],
        "inner_vina_run_metadata": inner_metadata,
        "execution": {
            "device": "cpu",
            "cpu_threads": 1,
            "purpose": "selection-pruning-only",
            "seed": request["seed"],
            "scoring": "vina",
            "exhaustiveness": 8,
            "num_modes": 1,
            "input_candidate_count": len(evaluations),
            "successful_candidate_count": success_count,
            "failed_candidate_count": failure_count,
            "box_source": quick_input["requests"][0]["box_source"],
            "coordinate_frame": quick_input["requests"][0]["coordinate_frame"],
            "docking_box_receipt_sha256": quick_input["docking_box_receipt"][
                "sha256"
            ],
            "site_derivation_verified": box_receipt["validation"][
                "site_derivation_verified"
            ],
            "site_scientific_interpretation": box_receipt["validation"][
                "scientific_interpretation"
            ],
        },
    },
    "fixture-vina-quick-run-metadata",
)
batch = _put_json(
    root,
    {
        "schema_version": "1.0",
        "kind": "protbind.quick-vina-evaluation-batch",
        "purpose": "selection-pruning-only",
        "input": request["input"],
        "selection_preparation_sha256": quick_input[
            "selection_preparation_sha256"
        ],
        "receptor": quick_input["receptor"],
        "docking_box_receipt": quick_input["docking_box_receipt"],
        "request_count": len(evaluations),
        "success_count": success_count,
        "failure_count": failure_count,
        "evaluations": evaluations,
        "run_metadata": metadata,
        "inner_docking_bundle": inner_bundle,
    },
    "fixture-vina-quick-batch",
)
print(
    json.dumps(
        {
            "schema_version": "1.0",
            "job_id": request["job_id"],
            "engine": request["engine"],
            "outputs": [
                batch,
                metadata,
                inner_bundle,
                inner_metadata,
                *referenced_outputs,
            ],
            "provenance": request["provenance"],
            "timings_seconds": {"fixture": 0.001},
            "peak_vram_bytes": None,
            "warnings": ["fixture quick Vina output; not scientific evidence"],
            "error": None,
        },
        separators=(",", ":"),
    )
)
