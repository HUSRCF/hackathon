#!/usr/bin/env python3
"""Run one offline ESMFold v1 worker smoke job with a hash-pinned checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import site
import sys
from pathlib import Path


def _exclude_user_site() -> None:
    """Keep parent-side provenance in the same pinned env as the worker."""

    raw_user_sites = site.getusersitepackages()
    user_sites = (
        (raw_user_sites,)
        if isinstance(raw_user_sites, str)
        else tuple(raw_user_sites)
    )
    resolved_user_sites = {
        str(Path(item).expanduser().resolve()) for item in user_sites
    }
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry
        or str(Path(entry).expanduser().resolve()) not in resolved_user_sites
    ]
    os.environ["PYTHONNOUSERSITE"] = "1"


_exclude_user_site()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protbind_agent.artifacts import (  # noqa: E402
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
)
from protbind_agent.worker_protocol import (  # noqa: E402
    JsonSubprocessWorker,
    WorkerProvenance,
    WorkerRequest,
)
from protbind_agent.workflow import (  # noqa: E402
    _gpu_lease,
    _stable_hardware_identity,
)
from radeon_agent.hardware import probe_hardware  # noqa: E402

WORKER = ROOT / "workers" / "esmfold_v1_worker.py"


def _checkpoint_set_sha256(
    model_path: Path, esm2_model_path: Path, esm2_regression_path: Path
) -> str:
    import importlib.util

    spec = importlib.util.spec_from_file_location("esmfold_worker", WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the ESMFold worker identity helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(
        module.checkpoint_set_sha256(
            model_path, esm2_model_path, esm2_regression_path
        )
    )


def _runtime_code_sha256(environment_lock: Path) -> str:
    import importlib.util

    import torch

    spec = importlib.util.spec_from_file_location("esmfold_worker_runtime", WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the ESMFold runtime identity helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.composite_code_sha256(environment_lock, torch))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline ESMFold v1 protocol smoke test using a local pinned checkpoint."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--esm2-model", type=Path, required=True)
    parser.add_argument("--esm2-regression", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--receipt-output",
        type=Path,
        required=True,
        help="Write a path-free receipt JSON for 'protbind case attach --name esmfold_receipt'.",
    )
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not all(
        path.is_file()
        for path in (
            args.model,
            args.esm2_model,
            args.esm2_regression,
            args.environment_lock,
        )
    ):
        raise FileNotFoundError("one or more local ESMFold checkpoint files do not exist")
    if args.receipt_output.exists():
        raise FileExistsError("receipt output already exists")
    if not args.receipt_output.parent.is_dir():
        raise FileNotFoundError("receipt output parent directory does not exist")
    if not args.device.isascii() or not args.device.isdecimal():
        raise ValueError("device must be one numeric HIP_VISIBLE_DEVICES index")
    # Capture the host identity before importing/initializing the model runtime.
    # Some ROCm diagnostic commands can transiently degrade immediately after a
    # large worker exits, while these stable fields describe the same host.
    hardware = probe_hardware().to_dict()
    hardware_sha256 = _stable_hardware_identity(hardware)
    store = ArtifactStore(args.workspace)
    input_artifact = store.put_json(
        {
            "schema_version": "1.0",
            "sequences": args.sequence,
        },
        producer="protbind.esmfold-v1.smoke-input",
        producer_version="1.0",
    )
    request = WorkerRequest(
        job_id="esmfold-v1-offline-smoke",
        engine="esmfold_v1",
        input=input_artifact,
        parameters={
            "model_path": str(args.model.resolve()),
            "esm2_model_path": str(args.esm2_model.resolve()),
            "esm2_regression_path": str(args.esm2_regression.resolve()),
            "environment_lock_path": str(args.environment_lock.resolve()),
            "chunk_sizes": [128, 64, 32],
            "minimum_free_vram_gib": 12.0,
        },
        seed=args.seed,
        provenance=WorkerProvenance(
            model_revision="esmfold_3B_v1",
            weight_sha256=_checkpoint_set_sha256(
                args.model, args.esm2_model, args.esm2_regression
            ),
            code_sha256=_runtime_code_sha256(args.environment_lock.resolve()),
        ),
    )
    with _gpu_lease(args.workspace.resolve(), args.device):
        response, elapsed = JsonSubprocessWorker(
            (sys.executable, str(WORKER)),
            timeout_seconds=args.timeout,
            environment={"HIP_VISIBLE_DEVICES": args.device},
            artifact_root=store.root,
            isolate_network=True,
        ).run(request)
    receipt_value = {
        "schema_version": "1.0",
        "kind": "protbind.esmfold-v1-smoke-receipt",
        "job_id": response.job_id,
        "engine": response.engine,
        "success": response.error is None,
        "input": input_artifact.to_dict(),
        "sequence_identity_sha256": [
            sha256_bytes(sequence.strip().upper().encode("ascii"))
            for sequence in args.sequence
        ],
        "outputs": [item.to_dict() for item in response.outputs],
        "timings_seconds": response.timings_seconds,
        "end_to_end_seconds": elapsed,
        "peak_vram_bytes": response.peak_vram_bytes,
        "warnings": list(response.warnings),
        "provenance": request.provenance.to_dict(),
        "hardware_sha256": hardware_sha256,
        "error": (
            {
                "code": response.error.code,
                "message": response.error.message,
                "recoverable": response.error.recoverable,
            }
            if response.error is not None
            else None
        ),
    }
    receipt = store.put_json(
        receipt_value,
        producer="protbind.esmfold-v1-smoke",
        producer_version="1.0",
        source=input_artifact.artifact_id,
    )
    args.receipt_output.write_bytes(canonical_json_bytes(receipt_value))
    result = {
        "schema_version": "1.0",
        "job_id": response.job_id,
        "engine": response.engine,
        "success": response.error is None,
        "output_artifact_ids": [item.artifact_id for item in response.outputs],
        "timings_seconds": response.timings_seconds,
        "end_to_end_seconds": elapsed,
        "peak_vram_bytes": response.peak_vram_bytes,
        "warnings": list(response.warnings),
        "provenance": request.provenance.to_dict(),
        "receipt_artifact_id": receipt.artifact_id,
        "hardware_sha256": hardware_sha256,
        "error": (
            {
                "code": response.error.code,
                "message": response.error.message,
                "recoverable": response.error.recoverable,
            }
            if response.error is not None
            else None
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if response.error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
