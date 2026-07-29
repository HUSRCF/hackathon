#!/usr/bin/env python3
"""Measure path-redacted AIAA provenance for quick and full Vina workers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
WORKER_ROOT = REPOSITORY_ROOT / "workers"

sys.path.insert(0, str(WORKER_ROOT))

import quick_vina_worker as quick  # noqa: E402
import vina_worker as vina  # noqa: E402

from protbind_agent.artifacts import sha256_file  # noqa: E402


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment_lock = args.environment_lock.resolve()
    if not environment_lock.is_file():
        raise FileNotFoundError("environment lock does not exist")
    parameters = {
        "vina_executable": str((REPOSITORY_ROOT / "tools/bin/vina").resolve()),
        "meeko_prepare_receptor_executable": str(
            (REPOSITORY_ROOT / ".venv-aiaa-protbind/bin/mk_prepare_receptor.py").resolve()
        ),
        "meeko_prepare_ligand_executable": str(
            (REPOSITORY_ROOT / ".venv-aiaa-protbind/bin/mk_prepare_ligand.py").resolve()
        ),
        "vina_version": vina.VINA_VERSION,
        "meeko_version": vina.MEEKO_VERSION,
        "rdkit_version": _version("rdkit"),
        "gemmi_version": _version("gemmi"),
        "numpy_version": _version("numpy"),
        "scipy_version": _version("scipy"),
        "scoring": "vina",
        "cpu": 1,
        "exhaustiveness": 8,
        "num_modes": 1,
        "energy_range": 3.0,
        "command_timeout_seconds": 900.0,
    }
    attestation = vina.runtime_asset_attestation(parameters)
    assets = str(attestation["runtime_assets_sha256"])
    lock_sha256 = sha256_file(environment_lock)
    base_revision = quick.base_model_revision(parameters)
    report = {
        "schema_version": "1.0",
        "environment_lock": {
            "name": environment_lock.name,
            "sha256": lock_sha256,
            "size_bytes": environment_lock.stat().st_size,
        },
        "runtime_attestation": attestation,
        "parameter_profile": {
            name: value
            for name, value in parameters.items()
            if not name.endswith("_executable")
        },
        "full_vina_provenance": {
            "model_revision": base_revision,
            "weight_sha256": assets,
            "code_sha256": vina.composite_code_sha256(lock_sha256, assets),
        },
        "quick_vina_provenance": {
            "model_revision": quick.quick_model_revision(parameters),
            "weight_sha256": assets,
            "code_sha256": quick.composite_code_sha256(lock_sha256, assets),
            "purpose": "selection-pruning-only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
