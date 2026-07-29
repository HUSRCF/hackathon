#!/usr/bin/env python3
"""Write a path-free audit for the AIAA-backed OpenFold3 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

# Direct execution puts ``scripts/`` rather than the repository root on
# sys.path. Add only the two reviewed local import roots; the wrapper still
# selects every third-party dependency from the AIAA-backed overlay.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

_runtime_attestation = import_module(
    "workers.openfold3_worker"
)._runtime_attestation


def _validator() -> dict[str, Any]:
    completed = subprocess.run(
        ("validate-openfold3-rocm",),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = (completed.stdout + completed.stderr).encode()
    return {
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }


def _gpu() -> dict[str, Any]:
    import torch

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "visible_index": index,
                "name": properties.name,
                "total_vram_bytes": properties.total_memory,
            }
        )
    return {"visible_device_count": len(devices), "devices": devices}


def _torch_overlay_state() -> dict[str, Any]:
    import torch

    local_site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    local_entries = sorted(
        path.name
        for path in local_site.iterdir()
        if path.name == "torch"
        or path.name.startswith("torch-")
        or path.name.startswith("torch_")
    ) if local_site.is_dir() else []
    imported_from_overlay = Path(torch.__file__).resolve().is_relative_to(
        local_site.resolve()
    )
    return {
        "inherits_aiaa_torch": not imported_from_overlay,
        "duplicate_torch_installed_in_overlay": bool(local_entries),
        "local_torch_distribution_entry_count": len(local_entries),
    }


def build_report() -> dict[str, Any]:
    local_packages = (
        "openfold3",
        "setuptools-scm",
        "ml-collections",
        "pytorch-lightning",
        "pdbeccdutils",
        "kalign-python",
        "ijson",
        "memory_profiler",
        "func_timeout",
    )
    packages = {name: metadata.version(name) for name in local_packages}
    package_bytes = json.dumps(
        packages, sort_keys=True, separators=(",", ":")
    ).encode()
    torch_overlay = _torch_overlay_state()
    return {
        "schema_version": "1.0",
        "environment": "AIAA + dedicated OpenFold3 overlay",
        **torch_overlay,
        "hsa_override_active": bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION")),
        "selected_physical_gpu": os.environ.get("HIP_VISIBLE_DEVICES"),
        "runtime_attestation": _runtime_attestation(),
        "packages": packages,
        "package_manifest_sha256": hashlib.sha256(package_bytes).hexdigest(),
        "gpu": _gpu(),
        "official_rocm_validator": _validator(),
        "checkpoint": {
            "installed": False,
            "policy": "separate explicit import with source receipt and SHA-256",
        },
        "fallback_policy": {
            "receptor_only": "RCSB/local exact cache, then legacy ESMFold v1",
            "complex_prediction": (
                "OpenFold3 only after checkpoint gate; ESMFold2 only after "
                "3-complex gfx1100 gate"
            ),
            "no_complex_predictor": "degrade explicitly to Vina plus validation evidence",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
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
