#!/usr/bin/env python3
"""Write a privacy-safe, machine-readable audit of the AIAA ProtBind runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_VINA_SHA256 = "f31f774f723bba7bbe6e9d1c47577020eea9a8da16424284c043d22593570644"
PACKAGES = (
    "torch",
    "rdkit",
    "gemmi",
    "openmm",
    "pdbfixer",
    "meeko",
    "vina",
    "posebusters",
    "prolif",
    "spyrmsd",
    "pyseekdb",
    "FlagEmbedding",
    "PyMuPDF",
    "fastapi",
    "uvicorn",
    "fair-esm",
)
MODULES = (
    "torch",
    "rdkit",
    "gemmi",
    "openmm",
    "pdbfixer",
    "meeko",
    "vina",
    "posebusters",
    "prolif",
    "spyrmsd",
    "pyseekdb",
    "FlagEmbedding",
    "pymupdf",
)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_report() -> dict[str, Any]:
    try:
        import torch

        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_vram_bytes": properties.total_memory,
                }
            )
        return {
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "device_count": len(devices),
            "devices": devices,
        }
    except Exception as error:  # pragma: no cover - diagnostic boundary
        return {"device_count": 0, "devices": [], "error": type(error).__name__}


def _openmm_platforms() -> list[str]:
    try:
        import openmm

        return [
            openmm.Platform.getPlatform(index).getName()
            for index in range(openmm.Platform.getNumPlatforms())
        ]
    except Exception:  # pragma: no cover - diagnostic boundary
        return []


def build_report(repo: Path) -> dict[str, Any]:
    vina = repo / "tools" / "bin" / "vina"
    package_versions = {name: _version(name) for name in PACKAGES}
    canonical_packages = json.dumps(
        package_versions, sort_keys=True, separators=(",", ":")
    ).encode()
    # Initialize ROCm through PyTorch before importing OpenMM.  On this AIAA build,
    # querying OpenMM's OpenCL platform first makes the same process report zero
    # ROCm devices even though both cards remain healthy.  Production folds and
    # validation already use separate workers; the audit mirrors that safe order.
    gpu = _gpu_report()
    openmm_platforms = _openmm_platforms()
    return {
        "schema_version": "1.0",
        "environment": "AIAA + workspace overlay",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "offline_defaults": {
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "tensorflow_disabled": os.environ.get("USE_TF") == "0",
            "user_site_disabled": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "hsa_override_active": bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION")),
        "packages": package_versions,
        "package_manifest_sha256": hashlib.sha256(canonical_packages).hexdigest(),
        "modules_available": {
            name: importlib.util.find_spec(name) is not None for name in MODULES
        },
        "executables": {
            "hipcc": shutil.which("hipcc") is not None,
            "vina": {
                "available": vina.is_file() and os.access(vina, os.X_OK),
                "version": "1.2.7" if vina.is_file() else None,
                "sha256": _sha256(vina) if vina.is_file() else None,
                "matches_expected_sha256": (
                    _sha256(vina) == EXPECTED_VINA_SHA256 if vina.is_file() else False
                ),
            },
        },
        "gpu": gpu,
        "openmm": {
            "platforms": openmm_platforms,
            "hip_available": "HIP" in openmm_platforms,
            "process_isolation_required_from_rocm_torch": True,
        },
        "resource_policy": {
            "openfold_environment": "dedicated AIAA-backed overlay; inherits AIAA Torch/Triton",
            "openfold_checkpoint": "openfold3-p2-155k",
            "openfold_checkpoint_size_bytes": 2_287_928_196,
            "openfold_has_small_memory_variant": False,
            "openfold_devices_per_job": 1,
            "openfold_max_concurrency": 1,
            "openfold_gpu": 0,
            "interactive_gpu": 1,
            "minimum_free_vram_gib": 28.0,
        },
        "prediction_fallback_policy": {
            "receptor_precedence": [
                "user_structure",
                "local_exact_sequence_cache",
                "explicitly_approved_rcsb",
                "legacy_esmfold_v1",
            ],
            "legacy_esmfold_v1_scope": "receptor_only_not_ligand_pose",
            "complex_predictor_precedence": [
                "openfold3_after_checkpoint_gate",
                "esmfold2_after_three_complex_gate",
            ],
            "no_complex_predictor": "explicit degraded state; never claim cofolding",
        },
        "deferred": {
            "powermem": "separate optional environment; core install would downgrade NumPy",
            "openmm_hip": "not exposed by the current AIAA OpenMM build",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    report = build_report(repo)
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
