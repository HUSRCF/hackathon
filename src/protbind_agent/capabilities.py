"""Read-only capability discovery for optional scientific runtimes."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radeon_agent.hardware import probe_hardware

from .external_predictors import drutai_admission_report
from .openfold_contract import OFFICIAL_CHECKPOINT_SIZES
from .privacy import redact_text

_BWRAP_PROBE_TIMEOUT_SECONDS = 2.0
_BWRAP_PROBE_DIAGNOSTIC_LIMIT = 512
_BUNDLED_EXECUTABLES = {
    "vina": Path(__file__).resolve().parents[2] / "tools" / "bin" / "vina",
}


@dataclass(frozen=True, slots=True)
class Capability:
    name: str
    available: bool
    version: str | None
    purpose: str
    required_for: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version,
            "purpose": self.purpose,
            "required_for": self.required_for,
        }


_PYTHON_CAPABILITIES = {
    "rdkit": ("rdkit", "structure/chemistry standardization and conformers", "index, screening"),
    "gemmi": ("gemmi", "PDB/mmCIF parsing and structural checks", "input validation"),
    "pdbfixer": ("pdbfixer", "conservative structure repair", "receptor preparation"),
    "meeko": ("meeko", "PDBQT preparation", "docking"),
    "posebusters": ("posebusters", "chemical and geometric pose gates", "validation"),
    "prolif": ("prolif", "interaction fingerprints", "validation"),
    "spyrmsd": ("spyrmsd", "symmetry-aware RMSD", "validation"),
    "openmm": ("openmm", "HIP minimization/stability checks", "validation"),
    "seekdb": ("pyseekdb", "authoritative structured/full-text/vector state", "knowledge"),
    "powermem": ("powermem", "workflow preference and failure memory", "memory"),
    "flagembedding": ("FlagEmbedding", "offline BGE-M3 dense embeddings", "knowledge"),
    "pymupdf": ("pymupdf", "PDF extraction with page citations", "knowledge"),
    "fastapi": ("fastapi", "local web application", "serve"),
}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def discover_capabilities() -> list[Capability]:
    result: list[Capability] = []
    for name, (module, purpose, required_for) in _PYTHON_CAPABILITIES.items():
        available = importlib.util.find_spec(module) is not None
        result.append(
            Capability(
                name=name,
                available=available,
                version=_package_version(
                    {
                        "seekdb": "pyseekdb",
                        "flagembedding": "FlagEmbedding",
                        "pymupdf": "PyMuPDF",
                    }.get(name, name)
                )
                if available
                else None,
                purpose=purpose,
                required_for=required_for,
            )
        )
    for name, executable, purpose, required_for in (
        ("vina", "vina", "reproducible CPU docking", "docking"),
        ("fpocket", "fpocket", "candidate pocket detection", "ligand_only"),
        ("p2rank", "prank", "candidate pocket detection", "ligand_only"),
        (
            "drutai",
            "drutai.predict",
            "optional sequence-SMILES DTI concordance annotation",
            "annotation",
        ),
        ("hipcc", "hipcc", "TriPharm HIP build", "hip-screening"),
        (
            "mmseqs",
            "mmseqs",
            "local protein homology search and sequence clustering",
            "homology",
        ),
    ):
        path = shutil.which(executable)
        source = "path"
        if path is None:
            bundled = _BUNDLED_EXECUTABLES.get(name)
            if bundled is not None and bundled.is_file() and os.access(bundled, os.X_OK):
                path = str(bundled)
                source = "bundled"
        result.append(
            Capability(
                name=name,
                available=path is not None,
                version=f"{source}-executable-found" if path else None,
                purpose=purpose,
                required_for=required_for,
            )
        )
    return sorted(result, key=lambda item: item.name)


def _sanitize_probe_diagnostic(value: object) -> str:
    """Return a bounded, single-line diagnostic safe for a doctor report."""

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif value is None:
        text = ""
    else:
        text = str(value)
    text = redact_text(text)
    printable = "".join(
        character if character.isprintable() else " " for character in text
    )
    text = " ".join(printable.split())
    if len(text) > _BWRAP_PROBE_DIAGNOSTIC_LIMIT:
        text = text[-_BWRAP_PROBE_DIAGNOSTIC_LIMIT:]
    return text


def _bubblewrap_network_isolation_preflight() -> dict[str, Any]:
    """Test whether the host can create the namespace used by production workers.

    The probe contains no request, artifact, path from the caller, or inherited
    environment.  Any ambiguity is reported as unusable; application-level
    offline flags are deliberately not considered an isolation mechanism.
    """

    executable = shutil.which("bwrap")
    base = {
        "mechanism": "bubblewrap-unshare-net",
        "executable_present": executable is not None,
        "probe_performed": False,
        "probe_timeout_seconds": _BWRAP_PROBE_TIMEOUT_SECONDS,
        "probe_return_code": None,
        "os_network_isolation_usable": False,
        "application_offline_env_is_os_isolation": False,
    }
    if executable is None:
        return {
            **base,
            "status": "missing",
            "reason": "bubblewrap executable was not found",
        }

    # Mirror the security-relevant namespace and root/device/proc mounts used by
    # WorkerRunner, while avoiding its workspace bind, cwd, environment, stdin,
    # and all user data.  A zero exit status attests namespace construction only.
    command = (
        executable,
        "--die-with-parent",
        "--unshare-net",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--chdir",
        "/",
        "--",
        "/bin/true",
    )
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=_BWRAP_PROBE_TIMEOUT_SECONDS,
            check=False,
            close_fds=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        return {
            **base,
            "probe_performed": True,
            "status": "present_but_unusable",
            "reason": (
                "bubblewrap namespace probe exceeded its "
                f"{_BWRAP_PROBE_TIMEOUT_SECONDS:g}-second timeout"
            ),
        }
    except Exception as exc:
        detail = _sanitize_probe_diagnostic(f"{type(exc).__name__}: {exc}")
        return {
            **base,
            "probe_performed": True,
            "status": "present_but_unusable",
            "reason": detail or "bubblewrap namespace probe could not be executed",
        }

    raw_return_code = getattr(completed, "returncode", None)
    if not isinstance(raw_return_code, int) or isinstance(raw_return_code, bool):
        return {
            **base,
            "probe_performed": True,
            "status": "present_but_unusable",
            "reason": "bubblewrap namespace probe returned an invalid status",
        }
    if raw_return_code != 0:
        detail = _sanitize_probe_diagnostic(
            getattr(completed, "stderr", None)
            or getattr(completed, "stdout", None)
        )
        reason = f"bubblewrap namespace probe exited with status {raw_return_code}"
        if detail:
            reason = f"{reason}: {detail}"
        reason = _sanitize_probe_diagnostic(reason)
        return {
            **base,
            "probe_performed": True,
            "probe_return_code": raw_return_code,
            "status": "present_but_unusable",
            "reason": reason,
        }
    return {
        **base,
        "probe_performed": True,
        "probe_return_code": 0,
        "os_network_isolation_usable": True,
        "status": "usable",
        "reason": "bubblewrap created the production-style unshared network namespace",
    }


def _openfold_resource_policy(device_count: int) -> dict[str, Any]:
    """Describe the non-oversubscribing default for 0/1/2/4-GPU hosts."""

    openfold_device = "0" if device_count else None
    other_tool_devices = [str(index) for index in range(1, device_count)]
    return {
        "detected_gpu_count": device_count,
        "openfold_profile": "openfold3-p2-155k/low_mem/rocm-triton/fp32",
        "openfold_devices_per_job": 1,
        "max_concurrent_openfold_jobs_default": 1,
        "minimum_free_vram_gib_default": 28.0,
        "openfold_visible_device_default": openfold_device,
        "reserved_interactive_device": (
            other_tool_devices[0] if other_tool_devices else None
        ),
        "reserved_other_tool_devices": other_tool_devices,
        "lease_scope": "same-user ProtBind workers across host workspaces",
        "external_services_coordinated": False,
        "single_gpu_policy": (
            "pause GPU LLM/OpenMM while OpenFold runs" if device_count == 1 else None
        ),
        "multi_gpu_semantics": (
            "multiple devices distribute independent queries; VRAM is not pooled"
        ),
        "checkpoint_policy": {
            "allowed": list(OFFICIAL_CHECKPOINT_SIZES),
            "size_bytes": dict(OFFICIAL_CHECKPOINT_SIZES),
            "has_small_memory_variant": False,
        },
    }


def _prediction_fallback_policy() -> dict[str, Any]:
    """Expose the fail-closed receptor/complex routing contract."""

    return {
        "receptor_precedence": [
            "user_structure",
            "local_exact_sequence_cache",
            "explicitly_approved_rcsb",
            "legacy_esmfold_v1",
        ],
        "complex_predictor_precedence": [
            "openfold3_after_checkpoint_gate",
            "esmfold2_after_three_complex_gate",
        ],
        "legacy_esmfold_v1_scope": "receptor_only_not_ligand_pose",
        "no_complex_predictor": (
            "explicit_degraded_state; retain Vina/validation evidence only and do not "
            "claim cofolding"
        ),
        "intercept_before_receptor_folding": True,
        "intercept_before_each_candidate_cofold": True,
    }


def doctor_report() -> dict[str, Any]:
    hardware = probe_hardware()
    hardware_value = hardware.to_dict()
    device_count = len(hardware_value.get("device_architectures", ()))
    network_isolation = _bubblewrap_network_isolation_preflight()
    openmm_platforms: list[str] = []
    if importlib.util.find_spec("openmm") is not None:
        try:
            import openmm

            openmm_platforms = [
                openmm.Platform.getPlatform(index).getName()
                for index in range(openmm.Platform.getNumPlatforms())
            ]
        except Exception:
            openmm_platforms = []
    return {
        "schema_version": "1.0",
        "offline_default": True,
        "hsa_override_forbidden": True,
        "hsa_override_active": bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION")),
        "hardware": hardware_value,
        "capabilities": [item.to_dict() for item in discover_capabilities()],
        "runtime_details": {
            "openmm_platforms": openmm_platforms,
            "openmm_hip_available": "HIP" in openmm_platforms,
            "worker_network_isolation": network_isolation,
            "drutai": drutai_admission_report(),
            "p2rank_semantics": (
                "candidate site hypotheses only; downstream receptor-frame, box, "
                "consensus, docking, and validation gates remain required"
            ),
        },
        "resource_policy": _openfold_resource_policy(device_count),
        "prediction_fallback_policy": _prediction_fallback_policy(),
        "scientific_boundary": {
            "llm_may_generate_scores": False,
            "vina_is_experimental_free_energy": False,
            "cofold_is_binding_fact": False,
        },
    }
