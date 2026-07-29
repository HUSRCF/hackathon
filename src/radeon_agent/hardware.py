"""ROCm/Radeon hardware discovery for reproducible benchmark manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

_GFX_PATTERN = re.compile(r"\bgfx[0-9a-f]+\b", re.IGNORECASE)
_DEVICE_NAME_PATTERN = re.compile(r"^\s*Name:\s*(gfx[0-9a-f]+)\s*$", re.MULTILINE)
_SMI_GFX_PATTERN = re.compile(r"GFX\s+Version:\s*(gfx[0-9a-f]+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    command: tuple[str, ...]
    available: bool
    return_code: int | None
    output_sha256: str | None
    relevant_output: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class HardwareManifest:
    captured_at: str
    host_fingerprint: str
    platform: str
    python_version: str
    rocm_version: str | None
    device_architectures: tuple[str, ...]
    architectures: tuple[str, ...]
    competition_roles: tuple[str, ...]
    hsa_override_active: bool
    evidence: tuple[CommandEvidence, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def extract_gfx_arches(text: str, *, unique: bool = True) -> tuple[str, ...]:
    values = [match.lower() for match in _GFX_PATTERN.findall(text)]
    if not unique:
        return tuple(values)
    return tuple(dict.fromkeys(values))


def extract_device_arches(rocminfo_text: str) -> tuple[str, ...]:
    """Extract one ISA name per GPU from rocminfo or rocm-smi output."""

    rocminfo_values = _DEVICE_NAME_PATTERN.findall(rocminfo_text)
    values = rocminfo_values or _SMI_GFX_PATTERN.findall(rocminfo_text)
    return tuple(match.lower() for match in values)


def competition_role(architecture: str) -> str:
    roles = {
        "gfx1100": "primary-radeon-rdna3",
        "gfx1201": "cross-verifier-radeon-rdna4",
        "gfx90a": "development-only-instinct-cdna2",
    }
    return roles.get(architecture.lower(), "unclassified")


def _relevant_lines(text: str, limit: int = 80) -> tuple[str, ...]:
    needles = (
        "gfx",
        "card series",
        "card model",
        "product name",
        "vram",
        "driver version",
        "rocm version",
        "compute unit",
    )
    lines = [line.strip() for line in text.splitlines() if any(n in line.lower() for n in needles)]
    return tuple(lines[:limit])


def _capture(
    command: tuple[str, ...], timeout_seconds: float = 10.0
) -> tuple[CommandEvidence, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return (
            CommandEvidence(command, False, None, None, (), "executable not found"),
            "",
        )
    resolved = (executable, *command[1:])
    try:
        process = subprocess.run(
            resolved,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = f"{process.stdout}\n{process.stderr}".strip()
        evidence = CommandEvidence(
            command=command,
            available=True,
            return_code=process.returncode,
            output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            relevant_output=_relevant_lines(output),
            error=None if process.returncode == 0 else "command returned non-zero",
        )
        return evidence, output
    except (subprocess.TimeoutExpired, OSError) as exc:
        return CommandEvidence(command, True, None, None, (), str(exc)), ""


def _rocm_version() -> str | None:
    candidates = (
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/.info/version-dev"),
    )
    for candidate in candidates:
        try:
            if candidate.is_file():
                value = candidate.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            continue
    return None


def probe_hardware() -> HardwareManifest:
    commands = (
        ("rocminfo",),
        ("amd-smi", "static", "--json"),
        ("rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showdriverversion"),
    )
    evidence: list[CommandEvidence] = []
    outputs: list[str] = []
    rocminfo_output = ""
    for command in commands:
        item, output = _capture(command)
        evidence.append(item)
        outputs.append(output)
        if command[0] == "rocminfo":
            rocminfo_output = output

    combined_output = "\n".join(outputs)
    device_arches = extract_device_arches(rocminfo_output)
    if not device_arches:
        device_arches = extract_device_arches(combined_output)
    architectures = extract_gfx_arches(combined_output)
    if not device_arches and architectures:
        device_arches = architectures
    return HardwareManifest(
        captured_at=datetime.now(UTC).isoformat(),
        host_fingerprint=hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16],
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        rocm_version=_rocm_version(),
        device_architectures=device_arches,
        architectures=architectures,
        competition_roles=tuple(competition_role(arch) for arch in architectures),
        hsa_override_active=bool(os.getenv("HSA_OVERRIDE_GFX_VERSION")),
        evidence=tuple(evidence),
    )


def hardware_json(*, indent: int = 2) -> str:
    return json.dumps(probe_hardware().to_dict(), ensure_ascii=False, indent=indent)
