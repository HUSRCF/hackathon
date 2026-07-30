"""Pinned, explicitly approved offline assets for the loopback research UI."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .artifacts import sha256_bytes, sha256_file
from .privacy import require_network_approval

THREEDMOL_VERSION = "2.5.4"
THREEDMOL_HOST = "cdn.jsdelivr.net"
THREEDMOL_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js"
THREEDMOL_SHA256 = "1297081865a4d6c0b2ac22d3e909724da8c03ba0caf7bfc78c8a3d9d8b143f4e"
THREEDMOL_LICENSE_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.5.4/LICENSE"
THREEDMOL_LICENSE_SHA256 = (
    "4c6eaaed856f3f28a3b1a98e74f4a8a71618de7d51ea4155c29f6f793bcef861"
)
_MAX_ASSET_BYTES = 2 * 1024 * 1024
_MAX_LICENSE_BYTES = 128 * 1024


class _ExactHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != THREEDMOL_HOST:
            raise PermissionError("3Dmol.js download redirected outside the approved host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url: str, *, max_bytes: int) -> bytes:
    opener = urllib.request.build_opener(_ExactHostRedirectHandler())
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ProtBind/0.1 offline-asset-installer"},
    )
    with opener.open(request, timeout=60) as response:
        final = urlsplit(response.geturl())
        if final.scheme != "https" or final.hostname != THREEDMOL_HOST:
            raise PermissionError("3Dmol.js response came from an unapproved host")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("3Dmol.js asset exceeded the pinned size limit")
    return data


def _verified(data: bytes, expected_sha256: str, label: str) -> bytes:
    actual = sha256_bytes(data)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return data


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_3dmol_asset(
    workspace: Path,
    *,
    approved_domains: tuple[str, ...] = (),
    javascript_file: Path | None = None,
    license_file: Path | None = None,
) -> dict[str, Any]:
    """Install an exact 3Dmol.js build from approved HTTPS or reviewed local files."""

    if (javascript_file is None) != (license_file is None):
        raise ValueError("--from-file and --license-file must be provided together")
    source_mode = "reviewed-local-files"
    if javascript_file is None:
        require_network_approval(THREEDMOL_URL, approved_domains)
        require_network_approval(THREEDMOL_LICENSE_URL, approved_domains)
        javascript = _download(THREEDMOL_URL, max_bytes=_MAX_ASSET_BYTES)
        license_text = _download(
            THREEDMOL_LICENSE_URL,
            max_bytes=_MAX_LICENSE_BYTES,
        )
        source_mode = "explicitly-approved-network"
    else:
        assert license_file is not None
        javascript = javascript_file.read_bytes()
        license_text = license_file.read_bytes()

    _verified(javascript, THREEDMOL_SHA256, "3Dmol-min.js")
    _verified(license_text, THREEDMOL_LICENSE_SHA256, "3Dmol.js LICENSE")
    static = workspace.resolve() / "static"
    asset_path = static / "3Dmol-min.js"
    license_path = static / "LICENSE.3Dmol.txt"
    manifest_path = static / "3Dmol.asset.json"
    manifest = {
        "schema_version": "1.0",
        "kind": "protbind.web-asset",
        "name": "3Dmol.js",
        "version": THREEDMOL_VERSION,
        "javascript": {
            "filename": asset_path.name,
            "sha256": THREEDMOL_SHA256,
            "source": THREEDMOL_URL,
            "size_bytes": len(javascript),
        },
        "license": {
            "filename": license_path.name,
            "sha256": THREEDMOL_LICENSE_SHA256,
            "source": THREEDMOL_LICENSE_URL,
            "spdx": "BSD-3-Clause",
        },
        "installation_source": source_mode,
        "runtime_network_required": False,
    }
    _atomic_write(asset_path, javascript)
    _atomic_write(license_path, license_text)
    _atomic_write(
        manifest_path,
        (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8"),
    )
    return manifest


def installed_3dmol_status(workspace: Path) -> dict[str, Any]:
    """Verify that the complete pinned asset set is present and untampered."""

    static = workspace.resolve() / "static"
    asset_path = static / "3Dmol-min.js"
    license_path = static / "LICENSE.3Dmol.txt"
    manifest_path = static / "3Dmol.asset.json"
    if not all(path.is_file() for path in (asset_path, license_path, manifest_path)):
        return {
            "installed": False,
            "verified": False,
            "reason": "one or more pinned 3Dmol.js asset files are missing",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "installed": True,
            "verified": False,
            "reason": "3Dmol.js asset manifest is unreadable",
        }
    valid_manifest = (
        isinstance(manifest, dict)
        and manifest.get("kind") == "protbind.web-asset"
        and manifest.get("version") == THREEDMOL_VERSION
        and isinstance(manifest.get("javascript"), dict)
        and manifest["javascript"].get("sha256") == THREEDMOL_SHA256
        and isinstance(manifest.get("license"), dict)
        and manifest["license"].get("sha256") == THREEDMOL_LICENSE_SHA256
    )
    if not valid_manifest:
        return {
            "installed": True,
            "verified": False,
            "reason": "3Dmol.js asset manifest does not match the pinned specification",
        }
    try:
        hashes_match = (
            sha256_file(asset_path) == THREEDMOL_SHA256
            and sha256_file(license_path) == THREEDMOL_LICENSE_SHA256
        )
    except OSError:
        hashes_match = False
    if not hashes_match:
        return {
            "installed": True,
            "verified": False,
            "reason": "3Dmol.js asset bytes failed SHA-256 verification",
        }
    return {
        "installed": True,
        "verified": True,
        "reason": None,
        "version": THREEDMOL_VERSION,
        "sha256": THREEDMOL_SHA256,
    }
