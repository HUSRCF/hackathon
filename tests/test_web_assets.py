from __future__ import annotations

import json

import pytest

from protbind_agent import web_assets
from protbind_agent.artifacts import sha256_bytes


def test_network_asset_install_requires_exact_domain(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        web_assets,
        "_download",
        lambda *_args, **_kwargs: pytest.fail("download must not start"),
    )

    with pytest.raises(PermissionError, match="explicitly approved"):
        web_assets.install_3dmol_asset(tmp_path)


def test_reviewed_local_asset_install_is_hash_bound(tmp_path, monkeypatch) -> None:
    javascript = b"window.$3Dmol={version:'test'};\n"
    license_text = b"test BSD-3-Clause license\n"
    javascript_file = tmp_path / "reviewed.js"
    license_file = tmp_path / "reviewed.LICENSE"
    javascript_file.write_bytes(javascript)
    license_file.write_bytes(license_text)
    monkeypatch.setattr(
        web_assets,
        "THREEDMOL_SHA256",
        sha256_bytes(javascript),
    )
    monkeypatch.setattr(
        web_assets,
        "THREEDMOL_LICENSE_SHA256",
        sha256_bytes(license_text),
    )

    manifest = web_assets.install_3dmol_asset(
        tmp_path / "workspace",
        javascript_file=javascript_file,
        license_file=license_file,
    )
    status = web_assets.installed_3dmol_status(tmp_path / "workspace")

    assert manifest["runtime_network_required"] is False
    assert manifest["installation_source"] == "reviewed-local-files"
    assert status["verified"] is True
    stored = json.loads(
        (tmp_path / "workspace/static/3Dmol.asset.json").read_text(encoding="utf-8")
    )
    assert stored["javascript"]["sha256"] == sha256_bytes(javascript)


def test_asset_tampering_disables_viewer(tmp_path, monkeypatch) -> None:
    javascript = b"verified javascript"
    license_text = b"verified license"
    javascript_file = tmp_path / "reviewed.js"
    license_file = tmp_path / "reviewed.LICENSE"
    javascript_file.write_bytes(javascript)
    license_file.write_bytes(license_text)
    monkeypatch.setattr(web_assets, "THREEDMOL_SHA256", sha256_bytes(javascript))
    monkeypatch.setattr(
        web_assets,
        "THREEDMOL_LICENSE_SHA256",
        sha256_bytes(license_text),
    )
    workspace = tmp_path / "workspace"
    web_assets.install_3dmol_asset(
        workspace,
        javascript_file=javascript_file,
        license_file=license_file,
    )

    (workspace / "static/3Dmol-min.js").write_bytes(b"tampered")

    assert web_assets.installed_3dmol_status(workspace)["verified"] is False
