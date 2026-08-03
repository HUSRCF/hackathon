from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

import protbind_agent.drutai as drutai_module
from protbind_agent.drutai import (
    DRUTAI_LICENSE_ACKNOWLEDGEMENT,
    DrutAIManager,
    DrutAIModelSpec,
)
from protbind_agent.models import ArtifactRef
from protbind_agent.public_data import CurlResult


def _blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _install_fixture_model(monkeypatch: pytest.MonkeyPatch) -> tuple[str, bytes]:
    data = b"\x08synthetic-onnx-fixture"
    spec = DrutAIModelSpec(
        name="fixture",
        filename="fixture.onnx",
        size_bytes=len(data),
        git_blob_sha1=_blob_sha1(data),
    )
    monkeypatch.setitem(drutai_module.DRUTAI_MODELS, "fixture", spec)
    return spec.name, data


class FakeTransport:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.calls = []

    def request(self, url: str, *, accept: str, max_bytes: int) -> CurlResult:
        self.calls.append((url, accept, max_bytes))
        return CurlResult(
            data=self.data,
            url=url,
            http_status=200,
            content_type="application/octet-stream",
            elapsed_seconds=0.01,
            size_download=len(self.data),
            curl_version="curl test",
        )


def _private_inputs(root: Path, *, smiles: str = "CCO") -> tuple[Path, Path]:
    input_tsv = root / "input.tsv"
    input_tsv.write_text(
        "sm\ttarget\tsmile\ncompound-1\tTARGET1\t" + smiles + "\n",
        encoding="utf-8",
    )
    fasta_directory = root / "fasta"
    fasta_directory.mkdir()
    (fasta_directory / "TARGET1.fasta").write_text(
        ">TARGET1\nACDEFGHIKLMNPQRSTVWY\n",
        encoding="utf-8",
    )
    return input_tsv, fasta_directory


def test_drutai_acquisition_requires_exact_license_and_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, data = _install_fixture_model(monkeypatch)
    transport = FakeTransport(data)
    manager = DrutAIManager(tmp_path / "workspace", transport=transport)

    with pytest.raises(PermissionError, match="GPL-3.0-only acknowledgement"):
        manager.acquire_model(
            model=model,
            approved_domain="raw.githubusercontent.com",
            license_acknowledgement="MIT",
        )
    with pytest.raises(PermissionError, match="not explicitly approved"):
        manager.acquire_model(
            model=model,
            approved_domain="github.com",
            license_acknowledgement=DRUTAI_LICENSE_ACKNOWLEDGEMENT,
        )

    assert transport.calls == []


def test_drutai_acquisition_pins_git_blob_and_writes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, data = _install_fixture_model(monkeypatch)
    transport = FakeTransport(data)
    manager = DrutAIManager(tmp_path / "workspace", transport=transport)

    result = manager.acquire_model(
        model=model,
        approved_domain="raw.githubusercontent.com",
        license_acknowledgement=DRUTAI_LICENSE_ACKNOWLEDGEMENT,
    )

    assert result["status"] == "ACQUIRED"
    assert result["license"] == "GPL-3.0-only"
    assert result["distributed_by_protbind"] is False
    assert result["upstream_code_copied_into_protbind"] is False
    assert result["observed_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["hard_filter_allowed"] is False
    receipt = ArtifactRef.from_dict(result["receipt_artifact"])
    assert manager.artifacts.resolve(receipt).is_file()

    repeated = manager.acquire_model(
        model=model,
        approved_domain="raw.githubusercontent.com",
        license_acknowledgement=DRUTAI_LICENSE_ACKNOWLEDGEMENT,
    )
    assert repeated["status"] == "PRESENT"
    assert repeated["network_request_performed"] is False
    assert len(transport.calls) == 1


def test_drutai_rejects_model_bytes_outside_pinned_git_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _data = _install_fixture_model(monkeypatch)
    manager = DrutAIManager(
        tmp_path / "workspace",
        transport=FakeTransport(b"\x08wrong-model"),
    )

    with pytest.raises(ValueError, match="size differs|commitment mismatch"):
        manager.acquire_model(
            model=model,
            approved_domain="raw.githubusercontent.com",
            license_acknowledgement=DRUTAI_LICENSE_ACKNOWLEDGEMENT,
        )


def test_drutai_invalid_smiles_fails_before_external_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rdkit")
    model, data = _install_fixture_model(monkeypatch)
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("worker must not run")

    manager = DrutAIManager(
        tmp_path / "workspace",
        transport=FakeTransport(data),
        runner=runner,
        executable="/opt/drutai.predict",
    )
    manager.acquire_model(
        model=model,
        approved_domain="raw.githubusercontent.com",
        license_acknowledgement=DRUTAI_LICENSE_ACKNOWLEDGEMENT,
    )
    input_tsv, fasta_directory = _private_inputs(tmp_path, smiles="not-a-smiles")

    with pytest.raises(ValueError, match="invalid SMILES"):
        manager.annotate(
            input_tsv=input_tsv,
            fasta_directory=fasta_directory,
            model=model,
            data_access_confirmed=True,
        )
    assert calls == []


def test_drutai_external_tsv_annotation_is_non_decisional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rdkit")
    model, data = _install_fixture_model(monkeypatch)
    input_tsv, fasta_directory = _private_inputs(tmp_path)

    def runner(command, **_kwargs):
        output = Path(command[command.index("-o") + 1])
        with input_tsv.open(newline="", encoding="utf-8") as source:
            row = next(csv.DictReader(source, delimiter="\t"))
        output.write_text(
            "sm\ttarget\tsmile\tprob_inter\tpred_type\n"
            f"{row['sm']}\t{row['target']}\t{row['smile']}\t0.91\tInteraction\n",
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    manager = DrutAIManager(
        tmp_path / "workspace",
        transport=FakeTransport(data),
        runner=runner,
        executable="/opt/drutai.predict",
    )
    manager.acquire_model(
        model=model,
        approved_domain="raw.githubusercontent.com",
        license_acknowledgement=DRUTAI_LICENSE_ACKNOWLEDGEMENT,
    )
    monkeypatch.setattr(
        drutai_module.shutil,
        "which",
        lambda executable: (
            "/usr/bin/bwrap" if executable == "bwrap" else "/opt/drutai.predict"
        ),
    )

    result = manager.annotate(
        input_tsv=input_tsv,
        fasta_directory=fasta_directory,
        model=model,
        data_access_confirmed=True,
    )

    assert result["concordance_counts"] == {
        "SUPPORTIVE": 1,
        "DISCORDANT": 0,
        "ABSTAIN": 0,
    }
    assert result["hard_filter_allowed"] is False
    bundle = manager.artifacts.read_json(ArtifactRef.from_dict(result["bundle_artifact"]))
    assert bundle["annotations"][0]["decision_eligible"] is False
    assert bundle["annotations"][0]["concordance"] == "SUPPORTIVE"
    assert bundle["input"]["raw_sequence_in_receipt"] is False
    assert bundle["input"]["raw_smiles_in_receipt"] is False
    assert bundle["evidence_grade_upgrade_allowed"] is False


def test_drutai_snap_uses_verified_strict_confinement_without_nested_bwrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rdkit")
    model, data = _install_fixture_model(monkeypatch)
    input_tsv, fasta_directory = _private_inputs(tmp_path)
    worker_commands = []
    info_output = """
notes:
  confinement: strict
  devmode: false
  trymode: false
  enabled: true
  broken: false
"""
    connections_output = "Interface Plug Slot Notes\nhome drutai:home :home -\n"

    def worker_runner(command, **_kwargs):
        worker_commands.append(command)
        output = Path(command[command.index("-o") + 1])
        output.write_text(
            "sm\ttarget\tsmile\tprob_inter\tpred_type\n"
            "compound-1\tTARGET1\tCCO\t0.49\tNon-interaction\n",
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def control_runner(command, **_kwargs):
        stdout = info_output if "info" in command else connections_output
        return type(
            "Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""}
        )()

    manager = DrutAIManager(
        tmp_path / "workspace",
        transport=FakeTransport(data),
        runner=worker_runner,
        control_runner=control_runner,
    )
    manager.acquire_model(
        model=model,
        approved_domain="raw.githubusercontent.com",
        license_acknowledgement=DRUTAI_LICENSE_ACKNOWLEDGEMENT,
    )
    monkeypatch.setattr(
        drutai_module.shutil,
        "which",
        lambda executable: (
            "/snap/bin/drutai.predict"
            if executable == "drutai.predict"
            else "/usr/bin/snap"
        ),
    )

    result = manager.annotate(
        input_tsv=input_tsv,
        fasta_directory=fasta_directory,
        model=model,
        data_access_confirmed=True,
    )

    assert worker_commands[0][0] == "/snap/bin/drutai.predict"
    bundle = manager.artifacts.read_json(ArtifactRef.from_dict(result["bundle_artifact"]))
    isolation = bundle["execution"]["network_isolation"]
    assert isolation["mode"] == "snap-strict-confinement"
    assert isolation["verified"] is True
    assert isolation["snap_instance"] == "drutai"
    assert isolation["connected_network_interfaces"] == []
    assert isolation["snap_info_sha256"] == hashlib.sha256(
        info_output.encode()
    ).hexdigest()
    assert isolation["snap_connections_sha256"] == hashlib.sha256(
        connections_output.encode()
    ).hexdigest()


def test_drutai_snap_fails_closed_with_connected_network_interface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rdkit")
    model, data = _install_fixture_model(monkeypatch)
    input_tsv, fasta_directory = _private_inputs(tmp_path)

    def control_runner(command, **_kwargs):
        if "info" in command:
            stdout = """
confinement: strict
devmode: false
trymode: false
enabled: true
broken: false
"""
        else:
            stdout = "Interface Plug Slot Notes\nnetwork drutai:network :network -\n"
        return type(
            "Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""}
        )()

    manager = DrutAIManager(
        tmp_path / "workspace",
        transport=FakeTransport(data),
        runner=lambda *_args, **_kwargs: pytest.fail("worker must not run"),
        control_runner=control_runner,
    )
    manager.acquire_model(
        model=model,
        approved_domain="raw.githubusercontent.com",
        license_acknowledgement=DRUTAI_LICENSE_ACKNOWLEDGEMENT,
    )
    monkeypatch.setattr(
        drutai_module.shutil,
        "which",
        lambda executable: (
            "/snap/bin/drutai.predict"
            if executable == "drutai.predict"
            else "/usr/bin/snap"
        ),
    )

    with pytest.raises(RuntimeError, match="connected network interface"):
        manager.annotate(
            input_tsv=input_tsv,
            fasta_directory=fasta_directory,
            model=model,
            data_access_confirmed=True,
        )
