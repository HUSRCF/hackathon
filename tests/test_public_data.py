from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from protbind_agent.public_data import (
    CurlTransport,
    PropkaAudit,
    PropkaAuditor,
    PublicDataFetcher,
    materialize_public_fetch,
    validate_public_output,
)

gemmi = pytest.importorskip("gemmi")
Chem = pytest.importorskip("rdkit.Chem")


def _mmcif(*, include_oxygen: bool = True) -> bytes:
    pdb = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00"
        "           N  \n"
        "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00"
        "           C  \n"
        "ATOM      3  C   ALA A   1       2.050   1.400   0.000  1.00 20.00"
        "           C  \n"
    )
    if include_oxygen:
        pdb += (
            "ATOM      4  O   ALA A   1       1.400   2.400   0.000  1.00 20.00"
            "           O  \n"
        )
    structure = gemmi.read_pdb_string(pdb + "END\n")
    structure.name = "TEST"
    return structure.make_mmcif_document().as_string().encode("utf-8")


def _sdf() -> bytes:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, (float(index), float(index % 2), 0.25 * index))
    molecule.AddConformer(conformer)
    return (Chem.MolToMolBlock(molecule) + "\n$$$$\n").encode("utf-8")


class FakeCurlRunner:
    def __init__(self, responses: dict[str, tuple[bytes, str]]) -> None:
        self.responses = responses
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        self.environments.append(dict(kwargs.get("env", {})))
        if "--version" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="curl 8.10.0 (test) OpenSSL/3\n",
                stderr="",
            )
        url = command[-1]
        data, content_type = self.responses[url]
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(data)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "http_code": 200,
                    "url_effective": url,
                    "content_type": content_type,
                    "size_download": len(data),
                }
            ),
            stderr="",
        )


class FakePropka:
    def audit(self, _data: bytes) -> PropkaAudit:
        output = b"PROPKA test report\n"
        return PropkaAudit(
            summary={
                "status": "SUCCEEDED",
                "scientific_gate": False,
                "output_size_bytes": len(output),
            },
            output=output,
            version="propka 3.5.1-test",
        )


def test_uniprot_fetch_uses_direct_curl_and_materializes_receipt(tmp_path) -> None:
    url = "https://rest.uniprot.org/uniprotkb/P69905.fasta"
    runner = FakeCurlRunner(
        {
            url: (
                b">sp|P69905|TEST\nACDEFGHIKLMNPQRSTVWY\n",
                "text/plain; charset=utf-8",
            )
        }
    )
    fetcher = PublicDataFetcher(
        tmp_path / "workspace",
        transport=CurlTransport(curl_binary="/usr/bin/curl", runner=runner),
        now=lambda: datetime(2026, 7, 30, tzinfo=UTC),
    )

    result = fetcher.fetch(
        source="uniprot-fasta",
        identifier="p69905",
        approved_domains=("rest.uniprot.org",),
    )
    materialized = materialize_public_fetch(
        result,
        fetcher.artifacts,
        project_root=tmp_path,
        output=Path("inputs/protein.fasta"),
    )

    transfer = runner.commands[-1]
    assert transfer[1] == "-q"
    assert "--location" not in transfer
    assert transfer[transfer.index("--noproxy") + 1] == "*"
    assert runner.environments[-1].get("HTTPS_PROXY") is None
    assert result.validation["v1_case_compatible"] is True
    assert (tmp_path / "inputs/protein.fasta").read_bytes().startswith(b">sp|")
    assert materialized["output"] == "inputs/protein.fasta"
    sidecar = json.loads(
        (tmp_path / "inputs/protein.fasta.protbind.json").read_text(encoding="utf-8")
    )
    assert sidecar["validation"]["sequence_length"] == 20
    assert sidecar["warnings"] == []
    receipt = fetcher.artifacts.read_json(result.receipt)
    assert receipt["privacy"]["sent_data"] == "public identifier only"
    assert receipt["privacy"]["private_sequence_uploaded"] is False


def test_fetch_rejects_unapproved_domain_before_curl_runs(tmp_path) -> None:
    runner = FakeCurlRunner({})
    fetcher = PublicDataFetcher(
        tmp_path,
        transport=CurlTransport(curl_binary="/usr/bin/curl", runner=runner),
    )

    with pytest.raises(PermissionError, match="explicitly approved"):
        fetcher.fetch(
            source="rcsb-ccd-sdf",
            identifier="ATP",
            approved_domains=("example.org",),
        )

    assert runner.commands == []


def test_rcsb_ligand_fetch_is_rdkit_parsed_and_license_bound(tmp_path) -> None:
    url = "https://files.rcsb.org/ligands/download/ATP_ideal.sdf"
    runner = FakeCurlRunner({url: (_sdf(), "chemical/x-mdl-sdfile")})
    fetcher = PublicDataFetcher(
        tmp_path,
        transport=CurlTransport(curl_binary="/usr/bin/curl", runner=runner),
    )

    result = fetcher.fetch(
        source="rcsb-ccd-sdf",
        identifier="ATP",
        approved_domains=("files.rcsb.org",),
    )

    assert result.artifact.license == "CC0-1.0"
    assert result.validation["parse_valid"] is True
    assert result.validation["declared_3d"] is True
    assert result.validation["v1_ordinary_ligand_candidate"] is True


def test_pubchem_fetch_surfaces_contributor_license_warning(tmp_path) -> None:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        "2244/SDF?record_type=3d"
    )
    runner = FakeCurlRunner({url: (_sdf(), "chemical/x-mdl-sdfile")})
    fetcher = PublicDataFetcher(
        tmp_path / "workspace",
        transport=CurlTransport(curl_binary="/usr/bin/curl", runner=runner),
    )

    result = fetcher.fetch(
        source="pubchem-cid-sdf-3d",
        identifier="2244",
        approved_domains=("pubchem.ncbi.nlm.nih.gov",),
    )

    assert result.artifact.license is None
    assert any("no uniform license" in warning for warning in result.warnings)
    assert result.to_dict()["warnings"] == list(result.warnings)


def test_alphafold_fetch_freezes_metadata_model_and_propka_outputs(tmp_path) -> None:
    metadata_url = "https://alphafold.ebi.ac.uk/api/prediction/P69905"
    model_url = "https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v6.cif"
    metadata = json.dumps(
        [
            {
                "uniprotAccession": "P69905",
                "isComplex": False,
                "latestVersion": 6,
                "entryId": "AF-P69905-F1",
                "cifUrl": model_url,
                "globalMetricValue": 98.0,
            }
        ]
    ).encode("utf-8")
    runner = FakeCurlRunner(
        {
            metadata_url: (metadata, "application/json"),
            model_url: (_mmcif(), "chemical/x-mmcif"),
        }
    )
    fetcher = PublicDataFetcher(
        tmp_path,
        transport=CurlTransport(curl_binary="/usr/bin/curl", runner=runner),
        propka=FakePropka(),
    )

    result = fetcher.fetch(
        source="alphafold-mmcif",
        identifier="P69905",
        approved_domains=("alphafold.ebi.ac.uk",),
    )

    assert len(runner.commands) == 3  # version probe plus metadata and model
    assert result.metadata_artifact is not None
    assert result.propka_artifact is not None
    assert result.validation["propka"]["status"] == "SUCCEEDED"
    assert result.validation["completeness"]["no_detected_completeness_findings"]


def test_gemmi_completeness_reports_missing_backbone_without_making_claim(tmp_path) -> None:
    url = "https://files.rcsb.org/download/1CRN.cif"
    runner = FakeCurlRunner({url: (_mmcif(include_oxygen=False), "chemical/x-mmcif")})
    fetcher = PublicDataFetcher(
        tmp_path,
        transport=CurlTransport(curl_binary="/usr/bin/curl", runner=runner),
    )

    result = fetcher.fetch(
        source="rcsb-mmcif",
        identifier="1crn",
        approved_domains=("files.rcsb.org",),
        run_propka=False,
    )

    completeness = result.validation["completeness"]
    assert completeness["standard_residue_missing_backbone_or_carbonyl_count"] == 1
    assert completeness["no_detected_completeness_findings"] is False
    assert completeness["scientific_gate"] is False
    assert result.validation["propka"]["status"] == "SKIPPED"


def test_propka_auditor_records_diagnostics_and_output(tmp_path) -> None:
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(list(command))
        if "--version" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="__main__.py 3.5.1\n",
                stderr="",
            )
        Path(kwargs["cwd"], "input.pka").write_text("pKa report\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr="WARNING: missing side-chain atom\n",
        )

    audit = PropkaAuditor(
        python_executable="/test/python",
        runner=runner,
    ).audit(_mmcif())

    assert audit.summary["status"] == "SUCCEEDED"
    assert audit.summary["scientific_gate"] is False
    assert audit.summary["diagnostics"] == ["WARNING: missing side-chain atom"]
    assert audit.output == b"pKa report\n"
    assert commands[-1][:3] == ["/test/python", "-m", "propka"]


def test_materialization_rejects_path_escape(tmp_path) -> None:
    url = "https://rest.uniprot.org/uniprotkb/P69905.fasta"
    runner = FakeCurlRunner({url: (b">P69905\nACDE\n", "text/plain")})
    fetcher = PublicDataFetcher(
        tmp_path / "workspace",
        transport=CurlTransport(curl_binary="/usr/bin/curl", runner=runner),
    )
    result = fetcher.fetch(
        source="uniprot-fasta",
        identifier="P69905",
        approved_domains=("rest.uniprot.org",),
    )

    with pytest.raises(ValueError, match="escapes"):
        materialize_public_fetch(
            result,
            fetcher.artifacts,
            project_root=tmp_path,
            output=Path("../outside.fasta"),
        )


def test_public_output_validation_rejects_wrong_suffix_before_fetch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"must use the \.cif suffix"):
        validate_public_output(
            "rcsb-mmcif",
            tmp_path,
            Path("inputs/receptor.pdb"),
        )


def test_materialization_does_not_overwrite_unrelated_sidecar(tmp_path) -> None:
    url = "https://rest.uniprot.org/uniprotkb/P69905.fasta"
    runner = FakeCurlRunner({url: (b">P69905\nACDE\n", "text/plain")})
    fetcher = PublicDataFetcher(
        tmp_path / "workspace",
        transport=CurlTransport(curl_binary="/usr/bin/curl", runner=runner),
    )
    result = fetcher.fetch(
        source="uniprot-fasta",
        identifier="P69905",
        approved_domains=("rest.uniprot.org",),
    )
    destination = tmp_path / "inputs/protein.fasta"
    destination.parent.mkdir()
    sidecar = destination.with_name(destination.name + ".protbind.json")
    sidecar.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="provenance sidecar"):
        materialize_public_fetch(
            result,
            fetcher.artifacts,
            project_root=tmp_path,
            output=Path("inputs/protein.fasta"),
        )

    assert sidecar.read_text(encoding="utf-8") == "operator-owned\n"
    assert not destination.exists()
