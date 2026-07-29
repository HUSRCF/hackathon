from __future__ import annotations

import json

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.caseio import ingest_case

_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  \n"
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C  \n"
    "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C  \n"
    "TER\nEND\n"
)


def test_caseio_builds_typed_site_evidence_from_real_source_artifact(tmp_path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(_PDB)
    source = tmp_path / "known-ligand.sdf"
    source.write_text("sealed source coordinates")
    case_file = tmp_path / "case.json"
    case_file.write_text(
        json.dumps(
            {
                "case_id": "typed-site",
                "target": {"name": "target", "structure_file": str(receptor)},
                "mode": "pocket_only",
                "pocket": {
                    "center": [0.0, 0.0, 0.0],
                    "box_size": [10.0, 10.0, 10.0],
                    "site_provenance_kind": "co-crystal-ligand",
                    "site_derivation_source_files": [str(source)],
                    "site_derivation_method": "co-crystal heavy-atom envelope",
                    "site_derivation_license": "test-only",
                },
            }
        )
    )
    store = ArtifactStore(tmp_path / "workspace")

    case = ingest_case(case_file, store)

    assert case.pocket is not None and case.pocket.site_evidence is not None
    evidence = store.read_json(case.pocket.site_evidence)
    assert evidence["source_kind"] == "co-crystal-ligand"
    assert evidence["reference_atom_coordinates_exposed_to_screening"] is False
    assert evidence["reference_derived_box_exposed_to_screening"] is True
    assert len(evidence["source_commitments"]) == 1
    store.resolve_sha256(evidence["source_commitments"][0])


def test_caseio_rejects_untyped_prebuilt_site_evidence_json(tmp_path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(_PDB)
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    case_file = tmp_path / "case.json"
    case_file.write_text(
        json.dumps(
            {
                "case_id": "untyped-site",
                "target": {"name": "target", "structure_file": str(receptor)},
                "mode": "pocket_only",
                "pocket": {
                    "center": [0.0, 0.0, 0.0],
                    "box_size": [10.0, 10.0, 10.0],
                    "site_provenance_kind": "co-crystal-ligand",
                    "site_evidence_file": str(evidence),
                },
            }
        )
    )

    with pytest.raises(ValueError, match="not trusted"):
        ingest_case(case_file, ArtifactStore(tmp_path / "workspace"))
