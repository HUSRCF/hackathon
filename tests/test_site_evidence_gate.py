from __future__ import annotations

import json

import pytest

from protbind_agent import selection as selection_module
from protbind_agent.artifacts import ArtifactStore
from protbind_agent.chemistry import MicrostateRecord
from protbind_agent.models import ArtifactRef, SiteProvenanceKind
from protbind_agent.selection import (
    build_quick_vina_input,
    build_selection_preparation,
    build_site_derivation_evidence,
)
from protbind_agent.structure import inspect_box_atom_overlap
from protbind_agent.tripharm import build_jsonl_index

gemmi = pytest.importorskip("gemmi")

_PDB = (
    b"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  \n"
    b"ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C  \n"
    b"ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C  \n"
    b"ATOM      4  H   ALA A   1       0.000   0.000   1.000  1.00 20.00           H  \n"
    b"HETATM    5  O   HOH B   1       0.000   1.000   0.000  1.00 20.00           O  \n"
    b"TER\nEND\n"
)


def _mmcif() -> bytes:
    structure = gemmi.read_pdb_string(_PDB.decode())
    return structure.make_mmcif_document().as_string().encode()


def _inputs(
    tmp_path, receptor_bytes: bytes = _PDB, media_type: str = "chemical/x-pdb"
) -> tuple[ArtifactStore, ArtifactRef, ArtifactRef, ArtifactRef]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    feature_file = tmp_path / "features.jsonl"
    feature_file.write_text(
        json.dumps(
            {
                "molecule_id": "mol-1",
                "smiles": "CCO",
                "conformers": [
                    {
                        "id": 0,
                        "features": [
                            {"type": "Donor", "position": [0, 0, 0]},
                            {"type": "Acceptor", "position": [3, 0, 0]},
                            {"type": "Hydrophobe", "position": [0, 4, 0]},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.sqlite"
    build_jsonl_index(feature_file, index_path)
    store = ArtifactStore(tmp_path / "workspace")
    index = store.import_file(index_path, media_type="application/x-sqlite3")
    screen = store.put_json(
        {"hits": [{"molecule_id": "mol-1"}]}, producer="test-screen"
    )
    receptor = store.put_bytes(
        receptor_bytes, media_type=media_type, producer="test-receptor"
    )
    return store, index, screen, receptor


def _stub_microstates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        selection_module,
        "enumerate_microstates",
        lambda smiles, max_states: (
            MicrostateRecord(
                microstate_id="state-01",
                canonical_isomeric_smiles=smiles,
                formal_charge=0,
                parent_standardized_smiles=smiles,
                enumeration_method="test-only",
            ),
        ),
    )


@pytest.mark.parametrize(
    ("suffix", "payload", "coordinate_format"),
    ((".pdb", _PDB, "pdb"), (".cif", None, "mmcif")),
)
def test_box_overlap_counts_heavy_and_protein_atoms_for_pdb_and_mmcif(
    tmp_path, suffix, payload, coordinate_format
) -> None:
    path = tmp_path / f"receptor{suffix}"
    path.write_bytes(_mmcif() if payload is None else payload)

    inspection = inspect_box_atom_overlap(
        path,
        center=(0.0, 0.0, 0.0),
        size=(10.0, 10.0, 10.0),
    )

    assert inspection.coordinate_format == coordinate_format
    assert inspection.receptor_heavy_atom_count == 4
    assert inspection.protein_heavy_atom_count == 3
    assert inspection.receptor_heavy_atom_count_inside_box == 4
    assert inspection.protein_heavy_atom_count_inside_box == 3
    assert inspection.protein_atom_overlap is True
    assert inspection.to_dict()["biological_site_validity_inferred"] is False


def test_selection_rejects_far_away_and_heavy_atom_empty_receptors(tmp_path) -> None:
    store, index, screen, receptor = _inputs(tmp_path / "far")
    with pytest.raises(ValueError, match="contains no protein heavy atoms"):
        build_selection_preparation(
            store,
            screening=screen,
            library_index=index,
            receptor=receptor,
            protein_chains=(("A", "A"),),
            box_center=(100.0, 100.0, 100.0),
            box_size=(10.0, 10.0, 10.0),
        )

    hydrogen_only = (
        b"ATOM      1  H   ALA A   1       0.000   0.000   0.000  "
        b"1.00 20.00           H  \nTER\nEND\n"
    )
    empty_store, empty_index, empty_screen, empty_receptor = _inputs(
        tmp_path / "empty", hydrogen_only
    )
    with pytest.raises(ValueError, match="contains no heavy atoms"):
        build_selection_preparation(
            empty_store,
            screening=empty_screen,
            library_index=empty_index,
            receptor=empty_receptor,
            protein_chains=(("A", "A"),),
            box_center=(0.0, 0.0, 0.0),
            box_size=(10.0, 10.0, 10.0),
        )


def test_site_derivation_evidence_is_bound_without_exposing_reference_coordinates(
    tmp_path, monkeypatch
) -> None:
    _stub_microstates(monkeypatch)
    store, index, screen, receptor = _inputs(tmp_path)
    source = store.put_bytes(
        b"sealed co-crystal coordinates",
        media_type="chemical/x-mdl-sdfile",
        producer="test-co-crystal-source",
    )
    evidence = build_site_derivation_evidence(
        store,
        receptor=receptor,
        source_kind=SiteProvenanceKind.COCRYSTAL_LIGAND,
        center=(0.0, 0.0, 0.0),
        size=(10.0, 10.0, 10.0),
        derivation_method="sanitized co-crystal ligand heavy-atom envelope",
        source_artifacts=(source,),
    )

    preparation = build_selection_preparation(
        store,
        screening=screen,
        library_index=index,
        receptor=receptor,
        protein_chains=(("A", "A"),),
        box_center=(0.0, 0.0, 0.0),
        box_size=(10.0, 10.0, 10.0),
        box_source=SiteProvenanceKind.COCRYSTAL_LIGAND,
        site_derivation_evidence=evidence,
    )
    value = store.read_json(preparation)
    assert value["site_evidence"]["site_derivation_verified"] is True
    assert value["site_evidence"]["scientific_interpretation"] == (
        "independently-supported-derivation"
    )
    lock = store.put_json({}, producer="test-lock")
    quick_input = store.read_json(
        build_quick_vina_input(store, preparation, lock, case_id="site-evidence")
    )
    assert quick_input["site_derivation_evidence"] == evidence.to_dict()

    evidence_value = store.read_json(evidence)
    changed_evidence = store.put_json(
        {**evidence_value, "center": [1.0, 0.0, 0.0]},
        producer=evidence.producer,
        producer_version=evidence.producer_version,
        source=evidence.source,
    )
    with pytest.raises(ValueError, match="coordinate-free receipt"):
        build_selection_preparation(
            store,
            screening=screen,
            library_index=index,
            receptor=receptor,
            protein_chains=(("A", "A"),),
            box_center=(0.0, 0.0, 0.0),
            box_size=(10.0, 10.0, 10.0),
            box_source=SiteProvenanceKind.COCRYSTAL_LIGAND,
            site_derivation_evidence=changed_evidence,
        )


def test_user_site_cannot_be_promoted_with_independent_evidence(
    tmp_path, monkeypatch
) -> None:
    _stub_microstates(monkeypatch)
    store, index, screen, receptor = _inputs(tmp_path)
    evidence = store.put_json({}, producer="test-site-derivation")

    with pytest.raises(ValueError, match="user-center cannot carry"):
        build_selection_preparation(
            store,
            screening=screen,
            library_index=index,
            receptor=receptor,
            protein_chains=(("A", "A"),),
            box_center=(0.0, 0.0, 0.0),
            box_size=(10.0, 10.0, 10.0),
            box_source=SiteProvenanceKind.USER_CENTER,
            site_derivation_evidence=evidence,
        )


def test_box_overlap_receipt_tampering_is_recomputed_from_receptor(
    tmp_path, monkeypatch
) -> None:
    _stub_microstates(monkeypatch)
    store, index, screen, receptor = _inputs(tmp_path)
    preparation = build_selection_preparation(
        store,
        screening=screen,
        library_index=index,
        receptor=receptor,
        protein_chains=(("A", "A"),),
        box_center=(0.0, 0.0, 0.0),
        box_size=(10.0, 10.0, 10.0),
    )
    plan = store.read_json(preparation)
    receipt_ref = ArtifactRef.from_dict(plan["docking_box_receipt"])
    receipt = store.read_json(receipt_ref)
    changed = json.loads(json.dumps(receipt))
    changed["validation"]["atom_overlap"][
        "protein_heavy_atom_count_inside_box"
    ] += 1
    changed_ref = store.put_json(
        changed,
        producer=receipt_ref.producer,
        producer_version=receipt_ref.producer_version,
        source=receipt_ref.source,
    )
    plan["docking_box_receipt"] = changed_ref.to_dict()
    tampered = store.put_json(plan, producer="tampered-preparation")
    lock = store.put_json({}, producer="test-lock")

    with pytest.raises(ValueError, match="atom-overlap receipt differs"):
        build_quick_vina_input(store, tampered, lock, case_id="tampered-site")
