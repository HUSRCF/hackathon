from __future__ import annotations

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.interaction_fingerprint import (
    interaction_fingerprint_metrics,
    prepare_prolif_ligand,
    prepare_prolif_receptor,
)


def test_ifp_metrics_report_recovery_precision_and_complete_counts() -> None:
    metrics = interaction_fingerprint_metrics(
        {"A:10|Hydrophobic", "A:20|HBDonor"},
        {"A:20|HBDonor", "A:30|HBAcceptor", "A:40|PiStacking"},
        comparison_name="reference",
    )

    assert metrics["ifp_similarity"] == pytest.approx(0.25)
    assert metrics["reference_interaction_recovery"] == pytest.approx(1 / 3)
    assert metrics["predicted_interaction_precision"] == pytest.approx(0.5)
    assert metrics["counts"] == {
        "docked": 2,
        "comparison": 3,
        "intersection": 1,
        "union": 4,
    }
    assert metrics["comparison"] == "reference"


def test_ifp_empty_sets_are_not_promoted_to_perfect_evidence() -> None:
    metrics = interaction_fingerprint_metrics(
        set(), set(), comparison_name="reference"
    )

    assert metrics["ifp_similarity"] is None
    assert metrics["reference_interaction_recovery"] is None
    assert metrics["predicted_interaction_precision"] is None
    assert metrics["counts"]["union"] == 0


def test_ifp_without_comparison_preserves_docked_labels_without_scores() -> None:
    metrics = interaction_fingerprint_metrics(
        {"A:10|Hydrophobic"}, None, comparison_name=None
    )

    assert metrics["docked_labels"] == ["A:10|Hydrophobic"]
    assert metrics["comparison_labels"] is None
    assert metrics["ifp_similarity"] is None
    assert metrics["counts"]["comparison"] is None


def test_ifp_requires_named_comparison_and_nonempty_labels() -> None:
    with pytest.raises(ValueError, match="comparison_name"):
        interaction_fingerprint_metrics({"A:1|HBDonor"}, {"A:1|HBDonor"}, comparison_name=None)
    with pytest.raises(ValueError, match="non-empty strings"):
        interaction_fingerprint_metrics({""}, None, comparison_name=None)


def test_prolif_preparation_adds_hydrogens_without_moving_heavy_atoms(tmp_path) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=20260721) == 0
    molecule = Chem.RemoveHs(molecule)
    store = ArtifactStore(tmp_path / "artifacts")
    source = store.put_bytes(
        (Chem.MolToMolBlock(molecule) + "\n$$$$\n").encode(),
        media_type="chemical/x-mdl-sdfile",
        producer="test-native-reference",
        license="test-only",
    )

    result = prepare_prolif_ligand(
        store, source, artifact_scope="VALIDATION_ONLY"
    )
    receipt = store.read_json(result.receipt)

    assert result.prepared_ligand != source
    assert result.hydrogens_added > 0
    assert result.heavy_atom_max_coordinate_delta_angstrom <= 1e-6
    assert receipt["input_ligand"] == source.to_dict()
    assert receipt["prepared_ligand"] == result.prepared_ligand.to_dict()
    assert receipt["heavy_atom_identity_preserved"] is True
    assert receipt["artifact_scope"] == "VALIDATION_ONLY"


def test_prolif_preparation_reuses_explicit_hydrogen_input(tmp_path) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    molecule = Chem.AddHs(Chem.MolFromSmiles("CO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=20260721) == 0
    store = ArtifactStore(tmp_path / "artifacts")
    source = store.put_bytes(
        (Chem.MolToMolBlock(molecule) + "\n$$$$\n").encode(),
        media_type="chemical/x-mdl-sdfile",
        producer="test-prepared-ligand",
    )

    result = prepare_prolif_ligand(
        store, source, artifact_scope="DOCKING_VISIBLE"
    )

    assert result.prepared_ligand == source
    assert result.hydrogens_added == 0
    assert store.read_json(result.receipt)["method"] == (
        "reuse-existing-explicit-hydrogens"
    )


def _pdb_atom(
    serial: int,
    name: str,
    residue_number: int,
    xyz: tuple[float, float, float],
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {name:>4} ALA A{residue_number:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}          {element:>2}  "
    )


def test_prolif_receptor_crop_is_union_bound_and_coordinate_preserving(tmp_path) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    pytest.importorskip("gemmi")
    ligand = Chem.AddHs(Chem.MolFromSmiles("CO"))
    assert AllChem.EmbedMolecule(ligand, randomSeed=20260721) == 0
    store = ArtifactStore(tmp_path / "artifacts")
    ligand_ref = store.put_bytes(
        (Chem.MolToMolBlock(ligand) + "\n$$$$\n").encode(),
        media_type="chemical/x-mdl-sdfile",
        producer="test-ligand",
    )
    lines = [
        _pdb_atom(1, "N", 1, (-1.0, 3.0, 0.0), "N"),
        _pdb_atom(2, "CA", 1, (0.0, 3.0, 0.0), "C"),
        _pdb_atom(3, "C", 1, (1.4, 3.0, 0.0), "C"),
        _pdb_atom(4, "O", 1, (2.5, 3.0, 0.0), "O"),
        _pdb_atom(5, "CB", 1, (0.0, 4.5, 0.0), "C"),
        _pdb_atom(6, "H", 1, (-1.6, 3.0, 0.0), "H"),
        _pdb_atom(7, "N", 2, (30.0, 0.0, 0.0), "N"),
        _pdb_atom(8, "CA", 2, (31.0, 0.0, 0.0), "C"),
        _pdb_atom(9, "C", 2, (32.4, 0.0, 0.0), "C"),
        _pdb_atom(10, "O", 2, (33.5, 0.0, 0.0), "O"),
        _pdb_atom(11, "CB", 2, (31.0, 1.5, 0.0), "C"),
        _pdb_atom(12, "H", 2, (29.4, 0.0, 0.0), "H"),
        "TER",
        "END",
    ]
    receptor_ref = store.put_bytes(
        ("\n".join(lines) + "\n").encode(),
        media_type="chemical/x-pdb",
        producer="test-receptor",
    )

    result = prepare_prolif_receptor(
        store,
        receptor_ref,
        ligand_ref,
        ligand_ref,
        cutoff_angstrom=8.0,
    )
    receipt = store.read_json(result.receipt)
    output = store.read_bytes(result.prepared_receptor).decode()

    assert result.selected_residue_count == 1
    assert "A:ALA:1" in receipt["selected_residue_ids"]
    assert "A:ALA:2" not in receipt["selected_residue_ids"]
    assert "ALA A   2" not in output
    assert receipt["atom_identity_preserved"] is True
    assert receipt["coordinate_max_delta_angstrom"] <= 0.002
