from __future__ import annotations

import io
import random

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.redocking import (
    RedockingBenchmarkCandidate,
    ReferenceAccessError,
    build_redocking_case,
    native_derived_box,
    persist_holdout_manifest,
    prepare_redocking_ligand,
    seal_validation_reference,
    select_holdout_manifest,
    symmetry_rmsd,
)

Chem = pytest.importorskip("rdkit.Chem")
AllChem = pytest.importorskip("rdkit.Chem.AllChem")
pytest.importorskip("spyrmsd")


def _sdf(store: ArtifactStore, smiles: str, *, seed: int = 17):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    handle = io.StringIO()
    writer = Chem.SDWriter(handle)
    writer.write(molecule)
    writer.close()
    return store.put_bytes(
        handle.getvalue().encode(),
        media_type="chemical/x-mdl-sdfile",
        producer="test.native-pose",
        license="CC0-1.0",
    )


def _two_atom_pose(store: ArtifactStore, positions: tuple[tuple[float, float, float], ...]):
    molecule = Chem.MolFromSmiles("CC")
    conformer = Chem.Conformer(2)
    for index, position in enumerate(positions):
        conformer.SetAtomPosition(index, position)
    molecule.AddConformer(conformer)
    handle = io.StringIO()
    writer = Chem.SDWriter(handle)
    writer.write(molecule)
    writer.close()
    return store.put_bytes(
        handle.getvalue().encode(),
        media_type="chemical/x-mdl-sdfile",
        producer="test.pose",
    )


def test_ligand_preparation_strips_coordinates_and_is_deterministic(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    native = _sdf(store, "C[NH2+]C[C@H](O)Cl")

    first = prepare_redocking_ligand(store, native, seed=20260721)
    second = prepare_redocking_ligand(store, native, seed=20260721)

    assert first == second
    assert first.ligand_3d.sha256 != native.sha256
    assert first.formal_charge == 1
    assert "@" in first.canonical_isomeric_smiles
    identity = store.read_json(first.identity)
    receipt = store.read_json(first.receipt)
    assert identity["contains_coordinates"] is False
    assert receipt["native_coordinates_discarded"] is True
    assert receipt["native_coordinates_copied"] is False
    assert receipt["identity_roundtrip_valid"] is True
    assert receipt["conformer_generation"] == "ETKDGv3"
    assert receipt["minimization_method"] in {"MMFF94s", "UFF"}


def test_native_box_and_sealed_reference_do_not_leak_into_docking_payload(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    native = _sdf(store, "C[C@H](O)Cl")
    ligand = prepare_redocking_ligand(store, native, seed=11)
    box = native_derived_box(store, native, padding_angstrom=5.0)
    receptor = store.put_bytes(
        b"prepared receptor", media_type="chemical/x-pdb", producer="test.receptor"
    )
    receptor_receipt = store.put_json({}, producer="test.receptor-receipt")
    sealed = seal_validation_reference("case-1", native, ligand.identity)
    case = build_redocking_case(
        case_id="case-1",
        receptor=receptor,
        receptor_preparation_receipt=receptor_receipt,
        ligand=ligand,
        box=box,
        sealed_reference=sealed,
        seed=11,
    )

    docking_payload = case.to_docking_dict()
    serialized = repr(docking_payload)
    assert docking_payload["artifact_scope"] == "DOCKING_VISIBLE"
    assert native.artifact_id not in serialized
    assert "native_pose" not in serialized
    assert box.definition == "redock-known-site"
    assert all(size >= 10.0 for size in box.size)

    docked = _sdf(store, "C[C@H](O)Cl", seed=31)
    released = sealed.release(case, committed_docking_pose=docked)
    assert released.to_validation_dict()["native_pose"]["sha256"] == native.sha256
    with pytest.raises(ReferenceAccessError, match="cannot be the native"):
        sealed.release(case, committed_docking_pose=native)


def test_symmetry_rmsd_uses_graph_symmetry_without_coordinate_fitting(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    reference = _two_atom_pose(store, ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    symmetric_swap = _two_atom_pose(store, ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    translated_swap = _two_atom_pose(store, ((2.0, 1.0, 0.0), (0.0, 1.0, 0.0)))

    assert symmetry_rmsd(store, reference, symmetric_swap).value_angstrom == pytest.approx(0.0)
    translated = symmetry_rmsd(store, reference, translated_swap)
    assert translated.value_angstrom == pytest.approx(1.0)
    assert translated.centered is False
    assert translated.minimized is False


def _candidate(store: ArtifactStore, complex_id: str, **overrides):
    artifact = store.put_bytes(
        complex_id.encode(),
        media_type="chemical/x-pdb",
        producer="test.dataset",
        license="CC-BY-4.0",
    )
    values = {
        "complex_id": complex_id,
        "ligand_instance_id": "Z:LIG:1",
        "source_complex": artifact,
        "receptor": artifact,
        "native_ligand": artifact,
        "license": "CC-BY-4.0",
        "protein_chain_count": 1,
        "protein_residue_count": 200,
        "ligand_count": 1,
        "ligand_heavy_atom_count": 24,
    }
    values.update(overrides)
    return RedockingBenchmarkCandidate(**values)


def test_holdout_selection_is_order_independent_and_freezes_artifacts(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    candidates = [_candidate(store, f"case-{index}") for index in range(7)]
    candidates.append(_candidate(store, "metal-case", contains_metal=True))
    shuffled = list(candidates)
    random.Random(9).shuffle(shuffled)

    first = select_holdout_manifest(
        candidates,
        dataset_name="fixture",
        dataset_version="1",
        dataset_license="CC-BY-4.0",
        count=3,
        excluded_complex_ids={"case-0"},
    )
    second = select_holdout_manifest(
        shuffled,
        dataset_name="fixture",
        dataset_version="1",
        dataset_license="CC-BY-4.0",
        count=3,
        excluded_complex_ids={"case-0"},
    )

    assert first.selection_hash == second.selection_hash
    assert [item.complex_id for item in first.selected] == [
        item.complex_id for item in second.selected
    ]
    exclusions = {item.complex_id: item.reasons for item in first.exclusions}
    assert exclusions["metal-case"] == ("metal_system",)
    assert exclusions["case-0"] == ("development_or_explicitly_excluded_case",)
    artifact = persist_holdout_manifest(store, first)
    frozen = store.read_json(artifact)
    assert frozen["selection_hash"] == first.selection_hash
    assert all("native_ligand" in selected for selected in frozen["selected"])

