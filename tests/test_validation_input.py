from __future__ import annotations

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.validation_input import build_validation_input_batch


def test_validation_batch_is_derived_from_docked_sdf_and_seals_reference(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    receptor = store.put_bytes(
        b"ATOM fixture\n", media_type="chemical/x-pdb", producer="test"
    )
    pose = store.put_bytes(
        b"fixture SDF\n", media_type="chemical/x-mdl-sdfile", producer="test"
    )
    reference = store.put_bytes(
        b"native SDF\n", media_type="chemical/x-mdl-sdfile", producer="test"
    )
    docking = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.docking-bundle",
            "receptor": receptor.to_dict(),
            "candidates": [
                {
                    "candidate_id": "vina-mol-a",
                    "molecule_id": "mol-a",
                    "microstate_id": "state-1",
                    "pose": pose.to_dict(),
                }
            ],
        },
        producer="test",
    )

    batch_ref = build_validation_input_batch(
        store, docking, reference_pose=reference
    )
    batch = store.read_json(batch_ref)

    assert batch["schema_version"] == "2.0"
    assert batch["reference_scope"] == "VALIDATION_ONLY"
    assert batch["docking_bundle"] == docking.to_dict()
    candidate = batch["candidates"][0]
    assert candidate["docked_pose"] == pose.to_dict()
    assert candidate["reference_pose"] == reference.to_dict()
    assert candidate["posebusters"]["reference_ligand"] == reference.to_dict()
    assert candidate["spyrmsd"]["predicted_ligand"] == pose.to_dict()


def test_validation_batch_rejects_pdbqt_as_canonical_pose(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    receptor = store.put_bytes(b"PDB\n", media_type="chemical/x-pdb", producer="test")
    pose = store.put_bytes(
        b"PDBQT\n", media_type="chemical/x-pdbqt", producer="test"
    )
    docking = store.put_json(
        {
            "kind": "protbind.docking-bundle",
            "receptor": receptor.to_dict(),
            "candidates": [
                {
                    "candidate_id": "vina-a",
                    "molecule_id": "a",
                    "microstate_id": "s1",
                    "pose": pose.to_dict(),
                }
            ],
        },
        producer="test",
    )

    try:
        build_validation_input_batch(store, docking)
    except ValueError as exc:
        assert "canonical docked pose" in str(exc)
    else:
        raise AssertionError("PDBQT-only pose must not enter validation")


def test_single_reference_cannot_be_applied_to_multiple_candidate_chemistries(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    receptor = store.put_bytes(b"PDB\n", media_type="chemical/x-pdb", producer="test")
    pose = store.put_bytes(
        b"pose SDF\n", media_type="chemical/x-mdl-sdfile", producer="test"
    )
    reference = store.put_bytes(
        b"native SDF\n", media_type="chemical/x-mdl-sdfile", producer="test"
    )
    docking = store.put_json(
        {
            "kind": "protbind.docking-bundle",
            "receptor": receptor.to_dict(),
            "candidates": [
                {
                    "candidate_id": f"vina-{index}",
                    "molecule_id": f"mol-{index}",
                    "microstate_id": "s1",
                    "pose": pose.to_dict(),
                }
                for index in range(2)
            ],
        },
        producer="test",
    )

    with pytest.raises(ValueError, match="one-candidate redocking control"):
        build_validation_input_batch(store, docking, reference_pose=reference)
