from __future__ import annotations

import json

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.chemistry import bemis_murcko_scaffold_smiles
from protbind_agent.models import ArtifactRef
from protbind_agent.selection import (
    build_quick_vina_input,
    build_selection_preparation,
    finalize_selection_bundle,
)
from protbind_agent.tripharm import build_jsonl_index

_RECEPTOR_PDB = (
    b"ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 20.00           N  \n"
    b"ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00 20.00           C  \n"
    b"ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00 20.00           C  \n"
    b"TER\nEND\n"
)


def test_acyclic_diversity_keys_do_not_collapse_unrelated_graphs() -> None:
    ethanol = bemis_murcko_scaffold_smiles("CCO")
    ethylamine = bemis_murcko_scaffold_smiles("CCN")

    assert ethanol.startswith("ACYCLIC:")
    assert ethylamine.startswith("ACYCLIC:")
    assert ethanol != ethylamine


def test_selection_builds_diverse_microstates_and_requires_complete_real_scores(
    tmp_path,
) -> None:
    feature_file = tmp_path / "features.jsonl"
    feature_file.write_text(
        "".join(
            json.dumps(
                {
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "conformers": [
                        {
                            "id": 0,
                            "features": [
                                {"type": "Donor", "position": [0, 0, 0]},
                                {"type": "Acceptor", "position": [3, 0, 0]},
                                {"type": "Aromatic", "position": [0, 4, 0]},
                            ],
                        }
                    ],
                }
            )
            + "\n"
            for molecule_id, smiles in (
                ("acid", "CC(=O)O"),
                ("benzene", "c1ccccc1"),
            )
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.sqlite"
    build_jsonl_index(feature_file, index_path)
    store = ArtifactStore(tmp_path / "workspace")
    index = store.import_file(index_path, media_type="application/x-sqlite3")
    screen = store.put_json(
        {"hits": [{"molecule_id": "acid"}, {"molecule_id": "benzene"}]},
        producer="test",
    )
    receptor = store.put_bytes(
        _RECEPTOR_PDB, media_type="chemical/x-pdb", producer="test"
    )
    preparation = build_selection_preparation(
        store,
        screening=screen,
        library_index=index,
        receptor=receptor,
        protein_chains=(("A", "ACDEFG"),),
        box_center=(1.0, 2.0, 3.0),
        box_size=(20.0, 20.0, 20.0),
    )
    plan = store.read_json(preparation)
    box_receipt_ref = ArtifactRef.from_dict(plan["docking_box_receipt"])
    box_receipt = store.read_json(box_receipt_ref)
    assert box_receipt == {
        "schema_version": "2.0",
        "kind": "protbind.docking-box-receipt",
        "source_kind": "user-center",
        "site_derivation_evidence": None,
        "receptor": receptor.to_dict(),
        "receptor_sha256": receptor.sha256,
        "coordinate_frame": "receptor-cartesian-angstrom",
        "center": [1.0, 2.0, 3.0],
        "size": [20.0, 20.0, 20.0],
        "validation": {
            "finite_geometry_checked": True,
            "quick_site_bounds_checked": True,
            "minimum_dimension_angstrom": 4.0,
            "maximum_dimension_angstrom": 60.0,
            "maximum_volume_angstrom3": 27000.0,
            "receptor_atom_overlap_checked": True,
            "atom_overlap": {
                "coordinate_format": "pdb",
                "model_index": 0,
                "receptor_heavy_atom_count": 3,
                "protein_heavy_atom_count": 3,
                "receptor_heavy_atom_count_inside_box": 3,
                "protein_heavy_atom_count_inside_box": 3,
                "nearest_receptor_heavy_atom_distance_to_center_angstrom": 0.0,
                "nearest_protein_heavy_atom_distance_to_center_angstrom": 0.0,
                "receptor_atom_overlap": True,
                "protein_atom_overlap": True,
                "interpretation": "coordinate-frame-plausibility-only",
                "biological_site_validity_inferred": False,
            },
            "site_derivation_verified": False,
            "scientific_interpretation": "user-hypothesis-only",
            "biological_site_validity_inferred": False,
        },
    }
    request_ids = [item["request_id"] for item in plan["quick_vina_requests"]]
    assert len(request_ids) == len(set(request_ids))
    assert all(item.startswith("quick-") for item in request_ids)
    assert all(
        item["docking_box_receipt_sha256"] == box_receipt_ref.sha256
        and item["coordinate_frame"] == "receptor-cartesian-angstrom"
        for item in plan["quick_vina_requests"]
    )
    environment_lock = store.put_json({}, producer="test-lock")
    quick_input = store.read_json(
        build_quick_vina_input(
            store,
            preparation,
            environment_lock,
            case_id="box-receipt",
        )
    )
    assert quick_input["docking_box_receipt"] == box_receipt_ref.to_dict()

    changed_receipt_value = {**box_receipt, "center": [2.0, 2.0, 3.0]}
    changed_receipt = store.put_json(
        changed_receipt_value,
        producer=box_receipt_ref.producer,
        producer_version=box_receipt_ref.producer_version,
        source=box_receipt_ref.source,
    )
    changed_plan = json.loads(json.dumps(plan))
    changed_plan["docking_box_receipt"] = changed_receipt.to_dict()
    changed_preparation = store.put_json(
        changed_plan, producer="tampered-box-preparation"
    )
    with pytest.raises(ValueError, match="differs from its docking box receipt"):
        build_quick_vina_input(
            store,
            changed_preparation,
            environment_lock,
            case_id="box-tamper",
        )

    other_receptor = store.put_bytes(
        b"OTHER PDB\n", media_type="chemical/x-pdb", producer="test"
    )
    wrong_receptor_receipt_value = {
        **box_receipt,
        "receptor": other_receptor.to_dict(),
        "receptor_sha256": other_receptor.sha256,
    }
    wrong_receptor_receipt = store.put_json(
        wrong_receptor_receipt_value,
        producer=box_receipt_ref.producer,
        producer_version=box_receipt_ref.producer_version,
        source=other_receptor.artifact_id,
    )
    wrong_receptor_plan = json.loads(json.dumps(plan))
    wrong_receptor_plan["docking_box_receipt"] = wrong_receptor_receipt.to_dict()
    wrong_receptor_preparation = store.put_json(
        wrong_receptor_plan, producer="wrong-receptor-box-preparation"
    )
    with pytest.raises(ValueError, match="receipt artifact provenance is invalid"):
        build_quick_vina_input(
            store,
            wrong_receptor_preparation,
            environment_lock,
            case_id="receptor-tamper",
        )

    tampered_plan = json.loads(json.dumps(plan))
    tampered_plan["quick_vina_requests"][0]["formal_charge"] += 1
    tampered_preparation = store.put_json(
        tampered_plan,
        producer="tampered-selection-preparation",
    )
    with pytest.raises(ValueError, match="formal charge differs"):
        build_quick_vina_input(
            store,
            tampered_preparation,
            environment_lock,
            case_id="charge-tamper",
        )
    evaluations = []
    for rank, request in enumerate(plan["quick_vina_requests"]):
        pose = store.put_bytes(
            f"pose-{rank}".encode(), media_type="chemical/x-pdbqt", producer="vina"
        )
        score = -7.0 + rank
        evidence = store.put_json(
            {
                "kind": "protbind.tool-evidence",
                "tool": "vina",
                "molecule_id": request["molecule_id"],
                "microstate_id": request["microstate_id"],
                "metrics": {"score": score},
            },
            producer="vina",
        )
        evaluations.append(
            {
                "molecule_id": request["molecule_id"],
                "microstate_id": request["microstate_id"],
                "score": score,
                "pose": pose.to_dict(),
                "evidence": evidence.to_dict(),
            }
        )

    selection = store.read_json(
        finalize_selection_bundle(store, preparation, evaluations)
    )

    assert selection["kind"] == "protbind.selection-bundle"
    assert selection["candidate_count"] == 2
    assert selection["candidates"][0]["molecule_id"] == "acid"
    assert selection["candidates"][0]["request_id"] in request_ids
    assert selection["candidates"][0]["receptor"] == receptor.to_dict()
    assert selection["known_site_calibration"]["claimed"] is False
    assert selection["candidates"][0]["site_evidence"] == {
        "source_kind": "user-center",
        "receptor_atom_overlap_checked": True,
        "protein_heavy_atom_count_inside_box": 3,
        "site_derivation_verified": False,
        "scientific_interpretation": "user-hypothesis-only",
        "biological_site_validity_inferred": False,
        "known_site_calibration": selection["known_site_calibration"],
    }
    assert "not an experimental binding free energy" in selection["candidates"][0][
        "quick_vina_score_semantics"
    ]


def test_selection_records_named_quick_vina_failures_without_fabricating_scores(
    tmp_path,
) -> None:
    feature_file = tmp_path / "features.jsonl"
    feature_file.write_text(
        "".join(
            json.dumps(
                {
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "conformers": [
                        {
                            "id": 0,
                            "features": [
                                {"type": "Donor", "position": [0, 0, 0]},
                                {"type": "Acceptor", "position": [3, 0, 0]},
                                {"type": "Aromatic", "position": [0, 4, 0]},
                            ],
                        }
                    ],
                }
            )
            + "\n"
            for molecule_id, smiles in (
                ("acid", "CC(=O)O"),
                ("benzene", "c1ccccc1"),
            )
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.sqlite"
    build_jsonl_index(feature_file, index_path)
    store = ArtifactStore(tmp_path / "workspace")
    index = store.import_file(index_path, media_type="application/x-sqlite3")
    screen = store.put_json(
        {"hits": [{"molecule_id": "acid"}, {"molecule_id": "benzene"}]},
        producer="test",
    )
    receptor = store.put_bytes(
        _RECEPTOR_PDB, media_type="chemical/x-pdb", producer="test"
    )
    preparation = build_selection_preparation(
        store,
        screening=screen,
        library_index=index,
        receptor=receptor,
        protein_chains=(("A", "ACDEFG"),),
        box_center=(1.0, 2.0, 3.0),
        box_size=(20.0, 20.0, 20.0),
    )
    requests = store.read_json(preparation)["quick_vina_requests"]
    evaluations = []
    for rank, request in enumerate(requests):
        if rank == 0:
            evaluations.append(
                {
                    "molecule_id": request["molecule_id"],
                    "microstate_id": request["microstate_id"],
                    "status": "failed",
                    "code": "LIGAND_PREPARATION_FAILED",
                    "reason": "Meeko rejected this microstate",
                }
            )
            continue
        pose = store.put_bytes(
            f"pose-{rank}".encode(), media_type="chemical/x-pdbqt", producer="vina"
        )
        score = -6.0 + rank
        evidence = store.put_json(
            {
                "tool": "vina",
                "molecule_id": request["molecule_id"],
                "microstate_id": request["microstate_id"],
                "metrics": {"score": score},
            },
            producer="vina",
        )
        evaluations.append(
            {
                "molecule_id": request["molecule_id"],
                "microstate_id": request["microstate_id"],
                "status": "completed",
                "score": score,
                "pose": pose.to_dict(),
                "evidence": evidence.to_dict(),
            }
        )

    selection = store.read_json(
        finalize_selection_bundle(store, preparation, evaluations)
    )
    failures = selection["quick_vina"]["failures"]
    assert failures == [
        {
            "request_id": requests[0]["request_id"],
            "molecule_id": requests[0]["molecule_id"],
            "microstate_id": requests[0]["microstate_id"],
            "status": "failed",
            "code": "LIGAND_PREPARATION_FAILED",
            "reason": "Meeko rejected this microstate",
        }
    ]
    assert "score" not in failures[0]
