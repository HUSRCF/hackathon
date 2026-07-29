from __future__ import annotations

import json
from pathlib import Path

import pytest

from protbind_agent.artifacts import (
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from protbind_agent.redock_regression import (
    RegressionIntegrityError,
    build_redock_regression,
    persist_redock_regression,
)


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_file(path)


def _json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def _artifact(
    sha256: str,
    *,
    file: str,
    scope: str,
    media_type: str,
    producer: str,
) -> dict[str, object]:
    return {
        "sha256": sha256,
        "size_bytes": 1,
        "media_type": media_type,
        "producer": producer,
        "producer_version": "test-1",
        "source": "dataset:fixture",
        "license": "CC0-1.0",
        "file": file,
        "access_scope": scope,
    }


def _completed_result(
    repo: Path,
    case_id: str,
    *,
    ligand_sdf: bytes | None = None,
) -> tuple[str, str, str]:
    Chem = pytest.importorskip("rdkit.Chem")
    directory = repo / "runs" / case_id
    if ligand_sdf is None:
        molecule = Chem.MolFromSmiles("CCO")
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        for index in range(molecule.GetNumAtoms()):
            conformer.SetAtomPosition(index, (float(index), 0.0, 0.0))
        molecule.AddConformer(conformer)
        ligand_sdf = (Chem.MolToMolBlock(molecule) + "\n$$$$\n").encode()
    poses_path = directory / "artifacts" / "vina-poses.sdf"
    poses_sha = _write(poses_path, ligand_sdf)
    records = [
        record
        for record in Chem.SDMolSupplier(str(poses_path), removeHs=False)
        if record is not None
    ]
    assert records
    top1_path = directory / "artifacts" / "vina-mode-01.sdf"
    writer = Chem.SDWriter(str(top1_path))
    writer.write(records[0])
    writer.close()
    top1_sha = sha256_file(top1_path)
    native_sha = _write(
        directory / "validation-only" / "native-reference.sdf",
        top1_path.read_bytes(),
    )
    prepared_lines = [
        "ATOM      1  N   ALA A   1      -1.000   3.000   0.000  1.00 20.00           N  ",
        "ATOM      2  CA  ALA A   1       0.000   3.000   0.000  1.00 20.00           C  ",
        "ATOM      3  C   ALA A   1       1.400   3.000   0.000  1.00 20.00           C  ",
        "ATOM      4  O   ALA A   1       2.500   3.000   0.000  1.00 20.00           O  ",
        "ATOM      5  CB  ALA A   1       0.000   4.500   0.000  1.00 20.00           C  ",
        "ATOM      6  H   ALA A   1      -1.600   3.000   0.000  1.00 20.00           H  ",
        "TER",
        "END",
    ]
    prepared_sha = _write(
        directory / "artifacts" / "receptor-prepared.pdb",
        ("\n".join(prepared_lines) + "\n").encode(),
    )
    receptor_input_sha = _write(
        directory / "artifacts" / "receptor-input.pdb",
        f"input-{case_id}".encode(),
    )
    result = {
        "schema_version": "1.2",
        "benchmark": "redock",
        "status": "COMPLETED",
        "failure": None,
        "top1": {
            "mode": 1,
            "file": "artifacts/vina-mode-01.sdf",
            "pose_artifact": {
                "sha256": top1_sha,
                "media_type": "chemical/x-mdl-sdfile",
                "producer": "protbind.redocking.mode-split",
                "producer_version": "test-1",
                "size_bytes": 1,
            },
            "posebusters_valid": True,
            "symmetry_rmsd_angstrom": 1.0,
            "recovered": True,
        },
        "top1_recovered": True,
        "top5_recovered": True,
        "pose_count": len(records),
        "top5_oracle": {
            "evaluated_modes": min(5, len(records)),
            "best_mode": 1,
            "best_symmetry_rmsd_angstrom": 1.0,
            "any_pb_valid_and_rmsd_le_2": True,
            "first_recovered_mode": 1,
        },
        "artifacts": {
            "native_reference": _artifact(
                native_sha,
                file="validation-only/native-reference.sdf",
                scope="VALIDATION_ONLY",
                media_type="chemical/x-mdl-sdfile",
                producer="protbind.redocking.validation-reference",
            ),
            "prepared_receptor": _artifact(
                prepared_sha,
                file="artifacts/receptor-prepared.pdb",
                scope="DOCKING_VISIBLE",
                media_type="chemical/x-pdb",
                producer="meeko.mk_prepare_receptor",
            ),
            "receptor_input": _artifact(
                receptor_input_sha,
                file="artifacts/receptor-input.pdb",
                scope="DOCKING_VISIBLE",
                media_type="chemical/x-pdb",
                producer="protbind.redocking.local-input",
            ),
            "vina_poses_sdf": _artifact(
                poses_sha,
                file="artifacts/vina-poses.sdf",
                scope="DOCKING_VISIBLE",
                media_type="chemical/x-mdl-sdfile",
                producer="meeko.mk_export",
            ),
        },
    }
    result_relative = f"runs/{case_id}/result.json"
    result_sha = _json(repo / result_relative, result)
    return result_relative, result_sha, receptor_input_sha


def _failed_result(repo: Path, case_id: str) -> tuple[str, str]:
    result = {
        "schema_version": "1.1",
        "benchmark": "redock",
        "status": "FAILED",
        "failure": {
            "stage": "vina",
            "code": "TOOL_NONZERO_EXIT",
            "message": "fixture failure",
        },
    }
    relative = f"runs/{case_id}/result.json"
    return relative, _json(repo / relative, result)


def _holdout_candidate(
    case_id: str,
    receptor_sha: str,
    native_sha: str,
) -> dict[str, object]:
    def artifact(sha: str) -> dict[str, object]:
        return {
            "sha256": sha,
            "size_bytes": 1,
            "media_type": "chemical/x-pdb",
            "producer": "fixture",
            "producer_version": "1",
            "source": "dataset:fixture",
            "license": "CC0-1.0",
        }

    return {
        "complex_id": case_id,
        "ligand_instance_id": "L:LIG:1",
        "source_complex": artifact("a" * 64),
        "receptor": artifact(receptor_sha),
        "native_ligand": {
            **artifact(native_sha),
            "media_type": "chemical/x-mdl-sdfile",
        },
        "license": "CC0-1.0",
        "protein_chain_count": 1,
        "protein_residue_count": 100,
        "ligand_count": 1,
        "ligand_heavy_atom_count": 20,
        "is_non_covalent": True,
        "ordinary_nonpolymer_ligand": True,
        "contains_metal": False,
        "requires_cofactor": False,
        "pocket_altloc_ambiguous": False,
        "missing_pocket_heavy_atoms": False,
    }


def _fixture_repo(
    repo: Path,
    *,
    design: str = "FROZEN_HOLDOUT",
    count: int = 10,
    failed: set[int] = frozenset(),
    unattempted: set[int] = frozenset(),
    ligand_sdf: bytes | None = None,
) -> Path:
    cases: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for index in range(count):
        case_id = f"case-{index:02d}"
        if index in failed:
            native_sha = sha256_bytes(f"native-{case_id}".encode())
            receptor_sha = sha256_bytes(f"input-{case_id}".encode())
            result_path, result_sha = _failed_result(repo, case_id)
        else:
            result_path, result_sha, receptor_sha = _completed_result(
                repo,
                case_id,
                ligand_sdf=ligand_sdf,
            )
            native_sha = sha256_file(
                repo / "runs" / case_id / "validation-only" / "native-reference.sdf"
            )
        selected.append(_holdout_candidate(case_id, receptor_sha, native_sha))
        cases.append(
            {
                "case_id": case_id,
                "result": (
                    None if index in unattempted else {"path": result_path, "sha256": result_sha}
                ),
            }
        )

    holdout_pointer = None
    if design == "FROZEN_HOLDOUT":
        holdout_body = {
            "schema_version": "1.0",
            "dataset_name": "fixture",
            "dataset_version": "1",
            "dataset_license": "CC0-1.0",
            "namespace": "fixture-v1",
            "requested_count": 10,
            "selection_rule": "sha256(namespace + ':' + complex_id), then complex_id",
            "selected": selected,
            "exclusions": [],
        }
        holdout = {
            **holdout_body,
            "selection_hash": sha256_bytes(canonical_json_bytes(holdout_body)),
        }
        holdout_sha = _json(repo / "holdout.json", holdout)
        holdout_pointer = {
            "path": "holdout.json",
            "sha256": holdout_sha,
            "selection_hash": holdout["selection_hash"],
        }

    manifest_body = {
        "schema_version": "1.0",
        "evaluation_design": design,
        "target_case_count": 10,
        "holdout": holdout_pointer,
        "cases": cases,
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest_body)),
    }
    _json(repo / "regression.json", manifest)
    return Path("regression.json")


def _rewrite_case_result(
    repo: Path,
    manifest_path: Path,
    case_id: str,
    update: dict[str, object],
) -> None:
    manifest_file = repo / manifest_path
    manifest = json.loads(manifest_file.read_text())
    case = next(item for item in manifest["cases"] if item["case_id"] == case_id)
    result_path = repo / case["result"]["path"]
    result = json.loads(result_path.read_text())
    result.update(update)
    case["result"]["sha256"] = _json(result_path, result)
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(body))
    _json(manifest_file, manifest)


def _rmsd(_native: Path, _predicted: Path) -> float:
    return 1.0


def _posebusters(_predicted: Path, _native: Path, _receptor: Path) -> dict[str, object]:
    return {
        "posebusters_valid": True,
        "posebusters_checks": {"geometry": True, "rmsd <= 2": True},
        "energy_ratio": 1.7,
        "mol_pred_energy": 3.0,
        "ensemble_avg_energy": 2.0,
        "protein_ligand_pairwise_clash_count": 0,
        "protein_ligand_clash_detected": False,
        "protein_ligand_volume_overlap": 0.02,
        "protein_ligand_minimum_distance_angstrom": 2.7,
    }


def _ifp(**_kwargs) -> dict[str, object]:
    return {
        "ifp_similarity": 0.75,
        "reference_interaction_recovery": 0.75,
        "predicted_interaction_precision": 1.0,
        "counts": {
            "docked": 3,
            "comparison": 4,
            "intersection": 3,
            "union": 4,
        },
    }


def _build(repo: Path, manifest: Path = Path("regression.json")) -> dict[str, object]:
    return build_redock_regression(
        repo,
        manifest,
        rmsd_evaluator=_rmsd,
        posebusters_evaluator=_posebusters,
        ifp_evaluator=_ifp,
    )


def test_exact_ten_hash_bound_holdout_completes_gate(tmp_path: Path) -> None:
    manifest = _fixture_repo(tmp_path)
    result = _build(tmp_path, manifest)

    assert result["evaluation_design"] == "FROZEN_HOLDOUT"
    assert result["mechanical_evaluation_complete"] is True
    assert result["gate_complete"] is False
    assert result["denominators"] == {
        "frozen": 10,
        "attempted": 10,
        "completed": 10,
        "failed": 0,
        "metric_failed": 0,
        "not_attempted": 0,
        "metrics_completed": 10,
    }
    assert result["pose_recovery_rates"]["top1"] == {
        "numerator": 10,
        "denominator": 10,
        "rate": 1.0,
    }
    assert result["pose_recovery_rates"]["top5"]["denominator"] == 10
    assert result["holdout"]["selection_hash"]
    assert result["runtime"]["evaluator_mode"] == "INJECTED_TEST_EVALUATORS"
    assert result["runtime"]["evaluator_bindings"]["symmetry_rmsd"]["mode"] == "INJECTED"
    assert len(result["runtime"]["runtime_sha256"]) == 64
    assert result["metric_summaries"]["posebusters_energy_ratio"]["median"] == 1.7
    assert "not binding energies" in result["metric_summaries"]["energy_semantics"]
    assert all(case["status"] == "METRICS_COMPLETED" for case in result["cases"])


def test_failed_and_unattempted_redocks_remain_in_frozen_denominator(
    tmp_path: Path,
) -> None:
    manifest = _fixture_repo(tmp_path, failed={1}, unattempted={2})
    result = _build(tmp_path, manifest)

    assert result["gate_complete"] is False
    assert result["denominators"]["frozen"] == 10
    assert result["denominators"]["attempted"] == 9
    assert result["denominators"]["completed"] == 8
    assert result["denominators"]["failed"] == 1
    assert result["denominators"]["not_attempted"] == 1
    assert result["pose_recovery_rates"]["top1"] == {
        "numerator": 8,
        "denominator": 10,
        "rate": 0.8,
    }
    assert result["cases"][1]["status"] == "REDOCK_FAILED"
    assert result["cases"][2]["status"] == "NOT_ATTEMPTED"


def test_metric_failure_stays_in_denominator_and_blocks_gate(tmp_path: Path) -> None:
    manifest = _fixture_repo(tmp_path)

    def reject_one(**kwargs):
        if "case-03" in kwargs["comparison_ligand_path"].as_posix():
            raise ValueError("ProLIF inputs require explicit hydrogens")
        return _ifp(**kwargs)

    result = build_redock_regression(
        tmp_path,
        manifest,
        rmsd_evaluator=_rmsd,
        posebusters_evaluator=_posebusters,
        ifp_evaluator=reject_one,
    )

    assert result["gate_complete"] is False
    assert result["denominators"]["completed"] == 10
    assert result["denominators"]["metric_failed"] == 1
    assert result["denominators"]["metrics_completed"] == 9
    assert result["pose_recovery_rates"]["top1"]["denominator"] == 10
    assert result["pose_recovery_rates"]["top1"]["numerator"] == 9
    failed = next(case for case in result["cases"] if case["case_id"] == "case-03")
    assert failed["status"] == "METRIC_FAILED"
    assert "explicit hydrogens" in failed["failure"]["message"]
    assert any("strict mode" in text for text in result["scientific_boundaries"])


def test_top5_is_recomputed_from_first_five_bound_vina_records(tmp_path: Path) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    molecule = Chem.MolFromSmiles("CCO")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, (float(index), 0.0, 0.0))
    molecule.AddConformer(conformer)
    source = tmp_path / "six-records.sdf"
    writer = Chem.SDWriter(str(source))
    for _ in range(6):
        writer.write(molecule)
    writer.close()
    manifest = _fixture_repo(
        tmp_path,
        design="PILOT_RETROSPECTIVE",
        count=1,
        ligand_sdf=source.read_bytes(),
    )
    _rewrite_case_result(
        tmp_path,
        manifest,
        "case-00",
        {
            "top1_recovered": False,
            "top5_recovered": True,
            "top1": {
                **json.loads((tmp_path / "runs/case-00/result.json").read_text())["top1"],
                "symmetry_rmsd_angstrom": 3.0,
                "recovered": False,
            },
            "top5_oracle": {
                "evaluated_modes": 5,
                "best_mode": 2,
                "best_symmetry_rmsd_angstrom": 1.0,
                "any_pb_valid_and_rmsd_le_2": True,
                "first_recovered_mode": 2,
            },
        },
    )

    def ranked_rmsd(_native: Path, predicted: Path) -> float:
        return 1.0 if predicted.name.endswith("02.sdf") else 3.0

    result = build_redock_regression(
        tmp_path,
        manifest,
        rmsd_evaluator=ranked_rmsd,
        posebusters_evaluator=_posebusters,
        ifp_evaluator=_ifp,
    )

    recovery = result["cases"][0]["metrics"]["pose_recovery"]
    assert len(recovery["top5_mode_audits"]) == 5
    assert recovery["top1_success"] is False
    assert recovery["top5_oracle_success"] is True
    assert result["pose_recovery_rates"]["top1"]["numerator"] == 0
    assert result["pose_recovery_rates"]["top5"]["numerator"] == 1


def test_source_posebusters_claim_cannot_override_recomputation(tmp_path: Path) -> None:
    manifest = _fixture_repo(
        tmp_path,
        design="PILOT_RETROSPECTIVE",
        count=1,
    )

    def invalid_posebusters(*_args):
        value = _posebusters(Path(), Path(), Path())
        value["posebusters_valid"] = False
        value["posebusters_checks"] = {"geometry": False, "rmsd <= 2": True}
        return value

    result = build_redock_regression(
        tmp_path,
        manifest,
        rmsd_evaluator=_rmsd,
        posebusters_evaluator=invalid_posebusters,
        ifp_evaluator=_ifp,
    )

    assert result["cases"][0]["status"] == "METRIC_FAILED"
    assert "source top1 recovery differs" in result["cases"][0]["failure"]["message"]
    assert result["pose_recovery_rates"]["top1"]["numerator"] == 0
    assert result["pose_recovery_rates"]["top5"]["numerator"] == 0


def test_unbound_split_mode_file_is_not_a_scientific_input(tmp_path: Path) -> None:
    manifest = _fixture_repo(
        tmp_path,
        design="PILOT_RETROSPECTIVE",
        count=1,
    )
    split = tmp_path / "runs/case-00/artifacts/vina-mode-01.sdf"
    split.write_bytes(b"not a scientific input")

    result = _build(tmp_path, manifest)

    assert result["cases"][0]["status"] == "METRICS_COMPLETED"
    binding = result["cases"][0]["metrics"]["artifact_bindings"]
    assert binding["vina_poses_sdf"]["sha256"] == sha256_file(
        tmp_path / "runs/case-00/artifacts/vina-poses.sdf"
    )


def test_prolif_derivation_store_cannot_overlap_source_run(tmp_path: Path) -> None:
    manifest = _fixture_repo(
        tmp_path,
        design="PILOT_RETROSPECTIVE",
        count=1,
    )
    store = ArtifactStore(tmp_path / "runs/case-00/derived-prolif")

    with pytest.raises(RegressionIntegrityError, match="must not overlap"):
        build_redock_regression(
            tmp_path,
            manifest,
            rmsd_evaluator=_rmsd,
            posebusters_evaluator=_posebusters,
            ifp_evaluator=_ifp,
            prolif_artifact_store=store,
        )


def test_ifp_scores_cannot_disagree_with_interaction_denominators(
    tmp_path: Path,
) -> None:
    manifest = _fixture_repo(
        tmp_path,
        design="PILOT_RETROSPECTIVE",
        count=1,
    )

    def inconsistent_ifp(**_kwargs):
        value = _ifp()
        value["ifp_similarity"] = 0.99
        return value

    result = build_redock_regression(
        tmp_path,
        manifest,
        rmsd_evaluator=_rmsd,
        posebusters_evaluator=_posebusters,
        ifp_evaluator=inconsistent_ifp,
    )

    assert result["denominators"]["metric_failed"] == 1
    assert "differs from its interaction counts" in result["cases"][0]["failure"][
        "message"
    ]


def test_retrospective_pilot_is_explicit_and_never_completes_gate(
    tmp_path: Path,
) -> None:
    manifest = _fixture_repo(
        tmp_path,
        design="PILOT_RETROSPECTIVE",
        count=3,
    )
    result = _build(tmp_path, manifest)

    assert result["evaluation_design"] == "PILOT_RETROSPECTIVE"
    assert result["gate_complete"] is False
    assert result["holdout"] is None
    assert result["denominators"]["frozen"] == 3
    assert result["pose_recovery_rates"]["top1"]["denominator"] == 3


def test_absolute_traversal_and_hash_tamper_are_rejected(tmp_path: Path) -> None:
    manifest = _fixture_repo(tmp_path)
    with pytest.raises(RegressionIntegrityError, match="absolute"):
        _build(tmp_path, tmp_path / manifest)

    manifest_data = json.loads((tmp_path / manifest).read_text())
    manifest_data["cases"][0]["result"]["path"] = "../outside.json"
    body = {key: value for key, value in manifest_data.items() if key != "manifest_sha256"}
    manifest_data["manifest_sha256"] = sha256_bytes(canonical_json_bytes(body))
    _json(tmp_path / manifest, manifest_data)
    with pytest.raises(RegressionIntegrityError, match="traversal"):
        _build(tmp_path, manifest)

    tmp_path_2 = tmp_path / "hash-tamper"
    manifest_2 = _fixture_repo(tmp_path_2)
    result_path = tmp_path_2 / "runs" / "case-00" / "result.json"
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(RegressionIntegrityError, match="SHA-256 mismatch"):
        _build(tmp_path_2, manifest_2)


def test_artifact_and_holdout_hash_tamper_are_rejected(tmp_path: Path) -> None:
    manifest = _fixture_repo(tmp_path)
    poses = tmp_path / "runs" / "case-00" / "artifacts" / "vina-poses.sdf"
    poses.write_bytes(b"tampered")
    with pytest.raises(RegressionIntegrityError, match="vina_poses_sdf file SHA-256 mismatch"):
        _build(tmp_path, manifest)

    repo_2 = tmp_path / "holdout-tamper"
    manifest_2 = _fixture_repo(repo_2)
    holdout = repo_2 / "holdout.json"
    holdout.write_bytes(holdout.read_bytes() + b" ")
    with pytest.raises(RegressionIntegrityError, match="holdout manifest SHA-256 mismatch"):
        _build(repo_2, manifest_2)


def test_atomic_persist_checks_regression_hash(tmp_path: Path) -> None:
    manifest = _fixture_repo(tmp_path)
    result = _build(tmp_path, manifest)
    output = tmp_path / "outputs" / "regression-result.json"
    persist_redock_regression(result, output)

    assert json.loads(output.read_text()) == result
    result["mechanical_evaluation_complete"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        persist_redock_regression(result, tmp_path / "tampered.json")


def test_optional_prolif_addhs_is_hash_bound_and_receipted(tmp_path: Path) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    molecule = Chem.MolFromSmiles("CCO")
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 7
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    ligand_sdf = (Chem.MolToMolBlock(molecule) + "\n$$$$\n").encode()
    manifest = _fixture_repo(tmp_path, ligand_sdf=ligand_sdf)
    store = ArtifactStore(tmp_path / "derived-prolif-artifacts")

    def prepared_ifp(**kwargs):
        for name in ("docked_ligand_path", "comparison_ligand_path"):
            records = [
                item
                for item in Chem.SDMolSupplier(
                    str(kwargs[name]),
                    removeHs=False,
                )
                if item is not None
            ]
            assert len(records) == 1
            assert any(atom.GetAtomicNum() == 1 for atom in records[0].GetAtoms())
        return _ifp(**kwargs)

    result = build_redock_regression(
        tmp_path,
        manifest,
        rmsd_evaluator=_rmsd,
        posebusters_evaluator=_posebusters,
        ifp_evaluator=prepared_ifp,
        prolif_artifact_store=store,
    )

    assert result["mechanical_evaluation_complete"] is True
    assert result["gate_complete"] is False
    assert result["config"]["prolif_ligand_preparation_mode"] == (
        "RECEIPTED_LIGAND_ADDHS_AND_RECEPTOR_POCKET_CROP"
    )
    preparation = result["cases"][0]["metrics"]["prolif_ligand_preparation"]
    assert preparation["docked_ligand"]["hydrogens_added"] > 0
    assert preparation["native_reference"]["hydrogens_added"] > 0
    assert preparation["receptor"]["selected_residue_count"] == 1
    assert preparation["receptor"]["atom_identity_preserved"] is True
    assert preparation["receptor"]["coordinate_max_delta_angstrom"] <= 0.002
    assert preparation["docked_ligand"]["heavy_atom_max_coordinate_delta_angstrom"] <= 1e-6
    assert len(preparation["docked_ligand"]["prepared_ligand"]["sha256"]) == 64
    assert len(preparation["docked_ligand"]["preparation_receipt"]["sha256"]) == 64
    summary = preparation["native_reference"]["receipt_summary"]
    assert summary["method"] == "RDKit AddHs(addCoords=True)"
    assert isinstance(summary["rdkit_version"], str)
    assert summary["input_sha256"] == preparation["native_reference"]["input_ligand"]["sha256"]
    assert summary["output_sha256"] == preparation["native_reference"]["prepared_ligand"]["sha256"]
    assert summary["heavy_atom_identity_preserved"] is True
    assert summary["coordinate_tolerance_angstrom"] == 1e-6
    assert any("AddHs(addCoords=True)" in text for text in result["scientific_boundaries"])
    assert any("whole-residue union" in text for text in result["scientific_boundaries"])
