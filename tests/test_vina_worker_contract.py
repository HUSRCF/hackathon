from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.models import ArtifactRef
from protbind_agent.worker_protocol import (
    JsonSubprocessWorker,
    WorkerProvenance,
    WorkerRequest,
)


def _worker_module() -> ModuleType:
    path = Path(__file__).parents[1] / "workers" / "vina_worker.py"
    spec = importlib.util.spec_from_file_location("protbind_vina_worker_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def _fake_tools(tmp_path: Path, *, score_record: bool = True) -> tuple[Path, Path, Path, Path]:
    atom = "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  0.00  0.00     0.000 C"
    meeko_source = (
        "#!/usr/bin/env python3\n"
        "import pathlib,sys\n"
        "a=sys.argv[1:]\n"
        "if '--write_pdbqt' in a:\n"
        " o=pathlib.Path(a[a.index('--write_pdbqt')+1]); ligand=False\n"
        "else:\n"
        " o=pathlib.Path(a[a.index('-o')+1]); ligand=True\n"
        f"atom={atom!r}\n"
        "text=('ROOT\\n'+atom+'\\nENDROOT\\nTORSDOF 0\\n') if ligand else atom+'\\n'\n"
        "o.write_text(text)\n"
    )
    receptor = _write_executable(tmp_path / "fake-mk-receptor", meeko_source)
    ligand = _write_executable(tmp_path / "fake-mk-ligand", meeko_source)
    argv_record = tmp_path / "vina-argv.json"
    result_line = "REMARK VINA RESULT: -7.250 0.000 0.000\n" if score_record else ""
    vina_source = (
        "#!/usr/bin/env python3\n"
        "import json,pathlib,sys\n"
        "a=sys.argv[1:]\n"
        "if a == ['--version']:\n"
        " print('AutoDock Vina v1.2.7'); raise SystemExit(0)\n"
        f"pathlib.Path({str(argv_record)!r}).write_text(json.dumps(a))\n"
        "out=pathlib.Path(a[a.index('--out')+1])\n"
        f"atom={atom!r}\n"
        f"result={result_line!r}\n"
        "out.write_text('MODEL 1\\n'+result+'ROOT\\n'+atom+"
        "'\\nENDROOT\\nTORSDOF 0\\nENDMDL\\n')\n"
    )
    vina = _write_executable(tmp_path / "fake-vina", vina_source)
    return vina, receptor, ligand, argv_record


def _request(
    tmp_path: Path,
    monkeypatch,
    *,
    score_record: bool = True,
    corrupt_lineage: bool = False,
) -> tuple[ArtifactStore, WorkerRequest, dict[str, str], Path]:
    store = ArtifactStore(tmp_path / "workspace")
    receptor = store.put_bytes(b"fixture receptor\n", media_type="chemical/x-pdb", producer="test")
    structure = store.put_bytes(
        b"fixture cofold complex\n", media_type="chemical/x-mmcif", producer="test"
    )
    batch_candidate_id = "cofold-mol-a"
    upstream_candidate_id = "wrong-candidate" if corrupt_lineage else batch_candidate_id
    batch = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.cofold-input-batch",
            "receptor": receptor.to_dict(),
            "microstates": [
                {
                    "molecule_id": "mol-a",
                    "microstate_id": "state-1",
                    "canonical_isomeric_smiles": "C",
                    "heavy_element_counts": {"C": 1},
                }
            ],
            "quick_vina": {
                "evaluated": [
                    {
                        "molecule_id": "mol-a",
                        "microstate_id": "state-1",
                        "box_center": [1.0, 2.0, 3.0],
                        "box_size": [10.0, 11.0, 12.0],
                    }
                ]
            },
            "cofold_candidates": [
                {
                    "candidate_id": batch_candidate_id,
                    "molecule_id": "mol-a",
                    "microstate_id": "state-1",
                }
            ],
        },
        producer="test",
    )
    cofold = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.cofold-bundle",
            "candidates": [
                {
                    "candidate_id": upstream_candidate_id,
                    "molecule_id": "mol-a",
                    "microstate_id": "state-1",
                    "structure": structure.to_dict(),
                }
            ],
        },
        producer="test",
    )
    environment_lock = store.put_bytes(
        b"fixture lock\n", media_type="application/octet-stream", producer="test"
    )
    case = store.put_json({"schema_version": "1.0"}, producer="test")
    envelope = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.stage-input",
            "stage": "DOCKED",
            "case": case.to_dict(),
            "input_artifacts": {},
            "supporting_artifacts": {
                "support_openfold_batch": batch.to_dict(),
                "support_vina_environment_lock": environment_lock.to_dict(),
            },
            "previous": {
                "stage": "COFOLDED",
                "scientific_outputs": [cofold.to_dict(), structure.to_dict()],
                "receipt": None,
            },
        },
        producer="test",
    )
    vina, receptor_tool, ligand_tool, argv_record = _fake_tools(tmp_path, score_record=score_record)
    parameters = {
        "vina_executable": str(vina),
        "meeko_prepare_receptor_executable": str(receptor_tool),
        "meeko_prepare_ligand_executable": str(ligand_tool),
        "vina_version": "1.2.7",
        "meeko_version": "0.7.1",
        "rdkit_version": "fixture-rdkit-1",
        "gemmi_version": "fixture-gemmi-1",
        "numpy_version": "fixture-numpy-1",
        "scipy_version": "fixture-scipy-1",
        "cpu": 1,
        "exhaustiveness": 32,
        "num_modes": 9,
        "energy_range": 3.0,
    }
    module = _worker_module()
    monkeypatch.setenv("PROTBIND_TEST_RUNTIME", "1")
    attestation = module.runtime_asset_attestation(parameters)
    provenance = WorkerProvenance(
        model_revision=(
            "autodock-vina-1.2.7+meeko-0.7.1+"
            "rdkit-fixture-rdkit-1+gemmi-fixture-gemmi-1+"
            "numpy-fixture-numpy-1+scipy-fixture-scipy-1"
        ),
        weight_sha256=attestation["runtime_assets_sha256"],
        code_sha256=module.composite_code_sha256(
            environment_lock.sha256, attestation["runtime_assets_sha256"]
        ),
    )
    request = WorkerRequest(
        job_id="vina-contract",
        engine="vina",
        input=envelope,
        parameters=parameters,
        seed=17,
        provenance=provenance,
    )
    return store, request, {"PROTBIND_TEST_RUNTIME": "1"}, argv_record


def _run(store: ArtifactStore, request: WorkerRequest, environment: dict[str, str]):
    script = Path(__file__).parents[1] / "workers" / "vina_worker.py"
    return JsonSubprocessWorker(
        (sys.executable, str(script)),
        artifact_root=store.root,
        environment=environment,
    ).run(request)[0]


def test_vina_worker_binds_score_lineage_box_seed_and_runtime(tmp_path, monkeypatch) -> None:
    store, request, environment, argv_record = _request(tmp_path, monkeypatch)

    response = _run(store, request, environment)

    assert response.error is None
    assert response.peak_vram_bytes is None
    assert "fixture" in response.warnings[0]
    bundle = store.read_json(response.outputs[0])
    assert bundle["kind"] == "protbind.docking-bundle"
    assert bundle["candidate_count"] == 1
    assert bundle["failure_count"] == 0
    assert bundle["upstream_candidate_ids"] == ["cofold-mol-a"]
    assert bundle["failures"] == []
    candidate = bundle["candidates"][0]
    assert candidate["parent_candidate_id"] == "cofold-mol-a"
    assert candidate["microstate_id"] == "state-1"
    assert candidate["engine"] == "test-fixture-vina"
    assert candidate["seed"] == 17
    assert candidate["vina_score"] == -7.25
    assert candidate["box_center"] == [1.0, 2.0, 3.0]
    assert candidate["box_size"] == [10.0, 11.0, 12.0]
    assert "not an experimental binding free energy" in candidate["vina_score_semantics"]
    evidence = store.read_json(ArtifactRef.from_dict(candidate["evidence"]))
    assert evidence["metrics"]["score"] == candidate["vina_score"]
    assert evidence["metrics"]["cpu"] == 1
    assert evidence["inputs"]["receptor"] == bundle["receptor"]
    assert evidence["inputs"]["pose"] == candidate["pose"]
    assert candidate["pose"] == candidate["pose_sdf"]
    assert ArtifactRef.from_dict(candidate["pose"]).media_type == "chemical/x-mdl-sdfile"
    assert ArtifactRef.from_dict(candidate["pose_pdbqt"]).media_type == "chemical/x-pdbqt"
    assert ArtifactRef.from_dict(candidate["all_modes_sdf"]).media_type == ("chemical/x-mdl-sdfile")
    assert candidate["all_modes"] == candidate["all_modes_pdbqt"]
    pose_receipt = store.read_json(ArtifactRef.from_dict(candidate["pose_extraction_receipt"]))
    assert pose_receipt["schema_version"] == "2.0"
    assert pose_receipt["kind"] == "protbind.pose-extraction-receipt"
    assert pose_receipt["test_fixture"] is True
    assert pose_receipt["pose_sdf"] == candidate["pose_sdf"]
    assert pose_receipt["pose_pdbqt"] == candidate["pose_pdbqt"]
    assert pose_receipt["all_modes_sdf"] == candidate["all_modes_sdf"]
    assert pose_receipt["all_modes_pdbqt"] == candidate["all_modes_pdbqt"]
    assert all(pose_receipt["checks"].values())
    receptor_receipt = store.read_json(
        ArtifactRef.from_dict(bundle["receptor_preparation_receipt"])
    )
    assert receptor_receipt["schema_version"] == "2.0"
    assert receptor_receipt["kind"] == "protbind.receptor-preparation-receipt"
    assert receptor_receipt["receptor"] == bundle["receptor"]
    assert receptor_receipt["receptor_preparation_input"] == (bundle["receptor_preparation_input"])
    assert receptor_receipt["prepared_receptor"] == bundle["prepared_receptor"]
    assert all(receptor_receipt["checks"].values())
    metadata = store.read_json(ArtifactRef.from_dict(bundle["run_metadata"]))
    assert metadata["toolchain"]["official_runtime"] is False
    assert metadata["toolchain"]["trust_level"] == "test-fixture"
    assert metadata["execution"]["device"] == "cpu"
    argv = json.loads(argv_record.read_text(encoding="utf-8"))
    assert argv[argv.index("--seed") + 1] == "17"
    assert argv[argv.index("--cpu") + 1] == "1"
    assert argv[argv.index("--center_y") + 1] == "2.0"


def test_vina_worker_never_invents_score_when_tool_output_has_none(tmp_path, monkeypatch) -> None:
    store, request, environment, _ = _request(tmp_path, monkeypatch, score_record=False)

    response = _run(store, request, environment)

    assert response.error is None
    bundle = store.read_json(response.outputs[0])
    assert bundle["candidates"] == []
    assert bundle["failure_count"] == 1
    assert bundle["upstream_candidate_ids"] == ["cofold-mol-a"]
    assert bundle["failures"][0]["candidate_id"] == "cofold-mol-a"
    assert bundle["failures"][0]["error"]["code"] == "OUTPUT_INVALID"
    assert "score" not in bundle["failures"][0]


def test_vina_worker_rejects_cofold_lineage_not_in_frozen_batch(tmp_path, monkeypatch) -> None:
    store, request, environment, _ = _request(tmp_path, monkeypatch, corrupt_lineage=True)

    response = _run(store, request, environment)

    assert response.error is not None
    assert response.error.code == "INPUT_NOT_PREPARED"
    assert not response.outputs


def test_vina_worker_rejects_runtime_asset_hash_mismatch(tmp_path, monkeypatch) -> None:
    store, request, environment, _ = _request(tmp_path, monkeypatch)
    request = WorkerRequest(
        job_id=request.job_id,
        engine=request.engine,
        input=request.input,
        parameters=request.parameters,
        seed=request.seed,
        provenance=WorkerProvenance(
            model_revision=request.provenance.model_revision,
            weight_sha256="0" * 64,
            code_sha256=request.provenance.code_sha256,
        ),
    )

    response = _run(store, request, environment)

    assert response.error is not None
    assert response.error.code == "ASSET_HASH_MISMATCH"
    assert not response.outputs


def test_vina_pdbqt_invariants_ignore_only_coordinates_and_support_macrocycle_glue() -> None:
    module = _worker_module()
    first = (
        "ROOT\n"
        "ATOM      1  C1  LIG A   1       0.000   0.000   0.000"
        "  0.00  0.00     0.000 CG0\n"
        "ATOM      2  G1  LIG A   1       1.000   0.000   0.000"
        "  0.00  0.00     0.000 G0\n"
        "ENDROOT\nTORSDOF 0\n"
    )
    moved = first.replace("   0.000   0.000   0.000", "   2.000   3.000   4.000", 1)
    changed_charge = moved.replace("0.000 CG0", "0.125 CG0")
    signatures = module._pdbqt_signatures(first)

    assert module._pdbqt_heavy_elements(signatures) == {"C": 1}
    assert module._pdbqt_invariants(first) == module._pdbqt_invariants(moved)
    assert module._pdbqt_invariants(first) != module._pdbqt_invariants(changed_charge)


def test_meeko_identity_metadata_is_complete_and_fails_closed_on_corruption(
    tmp_path,
) -> None:
    module = _worker_module()
    atom = "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  0.00  0.00     0.000 C"
    prepared = (
        "REMARK SMILES C\n"
        "REMARK SMILES IDX 1 1\n"
        "REMARK INDEX MAP 1 1\n"
        "ROOT\n"
        f"{atom}\n"
        "ENDROOT\nTORSDOF 0\n"
    )
    metadata = module._meeko_identity_metadata(prepared)
    assert metadata["smiles"] == "C"
    assert metadata["smiles_index"] == [[1, 1]]
    signatures = module._pdbqt_signatures(prepared)
    invariants = module._pdbqt_invariants(prepared)
    output = tmp_path / "pose.pdbqt"
    output.write_text(
        "MODEL 1\n"
        "REMARK VINA RESULT: -1.000 0.000 0.000\n"
        + prepared.replace("REMARK SMILES C", "REMARK SMILES [C+]")
        + "ENDMDL\n",
        encoding="utf-8",
    )

    with pytest.raises(module.WorkerFailure, match="chemical-identity metadata"):
        module._parse_vina_output(output, signatures, invariants, metadata, 9)

    with pytest.raises(module.WorkerFailure, match="do not cover"):
        module._meeko_identity_metadata(
            prepared.replace("REMARK INDEX MAP 1 1", "REMARK INDEX MAP 1 2")
        )


def test_vina_worker_accepts_selection_bundle_without_cofold_structure(
    tmp_path, monkeypatch
) -> None:
    store, request, environment, _ = _request(tmp_path, monkeypatch)
    legacy_envelope = store.read_json(request.input)
    batch = store.read_json(
        ArtifactRef.from_dict(legacy_envelope["supporting_artifacts"]["support_openfold_batch"])
    )
    selection = store.put_json(
        {
            "schema_version": "2.0",
            "kind": "protbind.selection-bundle",
            "case_id": "selection-contract",
            "receptor": batch["receptor"],
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "selected-mol-a",
                    "molecule_id": "mol-a",
                    "microstate_id": "state-1",
                    "canonical_isomeric_smiles": "C",
                    "heavy_element_counts": {"C": 1},
                    "receptor": batch["receptor"],
                    "box_center": [1.0, 2.0, 3.0],
                    "box_size": [10.0, 11.0, 12.0],
                }
            ],
        },
        producer="test",
    )
    envelope = store.put_json(
        {
            "schema_version": "2.0",
            "kind": "protbind.stage-input",
            "stage": "DOCKED",
            "case_id": "selection-contract",
            "input_artifacts": {},
            "supporting_artifacts": legacy_envelope["supporting_artifacts"],
            "previous": {
                "stage": "SELECTED",
                "scientific_outputs": [selection.to_dict()],
                "receipt": None,
            },
        },
        producer="test",
    )
    selected_request = WorkerRequest(
        job_id=request.job_id,
        engine=request.engine,
        input=envelope,
        parameters=request.parameters,
        seed=request.seed,
        provenance=request.provenance,
    )

    response = _run(store, selected_request, environment)

    assert response.error is None
    bundle = store.read_json(response.outputs[0])
    assert bundle["schema_version"] == "2.0"
    assert bundle["upstream_selection_bundle"] == selection.to_dict()
    assert "upstream_cofold_bundle" not in bundle
    assert bundle["upstream_candidate_ids"] == ["selected-mol-a"]
    assert "cofold_structure" not in bundle["candidates"][0]
    evidence = store.read_json(ArtifactRef.from_dict(bundle["candidates"][0]["evidence"]))
    assert evidence["inputs"]["selection_bundle"] == selection.to_dict()
    assert "cofold_bundle" not in evidence["inputs"]
