from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.models import ArtifactRef
from protbind_agent.selection import (
    build_quick_vina_input,
    build_selection_preparation,
    validate_quick_vina_batch,
)
from protbind_agent.tripharm import build_jsonl_index
from protbind_agent.worker_protocol import (
    JsonSubprocessWorker,
    WorkerProvenance,
    WorkerRequest,
    WorkerResponse,
)
from protbind_agent.worker_sdk import WorkerFailure


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "workers" / "quick_vina_worker.py"
    spec = importlib.util.spec_from_file_location("protbind_quick_vina_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def _fake_tools(
    tmp_path: Path, *, score_record: bool = True
) -> tuple[Path, Path, Path, Path]:
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
    argv_record = tmp_path / "quick-vina-argv.json"
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
    monkeypatch: pytest.MonkeyPatch,
    *,
    score_record: bool = True,
    two_candidates: bool = False,
) -> tuple[ArtifactStore, object, WorkerRequest, Path]:
    for name in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROTBIND_TEST_RUNTIME", "1")
    store = ArtifactStore(tmp_path / "workspace")
    feature_file = tmp_path / "features.jsonl"
    records = [
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
        for molecule_id, smiles in (
            (("mol-a", "C"), ("mol-b", "N"))
            if two_candidates
            else (("mol-a", "C"),)
        )
    ]
    feature_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.sqlite"
    build_jsonl_index(feature_file, index_path)
    index = store.import_file(index_path, media_type="application/vnd.sqlite3")
    screen = store.put_json(
        {"hits": [{"molecule_id": item["molecule_id"]} for item in records]},
        producer="test-screen",
    )
    receptor = store.put_bytes(
        (
            b"ATOM      1  N   ALA A   1       1.000   2.000   3.000  "
            b"1.00 20.00           N  \n"
            b"ATOM      2  CA  ALA A   1       2.000   2.000   3.000  "
            b"1.00 20.00           C  \n"
            b"ATOM      3  C   ALA A   1       3.000   2.000   3.000  "
            b"1.00 20.00           C  \nTER\nEND\n"
        ),
        media_type="chemical/x-pdb",
        producer="test",
    )
    environment_lock = store.put_json(
        {"fixture": True}, producer="test-environment-lock"
    )
    preparation = build_selection_preparation(
        store,
        screening=screen,
        library_index=index,
        receptor=receptor,
        protein_chains=(("A", "ACDEFG"),),
        box_center=(1.0, 2.0, 3.0),
        box_size=(10.0, 11.0, 12.0),
    )
    quick_input = build_quick_vina_input(
        store, preparation, environment_lock, case_id="quick-contract"
    )
    vina, receptor_tool, ligand_tool, argv_record = _fake_tools(
        tmp_path, score_record=score_record
    )
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
        "exhaustiveness": 8,
        "num_modes": 1,
        "energy_range": 3.0,
    }
    module = _module()
    attestation = module.vina.runtime_asset_attestation(parameters)
    assets = attestation["runtime_assets_sha256"]
    provenance = WorkerProvenance(
        model_revision=module.quick_model_revision(parameters),
        weight_sha256=assets,
        code_sha256=module.composite_code_sha256(environment_lock.sha256, assets),
    )
    request = WorkerRequest(
        job_id="quick-vina-contract",
        engine="vina-quick",
        input=quick_input,
        parameters=parameters,
        seed=17,
        provenance=provenance,
    )
    context = {
        "preparation": preparation,
        "quick_input": quick_input,
    }
    return store, context, request, argv_record


def _run(store: ArtifactStore, request: WorkerRequest):
    script = Path(__file__).parents[1] / "workers" / "quick_vina_worker.py"
    return JsonSubprocessWorker(
        (sys.executable, str(script)),
        artifact_root=store.root,
        environment={"PROTBIND_TEST_RUNTIME": "1"},
    ).run(request)[0]


def test_quick_vina_worker_binds_cpu_profile_and_complete_output_closure(
    tmp_path, monkeypatch
) -> None:
    store, context, request, argv_record = _request(tmp_path, monkeypatch)

    response = _run(store, request)

    assert response.error is None
    evaluations = validate_quick_vina_batch(
        store,
        context["preparation"],
        context["quick_input"],
        response.outputs,
        case_id="quick-contract",
        seed=17,
    )
    assert evaluations[0]["status"] == "completed"
    assert evaluations[0]["score"] == -7.25
    batch = store.read_json(response.outputs[0])
    assert batch["kind"] == "protbind.quick-vina-evaluation-batch"
    assert batch["purpose"] == "selection-pruning-only"
    argv = json.loads(argv_record.read_text(encoding="utf-8"))
    assert argv[argv.index("--cpu") + 1] == "1"
    assert argv[argv.index("--exhaustiveness") + 1] == "8"
    assert argv[argv.index("--num_modes") + 1] == "1"
    assert argv[argv.index("--seed") + 1] == "17"


def test_quick_vina_worker_rejects_tampered_box_coordinate_frame(
    tmp_path, monkeypatch
) -> None:
    store, _, request, argv_record = _request(tmp_path, monkeypatch)
    quick_input = store.read_json(request.input)
    receipt_ref = ArtifactRef.from_dict(quick_input["docking_box_receipt"])
    receipt = store.read_json(receipt_ref)
    forged_receipt = store.put_json(
        {**receipt, "coordinate_frame": "unbound-world-frame"},
        producer=receipt_ref.producer,
        producer_version=receipt_ref.producer_version,
        source=receipt_ref.source,
    )
    forged_input = store.put_json(
        {**quick_input, "docking_box_receipt": forged_receipt.to_dict()},
        producer=request.input.producer,
        producer_version=request.input.producer_version,
        source=request.input.source,
    )
    forged_request = WorkerRequest(
        job_id=request.job_id,
        engine=request.engine,
        input=forged_input,
        parameters=request.parameters,
        seed=request.seed,
        provenance=request.provenance,
    )

    response = _run(store, forged_request)

    assert response.error is not None
    assert response.error.code == "INVALID_INPUT"
    assert "docking box receipt" in response.error.message
    assert not argv_record.exists()


def test_quick_vina_validator_rejects_inner_evidence_for_another_pose(
    tmp_path, monkeypatch
) -> None:
    store, context, request, _ = _request(tmp_path, monkeypatch)
    response = _run(store, request)
    assert response.error is None
    batch = store.read_json(response.outputs[0])
    evaluation = batch["evaluations"][0]
    outer_ref = ArtifactRef.from_dict(evaluation["evidence"])
    outer = store.read_json(outer_ref)
    inner_ref = ArtifactRef.from_dict(outer["inputs"]["inner_vina_evidence"])
    inner = store.read_json(inner_ref)
    wrong_pose = store.put_bytes(
        b"different pose\n",
        media_type="chemical/x-mdl-sdfile",
        producer="tamper-test",
    )
    inner["inputs"]["pose"] = wrong_pose.to_dict()
    changed_inner = store.put_json(inner, producer="tamper-test-inner")
    outer["inputs"]["inner_vina_evidence"] = changed_inner.to_dict()
    changed_outer = store.put_json(outer, producer="tamper-test-outer")
    evaluation["evidence"] = changed_outer.to_dict()
    changed_batch = store.put_json(batch, producer="tamper-test-batch")

    with pytest.raises(ValueError, match="inner Vina evidence"):
        validate_quick_vina_batch(
            store,
            context["preparation"],
            context["quick_input"],
            (changed_batch,),
            case_id="quick-contract",
            seed=17,
        )


def test_quick_vina_worker_fails_whole_attempt_on_invalid_tool_output(
    tmp_path, monkeypatch
) -> None:
    store, _, request, _ = _request(tmp_path, monkeypatch, score_record=False)

    response = _run(store, request)

    assert response.error is not None
    assert response.error.code == "OUTPUT_INVALID"
    assert response.error.recoverable is False
    assert not response.outputs


def test_partial_quick_vina_timeout_fails_the_whole_attempt(
    tmp_path, monkeypatch
) -> None:
    store, _, request, _ = _request(
        tmp_path,
        monkeypatch,
        two_candidates=True,
    )
    module = _module()

    def partial_inner(inner_request, inner_store):
        envelope = inner_store.read_json(inner_request.input)
        selection_ref = ArtifactRef.from_dict(
            envelope["previous"]["scientific_outputs"][0]
        )
        candidates = inner_store.read_json(selection_ref)["candidates"]
        first, second = candidates
        pose = inner_store.put_bytes(
            b"partial pose\n",
            media_type="chemical/x-mdl-sdfile",
            producer="partial-test",
        )
        evidence = inner_store.put_json({}, producer="partial-test")
        bundle = inner_store.put_json(
            {
                "schema_version": "2.0",
                "kind": "protbind.docking-bundle",
                "candidates": [
                    {
                        "parent_candidate_id": first["candidate_id"],
                        "molecule_id": first["molecule_id"],
                        "microstate_id": first["microstate_id"],
                        "pose": pose.to_dict(),
                        "evidence": evidence.to_dict(),
                        "vina_score": -7.0,
                        "vina_score_semantics": module.vina.SCORE_SEMANTICS,
                        "box_center": first["box_center"],
                        "box_size": first["box_size"],
                    }
                ],
                "failures": [
                    {
                        "candidate_id": second["candidate_id"],
                        "error": {
                            "code": "TOOL_TIMEOUT",
                            "message": "candidate command timed out",
                            "recoverable": True,
                        },
                    }
                ],
            },
            producer="partial-test",
        )
        return WorkerResponse(
            job_id=inner_request.job_id,
            engine=inner_request.engine,
            outputs=(bundle,),
            provenance=inner_request.provenance,
        )

    monkeypatch.setattr(module.vina, "_handler", partial_inner)

    with pytest.raises(WorkerFailure) as captured:
        module._handler(request, store)

    assert captured.value.code == "TOOL_TIMEOUT"
    assert captured.value.recoverable is True


@pytest.mark.parametrize(
    ("name", "value"),
    (("cpu", 2), ("exhaustiveness", 17), ("num_modes", 4)),
)
def test_quick_vina_worker_rejects_non_quick_profile(
    tmp_path, monkeypatch, name, value
) -> None:
    store, _, request, _ = _request(tmp_path, monkeypatch)
    parameters = {**request.parameters, name: value}
    changed = WorkerRequest(
        job_id=request.job_id,
        engine=request.engine,
        input=request.input,
        parameters=parameters,
        seed=request.seed,
        provenance=request.provenance,
    )

    response = _run(store, changed)

    assert response.error is not None
    assert response.error.code == "INVALID_PARAMETERS"
    assert not response.outputs


def test_quick_and_full_vina_inputs_are_not_interchangeable(
    tmp_path, monkeypatch
) -> None:
    store, _, request, _ = _request(tmp_path, monkeypatch)
    full_request = WorkerRequest(
        job_id=request.job_id,
        engine="vina",
        input=request.input,
        parameters=request.parameters,
        seed=request.seed,
        provenance=request.provenance,
    )
    full_script = Path(__file__).parents[1] / "workers" / "vina_worker.py"

    response = JsonSubprocessWorker(
        (sys.executable, str(full_script)),
        artifact_root=store.root,
        environment={"PROTBIND_TEST_RUNTIME": "1"},
    ).run(full_request)[0]

    assert response.error is not None
    assert response.error.code == "INVALID_INPUT"
    assert not response.outputs
