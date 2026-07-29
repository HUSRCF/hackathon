from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.manifest import CofoldStatus, RunState
from protbind_agent.models import (
    LigandHypothesis,
    PocketHypothesis,
    ResearchCase,
    ResearchMode,
    TargetSpec,
)
from protbind_agent.tripharm import build_jsonl_index
from protbind_agent.worker_protocol import WorkerProvenance
from protbind_agent.workflow import (
    PipelineConfig,
    PipelineStageError,
    ProtBindWorkflow,
    WorkerConfig,
    _gpu_lease,
    _worker_resource_lease,
)


def _feature_payload(offset: float = 0.0):
    return [
        {"type": "Donor", "position": [offset, 0, 0], "atom_indices": [0]},
        {"type": "Acceptor", "position": [offset + 3, 0, 0], "atom_indices": [1]},
        {"type": "Aromatic", "position": [offset, 4, 0], "atom_indices": [2]},
    ]


def test_screening_run_resumes_without_recomputing_and_degrades_explicitly(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    feature_file = tmp_path / "features.jsonl"
    records = [
        {
            "molecule_id": "mol-a",
            "smiles": "CCO",
            "conformers": [{"id": 0, "features": _feature_payload(10)}],
        },
        {
            "molecule_id": "mol-b",
            "smiles": "CCN",
            "conformers": [{"id": 0, "features": _feature_payload(-5)}],
        },
    ]
    feature_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    index = tmp_path / "library.sqlite"
    build_jsonl_index(feature_file, index)
    artifacts = ArtifactStore(workspace)
    query = artifacts.put_json(
        {"features": _feature_payload()}, producer="test-query"
    )
    case = ResearchCase(
        case_id="screen-case",
        target=TargetSpec(name="target", sequences=("ACDEFG",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=query),
    )
    workflow = ProtBindWorkflow(workspace)

    manifest = workflow.create(case, index, run_id="screen-run")
    receptor_path = tmp_path / "screen-receptor.pdb"
    receptor_path.write_text("fixture receptor\n", encoding="utf-8")
    workflow.attach_support(
        manifest,
        "receptor_structure",
        receptor_path,
        media_type="chemical/x-pdb",
    )
    manifest = workflow.run(manifest, stop_after=RunState.SCREENED)
    screen_ref = manifest.stage_records[RunState.SCREENED.value].outputs[0]
    screen = artifacts.read_json(screen_ref)

    assert manifest.state is RunState.SCREENED
    assert [hit["molecule_id"] for hit in screen["hits"]] == ["mol-a", "mol-b"]
    resumed = workflow.run(workflow.manifests.load("screen-run"), stop_after=RunState.SCREENED)
    assert resumed.stage_records[RunState.SCREENED.value].outputs[0] == screen_ref
    with pytest.raises(ValueError, match="does not match current input/config"):
        ProtBindWorkflow(
            workspace, config=PipelineConfig(screen_top_k=1)
        ).run(workflow.manifests.load("screen-run"), stop_after=RunState.REPORTED)
    degraded = workflow.run(resumed, stop_after=RunState.REPORTED)
    assert degraded.state is RunState.DEGRADED
    assert degraded.last_completed_stage is RunState.SCREENED
    assert degraded.failures[-1].code == "INPUT_NOT_PREPARED"
    assert "degraded_report" in degraded.artifacts
    report = artifacts.read_bytes(degraded.artifacts["degraded_report"]).decode()
    assert "No missing scientific result was imputed" in report
    assert str(tmp_path) not in report


def test_explicit_fixture_workers_can_complete_state_machine(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    feature_file = tmp_path / "features.jsonl"
    feature_file.write_text(
        json.dumps(
            {
                "molecule_id": "mol-a",
                "smiles": "CCO",
                "conformers": [{"id": 0, "features": _feature_payload(3)}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = tmp_path / "library.sqlite"
    build_jsonl_index(feature_file, index)
    artifacts = ArtifactStore(workspace)
    query = artifacts.put_json({"features": _feature_payload()}, producer="test-query")
    case = ResearchCase(
        case_id="complete-case",
        target=TargetSpec(name="target", sequences=("ACDEFG",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=query),
    )
    worker_script = tmp_path / "fixture_worker.py"
    worker_script.write_text(
        "import hashlib,json,os,pathlib,sys\n"
        "r=json.loads(sys.stdin.readline())\n"
        "root=pathlib.Path(os.environ['PROTBIND_ARTIFACT_ROOT'])\n"
        "def put(b,media,producer):\n"
        " h=hashlib.sha256(b).hexdigest();d=root/'objects'/h[:2];"
        "d.mkdir(parents=True,exist_ok=True);(d/h[2:]).write_bytes(b);"
        "return {'sha256':h,'media_type':media,'size_bytes':len(b),"
        "'producer':producer,'producer_version':'1',"
        "'source':None,'license':None}\n"
        "def get(ref):\n"
        " p=root/'objects'/ref['sha256'][:2]/ref['sha256'][2:];"
        "return json.loads(p.read_text())\n"
        "envelope=get(r['input']);previous=get(envelope['previous']['scientific_outputs'][0])\n"
        "if r['engine']=='validate-fixture':\n"
        " up=previous['candidates'][0];dock=up['pose'];"
        "cid=up['candidate_id']\n"
        " specs=[('posebusters',{'valid':True},{'docked_pose':dock}),"
        "('vina',{'pose_valid':True},{'pose':dock})]\n"
        " evidence=[put(json.dumps({'schema_version':'1.0',"
        "'kind':'protbind.tool-evidence','tool':tool,'molecule_id':'mol-a',"
        "'candidate_id':cid,'metrics':metrics,'inputs':inputs},"
        "sort_keys=True,separators=(',',':')).encode(),"
        "'application/json','fixture-'+tool) for tool,metrics,inputs in specs]\n"
        " p={'schema_version':'1.0','kind':'protbind.validation-bundle',"
        "'candidates':[{'candidate_id':cid,'molecule_id':'mol-a',"
        "'microstate_id':up['microstate_id'],'docked_pose':dock,"
        "'engine':'fixture-validation','seed':r['seed'],'has_reference_pose':False,"
        "'decision_reason':'fixture geometry only; not scientific evidence',"
        "'bundle':{'posebusters_valid':True,'vina_pose_valid':True,"
        "'evidence':evidence}}]}\n"
        " extras=evidence\n"
        "elif r['engine']=='cofold-fixture':\n"
        " raw=put(('fixture-'+r['engine']).encode(),'chemical/x-pdb',"
        "'test-fixture-worker')\n"
        " p={'schema_version':'1.0','kind':'protbind.cofold-bundle',"
        "'candidates':[{'candidate_id':'cofold-mol-a','molecule_id':'mol-a',"
        "'microstate_id':'state-1','engine':'fixture-cofold','seed':r['seed'],"
        "'structure':raw,"
        "'confidence_name':'fixture_confidence','confidence_value':0.5,"
        "'confidence_semantics':'fixture model confidence; not binding affinity'}]}\n"
        " extras=[raw]\n"
        "else:\n"
        " up=previous['candidates'][0]\n"
        " raw=put(('fixture-'+r['engine']).encode(),'chemical/x-pdb',"
        "'test-fixture-worker')\n"
        " p={'schema_version':'1.0','kind':'protbind.docking-bundle',"
        "'candidates':[{'candidate_id':'dock-mol-a','molecule_id':'mol-a',"
        "'parent_candidate_id':up['candidate_id'],'microstate_id':up['microstate_id'],"
        "'engine':'fixture-vina','seed':r['seed'],"
        "'pose':raw,'vina_score':-1.0,"
        "'vina_score_semantics':'tool score only; not an experimental binding free energy',"
        "'box_center':[0,0,0],'box_size':[10,10,10]}]}\n"
        " extras=[raw]\n"
        "b=json.dumps(p,sort_keys=True,separators=(',',':')).encode()\n"
        "ref=put(b,'application/json','test-fixture-worker')\n"
        "print(json.dumps({'schema_version':'1.0','job_id':r['job_id'],"
        "'engine':r['engine'],'outputs':[ref]+extras,'provenance':r['provenance'],"
        "'timings_seconds':{'fixture':0.001},"
        "'peak_vram_bytes':0,'warnings':['fixture output; not scientific evidence']}))\n",
        encoding="utf-8",
    )
    provenance = WorkerProvenance(
        model_revision="fixture-only",
        weight_sha256="a" * 64,
        code_sha256="b" * 64,
    )
    workers = {
        RunState.DOCKED: WorkerConfig(
            engine="dock-fixture",
            argv=(sys.executable, str(worker_script)),
            provenance=provenance,
            isolate_network=False,
            allow_unisolated_test_fixture=True,
        ),
        RunState.VALIDATED: WorkerConfig(
            engine="validate-fixture",
            argv=(sys.executable, str(worker_script)),
            provenance=provenance,
            isolate_network=False,
            allow_unisolated_test_fixture=True,
        ),
    }
    workflow = ProtBindWorkflow(workspace, config=PipelineConfig(workers=workers))

    manifest = workflow.create(case, index, run_id="complete-run")
    receptor_path = tmp_path / "receptor.pdb"
    receptor_path.write_text("fixture receptor; not a scientific structure\n", encoding="utf-8")
    receptor_ref = workflow.attach_support(
        manifest,
        "receptor_structure",
        receptor_path,
        media_type="chemical/x-pdb",
    )
    manifest = workflow.run(manifest, stop_after=RunState.SCREENED)
    screen_ref = manifest.stage_records[RunState.SCREENED.value].outputs[0]
    quick_pose = artifacts.put_bytes(
        b"fixture quick Vina pose",
        media_type="chemical/x-pdbqt",
        producer="fixture-vina",
    )
    vina_evidence = artifacts.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.tool-evidence",
            "tool": "vina",
            "molecule_id": "mol-a",
            "microstate_id": "state-1",
            "metrics": {
                "score": -1.0,
                "box_center": [0.0, 0.0, 0.0],
                "box_size": [10.0, 10.0, 10.0],
            },
            "inputs": {
                "receptor": receptor_ref.to_dict(),
                "pose": quick_pose.to_dict(),
            },
        },
        producer="fixture-vina",
    )
    support = tmp_path / "openfold-batch.json"
    support.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "protbind.cofold-input-batch",
                "screening_artifact": screen_ref.to_dict(),
                "library_index": manifest.input_artifacts["library_index"].to_dict(),
                "receptor": receptor_ref.to_dict(),
                "protein_chains": [{"chain_id": "A", "sequence": "ACDEFG"}],
                "diversity": {
                    "method": "Bemis-Murcko",
                    "input_molecule_ids": ["mol-a"],
                    "retained": [
                        {"molecule_id": "mol-a", "scaffold_smiles": "CCO"}
                    ],
                },
                "microstates": [
                    {
                        "molecule_id": "mol-a",
                        "microstate_id": "state-1",
                        "canonical_isomeric_smiles": "CCO",
                        "parent_standardized_smiles": "CCO",
                        "heavy_element_counts": {"C": 2, "O": 1},
                    }
                ],
                "quick_vina": {
                    "evaluated": [
                        {
                            "molecule_id": "mol-a",
                            "microstate_id": "state-1",
                            "score": -1.0,
                            "score_semantics": (
                                "tool score only; not an experimental binding free energy"
                            ),
                            "pose": quick_pose.to_dict(),
                            "box_center": [0.0, 0.0, 0.0],
                            "box_size": [10.0, 10.0, 10.0],
                            "evidence": vina_evidence.to_dict(),
                        }
                    ],
                    "retained_molecule_ids": ["mol-a"],
                },
                "cofold_candidates": [
                    {
                        "candidate_id": "cofold-mol-a",
                        "molecule_id": "mol-a",
                        "microstate_id": "state-1",
                        "canonical_isomeric_smiles": "CCO",
                        "heavy_element_counts": {"C": 2, "O": 1},
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    support_ref = workflow.attach_support(
        manifest,
        "openfold_batch",
        support,
        media_type="application/json",
    )
    manifest = workflow.run(manifest)

    assert manifest.state is RunState.REPORTED
    assert manifest.cofold_status is CofoldStatus.NOT_REQUESTED
    previous_stage = {
        RunState.DOCKED: RunState.SELECTED,
        RunState.VALIDATED: RunState.DOCKED,
    }
    for stage, previous in previous_stage.items():
        envelope_ref = manifest.artifacts[f"worker_input_{stage.value.lower()}"]
        envelope = artifacts.read_json(envelope_ref)
        assert envelope["kind"] == "protbind.stage-input"
        assert envelope["stage"] == stage.value
        assert envelope["case_id"] == manifest.case_id
        assert "case" not in envelope
        assert envelope["previous"]["stage"] == previous.value
        assert envelope["previous"]["scientific_outputs"]
        assert envelope["input_artifacts"] == {}
        assert "query_ligand" not in envelope["supporting_artifacts"]
        if stage is RunState.DOCKED:
            assert (
                envelope["supporting_artifacts"]["support_openfold_batch"]
                == support_ref.to_dict()
            )
        else:
            assert "support_openfold_batch" not in envelope["supporting_artifacts"]
        if previous is RunState.DOCKED:
            assert envelope["previous"]["receipt"] is not None
        else:
            assert envelope["previous"]["receipt"] is None
    report = artifacts.read_bytes(manifest.artifacts["report_markdown"]).decode()
    assert "HYPOTHESIS_ONLY" in report
    assert "not experimental binding free energies" in report
    assert str(tmp_path) not in report
    with pytest.raises(ValueError, match="frozen"):
        workflow.attach_support(
            manifest,
            "late_input",
            support,
            media_type="application/json",
        )


def _automatic_selection_case(
    tmp_path,
    *,
    explicit_box: bool,
    all_fail: bool = False,
    transient_fail_once: bool = False,
    box_size: tuple[float, float, float] = (10.0, 11.0, 12.0),
):
    workspace = tmp_path / "workspace"
    feature_file = tmp_path / "features.jsonl"
    feature_file.write_text(
        json.dumps(
            {
                "molecule_id": "mol-a",
                "smiles": "CCO",
                "conformers": [{"id": 0, "features": _feature_payload()}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = tmp_path / "library.sqlite"
    build_jsonl_index(feature_file, index)
    artifacts = ArtifactStore(workspace)
    query = artifacts.put_json({"features": _feature_payload()}, producer="test-query")
    pocket = (
        PocketHypothesis(
            center=(1.0, 2.0, 3.0),
            box_size=box_size,
            pharmacophore=query,
        )
        if explicit_box
        else None
    )
    case = ResearchCase(
        case_id="automatic-selection-case",
        target=TargetSpec(name="target", sequences=("ACDEFG",)),
        mode=(ResearchMode.POCKET_ONLY if explicit_box else ResearchMode.LIGAND_ONLY),
        pocket=pocket,
        ligand=(None if explicit_box else LigandHypothesis(pharmacophore=query)),
        seed=17,
    )
    marker = tmp_path / "quick-worker-count.txt"
    fixture_worker = (
        Path(__file__).parent / "fixtures" / "quick_selection_worker.py"
    )
    quick = WorkerConfig(
        engine="quick-selection-fixture",
        argv=(sys.executable, str(fixture_worker)),
        provenance=WorkerProvenance(
            model_revision="fixture-only",
            weight_sha256="a" * 64,
            code_sha256="b" * 64,
        ),
        parameters={
            "fixture_marker": str(marker),
            "fixture_all_fail": all_fail,
            "fixture_transient_fail_once": transient_fail_once,
        },
        isolate_network=False,
        allow_unisolated_test_fixture=True,
    )
    workflow = ProtBindWorkflow(
        workspace,
        config=PipelineConfig(workers={RunState.SELECTED: quick}),
    )
    manifest = workflow.create(case, index, run_id="automatic-selection-run")
    receptor_path = tmp_path / "receptor.pdb"
    serial = 1
    lines = []
    for residue_index, residue_name in enumerate(
        ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY"), start=1
    ):
        for atom_offset, (atom_name, element) in enumerate(
            (("N", "N"), ("CA", "C"), ("C", "C"))
        ):
            x = float(residue_index - 1) + atom_offset * 0.2
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} "
                f"A{residue_index:4d}    {x:8.3f}{2.0:8.3f}{3.0:8.3f}"
                f"  1.00 20.00          {element:>2s}\n"
            )
            serial += 1
    receptor_path.write_text("".join(lines) + "TER\nEND\n", encoding="utf-8")
    workflow.attach_support(
        manifest,
        "receptor_structure",
        receptor_path,
        media_type="chemical/x-pdb",
    )
    lock_path = tmp_path / "vina-environment-lock.json"
    lock_path.write_text('{"fixture":true}\n', encoding="utf-8")
    workflow.attach_support(
        manifest,
        "vina_environment_lock",
        lock_path,
        media_type="application/json",
    )
    return workflow, manifest, marker


def test_automatic_quick_vina_completes_selected_and_is_idempotent(tmp_path) -> None:
    workflow, manifest, marker = _automatic_selection_case(
        tmp_path, explicit_box=True
    )

    selected = workflow.run(manifest, stop_after=RunState.SELECTED)

    assert selected.state is RunState.SELECTED
    assert marker.read_text(encoding="utf-8") == "1"
    assert {
        "selection_preparation",
        "worker_input_selected",
        "selection_quick_vina_bundle",
        "selection_quick_vina_receipt",
        "selection_bundle",
        "support_selection_batch",
    }.issubset(selected.artifacts)
    record = selected.stage_records[RunState.SELECTED.value]
    assert record.outputs == (
        selected.artifacts["selection_bundle"],
        selected.artifacts["selection_quick_vina_receipt"],
    )
    selection = workflow.artifacts.read_json(record.outputs[0])
    assert selection["automatic_orchestration"]["worker_receipt"] == (
        record.outputs[1].to_dict()
    )
    assert selection["candidates"][0]["receptor"] == selection["receptor"]
    cached = workflow.run(
        workflow.manifests.load(selected.run_id), stop_after=RunState.SELECTED
    )
    assert cached.stage_records[RunState.SELECTED.value].outputs == record.outputs
    assert marker.read_text(encoding="utf-8") == "1"


def test_automatic_selection_rejects_tampered_cached_receipt_contract(tmp_path) -> None:
    workflow, manifest, _ = _automatic_selection_case(tmp_path, explicit_box=True)
    selected = workflow.run(manifest, stop_after=RunState.SELECTED)
    receipt = selected.artifacts["selection_quick_vina_receipt"]
    quick_input = selected.artifacts["worker_input_selected"]
    batch = selected.artifacts["selection_quick_vina_bundle"]
    receipt_value = workflow.artifacts.read_json(receipt)
    worker_config = workflow.config.workers[RunState.SELECTED]

    for field, changed_value in (
        ("job_id", "another-job"),
        ("case_id", "another-case"),
        ("output_contract", "another-contract"),
        ("network_isolation", "another-isolation"),
        ("gpu_lease_device", "0"),
        ("peak_vram_bytes", 1),
        ("end_to_end_seconds", -1.0),
        ("timings_seconds", {"worker": -1.0}),
        ("warnings", [1]),
        ("scientific_scope", "evidence-grade"),
    ):
        forged_value = {**receipt_value, field: changed_value}
        forged = workflow.artifacts.put_json(
            forged_value,
            producer=receipt.producer,
            producer_version=receipt.producer_version,
            source=receipt.source,
        )
        with pytest.raises(ValueError, match="differs from current configuration"):
            workflow._quick_selection_receipt_outputs(
                forged,
                expected_job_id=f"{selected.run_id}-selected-quick-vina",
                expected_case_id="automatic-selection-case",
                quick_input=quick_input,
                batch=batch,
                worker_config=worker_config,
            )


def test_automatic_selection_without_explicit_box_never_launches_worker(tmp_path) -> None:
    workflow, manifest, marker = _automatic_selection_case(
        tmp_path, explicit_box=False
    )

    degraded = workflow.run(manifest, stop_after=RunState.SELECTED)

    assert degraded.state is RunState.DEGRADED
    assert degraded.last_completed_stage is RunState.SCREENED
    assert degraded.failures[-1].code == "SITE_DISCOVERY_UNAVAILABLE"
    assert "whole-protein box was guessed" in degraded.failures[-1].message
    assert not marker.exists()


@pytest.mark.parametrize(
    ("box_size", "message"),
    (
        ((3.99, 10.0, 10.0), "degenerate"),
        ((31.0, 31.0, 29.0), "implausibly broad"),
    ),
)
def test_automatic_selection_rejects_implausible_box_before_worker_launch(
    tmp_path, box_size, message
) -> None:
    workflow, manifest, marker = _automatic_selection_case(
        tmp_path,
        explicit_box=True,
        box_size=box_size,
    )

    failed = workflow.run(manifest, stop_after=RunState.SELECTED)

    assert failed.state is RunState.FAILED
    assert failed.last_completed_stage is RunState.SCREENED
    assert failed.failures[-1].code == "INPUT_NOT_PREPARED"
    assert message in failed.failures[-1].message
    assert not marker.exists()
    assert "worker_input_selected" not in failed.artifacts


def test_all_failed_quick_vina_is_cached_and_never_fabricates_selection(tmp_path) -> None:
    workflow, manifest, marker = _automatic_selection_case(
        tmp_path, explicit_box=True, all_fail=True
    )

    degraded = workflow.run(manifest, stop_after=RunState.SELECTED)

    assert degraded.state is RunState.DEGRADED
    assert degraded.failures[-1].code == "NO_SELECTABLE_CANDIDATES"
    assert "selection_bundle" not in degraded.artifacts
    assert "selection_quick_vina_bundle" in degraded.artifacts
    assert marker.read_text(encoding="utf-8") == "1"
    repeated = workflow.run(
        workflow.manifests.load(degraded.run_id), stop_after=RunState.SELECTED
    )
    assert repeated.state is RunState.DEGRADED
    assert repeated.failures[-1].code == "NO_SELECTABLE_CANDIDATES"
    assert marker.read_text(encoding="utf-8") == "1"


def test_transient_quick_vina_failure_is_not_cached_and_resume_retries(tmp_path) -> None:
    workflow, manifest, marker = _automatic_selection_case(
        tmp_path,
        explicit_box=True,
        transient_fail_once=True,
    )

    degraded = workflow.run(manifest, stop_after=RunState.SELECTED)

    assert degraded.state is RunState.DEGRADED
    assert degraded.failures[-1].code == "TOOL_TIMEOUT"
    assert "selection_quick_vina_bundle" not in degraded.artifacts
    assert "selection_quick_vina_receipt" not in degraded.artifacts
    assert marker.read_text(encoding="utf-8") == "1"
    resumed = workflow.run(
        workflow.manifests.load(degraded.run_id), stop_after=RunState.SELECTED
    )
    assert resumed.state is RunState.SELECTED
    assert marker.read_text(encoding="utf-8") == "2"


def test_production_quick_vina_policy_rejects_gpu_or_evidence_grade_profile() -> None:
    provenance = WorkerProvenance(
        model_revision="selection-quick-vina-1.0+fixture",
        weight_sha256="a" * 64,
        code_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="cpu=1"):
        WorkerConfig(
            engine="vina-quick",
            argv=(sys.executable, "worker.py"),
            provenance=provenance,
            parameters={"cpu": 2},
        )
    with pytest.raises(ValueError, match="forbids GPU masks"):
        WorkerConfig(
            engine="vina-quick",
            argv=(sys.executable, "worker.py"),
            provenance=provenance,
            environment={"HIP_VISIBLE_DEVICES": "0"},
        )
    with pytest.raises(ValueError, match="pinned vina-quick"):
        PipelineConfig(
            workers={
                RunState.SELECTED: WorkerConfig(
                    engine="vina",
                    argv=(sys.executable, "worker.py"),
                    provenance=provenance,
                )
            }
        )


def test_worker_config_identity_binds_environment_without_exposing_it() -> None:
    provenance = WorkerProvenance(
        model_revision="fixture-only",
        weight_sha256="a" * 64,
        code_sha256="b" * 64,
    )
    first = WorkerConfig(
        engine="fixture",
        argv=(sys.executable, "worker.py"),
        provenance=provenance,
        environment={"MODEL_CACHE": "/internal/cache-a"},
    )
    second = WorkerConfig(
        engine="fixture",
        argv=(sys.executable, "worker.py"),
        provenance=provenance,
        environment={"MODEL_CACHE": "/internal/cache-b"},
    )

    assert first.identity_hash != second.identity_hash
    assert "/internal" not in first.identity_hash
    with pytest.raises(ValueError, match="forbidden"):
        WorkerConfig(
            engine="fixture",
            argv=(sys.executable, "worker.py"),
            provenance=provenance,
            environment={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
        )
    with pytest.raises(ValueError, match="exactly one numeric"):
        WorkerConfig(
            engine="openfold3",
            argv=(sys.executable, "worker.py"),
            provenance=provenance,
            environment={"HIP_VISIBLE_DEVICES": "0,1"},
        )
    with pytest.raises(ValueError, match="canonical form"):
        WorkerConfig(
            engine="openfold3",
            argv=(sys.executable, "worker.py"),
            provenance=provenance,
            environment={"HIP_VISIBLE_DEVICES": "00"},
        )
    with pytest.raises(ValueError, match="must use HIP_VISIBLE_DEVICES only"):
        WorkerConfig(
            engine="openfold3",
            argv=(sys.executable, "worker.py"),
            provenance=provenance,
            environment={
                "HIP_VISIBLE_DEVICES": "0",
                "ROCR_VISIBLE_DEVICES": "1",
            },
        )
    with pytest.raises(ValueError, match="reserved for protocol tests"):
        WorkerConfig(
            engine="fixture",
            argv=(sys.executable, "worker.py"),
            provenance=provenance,
            environment={"PROTBIND_TEST_RUNTIME": "1"},
        )
    with pytest.raises(ValueError, match="pinned openfold3 adapter"):
        PipelineConfig(
            workers={
                RunState.COFOLDED: WorkerConfig(
                    engine="openfold3-typo",
                    argv=(sys.executable, "worker.py"),
                    provenance=provenance,
                    environment={"HIP_VISIBLE_DEVICES": "0"},
                )
            }
        )
    with pytest.raises(ValueError, match="forbidden"):
        WorkerConfig(
            engine="fixture",
            argv=(sys.executable, "worker.py"),
            provenance=provenance,
            environment={"REMOTE_API_KEY": "secret"},
        )


@pytest.mark.parametrize(
    "parameters",
    (
        {"low_mem": False},
        {"use_triton_triangle_kernels": False},
        {"use_msa_server": True},
    ),
)
def test_production_openfold_profile_is_rejected_before_launch(
    parameters: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="before launch"):
        WorkerConfig(
            engine="openfold3",
            argv=(sys.executable, "worker.py"),
            provenance=WorkerProvenance(
                model_revision="fixture-only",
                weight_sha256="a" * 64,
                code_sha256="b" * 64,
            ),
            parameters=parameters,
            environment={"HIP_VISIBLE_DEVICES": "0"},
        )


def test_openfold_launch_profile_is_rechecked_after_parameter_mutation() -> None:
    worker = WorkerConfig(
        engine="openfold3",
        argv=(sys.executable, "worker.py"),
        provenance=WorkerProvenance(
            model_revision="fixture-only",
            weight_sha256="a" * 64,
            code_sha256="b" * 64,
        ),
        environment={"HIP_VISIBLE_DEVICES": "0"},
    )
    worker.parameters["low_mem"] = False

    with pytest.raises(ValueError, match="before launch"):
        worker.validate_launch_profile()


def test_gpu_lease_rejects_two_workers_on_the_same_device(tmp_path) -> None:
    with (
        _gpu_lease(tmp_path / "workspace-a", "0"),
        pytest.raises(PipelineStageError, match="leased") as captured,
        _gpu_lease(tmp_path / "workspace-b", "0"),
    ):
        pass
    assert captured.value.code == "GPU_BUSY"
    assert captured.value.recoverable is True


def test_openfold_global_lease_rejects_a_second_device_across_workspaces(
    tmp_path,
) -> None:
    with (
        _worker_resource_lease(tmp_path / "workspace-a", "openfold3", "0"),
        pytest.raises(PipelineStageError, match="already running") as captured,
        _worker_resource_lease(tmp_path / "workspace-b", "openfold3", "1"),
    ):
        pass
    assert captured.value.code == "OPENFOLD_BUSY"
    assert captured.value.recoverable is True
