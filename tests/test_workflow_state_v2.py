from __future__ import annotations

from types import SimpleNamespace

import pytest

import protbind_agent.workflow as workflow_module
from protbind_agent.artifacts import ArtifactStore
from protbind_agent.manifest import RunManifest, RunState, StageRecord
from protbind_agent.worker_protocol import WorkerProvenance
from protbind_agent.workflow import (
    PipelineStageError,
    ProtBindWorkflow,
    WorkerConfig,
    _worker_input_payload,
)


def _manifest_through(
    store: ArtifactStore,
    stage: RunState,
    *,
    docking_bundle=None,
) -> RunManifest:
    case = store.put_json({"case_id": "case"}, producer="test")
    manifest = RunManifest(run_id="run", case_id="case", case_artifact=case)
    order = (
        RunState.INPUT_VALIDATED,
        RunState.RECEPTOR_READY,
        RunState.INDEXED,
        RunState.SCREENED,
        RunState.SELECTED,
        RunState.DOCKED,
        RunState.VALIDATED,
    )
    for index, current in enumerate(order, start=1):
        if current is RunState.DOCKED and docking_bundle is not None:
            output = docking_bundle
        else:
            output = store.put_json(
                {"stage": current.value}, producer=f"test-{current.value.lower()}"
            )
        manifest.complete_stage(
            StageRecord.create(
                current,
                input_hash=f"{index:x}" * 64,
                config_hash=f"{index + 7:x}" * 64,
                outputs=(output,),
                duration_seconds=0,
            )
        )
        if current is stage:
            break
    return manifest


def _docking_bundle(store: ArtifactStore):
    receptor = store.put_bytes(
        b"ATOM fixture\n", media_type="chemical/x-pdb", producer="test"
    )
    pose = store.put_bytes(
        b"fixture SDF\n",
        media_type="chemical/x-mdl-sdfile",
        producer="test-vina",
    )
    docking = store.put_json(
        {
            "schema_version": "2.0",
            "kind": "protbind.docking-bundle",
            "receptor": receptor.to_dict(),
            "candidates": [
                {
                    "candidate_id": "vina-selected-a",
                    "molecule_id": "a",
                    "microstate_id": "s1",
                    "pose": pose.to_dict(),
                    "pose_sdf": pose.to_dict(),
                }
            ],
        },
        producer="test-vina",
    )
    return docking


def test_validation_support_is_auto_built_from_vina_only_docking(
    tmp_path, monkeypatch
) -> None:
    workflow = ProtBindWorkflow(tmp_path / "workspace")
    store = workflow.artifacts
    docking = _docking_bundle(store)
    manifest = _manifest_through(store, RunState.DOCKED, docking_bundle=docking)
    provenance = WorkerProvenance(
        model_revision="validation-toolchain:fixture",
        weight_sha256="a" * 64,
        code_sha256="b" * 64,
    )
    toolchain = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.validation-toolchain-manifest",
            "test_fixture": False,
            "posebusters_configs": ["dock", "redock"],
            "tools": {
                "posebusters": {
                    "version": "1",
                    "package_source_sha256": "c" * 64,
                }
            },
            "assets": [],
        },
        producer="test-toolchain",
    )
    monkeypatch.setattr(
        workflow_module,
        "build_validation_toolchain",
        lambda *args, **kwargs: SimpleNamespace(
            artifact=toolchain, provenance=provenance
        ),
    )
    worker = WorkerConfig(
        engine="protbind-validation",
        argv=("/bin/true",),
        provenance=provenance,
    )

    effective = workflow._ensure_validation_support(manifest, worker)

    assert effective == provenance
    batch_ref = manifest.artifacts["support_validation_batch"]
    batch = store.read_json(batch_ref)
    assert batch["schema_version"] == "2.0"
    assert batch["reference_scope"] == "NOT_PROVIDED"
    assert "cofold_pose" not in batch["candidates"][0]
    assert manifest.artifacts["support_validation_toolchain"] == toolchain
    envelope = _worker_input_payload(manifest, RunState.VALIDATED)
    assert envelope["schema_version"] == "2.0"
    assert envelope["previous"]["stage"] == "DOCKED"
    assert (
        envelope["supporting_artifacts"]["support_validation_batch"]
        == batch_ref.to_dict()
    )


def test_generated_validation_toolchain_must_match_worker_provenance(
    tmp_path, monkeypatch
) -> None:
    workflow = ProtBindWorkflow(tmp_path / "workspace")
    store = workflow.artifacts
    manifest = _manifest_through(
        store, RunState.DOCKED, docking_bundle=_docking_bundle(store)
    )
    generated = WorkerProvenance(
        model_revision="generated", weight_sha256="a" * 64, code_sha256="b" * 64
    )
    configured = WorkerProvenance(
        model_revision="configured", weight_sha256="c" * 64, code_sha256="d" * 64
    )
    toolchain = store.put_json({"kind": "toolchain"}, producer="test")
    monkeypatch.setattr(
        workflow_module,
        "build_validation_toolchain",
        lambda *args, **kwargs: SimpleNamespace(
            artifact=toolchain, provenance=generated
        ),
    )
    worker = WorkerConfig(
        engine="protbind-validation",
        argv=("/bin/true",),
        provenance=configured,
    )

    with pytest.raises(PipelineStageError, match="provenance differs") as error:
        workflow._ensure_validation_support(manifest, worker)

    assert error.value.code == "VALIDATION_RUNTIME_MISMATCH"
    assert error.value.recoverable is False


def test_validation_only_support_never_changes_committed_docking_input(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    selected = _manifest_through(store, RunState.SELECTED)
    docking_before = _worker_input_payload(selected, RunState.DOCKED)
    reference = store.put_bytes(
        b"native SDF\n",
        media_type="chemical/x-mdl-sdfile",
        producer="test-reference",
    )
    validation_batch = store.put_json(
        {"kind": "protbind.validation-input-batch"}, producer="test-validation"
    )
    selected.artifacts["support_reference_pose"] = reference
    selected.artifacts["support_validation_batch"] = validation_batch

    assert _worker_input_payload(selected, RunState.DOCKED) == docking_before

    docked = _manifest_through(
        store, RunState.DOCKED, docking_bundle=_docking_bundle(store)
    )
    docked.artifacts["support_reference_pose"] = reference
    docked.artifacts["support_validation_batch"] = validation_batch
    validation_envelope = _worker_input_payload(docked, RunState.VALIDATED)
    assert (
        validation_envelope["supporting_artifacts"]["support_reference_pose"]
        == reference.to_dict()
    )
    assert (
        validation_envelope["supporting_artifacts"]["support_validation_batch"]
        == validation_batch.to_dict()
    )


def test_reference_pose_can_only_be_released_between_docked_and_validated(
    tmp_path, monkeypatch
) -> None:
    workflow = ProtBindWorkflow(tmp_path / "workspace")
    store = workflow.artifacts
    reference_path = tmp_path / "native.sdf"
    reference_path.write_text("native SDF\n", encoding="utf-8")
    selected = _manifest_through(store, RunState.SELECTED)

    with pytest.raises(ValueError, match="only after DOCKED"):
        workflow.attach_support(
            selected,
            "reference_pose",
            reference_path,
            media_type="chemical/x-mdl-sdfile",
        )

    docked = _manifest_through(
        store, RunState.DOCKED, docking_bundle=_docking_bundle(store)
    )
    monkeypatch.setattr(workflow, "_audit_artifacts", lambda manifest: None)
    monkeypatch.setattr(workflow, "load_case", lambda manifest: object())
    monkeypatch.setattr(workflow, "_verify_case_artifacts", lambda case: None)
    monkeypatch.setattr(
        workflow, "_audit_configuration", lambda manifest, case: None
    )
    reference = workflow.attach_support(
        docked,
        "reference_pose",
        reference_path,
        media_type="chemical/x-mdl-sdfile",
    )
    assert docked.artifacts["support_reference_pose"] == reference

    validated_output = store.put_json({"stage": "VALIDATED"}, producer="test")
    docked.complete_stage(
        StageRecord.create(
            RunState.VALIDATED,
            input_hash="e" * 64,
            config_hash="f" * 64,
            outputs=(validated_output,),
            duration_seconds=0,
        )
    )
    with pytest.raises(ValueError, match="frozen once VALIDATED"):
        workflow.attach_support(
            docked,
            "reference_pose",
            reference_path,
            media_type="chemical/x-mdl-sdfile",
            replace=True,
        )
