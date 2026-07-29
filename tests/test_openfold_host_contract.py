from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import protbind_agent.workflow as workflow_module
from protbind_agent.artifacts import ArtifactStore
from protbind_agent.manifest import RunState
from protbind_agent.models import LigandHypothesis, ResearchCase, ResearchMode, TargetSpec
from protbind_agent.openfold_contract import (
    OFFICIAL_RUNTIME_FILE_COUNT,
    OFFICIAL_RUNTIME_SHA256,
    OPENFOLD_BUNDLE_PRODUCER,
    OPENFOLD_QUERY_MANIFEST_PRODUCER,
    OPENFOLD_REVISION,
    OPENFOLD_RUN_METADATA_PRODUCER,
    OPENFOLD_RUNNER_PRODUCER,
    OPENFOLD_RUNTIME_ENGINE,
    OPENFOLD_SCM_NODE,
    OPENFOLD_VERSION,
)
from protbind_agent.worker_protocol import WorkerProvenance
from protbind_agent.workflow import PipelineConfig, ProtBindWorkflow, WorkerConfig


def _contract(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    case = ResearchCase(
        case_id="openfold-host-contract",
        target=TargetSpec(name="target", sequences=("ACDEFG",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(smiles="CCO"),
        seed=17,
    )
    case_ref = store.put_json(case.to_dict(), producer="protbind.case")
    batch = store.put_json(
        {"schema_version": "1.0", "kind": "protbind.cofold-input-batch"},
        producer="protbind.cofold-input",
    )
    checkpoint = store.put_bytes(
        b"checkpoint",
        media_type="application/octet-stream",
        producer="openfold.checkpoint",
    )
    environment_lock = store.put_bytes(
        b"lock", media_type="application/octet-stream", producer="pixi.lock"
    )
    envelope = store.put_json(
        {"schema_version": "1.0", "kind": "protbind.stage-input"},
        producer="protbind.worker-stage-input",
    )
    monkeypatch.setitem(
        workflow_module.OFFICIAL_CHECKPOINT_SIZES,
        "openfold3-p2-155k",
        checkpoint.size_bytes,
    )
    provenance = WorkerProvenance(
        model_revision=OPENFOLD_REVISION,
        weight_sha256=checkpoint.sha256,
        code_sha256="a" * 64,
    )
    worker = WorkerConfig(
        engine="openfold3",
        argv=("/bin/true",),
        provenance=provenance,
        parameters={
            "num_diffusion_samples": 1,
            "low_mem": True,
            "use_triton_triangle_kernels": True,
            "use_msa_server": False,
            "checkpoint_name": "openfold3-p2-155k",
            "minimum_free_vram_gib": 28.0,
        },
        environment={"HIP_VISIBLE_DEVICES": "0"},
    )
    workflow = ProtBindWorkflow(
        workspace,
        config=PipelineConfig(workers={RunState.COFOLDED: worker}),
    )
    structure = store.put_bytes(
        b"data_model\n#\n",
        media_type="chemical/x-mmcif",
        producer=OPENFOLD_RUNTIME_ENGINE,
        producer_version=OPENFOLD_REVISION,
        source="local-output:query_model.cif",
    )
    confidence = store.put_json(
        {"sample_ranking_score": 0.7},
        producer=f"{OPENFOLD_RUNTIME_ENGINE}.sanitized-output",
        producer_version=OPENFOLD_REVISION,
        source="local-output:query_confidences_aggregated.json",
    )
    full_confidence = store.put_json(
        {"plddt": [80.0], "pde": [[0.0]], "pae": [[0.0]]},
        producer=f"{OPENFOLD_RUNTIME_ENGINE}.sanitized-output",
        producer_version=OPENFOLD_REVISION,
        source="local-output:query_confidences.json",
    )
    timing = store.put_json(
        {"runtime_s": 1.0},
        producer=f"{OPENFOLD_RUNTIME_ENGINE}.sanitized-output",
        producer_version=OPENFOLD_REVISION,
        source="local-output:timing.json",
    )
    effective = {
        name: store.put_json(
            {"name": name},
            producer=f"{OPENFOLD_RUNTIME_ENGINE}.sanitized-output",
            producer_version=OPENFOLD_REVISION,
            source=f"local-output:{name}.json",
        )
        for name in ("experiment_config", "inference_query_set", "model_config")
    }
    runtime = {
        "distribution": "openfold3",
        "version": OPENFOLD_VERSION,
        "scm_tag": OPENFOLD_VERSION,
        "scm_distance": 0,
        "scm_node": OPENFOLD_SCM_NODE,
        "scm_dirty": False,
        "entry_point": "openfold3.run_openfold:cli",
        "package_source_sha256": OFFICIAL_RUNTIME_SHA256,
        "runtime_file_count": OFFICIAL_RUNTIME_FILE_COUNT,
        "official_release": True,
        "torch_version": "2.12.0",
        "torch_hip_version": "7.2",
        "triton_distribution": "triton-rocm",
        "triton_version": "3.5.0",
        "triton_module_version": "3.5.0",
    }
    bindings = {
        "stage_envelope": envelope.to_dict(),
        "input_batch": batch.to_dict(),
        "checkpoint": checkpoint.to_dict(),
        "environment_lock": environment_lock.to_dict(),
        "provenance": provenance.to_dict(),
    }
    raw_outputs = [
        structure,
        confidence,
        full_confidence,
        timing,
        *effective.values(),
    ]
    query_manifest = store.put_json(
        {
            "schema_version": "1.0",
            "query_ids": ["pb_0001_deadbeef"],
            "candidate_ids": ["cofold-mol-a"],
            "msa_server": False,
            "templates": False,
            **bindings,
            "runtime_attestation": runtime,
            "effective_config_artifacts": {
                name: reference.to_dict() for name, reference in effective.items()
            },
            "raw_outputs": [reference.to_dict() for reference in raw_outputs],
        },
        producer=OPENFOLD_QUERY_MANIFEST_PRODUCER,
        producer_version=OPENFOLD_REVISION,
    )
    runner = store.put_bytes(
        (
            b"use_msa_server: false\nuse_triton_triangle_kernels: true\n"
            b"precision: 32-true\ndevices: 1\n    - low_mem\n"
        ),
        media_type="application/yaml",
        producer=OPENFOLD_RUNNER_PRODUCER,
        producer_version=OPENFOLD_REVISION,
    )
    metadata_value = {
        "schema_version": "1.0",
        "openfold_revision": OPENFOLD_REVISION,
        **bindings,
        "checkpoint_name": "openfold3-p2-155k",
        "seed": 17,
        "num_diffusion_samples": 1,
        "low_mem": True,
        "rocm_triton": True,
        "msa_server": False,
        "templates": False,
        "precision": "32-true",
        "runtime_attestation": runtime,
        "resource_policy": {
            "hip_visible_device": "0",
            "trainer_devices": 1,
            "concurrent_openfold_jobs": 1,
            "minimum_free_vram_gib": 28.0,
            "free_vram_bytes_before_run": 30 * 1024**3,
            "total_vram_bytes": 48 * 1024**3,
        },
    }
    metadata = store.put_json(
        metadata_value,
        producer=OPENFOLD_RUN_METADATA_PRODUCER,
        producer_version=OPENFOLD_REVISION,
    )
    candidate = {
        "candidate_id": "cofold-mol-a",
        "molecule_id": "mol-a",
        "microstate_id": "state-1",
        "engine": OPENFOLD_RUNTIME_ENGINE,
        "seed": 17,
        "structure": structure.to_dict(),
        "confidence_name": "sample_ranking_score",
        "confidence_value": 0.7,
        "confidence_semantics": "OpenFold3 model confidence; not binding affinity",
        "samples": [
            {
                "structure": structure.to_dict(),
                "confidence": confidence.to_dict(),
                "sample_ranking_score": 0.7,
            }
        ],
    }
    value = {
        "schema_version": "1.0",
        "kind": "protbind.cofold-bundle",
        "score_semantics": "model confidence only; not binding affinity",
        "candidates": [candidate],
        "query_manifest": query_manifest.to_dict(),
        "runner": runner.to_dict(),
        "run_metadata": metadata.to_dict(),
    }
    bundle = store.put_json(
        value,
        producer=OPENFOLD_BUNDLE_PRODUCER,
        producer_version=OPENFOLD_REVISION,
    )
    outputs = (
        bundle,
        *raw_outputs,
        query_manifest,
        runner,
        metadata,
    )
    manifest = SimpleNamespace(
        case_id=case.case_id,
        case_artifact=case_ref,
        artifacts={
            "worker_input_cofolded": envelope,
            "support_openfold_batch": batch,
            "support_openfold_checkpoint": checkpoint,
            "support_openfold_environment_lock": environment_lock,
        },
    )
    upstream = [
        {
            "candidate_id": "cofold-mol-a",
            "molecule_id": "mol-a",
            "microstate_id": "state-1",
        }
    ]
    return workflow, store, manifest, value, outputs, upstream, metadata_value


def test_production_cofold_host_contract_accepts_hash_bound_output(
    tmp_path, monkeypatch
) -> None:
    workflow, _, manifest, value, outputs, upstream, _ = _contract(
        tmp_path, monkeypatch
    )

    workflow._validate_cofolded_production_contract(
        manifest, value, outputs, upstream
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("producer", "bundle producer"),
        ("engine", "not official OpenFold3"),
        ("extra", "unreferenced artifacts"),
        ("metadata", "changed checkpoint_name"),
    ],
)
def test_production_cofold_host_contract_rejects_bypass_mutations(
    tmp_path, monkeypatch, mutation: str, message: str
) -> None:
    workflow, store, manifest, value, outputs, upstream, metadata_value = _contract(
        tmp_path, monkeypatch
    )
    values = {**value, "candidates": [dict(value["candidates"][0])]}
    output_values = list(outputs)
    if mutation == "producer":
        output_values[0] = replace(output_values[0], producer="fake-openfold.bundle")
    elif mutation == "engine":
        values["candidates"][0]["engine"] = "some-other-engine"
    elif mutation == "extra":
        output_values.append(
            store.put_json(
                {"extra": True},
                producer=f"{OPENFOLD_RUNTIME_ENGINE}.sanitized-output",
                producer_version=OPENFOLD_REVISION,
                source="local-output:extra.json",
            )
        )
    else:
        changed_metadata = {**metadata_value, "checkpoint_name": "openfold3-p2-145k"}
        metadata = store.put_json(
            changed_metadata,
            producer=OPENFOLD_RUN_METADATA_PRODUCER,
            producer_version=OPENFOLD_REVISION,
        )
        values["run_metadata"] = metadata.to_dict()
        output_values[-1] = metadata

    with pytest.raises(ValueError, match=message):
        workflow._validate_cofolded_production_contract(
            manifest, values, tuple(output_values), upstream
        )
