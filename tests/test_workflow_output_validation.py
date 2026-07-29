from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.manifest import RunState
from protbind_agent.models import LigandHypothesis, ResearchCase, ResearchMode, TargetSpec
from protbind_agent.worker_protocol import WorkerProvenance
from protbind_agent.workflow import (
    PipelineConfig,
    PipelineStageError,
    ProtBindWorkflow,
    WorkerConfig,
)


def _artifact(store: ArtifactStore, name: str, *, producer: str = "prepared"):
    return store.put_bytes(
        name.encode(),
        media_type="chemical/x-pdbqt",
        producer=producer,
        producer_version="1.0",
    )


def _case(store: ArtifactStore):
    case = ResearchCase(
        case_id="output-contract",
        target=TargetSpec(name="target", sequences=("ACDEFG",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(smiles="CCO"),
        seed=17,
    )
    return store.put_json(case.to_dict(), producer="protbind.case")


def _record(*outputs):
    return SimpleNamespace(outputs=tuple(outputs))


def _rewrite_json(
    store: ArtifactStore,
    reference,
    value: dict[str, Any],
    *,
    producer: str | None = None,
    producer_version: str | None = None,
):
    return store.put_json(
        value,
        producer=producer or reference.producer,
        producer_version=producer_version or reference.producer_version,
        source=reference.source,
    )


def _docked_contract(tmp_path):
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    runtime_sha = "a" * 64
    worker = WorkerConfig(
        engine="vina",
        argv=("/bin/true",),
        provenance=WorkerProvenance(
            model_revision=(
                "autodock-vina-1.2.7+meeko-0.7.1+rdkit-1+gemmi-1+"
                "numpy-1+scipy-1"
            ),
            weight_sha256=runtime_sha,
            code_sha256="b" * 64,
        ),
        parameters={"cpu": 2, "exhaustiveness": 8, "num_modes": 3, "energy_range": 4.0},
    )
    workflow = ProtBindWorkflow(
        workspace,
        config=PipelineConfig(workers={RunState.DOCKED: worker}),
    )
    receptor = store.put_bytes(
        b"receptor", media_type="chemical/x-pdb", producer="structure"
    )
    structure = store.put_bytes(
        b"complex", media_type="chemical/x-mmcif", producer="openfold3"
    )
    cofold = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.cofold-bundle",
            "candidates": [
                {
                    "candidate_id": "cofold-mol-a",
                    "molecule_id": "mol-a",
                    "microstate_id": "state-1",
                    "structure": structure.to_dict(),
                }
            ],
        },
        producer="official-openfold3.bundle",
    )
    batch = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.cofold-input-batch",
            "receptor": receptor.to_dict(),
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
        },
        producer="protbind.cofold-input",
    )
    environment_lock = store.put_bytes(
        b"lock", media_type="application/octet-stream", producer="environment"
    )
    envelope = store.put_json(
        {"schema_version": "1.0", "kind": "protbind.stage-input", "stage": "DOCKED"},
        producer="protbind.worker-stage-input",
    )
    prepared_receptor = _artifact(store, "prepared-receptor", producer="vina.meeko-receptor")
    ligand_sdf = store.put_bytes(
        b"ligand-sdf",
        media_type="chemical/x-mdl-sdfile",
        producer="vina.rdkit-etkdgv3",
    )
    prepared_ligand = _artifact(store, "prepared-ligand", producer="vina.meeko-ligand")
    all_modes = _artifact(store, "all-modes", producer="vina.all-modes")
    pose = _artifact(store, "pose", producer="attested-local-autodock-vina")
    score_semantics = (
        "AutoDock Vina tool score only; not an experimental binding free energy"
    )
    evidence_value = {
        "schema_version": "1.0",
        "kind": "protbind.tool-evidence",
        "tool": "vina",
        "tool_version": "1.2.7",
        "candidate_id": "vina-cofold-mol-a",
        "parent_candidate_id": "cofold-mol-a",
        "molecule_id": "mol-a",
        "microstate_id": "state-1",
        "seed": 17,
        "metrics": {
            "score": -7.25,
            "score_semantics": score_semantics,
            "box_center": [1.0, 2.0, 3.0],
            "box_size": [10.0, 11.0, 12.0],
            "scoring": "vina",
            "cpu": 2,
            "exhaustiveness": 8,
            "num_modes": 3,
            "energy_range": 4.0,
        },
        "inputs": {
            "stage_envelope": envelope.to_dict(),
            "cofold_bundle": cofold.to_dict(),
            "cofold_structure": structure.to_dict(),
            "receptor": receptor.to_dict(),
            "prepared_receptor": prepared_receptor.to_dict(),
            "ligand_sdf": ligand_sdf.to_dict(),
            "prepared_ligand": prepared_ligand.to_dict(),
            "pose": pose.to_dict(),
            "all_modes": all_modes.to_dict(),
        },
        "runtime_assets_sha256": runtime_sha,
        "timings_seconds": {"ligand_preparation": 0.1, "vina_command": 0.2},
    }
    evidence = store.put_json(
        evidence_value,
        producer="attested-local-autodock-vina.evidence",
        producer_version="1.2.7",
    )
    candidate = {
        "candidate_id": "vina-cofold-mol-a",
        "molecule_id": "mol-a",
        "parent_candidate_id": "cofold-mol-a",
        "microstate_id": "state-1",
        "cofold_structure": structure.to_dict(),
        "engine": "attested-local-autodock-vina",
        "seed": 17,
        "pose": pose.to_dict(),
        "vina_score": -7.25,
        "vina_score_semantics": score_semantics,
        "box_center": [1.0, 2.0, 3.0],
        "box_size": [10.0, 11.0, 12.0],
        "receptor": receptor.to_dict(),
        "prepared_receptor": prepared_receptor.to_dict(),
        "prepared_ligand": prepared_ligand.to_dict(),
        "all_modes": all_modes.to_dict(),
        "evidence": evidence.to_dict(),
    }
    metadata_value = {
        "schema_version": "1.0",
        "toolchain": {
            "runtime_assets_sha256": runtime_sha,
            "official_runtime": False,
            "trust_level": "hash-attested-local-without-reviewed-upstream-allowlist",
            "vina": {"version": "1.2.7"},
        },
        "environment_lock": environment_lock.to_dict(),
        "execution": {
            "device": "cpu",
            "cpu_threads": 2,
            "seed": 17,
            "scoring": "vina",
            "exhaustiveness": 8,
            "num_modes": 3,
            "energy_range": 4.0,
            "input_candidate_count": 1,
            "successful_candidate_count": 1,
            "failed_candidate_count": 0,
        },
    }
    metadata = store.put_json(
        metadata_value,
        producer="attested-local-autodock-vina.run-metadata",
        producer_version="1.2.7",
    )
    bundle_value = {
        "schema_version": "1.0",
        "kind": "protbind.docking-bundle",
        "score_semantics": score_semantics,
        "receptor": receptor.to_dict(),
        "prepared_receptor": prepared_receptor.to_dict(),
        "upstream_cofold_bundle": cofold.to_dict(),
        "upstream_candidate_ids": ["cofold-mol-a"],
        "candidate_count": 1,
        "failure_count": 0,
        "candidates": [candidate],
        "failures": [],
        "run_metadata": metadata.to_dict(),
    }
    bundle = store.put_json(
        bundle_value,
        producer="attested-local-autodock-vina.bundle",
        producer_version="1.2.7",
    )
    outputs = (
        bundle,
        prepared_receptor,
        ligand_sdf,
        prepared_ligand,
        all_modes,
        pose,
        evidence,
        metadata,
    )
    manifest = SimpleNamespace(
        case_id="output-contract",
        case_artifact=_case(store),
        artifacts={
            "support_openfold_batch": batch,
            "support_vina_environment_lock": environment_lock,
            "worker_input_docked": envelope,
        },
        stage_records={RunState.COFOLDED.value: _record(cofold)},
    )
    return workflow, store, manifest, bundle_value, outputs, evidence_value, metadata_value


def test_docked_production_contract_accepts_complete_hash_bound_bundle(tmp_path) -> None:
    workflow, _, manifest, _, outputs, _, _ = _docked_contract(tmp_path)

    workflow._validate_worker_outputs(manifest, RunState.DOCKED, outputs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("coverage", "upstream_candidate_ids"),
        ("candidate_id", "derived from its frozen parent"),
        ("seed", "seed does not match"),
        ("box", "frozen quick-docking box"),
        ("lock", "environment lock"),
        ("runtime", "runtime assets"),
    ],
)
def test_docked_production_contract_rejects_lineage_and_runtime_mutations(
    tmp_path, mutation: str, message: str
) -> None:
    workflow, store, manifest, bundle_value, outputs, evidence_value, metadata_value = (
        _docked_contract(tmp_path)
    )
    bundle_value = {**bundle_value, "candidates": [dict(bundle_value["candidates"][0])]}
    output_values = list(outputs)
    if mutation == "coverage":
        bundle_value["upstream_candidate_ids"] = []
    elif mutation == "candidate_id":
        bundle_value["candidates"][0]["candidate_id"] = "not-derived"
    elif mutation == "seed":
        bundle_value["candidates"][0]["seed"] = 99
    elif mutation == "box":
        bundle_value["candidates"][0]["box_center"] = [9.0, 2.0, 3.0]
    elif mutation == "lock":
        wrong_lock = store.put_bytes(
            b"wrong-lock", media_type="application/octet-stream", producer="environment"
        )
        changed_metadata = {**metadata_value, "environment_lock": wrong_lock.to_dict()}
        metadata = _rewrite_json(store, outputs[-1], changed_metadata)
        bundle_value["run_metadata"] = metadata.to_dict()
        output_values[-1] = metadata
    else:
        changed_evidence = {**evidence_value, "runtime_assets_sha256": "f" * 64}
        evidence = _rewrite_json(store, outputs[-2], changed_evidence)
        bundle_value["candidates"][0]["evidence"] = evidence.to_dict()
        output_values[-2] = evidence
    bundle = _rewrite_json(store, outputs[0], bundle_value)
    output_values[0] = bundle

    with pytest.raises(PipelineStageError, match=message):
        workflow._validate_worker_outputs(
            manifest, RunState.DOCKED, tuple(output_values)
        )


def test_docked_production_contract_rejects_omitted_returned_evidence(tmp_path) -> None:
    workflow, _, manifest, _, outputs, _, _ = _docked_contract(tmp_path)

    with pytest.raises(PipelineStageError, match="must be returned"):
        workflow._validate_worker_outputs(
            manifest, RunState.DOCKED, tuple(item for item in outputs if item != outputs[-2])
        )


@pytest.mark.parametrize("forbidden", ["pose", "evidence", "score"])
def test_docked_failure_cannot_carry_scientific_outputs(tmp_path, forbidden: str) -> None:
    workflow, store, manifest, bundle_value, outputs, _, metadata_value = (
        _docked_contract(tmp_path)
    )
    original = bundle_value["candidates"][0]
    failure = {
        "candidate_id": "cofold-mol-a",
        "parent_candidate_id": "cofold-mol-a",
        "molecule_id": "mol-a",
        "microstate_id": "state-1",
        "engine": "attested-local-autodock-vina",
        "cofold_structure": original["cofold_structure"],
        "receptor": original["receptor"],
        "seed": 17,
        "box_center": original["box_center"],
        "box_size": original["box_size"],
        "stage": "ligand_preparation_or_vina",
        "error": {"code": "OUTPUT_INVALID", "message": "no score", "recoverable": False},
    }
    failure[forbidden] = (
        original["pose"]
        if forbidden == "pose"
        else original["evidence"]
        if forbidden == "evidence"
        else -9.0
    )
    execution = {
        **metadata_value["execution"],
        "successful_candidate_count": 0,
        "failed_candidate_count": 1,
    }
    metadata = _rewrite_json(
        store, outputs[-1], {**metadata_value, "execution": execution}
    )
    failed_bundle = {
        **bundle_value,
        "candidate_count": 0,
        "failure_count": 1,
        "candidates": [],
        "failures": [failure],
        "run_metadata": metadata.to_dict(),
    }
    bundle = _rewrite_json(store, outputs[0], failed_bundle)

    with pytest.raises(PipelineStageError, match="cannot carry pose/evidence/score"):
        workflow._validate_worker_outputs(
            manifest, RunState.DOCKED, (bundle, outputs[1], metadata)
        )


def _validated_contract(tmp_path):
    workspace = tmp_path / "validation-workspace"
    store = ArtifactStore(workspace)
    workflow = ProtBindWorkflow(
        workspace,
        config=PipelineConfig(
            workers={
                RunState.VALIDATED: WorkerConfig(
                    engine="protbind-validation",
                    argv=("/bin/true",),
                    provenance=WorkerProvenance(
                        model_revision="validation-toolchain:test",
                        weight_sha256="c" * 64,
                        code_sha256="d" * 64,
                    ),
                )
            }
        ),
    )
    docked_pose = _artifact(store, "docked-pose", producer="vina")
    cofold_pose = store.put_bytes(
        b"cofold-pose", media_type="chemical/x-mmcif", producer="openfold3"
    )
    docking = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.docking-bundle",
            "candidates": [
                {
                    "candidate_id": "vina-cofold-mol-a",
                    "molecule_id": "mol-a",
                    "microstate_id": "state-1",
                    "pose": docked_pose.to_dict(),
                    "cofold_structure": cofold_pose.to_dict(),
                }
            ],
        },
        producer="attested-local-autodock-vina.bundle",
    )
    prepared = {
        name: store.put_bytes(
            name.encode(),
            media_type=(
                "chemical/x-mdl-sdfile" if "ligand" in name else "chemical/x-pdb"
            ),
            producer="validation-preparation",
        )
        for name in (
            "docked_ligand",
            "docked_receptor",
            "cofold_ligand",
            "cofold_receptor",
        )
    }
    batch = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.validation-input-batch",
            "docking_bundle": docking.to_dict(),
            "candidates": [
                {
                    "candidate_id": "vina-cofold-mol-a",
                    "molecule_id": "mol-a",
                    "microstate_id": "state-1",
                    "docked_pose": docked_pose.to_dict(),
                    "cofold_pose": cofold_pose.to_dict(),
                    "posebusters": {
                        name: reference.to_dict() for name, reference in prepared.items()
                    },
                }
            ],
        },
        producer="protbind.validation-input",
    )
    source_sha = "e" * 64
    toolchain = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.validation-toolchain-manifest",
            "test_fixture": False,
            "posebusters_config": "dock",
            "tools": {
                "posebusters": {
                    "version": "1.2.0",
                    "package_source_sha256": source_sha,
                }
            },
            "assets": [],
        },
        producer="protbind.validation-toolchain",
    )
    evidence_value = {
        "schema_version": "1.0",
        "kind": "protbind.tool-evidence",
        "tool": "posebusters",
        "molecule_id": "mol-a",
        "candidate_id": "vina-cofold-mol-a",
        "metrics": {
            "valid": True,
            "docked_valid": True,
            "cofold_valid": True,
            "config": "dock",
        },
        "inputs": {
            "docked_pose": docked_pose.to_dict(),
            "cofold_pose": cofold_pose.to_dict(),
            **{name: reference.to_dict() for name, reference in prepared.items()},
        },
        "runtime": {
            "version": "1.2.0",
            "package_source_sha256": source_sha,
        },
        "test_fixture": False,
    }
    evidence = store.put_json(
        evidence_value,
        producer="protbind.posebusters",
        producer_version="1.2.0",
    )
    candidate = {
        "candidate_id": "vina-cofold-mol-a",
        "molecule_id": "mol-a",
        "microstate_id": "state-1",
        "docked_pose": docked_pose.to_dict(),
        "cofold_pose": cofold_pose.to_dict(),
        "engine": "protbind-validation",
        "seed": 17,
        "has_reference_pose": False,
        "decision_reason": "PoseBusters passed; this does not establish binding.",
        "bundle": {
            "preparation_attested": False,
            "posebusters_valid": True,
            "vina_pose_valid": True,
            "cofold_pose_valid": True,
            "unsupported_reasons": [],
            "evidence": [evidence.to_dict()],
        },
    }
    bundle_value = {
        "schema_version": "1.0",
        "kind": "protbind.validation-bundle",
        "candidates": [candidate],
        "toolchain": toolchain.to_dict(),
        "test_fixture": False,
    }
    bundle = store.put_json(
        bundle_value,
        producer="protbind-validation",
        producer_version="1.0",
    )
    manifest = SimpleNamespace(
        case_id="output-contract",
        case_artifact=_case(store),
        artifacts={
            "support_validation_batch": batch,
            "support_validation_toolchain": toolchain,
        },
        stage_records={RunState.DOCKED.value: _record(docking)},
    )
    return workflow, store, manifest, bundle_value, (bundle, evidence), evidence_value


def test_validated_production_contract_accepts_toolchain_and_prepared_lineage(
    tmp_path,
) -> None:
    workflow, _, manifest, _, outputs, _ = _validated_contract(tmp_path)

    workflow._validate_worker_outputs(manifest, RunState.VALIDATED, outputs)


def test_optional_invalid_cofold_does_not_veto_valid_docked_pose(tmp_path) -> None:
    workflow, store, manifest, bundle_value, _, evidence_value = (
        _validated_contract(tmp_path)
    )
    evidence_value["metrics"]["cofold_valid"] = False
    evidence = store.put_json(
        evidence_value,
        producer="protbind.posebusters",
        producer_version="1.2.0",
    )
    bundle_value["candidates"][0]["bundle"]["cofold_pose_valid"] = False
    bundle_value["candidates"][0]["bundle"]["evidence"] = [evidence.to_dict()]
    bundle = store.put_json(
        bundle_value,
        producer="protbind-validation",
        producer_version="1.0",
    )

    workflow._validate_worker_outputs(
        manifest, RunState.VALIDATED, (bundle, evidence)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("prepared", "frozen preparation"),
        ("runtime", "frozen toolchain"),
        ("producer_version", "producer_version"),
        ("fixture", "fixture label"),
        ("coverage", "exactly cover"),
        ("engine", "wrong engine"),
        ("tool", "unsupported validation tool"),
        ("unattested_promotion", "self-certified preparation"),
    ],
)
def test_validated_production_contract_rejects_unbound_or_fixture_evidence(
    tmp_path, mutation: str, message: str
) -> None:
    workflow, store, manifest, bundle_value, outputs, evidence_value = (
        _validated_contract(tmp_path)
    )
    bundle_value = {**bundle_value, "candidates": [dict(bundle_value["candidates"][0])]}
    output_values = list(outputs)
    if mutation == "coverage":
        bundle_value["candidates"] = []
        output_values = output_values[:1]
    elif mutation == "engine":
        bundle_value["candidates"][0]["engine"] = "some-other-validator"
    elif mutation == "unattested_promotion":
        bundle_value["candidates"][0]["bundle"] = {
            **bundle_value["candidates"][0]["bundle"],
            "preparation_attested": True,
        }
    else:
        changed = dict(evidence_value)
        if mutation == "prepared":
            changed["inputs"] = dict(changed["inputs"])
            changed["inputs"]["docked_ligand"] = changed["inputs"]["cofold_ligand"]
        elif mutation == "runtime":
            changed["runtime"] = {
                **changed["runtime"],
                "package_source_sha256": "f" * 64,
            }
        elif mutation == "fixture":
            changed["test_fixture"] = True
        elif mutation == "tool":
            changed["tool"] = "vina"
        evidence = _rewrite_json(
            store,
            outputs[1],
            changed,
            producer="protbind.vina" if mutation == "tool" else None,
        )
        if mutation == "producer_version":
            evidence = replace(evidence, producer_version="9.9.9")
        bundle_value["candidates"][0] = {
            **bundle_value["candidates"][0],
            "bundle": {
                **bundle_value["candidates"][0]["bundle"],
                "evidence": [evidence.to_dict()],
            },
        }
        output_values[1] = evidence
    bundle = _rewrite_json(store, outputs[0], bundle_value)
    output_values[0] = bundle

    with pytest.raises(PipelineStageError, match=message):
        workflow._validate_worker_outputs(
            manifest, RunState.VALIDATED, tuple(output_values)
        )
