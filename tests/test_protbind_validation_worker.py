from __future__ import annotations

import json
import sys

import pytest

from protbind_agent.artifacts import sha256_bytes
from protbind_agent.models import ArtifactRef, EvidenceGrade, ValidationBundle
from protbind_agent.validation import classify_evidence, vina_score_language
from protbind_agent.worker_protocol import (
    JsonSubprocessWorker,
    WorkerError,
    WorkerProvenance,
    WorkerRequest,
    WorkerResponse,
)


def _ref() -> ArtifactRef:
    return ArtifactRef(
        sha256=sha256_bytes(b"input"),
        media_type="application/json",
        size_bytes=5,
        producer="test",
    )


def test_evidence_grade_rules_and_vina_language() -> None:
    reference = ValidationBundle(
        preparation_attested=True,
        posebusters_valid=True,
        symmetry_rmsd_angstrom=1.8,
    )
    consensus = ValidationBundle(
        preparation_attested=True,
        posebusters_valid=True,
        vina_pose_valid=True,
        cofold_pose_valid=True,
        ifp_similarity=0.75,
        ifp_reference_recovery=0.75,
        ifp_predicted_precision=1.0,
        ifp_docked_label_count=3,
        ifp_comparison_label_count=4,
        ifp_intersection_count=3,
        ifp_union_count=4,
    )
    rejected = ValidationBundle(posebusters_valid=False)

    assert (
        classify_evidence(reference, has_reference_pose=True)
        is EvidenceGrade.REDOCKING_RECOVERED
    )
    assert (
        classify_evidence(consensus, has_reference_pose=False)
        is EvidenceGrade.METHOD_CONSENSUS
    )
    assert classify_evidence(rejected, has_reference_pose=False) is EvidenceGrade.REJECTED
    assert (
        classify_evidence(
            ValidationBundle(
                preparation_attested=False,
                posebusters_valid=True,
                vina_pose_valid=True,
                cofold_pose_valid=True,
                ifp_similarity=1.0,
                ifp_reference_recovery=1.0,
                ifp_predicted_precision=1.0,
                ifp_docked_label_count=1,
                ifp_comparison_label_count=1,
                ifp_intersection_count=1,
                ifp_union_count=1,
            ),
            has_reference_pose=False,
        )
        is EvidenceGrade.HYPOTHESIS_ONLY
    )
    assert (
        classify_evidence(
            ValidationBundle(
                posebusters_valid=True,
                openmm_parameterized=True,
                openmm_stable=False,
            ),
            has_reference_pose=False,
        )
        is EvidenceGrade.REJECTED
    )
    assert "not an experimental binding free energy" in vina_score_language(-7.2)
    with pytest.raises(ValueError, match="PoseBusters"):
        classify_evidence(ValidationBundle(), has_reference_pose=False)


def test_validation_bundle_rejects_invalid_numeric_or_boolean_evidence() -> None:
    with pytest.raises(ValueError, match="IFP similarity"):
        ValidationBundle(
            ifp_similarity=1.01,
            ifp_reference_recovery=1.0,
            ifp_predicted_precision=1.0,
            ifp_docked_label_count=1,
            ifp_comparison_label_count=1,
            ifp_intersection_count=1,
            ifp_union_count=1,
        )
    with pytest.raises(ValueError, match="internally inconsistent"):
        ValidationBundle(
            ifp_similarity=0.5,
            ifp_reference_recovery=0.5,
            ifp_predicted_precision=0.5,
            ifp_docked_label_count=2,
            ifp_comparison_label_count=2,
            ifp_intersection_count=2,
            ifp_union_count=3,
        )
    with pytest.raises(ValueError, match="symmetry RMSD"):
        ValidationBundle(symmetry_rmsd_angstrom=-0.1)
    with pytest.raises(ValueError, match="must be boolean"):
        ValidationBundle(posebusters_valid="false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires successful parameterization"):
        ValidationBundle(openmm_parameterized=None, openmm_stable=True)


def test_worker_response_rejects_success_without_artifact() -> None:
    with pytest.raises(ValueError, match="requires at least one"):
        WorkerResponse(job_id="job", engine="engine")
    failed = WorkerResponse(
        job_id="job",
        engine="engine",
        error=WorkerError(code="OOM", message="out of memory", recoverable=True),
    )
    assert failed.error and failed.error.recoverable


def test_worker_protocol_rejects_boolean_numeric_fields() -> None:
    request_value = {
        "schema_version": "1.0",
        "job_id": "job",
        "engine": "engine",
        "input": _ref().to_dict(),
        "parameters": {},
        "seed": True,
        "provenance": {
            "model_revision": "test",
            "weight_sha256": "a" * 64,
            "code_sha256": "b" * 64,
        },
    }
    with pytest.raises(ValueError, match="seed must be an integer"):
        WorkerRequest.from_dict(request_value)

    response_value = {
        "schema_version": "1.0",
        "job_id": "job",
        "engine": "engine",
        "outputs": [],
        "provenance": None,
        "timings_seconds": {"runtime": True},
        "peak_vram_bytes": None,
        "warnings": [],
        "error": {"code": "FAILED", "message": "failed", "recoverable": False},
    }
    with pytest.raises(ValueError, match="timings must contain numeric"):
        WorkerResponse.from_dict(response_value)
    response_value["timings_seconds"] = {}
    response_value["peak_vram_bytes"] = False
    with pytest.raises(ValueError, match="peak_vram_bytes"):
        WorkerResponse.from_dict(response_value)
    with pytest.raises(ValueError, match="timeout"):
        JsonSubprocessWorker((sys.executable, "worker.py"), timeout_seconds=float("nan"))


def test_explicit_hip_assignment_drops_ambient_cuda_alias(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    worker = JsonSubprocessWorker(
        (sys.executable, "worker.py"),
        environment={"HIP_VISIBLE_DEVICES": "0"},
    )

    environment = worker._environment(None)

    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert "CUDA_VISIBLE_DEVICES" not in environment


def test_json_worker_contract_does_not_use_shell(tmp_path) -> None:
    script = tmp_path / "worker.py"
    script.write_text(
        "import json, sys\n"
        "r=json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'schema_version':'1.0','job_id':r['job_id'],"
        "'engine':r['engine'],'error':{'code':'UNAVAILABLE',"
        "'message':'missing /private/model/file','recoverable':True}}))\n",
        encoding="utf-8",
    )
    request = WorkerRequest(
        job_id="job-1",
        engine="openfold3",
        input=_ref(),
        parameters={},
        seed=7,
        provenance=WorkerProvenance(
            model_revision="test",
            weight_sha256="a" * 64,
            code_sha256="b" * 64,
        ),
    )

    response, elapsed = JsonSubprocessWorker((sys.executable, str(script))).run(request)

    assert response.error and response.error.code == "UNAVAILABLE"
    assert "/private/model/file" not in response.error.message
    assert elapsed >= 0
    assert json.loads(json.dumps(request.to_dict()))["seed"] == 7
