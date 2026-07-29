from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from protbind_agent.artifacts import ArtifactStore, sha256_file
from protbind_agent.worker_protocol import (
    JsonSubprocessWorker,
    WorkerProvenance,
    WorkerRequest,
)

WORKER = Path(__file__).parents[1] / "workers" / "esmfold_v1_worker.py"


def _worker_module():
    spec = importlib.util.spec_from_file_location("esmfold_worker_test", WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ESMFold worker module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pdb(sequence: str, *, coordinate: str | None = None) -> str:
    names = {"A": "ALA", "C": "CYS", "D": "ASP"}
    lines: list[str] = []
    serial = 1
    for residue_number, code in enumerate(sequence, start=1):
        for atom_name in ("N", "CA", "C"):
            x = f"{float(serial):8.3f}" if coordinate is None else f"{coordinate:>8}"
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {names[code]:>3s} A"
                f"{residue_number:4d}    {x}{0.0:8.3f}{0.0:8.3f}"
                "  1.00 80.00           C"
            )
            serial += 1
    return "\n".join(lines) + "\nEND\n"


def test_esmfold_worker_requires_local_weights_without_downloading(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    input_artifact = store.put_json(
        {"schema_version": "1.0", "sequences": ["ACDEFG"]}, producer="test"
    )
    request = WorkerRequest(
        job_id="esmfold-contract",
        engine="esmfold_v1",
        input=input_artifact,
        parameters={},
        seed=7,
        provenance=WorkerProvenance(
            model_revision="esmfold_3B_v1",
            weight_sha256="a" * 64,
            code_sha256="b" * 64,
        ),
    )
    script = WORKER

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(script)), artifact_root=store.root
    ).run(request)

    assert response.error is not None
    assert response.error.code == "MODEL_UNAVAILABLE"
    assert response.error.recoverable


def test_esmfold_worker_requires_the_pinned_model_revision(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    input_artifact = store.put_json(
        {"schema_version": "1.0", "sequences": ["ACDEFG"]}, producer="test"
    )
    request = WorkerRequest(
        job_id="esmfold-revision-contract",
        engine="esmfold_v1",
        input=input_artifact,
        parameters={"model_path": str(tmp_path / "model.pt")},
        seed=7,
        provenance=WorkerProvenance(
            model_revision="unreviewed-model",
            weight_sha256="a" * 64,
            code_sha256="b" * 64,
        ),
    )
    script = WORKER

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(script)), artifact_root=store.root
    ).run(request)

    assert response.error is not None
    assert response.error.code == "MODEL_REVISION_MISMATCH"
    assert response.error.recoverable is False


def test_esmfold_worker_requires_offline_esm2_backbone_files(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    input_artifact = store.put_json(
        {"schema_version": "1.0", "sequences": ["ACDEFG"]}, producer="test"
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"not loaded")
    request = WorkerRequest(
        job_id="esmfold-backbone-contract",
        engine="esmfold_v1",
        input=input_artifact,
        parameters={"model_path": str(model)},
        seed=7,
        provenance=WorkerProvenance(
            model_revision="esmfold_3B_v1",
            weight_sha256="a" * 64,
            code_sha256="b" * 64,
        ),
    )
    script = WORKER

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(script)), artifact_root=store.root
    ).run(request)

    assert response.error is not None
    assert response.error.code == "MODEL_UNAVAILABLE"
    assert response.error.recoverable is True


def test_esmfold_worker_requires_torch_hub_checkpoints_layout(tmp_path) -> None:
    module = _worker_module()
    store = ArtifactStore(tmp_path / "workspace")
    input_artifact = store.put_json({"sequences": ["AC"]}, producer="test")
    model = tmp_path / "esmfold.pt"
    cache = tmp_path / "wrong-cache-directory"
    cache.mkdir()
    esm2 = cache / "esm2_t36_3B_UR50D.pt"
    regression = cache / "esm2_t36_3B_UR50D-contact-regression.pt"
    lock = tmp_path / "environment.lock"
    for path in (model, esm2, regression, lock):
        path.write_bytes(path.name.encode())
    request = WorkerRequest(
        job_id="esmfold-layout-contract",
        engine="esmfold_v1",
        input=input_artifact,
        parameters={
            "model_path": str(model),
            "esm2_model_path": str(esm2),
            "esm2_regression_path": str(regression),
            "environment_lock_path": str(lock),
        },
        seed=7,
        provenance=WorkerProvenance(
            model_revision="esmfold_3B_v1",
            weight_sha256=module.checkpoint_set_sha256(model, esm2, regression),
            code_sha256=sha256_file(WORKER),
        ),
    )

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(WORKER)), artifact_root=store.root
    ).run(request)

    assert response.error is not None
    assert response.error.code == "MODEL_UNAVAILABLE"
    assert response.error.recoverable is False


def test_esmfold_output_gate_checks_sequence_backbone_and_finite_coordinates() -> None:
    module = _worker_module()

    qc = module.validate_predicted_pdb(_pdb("AC"), ("AC",))

    assert qc["chain_lengths"] == [2]
    assert qc["coordinate_finite"] is True
    with pytest.raises(module.WorkerFailure, match="non-finite"):
        module.validate_predicted_pdb(_pdb("AC", coordinate="nan"), ("AC",))
    with pytest.raises(module.WorkerFailure, match="sequences differ"):
        module.validate_predicted_pdb(_pdb("AD"), ("AC",))
