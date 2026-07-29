from __future__ import annotations

import os
import sys
from pathlib import Path

from protbind_agent.artifacts import (
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from protbind_agent.worker_protocol import (
    JsonSubprocessWorker,
    WorkerProvenance,
    WorkerRequest,
)

ROOT = Path(__file__).parents[1]
WORKER = ROOT / "workers" / "validation_worker.py"


def _tree_sha256(root: Path) -> str:
    entries = [
        (path.relative_to(root).as_posix(), sha256_file(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    return sha256_bytes(canonical_json_bytes(entries))


def _code_sha256() -> str:
    sources = [
        (str(path.relative_to(ROOT)), sha256_file(path))
        for path in sorted((ROOT / "src" / "protbind_agent").rglob("*.py"))
    ]
    runtime_sha = sha256_bytes(canonical_json_bytes(sources))
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "adapter_sha256": sha256_file(WORKER),
                "protbind_runtime_sha256": runtime_sha,
            }
        )
    )


def _fake_posebusters(path: Path) -> str:
    package = path / "posebusters"
    metadata = path / "posebusters-0.0.test.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    (package / "__init__.py").write_text(
        """from pathlib import Path
import pandas as pd

class PoseBusters:
    def __init__(self, *, config, top_n, max_workers, chunk_size):
        assert config in {"dock", "redock"}
        self.config = config
        assert top_n == 1
        assert max_workers == 0
        assert chunk_size == 1

    def bust(self, *, mol_pred, mol_cond, full_report, mol_true=None):
        assert full_report is False
        assert Path(mol_cond).is_file()
        assert (mol_true is not None) == (self.config == "redock")
        if mol_true is not None:
            assert Path(mol_true).is_file()
        valid = b"PB_INVALID" not in Path(mol_pred).read_bytes()
        row = {"loaded": True, "geometry": valid}
        if mol_true is not None:
            row["rmsd_≤_2a"] = False
        return pd.DataFrame([row])
""",
        encoding="utf-8",
    )
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: posebusters\nVersion: 0.0.test\n",
        encoding="utf-8",
    )
    return _tree_sha256(package)


def _fixture(
    tmp_path: Path,
    *,
    invalid_pose: bool = False,
    wrong_lineage: bool = False,
    without_cofold: bool = False,
    with_reference: bool = False,
    docking_schema: str = "1.0",
    code_sha256: str | None = None,
) -> tuple[ArtifactStore, WorkerRequest, Path]:
    fake_root = tmp_path / "fake-runtime"
    pb_source_sha = _fake_posebusters(fake_root)
    store = ArtifactStore(tmp_path / "workspace")

    docked_pose = store.put_bytes(
        b"original docked pose",
        media_type="chemical/x-pdbqt",
        producer="fixture-vina",
    )
    cofold_pose = store.put_bytes(
        b"original cofold complex",
        media_type="chemical/x-mmcif",
        producer="fixture-openfold3",
    )
    docking_candidate = {
        "candidate_id": "dock-mol-a",
        "molecule_id": "mol-a",
        "microstate_id": "state-1",
        "pose": docked_pose.to_dict(),
        "engine": "fixture-vina",
        "seed": 20260721,
    }
    if not without_cofold:
        docking_candidate["cofold_structure"] = cofold_pose.to_dict()
    docking = store.put_json(
        {
            "schema_version": docking_schema,
            "kind": "protbind.docking-bundle",
            "candidates": [docking_candidate],
        },
        producer="fixture-docking-worker",
    )
    ligand_bytes = b"PB_INVALID" if invalid_pose else b"fixture prepared ligand"
    docked_ligand = store.put_bytes(
        ligand_bytes,
        media_type="chemical/x-mdl-sdfile",
        producer="fixture-preparation",
    )
    cofold_ligand = store.put_bytes(
        b"fixture prepared cofold ligand",
        media_type="chemical/x-mdl-sdfile",
        producer="fixture-preparation",
    )
    docked_receptor = store.put_bytes(
        b"fixture docked receptor",
        media_type="chemical/x-pdb",
        producer="fixture-preparation",
    )
    cofold_receptor = store.put_bytes(
        b"fixture cofold receptor",
        media_type="chemical/x-pdb",
        producer="fixture-preparation",
    )
    reference = store.put_bytes(
        b"fixture native ligand",
        media_type="chemical/x-mdl-sdfile",
        producer="fixture-reference",
    )
    batch_docked_pose = cofold_pose if wrong_lineage else docked_pose
    prepared_candidate = {
        "candidate_id": "dock-mol-a",
        "molecule_id": "mol-a",
        "microstate_id": "state-1",
        "docked_pose": batch_docked_pose.to_dict(),
        "posebusters": {
            "docked_ligand": docked_ligand.to_dict(),
            "docked_receptor": docked_receptor.to_dict(),
        },
    }
    if not without_cofold:
        prepared_candidate["cofold_pose"] = cofold_pose.to_dict()
        prepared_candidate["posebusters"].update(
            {
                "cofold_ligand": cofold_ligand.to_dict(),
                "cofold_receptor": cofold_receptor.to_dict(),
            }
        )
    if with_reference:
        prepared_candidate["reference_pose"] = reference.to_dict()
        prepared_candidate["posebusters"]["reference_ligand"] = reference.to_dict()
    batch = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.validation-input-batch",
            "docking_bundle": docking.to_dict(),
            "candidates": [prepared_candidate],
        },
        producer="test-fixture-validation-input",
    )
    toolchain = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.validation-toolchain-manifest",
            "test_fixture": True,
            "posebusters_configs": ["dock", "redock"] if with_reference else ["dock"],
            "tools": {
                "posebusters": {
                    "version": "0.0.test",
                    "package_source_sha256": pb_source_sha,
                }
            },
            "assets": [],
        },
        producer="test-fixture-validation-toolchain",
    )
    case = store.put_json(
        {"case_id": "fixture-case", "ligand": None},
        producer="test-fixture-case",
    )
    envelope = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.stage-input",
            "stage": "VALIDATED",
            "case": case.to_dict(),
            "input_artifacts": {},
            "supporting_artifacts": {
                "support_validation_batch": batch.to_dict(),
                "support_validation_toolchain": toolchain.to_dict(),
                **(
                    {"support_reference_pose": reference.to_dict()}
                    if with_reference
                    else {}
                ),
            },
            "previous": {
                "stage": "DOCKED",
                "scientific_outputs": [
                    docking.to_dict(),
                    docked_pose.to_dict(),
                    *([] if without_cofold else [cofold_pose.to_dict()]),
                ],
                "receipt": None,
            },
        },
        producer="protbind.worker-stage-input",
    )
    request = WorkerRequest(
        job_id="fixture-validated",
        engine="protbind-validation",
        input=envelope,
        parameters={},
        seed=20260721,
        provenance=WorkerProvenance(
            model_revision=f"validation-toolchain:{toolchain.sha256}",
            weight_sha256=toolchain.sha256,
            code_sha256=code_sha256 or _code_sha256(),
        ),
    )
    return store, request, fake_root


def _run(
    store: ArtifactStore, request: WorkerRequest, fake_root: Path
):
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(fake_root)
    if existing_pythonpath:
        pythonpath = os.pathsep.join((pythonpath, existing_pythonpath))
    worker = JsonSubprocessWorker(
        (sys.executable, str(WORKER)),
        artifact_root=store.root,
        environment={
            "PROTBIND_TEST_RUNTIME": "1",
            "PYTHONPATH": pythonpath,
        },
    )
    return worker.run(request)[0]


def test_fixture_posebusters_contract_is_explicit_and_lineage_bound(tmp_path: Path) -> None:
    store, request, fake_root = _fixture(tmp_path)

    response = _run(store, request, fake_root)

    assert response.error is None
    assert response.warnings == (
        "protocol fixture runtime; outputs are not scientific evidence",
    )
    bundle = store.read_json(response.outputs[0])
    assert bundle["test_fixture"] is True
    candidate = bundle["candidates"][0]
    assert candidate["candidate_id"] == "dock-mol-a"
    assert candidate["has_reference_pose"] is False
    assert candidate["bundle"]["posebusters_valid"] is True
    assert candidate["bundle"]["vina_pose_valid"] is True
    assert candidate["bundle"]["cofold_pose_valid"] is True
    assert candidate["bundle"]["preparation_attested"] is False
    assert candidate["bundle"]["unsupported_reasons"] == [
        "validation-preparation: exact pose/receptor derivation receipts are missing, "
        "incomplete, or fixture-only; grade is capped at HYPOTHESIS_ONLY",
        "spyrmsd: not pinned in the validation toolchain",
        "prolif: not pinned in the validation toolchain",
        "openmm: not pinned in the validation toolchain",
    ]
    assert candidate["engine"] == "test-fixture-validation"
    assert len(response.outputs) == 2
    evidence_ref = response.outputs[1]
    assert evidence_ref.producer == "test-fixture-posebusters"
    evidence = store.read_json(evidence_ref)
    assert evidence["tool"] == "posebusters"
    assert evidence["test_fixture"] is True
    assert evidence["metrics"]["valid"] is True
    assert evidence["inputs"]["docked_pose"] == candidate["docked_pose"]
    assert evidence["inputs"]["cofold_pose"] == candidate["cofold_pose"]
    assert evidence["runtime"]["version"] == "0.0.test"


def test_real_posebusters_false_is_a_result_not_a_worker_error(tmp_path: Path) -> None:
    store, request, fake_root = _fixture(tmp_path, invalid_pose=True)

    response = _run(store, request, fake_root)

    assert response.error is None
    bundle = store.read_json(response.outputs[0])
    candidate = bundle["candidates"][0]
    assert candidate["bundle"]["posebusters_valid"] is False
    assert candidate["bundle"]["vina_pose_valid"] is False
    assert candidate["bundle"]["cofold_pose_valid"] is True
    assert "docked Vina pose failed" in candidate["decision_reason"]
    evidence = store.read_json(response.outputs[1])
    assert evidence["metrics"]["docked_valid"] is False
    assert evidence["metrics"]["cofold_valid"] is True


def test_vina_only_reference_uses_redock_without_requiring_cofold(tmp_path: Path) -> None:
    store, request, fake_root = _fixture(
        tmp_path,
        without_cofold=True,
        with_reference=True,
        docking_schema="2.0",
    )

    response = _run(store, request, fake_root)

    assert response.error is None
    bundle = store.read_json(response.outputs[0])
    candidate = bundle["candidates"][0]
    assert candidate["has_reference_pose"] is True
    assert "cofold_pose" not in candidate
    assert candidate["bundle"]["posebusters_valid"] is True
    assert candidate["bundle"]["vina_pose_valid"] is True
    assert candidate["bundle"]["cofold_pose_valid"] is None
    evidence = store.read_json(response.outputs[1])
    assert evidence["metrics"]["config"] == "redock"


def test_validation_worker_rejects_pose_lineage_mismatch(tmp_path: Path) -> None:
    store, request, fake_root = _fixture(tmp_path, wrong_lineage=True)

    response = _run(store, request, fake_root)

    assert response.error is not None
    assert response.error.code == "LINEAGE_MISMATCH"
    assert response.error.recoverable is False
    assert response.outputs == ()


def test_validation_worker_attests_adapter_code(tmp_path: Path) -> None:
    store, request, fake_root = _fixture(tmp_path, code_sha256="f" * 64)

    response = _run(store, request, fake_root)

    assert response.error is not None
    assert response.error.code == "CODE_HASH_MISMATCH"
    assert response.outputs == ()
