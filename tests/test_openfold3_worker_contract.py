from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from protbind_agent.artifacts import (
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from protbind_agent.models import ArtifactRef
from protbind_agent.worker_protocol import (
    JsonSubprocessWorker,
    WorkerProvenance,
    WorkerRequest,
)

ROOT = Path(__file__).parents[1]
WORKER = ROOT / "workers" / "openfold3_worker.py"
OPENFOLD_REVISION = (
    "openfold3-0.4.3@0bb17be5199846e806b6347b6e17c6249c88ff1b"
)
OPENFOLD_SCM_NODE = "g0bb17be5199846e806b6347b6e17c6249c88ff1b"


def _fake_openfold_distribution(
    path: Path,
    *,
    scm_node: str = OPENFOLD_SCM_NODE,
) -> str:
    package = path / "openfold3"
    metadata = path / "openfold3-0.4.3.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    init_path = package / "__init__.py"
    init_path.write_text('__version__ = "0.4.3"\n', encoding="utf-8")
    module_path = package / "run_openfold.py"
    module_path.write_text(
        """from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path

import gemmi

MODEL_MODE = os.environ.get("OPENFOLD_FAKE_MODEL_MODE", "valid")

def write_complex(path, smiles):
    structure = gemmi.Structure()
    structure.name = "model"
    model = gemmi.Model(1)
    protein = gemmi.Chain("A")
    for residue_index, residue_name in enumerate(
        ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY"), start=1
    ):
        residue = gemmi.Residue()
        residue.name = residue_name
        residue.seqid = gemmi.SeqId(residue_index, " ")
        for atom_index, (atom_name, element) in enumerate(
            (("N", "N"), ("CA", "C"), ("C", "C"))
        ):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position(float(residue_index), float(atom_index), 0.0)
            residue.add_atom(atom)
        protein.add_residue(residue)
    model.add_chain(protein)
    if MODEL_MODE != "missing-ligand":
        ligand = gemmi.Chain("Z")
        residue = gemmi.Residue()
        residue.name = "LIG"
        residue.seqid = gemmi.SeqId(1, " ")
        elements = ("C", "C", "N") if smiles == "CCN" else ("C", "C", "C", "N", "O", "O")
        for atom_index, element in enumerate(elements, start=1):
            atom = gemmi.Atom()
            atom.name = f"{element}{atom_index}"
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position(float(atom_index), 5.0, 0.0)
            residue.add_atom(atom)
        ligand.add_residue(residue)
        model.add_chain(ligand)
    structure.add_model(model)
    structure.make_mmcif_document().write_file(str(path))

arguments = sys.argv[1:]

def option(name):
    index = arguments.index(name)
    return arguments[index + 1]

query_path = Path(option("--query-json"))
checkpoint_path = Path(option("--inference-ckpt-path"))
output_path = Path(option("--output-dir"))
runner_path = Path(option("--runner-yaml"))
samples = int(option("--num-diffusion-samples"))
query = json.loads(query_path.read_text(encoding="utf-8"))
templates_enabled = "--use-templates=true" in arguments
capture = {
    "arguments": arguments,
    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "query": query,
    "runner": runner_path.read_text(encoding="utf-8"),
}
Path(os.environ["OPENFOLD_FAKE_CAPTURE"]).write_text(
    json.dumps(capture, sort_keys=True), encoding="utf-8"
)
(output_path / "experiment_config.json").write_text(
    json.dumps({
        "experiment_settings": {
            "seeds": [20260721],
            "use_msa_server": False,
            "use_templates": templates_enabled,
        },
        "output_writer_settings": {
            "structure_format": "cif",
            "write_full_confidence_scores": True,
        },
    }),
    encoding="utf-8",
)
(output_path / "model_config.json").write_text(
    json.dumps({
        "settings": {
            "clear_cache_between_steps": True,
            "memory": {
                "eval": {
                    "use_triton_triangle_kernels": True,
                    "use_deepspeed_evo_attention": False,
                    "use_cueq_triangle_kernels": False,
                    "per_sample_token_cutoff": 0,
                    "per_sample_atom_cutoff": 0,
                    "offload_inference": {
                        "template_module": True,
                        "msa_module": True,
                        "confidence_heads": True,
                        "token_cutoff": 0,
                    },
                }
            },
        },
        "architecture": {
            "shared": {"diffusion": {"no_full_rollout_samples": samples}}
        },
    }),
    encoding="utf-8",
)
(output_path / "inference_query_set.json").write_text(
    json.dumps(query), encoding="utf-8"
)
for query_id in query["queries"]:
    query_output = output_path / query_id / "seed_20260721"
    query_output.mkdir(parents=True)
    for sample in range(1, samples + 1):
        prefix = f"{query_id}_seed_20260721_sample_{sample}"
        ligand_smiles = query["queries"][query_id]["chains"][-1]["smiles"]
        write_complex(query_output / f"{prefix}_model.cif", ligand_smiles)
        (query_output / f"{prefix}_confidences_aggregated.json").write_text(
            json.dumps({
                "sample_ranking_score": 0.76 - sample / 100,
                "avg_plddt": 0.8,
                "gpde": 0.2,
                "ptm": 0.7,
                "iptm": 0.6,
                "has_clash": False,
            }),
            encoding="utf-8",
        )
        (query_output / f"{prefix}_confidences.json").write_text(
            json.dumps({"plddt": [0.5], "pde": [[0.1]], "pae": [[0.2]]}),
            encoding="utf-8",
        )
    (query_output / "timing.json").write_text(
        json.dumps({"runtime_s": 1.25}), encoding="utf-8"
    )
""",
        encoding="utf-8",
    )
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: openfold3\nVersion: 0.4.3\n",
        encoding="utf-8",
    )
    (metadata / "entry_points.txt").write_text(
        "[console_scripts]\nrun_openfold = openfold3.run_openfold:cli\n",
        encoding="utf-8",
    )
    (metadata / "scm_version.json").write_text(
        json.dumps(
            {
                "tag": "0.4.3",
                "distance": 0,
                "node": scm_node,
                "dirty": False,
                "branch": "HEAD",
                "node_date": "2026-07-03",
            }
        ),
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text(
        "openfold3/__init__.py,,\n"
        "openfold3/run_openfold.py,,\n"
        "openfold3-0.4.3.dist-info/METADATA,,\n"
        "openfold3-0.4.3.dist-info/entry_points.txt,,\n"
        "openfold3-0.4.3.dist-info/scm_version.json,,\n"
        "openfold3-0.4.3.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    source_hashes = sorted(
        (
            str(file.relative_to(path)),
            sha256_file(file),
        )
        for file in (init_path, module_path)
    )
    return sha256_bytes(canonical_json_bytes(source_hashes))


def _request_fixture(
    tmp_path: Path,
    *,
    code_sha256: str | None = None,
    candidate_count: int = 1,
    with_template: bool = False,
    scm_node: str = OPENFOLD_SCM_NODE,
) -> tuple[ArtifactStore, WorkerRequest, Path, ArtifactRef]:
    store = ArtifactStore(tmp_path / "workspace")
    checkpoint = store.put_bytes(
        b"pinned OpenFold3 checkpoint fixture",
        media_type="application/octet-stream",
        producer="test.openfold-checkpoint",
        producer_version=OPENFOLD_REVISION,
    )
    environment_lock = store.put_bytes(
        b"[environments]\nopenfold3-rocm7 = ['openfold3']\n",
        media_type="text/plain",
        producer="test.openfold-environment-lock",
        producer_version=OPENFOLD_REVISION,
    )
    protein_chain: dict[str, object] = {"chain_id": "A", "sequence": "ACDEFG"}
    if with_template:
        template = store.put_bytes(
            b"data_direct_cif_template\n#\n",
            media_type="chemical/x-mmcif",
            producer="test.openfold-template",
        )
        protein_chain.update(
            {
                "template_cif": template.to_dict(),
                "template_chain_id": "A",
            }
        )
    batch = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.cofold-input-batch",
            "protein_chains": [protein_chain],
            "cofold_candidates": [
                {
                    "candidate_id": f"candidate-{index:03d}",
                    "molecule_id": f"molecule-{index:03d}",
                    "microstate_id": f"microstate-{index:03d}",
                    "canonical_isomeric_smiles": (
                        "C[C@H](N)C(=O)O" if index == 1 else "CCN"
                    ),
                    "heavy_element_counts": (
                        {"C": 3, "N": 1, "O": 2}
                        if index == 1
                        else {"C": 2, "N": 1}
                    ),
                }
                for index in range(1, candidate_count + 1)
            ],
        },
        producer="test.openfold-batch",
    )
    envelope = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.stage-input",
            "stage": "COFOLDED",
            "supporting_artifacts": {
                "support_openfold_batch": batch.to_dict(),
                "support_openfold_checkpoint": checkpoint.to_dict(),
                "support_openfold_environment_lock": environment_lock.to_dict(),
            },
        },
        producer="test.stage-envelope",
    )
    fake_site = tmp_path / "fake-site"
    package_source_sha256 = _fake_openfold_distribution(
        fake_site,
        scm_node=scm_node,
    )
    composite = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "adapter_sha256": sha256_file(WORKER),
                "protbind_runtime_sha256": sha256_bytes(
                    canonical_json_bytes(
                        [
                            (
                                str(path.relative_to(ROOT)),
                                sha256_file(path),
                            )
                            for path in sorted(
                                (ROOT / "src" / "protbind_agent").rglob("*.py")
                            )
                        ]
                    )
                ),
                "environment_lock_sha256": environment_lock.sha256,
                "openfold_package_source_sha256": package_source_sha256,
                "openfold_revision": OPENFOLD_REVISION,
            }
        )
    )
    request = WorkerRequest(
        job_id="openfold3-contract",
        engine="openfold3",
        input=envelope,
        parameters={
            "num_diffusion_samples": 2,
            "command_timeout_seconds": 30.0,
            "low_mem": True,
            "use_triton_triangle_kernels": True,
            "use_msa_server": False,
        },
        seed=20260721,
        provenance=WorkerProvenance(
            model_revision=OPENFOLD_REVISION,
            weight_sha256=checkpoint.sha256,
            code_sha256=code_sha256 or composite,
        ),
    )
    return store, request, fake_site, checkpoint


def test_openfold3_worker_runs_pinned_offline_cli_and_imports_artifacts(tmp_path) -> None:
    store, request, fake_site, checkpoint = _request_fixture(tmp_path)
    capture_path = tmp_path / "openfold-capture.json"

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(WORKER)),
        artifact_root=store.root,
        environment={
            "OPENFOLD_FAKE_CAPTURE": str(capture_path),
            "PROTBIND_TEST_RUNTIME": "1",
            "HIP_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(fake_site),
        },
    ).run(request)

    assert response.error is None
    assert response.provenance == request.provenance
    assert response.peak_vram_bytes is None
    assert response.timings_seconds["openfold_command"] >= 0
    assert response.warnings == (
        "peak VRAM is unavailable from the child CLI and was not fabricated",
    )

    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    arguments = capture["arguments"]
    assert len(arguments) == 13
    assert arguments[0] == "predict"
    assert arguments[1] == "--query-json"
    assert Path(arguments[2]).name == "query.json"
    assert arguments[3] == "--inference-ckpt-path"
    assert Path(arguments[4]).name == "checkpoint.pt"
    assert arguments[5:7] == ["--use-msa-server=False", "--use-templates=false"]
    assert arguments[7] == "--output-dir"
    assert Path(arguments[8]).name == "output"
    assert arguments[9:11] == ["--num-diffusion-samples", "2"]
    assert arguments[11] == "--runner-yaml"
    assert Path(arguments[12]).name == "runner.yaml"
    assert capture["checkpoint_sha256"] == checkpoint.sha256
    assert str(fake_site) not in arguments

    query_id = f"pb_0001_{sha256_bytes(b'candidate-001')[:8]}"
    assert capture["query"] == {
        "seeds": [20260721],
        "queries": {
            query_id: {
                "use_msas": False,
                "use_paired_msas": False,
                "use_main_msas": False,
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": "A",
                        "sequence": "ACDEFG",
                    },
                    {
                        "molecule_type": "ligand",
                        "chain_ids": "Z",
                        "smiles": "C[C@H](N)C(=O)O",
                    },
                ]
            }
        }
    }
    runner = capture["runner"]
    assert "seeds: [20260721]" in runner
    assert "use_msa_server: false" in runner
    assert "use_templates: false" in runner
    assert "    - low_mem" in runner
    assert "use_triton_triangle_kernels: true" in runner
    assert "precision: 32-true" in runner

    output_ids = {reference.artifact_id for reference in response.outputs}
    bundle = store.read_json(response.outputs[0])
    assert bundle["kind"] == "protbind.cofold-bundle"
    assert bundle["score_semantics"] == "model confidence only; not binding affinity"
    candidate = bundle["candidates"][0]
    assert candidate["candidate_id"] == "candidate-001"
    assert candidate["molecule_id"] == "molecule-001"
    assert candidate["microstate_id"] == "microstate-001"
    assert candidate["engine"] == "test-fixture-openfold3"
    assert candidate["confidence_value"] == 0.75
    assert len(candidate["samples"]) == 2
    nested = [
        ArtifactRef.from_dict(candidate["structure"]),
        ArtifactRef.from_dict(candidate["samples"][0]["confidence"]),
        ArtifactRef.from_dict(candidate["samples"][1]["structure"]),
        ArtifactRef.from_dict(candidate["samples"][1]["confidence"]),
        ArtifactRef.from_dict(bundle["query_manifest"]),
        ArtifactRef.from_dict(bundle["runner"]),
        ArtifactRef.from_dict(bundle["run_metadata"]),
    ]
    for reference in nested:
        assert reference.artifact_id in output_ids
        store.resolve(reference)
    structure = store.read_bytes(ArtifactRef.from_dict(candidate["structure"]))
    assert structure.startswith(b"data_model")

    manifest = store.read_json(ArtifactRef.from_dict(bundle["query_manifest"]))
    assert manifest["query_ids"] == [query_id]
    assert manifest["candidate_ids"] == ["candidate-001"]
    assert manifest["msa_server"] is False
    assert ArtifactRef.from_dict(manifest["checkpoint"]) == checkpoint
    assert ArtifactRef.from_dict(manifest["input_batch"]).producer == (
        "test.openfold-batch"
    )
    assert ArtifactRef.from_dict(manifest["environment_lock"]).producer == (
        "test.openfold-environment-lock"
    )
    assert manifest["provenance"] == request.provenance.to_dict()
    assert manifest["raw_outputs"]
    metadata = store.read_json(ArtifactRef.from_dict(bundle["run_metadata"]))
    assert metadata["checkpoint_name"] == "openfold3-p2-155k"
    assert metadata["precision"] == "32-true"
    assert metadata["runtime_attestation"]["official_release"] is False
    assert metadata["resource_policy"]["trainer_devices"] == 1
    assert metadata["resource_policy"]["hip_visible_device"] == "0"


def test_openfold3_worker_rejects_model_missing_ligand_z(tmp_path) -> None:
    store, request, fake_site, _ = _request_fixture(tmp_path)
    capture_path = tmp_path / "missing-ligand-capture.json"

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(WORKER)),
        artifact_root=store.root,
        environment={
            "OPENFOLD_FAKE_CAPTURE": str(capture_path),
            "OPENFOLD_FAKE_MODEL_MODE": "missing-ligand",
            "PROTBIND_TEST_RUNTIME": "1",
            "HIP_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(fake_site),
        },
    ).run(request)

    assert capture_path.is_file()
    assert response.error is not None
    assert response.error.code == "OUTPUT_INVALID"
    assert response.error.recoverable is False
    assert response.outputs == ()
    assert response.provenance is None


def test_openfold3_worker_supports_two_candidates_sharing_direct_cif_template(
    tmp_path: Path,
) -> None:
    store, request, fake_site, _ = _request_fixture(
        tmp_path,
        candidate_count=2,
        with_template=True,
    )
    capture_path = tmp_path / "shared-template-capture.json"

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(WORKER)),
        artifact_root=store.root,
        environment={
            "OPENFOLD_FAKE_CAPTURE": str(capture_path),
            "PROTBIND_TEST_RUNTIME": "1",
            "HIP_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(fake_site),
        },
    ).run(request)

    assert response.error is None
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert capture["arguments"][6] == "--use-templates=true"
    query_templates = []
    for query_id, query in capture["query"]["queries"].items():
        protein = query["chains"][0]
        assert protein["template_cif_chain_ids"] == ["A"]
        assert len(protein["template_cif_paths"]) == 1
        template_path = protein["template_cif_paths"][0]
        assert query_id in Path(template_path).name
        query_templates.append(template_path)
    assert len(query_templates) == 2
    assert len(set(query_templates)) == 2

    bundle = store.read_json(response.outputs[0])
    assert [item["candidate_id"] for item in bundle["candidates"]] == [
        "candidate-001",
        "candidate-002",
    ]
    manifest = store.read_json(ArtifactRef.from_dict(bundle["query_manifest"]))
    assert manifest["templates"] is True
    assert manifest["candidate_ids"] == ["candidate-001", "candidate-002"]


def test_openfold3_worker_rejects_wrong_scm_runtime_attestation(tmp_path) -> None:
    store, request, fake_site, _ = _request_fixture(
        tmp_path,
        scm_node="gdeadbeef",
    )
    capture_path = tmp_path / "wrong-scm-should-not-run.json"

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(WORKER)),
        artifact_root=store.root,
        environment={
            "OPENFOLD_FAKE_CAPTURE": str(capture_path),
            "PROTBIND_TEST_RUNTIME": "1",
            "HIP_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(fake_site),
        },
    ).run(request)

    assert response.error is not None
    assert response.error.code == "MODEL_RUNTIME_INVALID"
    assert response.error.recoverable is False
    assert response.outputs == ()
    assert response.provenance is None
    assert not capture_path.exists()


def test_openfold3_worker_rejects_composite_code_hash_mismatch(tmp_path) -> None:
    store, request, fake_site, _ = _request_fixture(tmp_path, code_sha256="f" * 64)
    capture_path = tmp_path / "should-not-run.json"

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(WORKER)),
        artifact_root=store.root,
        environment={
            "OPENFOLD_FAKE_CAPTURE": str(capture_path),
            "PROTBIND_TEST_RUNTIME": "1",
            "HIP_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(fake_site),
        },
    ).run(request)

    assert response.error is not None
    assert response.error.code == "PROVENANCE_MISMATCH"
    assert response.error.recoverable is False
    assert response.outputs == ()
    assert response.provenance is None
    assert not capture_path.exists()


def test_openfold3_worker_rejects_noncanonical_device_index(tmp_path) -> None:
    store, request, fake_site, _ = _request_fixture(tmp_path)
    capture_path = tmp_path / "noncanonical-device-should-not-run.json"

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(WORKER)),
        artifact_root=store.root,
        environment={
            "OPENFOLD_FAKE_CAPTURE": str(capture_path),
            "PROTBIND_TEST_RUNTIME": "1",
            "HIP_VISIBLE_DEVICES": "00",
            "PYTHONPATH": str(fake_site),
        },
    ).run(request)

    assert response.error is not None
    assert response.error.code == "RESOURCE_POLICY_VIOLATION"
    assert response.error.recoverable is False
    assert response.outputs == ()
    assert response.provenance is None
    assert not capture_path.exists()


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("low_mem", False),
        ("use_triton_triangle_kernels", False),
    ),
)
def test_openfold3_worker_rejects_nonproduction_memory_profile_before_cli(
    tmp_path, parameter: str, value: bool
) -> None:
    store, request, fake_site, _ = _request_fixture(tmp_path)
    capture_path = tmp_path / f"{parameter}-should-not-run.json"
    request = replace(
        request,
        parameters={**request.parameters, parameter: value},
    )

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(WORKER)),
        artifact_root=store.root,
        environment={
            "OPENFOLD_FAKE_CAPTURE": str(capture_path),
            "PROTBIND_TEST_RUNTIME": "1",
            "HIP_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(fake_site),
        },
    ).run(request)

    assert response.error is not None
    assert response.error.code == "OPENFOLD_INPUT_REJECTED"
    assert response.error.recoverable is False
    assert response.outputs == ()
    assert not capture_path.exists()


def test_openfold3_worker_rejects_conflicting_rocr_device_mask(tmp_path) -> None:
    store, request, fake_site, _ = _request_fixture(tmp_path)
    capture_path = tmp_path / "conflicting-mask-should-not-run.json"

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(WORKER)),
        artifact_root=store.root,
        environment={
            "OPENFOLD_FAKE_CAPTURE": str(capture_path),
            "PROTBIND_TEST_RUNTIME": "1",
            "HIP_VISIBLE_DEVICES": "0",
            "ROCR_VISIBLE_DEVICES": "1",
            "PYTHONPATH": str(fake_site),
        },
    ).run(request)

    assert response.error is not None
    assert response.error.code == "RESOURCE_POLICY_VIOLATION"
    assert response.error.recoverable is False
    assert response.outputs == ()
    assert response.provenance is None
    assert not capture_path.exists()
