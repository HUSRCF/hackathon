from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.dossier import (
    build_run_dossier,
    dossier_content,
    persist_run_dossier,
)
from protbind_agent.manifest import STAGE_ORDER, RunManifest, RunState, StageRecord
from protbind_agent.pose_view import build_pose_scene_summary
from protbind_agent.web import create_app

pytest.importorskip("gemmi")
Chem = pytest.importorskip("rdkit.Chem")


def _pdb() -> bytes:
    return (
        b"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00"
        b"           N  \n"
        b"ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00"
        b"           C  \n"
        b"ATOM      3  C   ALA A   1       2.050   1.400   0.000  1.00 20.00"
        b"           C  \n"
        b"ATOM      4  O   ALA A   1       1.400   2.400   0.000  1.00 20.00"
        b"           O  \n"
        b"END\n"
    )


def _sdf() -> bytes:
    molecule = Chem.MolFromSmiles("CO")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.SetAtomPosition(0, (2.5, 0.0, 0.0))
    conformer.SetAtomPosition(1, (3.6, 0.0, 0.0))
    molecule.AddConformer(conformer)
    return (Chem.MolToMolBlock(molecule) + "\n$$$$\n").encode()


def _stage_record(stage: RunState, output, index: int) -> StageRecord:
    return StageRecord.create(
        stage,
        input_hash=f"{index + 1:064x}",
        config_hash=f"{index + 100:064x}",
        outputs=(output,),
        duration_seconds=float(index) / 10.0,
        warnings=(f"{stage.value} fixture warning",) if stage is RunState.SCREENED else (),
    )


def _workspace_with_pose(tmp_path: Path) -> tuple[ArtifactStore, RunManifest]:
    store = ArtifactStore(tmp_path)
    case = store.put_json({"case_id": "view-case"}, producer="test-case")
    receptor = store.put_bytes(
        _pdb(),
        media_type="chemical/x-pdb",
        producer="test-receptor",
    )
    pose = store.put_bytes(
        _sdf(),
        media_type="chemical/x-mdl-sdfile",
        producer="test-vina",
    )
    dummy = store.put_json({"fixture": True}, producer="test-stage")
    docking = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.docking-bundle",
            "receptor": receptor.to_dict(),
            "candidates": [
                {
                    "candidate_id": "vina-mol-1",
                    "parent_candidate_id": "selected-mol-1",
                    "molecule_id": "mol-1",
                    "microstate_id": "state-1",
                    "engine": "attested-local-autodock-vina",
                    "pose": pose.to_dict(),
                    "vina_score": -7.25,
                    "vina_score_semantics": (
                        "tool score only; not an experimental binding free energy"
                    ),
                    "box_center": [0.0, 0.0, 0.0],
                    "box_size": [20.0, 20.0, 20.0],
                }
            ],
        },
        producer="test-docking",
    )
    validation = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.validation-bundle",
            "candidates": [
                {
                    "candidate_id": "vina-mol-1",
                    "molecule_id": "mol-1",
                    "microstate_id": "state-1",
                    "docked_pose": pose.to_dict(),
                    "has_reference_pose": False,
                    "decision_reason": "geometry passed the fixture checks",
                    "bundle": {
                        "preparation_attested": True,
                        "posebusters_valid": True,
                        "vina_pose_valid": True,
                        "evidence": [],
                    },
                }
            ],
        },
        producer="test-validation",
    )
    manifest = RunManifest(
        run_id="view-run",
        case_id="view-case",
        case_artifact=case,
        artifacts={"receptor_ready_structure": receptor},
    )
    for index, stage in enumerate(STAGE_ORDER[1:-1], start=1):
        output = (
            docking
            if stage is RunState.DOCKED
            else validation
            if stage is RunState.VALIDATED
            else dummy
        )
        manifest.complete_stage(_stage_record(stage, output, index))
    return store, manifest


def test_pose_scene_is_coordinate_free_and_geometry_is_tool_derived(tmp_path) -> None:
    store, manifest = _workspace_with_pose(tmp_path)

    summary = build_pose_scene_summary(manifest, store)

    assert summary["available"] is True
    assert summary["candidate_count"] == 1
    scene = summary["candidates"][0]
    assert scene["validation"]["posebusters_valid"] is True
    assert scene["validation"]["evidence_grade"] == "HYPOTHESIS_ONLY"
    assert scene["geometry"]["available"] is True
    assert scene["geometry"]["minimum_heavy_atom_distance_angstrom"] == 1.05
    assert scene["geometry"]["all_ligand_heavy_atoms_inside_declared_box"] is True
    assert scene["coordinates_disclosed_to_agent"] is False
    assert scene["scene_artifact_id"].startswith("sha256:")
    serialized = json.dumps(summary)
    assert "ATOM      1" not in serialized
    assert "$$$$" not in serialized


def test_dossier_distinguishes_completed_from_closed_loop_accepted(tmp_path) -> None:
    store, manifest = _workspace_with_pose(tmp_path)
    acceptance = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.stage-acceptance",
            "phase": "POSTFLIGHT",
            "run_id": manifest.run_id,
            "stage": "INPUT_VALIDATED",
            "decision": "ACCEPTED",
            "manifest_sha256": "a" * 64,
            "checks": [],
            "required_actions": [],
            "continuation_token": "must-not-leak",
            "automatic_retry": False,
        },
        producer="protbind.stage-gate-receipt",
        producer_version="test",
    )
    history = {
        "schema_version": "1.0",
        "run_id": manifest.run_id,
        "policy_sha256": "b" * 64,
        "receipts": [acceptance.to_dict()],
    }

    dossier = build_run_dossier(
        manifest,
        store,
        control_history=history,
        pose_summary=build_pose_scene_summary(manifest, store),
    )
    references = persist_run_dossier(dossier, store)

    assert dossier["completion"]["completed_stage_count"] == 7
    assert dossier["completion"]["accepted_stage_count"] == 1
    assert dossier["completion"]["closed_loop_complete"] is False
    assert dossier["stages"][0]["status"] == "COMPLETED_ACCEPTED"
    assert dossier["stages"][1]["status"] == "COMPLETED_UNRECEIPTED"
    assert dossier["stages"][-1]["status"] == "NEXT"
    assert "must-not-leak" not in json.dumps(dossier)
    markdown = dossier_content(dossier, "markdown")
    assert "7/8" in markdown
    assert "1/8" in markdown
    assert "not experimental binding free energies" in markdown
    for reference in references.values():
        store.resolve(reference)


def test_loopback_web_pose_endpoints_serve_only_selected_artifacts(tmp_path) -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    store, manifest = _workspace_with_pose(tmp_path)
    from protbind_agent.manifest import ManifestStore

    ManifestStore(tmp_path).save(manifest)
    static = tmp_path / "static"
    static.mkdir()
    (static / "3Dmol-min.js").write_text("window.$3Dmol={};", encoding="utf-8")
    async def request_views():
        transport = httpx.ASGITransport(app=create_app(tmp_path))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1",
        ) as client:
            return (
                await client.get("/runs/view-run/poses/vina-mol-1"),
                await client.get("/api/runs/view-run/poses/vina-mol-1/receptor"),
                await client.get("/api/runs/view-run/poses/vina-mol-1/ligand"),
                await client.get("/api/runs/view-run/poses/vina-mol-1"),
            )

    page, receptor, ligand, metadata = asyncio.run(request_views())

    assert page.status_code == 200
    assert "Download local PNG" in page.text
    assert "/static/3Dmol-min.js" in page.text
    assert receptor.content == _pdb()
    assert ligand.content == _sdf()
    assert metadata.json()["coordinates_disclosed_to_agent"] is False
    assert "ATOM      1" not in metadata.text
