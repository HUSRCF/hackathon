from __future__ import annotations

import json
from types import SimpleNamespace

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.experience import ExperienceStore
from protbind_agent.manifest import RunState
from protbind_agent.models import (
    LigandHypothesis,
    PrivacyPolicy,
    ResearchCase,
    ResearchMode,
    TargetSpec,
)


def test_experience_is_derived_from_reported_artifacts_and_is_only_a_hint(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    artifacts = ArtifactStore(workspace)
    receptor = artifacts.put_bytes(
        b"ATOM\n", media_type="chemical/x-pdb", producer="test-receptor"
    )
    selected = artifacts.put_json(
        {
            "candidates": [
                {"candidate_id": "candidate-a", "molecule_id": "molecule-a"}
            ]
        },
        producer="test-selection",
    )
    validation = artifacts.put_json(
        {
            "candidates": [
                {
                    "candidate_id": "candidate-a",
                    "molecule_id": "molecule-a",
                    "has_reference_pose": False,
                    "bundle": {
                        "preparation_attested": True,
                        "posebusters_valid": True,
                        "vina_pose_valid": True,
                        "evidence": [],
                    },
                }
            ]
        },
        producer="test-validation",
    )
    report = artifacts.put_bytes(
        b"# report\n",
        media_type="text/markdown",
        producer="test-report",
    )
    manifest_path = workspace / "runs" / "run-1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"run_id":"run-1"}\n', encoding="utf-8")
    stage_records = {
        RunState.SELECTED.value: SimpleNamespace(
            stage=RunState.SELECTED,
            outputs=(selected,),
        ),
        RunState.VALIDATED.value: SimpleNamespace(
            stage=RunState.VALIDATED,
            outputs=(validation,),
        ),
        RunState.REPORTED.value: SimpleNamespace(
            stage=RunState.REPORTED,
            outputs=(report,),
        ),
    }
    manifest = SimpleNamespace(
        run_id="run-1",
        last_completed_stage=RunState.REPORTED,
        input_artifacts={"target": receptor},
        artifacts={"report_markdown": report},
        stage_records=stage_records,
        failures=[],
    )
    case = ResearchCase(
        case_id="case-1",
        target=TargetSpec(name="target", sequences=("ACDEFG",), structure=receptor),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(smiles="CCO"),
        privacy=PrivacyPolicy(),
    )
    workflow = SimpleNamespace(
        artifacts=artifacts,
        manifests=SimpleNamespace(
            load=lambda run_id: manifest,
            path_for=lambda run_id: manifest_path,
        ),
        audit_manifest=lambda value: case,
    )
    store = ExperienceStore(workspace, workflow)

    written = store.write("run-1", preference="优先离线运行")
    hits = store.search("离线 molecule-a")
    record = artifacts.read_json(
        type(report).from_dict(written["artifact"])
    )

    assert written["evidence_grade"] == "HYPOTHESIS_ONLY"
    assert written["scientific_state_changed"] is False
    assert record["selected_candidates"] == ["candidate-a"]
    assert record["receptor_identity"] == receptor.artifact_id
    assert "ACDEFG" not in json.dumps(record)
    assert hits["hits"][0]["artifact"]["artifact_id"] == (
        f"sha256:{written['artifact']['sha256']}"
    )
    assert "must not copy boxes" in hits["semantics"]
