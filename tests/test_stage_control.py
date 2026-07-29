from __future__ import annotations

import json

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.control import GateDecision, StageGateController
from protbind_agent.manifest import RunState
from protbind_agent.models import (
    LigandHypothesis,
    ResearchCase,
    ResearchMode,
    TargetSpec,
)
from protbind_agent.tripharm import build_jsonl_index
from protbind_agent.workflow import ProtBindWorkflow


def _features(offset: float = 0.0) -> list[dict[str, object]]:
    return [
        {"type": "Donor", "position": [offset, 0.0, 0.0], "atom_indices": [0]},
        {"type": "Acceptor", "position": [offset + 3.0, 0.0, 0.0], "atom_indices": [1]},
        {"type": "Aromatic", "position": [offset, 4.0, 0.0], "atom_indices": [2]},
    ]


def _created_workflow(tmp_path, *, attach_receptor: bool) -> tuple[ProtBindWorkflow, str]:
    workspace = tmp_path / "workspace"
    records = tmp_path / "library.jsonl"
    records.write_text(
        json.dumps(
            {
                "molecule_id": "mol-a",
                "smiles": "CCO",
                "conformers": [{"id": 0, "features": _features(2.0)}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = tmp_path / "library.sqlite"
    build_jsonl_index(records, index)
    artifacts = ArtifactStore(workspace)
    query = artifacts.put_json(
        {"features": _features()},
        producer="test-stage-control-query",
    )
    case = ResearchCase(
        case_id="closed-loop-case",
        target=TargetSpec(name="target", sequences=("ACDEFG",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=query),
    )
    workflow = ProtBindWorkflow(workspace)
    manifest = workflow.create(case, index, run_id="closed-loop-run")
    if attach_receptor:
        receptor = tmp_path / "receptor.pdb"
        receptor.write_text("fixture receptor; not scientific coordinates\n", encoding="utf-8")
        workflow.attach_support(
            manifest,
            "receptor_structure",
            receptor,
            media_type="chemical/x-pdb",
        )
    return workflow, manifest.run_id


def test_controller_advances_exactly_one_stage_and_rejects_stale_token(tmp_path) -> None:
    workflow, run_id = _created_workflow(tmp_path, attach_receptor=True)
    controller = StageGateController(workflow)

    first = controller.inspect(run_id)
    assert first["gate"]["decision"] == GateDecision.READY
    assert first["gate"]["stage"] == RunState.INPUT_VALIDATED
    token = first["gate"]["continuation_token"]

    advanced = controller.advance(run_id, token)

    assert advanced["acceptance"]["decision"] == GateDecision.ACCEPTED
    assert advanced["acceptance"]["stage"] == RunState.INPUT_VALIDATED
    assert advanced["next_gate"]["gate"]["stage"] == RunState.RECEPTOR_READY
    manifest = workflow.manifests.load(run_id)
    assert manifest.last_completed_stage is RunState.INPUT_VALIDATED
    assert RunState.RECEPTOR_READY.value not in manifest.stage_records

    with pytest.raises(ValueError, match="stale or mismatched"):
        controller.advance(run_id, token)


def test_missing_receptor_stops_then_fresh_support_reopens_gate(tmp_path) -> None:
    workflow, run_id = _created_workflow(tmp_path, attach_receptor=False)
    controller = StageGateController(workflow)

    first = controller.inspect(run_id)
    advanced = controller.advance(run_id, first["gate"]["continuation_token"])
    blocked = advanced["next_gate"]

    assert blocked["gate"]["decision"] == GateDecision.NEEDS_ACTION
    assert blocked["gate"]["stage"] == RunState.RECEPTOR_READY
    assert blocked["gate"]["continuation_token"] is None
    assert "receptor" in " ".join(blocked["gate"]["required_actions"]).lower()

    receptor = tmp_path / "late-receptor.pdb"
    receptor.write_text("late fixture receptor\n", encoding="utf-8")
    manifest = workflow.manifests.load(run_id)
    workflow.attach_support(
        manifest,
        "receptor_structure",
        receptor,
        media_type="chemical/x-pdb",
    )
    reopened = controller.inspect(run_id)

    assert reopened["gate"]["decision"] == GateDecision.READY
    assert reopened["gate"]["stage"] == RunState.RECEPTOR_READY
    assert reopened["gate"]["continuation_token"] != first["gate"]["continuation_token"]


def test_controller_stops_before_unconfigured_selection_worker(tmp_path) -> None:
    workflow, run_id = _created_workflow(tmp_path, attach_receptor=True)
    controller = StageGateController(workflow)

    expected = (
        RunState.INPUT_VALIDATED,
        RunState.RECEPTOR_READY,
        RunState.INDEXED,
        RunState.SCREENED,
    )
    for stage in expected:
        gate = controller.inspect(run_id)["gate"]
        assert gate["stage"] == stage
        result = controller.advance(run_id, gate["continuation_token"])
        assert result["acceptance"]["decision"] == GateDecision.ACCEPTED

    blocked = controller.inspect(run_id)
    assert blocked["gate"]["stage"] == RunState.SELECTED
    assert blocked["gate"]["decision"] == GateDecision.NEEDS_ACTION
    assert blocked["gate"]["continuation_token"] is None
    assert workflow.manifests.load(run_id).last_completed_stage is RunState.SCREENED


def test_control_history_is_content_addressed_and_deduplicated(tmp_path) -> None:
    workflow, run_id = _created_workflow(tmp_path, attach_receptor=True)
    controller = StageGateController(workflow)

    first = controller.inspect(run_id)
    repeated = controller.inspect(run_id)
    history = controller.store.read(run_id)

    assert first["gate_receipt"] == repeated["gate_receipt"]
    assert len(history["receipts"]) == 1
    assert history["receipts"][0] == first["gate_receipt"]
