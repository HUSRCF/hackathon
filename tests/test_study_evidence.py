from __future__ import annotations

import json
from pathlib import Path

import pytest

import protbind_agent.cli as cli_module
from protbind_agent.artifacts import canonical_json_bytes, sha256_bytes
from protbind_agent.study_evidence import (
    StudyEvidenceIntegrityError,
    build_academic_evidence,
    exact_mcnemar_two_sided,
    freeze_study_protocol,
    load_frozen_study_protocol,
    persist_academic_evidence,
    persist_frozen_study_protocol,
    wilson_interval,
)


def _protocol_draft(
    *,
    timing: str = "RETROSPECTIVE_AFTER_OUTCOME",
    scope: str = "PILOT_METHOD_VERIFICATION",
    frozen_case_count: int = 10,
    holdout_sha256: str = "b" * 64,
    candidate_manifest_sha256: str = "e" * 64,
    baseline_manifest_sha256: str = "f" * 64,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "kind": "PROTBIND_STUDY_PROTOCOL",
        "study_id": "repair-v2-pilot",
        "title": "Restrained side-chain repair pilot",
        "analysis_timing": timing,
        "scope": scope,
        "design": "PAIRED_FROZEN_HOLDOUT",
        "dataset_binding": {
            "name": "test holdout",
            "holdout_sha256": holdout_sha256,
            "holdout_selection_hash": "c" * 64,
            "frozen_case_count": frozen_case_count,
        },
        "arm_bindings": {
            "candidate": {
                "label": "candidate",
                "regression_manifest_sha256": candidate_manifest_sha256,
            },
            "baseline": {
                "label": "baseline",
                "regression_manifest_sha256": baseline_manifest_sha256,
            },
        },
        "primary_endpoint": {
            "id": "candidate_top1_pose_recovery_rate",
            "success_definition": (
                "PoseBusters-valid AND symmetry-aware heavy-atom RMSD <= 2.0 angstrom"
            ),
            "denominator": "all_frozen_cases",
        },
        "statistical_plan": {
            "confidence_level": 0.95,
            "alpha": 0.05,
            "paired_test": "EXACT_MCNEMAR_TWO_SIDED",
            "missing_case_policy": "COUNT_AS_FAILURE",
            "multiplicity": "ONE_PRIMARY_OTHERS_DESCRIPTIVE",
        },
        "claim_rules": [
            {
                "claim_id": "C1",
                "claim": "Candidate reaches at least 80% top-1 recovery.",
                "primary": True,
                "endpoint": "candidate_top1_pose_recovery_rate",
                "operator": "gte",
                "threshold": 0.8,
            },
            {
                "claim_id": "C2",
                "claim": "Candidate completes every frozen case.",
                "primary": False,
                "endpoint": "candidate_workflow_completion_rate",
                "operator": "eq",
                "threshold": 1.0,
            },
            {
                "claim_id": "C3",
                "claim": "Candidate is superior to baseline on paired top-1 recovery.",
                "primary": False,
                "endpoint": "paired_top1_superiority",
                "operator": "paired_exact_superiority",
            },
        ],
        "negative_controls": [
            {
                "id": "box-perturbation",
                "status": "NOT_RUN",
                "purpose": "Test sensitivity to an incorrect pocket.",
            }
        ],
        "ablations": [
            {
                "id": "repair-off",
                "status": "NOT_RUN",
                "purpose": "Separate repair availability from pose ranking.",
            }
        ],
        "scientific_boundaries": [
            "Docking pose recovery is not experimental binding evidence.",
        ],
    }


def _regression(
    outcomes: list[bool | None],
    *,
    completed: int,
    holdout_sha256: str = "b" * 64,
    manifest_sha256: str = "e" * 64,
) -> dict[str, object]:
    cases = []
    for index, outcome in enumerate(outcomes):
        metrics = (
            {"pose_recovery": {"top1_success": outcome, "top5_success": outcome}}
            if outcome is not None
            else None
        )
        cases.append(
            {
                "case_id": f"case-{index:02d}",
                "status": "METRICS_COMPLETED" if metrics is not None else "REDOCK_FAILED",
                "metrics": metrics,
            }
        )
    top1 = sum(outcome is True for outcome in outcomes)
    top5 = top1
    core = {
        "schema_version": "1.1",
        "analysis": "PROTBIND_REDOCK_REGRESSION",
        "evaluation_design": "FROZEN_HOLDOUT",
        "gate_complete": completed == len(outcomes),
        "denominators": {
            "frozen": len(outcomes),
            "completed": completed,
        },
        "pose_recovery_rates": {
            "top1": {
                "numerator": top1,
                "denominator": len(outcomes),
                "rate": top1 / len(outcomes),
            },
            "top5": {
                "numerator": top5,
                "denominator": len(outcomes),
                "rate": top5 / len(outcomes),
            },
        },
        "holdout": {
            "sha256": holdout_sha256,
            "selection_hash": "c" * 64,
        },
        "manifest": {
            "manifest_sha256": manifest_sha256,
        },
        "cases": cases,
    }
    return {
        **core,
        "regression_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_wilson_interval_and_exact_mcnemar_small_sample() -> None:
    interval = wilson_interval(9, 10)

    assert interval["lower"] == pytest.approx(0.59585, abs=1e-5)
    assert interval["upper"] == pytest.approx(0.98212, abs=1e-5)
    assert exact_mcnemar_two_sided(1, 0) == 1.0
    assert exact_mcnemar_two_sided(10, 0) == pytest.approx(0.001953125)


def test_retrospective_protocol_cannot_claim_confirmatory_scope() -> None:
    with pytest.raises(StudyEvidenceIntegrityError, match="retrospective"):
        freeze_study_protocol(_protocol_draft(scope="CONFIRMATORY_BENCHMARK"))

def test_frozen_protocol_is_self_hashed_and_tamper_evident(tmp_path: Path) -> None:
    output = tmp_path / "protocol.json"
    protocol = freeze_study_protocol(_protocol_draft())
    persist_frozen_study_protocol(protocol, output)

    assert load_frozen_study_protocol(output) == protocol
    value = json.loads(output.read_text())
    value["title"] = "tampered"
    _write_json(output, value)

    with pytest.raises(StudyEvidenceIntegrityError, match="hash mismatch"):
        load_frozen_study_protocol(output)


def test_build_paired_pilot_keeps_raw_improvement_inconclusive(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    protocol = freeze_study_protocol(_protocol_draft())
    persist_frozen_study_protocol(protocol, protocol_path)
    _write_json(
        candidate_path,
        _regression(
            [False, True, True, True, True, True, True, True, True, True],
            completed=10,
        ),
    )
    _write_json(
        baseline_path,
        _regression(
            [False, True, True, True, None, True, True, True, True, True],
            completed=9,
            manifest_sha256="f" * 64,
        ),
    )

    packet = build_academic_evidence(
        protocol_path,
        candidate_path,
        baseline_result_path=baseline_path,
    )

    assert packet["candidate"]["top1_pose_recovery"]["rate"] == 0.9
    assert packet["candidate"]["top1_pose_recovery"]["interval"]["lower"] == pytest.approx(
        0.59585,
        abs=1e-5,
    )
    assert packet["paired_comparison"]["absolute_rate_difference"] == pytest.approx(0.1)
    assert packet["paired_comparison"]["candidate_only_success"] == 1
    assert packet["paired_comparison"]["baseline_only_success"] == 0
    assert packet["paired_comparison"]["exact_mcnemar_two_sided_p_value"] == 1.0
    statuses = {claim["claim_id"]: claim["status"] for claim in packet["claims"]}
    assert statuses == {
        "C1": "SUPPORTED",
        "C2": "SUPPORTED",
        "C3": "INCONCLUSIVE",
    }
    assert packet["interpretation"]["generalisation_claim"] == "NOT_EVALUATED"

    output = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    persist_academic_evidence(packet, output, markdown_output=markdown)
    assert json.loads(output.read_text())["evidence_sha256"] == packet["evidence_sha256"]
    report = markdown.read_text()
    assert "Exact two-sided McNemar p-value: 1.000000" in report
    assert "Generalisation, affinity, and screening hit-rate" in report


def test_pairing_rejects_different_holdouts(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    persist_frozen_study_protocol(
        freeze_study_protocol(_protocol_draft(frozen_case_count=2)),
        protocol_path,
    )
    _write_json(candidate_path, _regression([True, False], completed=2))
    _write_json(
        baseline_path,
        _regression(
            [True, False],
            completed=2,
            holdout_sha256="d" * 64,
            manifest_sha256="f" * 64,
        ),
    )

    with pytest.raises(StudyEvidenceIntegrityError, match="baseline holdout"):
        build_academic_evidence(
            protocol_path,
            candidate_path,
            baseline_result_path=baseline_path,
        )


def test_candidate_must_match_frozen_arm_manifest(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    candidate_path = tmp_path / "candidate.json"
    persist_frozen_study_protocol(
        freeze_study_protocol(_protocol_draft(frozen_case_count=2)),
        protocol_path,
    )
    _write_json(
        candidate_path,
        _regression([True, False], completed=2, manifest_sha256="a" * 64),
    )

    with pytest.raises(StudyEvidenceIntegrityError, match="candidate regression manifest"):
        build_academic_evidence(protocol_path, candidate_path)


def test_cli_freeze_and_evidence_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    draft_path = tmp_path / "draft.json"
    protocol_path = tmp_path / "protocol.json"
    candidate_path = tmp_path / "candidate.json"
    evidence_path = tmp_path / "evidence.json"
    _write_json(draft_path, _protocol_draft(frozen_case_count=2))
    _write_json(candidate_path, _regression([True, False], completed=2))

    freeze_args = cli_module._build_parser().parse_args(
        [
            "benchmark",
            "study-freeze",
            "--protocol",
            str(draft_path),
            "--output",
            str(protocol_path),
        ]
    )
    assert cli_module._run(freeze_args) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "PROTBIND_STUDY_PROTOCOL"

    evidence_args = cli_module._build_parser().parse_args(
        [
            "benchmark",
            "study-evidence",
            "--protocol",
            str(protocol_path),
            "--candidate-results",
            str(candidate_path),
            "--output",
            str(evidence_path),
        ]
    )
    assert cli_module._run(evidence_args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == "PROTBIND_ACADEMIC_EVIDENCE"
    assert summary["claim_statuses"]["C3"] == "NOT_EVALUATED"


def test_cli_refuses_to_overwrite_frozen_protocol_without_force(tmp_path: Path) -> None:
    draft_path = tmp_path / "draft.json"
    output = tmp_path / "protocol.json"
    _write_json(draft_path, _protocol_draft())
    output.write_text("prior protocol\n", encoding="utf-8")
    args = cli_module._build_parser().parse_args(
        [
            "benchmark",
            "study-freeze",
            "--protocol",
            str(draft_path),
            "--output",
            str(output),
        ]
    )

    with pytest.raises(FileExistsError, match="already exists"):
        cli_module._run(args)
    assert output.read_text(encoding="utf-8") == "prior protocol\n"
