from __future__ import annotations

import json
from pathlib import Path

import pytest

import protbind_agent.cli as cli_module
from protbind_agent.redock_regression import RegressionIntegrityError


def _result(*, design: str, gate_complete: bool) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "analysis": "PROTBIND_REDOCK_REGRESSION",
        "evaluation_design": design,
        "gate_complete": gate_complete,
        "regression_sha256": "a" * 64,
        "config": {
            "prolif_ligand_preparation_mode": (
                "RECEIPTED_LIGAND_ADDHS_AND_RECEPTOR_POCKET_CROP"
            ),
        },
        "denominators": {
            "frozen": 10,
            "attempted": 9,
            "completed": 8,
            "failed": 1,
            "metric_failed": 0,
            "not_attempted": 1,
            "metrics_completed": 8,
        },
        "pose_recovery_rates": {
            "top1": {"numerator": 5, "denominator": 10, "rate": 0.5},
            "top5": {"numerator": 7, "denominator": 10, "rate": 0.7},
        },
    }


def test_parser_exposes_hash_bound_regression_and_optional_addhs_store() -> None:
    args = cli_module._build_parser().parse_args(
        [
            "benchmark",
            "redock-regression",
            "--repo-root",
            "repo",
            "--manifest",
            "configs/pilot.json",
            "--output",
            "results/pilot.json",
            "--prolif-addhs-artifacts",
            "results/derived-prolif",
        ]
    )

    assert args.benchmark_command == "redock-regression"
    assert args.repo_root == Path("repo")
    assert args.manifest == Path("configs/pilot.json")
    assert args.output == Path("results/pilot.json")
    assert args.prolif_addhs_artifacts == Path("results/derived-prolif")


def test_pilot_persists_result_and_reports_non_gate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _result(design="PILOT_RETROSPECTIVE", gate_complete=False)
    observed: dict[str, object] = {}

    def fake_build(repo_root, manifest, *, prolif_artifact_store):
        observed["repo_root"] = repo_root
        observed["manifest"] = manifest
        observed["artifact_root"] = prolif_artifact_store.root
        return result

    def fake_persist(value, output):
        observed["persisted"] = value
        observed["output"] = output

    monkeypatch.setattr(cli_module, "build_redock_regression", fake_build)
    monkeypatch.setattr(cli_module, "persist_redock_regression", fake_persist)
    args = cli_module._build_parser().parse_args(
        [
            "benchmark",
            "redock-regression",
            "--repo-root",
            str(tmp_path),
            "--manifest",
            "configs/pilot.json",
            "--output",
            str(tmp_path / "pilot-result.json"),
            "--prolif-addhs-artifacts",
            str(tmp_path / "derived-prolif"),
        ]
    )

    assert cli_module._run(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "schema_version": "1.0",
        "analysis": "PROTBIND_REDOCK_REGRESSION",
        "evaluation_design": "PILOT_RETROSPECTIVE",
        "gate_status": "PILOT_NOT_ELIGIBLE",
        "gate_complete": False,
        "process_exit_code": 0,
        "regression_sha256": "a" * 64,
        "prolif_ligand_preparation_mode": (
            "RECEIPTED_LIGAND_ADDHS_AND_RECEPTOR_POCKET_CROP"
        ),
        "denominators": result["denominators"],
        "pose_recovery_rates": result["pose_recovery_rates"],
    }
    assert observed["repo_root"] == tmp_path
    assert observed["manifest"] == Path("configs/pilot.json")
    assert observed["artifact_root"] == (tmp_path / "derived-prolif").resolve()
    assert observed["persisted"] is result
    assert observed["output"] == tmp_path / "pilot-result.json"


@pytest.mark.parametrize(
    ("gate_complete", "expected_status", "expected_exit"),
    ((True, "PASS", 0), (False, "INCOMPLETE", 3)),
)
def test_frozen_gate_exit_status_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gate_complete: bool,
    expected_status: str,
    expected_exit: int,
) -> None:
    result = _result(design="FROZEN_HOLDOUT", gate_complete=gate_complete)
    persisted: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        cli_module,
        "build_redock_regression",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        cli_module,
        "persist_redock_regression",
        lambda value, output: persisted.append((value, output)),
    )
    args = cli_module._build_parser().parse_args(
        [
            "benchmark",
            "redock-regression",
            "--manifest",
            "configs/frozen.json",
            "--output",
            str(tmp_path / "frozen-result.json"),
        ]
    )

    assert cli_module._run(args) == expected_exit
    summary = json.loads(capsys.readouterr().out)
    assert summary["gate_status"] == expected_status
    assert summary["process_exit_code"] == expected_exit
    assert summary["gate_complete"] is gate_complete
    assert persisted == [(result, tmp_path / "frozen-result.json")]


def test_regression_command_requires_manifest(tmp_path: Path) -> None:
    args = cli_module._build_parser().parse_args(
        [
            "benchmark",
            "redock-regression",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    with pytest.raises(ValueError, match="requires --manifest"):
        cli_module._run(args)


def test_integrity_failure_uses_standard_error_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args, **_kwargs):
        raise RegressionIntegrityError("manifest SHA-256 mismatch")

    monkeypatch.setattr(cli_module, "build_redock_regression", fail)

    assert cli_module.main(
        [
            "benchmark",
            "redock-regression",
            "--manifest",
            "configs/frozen.json",
            "--output",
            str(tmp_path / "result.json"),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: manifest SHA-256 mismatch\n"
    assert not (tmp_path / "result.json").exists()
