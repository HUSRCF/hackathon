from __future__ import annotations

import json
from pathlib import Path

import pytest

from protbind_agent.screening_benchmark import (
    bootstrap_complete_score_metrics,
    complete_score_metrics,
)
from protbind_agent.screening_protocol import (
    authorize_validation_once,
    freeze_screening_protocol,
    verify_validation_authorization,
)


def test_complete_metrics_are_tie_invariant_and_separate_perfect_from_random() -> None:
    labels = {"a": True, "b": True, "i1": False, "i2": False, "i3": False, "i4": False}
    perfect = complete_score_metrics(
        {"a": 2.0, "b": 1.0, "i1": 0.0, "i2": 0.0, "i3": 0.0, "i4": 0.0},
        labels,
    )
    tied_left = complete_score_metrics({key: 0.0 for key in labels}, labels)
    tied_right = complete_score_metrics(
        {key: 0.0 for key in reversed(tuple(labels))},
        dict(reversed(tuple(labels.items()))),
    )

    assert perfect["average_precision"] == 1.0
    assert perfect["roc_auc"] == 1.0
    assert perfect["bedroc"]["value"] == pytest.approx(1.0)
    assert tied_left == tied_right
    assert tied_left["average_precision"] == pytest.approx(2 / 6)
    assert tied_left["roc_auc"] == 0.5
    assert tied_left["cutoffs"]["0.010000"]["boundary_is_tied"] is True


def test_bootstrap_is_deterministic() -> None:
    labels = {"a": True, "b": True, "i1": False, "i2": False}
    scores = {"a": 1.0, "b": 0.0, "i1": 0.5, "i2": 0.0}
    first = bootstrap_complete_score_metrics(scores, labels, replicates=20, seed=7)
    second = bootstrap_complete_score_metrics(scores, labels, replicates=20, seed=7)
    assert first == second


def _split_files(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for role, contents in (
        ("active_train", "CC a\n"),
        ("inactive_train", "CCC i\n"),
        ("active_validation", "CCCC av\n"),
        ("inactive_validation", "CCCCC iv\n"),
    ):
        path = tmp_path / f"{role}.smi"
        path.write_text(contents, encoding="utf-8")
        paths[role] = path
    return paths


def test_protocol_rejects_exposed_target_and_consumes_authorization_once(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "dataset.tar.gz"
    archive.write_bytes(b"fixture")
    code = tmp_path / "code.py"
    code.write_text("# fixed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exposed"):
        freeze_screening_protocol(
            dataset_name="fixture",
            dataset_archive=archive,
            targets={"TP53": _split_files(tmp_path)},
            exposed_targets={"TP53"},
            hyperparameters={},
            code_paths=(code,),
            output=tmp_path / "bad.json",
        )

    protocol_path = tmp_path / "protocol.json"
    protocol = freeze_screening_protocol(
        dataset_name="fixture",
        dataset_archive=archive,
        targets={"ALDH1": _split_files(tmp_path)},
        exposed_targets={"TP53"},
        hyperparameters={"ensemble_sizes": [4, 8, 16]},
        code_paths=(code,),
        output=protocol_path,
    )
    assert protocol["prospective_targets"]["ALDH1"]["active_validation"][
        "record_count"
    ] == 1
    receipt_path = tmp_path / "authorization.json"
    receipt = authorize_validation_once(
        protocol_path=protocol_path,
        target="ALDH1",
        receipt_path=receipt_path,
    )
    assert receipt["status"] == "CONSUMED"
    verified = verify_validation_authorization(
        protocol_path=protocol_path,
        target="ALDH1",
        receipt_path=receipt_path,
    )
    assert verified["authorization"]["target"] == "ALDH1"
    with pytest.raises(FileExistsError):
        authorize_validation_once(
            protocol_path=protocol_path,
            target="ALDH1",
            receipt_path=receipt_path,
        )

    tampered = json.loads(protocol_path.read_text(encoding="utf-8"))
    tampered["hyperparameters"]["ensemble_sizes"] = [999]
    protocol_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        authorize_validation_once(
            protocol_path=protocol_path,
            target="ALDH1",
            receipt_path=tmp_path / "tampered.json",
        )
