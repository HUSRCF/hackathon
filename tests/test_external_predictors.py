from __future__ import annotations

from pathlib import Path

import pytest

from protbind_agent.external_predictors import (
    drutai_admission_report,
    p2rank_command,
    parse_p2rank_predictions,
)


def test_p2rank_parser_emits_hypotheses_not_site_truth(tmp_path: Path) -> None:
    predictions = tmp_path / "target_predictions.csv"
    predictions.write_text(
        "name, rank, score, probability, center_x, center_y, center_z, "
        "residue_ids, surf_atom_ids\n"
        "target, 2, 5.1, 0.7, 4, 5, 6, A_2 A_3, 5 6\n"
        "target, 1, 9.2, 0.9, 1, 2, 3, A_1, 1 2\n",
        encoding="utf-8",
    )

    result = parse_p2rank_predictions(
        predictions,
        receptor_sha256="a" * 64,
        p2rank_version="P2Rank 2.5",
        top_k=1,
    )

    assert result["kind"] == "p2rank-site-hypotheses"
    assert result["pockets"][0]["rank"] == 1
    assert result["p2rank_version"] == "P2Rank 2.5"
    assert result["pockets"][0]["center"] == [1.0, 2.0, 3.0]
    assert result["biological_site_validity_inferred"] is False
    assert result["docking_box_validated"] is False


def test_p2rank_command_is_fixed_and_profile_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receptor = tmp_path / "target.pdb"
    receptor.write_text("END\n", encoding="utf-8")
    monkeypatch.setattr(
        "protbind_agent.external_predictors.shutil.which",
        lambda executable: "/opt/p2rank/prank",
    )

    command = p2rank_command(
        receptor,
        tmp_path / "output",
        profile="alphafold",
    )

    assert command[:2] == ("/opt/p2rank/prank", "predict")
    assert command[-2:] == ("-c", "alphafold")
    assert ";" not in " ".join(command)


def test_p2rank_parser_rejects_unpinned_version(tmp_path: Path) -> None:
    predictions = tmp_path / "target_predictions.csv"
    predictions.write_text(
        "rank,score,probability,center_x,center_y,center_z\n"
        "1,1,0.5,0,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pinned 2.5"):
        parse_p2rank_predictions(
            predictions,
            receptor_sha256="a" * 64,
            p2rank_version="P2Rank 2.6-alpha",
        )


def test_drutai_is_fail_closed_and_annotation_only() -> None:
    report = drutai_admission_report()

    assert report["status"] == "BLOCKED_PENDING_BAKEOFF"
    assert report["model_or_score_may_be_used"] is False
    assert "annotation-only" in report["scientific_role_if_admitted"]
    assert any("license conflict" in gate for gate in report["blocking_gates"])
