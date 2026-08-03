from __future__ import annotations

from pathlib import Path

import pytest

from protbind_agent.experimental_assays import ExperimentalAssayStore
from protbind_agent.models import ArtifactRef

HEADER = (
    "experiment_id,assay_type,target_id,candidate_id,batch_id,lab_id,"
    "condition_id,replicate,concentration,concentration_unit,response,"
    "response_unit,control_type\n"
)


def _assay_csv(path: Path, *, experiment_id: str = "exp-1") -> None:
    concentrations = (0.01, 0.01, 0.1, 0.1, 1.0, 1.0, 10.0, 10.0)
    responses = (1.0, 1.1, 1.8, 1.9, 5.4, 5.6, 9.1, 9.3)
    rows = []
    for index, (concentration, response) in enumerate(
        zip(concentrations, responses, strict=True), start=1
    ):
        rows.append(
            f"{experiment_id},enzyme-activity,MTORC1,compound-1,batch-1,lab-a,"
            f"dose-{concentration},{1 if index % 2 else 2},{concentration},uM,"
            f"{response},percent,treatment"
        )
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")


def test_assay_preview_is_hash_bound_and_non_mutating(tmp_path: Path) -> None:
    source = tmp_path / "assay.csv"
    _assay_csv(source)
    store = ExperimentalAssayStore(tmp_path / "workspace")

    preview = store.preview_import(source)

    assert preview["experiment_id"] == "exp-1"
    assert preview["row_count"] == 8
    assert preview["writes_performed"] is False
    assert preview["raw_measurements_returned"] is False
    assert not store.database.exists()


def test_assay_commit_requires_fresh_matching_plan_and_is_append_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "assay.csv"
    _assay_csv(source)
    store = ExperimentalAssayStore(tmp_path / "workspace")
    preview = store.preview_import(source)

    with pytest.raises(PermissionError, match="fresh private-data approval"):
        store.commit_import(
            source,
            plan_id=preview["plan_id"],
            data_access_confirmed=False,
        )
    with pytest.raises(PermissionError, match="stale"):
        store.commit_import(
            source,
            plan_id="0" * 64,
            data_access_confirmed=True,
        )

    result = store.commit_import(
        source,
        plan_id=preview["plan_id"],
        data_access_confirmed=True,
    )

    assert result["status"] == "IMPORTED"
    assert result["revision"] == 1
    assert store.database.is_file()
    receipt = store.artifacts.read_json(
        ArtifactRef.from_dict(result["receipt_artifact"])
    )
    assert receipt["mutation_policy"].startswith("append-only")

    with pytest.raises(FileExistsError, match="supersede"):
        store.commit_import(
            source,
            plan_id=preview["plan_id"],
            data_access_confirmed=True,
        )


def test_assay_plan_becomes_stale_after_source_change(tmp_path: Path) -> None:
    source = tmp_path / "assay.csv"
    _assay_csv(source)
    store = ExperimentalAssayStore(tmp_path / "workspace")
    preview = store.preview_import(source)
    source.write_text(source.read_text(encoding="utf-8").replace("9.3", "9.4"), encoding="utf-8")

    with pytest.raises(PermissionError, match="stale"):
        store.commit_import(
            source,
            plan_id=preview["plan_id"],
            data_access_confirmed=True,
        )


def test_assay_four_parameter_fit_writes_cited_receipt(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    source = tmp_path / "assay.csv"
    _assay_csv(source)
    store = ExperimentalAssayStore(tmp_path / "workspace")
    preview = store.preview_import(source)
    store.commit_import(
        source,
        plan_id=preview["plan_id"],
        data_access_confirmed=True,
    )

    result = store.fit_curve(
        experiment_id="exp-1",
        model="four-parameter-logistic",
        data_access_confirmed=True,
    )

    assert result["status"] == "FITTED"
    assert result["point_count"] == 8
    assert result["r_squared"] > 0.98
    assert result["parameters"]["ec50"] > 0
    fit = store.artifacts.read_json(ArtifactRef.from_dict(result["fit_artifact"]))
    assert "does not by itself establish direct binding" in fit["scientific_semantics"]


def test_assay_rejects_nonfinite_measurements(tmp_path: Path) -> None:
    source = tmp_path / "assay.csv"
    _assay_csv(source)
    source.write_text(source.read_text(encoding="utf-8").replace("9.3", "NaN"), encoding="utf-8")

    with pytest.raises(ValueError, match="responses must be finite"):
        ExperimentalAssayStore(tmp_path / "workspace").preview_import(source)
