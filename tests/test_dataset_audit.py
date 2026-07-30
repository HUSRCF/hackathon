from __future__ import annotations

import json
from pathlib import Path

import pytest

import protbind_agent.cli as cli_module
from protbind_agent.dataset_audit import (
    DatasetAuditConfig,
    DatasetAuditIntegrityError,
    build_dataset_leakage_audit,
    load_dataset_leakage_audit,
    parse_split_spec,
    persist_dataset_leakage_audit,
)


def _write_smiles(path: Path, values: list[str]) -> None:
    path.write_text(
        "".join(f"{value} molecule-{index:02d}\n" for index, value in enumerate(values)),
        encoding="utf-8",
    )


def _build(
    train: Path,
    test: Path,
    *,
    config: DatasetAuditConfig | None = None,
) -> dict[str, object]:
    return build_dataset_leakage_audit(
        {"train": train, "test": test},
        dataset_name="synthetic audit fixture",
        dataset_version="1",
        dataset_license="CC0-1.0",
        dataset_source="synthetic:test",
        config=config,
    )


def test_exact_identity_and_scaffold_leakage_fail_broad_precondition(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.smi"
    test = tmp_path / "test.smi"
    _write_smiles(train, ["CCO", "c1ccccc1", "CC(=O)O"])
    _write_smiles(test, ["OCC", "Cc1ccccc1", "N#N"])

    result = _build(
        train,
        test,
        config=DatasetAuditConfig(similarity_threshold=0.5),
    )
    pair = result["pairwise"][0]

    assert pair["exact_parent_identity_overlap"]["unique_count"] == 1
    assert pair["scaffold_overlap"]["unique_count"] >= 2
    assert pair["morgan_similarity"]["status"] == "FULL"
    assert pair["morgan_similarity"]["maximum"] == 1.0
    assert result["gate"]["identity_novelty"]["status"] == "FAIL"
    assert result["gate"]["scaffold_novelty"]["status"] == "FAIL"
    assert result["gate"]["broad_generalisation_precondition"]["status"] == "FAIL"
    assert "CCO" not in json.dumps(result)
    assert result["privacy"]["raw_smiles_in_receipt"] is False


def test_clean_small_splits_pass_integrity_preconditions(tmp_path: Path) -> None:
    train = tmp_path / "train.smi"
    test = tmp_path / "test.smi"
    _write_smiles(train, ["CCO", "c1ccccc1"])
    _write_smiles(test, ["N#N", "C1CCCCC1"])

    result = _build(train, test)

    assert result["gate"]["parsing_complete"]["status"] == "PASS"
    assert result["gate"]["identity_novelty"]["status"] == "PASS"
    assert result["gate"]["analogue_novelty"]["status"] == "PASS"
    assert result["gate"]["scaffold_novelty"]["status"] == "PASS"
    assert result["gate"]["broad_generalisation_precondition"]["status"] == "PASS"
    assert "does not demonstrate" in result["gate"]["scientific_semantics"]


def test_partial_similarity_audit_cannot_establish_absence_of_analogues(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.smi"
    test = tmp_path / "test.smi"
    _write_smiles(train, ["CCO", "c1ccccc1"])
    _write_smiles(test, ["N#N", "C1CCCCC1"])

    result = _build(
        train,
        test,
        config=DatasetAuditConfig(max_similarity_comparisons=1),
    )
    similarity = result["pairwise"][0]["morgan_similarity"]

    assert similarity["status"] == "PARTIAL_DETERMINISTIC_SAMPLE"
    assert similarity["executed_comparison_count"] == 1
    assert result["gate"]["analogue_novelty"]["status"] == "INCOMPLETE"
    assert (
        result["gate"]["broad_generalisation_precondition"]["status"]
        == "INCOMPLETE"
    )


def test_invalid_and_empty_splits_fail_parsing_gate(tmp_path: Path) -> None:
    train = tmp_path / "train.smi"
    test = tmp_path / "test.smi"
    train.write_text("not-a-smiles broken\nCCO valid\n", encoding="utf-8")
    test.write_text("# no records\n", encoding="utf-8")

    result = _build(train, test)

    assert result["splits"]["train"]["invalid_record_count"] == 1
    assert result["splits"]["test"]["parsed_record_count"] == 0
    blockers = result["gate"]["parsing_complete"]["blockers"]
    assert "train:invalid_records=1" in blockers
    assert "test:no_valid_records" in blockers
    assert result["gate"]["broad_generalisation_precondition"]["status"] == "FAIL"


def test_within_split_duplicates_are_not_silently_deduplicated(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.smi"
    test = tmp_path / "test.smi"
    _write_smiles(train, ["CCO", "OCC"])
    _write_smiles(test, ["N#N"])

    result = _build(train, test)

    assert result["splits"]["train"]["duplicate_parent_record_count"] == 1
    uniqueness = result["gate"]["within_split_identity_uniqueness"]
    assert uniqueness["status"] == "FAIL"
    assert "train:duplicate_parent_records=1" in uniqueness["blockers"]
    assert result["gate"]["broad_generalisation_precondition"]["status"] == "FAIL"


def test_receipt_is_hash_bound_and_tamper_evident(tmp_path: Path) -> None:
    train = tmp_path / "train.smi"
    test = tmp_path / "test.smi"
    output = tmp_path / "audit.json"
    _write_smiles(train, ["CCO"])
    _write_smiles(test, ["N#N"])
    result = _build(train, test)

    persist_dataset_leakage_audit(result, output)
    assert load_dataset_leakage_audit(output)["audit_sha256"] == result["audit_sha256"]
    result["dataset"]["version"] = "tampered"

    with pytest.raises(DatasetAuditIntegrityError, match="hash mismatch"):
        persist_dataset_leakage_audit(result, tmp_path / "tampered.json")


@pytest.mark.parametrize(
    "source",
    (
        "/private/internal/dataset",
        "~/private/dataset",
        "https://user:secret@example.org/data.smi",
        "https://example.org/data.smi?token=secret",
        "file:///private/internal/dataset.smi",
        "C:\\private\\internal\\dataset.smi",
    ),
)
def test_dataset_source_rejects_internal_paths_and_url_secrets(
    tmp_path: Path,
    source: str,
) -> None:
    train = tmp_path / "train.smi"
    test = tmp_path / "test.smi"
    _write_smiles(train, ["CCO"])
    _write_smiles(test, ["N#N"])

    with pytest.raises(ValueError, match="dataset_source"):
        build_dataset_leakage_audit(
            {"train": train, "test": test},
            dataset_name="fixture",
            dataset_version="1",
            dataset_license="CC0-1.0",
            dataset_source=source,
        )


@pytest.mark.parametrize(
    "value",
    ("train", "=path.smi", "train=", "bad name=path.smi"),
)
def test_split_spec_rejects_ambiguous_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_split_spec(value)


def test_cli_dataset_audit_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train = tmp_path / "train.smi"
    test = tmp_path / "test.smi"
    output = tmp_path / "audit.json"
    _write_smiles(train, ["CCO"])
    _write_smiles(test, ["N#N"])
    args = cli_module._build_parser().parse_args(
        [
            "benchmark",
            "dataset-audit",
            "--split",
            f"train={train}",
            "--split",
            f"test={test}",
            "--dataset-name",
            "fixture",
            "--dataset-version",
            "1",
            "--dataset-license",
            "CC0-1.0",
            "--dataset-source",
            "synthetic:test",
            "--output",
            str(output),
        ]
    )

    assert cli_module._run(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == "PROTBIND_DATASET_LEAKAGE_AUDIT"
    assert summary["split_count"] == 2
    assert summary["gate"]["broad_generalisation_precondition"]["status"] == "PASS"
    assert output.is_file()
