from __future__ import annotations

import json
from pathlib import Path

import pytest

import protbind_agent.cli as cli_module
from protbind_agent.research_leakage import (
    ResearchLeakageConfig,
    ResearchLeakageIntegrityError,
    build_research_leakage_audit,
    global_edit_identity,
    load_research_leakage_audit,
    persist_research_leakage_audit,
)


def _record(
    record_id: str,
    split: str,
    sequence: str,
    pocket_sha: str,
    pocket_cluster: str,
    pdb_id: str,
    release_date: str,
    assay_id: str,
    replicate_id: str,
    target: str,
    compound: str,
    label: object,
    *,
    sequence_cluster: str | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "split": split,
        "protein_sequence": sequence,
        "sequence_cluster_id": sequence_cluster or f"sequence-cluster-{record_id}",
        "pocket_artifact_sha256": pocket_sha,
        "pocket_cluster_id": pocket_cluster,
        "pdb_id": pdb_id,
        "pdb_release_date": release_date,
        "assay_id": assay_id,
        "replicate_group_id": replicate_id,
        "target_identity": target,
        "compound_parent_identity": compound,
        "label": label,
    }


def _manifest(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset": {
            "name": "synthetic cross-modal leakage fixture",
            "version": "1",
            "license": "CC0-1.0",
            "source": "synthetic:research-leakage-fixture",
        },
        "split_roles": {
            "train": "TRAIN",
            "test": "EVALUATION",
        },
        "pdb_training_cutoff_date": "2022-12-31",
        "sequence_cluster_protocol": {
            "method": "synthetic declared sequence clusters",
            "version": "1",
            "threshold_semantics": "identical fixture cluster IDs",
            "assignment_artifact_sha256": "9" * 64,
        },
        "pocket_cluster_protocol": {
            "method": "synthetic declared pocket clusters",
            "version": "1",
            "threshold_semantics": "identical fixture cluster IDs",
            "assignment_artifact_sha256": "a" * 64,
        },
        "provenance": {
            "sequence_source_artifact_sha256": "b" * 64,
            "pocket_source_artifact_sha256": "c" * 64,
            "pdb_metadata_artifact_sha256": "d" * 64,
            "assay_metadata_artifact_sha256": "e" * 64,
        },
        "records": records,
    }


def _positive_records() -> list[dict[str, object]]:
    return [
        _record(
            "train-1",
            "train",
            "ACDEFGHIK",
            "1" * 64,
            "pocket-cluster-1",
            "1AAA",
            "2020-01-01",
            "assay-1",
            "replicate-1",
            "target-1",
            "compound-1",
            1,
            sequence_cluster="sequence-cluster-1",
        ),
        _record(
            "train-2",
            "train",
            "MNPQRSTVW",
            "2" * 64,
            "pocket-cluster-2",
            "2BBB",
            "2022-01-01",
            "assay-2",
            "replicate-2",
            "target-2",
            "compound-2",
            0,
            sequence_cluster="sequence-cluster-2",
        ),
        _record(
            "test-1",
            "test",
            "ACDEFGHIK",
            "1" * 64,
            "pocket-cluster-1",
            "1AAA",
            "2020-01-01",
            "assay-1",
            "replicate-1",
            "target-1",
            "compound-1",
            0,
            sequence_cluster="sequence-cluster-1",
        ),
        _record(
            "test-2",
            "test",
            "YYYYYYYYY",
            "3" * 64,
            "pocket-cluster-3",
            "3CCC",
            "2023-01-01",
            "assay-3",
            "replicate-3",
            "target-3",
            "compound-3",
            1,
            sequence_cluster="sequence-cluster-3",
        ),
    ]


def _clean_records() -> list[dict[str, object]]:
    return [
        _record(
            "train-1",
            "train",
            "AAAAAAAAA",
            "1" * 64,
            "pocket-cluster-1",
            "1AAA",
            "2020-01-01",
            "assay-1",
            "replicate-1",
            "target-1",
            "compound-1",
            1,
        ),
        _record(
            "test-1",
            "test",
            "RRRRRRRRR",
            "2" * 64,
            "pocket-cluster-2",
            "2BBB",
            "2023-01-01",
            "assay-2",
            "replicate-2",
            "target-2",
            "compound-2",
            0,
        ),
    ]


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_global_edit_identity_has_explicit_deterministic_semantics() -> None:
    assert global_edit_identity("AAAA", "AAAA") == 1.0
    assert global_edit_identity("AAAA", "AAAT") == 0.75
    assert global_edit_identity("AAAA", "AAA") == 0.75
    assert global_edit_identity("AAAA", "RRRR") == 0.0


def test_positive_control_fails_all_four_receipts_without_raw_private_values(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write(manifest, _manifest(_positive_records()))

    result = build_research_leakage_audit(manifest)
    statuses = result["gate"]["component_statuses"]

    assert statuses == {
        "sequence_cluster": "FAIL",
        "pocket_cluster": "FAIL",
        "pdb_temporal": "FAIL",
        "assay_label": "FAIL",
    }
    assert (
        result["gate"]["broad_cross_modal_novelty_precondition"]["status"]
        == "FAIL"
    )
    sequence_pair = result["receipts"]["sequence_cluster"]["pairwise"][0]
    assert sequence_pair["exact_sequence_overlap_count"] == 1
    assert sequence_pair["declared_sequence_cluster_overlap_count"] == 1
    pocket_pair = result["receipts"]["pocket_cluster"]["pairwise"][0]
    assert pocket_pair["pocket_artifact_overlap_count"] == 1
    assert pocket_pair["declared_pocket_cluster_overlap_count"] == 1
    temporal = result["receipts"]["pdb_temporal"]
    assert temporal["pairwise"][0]["pdb_id_overlap_count"] == 1
    assert temporal["temporal_violations"]
    assay = result["receipts"]["assay_label"]["pairwise"][0]
    assert assay["assay_id_overlap_count"] == 1
    assert assay["replicate_group_overlap_count"] == 1
    assert assay["target_compound_pair_overlap_count"] == 1
    assert assay["conflicting_label_pair_count"] == 1

    receipt_text = json.dumps(result, ensure_ascii=False)
    for private_value in (
        "ACDEFGHIK",
        "assay-1",
        "replicate-1",
        "target-1",
        "compound-1",
        "train-1",
    ):
        assert private_value not in receipt_text
    assert result["privacy"]["raw_sequences_in_receipt"] is False
    assert result["privacy"]["raw_labels_in_receipt"] is False


def test_clean_control_passes_all_data_integrity_preconditions(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write(manifest, _manifest(_clean_records()))

    result = build_research_leakage_audit(manifest)

    assert result["gate"]["component_statuses"] == {
        "sequence_cluster": "PASS",
        "pocket_cluster": "PASS",
        "pdb_temporal": "PASS",
        "assay_label": "PASS",
    }
    assert (
        result["gate"]["broad_cross_modal_novelty_precondition"]["status"]
        == "PASS"
    )
    assert "does not demonstrate" in result["gate"][
        "broad_cross_modal_novelty_precondition"
    ]["semantics"]
    assert (
        result["receipts"]["sequence_cluster"]["cluster_method_verification"]
        == "NOT_EVALUATED"
    )
    assert (
        result["receipts"]["pocket_cluster"]["cluster_method_verification"]
        == "NOT_EVALUATED"
    )
    assert (
        result["receipts"]["pdb_temporal"]["rcsb_metadata_verification"]
        == "NOT_EVALUATED"
    )


def test_partial_sequence_comparison_keeps_bundle_incomplete(tmp_path: Path) -> None:
    records = [
        *_clean_records(),
        _record(
            "train-2",
            "train",
            "CCCCCCCCC",
            "3" * 64,
            "pocket-cluster-3",
            "3CCC",
            "2021-01-01",
            "assay-3",
            "replicate-3",
            "target-3",
            "compound-3",
            1,
        ),
        _record(
            "test-2",
            "test",
            "TTTTTTTTT",
            "4" * 64,
            "pocket-cluster-4",
            "4DDD",
            "2024-01-01",
            "assay-4",
            "replicate-4",
            "target-4",
            "compound-4",
            0,
        ),
    ]
    manifest = tmp_path / "manifest.json"
    _write(manifest, _manifest(records))

    result = build_research_leakage_audit(
        manifest,
        config=ResearchLeakageConfig(max_sequence_comparisons=1),
    )

    assert result["gate"]["component_statuses"]["sequence_cluster"] == "INCOMPLETE"
    assert result["receipts"]["sequence_cluster"]["pairwise"][0]["status"] == (
        "PARTIAL_DETERMINISTIC_SAMPLE"
    )
    assert (
        result["gate"]["broad_cross_modal_novelty_precondition"]["status"]
        == "INCOMPLETE"
    )


def test_nested_receipts_and_bundle_are_tamper_evident(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "receipt.json"
    _write(manifest, _manifest(_clean_records()))
    result = build_research_leakage_audit(manifest)

    persist_research_leakage_audit(result, output)
    assert load_research_leakage_audit(output)["bundle_sha256"] == result["bundle_sha256"]
    result["receipts"]["sequence_cluster"]["pairwise"][0][
        "maximum_global_edit_identity"
    ] = 1.0

    with pytest.raises(ResearchLeakageIntegrityError, match="sequence_cluster receipt"):
        persist_research_leakage_audit(result, tmp_path / "tampered.json")


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["dataset"].update({"source": "/private/data"}),
        lambda value: value["split_roles"].update({"test": "VALIDATION"}),
        lambda value: value["records"][0].update({"protein_sequence": "ACD-EF"}),
        lambda value: value["records"][0].update({"pdb_release_date": "2020"}),
        lambda value: value["records"][0].update({"label": None}),
    ),
)
def test_manifest_rejects_ambiguous_or_private_metadata(
    tmp_path: Path,
    mutation,
) -> None:
    value = _manifest(_clean_records())
    mutation(value)
    manifest = tmp_path / "manifest.json"
    _write(manifest, value)

    with pytest.raises(ResearchLeakageIntegrityError):
        build_research_leakage_audit(manifest)


def test_cli_research_leakage_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "receipt.json"
    _write(manifest, _manifest(_clean_records()))
    args = cli_module._build_parser().parse_args(
        [
            "benchmark",
            "research-leakage-audit",
            "--leakage-manifest",
            str(manifest),
            "--output",
            str(output),
        ]
    )

    assert cli_module._run(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == "PROTBIND_RESEARCH_LEAKAGE_AUDIT"
    assert summary["component_statuses"] == {
        "sequence_cluster": "PASS",
        "pocket_cluster": "PASS",
        "pdb_temporal": "PASS",
        "assay_label": "PASS",
    }
    assert summary["broad_cross_modal_novelty_precondition"]["status"] == "PASS"
    assert output.is_file()
