from __future__ import annotations

import json
from pathlib import Path

from protbind_agent.artifacts import sha256_file
from protbind_agent.screening_benchmark import (
    export_training_ensemble_pharmer_panel,
    freeze_pharmer_query_subset,
    freeze_pharmer_triangle_panel,
    prepare_smiles_conformer_sdf,
    ranked_screen_metrics,
)


def test_freeze_query_subset_is_label_blind_and_shared(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "points": [
                    {"enabled": True, "name": "Aromatic", "x": 0, "y": 0, "z": 0},
                    {
                        "enabled": True,
                        "name": "HydrogenDonor",
                        "x": 3,
                        "y": 0,
                        "z": 0,
                    },
                    {
                        "enabled": True,
                        "name": "HydrogenAcceptor",
                        "x": 0,
                        "y": 4,
                        "z": 0,
                    },
                    {"enabled": True, "name": "NegativeIon", "x": 0, "y": 0, "z": 5},
                    {"enabled": True, "name": "Hydrophobic", "x": 0, "y": 0, "z": 0},
                ]
            }
        ),
        encoding="utf-8",
    )
    pharmer = tmp_path / "pharmer.json"
    tripharm = tmp_path / "tripharm.json"
    receipt = tmp_path / "receipt.json"
    result = freeze_pharmer_query_subset(
        source, pharmer, tripharm, receipt, max_points=4
    )
    assert result["selected_source_indices"] == [0, 1, 2, 3]
    assert sum(point["enabled"] for point in json.loads(pharmer.read_text())["points"]) == 4
    assert len(json.loads(tripharm.read_text())["features"]) == 4


def test_ranked_screen_metrics_marks_short_ranking_incomplete() -> None:
    metrics = ranked_screen_metrics(
        ["a", "x"],
        {"a": True, "b": True, "x": False, "y": False},
        fractions=(0.5, 1.0),
    )
    assert metrics["active_retrieved_total"] == 1
    assert metrics["cutoffs"]["0.500000"] == {
        "cutoff": 2,
        "retrieved": 2,
        "true_positives": 1,
        "status": "COMPLETE",
        "recall": 0.5,
        "enrichment_factor": 1.0,
    }
    assert metrics["cutoffs"]["1.000000"] == {
        "cutoff": 4,
        "retrieved": 2,
        "true_positives": 1,
        "status": "INCOMPLETE",
    }


def test_freeze_triangle_panel_uses_same_full_feature_set(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "points": [
                    {"enabled": True, "name": "Aromatic", "x": 0, "y": 0, "z": 0},
                    {
                        "enabled": True,
                        "name": "HydrogenDonor",
                        "x": 3,
                        "y": 0,
                        "z": 0,
                    },
                    {
                        "enabled": True,
                        "name": "HydrogenAcceptor",
                        "x": 0,
                        "y": 4,
                        "z": 0,
                    },
                    {"enabled": True, "name": "NegativeIon", "x": 0, "y": 0, "z": 5},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = freeze_pharmer_triangle_panel(
        source,
        tmp_path / "panel",
        tmp_path / "tripharm.json",
        tmp_path / "receipt.json",
        max_triangles=3,
    )
    assert result["feature_count"] == 4
    assert result["triangle_count"] == 3
    assert len(json.loads((tmp_path / "tripharm.json").read_text())["features"]) == 4
    for item in result["panel"]:
        query = json.loads((tmp_path / "panel" / item["file"]).read_text())
        assert sum(point["enabled"] for point in query["points"]) == 3


def test_failed_conformer_records_retain_label_and_denominator_identity(
    tmp_path: Path,
) -> None:
    active = tmp_path / "active.smi"
    inactive = tmp_path / "inactive.smi"
    active.write_text("invalid active-source\nCCO active-ok\n", encoding="utf-8")
    inactive.write_text("CCC inactive-ok\n", encoding="utf-8")
    result = prepare_smiles_conformer_sdf(
        active,
        inactive,
        tmp_path / "library.sdf",
        tmp_path / "labels.json",
        max_conformers=1,
    )
    assert result["counts"]["failed"] == 1
    assert result["failures"][0]["record_id"] == "active-000001"
    assert result["failures"][0]["label"] == "active"


def test_selected_training_queries_export_to_pharmer_triangles(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    query = candidates / "query-00.json"
    query.write_text(
        json.dumps(
            {
                "features": [
                    {"type": "Donor", "position": [0, 0, 0]},
                    {"type": "Acceptor", "position": [3, 0, 0]},
                    {"type": "Aromatic", "position": [0, 4, 0]},
                ]
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"selected_query_sha256": {query.name: sha256_file(query)}}),
        encoding="utf-8",
    )
    result = export_training_ensemble_pharmer_panel(
        selection_receipt=selection,
        candidate_dir=candidates,
        output_dir=tmp_path / "panel",
    )
    assert result["triangle_query_count"] == 1
    exported = json.loads((tmp_path / "panel" / "q00-t000.json").read_text())
    assert {point["name"] for point in exported["points"]} == {
        "HydrogenDonor",
        "HydrogenAcceptor",
        "Aromatic",
    }
