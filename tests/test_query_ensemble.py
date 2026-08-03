from pathlib import Path
from types import SimpleNamespace

from protbind_agent.query_ensemble import (
    freeze_training_query_candidates,
    prepare_inner_selection_split,
    select_training_query_ensemble,
)


def test_train_only_query_candidates_are_deterministic_and_private(tmp_path: Path) -> None:
    train = tmp_path / "active_T.smi"
    train.write_text(
        "CC(=O)O source-a\n"
        "c1ccccc1O source-b\n"
        "CCN source-c\n",
        encoding="utf-8",
    )
    first = freeze_training_query_candidates(train, tmp_path / "first", max_queries=2)
    second = freeze_training_query_candidates(train, tmp_path / "second", max_queries=2)
    assert first["query_sha256"] == second["query_sha256"]
    assert first["validation_inputs_read"] == []
    rendered = (tmp_path / "first" / "candidate-bank-receipt.json").read_text()
    assert "source-a" not in rendered
    assert "CC(=O)O" not in rendered


def test_inner_selection_split_is_hash_fixed_and_validation_free(tmp_path: Path) -> None:
    active = tmp_path / "active_T.smi"
    inactive = tmp_path / "inactive_T.smi"
    active.write_text(
        "".join(f"CCO active-{index}\n" for index in range(30)), encoding="utf-8"
    )
    inactive.write_text(
        "".join(f"CCC inactive-{index}\n" for index in range(30)), encoding="utf-8"
    )
    first = prepare_inner_selection_split(active, inactive, tmp_path / "first", inactive_limit=10)
    second = prepare_inner_selection_split(
        active, inactive, tmp_path / "second", inactive_limit=10
    )
    assert first["outputs_sha256"] == second["outputs_sha256"]
    assert first["counts"]["inactive_selection"] == 10
    assert first["validation_inputs_read"] == []


def test_query_grid_selection_uses_complete_denominator(tmp_path: Path, monkeypatch) -> None:
    index = tmp_path / "index.sqlite"
    index.write_bytes(b"index")
    labels = tmp_path / "labels.json"
    labels.write_text(
        '{"records":[{"record_id":"active-1","label":"active"},'
        '{"record_id":"inactive-1","label":"inactive"}],'
        '"failures":[{"record_id":"active-2","label":"active"}]}',
        encoding="utf-8",
    )
    candidates = tmp_path / "queries"
    candidates.mkdir()
    (candidates / "candidate-bank-receipt.json").write_text("{}", encoding="utf-8")
    for index_number in range(2):
        (candidates / f"query-{index_number:02d}.json").write_text(
            "{}", encoding="utf-8"
        )

    monkeypatch.setattr(
        "protbind_agent.query_ensemble.read_query", lambda path: path.name
    )

    def fake_query(_index, query, **_kwargs):
        if query == "query-00.json":
            return [SimpleNamespace(molecule_id="active-1", geometric_match_score=1.0)]
        return [SimpleNamespace(molecule_id="inactive-1", geometric_match_score=1.0)]

    monkeypatch.setattr("protbind_agent.query_ensemble.query_index", fake_query)
    result = select_training_query_ensemble(
        index_path=index,
        labels_path=labels,
        candidate_dir=candidates,
        output=tmp_path / "selection.json",
        ensemble_sizes=(1, 2),
        tolerances=(1.0,),
    )
    assert result["chosen"]["ensemble_size"] == 1
    assert result["chosen"]["metrics"]["library_size"] == 3
    assert result["validation_inputs_read"] == []
