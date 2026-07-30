from __future__ import annotations

import json
from pathlib import Path

import pytest

from protbind_agent.library import (
    ImportState,
    LibraryManager,
    VerificationState,
    load_library_config,
    save_library_config,
)


def _manager(tmp_path: Path) -> tuple[LibraryManager, Path]:
    config_path = tmp_path / "private-library.json"
    save_library_config(
        config_path,
        protein_root=tmp_path / "proteins",
        ligand_root=tmp_path / "ligands",
    )
    return LibraryManager(load_library_config(config_path)), config_path


def test_library_init_is_private_and_uses_separate_roots(tmp_path: Path) -> None:
    manager, config_path = _manager(tmp_path)

    status = manager.status()

    assert config_path.stat().st_mode & 0o077 == 0
    assert status["absolute_paths_disclosed"] is False
    assert set(status["libraries"]) == {"protein", "ligand"}
    assert status["libraries"]["protein"]["root_id"].startswith("sha256:")
    assert str(tmp_path) not in json.dumps(status)
    for name in ("objects", "incoming", "quarantine", "derived", "receipts"):
        assert (tmp_path / "proteins" / name).is_dir()
        assert (tmp_path / "ligands" / name).is_dir()


def test_fasta_scan_apply_is_hash_bound_deduplicated_and_path_redacted(
    tmp_path: Path,
) -> None:
    manager, _ = _manager(tmp_path)
    source = tmp_path / "batch"
    source.mkdir()
    fasta = source / "target.fasta"
    fasta.write_text(">private-target\nACDEFGHIK\n", encoding="utf-8")
    (source / "notes.txt").write_text("not input", encoding="utf-8")

    plan = manager.scan("protein", source)
    repeated_plan = manager.scan("protein", source)
    result = manager.apply(plan)
    replay = manager.apply(plan)

    assert len(plan["files"]) == 1
    assert repeated_plan["plan_id"] == plan["plan_id"]
    assert repeated_plan["idempotent_replay"] is True
    assert plan["skipped"] == [{"name": "notes.txt", "reason": "unsupported_suffix"}]
    assert result["imported_count"] == 1
    assert result["results"][0]["state"] == ImportState.ACTIVE
    assert result["results"][0]["verification_state"] == VerificationState.UNVERIFIED
    assert "sequences" not in result["results"][0]["qc"]
    assert result["results"][0]["qc"]["sequence_values_disclosed"] is False
    assert replay["idempotent_replay"] is True
    listed = manager.list_entries("protein")
    assert len(listed["entries"]) == 1
    assert str(tmp_path) not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(listed)
    assert fasta.is_file()


def test_import_rechecks_source_and_move_needs_exact_plan_confirmation(
    tmp_path: Path,
) -> None:
    manager, _ = _manager(tmp_path)
    source = tmp_path / "moving.fasta"
    source.write_text(">target\nACDEFG\n", encoding="utf-8")
    plan = manager.scan("protein", source)

    with pytest.raises(PermissionError, match="exact second confirmation"):
        manager.apply(plan, mode="move")

    source.write_text(">target\nACDEFGH\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed before import"):
        manager.apply(plan, mode="move", confirm_move=plan["plan_id"])
    assert source.is_file()


def test_import_rejects_tampered_plan_content(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    source = tmp_path / "target.fasta"
    source.write_text(">target\nACDEFG\n", encoding="utf-8")
    plan = manager.scan("protein", source)
    plan["files"][0]["size_bytes"] += 1

    with pytest.raises(ValueError, match="does not match its plan_id"):
        manager.apply(plan)


def test_move_removes_source_only_after_content_addressed_copy(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    source = tmp_path / "moving.fasta"
    source.write_text(">target\nACDEFG\n", encoding="utf-8")
    plan = manager.scan("protein", source)

    result = manager.apply(plan, mode="move", confirm_move=plan["plan_id"])

    assert not source.exists()
    imported = result["results"][0]
    assert imported["source_removed_after_verified_copy"] is True
    assert imported["raw_artifact"]["sha256"] == plan["files"][0]["sha256"]


def test_invalid_protein_is_quarantined_without_becoming_verified(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    source = tmp_path / "bad.fasta"
    source.write_text(">not-protein\nACDZZZ\n", encoding="utf-8")

    result = manager.apply(manager.scan("protein", source))

    imported = result["results"][0]
    assert imported["state"] == ImportState.QUARANTINED
    assert imported["verification_state"] == VerificationState.UNVERIFIED
    assert imported["qc"]["parse_valid"] is False
    marker = tmp_path / "proteins" / "quarantine" / f"{imported['entry_id']}.json"
    assert marker.is_file()
    assert marker.stat().st_mode & 0o077 == 0


def test_ligand_import_keeps_raw_and_separate_standardized_derivative(
    tmp_path: Path,
) -> None:
    pytest.importorskip("rdkit")
    manager, _ = _manager(tmp_path)
    source = tmp_path / "ethanol.smi"
    source.write_text("CCO ethanol\n", encoding="utf-8")

    imported = manager.apply(manager.scan("ligand", source))["results"][0]

    assert imported["state"] == ImportState.ACTIVE
    assert imported["raw_artifact"]["sha256"] != imported["derived_artifact"]["sha256"]
    assert imported["qc"]["record_count"] == 1
    assert "canonical_isomeric_smiles" not in json.dumps(imported["qc"])


def test_move_preserves_quarantined_source_for_operator_review(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    source = tmp_path / "bad.fasta"
    source.write_text(">not-protein\nZZZ\n", encoding="utf-8")
    plan = manager.scan("protein", source)

    result = manager.apply(plan, mode="move", confirm_move=plan["plan_id"])

    imported = result["results"][0]
    assert imported["state"] == ImportState.QUARANTINED
    assert imported["source_removed_after_verified_copy"] is False
    assert imported["source_preserved_due_to_quarantine"] is True
    assert source.is_file()


@pytest.mark.parametrize(
    ("reference", "expected"),
    (
        ("ACDEFGHIK", VerificationState.EXACT_SEQUENCE),
        ("MACDEFGHIKM", VerificationState.PARTIAL_COORDINATE_MATCH),
        ("YYYYYYYYY", VerificationState.CONFLICT),
    ),
)
def test_uniprot_verification_states_are_sequence_bounded(
    tmp_path: Path,
    reference: str,
    expected: VerificationState,
) -> None:
    manager, _ = _manager(tmp_path)
    source = tmp_path / "target.fasta"
    source.write_text(">target\nACDEFGHIK\n", encoding="utf-8")
    imported = manager.apply(manager.scan("protein", source))["results"][0]

    verification = manager.verify_uniprot_bytes(
        imported["entry_id"],
        "P12345",
        f">sp|P12345|TARGET\n{reference}\n".encode(),
    )

    assert verification["state"] == expected
    assert verification["private_sequence_uploaded"] is False
    assert verification["absolute_paths_disclosed"] is False
    assert verification["receipt_artifact"]["producer"] == (
        "protbind.library.uniprot-verification"
    )


def test_scan_rejects_symlink_sources(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    actual = tmp_path / "actual.fasta"
    actual.write_text(">target\nACDEFG\n", encoding="utf-8")
    linked = tmp_path / "linked.fasta"
    linked.symlink_to(actual)

    with pytest.raises(ValueError, match="symbolic-link"):
        manager.scan("protein", linked)
