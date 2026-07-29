from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from protbind_agent.artifacts import ArtifactStore, sha256_bytes
from protbind_agent.posebusters_holdout import (
    PoseBustersHoldoutError,
    freeze_posebusters_holdout,
    write_holdout_manifest,
)

pytest.importorskip("gemmi")
Chem = pytest.importorskip("rdkit.Chem")
AllChem = pytest.importorskip("rdkit.Chem.AllChem")


def _atom_line(
    record: str,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain: str,
    residue_number: int,
    xyz: tuple[float, float, float],
    *,
    element: str,
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:>4} {residue_name:>3} "
        f"{chain:1}{residue_number:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}          {element:>2}  "
    )


def _protein(*, missing_cb: bool = False, cofactor: bool = False) -> bytes:
    lines = [
        _atom_line("ATOM", 1, "N", "ALA", "A", 1, (2.0, 0.0, 0.0), element="N"),
        _atom_line("ATOM", 2, "CA", "ALA", "A", 1, (2.5, 0.5, 0.0), element="C"),
        _atom_line("ATOM", 3, "C", "ALA", "A", 1, (3.0, 0.0, 0.0), element="C"),
        _atom_line("ATOM", 4, "O", "ALA", "A", 1, (3.5, 0.0, 0.0), element="O"),
    ]
    if not missing_cb:
        lines.append(
            _atom_line("ATOM", 5, "CB", "ALA", "A", 1, (2.5, 1.5, 0.0), element="C")
        )
    if cofactor:
        lines.append(
            _atom_line("HETATM", 6, "C1", "FAD", "X", 9, (1.0, 1.0, 0.0), element="C")
        )
    lines.extend(("TER", "END"))
    return ("\n".join(lines) + "\n").encode()


def _ligand() -> bytes:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=17) == 0
    handle = io.StringIO()
    writer = Chem.SDWriter(handle)
    writer.write(molecule)
    writer.close()
    return handle.getvalue().encode()


def _fixture(tmp_path: Path, *, unsafe_member: bool = False):
    identifiers = tuple(f"7A{index:02d}_L{index % 10}" for index in range(12))
    candidate_list = tmp_path / "ids.txt"
    candidate_data = ("\n".join(identifiers) + "\n").encode()
    candidate_list.write_bytes(candidate_data)
    archive = tmp_path / "posebusters.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        if unsafe_member:
            output.writestr("../escape", b"bad")
        for index, case_id in enumerate(identifiers):
            base = f"posebusters_benchmark_set/{case_id}/{case_id}"
            output.writestr(
                base + "_protein.pdb",
                _protein(missing_cb=index == 10, cofactor=index == 11),
            )
            output.writestr(base + "_ligand.sdf", _ligand())
    archive_data = archive.read_bytes()
    return {
        "archive": archive,
        "candidate_list": candidate_list,
        "archive_md5": hashlib.md5(archive_data, usedforsecurity=False).hexdigest(),
        "archive_sha256": sha256_bytes(archive_data),
        "candidate_sha256": sha256_bytes(candidate_data),
        "identifiers": identifiers,
    }


def _freeze(inputs, root: Path):
    return freeze_posebusters_holdout(
        inputs["archive"],
        inputs["candidate_list"],
        ArtifactStore(root),
        count=3,
        namespace="fixture-result-blind",
        expected_archive_md5=inputs["archive_md5"],
        expected_archive_sha256=inputs["archive_sha256"],
        expected_candidate_list_sha256=inputs["candidate_sha256"],
        expected_candidate_count=12,
    )


def test_freeze_audits_pool_hash_sorts_and_materializes_only_selected(tmp_path: Path):
    inputs = _fixture(tmp_path)
    first_store = ArtifactStore(tmp_path / "first")
    first = _freeze(inputs, first_store.root)
    second = _freeze(inputs, tmp_path / "second")

    assert first.candidate_count == 12
    assert first.eligible_count == 10
    assert first.manifest.selection_hash == second.manifest.selection_hash
    assert [item.complex_id for item in first.manifest.selected] == [
        item.complex_id for item in second.manifest.selected
    ]
    assert first.exclusion_reason_counts["missing_pocket_heavy_atoms"] == 1
    assert first.exclusion_reason_counts["required_cofactor"] == 1
    assert first.manifest.eligibility_policy["selection_reads_docking_results"] is False
    assert first.manifest.dataset_source.sha256 == inputs["archive_sha256"]
    assert first.manifest.candidate_list.sha256 == inputs["candidate_sha256"]
    for candidate in first.manifest.selected:
        first_store.resolve(candidate.receptor)
        first_store.resolve(candidate.native_ligand)
        first_store.resolve(candidate.source_complex)

    output = tmp_path / "holdout.json"
    write_holdout_manifest(output, first.manifest)
    with pytest.raises(FileExistsError):
        write_holdout_manifest(output, first.manifest)


def test_freeze_rejects_source_tamper_before_selection(tmp_path: Path):
    inputs = _fixture(tmp_path)
    inputs["candidate_list"].write_bytes(inputs["candidate_list"].read_bytes() + b"7ZZZ_BAD\n")
    with pytest.raises(PoseBustersHoldoutError, match="candidate-list SHA-256 mismatch"):
        _freeze(inputs, tmp_path / "store")


def test_freeze_rejects_unsafe_archive_member(tmp_path: Path):
    inputs = _fixture(tmp_path, unsafe_member=True)
    with pytest.raises(PoseBustersHoldoutError, match="unsafe archive member path"):
        _freeze(inputs, tmp_path / "store")
