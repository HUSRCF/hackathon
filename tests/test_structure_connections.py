from __future__ import annotations

from protbind_agent.structure import inspect_declared_connections


def _protein_and_ligand_pdb(*records: str) -> str:
    atoms = (
        "ATOM      1  N   CYS A   1       0.000   0.000   0.000  1.00 20.00           N  \n"
        "ATOM      2  CA  CYS A   1       1.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      3  C   CYS A   1       2.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      4  SG  CYS A   1       1.000   1.000   0.000  1.00 20.00           S  \n"
        "HETATM    5  C1  LIG Z   1       1.000   2.000   0.000  1.00 20.00           C  \n"
        "HETATM    6  C2  LIG Z   1       1.000   3.000   0.000  1.00 20.00           C  \n"
    )
    return atoms + "".join(f"{record}\n" for record in records) + "END\n"


def test_pdb_link_is_declared_covalent_connection(tmp_path) -> None:
    path = tmp_path / "linked.pdb"
    path.write_text(
        _protein_and_ligand_pdb(
            "LINK         SG  CYS A   1                 C1  LIG Z   1     1555   1555  1.80"
        ),
        encoding="utf-8",
    )

    inspection = inspect_declared_connections(path)

    assert inspection.covalent_detected is True
    assert inspection.declared_covalent_connections == 1
    assert inspection.status == "covalent_detected"


def test_pdb_protein_ligand_conect_is_covalent_but_ligand_internal_is_not(
    tmp_path,
) -> None:
    cross = tmp_path / "cross.pdb"
    cross.write_text(
        _protein_and_ligand_pdb("CONECT    4    5"), encoding="utf-8"
    )
    internal = tmp_path / "internal.pdb"
    internal.write_text(
        _protein_and_ligand_pdb("CONECT    5    6"), encoding="utf-8"
    )

    cross_inspection = inspect_declared_connections(cross)
    internal_inspection = inspect_declared_connections(internal)

    assert cross_inspection.covalent_detected is True
    assert cross_inspection.protein_ligand_conect_edges == 1
    assert internal_inspection.covalent_detected is False
    assert internal_inspection.protein_ligand_conect_edges == 0
    assert internal_inspection.status == "partial_no_declared_crosslink"


def test_pdb_disulfide_is_not_mislabeled_as_covalent_ligand(tmp_path) -> None:
    path = tmp_path / "disulfide.pdb"
    path.write_text(
        (
            "ATOM      1  SG  CYS A   1       0.000   0.000   0.000  1.00 20.00           S  \n"
            "ATOM      2  SG  CYS B   1       2.000   0.000   0.000  1.00 20.00           S  \n"
            "SSBOND   1 CYS A    1    CYS B    1                          1555   1555  2.03\n"
            "END\n"
        ),
        encoding="utf-8",
    )

    inspection = inspect_declared_connections(path)

    assert inspection.covalent_detected is False
    assert inspection.declared_covalent_connections == 0
    assert inspection.status == "partial_no_declared_crosslink"
