from __future__ import annotations

from collections import Counter

import pytest

from protbind_agent.chemistry import smiles_pharmacophore
from protbind_agent.structure import pocket_pharmacophore
from protbind_agent.tripharm import (
    FeatureType,
    enumerate_triangles,
    select_query_triangles,
)

pytest.importorskip("gemmi")
pytest.importorskip("rdkit")

_ONE_IEP_LIKE_SMILES = (
    "Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1"
    "Nc1nccc(-c2cccnc2)n1"
)


def _atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    xyz: tuple[float, float, float],
    *,
    element: str,
) -> str:
    return (
        f"{'ATOM':<6}{serial:5d} {atom_name:>4} {residue_name:>3} "
        f"{'A':1}{residue_number:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}          {element:>2}  "
    )


def _synthetic_charge_rich_pocket() -> bytes:
    """Return a synthetic protein pocket; no native/reference pose is consulted."""

    lines: list[str] = []
    serial = 1

    def add(
        atom_name: str,
        residue_name: str,
        residue_number: int,
        xyz: tuple[float, float, float],
        element: str,
    ) -> None:
        nonlocal serial
        lines.append(
            _atom_line(
                serial,
                atom_name,
                residue_name,
                residue_number,
                xyz,
                element=element,
            )
        )
        serial += 1

    # Deliberately make charged complementary points the numerical majority.
    for offset in range(8):
        add("OD1", "ASP", 1 + offset, (-14.0 + 4.0 * offset, -8.0, 4.0), "O")
        add("NH1", "ARG", 21 + offset, (-14.0 + 4.0 * offset, 8.0, -4.0), "N")

    # Two representatives of each non-charge family keep the expected
    # complementary chemistry explicit.
    for offset, x_coordinate in enumerate((-5.0, 5.0)):
        add("OD1", "ASN", 41 + offset, (x_coordinate, -4.0, -3.0), "O")
        add("ND2", "ASN", 51 + offset, (x_coordinate, 4.0, 3.0), "N")
        add("CB", "LEU", 61 + offset, (x_coordinate, -2.0, 7.0), "C")

    ring = (
        ("CG", (1.4, 0.0, 0.0)),
        ("CD1", (0.7, 1.2, 0.0)),
        ("CD2", (0.7, -1.2, 0.0)),
        ("CE1", (-0.7, 1.2, 0.0)),
        ("CE2", (-0.7, -1.2, 0.0)),
        ("CZ", (-1.4, 0.0, 0.0)),
    )
    for residue_number, x_offset in ((71, -6.0), (72, 6.0)):
        for atom_name, (x, y, z) in ring:
            add(atom_name, "PHE", residue_number, (x + x_offset, y, z - 7.0), "C")
    return ("\n".join([*lines, "TER", "END"]) + "\n").encode()


def test_charge_rich_pocket_selection_preserves_type_diversity_and_matchable_triangles(
    tmp_path,
):
    receptor = tmp_path / "synthetic-pocket.pdb"
    receptor.write_bytes(_synthetic_charge_rich_pocket())

    first = pocket_pharmacophore(
        receptor,
        center=(0.0, 0.0, 0.0),
        box_size=(40.0, 40.0, 40.0),
        max_points=12,
    )
    second = pocket_pharmacophore(
        receptor,
        center=(0.0, 0.0, 0.0),
        box_size=(40.0, 40.0, 40.0),
        max_points=12,
    )

    assert first == second
    counts = Counter(point.feature_type for point in first)
    assert set(counts) == set(FeatureType)
    assert max(counts.values()) == 2

    # The ligand supplies coordinate-free chemistry and an ETKDG conformer,
    # not a crystallographic/native pose.  At least one of the pocket's
    # deterministic top-64 high-information triangle types must be reachable.
    ligand = smiles_pharmacophore(_ONE_IEP_LIKE_SMILES, seed=20260721)
    pocket_type_keys = {
        triangle.type_key
        for triangle in select_query_triangles(first, max_triangles=64)
    }
    ligand_type_keys = {
        triangle.type_key for triangle in enumerate_triangles(ligand)
    }
    assert pocket_type_keys & ligand_type_keys
