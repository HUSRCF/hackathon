from __future__ import annotations

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.preparation import (
    ReceptorPreparationUnsupportedError,
    ResidueSelector,
    conservative_heavy_atom_repair,
    extract_redocking_receptor,
    restrained_sidechain_geometry_optimize,
)

pytest.importorskip("gemmi")


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
    altloc: str = "",
    occupancy: float = 1.0,
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:>4}{altloc:1}{residue_name:>3} "
        f"{chain:1}{residue_number:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{occupancy:6.2f}{20.0:6.2f}          {element:>2}  "
    )


def _complex_pdb(
    *,
    equal_pocket_altloc: bool = False,
    include_cb: bool = True,
    extra_component: tuple[str, str] | None = None,
    covalent: bool = False,
) -> bytes:
    lines = [
        _atom_line("ATOM", 1, "N", "ALA", "A", 1, (2.0, 0.0, 0.0), element="N"),
        _atom_line("ATOM", 2, "CA", "ALA", "A", 1, (2.5, 0.5, 0.0), element="C"),
        _atom_line("ATOM", 3, "C", "ALA", "A", 1, (3.0, 0.0, 0.0), element="C"),
        _atom_line("ATOM", 4, "O", "ALA", "A", 1, (3.5, 0.0, 0.0), element="O"),
    ]
    if include_cb:
        lines.extend(
            [
                _atom_line(
                    "ATOM",
                    5,
                    "CB",
                    "ALA",
                    "A",
                    1,
                    (2.5, 1.5, 0.0),
                    element="C",
                    altloc="A",
                    occupancy=0.5 if equal_pocket_altloc else 0.6,
                ),
                _atom_line(
                    "ATOM",
                    6,
                    "CB",
                    "ALA",
                    "A",
                    1,
                    (2.5, -1.5, 0.0),
                    element="C",
                    altloc="B",
                    occupancy=0.5 if equal_pocket_altloc else 0.4,
                ),
            ]
        )
    ligand_serial = 7
    lines.extend(
        [
            _atom_line(
                "HETATM", ligand_serial, "C1", "LIG", "Z", 100, (0.0, 0.0, 0.0), element="C"
            ),
            _atom_line(
                "HETATM", ligand_serial + 1, "O1", "LIG", "Z", 100, (0.8, 0.0, 0.0), element="O"
            ),
            _atom_line(
                "HETATM", ligand_serial + 2, "O", "HOH", "W", 1, (1.0, 1.0, 0.0), element="O"
            ),
        ]
    )
    if extra_component is not None:
        residue_name, element = extra_component
        lines.append(
            _atom_line(
                "HETATM",
                ligand_serial + 3,
                element,
                residue_name,
                "X",
                9,
                (1.0, 2.0, 0.0),
                element=element,
            )
        )
    if covalent:
        lines.append(f"CONECT{1:5d}{ligand_serial:5d}")
    lines.extend(["TER", "END"])
    return ("\n".join(lines) + "\n").encode()


def _extract(store: ArtifactStore, data: bytes):
    source = store.put_bytes(
        data,
        media_type="chemical/x-pdb",
        producer="test.holo",
        license="CC0-1.0",
    )
    return extract_redocking_receptor(
        store,
        source,
        native_ligand=ResidueSelector("Z", "LIG", 100),
        site_center=(0.0, 0.0, 0.0),
        pocket_radius=6.0,
    )


def test_extract_receptor_resolves_altloc_and_records_only_allowed_removals(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    result = _extract(store, _complex_pdb())

    prepared = store.read_bytes(result.structure).decode()
    receipt = store.read_json(result.receipt)
    assert "LIG" not in prepared
    assert "HOH" not in prepared
    assert prepared.count(" CB ") == 1
    assert result.alternate_location_atoms == 2
    assert result.alternate_conformers_removed == 1
    assert {item["category"] for item in receipt["removed_components"]} == {
        "native_ligand",
        "water",
    }
    assert receipt["possible_cofactors_silently_removed"] is False
    assert receipt["pocket_missing_heavy_atoms"] == []


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (_complex_pdb(equal_pocket_altloc=True), "AMBIGUOUS_POCKET_ALTLOC"),
        (_complex_pdb(include_cb=False), "MISSING_POCKET_HEAVY_ATOMS"),
        (_complex_pdb(extra_component=("FAD", "C")), "POSSIBLE_REQUIRED_COFACTOR"),
        (_complex_pdb(extra_component=("ZN", "ZN")), "METAL_SYSTEM"),
        (_complex_pdb(covalent=True), "COVALENT_LIGAND"),
    ],
)
def test_extract_receptor_fails_closed_for_unsupported_pocket_chemistry(
    tmp_path, data, code
):
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ReceptorPreparationUnsupportedError) as error:
        _extract(store, data)
    assert error.value.code == code


def _dry_receptor_with_missing_sidechain(*, near_pocket: bool) -> bytes:
    residue_number = 1 if near_pocket else 2
    offset = 2.0 if near_pocket else 20.0
    residue_name = "ALA" if near_pocket else "LYS"
    lines = [
        _atom_line(
            "ATOM", 1, "N", residue_name, "A", residue_number,
            (offset, 0.0, 0.0), element="N"
        ),
        _atom_line(
            "ATOM", 2, "CA", residue_name, "A", residue_number,
            (offset + 1.2, 0.0, 0.0), element="C"
        ),
        _atom_line(
            "ATOM", 3, "C", residue_name, "A", residue_number,
            (offset + 2.2, 0.8, 0.0), element="C"
        ),
        _atom_line(
            "ATOM", 4, "O", residue_name, "A", residue_number,
            (offset + 3.3, 0.4, 0.0), element="O"
        ),
        "TER",
        "END",
    ]
    return ("\n".join(lines) + "\n").encode()


def _two_residue_receptor_with_missing_glu_sidechain() -> bytes:
    atoms = (
        ("N", "GLU", 4, (18.752, -46.623, -32.975), "N"),
        ("CA", "GLU", 4, (19.695, -45.953, -32.074), "C"),
        ("C", "GLU", 4, (19.240, -44.496, -31.971), "C"),
        ("O", "GLU", 4, (19.675, -43.702, -32.781), "O"),
        ("CB", "GLU", 4, (21.137, -46.096, -32.593), "C"),
        ("N", "LEU", 5, (18.344, -44.185, -31.036), "N"),
        ("CA", "LEU", 5, (17.659, -42.862, -31.000), "C"),
        ("C", "LEU", 5, (18.452, -41.857, -30.155), "C"),
        ("O", "LEU", 5, (19.036, -42.264, -29.160), "O"),
        ("CB", "LEU", 5, (16.231, -43.047, -30.467), "C"),
        ("CG", "LEU", 5, (15.323, -43.936, -31.329), "C"),
        ("CD1", "LEU", 5, (13.950, -44.099, -30.693), "C"),
        ("CD2", "LEU", 5, (15.191, -43.399, -32.745), "C"),
    )
    lines = [
        _atom_line(
            "ATOM",
            serial,
            atom_name,
            residue_name,
            "A",
            residue_number,
            xyz,
            element=element,
        )
        for serial, (atom_name, residue_name, residue_number, xyz, element) in enumerate(
            atoms, start=1
        )
    ]
    return ("\n".join((*lines, "TER", "END")) + "\n").encode()


def test_heavy_atom_only_repair_is_receipted_and_never_repairs_pocket(tmp_path):
    pytest.importorskip("pdbfixer")
    pytest.importorskip("openmm")
    store = ArtifactStore(tmp_path / "artifacts")
    outside = store.put_bytes(
        _dry_receptor_with_missing_sidechain(near_pocket=False),
        media_type="chemical/x-pdb",
        producer="test.dry-receptor",
        license="CC0-1.0",
    )
    repaired = conservative_heavy_atom_repair(
        store,
        outside,
        protected_points=((0.0, 0.0, 0.0),),
        protected_radius=6.0,
    )
    receipt = store.read_json(repaired.receipt)
    output = store.read_bytes(repaired.structure).decode()
    assert repaired.added_heavy_atom_count >= 5
    assert receipt["repair_required"] is True
    assert receipt["missing_residues_rebuilt"] is False
    assert receipt["hydrogens_added"] is False
    assert receipt["original_heavy_atom_identity_preserved"] is True
    assert receipt["original_heavy_atom_max_coordinate_delta_angstrom"] <= 0.002
    assert not any(line[76:78].strip() == "H" for line in output.splitlines())

    pocket = store.put_bytes(
        _dry_receptor_with_missing_sidechain(near_pocket=True),
        media_type="chemical/x-pdb",
        producer="test.dry-receptor",
        license="CC0-1.0",
    )
    with pytest.raises(ReceptorPreparationUnsupportedError) as error:
        conservative_heavy_atom_repair(
            store,
            pocket,
            protected_points=((0.0, 0.0, 0.0),),
            protected_radius=6.0,
        )
    assert error.value.code == "MISSING_POCKET_HEAVY_ATOMS"


def test_restrained_sidechain_geometry_fixes_original_atoms_and_checks_chirality(
    tmp_path,
):
    pytest.importorskip("pdbfixer")
    pytest.importorskip("openmm")
    store = ArtifactStore(tmp_path / "artifacts")
    original = store.put_bytes(
        _two_residue_receptor_with_missing_glu_sidechain(),
        media_type="chemical/x-pdb",
        producer="test.dry-receptor",
        license="CC0-1.0",
    )
    repaired = conservative_heavy_atom_repair(
        store,
        original,
        protected_points=((0.0, 0.0, 0.0),),
        protected_radius=6.0,
    )
    optimized = restrained_sidechain_geometry_optimize(
        store,
        original,
        repaired.structure,
        iteration_limit=50,
    )
    receipt = store.read_json(optimized.receipt)
    output = store.read_bytes(optimized.structure).decode()

    assert optimized.mobile_added_heavy_atom_count == repaired.added_heavy_atom_count
    assert optimized.original_heavy_atom_max_coordinate_delta_angstrom <= 0.002
    assert optimized.minimum_nonbonded_distance_ratio >= 0.60
    assert receipt["fixed_original_heavy_atom_count"] == 13
    assert receipt["original_heavy_atom_identity_preserved"] is True
    assert receipt["chirality_signs_preserved"] is True
    assert receipt["meeko_rdkit_validation_required_downstream"] is True
    assert receipt["output_hydrogen_count"] == 0
    assert "not binding energy" in receipt["energy_semantics"]
    assert not any(line[76:78].strip() == "H" for line in output.splitlines())
