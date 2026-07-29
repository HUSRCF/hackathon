from __future__ import annotations

import pytest

from protbind_agent.chemistry import enumerate_microstates, molecule_to_index_record


def test_index_standardization_preserves_formal_charge() -> None:
    from rdkit import Chem

    smiles = "C[NH2+]CC(=O)[O-]"
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    record = molecule_to_index_record(
        "charged",
        molecule,
        original_smiles=smiles,
        source="test",
        seed=20260721,
        max_conformers=1,
    )
    indexed = Chem.MolFromSmiles(record.standardized_smiles)

    assert indexed is not None
    assert sorted(
        atom.GetFormalCharge() for atom in indexed.GetAtoms() if atom.GetFormalCharge()
    ) == [-1, 1]


def test_microstates_are_bounded_deterministic_and_parent_preserving() -> None:
    first = enumerate_microstates("CC(=O)O", max_states=4)
    second = enumerate_microstates("CC(=O)O", max_states=4)

    assert first == second
    assert 1 <= len(first) <= 4
    assert [item.microstate_id for item in first] == [
        f"state-{index:02d}" for index in range(1, len(first) + 1)
    ]
    assert all("experimental assignment" in item.uncertainty for item in first)


def test_microstates_reject_unassigned_stereochemistry() -> None:
    with pytest.raises(ValueError, match="stereochemistry"):
        enumerate_microstates("CC(O)Cl")
