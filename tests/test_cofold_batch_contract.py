from __future__ import annotations

import copy
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.cofold_batch import validate_cofold_batch
from protbind_agent.models import (
    ArtifactRef,
    LigandHypothesis,
    ResearchCase,
    ResearchMode,
    TargetSpec,
)


@dataclass(slots=True)
class _BatchFixture:
    store: ArtifactStore
    case: ResearchCase
    screening: ArtifactRef
    library_index: ArtifactRef
    receptor: ArtifactRef
    batch: dict[str, Any]

    def validate(self, value: dict[str, Any] | None = None) -> dict[str, Any]:
        reference = self.store.put_json(
            value if value is not None else self.batch,
            producer="test.cofold-batch",
        )
        return validate_cofold_batch(
            self.store,
            reference,
            case=self.case,
            screening_artifact=self.screening,
            library_index=self.library_index,
            allowed_receptors=(self.receptor,),
            verify_chemistry=False,
        )


def _library_index(
    tmp_path: Path,
    store: ArtifactStore,
    molecule_smiles: dict[str, str],
) -> ArtifactRef:
    path = tmp_path / "library.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE molecules "
            "(molecule_id TEXT PRIMARY KEY, standardized_smiles TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO molecules (molecule_id, standardized_smiles) VALUES (?, ?)",
            molecule_smiles.items(),
        )
        connection.commit()
    finally:
        connection.close()
    return store.import_file(
        path,
        media_type="application/vnd.sqlite3",
        producer="test.tripharm-index",
    )


def _make_fixture(tmp_path: Path) -> _BatchFixture:
    store = ArtifactStore(tmp_path / "workspace")
    molecule_ids = [f"mol-{index:02d}" for index in range(9)]
    molecule_smiles = {
        molecule_id: "C" * (index + 1)
        for index, molecule_id in enumerate(molecule_ids)
    }
    library_index = _library_index(tmp_path, store, molecule_smiles)
    screening = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.screening-result",
            "hits": [
                {"molecule_id": molecule_id, "rank": rank}
                for rank, molecule_id in enumerate(molecule_ids, start=1)
            ],
        },
        producer="test.screening",
    )
    receptor = store.put_bytes(
        b"ATOM      1  CA  ALA A   1       0.0  0.0  0.0\nEND\n",
        media_type="chemical/x-pdb",
        producer="test.receptor",
    )
    case = ResearchCase(
        case_id="cofold-batch-contract",
        target=TargetSpec(name="fixture target", sequences=("ACDEFG",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(smiles="CC"),
    )

    microstates: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    best_states: dict[str, tuple[str, str]] = {}
    for index, molecule_id in enumerate(molecule_ids):
        parent = molecule_smiles[molecule_id]
        states = [("state-a", parent, -9.0 + index)]
        if index == 0:
            states.append(("state-b", "[CH3]", -10.0))
        for microstate_id, smiles, score in states:
            microstates.append(
                {
                    "molecule_id": molecule_id,
                    "microstate_id": microstate_id,
                    "canonical_isomeric_smiles": smiles,
                    "parent_standardized_smiles": parent,
                    "heavy_element_counts": {"C": smiles.count("C")},
                }
            )
            pose = store.put_bytes(
                f"REMARK {molecule_id} {microstate_id}\n".encode(),
                media_type="chemical/x-pdbqt",
                producer="fixture-vina.pose",
            )
            evidence = store.put_json(
                {
                    "schema_version": "1.0",
                    "kind": "protbind.tool-evidence",
                    "tool": "vina",
                    "molecule_id": molecule_id,
                    "microstate_id": microstate_id,
                    "inputs": {
                        "receptor": receptor.to_dict(),
                        "pose": pose.to_dict(),
                    },
                    "metrics": {
                        "score": score,
                        "box_center": [1.0, 2.0, 3.0],
                        "box_size": [20.0, 20.0, 20.0],
                    },
                },
                producer="fixture-vina.evidence",
            )
            evaluated.append(
                {
                    "molecule_id": molecule_id,
                    "microstate_id": microstate_id,
                    "score": score,
                    "score_semantics": (
                        "Vina tool score only; not an experimental binding free energy"
                    ),
                    "pose": pose.to_dict(),
                    "evidence": evidence.to_dict(),
                    "box_center": [1.0, 2.0, 3.0],
                    "box_size": [20.0, 20.0, 20.0],
                }
            )
        best = min(states, key=lambda item: (item[2], item[0]))
        best_states[molecule_id] = (best[0], best[1])

    retained_ids = molecule_ids.copy()
    cofold_candidates = []
    for molecule_id in retained_ids[:8]:
        microstate_id, smiles = best_states[molecule_id]
        cofold_candidates.append(
            {
                "candidate_id": f"cofold-{molecule_id}",
                "molecule_id": molecule_id,
                "microstate_id": microstate_id,
                "canonical_isomeric_smiles": smiles,
                "heavy_element_counts": {"C": smiles.count("C")},
            }
        )
    batch = {
        "schema_version": "1.0",
        "kind": "protbind.cofold-input-batch",
        "screening_artifact": screening.to_dict(),
        "library_index": library_index.to_dict(),
        "receptor": receptor.to_dict(),
        "protein_chains": [{"chain_id": "A", "sequence": "ACDEFG"}],
        "diversity": {
            "method": "Bemis-Murcko",
            "input_molecule_ids": molecule_ids,
            "retained": [
                {
                    "molecule_id": molecule_id,
                    "scaffold_smiles": f"scaffold-{index:02d}",
                }
                for index, molecule_id in enumerate(molecule_ids)
            ],
        },
        "microstates": microstates,
        # Deliberately reverse execution receipt order: selection must use scores.
        "quick_vina": {
            "evaluated": list(reversed(evaluated)),
            "retained_molecule_ids": retained_ids,
        },
        "cofold_candidates": cofold_candidates,
    }
    return _BatchFixture(
        store=store,
        case=case,
        screening=screening,
        library_index=library_index,
        receptor=receptor,
        batch=batch,
    )


def test_cofold_batch_accepts_deterministic_score_ranked_receipt(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    validated = fixture.validate()

    assert validated == fixture.batch
    assert validated["quick_vina"]["retained_molecule_ids"] == [
        f"mol-{index:02d}" for index in range(9)
    ]
    assert validated["cofold_candidates"][0]["microstate_id"] == "state-b"


def test_cofold_batch_rejects_duplicate_bemis_murcko_scaffolds(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    batch = copy.deepcopy(fixture.batch)
    batch["diversity"]["retained"][1]["scaffold_smiles"] = batch["diversity"][
        "retained"
    ][0]["scaffold_smiles"]

    with pytest.raises(ValueError, match="retained scaffolds cannot repeat"):
        fixture.validate(batch)


def test_structure_only_case_rejects_batch_sequence_not_in_bound_receptor(
    tmp_path: Path,
) -> None:
    pytest.importorskip("gemmi")
    fixture = _make_fixture(tmp_path)
    receptor = fixture.store.put_bytes(
        (
            b"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            b"ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C\n"
            b"ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C\n"
            b"END\n"
        ),
        media_type="chemical/x-pdb",
        producer="test.receptor",
    )
    fixture.receptor = receptor
    fixture.case = ResearchCase(
        case_id="structure-only-contract",
        target=TargetSpec(name="structure target", structure=receptor),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(smiles="CC"),
    )
    batch = copy.deepcopy(fixture.batch)
    batch["receptor"] = receptor.to_dict()

    with pytest.raises(ValueError, match="bound receptor structure"):
        fixture.validate(batch)


def test_cofold_batch_rejects_non_deterministic_quick_vina_order(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    batch = copy.deepcopy(fixture.batch)
    batch["quick_vina"]["retained_molecule_ids"][0:2] = ["mol-01", "mol-00"]

    with pytest.raises(ValueError, match="deterministic score-ranked top 16"):
        fixture.validate(batch)


@pytest.mark.parametrize("mutation", ["order", "microstate", "candidate_id"])
def test_cofold_batch_rejects_non_exact_top_eight(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    batch = copy.deepcopy(fixture.batch)
    candidates = batch["cofold_candidates"]
    if mutation == "order":
        candidates[0], candidates[1] = candidates[1], candidates[0]
        expected = "traceable to retained Vina evidence"
    elif mutation == "microstate":
        candidates[0]["microstate_id"] = "state-a"
        candidates[0]["canonical_isomeric_smiles"] = "C"
        expected = "traceable to retained Vina evidence"
    else:
        candidates[1]["candidate_id"] = candidates[0]["candidate_id"]
        expected = "candidate IDs cannot repeat"

    with pytest.raises(ValueError, match=expected):
        fixture.validate(batch)


@pytest.mark.parametrize(
    ("input_name", "media_type"),
    [("receptor", "chemical/x-pdb"), ("pose", "chemical/x-pdbqt")],
)
def test_cofold_batch_rejects_vina_evidence_bound_to_wrong_artifact_hash(
    tmp_path: Path,
    input_name: str,
    media_type: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    batch = copy.deepcopy(fixture.batch)
    evaluation = batch["quick_vina"]["evaluated"][0]
    evidence_ref = ArtifactRef.from_dict(evaluation["evidence"])
    evidence = fixture.store.read_json(evidence_ref)
    wrong_artifact = fixture.store.put_bytes(
        f"wrong {input_name}\n".encode(),
        media_type=media_type,
        producer="test.mismatched-artifact",
    )
    evidence["inputs"][input_name] = wrong_artifact.to_dict()
    evaluation["evidence"] = fixture.store.put_json(
        evidence,
        producer="fixture-vina.evidence",
    ).to_dict()

    with pytest.raises(ValueError, match="wrong receptor or pose"):
        fixture.validate(batch)
