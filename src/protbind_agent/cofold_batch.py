"""Strict funnel receipt required before the OpenFold3 cofold stage."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from typing import Any

from .artifacts import ArtifactStore
from .chemistry import (
    bemis_murcko_scaffold_smiles,
    canonical_microstate_parent_identity,
    heavy_element_counts,
)
from .models import ArtifactRef, ResearchCase
from .structure import inspect_structure

_PROTEIN_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+\*?$")


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _unique_strings(values: list[Any], name: str) -> list[str]:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    result = [str(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _box(value: Any, name: str, *, positive: bool = False) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            not isinstance(item, int | float)
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain three finite numbers")
    result = tuple(float(item) for item in value)
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _protein_sequence(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("protein chain sequence must be a string")
    sequence = value.strip().upper()
    if not _PROTEIN_SEQUENCE.fullmatch(sequence):
        raise ValueError("protein chain sequence must use the 20 canonical amino acids")
    return sequence.removesuffix("*")


def _index_molecules(path: Any, molecule_ids: list[str]) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT molecule_id, standardized_smiles FROM molecules"
        )
        all_molecules = {str(row[0]): str(row[1]) for row in rows}
    except sqlite3.Error as exc:
        raise ValueError("library index does not expose molecule identities") from exc
    finally:
        connection.close()
    missing = [molecule_id for molecule_id in molecule_ids if molecule_id not in all_molecules]
    if missing:
        raise ValueError("screened molecules are absent from the bound library index")
    return {molecule_id: all_molecules[molecule_id] for molecule_id in molecule_ids}


def validate_cofold_batch(
    store: ArtifactStore,
    reference: ArtifactRef,
    *,
    case: ResearchCase,
    screening_artifact: ArtifactRef,
    library_index: ArtifactRef,
    allowed_receptors: tuple[ArtifactRef, ...],
    verify_chemistry: bool,
) -> dict[str, Any]:
    """Validate 512→128→microstate/Vina→16→8 selection without inventing results."""

    value = store.read_json(reference)
    if not isinstance(value, dict):
        raise ValueError("cofold input batch must be a JSON object")
    if value.get("schema_version") != "1.0" or value.get("kind") != (
        "protbind.cofold-input-batch"
    ):
        raise ValueError("cofold input must satisfy protbind.cofold-input-batch v1.0")
    bound_screen = ArtifactRef.from_dict(value["screening_artifact"])
    if bound_screen.sha256 != screening_artifact.sha256:
        raise ValueError("cofold batch is not bound to this run's screening artifact")
    store.resolve(bound_screen)
    bound_index = ArtifactRef.from_dict(value["library_index"])
    if bound_index.sha256 != library_index.sha256:
        raise ValueError("cofold batch is not bound to this run's library index")
    index_path = store.resolve(bound_index)

    receptor = ArtifactRef.from_dict(value["receptor"])
    store.resolve(receptor)
    for allowed in allowed_receptors:
        store.resolve(allowed)
    if not allowed_receptors or receptor.sha256 not in {
        allowed.sha256 for allowed in allowed_receptors
    }:
        raise ValueError("cofold batch receptor is not bound to the research target")

    chains = _array(value.get("protein_chains"), "protein_chains")
    if not 1 <= len(chains) <= 2:
        raise ValueError("cofold batch requires one or two protein chains")
    sequences: list[str] = []
    for index, chain in enumerate(chains):
        if not isinstance(chain, dict):
            raise ValueError("protein chain entries must be objects")
        expected_id = chr(ord("A") + index)
        if chain.get("chain_id") != expected_id:
            raise ValueError("protein chain IDs must be deterministic A then B")
        sequences.append(_protein_sequence(chain.get("sequence")))
        template_value = chain.get("template_cif")
        if template_value is not None:
            template = ArtifactRef.from_dict(template_value)
            template_path = store.resolve(template)
            template_chain = chain.get("template_chain_id")
            if template.media_type not in {"chemical/x-mmcif", "chemical/mmcif"}:
                raise ValueError("template_cif must be an mmCIF artifact")
            if (
                not isinstance(template_chain, str)
                or not template_chain.strip()
                or not template_chain.isalnum()
            ):
                raise ValueError("template_cif requires an alphanumeric template_chain_id")
            template_inspection = inspect_structure(template_path)
            if template_chain not in template_inspection.chain_ids:
                raise ValueError("template_chain_id is absent from the direct-CIF template")
    if sum(len(sequence) for sequence in sequences) > 700:
        raise ValueError("cofold batch exceeds the 700-residue v1 limit")
    case_sequences = tuple(_protein_sequence(item) for item in case.target.sequences)
    if case_sequences and tuple(sequences) != case_sequences:
        raise ValueError("cofold batch protein sequences differ from the research case")
    if verify_chemistry or case.target.structure is not None:
        receptor_sequences = inspect_structure(store.resolve(receptor)).sequences
        if tuple(sequences) != receptor_sequences:
            raise ValueError(
                "cofold batch protein sequences differ from the bound receptor structure"
            )

    screen = store.read_json(screening_artifact)
    hits = screen.get("hits") if isinstance(screen, dict) else None
    if not isinstance(hits, list):
        raise ValueError("screening artifact has no hits array")
    screen_ids = [
        str(hit["molecule_id"])
        for hit in hits
        if isinstance(hit, dict) and "molecule_id" in hit
    ]
    if len(screen_ids) != len(hits) or len(set(screen_ids)) != len(screen_ids):
        raise ValueError("screening hits must have unique molecule identities")
    index_smiles = _index_molecules(index_path, screen_ids)

    diversity = value.get("diversity")
    if not isinstance(diversity, dict) or diversity.get("method") != "Bemis-Murcko":
        raise ValueError("cofold batch requires Bemis-Murcko diversity evidence")
    diversity_input = _unique_strings(
        _array(diversity.get("input_molecule_ids"), "diversity.input_molecule_ids"),
        "diversity.input_molecule_ids",
    )
    if diversity_input != screen_ids:
        raise ValueError("diversity input must preserve the complete screening ranking")
    retained_entries = _array(diversity.get("retained"), "diversity.retained")
    if not 1 <= len(retained_entries) <= 128:
        raise ValueError("Bemis-Murcko diversity must retain between 1 and 128 molecules")
    retained_ids: list[str] = []
    retained_scaffolds: list[str] = []
    for entry in retained_entries:
        if not isinstance(entry, dict):
            raise ValueError("diversity retained entries must be objects")
        molecule_id = entry.get("molecule_id")
        scaffold = entry.get("scaffold_smiles")
        if molecule_id not in screen_ids or not isinstance(scaffold, str) or not scaffold:
            raise ValueError("diversity entries require a screened ID and scaffold SMILES")
        retained_ids.append(str(molecule_id))
        retained_scaffolds.append(scaffold)
    if len(set(retained_ids)) != len(retained_ids):
        raise ValueError("diversity retained molecule IDs cannot repeat")
    if len(set(retained_scaffolds)) != len(retained_scaffolds):
        raise ValueError("Bemis-Murcko retained scaffolds cannot repeat")
    screen_positions = {molecule_id: index for index, molecule_id in enumerate(screen_ids)}
    if retained_ids != sorted(retained_ids, key=screen_positions.__getitem__):
        raise ValueError("diversity selection must preserve the screening ranking")
    if verify_chemistry:
        expected_diversity: list[tuple[str, str]] = []
        observed_scaffolds: set[str] = set()
        for molecule_id in screen_ids:
            scaffold = bemis_murcko_scaffold_smiles(index_smiles[molecule_id])
            if scaffold in observed_scaffolds:
                continue
            observed_scaffolds.add(scaffold)
            expected_diversity.append((molecule_id, scaffold))
            if len(expected_diversity) == 128:
                break
        if list(zip(retained_ids, retained_scaffolds, strict=True)) != expected_diversity:
            raise ValueError(
                "diversity receipt does not reproduce deterministic Bemis-Murcko selection"
            )

    microstates = _array(value.get("microstates"), "microstates")
    microstate_keys: set[tuple[str, str]] = set()
    microstate_smiles: dict[tuple[str, str], str] = {}
    microstate_elements: dict[tuple[str, str], dict[str, int]] = {}
    per_molecule: Counter[str] = Counter()
    for entry in microstates:
        if not isinstance(entry, dict):
            raise ValueError("microstate entries must be objects")
        molecule_id = entry.get("molecule_id")
        microstate_id = entry.get("microstate_id")
        smiles = entry.get("canonical_isomeric_smiles")
        parent_smiles = entry.get("parent_standardized_smiles")
        element_counts = entry.get("heavy_element_counts")
        if (
            molecule_id not in retained_ids
            or not isinstance(microstate_id, str)
            or not microstate_id
            or not isinstance(smiles, str)
            or not smiles
            or parent_smiles != index_smiles.get(str(molecule_id))
            or not isinstance(element_counts, dict)
            or not element_counts
            or any(
                not isinstance(element, str)
                or not element
                or element != element.upper()
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
                for element, count in element_counts.items()
            )
        ):
            raise ValueError(
                "microstates require retained IDs, canonical SMILES, and the bound parent identity"
            )
        key = (str(molecule_id), microstate_id)
        if key in microstate_keys:
            raise ValueError("duplicate molecule/microstate pair")
        microstate_keys.add(key)
        microstate_smiles[key] = smiles
        microstate_elements[key] = dict(sorted(element_counts.items()))
        per_molecule[str(molecule_id)] += 1
        if verify_chemistry:
            if canonical_microstate_parent_identity(
                index_smiles[str(molecule_id)]
            ) != canonical_microstate_parent_identity(smiles):
                raise ValueError("microstate does not preserve its indexed molecular parent")
            if microstate_elements[key] != heavy_element_counts(smiles):
                raise ValueError("microstate heavy-element counts do not match its SMILES")
    if (
        not microstates
        or set(per_molecule) != set(retained_ids)
        or any(not 1 <= count <= 4 for count in per_molecule.values())
    ):
        raise ValueError("microstate enumeration must retain 1-4 states for every molecule")

    quick_vina = value.get("quick_vina")
    if not isinstance(quick_vina, dict):
        raise ValueError("cofold batch requires quick_vina evidence")
    evaluated = _array(quick_vina.get("evaluated"), "quick_vina.evaluated")
    evaluated_keys: set[tuple[str, str]] = set()
    vina_counts: Counter[str] = Counter()
    for entry in evaluated:
        if not isinstance(entry, dict):
            raise ValueError("quick Vina entries must be objects")
        key = (str(entry.get("molecule_id")), str(entry.get("microstate_id")))
        score = entry.get("score")
        semantics = entry.get("score_semantics")
        evidence = ArtifactRef.from_dict(entry["evidence"])
        pose = ArtifactRef.from_dict(entry["pose"])
        store.resolve(evidence)
        store.resolve(pose)
        if key not in microstate_keys:
            raise ValueError("quick Vina entry does not reference an enumerated microstate")
        if key in evaluated_keys:
            raise ValueError("a microstate cannot be evaluated by quick Vina twice")
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError("quick Vina score must be finite")
        if not isinstance(semantics, str) or (
            "not an experimental binding free energy" not in semantics.lower()
        ):
            raise ValueError("quick Vina score must carry the scientific disclaimer")
        if "vina" not in evidence.producer.lower():
            raise ValueError("quick Vina evidence producer must identify Vina")
        evidence_value = store.read_json(evidence)
        if (
            not isinstance(evidence_value, dict)
            or evidence_value.get("schema_version") != "1.0"
            or evidence_value.get("kind") != "protbind.tool-evidence"
            or evidence_value.get("tool") != "vina"
            or evidence_value.get("molecule_id") != key[0]
            or evidence_value.get("microstate_id") != key[1]
            or not isinstance(evidence_value.get("metrics"), dict)
            or not isinstance(evidence_value.get("inputs"), dict)
        ):
            raise ValueError("quick Vina evidence is not bound to its molecule/microstate")
        evidence_score = evidence_value["metrics"].get("score")
        if (
            not isinstance(evidence_score, int | float)
            or isinstance(evidence_score, bool)
            or not math.isfinite(float(evidence_score))
            or not math.isclose(
                float(evidence_score), float(score), rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise ValueError("quick Vina evidence score does not match the batch")
        evidence_receptor = ArtifactRef.from_dict(evidence_value["inputs"]["receptor"])
        evidence_pose = ArtifactRef.from_dict(evidence_value["inputs"]["pose"])
        store.resolve(evidence_receptor)
        store.resolve(evidence_pose)
        if evidence_receptor.sha256 != receptor.sha256 or evidence_pose.sha256 != pose.sha256:
            raise ValueError("quick Vina evidence is bound to the wrong receptor or pose")
        center = _box(entry.get("box_center"), "quick_vina.box_center")
        size = _box(entry.get("box_size"), "quick_vina.box_size", positive=True)
        evidence_center = _box(
            evidence_value["metrics"].get("box_center"),
            "quick Vina evidence box_center",
        )
        evidence_size = _box(
            evidence_value["metrics"].get("box_size"),
            "quick Vina evidence box_size",
            positive=True,
        )
        if center != evidence_center or size != evidence_size:
            raise ValueError("quick Vina evidence box does not match the batch")
        evaluated_keys.add(key)
        vina_counts[key[0]] += 1
    if (
        not evaluated
        or set(vina_counts) != set(retained_ids)
        or any(not 1 <= count <= 2 for count in vina_counts.values())
    ):
        raise ValueError("quick Vina must evaluate 1-2 microstates for every molecule")
    vina_retained = _unique_strings(
        _array(
            quick_vina.get("retained_molecule_ids"),
            "quick_vina.retained_molecule_ids",
        ),
        "quick_vina.retained_molecule_ids",
    )
    ordered_evaluations = sorted(
        evaluated,
        key=lambda entry: (
            float(entry["score"]),
            str(entry["molecule_id"]),
            str(entry["microstate_id"]),
        ),
    )
    best_evaluation: dict[str, dict[str, Any]] = {}
    for entry in ordered_evaluations:
        best_evaluation.setdefault(str(entry["molecule_id"]), entry)
    expected_vina_retained = list(best_evaluation)[:16]
    if vina_retained != expected_vina_retained:
        raise ValueError("quick Vina retained IDs must be the deterministic score-ranked top 16")

    cofold = _array(value.get("cofold_candidates"), "cofold_candidates")
    expected_cofold_ids = vina_retained[:8]
    if len(cofold) != len(expected_cofold_ids):
        raise ValueError("OpenFold3 batch must contain the deterministic top 8")
    cofold_ids: set[str] = set()
    candidate_ids: set[str] = set()
    for position, entry in enumerate(cofold):
        if not isinstance(entry, dict):
            raise ValueError("cofold candidate entries must be objects")
        molecule_id = entry.get("molecule_id")
        key = (str(molecule_id), str(entry.get("microstate_id")))
        expected_evaluation = best_evaluation.get(str(molecule_id), {})
        if (
            molecule_id != expected_cofold_ids[position]
            or key not in evaluated_keys
            or entry.get("microstate_id") != expected_evaluation.get("microstate_id")
            or not isinstance(entry.get("candidate_id"), str)
            or not entry["candidate_id"]
            or not isinstance(entry.get("canonical_isomeric_smiles"), str)
            or not entry["canonical_isomeric_smiles"]
            or entry["canonical_isomeric_smiles"] != microstate_smiles.get(key)
            or entry.get("heavy_element_counts") != microstate_elements.get(key)
        ):
            raise ValueError("cofold candidate is not traceable to retained Vina evidence")
        if str(molecule_id) in cofold_ids:
            raise ValueError("OpenFold3 batch permits one candidate per molecule")
        if entry["candidate_id"] in candidate_ids:
            raise ValueError("OpenFold3 candidate IDs cannot repeat")
        cofold_ids.add(str(molecule_id))
        candidate_ids.add(entry["candidate_id"])
    return value
