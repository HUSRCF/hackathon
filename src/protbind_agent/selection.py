"""Deterministic scaffold, microstate, and quick-docking selection receipts."""

from __future__ import annotations

import math
import re
import sqlite3
from typing import Any

from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes
from .chemistry import (
    bemis_murcko_scaffold_smiles,
    enumerate_microstates,
    heavy_element_counts,
    smiles_formal_charge,
)
from .models import ArtifactRef, SiteProvenanceKind
from .redock_calibration import validate_known_site_calibration_artifact
from .structure import inspect_box_atom_overlap

_SCORE_SEMANTICS = (
    "AutoDock Vina tool score only; not an experimental binding free energy"
)
QUICK_VINA_PURPOSE = "selection-pruning-only"
QUICK_VINA_INPUT_KIND = "protbind.quick-vina-input"
QUICK_VINA_BATCH_KIND = "protbind.quick-vina-evaluation-batch"
DOCKING_BOX_RECEIPT_KIND = "protbind.docking-box-receipt"
SITE_DERIVATION_EVIDENCE_KIND = "protbind.site-derivation-evidence"
DOCKING_BOX_COORDINATE_FRAME = "receptor-cartesian-angstrom"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIN_BOX_DIMENSION_ANGSTROM = 4.0
_MAX_BOX_DIMENSION_ANGSTROM = 60.0
_MAX_BOX_VOLUME_ANGSTROM3 = 27_000.0


def _box(value: Any, name: str, *, positive: bool) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != 3 or any(
        not isinstance(item, int | float)
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"{name} must contain three finite numbers")
    result = [float(item) for item in value]
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"{name} must be positive")
    return result


def _plausible_docking_box(
    center: Any, size: Any
) -> tuple[list[float], list[float]]:
    normalized_center = _box(center, "box_center", positive=False)
    normalized_size = _box(size, "box_size", positive=True)
    if any(item < _MIN_BOX_DIMENSION_ANGSTROM for item in normalized_size):
        raise ValueError(
            "docking box is degenerate for ordinary non-covalent ligand docking: "
            f"every dimension must be >= {_MIN_BOX_DIMENSION_ANGSTROM:g} angstrom"
        )
    volume = math.prod(normalized_size)
    if (
        any(item > _MAX_BOX_DIMENSION_ANGSTROM for item in normalized_size)
        or volume > _MAX_BOX_VOLUME_ANGSTROM3
    ):
        raise ValueError(
            "docking box is implausibly broad for quick site docking: dimensions must "
            f"be <= {_MAX_BOX_DIMENSION_ANGSTROM:g} angstrom and volume <= "
            f"{_MAX_BOX_VOLUME_ANGSTROM3:g} cubic angstrom"
        )
    return normalized_center, normalized_size


def _site_provenance_kind(value: str | SiteProvenanceKind) -> SiteProvenanceKind:
    try:
        return value if isinstance(value, SiteProvenanceKind) else SiteProvenanceKind(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in SiteProvenanceKind)
        raise ValueError(f"site provenance kind must be one of: {allowed}") from exc


def validate_site_derivation_evidence(
    store: ArtifactStore,
    evidence: ArtifactRef,
    *,
    receptor: ArtifactRef,
    source_kind: SiteProvenanceKind,
    center: list[float],
    size: list[float],
) -> dict[str, Any]:
    """Validate a coordinate-free receipt for how a site box was derived."""

    if (
        evidence.media_type != "application/json"
        or evidence.producer != "protbind.site-derivation-evidence"
        or evidence.producer_version != "1.1"
        or evidence.source != receptor.artifact_id
    ):
        raise ValueError("site derivation evidence artifact provenance is invalid")
    value = store.read_json(evidence)
    allowed_fields = {
        "schema_version",
        "kind",
        "artifact_scope",
        "source_kind",
        "receptor",
        "receptor_sha256",
        "coordinate_frame",
        "center",
        "size",
        "derivation_method",
        "source_commitments",
        "reference_atom_coordinates_exposed_to_screening",
        "reference_derived_box_exposed_to_screening",
        "biological_site_validity_inferred",
    }
    commitments = value.get("source_commitments") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != allowed_fields
        or value.get("schema_version") != "1.1"
        or value.get("kind") != SITE_DERIVATION_EVIDENCE_KIND
        or value.get("artifact_scope") != "site-derivation-only"
        or value.get("source_kind") != source_kind.value
        or _reference(value.get("receptor"), "site evidence receptor") != receptor
        or value.get("receptor_sha256") != receptor.sha256
        or value.get("coordinate_frame") != DOCKING_BOX_COORDINATE_FRAME
        or value.get("center") != center
        or value.get("size") != size
        or not isinstance(value.get("derivation_method"), str)
        or not value["derivation_method"].strip()
        or not isinstance(commitments, list)
        or not commitments
        or any(not isinstance(item, str) or _SHA256.fullmatch(item) is None for item in commitments)
        or len(set(commitments)) != len(commitments)
        or value.get("reference_atom_coordinates_exposed_to_screening") is not False
        or value.get("reference_derived_box_exposed_to_screening") is not True
        or value.get("biological_site_validity_inferred") is not False
    ):
        raise ValueError(
            "site derivation evidence is not a coordinate-free receipt bound to "
            "this receptor and box"
        )
    for commitment in commitments:
        store.resolve_sha256(commitment)
    return value


def build_site_derivation_evidence(
    store: ArtifactStore,
    *,
    receptor: ArtifactRef,
    source_kind: str | SiteProvenanceKind,
    center: tuple[float, float, float] | list[float],
    size: tuple[float, float, float] | list[float],
    derivation_method: str,
    source_artifacts: tuple[ArtifactRef, ...],
    license: str | None = None,
) -> ArtifactRef:
    """Commit real derivation inputs without exposing their atom coordinates."""

    kind = _site_provenance_kind(source_kind)
    if not kind.requires_derivation_evidence:
        raise ValueError(
            "user-declared sites cannot be promoted by derivation evidence"
        )
    if not isinstance(derivation_method, str) or not derivation_method.strip():
        raise ValueError("site derivation method must be explicit")
    normalized_center, normalized_size = _plausible_docking_box(center, size)
    store.resolve(receptor)
    if not source_artifacts:
        raise ValueError("site derivation requires at least one source artifact")
    commitments: list[str] = []
    for source in source_artifacts:
        store.resolve(source)
        if source.sha256 == receptor.sha256:
            raise ValueError(
                "receptor alone cannot serve as independent site derivation evidence"
            )
        if source.sha256 in commitments:
            raise ValueError("site derivation source artifacts must be unique")
        commitments.append(source.sha256)
    return store.put_json(
        {
            "schema_version": "1.1",
            "kind": SITE_DERIVATION_EVIDENCE_KIND,
            "artifact_scope": "site-derivation-only",
            "source_kind": kind.value,
            "receptor": receptor.to_dict(),
            "receptor_sha256": receptor.sha256,
            "coordinate_frame": DOCKING_BOX_COORDINATE_FRAME,
            "center": normalized_center,
            "size": normalized_size,
            "derivation_method": derivation_method.strip(),
            "source_commitments": commitments,
            "reference_atom_coordinates_exposed_to_screening": False,
            "reference_derived_box_exposed_to_screening": True,
            "biological_site_validity_inferred": False,
        },
        producer="protbind.site-derivation-evidence",
        producer_version="1.1",
        source=receptor.artifact_id,
        license=license,
    )


def known_site_calibration_summary(
    store: ArtifactStore,
    *,
    calibration: ArtifactRef | None,
    target_id: str | None,
    receptor: ArtifactRef,
    source_kind: SiteProvenanceKind,
    center: list[float],
    size: list[float],
) -> dict[str, Any]:
    """Validate an opt-in calibration gate without exposing its reference ligand."""

    if (calibration is None) != (target_id is None):
        raise ValueError(
            "known-site calibration receipt and target_id must be provided together"
        )
    if calibration is None:
        return {
            "claimed": False,
            "receipt": None,
            "target_id": None,
            "decision": None,
            "scientific_interpretation": (
                "not claimed; no calibration authorization is inferred"
            ),
        }
    if not source_kind.requires_derivation_evidence:
        raise ValueError(
            "user-declared site provenance cannot be promoted by calibration"
        )
    assert target_id is not None
    value = validate_known_site_calibration_artifact(
        store,
        calibration,
        expected_target_id=target_id,
        expected_prepared_receptor_sha256=receptor.sha256,
        expected_box_center=center,
        expected_box_size=size,
    )
    authorized = value["authorized_inputs"]
    return {
        "claimed": True,
        "receipt": calibration.to_dict(),
        "target_id": target_id,
        "decision": "PASS",
        "required_rank": value["decision"]["required_rank"],
        "calibration_config_sha256": value["calibration_config_sha256"],
        "source_result_sha256": value["source_redock"]["result_artifact"][
            "sha256"
        ],
        "prepared_receptor_sha256": authorized["prepared_receptor"]["sha256"],
        "receptor_preparation_receipt_sha256": authorized[
            "receptor_preparation_receipt"
        ]["sha256"],
        "known_site_box_receipt_sha256": authorized["known_site_box_receipt"][
            "sha256"
        ],
        "scientific_interpretation": (
            "target-specific known-site pose-recovery calibration only; not evidence "
            "of binding or affinity"
        ),
    }


def validate_docking_box_receipt(
    store: ArtifactStore,
    receipt: ArtifactRef,
    *,
    receptor: ArtifactRef,
    center: Any | None = None,
    size: Any | None = None,
    source_kind: str | None = None,
) -> dict[str, Any]:
    """Validate one box in the exact Cartesian frame of one receptor artifact.

    Atom overlap establishes coordinate-frame plausibility only.  Independent
    derivation evidence is validated separately and never inferred from overlap.
    """

    if (
        receipt.media_type != "application/json"
        or receipt.producer != "protbind.selection-box-receipt"
        or receipt.producer_version != "2.0"
        or receipt.source != receptor.artifact_id
    ):
        raise ValueError("docking box receipt artifact provenance is invalid")
    value = store.read_json(receipt)
    validation = value.get("validation") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "2.0"
        or value.get("kind") != DOCKING_BOX_RECEIPT_KIND
        or value.get("coordinate_frame") != DOCKING_BOX_COORDINATE_FRAME
        or value.get("receptor_sha256") != receptor.sha256
        or _reference(value.get("receptor"), "box receipt receptor") != receptor
        or not isinstance(value.get("source_kind"), str)
        or not value["source_kind"].strip()
        or not isinstance(validation, dict)
        or validation.get("finite_geometry_checked") is not True
        or validation.get("quick_site_bounds_checked") is not True
        or validation.get("receptor_atom_overlap_checked") is not True
        or validation.get("minimum_dimension_angstrom")
        != _MIN_BOX_DIMENSION_ANGSTROM
        or validation.get("maximum_dimension_angstrom")
        != _MAX_BOX_DIMENSION_ANGSTROM
        or validation.get("maximum_volume_angstrom3")
        != _MAX_BOX_VOLUME_ANGSTROM3
    ):
        raise ValueError("docking box receipt schema or receptor binding is invalid")
    receipt_center, receipt_size = _plausible_docking_box(
        value.get("center"), value.get("size")
    )
    if center is not None and receipt_center != _box(
        center, "expected box_center", positive=False
    ):
        raise ValueError("docking box receipt center differs from the frozen box")
    if size is not None and receipt_size != _box(
        size, "expected box_size", positive=True
    ):
        raise ValueError("docking box receipt size differs from the frozen box")
    if source_kind is not None and value["source_kind"] != source_kind:
        raise ValueError("docking box receipt source differs from the frozen box")
    kind = _site_provenance_kind(value["source_kind"])
    receptor_path = store.resolve(receptor)
    overlap = inspect_box_atom_overlap(
        receptor_path,
        center=receipt_center,
        size=receipt_size,
    ).to_dict()
    if validation.get("atom_overlap") != overlap:
        raise ValueError("docking box atom-overlap receipt differs from the receptor")
    if not overlap["protein_atom_overlap"]:
        raise ValueError("docking box contains no protein heavy atoms")
    evidence_value = value.get("site_derivation_evidence")
    evidence = (
        None
        if evidence_value is None
        else _reference(evidence_value, "site derivation evidence")
    )
    if not kind.requires_derivation_evidence and evidence is not None:
        raise ValueError(
            f"{kind.value} cannot carry independent site derivation evidence"
        )
    if kind.requires_derivation_evidence and evidence is None:
        raise ValueError(f"{kind.value} requires site derivation evidence")
    derivation_verified = evidence is not None
    if evidence is not None:
        validate_site_derivation_evidence(
            store,
            evidence,
            receptor=receptor,
            source_kind=kind,
            center=receipt_center,
            size=receipt_size,
        )
    expected_grade = (
        "independently-supported-derivation"
        if derivation_verified
        else "user-hypothesis-only"
    )
    if (
        validation.get("site_derivation_verified") is not derivation_verified
        or validation.get("scientific_interpretation") != expected_grade
        or validation.get("biological_site_validity_inferred") is not False
    ):
        raise ValueError("docking box site-evidence interpretation is invalid")
    return {
        **value,
        "center": receipt_center,
        "size": receipt_size,
    }


def _screened_smiles(
    store: ArtifactStore,
    screening: ArtifactRef,
    library_index: ArtifactRef,
) -> tuple[list[str], dict[str, str]]:
    screen = store.read_json(screening)
    hits = screen.get("hits") if isinstance(screen, dict) else None
    if not isinstance(hits, list) or not hits:
        raise ValueError("screening artifact has no ranked hits")
    ids = [hit.get("molecule_id") for hit in hits if isinstance(hit, dict)]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(
        set(ids)
    ):
        raise ValueError("screening hits require unique molecule IDs")
    path = store.resolve(library_index)
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT molecule_id, standardized_smiles FROM molecules"
        )
        all_smiles = {str(row[0]): str(row[1]) for row in rows}
    finally:
        connection.close()
    if any(item not in all_smiles for item in ids):
        raise ValueError("screening hit is absent from the bound library index")
    return list(ids), {item: all_smiles[item] for item in ids}


def _quick_request_id(
    *,
    screening: ArtifactRef,
    library_index: ArtifactRef,
    receptor: ArtifactRef,
    request: dict[str, Any],
) -> str:
    digest = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "purpose": QUICK_VINA_PURPOSE,
                "screening_sha256": screening.sha256,
                "library_index_sha256": library_index.sha256,
                "receptor_sha256": receptor.sha256,
                "molecule_id": request["molecule_id"],
                "microstate_id": request["microstate_id"],
                "canonical_isomeric_smiles": request[
                    "canonical_isomeric_smiles"
                ],
                "heavy_element_counts": request["heavy_element_counts"],
                "formal_charge": request["formal_charge"],
                "box_center": request["box_center"],
                "box_size": request["box_size"],
                "box_source": request["box_source"],
                "coordinate_frame": request["coordinate_frame"],
                "docking_box_receipt_sha256": request[
                    "docking_box_receipt_sha256"
                ],
            }
        )
    )
    return f"quick-{digest[:24]}"


def build_selection_preparation(
    store: ArtifactStore,
    *,
    screening: ArtifactRef,
    library_index: ArtifactRef,
    receptor: ArtifactRef,
    protein_chains: tuple[tuple[str, str], ...],
    box_center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    diversity_limit: int = 128,
    max_microstates: int = 4,
    quick_states_per_molecule: int = 2,
    box_source: str | SiteProvenanceKind = SiteProvenanceKind.USER_CENTER,
    site_derivation_evidence: ArtifactRef | None = None,
    known_site_calibration_receipt: ArtifactRef | None = None,
    known_site_calibration_target_id: str | None = None,
) -> ArtifactRef:
    if not 1 <= diversity_limit <= 128:
        raise ValueError("diversity_limit must be in [1, 128]")
    if not 1 <= quick_states_per_molecule <= 2:
        raise ValueError("quick_states_per_molecule must be in [1, 2]")
    kind = _site_provenance_kind(box_source)
    receptor_path = store.resolve(receptor)
    screen_ids, smiles = _screened_smiles(store, screening, library_index)
    retained: list[dict[str, str]] = []
    observed_scaffolds: set[str] = set()
    for molecule_id in screen_ids:
        scaffold = bemis_murcko_scaffold_smiles(smiles[molecule_id])
        if scaffold in observed_scaffolds:
            continue
        observed_scaffolds.add(scaffold)
        retained.append({"molecule_id": molecule_id, "scaffold_smiles": scaffold})
        if len(retained) == diversity_limit:
            break
    microstates: list[dict[str, Any]] = []
    eliminated: list[dict[str, str]] = []
    quick_requests: list[dict[str, Any]] = []
    center, size = _plausible_docking_box(box_center, box_size)
    source_kind = kind.value
    overlap = inspect_box_atom_overlap(
        receptor_path,
        center=center,
        size=size,
    ).to_dict()
    if overlap["receptor_heavy_atom_count"] < 1:
        raise ValueError("receptor contains no heavy atoms")
    if overlap["protein_heavy_atom_count"] < 1:
        raise ValueError("receptor contains no standard-protein heavy atoms")
    if not overlap["protein_atom_overlap"]:
        raise ValueError("docking box contains no protein heavy atoms")
    if kind.requires_derivation_evidence and site_derivation_evidence is None:
        raise ValueError(f"{kind.value} requires site derivation evidence")
    if not kind.requires_derivation_evidence and site_derivation_evidence is not None:
        raise ValueError(
            f"{kind.value} cannot carry independent site derivation evidence"
        )
    derivation_verified = site_derivation_evidence is not None
    if site_derivation_evidence is not None:
        validate_site_derivation_evidence(
            store,
            site_derivation_evidence,
            receptor=receptor,
            source_kind=kind,
            center=center,
            size=size,
        )
    calibration_summary = known_site_calibration_summary(
        store,
        calibration=known_site_calibration_receipt,
        target_id=known_site_calibration_target_id,
        receptor=receptor,
        source_kind=kind,
        center=center,
        size=size,
    )
    scientific_interpretation = (
        "independently-supported-derivation"
        if derivation_verified
        else "user-hypothesis-only"
    )
    box_receipt = store.put_json(
        {
            "schema_version": "2.0",
            "kind": DOCKING_BOX_RECEIPT_KIND,
            "source_kind": source_kind,
            "site_derivation_evidence": (
                site_derivation_evidence.to_dict()
                if site_derivation_evidence is not None
                else None
            ),
            "receptor": receptor.to_dict(),
            "receptor_sha256": receptor.sha256,
            "coordinate_frame": DOCKING_BOX_COORDINATE_FRAME,
            "center": center,
            "size": size,
            "validation": {
                "finite_geometry_checked": True,
                "quick_site_bounds_checked": True,
                "minimum_dimension_angstrom": _MIN_BOX_DIMENSION_ANGSTROM,
                "maximum_dimension_angstrom": _MAX_BOX_DIMENSION_ANGSTROM,
                "maximum_volume_angstrom3": _MAX_BOX_VOLUME_ANGSTROM3,
                "receptor_atom_overlap_checked": True,
                "atom_overlap": overlap,
                "site_derivation_verified": derivation_verified,
                "scientific_interpretation": scientific_interpretation,
                "biological_site_validity_inferred": False,
            },
        },
        producer="protbind.selection-box-receipt",
        producer_version="2.0",
        source=receptor.artifact_id,
    )
    for item in retained:
        molecule_id = item["molecule_id"]
        try:
            states = enumerate_microstates(
                smiles[molecule_id], max_states=max_microstates
            )
        except ValueError as exc:
            eliminated.append(
                {
                    "molecule_id": molecule_id,
                    "stage": "microstate_enumeration",
                    "reason": str(exc),
                }
            )
            continue
        for rank, state in enumerate(states):
            entry = {
                "molecule_id": molecule_id,
                "microstate_id": state.microstate_id,
                "canonical_isomeric_smiles": state.canonical_isomeric_smiles,
                "parent_standardized_smiles": smiles[molecule_id],
                "formal_charge": state.formal_charge,
                "heavy_element_counts": heavy_element_counts(
                    state.canonical_isomeric_smiles
                ),
                "enumeration_method": state.enumeration_method,
                "uncertainty": state.uncertainty,
            }
            microstates.append(entry)
            if rank < quick_states_per_molecule:
                request = {
                    **entry,
                    "purpose": QUICK_VINA_PURPOSE,
                    "box_center": center,
                    "box_size": size,
                    "box_source": source_kind,
                    "coordinate_frame": DOCKING_BOX_COORDINATE_FRAME,
                    "docking_box_receipt_sha256": box_receipt.sha256,
                }
                request["request_id"] = _quick_request_id(
                    screening=screening,
                    library_index=library_index,
                    receptor=receptor,
                    request=request,
                )
                quick_requests.append(request)
    if not quick_requests:
        raise ValueError("no chemically supported microstate can enter quick docking")
    return store.put_json(
        {
            "schema_version": "2.0",
            "kind": "protbind.selection-preparation",
            "screening_artifact": screening.to_dict(),
            "library_index": library_index.to_dict(),
            "receptor": receptor.to_dict(),
            "docking_box_receipt": box_receipt.to_dict(),
            "site_derivation_evidence": (
                site_derivation_evidence.to_dict()
                if site_derivation_evidence is not None
                else None
            ),
            "known_site_calibration_receipt": (
                known_site_calibration_receipt.to_dict()
                if known_site_calibration_receipt is not None
                else None
            ),
            "known_site_calibration": calibration_summary,
            "site_evidence": {
                "source_kind": source_kind,
                "receptor_atom_overlap_checked": True,
                "protein_heavy_atom_count_inside_box": overlap[
                    "protein_heavy_atom_count_inside_box"
                ],
                "site_derivation_verified": derivation_verified,
                "scientific_interpretation": scientific_interpretation,
                "biological_site_validity_inferred": False,
            },
            "protein_chains": [
                {"chain_id": chain_id, "sequence": sequence}
                for chain_id, sequence in protein_chains
            ],
            "diversity": {
                "method": "Bemis-Murcko",
                "input_molecule_ids": screen_ids,
                "retained": retained,
            },
            "microstates": microstates,
            "quick_vina_requests": quick_requests,
            "quick_vina": {
                "purpose": QUICK_VINA_PURPOSE,
                "score_semantics": _SCORE_SEMANTICS,
                "box_source": source_kind,
                "coordinate_frame": DOCKING_BOX_COORDINATE_FRAME,
                "docking_box_receipt": box_receipt.to_dict(),
                "site_derivation_verified": derivation_verified,
                "scientific_interpretation": scientific_interpretation,
            },
            "eliminated": eliminated,
        },
        producer="protbind.selection-preparation",
        producer_version="2.5",
        source=screening.artifact_id,
    )


def _validated_requests(
    store: ArtifactStore, value: dict[str, Any]
) -> list[dict[str, Any]]:
    requests = value.get("quick_vina_requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("selection preparation has no quick Vina requests")
    if len(requests) > 256:
        raise ValueError("selection preparation exceeds 256 quick Vina requests")
    request_ids: set[str] = set()
    identities: set[tuple[str, str]] = set()
    screening = _reference(value.get("screening_artifact"), "selection screening")
    library = _reference(value.get("library_index"), "selection library index")
    receptor = _reference(value.get("receptor"), "selection receptor")
    box_receipt = _reference(
        value.get("docking_box_receipt"), "selection docking box receipt"
    )
    receipt_value = validate_docking_box_receipt(
        store,
        box_receipt,
        receptor=receptor,
    )
    calibration_value = value.get("known_site_calibration")
    calibration_ref_value = value.get("known_site_calibration_receipt")
    calibration_ref = (
        None
        if calibration_ref_value is None
        else _reference(calibration_ref_value, "known-site calibration receipt")
    )
    calibration_target = (
        calibration_value.get("target_id")
        if isinstance(calibration_value, dict)
        else None
    )
    expected_calibration = known_site_calibration_summary(
        store,
        calibration=calibration_ref,
        target_id=calibration_target,
        receptor=receptor,
        source_kind=_site_provenance_kind(receipt_value["source_kind"]),
        center=receipt_value["center"],
        size=receipt_value["size"],
    )
    if calibration_value != expected_calibration:
        raise ValueError(
            "selection known-site calibration summary differs from its source receipt"
        )
    receipt_site_evidence = receipt_value["site_derivation_evidence"]
    if value.get("site_derivation_evidence") != receipt_site_evidence:
        raise ValueError("selection site evidence differs from its docking box receipt")
    site_summary = value.get("site_evidence")
    overlap = receipt_value["validation"]["atom_overlap"]
    if site_summary != {
        "source_kind": receipt_value["source_kind"],
        "receptor_atom_overlap_checked": True,
        "protein_heavy_atom_count_inside_box": overlap[
            "protein_heavy_atom_count_inside_box"
        ],
        "site_derivation_verified": receipt_value["validation"][
            "site_derivation_verified"
        ],
        "scientific_interpretation": receipt_value["validation"][
            "scientific_interpretation"
        ],
        "biological_site_validity_inferred": False,
    }:
        raise ValueError("selection site-evidence summary differs from its receipt")
    quick_summary = value.get("quick_vina")
    if (
        not isinstance(quick_summary, dict)
        or quick_summary.get("box_source") != receipt_value["source_kind"]
        or quick_summary.get("coordinate_frame")
        != receipt_value["coordinate_frame"]
        or _reference(
            quick_summary.get("docking_box_receipt"),
            "quick Vina summary docking box receipt",
        )
        != box_receipt
        or quick_summary.get("site_derivation_verified")
        is not receipt_value["validation"]["site_derivation_verified"]
        or quick_summary.get("scientific_interpretation")
        != receipt_value["validation"]["scientific_interpretation"]
    ):
        raise ValueError(
            "selection quick Vina summary differs from its docking box receipt"
        )
    validated: list[dict[str, Any]] = []
    for raw in requests:
        if not isinstance(raw, dict):
            raise ValueError("quick Vina request must be an object")
        request = dict(raw)
        request_id = request.get("request_id")
        molecule_id = request.get("molecule_id")
        microstate_id = request.get("microstate_id")
        smiles = request.get("canonical_isomeric_smiles")
        counts = request.get("heavy_element_counts")
        if (
            any(
                not isinstance(item, str) or _SAFE_ID.fullmatch(item) is None
                for item in (request_id, molecule_id, microstate_id)
            )
            or not isinstance(smiles, str)
            or not smiles.strip()
        ):
            raise ValueError(
                "quick Vina request requires request/molecule/microstate IDs and SMILES"
            )
        if request.get("purpose") != QUICK_VINA_PURPOSE:
            raise ValueError("quick Vina request has the wrong scientific purpose")
        if (
            not isinstance(counts, dict)
            or not counts
            or any(
                not isinstance(element, str)
                or element != element.upper()
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
                for element, count in counts.items()
            )
        ):
            raise ValueError("quick Vina request has invalid heavy-element identity")
        try:
            observed_counts = heavy_element_counts(str(smiles))
        except ValueError as exc:
            raise ValueError("quick Vina request has invalid ligand chemistry") from exc
        if observed_counts != dict(sorted(counts.items())):
            raise ValueError("quick Vina request heavy elements differ from its SMILES")
        formal_charge = request.get("formal_charge")
        if not isinstance(formal_charge, int) or isinstance(formal_charge, bool):
            raise ValueError("quick Vina request formal_charge must be an integer")
        if formal_charge != smiles_formal_charge(str(smiles)):
            raise ValueError("quick Vina request formal charge differs from its SMILES")
        box_source = request.get("box_source")
        if not isinstance(box_source, str) or not box_source.strip():
            raise ValueError("quick Vina request requires a non-empty box_source")
        center, size = _plausible_docking_box(
            request.get("box_center"), request.get("box_size")
        )
        if (
            center != receipt_value["center"]
            or size != receipt_value["size"]
            or box_source != receipt_value["source_kind"]
            or request.get("coordinate_frame")
            != receipt_value["coordinate_frame"]
            or request.get("docking_box_receipt_sha256") != box_receipt.sha256
        ):
            raise ValueError(
                "quick Vina request docking box differs from its provenance receipt"
            )
        identity = (str(molecule_id), str(microstate_id))
        if request_id in request_ids or identity in identities:
            raise ValueError("quick Vina planned requests must have unique identities")
        request_ids.add(str(request_id))
        identities.add(identity)
        expected_request_id = _quick_request_id(
            screening=screening,
            library_index=library,
            receptor=receptor,
            request=request,
        )
        if request_id != expected_request_id:
            raise ValueError("quick Vina request_id differs from its frozen inputs")
        validated.append(request)
    return validated


def build_quick_vina_input(
    store: ArtifactStore,
    preparation: ArtifactRef,
    environment_lock: ArtifactRef,
    *,
    case_id: str,
) -> ArtifactRef:
    """Project selection preparation into the minimum worker-visible payload."""

    value = store.read_json(preparation)
    if not isinstance(value, dict) or value.get("kind") != (
        "protbind.selection-preparation"
    ):
        raise ValueError("preparation is not a selection-preparation artifact")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a non-empty string")
    receptor = _reference(value.get("receptor"), "selection receptor")
    store.resolve(receptor)
    store.resolve(environment_lock)
    screening = _reference(value.get("screening_artifact"), "selection screening")
    library = _reference(value.get("library_index"), "selection library index")
    box_receipt = _reference(
        value.get("docking_box_receipt"), "selection docking box receipt"
    )
    receipt_value = validate_docking_box_receipt(
        store, box_receipt, receptor=receptor
    )
    site_evidence = (
        None
        if receipt_value["site_derivation_evidence"] is None
        else _reference(
            receipt_value["site_derivation_evidence"], "site derivation evidence"
        )
    )
    requests = _validated_requests(store, value)
    projected = [
        {
            name: request[name]
            for name in (
                "request_id",
                "molecule_id",
                "microstate_id",
                "canonical_isomeric_smiles",
                "heavy_element_counts",
                "formal_charge",
                "box_center",
                "box_size",
                "box_source",
                "coordinate_frame",
                "docking_box_receipt_sha256",
                "purpose",
            )
        }
        for request in requests
    ]
    return store.put_json(
        {
            "schema_version": "1.0",
            "kind": QUICK_VINA_INPUT_KIND,
            "purpose": QUICK_VINA_PURPOSE,
            "case_id": case_id,
            # Scalar commitments avoid staging the screening JSON and 100k index.
            "selection_preparation_sha256": preparation.sha256,
            "screening_sha256": screening.sha256,
            "library_index_sha256": library.sha256,
            "receptor": receptor.to_dict(),
            "docking_box_receipt": box_receipt.to_dict(),
            "site_derivation_evidence": (
                site_evidence.to_dict() if site_evidence is not None else None
            ),
            "environment_lock": environment_lock.to_dict(),
            "request_count": len(projected),
            "requests": projected,
        },
        producer="protbind.quick-vina-input",
        producer_version="1.2",
        source=preparation.artifact_id,
    )


def _nested_references(value: Any) -> list[ArtifactRef]:
    references: list[ArtifactRef] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            required = {"sha256", "media_type", "size_bytes", "producer"}
            allowed = required | {"producer_version", "source", "license"}
            if required.issubset(item) and set(item).issubset(allowed):
                references.append(_reference(item, "nested artifact"))
                return
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return references


def _validate_output_closure(
    store: ArtifactStore,
    outputs: tuple[ArtifactRef, ...],
    *,
    allowed_inputs: set[ArtifactRef],
) -> None:
    if not outputs:
        raise ValueError("quick Vina worker returned no artifacts")
    if len(set(outputs)) != len(outputs):
        raise ValueError("quick Vina WorkerResponse contains duplicate artifacts")
    returned = set(outputs)
    reachable: set[ArtifactRef] = {outputs[0]}
    queued = [outputs[0]]
    visited: set[ArtifactRef] = set()
    while queued:
        reference = queued.pop()
        if reference in visited:
            continue
        visited.add(reference)
        store.resolve(reference)
        if reference.media_type != "application/json":
            continue
        value = store.read_json(reference)
        for nested in _nested_references(value):
            store.resolve(nested)
            if nested in allowed_inputs:
                continue
            if nested not in returned:
                raise ValueError(
                    "quick Vina output references an artifact absent from WorkerResponse"
                )
            reachable.add(nested)
            if nested.media_type == "application/json":
                queued.append(nested)
    if returned != reachable:
        raise ValueError("quick Vina WorkerResponse contains an unreferenced artifact")


def validate_quick_vina_batch(
    store: ArtifactStore,
    preparation: ArtifactRef,
    quick_input: ArtifactRef,
    outputs: tuple[ArtifactRef, ...],
    *,
    case_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Validate exact request coverage and return finalizer-ready evaluations."""

    if not outputs or outputs[0].media_type != "application/json":
        raise ValueError("quick Vina primary output must be an application/json batch")
    batch_ref = outputs[0]
    batch = store.read_json(batch_ref)
    input_value = store.read_json(quick_input)
    preparation_value = store.read_json(preparation)
    if (
        not isinstance(batch, dict)
        or batch.get("schema_version") != "1.0"
        or batch.get("kind") != QUICK_VINA_BATCH_KIND
        or batch.get("purpose") != QUICK_VINA_PURPOSE
    ):
        raise ValueError("quick Vina primary output has the wrong schema or purpose")
    if (
        not isinstance(input_value, dict)
        or input_value.get("schema_version") != "1.0"
        or input_value.get("kind") != QUICK_VINA_INPUT_KIND
        or input_value.get("purpose") != QUICK_VINA_PURPOSE
        or input_value.get("case_id") != case_id
        or input_value.get("selection_preparation_sha256") != preparation.sha256
    ):
        raise ValueError(
            "quick Vina input does not bind the case or selection preparation"
        )
    if not isinstance(preparation_value, dict):
        raise ValueError("selection preparation is not a JSON object")
    receptor = _reference(input_value.get("receptor"), "quick Vina receptor")
    environment_lock = _reference(
        input_value.get("environment_lock"), "quick Vina environment lock"
    )
    preparation_receptor = _reference(
        preparation_value.get("receptor"), "selection preparation receptor"
    )
    preparation_screening = _reference(
        preparation_value.get("screening_artifact"),
        "selection preparation screening",
    )
    preparation_library = _reference(
        preparation_value.get("library_index"), "selection preparation library"
    )
    if (
        receptor != preparation_receptor
        or input_value.get("screening_sha256") != preparation_screening.sha256
        or input_value.get("library_index_sha256") != preparation_library.sha256
    ):
        raise ValueError("quick Vina input commitments differ from preparation")
    if _reference(batch.get("input"), "quick Vina batch input") != quick_input:
        raise ValueError("quick Vina batch is bound to a different input")
    if batch.get("selection_preparation_sha256") != preparation.sha256:
        raise ValueError("quick Vina batch is bound to a different preparation")
    if _reference(batch.get("receptor"), "quick Vina batch receptor") != receptor:
        raise ValueError("quick Vina batch receptor differs from its input")
    preparation_box_receipt = _reference(
        preparation_value.get("docking_box_receipt"),
        "selection docking box receipt",
    )
    input_box_receipt = _reference(
        input_value.get("docking_box_receipt"), "quick Vina docking box receipt"
    )
    if input_box_receipt != preparation_box_receipt:
        raise ValueError("quick Vina input has a different docking box receipt")
    receipt_value = validate_docking_box_receipt(
        store, input_box_receipt, receptor=receptor
    )
    receipt_site_evidence = receipt_value["site_derivation_evidence"]
    if (
        input_value.get("site_derivation_evidence") != receipt_site_evidence
        or preparation_value.get("site_derivation_evidence")
        != receipt_site_evidence
    ):
        raise ValueError("quick Vina site evidence differs from its box receipt")
    if (
        _reference(batch.get("docking_box_receipt"), "batch docking box receipt")
        != input_box_receipt
    ):
        raise ValueError("quick Vina batch has a different docking box receipt")
    requests = _validated_requests(store, preparation_value)
    projected_requests = input_value.get("requests")
    if not isinstance(projected_requests, list) or len(projected_requests) != len(
        requests
    ) or input_value.get("request_count") != len(requests):
        raise ValueError("quick Vina input request count differs from preparation")
    request_by_id = {str(item["request_id"]): item for item in requests}
    projected_by_id = {
        str(item.get("request_id")): item
        for item in projected_requests
        if isinstance(item, dict)
    }
    if set(projected_by_id) != set(request_by_id) or len(projected_by_id) != len(
        projected_requests
    ):
        raise ValueError("quick Vina input request identities differ from preparation")
    for request_id, request in request_by_id.items():
        projected = projected_by_id[request_id]
        for name in (
            "molecule_id",
            "microstate_id",
            "canonical_isomeric_smiles",
            "heavy_element_counts",
            "formal_charge",
            "box_center",
            "box_size",
            "box_source",
            "coordinate_frame",
            "docking_box_receipt_sha256",
            "purpose",
        ):
            if projected.get(name) != request.get(name):
                raise ValueError(f"quick Vina input changed request field {name}")
    raw_evaluations = batch.get("evaluations")
    if not isinstance(raw_evaluations, list):
        raise ValueError("quick Vina batch evaluations must be an array")
    observed: set[str] = set()
    evaluations: list[dict[str, Any]] = []
    evidence_profiles: set[tuple[int, int, int, str]] = set()
    completed = 0
    failed = 0
    for raw in raw_evaluations:
        if not isinstance(raw, dict):
            raise ValueError("quick Vina evaluation must be an object")
        request_id = raw.get("request_id")
        if (
            not isinstance(request_id, str)
            or request_id not in request_by_id
            or request_id in observed
        ):
            raise ValueError("quick Vina evaluation identity is unexpected or duplicated")
        observed.add(request_id)
        request = request_by_id[request_id]
        if raw.get("molecule_id") != request["molecule_id"] or raw.get(
            "microstate_id"
        ) != request["microstate_id"]:
            raise ValueError("quick Vina evaluation changed molecule/microstate identity")
        if raw.get("box_center") != request["box_center"] or raw.get(
            "box_size"
        ) != request["box_size"]:
            raise ValueError("quick Vina evaluation changed the frozen docking box")
        if (
            raw.get("coordinate_frame") != receipt_value["coordinate_frame"]
            or raw.get("box_source") != receipt_value["source_kind"]
            or raw.get("docking_box_receipt_sha256") != input_box_receipt.sha256
        ):
            raise ValueError(
                "quick Vina evaluation changed the docking box provenance"
            )
        if raw.get("seed") != seed:
            raise ValueError("quick Vina evaluation used a different seed")
        status = raw.get("status")
        if status == "failed":
            if any(name in raw for name in ("score", "pose", "evidence")):
                raise ValueError(
                    "failed quick Vina evaluation cannot contain score, pose, or evidence"
                )
            code = raw.get("code")
            reason = raw.get("reason")
            if (
                code != "UNSUPPORTED_CHEMISTRY"
                or raw.get("recoverable") is not False
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError(
                    "quick Vina batches may eliminate only deterministic unsupported "
                    "chemistry; tool/runtime failures must fail the whole worker"
                )
            evaluations.append(
                {
                    "request_id": request_id,
                    "molecule_id": request["molecule_id"],
                    "microstate_id": request["microstate_id"],
                    "status": "failed",
                    "code": code.strip(),
                    "reason": reason.strip(),
                    "recoverable": False,
                }
            )
            failed += 1
            continue
        if status != "completed":
            raise ValueError("quick Vina evaluation status must be completed or failed")
        score = raw.get("score")
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError("completed quick Vina evaluation requires a finite score")
        semantics = raw.get("score_semantics")
        if semantics != _SCORE_SEMANTICS:
            raise ValueError("quick Vina score semantics must disclaim experimental energy")
        pose = _reference(raw.get("pose"), "quick Vina pose")
        evidence = _reference(raw.get("evidence"), "quick Vina evidence")
        store.resolve(pose)
        evidence_value = store.read_json(evidence)
        if (
            not isinstance(evidence_value, dict)
            or evidence_value.get("schema_version") != "1.0"
            or evidence_value.get("kind") != "protbind.tool-evidence"
            or evidence_value.get("tool") != "vina"
            or evidence_value.get("purpose") != QUICK_VINA_PURPOSE
            or evidence_value.get("request_id") != request_id
            or evidence_value.get("molecule_id") != request["molecule_id"]
            or evidence_value.get("microstate_id") != request["microstate_id"]
            or evidence_value.get("seed") != seed
        ):
            raise ValueError("quick Vina evidence identity or purpose is invalid")
        inputs = evidence_value.get("inputs")
        metrics = evidence_value.get("metrics")
        if not isinstance(inputs, dict) or not isinstance(metrics, dict):
            raise ValueError("quick Vina evidence lacks inputs or metrics")
        if (
            _reference(inputs.get("quick_vina_input"), "evidence quick input")
            != quick_input
            or _reference(inputs.get("receptor"), "evidence receptor") != receptor
            or _reference(inputs.get("pose"), "evidence pose") != pose
            or _reference(
                inputs.get("docking_box_receipt"), "evidence docking box receipt"
            )
            != input_box_receipt
        ):
            raise ValueError("quick Vina evidence is bound to different input artifacts")
        inner = _reference(
            inputs.get("inner_vina_evidence"), "inner Vina evidence"
        )
        inner_value = store.read_json(inner)
        inner_inputs = inner_value.get("inputs") if isinstance(inner_value, dict) else None
        inner_metrics = inner_value.get("metrics") if isinstance(inner_value, dict) else None
        if (
            not isinstance(inner_value, dict)
            or inner_value.get("schema_version") != "1.0"
            or inner_value.get("kind") != "protbind.tool-evidence"
            or inner_value.get("tool") != "vina"
            or not isinstance(inner_value.get("tool_version"), str)
            or not inner_value["tool_version"].strip()
            or inner_value.get("candidate_id") != f"vina-{request_id}"
            or inner_value.get("parent_candidate_id") != request_id
            or inner_value.get("molecule_id") != request["molecule_id"]
            or inner_value.get("microstate_id") != request["microstate_id"]
            or inner_value.get("seed") != seed
            or not isinstance(inner_inputs, dict)
            or _reference(inner_inputs.get("receptor"), "inner evidence receptor")
            != receptor
            or _reference(inner_inputs.get("pose"), "inner evidence pose") != pose
            or _reference(inner_inputs.get("pose_sdf"), "inner evidence pose SDF")
            != pose
            or not isinstance(inner_metrics, dict)
            or not math.isclose(
                float(inner_metrics.get("score", math.nan)),
                float(score),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or inner_metrics.get("box_center") != request["box_center"]
            or inner_metrics.get("box_size") != request["box_size"]
            or inner_metrics.get("score_semantics") != _SCORE_SEMANTICS
            or inner_metrics.get("scoring") != "vina"
            or inner_metrics.get("cpu") != 1
            or not isinstance(inner_metrics.get("exhaustiveness"), int)
            or not 1 <= inner_metrics["exhaustiveness"] <= 16
            or not isinstance(inner_metrics.get("num_modes"), int)
            or not 1 <= inner_metrics["num_modes"] <= 3
            or inner_metrics.get("heavy_element_counts")
            != request["heavy_element_counts"]
            or inner_metrics.get("formal_charge") != request["formal_charge"]
        ):
            raise ValueError("inner Vina evidence does not bind the quick result")
        if (
            not math.isclose(
                float(metrics.get("score", math.nan)),
                float(score),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or metrics.get("box_center") != request["box_center"]
            or metrics.get("box_size") != request["box_size"]
            or metrics.get("coordinate_frame")
            != receipt_value["coordinate_frame"]
            or metrics.get("box_source") != receipt_value["source_kind"]
            or metrics.get("docking_box_receipt_sha256")
            != input_box_receipt.sha256
            or metrics.get("purpose") != QUICK_VINA_PURPOSE
            or metrics.get("score_semantics") != semantics
            or metrics.get("scoring") != inner_metrics["scoring"]
            or metrics.get("cpu") != inner_metrics["cpu"]
            or metrics.get("exhaustiveness") != inner_metrics["exhaustiveness"]
            or metrics.get("num_modes") != inner_metrics["num_modes"]
            or metrics.get("formal_charge") != request["formal_charge"]
        ):
            raise ValueError("quick Vina evidence metrics differ from the evaluation")
        evidence_profiles.add(
            (
                int(metrics["cpu"]),
                int(metrics["exhaustiveness"]),
                int(metrics["num_modes"]),
                str(metrics["scoring"]),
            )
        )
        evaluations.append(
            {
                "request_id": request_id,
                "molecule_id": request["molecule_id"],
                "microstate_id": request["microstate_id"],
                "status": "completed",
                "score": float(score),
                "score_semantics": semantics,
                "pose": pose.to_dict(),
                "evidence": evidence.to_dict(),
            }
        )
        completed += 1
    if observed != set(request_by_id):
        raise ValueError("quick Vina evaluations do not exactly cover every request")
    if (
        batch.get("request_count") != len(requests)
        or batch.get("success_count") != completed
        or batch.get("failure_count") != failed
    ):
        raise ValueError("quick Vina batch counts do not match its evaluations")
    metadata = _reference(batch.get("run_metadata"), "quick Vina run metadata")
    metadata_value = store.read_json(metadata)
    execution = metadata_value.get("execution") if isinstance(metadata_value, dict) else None
    if (
        not isinstance(metadata_value, dict)
        or metadata_value.get("schema_version") != "1.0"
        or _reference(
            metadata_value.get("docking_box_receipt"),
            "metadata docking box receipt",
        )
        != input_box_receipt
        or not isinstance(execution, dict)
        or execution.get("device") != "cpu"
        or execution.get("cpu_threads") != 1
        or execution.get("purpose") != QUICK_VINA_PURPOSE
        or execution.get("seed") != seed
        or execution.get("scoring") != "vina"
        or not isinstance(execution.get("exhaustiveness"), int)
        or not 1 <= execution["exhaustiveness"] <= 16
        or not isinstance(execution.get("num_modes"), int)
        or not 1 <= execution["num_modes"] <= 3
        or execution.get("input_candidate_count") != len(requests)
        or execution.get("successful_candidate_count") != completed
        or execution.get("failed_candidate_count") != failed
        or execution.get("coordinate_frame")
        != receipt_value["coordinate_frame"]
        or execution.get("box_source") != receipt_value["source_kind"]
        or execution.get("docking_box_receipt_sha256")
        != input_box_receipt.sha256
        or execution.get("site_derivation_verified")
        is not receipt_value["validation"]["site_derivation_verified"]
        or execution.get("site_scientific_interpretation")
        != receipt_value["validation"]["scientific_interpretation"]
    ):
        raise ValueError("quick Vina run metadata does not prove the CPU ranking profile")
    if (
        _reference(metadata_value.get("environment_lock"), "metadata environment lock")
        != environment_lock
    ):
        raise ValueError("quick Vina run metadata has a different environment lock")
    expected_profile = (
        int(execution["cpu_threads"]),
        int(execution["exhaustiveness"]),
        int(execution["num_modes"]),
        str(execution["scoring"]),
    )
    if evidence_profiles and evidence_profiles != {expected_profile}:
        raise ValueError("quick Vina evidence differs from the recorded execution profile")
    inner_metadata = _reference(
        metadata_value.get("inner_vina_run_metadata"), "inner Vina run metadata"
    )
    inner_metadata_value = store.read_json(inner_metadata)
    inner_execution = (
        inner_metadata_value.get("execution")
        if isinstance(inner_metadata_value, dict)
        else None
    )
    if (
        not isinstance(inner_metadata_value, dict)
        or inner_metadata_value.get("schema_version") != "1.0"
        or _reference(
            inner_metadata_value.get("environment_lock"),
            "inner metadata environment lock",
        )
        != environment_lock
        or not isinstance(inner_execution, dict)
        or inner_execution.get("device") != "cpu"
        or inner_execution.get("cpu_threads") != execution["cpu_threads"]
        or inner_execution.get("seed") != seed
        or inner_execution.get("scoring") != execution["scoring"]
        or inner_execution.get("exhaustiveness") != execution["exhaustiveness"]
        or inner_execution.get("num_modes") != execution["num_modes"]
        or inner_execution.get("input_candidate_count") != len(requests)
        or inner_execution.get("successful_candidate_count") != completed
        or inner_execution.get("failed_candidate_count") != failed
    ):
        raise ValueError("inner Vina run metadata differs from the quick CPU profile")
    inner_bundle = _reference(
        batch.get("inner_docking_bundle"), "inner Vina docking bundle"
    )
    inner_bundle_value = store.read_json(inner_bundle)
    if (
        not isinstance(inner_bundle_value, dict)
        or inner_bundle_value.get("kind") != "protbind.docking-bundle"
        or _reference(inner_bundle_value.get("receptor"), "inner bundle receptor")
        != receptor
        or _reference(
            inner_bundle_value.get("run_metadata"), "inner bundle run metadata"
        )
        != inner_metadata
        or inner_bundle_value.get("candidate_count") != completed
        or inner_bundle_value.get("failure_count") != failed
        or set(inner_bundle_value.get("upstream_candidate_ids", []))
        != set(request_by_id)
    ):
        raise ValueError("inner Vina docking bundle differs from the quick batch")
    _validate_output_closure(
        store,
        outputs,
        allowed_inputs={
            quick_input,
            receptor,
            environment_lock,
            input_box_receipt,
            *(
                {
                    _reference(
                        receipt_site_evidence, "site derivation evidence"
                    )
                }
                if receipt_site_evidence is not None
                else set()
            ),
        },
    )
    return evaluations


def finalize_selection_bundle(
    store: ArtifactStore,
    preparation: ArtifactRef,
    evaluations: list[dict[str, Any]],
    *,
    quick_vina_input: ArtifactRef | None = None,
    quick_vina_batch: ArtifactRef | None = None,
    worker_receipt: ArtifactRef | None = None,
) -> ArtifactRef:
    value = store.read_json(preparation)
    if not isinstance(value, dict) or value.get("kind") != (
        "protbind.selection-preparation"
    ):
        raise ValueError("preparation is not a selection-preparation artifact")
    requests = _validated_requests(store, value)
    box_receipt = _reference(
        value.get("docking_box_receipt"), "selection docking box receipt"
    )
    receptor = _reference(value.get("receptor"), "selection receptor")
    receipt_value = validate_docking_box_receipt(
        store, box_receipt, receptor=receptor
    )
    site_validation = receipt_value["validation"]
    overlap = site_validation["atom_overlap"]
    request_by_key = {
        (item["molecule_id"], item["microstate_id"]): item for item in requests
    }
    orchestration_references = (
        quick_vina_input,
        quick_vina_batch,
        worker_receipt,
    )
    if any(item is not None for item in orchestration_references) and not all(
        item is not None for item in orchestration_references
    ):
        raise ValueError(
            "automatic orchestration requires quick input, batch, and worker receipt"
        )
    if all(item is not None for item in orchestration_references):
        for reference in orchestration_references:
            assert reference is not None
            store.resolve(reference)
    automatic = all(item is not None for item in orchestration_references)
    evaluated: dict[tuple[str, str], dict[str, Any]] = {}
    failed: dict[tuple[str, str], dict[str, Any]] = {}
    for result in evaluations:
        key = (str(result.get("molecule_id")), str(result.get("microstate_id")))
        if key not in request_by_key or key in evaluated or key in failed:
            raise ValueError("quick Vina result is unexpected or duplicated")
        status = result.get("status", "completed")
        if status == "failed":
            code = result.get("code")
            reason = result.get("reason")
            if (
                not isinstance(code, str)
                or not code.strip()
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError("failed quick Vina result requires code and reason")
            if any(name in result for name in ("score", "pose", "evidence")):
                raise ValueError(
                    "failed quick Vina result cannot contain score, pose, or evidence"
                )
            if result.get("request_id") not in {
                None,
                request_by_key[key]["request_id"],
            }:
                raise ValueError("failed quick Vina result has the wrong request_id")
            if automatic and (
                code != "UNSUPPORTED_CHEMISTRY"
                or result.get("recoverable") is not False
            ):
                raise ValueError(
                    "automatic selection may eliminate only deterministic unsupported "
                    "chemistry"
                )
            failed[key] = {
                "request_id": request_by_key[key]["request_id"],
                "molecule_id": key[0],
                "microstate_id": key[1],
                "status": "failed",
                "code": code.strip(),
                "reason": reason.strip(),
                **(
                    {"recoverable": False}
                    if result.get("recoverable") is False
                    else {}
                ),
            }
            continue
        if status != "completed":
            raise ValueError("quick Vina status must be completed or failed")
        if result.get("request_id") not in {
            None,
            request_by_key[key]["request_id"],
        }:
            raise ValueError("completed quick Vina result has the wrong request_id")
        score = result.get("score")
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError("quick Vina score must be finite")
        pose = _reference(result.get("pose"), "quick Vina pose")
        evidence = _reference(result.get("evidence"), "quick Vina evidence")
        store.resolve(pose)
        evidence_value = store.read_json(evidence)
        if (
            not isinstance(evidence_value, dict)
            or evidence_value.get("tool") != "vina"
            or evidence_value.get("molecule_id") != key[0]
            or evidence_value.get("microstate_id") != key[1]
            or not math.isclose(
                float(evidence_value.get("metrics", {}).get("score", math.nan)),
                float(score),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("quick Vina evidence does not bind the result")
        evaluated[key] = {
            **result,
            "request_id": request_by_key[key]["request_id"],
            "status": "completed",
            "score": float(score),
            "score_semantics": _SCORE_SEMANTICS,
            "pose": pose.to_dict(),
            "evidence": evidence.to_dict(),
        }
    if set(evaluated) | set(failed) != set(request_by_key):
        raise ValueError(
            "quick Vina successes and failures must exactly cover every planned request"
        )
    if not evaluated:
        raise ValueError("every planned quick Vina request failed; no candidate can be selected")
    ordered = sorted(
        evaluated.values(),
        key=lambda item: (
            float(item["score"]),
            str(item["molecule_id"]),
            str(item["microstate_id"]),
        ),
    )
    best_by_molecule: dict[str, dict[str, Any]] = {}
    for item in ordered:
        best_by_molecule.setdefault(str(item["molecule_id"]), item)
    best = list(best_by_molecule.values())[:16]
    candidates: list[dict[str, Any]] = []
    for item in best:
        request = request_by_key[(item["molecule_id"], item["microstate_id"])]
        candidate_digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "2.0",
                    "molecule_id": item["molecule_id"],
                    "microstate_id": item["microstate_id"],
                    "request_id": item["request_id"],
                }
            )
        )
        candidates.append(
            {
                "candidate_id": f"selected-{candidate_digest[:20]}",
                "request_id": item["request_id"],
                "molecule_id": item["molecule_id"],
                "microstate_id": item["microstate_id"],
                "canonical_isomeric_smiles": request["canonical_isomeric_smiles"],
                "heavy_element_counts": request["heavy_element_counts"],
                "formal_charge": request["formal_charge"],
                "receptor": value["receptor"],
                "box_center": request["box_center"],
                "box_size": request["box_size"],
                "box_source": request["box_source"],
                "coordinate_frame": request["coordinate_frame"],
                "docking_box_receipt": value["docking_box_receipt"],
                "site_evidence": {
                    "source_kind": receipt_value["source_kind"],
                    "receptor_atom_overlap_checked": True,
                    "protein_heavy_atom_count_inside_box": overlap[
                        "protein_heavy_atom_count_inside_box"
                    ],
                    "site_derivation_verified": site_validation[
                        "site_derivation_verified"
                    ],
                    "scientific_interpretation": site_validation[
                        "scientific_interpretation"
                    ],
                    "biological_site_validity_inferred": False,
                    "known_site_calibration": value["known_site_calibration"],
                },
                "quick_vina_score": item["score"],
                "quick_vina_score_semantics": _SCORE_SEMANTICS,
                "quick_vina_pose": item["pose"],
                "quick_vina_evidence": item["evidence"],
            }
        )
    bundle = {
        "schema_version": "2.0",
        "kind": "protbind.selection-bundle",
        "preparation": preparation.to_dict(),
        "screening_artifact": value["screening_artifact"],
        "library_index": value["library_index"],
        "receptor": value["receptor"],
        "docking_box_receipt": value["docking_box_receipt"],
        "site_derivation_evidence": value["site_derivation_evidence"],
        "site_evidence": value["site_evidence"],
        "known_site_calibration_receipt": value[
            "known_site_calibration_receipt"
        ],
        "known_site_calibration": value["known_site_calibration"],
        "protein_chains": value["protein_chains"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "quick_vina": {
            "purpose": QUICK_VINA_PURPOSE,
            "score_semantics": _SCORE_SEMANTICS,
            "evaluated": ordered,
            "failures": [failed[key] for key in sorted(failed)],
            "retained_molecule_ids": [item["molecule_id"] for item in best],
        },
        "eliminated": value["eliminated"],
    }
    if all(item is not None for item in orchestration_references):
        assert quick_vina_input is not None
        assert quick_vina_batch is not None
        assert worker_receipt is not None
        bundle["automatic_orchestration"] = {
            "quick_vina_input": quick_vina_input.to_dict(),
            "quick_vina_batch": quick_vina_batch.to_dict(),
            "worker_receipt": worker_receipt.to_dict(),
        }
    return store.put_json(
        bundle,
        producer="protbind.selection-builder",
        producer_version="2.5",
        source=preparation.artifact_id,
    )


def _reference(value: Any, name: str) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an ArtifactRef") from exc
