"""Public domain models and scientific boundary checks."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_AA_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+\*?$")
_PDB_ID = re.compile(r"(?:[0-9][A-Za-z0-9]{3}|PDB_[A-Za-z0-9]{8})")
_UNIPROT_ACCESSION = re.compile(r"[A-Z0-9][A-Z0-9-]{5,15}")
_CHAIN_ID = re.compile(r"[A-Za-z0-9_.-]{1,16}")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


class ResearchMode(StrEnum):
    BOTH = "both"
    LIGAND_ONLY = "ligand_only"
    POCKET_ONLY = "pocket_only"


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class EvidenceGrade(StrEnum):
    # Schema 2 terminology deliberately describes what the computation
    # established.  Redocking recovery and agreement between two methods are
    # not experimental evidence of binding or activity.
    REDOCKING_RECOVERED = "REDOCKING_RECOVERED"
    METHOD_CONSENSUS = "METHOD_CONSENSUS"
    # Retained so schema-1 reports remain readable.  New reports must not emit
    # these ambiguous labels.
    REFERENCE_SUPPORTED = "REFERENCE_SUPPORTED"
    CONSENSUS_SUPPORTED = "CONSENSUS_SUPPORTED"
    HYPOTHESIS_ONLY = "HYPOTHESIS_ONLY"
    REJECTED = "REJECTED"


class RCSBCoordinatePolicy(StrEnum):
    """Which RCSB coordinate representation an explicit PDB request means."""

    DEPOSITED_ASYMMETRIC_UNIT = "deposited_asymmetric_unit"
    BIOLOGICAL_ASSEMBLY = "biological_assembly"


class SiteProvenanceKind(StrEnum):
    """Declared origin of a prospective docking site hypothesis."""

    USER_CENTER = "user-center"
    USER_RESIDUES = "user-residues"
    COCRYSTAL_LIGAND = "co-crystal-ligand"
    FPOCKET_P2RANK_CONSENSUS = "fpocket-p2rank-consensus"
    PUBLIC_BENCHMARK_REFERENCE = "public-benchmark-reference"

    @property
    def requires_derivation_evidence(self) -> bool:
        return self not in {self.USER_CENTER, self.USER_RESIDUES}


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Content-addressed reference without an internal filesystem path."""

    sha256: str
    media_type: str
    size_bytes: int
    producer: str
    producer_version: str = "unknown"
    source: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("artifact sha256 must be 64 lowercase hexadecimal characters")
        if not self.media_type or "/" not in self.media_type:
            raise ValueError("artifact media_type must be a non-empty MIME type")
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be >= 0")
        if not self.producer.strip():
            raise ValueError("artifact producer cannot be empty")
        if self.source is not None:
            if any(ord(character) < 32 for character in self.source):
                raise ValueError("artifact source cannot contain control characters")
            if (
                self.source.lower().startswith("file:")
                or self.source.startswith(("/", "\\"))
                or _WINDOWS_ABSOLUTE.match(self.source)
            ):
                raise ValueError("artifact source cannot contain an absolute local path")

    @property
    def artifact_id(self) -> str:
        return f"sha256:{self.sha256}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArtifactRef:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class PrivacyPolicy:
    network_allowed: bool = False
    approved_domains: tuple[str, ...] = ()
    sequence_upload_allowed: bool = False
    redact_internal_paths: bool = True

    def __post_init__(self) -> None:
        if self.sequence_upload_allowed and not self.network_allowed:
            raise ValueError("sequence upload requires network_allowed=true")
        for domain in self.approved_domains:
            if not domain or "/" in domain or ":" in domain:
                raise ValueError(f"invalid approved domain: {domain!r}")


@dataclass(frozen=True, slots=True)
class TargetSpec:
    name: str
    sequences: tuple[str, ...] = ()
    structure: ArtifactRef | None = None
    pdb_id: str | None = None
    uniprot_accession: str | None = None
    rcsb_chain_ids: tuple[str, ...] = ()
    rcsb_coordinate_policy: RCSBCoordinatePolicy = (
        RCSBCoordinatePolicy.DEPOSITED_ASYMMETRIC_UNIT
    )
    rcsb_assembly_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("target name cannot be empty")
        if not self.sequences and self.structure is None:
            raise ValueError("target requires sequences or a structure artifact")
        if len(self.sequences) > 2:
            raise ValueError("v1 supports at most two protein chains")
        cleaned = tuple(sequence.strip().upper() for sequence in self.sequences)
        if any(not sequence or not _AA_SEQUENCE.fullmatch(sequence) for sequence in cleaned):
            raise ValueError(
                "protein sequences must use the 20 canonical amino acids with at most "
                "one terminal stop"
            )
        if sum(len(sequence.replace("*", "")) for sequence in cleaned) > 700:
            raise ValueError("v1 supports a total protein length of at most 700 residues")
        if self.pdb_id is not None:
            pdb_id = self.pdb_id.strip().upper()
            if not _PDB_ID.fullmatch(pdb_id):
                raise ValueError("pdb_id must be a 4-character or extended RCSB PDB ID")
            object.__setattr__(self, "pdb_id", pdb_id)
        if self.uniprot_accession is not None:
            accession = self.uniprot_accession.strip().upper()
            if not _UNIPROT_ACCESSION.fullmatch(accession):
                raise ValueError("uniprot_accession has an invalid format")
            object.__setattr__(self, "uniprot_accession", accession)
        if self.pdb_id is not None and self.uniprot_accession is not None:
            raise ValueError("target cannot request both pdb_id and uniprot_accession")
        if not isinstance(self.rcsb_coordinate_policy, RCSBCoordinatePolicy):
            object.__setattr__(
                self,
                "rcsb_coordinate_policy",
                RCSBCoordinatePolicy(self.rcsb_coordinate_policy),
            )
        assembly_id = (
            self.rcsb_assembly_id.strip() if self.rcsb_assembly_id is not None else None
        )
        if assembly_id is not None and not re.fullmatch(r"[1-9][0-9]{0,5}", assembly_id):
            raise ValueError("rcsb_assembly_id must be a positive decimal assembly ID")
        if self.rcsb_coordinate_policy is RCSBCoordinatePolicy.BIOLOGICAL_ASSEMBLY:
            if self.pdb_id is None or assembly_id is None:
                raise ValueError(
                    "biological_assembly requires an explicit pdb_id and rcsb_assembly_id"
                )
        elif assembly_id is not None:
            raise ValueError(
                "rcsb_assembly_id requires rcsb_coordinate_policy=biological_assembly"
            )
        object.__setattr__(self, "rcsb_assembly_id", assembly_id)
        chain_ids = tuple(chain.strip() for chain in self.rcsb_chain_ids)
        if len(chain_ids) > 2 or len(set(chain_ids)) != len(chain_ids):
            raise ValueError("rcsb_chain_ids must contain at most two unique chain IDs")
        if any(not _CHAIN_ID.fullmatch(chain) for chain in chain_ids):
            raise ValueError("rcsb_chain_ids contains an invalid chain ID")
        if chain_ids and self.pdb_id is None:
            raise ValueError("rcsb_chain_ids requires an explicit pdb_id")
        object.__setattr__(self, "sequences", cleaned)
        object.__setattr__(self, "rcsb_chain_ids", chain_ids)


@dataclass(frozen=True, slots=True)
class LigandHypothesis:
    smiles: str | None = None
    structure: ArtifactRef | None = None
    pharmacophore: ArtifactRef | None = None
    heavy_atom_count: int | None = None
    is_covalent: bool = False
    is_polymer: bool = False
    contains_metal: bool = False

    def __post_init__(self) -> None:
        if not (self.smiles or self.structure or self.pharmacophore):
            raise ValueError("ligand hypothesis requires SMILES, structure, or pharmacophore")
        if self.heavy_atom_count is not None:
            if self.heavy_atom_count < 1:
                raise ValueError("ligand heavy_atom_count must be >= 1")
            if self.heavy_atom_count > 100:
                raise ValueError("v1 supports ligands with at most 100 heavy atoms")
        unsupported = []
        if self.is_covalent:
            unsupported.append("covalent ligand")
        if self.is_polymer:
            unsupported.append("polymer ligand")
        if self.contains_metal:
            unsupported.append("metal-containing ligand")
        if unsupported:
            raise ValueError("unsupported v1 chemistry: " + ", ".join(unsupported))


@dataclass(frozen=True, slots=True)
class PocketHypothesis:
    residues: tuple[str, ...] = ()
    center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] | None = None
    pharmacophore: ArtifactRef | None = None
    site_provenance_kind: SiteProvenanceKind | None = None
    site_evidence: ArtifactRef | None = None
    known_site_calibration_receipt: ArtifactRef | None = None
    known_site_calibration_target_id: str | None = None

    def __post_init__(self) -> None:
        if not (self.residues or self.center or self.pharmacophore):
            raise ValueError("pocket hypothesis requires residues, center, or pharmacophore")
        if (self.center is None) != (self.box_size is None):
            raise ValueError("pocket center and box_size must be provided together")
        if self.center is not None and self.box_size is not None:
            if len(self.center) != 3 or len(self.box_size) != 3 or any(
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in (*self.center, *self.box_size)
            ):
                raise ValueError(
                    "pocket center and box_size must contain three finite numbers"
                )
            if any(value <= 0 for value in self.box_size):
                raise ValueError("pocket box dimensions must be positive")
        if any(not residue.strip() for residue in self.residues):
            raise ValueError("pocket residue identifiers cannot be empty")
        if self.site_provenance_kind is not None and not isinstance(
            self.site_provenance_kind, SiteProvenanceKind
        ):
            object.__setattr__(
                self,
                "site_provenance_kind",
                SiteProvenanceKind(self.site_provenance_kind),
            )
        calibration_claimed = self.known_site_calibration_receipt is not None
        if calibration_claimed != (self.known_site_calibration_target_id is not None):
            raise ValueError(
                "known-site calibration receipt and target_id must be provided together"
            )
        if calibration_claimed:
            if self.center is None or self.box_size is None:
                raise ValueError(
                    "known-site calibration requires an explicit center and box_size"
                )
            if (
                self.site_provenance_kind is None
                or not self.site_provenance_kind.requires_derivation_evidence
            ):
                raise ValueError(
                    "known-site calibration requires independent site provenance"
                )
            target_id = self.known_site_calibration_target_id
            assert target_id is not None
            if _SAFE_ID.fullmatch(target_id) is None:
                raise ValueError("known-site calibration target_id is invalid")
        if self.site_evidence is not None and self.site_provenance_kind is None:
            raise ValueError("site_evidence requires an explicit site_provenance_kind")
        if (
            self.site_evidence is not None
            and self.site_provenance_kind is not None
            and not self.site_provenance_kind.requires_derivation_evidence
        ):
            raise ValueError(
                "user-declared site provenance cannot carry independent derivation "
                "evidence; declare the exact independent source kind"
            )
        if (
            self.site_provenance_kind is not None
            and self.site_provenance_kind.requires_derivation_evidence
            and self.site_evidence is None
        ):
            raise ValueError(
                f"{self.site_provenance_kind.value} requires site derivation evidence"
            )


@dataclass(frozen=True, slots=True)
class ResearchCase:
    case_id: str
    target: TargetSpec
    mode: ResearchMode
    ligand: LigandHypothesis | None = None
    pocket: PocketHypothesis | None = None
    privacy: PrivacyPolicy = field(default_factory=PrivacyPolicy)
    seed: int = 20260721
    title: str | None = None

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.case_id):
            raise ValueError("case_id must be a safe 1-128 character identifier")
        if not isinstance(self.mode, ResearchMode):
            object.__setattr__(self, "mode", ResearchMode(self.mode))
        if self.seed < 0 or self.seed > 2**32 - 1:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
        if self.mode is ResearchMode.BOTH and not (self.ligand and self.pocket):
            raise ValueError("mode=both requires ligand and pocket hypotheses")
        if self.mode is ResearchMode.LIGAND_ONLY and self.ligand is None:
            raise ValueError("mode=ligand_only requires a ligand hypothesis")
        if self.mode is ResearchMode.POCKET_ONLY and self.pocket is None:
            raise ValueError("mode=pocket_only requires a pocket hypothesis")
        if (
            self.pocket is not None
            and self.pocket.known_site_calibration_receipt is not None
            and self.mode is not ResearchMode.BOTH
        ):
            raise ValueError("known-site calibration is authorized only for mode=both")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResearchCase:
        payload = dict(value)
        payload["mode"] = ResearchMode(payload["mode"])
        payload["target"] = _target_from_dict(payload["target"])
        if payload.get("ligand") is not None:
            payload["ligand"] = _ligand_from_dict(payload["ligand"])
        if payload.get("pocket") is not None:
            payload["pocket"] = _pocket_from_dict(payload["pocket"])
        if payload.get("privacy") is not None:
            privacy = dict(payload["privacy"])
            privacy["approved_domains"] = tuple(privacy.get("approved_domains", ()))
            payload["privacy"] = PrivacyPolicy(**privacy)
        return cls(**payload)


def _artifact_or_none(value: dict[str, Any] | None) -> ArtifactRef | None:
    return ArtifactRef.from_dict(value) if value is not None else None


def _target_from_dict(value: dict[str, Any]) -> TargetSpec:
    payload = dict(value)
    payload["sequences"] = tuple(payload.get("sequences", ()))
    payload["rcsb_chain_ids"] = tuple(payload.get("rcsb_chain_ids", ()))
    payload["structure"] = _artifact_or_none(payload.get("structure"))
    payload["rcsb_coordinate_policy"] = RCSBCoordinatePolicy(
        payload.get(
            "rcsb_coordinate_policy",
            RCSBCoordinatePolicy.DEPOSITED_ASYMMETRIC_UNIT.value,
        )
    )
    return TargetSpec(**payload)


def _ligand_from_dict(value: dict[str, Any]) -> LigandHypothesis:
    payload = dict(value)
    payload["structure"] = _artifact_or_none(payload.get("structure"))
    payload["pharmacophore"] = _artifact_or_none(payload.get("pharmacophore"))
    return LigandHypothesis(**payload)


def _pocket_from_dict(value: dict[str, Any]) -> PocketHypothesis:
    payload = dict(value)
    payload["residues"] = tuple(payload.get("residues", ()))
    if payload.get("center") is not None:
        payload["center"] = tuple(float(item) for item in payload["center"])
    if payload.get("box_size") is not None:
        payload["box_size"] = tuple(float(item) for item in payload["box_size"])
    payload["pharmacophore"] = _artifact_or_none(payload.get("pharmacophore"))
    payload["site_evidence"] = _artifact_or_none(payload.get("site_evidence"))
    payload["known_site_calibration_receipt"] = _artifact_or_none(
        payload.get("known_site_calibration_receipt")
    )
    if payload.get("site_provenance_kind") is not None:
        payload["site_provenance_kind"] = SiteProvenanceKind(
            payload["site_provenance_kind"]
        )
    return PocketHypothesis(**payload)


@dataclass(frozen=True, slots=True)
class MoleculeRecord:
    molecule_id: str
    original_smiles: str
    standardized_smiles: str
    stereochemistry: str | None = None
    microstate_id: str | None = None
    conformer_ids: tuple[int, ...] = ()
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ScreenHit:
    molecule_id: str
    rank: int
    query_coverage: float
    median_normalized_distance_error: float
    branch_scores: dict[str, float] = field(default_factory=dict)
    branch_ranks: dict[str, int] = field(default_factory=dict)
    matched_feature_indices: tuple[int, ...] = ()
    conformer_id: int | None = None
    overlay: tuple[tuple[float, ...], ...] | None = None
    elimination_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PoseCandidate:
    candidate_id: str
    molecule_id: str
    engine: str
    structure: ArtifactRef
    seed: int
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] | None = None
    confidence_name: str | None = None
    confidence_value: float | None = None
    confidence_semantics: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationBundle:
    preparation_attested: bool = False
    posebusters_valid: bool | None = None
    symmetry_rmsd_angstrom: float | None = None
    ifp_similarity: float | None = None
    ifp_reference_recovery: float | None = None
    ifp_predicted_precision: float | None = None
    ifp_docked_label_count: int | None = None
    ifp_comparison_label_count: int | None = None
    ifp_intersection_count: int | None = None
    ifp_union_count: int | None = None
    vina_pose_valid: bool | None = None
    cofold_pose_valid: bool | None = None
    openmm_parameterized: bool | None = None
    openmm_stable: bool | None = None
    unsupported_reasons: tuple[str, ...] = ()
    evidence: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.preparation_attested, bool):
            raise ValueError("preparation_attested must be boolean")
        for name in (
            "posebusters_valid",
            "vina_pose_valid",
            "cofold_pose_valid",
            "openmm_parameterized",
            "openmm_stable",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean or null")
        if self.symmetry_rmsd_angstrom is not None and (
            not math.isfinite(self.symmetry_rmsd_angstrom)
            or self.symmetry_rmsd_angstrom < 0
        ):
            raise ValueError("symmetry RMSD must be a finite non-negative value")
        for name in (
            "ifp_similarity",
            "ifp_reference_recovery",
            "ifp_predicted_precision",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                label = {
                    "ifp_similarity": "IFP similarity",
                    "ifp_reference_recovery": "IFP reference recovery",
                    "ifp_predicted_precision": "IFP predicted precision",
                }[name]
                raise ValueError(f"{label} must be a finite value in [0, 1]")
        count_names = (
            "ifp_docked_label_count",
            "ifp_comparison_label_count",
            "ifp_intersection_count",
            "ifp_union_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if any(getattr(self, name) is not None for name in count_names) or any(
            getattr(self, name) is not None
            for name in (
                "ifp_similarity",
                "ifp_reference_recovery",
                "ifp_predicted_precision",
            )
        ):
            docked = self.ifp_docked_label_count
            comparison = self.ifp_comparison_label_count
            intersection = self.ifp_intersection_count
            union = self.ifp_union_count
            if docked is None:
                raise ValueError("IFP metrics require docked label count")
            if comparison is None:
                if any(
                    value is not None
                    for value in (
                        self.ifp_similarity,
                        self.ifp_reference_recovery,
                        self.ifp_predicted_precision,
                        intersection,
                        union,
                    )
                ):
                    raise ValueError("IFP comparison metrics require comparison label count")
            else:
                if intersection is None or union is None:
                    raise ValueError("IFP comparison metrics require intersection/union counts")
                if intersection > min(docked, comparison) or union != (
                    docked + comparison - intersection
                ):
                    raise ValueError("IFP label counts are internally inconsistent")

                def require_ratio(
                    value: float | None, numerator: int, denominator: int, name: str
                ) -> None:
                    if denominator == 0:
                        if value is not None:
                            raise ValueError(f"{name} must be null for an empty denominator")
                    elif value is None or not math.isclose(
                        float(value), numerator / denominator, rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise ValueError(f"{name} differs from its label counts")

                require_ratio(self.ifp_similarity, intersection, union, "IFP similarity")
                require_ratio(
                    self.ifp_reference_recovery,
                    intersection,
                    comparison,
                    "IFP reference recovery",
                )
                require_ratio(
                    self.ifp_predicted_precision,
                    intersection,
                    docked,
                    "IFP predicted precision",
                )
        if self.openmm_stable is True and self.openmm_parameterized is not True:
            raise ValueError("OpenMM stability requires successful parameterization")
        if any(not reason.strip() for reason in self.unsupported_reasons):
            raise ValueError("unsupported reasons cannot be empty")


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    claim_id: str
    text: str
    status: EvidenceStatus
    evidence: tuple[ArtifactRef, ...] = ()
    caveat: str | None = None

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.claim_id):
            raise ValueError("claim_id must be a safe identifier")
        if not self.text.strip():
            raise ValueError("claim text cannot be empty")
        if not isinstance(self.status, EvidenceStatus):
            object.__setattr__(self, "status", EvidenceStatus(self.status))
        if self.status is not EvidenceStatus.UNKNOWN and not self.evidence:
            raise ValueError("supported or contradicted claims require artifact evidence")
