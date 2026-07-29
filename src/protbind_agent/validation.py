"""Evidence-level decisions; no learned or opaque aggregate score is used."""

from __future__ import annotations

from .models import EvidenceGrade, ValidationBundle


def classify_evidence(
    bundle: ValidationBundle,
    *,
    has_reference_pose: bool,
    rmsd_threshold_angstrom: float = 2.0,
    ifp_consensus_threshold: float = 0.5,
) -> EvidenceGrade:
    if rmsd_threshold_angstrom <= 0:
        raise ValueError("RMSD threshold must be positive")
    if not 0 <= ifp_consensus_threshold <= 1:
        raise ValueError("IFP consensus threshold must be in [0, 1]")
    if bundle.posebusters_valid is None:
        raise ValueError(
            "evidence grading requires an explicit PoseBusters validity result"
        )
    # The docked pose is the primary object being validated.  Optional cofold
    # evidence may be absent or invalid without rejecting an otherwise valid
    # Vina pose.  Likewise, an unavailable parameterization is an explicit
    # unsupported result, not fabricated negative evidence.
    hard_failure = (
        bundle.posebusters_valid is False
        or bundle.vina_pose_valid is False
        or bundle.openmm_stable is False
        or any(reason.startswith("hard:") for reason in bundle.unsupported_reasons)
    )
    if hard_failure:
        return EvidenceGrade.REJECTED
    # Promotion requires the separate preparation receipts to attest that the
    # canonical SDF pose and normalized receptor were deterministically derived
    # from the exact Vina inputs.  A caller-provided batch cannot self-certify
    # this property.
    if not bundle.preparation_attested:
        return EvidenceGrade.HYPOTHESIS_ONLY
    if (
        has_reference_pose
        and bundle.posebusters_valid is True
        and bundle.symmetry_rmsd_angstrom is not None
        and bundle.symmetry_rmsd_angstrom <= rmsd_threshold_angstrom
    ):
        return EvidenceGrade.REDOCKING_RECOVERED
    if (
        not has_reference_pose
        and bundle.posebusters_valid is True
        and bundle.vina_pose_valid is True
        and bundle.cofold_pose_valid is True
        and bundle.ifp_similarity is not None
        and bundle.ifp_similarity >= ifp_consensus_threshold
    ):
        return EvidenceGrade.METHOD_CONSENSUS
    return EvidenceGrade.HYPOTHESIS_ONLY


def vina_score_language(value: float) -> str:
    """Render Vina output without misrepresenting it as experimental thermodynamics."""

    return (
        f"AutoDock Vina pose-ranking score: {value:.3f} kcal/mol (tool score only; "
        "not an experimental binding free energy)."
    )
