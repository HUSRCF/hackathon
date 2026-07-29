"""Local-first protein--ligand research workflow for AMD Radeon systems.

The package keeps scientific computation deterministic and tool-owned.  Language
models may plan and explain a run, but they never manufacture structures or scores.
"""

from .models import (
    ArtifactRef,
    EvidenceClaim,
    EvidenceGrade,
    EvidenceStatus,
    LigandHypothesis,
    MoleculeRecord,
    PocketHypothesis,
    PoseCandidate,
    PrivacyPolicy,
    ResearchCase,
    ResearchMode,
    ScreenHit,
    SiteProvenanceKind,
    TargetSpec,
    ValidationBundle,
)

__all__ = [
    "ArtifactRef",
    "EvidenceClaim",
    "EvidenceGrade",
    "EvidenceStatus",
    "LigandHypothesis",
    "MoleculeRecord",
    "PocketHypothesis",
    "PoseCandidate",
    "PrivacyPolicy",
    "ResearchCase",
    "ResearchMode",
    "ScreenHit",
    "SiteProvenanceKind",
    "TargetSpec",
    "ValidationBundle",
]

__version__ = "0.1.0"
