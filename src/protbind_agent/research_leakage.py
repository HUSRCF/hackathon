"""Cross-modal leakage receipts for scientific benchmark manifests.

The bundle audits four distinct questions without collapsing them into a
single opaque score:

* protein sequence-cluster separation;
* pocket artifact and declared-cluster separation;
* PDB release-time separation;
* assay, replicate, and target-compound label separation.

Raw sequences, labels, and private identifiers never enter the receipt.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib import metadata as importlib_metadata
from itertools import combinations
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file

MANIFEST_SCHEMA_VERSION = "1.0"
BUNDLE_SCHEMA_VERSION = "1.0"
BUNDLE_KIND = "PROTBIND_RESEARCH_LEAKAGE_AUDIT"
_SHA256_LENGTH = 64
_SEQUENCE_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
_SPLIT_ROLES = {"TRAIN", "EVALUATION"}


class ResearchLeakageIntegrityError(ValueError):
    """A leakage manifest or receipt cannot be trusted."""


@dataclass(frozen=True, slots=True)
class ResearchLeakageConfig:
    sequence_identity_threshold: float = 0.3
    max_sequence_comparisons: int = 10_000
    sequence_sampling_namespace: str = "protbind-sequence-leakage-v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence_identity_threshold, bool)
            or not isinstance(self.sequence_identity_threshold, int | float)
            or not math.isfinite(float(self.sequence_identity_threshold))
            or not 0.0 < float(self.sequence_identity_threshold) <= 1.0
        ):
            raise ValueError("sequence_identity_threshold must be in (0, 1]")
        if (
            isinstance(self.max_sequence_comparisons, bool)
            or not isinstance(self.max_sequence_comparisons, int)
            or self.max_sequence_comparisons < 1
        ):
            raise ValueError("max_sequence_comparisons must be a positive integer")
        if not self.sequence_sampling_namespace.strip():
            raise ValueError("sequence_sampling_namespace must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_identity": {
                "definition": (
                    "1 - Levenshtein_edit_distance / max(sequence lengths)"
                ),
                "threshold": float(self.sequence_identity_threshold),
                "max_comparisons_per_split_pair": self.max_sequence_comparisons,
                "large_pair_policy": (
                    "deterministic SHA-256 sampling; a partial audit can detect "
                    "leakage but cannot establish its absence"
                ),
                "sampling_namespace": self.sequence_sampling_namespace,
            }
        }


@dataclass(frozen=True, slots=True)
class _Record:
    record_id: str
    split: str
    split_role: str
    protein_sequence: str
    sequence_sha256: str
    sequence_cluster_id: str
    pocket_artifact_sha256: str
    pocket_cluster_id: str
    pdb_id: str
    pdb_release_date: date
    assay_id: str
    replicate_group_id: str
    target_identity: str
    compound_parent_identity: str
    label_key: str


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise ResearchLeakageIntegrityError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchLeakageIntegrityError(f"{name} must be a non-empty string")
    return value


def _validate_source(value: str) -> None:
    if (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith("~")
    ):
        raise ResearchLeakageIntegrityError(
            "dataset source must not disclose an absolute internal path"
        )
    parsed = urlsplit(value)
    if parsed.scheme == "file":
        raise ResearchLeakageIntegrityError(
            "dataset source must not disclose a local file URL"
        )
    if parsed.scheme in {"http", "https"}:
        if parsed.username or parsed.password:
            raise ResearchLeakageIntegrityError(
                "dataset source URL must not contain credentials"
            )
        if parsed.query or parsed.fragment:
            raise ResearchLeakageIntegrityError(
                "dataset source URL must not contain query parameters or fragments"
            )


def _parse_date(value: Any, name: str) -> date:
    text = _require_string(value, name)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ResearchLeakageIntegrityError(
            f"{name} must be an ISO YYYY-MM-DD date"
        ) from exc
    if parsed.isoformat() != text:
        raise ResearchLeakageIntegrityError(
            f"{name} must be a canonical ISO YYYY-MM-DD date"
        )
    return parsed


def _label_key(value: Any, name: str) -> str:
    if value is None or isinstance(value, dict | list):
        raise ResearchLeakageIntegrityError(f"{name} must be a non-null JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ResearchLeakageIntegrityError(f"{name} must be finite")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _private_ref(value: str) -> str:
    return f"sha256:{sha256_bytes(value.encode())}"


def _subreceipt(kind: str, core: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": kind,
        **core,
    }
    return {
        **value,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(value)),
    }


def _python_edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (0 if left_value == right_value else 1),
                )
            )
        previous = current
    return previous[-1]


@lru_cache(maxsize=1)
def _alignment_backend() -> tuple[str, str | None, Any | None]:
    try:
        from Bio.Align import PairwiseAligner
    except ImportError:
        return "python-dynamic-programming", None, None
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 0.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -1.0
    try:
        version = importlib_metadata.version("biopython")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        version = None
    return "biopython.PairwiseAligner", version, aligner


def _edit_distance(left: str, right: str) -> int:
    _, _, aligner = _alignment_backend()
    if aligner is None:
        return _python_edit_distance(left, right)
    distance = -float(aligner.score(left, right))
    rounded = round(distance)
    if not math.isclose(distance, rounded, abs_tol=1e-8):
        raise ResearchLeakageIntegrityError(
            "sequence alignment backend emitted a non-integral edit distance"
        )
    return int(rounded)


def global_edit_identity(left: str, right: str) -> float:
    """Deterministic global identity based on normalized Levenshtein distance."""

    if not left or not right:
        raise ValueError("sequences must be non-empty")
    return 1.0 - _edit_distance(left, right) / max(len(left), len(right))


def _sample_sizes(left: int, right: int, maximum: int) -> tuple[int, int]:
    if left * right <= maximum:
        return left, right
    smaller = min(left, right)
    if smaller <= math.isqrt(maximum):
        if left <= right:
            return left, max(1, maximum // left)
        return max(1, maximum // right), right
    left_count = min(left, math.isqrt(maximum))
    right_count = min(right, max(1, maximum // left_count))
    return left_count, right_count


def _subset(
    values: dict[str, str],
    *,
    count: int,
    split: str,
    namespace: str,
) -> dict[str, str]:
    if count >= len(values):
        return dict(sorted(values.items()))
    selected = sorted(
        values,
        key=lambda value: sha256_bytes(
            f"{namespace}\0{split}\0{value}".encode()
        ),
    )[:count]
    return {key: values[key] for key in selected}


def _parse_manifest(path: Path) -> tuple[dict[str, Any], list[_Record]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ResearchLeakageIntegrityError("leakage manifest must be an object")
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ResearchLeakageIntegrityError(
            f"leakage manifest schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise ResearchLeakageIntegrityError("manifest.dataset must be an object")
    for field in ("name", "version", "license", "source"):
        _require_string(dataset.get(field), f"dataset.{field}")
    _validate_source(dataset["source"])

    split_roles = raw.get("split_roles")
    if not isinstance(split_roles, dict) or len(split_roles) < 2:
        raise ResearchLeakageIntegrityError(
            "split_roles must declare at least two splits"
        )
    normalized_roles: dict[str, str] = {}
    for split, role in split_roles.items():
        split_name = _require_string(split, "split_roles key")
        if role not in _SPLIT_ROLES:
            raise ResearchLeakageIntegrityError(
                f"split role for {split_name} must be TRAIN or EVALUATION"
            )
        normalized_roles[split_name] = role
    if set(normalized_roles.values()) != _SPLIT_ROLES:
        raise ResearchLeakageIntegrityError(
            "split_roles must contain TRAIN and EVALUATION"
        )

    _parse_date(raw.get("pdb_training_cutoff_date"), "pdb_training_cutoff_date")
    sequence_protocol = raw.get("sequence_cluster_protocol")
    if not isinstance(sequence_protocol, dict):
        raise ResearchLeakageIntegrityError(
            "sequence_cluster_protocol must be an object"
        )
    pocket_protocol = raw.get("pocket_cluster_protocol")
    if not isinstance(pocket_protocol, dict):
        raise ResearchLeakageIntegrityError(
            "pocket_cluster_protocol must be an object"
        )
    for protocol_name, protocol in (
        ("sequence_cluster_protocol", sequence_protocol),
        ("pocket_cluster_protocol", pocket_protocol),
    ):
        for field in ("method", "version", "threshold_semantics"):
            _require_string(
                protocol.get(field),
                f"{protocol_name}.{field}",
            )
        _require_sha256(
            protocol.get("assignment_artifact_sha256"),
            f"{protocol_name}.assignment_artifact_sha256",
        )
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise ResearchLeakageIntegrityError("provenance must be an object")
    for field in (
        "sequence_source_artifact_sha256",
        "pocket_source_artifact_sha256",
        "pdb_metadata_artifact_sha256",
        "assay_metadata_artifact_sha256",
    ):
        _require_sha256(provenance.get(field), f"provenance.{field}")

    values = raw.get("records")
    if not isinstance(values, list) or not values:
        raise ResearchLeakageIntegrityError("records must be a non-empty list")
    records: list[_Record] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(values):
        name = f"records[{index}]"
        if not isinstance(value, dict):
            raise ResearchLeakageIntegrityError(f"{name} must be an object")
        record_id = _require_string(value.get("record_id"), f"{name}.record_id")
        if record_id in seen_ids:
            raise ResearchLeakageIntegrityError(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        split = _require_string(value.get("split"), f"{name}.split")
        if split not in normalized_roles:
            raise ResearchLeakageIntegrityError(
                f"{name}.split is not declared in split_roles"
            )
        sequence = _require_string(
            value.get("protein_sequence"),
            f"{name}.protein_sequence",
        ).upper()
        if (
            len(sequence) > 5000
            or any(character not in _SEQUENCE_ALPHABET for character in sequence)
        ):
            raise ResearchLeakageIntegrityError(
                f"{name}.protein_sequence has unsupported characters or length"
            )
        pdb_id = _require_string(value.get("pdb_id"), f"{name}.pdb_id").upper()
        if len(pdb_id) != 4 or not pdb_id.isalnum():
            raise ResearchLeakageIntegrityError(
                f"{name}.pdb_id must be a four-character PDB identifier"
            )
        records.append(
            _Record(
                record_id=record_id,
                split=split,
                split_role=normalized_roles[split],
                protein_sequence=sequence,
                sequence_sha256=sha256_bytes(sequence.encode()),
                sequence_cluster_id=_require_string(
                    value.get("sequence_cluster_id"),
                    f"{name}.sequence_cluster_id",
                ),
                pocket_artifact_sha256=_require_sha256(
                    value.get("pocket_artifact_sha256"),
                    f"{name}.pocket_artifact_sha256",
                ),
                pocket_cluster_id=_require_string(
                    value.get("pocket_cluster_id"),
                    f"{name}.pocket_cluster_id",
                ),
                pdb_id=pdb_id,
                pdb_release_date=_parse_date(
                    value.get("pdb_release_date"),
                    f"{name}.pdb_release_date",
                ),
                assay_id=_require_string(
                    value.get("assay_id"),
                    f"{name}.assay_id",
                ),
                replicate_group_id=_require_string(
                    value.get("replicate_group_id"),
                    f"{name}.replicate_group_id",
                ),
                target_identity=_require_string(
                    value.get("target_identity"),
                    f"{name}.target_identity",
                ),
                compound_parent_identity=_require_string(
                    value.get("compound_parent_identity"),
                    f"{name}.compound_parent_identity",
                ),
                label_key=_label_key(value.get("label"), f"{name}.label"),
            )
        )
    return raw, records


def _split_records(records: list[_Record]) -> dict[str, list[_Record]]:
    result: dict[str, list[_Record]] = {}
    for record in records:
        result.setdefault(record.split, []).append(record)
    return dict(sorted(result.items()))


def _sequence_receipt(
    records_by_split: dict[str, list[_Record]],
    *,
    config: ResearchLeakageConfig,
    protocol: dict[str, Any],
    provenance_sha256: str,
) -> dict[str, Any]:
    sequences = {
        split: {
            record.sequence_sha256: record.protein_sequence
            for record in records
        }
        for split, records in records_by_split.items()
    }
    clusters = {
        split: {record.sequence_cluster_id for record in records}
        for split, records in records_by_split.items()
    }
    pairwise: list[dict[str, Any]] = []
    any_leakage = False
    any_partial = False
    for left_name, right_name in combinations(sorted(sequences), 2):
        left = sequences[left_name]
        right = sequences[right_name]
        exact_sequence_overlap = sorted(set(left) & set(right))
        exact_sequence_examples = [
            f"sha256:{value}" for value in exact_sequence_overlap[:20]
        ]
        cluster_count, cluster_examples = _overlap(
            clusters[left_name],
            clusters[right_name],
        )
        left_count, right_count = _sample_sizes(
            len(left),
            len(right),
            config.max_sequence_comparisons,
        )
        selected_left = _subset(
            left,
            count=left_count,
            split=left_name,
            namespace=config.sequence_sampling_namespace,
        )
        selected_right = _subset(
            right,
            count=right_count,
            split=right_name,
            namespace=config.sequence_sampling_namespace,
        )
        full = left_count == len(left) and right_count == len(right)
        any_partial = any_partial or not full
        above_threshold = 0
        maximum = 0.0
        examples: list[dict[str, Any]] = []
        for left_hash, left_sequence in selected_left.items():
            for right_hash, right_sequence in selected_right.items():
                identity = global_edit_identity(left_sequence, right_sequence)
                maximum = max(maximum, identity)
                if identity >= config.sequence_identity_threshold:
                    above_threshold += 1
                    if len(examples) < 20:
                        examples.append(
                            {
                                "left_sequence": f"sha256:{left_hash}",
                                "right_sequence": f"sha256:{right_hash}",
                                "global_edit_identity": identity,
                            }
                        )
        any_leakage = (
            any_leakage
            or bool(exact_sequence_overlap)
            or cluster_count > 0
            or above_threshold > 0
        )
        pairwise.append(
            {
                "left_split": left_name,
                "right_split": right_name,
                "status": "FULL" if full else "PARTIAL_DETERMINISTIC_SAMPLE",
                "full_comparison_count": len(left) * len(right),
                "executed_comparison_count": len(selected_left) * len(selected_right),
                "threshold": float(config.sequence_identity_threshold),
                "exact_sequence_overlap_count": len(exact_sequence_overlap),
                "exact_sequence_overlap_examples": exact_sequence_examples,
                "declared_sequence_cluster_overlap_count": cluster_count,
                "declared_sequence_cluster_overlap_examples": cluster_examples,
                "maximum_global_edit_identity": maximum,
                "at_or_above_threshold_pair_count": above_threshold,
                "leakage_examples": examples,
            }
        )
    if any_leakage:
        status = "FAIL"
    elif any_partial:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    return _subreceipt(
        "PROTBIND_SEQUENCE_CLUSTER_LEAKAGE_RECEIPT",
        {
            "provenance_artifact_sha256": provenance_sha256,
            "cluster_protocol": protocol,
            "identity_definition": (
                "1 - Levenshtein_edit_distance / max(sequence lengths)"
            ),
            "split_unique_sequence_counts": {
                split: len(values) for split, values in sequences.items()
            },
            "pairwise": pairwise,
            "sequence_cluster_precondition": {
                "status": status,
                "semantics": (
                    "PASS means no cross-split edge met the frozen global-edit identity "
                    "threshold in a FULL comparison, no exact sequence was shared, and no "
                    "declared sequence cluster crossed splits. It does not establish "
                    "model generalisation."
                ),
            },
            "cluster_method_verification": "NOT_EVALUATED",
        },
    )


def _overlap(
    left: set[str],
    right: set[str],
) -> tuple[int, list[str]]:
    values = sorted(left & right)
    return len(values), [_private_ref(value) for value in values[:20]]


def _pocket_receipt(
    records_by_split: dict[str, list[_Record]],
    protocol: dict[str, Any],
    provenance_sha256: str,
) -> dict[str, Any]:
    artifacts = {
        split: {record.pocket_artifact_sha256 for record in records}
        for split, records in records_by_split.items()
    }
    clusters = {
        split: {record.pocket_cluster_id for record in records}
        for split, records in records_by_split.items()
    }
    pairwise: list[dict[str, Any]] = []
    failed = False
    for left_name, right_name in combinations(sorted(records_by_split), 2):
        artifact_count, artifact_examples = _overlap(
            artifacts[left_name],
            artifacts[right_name],
        )
        cluster_count, cluster_examples = _overlap(
            clusters[left_name],
            clusters[right_name],
        )
        failed = failed or artifact_count > 0 or cluster_count > 0
        pairwise.append(
            {
                "left_split": left_name,
                "right_split": right_name,
                "pocket_artifact_overlap_count": artifact_count,
                "pocket_artifact_overlap_examples": artifact_examples,
                "declared_pocket_cluster_overlap_count": cluster_count,
                "declared_pocket_cluster_overlap_examples": cluster_examples,
            }
        )
    return _subreceipt(
        "PROTBIND_POCKET_CLUSTER_LEAKAGE_RECEIPT",
        {
            "provenance_artifact_sha256": provenance_sha256,
            "cluster_protocol": protocol,
            "pairwise": pairwise,
            "pocket_cluster_precondition": {
                "status": "FAIL" if failed else "PASS",
                "semantics": (
                    "Artifact overlap is verified by SHA-256. Cluster overlap is audited "
                    "from hash-bound external assignments; this receipt does not recompute "
                    "or validate the declared pocket clustering method."
                ),
            },
            "cluster_method_verification": "NOT_EVALUATED",
        },
    )


def _pdb_temporal_receipt(
    records_by_split: dict[str, list[_Record]],
    *,
    cutoff: date,
    provenance_sha256: str,
) -> dict[str, Any]:
    split_ids = {
        split: {record.pdb_id for record in records}
        for split, records in records_by_split.items()
    }
    pairwise: list[dict[str, Any]] = []
    overlap_failed = False
    for left_name, right_name in combinations(sorted(split_ids), 2):
        count, examples = _overlap(split_ids[left_name], split_ids[right_name])
        overlap_failed = overlap_failed or count > 0
        pairwise.append(
            {
                "left_split": left_name,
                "right_split": right_name,
                "pdb_id_overlap_count": count,
                "pdb_id_overlap_examples": examples,
            }
        )
    violation_count = 0
    violations: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        for record in records:
            valid = (
                record.pdb_release_date <= cutoff
                if record.split_role == "TRAIN"
                else record.pdb_release_date > cutoff
            )
            if not valid:
                violation_count += 1
                if len(violations) < 50:
                    violations.append(
                        {
                            "record": _private_ref(record.record_id),
                            "split": split,
                            "role": record.split_role,
                            "pdb": _private_ref(record.pdb_id),
                            "release_date": record.pdb_release_date.isoformat(),
                            "expected": (
                                f"<= {cutoff.isoformat()}"
                                if record.split_role == "TRAIN"
                                else f"> {cutoff.isoformat()}"
                            ),
                        }
                    )
    date_ranges = {
        split: {
            "minimum": min(record.pdb_release_date for record in records).isoformat(),
            "maximum": max(record.pdb_release_date for record in records).isoformat(),
        }
        for split, records in records_by_split.items()
    }
    failed = overlap_failed or violation_count > 0
    return _subreceipt(
        "PROTBIND_PDB_RELEASE_TIME_LEAKAGE_RECEIPT",
        {
            "provenance_artifact_sha256": provenance_sha256,
            "training_cutoff_date": cutoff.isoformat(),
            "split_release_date_ranges": date_ranges,
            "pairwise": pairwise,
            "temporal_violation_count": violation_count,
            "temporal_violations": violations,
            "pdb_temporal_precondition": {
                "status": "FAIL" if failed else "PASS",
                "semantics": (
                    "TRAIN records must be released on/before the cutoff and EVALUATION "
                    "records strictly after it, with no exact PDB ID shared across splits. "
                    "Dates are hash-bound declarations and are not re-fetched from RCSB."
                ),
            },
            "rcsb_metadata_verification": "NOT_EVALUATED",
        },
    )


def _assay_label_receipt(
    records_by_split: dict[str, list[_Record]],
    *,
    provenance_sha256: str,
) -> dict[str, Any]:
    pairwise: list[dict[str, Any]] = []
    failed = False
    for left_name, right_name in combinations(sorted(records_by_split), 2):
        left_records = records_by_split[left_name]
        right_records = records_by_split[right_name]
        left_assays = {record.assay_id for record in left_records}
        right_assays = {record.assay_id for record in right_records}
        left_replicates = {record.replicate_group_id for record in left_records}
        right_replicates = {record.replicate_group_id for record in right_records}
        left_pairs = {
            (record.target_identity, record.compound_parent_identity)
            for record in left_records
        }
        right_pairs = {
            (record.target_identity, record.compound_parent_identity)
            for record in right_records
        }
        assay_count, assay_examples = _overlap(left_assays, right_assays)
        replicate_count, replicate_examples = _overlap(
            left_replicates,
            right_replicates,
        )
        entity_pairs = sorted(left_pairs & right_pairs)
        entity_pair_examples = [
            _private_ref(f"{target}\0{compound}")
            for target, compound in entity_pairs[:20]
        ]
        left_labels: dict[tuple[str, str], set[str]] = {}
        right_labels: dict[tuple[str, str], set[str]] = {}
        for record in left_records:
            left_labels.setdefault(
                (record.target_identity, record.compound_parent_identity),
                set(),
            ).add(record.label_key)
        for record in right_records:
            right_labels.setdefault(
                (record.target_identity, record.compound_parent_identity),
                set(),
            ).add(record.label_key)
        conflicting = [
            pair
            for pair in entity_pairs
            if left_labels[pair] != right_labels[pair]
        ]
        failed = failed or any(
            (
                assay_count,
                replicate_count,
                len(entity_pairs),
            )
        )
        pairwise.append(
            {
                "left_split": left_name,
                "right_split": right_name,
                "assay_id_overlap_count": assay_count,
                "assay_id_overlap_examples": assay_examples,
                "replicate_group_overlap_count": replicate_count,
                "replicate_group_overlap_examples": replicate_examples,
                "target_compound_pair_overlap_count": len(entity_pairs),
                "target_compound_pair_overlap_examples": entity_pair_examples,
                "conflicting_label_pair_count": len(conflicting),
                "conflicting_label_pair_examples": [
                    _private_ref(f"{target}\0{compound}")
                    for target, compound in conflicting[:20]
                ],
            }
        )
    within_split: dict[str, dict[str, int]] = {}
    within_failed = False
    for split, records in records_by_split.items():
        pairs = [
            (record.target_identity, record.compound_parent_identity)
            for record in records
        ]
        duplicate_count = len(pairs) - len(set(pairs))
        within_failed = within_failed or duplicate_count > 0
        within_split[split] = {
            "record_count": len(records),
            "unique_target_compound_pair_count": len(set(pairs)),
            "duplicate_target_compound_pair_record_count": duplicate_count,
        }
    failed = failed or within_failed
    return _subreceipt(
        "PROTBIND_ASSAY_LABEL_LEAKAGE_RECEIPT",
        {
            "provenance_artifact_sha256": provenance_sha256,
            "within_split": within_split,
            "pairwise": pairwise,
            "assay_label_precondition": {
                "status": "FAIL" if failed else "PASS",
                "semantics": (
                    "The strict broad-novelty gate requires assay IDs, replicate groups, "
                    "and target-compound entity pairs to be disjoint across splits and "
                    "entity pairs to be unique within each split. This may be stricter "
                    "than a target-specific random split and must match the intended claim."
                ),
            },
        },
    )


def build_research_leakage_audit(
    manifest_path: Path,
    *,
    config: ResearchLeakageConfig | None = None,
) -> dict[str, Any]:
    """Build four independent leakage receipts and a hash-bound bundle."""

    config = config or ResearchLeakageConfig()
    manifest, records = _parse_manifest(manifest_path)
    records_by_split = _split_records(records)
    if set(records_by_split) != set(manifest["split_roles"]):
        missing = sorted(set(manifest["split_roles"]) - set(records_by_split))
        raise ResearchLeakageIntegrityError(
            f"every declared split must contain records; empty splits: {missing}"
        )
    sequence = _sequence_receipt(
        records_by_split,
        config=config,
        protocol=manifest["sequence_cluster_protocol"],
        provenance_sha256=manifest["provenance"][
            "sequence_source_artifact_sha256"
        ],
    )
    pocket = _pocket_receipt(
        records_by_split,
        manifest["pocket_cluster_protocol"],
        manifest["provenance"]["pocket_source_artifact_sha256"],
    )
    temporal = _pdb_temporal_receipt(
        records_by_split,
        cutoff=_parse_date(
            manifest["pdb_training_cutoff_date"],
            "pdb_training_cutoff_date",
        ),
        provenance_sha256=manifest["provenance"][
            "pdb_metadata_artifact_sha256"
        ],
    )
    assay = _assay_label_receipt(
        records_by_split,
        provenance_sha256=manifest["provenance"][
            "assay_metadata_artifact_sha256"
        ],
    )
    statuses = {
        "sequence_cluster": sequence["sequence_cluster_precondition"]["status"],
        "pocket_cluster": pocket["pocket_cluster_precondition"]["status"],
        "pdb_temporal": temporal["pdb_temporal_precondition"]["status"],
        "assay_label": assay["assay_label_precondition"]["status"],
    }
    if any(value == "FAIL" for value in statuses.values()):
        broad_status = "FAIL"
    elif any(value == "INCOMPLETE" for value in statuses.values()):
        broad_status = "INCOMPLETE"
    else:
        broad_status = "PASS"
    core = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "dataset": manifest["dataset"],
        "input_binding": {
            "filename": manifest_path.name,
            "file_sha256": sha256_file(manifest_path),
        },
        "config": config.to_dict(),
        "implementation": {
            "module": "protbind_agent.research_leakage",
            "source_sha256": sha256_file(Path(__file__)),
            "sequence_identity_backend": {
                "name": _alignment_backend()[0],
                "version": _alignment_backend()[1],
            },
        },
        "split_summary": {
            split: {
                "role": manifest["split_roles"][split],
                "record_count": len(values),
                "unique_sequence_count": len(
                    {record.sequence_sha256 for record in values}
                ),
                "unique_pdb_count": len({record.pdb_id for record in values}),
            }
            for split, values in records_by_split.items()
        },
        "receipts": {
            "sequence_cluster": sequence,
            "pocket_cluster": pocket,
            "pdb_temporal": temporal,
            "assay_label": assay,
        },
        "gate": {
            "component_statuses": statuses,
            "broad_cross_modal_novelty_precondition": {
                "status": broad_status,
                "semantics": (
                    "PASS is only a strict cross-modal data-integrity precondition. "
                    "It does not demonstrate model performance or biological validity."
                ),
            },
        },
        "privacy": {
            "raw_sequences_in_receipt": False,
            "raw_labels_in_receipt": False,
            "raw_record_ids_in_receipt": False,
            "raw_assay_or_entity_ids_in_receipt": False,
            "examples": "SHA-256 commitments only",
        },
        "scientific_boundaries": [
            "Global edit identity is deterministic but is not MMseqs local sequence "
            "identity; claims must name the metric actually used.",
            "Partial sequence comparisons can detect leakage but cannot establish its "
            "absence.",
            "Pocket cluster assignments and PDB release dates are hash-bound declarations; "
            "their upstream computation or RCSB correctness is not independently verified.",
            "The assay gate is intentionally strict for broad novelty and may not match a "
            "target-specific random-split estimand.",
            "A PASS receipt does not establish docking accuracy, enrichment, affinity, "
            "activity, or external generalisation.",
        ],
    }
    return {
        **core,
        "bundle_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def verify_research_leakage_audit(result: dict[str, Any]) -> None:
    """Verify nested receipt commitments and the bundle commitment."""

    if not isinstance(result, dict):
        raise ResearchLeakageIntegrityError("research leakage audit must be an object")
    if result.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ResearchLeakageIntegrityError(
            f"research leakage schema_version must be {BUNDLE_SCHEMA_VERSION}"
        )
    if result.get("kind") != BUNDLE_KIND:
        raise ResearchLeakageIntegrityError(
            f"research leakage kind must be {BUNDLE_KIND}"
        )
    receipts = result.get("receipts")
    if not isinstance(receipts, dict) or set(receipts) != {
        "sequence_cluster",
        "pocket_cluster",
        "pdb_temporal",
        "assay_label",
    }:
        raise ResearchLeakageIntegrityError(
            "research leakage bundle has incomplete component receipts"
        )
    expected_kinds = {
        "sequence_cluster": "PROTBIND_SEQUENCE_CLUSTER_LEAKAGE_RECEIPT",
        "pocket_cluster": "PROTBIND_POCKET_CLUSTER_LEAKAGE_RECEIPT",
        "pdb_temporal": "PROTBIND_PDB_RELEASE_TIME_LEAKAGE_RECEIPT",
        "assay_label": "PROTBIND_ASSAY_LABEL_LEAKAGE_RECEIPT",
    }
    for name, receipt in receipts.items():
        if not isinstance(receipt, dict):
            raise ResearchLeakageIntegrityError(f"{name} receipt must be an object")
        if receipt.get("kind") != expected_kinds[name]:
            raise ResearchLeakageIntegrityError(
                f"{name} receipt kind must be {expected_kinds[name]}"
            )
        core = {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
        if receipt.get("receipt_sha256") != sha256_bytes(
            canonical_json_bytes(core)
        ):
            raise ResearchLeakageIntegrityError(f"{name} receipt hash mismatch")
    core = {key: value for key, value in result.items() if key != "bundle_sha256"}
    if result.get("bundle_sha256") != sha256_bytes(canonical_json_bytes(core)):
        raise ResearchLeakageIntegrityError("research leakage bundle hash mismatch")


def load_research_leakage_audit(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    verify_research_leakage_audit(result)
    return result


def persist_research_leakage_audit(result: dict[str, Any], output: Path) -> None:
    """Atomically persist one verified cross-modal leakage bundle."""

    verify_research_leakage_audit(result)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
