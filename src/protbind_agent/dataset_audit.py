"""Deterministic molecular split-leakage audit for benchmark claims.

Identity and scaffold overlap are always evaluated over every parsed record.
Morgan similarity is exact only when the declared comparison budget covers
the complete Cartesian product.  A deterministic partial audit is useful for
triage, but deliberately cannot pass the broad-generalisation precondition.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from .chemistry import ChemistryCapabilityError, bemis_murcko_scaffold_smiles

AUDIT_SCHEMA_VERSION = "1.0"
AUDIT_KIND = "PROTBIND_DATASET_LEAKAGE_AUDIT"


class DatasetAuditIntegrityError(ValueError):
    """An audit input, configuration, or receipt is not trustworthy."""


@dataclass(frozen=True, slots=True)
class DatasetAuditConfig:
    similarity_threshold: float = 0.8
    max_similarity_comparisons: int = 1_000_000
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    include_chirality: bool = True
    sampling_namespace: str = "protbind-dataset-audit-v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.similarity_threshold, bool)
            or not isinstance(self.similarity_threshold, int | float)
            or not math.isfinite(float(self.similarity_threshold))
            or not 0.0 < float(self.similarity_threshold) <= 1.0
        ):
            raise ValueError("similarity_threshold must be in (0, 1]")
        if (
            isinstance(self.max_similarity_comparisons, bool)
            or not isinstance(self.max_similarity_comparisons, int)
            or self.max_similarity_comparisons < 1
        ):
            raise ValueError("max_similarity_comparisons must be a positive integer")
        if not 1 <= self.fingerprint_radius <= 4:
            raise ValueError("fingerprint_radius must be in [1, 4]")
        if self.fingerprint_bits not in {512, 1024, 2048, 4096}:
            raise ValueError("fingerprint_bits must be one of 512, 1024, 2048, 4096")
        if not self.sampling_namespace.strip():
            raise ValueError("sampling_namespace must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "similarity_threshold": float(self.similarity_threshold),
            "max_similarity_comparisons_per_split_pair": (
                self.max_similarity_comparisons
            ),
            "fingerprint": {
                "type": "Morgan",
                "radius": self.fingerprint_radius,
                "bits": self.fingerprint_bits,
                "include_chirality": self.include_chirality,
            },
            "sampling_namespace": self.sampling_namespace,
            "large_pair_policy": (
                "deterministic SHA-256 sampling; partial similarity audits remain "
                "INCOMPLETE and cannot pass broad-generalisation preconditions"
            ),
        }


@dataclass(frozen=True, slots=True)
class _MoleculeIdentity:
    standardized_isomeric_smiles: str
    parent_identity: str
    connectivity_identity: str
    scaffold: str


def _rdkit() -> tuple[Any, Any, Any, Any]:
    try:
        from rdkit import Chem, DataStructs, rdBase
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:
        raise ChemistryCapabilityError(
            "RDKit is required for molecular dataset leakage audits"
        ) from exc
    return Chem, DataStructs, rdBase, (rdFingerprintGenerator, rdMolStandardize)


def parse_split_spec(value: str) -> tuple[str, Path]:
    """Parse one CLI NAME=PATH split declaration."""

    if not isinstance(value, str) or "=" not in value:
        raise ValueError("split must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if (
        not name
        or not raw_path
        or not name.replace("_", "-").replace("-", "").isalnum()
    ):
        raise ValueError("split name must contain only letters, digits, '_' or '-'")
    return name, Path(raw_path)


def _validate_dataset_source(value: str) -> None:
    if (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith("~")
    ):
        raise ValueError("dataset_source must not disclose an absolute internal path")
    parsed = urlsplit(value)
    if parsed.scheme == "file":
        raise ValueError("dataset_source must not disclose a local file URL")
    if parsed.scheme in {"http", "https"}:
        if parsed.username or parsed.password:
            raise ValueError("dataset_source URL must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError(
                "dataset_source URL must not contain query parameters or fragments"
            )


def _iter_raw_molecules(
    path: Path,
) -> Iterator[tuple[int, Any | None, str | None]]:
    chem, _, _, _ = _rdkit()
    suffix = path.suffix.lower()
    if suffix in {".smi", ".smiles", ".txt"}:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                smiles = line.split()[0]
                molecule = chem.MolFromSmiles(smiles)
                if molecule is None:
                    yield line_number, None, "INVALID_SMILES"
                else:
                    yield line_number, molecule, None
        return
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle), start=2):
                lowered = {str(key).lower(): value for key, value in row.items()}
                raw_smiles = lowered.get("smiles") or lowered.get("canonical_smiles")
                molecule = (
                    chem.MolFromSmiles(str(raw_smiles)) if raw_smiles else None
                )
                if molecule is None:
                    yield row_index, None, "MISSING_OR_INVALID_SMILES"
                else:
                    yield row_index, molecule, None
        return
    if suffix in {".sdf", ".sd"}:
        supplier = chem.SDMolSupplier(str(path), removeHs=False)
        for record_index, molecule in enumerate(supplier, start=1):
            if molecule is None:
                yield record_index, None, "INVALID_SDF_RECORD"
            else:
                yield record_index, molecule, None
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ChemistryCapabilityError(
                "PyArrow is required to audit Parquet molecular splits"
            ) from exc
        row_index = 0
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=10_000):
            for row in batch.to_pylist():
                row_index += 1
                lowered = {str(key).lower(): value for key, value in row.items()}
                raw_smiles = lowered.get("smiles") or lowered.get("canonical_smiles")
                molecule = (
                    chem.MolFromSmiles(str(raw_smiles)) if raw_smiles else None
                )
                if molecule is None:
                    yield row_index, None, "MISSING_OR_INVALID_SMILES"
                else:
                    yield row_index, molecule, None
        return
    raise ValueError(
        f"unsupported audit split {path.name}; use SMI/SMILES/TXT, CSV, SDF, or Parquet"
    )


def _identity(molecule: Any) -> _MoleculeIdentity:
    chem, _, _, modules = _rdkit()
    _, standardize = modules
    cleaned = standardize.Cleanup(chem.Mol(molecule))
    fragment_parent = standardize.FragmentParent(cleaned)
    chem.AssignStereochemistry(fragment_parent, cleanIt=True, force=True)
    standardized = str(
        chem.MolToSmiles(
            fragment_parent,
            canonical=True,
            isomericSmiles=True,
        )
    )
    charge_parent = standardize.ChargeParent(fragment_parent)
    tautomer_parent = standardize.TautomerParent(charge_parent)
    chem.AssignStereochemistry(tautomer_parent, cleanIt=True, force=True)
    parent_identity = str(
        chem.MolToSmiles(
            tautomer_parent,
            canonical=True,
            isomericSmiles=True,
        )
    )
    connectivity = str(
        chem.MolToSmiles(
            tautomer_parent,
            canonical=True,
            isomericSmiles=False,
        )
    )
    if not standardized or not parent_identity or not connectivity:
        raise ValueError("RDKit emitted an empty standardized identity")
    return _MoleculeIdentity(
        standardized_isomeric_smiles=standardized,
        parent_identity=parent_identity,
        connectivity_identity=connectivity,
        scaffold=bemis_murcko_scaffold_smiles(parent_identity),
    )


def _audit_split(name: str, path: Path) -> tuple[dict[str, Any], dict[str, _MoleculeIdentity]]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset split is not a file: {path}")
    record_count = 0
    parsed_record_count = 0
    invalid_record_count = 0
    invalid_record_examples: list[dict[str, Any]] = []
    unique: dict[str, _MoleculeIdentity] = {}
    stereo_unique: set[str] = set()
    connectivity_unique: set[str] = set()
    scaffold_unique: set[str] = set()
    for record_index, molecule, parse_error in _iter_raw_molecules(path):
        record_count += 1
        if parse_error is not None:
            invalid_record_count += 1
            if len(invalid_record_examples) < 20:
                invalid_record_examples.append(
                    {
                        "record_index": record_index,
                        "code": parse_error,
                    }
                )
            continue
        if molecule is None:
            raise AssertionError("molecule and parse_error cannot both be empty")
        try:
            identity = _identity(molecule)
        except Exception as exc:
            invalid_record_count += 1
            if len(invalid_record_examples) < 20:
                invalid_record_examples.append(
                    {
                        "record_index": record_index,
                        "code": "STANDARDIZATION_FAILED",
                        "error_type": type(exc).__name__,
                    }
                )
            continue
        parsed_record_count += 1
        unique.setdefault(identity.parent_identity, identity)
        stereo_unique.add(identity.standardized_isomeric_smiles)
        connectivity_unique.add(identity.connectivity_identity)
        scaffold_unique.add(identity.scaffold)
    input_binding = {
        "split": name,
        "filename": path.name,
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    summary = {
        "input": input_binding,
        "record_count": record_count,
        "parsed_record_count": parsed_record_count,
        "invalid_record_count": invalid_record_count,
        "invalid_record_examples": invalid_record_examples,
        "unique_parent_identity_count": len(unique),
        "duplicate_parent_record_count": parsed_record_count - len(unique),
        "unique_stereo_identity_count": len(stereo_unique),
        "unique_connectivity_identity_count": len(connectivity_unique),
        "unique_scaffold_count": len(scaffold_unique),
    }
    return summary, unique


def _private_identity_ref(value: str) -> str:
    return f"sha256:{sha256_bytes(value.encode('utf-8'))}"


def _deterministic_subset(
    values: list[str],
    *,
    count: int,
    split_name: str,
    namespace: str,
) -> list[str]:
    if count >= len(values):
        return sorted(values)
    return [
        value
        for _, value in sorted(
            (
                sha256_bytes(
                    f"{namespace}\0{split_name}\0{value}".encode()
                ),
                value,
            )
            for value in values
        )[:count]
    ]


def _comparison_sample_sizes(left: int, right: int, maximum: int) -> tuple[int, int]:
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


def _fingerprint(value: str, *, generator: Any) -> Any:
    chem, _, _, _ = _rdkit()
    molecule = chem.MolFromSmiles(value)
    if molecule is None:
        raise DatasetAuditIntegrityError("RDKit could not reparse a canonical parent")
    return generator.GetFingerprint(molecule)


def _pair_audit(
    left_name: str,
    left: dict[str, _MoleculeIdentity],
    right_name: str,
    right: dict[str, _MoleculeIdentity],
    *,
    config: DatasetAuditConfig,
) -> dict[str, Any]:
    chem, data_structs, _, modules = _rdkit()
    del chem
    fingerprint_module, _ = modules
    left_parents = set(left)
    right_parents = set(right)
    identity_overlap = sorted(left_parents & right_parents)
    stereo_overlap = sorted(
        {value.standardized_isomeric_smiles for value in left.values()}
        & {value.standardized_isomeric_smiles for value in right.values()}
    )
    connectivity_overlap = sorted(
        {value.connectivity_identity for value in left.values()}
        & {value.connectivity_identity for value in right.values()}
    )
    scaffold_overlap = sorted(
        {value.scaffold for value in left.values()}
        & {value.scaffold for value in right.values()}
    )

    left_count, right_count = _comparison_sample_sizes(
        len(left),
        len(right),
        config.max_similarity_comparisons,
    )
    left_selected = _deterministic_subset(
        list(left),
        count=left_count,
        split_name=left_name,
        namespace=config.sampling_namespace,
    )
    right_selected = _deterministic_subset(
        list(right),
        count=right_count,
        split_name=right_name,
        namespace=config.sampling_namespace,
    )
    full = left_count == len(left) and right_count == len(right)
    generator = fingerprint_module.GetMorganGenerator(
        radius=config.fingerprint_radius,
        fpSize=config.fingerprint_bits,
        includeChirality=config.include_chirality,
    )
    left_fingerprints = {
        value: _fingerprint(value, generator=generator)
        for value in left_selected
    }
    right_fingerprints = {
        value: _fingerprint(value, generator=generator)
        for value in right_selected
    }
    maximum_similarity = 0.0
    maximum_nonidentical_similarity = 0.0
    at_or_above_threshold = 0
    nonidentical_at_or_above_threshold = 0
    comparisons = 0
    for left_value in left_selected:
        similarities = data_structs.BulkTanimotoSimilarity(
            left_fingerprints[left_value],
            [right_fingerprints[value] for value in right_selected],
        )
        for right_value, similarity_value in zip(
            right_selected,
            similarities,
            strict=True,
        ):
            similarity = float(similarity_value)
            comparisons += 1
            maximum_similarity = max(maximum_similarity, similarity)
            if similarity >= config.similarity_threshold:
                at_or_above_threshold += 1
            if left_value != right_value:
                maximum_nonidentical_similarity = max(
                    maximum_nonidentical_similarity,
                    similarity,
                )
                if similarity >= config.similarity_threshold:
                    nonidentical_at_or_above_threshold += 1

    smaller_parent_count = min(len(left), len(right))
    smaller_scaffold_count = min(
        len({value.scaffold for value in left.values()}),
        len({value.scaffold for value in right.values()}),
    )
    return {
        "left_split": left_name,
        "right_split": right_name,
        "exact_parent_identity_overlap": {
            "unique_count": len(identity_overlap),
            "fraction_of_smaller_unique_split": (
                len(identity_overlap) / smaller_parent_count
                if smaller_parent_count
                else None
            ),
            "identity_hash_examples": [
                _private_identity_ref(value) for value in identity_overlap[:20]
            ],
        },
        "stereo_identity_overlap_unique_count": len(stereo_overlap),
        "connectivity_identity_overlap_unique_count": len(connectivity_overlap),
        "scaffold_overlap": {
            "unique_count": len(scaffold_overlap),
            "fraction_of_smaller_scaffold_set": (
                len(scaffold_overlap) / smaller_scaffold_count
                if smaller_scaffold_count
                else None
            ),
            "scaffold_hash_examples": [
                _private_identity_ref(value) for value in scaffold_overlap[:20]
            ],
        },
        "morgan_similarity": {
            "status": "FULL" if full else "PARTIAL_DETERMINISTIC_SAMPLE",
            "full_cartesian_comparison_count": len(left) * len(right),
            "executed_comparison_count": comparisons,
            "sampled_unique_parent_counts": {
                left_name: left_count,
                right_name: right_count,
            },
            "threshold": float(config.similarity_threshold),
            "maximum": maximum_similarity,
            "maximum_nonidentical": maximum_nonidentical_similarity,
            "at_or_above_threshold_pair_count": at_or_above_threshold,
            "nonidentical_at_or_above_threshold_pair_count": (
                nonidentical_at_or_above_threshold
            ),
            "semantics": (
                "Counts are exact only when status=FULL. A partial deterministic sample "
                "can reveal leakage but cannot establish its absence."
            ),
        },
    }


def _precondition(status: str, blockers: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "blockers": blockers,
    }


def _gate(
    splits: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    parse_blockers = [
        f"{name}:invalid_records={summary['invalid_record_count']}"
        for name, summary in splits.items()
        if summary["invalid_record_count"]
    ]
    parse_blockers.extend(
        f"{name}:no_valid_records"
        for name, summary in splits.items()
        if summary["parsed_record_count"] == 0
    )
    identity_blockers = [
        (
            f"{pair['left_split']}<->{pair['right_split']}:"
            f"parent_identity_overlap="
            f"{pair['exact_parent_identity_overlap']['unique_count']}"
        )
        for pair in pairs
        if pair["exact_parent_identity_overlap"]["unique_count"]
    ]
    partial_pairs = [
        f"{pair['left_split']}<->{pair['right_split']}"
        for pair in pairs
        if pair["morgan_similarity"]["status"] != "FULL"
    ]
    analogue_blockers = [
        (
            f"{pair['left_split']}<->{pair['right_split']}:"
            f"nonidentical_pairs_at_or_above_threshold="
            f"{pair['morgan_similarity']['nonidentical_at_or_above_threshold_pair_count']}"
        )
        for pair in pairs
        if pair["morgan_similarity"]["nonidentical_at_or_above_threshold_pair_count"]
    ]
    scaffold_blockers = [
        (
            f"{pair['left_split']}<->{pair['right_split']}:"
            f"scaffold_overlap={pair['scaffold_overlap']['unique_count']}"
        )
        for pair in pairs
        if pair["scaffold_overlap"]["unique_count"]
    ]
    within_split_duplicate_blockers = [
        f"{name}:duplicate_parent_records={summary['duplicate_parent_record_count']}"
        for name, summary in splits.items()
        if summary["duplicate_parent_record_count"]
    ]
    parsing = _precondition("PASS" if not parse_blockers else "FAIL", parse_blockers)
    within_split_uniqueness = _precondition(
        (
            "PASS"
            if not parse_blockers and not within_split_duplicate_blockers
            else "FAIL"
        ),
        [*parse_blockers, *within_split_duplicate_blockers],
    )
    identity = _precondition(
        "PASS" if not parse_blockers and not identity_blockers else "FAIL",
        [*parse_blockers, *identity_blockers],
    )
    if parse_blockers or analogue_blockers:
        analogue_status = "FAIL"
    elif partial_pairs:
        analogue_status = "INCOMPLETE"
    else:
        analogue_status = "PASS"
    analogue = _precondition(
        analogue_status,
        [
            *parse_blockers,
            *analogue_blockers,
            *[f"{value}:similarity_audit_partial" for value in partial_pairs],
        ],
    )
    scaffold = _precondition(
        "PASS" if not parse_blockers and not scaffold_blockers else "FAIL",
        [*parse_blockers, *scaffold_blockers],
    )
    broad_components = {
        "parsing_complete": parsing,
        "within_split_identity_uniqueness": within_split_uniqueness,
        "identity_novelty": identity,
        "analogue_novelty": analogue,
        "scaffold_novelty": scaffold,
    }
    broad_blockers = [
        f"{name}:{blocker}"
        for name, value in broad_components.items()
        for blocker in value["blockers"]
    ]
    if all(value["status"] == "PASS" for value in broad_components.values()):
        broad_status = "PASS"
    elif any(value["status"] == "FAIL" for value in broad_components.values()):
        broad_status = "FAIL"
    else:
        broad_status = "INCOMPLETE"
    return {
        **broad_components,
        "broad_generalisation_precondition": _precondition(
            broad_status,
            broad_blockers,
        ),
        "scientific_semantics": (
            "PASS is only a dataset-integrity precondition. It does not demonstrate "
            "model accuracy, enrichment, affinity, biological activity, or external "
            "generalisation."
        ),
    }


def build_dataset_leakage_audit(
    split_paths: dict[str, Path],
    *,
    dataset_name: str,
    dataset_version: str,
    dataset_license: str,
    dataset_source: str,
    config: DatasetAuditConfig | None = None,
) -> dict[str, Any]:
    """Audit molecular split leakage and return a self-hashed receipt."""

    if len(split_paths) < 2:
        raise ValueError("dataset audit requires at least two named splits")
    if len(split_paths) > 8:
        raise ValueError("dataset audit supports at most eight named splits")
    if any(not value.strip() for value in split_paths):
        raise ValueError("dataset split names must be non-empty")
    for field_name, value in (
        ("dataset_name", dataset_name),
        ("dataset_version", dataset_version),
        ("dataset_license", dataset_license),
        ("dataset_source", dataset_source),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    _validate_dataset_source(dataset_source)
    config = config or DatasetAuditConfig()
    _, _, rd_base, _ = _rdkit()
    split_summaries: dict[str, dict[str, Any]] = {}
    split_identities: dict[str, dict[str, _MoleculeIdentity]] = {}
    for name in sorted(split_paths):
        summary, identities = _audit_split(name, split_paths[name])
        split_summaries[name] = summary
        split_identities[name] = identities
    pairwise = [
        _pair_audit(
            left_name,
            split_identities[left_name],
            right_name,
            split_identities[right_name],
            config=config,
        )
        for left_name, right_name in combinations(sorted(split_paths), 2)
    ]
    core = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "dataset": {
            "name": dataset_name,
            "version": dataset_version,
            "license": dataset_license,
            "source": dataset_source,
        },
        "config": config.to_dict(),
        "implementation": {
            "module": "protbind_agent.dataset_audit",
            "source_sha256": sha256_file(Path(__file__)),
            "rdkit_version": rd_base.rdkitVersion,
        },
        "splits": split_summaries,
        "pairwise": pairwise,
        "gate": _gate(split_summaries, pairwise),
        "privacy": {
            "raw_smiles_in_receipt": False,
            "record_ids_in_receipt": False,
            "overlap_examples": "SHA-256 commitments of canonical identities/scaffolds",
        },
        "scientific_boundaries": [
            "This receipt audits molecular split integrity; it does not measure docking "
            "or screening performance.",
            "Identity standardization uses RDKit FragmentParent, ChargeParent, and "
            "TautomerParent and may not match every dataset's published identity policy.",
            "Scaffold disjointness is a strict precondition for scaffold-novelty claims, "
            "not a universal requirement for every assay question.",
            "PARTIAL_DETERMINISTIC_SAMPLE similarity results can find leakage but cannot "
            "establish the absence of analog leakage.",
            "Protein sequence, pocket, temporal, assay, and label leakage require separate "
            "audits and are NOT_EVALUATED here.",
        ],
    }
    return {
        **core,
        "audit_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def verify_dataset_leakage_audit(result: dict[str, Any]) -> None:
    """Verify the schema, kind, and content commitment of one audit receipt."""

    if not isinstance(result, dict):
        raise DatasetAuditIntegrityError("dataset audit result must be an object")
    if result.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise DatasetAuditIntegrityError(
            f"dataset audit schema_version must be {AUDIT_SCHEMA_VERSION}"
        )
    if result.get("kind") != AUDIT_KIND:
        raise DatasetAuditIntegrityError(f"dataset audit kind must be {AUDIT_KIND}")
    core = {key: value for key, value in result.items() if key != "audit_sha256"}
    if result.get("audit_sha256") != sha256_bytes(canonical_json_bytes(core)):
        raise DatasetAuditIntegrityError("dataset audit receipt hash mismatch")


def load_dataset_leakage_audit(path: Path) -> dict[str, Any]:
    """Load and self-verify one persisted audit receipt."""

    result = json.loads(path.read_text(encoding="utf-8"))
    verify_dataset_leakage_audit(result)
    return result


def persist_dataset_leakage_audit(result: dict[str, Any], output: Path) -> None:
    """Atomically persist one verified audit receipt."""

    verify_dataset_leakage_audit(result)
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
