"""Private, content-addressed protein and ligand libraries.

The command-line interface is the operator surface and may be given explicit
paths.  Agent integrations must only expose configured library aliases and the
bounded ``incoming`` directories; this module never grants filesystem access.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes, sha256_file
from .models import ArtifactRef
from .structure import (
    StructureCapabilityError,
    inspect_declared_connections,
    inspect_structure,
)

LIBRARY_SCHEMA_VERSION = "1.0"
PLAN_SCHEMA_VERSION = "1.0"
_KINDS = ("protein", "ligand")
_PROTEIN_SUFFIXES = frozenset({".pdb", ".cif", ".mmcif", ".fa", ".faa", ".fasta"})
_LIGAND_SUFFIXES = frozenset({".sdf", ".mol", ".smi", ".smiles"})
_SAFE_ACCESSION = re.compile(r"^[A-Z0-9][A-Z0-9-]{5,15}$")
_FASTA_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+\*?$")


class ImportState(StrEnum):
    DISCOVERED = "DISCOVERED"
    STAGED = "STAGED"
    PARSED = "PARSED"
    QC_PASSED = "QC_PASSED"
    QUARANTINED = "QUARANTINED"
    IDENTITY_CHECKED = "IDENTITY_CHECKED"
    ACTIVE = "ACTIVE"


class VerificationState(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    EXACT_SEQUENCE = "EXACT_SEQUENCE"
    CONSISTENT_VARIANT = "CONSISTENT_VARIANT"
    PARTIAL_COORDINATE_MATCH = "PARTIAL_COORDINATE_MATCH"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class LibraryConfig:
    protein_root: Path
    ligand_root: Path

    def root(self, kind: str) -> Path:
        _validate_kind(kind)
        return self.protein_root if kind == "protein" else self.ligand_root

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIBRARY_SCHEMA_VERSION,
            "libraries": {
                kind: {
                    "root_id": _root_id(self.root(kind)),
                    "alias": f"{kind}-library",
                }
                for kind in _KINDS
            },
            "absolute_paths_disclosed": False,
        }


def save_library_config(
    path: Path,
    *,
    protein_root: Path,
    ligand_root: Path,
    replace: bool = False,
) -> LibraryConfig:
    """Create two independent library roots and a private operator config."""

    destination = path.resolve()
    if destination.exists() and not replace:
        raise FileExistsError(f"library config already exists: {destination.name}")
    config = LibraryConfig(
        protein_root=protein_root.resolve(),
        ligand_root=ligand_root.resolve(),
    )
    if config.protein_root == config.ligand_root:
        raise ValueError("protein and ligand library roots must be different")
    for kind in _KINDS:
        _initialize_root(config.root(kind), kind)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    value = {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "protein_root": str(config.protein_root),
        "ligand_root": str(config.ligand_root),
    }
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    return config


def load_library_config(path: Path) -> LibraryConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != LIBRARY_SCHEMA_VERSION:
        raise ValueError("unsupported library config schema")
    config = LibraryConfig(
        protein_root=Path(value["protein_root"]).resolve(),
        ligand_root=Path(value["ligand_root"]).resolve(),
    )
    if config.protein_root == config.ligand_root:
        raise ValueError("protein and ligand library roots must be different")
    return config


class LibraryManager:
    """Operate configured libraries without returning absolute paths."""

    def __init__(self, config: LibraryConfig) -> None:
        self.config = config

    def initialize(self) -> dict[str, Any]:
        for kind in _KINDS:
            _initialize_root(self.config.root(kind), kind)
        return self.status()

    def status(self) -> dict[str, Any]:
        libraries: dict[str, Any] = {}
        for kind in _KINDS:
            root = self.config.root(kind)
            catalog = root / "catalog.sqlite"
            counts: dict[str, int] = {}
            if catalog.is_file():
                with _connect(root) as connection:
                    rows = connection.execute(
                        "SELECT state, COUNT(*) AS count FROM entries GROUP BY state"
                    ).fetchall()
                    counts = {str(row["state"]): int(row["count"]) for row in rows}
            libraries[kind] = {
                "alias": f"{kind}-library",
                "root_id": _root_id(root),
                "initialized": catalog.is_file(),
                "entry_count": sum(counts.values()),
                "state_counts": counts,
                "incoming_file_count": _bounded_incoming_count(root / "incoming"),
            }
        return {
            "schema_version": LIBRARY_SCHEMA_VERSION,
            "libraries": libraries,
            "absolute_paths_disclosed": False,
        }

    def scan(
        self,
        kind: str,
        source: Path,
        *,
        recursive: bool = False,
        max_files: int = 10_000,
        max_file_bytes: int = 64 * 1024 * 1024,
        save_receipt: bool = True,
    ) -> dict[str, Any]:
        """Hash a bounded source selection and freeze an immutable import plan."""

        _validate_kind(kind)
        if max_files < 1 or max_files > 100_000:
            raise ValueError("max_files must be between 1 and 100000")
        if max_file_bytes < 1 or max_file_bytes > 1024 * 1024 * 1024:
            raise ValueError("max_file_bytes must be between 1 byte and 1 GiB")
        if source.is_symlink():
            raise ValueError("symbolic-link import sources are not supported")
        selected = source.resolve()
        if selected.is_file():
            source_root = selected.parent
            candidates = [selected]
        elif selected.is_dir():
            source_root = selected
            iterator = selected.rglob("*") if recursive else selected.glob("*")
            candidates = sorted(
                (path for path in iterator if path.is_file()),
                key=lambda path: path.relative_to(source_root).as_posix(),
            )
        else:
            raise FileNotFoundError(f"import source does not exist: {selected.name}")
        if len(candidates) > max_files:
            raise ValueError(f"source selection exceeds max_files={max_files}")
        suffixes = _PROTEIN_SUFFIXES if kind == "protein" else _LIGAND_SUFFIXES
        files: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for path in candidates:
            relative = path.relative_to(source_root)
            parent_symlink = any(
                parent.is_symlink()
                for parent in path.parents
                if parent != source_root
            )
            if path.is_symlink() or parent_symlink:
                skipped.append({"name": relative.name, "reason": "symbolic_link"})
                continue
            if path.suffix.lower() not in suffixes:
                skipped.append({"name": relative.name, "reason": "unsupported_suffix"})
                continue
            size = path.stat().st_size
            if size > max_file_bytes:
                skipped.append({"name": relative.name, "reason": "file_too_large"})
                continue
            files.append(
                {
                    "relative_path": relative.as_posix(),
                    "filename": path.name,
                    "size_bytes": size,
                    "sha256": sha256_file(path),
                    "format": path.suffix.lower().lstrip("."),
                }
            )
        commitment = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "kind": kind,
            "library_root_id": _root_id(self.config.root(kind)),
            "source_root": str(source_root),
            "recursive": recursive,
            "files": files,
            "skipped": skipped,
        }
        plan_id = sha256_bytes(canonical_json_bytes(commitment))
        plan = {
            **commitment,
            "plan_id": plan_id,
            "created_at": _now(),
            "semantics": (
                "A hash-bound import proposal only. Files are not trusted, imported, "
                "or scientifically verified until apply/QC completes."
            ),
        }
        if save_receipt:
            receipt_path = self._plan_path(kind, plan_id)
            if receipt_path.is_file():
                existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                if all(existing.get(key) == value for key, value in commitment.items()):
                    return {**existing, "idempotent_replay": True}
                raise ValueError("existing import plan does not match its commitment")
            _write_once(receipt_path, canonical_json_bytes(plan) + b"\n")
        return plan

    def scan_incoming(self, kind: str, **kwargs: Any) -> dict[str, Any]:
        return self.scan(kind, self.config.root(kind) / "incoming", **kwargs)

    def load_plan(self, kind: str, plan_id: str) -> dict[str, Any]:
        _validate_kind(kind)
        _validate_digest(plan_id, "plan_id")
        plan = json.loads(self._plan_path(kind, plan_id).read_text(encoding="utf-8"))
        if (
            plan.get("plan_id") != plan_id
            or plan.get("kind") != kind
            or _committed_plan_id(plan) != plan_id
        ):
            raise ValueError("import plan identity mismatch")
        return plan

    def apply(
        self,
        plan: dict[str, Any],
        *,
        mode: str = "copy",
        confirm_move: str | None = None,
    ) -> dict[str, Any]:
        """Apply a frozen plan. Move deletes sources only after verified CAS import."""

        kind = str(plan.get("kind", ""))
        _validate_kind(kind)
        if mode not in {"copy", "move"}:
            raise ValueError("mode must be copy or move")
        plan_id = str(plan.get("plan_id", ""))
        _validate_digest(plan_id, "plan_id")
        if _committed_plan_id(plan) != plan_id:
            raise ValueError("import plan content does not match its plan_id")
        if plan.get("library_root_id") != _root_id(self.config.root(kind)):
            raise ValueError("plan targets a different library root")
        if mode == "move" and confirm_move != plan_id:
            raise PermissionError(
                "move requires an exact second confirmation: --confirm-move PLAN_ID"
            )
        receipt_path = (
            self.config.root(kind) / "receipts" / f"{plan_id}.apply.json"
        )
        if receipt_path.is_file():
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            if previous.get("mode") != mode:
                raise ValueError("plan was already applied using a different mode")
            return {**previous, "idempotent_replay": True}
        source_root = Path(str(plan["source_root"])).resolve()
        results: list[dict[str, Any]] = []
        for item in plan.get("files", []):
            path = _resolve_planned_source(source_root, item)
            if sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
                raise ValueError(f"planned source changed before import: {path.name}")
            result = self._import_one(kind, path, item)
            if mode == "move" and result["state"] != ImportState.QUARANTINED:
                stored = ArtifactRef.from_dict(result["raw_artifact"])
                ArtifactStore(self.config.root(kind)).resolve(stored)
                path.unlink()
                result["source_removed_after_verified_copy"] = True
            else:
                result["source_removed_after_verified_copy"] = False
            result["source_preserved_due_to_quarantine"] = (
                mode == "move" and result["state"] == ImportState.QUARANTINED
            )
            results.append(result)
        receipt = {
            "schema_version": LIBRARY_SCHEMA_VERSION,
            "plan_id": plan_id,
            "kind": kind,
            "mode": mode,
            "completed_at": _now(),
            "imported_count": sum(not result["deduplicated"] for result in results),
            "deduplicated_count": sum(result["deduplicated"] for result in results),
            "quarantined_count": sum(
                result["state"] == ImportState.QUARANTINED for result in results
            ),
            "results": results,
            "absolute_paths_disclosed": False,
        }
        _write_once(receipt_path, canonical_json_bytes(receipt) + b"\n")
        return receipt

    def apply_saved(
        self,
        kind: str,
        plan_id: str,
        *,
        mode: str = "copy",
        confirm_move: str | None = None,
    ) -> dict[str, Any]:
        return self.apply(
            self.load_plan(kind, plan_id),
            mode=mode,
            confirm_move=confirm_move,
        )

    def list_entries(
        self,
        kind: str,
        *,
        state: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        _validate_kind(kind)
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = (
            "SELECT entry_id, filename, state, verification_state, raw_sha256, "
            "created_at FROM entries"
        )
        parameters: tuple[Any, ...] = ()
        if state is not None:
            if state not in {item.value for item in ImportState}:
                raise ValueError("invalid import state")
            query += " WHERE state = ?"
            parameters = (state,)
        query += " ORDER BY created_at, entry_id LIMIT ?"
        parameters += (limit,)
        with _connect(self.config.root(kind)) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return {
            "kind": kind,
            "entries": [
                {
                    "entry_id": row["entry_id"],
                    "filename": row["filename"],
                    "state": row["state"],
                    "verification_state": row["verification_state"],
                    "artifact_id": f"sha256:{row['raw_sha256']}",
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
            "limit": limit,
            "absolute_paths_disclosed": False,
        }

    def show_entry(self, kind: str, entry_id: str) -> dict[str, Any]:
        row = self._entry(kind, entry_id)
        return _public_entry(row)

    def rag_projection(
        self,
        kind: str = "protein",
        *,
        include_quarantined: bool = False,
        limit: int = 10_000,
    ) -> dict[str, Any]:
        """Build a path/sequence/coordinate-free projection for a derived RAG index."""

        _validate_kind(kind)
        if limit < 1 or limit > 100_000:
            raise ValueError("limit must be between 1 and 100000")
        query = "SELECT * FROM entries"
        parameters: tuple[Any, ...] = ()
        if not include_quarantined:
            query += " WHERE state != ?"
            parameters = (ImportState.QUARANTINED.value,)
        query += " ORDER BY created_at, entry_id LIMIT ?"
        parameters += (limit,)
        with _connect(self.config.root(kind)) as connection:
            rows = connection.execute(query, parameters).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            qc = json.loads(row["qc_json"])
            verification = json.loads(row["verification_json"])
            record: dict[str, Any] = {
                "entry_id": row["entry_id"],
                "kind": row["kind"],
                "state": row["state"],
                "verification_state": row["verification_state"],
                "accession": verification.get("accession"),
                "format": qc.get("format"),
                "parse_valid": bool(qc.get("parse_valid", False)),
                "workflow_v1_compatible": qc.get("workflow_v1_compatible"),
                "workflow_v1_blockers": list(qc.get("workflow_v1_blockers", [])),
            }
            if kind == "protein":
                record.update(
                    {
                        "coordinate_model_present": qc.get(
                            "coordinate_model_present",
                            qc.get("format") in {"pdb", "mmcif"},
                        ),
                        "chain_count": qc.get("chain_count", qc.get("sequence_count")),
                        "residue_count": qc.get(
                            "residue_count",
                            sum(qc.get("sequence_lengths", [])),
                        ),
                        "sequence_lengths": list(qc.get("sequence_lengths", [])),
                        "metal_element_count": len(qc.get("metal_elements", [])),
                        "nonstandard_residue_type_count": len(
                            qc.get("nonstandard_residues", [])
                        ),
                        "missing_backbone_residue_count": len(
                            qc.get("missing_backbone_residues", [])
                        ),
                    }
                )
            else:
                record.update(
                    {
                        "record_count": qc.get("record_count"),
                        "maximum_heavy_atom_count": qc.get(
                            "maximum_heavy_atom_count"
                        ),
                    }
                )
            records.append(record)
        return {
            "schema_version": LIBRARY_SCHEMA_VERSION,
            "kind": kind,
            "library_root_id": _root_id(self.config.root(kind)),
            "records": records,
            "record_count": len(records),
            "projection_policy": (
                "Derived retrieval projection only; catalog.sqlite remains authoritative. "
                "No filenames, paths, sequences, SMILES, molecule bytes, or coordinates."
            ),
            "absolute_paths_disclosed": False,
            "private_values_disclosed": False,
        }

    def protein_sequences(self, entry_id: str) -> tuple[str, ...]:
        row = self._entry("protein", entry_id)
        qc = json.loads(row["qc_json"])
        return tuple(str(value) for value in qc.get("sequences", []))

    def verify_uniprot_bytes(
        self,
        entry_id: str,
        accession: str,
        fasta_data: bytes,
        *,
        source_artifact: ArtifactRef | None = None,
    ) -> dict[str, Any]:
        """Compare local observed sequences with an accession-bound UniProt FASTA."""

        accession = accession.strip().upper()
        if _SAFE_ACCESSION.fullmatch(accession) is None:
            raise ValueError("invalid UniProt accession")
        reference = _parse_fasta(fasta_data)
        row = self._entry("protein", entry_id)
        qc = json.loads(row["qc_json"])
        observed = tuple(str(value) for value in qc.get("sequences", []))
        if not observed:
            raise ValueError("protein entry has no sequence available for verification")
        comparisons = [_compare_sequences(sequence, reference) for sequence in observed]
        best_index, best = max(
            enumerate(comparisons),
            key=lambda item: (
                item[1]["exact"],
                item[1]["identity"],
                item[1]["reference_coverage"],
            ),
        )
        if best["exact"]:
            state = VerificationState.EXACT_SEQUENCE
        elif best["observed_is_reference_subsequence"]:
            state = VerificationState.PARTIAL_COORDINATE_MATCH
        elif best["identity"] >= 0.98 and best["reference_coverage"] >= 0.9:
            state = VerificationState.CONSISTENT_VARIANT
        else:
            state = VerificationState.CONFLICT
        verification = {
            "state": state.value,
            "accession": accession,
            "best_observed_sequence_index": best_index,
            "comparison": best,
            "all_comparisons": comparisons,
            "reference_artifact": (
                source_artifact.to_dict() if source_artifact is not None else None
            ),
            "method": "deterministic global alignment; accession-only network lookup",
            "private_sequence_uploaded": False,
            "semantics": (
                "Identity evidence for the observed sequence only. It does not prove "
                "the coordinate model, biological assembly, ligand state, or activity."
            ),
            "checked_at": _now(),
        }
        root = self.config.root("protein")
        raw_artifact = ArtifactRef.from_dict(json.loads(row["raw_artifact_json"]))
        receipt = ArtifactStore(root).put_json(
            verification,
            producer="protbind.library.uniprot-verification",
            producer_version=__version__,
            source=raw_artifact.artifact_id,
        )
        stored_verification = {**verification, "receipt_artifact": receipt.to_dict()}
        with _connect(root) as connection:
            connection.execute(
                "UPDATE entries SET verification_state = ?, verification_json = ?, "
                "state = ? WHERE entry_id = ?",
                (
                    state.value,
                    json.dumps(stored_verification, sort_keys=True),
                    (
                        ImportState.IDENTITY_CHECKED.value
                        if state == VerificationState.CONFLICT
                        else ImportState.ACTIVE.value
                    ),
                    entry_id,
                ),
            )
        return {
            "entry_id": entry_id,
            **stored_verification,
            "absolute_paths_disclosed": False,
        }

    def _entry(self, kind: str, entry_id: str) -> sqlite3.Row:
        _validate_kind(kind)
        with _connect(self.config.root(kind)) as connection:
            row = connection.execute(
                "SELECT * FROM entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown {kind} library entry")
        _verify_catalog_artifacts(self.config.root(kind), row)
        return row

    def _plan_path(self, kind: str, plan_id: str) -> Path:
        return self.config.root(kind) / "receipts" / f"{plan_id}.plan.json"

    def _import_one(
        self,
        kind: str,
        path: Path,
        planned: dict[str, Any],
    ) -> dict[str, Any]:
        root = self.config.root(kind)
        _initialize_root(root, kind)
        with _connect(root) as connection:
            existing = connection.execute(
                "SELECT * FROM entries WHERE raw_sha256 = ?", (planned["sha256"],)
            ).fetchone()
        if existing is not None:
            _verify_catalog_artifacts(root, existing)
            return {
                "entry_id": existing["entry_id"],
                "filename": existing["filename"],
                "state": existing["state"],
                "verification_state": existing["verification_state"],
                "raw_artifact": json.loads(existing["raw_artifact_json"]),
                "deduplicated": True,
                "state_history": [existing["state"]],
            }
        media_type = _media_type(kind, path)
        raw = ArtifactStore(root).import_file(
            path,
            media_type=media_type,
            producer="protbind.library.import",
            producer_version=__version__,
            source=f"local-import:{path.name}",
        )
        try:
            qc, derived = _qc_file(kind, path, ArtifactStore(root))
            state = ImportState.ACTIVE
            verification = VerificationState.UNVERIFIED
        except (ImportError, OSError, RuntimeError, ValueError, StructureCapabilityError) as exc:
            qc = {
                "parse_valid": False,
                "status": "QUARANTINED",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            derived = None
            state = ImportState.QUARANTINED
            verification = VerificationState.UNVERIFIED
        entry_id = f"{kind}-{raw.sha256[:20]}"
        created_at = _now()
        with _connect(root) as connection:
            connection.execute(
                """
                INSERT INTO entries (
                    entry_id, kind, filename, state, verification_state,
                    raw_sha256, raw_artifact_json, derived_artifact_json,
                    source_kind, source_name, qc_json, verification_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    kind,
                    path.name,
                    state.value,
                    verification.value,
                    raw.sha256,
                    json.dumps(raw.to_dict(), sort_keys=True),
                    json.dumps(derived.to_dict(), sort_keys=True) if derived else None,
                    "local-import",
                    planned["relative_path"],
                    json.dumps(qc, sort_keys=True),
                    json.dumps(
                        {
                            "state": VerificationState.UNVERIFIED.value,
                            "reason": "No identifier-bound identity check has been completed.",
                        },
                        sort_keys=True,
                    ),
                    created_at,
                ),
            )
        if state == ImportState.QUARANTINED:
            _write_once(
                root / "quarantine" / f"{entry_id}.json",
                canonical_json_bytes(
                    {
                        "schema_version": LIBRARY_SCHEMA_VERSION,
                        "entry_id": entry_id,
                        "raw_artifact": raw.to_dict(),
                        "qc": _public_qc(qc),
                        "source_bytes_preserved_in_cas": True,
                        "scientific_identity_inferred": False,
                    }
                )
                + b"\n",
            )
        return {
            "entry_id": entry_id,
            "filename": path.name,
            "state": state.value,
            "verification_state": verification.value,
            "raw_artifact": raw.to_dict(),
            "derived_artifact": derived.to_dict() if derived else None,
            "qc": _public_qc(qc),
            "deduplicated": False,
            "state_history": (
                [
                    ImportState.DISCOVERED.value,
                    ImportState.STAGED.value,
                    ImportState.PARSED.value,
                    ImportState.QC_PASSED.value,
                    ImportState.ACTIVE.value,
                ]
                if state == ImportState.ACTIVE
                else [
                    ImportState.DISCOVERED.value,
                    ImportState.STAGED.value,
                    ImportState.QUARANTINED.value,
                ]
            ),
        }


def _initialize_root(root: Path, kind: str) -> None:
    _validate_kind(kind)
    existed = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    if not existed:
        os.chmod(root, 0o700)
    for name in ("objects", "incoming", "quarantine", "derived", "receipts"):
        directory = root / name
        directory.mkdir(exist_ok=True)
        os.chmod(directory, 0o700)
    with _connect(root) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entries (
                entry_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                filename TEXT NOT NULL,
                state TEXT NOT NULL,
                verification_state TEXT NOT NULL,
                raw_sha256 TEXT NOT NULL UNIQUE,
                raw_artifact_json TEXT NOT NULL,
                derived_artifact_json TEXT,
                source_kind TEXT NOT NULL,
                source_name TEXT NOT NULL,
                qc_json TEXT NOT NULL,
                verification_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS entries_state_idx ON entries(state);
            """
        )
        existing = connection.execute(
            "SELECT value FROM metadata WHERE key = 'kind'"
        ).fetchone()
        if existing is not None and existing["value"] != kind:
            raise ValueError("library root was initialized for a different data kind")
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
            (LIBRARY_SCHEMA_VERSION,),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('kind', ?)", (kind,)
        )
    os.chmod(root / "catalog.sqlite", 0o600)


@contextmanager
def _connect(root: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(root / "catalog.sqlite")
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _qc_file(
    kind: str,
    path: Path,
    store: ArtifactStore,
) -> tuple[dict[str, Any], ArtifactRef | None]:
    if kind == "protein":
        if path.suffix.lower() in {".fa", ".faa", ".fasta"}:
            sequence = _parse_fasta(path.read_bytes())
            return (
                {
                    "parse_valid": True,
                    "format": "fasta",
                    "sequence_count": 1,
                    "sequences": [sequence],
                    "sequence_lengths": [len(sequence)],
                    "coordinate_model_present": False,
                    "workflow_v1_compatible": len(sequence) <= 700,
                },
                None,
            )
        inspection = inspect_structure(path, max_chains=None, max_residues=None)
        connections = inspect_declared_connections(path)
        qc = {
            "parse_valid": True,
            "format": "mmcif" if path.suffix.lower() in {".cif", ".mmcif"} else "pdb",
            "chain_count": inspection.chain_count,
            "residue_count": inspection.residue_count,
            "chain_ids": list(inspection.chain_ids),
            "sequences": list(inspection.sequences),
            "sequence_lengths": [len(sequence) for sequence in inspection.sequences],
            "metal_elements": list(inspection.metal_elements),
            "nonstandard_residues": list(inspection.nonstandard_residues),
            "alternate_location_atoms": inspection.alternate_location_atoms,
            "missing_backbone_residues": list(inspection.missing_backbone_residues),
            "declared_connections": connections.to_dict(),
            "workflow_v1_compatible": (
                inspection.chain_count <= 2 and inspection.residue_count <= 700
            ),
            "workflow_v1_blockers": [
                reason
                for condition, reason in (
                    (inspection.chain_count > 2, "more_than_two_protein_chains"),
                    (inspection.residue_count > 700, "more_than_700_residues"),
                    (bool(inspection.metal_elements), "metal_present"),
                    (connections.covalent_detected, "declared_covalent_connection"),
                )
                if condition
            ],
        }
        return qc, None
    return _qc_ligand(path, store)


def _qc_ligand(
    path: Path,
    store: ArtifactStore,
) -> tuple[dict[str, Any], ArtifactRef]:
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:
        raise RuntimeError("RDKit is required for ligand library QC") from exc
    suffix = path.suffix.lower()
    molecules: list[Any] = []
    if suffix == ".sdf":
        molecules = [molecule for molecule in Chem.SDMolSupplier(str(path)) if molecule]
    elif suffix == ".mol":
        molecule = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
        molecules = [molecule] if molecule is not None else []
    else:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for line in lines:
            token = line.split()[0]
            molecule = Chem.MolFromSmiles(token)
            if molecule is not None:
                molecules.append(molecule)
    if not molecules:
        raise ValueError("RDKit parsed no valid molecules")
    standardized: list[dict[str, Any]] = []
    for index, molecule in enumerate(molecules):
        cleaned = rdMolStandardize.Cleanup(molecule)
        parent = rdMolStandardize.FragmentParent(cleaned)
        standardized.append(
            {
                "record_index": index,
                "canonical_isomeric_smiles": Chem.MolToSmiles(
                    parent, canonical=True, isomericSmiles=True
                ),
                "heavy_atom_count": parent.GetNumHeavyAtoms(),
                "formal_charge": Chem.GetFormalCharge(parent),
                "chiral_center_count": len(
                    Chem.FindMolChiralCenters(parent, includeUnassigned=True)
                ),
            }
        )
    derivative = store.put_json(
        {
            "schema_version": LIBRARY_SCHEMA_VERSION,
            "semantics": (
                "RDKit normalized parent representations derived from an immutable raw "
                "file; these do not overwrite or authenticate the source records."
            ),
            "records": standardized,
        },
        producer="protbind.library.rdkit-standardize",
        producer_version=rdBase.rdkitVersion,
        source=f"derived-from:{sha256_file(path)}",
    )
    return (
        {
            "parse_valid": True,
            "format": suffix.lstrip("."),
            "record_count": len(molecules),
            "standardized_parent_count": len(standardized),
            "maximum_heavy_atom_count": max(
                record["heavy_atom_count"] for record in standardized
            ),
            "workflow_v1_compatible": all(
                record["heavy_atom_count"] <= 100 for record in standardized
            ),
        },
        derivative,
    )


def _parse_fasta(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("FASTA is not UTF-8 text") from exc
    records = sum(
        line.lstrip().startswith(">")
        for line in text.splitlines()
        if line.strip()
    )
    if records > 1:
        raise ValueError("FASTA library entries must contain exactly one sequence")
    sequence = "".join(
        line.strip().upper()
        for line in text.splitlines()
        if line.strip() and not line.startswith(">")
    )
    if not sequence or _FASTA_SEQUENCE.fullmatch(sequence) is None:
        raise ValueError("FASTA must contain one canonical amino-acid sequence")
    return sequence.removesuffix("*")


def _compare_sequences(observed: str, reference: str) -> dict[str, Any]:
    if len(observed) > 2000 or len(reference) > 2000:
        raise ValueError(
            "sequence identity verification supports at most 2000 residues per sequence"
        )
    if observed == reference:
        return {
            "exact": True,
            "observed_is_reference_subsequence": True,
            "identity": 1.0,
            "reference_coverage": 1.0,
            "observed_coverage": 1.0,
            "matches": len(reference),
            "aligned_columns": len(reference),
            "observed_length": len(observed),
            "reference_length": len(reference),
        }
    rows, columns = len(observed) + 1, len(reference) + 1
    score = [[0] * columns for _ in range(rows)]
    trace = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        score[row][0] = -2 * row
        trace[row][0] = "U"
    for column in range(1, columns):
        score[0][column] = -2 * column
        trace[0][column] = "L"
    for row in range(1, rows):
        for column in range(1, columns):
            choices = (
                (
                    score[row - 1][column - 1]
                    + (2 if observed[row - 1] == reference[column - 1] else -1),
                    "D",
                ),
                (score[row - 1][column] - 2, "U"),
                (score[row][column - 1] - 2, "L"),
            )
            score[row][column], trace[row][column] = max(
                choices, key=lambda value: (value[0], value[1] == "D", value[1] == "U")
            )
    row, column = len(observed), len(reference)
    matches = 0
    aligned = 0
    observed_aligned = 0
    reference_aligned = 0
    while row or column:
        direction = trace[row][column]
        aligned += 1
        if direction == "D":
            observed_aligned += 1
            reference_aligned += 1
            matches += observed[row - 1] == reference[column - 1]
            row -= 1
            column -= 1
        elif direction == "U":
            observed_aligned += 1
            row -= 1
        else:
            reference_aligned += 1
            column -= 1
    return {
        "exact": False,
        "observed_is_reference_subsequence": observed in reference,
        "identity": round(matches / max(1, aligned), 6),
        "reference_coverage": round(matches / max(1, len(reference)), 6),
        "observed_coverage": round(matches / max(1, len(observed)), 6),
        "matches": matches,
        "aligned_columns": aligned,
        "observed_length": len(observed),
        "reference_length": len(reference),
    }


def _public_entry(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "entry_id": row["entry_id"],
        "kind": row["kind"],
        "filename": row["filename"],
        "state": row["state"],
        "verification_state": row["verification_state"],
        "raw_artifact": json.loads(row["raw_artifact_json"]),
        "derived_artifact": (
            json.loads(row["derived_artifact_json"])
            if row["derived_artifact_json"]
            else None
        ),
        "source_kind": row["source_kind"],
        "source_name": Path(row["source_name"]).name,
        "qc": _public_qc(json.loads(row["qc_json"])),
        "verification": json.loads(row["verification_json"]),
        "created_at": row["created_at"],
        "absolute_paths_disclosed": False,
    }


def _verify_catalog_artifacts(root: Path, row: sqlite3.Row) -> None:
    store = ArtifactStore(root)
    store.resolve(ArtifactRef.from_dict(json.loads(row["raw_artifact_json"])))
    if row["derived_artifact_json"]:
        store.resolve(
            ArtifactRef.from_dict(json.loads(row["derived_artifact_json"]))
        )


def _public_qc(qc: dict[str, Any]) -> dict[str, Any]:
    """Remove private sequence values from CLI/MCP output and receipts."""

    value = {key: item for key, item in qc.items() if key != "sequences"}
    if "sequences" in qc:
        value["sequence_values_disclosed"] = False
    return value


def _resolve_planned_source(source_root: Path, item: dict[str, Any]) -> Path:
    relative = Path(str(item["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("planned source path is unsafe")
    path = (source_root / relative).resolve()
    if not path.is_relative_to(source_root) or not path.is_file() or path.is_symlink():
        raise ValueError(f"planned source is unavailable or unsafe: {relative.name}")
    return path


def _committed_plan_id(plan: dict[str, Any]) -> str:
    fields = (
        "schema_version",
        "kind",
        "library_root_id",
        "source_root",
        "recursive",
        "files",
        "skipped",
    )
    if any(field not in plan for field in fields):
        raise ValueError("import plan is missing committed fields")
    commitment = {field: plan[field] for field in fields}
    return sha256_bytes(canonical_json_bytes(commitment))


def _bounded_incoming_count(path: Path) -> int:
    count = 0
    for item in path.iterdir() if path.is_dir() else ():
        if item.is_file() and not item.is_symlink():
            count += 1
            if count >= 10_000:
                break
    return count


def _root_id(path: Path) -> str:
    return f"sha256:{sha256_bytes(str(path.resolve()).encode('utf-8'))}"


def _media_type(kind: str, path: Path) -> str:
    suffix = path.suffix.lower()
    known = {
        ".pdb": "chemical/x-pdb",
        ".cif": "chemical/x-mmcif",
        ".mmcif": "chemical/x-mmcif",
        ".fa": "text/x-fasta",
        ".faa": "text/x-fasta",
        ".fasta": "text/x-fasta",
        ".sdf": "chemical/x-mdl-sdfile",
        ".mol": "chemical/x-mdl-molfile",
        ".smi": "chemical/x-daylight-smiles",
        ".smiles": "chemical/x-daylight-smiles",
    }
    return known.get(suffix, mimetypes.guess_type(path.name)[0] or f"application/x-{kind}")


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
    except FileExistsError:
        if path.read_bytes() != data:
            raise FileExistsError(f"immutable receipt already exists: {path.name}") from None


def _validate_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError("kind must be protein or ligand")


def _validate_digest(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _now() -> str:
    return datetime.now(UTC).isoformat()
