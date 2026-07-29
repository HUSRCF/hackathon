"""Privacy-gated receptor resolution before expensive protein folding.

Resolution is deliberately conservative: a supplied structure wins, then a local
exact-sequence cache, then an explicitly approved RCSB route.  A remote candidate
is accepted only after unique chain assignment, exact coordinate-sequence identity,
backbone completeness, and a metal-free structural inspection.  Otherwise the
caller receives an auditable ``folding_required`` decision.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import permutations
from pathlib import Path
from typing import Protocol

from . import __version__
from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes
from .models import ArtifactRef, PrivacyPolicy, RCSBCoordinatePolicy, TargetSpec
from .privacy import require_network_approval
from .structure import (
    ConnectionInspection,
    StructureCapabilityError,
    StructureInspection,
    inspect_declared_connections,
    inspect_structure,
    select_protein_chains,
)

_PDB_ID = re.compile(r"(?:[0-9][A-Z0-9]{3}|PDB_[A-Z0-9]{8})")
_RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_RCSB_FILE_ROOT = "https://files.rcsb.org/download"
_RCSB_LICENSE = "CC0-1.0"
_CACHE_SCHEMA = "protbind.structure-cache/3.0"
_RECEIPT_KIND = "protbind.structure-resolution"
_MAX_SEARCH_BYTES = 2 * 1024 * 1024
_MAX_STRUCTURE_BYTES = 64 * 1024 * 1024


class ResolutionDecision(StrEnum):
    USER_SUPPLIED = "user_supplied"
    LOCAL_EXACT_CACHE = "local_exact_sequence_cache"
    RCSB_IMPORTED = "rcsb_imported"
    FOLDING_REQUIRED = "folding_required"


class StructureResolutionError(ValueError):
    """A supplied or cached structure violated a non-negotiable identity gate."""


@dataclass(frozen=True, slots=True)
class HTTPResult:
    data: bytes
    headers: Mapping[str, str]


class HTTPTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None,
        headers: Mapping[str, str],
        max_bytes: int,
    ) -> HTTPResult: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):  # noqa: ANN001
        return None


class UrllibHTTPTransport:
    """Direct, redirect-refusing HTTPS transport with a hard response limit.

    An exact-domain approval applies to the actual peer selected by ProtBind.  It
    must not be silently re-routed through ambient ``HTTP(S)_PROXY`` settings.
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None,
        headers: Mapping[str, str],
        max_bytes: int,
    ) -> HTTPResult:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )
        try:
            with opener.open(request, timeout=60) as response:
                announced = response.headers.get("Content-Length")
                if announced is not None:
                    try:
                        announced_size = int(announced)
                    except ValueError:
                        announced_size = 0
                    if announced_size > max_bytes:
                        raise ValueError("RCSB response exceeds its configured size limit")
                data = response.read(max_bytes + 1)
                response_headers = {
                    str(name): str(value) for name, value in response.headers.items()
                }
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise PermissionError(
                    "RCSB redirects are not followed; approve and request the final domain"
                ) from exc
            raise OSError(f"RCSB returned HTTP {exc.code}") from exc
        if len(data) > max_bytes:
            raise ValueError("RCSB response exceeds its configured size limit")
        return HTTPResult(data=data, headers=response_headers)


@dataclass(frozen=True, slots=True)
class StructureResolutionResult:
    decision: ResolutionDecision
    structure: ArtifactRef | None
    receipt: ArtifactRef
    raw_source: ArtifactRef | None = None

    @property
    def folding_required(self) -> bool:
        return self.decision is ResolutionDecision.FOLDING_REQUIRED


@dataclass(frozen=True, slots=True)
class _AcceptedCandidate:
    structure: ArtifactRef
    raw_source: ArtifactRef
    pdb_id: str
    selected_chain_ids: tuple[str, ...]
    requested_chain_ids: tuple[str, ...]
    chain_selection_policy: str
    discarded_chain_ids: tuple[str, ...]
    inspection: StructureInspection
    connection_check: ConnectionInspection
    selected_connection_check: ConnectionInspection
    source_url: str
    archive_revision: str | None
    experimental_methods: tuple[str, ...]
    resolution_angstrom: float | None
    retrieval_headers: dict[str, str]
    coordinate_file_policy: RCSBCoordinatePolicy
    assembly_id: str | None


class StructureResolver:
    """Resolve a receptor while preserving private-by-default network semantics."""

    def __init__(
        self,
        workspace: Path,
        *,
        artifacts: ArtifactStore | None = None,
        transport: HTTPTransport | None = None,
        now: Callable[[], datetime] | None = None,
        max_candidates: int = 8,
    ) -> None:
        if max_candidates < 1 or max_candidates > 25:
            raise ValueError("max_candidates must be between 1 and 25")
        self.workspace = workspace.resolve()
        self.artifacts = artifacts or ArtifactStore(self.workspace)
        self.transport = transport or UrllibHTTPTransport()
        self.now = now or (lambda: datetime.now(UTC))
        self.max_candidates = max_candidates
        self.cache_path = self.workspace / "structure-cache-v3.json"

    def resolve(
        self,
        target: TargetSpec,
        privacy: PrivacyPolicy,
    ) -> StructureResolutionResult:
        expected = tuple(sequence.removesuffix("*") for sequence in target.sequences)
        identity = _sequence_identity(expected)

        if target.structure is not None:
            inspection, connection_check = self._validate_structure(
                target.structure, expected
            )
            self._register_validated(
                target.structure,
                inspection,
                connection_check,
                origin="user_supplied",
            )
            return self._result(
                ResolutionDecision.USER_SUPPLIED,
                target.structure,
                {
                    "sequence_identity": identity,
                    "selected_chain_ids": list(inspection.chain_ids),
                    "model_policy": "first coordinate model only",
                    "sequence_match": "exact" if expected else "structure_defined",
                    "qc": _inspection_receipt(inspection),
                    "connection_check": connection_check.to_dict(),
                    "selected_connection_check": connection_check.to_dict(),
                    "source_artifact": target.structure.to_dict(),
                    "selected_receptor_artifact": target.structure.to_dict(),
                    "coordinate_file_policy": "user_supplied",
                    "network_requests": [],
                },
            )

        cached = self._cache_lookup(expected, target)
        if cached is not None:
            structure, inspection, connection_check, cache_metadata, raw_source = cached
            return self._result(
                ResolutionDecision.LOCAL_EXACT_CACHE,
                structure,
                {
                    "sequence_identity": identity,
                    "selected_chain_ids": list(inspection.chain_ids),
                    "model_policy": "first coordinate model only",
                    "sequence_match": "exact",
                    "qc": _inspection_receipt(inspection),
                    "connection_check": cache_metadata.get(
                        "source_metadata", {}
                    ).get("connection_check", connection_check.to_dict()),
                    "selected_connection_check": connection_check.to_dict(),
                    "cache_origin": cache_metadata.get("origin", "unknown"),
                    "cache_source_metadata": cache_metadata.get(
                        "source_metadata", {}
                    ),
                    "source_artifact": structure.to_dict(),
                    "selected_receptor_artifact": structure.to_dict(),
                    "raw_source_artifact": (
                        raw_source.to_dict() if raw_source is not None else None
                    ),
                    "coordinate_file_policy": cache_metadata.get(
                        "source_metadata", {}
                    ).get("coordinate_file_policy", "local_structure"),
                    "assembly_id": cache_metadata.get("source_metadata", {}).get(
                        "assembly_id"
                    ),
                    "requested_chain_ids": cache_metadata.get(
                        "source_metadata", {}
                    ).get("requested_chain_ids", []),
                    "chain_selection_policy": cache_metadata.get(
                        "source_metadata", {}
                    ).get("chain_selection_policy", "local_exact_sequence"),
                    "network_requests": [],
                },
                raw_source=raw_source,
            )

        route, required_domains = self._remote_route(target, privacy)
        if route is None:
            return self._folding_required(
                identity,
                "no explicitly approved RCSB identifier or sequence-upload route",
                required_domains=(),
                coordinate_file_policy=target.rcsb_coordinate_policy,
                assembly_id=target.rcsb_assembly_id,
                requested_chain_ids=target.rcsb_chain_ids,
            )
        if not privacy.network_allowed:
            return self._folding_required(
                identity,
                "network access is disabled",
                required_domains=required_domains,
                coordinate_file_policy=target.rcsb_coordinate_policy,
                assembly_id=target.rcsb_assembly_id,
                requested_chain_ids=target.rcsb_chain_ids,
            )
        approved = {domain.lower().rstrip(".") for domain in privacy.approved_domains}
        missing_domains = tuple(domain for domain in required_domains if domain not in approved)
        if missing_domains:
            return self._folding_required(
                identity,
                "required RCSB domains were not explicitly approved",
                required_domains=required_domains,
                missing_domains=missing_domains,
                coordinate_file_policy=target.rcsb_coordinate_policy,
                assembly_id=target.rcsb_assembly_id,
                requested_chain_ids=target.rcsb_chain_ids,
            )

        retrieved_at = self.now().astimezone(UTC).isoformat()
        attempts: list[dict[str, object]] = []
        network_requests: list[dict[str, object]] = []
        try:
            pdb_ids = self._candidate_ids(target, privacy, network_requests)
        except (OSError, PermissionError, ValueError, json.JSONDecodeError) as exc:
            return self._folding_required(
                identity,
                f"RCSB discovery failed: {type(exc).__name__}",
                required_domains=required_domains,
                attempts=attempts,
                network_requests=network_requests,
                coordinate_file_policy=target.rcsb_coordinate_policy,
                assembly_id=target.rcsb_assembly_id,
                requested_chain_ids=target.rcsb_chain_ids,
            )

        for pdb_id in pdb_ids[: self.max_candidates]:
            source_url = _coordinate_source_url(
                pdb_id,
                target.rcsb_coordinate_policy,
                target.rcsb_assembly_id,
            )
            network_requests.append(
                {
                    "method": "GET",
                    "url": source_url,
                    "sequence_uploaded": False,
                    "coordinate_file_policy": target.rcsb_coordinate_policy.value,
                    "assembly_id": target.rcsb_assembly_id,
                }
            )
            raw_source: ArtifactRef | None = None
            try:
                result = self._network_request(
                    privacy,
                    "GET",
                    source_url,
                    body=None,
                    max_bytes=_MAX_STRUCTURE_BYTES,
                )
                raw_source = self._persist_raw_mmcif(result.data, source_url)
                accepted = self._evaluate_candidate(
                    result.data,
                    result.headers,
                    raw_source=raw_source,
                    pdb_id=pdb_id,
                    source_url=source_url,
                    expected_sequences=expected,
                    requested_chain_ids=target.rcsb_chain_ids,
                    coordinate_file_policy=target.rcsb_coordinate_policy,
                    assembly_id=target.rcsb_assembly_id,
                )
            except (
                OSError,
                PermissionError,
                RuntimeError,
                ValueError,
                StructureCapabilityError,
            ) as exc:
                attempt: dict[str, object] = {
                    "pdb_id": pdb_id,
                    "accepted": False,
                    "reason_code": _reason_code(exc),
                }
                if raw_source is not None:
                    attempt["raw_source_artifact"] = raw_source.to_dict()
                attempts.append(attempt)
                continue
            attempts.append(
                {
                    "pdb_id": pdb_id,
                    "accepted": True,
                    "selected_chain_ids": list(accepted.selected_chain_ids),
                    "discarded_chain_ids": list(accepted.discarded_chain_ids),
                    "raw_source_artifact": accepted.raw_source.to_dict(),
                }
            )
            self._register_validated(
                accepted.structure,
                accepted.inspection,
                accepted.selected_connection_check,
                origin=f"rcsb:{accepted.pdb_id}",
                source_metadata={
                    "pdb_id": accepted.pdb_id,
                    "source_url": accepted.source_url,
                    "retrieved_at": retrieved_at,
                    "archive_revision": accepted.archive_revision,
                    "license": _RCSB_LICENSE,
                    "coordinate_file_policy": accepted.coordinate_file_policy.value,
                    "assembly_id": accepted.assembly_id,
                    "raw_source_artifact": accepted.raw_source.to_dict(),
                    "selected_receptor_artifact": accepted.structure.to_dict(),
                    "selected_chain_ids": list(accepted.selected_chain_ids),
                    "requested_chain_ids": list(accepted.requested_chain_ids),
                    "chain_selection_policy": accepted.chain_selection_policy,
                    "discarded_protein_chain_ids": list(
                        accepted.discarded_chain_ids
                    ),
                    "connection_check": accepted.connection_check.to_dict(),
                    "selected_connection_check": (
                        accepted.selected_connection_check.to_dict()
                    ),
                },
            )
            return self._result(
                ResolutionDecision.RCSB_IMPORTED,
                accepted.structure,
                {
                    "sequence_identity": identity,
                    "sequence_match": "exact",
                    "pdb_id": accepted.pdb_id,
                    "selected_chain_ids": list(accepted.selected_chain_ids),
                    "requested_chain_ids": list(accepted.requested_chain_ids),
                    "chain_selection_policy": accepted.chain_selection_policy,
                    "discarded_protein_chain_ids": list(accepted.discarded_chain_ids),
                    "multi_chain_policy": (
                        "unique exact-sequence assignment or explicit chain IDs; "
                        "ambiguous assignments rejected"
                    ),
                    "receptor_sanitization": (
                        "selected standard protein residues only; waters and "
                        "non-polymers removed"
                    ),
                    "model_policy": "first coordinate model only",
                    "qc": _inspection_receipt(accepted.inspection),
                    "connection_check": accepted.connection_check.to_dict(),
                    "selected_connection_check": (
                        accepted.selected_connection_check.to_dict()
                    ),
                    "source_url": accepted.source_url,
                    "retrieved_at": retrieved_at,
                    "archive_revision": accepted.archive_revision,
                    "experimental_methods": list(accepted.experimental_methods),
                    "resolution_angstrom": accepted.resolution_angstrom,
                    "retrieval_headers": accepted.retrieval_headers,
                    "license": _RCSB_LICENSE,
                    "coordinate_file_policy": accepted.coordinate_file_policy.value,
                    "assembly_id": accepted.assembly_id,
                    "source_artifact": accepted.raw_source.to_dict(),
                    "raw_source_artifact": accepted.raw_source.to_dict(),
                    "selected_receptor_artifact": accepted.structure.to_dict(),
                    "candidate_attempts": attempts,
                    "network_requests": network_requests,
                },
                raw_source=accepted.raw_source,
            )

        return self._folding_required(
            identity,
            "no RCSB candidate passed exact identity and structural QC",
            required_domains=required_domains,
            attempts=attempts,
            network_requests=network_requests,
            coordinate_file_policy=target.rcsb_coordinate_policy,
            assembly_id=target.rcsb_assembly_id,
            requested_chain_ids=target.rcsb_chain_ids,
        )

    def register(
        self,
        structure: ArtifactRef,
        sequences: tuple[str, ...],
        *,
        origin: str,
        source_metadata: dict[str, object] | None = None,
    ) -> StructureInspection:
        """Register a completed local fold for reuse by later exact-sequence cases."""

        expected = tuple(sequence.removesuffix("*") for sequence in sequences)
        inspection, connection_check = self._validate_structure(structure, expected)
        self._register_validated(
            structure,
            inspection,
            connection_check,
            origin=origin,
            source_metadata=source_metadata,
        )
        return inspection

    def _validate_structure(
        self,
        structure: ArtifactRef,
        expected_sequences: tuple[str, ...],
    ) -> tuple[StructureInspection, ConnectionInspection]:
        path = self.artifacts.resolve(structure)
        inspection = inspect_structure(path)
        connection_check = inspect_declared_connections(path)
        if expected_sequences and inspection.sequences != expected_sequences:
            raise StructureResolutionError(
                "protein coordinate sequences do not exactly match the requested target"
            )
        if inspection.missing_backbone_residues:
            raise StructureResolutionError(
                "protein structure has residues missing N/CA/C backbone atoms"
            )
        if inspection.alternate_location_atoms:
            raise StructureResolutionError(
                "protein structure contains unresolved alternate-location atoms"
            )
        if inspection.metal_elements:
            raise StructureResolutionError(
                "metal-containing receptor structures are unsupported in v1"
            )
        if connection_check.covalent_detected:
            raise StructureResolutionError(
                "receptor structure declares a covalent protein/ligand connection"
            )
        return inspection, connection_check

    def _remote_route(
        self, target: TargetSpec, privacy: PrivacyPolicy
    ) -> tuple[str | None, tuple[str, ...]]:
        if target.pdb_id is not None:
            return "pdb_id", ("files.rcsb.org",)
        if target.uniprot_accession is not None:
            return "uniprot", ("search.rcsb.org", "files.rcsb.org")
        if privacy.sequence_upload_allowed and target.sequences:
            return "sequence", ("search.rcsb.org", "files.rcsb.org")
        return None, ()

    def _candidate_ids(
        self,
        target: TargetSpec,
        privacy: PrivacyPolicy,
        network_requests: list[dict[str, object]],
    ) -> tuple[str, ...]:
        if target.pdb_id is not None:
            return (target.pdb_id,)
        if target.uniprot_accession is not None:
            payload = _uniprot_query(target.uniprot_accession, self.max_candidates)
            network_requests.append(
                {
                    "method": "POST",
                    "url": _RCSB_SEARCH_URL,
                    "sequence_uploaded": False,
                    "query_type": "uniprot_accession",
                }
            )
            identifiers = self._search(payload, privacy, sequence_uploaded=False)
            return _entry_ids(identifiers)
        if not privacy.sequence_upload_allowed:
            raise PermissionError("RCSB sequence search requires sequence-upload approval")
        entry_lists: list[tuple[str, ...]] = []
        for sequence in target.sequences:
            payload = _sequence_query(sequence.removesuffix("*"), self.max_candidates)
            network_requests.append(
                {
                    "method": "POST",
                    "url": _RCSB_SEARCH_URL,
                    "sequence_uploaded": True,
                    "query_type": "exact_protein_sequence",
                }
            )
            identifiers = self._search(payload, privacy, sequence_uploaded=True)
            entry_lists.append(_entry_ids(identifiers))
        if not entry_lists:
            return ()
        common = set(entry_lists[0])
        for entries in entry_lists[1:]:
            common.intersection_update(entries)
        return tuple(entry for entry in entry_lists[0] if entry in common)

    def _search(
        self,
        payload: dict[str, object],
        privacy: PrivacyPolicy,
        *,
        sequence_uploaded: bool,
    ) -> tuple[str, ...]:
        if sequence_uploaded and not privacy.sequence_upload_allowed:
            raise PermissionError("RCSB sequence search requires sequence-upload approval")
        result = self._network_request(
            privacy,
            "POST",
            _RCSB_SEARCH_URL,
            body=canonical_json_bytes(payload),
            max_bytes=_MAX_SEARCH_BYTES,
        )
        # RCSB Search uses an empty/204-style response for a normal no-hit
        # outcome.  That is not a malformed scientific result or network fault.
        if not result.data.strip():
            return ()
        value = json.loads(result.data)
        result_set = value.get("result_set") if isinstance(value, dict) else None
        if not isinstance(result_set, list):
            raise ValueError("RCSB search response has no result_set array")
        identifiers: list[str] = []
        for item in result_set:
            identifier = item.get("identifier") if isinstance(item, dict) else None
            if isinstance(identifier, str):
                identifiers.append(identifier)
        return tuple(identifiers)

    def _network_request(
        self,
        privacy: PrivacyPolicy,
        method: str,
        url: str,
        *,
        body: bytes | None,
        max_bytes: int,
    ) -> HTTPResult:
        require_network_approval(url, privacy.approved_domains)
        headers = {"User-Agent": f"ProtBind/{__version__} explicit-rcsb-import"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        return self.transport.request(
            method,
            url,
            body=body,
            headers=headers,
            max_bytes=max_bytes,
        )

    def _evaluate_candidate(
        self,
        data: bytes,
        headers: Mapping[str, str],
        *,
        raw_source: ArtifactRef,
        pdb_id: str,
        source_url: str,
        expected_sequences: tuple[str, ...],
        requested_chain_ids: tuple[str, ...],
        coordinate_file_policy: RCSBCoordinatePolicy,
        assembly_id: str | None,
    ) -> _AcceptedCandidate:
        connection_check = inspect_declared_connections(
            self.artifacts.resolve(raw_source)
        )
        if connection_check.covalent_detected:
            raise ValueError("RCSB candidate declares a covalent structure connection")
        path = self.artifacts.resolve(raw_source)
        full = inspect_structure(path, max_chains=None, max_residues=None)
        if full.metal_elements:
            raise ValueError("RCSB candidate contains a metal center")
        if full.alternate_location_atoms:
            raise ValueError("RCSB candidate contains unresolved alternate locations")
        selected_ids = _select_chain_ids(
            full,
            expected_sequences=expected_sequences,
            requested_chain_ids=requested_chain_ids,
        )
        selected_data = select_protein_chains(path, selected_ids)
        archive_revision = _mmcif_revision(data)
        experimental_methods, resolution_angstrom = _mmcif_experiment_metadata(data)
        structure = self.artifacts.put_bytes(
            selected_data,
            media_type="chemical/x-mmcif",
            producer="protbind.rcsb-import",
            producer_version=archive_revision or "archive-current",
            source=source_url,
            license=_RCSB_LICENSE,
        )
        inspection, selected_connection_check = self._validate_structure(
            structure, expected_sequences
        )
        if inspection.chain_ids != selected_ids:
            raise ValueError("selected-chain identities changed during mmCIF extraction")
        safe_headers = {
            name.lower(): str(value)
            for name, value in headers.items()
            if name.lower() in {"etag", "last-modified", "content-type"}
            and not any(ord(character) < 32 for character in str(value))
        }
        return _AcceptedCandidate(
            structure=structure,
            raw_source=raw_source,
            pdb_id=pdb_id,
            selected_chain_ids=selected_ids,
            requested_chain_ids=requested_chain_ids,
            chain_selection_policy=(
                "explicit" if requested_chain_ids else "unique_exact"
            ),
            discarded_chain_ids=tuple(
                chain for chain in full.chain_ids if chain not in selected_ids
            ),
            inspection=inspection,
            connection_check=connection_check,
            selected_connection_check=selected_connection_check,
            source_url=source_url,
            archive_revision=archive_revision,
            experimental_methods=experimental_methods,
            resolution_angstrom=resolution_angstrom,
            retrieval_headers=dict(sorted(safe_headers.items())),
            coordinate_file_policy=coordinate_file_policy,
            assembly_id=assembly_id,
        )

    def _persist_raw_mmcif(self, data: bytes, source_url: str) -> ArtifactRef:
        if not data.lstrip().lower().startswith(b"data_"):
            raise ValueError("RCSB coordinate response is not an mmCIF document")
        archive_revision = _mmcif_revision(data)
        return self.artifacts.put_bytes(
            data,
            media_type="chemical/x-mmcif",
            producer="protbind.rcsb-download",
            producer_version=archive_revision or "archive-current",
            source=source_url,
            license=_RCSB_LICENSE,
        )

    def _cache_lookup(
        self,
        expected_sequences: tuple[str, ...],
        target: TargetSpec,
    ) -> tuple[
        ArtifactRef,
        StructureInspection,
        ConnectionInspection,
        dict[str, object],
        ArtifactRef | None,
    ] | None:
        if not expected_sequences:
            return None
        with self._cache_lock():
            cache = self._load_cache()
        entries = cache["entries"]
        sequence_key = _sequence_cache_key(expected_sequences)
        matching_entries_found = False
        if target.pdb_id is not None:
            lookup_key = _sequence_cache_entry_key(
                expected_sequences,
                scope=_rcsb_cache_scope(
                    target.pdb_id,
                    target.rcsb_coordinate_policy,
                    target.rcsb_assembly_id,
                    target.rcsb_chain_ids,
                ),
            )
            entry = entries.get(lookup_key)
            matching_entries_found = entry is not None
        else:
            entry = entries.get(
                _sequence_cache_entry_key(expected_sequences, scope="local")
            )
            if entry is None:
                matches = [
                    value
                    for _, value in sorted(entries.items())
                    if isinstance(value, dict)
                    and value.get("sequence_cache_key") == sequence_key
                ]
                matching_entries_found = bool(matches)
                artifact_ids = {
                    json.dumps(value.get("artifact"), sort_keys=True)
                    for value in matches
                    if isinstance(value.get("artifact"), dict)
                }
                entry = matches[0] if len(artifact_ids) == 1 else None
            else:
                matching_entries_found = True
        if entry is None:
            if matching_entries_found:
                raise StructureResolutionError(
                    "matching local structure cache entries are corrupt or conflicting"
                )
            return None
        if not isinstance(entry, dict) or not isinstance(entry.get("artifact"), dict):
            raise StructureResolutionError(
                "matching local structure cache entry is malformed"
            )
        source_metadata = entry.get("source_metadata", {})
        if not isinstance(source_metadata, dict):
            raise StructureResolutionError(
                "matching local structure cache source metadata is malformed"
            )
        # An explicit archive representation is a scientific input, not a hint.
        # Never satisfy an assembly request from an ASU cache entry (or vice versa).
        if target.pdb_id is not None and (
            source_metadata.get("pdb_id") != target.pdb_id
            or source_metadata.get("coordinate_file_policy")
            != target.rcsb_coordinate_policy.value
            or source_metadata.get("assembly_id") != target.rcsb_assembly_id
            or tuple(source_metadata.get("requested_chain_ids", ()))
            != target.rcsb_chain_ids
        ):
            return None
        try:
            structure = ArtifactRef.from_dict(entry["artifact"])
            inspection, connection_check = self._validate_structure(
                structure, expected_sequences
            )
            if tuple(entry.get("selected_chain_ids", ())) != inspection.chain_ids:
                raise StructureResolutionError(
                    "cached selected-chain evidence differs from the artifact"
                )
            self._validate_rcsb_cache_binding(
                structure, inspection, source_metadata
            )
            if entry.get("connection_check") != connection_check.to_dict():
                raise StructureResolutionError(
                    "cached structure connection evidence differs from the artifact"
                )
            raw_value = source_metadata.get("raw_source_artifact")
            raw_source = (
                ArtifactRef.from_dict(raw_value) if isinstance(raw_value, dict) else None
            )
            if raw_source is not None:
                raw_connection = inspect_declared_connections(
                    self.artifacts.resolve(raw_source)
                )
                if raw_connection.covalent_detected:
                    raise StructureResolutionError(
                        "cached raw source declares a covalent connection"
                    )
                if source_metadata.get("connection_check") != raw_connection.to_dict():
                    raise StructureResolutionError(
                        "cached raw connection evidence differs from the artifact"
                    )
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            raise StructureResolutionError(
                "matching local structure cache entry is missing or invalid"
            ) from exc
        return structure, inspection, connection_check, entry, raw_source

    def _register_validated(
        self,
        structure: ArtifactRef,
        inspection: StructureInspection,
        connection_check: ConnectionInspection,
        *,
        origin: str,
        source_metadata: dict[str, object] | None = None,
    ) -> None:
        sequences = inspection.sequences
        if not sequences:
            return
        if source_metadata is not None:
            self._validate_rcsb_cache_binding(
                structure, inspection, source_metadata
            )
        with self._cache_lock():
            cache = self._load_cache()
            entry: dict[str, object] = {
                "artifact": structure.to_dict(),
                "origin": origin,
                "selected_chain_ids": list(inspection.chain_ids),
                "sequence_identity": _sequence_identity(sequences),
                "sequence_cache_key": _sequence_cache_key(sequences),
                "connection_check": connection_check.to_dict(),
            }
            if source_metadata is not None:
                entry["source_metadata"] = source_metadata
            scope = "local"
            if source_metadata is not None and isinstance(
                source_metadata.get("pdb_id"), str
            ):
                scope = _rcsb_cache_scope(
                    str(source_metadata["pdb_id"]),
                    RCSBCoordinatePolicy(
                        source_metadata.get(
                            "coordinate_file_policy",
                            RCSBCoordinatePolicy.DEPOSITED_ASYMMETRIC_UNIT.value,
                        )
                    ),
                    (
                        str(source_metadata["assembly_id"])
                        if source_metadata.get("assembly_id") is not None
                        else None
                    ),
                    tuple(
                        str(item)
                        for item in source_metadata.get("requested_chain_ids", ())
                    ),
                )
            cache_key = _sequence_cache_entry_key(sequences, scope=scope)
            existing = cache["entries"].get(cache_key)
            if isinstance(existing, dict) and existing.get("artifact") != entry["artifact"]:
                raise StructureResolutionError(
                    "an exact cache scope already names a different structure artifact"
                )
            cache["entries"][cache_key] = entry
            self._save_cache(cache)

    @staticmethod
    def _validate_rcsb_cache_binding(
        structure: ArtifactRef,
        inspection: StructureInspection,
        source_metadata: Mapping[str, object],
    ) -> None:
        """Prevent public cache registration from forging an explicit chain scope."""

        if not isinstance(source_metadata.get("pdb_id"), str):
            return
        selected = source_metadata.get("selected_chain_ids")
        requested = source_metadata.get("requested_chain_ids")
        if (
            not isinstance(selected, list | tuple)
            or not all(isinstance(item, str) for item in selected)
            or tuple(selected) != inspection.chain_ids
            or not isinstance(requested, list | tuple)
            or not all(isinstance(item, str) for item in requested)
        ):
            raise StructureResolutionError(
                "RCSB cache chain metadata differs from the selected receptor"
            )
        requested_ids = tuple(requested)
        expected_policy = "explicit" if requested_ids else "unique_exact"
        if (
            (requested_ids and requested_ids != inspection.chain_ids)
            or source_metadata.get("chain_selection_policy") != expected_policy
            or source_metadata.get("selected_receptor_artifact")
            != structure.to_dict()
        ):
            raise StructureResolutionError(
                "RCSB cache scope is not bound to the selected receptor artifact"
            )

    @contextmanager
    def _cache_lock(self):
        """Serialize cache metadata updates without widening artifact permissions."""

        try:
            import fcntl
        except ImportError:  # pragma: no cover - production target is Linux/ROCm
            yield
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_path.with_suffix(".lock")
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_cache(self) -> dict[str, object]:
        if not self.cache_path.is_file():
            return {"schema_version": _CACHE_SCHEMA, "entries": {}}
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StructureResolutionError("local structure cache metadata is invalid") from exc
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != _CACHE_SCHEMA
            or not isinstance(value.get("entries"), dict)
        ):
            raise StructureResolutionError("local structure cache schema is invalid")
        return value

    def _save_cache(self, value: dict[str, object]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical_json_bytes(value)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.cache_path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _result(
        self,
        decision: ResolutionDecision,
        structure: ArtifactRef | None,
        details: dict[str, object],
        *,
        raw_source: ArtifactRef | None = None,
    ) -> StructureResolutionResult:
        payload = {
            "schema_version": "1.0",
            "kind": _RECEIPT_KIND,
            "decision": decision.value,
            "folding_required": decision is ResolutionDecision.FOLDING_REQUIRED,
            **details,
        }
        receipt = self.artifacts.put_json(
            payload,
            producer="protbind.structure-resolver",
            producer_version=__version__,
        )
        return StructureResolutionResult(
            decision=decision,
            structure=structure,
            receipt=receipt,
            raw_source=raw_source,
        )

    def _folding_required(
        self,
        identity: dict[str, object],
        reason: str,
        *,
        required_domains: tuple[str, ...],
        missing_domains: tuple[str, ...] = (),
        attempts: list[dict[str, object]] | None = None,
        network_requests: list[dict[str, object]] | None = None,
        coordinate_file_policy: RCSBCoordinatePolicy = (
            RCSBCoordinatePolicy.DEPOSITED_ASYMMETRIC_UNIT
        ),
        assembly_id: str | None = None,
        requested_chain_ids: tuple[str, ...] = (),
    ) -> StructureResolutionResult:
        return self._result(
            ResolutionDecision.FOLDING_REQUIRED,
            None,
            {
                "sequence_identity": identity,
                "reason": reason,
                "required_domains": list(required_domains),
                "missing_domains": list(missing_domains),
                "candidate_attempts": attempts or [],
                "network_requests": network_requests or [],
                "coordinate_file_policy": coordinate_file_policy.value,
                "assembly_id": assembly_id,
                "requested_chain_ids": list(requested_chain_ids),
            },
        )


def _sequence_cache_key(sequences: tuple[str, ...]) -> str:
    return sha256_bytes(canonical_json_bytes({"sequences": sequences}))


def _sequence_cache_entry_key(sequences: tuple[str, ...], *, scope: str) -> str:
    return sha256_bytes(
        canonical_json_bytes({"sequence_key": _sequence_cache_key(sequences), "scope": scope})
    )


def _rcsb_cache_scope(
    pdb_id: str,
    policy: RCSBCoordinatePolicy,
    assembly_id: str | None,
    requested_chain_ids: tuple[str, ...] = (),
) -> str:
    return ":".join(
        (
            "rcsb",
            pdb_id.upper(),
            policy.value,
            assembly_id or "none",
            ",".join(requested_chain_ids) if requested_chain_ids else "auto",
        )
    )


def _coordinate_source_url(
    pdb_id: str,
    policy: RCSBCoordinatePolicy,
    assembly_id: str | None,
) -> str:
    stem = pdb_id.lower()
    if policy is RCSBCoordinatePolicy.BIOLOGICAL_ASSEMBLY:
        if assembly_id is None:  # protected by TargetSpec; keep the boundary local too
            raise ValueError("biological assembly coordinate request has no assembly ID")
        stem = f"{stem}-assembly{assembly_id}"
    return f"{_RCSB_FILE_ROOT}/{stem}.cif"


def _sequence_identity(sequences: tuple[str, ...]) -> dict[str, object]:
    return {
        "chain_count": len(sequences),
        "lengths": [len(sequence) for sequence in sequences],
        "sha256": [sha256_bytes(sequence.encode("ascii")) for sequence in sequences],
        "ordering": "target_chain_order",
    }


def _inspection_receipt(inspection: StructureInspection) -> dict[str, object]:
    value = asdict(inspection)
    value.pop("sequences")
    value["sequence_identity"] = _sequence_identity(inspection.sequences)
    return value


def _select_chain_ids(
    inspection: StructureInspection,
    *,
    expected_sequences: tuple[str, ...],
    requested_chain_ids: tuple[str, ...],
) -> tuple[str, ...]:
    sequence_by_chain = dict(zip(inspection.chain_ids, inspection.sequences, strict=True))
    if not expected_sequences:
        raise ValueError("RCSB resolution requires target sequences for identity verification")
    if requested_chain_ids:
        if any(chain not in sequence_by_chain for chain in requested_chain_ids):
            raise ValueError("explicit RCSB chain selection is absent from the entry")
        actual = tuple(sequence_by_chain[chain] for chain in requested_chain_ids)
        if actual != expected_sequences:
            raise ValueError("explicit RCSB chains do not exactly match target sequences")
        return requested_chain_ids
    if len(expected_sequences) > len(inspection.chain_ids):
        raise ValueError("RCSB candidate has fewer protein chains than the target")
    assignments = {
        candidate
        for candidate in permutations(inspection.chain_ids, len(expected_sequences))
        if tuple(sequence_by_chain[chain] for chain in candidate) == expected_sequences
    }
    if not assignments:
        raise ValueError("RCSB candidate has no exact coordinate-sequence chain assignment")
    selected_sets = {frozenset(assignment) for assignment in assignments}
    if len(selected_sets) != 1:
        raise ValueError("RCSB candidate has an ambiguous multi-chain sequence assignment")
    selected_set = next(iter(selected_sets))
    return tuple(
        chain
        for sequence in expected_sequences
        for chain in inspection.chain_ids
        if chain in selected_set and sequence_by_chain[chain] == sequence
    )[: len(expected_sequences)]


def _uniprot_query(accession: str, rows: int) -> dict[str, object]:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "exact_match",
                        "value": accession,
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_accession"
                        ),
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "exact_match",
                        "value": "UniProt",
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_name"
                        ),
                    },
                },
            ],
        },
        "request_options": {"paginate": {"start": 0, "rows": rows}},
        "return_type": "polymer_entity",
    }


def _sequence_query(sequence: str, rows: int) -> dict[str, object]:
    return {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 0.1,
                "identity_cutoff": 1.0,
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "request_options": {
            "scoring_strategy": "sequence",
            "paginate": {"start": 0, "rows": rows},
        },
        "return_type": "polymer_entity",
    }


def _entry_ids(identifiers: tuple[str, ...]) -> tuple[str, ...]:
    entries: list[str] = []
    for identifier in identifiers:
        entry = identifier.rsplit("_", 1)[0].upper()
        if _PDB_ID.fullmatch(entry) and entry not in entries:
            entries.append(entry)
    return tuple(entries)


def _mmcif_revision(data: bytes) -> str | None:
    try:
        import gemmi

        block = gemmi.cif.read_string(data.decode("utf-8")).sole_block()
        revisions = [
            gemmi.cif.as_string(str(value)).strip()
            for value in block.find_values("_pdbx_audit_revision_history.revision_date")
            if gemmi.cif.as_string(str(value)).strip()
        ]
    except (ImportError, RuntimeError, UnicodeDecodeError, ValueError):
        return None
    return max(revisions) if revisions else None


def _mmcif_experiment_metadata(data: bytes) -> tuple[tuple[str, ...], float | None]:
    try:
        import gemmi

        block = gemmi.cif.read_string(data.decode("utf-8")).sole_block()
        methods = tuple(
            dict.fromkeys(
                gemmi.cif.as_string(str(value)).strip()
                for value in block.find_values("_exptl.method")
                if gemmi.cif.as_string(str(value)).strip()
            )
        )
        resolutions: list[float] = []
        for tag in (
            "_refine.ls_d_res_high",
            "_em_3d_reconstruction.resolution",
            "_reflns.d_resolution_high",
        ):
            for raw in block.find_values(tag):
                try:
                    value = float(gemmi.cif.as_string(str(raw)))
                except ValueError:
                    continue
                if math.isfinite(value) and value > 0:
                    resolutions.append(value)
    except (ImportError, RuntimeError, UnicodeDecodeError, ValueError):
        return (), None
    return methods, min(resolutions) if resolutions else None


def _reason_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "non-finite" in message:
        return "NON_FINITE_COORDINATES"
    if "alternate" in message or "altloc" in message:
        return "UNRESOLVED_ALTLOC"
    if "metal" in message:
        return "METAL_STRUCTURE_REJECTED"
    if "covalent" in message:
        return "COVALENT_STRUCTURE_REJECTED"
    if "ambiguous" in message or "multi-chain" in message:
        return "AMBIGUOUS_CHAIN_ASSIGNMENT"
    if "sequence" in message or "chains do not exactly" in message:
        return "SEQUENCE_MISMATCH"
    if "backbone" in message:
        return "MISSING_BACKBONE"
    if isinstance(exc, StructureCapabilityError):
        return "STRUCTURE_CAPABILITY_UNAVAILABLE"
    if isinstance(exc, PermissionError):
        return "NETWORK_NOT_APPROVED"
    if isinstance(exc, OSError):
        return "NETWORK_ERROR"
    return "STRUCTURE_QC_REJECTED"
