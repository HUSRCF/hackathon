"""Frozen train-only screening protocol and one-shot validation authorization."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file

_ROLES = ("active_train", "inactive_train", "active_validation", "inactive_validation")


def _record_count(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def freeze_screening_protocol(
    *,
    dataset_name: str,
    dataset_archive: Path,
    targets: Mapping[str, Mapping[str, Path]],
    exposed_targets: set[str] | frozenset[str],
    hyperparameters: Mapping[str, Any],
    code_paths: tuple[Path, ...],
    output: Path,
) -> dict[str, Any]:
    """Write an immutable protocol commitment before prospective screening."""

    if output.exists():
        raise FileExistsError("screening protocol already exists and cannot be overwritten")
    if not dataset_name.strip() or not targets:
        raise ValueError("dataset name and at least one prospective target are required")
    overlap = sorted(set(targets) & set(exposed_targets))
    if overlap:
        raise ValueError(f"exposed targets cannot be prospective: {overlap}")
    target_payload: dict[str, Any] = {}
    for target, paths in sorted(targets.items()):
        if set(paths) != set(_ROLES):
            raise ValueError(f"target {target} must define exactly {_ROLES}")
        resolved = {role: Path(paths[role]).resolve() for role in _ROLES}
        if len(set(resolved.values())) != len(_ROLES):
            raise ValueError(f"target {target} reuses a file across split roles")
        target_payload[target] = {
            role: {
                "sha256": sha256_file(path),
                "record_count": _record_count(path),
                "source_name": path.name,
            }
            for role, path in resolved.items()
        }
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.prospective-screening-protocol",
        "dataset": {
            "name": dataset_name,
            "archive_sha256": sha256_file(dataset_archive),
        },
        "prospective_targets": target_payload,
        "exposed_targets": sorted(exposed_targets),
        "hyperparameters": dict(hyperparameters),
        "code_sha256": {
            path.resolve().as_posix(): sha256_file(path) for path in sorted(code_paths)
        },
        "validation_policy": {
            "query_or_metric_before_freeze": "FORBIDDEN",
            "identity_count_and_parse_qc": "ALLOWED",
            "one_shot_per_target_and_protocol": True,
            "failed_records_remain_in_denominator": True,
        },
    }
    payload["protocol_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def authorize_validation_once(
    *,
    protocol_path: Path,
    target: str,
    receipt_path: Path,
) -> dict[str, Any]:
    """Atomically consume one validation authorization for a frozen protocol."""

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    committed = protocol.get("protocol_sha256")
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    observed = sha256_bytes(canonical_json_bytes(unsigned))
    if not isinstance(committed, str) or committed != observed:
        raise ValueError("screening protocol hash is invalid")
    if target not in protocol.get("prospective_targets", {}):
        raise ValueError("target is not committed as prospective")
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "protbind.validation-one-shot-authorization",
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_sha256": committed,
        "target": target,
        "status": "CONSUMED",
    }
    receipt["authorization_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(receipt) + b"\n")
    except BaseException:
        receipt_path.unlink(missing_ok=True)
        raise
    return receipt


def verify_validation_authorization(
    *,
    protocol_path: Path,
    target: str,
    receipt_path: Path,
) -> dict[str, Any]:
    """Fail closed unless a consumed receipt matches an untampered protocol."""

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    committed = protocol.get("protocol_sha256")
    unsigned_protocol = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
    if committed != sha256_bytes(canonical_json_bytes(unsigned_protocol)):
        raise ValueError("screening protocol hash is invalid")
    for path_name, expected_sha256 in protocol.get("code_sha256", {}).items():
        path = Path(path_name)
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"frozen screening code hash changed: {path.name}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authorization_sha256 = receipt.get("authorization_sha256")
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "authorization_sha256"
    }
    if authorization_sha256 != sha256_bytes(canonical_json_bytes(unsigned_receipt)):
        raise ValueError("validation authorization hash is invalid")
    if (
        receipt.get("kind") != "protbind.validation-one-shot-authorization"
        or receipt.get("status") != "CONSUMED"
        or receipt.get("target") != target
        or receipt.get("protocol_sha256") != committed
        or receipt.get("protocol_file_sha256") != sha256_file(protocol_path)
        or target not in protocol.get("prospective_targets", {})
    ):
        raise ValueError("validation authorization does not match protocol and target")
    return {"protocol": protocol, "authorization": receipt}
