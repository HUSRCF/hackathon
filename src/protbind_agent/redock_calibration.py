"""Hash-bound known-site calibration receipts derived from redocking results.

The receipt is deliberately smaller than a redocking result.  It carries only
the prepared receptor and known-site box authorization needed by a later
``both``-mode screen.  Native/reference ligand artifacts, identities, and
coordinates remain outside this downstream contract.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes
from .models import ArtifactRef

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_RANKS = frozenset({"top1", "top5"})

_REFERENCE_BOUNDARY = {
    "reference_ligand_scope": "VALIDATION_AND_CALIBRATION_ONLY",
    "native_coordinates_visible_to_pose_generation": False,
    "reference_ligand_visible_to_candidate_pose_generation": False,
    "permitted_downstream_derivative": "KNOWN_SITE_BOX_RECEIPT_ONLY",
    "candidate_pose_start_coordinates": "CANDIDATE_SPECIFIC_AND_REFERENCE_INDEPENDENT",
}
_DISCLAIMERS = (
    "A passing receipt establishes only target-specific known-site pose-recovery calibration.",
    "The reference ligand is not an authorized candidate pose-generation input.",
    "The known-site box may be reused only with the hash-bound prepared receptor.",
    "Vina scores are model scores, not experimental binding free energies or affinities.",
)


@dataclass(frozen=True, slots=True)
class KnownSiteCalibrationConfig:
    """Explicit scientific gate for one target and one frozen docking setup."""

    target_id: str
    required_rank: str = "top1"
    rmsd_threshold_angstrom: float = 2.0

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.target_id):
            raise ValueError("calibration target_id must be a safe explicit identifier")
        if self.required_rank not in _REQUIRED_RANKS:
            raise ValueError("calibration required_rank must be top1 or top5")
        if (
            isinstance(self.rmsd_threshold_angstrom, bool)
            or not isinstance(self.rmsd_threshold_angstrom, int | float)
            or not math.isfinite(float(self.rmsd_threshold_angstrom))
            or float(self.rmsd_threshold_angstrom) <= 0
            or float(self.rmsd_threshold_angstrom) > 2.0
        ):
            raise ValueError(
                "calibration authorization RMSD threshold must be in (0, 2.0] angstrom"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "target_id": self.target_id,
            "screening_mode": "both",
            "required_rank": self.required_rank,
            "rmsd_threshold_angstrom": float(self.rmsd_threshold_angstrom),
            "posebusters_required": True,
            "rmsd_implementation": "spyrmsd",
            "rmsd_semantics": "same-frame symmetry-aware; no centering or fitting",
            "top5_semantics": "oracle over at most five highest-ranked Vina modes",
        }


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _require_non_negative_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _require_vector(value: Any, name: str, *, positive: bool = False) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    result: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{name} must contain three finite numbers")
        result.append(float(item))
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"{name} must be positive")
    return result


def _artifact_pointer(record: Any, name: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{name} artifact is missing")
    if record.get("access_scope") != "DOCKING_VISIBLE":
        raise ValueError(f"{name} must be docking-visible")
    sha256 = _require_sha256(record.get("sha256"), f"{name}.sha256")
    media_type = record.get("media_type")
    producer = record.get("producer")
    if not isinstance(media_type, str) or not media_type:
        raise ValueError(f"{name}.media_type is missing")
    if not isinstance(producer, str) or not producer:
        raise ValueError(f"{name}.producer is missing")
    producer_version = record.get("producer_version")
    if producer_version is not None and (
        not isinstance(producer_version, str) or not producer_version
    ):
        raise ValueError(f"{name}.producer_version is invalid")
    return {
        "sha256": sha256,
        "media_type": media_type,
        "producer": producer,
        "producer_version": producer_version,
    }


def _require_authorized_artifact(
    pointer: dict[str, Any],
    *,
    name: str,
    media_type: str,
    producer: str,
) -> None:
    if pointer["media_type"] != media_type or pointer["producer"] != producer:
        raise ValueError(f"{name} is not the authorized redocking artifact type")


def _top5_metrics(redock_result: dict[str, Any]) -> dict[str, Any]:
    top1 = redock_result.get("top1")
    if not isinstance(top1, dict):
        raise ValueError("completed redocking result has no top1 metrics")
    top1_pb_valid = _require_bool(
        top1.get("posebusters_valid"), "top1.posebusters_valid"
    )
    top1_rmsd = _require_non_negative_float(
        top1.get("symmetry_rmsd_angstrom"), "top1.symmetry_rmsd_angstrom"
    )
    top1_mode = top1.get("mode")
    if type(top1_mode) is not int or top1_mode != 1:
        raise ValueError("top1 mode must be Vina rank 1")

    modes = redock_result.get("top5_modes")
    if modes is not None:
        if not isinstance(modes, list) or not 1 <= len(modes) <= 5:
            raise ValueError("top5_modes must contain one to five evaluated modes")
        pb_valid_rmsds: list[tuple[int, float]] = []
        for expected_mode, mode in enumerate(modes, start=1):
            if not isinstance(mode, dict) or mode.get("mode") != expected_mode:
                raise ValueError("top5_modes must be ordered contiguous Vina ranks")
            pb_valid = _require_bool(
                mode.get("posebusters_valid"),
                f"top5_modes[{expected_mode}].posebusters_valid",
            )
            rmsd = _require_non_negative_float(
                mode.get("symmetry_rmsd_angstrom"),
                f"top5_modes[{expected_mode}].symmetry_rmsd_angstrom",
            )
            if expected_mode == 1 and (
                pb_valid != top1_pb_valid or rmsd != top1_rmsd
            ):
                raise ValueError("top1 and top5_modes[0] metrics disagree")
            if pb_valid:
                pb_valid_rmsds.append((expected_mode, rmsd))
        best = min(pb_valid_rmsds, key=lambda item: (item[1], item[0]), default=None)
        return {
            "top1_posebusters_valid": top1_pb_valid,
            "top1_symmetry_rmsd_angstrom": top1_rmsd,
            "evaluated_modes": len(modes),
            "pb_valid_mode_count": len(pb_valid_rmsds),
            "pb_valid_mode_count_complete": True,
            "best_pb_valid_mode": best[0] if best is not None else None,
            "best_pb_valid_symmetry_rmsd_angstrom": (
                best[1] if best is not None else None
            ),
        }

    # Legacy schema 1.1 did not persist per-mode metrics.  It is usable only when
    # the recorded oracle-best mode is also explicitly the first recovered
    # (therefore PB-valid) mode; otherwise the PB-valid best RMSD is unknowable.
    oracle = redock_result.get("top5_oracle")
    if not isinstance(oracle, dict):
        raise ValueError("redocking result has no top5 metrics")
    evaluated_modes = oracle.get("evaluated_modes")
    best_mode = oracle.get("best_mode")
    first_recovered = oracle.get("first_recovered_mode")
    any_recovered = _require_bool(
        oracle.get("any_pb_valid_and_rmsd_le_2"),
        "top5_oracle.any_pb_valid_and_rmsd_le_2",
    )
    if (
        type(evaluated_modes) is not int
        or not 1 <= evaluated_modes <= 5
        or type(best_mode) is not int
        or not 1 <= best_mode <= evaluated_modes
    ):
        raise ValueError("legacy top5 oracle ranks are invalid")
    best_rmsd = _require_non_negative_float(
        oracle.get("best_symmetry_rmsd_angstrom"),
        "top5_oracle.best_symmetry_rmsd_angstrom",
    )
    if not any_recovered or first_recovered != best_mode:
        raise ValueError(
            "legacy redocking result cannot identify the best PB-valid top5 RMSD"
        )
    return {
        "top1_posebusters_valid": top1_pb_valid,
        "top1_symmetry_rmsd_angstrom": top1_rmsd,
        "evaluated_modes": evaluated_modes,
        "pb_valid_mode_count": None,
        "pb_valid_mode_count_complete": False,
        "best_pb_valid_mode": best_mode,
        "best_pb_valid_symmetry_rmsd_angstrom": best_rmsd,
    }


def build_known_site_calibration_receipt(
    redock_result: dict[str, Any],
    config: KnownSiteCalibrationConfig,
    *,
    source_result: ArtifactRef,
) -> dict[str, Any]:
    """Build a downstream-safe calibration receipt from one completed redock run."""

    if not isinstance(redock_result, dict):
        raise ValueError("redocking result must be an object")
    if redock_result.get("benchmark") != "redock":
        raise ValueError("calibration source must be a redock benchmark")
    if redock_result.get("status") != "COMPLETED":
        raise ValueError("calibration source redock must be completed")
    source_payload = canonical_json_bytes(redock_result)
    if (
        source_result.media_type != "application/json"
        or source_result.producer != "protbind.redocking.calibration-source"
        or source_result.sha256 != sha256_bytes(source_payload)
        or source_result.size_bytes != len(source_payload)
    ):
        raise ValueError(
            "calibration source_result must be the exact canonical redock result artifact"
        )
    source_redock = {
        "result_artifact": {
            **source_result.to_dict(),
            "access_scope": "VALIDATION_AND_CALIBRATION_ONLY",
        },
        "schema_version": str(redock_result.get("schema_version", "")),
        "run_identity_sha256": _require_sha256(
            redock_result.get("run_identity_sha256"), "run_identity_sha256"
        ),
        "redock_config_sha256": _require_sha256(
            redock_result.get("config_sha256"), "config_sha256"
        ),
        "toolchain_sha256": _require_sha256(
            redock_result.get("toolchain_sha256"), "toolchain_sha256"
        ),
        "code_sha256": _require_sha256(
            redock_result.get("code_sha256"), "code_sha256"
        ),
    }
    artifacts = redock_result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("redocking result artifacts are missing")
    authorized_inputs = {
        "prepared_receptor": _artifact_pointer(
            artifacts.get("prepared_receptor"), "prepared_receptor"
        ),
        "receptor_preparation_receipt": _artifact_pointer(
            artifacts.get("receptor_preparation_receipt"),
            "receptor_preparation_receipt",
        ),
        "known_site_box_receipt": _artifact_pointer(
            artifacts.get("native_box_receipt"), "native_box_receipt"
        ),
        "coordinate_frame": "receptor-cartesian-angstrom",
    }
    metrics = _top5_metrics(redock_result)
    threshold = float(config.rmsd_threshold_angstrom)
    top1_pass = (
        metrics["top1_posebusters_valid"]
        and metrics["top1_symmetry_rmsd_angstrom"] <= threshold
    )
    top5_rmsd = metrics["best_pb_valid_symmetry_rmsd_angstrom"]
    top5_pass = top5_rmsd is not None and top5_rmsd <= threshold
    metrics.update(
        {
            "top1_pass": top1_pass,
            "top5_pass": top5_pass,
        }
    )
    calibration_config = config.to_dict()
    calibration_config_sha256 = sha256_bytes(canonical_json_bytes(calibration_config))
    criterion_met = top1_pass if config.required_rank == "top1" else top5_pass
    core = {
        "schema_version": "1.1",
        "receipt_type": "PROTBIND_TARGET_KNOWN_SITE_CALIBRATION",
        "artifact_scope": "DOCKING_VISIBLE",
        "target_id": config.target_id,
        "screening_mode": "both",
        "source_redock": source_redock,
        "authorized_inputs": authorized_inputs,
        "reference_boundary": dict(_REFERENCE_BOUNDARY),
        "calibration_config": calibration_config,
        "calibration_config_sha256": calibration_config_sha256,
        "metrics": metrics,
        "decision": {
            "status": "PASS" if criterion_met else "FAIL",
            "criterion_met": criterion_met,
            "required_rank": config.required_rank,
            "criterion": (
                "PoseBusters-valid same-frame symmetry RMSD <= "
                f"{threshold:g} A at {config.required_rank}"
            ),
            "authorized_before_screening": criterion_met,
        },
        "disclaimers": list(_DISCLAIMERS),
    }
    receipt = {
        **core,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(core)),
    }
    validate_known_site_calibration_receipt(
        receipt,
        expected_target_id=config.target_id,
        expected_prepared_receptor_sha256=authorized_inputs["prepared_receptor"][
            "sha256"
        ],
        require_pass=False,
    )
    return receipt


def validate_known_site_calibration_receipt(
    receipt: dict[str, Any],
    *,
    expected_target_id: str | None = None,
    expected_prepared_receptor_sha256: str | None = None,
    expected_box_center: tuple[float, float, float] | list[float] | None = None,
    expected_box_size: tuple[float, float, float] | list[float] | None = None,
    require_pass: bool = True,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    """Fail closed on malformed, tampered, failed, or target-mismatched receipts."""

    if not isinstance(receipt, dict):
        raise ValueError("calibration receipt must be an object")
    expected_keys = {
        "schema_version",
        "receipt_type",
        "artifact_scope",
        "target_id",
        "screening_mode",
        "source_redock",
        "authorized_inputs",
        "reference_boundary",
        "calibration_config",
        "calibration_config_sha256",
        "metrics",
        "decision",
        "disclaimers",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("calibration receipt has missing or unauthorized fields")
    if (
        receipt["schema_version"] != "1.1"
        or receipt["receipt_type"] != "PROTBIND_TARGET_KNOWN_SITE_CALIBRATION"
        or receipt["artifact_scope"] != "DOCKING_VISIBLE"
        or receipt["screening_mode"] != "both"
    ):
        raise ValueError("calibration receipt identity or scope is invalid")
    target_id = receipt["target_id"]
    if not isinstance(target_id, str) or not _SAFE_ID.fullmatch(target_id):
        raise ValueError("calibration receipt target_id is invalid")
    if expected_target_id is not None and target_id != expected_target_id:
        raise ValueError("calibration receipt belongs to a different target")

    source = receipt["source_redock"]
    if not isinstance(source, dict) or set(source) != {
        "result_artifact",
        "schema_version",
        "run_identity_sha256",
        "redock_config_sha256",
        "toolchain_sha256",
        "code_sha256",
    }:
        raise ValueError("calibration source redock binding is invalid")
    source_record = source["result_artifact"]
    if not isinstance(source_record, dict) or source_record.get(
        "access_scope"
    ) != "VALIDATION_AND_CALIBRATION_ONLY":
        raise ValueError("calibration source result scope is invalid")
    source_artifact_value = dict(source_record)
    source_artifact_value.pop("access_scope", None)
    try:
        source_artifact = ArtifactRef.from_dict(source_artifact_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibration source result artifact is invalid") from exc
    if (
        source_artifact.media_type != "application/json"
        or source_artifact.producer != "protbind.redocking.calibration-source"
    ):
        raise ValueError("calibration source result artifact provenance is invalid")
    if not isinstance(source["schema_version"], str) or not source["schema_version"]:
        raise ValueError("source redock schema version is missing")
    for key in (
        "run_identity_sha256",
        "redock_config_sha256",
        "toolchain_sha256",
        "code_sha256",
    ):
        _require_sha256(source[key], f"source_redock.{key}")

    source_metrics: dict[str, Any] | None = None
    if (expected_box_center is None) != (expected_box_size is None):
        raise ValueError(
            "expected calibration box center and size must be provided together"
        )
    if store is not None:
        source_value = store.read_json(source_artifact)
        if not isinstance(source_value, dict):
            raise ValueError("calibration source result is not a JSON object")
        if source_value.get("benchmark") != "redock":
            raise ValueError("calibration source artifact is not a redock benchmark")
        if source_value.get("status") != "COMPLETED":
            raise ValueError("calibration source redock is not completed")
        expected_source = {
            "schema_version": str(source_value.get("schema_version", "")),
            "run_identity_sha256": source_value.get("run_identity_sha256"),
            "redock_config_sha256": source_value.get("config_sha256"),
            "toolchain_sha256": source_value.get("toolchain_sha256"),
            "code_sha256": source_value.get("code_sha256"),
        }
        if any(source.get(key) != value for key, value in expected_source.items()):
            raise ValueError("calibration receipt differs from its source redock result")
        source_metrics = _top5_metrics(source_value)

    inputs = receipt["authorized_inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {
        "prepared_receptor",
        "receptor_preparation_receipt",
        "known_site_box_receipt",
        "coordinate_frame",
    }:
        raise ValueError("calibration authorized inputs are invalid")
    if inputs["coordinate_frame"] != "receptor-cartesian-angstrom":
        raise ValueError("calibration coordinate frame is invalid")
    for name in (
        "prepared_receptor",
        "receptor_preparation_receipt",
        "known_site_box_receipt",
    ):
        pointer = inputs[name]
        if not isinstance(pointer, dict) or set(pointer) != {
            "sha256",
            "media_type",
            "producer",
            "producer_version",
        }:
            raise ValueError(f"{name} pointer is invalid")
        _require_sha256(pointer["sha256"], f"{name}.sha256")
        if not isinstance(pointer["media_type"], str) or not pointer["media_type"]:
            raise ValueError(f"{name}.media_type is invalid")
        if not isinstance(pointer["producer"], str) or not pointer["producer"]:
            raise ValueError(f"{name}.producer is invalid")
        if pointer["producer_version"] is not None and (
            not isinstance(pointer["producer_version"], str)
            or not pointer["producer_version"]
        ):
            raise ValueError(f"{name}.producer_version is invalid")
    _require_authorized_artifact(
        inputs["prepared_receptor"],
        name="prepared_receptor",
        media_type="chemical/x-pdb",
        producer="meeko.mk_prepare_receptor",
    )
    _require_authorized_artifact(
        inputs["receptor_preparation_receipt"],
        name="receptor_preparation_receipt",
        media_type="application/json",
        producer="protbind.redocking.meeko-receptor-receipt",
    )
    _require_authorized_artifact(
        inputs["known_site_box_receipt"],
        name="known_site_box_receipt",
        media_type="application/json",
        producer="protbind.redocking.native-box-receipt",
    )
    receptor_sha = inputs["prepared_receptor"]["sha256"]
    if (
        expected_prepared_receptor_sha256 is not None
        and receptor_sha != expected_prepared_receptor_sha256
    ):
        raise ValueError("calibration receipt is bound to a different prepared receptor")
    if store is not None:
        source_artifacts = source_value.get("artifacts")
        if not isinstance(source_artifacts, dict):
            raise ValueError("calibration source redock artifacts are missing")
        source_inputs = {
            "prepared_receptor": _artifact_pointer(
                source_artifacts.get("prepared_receptor"),
                "source prepared_receptor",
            ),
            "receptor_preparation_receipt": _artifact_pointer(
                source_artifacts.get("receptor_preparation_receipt"),
                "source receptor_preparation_receipt",
            ),
            "known_site_box_receipt": _artifact_pointer(
                source_artifacts.get("native_box_receipt"),
                "source native_box_receipt",
            ),
        }
        if any(inputs[name] != pointer for name, pointer in source_inputs.items()):
            raise ValueError(
                "calibration authorized inputs differ from its source redock artifacts"
            )
        artifact_values: dict[str, Any] = {}
        for name, pointer in source_inputs.items():
            try:
                path = store.resolve_sha256(pointer["sha256"])
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    f"calibration authorized {name} artifact is not store-resolvable"
                ) from exc
            if name == "prepared_receptor":
                continue
            try:
                artifact_values[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"calibration authorized {name} artifact is not valid JSON"
                ) from exc
        preparation = artifact_values["receptor_preparation_receipt"]
        if (
            not isinstance(preparation, dict)
            or preparation.get("schema_version") != "1.0"
            or preparation.get("prepared_receptor")
            != f"sha256:{source_inputs['prepared_receptor']['sha256']}"
            or preparation.get("box_receipt")
            != f"sha256:{source_inputs['known_site_box_receipt']['sha256']}"
            or preparation.get("protein_only_input_required") is not True
            or preparation.get("allow_bad_residues") is not False
            or preparation.get("possible_cofactors_silently_removed") is not False
        ):
            raise ValueError(
                "calibration receptor-preparation receipt does not bind the "
                "authorized receptor and box"
            )
        box = artifact_values["known_site_box_receipt"]
        if (
            not isinstance(box, dict)
            or box.get("schema_version") != "1.0"
            or box.get("definition") != "redock-known-site"
            or box.get("native_coordinates_exposed_to_docking") is not False
            or box.get("native_coordinates_used_for_box_derivation") is not True
        ):
            raise ValueError("calibration known-site box receipt semantics are invalid")
        box_center = _require_vector(box.get("center"), "calibration box center")
        box_size = _require_vector(
            box.get("size"), "calibration box size", positive=True
        )
        if expected_box_center is not None:
            requested_center = _require_vector(
                expected_box_center, "expected calibration box center"
            )
            requested_size = _require_vector(
                expected_box_size, "expected calibration box size", positive=True
            )
            if box_center != requested_center or box_size != requested_size:
                raise ValueError(
                    "calibration receipt is bound to a different known-site box"
                )
    if receipt["reference_boundary"] != _REFERENCE_BOUNDARY:
        raise ValueError("calibration reference-ligand boundary was weakened")

    config_payload = receipt["calibration_config"]
    if not isinstance(config_payload, dict):
        raise ValueError("calibration config is missing")
    config_target = config_payload.get("target_id")
    config_rank = config_payload.get("required_rank")
    config_threshold = config_payload.get("rmsd_threshold_angstrom")
    if (
        not isinstance(config_target, str)
        or not isinstance(config_rank, str)
        or isinstance(config_threshold, bool)
        or not isinstance(config_threshold, int | float)
    ):
        raise ValueError("calibration config field types are invalid")
    config = KnownSiteCalibrationConfig(
        target_id=config_target,
        required_rank=config_rank,
        rmsd_threshold_angstrom=config_threshold,
    )
    if config_payload != config.to_dict() or config.target_id != target_id:
        raise ValueError("calibration config is noncanonical or target-mismatched")
    config_sha = sha256_bytes(canonical_json_bytes(config_payload))
    if receipt["calibration_config_sha256"] != config_sha:
        raise ValueError("calibration config hash mismatch")

    metrics = receipt["metrics"]
    metric_keys = {
        "top1_posebusters_valid",
        "top1_symmetry_rmsd_angstrom",
        "evaluated_modes",
        "pb_valid_mode_count",
        "pb_valid_mode_count_complete",
        "best_pb_valid_mode",
        "best_pb_valid_symmetry_rmsd_angstrom",
        "top1_pass",
        "top5_pass",
    }
    if not isinstance(metrics, dict) or set(metrics) != metric_keys:
        raise ValueError("calibration metrics are invalid")
    top1_pb = _require_bool(
        metrics["top1_posebusters_valid"], "metrics.top1_posebusters_valid"
    )
    top1_rmsd = _require_non_negative_float(
        metrics["top1_symmetry_rmsd_angstrom"],
        "metrics.top1_symmetry_rmsd_angstrom",
    )
    evaluated = metrics["evaluated_modes"]
    if type(evaluated) is not int or not 1 <= evaluated <= 5:
        raise ValueError("calibration evaluated_modes is invalid")
    count_complete = _require_bool(
        metrics["pb_valid_mode_count_complete"],
        "metrics.pb_valid_mode_count_complete",
    )
    count = metrics["pb_valid_mode_count"]
    if count_complete:
        if type(count) is not int or not 0 <= count <= evaluated:
            raise ValueError("calibration PB-valid mode count is invalid")
    elif count is not None:
        raise ValueError("incomplete PB-valid mode count must be null")
    best_mode = metrics["best_pb_valid_mode"]
    best_rmsd_value = metrics["best_pb_valid_symmetry_rmsd_angstrom"]
    if (best_mode is None) != (best_rmsd_value is None):
        raise ValueError("best PB-valid mode and RMSD must both be null or present")
    if best_mode is not None and (
        type(best_mode) is not int or not 1 <= best_mode <= evaluated
    ):
        raise ValueError("best PB-valid mode is invalid")
    if count_complete and ((count == 0) != (best_mode is None)):
        raise ValueError("PB-valid mode count and best mode disagree")
    if top1_pb and count_complete and count < 1:
        raise ValueError("PB-valid top1 is missing from the PB-valid mode count")
    best_rmsd = (
        None
        if best_rmsd_value is None
        else _require_non_negative_float(best_rmsd_value, "metrics.best_pb_valid_rmsd")
    )
    if top1_pb and (best_rmsd is None or best_rmsd > top1_rmsd):
        raise ValueError("best PB-valid top5 RMSD cannot be worse than PB-valid top1")
    threshold = float(config.rmsd_threshold_angstrom)
    expected_top1_pass = top1_pb and top1_rmsd <= threshold
    expected_top5_pass = best_rmsd is not None and best_rmsd <= threshold
    if (
        _require_bool(metrics["top1_pass"], "metrics.top1_pass")
        != expected_top1_pass
        or _require_bool(metrics["top5_pass"], "metrics.top5_pass")
        != expected_top5_pass
    ):
        raise ValueError("calibration pass metrics are internally inconsistent")
    if source_metrics is not None:
        recomputed_metrics = {
            **source_metrics,
            "top1_pass": expected_top1_pass,
            "top5_pass": expected_top5_pass,
        }
        if metrics != recomputed_metrics:
            raise ValueError(
                "calibration metrics differ from recomputed source redock metrics"
            )

    decision = receipt["decision"]
    if not isinstance(decision, dict) or set(decision) != {
        "status",
        "criterion_met",
        "required_rank",
        "criterion",
        "authorized_before_screening",
    }:
        raise ValueError("calibration decision is invalid")
    criterion_met = (
        expected_top1_pass if config.required_rank == "top1" else expected_top5_pass
    )
    expected_status = "PASS" if criterion_met else "FAIL"
    expected_criterion = (
        "PoseBusters-valid same-frame symmetry RMSD <= "
        f"{threshold:g} A at {config.required_rank}"
    )
    if (
        decision["status"] != expected_status
        or decision["criterion_met"] is not criterion_met
        or decision["authorized_before_screening"] is not criterion_met
        or decision["required_rank"] != config.required_rank
        or decision["criterion"] != expected_criterion
    ):
        raise ValueError("calibration gate decision is internally inconsistent")
    if receipt["disclaimers"] != list(_DISCLAIMERS):
        raise ValueError("calibration scientific disclaimers are invalid")

    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected_receipt_sha = sha256_bytes(canonical_json_bytes(core))
    if receipt["receipt_sha256"] != expected_receipt_sha:
        raise ValueError("calibration receipt hash mismatch")
    if require_pass and not criterion_met:
        raise ValueError("target-specific known-site calibration gate did not pass")
    return receipt


def validate_known_site_calibration_artifact(
    store: ArtifactStore,
    calibration: ArtifactRef,
    *,
    expected_target_id: str,
    expected_prepared_receptor_sha256: str,
    expected_box_center: tuple[float, float, float] | list[float],
    expected_box_size: tuple[float, float, float] | list[float],
) -> dict[str, Any]:
    """Validate a stored passing receipt against its exact downstream inputs."""

    if (
        calibration.media_type != "application/json"
        or calibration.producer != "protbind.redocking.known-site-calibration"
    ):
        raise ValueError("known-site calibration artifact provenance is invalid")
    value = store.read_json(calibration)
    if not isinstance(value, dict):
        raise ValueError("known-site calibration artifact is not a JSON object")
    return validate_known_site_calibration_receipt(
        value,
        expected_target_id=expected_target_id,
        expected_prepared_receptor_sha256=expected_prepared_receptor_sha256,
        expected_box_center=expected_box_center,
        expected_box_size=expected_box_size,
        require_pass=True,
        store=store,
    )
