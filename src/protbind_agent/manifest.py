"""Hash-bound run state, cache decisions, and resumable failure records."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes
from .models import ArtifactRef


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RunState(StrEnum):
    CREATED = "CREATED"
    INPUT_VALIDATED = "INPUT_VALIDATED"
    RECEPTOR_READY = "RECEPTOR_READY"
    INDEXED = "INDEXED"
    SCREENED = "SCREENED"
    SELECTED = "SELECTED"
    # COFOLDED is retained as a protocol/task name for schema-1 manifests and
    # worker configuration.  It is deliberately not part of the schema-2 main
    # state machine; cofolding is optional evidence recorded by CofoldStatus.
    COFOLDED = "COFOLDED"
    DOCKED = "DOCKED"
    VALIDATED = "VALIDATED"
    REPORTED = "REPORTED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class CofoldStatus(StrEnum):
    NOT_REQUESTED = "NOT_REQUESTED"
    UNAVAILABLE = "UNAVAILABLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED_RECOVERABLE = "FAILED_RECOVERABLE"


MANIFEST_SCHEMA_VERSION = "2.0"
LEGACY_MANIFEST_SCHEMA_VERSION = "1.0"


STAGE_ORDER = (
    RunState.CREATED,
    RunState.INPUT_VALIDATED,
    RunState.RECEPTOR_READY,
    RunState.INDEXED,
    RunState.SCREENED,
    RunState.SELECTED,
    RunState.DOCKED,
    RunState.VALIDATED,
    RunState.REPORTED,
)

_LEGACY_STAGE_ORDER = (
    RunState.CREATED,
    RunState.INPUT_VALIDATED,
    RunState.INDEXED,
    RunState.SCREENED,
    RunState.COFOLDED,
    RunState.DOCKED,
    RunState.VALIDATED,
    RunState.REPORTED,
)


@dataclass(frozen=True, slots=True)
class StageRecord:
    stage: RunState
    input_hash: str
    config_hash: str
    cache_key: str
    outputs: tuple[ArtifactRef, ...]
    duration_seconds: float
    completed_at: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in {*STAGE_ORDER[1:], RunState.COFOLDED}:
            raise ValueError("stage record must represent a normal executable stage")
        for name, digest in (
            ("input_hash", self.input_hash),
            ("config_hash", self.config_hash),
            ("cache_key", self.cache_key),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.cache_key != stage_cache_key(
            self.stage, self.input_hash, self.config_hash
        ):
            raise ValueError("stage record cache key does not match its input/config")
        if not self.outputs:
            raise ValueError("stage record requires at least one output artifact")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and >= 0")

    @classmethod
    def create(
        cls,
        stage: RunState,
        *,
        input_hash: str,
        config_hash: str,
        outputs: tuple[ArtifactRef, ...],
        duration_seconds: float,
        warnings: tuple[str, ...] = (),
    ) -> StageRecord:
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and >= 0")
        return cls(
            stage=stage,
            input_hash=input_hash,
            config_hash=config_hash,
            cache_key=stage_cache_key(stage, input_hash, config_hash),
            outputs=outputs,
            duration_seconds=duration_seconds,
            completed_at=utc_now(),
            warnings=warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "input_hash": self.input_hash,
            "config_hash": self.config_hash,
            "cache_key": self.cache_key,
            "outputs": [output.to_dict() for output in self.outputs],
            "duration_seconds": self.duration_seconds,
            "completed_at": self.completed_at,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StageRecord:
        payload = dict(value)
        payload["stage"] = RunState(payload["stage"])
        payload["outputs"] = tuple(ArtifactRef.from_dict(item) for item in payload["outputs"])
        payload["warnings"] = tuple(payload.get("warnings", ()))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class FailureRecord:
    stage: RunState
    code: str
    message: str
    recoverable: bool
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FailureRecord:
        return cls(**{**value, "stage": RunState(value["stage"])})


def stage_cache_key(stage: RunState, input_hash: str, config_hash: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "stage": stage.value,
                "input": input_hash,
                "config": config_hash,
            }
        )
    )


@dataclass(slots=True)
class RunManifest:
    run_id: str
    case_id: str
    case_artifact: ArtifactRef
    input_artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    state: RunState = RunState.CREATED
    last_completed_stage: RunState = RunState.CREATED
    stage_records: dict[str, StageRecord] = field(default_factory=dict)
    failures: list[FailureRecord] = field(default_factory=list)
    cofold_status: CofoldStatus = CofoldStatus.NOT_REQUESTED
    cofold_record: StageRecord | None = None
    cofold_failure: FailureRecord | None = None
    provenance: dict[str, str | None] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.state, RunState):
            self.state = RunState(self.state)
        if not isinstance(self.last_completed_stage, RunState):
            self.last_completed_stage = RunState(self.last_completed_stage)
        if not isinstance(self.cofold_status, CofoldStatus):
            self.cofold_status = CofoldStatus(self.cofold_status)
        if self.schema_version not in {
            MANIFEST_SCHEMA_VERSION,
            LEGACY_MANIFEST_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported manifest schema: {self.schema_version}")
        order = (
            STAGE_ORDER
            if self.schema_version == MANIFEST_SCHEMA_VERSION
            else _LEGACY_STAGE_ORDER
        )
        if self.last_completed_stage not in order:
            raise ValueError("last_completed_stage must be a normal workflow stage")
        last_position = order.index(self.last_completed_stage)
        expected_names = {stage.value for stage in order[1 : last_position + 1]}
        if set(self.stage_records) != expected_names:
            raise ValueError("manifest stage records must be contiguous through the last stage")
        for name, record in self.stage_records.items():
            if name != record.stage.value:
                raise ValueError("manifest stage record key does not match its stage")
        if self.state not in {*order, RunState.DEGRADED, RunState.FAILED}:
            raise ValueError("manifest state is not valid for its schema")
        if self.state in order and self.state is not self.last_completed_stage:
            raise ValueError("normal manifest state must equal last_completed_stage")
        if self.state is RunState.DEGRADED and (
            not self.failures or not self.failures[-1].recoverable
        ):
            raise ValueError("degraded manifest requires a recoverable failure record")
        if self.state is RunState.FAILED and (
            not self.failures or self.failures[-1].recoverable
        ):
            raise ValueError("failed manifest requires a non-recoverable failure record")
        if self.schema_version == MANIFEST_SCHEMA_VERSION:
            if RunState.COFOLDED.value in self.stage_records:
                raise ValueError("schema-2 COFOLDED evidence must not be a main stage record")
            if self.cofold_record is not None and self.cofold_record.stage is not (
                RunState.COFOLDED
            ):
                raise ValueError("cofold_record must represent the COFOLDED side task")
            if self.cofold_failure is not None and self.cofold_failure.stage is not (
                RunState.COFOLDED
            ):
                raise ValueError("cofold_failure must represent the COFOLDED side task")
            if self.cofold_status is CofoldStatus.COMPLETED:
                if self.cofold_record is None or self.cofold_failure is not None:
                    raise ValueError("completed cofold evidence requires exactly one record")
            elif self.cofold_record is not None:
                raise ValueError("cofold_record is only valid when cofold is completed")
            if self.cofold_status in {
                CofoldStatus.UNAVAILABLE,
                CofoldStatus.FAILED_RECOVERABLE,
            }:
                if self.cofold_failure is None or not self.cofold_failure.recoverable:
                    raise ValueError(
                        "unavailable/failed cofold evidence requires a recoverable failure"
                    )
            elif self.cofold_failure is not None:
                raise ValueError(
                    "cofold_failure is only valid for unavailable/recoverable failure"
                )

    @property
    def is_read_only(self) -> bool:
        """Schema-1 manifests remain inspectable but cannot be resumed or rewritten."""

        return self.schema_version != MANIFEST_SCHEMA_VERSION

    def _ensure_writable(self) -> None:
        if self.is_read_only:
            raise ValueError(
                "manifest schema 1.0 is read-only; create a new schema-2 run instead of "
                "resuming or rewriting it"
            )

    @property
    def next_stage(self) -> RunState | None:
        order = STAGE_ORDER if not self.is_read_only else _LEGACY_STAGE_ORDER
        position = order.index(self.last_completed_stage)
        return order[position + 1] if position + 1 < len(order) else None

    def cached_outputs(
        self, stage: RunState, *, input_hash: str, config_hash: str
    ) -> tuple[ArtifactRef, ...] | None:
        record = self.stage_records.get(stage.value)
        expected = stage_cache_key(stage, input_hash, config_hash)
        if record is not None and record.cache_key == expected:
            return record.outputs
        return None

    def complete_stage(self, record: StageRecord) -> None:
        self._ensure_writable()
        if record.stage in {RunState.CREATED, RunState.DEGRADED, RunState.FAILED}:
            raise ValueError(f"cannot complete synthetic stage {record.stage.value}")
        existing = self.stage_records.get(record.stage.value)
        if existing is not None:
            if existing.cache_key != record.cache_key:
                raise ValueError(
                    f"stage {record.stage.value} already has different cached inputs"
                )
            # Idempotently observing an earlier record must never roll a later run
            # backward to that stage.
            return
        if record.stage is not self.next_stage:
            expected = self.next_stage.value if self.next_stage else "none"
            raise ValueError(
                f"invalid transition {self.last_completed_stage.value} -> "
                f"{record.stage.value}; expected {expected}"
            )
        self.stage_records[record.stage.value] = record
        self.state = record.stage
        self.last_completed_stage = record.stage
        self.updated_at = utc_now()

    def begin_cofold(self) -> None:
        self._ensure_writable()
        if self.cofold_status is CofoldStatus.COMPLETED:
            return
        self.cofold_status = CofoldStatus.RUNNING
        self.cofold_record = None
        self.cofold_failure = None
        self.updated_at = utc_now()

    def complete_cofold(self, record: StageRecord) -> None:
        self._ensure_writable()
        if record.stage is not RunState.COFOLDED:
            raise ValueError("optional cofold record must use the COFOLDED task name")
        if self.last_completed_stage not in {
            RunState.SELECTED,
            RunState.DOCKED,
            RunState.VALIDATED,
            RunState.REPORTED,
        }:
            raise ValueError("optional cofold evidence requires completed selection")
        if self.cofold_record is not None:
            if self.cofold_record.cache_key != record.cache_key:
                raise ValueError("cofold evidence already has different cached inputs")
            return
        self.cofold_status = CofoldStatus.COMPLETED
        self.cofold_record = record
        self.cofold_failure = None
        self.updated_at = utc_now()

    def mark_cofold_unavailable(self, *, code: str, message: str) -> None:
        self._mark_cofold_failure(
            status=CofoldStatus.UNAVAILABLE, code=code, message=message
        )

    def mark_cofold_failed(self, *, code: str, message: str) -> None:
        self._mark_cofold_failure(
            status=CofoldStatus.FAILED_RECOVERABLE, code=code, message=message
        )

    def _mark_cofold_failure(
        self, *, status: CofoldStatus, code: str, message: str
    ) -> None:
        self._ensure_writable()
        if status not in {
            CofoldStatus.UNAVAILABLE,
            CofoldStatus.FAILED_RECOVERABLE,
        }:
            raise ValueError("invalid cofold failure status")
        self.cofold_status = status
        self.cofold_record = None
        self.cofold_failure = FailureRecord(
            stage=RunState.COFOLDED,
            code=code,
            message=message,
            recoverable=True,
        )
        self.updated_at = utc_now()

    def degrade(self, *, stage: RunState, code: str, message: str) -> None:
        self._ensure_writable()
        self.failures.append(
            FailureRecord(
                stage=stage,
                code=code,
                message=message,
                recoverable=True,
            )
        )
        self.state = RunState.DEGRADED
        self.updated_at = utc_now()

    def fail(self, *, stage: RunState, code: str, message: str) -> None:
        self._ensure_writable()
        self.failures.append(
            FailureRecord(
                stage=stage,
                code=code,
                message=message,
                recoverable=False,
            )
        )
        self.state = RunState.FAILED
        self.updated_at = utc_now()

    def prepare_resume(self) -> None:
        self._ensure_writable()
        if self.state is RunState.FAILED:
            raise ValueError("failed run is not resumable; create a new run with corrected inputs")
        if self.state is RunState.DEGRADED:
            self.state = self.last_completed_stage
            self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "case_artifact": self.case_artifact.to_dict(),
            "input_artifacts": {
                name: artifact.to_dict()
                for name, artifact in sorted(self.input_artifacts.items())
            },
            "artifacts": {
                name: artifact.to_dict()
                for name, artifact in sorted(self.artifacts.items())
            },
            "state": self.state.value,
            "last_completed_stage": self.last_completed_stage.value,
            "stage_records": {
                name: record.to_dict() for name, record in sorted(self.stage_records.items())
            },
            "failures": [failure.to_dict() for failure in self.failures],
            "provenance": dict(sorted(self.provenance.items())),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.schema_version == MANIFEST_SCHEMA_VERSION:
            result["cofold"] = {
                "status": self.cofold_status.value,
                "record": (
                    self.cofold_record.to_dict()
                    if self.cofold_record is not None
                    else None
                ),
                "failure": (
                    self.cofold_failure.to_dict()
                    if self.cofold_failure is not None
                    else None
                ),
            }
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunManifest:
        schema_version = value["schema_version"]
        cofold = value.get("cofold", {})
        if not isinstance(cofold, dict):
            raise ValueError("manifest cofold side-task state must be an object")
        return cls(
            schema_version=schema_version,
            run_id=value["run_id"],
            case_id=value["case_id"],
            case_artifact=ArtifactRef.from_dict(value["case_artifact"]),
            input_artifacts={
                name: ArtifactRef.from_dict(artifact)
                for name, artifact in value.get("input_artifacts", {}).items()
            },
            artifacts={
                name: ArtifactRef.from_dict(artifact)
                for name, artifact in value.get("artifacts", {}).items()
            },
            state=RunState(value["state"]),
            last_completed_stage=RunState(value["last_completed_stage"]),
            stage_records={
                name: StageRecord.from_dict(record)
                for name, record in value.get("stage_records", {}).items()
            },
            failures=[FailureRecord.from_dict(item) for item in value.get("failures", ())],
            cofold_status=CofoldStatus(
                cofold.get("status", CofoldStatus.NOT_REQUESTED.value)
            ),
            cofold_record=(
                StageRecord.from_dict(cofold["record"])
                if cofold.get("record") is not None
                else None
            ),
            cofold_failure=(
                FailureRecord.from_dict(cofold["failure"])
                if cofold.get("failure") is not None
                else None
            ),
            provenance=dict(value.get("provenance", {})),
            created_at=value["created_at"],
            updated_at=value["updated_at"],
        )


class ManifestStore:
    """Atomic local manifest storage; paths are derived only from safe run IDs."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @staticmethod
    def _validate_id(run_id: str) -> None:
        if not run_id or len(run_id) > 128 or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in run_id
        ):
            raise ValueError("run_id must be a safe 1-128 character identifier")

    def path_for(self, run_id: str) -> Path:
        self._validate_id(run_id)
        return self.root / "runs" / run_id / "manifest.json"

    def save(self, manifest: RunManifest) -> Path:
        if manifest.is_read_only:
            raise ValueError(
                "manifest schema 1.0 is read-only and cannot be rewritten in place"
            )
        path = self.path_for(manifest.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8") + b"\n"
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load(self, run_id: str) -> RunManifest:
        path = self.path_for(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"run manifest not found: {run_id}")
        manifest = RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if manifest.run_id != run_id:
            raise ValueError("manifest path and embedded run_id do not match")
        return manifest
