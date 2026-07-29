"""Versioned JSON subprocess contract for isolated model/science environments."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .models import ArtifactRef
from .privacy import redact_text

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class WorkerProvenance:
    model_revision: str
    weight_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        if not self.model_revision.strip():
            raise ValueError("model_revision cannot be empty")
        for label, digest in (
            ("weight_sha256", self.weight_sha256),
            ("code_sha256", self.code_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "model_revision": self.model_revision,
            "weight_sha256": self.weight_sha256,
            "code_sha256": self.code_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkerProvenance:
        return cls(
            model_revision=str(value["model_revision"]),
            weight_sha256=str(value["weight_sha256"]),
            code_sha256=str(value["code_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    job_id: str
    engine: str
    input: ArtifactRef
    parameters: dict[str, Any]
    seed: int
    provenance: WorkerProvenance
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported worker request schema: {self.schema_version}")
        if not self.job_id or not self.engine:
            raise ValueError("worker job_id and engine are required")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not (
            0 <= self.seed <= 2**32 - 1
        ):
            raise ValueError("worker seed must fit in an unsigned 32-bit integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "engine": self.engine,
            "input": self.input.to_dict(),
            "parameters": self.parameters,
            "seed": self.seed,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkerRequest:
        provenance = value["provenance"]
        seed = value["seed"]
        parameters = value.get("parameters", {})
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("worker seed must be an integer")
        if not isinstance(parameters, dict):
            raise ValueError("worker parameters must be an object")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            job_id=str(value["job_id"]),
            engine=str(value["engine"]),
            input=ArtifactRef.from_dict(value["input"]),
            parameters=dict(parameters),
            seed=seed,
            provenance=WorkerProvenance.from_dict(provenance),
        )


@dataclass(frozen=True, slots=True)
class WorkerError:
    code: str
    message: str
    recoverable: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkerError:
        recoverable = value.get("recoverable", False)
        if not isinstance(recoverable, bool):
            raise ValueError("worker error recoverable must be boolean")
        return cls(
            code=str(value["code"]),
            message=redact_text(str(value["message"])),
            recoverable=recoverable,
        )


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    job_id: str
    engine: str
    outputs: tuple[ArtifactRef, ...] = ()
    provenance: WorkerProvenance | None = None
    timings_seconds: dict[str, float] = field(default_factory=dict)
    peak_vram_bytes: int | None = None
    warnings: tuple[str, ...] = ()
    error: WorkerError | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported worker response schema: {self.schema_version}")
        if self.error is None and not self.outputs:
            raise ValueError("successful worker response requires at least one output artifact")
        if self.error is None and self.provenance is None:
            raise ValueError("successful worker response requires verified provenance")
        if self.error is not None and self.outputs:
            raise ValueError("failed worker response cannot also claim output artifacts")
        if self.error is not None and self.provenance is not None:
            raise ValueError("failed worker response cannot claim verified provenance")
        if self.peak_vram_bytes is not None and (
            not isinstance(self.peak_vram_bytes, int)
            or isinstance(self.peak_vram_bytes, bool)
            or self.peak_vram_bytes < 0
        ):
            raise ValueError("peak_vram_bytes must be a non-negative integer or null")
        if any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            for value in self.timings_seconds.values()
        ):
            raise ValueError("worker timings must be finite and non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkerResponse:
        raw_timings = value.get("timings_seconds", {})
        raw_peak = value.get("peak_vram_bytes")
        if not isinstance(raw_timings, dict) or any(
            not isinstance(duration, int | float) or isinstance(duration, bool)
            for duration in raw_timings.values()
        ):
            raise ValueError("worker timings must contain numeric values")
        if raw_peak is not None and (
            not isinstance(raw_peak, int) or isinstance(raw_peak, bool)
        ):
            raise ValueError("peak_vram_bytes must be an integer or null")
        return cls(
            schema_version=value.get("schema_version", ""),
            job_id=str(value["job_id"]),
            engine=str(value["engine"]),
            outputs=tuple(ArtifactRef.from_dict(item) for item in value.get("outputs", ())),
            provenance=(
                WorkerProvenance.from_dict(value["provenance"])
                if value.get("provenance") is not None
                else None
            ),
            timings_seconds={
                str(name): float(duration)
                for name, duration in raw_timings.items()
            },
            peak_vram_bytes=(
                raw_peak if raw_peak is not None else None
            ),
            warnings=tuple(redact_text(str(item)) for item in value.get("warnings", ())),
            error=(WorkerError.from_dict(value["error"]) if value.get("error") else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "engine": self.engine,
            "outputs": [output.to_dict() for output in self.outputs],
            "provenance": (
                self.provenance.to_dict() if self.provenance is not None else None
            ),
            "timings_seconds": self.timings_seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
            "warnings": list(self.warnings),
            "error": (
                {
                    "code": self.error.code,
                    "message": self.error.message,
                    "recoverable": self.error.recoverable,
                }
                if self.error is not None
                else None
            ),
        }


class WorkerExecutionError(RuntimeError):
    pass


class JsonSubprocessWorker:
    """Run an explicit argv without a shell and parse exactly one JSON response."""

    def __init__(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: float = 3600.0,
        environment: dict[str, str] | None = None,
        artifact_root: Path | None = None,
        isolate_network: bool = False,
    ) -> None:
        if not argv or any(not argument for argument in argv):
            raise ValueError("worker argv cannot be empty")
        if (
            not isinstance(timeout_seconds, int | float)
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("worker timeout must be positive")
        self.argv = argv
        self.timeout_seconds = timeout_seconds
        self.environment = environment or {}
        self.artifact_root = artifact_root
        self.isolate_network = isolate_network

    def _environment(self, worker_root: Path | None) -> dict[str, str]:
        # Do not implicitly forward API keys or the full parent environment.
        allowed = (
            "PATH",
            "LD_LIBRARY_PATH",
            "ROCM_PATH",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
        )
        environment = {name: os.environ[name] for name in allowed if name in os.environ}
        environment.update(self.environment)
        # ROCm accepts CUDA_VISIBLE_DEVICES as a compatibility alias. Forwarding an
        # ambient value alongside an explicit HIP_VISIBLE_DEVICES assignment makes
        # child selection ambiguous and can desynchronise the host lease from the
        # GPU used by the worker. HIP_VISIBLE_DEVICES is authoritative here.
        if environment.get("HIP_VISIBLE_DEVICES"):
            environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment["PROTBIND_NETWORK_POLICY"] = "deny"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        # A worker must execute against its pinned environment, never packages
        # inherited from the invoking account's user site.  HOME isolation alone
        # is insufficient because Python may have already selected a user site
        # during interpreter startup.
        environment["PYTHONNOUSERSITE"] = "1"
        if worker_root is not None:
            environment["PROTBIND_ARTIFACT_ROOT"] = str(worker_root.resolve())
            environment["HOME"] = str((worker_root / "home").resolve())
            environment["TMPDIR"] = str((worker_root / "tmp").resolve())
        return environment

    def _argv(self, worker_root: Path | None) -> tuple[str, ...]:
        if not self.isolate_network:
            return self.argv
        if worker_root is None:
            raise WorkerExecutionError(
                "OS network isolation requires an artifact workspace"
            )
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise WorkerExecutionError(
                "bubblewrap is required for OS-level worker network isolation"
            )
        root = str(worker_root.resolve())
        return (
            bwrap,
            "--die-with-parent",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--proc",
            "/proc",
            "--bind",
            root,
            root,
            "--chdir",
            str(Path.cwd()),
            "--",
            *self.argv,
        )

    @staticmethod
    def _nested_artifact(value: Any) -> ArtifactRef | None:
        if not isinstance(value, dict):
            return None
        required = {"sha256", "media_type", "size_bytes", "producer"}
        allowed = required | {"producer_version", "source", "license"}
        if not required.issubset(value) or not set(value).issubset(allowed):
            return None
        try:
            return ArtifactRef.from_dict(value)
        except (TypeError, ValueError):
            return None

    def _copy_dependency_graph(
        self,
        reference: ArtifactRef,
        source: ArtifactStore,
        destination: ArtifactStore,
        seen: set[str],
    ) -> None:
        if reference.sha256 in seen:
            return
        seen.add(reference.sha256)
        data = source.read_bytes(reference)
        copied = destination.put_bytes(
            data,
            media_type=reference.media_type,
            producer=reference.producer,
            producer_version=reference.producer_version,
            source=reference.source,
            license=reference.license,
        )
        if copied != reference:
            raise WorkerExecutionError("artifact metadata changed while staging worker input")
        if reference.media_type != "application/json":
            return
        try:
            value = json.loads(data)
        except json.JSONDecodeError as exc:
            raise WorkerExecutionError(
                "an application/json dependency contains invalid JSON"
            ) from exc

        def visit(item: Any) -> None:
            nested = self._nested_artifact(item)
            if nested is not None:
                self._copy_dependency_graph(
                    nested, source, destination, seen
                )
                return
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)

    def _stage_exchange(self, request: WorkerRequest, worker_root: Path) -> None:
        if self.artifact_root is None:
            return
        worker_root.mkdir(parents=True, exist_ok=True)
        (worker_root / "home").mkdir()
        (worker_root / "tmp").mkdir()
        self._copy_dependency_graph(
            request.input,
            ArtifactStore(self.artifact_root),
            ArtifactStore(worker_root),
            set(),
        )

    def _import_outputs(
        self, response: WorkerResponse, worker_root: Path | None
    ) -> None:
        if self.artifact_root is None or worker_root is None or response.error is not None:
            return
        source = ArtifactStore(worker_root)
        destination = ArtifactStore(self.artifact_root)
        for reference in response.outputs:
            data = source.read_bytes(reference)
            imported = destination.put_bytes(
                data,
                media_type=reference.media_type,
                producer=reference.producer,
                producer_version=reference.producer_version,
                source=reference.source,
                license=reference.license,
            )
            if imported != reference:
                raise WorkerExecutionError(
                    "worker output metadata changed while importing its artifact"
                )

    def _finish(
        self,
        completed: subprocess.CompletedProcess[str],
        request: WorkerRequest,
        worker_root: Path | None,
        elapsed: float,
    ) -> tuple[WorkerResponse, float]:
        if completed.returncode != 0:
            detail = redact_text((completed.stderr or completed.stdout or "").strip())
            detail = detail[-2000:] if detail else "no diagnostic output"
            raise WorkerExecutionError(
                f"worker exited with status {completed.returncode}: {detail}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise WorkerExecutionError(
                "worker must emit exactly one non-empty JSON line on stdout"
            )
        try:
            value = json.loads(lines[0])
            if not isinstance(value, dict):
                raise TypeError("response is not an object")
            response = WorkerResponse.from_dict(value)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkerExecutionError(f"invalid worker response: {exc}") from exc
        if response.job_id != request.job_id or response.engine != request.engine:
            raise WorkerExecutionError("worker response job_id/engine does not match request")
        if response.error is None and response.provenance != request.provenance:
            raise WorkerExecutionError(
                "worker response provenance does not match the request"
            )
        self._import_outputs(response, worker_root)
        return response, elapsed

    def run(self, request: WorkerRequest) -> tuple[WorkerResponse, float]:
        started = time.perf_counter()
        temporary: tempfile.TemporaryDirectory[str] | None = None
        worker_root: Path | None = None
        try:
            if self.artifact_root is not None:
                temporary = tempfile.TemporaryDirectory(prefix="protbind-worker-")
                worker_root = Path(temporary.name)
                self._stage_exchange(request, worker_root)
            completed = subprocess.run(
                self._argv(worker_root),
                input=json.dumps(request.to_dict(), ensure_ascii=False) + "\n",
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=self._environment(worker_root),
            )
        except (OSError, subprocess.TimeoutExpired, WorkerExecutionError) as exc:
            if temporary is not None:
                temporary.cleanup()
            raise WorkerExecutionError(redact_text(f"worker launch failed: {exc}")) from exc
        try:
            return self._finish(
                completed,
                request,
                worker_root,
                time.perf_counter() - started,
            )
        finally:
            if temporary is not None:
                temporary.cleanup()
