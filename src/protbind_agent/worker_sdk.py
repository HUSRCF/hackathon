"""Helpers for implementing a model-specific worker in its isolated environment."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from .artifacts import ArtifactStore
from .privacy import redact_text
from .worker_protocol import WorkerError, WorkerRequest, WorkerResponse

WorkerHandler = Callable[[WorkerRequest, ArtifactStore], WorkerResponse]


class WorkerFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, recoverable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


def serve_worker(engine: str, handler: WorkerHandler) -> int:
    """Read one request and emit one response; diagnostics go to stderr only."""

    request: WorkerRequest | None = None
    try:
        lines = [line for line in sys.stdin.read().splitlines() if line.strip()]
        if len(lines) != 1:
            raise WorkerFailure(
                "PROTOCOL_ERROR",
                "worker expects exactly one non-empty request line",
                recoverable=False,
            )
        value = json.loads(lines[0])
        if not isinstance(value, dict):
            raise TypeError("request is not a JSON object")
        request = WorkerRequest.from_dict(value)
        if request.engine != engine:
            raise WorkerFailure(
                "ENGINE_MISMATCH",
                f"worker implements {engine!r}, request selected {request.engine!r}",
                recoverable=False,
            )
        if os.environ.get("PROTBIND_NETWORK_POLICY") != "deny":
            raise WorkerFailure(
                "OFFLINE_POLICY_VIOLATION",
                "scientific workers require PROTBIND_NETWORK_POLICY=deny",
                recoverable=False,
            )
        artifact_root = os.environ.get("PROTBIND_ARTIFACT_ROOT")
        if not artifact_root:
            raise WorkerFailure(
                "PROTOCOL_ERROR",
                "PROTBIND_ARTIFACT_ROOT was not supplied by the host",
                recoverable=False,
            )
        store = ArtifactStore(Path(artifact_root))
        store.resolve(request.input)
        # Third-party scientific libraries occasionally print progress or
        # warnings to stdout.  Keep stdout reserved for the single-line JSON
        # protocol and route such diagnostics to the captured stderr channel.
        with contextlib.redirect_stdout(sys.stderr):
            response = handler(request, store)
        if response.job_id != request.job_id or response.engine != engine:
            raise WorkerFailure(
                "PROTOCOL_ERROR",
                "handler returned mismatched job_id or engine",
                recoverable=False,
            )
    except WorkerFailure as exc:
        response = WorkerResponse(
            job_id=request.job_id if request else "invalid-request",
            engine=request.engine if request else engine,
            error=WorkerError(
                code=exc.code,
                message=redact_text(str(exc)),
                recoverable=exc.recoverable,
            ),
        )
    except Exception as exc:  # worker boundary must return a structured failure
        response = WorkerResponse(
            job_id=request.job_id if request else "invalid-request",
            engine=request.engine if request else engine,
            error=WorkerError(
                code="WORKER_EXCEPTION",
                message=redact_text(f"{type(exc).__name__}: {exc}"),
                recoverable=False,
            ),
        )
    print(json.dumps(response.to_dict(), ensure_ascii=False, separators=(",", ":")))
    return 0
