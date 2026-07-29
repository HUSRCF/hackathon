"""Restricted MCP facade for interactive local ProtBind clients."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .capabilities import doctor_report
from .caseio import ingest_case
from .control import StageGateController
from .dossier import (
    build_run_dossier,
    dossier_content,
    persist_run_dossier,
)
from .models import ArtifactRef
from .pose_view import build_pose_scene_summary
from .workflow import PipelineConfig, ProtBindWorkflow

_REPORT_CHARACTER_LIMIT = 32_000
_PATH_FIELDS = {
    "structure_file",
    "pharmacophore_file",
    "site_derivation_source_files",
}


class ProtBindMCPService:
    """Host methods with no arbitrary shell, filesystem, or network surface."""

    def __init__(
        self,
        *,
        workspace: Path,
        project_root: Path,
        config: PipelineConfig | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.project_root = project_root.resolve()
        self.workflow = ProtBindWorkflow(self.workspace, config=config)
        self.controller = StageGateController(self.workflow)

    def _project_file(self, value: str, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty project-relative path")
        raw = Path(value)
        if raw.is_absolute():
            raise ValueError(f"{name} must be project-relative")
        path = (self.project_root / raw).resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError(f"{name} escapes the configured project root")
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {raw.name}")
        return path

    def _validate_nested_case_paths(self, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key not in _PATH_FIELDS:
                    self._validate_nested_case_paths(item)
                    continue
                paths = item if key == "site_derivation_source_files" else [item]
                if not isinstance(paths, list) or any(
                    not isinstance(path, str) for path in paths
                ):
                    raise ValueError(f"{key} must contain project-relative paths")
                for path in paths:
                    self._project_file(path, key)
        elif isinstance(value, list):
            for item in value:
                self._validate_nested_case_paths(item)

    def doctor(self) -> dict[str, Any]:
        """Return path-free local capability and Radeon admission evidence."""

        return doctor_report()

    def case_create(
        self,
        *,
        case_path: str,
        index_path: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an offline run and return its first stage gate.

        Both paths and every nested case ``*_file`` input must remain inside the
        configured project root. Network-enabled case policies are rejected by
        this MCP surface; authorized imports stay a separate user-mediated flow.
        """

        case_file = self._project_file(case_path, "case_path")
        index_file = self._project_file(index_path, "index_path")
        raw_case = json.loads(case_file.read_text(encoding="utf-8"))
        self._validate_nested_case_paths(raw_case)
        case = ingest_case(
            case_file,
            self.workflow.artifacts,
            input_root=self.project_root,
        )
        if case.privacy.network_allowed or case.privacy.sequence_upload_allowed:
            raise ValueError(
                "MCP case_create is offline-only; perform approved network resolution "
                "outside this tool and pass content-addressed local inputs"
            )
        manifest = self.workflow.create(case, index_file, run_id=run_id)
        return {
            "created": True,
            "run_id": manifest.run_id,
            "case_id": manifest.case_id,
            "case_file_sha256": sha256_file(case_file),
            "index_file_sha256": sha256_file(index_file),
            "stage_gate": self.controller.inspect(manifest.run_id),
        }

    def case_status(self, *, run_id: str) -> dict[str, Any]:
        """Re-audit a run and return the current preflight gate."""

        return self.controller.inspect(run_id)

    def case_advance(
        self,
        *,
        run_id: str,
        continuation_token: str,
    ) -> dict[str, Any]:
        """Execute and postflight exactly one main stage."""

        return self.controller.advance(run_id, continuation_token)

    def case_attach_support(
        self,
        *,
        run_id: str,
        name: str,
        project_path: str,
        media_type: str,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Attach one project-local support file through core stage-freeze rules."""

        path = self._project_file(project_path, "project_path")
        with self.controller.store.lock(run_id):
            manifest = self.workflow.manifests.load(run_id)
            reference = self.workflow.attach_support(
                manifest,
                name,
                path,
                media_type=media_type,
                replace=replace,
            )
            return {
                "run_id": run_id,
                "support_name": name,
                "artifact": reference.to_dict(),
                "stage_gate": self.controller.inspect(run_id),
            }

    def case_report(self, *, run_id: str, format: str = "markdown") -> dict[str, Any]:
        """Return a bounded deterministic report; never return structure coordinates."""

        keys = {
            "markdown": "report_markdown",
            "html": "report_html",
            "degraded": "degraded_report",
        }
        if format not in keys:
            raise ValueError("format must be markdown, html, or degraded")
        manifest = self.workflow.manifests.load(run_id)
        reference = manifest.artifacts.get(keys[format])
        if reference is None:
            return {
                "run_id": run_id,
                "available": False,
                "format": format,
                "state": manifest.state.value,
                "last_completed_stage": manifest.last_completed_stage.value,
            }
        content = self.workflow.artifacts.read_bytes(reference).decode(
            "utf-8", errors="replace"
        )
        truncated = len(content) > _REPORT_CHARACTER_LIMIT
        if truncated:
            content = content[:_REPORT_CHARACTER_LIMIT]
        return {
            "run_id": run_id,
            "available": True,
            "format": format,
            "artifact": reference.to_dict(),
            "content": content,
            "truncated": truncated,
        }

    def case_dossier(self, *, run_id: str, format: str = "markdown") -> dict[str, Any]:
        """Build a detailed stage/control/artifact dossier at the current checkpoint."""

        if format not in {"json", "markdown", "html"}:
            raise ValueError("format must be json, markdown, or html")
        manifest = self.workflow.manifests.load(run_id)
        self.workflow.audit_manifest(manifest)
        try:
            pose_summary = build_pose_scene_summary(manifest, self.workflow.artifacts)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            pose_summary = {
                "available": False,
                "reason": f"Pose scene audit failed: {type(exc).__name__}: {exc}",
                "candidate_count": 0,
                "geometry_summary_count": 0,
                "candidates": [],
            }
        dossier = build_run_dossier(
            manifest,
            self.workflow.artifacts,
            control_history=self.controller.store.read(run_id),
            pose_summary=pose_summary,
        )
        references = persist_run_dossier(dossier, self.workflow.artifacts)
        content = dossier_content(dossier, format)
        truncated = len(content) > _REPORT_CHARACTER_LIMIT
        if truncated:
            content = content[:_REPORT_CHARACTER_LIMIT]
        return {
            "run_id": run_id,
            "available": True,
            "format": format,
            "artifact": references[format].to_dict(),
            "all_formats": {
                name: reference.to_dict() for name, reference in references.items()
            },
            "content": content,
            "truncated": truncated,
            "coordinates_disclosed": False,
        }

    def case_pose_view(self, *, run_id: str) -> dict[str, Any]:
        """Return coordinate-free docking scene QA for Agent interpretation."""

        manifest = self.workflow.manifests.load(run_id)
        self.workflow.audit_manifest(manifest)
        return build_pose_scene_summary(manifest, self.workflow.artifacts)

    def artifact_metadata(self, *, artifact_ref_json: str) -> dict[str, Any]:
        """Verify one caller-supplied ArtifactRef and return metadata, never bytes."""

        value = json.loads(artifact_ref_json)
        if not isinstance(value, dict):
            raise ValueError("artifact_ref_json must contain an ArtifactRef object")
        reference = ArtifactRef.from_dict(value)
        self.workflow.artifacts.resolve(reference)
        return {
            "verified": True,
            "artifact": reference.to_dict(),
            "coordinates_disclosed": False,
        }

    def control_history(self, *, run_id: str) -> dict[str, Any]:
        """Return content-addressed gate/acceptance receipt references."""

        self.workflow.manifests.load(run_id)
        return self.controller.store.read(run_id)


def create_mcp_server(service: ProtBindMCPService) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "MCP support requires the optional dependency 'mcp>=1.14,<2'"
        ) from exc

    server = FastMCP(
        "ProtBind",
        instructions=(
            "Use stage gates. Call case_status before case_advance, execute exactly one "
            "stage per continuation token, and never invent scientific outputs."
        ),
        json_response=True,
    )

    server.tool(name="doctor", structured_output=True)(service.doctor)
    server.tool(name="case_create", structured_output=True)(service.case_create)
    server.tool(name="case_status", structured_output=True)(service.case_status)
    server.tool(name="case_advance", structured_output=True)(service.case_advance)
    server.tool(name="case_attach_support", structured_output=True)(
        service.case_attach_support
    )
    server.tool(name="case_report", structured_output=True)(service.case_report)
    server.tool(name="case_dossier", structured_output=True)(service.case_dossier)
    server.tool(name="case_pose_view", structured_output=True)(service.case_pose_view)
    server.tool(name="artifact_metadata", structured_output=True)(
        service.artifact_metadata
    )
    server.tool(name="control_history", structured_output=True)(
        service.control_history
    )

    @server.resource(
        "protbind://runs/{run_id}/control",
        name="run-control-history",
        mime_type="application/json",
    )
    def run_control_history(run_id: str) -> str:
        return json.dumps(
            service.control_history(run_id=run_id),
            ensure_ascii=False,
            sort_keys=True,
        )

    return server


@asynccontextmanager
async def _nonblocking_stdio_server() -> AsyncIterator[tuple[Any, Any]]:
    """MCP stdio transport that does not delegate pipe I/O to worker threads.

    The stable MCP Python SDK 1.14 wraps ``sys.stdin`` and ``sys.stdout`` with
    ``anyio.AsyncFile``. Some restricted AIAA/container environments can create
    the helper thread but never dispatch its queued pipe read, leaving MCP
    initialization blocked forever. File-descriptor readiness is sufficient for
    local POSIX stdio and keeps the SDK's protocol/session implementation intact.
    """

    try:
        import anyio
        import mcp.types as mcp_types
        from mcp.shared.message import SessionMessage
    except ImportError as exc:
        raise RuntimeError(
            "MCP support requires the optional dependency 'mcp>=1.14,<2'"
        ) from exc

    input_writer, input_reader = anyio.create_memory_object_stream(0)
    output_writer, output_reader = anyio.create_memory_object_stream(0)
    stdin_fd = 0
    stdout_fd = 1
    stdin_blocking = os.get_blocking(stdin_fd)
    stdout_blocking = os.get_blocking(stdout_fd)
    os.set_blocking(stdin_fd, False)
    os.set_blocking(stdout_fd, False)

    async def read_stdin() -> None:
        buffered = b""
        try:
            async with input_writer:
                while True:
                    await anyio.wait_readable(stdin_fd)
                    try:
                        chunk = os.read(stdin_fd, 65_536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        if buffered.strip():
                            await input_writer.send(
                                ValueError("MCP stdio input ended before a newline")
                            )
                        return
                    lines = (buffered + chunk).split(b"\n")
                    buffered = lines.pop()
                    for raw_line in lines:
                        if not raw_line.strip():
                            continue
                        try:
                            message = mcp_types.JSONRPCMessage.model_validate_json(
                                raw_line
                            )
                        except Exception as exc:  # noqa: BLE001 - protocol error item
                            await input_writer.send(exc)
                            continue
                        await input_writer.send(SessionMessage(message))
        except (anyio.ClosedResourceError, anyio.EndOfStream):
            return

    async def write_bytes(data: bytes) -> None:
        view = memoryview(data)
        while view:
            await anyio.wait_writable(stdout_fd)
            try:
                written = os.write(stdout_fd, view)
            except BlockingIOError:
                continue
            view = view[written:]

    async def write_stdout() -> None:
        try:
            async with output_reader:
                async for session_message in output_reader:
                    encoded = (
                        session_message.message.model_dump_json(
                            by_alias=True,
                            exclude_none=True,
                        ).encode("utf-8")
                        + b"\n"
                    )
                    await write_bytes(encoded)
        except (BrokenPipeError, anyio.ClosedResourceError, anyio.EndOfStream):
            return

    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(read_stdin)
            tasks.start_soon(write_stdout)
            try:
                yield input_reader, output_writer
            finally:
                tasks.cancel_scope.cancel()
    finally:
        os.set_blocking(stdin_fd, stdin_blocking)
        os.set_blocking(stdout_fd, stdout_blocking)


def serve_mcp(
    *,
    workspace: Path,
    project_root: Path,
    config: PipelineConfig | None = None,
    transport: str = "stdio",
) -> None:
    if transport != "stdio":
        raise ValueError("ProtBind MCP currently permits only local stdio transport")
    try:
        import anyio
    except ImportError as exc:
        raise RuntimeError(
            "MCP support requires the optional dependency 'mcp>=1.14,<2'"
        ) from exc
    service = ProtBindMCPService(
        workspace=workspace,
        project_root=project_root,
        config=config,
    )
    server = create_mcp_server(service)

    async def run() -> None:
        async with _nonblocking_stdio_server() as (read_stream, write_stream):
            await server._mcp_server.run(  # noqa: SLF001 - stable SDK 1.x facade
                read_stream,
                write_stream,
                server._mcp_server.create_initialization_options(),  # noqa: SLF001
            )

    anyio.run(run)
