"""Restricted MCP facade for interactive local ProtBind clients."""

from __future__ import annotations

import json
import os
import threading
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
from .drutai import DrutAIManager
from .experimental_assays import ExperimentalAssayStore
from .knowledge import (
    SeekDBKnowledgeStore,
    extract_document_bytes,
    import_document,
    inspect_embedding_model,
    sync_library_rag,
)
from .library import LibraryManager, load_library_config
from .models import ArtifactRef
from .pose_view import build_pose_scene_summary
from .public_data import (
    PUBLIC_DATA_SOURCES,
    PublicDataFetcher,
    materialize_public_fetch,
    validate_public_output,
)
from .workflow import PipelineConfig, ProtBindWorkflow

_REPORT_CHARACTER_LIMIT = 32_000
_PATH_FIELDS = {
    "structure_file",
    "pharmacophore_file",
    "site_derivation_source_files",
}


class ProtBindMCPService:
    """Host methods with no arbitrary shell, filesystem, URL, or open network surface."""

    def __init__(
        self,
        *,
        workspace: Path,
        project_root: Path,
        config: PipelineConfig | None = None,
        library_config: Path | None = None,
        knowledge_model: Path | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.project_root = project_root.resolve()
        self.workflow = ProtBindWorkflow(self.workspace, config=config)
        self.controller = StageGateController(self.workflow)
        self.library = (
            LibraryManager(load_library_config(library_config))
            if library_config is not None and library_config.is_file()
            else None
        )
        self.knowledge_model = (
            knowledge_model.resolve() if knowledge_model is not None else None
        )
        self._knowledge_store_instance: SeekDBKnowledgeStore | None = None
        self._knowledge_store_lock = threading.Lock()

    @staticmethod
    def _require_data_access_confirmation(data_access_confirmed: bool) -> None:
        if data_access_confirmed is not True:
            raise PermissionError(
                "private data access requires a fresh explicit user confirmation"
            )

    def _library(self, data_access_confirmed: bool) -> LibraryManager:
        self._require_data_access_confirmation(data_access_confirmed)
        if self.library is None:
            raise RuntimeError(
                "private libraries are not configured; an operator must start the MCP "
                "server with --library-config"
            )
        return self.library

    def _knowledge_store(self, data_access_confirmed: bool) -> SeekDBKnowledgeStore:
        self._require_data_access_confirmation(data_access_confirmed)
        if self.knowledge_model is None:
            raise RuntimeError(
                "knowledge retrieval is not configured; an operator must start the MCP "
                "server with --knowledge-model"
            )
        with self._knowledge_store_lock:
            if self._knowledge_store_instance is None:
                self._knowledge_store_instance = SeekDBKnowledgeStore(
                    self.workspace,
                    self.knowledge_model,
                )
            return self._knowledge_store_instance

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

    def _project_directory(self, value: str, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty project-relative path")
        raw = Path(value)
        if raw.is_absolute():
            raise ValueError(f"{name} must be project-relative")
        path = (self.project_root / raw).resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError(f"{name} escapes the configured project root")
        if not path.is_dir():
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

        report = doctor_report()
        report["runtime_details"]["drutai_workspace"] = DrutAIManager(
            self.workspace
        ).status()
        return report

    def drutai_status(self) -> dict[str, Any]:
        """Return path-free optional model and scientific-admission state."""

        return DrutAIManager(self.workspace).status()

    def drutai_model_acquire(
        self,
        *,
        model: str,
        approved_domain: str,
        license_acknowledgement: str,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Acquire one fixed-commit ONNX model after explicit GPL acknowledgement."""

        return DrutAIManager(self.workspace).acquire_model(
            model=model,
            approved_domain=approved_domain,
            license_acknowledgement=license_acknowledgement,
            replace=replace,
        )

    def drutai_annotate(
        self,
        *,
        input_path: str,
        fasta_directory: str,
        model: str,
        data_access_confirmed: bool,
        threads: int | None = None,
        batch_size: int = 2000,
        abstention_margin: float = 0.05,
    ) -> dict[str, Any]:
        """Run a network-isolated, non-decisional DrutAI annotation."""

        self._require_data_access_confirmation(data_access_confirmed)
        return DrutAIManager(self.workspace).annotate(
            input_tsv=self._project_file(input_path, "input_path"),
            fasta_directory=self._project_directory(
                fasta_directory, "fasta_directory"
            ),
            model=model,
            data_access_confirmed=True,
            threads=threads,
            batch_size=batch_size,
            abstention_margin=abstention_margin,
        )

    def experiment_import_preview(
        self,
        *,
        source_path: str,
        data_access_confirmed: bool,
    ) -> dict[str, Any]:
        """Validate a private assay table and return a non-mutating hash-bound plan."""

        self._require_data_access_confirmation(data_access_confirmed)
        return ExperimentalAssayStore(self.workspace).preview_import(
            self._project_file(source_path, "source_path")
        )

    def experiment_import_commit(
        self,
        *,
        source_path: str,
        plan_id: str,
        data_access_confirmed: bool,
    ) -> dict[str, Any]:
        """Commit the exact previewed assay table to immutable artifacts and catalog."""

        self._require_data_access_confirmation(data_access_confirmed)
        return ExperimentalAssayStore(self.workspace).commit_import(
            self._project_file(source_path, "source_path"),
            plan_id=plan_id,
            data_access_confirmed=True,
        )

    def experiment_list(
        self,
        *,
        data_access_confirmed: bool,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List private experiment metadata without returning measurements."""

        self._require_data_access_confirmation(data_access_confirmed)
        return ExperimentalAssayStore(self.workspace).list_experiments(limit=limit)

    def experiment_fit_curve(
        self,
        *,
        experiment_id: str,
        model: str,
        data_access_confirmed: bool,
    ) -> dict[str, Any]:
        """Fit one explicitly selected deterministic model and persist its receipt."""

        self._require_data_access_confirmation(data_access_confirmed)
        return ExperimentalAssayStore(self.workspace).fit_curve(
            experiment_id=experiment_id,
            model=model,
            data_access_confirmed=True,
        )

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
        this method; public acquisition stays a separately approved identifier-only tool.
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
                "MCP case_create is offline-only; use separately approved "
                "fetch_public_data and then pass content-addressed local inputs"
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

    def fetch_public_data(
        self,
        *,
        source: str,
        identifier: str,
        project_path: str,
        approved_domain: str,
        run_propka: bool = True,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Fetch one registry ID into a project-relative file; arbitrary URLs are forbidden."""

        if source not in PUBLIC_DATA_SOURCES:
            raise ValueError("unsupported public data source")
        output = Path(project_path)
        validate_public_output(source, self.project_root, output)
        fetcher = PublicDataFetcher(self.workspace)
        result = fetcher.fetch(
            source=source,
            identifier=identifier,
            approved_domains=(approved_domain,),
            run_propka=run_propka,
        )
        return materialize_public_fetch(
            result,
            fetcher.artifacts,
            project_root=self.project_root,
            output=output,
            replace=replace,
        )

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

    def library_status(self, *, data_access_confirmed: bool) -> dict[str, Any]:
        """Inspect only preconfigured library aliases after explicit user consent."""

        if self.library is None:
            self._require_data_access_confirmation(data_access_confirmed)
            return {
                "configured": False,
                "next_operator_action": (
                    "Run protbind library init, then add --library-config to MCP serve."
                ),
                "absolute_paths_disclosed": False,
            }
        return {"configured": True, **self._library(data_access_confirmed).status()}

    def library_list(
        self,
        *,
        kind: str,
        data_access_confirmed: bool,
        state: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List path-redacted entries in one preconfigured library."""

        return self._library(data_access_confirmed).list_entries(
            kind, state=state, limit=limit
        )

    def library_show(
        self,
        *,
        kind: str,
        entry_id: str,
        data_access_confirmed: bool,
    ) -> dict[str, Any]:
        """Show bounded QC and identity metadata, never source file bytes."""

        return self._library(data_access_confirmed).show_entry(kind, entry_id)

    def library_plan_import(
        self,
        *,
        kind: str,
        data_access_confirmed: bool,
        recursive: bool = False,
        max_files: int = 10_000,
    ) -> dict[str, Any]:
        """Plan only from the configured incoming directory; arbitrary paths are absent."""

        plan = self._library(data_access_confirmed).scan_incoming(
            kind,
            recursive=recursive,
            max_files=max_files,
        )
        return {
            "plan_id": plan["plan_id"],
            "kind": kind,
            "file_count": len(plan["files"]),
            "skipped": plan["skipped"],
            "semantics": plan["semantics"],
            "source_path_disclosed": False,
            "requires_separate_apply_confirmation": True,
        }

    def library_apply_import(
        self,
        *,
        kind: str,
        plan_id: str,
        data_access_confirmed: bool,
        mode: str = "copy",
        confirm_move: str | None = None,
    ) -> dict[str, Any]:
        """Apply one hash-bound plan; move requires the exact plan ID again."""

        return self._library(data_access_confirmed).apply_saved(
            kind,
            plan_id,
            mode=mode,
            confirm_move=confirm_move,
        )

    def library_verify_uniprot(
        self,
        *,
        entry_id: str,
        accession: str,
        approved_domain: str,
        data_access_confirmed: bool,
    ) -> dict[str, Any]:
        """Verify via accession-only UniProt lookup; no private sequence is uploaded."""

        manager = self._library(data_access_confirmed)
        fetcher = PublicDataFetcher(self.workspace)
        result = fetcher.fetch(
            source="uniprot-fasta",
            identifier=accession,
            approved_domains=(approved_domain,),
            run_propka=False,
        )
        verification = manager.verify_uniprot_bytes(
            entry_id,
            accession,
            fetcher.artifacts.read_bytes(result.artifact),
            source_artifact=result.artifact,
        )
        verification["network_receipt"] = result.receipt.to_dict()
        return verification

    def knowledge_document_inspect(
        self,
        *,
        project_path: str,
        data_access_confirmed: bool,
        pdf_backend: str = "auto",
        ocr: str = "off",
        ocr_language: str = "eng",
    ) -> dict[str, Any]:
        """Inspect extraction readiness for one project-local document without returning text."""

        self._require_data_access_confirmation(data_access_confirmed)
        path = self._project_file(project_path, "project_path")
        extraction = extract_document_bytes(
            path.read_bytes(),
            suffix=path.suffix,
            pdf_backend=pdf_backend,
            ocr=ocr,
            ocr_language=ocr_language,
        )
        return {
            **extraction.receipt,
            "source_name": path.name,
            "text_returned": False,
        }

    def knowledge_import(
        self,
        *,
        project_path: str,
        data_access_confirmed: bool,
        license: str | None = None,
        pdf_backend: str = "auto",
        ocr: str = "off",
        ocr_language: str = "eng",
    ) -> dict[str, Any]:
        """Import one project-local document into the configured cited evidence index."""

        self._require_data_access_confirmation(data_access_confirmed)
        if self.knowledge_model is None:
            raise RuntimeError(
                "knowledge retrieval is not configured; start MCP with --knowledge-model"
            )
        path = self._project_file(project_path, "project_path")
        artifact, count, receipt = import_document(
            self.workspace,
            path,
            self.knowledge_model,
            license=license,
            pdf_backend=pdf_backend,
            ocr=ocr,
            ocr_language=ocr_language,
        )
        return {
            "artifact_id": artifact.artifact_id,
            "chunks_indexed": count,
            "extraction_receipt_artifact_id": receipt.artifact_id,
            "source_name": path.name,
        }

    def knowledge_search(
        self,
        *,
        query: str,
        data_access_confirmed: bool,
        scope: str | None = "evidence",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Return cited local retrieval results; this tool performs no answer synthesis."""

        if scope not in {None, "evidence", "protein-library", "ligand-library"}:
            raise ValueError("unsupported knowledge scope")
        hits = self._knowledge_store(data_access_confirmed).search(
            query,
            top_k=top_k,
            scope=scope,
        )
        return {
            "query": query,
            "scope": scope,
            "answer_mode": "retrieval-only; citations and scientific gates remain required",
            "evidence": hits,
        }

    def library_rag_sync(
        self,
        *,
        kind: str,
        data_access_confirmed: bool,
        include_quarantined: bool = False,
    ) -> dict[str, Any]:
        """Rebuild a sanitized catalog projection after fresh private-data consent."""

        manager = self._library(data_access_confirmed)
        if self.knowledge_model is None:
            raise RuntimeError(
                "knowledge retrieval is not configured; start MCP with --knowledge-model"
            )
        return sync_library_rag(
            self.workspace,
            manager,
            self.knowledge_model,
            kind=kind,
            include_quarantined=include_quarantined,
        )

    def library_rag_search(
        self,
        *,
        query: str,
        kind: str,
        data_access_confirmed: bool,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve sanitized library candidates, never raw sequences or coordinates."""

        return self.knowledge_search(
            query=query,
            data_access_confirmed=data_access_confirmed,
            scope=f"{kind}-library",
            top_k=top_k,
        )

    def knowledge_model_status(self) -> dict[str, Any]:
        """Return path-free offline model admission evidence."""

        return inspect_embedding_model(self.knowledge_model)


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
    server.tool(name="drutai_status", structured_output=True)(service.drutai_status)
    server.tool(name="drutai_model_acquire", structured_output=True)(
        service.drutai_model_acquire
    )
    server.tool(name="drutai_annotate", structured_output=True)(
        service.drutai_annotate
    )
    server.tool(name="experiment_import_preview", structured_output=True)(
        service.experiment_import_preview
    )
    server.tool(name="experiment_import_commit", structured_output=True)(
        service.experiment_import_commit
    )
    server.tool(name="experiment_list", structured_output=True)(
        service.experiment_list
    )
    server.tool(name="experiment_fit_curve", structured_output=True)(
        service.experiment_fit_curve
    )
    server.tool(name="fetch_public_data", structured_output=True)(
        service.fetch_public_data
    )
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
    server.tool(name="library_status", structured_output=True)(service.library_status)
    server.tool(name="library_list", structured_output=True)(service.library_list)
    server.tool(name="library_show", structured_output=True)(service.library_show)
    server.tool(name="library_plan_import", structured_output=True)(
        service.library_plan_import
    )
    server.tool(name="library_apply_import", structured_output=True)(
        service.library_apply_import
    )
    server.tool(name="library_verify_uniprot", structured_output=True)(
        service.library_verify_uniprot
    )
    server.tool(name="knowledge_document_inspect", structured_output=True)(
        service.knowledge_document_inspect
    )
    server.tool(name="knowledge_import", structured_output=True)(
        service.knowledge_import
    )
    server.tool(name="knowledge_search", structured_output=True)(
        service.knowledge_search
    )
    server.tool(name="library_rag_sync", structured_output=True)(
        service.library_rag_sync
    )
    server.tool(name="library_rag_search", structured_output=True)(
        service.library_rag_search
    )
    server.tool(name="knowledge_model_status", structured_output=True)(
        service.knowledge_model_status
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
    library_config: Path | None = None,
    knowledge_model: Path | None = None,
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
        library_config=library_config,
        knowledge_model=knowledge_model,
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
