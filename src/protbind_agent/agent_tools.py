"""Convert the bounded ProtBind service into in-process Agent ToolSpecs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from radeon_agent.tools import SideEffect, ToolPermissionError, ToolRegistry, ToolSpec

from .experience import ExperienceStore
from .mcp_server import ProtBindMCPService


@dataclass(frozen=True, slots=True)
class ActionPreview:
    tool: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    network: str
    scientific_state_change: bool
    expected_next_state: str
    recovery: str


@dataclass(frozen=True, slots=True)
class ToolAuditEvent:
    tool: str
    confirmed: bool
    started_at_monotonic: float
    duration_seconds: float
    ok: bool
    error_type: str | None


ConfirmationCallback = Callable[[ActionPreview], bool]


def deny_confirmation(_preview: ActionPreview) -> bool:
    return False


def _object(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _string(*, enum: tuple[str, ...] | None = None, maximum: int = 1000) -> dict[str, Any]:
    value: dict[str, Any] = {"type": "string", "minLength": 1, "maxLength": maximum}
    if enum is not None:
        value["enum"] = list(enum)
    return value


class ProtBindAgentTools:
    """In-process allowlist; it never discovers or calls arbitrary MCP tools."""

    def __init__(
        self,
        service: ProtBindMCPService,
        *,
        confirmation: ConfirmationCallback = deny_confirmation,
    ) -> None:
        self.service = service
        self.confirmation = confirmation
        self.experience = ExperienceStore(service.workspace, service.workflow)
        self.audit_events: list[ToolAuditEvent] = []

    def _handler(
        self,
        name: str,
        function: Callable[..., Any],
        *,
        preview: Callable[[dict[str, Any]], ActionPreview] | None = None,
        inject_data_confirmation: bool = False,
    ) -> Callable[[dict[str, Any]], Any]:
        def execute(arguments: dict[str, Any]) -> Any:
            started = time.monotonic()
            confirmed = preview is None
            try:
                values = dict(arguments)
                if preview is not None:
                    confirmed = self.confirmation(preview(values))
                    if not confirmed:
                        raise ToolPermissionError(
                            f"user declined or did not confirm tool {name!r}"
                        )
                if inject_data_confirmation:
                    values["data_access_confirmed"] = True
                result = function(**values)
            except Exception as exc:
                self.audit_events.append(
                    ToolAuditEvent(
                        tool=name,
                        confirmed=confirmed,
                        started_at_monotonic=started,
                        duration_seconds=time.monotonic() - started,
                        ok=False,
                        error_type=type(exc).__name__,
                    )
                )
                raise
            self.audit_events.append(
                ToolAuditEvent(
                    tool=name,
                    confirmed=confirmed,
                    started_at_monotonic=started,
                    duration_seconds=time.monotonic() - started,
                    ok=True,
                    error_type=None,
                )
            )
            return result

        return execute

    @staticmethod
    def _preview(
        tool: str,
        *,
        reads: tuple[str, ...],
        writes: tuple[str, ...],
        network: str = "none",
        scientific_state_change: bool = False,
        expected_next_state: str = "unchanged",
        recovery: str = "No scientific state is changed on failure.",
    ) -> Callable[[dict[str, Any]], ActionPreview]:
        def build(_arguments: dict[str, Any]) -> ActionPreview:
            return ActionPreview(
                tool=tool,
                reads=reads,
                writes=writes,
                network=network,
                scientific_state_change=scientific_state_change,
                expected_next_state=expected_next_state,
                recovery=recovery,
            )

        return build

    def specs(self) -> tuple[ToolSpec, ...]:
        run = {"run_id": _string(maximum=128)}
        format_schema = _string(enum=("markdown", "html", "degraded"))
        dossier_format = _string(enum=("json", "markdown", "html"))
        read_only = (
            ToolSpec(
                "doctor",
                "Return path-free scientific runtime and Radeon capability evidence.",
                _object({}),
                self._handler("doctor", self.service.doctor),
            ),
            ToolSpec(
                "case_status",
                "Deep-audit one run and return its fresh one-stage continuation gate.",
                _object(run, required=("run_id",)),
                self._handler("case_status", self.service.case_status),
            ),
            ToolSpec(
                "case_report",
                "Read a bounded deterministic report; never returns coordinates.",
                _object({**run, "format": format_schema}, required=("run_id",)),
                self._handler("case_report", self.service.case_report),
            ),
            ToolSpec(
                "case_dossier",
                "Read accepted/computed stage, timing, failure, and artifact evidence.",
                _object(
                    {**run, "format": dossier_format},
                    required=("run_id",),
                ),
                self._handler("case_dossier", self.service.case_dossier),
            ),
            ToolSpec(
                "case_pose_view",
                "Read coordinate-free pose QA metadata; not scientific validation.",
                _object(run, required=("run_id",)),
                self._handler("case_pose_view", self.service.case_pose_view),
            ),
            ToolSpec(
                "artifact_metadata",
                "Verify one complete serialized ArtifactRef and return metadata only.",
                _object(
                    {"artifact_ref_json": _string(maximum=8000)},
                    required=("artifact_ref_json",),
                ),
                self._handler("artifact_metadata", self.service.artifact_metadata),
            ),
            ToolSpec(
                "knowledge_model_status",
                "Return path-free local embedding-model admission evidence.",
                _object({}),
                self._handler(
                    "knowledge_model_status", self.service.knowledge_model_status
                ),
            ),
            ToolSpec(
                "memory_search",
                "Search prior deterministic experience hints; never changes a protocol.",
                _object(
                    {
                        "query": _string(maximum=1000),
                        "top_k": {"type": "integer"},
                    },
                    required=("query",),
                ),
                self._handler("memory_search", self.experience.search),
            ),
        )
        confirmed = (
            ToolSpec(
                "fetch_public_data",
                "Fetch one public registry identifier after exact-domain confirmation.",
                _object(
                    {
                        "source": _string(maximum=64),
                        "identifier": _string(maximum=128),
                        "project_path": _string(maximum=1000),
                        "approved_domain": _string(maximum=253),
                        "run_propka": {"type": "boolean"},
                        "replace": {"type": "boolean"},
                    },
                    required=(
                        "source",
                        "identifier",
                        "project_path",
                        "approved_domain",
                    ),
                ),
                self._handler(
                    "fetch_public_data",
                    self.service.fetch_public_data,
                    preview=self._preview(
                        "fetch_public_data",
                        reads=("public registry identifier",),
                        writes=("project-local candidate file and provenance sidecar",),
                        network="exact approved registry domain",
                        recovery="Partial downloads are not accepted or attached to a case.",
                    ),
                ),
                SideEffect.EXTERNAL,
            ),
            ToolSpec(
                "case_create",
                "Create an offline ResearchCase from project-relative inputs.",
                _object(
                    {
                        "case_path": _string(maximum=1000),
                        "index_path": _string(maximum=1000),
                        "run_id": _string(maximum=128),
                    },
                    required=("case_path", "index_path"),
                ),
                self._handler(
                    "case_create",
                    self.service.case_create,
                    preview=self._preview(
                        "case_create",
                        reads=("project-local case JSON", "frozen TriPharm index"),
                        writes=("new run manifest and imported input artifacts",),
                        scientific_state_change=True,
                        expected_next_state="CREATED with a fresh INPUT_VALIDATED gate",
                        recovery="A failed create does not advance a scientific stage.",
                    ),
                ),
                SideEffect.LOCAL_WRITE,
            ),
            ToolSpec(
                "case_advance",
                "Advance exactly one stage with a fresh continuation token.",
                _object(
                    {
                        **run,
                        "continuation_token": _string(maximum=4096),
                    },
                    required=("run_id", "continuation_token"),
                ),
                self._handler(
                    "case_advance",
                    self.service.case_advance,
                    preview=self._preview(
                        "case_advance",
                        reads=("current manifest and all upstream artifacts",),
                        writes=("one stage record and postflight acceptance receipt",),
                        scientific_state_change=True,
                        expected_next_state="exactly one accepted main stage or explicit failure",
                        recovery=(
                            "Retry requires a newly issued gate token; tokens are never reused."
                        ),
                    ),
                ),
                SideEffect.LOCAL_WRITE,
            ),
            ToolSpec(
                "case_attach_support",
                "Attach one reviewed project-local support artifact under freeze rules.",
                _object(
                    {
                        **run,
                        "name": _string(maximum=128),
                        "project_path": _string(maximum=1000),
                        "media_type": _string(maximum=200),
                        "replace": {"type": "boolean"},
                    },
                    required=("run_id", "name", "project_path", "media_type"),
                ),
                self._handler(
                    "case_attach_support",
                    self.service.case_attach_support,
                    preview=self._preview(
                        "case_attach_support",
                        reads=("one reviewed project-local file",),
                        writes=("content-addressed support artifact and manifest binding",),
                        scientific_state_change=True,
                        recovery="Frozen or mismatched support is rejected without stage advance.",
                    ),
                ),
                SideEffect.LOCAL_WRITE,
            ),
            ToolSpec(
                "library_plan_import",
                "Hash configured incoming files and freeze a non-mutating import plan.",
                _object(
                    {
                        "kind": _string(enum=("protein", "ligand")),
                        "recursive": {"type": "boolean"},
                        "max_files": {"type": "integer"},
                    },
                    required=("kind",),
                ),
                self._handler(
                    "library_plan_import",
                    self.service.library_plan_import,
                    preview=self._preview(
                        "library_plan_import",
                        reads=("configured private library incoming directory",),
                        writes=("hash-bound import plan receipt only",),
                        recovery="No library object is imported by planning.",
                    ),
                    inject_data_confirmation=True,
                ),
                SideEffect.LOCAL_WRITE,
            ),
            ToolSpec(
                "library_apply_import",
                "Apply one reviewed hash-bound library plan; copy is the default.",
                _object(
                    {
                        "kind": _string(enum=("protein", "ligand")),
                        "plan_id": _string(maximum=64),
                        "mode": _string(enum=("copy", "move")),
                        "confirm_move": _string(maximum=64),
                    },
                    required=("kind", "plan_id"),
                ),
                self._handler(
                    "library_apply_import",
                    self.service.library_apply_import,
                    preview=self._preview(
                        "library_apply_import",
                        reads=("files committed by the reviewed import plan",),
                        writes=("private CAS, QC catalog, and import receipt",),
                        scientific_state_change=False,
                        recovery=(
                            "Quarantined inputs are preserved; move deletes only after "
                            "CAS verification."
                        ),
                    ),
                    inject_data_confirmation=True,
                ),
                SideEffect.LOCAL_WRITE,
            ),
            ToolSpec(
                "knowledge_import",
                "Import one project-local PDF/Markdown with extraction receipt.",
                _object(
                    {
                        "project_path": _string(maximum=1000),
                        "license": _string(maximum=500),
                        "pdf_backend": _string(
                            enum=("auto", "pymupdf", "pdftotext")
                        ),
                        "ocr": _string(enum=("off", "auto", "required")),
                        "ocr_language": _string(maximum=32),
                    },
                    required=("project_path",),
                ),
                self._handler(
                    "knowledge_import",
                    self.service.knowledge_import,
                    preview=self._preview(
                        "knowledge_import",
                        reads=("one project-local document",),
                        writes=("document artifact, extraction receipt, and seekdb chunks",),
                        recovery=(
                            "Failed extraction is not indexed; unresolved OCR pages "
                            "remain explicit."
                        ),
                    ),
                    inject_data_confirmation=True,
                ),
                SideEffect.LOCAL_WRITE,
            ),
            ToolSpec(
                "knowledge_search",
                "Retrieve cited local evidence; results are not scientific validation.",
                _object(
                    {
                        "query": _string(maximum=1000),
                        "scope": _string(
                            enum=("evidence", "protein-library", "ligand-library")
                        ),
                        "top_k": {"type": "integer"},
                    },
                    required=("query",),
                ),
                self._handler(
                    "knowledge_search",
                    self.service.knowledge_search,
                    preview=self._preview(
                        "knowledge_search",
                        reads=("private local knowledge index",),
                        writes=(),
                    ),
                    inject_data_confirmation=True,
                ),
            ),
            ToolSpec(
                "library_rag_sync",
                "Rebuild the sanitized seekdb projection of one private library.",
                _object(
                    {
                        "kind": _string(enum=("protein", "ligand")),
                        "include_quarantined": {"type": "boolean"},
                    },
                    required=("kind",),
                ),
                self._handler(
                    "library_rag_sync",
                    self.service.library_rag_sync,
                    preview=self._preview(
                        "library_rag_sync",
                        reads=("private catalog metadata and bounded QC",),
                        writes=("sanitized projection artifact and seekdb scope",),
                        recovery=(
                            "catalog.sqlite remains authoritative if projection rebuild fails."
                        ),
                    ),
                    inject_data_confirmation=True,
                ),
                SideEffect.LOCAL_WRITE,
            ),
            ToolSpec(
                "memory_write",
                "Derive and persist one cited experience record from a REPORTED run.",
                _object(
                    {
                        **run,
                        "preference": _string(maximum=1000),
                    },
                    required=("run_id",),
                ),
                self._handler(
                    "memory_write",
                    self.experience.write,
                    preview=self._preview(
                        "memory_write",
                        reads=("audited REPORTED manifest and cited scientific artifacts",),
                        writes=("deterministic experience artifact and derived search catalog",),
                        recovery="Failure leaves the scientific run unchanged.",
                    ),
                ),
                SideEffect.LOCAL_WRITE,
            ),
        )
        return (*read_only, *confirmed)

    def registry(self) -> ToolRegistry:
        return ToolRegistry(self.specs(), max_side_effect=SideEffect.EXTERNAL)

    def audit_dicts(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.audit_events]
