"""Standalone ProtBind Agent runtime backed by the bounded in-process service."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

from radeon_agent.agent import Agent, AgentPendingResult, AgentResult
from radeon_agent.backends import HipFireBackend
from radeon_agent.backends.base import LLMBackend

from .agent_routing import ProtBindToolRouter
from .agent_tools import ConfirmationCallback, ProtBindAgentTools
from .approval_runtime import ApprovalCoordinator
from .mcp_server import ProtBindMCPService
from .workflow import PipelineConfig

PROTBIND_AGENT_SYSTEM_PROMPT = """You are the local private ProtBind research Agent.
Use only the supplied typed tools. Never invent an artifact, run ID, continuation token, score,
pose, validation result, citation, or completed stage. Explain that private sequences are not
uploaded. Before advancing, call case_status and use only its fresh continuation token; advance
exactly one stage per call and require an ACCEPTED postflight result. Stop on NEEDS_ACTION,
UNSUPPORTED, FAILED, or RETRYABLE unless the user explicitly chooses the bounded recovery.
TriPharm is geometric matching, Vina is a pose-ranking tool score, model poses are hypotheses,
and visual QA is not scientific validation. Retrieval and prior experience are hints only.
Every scientific statement must cite an artifact ID or document page/section. Write experience
memory only after the user agrees and only from a fully audited REPORTED run. Never retry a failed
tool call automatically; report the failure and wait for an explicit user choice or a declared
deterministic recovery policy."""
_ARTIFACT_CITATION = re.compile(r"sha256:[0-9a-f]{64}")


def require_loopback_hipfire_url(base_url: str) -> None:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise PermissionError(
            "built-in HipFire Agent requires an exact loopback HTTP /v1 base URL"
        )


@dataclass(frozen=True, slots=True)
class ProtBindAgentResult:
    answer: str
    model_calls: int
    tool_calls: int
    elapsed_seconds: float
    usage: dict[str, int | None]
    model_timings: tuple[dict[str, float | None], ...]
    model_usages: tuple[dict[str, int | None], ...]
    tool_results: tuple[dict[str, str | bool], ...]
    validated_artifact_citations: tuple[str, ...]
    citation_warnings: tuple[str, ...]
    tool_timeline: tuple[dict[str, Any], ...]
    exposed_tool_names: tuple[tuple[str, ...], ...]
    exposed_tool_schema_bytes: tuple[int, ...]
    tool_routes: tuple[dict[str, object], ...]
    shadow_plans: tuple[dict[str, Any], ...]
    approvals: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ProtBindAgentPendingResult:
    status: str
    session_id: str
    approval_id: str
    approval: dict[str, Any]
    model_calls: int
    tool_calls: int
    active_elapsed_seconds: float
    tool_timeline: tuple[dict[str, Any], ...]
    shadow_plans: tuple[dict[str, Any], ...]


AgentRuntimeResult = ProtBindAgentResult | ProtBindAgentPendingResult


class ProtBindAgentRuntime:
    def __init__(
        self,
        service: ProtBindMCPService,
        backend: LLMBackend,
        *,
        model: str,
        confirmation: ConfirmationCallback | None = None,
        max_steps: int = 16,
        timeout_seconds: float = 1800.0,
        max_tokens: int = 4096,
        stream: bool = True,
        route_tools: bool = True,
    ) -> None:
        self.approvals = ApprovalCoordinator() if confirmation is None else None
        approval_callback = confirmation or self.approvals
        if approval_callback is None:  # narrowed for static type checkers
            raise RuntimeError("approval callback could not be initialized")
        self.tools = ProtBindAgentTools(service, confirmation=approval_callback)
        self.tool_router = ProtBindToolRouter()
        self.agent = Agent(
            backend,
            model=model,
            tools=self.tools.registry(),
            system_prompt=PROTBIND_AGENT_SYSTEM_PROMPT,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            stream=stream,
            request_extra={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            tool_schema_selector=self.tool_router if route_tools else None,
            allow_failed_tool_retries=False,
        )

    def run(self, prompt: str) -> ProtBindAgentResult:
        """Compatibility path for pre-approved tests and controlled benchmarks."""

        result: AgentResult = self.agent.run(prompt)
        return self._final_result(result)

    def start(self, prompt: str) -> AgentRuntimeResult:
        result = self.agent.start(prompt)
        if isinstance(result, AgentPendingResult):
            return self._pending_result(result)
        return self._final_result(result)

    def resume(
        self,
        session_id: str,
        approval_id: str,
        *,
        approved: bool,
    ) -> AgentRuntimeResult:
        if self.approvals is None:
            raise RuntimeError(
                "this runtime uses a synchronous confirmation callback and cannot resume"
            )
        if self.agent.pending_approval_id(session_id) != approval_id:
            raise ValueError("approval_id does not match the paused agent session")
        with self.approvals.resume_scope(approval_id, approved=approved):
            result = self.agent.resume(
                session_id,
                approval_id=approval_id,
            )
        if isinstance(result, AgentPendingResult):
            return self._pending_result(result)
        return self._final_result(result)

    def approval_status(self, approval_id: str) -> dict[str, Any]:
        if self.approvals is None:
            raise RuntimeError("this runtime has no non-blocking approval coordinator")
        return self.approvals.get(approval_id)

    def _pending_result(
        self,
        result: AgentPendingResult,
    ) -> ProtBindAgentPendingResult:
        return ProtBindAgentPendingResult(
            status=result.status,
            session_id=result.session_id,
            approval_id=result.pending_tool.approval_id,
            approval=dict(result.pending_tool.payload),
            model_calls=result.model_calls,
            tool_calls=result.tool_calls,
            active_elapsed_seconds=result.active_elapsed_seconds,
            tool_timeline=tuple(self.tools.audit_dicts()),
            shadow_plans=tuple(self.tools.shadow_plan_dicts()),
        )

    def _final_result(self, result: AgentResult) -> ProtBindAgentResult:
        tool_citations = sorted(
            {
                citation
                for message in result.messages
                if message.role == "tool"
                for citation in _ARTIFACT_CITATION.findall(message.content)
            }
        )
        answer = result.answer
        answer_citations = set(_ARTIFACT_CITATION.findall(answer))
        invalid_citations = sorted(answer_citations - set(tool_citations))
        for citation in invalid_citations:
            answer = answer.replace(citation, "[未验证的 artifact 引用已移除]")
        validated = sorted(
            set(_ARTIFACT_CITATION.findall(answer)) & set(tool_citations)
        )
        if tool_citations and not validated:
            answer = (
                answer.rstrip()
                + "\n\n工具证据引用："
                + tool_citations[0]
            )
            validated = [tool_citations[0]]
        return ProtBindAgentResult(
            answer=answer,
            model_calls=result.model_calls,
            tool_calls=result.tool_calls,
            elapsed_seconds=result.elapsed_seconds,
            usage={
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            model_timings=tuple(asdict(timing) for timing in result.model_timings),
            model_usages=tuple(asdict(usage) for usage in result.model_usages),
            tool_results=tuple(asdict(item) for item in result.tool_results),
            validated_artifact_citations=tuple(validated),
            citation_warnings=(
                ("model emitted artifact IDs absent from tool results",)
                if invalid_citations
                else ()
            ),
            tool_timeline=tuple(self.tools.audit_dicts()),
            exposed_tool_names=result.exposed_tool_names,
            exposed_tool_schema_bytes=result.exposed_tool_schema_bytes,
            tool_routes=tuple(self.tool_router.decision_dicts()),
            shadow_plans=tuple(self.tools.shadow_plan_dicts()),
            approvals=(
                self.approvals.requests() if self.approvals is not None else ()
            ),
        )

    def run_interactive(
        self,
        prompt: str,
        *,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stderr,
    ) -> ProtBindAgentResult:
        result = self.start(prompt)
        while isinstance(result, ProtBindAgentPendingResult):
            output_stream.write(
                "\nProtBind approval is waiting; deterministic idle work is cancellable:\n"
                + json.dumps(result.approval, ensure_ascii=False, indent=2)
                + "\nApprove this one tool call? [y/N] "
            )
            output_stream.flush()
            answer = input_stream.readline()
            result = self.resume(
                result.session_id,
                result.approval_id,
                approved=answer.strip().lower() in {"y", "yes"},
            )
        return result


def create_runtime(
    *,
    workspace: Path,
    project_root: Path,
    model: str = "qwen3.5:9b",
    base_url: str = "http://127.0.0.1:11435/v1",
    library_config: Path | None = None,
    knowledge_model: Path | None = None,
    pipeline_config: PipelineConfig | None = None,
    confirmation: ConfirmationCallback | None = None,
    backend: LLMBackend | None = None,
    max_steps: int = 16,
    stream: bool = True,
    route_tools: bool = True,
) -> ProtBindAgentRuntime:
    require_loopback_hipfire_url(base_url)
    service = ProtBindMCPService(
        workspace=workspace,
        project_root=project_root,
        config=pipeline_config,
        library_config=library_config,
        knowledge_model=knowledge_model,
    )
    return ProtBindAgentRuntime(
        service,
        backend or HipFireBackend(base_url),
        model=model,
        confirmation=confirmation,
        max_steps=max_steps,
        stream=stream,
        route_tools=route_tools,
    )
