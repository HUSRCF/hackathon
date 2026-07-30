"""Standalone ProtBind Agent runtime backed by the bounded in-process service."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

from radeon_agent.agent import Agent, AgentResult
from radeon_agent.backends import HipFireBackend
from radeon_agent.backends.base import LLMBackend

from .agent_tools import ActionPreview, ConfirmationCallback, ProtBindAgentTools
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
memory only after the user agrees and only from a fully audited REPORTED run."""
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


class TerminalConfirmation:
    """Fresh, human-visible confirmation for every permissioned tool invocation."""

    def __init__(
        self,
        *,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stderr,
    ) -> None:
        self.input_stream = input_stream
        self.output_stream = output_stream

    def __call__(self, preview: ActionPreview) -> bool:
        self.output_stream.write(
            "\nProtBind confirmation required:\n"
            + json.dumps(asdict(preview), ensure_ascii=False, indent=2)
            + "\nApprove this one tool call? [y/N] "
        )
        self.output_stream.flush()
        answer = self.input_stream.readline()
        return answer.strip().lower() in {"y", "yes"}


class ProtBindAgentRuntime:
    def __init__(
        self,
        service: ProtBindMCPService,
        backend: LLMBackend,
        *,
        model: str,
        confirmation: ConfirmationCallback,
        max_steps: int = 16,
        timeout_seconds: float = 1800.0,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> None:
        self.tools = ProtBindAgentTools(service, confirmation=confirmation)
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
        )

    def run(self, prompt: str) -> ProtBindAgentResult:
        result: AgentResult = self.agent.run(prompt)
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
        )


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
        confirmation=confirmation or TerminalConfirmation(),
        max_steps=max_steps,
        stream=stream,
    )
