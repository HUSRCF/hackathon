from __future__ import annotations

from protbind_agent.agent_routing import ProtBindToolRouter
from radeon_agent.models import Message

_AVAILABLE = (
    "doctor",
    "case_status",
    "case_advance",
    "case_report",
    "knowledge_search",
    "memory_write",
    "library_plan_import",
)


def _messages(text: str) -> tuple[Message, ...]:
    return (Message(role="user", content=text),)


def test_router_uses_exact_explicit_tools_for_benchmark_style_prompt() -> None:
    router = ProtBindToolRouter()

    selected = router(
        _messages("依次调用 case_status、knowledge_search 和 memory_write。"),
        _AVAILABLE,
    )

    assert selected == ("case_status", "knowledge_search", "memory_write")
    assert router.decisions[-1].mode == "routed"


def test_router_adds_case_status_dependency_for_advance() -> None:
    router = ProtBindToolRouter()

    selected = router(_messages("请调用 case_advance 推进一步。"), _AVAILABLE)

    assert selected == ("case_status", "case_advance")


def test_router_falls_back_to_full_allowlist_for_ambiguous_research_prompt() -> None:
    router = ProtBindToolRouter()

    selected = router(_messages("分析这个蛋白质和候选小分子。"), _AVAILABLE)

    assert selected == _AVAILABLE
    assert router.decisions[-1].mode == "full-fallback"
