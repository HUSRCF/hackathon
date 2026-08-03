"""Deterministic schema routing for the bounded ProtBind Agent.

Routing only reduces the tools described to the model. It never registers a new
tool, changes side-effect policy, or bypasses a host confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass

from radeon_agent.models import Message

_PACKS: dict[str, frozenset[str]] = {
    "doctor": frozenset({"doctor", "knowledge_model_status"}),
    "case-control": frozenset(
        {
            "case_create",
            "case_status",
            "case_advance",
            "case_attach_support",
        }
    ),
    "report": frozenset(
        {
            "case_status",
            "case_report",
            "case_dossier",
            "case_pose_view",
            "artifact_metadata",
        }
    ),
    "knowledge-memory": frozenset(
        {
            "case_status",
            "knowledge_model_status",
            "knowledge_import",
            "knowledge_search",
            "memory_search",
            "memory_write",
        }
    ),
    "library": frozenset(
        {
            "library_plan_import",
            "library_apply_import",
            "library_rag_sync",
            "knowledge_search",
        }
    ),
    "public-fetch": frozenset({"doctor", "fetch_public_data"}),
    "drutai": frozenset(
        {"drutai_status", "drutai_model_acquire", "drutai_annotate"}
    ),
    "experiment": frozenset(
        {
            "experiment_import_preview",
            "experiment_import_commit",
            "experiment_list",
            "experiment_fit_curve",
        }
    ),
}

_PACK_TERMS: dict[str, tuple[str, ...]] = {
    "doctor": (
        "doctor",
        "hardware",
        "gpu",
        "capability",
        "环境检查",
        "硬件",
        "显卡",
        "能力检查",
        "检查",
    ),
    "case-control": (
        "case",
        "run_id",
        "stage",
        "advance",
        "resume",
        "continuation",
        "案例",
        "阶段",
        "继续",
        "恢复",
        "推进",
        "附加支持",
    ),
    "report": (
        "report",
        "dossier",
        "pose view",
        "artifact",
        "报告",
        "档案",
        "姿态",
        "证据",
        "可视化",
    ),
    "knowledge-memory": (
        "knowledge",
        "rag",
        "paper",
        "memory",
        "文献",
        "论文",
        "检索",
        "记忆",
        "经验",
    ),
    "library": (
        "library",
        "import",
        "protein library",
        "ligand library",
        "数据库",
        "资料库",
        "蛋白质库",
        "小分子库",
        "导入",
        "迁移",
    ),
    "public-fetch": (
        "fetch",
        "rcsb",
        "uniprot",
        "pubchem",
        "alphafold db",
        "下载",
        "获取公开",
        "拉取",
    ),
    "drutai": (
        "drutai",
        "dti",
        "sequence-smiles",
        "结合注释",
        "一致性注释",
    ),
    "experiment": (
        "assay",
        "experiment",
        "dose response",
        "curve fit",
        "实验数据",
        "剂量响应",
        "拟合",
        "批次",
    ),
}

_DEPENDENCIES: dict[str, frozenset[str]] = {
    "case_advance": frozenset({"case_status"}),
    "memory_write": frozenset({"case_status"}),
}


@dataclass(frozen=True, slots=True)
class ToolRoute:
    mode: str
    packs: tuple[str, ...]
    tools: tuple[str, ...]


class ProtBindToolRouter:
    """Select a stable high-confidence tool subset or fall back to all tools."""

    def __init__(self) -> None:
        self.decisions: list[ToolRoute] = []

    @staticmethod
    def _user_text(messages: tuple[Message, ...]) -> str:
        return "\n".join(
            message.content or "" for message in messages if message.role == "user"
        ).lower()

    def route(
        self,
        messages: tuple[Message, ...],
        available: tuple[str, ...],
    ) -> ToolRoute:
        text = self._user_text(messages)
        available_set = set(available)
        explicit = {name for name in available if name.lower() in text}
        selected = set(explicit)
        packs = (
            set()
            if len(explicit) >= 2
            else {
                pack
                for pack, terms in _PACK_TERMS.items()
                if any(term in text for term in terms)
            }
        )
        for pack in packs:
            selected.update(_PACKS[pack])
        for name in tuple(selected):
            selected.update(_DEPENDENCIES.get(name, ()))

        selected.intersection_update(available_set)
        if not selected:
            decision = ToolRoute(
                mode="full-fallback",
                packs=(),
                tools=available,
            )
        else:
            decision = ToolRoute(
                mode="routed",
                packs=tuple(sorted(packs)),
                tools=tuple(name for name in available if name in selected),
            )
        self.decisions.append(decision)
        return decision

    def __call__(
        self,
        messages: tuple[Message, ...],
        available: tuple[str, ...],
    ) -> tuple[str, ...]:
        return self.route(messages, available).tools

    def decision_dicts(self) -> list[dict[str, object]]:
        return [
            {
                "mode": decision.mode,
                "packs": decision.packs,
                "tools": decision.tools,
            }
            for decision in self.decisions
        ]
