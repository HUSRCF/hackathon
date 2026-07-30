"""Bounded, evidence-preserving views of tool results for the LLM context."""

from __future__ import annotations

from typing import Any

_TEXT_LIMIT = 1200


def _artifact_view(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = {
        key: value[key]
        for key in (
            "sha256",
            "media_type",
            "size_bytes",
            "producer",
            "producer_version",
            "source",
            "license",
        )
        if key in value
    }
    sha256 = result.get("sha256")
    if isinstance(sha256, str) and len(sha256) == 64:
        result["artifact_id"] = f"sha256:{sha256}"
    return result or value


def _gate_view(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: value.get(key)
        for key in (
            "phase",
            "run_id",
            "stage",
            "decision",
            "manifest_sha256",
            "policy_sha256",
            "checks",
            "required_actions",
            "continuation_token",
            "automatic_retry",
        )
        if key in value
    }


def compact_case_status(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        "gate": _gate_view(value.get("gate")),
        "gate_receipt": _artifact_view(value.get("gate_receipt")),
        "run": value.get("run"),
        "semantics": (
            "Fresh preflight only. A mutation still requires host confirmation, "
            "the exact continuation token, one-stage execution, and ACCEPTED postflight."
        ),
    }


def compact_case_advance(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    acceptance = value.get("acceptance")
    next_gate = value.get("next_gate")
    return {
        "acceptance": _gate_view(acceptance),
        "acceptance_receipt": _artifact_view(value.get("acceptance_receipt")),
        "next_gate": compact_case_status(next_gate),
        "semantics": (
            "Only acceptance.decision=ACCEPTED completes the attempted stage. "
            "The next stage requires a new user decision and fresh gate."
        ),
    }


def compact_knowledge_search(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    evidence = []
    for raw in value.get("evidence", []):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", ""))
        if len(text) > _TEXT_LIMIT:
            text = text[:_TEXT_LIMIT].rstrip() + "…"
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            metadata = {
                key: metadata[key]
                for key in (
                    "artifact_id",
                    "source_name",
                    "section",
                    "page",
                    "scope",
                    "entry_id",
                    "kind",
                )
                if key in metadata
            }
        evidence.append(
            {
                "id": raw.get("id"),
                "text": text,
                "metadata": metadata,
            }
        )
    return {
        "query": value.get("query"),
        "scope": value.get("scope"),
        "answer_mode": value.get("answer_mode"),
        "evidence": evidence,
    }


def compact_memory_write(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: (
            _artifact_view(item)
            if key == "artifact"
            else item
        )
        for key, item in value.items()
        if key
        in {
            "written",
            "experience_id",
            "artifact",
            "case_id",
            "run_id",
            "evidence_grade",
            "scientific_state_changed",
            "semantics",
        }
    }
