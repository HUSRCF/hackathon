from __future__ import annotations

from protbind_agent.agent_views import (
    compact_case_status,
    compact_knowledge_search,
)


def test_case_status_view_preserves_gate_token_and_adds_artifact_id() -> None:
    digest = "a" * 64
    result = compact_case_status(
        {
            "gate": {
                "phase": "PREFLIGHT",
                "run_id": "run-1",
                "stage": "SCREENED",
                "decision": "READY",
                "manifest_sha256": "b" * 64,
                "policy_sha256": "c" * 64,
                "checks": [],
                "required_actions": [],
                "continuation_token": "d" * 64,
                "automatic_retry": False,
                "manifest_updated_at": "not-needed-by-model",
            },
            "gate_receipt": {
                "sha256": digest,
                "media_type": "application/json",
                "size_bytes": 10,
                "producer": "test",
            },
            "run": {"state": "INDEXED", "next_stage": "SCREENED"},
        }
    )

    assert result["gate"]["continuation_token"] == "d" * 64
    assert result["gate_receipt"]["artifact_id"] == f"sha256:{digest}"
    assert "manifest_updated_at" not in result["gate"]


def test_knowledge_view_bounds_text_and_preserves_page_citation() -> None:
    result = compact_knowledge_search(
        {
            "query": "kinase",
            "scope": "evidence",
            "answer_mode": "retrieval-only",
            "evidence": [
                {
                    "id": "doc:1",
                    "text": "x" * 2000,
                    "metadata": {
                        "artifact_id": f"sha256:{'e' * 64}",
                        "source_name": "paper.pdf",
                        "page": 7,
                        "section": "Results",
                        "private_internal_field": "drop",
                    },
                }
            ],
        }
    )

    hit = result["evidence"][0]
    assert len(hit["text"]) == 1201
    assert hit["text"].endswith("…")
    assert hit["metadata"]["page"] == 7
    assert "private_internal_field" not in hit["metadata"]
