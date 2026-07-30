from __future__ import annotations

import hashlib
import json

import pytest

from protbind_agent.agent_benchmark import (
    AgentBenchmarkConfig,
    _model_weights_receipt,
    _successful_subsequence,
    load_agent_workload,
)


def test_agent_workload_is_raw_hash_bound_and_formats_only_named_inputs(
    tmp_path,
) -> None:
    workload = tmp_path / "workload.json"
    value = {
        "schema_version": "1.0",
        "prompt_template": "{run_id}|{knowledge_query}|{preference}",
        "required_tools": ["case_status", "knowledge_search", "memory_write"],
    }
    raw = json.dumps(value).encode()
    workload.write_bytes(raw)

    parsed, prompt, digest = load_agent_workload(
        workload,
        run_id="run-1",
        knowledge_query="kinase",
        preference="offline",
    )

    assert parsed == value
    assert prompt == "run-1|kinase|offline"
    assert digest == hashlib.sha256(raw).hexdigest()


def test_agent_benchmark_requires_measured_model_hash(tmp_path) -> None:
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"local model fixture")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()

    receipt = _model_weights_receipt(weights, digest)

    assert receipt["sha256"] == digest
    assert receipt["file_count"] == 1
    assert receipt["path_disclosed"] is False
    with pytest.raises(ValueError, match="does not match"):
        _model_weights_receipt(weights, "0" * 64)


def test_required_agent_tools_must_succeed_in_order() -> None:
    required = ["case_status", "knowledge_search", "memory_write"]
    assert _successful_subsequence(
        [
            {"tool": "doctor", "ok": True},
            {"tool": "case_status", "ok": True},
            {"tool": "knowledge_search", "ok": True},
            {"tool": "memory_write", "ok": True},
        ],
        required,
    )
    assert not _successful_subsequence(
        [
            {"tool": "knowledge_search", "ok": True},
            {"tool": "case_status", "ok": True},
            {"tool": "memory_write", "ok": True},
        ],
        required,
    )
    assert not _successful_subsequence(
        [
            {"tool": "case_status", "ok": True},
            {"tool": "knowledge_search", "ok": False},
            {"tool": "memory_write", "ok": True},
        ],
        required,
    )


def test_agent_benchmark_config_rejects_bootstrap_grade_provenance() -> None:
    with pytest.raises(ValueError, match="at least three"):
        AgentBenchmarkConfig(
            label="w7900",
            model="qwen",
            model_revision="revision",
            model_sha256="a" * 64,
            quantization="q4",
            hipfire_revision="revision",
            hipfire_visible_device=0,
            hipfire_speculation="off",
            hipfire_jinja_mode="default-on",
            code_revision="revision",
            repetitions=1,
        )
