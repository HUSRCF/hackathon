from __future__ import annotations

from radeon_agent.memory import JsonlMemoryStore
from radeon_agent.tools import ToolRegistry, ToolSpec


def test_jsonl_memory_round_trip(tmp_path) -> None:
    store = JsonlMemoryStore(tmp_path / "memory.jsonl")
    record = store.add("User prefers ROCm kernels", tags=("gpu", "preference"))

    hits = store.search("ROCm preference", top_k=2)

    assert hits[0].record.id == record.id
    assert hits[0].score > 0


def test_registry_validates_required_and_unknown_arguments() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                handler=lambda arguments: arguments["text"],
            )
        ]
    )

    assert not registry.execute("echo", {}).ok
    assert not registry.execute("echo", {"text": "ok", "extra": 1}).ok
    assert registry.execute("echo", {"text": "ok"}).value == "ok"

