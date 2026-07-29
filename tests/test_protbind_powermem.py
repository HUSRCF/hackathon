from __future__ import annotations

from protbind_agent.artifacts import sha256_bytes
from protbind_agent.models import ArtifactRef
from protbind_agent.powermem_store import PowerMemWorkflowStore, WorkflowMemory


class _FakeMemory:
    def __init__(self) -> None:
        self.values: list[str] = []

    def add(self, text: str, *, user_id: str):
        self.values.append(text)
        return {"user_id": user_id}

    def search(self, query: str, *, user_id: str):
        del query, user_id
        return {"results": [{"memory": value} for value in self.values]}


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        sha256=sha256_bytes(b"evidence"),
        media_type="application/json",
        size_bytes=8,
        producer="test",
    )


def test_powermem_records_must_retain_seekdb_and_artifact_pointers() -> None:
    backend = _FakeMemory()
    store = PowerMemWorkflowStore(backend)

    store.add(
        "Use low_mem after an OOM.",
        user_id="researcher",
        seekdb_job_id="run-42",
        artifacts=(_artifact(),),
    )
    backend.values.append("unreferenced memory must be ignored")
    results = store.search("OOM", user_id="researcher")

    assert results == [
        WorkflowMemory(
            text="Use low_mem after an OOM.",
            seekdb_job_id="run-42",
            artifact_ids=(_artifact().artifact_id,),
        )
    ]
