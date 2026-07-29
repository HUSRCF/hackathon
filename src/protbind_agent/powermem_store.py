"""PowerMem adapter that can never replace seekdb/artifact evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .models import ArtifactRef

_MARKER = "PROTBIND_SEEKDB_POINTER_V1:"
_ARTIFACT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class PowerMemCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WorkflowMemory:
    text: str
    seekdb_job_id: str
    artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.seekdb_job_id.strip():
            raise ValueError("workflow memory requires text and a seekdb job ID")
        if not self.artifact_ids:
            raise ValueError("workflow memory requires at least one artifact pointer")
        if any(not _ARTIFACT_ID.fullmatch(value) for value in self.artifact_ids):
            raise ValueError("workflow memory artifact pointers must be SHA-256 IDs")

    def serialize(self) -> str:
        pointer = json.dumps(
            {
                "seekdb_job_id": self.seekdb_job_id,
                "artifact_ids": list(self.artifact_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{self.text.strip()}\n{_MARKER}{pointer}"

    @classmethod
    def parse(cls, value: str) -> WorkflowMemory | None:
        if _MARKER not in value:
            return None
        text, pointer = value.rsplit(_MARKER, 1)
        try:
            parsed = json.loads(pointer.strip())
            return cls(
                text=text.strip(),
                seekdb_job_id=str(parsed["seekdb_job_id"]),
                artifact_ids=tuple(str(item) for item in parsed["artifact_ids"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None


class PowerMemWorkflowStore:
    """Stores preferences/experience only when an authoritative pointer is present."""

    def __init__(self, memory_client: Any) -> None:
        self.memory_client = memory_client

    @classmethod
    def from_config(cls, config: Any) -> PowerMemWorkflowStore:
        try:
            from powermem import Memory
        except ImportError as exc:
            raise PowerMemCapabilityError("PowerMem is not installed") from exc
        # The caller must provide an explicitly local/offline config.  We do not call
        # auto_config(), because its provider choices may include external services.
        return cls(Memory(config=config))

    def add(
        self,
        text: str,
        *,
        user_id: str,
        seekdb_job_id: str,
        artifacts: tuple[ArtifactRef, ...],
    ) -> Any:
        record = WorkflowMemory(
            text=text,
            seekdb_job_id=seekdb_job_id,
            artifact_ids=tuple(artifact.artifact_id for artifact in artifacts),
        )
        return self.memory_client.add(record.serialize(), user_id=user_id)

    def search(
        self, query: str, *, user_id: str, top_k: int = 5
    ) -> list[WorkflowMemory]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        result = self.memory_client.search(query, user_id=user_id)
        values = result.get("results", []) if isinstance(result, dict) else []
        records: list[WorkflowMemory] = []
        for value in values:
            raw = value.get("memory") if isinstance(value, dict) else None
            record = WorkflowMemory.parse(str(raw)) if raw is not None else None
            if record is not None:
                records.append(record)
        return records[:top_k]
