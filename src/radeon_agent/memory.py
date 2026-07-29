"""Small local memory seam; seekdb/PowerMem can implement the same protocol later."""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .tools import SideEffect, ToolSpec


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    text: str
    tags: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class MemoryHit:
    record: MemoryRecord
    score: float


class MemoryStore(Protocol):
    def add(self, text: str, *, tags: tuple[str, ...] = ()) -> MemoryRecord: ...

    def search(self, query: str, *, top_k: int = 5) -> list[MemoryHit]: ...


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


class JsonlMemoryStore:
    """Append-only memory suitable for a private, single-process MVP."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _records(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        records: list[MemoryRecord] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    item["tags"] = tuple(item.get("tags", ()))
                    records.append(MemoryRecord(**item))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(
                        f"invalid memory record at {self.path}:{line_number}"
                    ) from exc
        return records

    def add(self, text: str, *, tags: tuple[str, ...] = ()) -> MemoryRecord:
        clean = text.strip()
        if not clean:
            raise ValueError("memory text cannot be empty")
        record = MemoryRecord(
            id=uuid.uuid4().hex,
            text=clean,
            tags=tuple(tag.strip() for tag in tags if tag.strip()),
            created_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record

    def search(self, query: str, *, top_k: int = 5) -> list[MemoryHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_terms = _terms(query)
        if not query_terms:
            return []
        hits: list[MemoryHit] = []
        for record in self._records():
            record_terms = _terms(f"{record.text} {' '.join(record.tags)}")
            overlap = len(query_terms & record_terms)
            if overlap:
                score = overlap / max(len(query_terms), 1)
                hits.append(MemoryHit(record=record, score=score))
        hits.sort(key=lambda hit: (hit.score, hit.record.created_at), reverse=True)
        return hits[:top_k]


def memory_tools(store: MemoryStore) -> tuple[ToolSpec, ...]:
    def remember(arguments: dict) -> dict:
        tags = tuple(arguments.get("tags", []))
        return asdict(store.add(arguments["text"], tags=tags))

    def recall(arguments: dict) -> list[dict]:
        return [
            {"score": hit.score, **asdict(hit.record)}
            for hit in store.search(arguments["query"], top_k=arguments.get("top_k", 5))
        ]

    return (
        ToolSpec(
            name="remember",
            description=(
                "Persist a useful user preference or reusable fact in local private memory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 4000},
                    "tags": {"type": "array"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=remember,
            side_effect=SideEffect.LOCAL_WRITE,
        ),
        ToolSpec(
            name="recall",
            description="Search local private memory for facts relevant to the current task.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=recall,
            side_effect=SideEffect.NONE,
        ),
    )
