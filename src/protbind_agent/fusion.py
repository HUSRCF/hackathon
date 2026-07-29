"""Branch-preserving reciprocal-rank fusion for mode=both."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tripharm import TriPharmHit


@dataclass(frozen=True, slots=True)
class FusedHit:
    molecule_id: str
    rank: int
    rrf_score: float
    branch_ranks: dict[str, int]
    branch_scores: dict[str, float]
    branch_hits: dict[str, TriPharmHit]


def reciprocal_rank_fusion(
    branches: dict[str, list[TriPharmHit]],
    *,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedHit]:
    if not branches:
        return []
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    if weights is None:
        weights = {name: 1.0 / len(branches) for name in branches}
    if set(weights) != set(branches):
        raise ValueError("RRF weights must name every branch exactly once")
    if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("RRF weights must be non-negative with a positive sum")
    total_weight = sum(weights.values())
    normalized = {name: value / total_weight for name, value in weights.items()}
    aggregate: dict[str, dict[str, Any]] = {}
    for branch_name in sorted(branches):
        seen: set[str] = set()
        for rank, hit in enumerate(branches[branch_name], start=1):
            if hit.molecule_id in seen:
                raise ValueError(
                    f"branch {branch_name!r} contains duplicate molecule {hit.molecule_id!r}"
                )
            seen.add(hit.molecule_id)
            item = aggregate.setdefault(
                hit.molecule_id,
                {"score": 0.0, "ranks": {}, "scores": {}, "hits": {}},
            )
            contribution = normalized[branch_name] / (rrf_k + rank)
            item["score"] += contribution
            item["ranks"][branch_name] = rank
            item["scores"][branch_name] = hit.geometric_match_score
            item["hits"][branch_name] = hit
    ordered = sorted(aggregate.items(), key=lambda item: (-item[1]["score"], item[0]))
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        ordered = ordered[:top_k]
    return [
        FusedHit(
            molecule_id=molecule_id,
            rank=rank,
            rrf_score=item["score"],
            branch_ranks=item["ranks"],
            branch_scores=item["scores"],
            branch_hits=item["hits"],
        )
        for rank, (molecule_id, item) in enumerate(ordered, start=1)
    ]
