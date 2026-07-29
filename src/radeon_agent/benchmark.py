"""Reproducible client-side inference benchmark for OpenAI-compatible backends."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backends.base import LLMBackend
from .hardware import HardwareManifest, probe_hardware
from .models import ChatRequest, Message

BENCHMARK_SCHEMA_VERSION = "1.0"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: bytes, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, value).hexdigest()


@dataclass(frozen=True, slots=True)
class QualityChecks:
    min_chars: int = 1
    contains: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> QualityChecks:
        raw = raw or {}
        return cls(
            min_chars=int(raw.get("min_chars", 1)),
            contains=tuple(str(item) for item in raw.get("contains", [])),
            excludes=tuple(str(item) for item in raw.get("excludes", [])),
        )

    def evaluate(self, content: str) -> tuple[bool, tuple[str, ...]]:
        failures: list[str] = []
        if len(content) < self.min_chars:
            failures.append(f"output shorter than {self.min_chars} chars")
        for needle in self.contains:
            if needle not in content:
                failures.append(f"missing required text: {needle!r}")
        for needle in self.excludes:
            if needle in content:
                failures.append(f"contains forbidden text: {needle!r}")
        return not failures, tuple(failures)


@dataclass(frozen=True, slots=True)
class PromptCase:
    id: str
    messages: tuple[Message, ...]
    max_tokens: int = 256
    checks: QualityChecks = field(default_factory=QualityChecks)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PromptCase:
        case_id = str(raw.get("id") or "").strip()
        if not case_id:
            raise ValueError("benchmark case requires a non-empty id")
        messages: list[Message] = []
        for item in raw.get("messages", []):
            role = item.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"case {case_id!r} has invalid role {role!r}")
            content = item.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError(f"case {case_id!r} has non-string content")
            messages.append(Message(role=role, content=content))
        if not messages:
            raise ValueError(f"case {case_id!r} has no messages")
        return cls(
            id=case_id,
            messages=tuple(messages),
            max_tokens=int(raw.get("max_tokens", 256)),
            checks=QualityChecks.from_dict(raw.get("checks")),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    path: str
    raw_sha256: str
    raw_md5: str
    cases: tuple[PromptCase, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    label: str
    model: str
    repetitions: int = 5
    warmup_runs: int = 1
    temperature: float = 0.0
    quantization: str = "unspecified"
    model_revision: str = "unspecified"
    model_sha256: str | None = None
    code_revision: str | None = None
    runtime_revision: str | None = None
    workload_config_sha256: str | None = None
    tuning_config_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.repetitions < 3:
            raise ValueError("repetitions must be >= 3 for cross-device evidence")
        if self.warmup_runs < 1:
            raise ValueError("warmup_runs must be >= 1 so JIT/model load is excluded")
        if not self.label.strip() or not self.model.strip():
            raise ValueError("label and model cannot be empty")
        if (
            self.model_sha256 is None
            or len(self.model_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.model_sha256)
        ):
            raise ValueError("model_sha256 must be a lowercase SHA-256 digest")
        for name, value in (
            ("quantization", self.quantization),
            ("model_revision", self.model_revision),
            ("code_revision", self.code_revision),
            ("runtime_revision", self.runtime_revision),
        ):
            if value is None or not value.strip() or value == "unspecified":
                raise ValueError(f"{name} is required for competition evidence")
        if (
            self.workload_config_sha256 is None
            or len(self.workload_config_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.workload_config_sha256
            )
        ):
            raise ValueError(
                "workload_config_sha256 must be a lowercase SHA-256 digest"
            )

    def semantic_dict(self) -> dict[str, Any]:
        """Fields that must remain identical across Radeon architectures."""

        return {
            "model": self.model,
            "temperature": self.temperature,
            "quantization": self.quantization,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "code_revision": self.code_revision,
            "runtime_revision": self.runtime_revision,
            "workload_config_sha256": self.workload_config_sha256,
        }


def load_suite(path: Path) -> BenchmarkSuite:
    raw = path.read_bytes()
    cases: list[PromptCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path.name}:{line_number}") from exc
        case = PromptCase.from_dict(item)
        if case.id in seen_ids:
            raise ValueError(f"duplicate benchmark case id: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"benchmark suite is empty: {path.name}")
    return BenchmarkSuite(
        path=path.name,
        raw_sha256=_digest(raw),
        raw_md5=_digest(raw, "md5"),  # noqa: S324 - identity, not security
        cases=tuple(cases),
    )


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _metric_summary(samples: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    values = [float(sample[field_name]) for sample in samples if sample.get(field_name) is not None]
    if not values:
        return {"count": 0, "median": None, "p95": None, "cv": None}
    mean = statistics.fmean(values)
    cv = statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": _nearest_rank(values, 0.95),
        "cv": cv,
    }


def run_benchmark(
    backend: LLMBackend,
    suite: BenchmarkSuite,
    config: BenchmarkConfig,
    *,
    hardware: HardwareManifest | None = None,
) -> dict[str, Any]:
    hardware = hardware or probe_hardware()

    # A full suite warmup catches model loading and prompt-shape JIT without contaminating samples.
    for _ in range(config.warmup_runs):
        for case in suite.cases:
            backend.stream_complete(
                ChatRequest(
                    model=config.model,
                    messages=case.messages,
                    max_tokens=case.max_tokens,
                    temperature=config.temperature,
                )
            )

    samples: list[dict[str, Any]] = []
    for case in suite.cases:
        request = ChatRequest(
            model=config.model,
            messages=case.messages,
            max_tokens=case.max_tokens,
            temperature=config.temperature,
        )
        request_hash = _digest(_canonical_json(request.to_openai(stream=True)))
        for repetition in range(config.repetitions):
            response, timing = backend.stream_complete(request)
            passed, failures = case.checks.evaluate(response.content)
            completion_tokens = response.usage.completion_tokens
            end_to_end_tokens_per_second = (
                completion_tokens / timing.total_seconds
                if completion_tokens is not None and timing.total_seconds > 0
                else None
            )
            post_first_content_seconds = None
            post_first_content_tokens_per_second = None
            if timing.time_to_first_token_seconds is not None:
                post_first_content_seconds = max(
                    timing.total_seconds - timing.time_to_first_token_seconds,
                    0.0,
                )
                if completion_tokens is not None and post_first_content_seconds > 0:
                    post_first_content_tokens_per_second = max(completion_tokens - 1, 0) / (
                        post_first_content_seconds
                    )
            samples.append(
                {
                    "case_id": case.id,
                    "repetition": repetition,
                    "request_sha256": request_hash,
                    "total_seconds": timing.total_seconds,
                    "client_ttft_seconds": timing.time_to_first_token_seconds,
                    "end_to_end_tokens_per_second": end_to_end_tokens_per_second,
                    "post_first_content_tokens_per_second": (
                        post_first_content_tokens_per_second
                    ),
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "output_chars": len(response.content),
                    "output_sha256": _digest(response.content.encode("utf-8")),
                    "quality_passed": passed,
                    "quality_failures": failures,
                    "finish_reason": response.finish_reason,
                }
            )

    case_summaries: dict[str, Any] = {}
    for case in suite.cases:
        case_samples = [sample for sample in samples if sample["case_id"] == case.id]
        case_summaries[case.id] = {
            "quality_pass_rate": sum(sample["quality_passed"] for sample in case_samples)
            / len(case_samples),
            "total_seconds": _metric_summary(case_samples, "total_seconds"),
            "client_ttft_seconds": _metric_summary(case_samples, "client_ttft_seconds"),
            "end_to_end_tokens_per_second": _metric_summary(
                case_samples, "end_to_end_tokens_per_second"
            ),
            "post_first_content_tokens_per_second": _metric_summary(
                case_samples, "post_first_content_tokens_per_second"
            ),
        }

    summary = {
        "sample_count": len(samples),
        "quality_pass_rate": sum(sample["quality_passed"] for sample in samples) / len(samples),
        "total_seconds": _metric_summary(samples, "total_seconds"),
        "client_ttft_seconds": _metric_summary(samples, "client_ttft_seconds"),
        "end_to_end_tokens_per_second": _metric_summary(
            samples, "end_to_end_tokens_per_second"
        ),
        "post_first_content_tokens_per_second": _metric_summary(
            samples, "post_first_content_tokens_per_second"
        ),
        "by_case": case_summaries,
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "label": config.label,
        "backend": backend.name,
        "hardware": hardware.to_dict(),
        "suite": {
            "path": suite.path,
            "raw_sha256": suite.raw_sha256,
            "raw_md5": suite.raw_md5,
            "case_ids": [case.id for case in suite.cases],
        },
        "config": asdict(config),
        "semantic_config": config.semantic_dict(),
        "semantic_config_sha256": _digest(_canonical_json(config.semantic_dict())),
        "samples": samples,
        "summary": summary,
    }


def save_benchmark(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
