"""Compatibility and performance comparison across two real Radeon architectures."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class VerificationReport:
    compatible: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def load_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"benchmark result must be a JSON object: {path}")
    return result


def _architectures(result: dict[str, Any]) -> set[str]:
    hardware = result.get("hardware") or {}
    return {str(item).lower() for item in hardware.get("architectures", [])}


def _request_map(result: dict[str, Any]) -> dict[tuple[str, int], str]:
    return {
        (str(sample.get("case_id")), int(sample.get("repetition", -1))): str(
            sample.get("request_sha256")
        )
        for sample in result.get("samples", [])
    }


def _median(result: dict[str, Any], metric: str) -> float | None:
    value = (((result.get("summary") or {}).get(metric) or {}).get("median"))
    return float(value) if value is not None else None


def compare_results(
    primary: dict[str, Any],
    verifier: dict[str, Any],
    *,
    expected_primary_arch: str = "gfx1100",
    expected_verifier_arch: str = "gfx1201",
    strict_output: bool = False,
) -> VerificationReport:
    errors: list[str] = []
    warnings: list[str] = []

    if primary.get("schema_version") != verifier.get("schema_version"):
        errors.append("benchmark schema versions differ")
    if primary.get("backend") != verifier.get("backend"):
        errors.append("inference backends differ")
    if primary.get("backend") == "deepseek" or verifier.get("backend") == "deepseek":
        errors.append("cloud DeepSeek results cannot prove local Radeon execution")
    if (primary.get("suite") or {}).get("raw_sha256") != (
        verifier.get("suite") or {}
    ).get("raw_sha256"):
        errors.append("benchmark suite SHA-256 differs")
    if (primary.get("suite") or {}).get("raw_md5") != (
        verifier.get("suite") or {}
    ).get("raw_md5"):
        errors.append("benchmark suite MD5 differs")
    if primary.get("semantic_config_sha256") != verifier.get("semantic_config_sha256"):
        errors.append("model or public semantic configuration differs")
    if _request_map(primary) != _request_map(verifier):
        errors.append("serialized request hashes differ")

    primary_arches = _architectures(primary)
    verifier_arches = _architectures(verifier)
    if expected_primary_arch not in primary_arches:
        errors.append(f"primary result does not prove {expected_primary_arch}")
    if expected_verifier_arch not in verifier_arches:
        errors.append(f"verifier result does not prove {expected_verifier_arch}")
    if (primary.get("hardware") or {}).get("hsa_override_active"):
        errors.append("primary used HSA_OVERRIDE_GFX_VERSION")
    if (verifier.get("hardware") or {}).get("hsa_override_active"):
        errors.append("verifier used HSA_OVERRIDE_GFX_VERSION")

    primary_config = primary.get("config") or {}
    verifier_config = verifier.get("config") or {}
    for label, config in (("primary", primary_config), ("verifier", verifier_config)):
        if not _SHA256.fullmatch(str(config.get("model_sha256") or "")):
            errors.append(f"{label} model weight SHA-256 is missing or invalid")
        if not _SHA256.fullmatch(str(config.get("workload_config_sha256") or "")):
            errors.append(f"{label} workload configuration SHA-256 is missing or invalid")
        for field in (
            "quantization",
            "model_revision",
            "code_revision",
            "runtime_revision",
        ):
            value = config.get(field)
            if not isinstance(value, str) or not value.strip() or value == "unspecified":
                errors.append(f"{label} {field} is missing")

    primary_outputs = {
        (sample.get("case_id"), sample.get("repetition")): sample.get("output_sha256")
        for sample in primary.get("samples", [])
    }
    verifier_outputs = {
        (sample.get("case_id"), sample.get("repetition")): sample.get("output_sha256")
        for sample in verifier.get("samples", [])
    }
    common = set(primary_outputs) & set(verifier_outputs)
    exact_matches = sum(primary_outputs[key] == verifier_outputs[key] for key in common)
    exact_match_rate = exact_matches / len(common) if common else None
    if strict_output and exact_match_rate != 1.0:
        errors.append("generated outputs are not byte-identical under strict mode")
    elif exact_match_rate != 1.0:
        warnings.append(
            "generated text differs across architectures; inspect quality checks instead of "
            "assuming byte identity"
        )

    if (primary.get("summary") or {}).get("quality_pass_rate") != 1.0:
        errors.append("primary quality checks did not pass 100%")
    if (verifier.get("summary") or {}).get("quality_pass_rate") != 1.0:
        errors.append("verifier quality checks did not pass 100%")

    metric_names = (
        "total_seconds",
        "client_ttft_seconds",
        "end_to_end_tokens_per_second",
        "post_first_content_tokens_per_second",
    )
    metrics: dict[str, Any] = {"exact_output_match_rate": exact_match_rate}
    for metric in metric_names:
        left = _median(primary, metric)
        right = _median(verifier, metric)
        ratio = right / left if left not in {None, 0.0} and right is not None else None
        metrics[metric] = {
            "primary_median": left,
            "verifier_median": right,
            "verifier_over_primary": ratio,
        }

    for label, result in (("primary", primary), ("verifier", verifier)):
        by_case = ((result.get("summary") or {}).get("by_case") or {})
        unstable_cases = [
            case_id
            for case_id, summary in by_case.items()
            if (((summary.get("total_seconds") or {}).get("cv")) or 0.0) > 0.05
        ]
        if unstable_cases:
            warnings.append(
                f"{label} total-latency CV exceeds 5% for cases: "
                f"{', '.join(sorted(unstable_cases))}; rerun in a stable thermal state"
            )

    return VerificationReport(
        compatible=not errors,
        errors=tuple(errors),
        warnings=tuple(dict.fromkeys(warnings)),
        metrics=metrics,
    )


def report_json(report: VerificationReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
