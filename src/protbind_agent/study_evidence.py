"""Hash-bound, falsifiable study evidence for ProtBind benchmarks.

The case workflow answers whether one run completed and what artifacts it
produced.  This module deliberately operates one level above it: a frozen
study protocol declares claims and endpoints before an evidence packet binds
them to one or two redocking regression artifacts.

It never turns docking scores into affinity claims and never treats a pilot
sample as proof of generalisation.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file

PROTOCOL_SCHEMA_VERSION = "1.0"
EVIDENCE_SCHEMA_VERSION = "1.0"
PROTOCOL_KIND = "PROTBIND_STUDY_PROTOCOL"
EVIDENCE_KIND = "PROTBIND_ACADEMIC_EVIDENCE"

_ANALYSIS_TIMINGS = {
    "PROSPECTIVE_BEFORE_OUTCOME",
    "RETROSPECTIVE_AFTER_OUTCOME",
}
_SCOPES = {
    "PILOT_METHOD_VERIFICATION",
    "CONFIRMATORY_BENCHMARK",
}
_CLAIM_STATUSES = {
    "SUPPORTED",
    "CONTRADICTED",
    "INCONCLUSIVE",
    "NOT_EVALUATED",
}
_ENDPOINTS = {
    "candidate_workflow_completion_rate",
    "candidate_top1_pose_recovery_rate",
    "candidate_top5_oracle_pose_recovery_rate",
    "paired_top1_superiority",
}
_OPERATORS = {"gte", "eq", "paired_exact_superiority"}
_SHA256_LENGTH = 64


class StudyEvidenceIntegrityError(ValueError):
    """A study protocol or result binding is incomplete or inconsistent."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise StudyEvidenceIntegrityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StudyEvidenceIntegrityError(f"{name} must be an object")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudyEvidenceIntegrityError(f"{name} must be a non-empty string")
    return value


def _require_probability(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise StudyEvidenceIntegrityError(f"{name} must be a finite probability")
    return float(value)


def _protocol_core(protocol: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in protocol.items() if key != "protocol_sha256"}


def _validate_protocol_core(core: dict[str, Any]) -> None:
    if core.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise StudyEvidenceIntegrityError(
            f"study protocol schema_version must be {PROTOCOL_SCHEMA_VERSION}"
        )
    if core.get("kind") != PROTOCOL_KIND:
        raise StudyEvidenceIntegrityError(f"study protocol kind must be {PROTOCOL_KIND}")
    _require_nonempty_string(core.get("study_id"), "study_id")
    _require_nonempty_string(core.get("title"), "title")
    timing = core.get("analysis_timing")
    if timing not in _ANALYSIS_TIMINGS:
        raise StudyEvidenceIntegrityError(
            f"analysis_timing must be one of {sorted(_ANALYSIS_TIMINGS)}"
        )
    scope = core.get("scope")
    if scope not in _SCOPES:
        raise StudyEvidenceIntegrityError(f"scope must be one of {sorted(_SCOPES)}")
    if core.get("design") != "PAIRED_FROZEN_HOLDOUT":
        raise StudyEvidenceIntegrityError("design must be PAIRED_FROZEN_HOLDOUT")

    dataset = _require_mapping(core.get("dataset_binding"), "dataset_binding")
    _require_nonempty_string(dataset.get("name"), "dataset_binding.name")
    _require_sha256(dataset.get("holdout_sha256"), "dataset_binding.holdout_sha256")
    _require_sha256(
        dataset.get("holdout_selection_hash"),
        "dataset_binding.holdout_selection_hash",
    )
    frozen_case_count = dataset.get("frozen_case_count")
    if (
        isinstance(frozen_case_count, bool)
        or not isinstance(frozen_case_count, int)
        or frozen_case_count <= 0
    ):
        raise StudyEvidenceIntegrityError(
            "dataset_binding.frozen_case_count must be a positive integer"
        )

    arms = _require_mapping(core.get("arm_bindings"), "arm_bindings")
    for arm_name in ("candidate", "baseline"):
        arm = _require_mapping(arms.get(arm_name), f"arm_bindings.{arm_name}")
        _require_nonempty_string(arm.get("label"), f"arm_bindings.{arm_name}.label")
        _require_sha256(
            arm.get("regression_manifest_sha256"),
            f"arm_bindings.{arm_name}.regression_manifest_sha256",
        )

    primary = _require_mapping(core.get("primary_endpoint"), "primary_endpoint")
    if primary.get("id") != "candidate_top1_pose_recovery_rate":
        raise StudyEvidenceIntegrityError(
            "primary endpoint is fixed to candidate_top1_pose_recovery_rate"
        )
    if primary.get("success_definition") != (
        "PoseBusters-valid AND symmetry-aware heavy-atom RMSD <= 2.0 angstrom"
    ):
        raise StudyEvidenceIntegrityError(
            "primary endpoint must retain the frozen PB-valid/RMSD success definition"
        )
    if primary.get("denominator") != "all_frozen_cases":
        raise StudyEvidenceIntegrityError("primary endpoint denominator must be all_frozen_cases")

    plan = _require_mapping(core.get("statistical_plan"), "statistical_plan")
    confidence_level = _require_probability(
        plan.get("confidence_level"), "statistical_plan.confidence_level"
    )
    if not 0.5 < confidence_level < 1.0:
        raise StudyEvidenceIntegrityError("confidence_level must be between 0.5 and 1")
    alpha = _require_probability(plan.get("alpha"), "statistical_plan.alpha")
    if not 0.0 < alpha < 0.5:
        raise StudyEvidenceIntegrityError("alpha must be between 0 and 0.5")
    if plan.get("paired_test") != "EXACT_MCNEMAR_TWO_SIDED":
        raise StudyEvidenceIntegrityError("paired_test must be EXACT_MCNEMAR_TWO_SIDED")
    if plan.get("missing_case_policy") != "COUNT_AS_FAILURE":
        raise StudyEvidenceIntegrityError("missing_case_policy must be COUNT_AS_FAILURE")
    if plan.get("multiplicity") != "ONE_PRIMARY_OTHERS_DESCRIPTIVE":
        raise StudyEvidenceIntegrityError(
            "multiplicity must be ONE_PRIMARY_OTHERS_DESCRIPTIVE"
        )

    rules = core.get("claim_rules")
    if not isinstance(rules, list) or not rules:
        raise StudyEvidenceIntegrityError("claim_rules must be a non-empty list")
    claim_ids: set[str] = set()
    primary_claim_count = 0
    for index, value in enumerate(rules):
        rule = _require_mapping(value, f"claim_rules[{index}]")
        claim_id = _require_nonempty_string(rule.get("claim_id"), f"claim_rules[{index}].claim_id")
        if claim_id in claim_ids:
            raise StudyEvidenceIntegrityError(f"duplicate claim_id: {claim_id}")
        claim_ids.add(claim_id)
        _require_nonempty_string(rule.get("claim"), f"claim_rules[{index}].claim")
        endpoint = rule.get("endpoint")
        operator = rule.get("operator")
        if endpoint not in _ENDPOINTS:
            raise StudyEvidenceIntegrityError(f"unsupported claim endpoint: {endpoint}")
        if operator not in _OPERATORS:
            raise StudyEvidenceIntegrityError(f"unsupported claim operator: {operator}")
        if operator in {"gte", "eq"}:
            _require_probability(rule.get("threshold"), f"claim_rules[{index}].threshold")
        if endpoint == "paired_top1_superiority" and operator != "paired_exact_superiority":
            raise StudyEvidenceIntegrityError(
                "paired_top1_superiority requires paired_exact_superiority"
            )
        if bool(rule.get("primary")):
            primary_claim_count += 1
            if endpoint != primary["id"]:
                raise StudyEvidenceIntegrityError(
                    "the primary claim must use the declared primary endpoint"
                )
    if primary_claim_count != 1:
        raise StudyEvidenceIntegrityError("exactly one claim rule must be primary")

    for field in ("negative_controls", "ablations", "scientific_boundaries"):
        if not isinstance(core.get(field), list):
            raise StudyEvidenceIntegrityError(f"{field} must be a list")

    if (
        timing == "RETROSPECTIVE_AFTER_OUTCOME"
        and scope == "CONFIRMATORY_BENCHMARK"
    ):
        raise StudyEvidenceIntegrityError(
            "a retrospective analysis cannot claim confirmatory benchmark scope"
        )


def freeze_study_protocol(draft: dict[str, Any]) -> dict[str, Any]:
    """Validate and content-bind a study protocol draft."""

    draft = _require_mapping(draft, "protocol")
    if "protocol_sha256" in draft:
        raise StudyEvidenceIntegrityError("protocol draft is already frozen")
    core = dict(draft)
    _validate_protocol_core(core)
    return {
        **core,
        "protocol_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def load_frozen_study_protocol(path: Path) -> dict[str, Any]:
    """Load a frozen protocol and verify its self-commitment."""

    value = json.loads(path.read_text(encoding="utf-8"))
    protocol = _require_mapping(value, "protocol")
    _validate_protocol_core(_protocol_core(protocol))
    expected = sha256_bytes(canonical_json_bytes(_protocol_core(protocol)))
    if protocol.get("protocol_sha256") != expected:
        raise StudyEvidenceIntegrityError("study protocol hash mismatch")
    return protocol


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Return a two-sided Wilson score interval for one binomial proportion."""

    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
        or total <= 0
        or not 0 <= successes <= total
    ):
        raise ValueError("successes and total must satisfy 0 <= successes <= total, total > 0")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1")
    z = statistics.NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "confidence_level": confidence_level,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def exact_mcnemar_two_sided(candidate_wins: int, baseline_wins: int) -> float:
    """Exact two-sided McNemar p-value over discordant pairs."""

    if (
        isinstance(candidate_wins, bool)
        or isinstance(baseline_wins, bool)
        or not isinstance(candidate_wins, int)
        or not isinstance(baseline_wins, int)
        or candidate_wins < 0
        or baseline_wins < 0
    ):
        raise ValueError("discordant counts must be non-negative integers")
    discordant = candidate_wins + baseline_wins
    if discordant == 0:
        return 1.0
    smaller = min(candidate_wins, baseline_wins)
    lower_tail = sum(
        math.comb(discordant, index) for index in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def _load_regression(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    result = _require_mapping(value, name)
    if result.get("analysis") != "PROTBIND_REDOCK_REGRESSION":
        raise StudyEvidenceIntegrityError(f"{name} is not a redock regression result")
    core = {key: item for key, item in result.items() if key != "regression_sha256"}
    expected = sha256_bytes(canonical_json_bytes(core))
    if result.get("regression_sha256") != expected:
        raise StudyEvidenceIntegrityError(f"{name} regression hash mismatch")
    if result.get("evaluation_design") != "FROZEN_HOLDOUT":
        raise StudyEvidenceIntegrityError(f"{name} must use FROZEN_HOLDOUT design")
    cases = result.get("cases")
    if not isinstance(cases, list) or not cases:
        raise StudyEvidenceIntegrityError(f"{name}.cases must be non-empty")
    frozen = _require_mapping(result.get("denominators"), f"{name}.denominators").get(
        "frozen"
    )
    if frozen != len(cases):
        raise StudyEvidenceIntegrityError(f"{name} frozen denominator does not match cases")
    holdout = _require_mapping(result.get("holdout"), f"{name}.holdout")
    _require_sha256(holdout.get("sha256"), f"{name}.holdout.sha256")
    _require_sha256(
        holdout.get("selection_hash"),
        f"{name}.holdout.selection_hash",
    )
    manifest = _require_mapping(result.get("manifest"), f"{name}.manifest")
    _require_sha256(
        manifest.get("manifest_sha256"),
        f"{name}.manifest.manifest_sha256",
    )
    return result


def _verify_protocol_bindings(
    protocol: dict[str, Any],
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> None:
    dataset = protocol["dataset_binding"]
    candidate_holdout = candidate["holdout"]
    if candidate_holdout["sha256"] != dataset["holdout_sha256"]:
        raise StudyEvidenceIntegrityError(
            "candidate holdout does not match the frozen study protocol"
        )
    if candidate_holdout["selection_hash"] != dataset["holdout_selection_hash"]:
        raise StudyEvidenceIntegrityError(
            "candidate holdout selection does not match the frozen study protocol"
        )
    if len(candidate["cases"]) != dataset["frozen_case_count"]:
        raise StudyEvidenceIntegrityError(
            "candidate case count does not match the frozen study protocol"
        )
    arms = protocol["arm_bindings"]
    if (
        candidate["manifest"]["manifest_sha256"]
        != arms["candidate"]["regression_manifest_sha256"]
    ):
        raise StudyEvidenceIntegrityError(
            "candidate regression manifest does not match the frozen study protocol"
        )
    if baseline is None:
        return
    if baseline["holdout"]["sha256"] != dataset["holdout_sha256"]:
        raise StudyEvidenceIntegrityError(
            "baseline holdout does not match the frozen study protocol"
        )
    if baseline["holdout"]["selection_hash"] != dataset["holdout_selection_hash"]:
        raise StudyEvidenceIntegrityError(
            "baseline holdout selection does not match the frozen study protocol"
        )
    if len(baseline["cases"]) != dataset["frozen_case_count"]:
        raise StudyEvidenceIntegrityError(
            "baseline case count does not match the frozen study protocol"
        )
    if (
        baseline["manifest"]["manifest_sha256"]
        != arms["baseline"]["regression_manifest_sha256"]
    ):
        raise StudyEvidenceIntegrityError(
            "baseline regression manifest does not match the frozen study protocol"
        )


def _metric_estimate(
    successes: int,
    total: int,
    *,
    confidence_level: float,
    semantics: str,
) -> dict[str, Any]:
    return {
        "numerator": successes,
        "denominator": total,
        "rate": successes / total,
        "interval": {
            "method": "WILSON_SCORE_TWO_SIDED",
            **wilson_interval(
                successes,
                total,
                confidence_level=confidence_level,
            ),
        },
        "semantics": semantics,
    }


def _arm_summary(
    result: dict[str, Any],
    *,
    confidence_level: float,
) -> dict[str, Any]:
    denominators = _require_mapping(result["denominators"], "denominators")
    frozen = int(denominators["frozen"])
    top1 = _require_mapping(
        _require_mapping(result["pose_recovery_rates"], "pose_recovery_rates")["top1"],
        "pose_recovery_rates.top1",
    )
    top5 = _require_mapping(result["pose_recovery_rates"]["top5"], "pose_recovery_rates.top5")
    if int(top1["denominator"]) != frozen or int(top5["denominator"]) != frozen:
        raise StudyEvidenceIntegrityError("pose recovery denominator must include all frozen cases")
    completed = int(denominators["completed"])
    return {
        "gate_complete": bool(result.get("gate_complete")),
        "frozen_case_count": frozen,
        "completion": _metric_estimate(
            completed,
            frozen,
            confidence_level=confidence_level,
            semantics="Completed redocking cases over all frozen cases.",
        ),
        "top1_pose_recovery": _metric_estimate(
            int(top1["numerator"]),
            frozen,
            confidence_level=confidence_level,
            semantics=(
                "PB-valid AND symmetry-aware RMSD <= 2.0 A for the top-ranked pose."
            ),
        ),
        "top5_oracle_pose_recovery": _metric_estimate(
            int(top5["numerator"]),
            frozen,
            confidence_level=confidence_level,
            semantics=(
                "At least one of the first five poses passes the endpoint. This is an "
                "oracle diagnostic, not prospective top-1 selection performance."
            ),
        ),
        "regression_sha256": result["regression_sha256"],
        "holdout_sha256": _require_mapping(result.get("holdout"), "holdout").get("sha256"),
        "holdout_selection_hash": result["holdout"].get("selection_hash"),
    }


def _top1_outcomes(result: dict[str, Any]) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for index, value in enumerate(result["cases"]):
        case = _require_mapping(value, f"cases[{index}]")
        case_id = _require_nonempty_string(case.get("case_id"), f"cases[{index}].case_id")
        if case_id in outcomes:
            raise StudyEvidenceIntegrityError(f"duplicate regression case_id: {case_id}")
        metrics = case.get("metrics")
        success = False
        if isinstance(metrics, dict):
            recovery = metrics.get("pose_recovery")
            if isinstance(recovery, dict) and isinstance(
                recovery.get("top1_success"), bool
            ):
                success = recovery["top1_success"]
        outcomes[case_id] = success
    return outcomes


def _paired_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    confidence_level: float,
) -> dict[str, Any]:
    if candidate["holdout"].get("sha256") != baseline["holdout"].get("sha256"):
        raise StudyEvidenceIntegrityError("candidate and baseline holdout hashes differ")
    candidate_outcomes = _top1_outcomes(candidate)
    baseline_outcomes = _top1_outcomes(baseline)
    if set(candidate_outcomes) != set(baseline_outcomes):
        raise StudyEvidenceIntegrityError("candidate and baseline frozen case IDs differ")
    candidate_wins = 0
    baseline_wins = 0
    both_success = 0
    both_failure = 0
    discordant_cases: list[dict[str, Any]] = []
    for case_id in sorted(candidate_outcomes):
        candidate_success = candidate_outcomes[case_id]
        baseline_success = baseline_outcomes[case_id]
        if candidate_success and baseline_success:
            both_success += 1
        elif candidate_success:
            candidate_wins += 1
            discordant_cases.append(
                {
                    "case_id": case_id,
                    "direction": "CANDIDATE_ONLY_SUCCESS",
                }
            )
        elif baseline_success:
            baseline_wins += 1
            discordant_cases.append(
                {
                    "case_id": case_id,
                    "direction": "BASELINE_ONLY_SUCCESS",
                }
            )
        else:
            both_failure += 1
    total = len(candidate_outcomes)
    candidate_rate = sum(candidate_outcomes.values()) / total
    baseline_rate = sum(baseline_outcomes.values()) / total
    discordant = candidate_wins + baseline_wins
    return {
        "endpoint": "top1_pose_recovery",
        "case_count": total,
        "both_success": both_success,
        "both_failure": both_failure,
        "candidate_only_success": candidate_wins,
        "baseline_only_success": baseline_wins,
        "discordant_pair_count": discordant,
        "absolute_rate_difference": candidate_rate - baseline_rate,
        "exact_mcnemar_two_sided_p_value": exact_mcnemar_two_sided(
            candidate_wins,
            baseline_wins,
        ),
        "candidate_win_fraction_among_discordant": (
            _metric_estimate(
                candidate_wins,
                discordant,
                confidence_level=confidence_level,
                semantics=(
                    "Candidate wins among discordant pairs; undefined when no pairs differ."
                ),
            )
            if discordant
            else None
        ),
        "discordant_cases": discordant_cases,
    }


def _claim_result(
    rule: dict[str, Any],
    *,
    candidate: dict[str, Any],
    paired: dict[str, Any] | None,
    alpha: float,
) -> dict[str, Any]:
    endpoint = rule["endpoint"]
    operator = rule["operator"]
    observed: float | None
    p_value: float | None = None
    rationale: str
    if endpoint == "candidate_workflow_completion_rate":
        observed = candidate["completion"]["rate"]
    elif endpoint == "candidate_top1_pose_recovery_rate":
        observed = candidate["top1_pose_recovery"]["rate"]
    elif endpoint == "candidate_top5_oracle_pose_recovery_rate":
        observed = candidate["top5_oracle_pose_recovery"]["rate"]
    else:
        observed = paired["absolute_rate_difference"] if paired is not None else None

    if operator == "gte":
        threshold = float(rule["threshold"])
        status = "SUPPORTED" if observed is not None and observed >= threshold else "CONTRADICTED"
        rationale = (
            f"Observed rate {observed:.6f} is "
            f"{'at least' if status == 'SUPPORTED' else 'below'} "
            f"the frozen descriptive threshold {threshold:.6f}."
        )
    elif operator == "eq":
        threshold = float(rule["threshold"])
        status = "SUPPORTED" if observed == threshold else "CONTRADICTED"
        rationale = (
            f"Observed rate {observed:.6f} "
            f"{'equals' if status == 'SUPPORTED' else 'does not equal'} "
            f"the frozen threshold {threshold:.6f}."
        )
    else:
        if paired is None:
            status = "NOT_EVALUATED"
            rationale = "No hash-matched baseline result was supplied."
        else:
            p_value = paired["exact_mcnemar_two_sided_p_value"]
            if p_value < alpha and observed is not None and observed > 0:
                status = "SUPPORTED"
                rationale = (
                    "The candidate has a positive paired rate difference and the exact "
                    "two-sided McNemar p-value is below alpha."
                )
            elif p_value < alpha and observed is not None and observed < 0:
                status = "CONTRADICTED"
                rationale = (
                    "The candidate has a negative paired rate difference and the exact "
                    "two-sided McNemar p-value is below alpha."
                )
            else:
                status = "INCONCLUSIVE"
                rationale = (
                    "The exact paired comparison does not cross the frozen alpha; a raw "
                    "rate difference alone is not evidence of superiority."
                )
    if status not in _CLAIM_STATUSES:
        raise AssertionError("invalid internal claim status")
    return {
        "claim_id": rule["claim_id"],
        "claim": rule["claim"],
        "primary": bool(rule.get("primary")),
        "endpoint": endpoint,
        "status": status,
        "observed": observed,
        "threshold": rule.get("threshold"),
        "p_value": p_value,
        "rationale": rationale,
        "evidence_refs": [
            f"sha256:{candidate['regression_sha256']}",
            *(
                [f"sha256:{paired['baseline_regression_sha256']}"]
                if paired is not None and endpoint == "paired_top1_superiority"
                else []
            ),
        ],
    }


def build_academic_evidence(
    protocol_path: Path,
    candidate_result_path: Path,
    *,
    baseline_result_path: Path | None = None,
) -> dict[str, Any]:
    """Build one deterministic academic evidence packet."""

    protocol = load_frozen_study_protocol(protocol_path)
    candidate_result = _load_regression(candidate_result_path, "candidate_result")
    baseline_result = (
        _load_regression(baseline_result_path, "baseline_result")
        if baseline_result_path is not None
        else None
    )
    _verify_protocol_bindings(protocol, candidate_result, baseline_result)
    plan = protocol["statistical_plan"]
    confidence_level = float(plan["confidence_level"])
    alpha = float(plan["alpha"])
    candidate = _arm_summary(
        candidate_result,
        confidence_level=confidence_level,
    )
    baseline = (
        _arm_summary(baseline_result, confidence_level=confidence_level)
        if baseline_result is not None
        else None
    )
    paired = (
        _paired_comparison(
            candidate_result,
            baseline_result,
            confidence_level=confidence_level,
        )
        if baseline_result is not None
        else None
    )
    if paired is not None and baseline is not None:
        paired["candidate_regression_sha256"] = candidate["regression_sha256"]
        paired["baseline_regression_sha256"] = baseline["regression_sha256"]
    claims = [
        _claim_result(
            rule,
            candidate=candidate,
            paired=paired,
            alpha=alpha,
        )
        for rule in protocol["claim_rules"]
    ]
    primary_claim = next(claim for claim in claims if claim["primary"])
    core = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "study_id": protocol["study_id"],
        "title": protocol["title"],
        "scope": protocol["scope"],
        "analysis_timing": protocol["analysis_timing"],
        "design": protocol["design"],
        "input_bindings": {
            "protocol": {
                "filename": protocol_path.name,
                "file_sha256": sha256_file(protocol_path),
                "protocol_sha256": protocol["protocol_sha256"],
            },
            "candidate_result": {
                "filename": candidate_result_path.name,
                "file_sha256": sha256_file(candidate_result_path),
                "regression_sha256": candidate["regression_sha256"],
            },
            "baseline_result": (
                {
                    "filename": baseline_result_path.name,
                    "file_sha256": sha256_file(baseline_result_path),
                    "regression_sha256": baseline["regression_sha256"],
                }
                if baseline_result_path is not None and baseline is not None
                else None
            ),
        },
        "implementation": {
            "module": "protbind_agent.study_evidence",
            "source_sha256": sha256_file(Path(__file__)),
        },
        "statistical_plan": plan,
        "candidate": candidate,
        "baseline": baseline,
        "paired_comparison": paired,
        "claims": claims,
        "primary_claim_status": primary_claim["status"],
        "registered_negative_controls": protocol["negative_controls"],
        "registered_ablations": protocol["ablations"],
        "scientific_boundaries": protocol["scientific_boundaries"],
        "interpretation": {
            "evidence_grade": (
                "RETROSPECTIVE_PILOT"
                if protocol["analysis_timing"] == "RETROSPECTIVE_AFTER_OUTCOME"
                else "PROSPECTIVE_PROTOCOL_BOUND"
            ),
            "generalisation_claim": "NOT_EVALUATED",
            "affinity_claim": "NOT_EVALUATED",
            "screening_hit_rate_claim": "NOT_EVALUATED",
            "statement": (
                "SUPPORTED means the frozen rule was met by these bound artifacts. It "
                "does not mean biological truth, experimental binding, or universal "
                "generalisation was proven."
            ),
        },
    }
    return {
        **core,
        "evidence_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_academic_evidence_markdown(packet: dict[str, Any]) -> str:
    """Render a reviewer-readable report without changing packet semantics."""

    candidate = packet["candidate"]
    lines = [
        f"# {_escape_markdown(packet['title'])}",
        "",
        "> This is a hash-bound computational evidence packet. It is not evidence of "
        "experimental binding affinity or clinical effect.",
        "",
        "## Study identity",
        "",
        f"- Study: `{packet['study_id']}`",
        f"- Scope: `{packet['scope']}`",
        f"- Analysis timing: `{packet['analysis_timing']}`",
        f"- Protocol: `sha256:{packet['input_bindings']['protocol']['protocol_sha256']}`",
        f"- Evidence packet: `sha256:{packet['evidence_sha256']}`",
        "",
        "## Candidate estimates",
        "",
        "| Endpoint | Successes / total | Estimate | Wilson interval |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Workflow completion", "completion"),
        ("Top-1 PB-valid + RMSD ≤ 2 Å", "top1_pose_recovery"),
        ("Top-5 oracle PB-valid + RMSD ≤ 2 Å", "top5_oracle_pose_recovery"),
    ):
        metric = candidate[key]
        interval = metric["interval"]
        lines.append(
            f"| {label} | {metric['numerator']} / {metric['denominator']} | "
            f"{metric['rate']:.3f} | [{interval['lower']:.3f}, "
            f"{interval['upper']:.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Claim–evidence matrix",
            "",
            "| Claim | Role | Status | Observed | Evidence |",
            "|---|---|---|---:|---|",
        ]
    )
    for claim in packet["claims"]:
        observed = "n/a" if claim["observed"] is None else f"{claim['observed']:.6f}"
        evidence = ", ".join(f"`{value}`" for value in claim["evidence_refs"])
        lines.append(
            f"| {_escape_markdown(claim['claim'])} | "
            f"{'primary' if claim['primary'] else 'secondary'} | "
            f"`{claim['status']}` | {observed} | {evidence} |"
        )
        lines.append(
            f"| ↳ rationale |  |  |  | {_escape_markdown(claim['rationale'])} |"
        )

    paired = packet.get("paired_comparison")
    if paired is not None:
        lines.extend(
            [
                "",
                "## Paired protocol comparison",
                "",
                f"- Candidate-only successes: {paired['candidate_only_success']}",
                f"- Baseline-only successes: {paired['baseline_only_success']}",
                f"- Absolute top-1 rate difference: "
                f"{paired['absolute_rate_difference']:.3f}",
                f"- Exact two-sided McNemar p-value: "
                f"{paired['exact_mcnemar_two_sided_p_value']:.6f}",
                "",
                "A positive raw difference is descriptive. Superiority is only supported "
                "when the frozen paired test crosses alpha in the declared direction.",
            ]
        )

    lines.extend(
        [
            "",
            "## Registered but not automatically credited",
            "",
            "| Type | ID | Status | Purpose |",
            "|---|---|---|---|",
        ]
    )
    for kind, values in (
        ("negative control", packet["registered_negative_controls"]),
        ("ablation", packet["registered_ablations"]),
    ):
        for value in values:
            lines.append(
                f"| {kind} | `{_escape_markdown(value.get('id', 'unknown'))}` | "
                f"`{_escape_markdown(value.get('status', 'NOT_RUN'))}` | "
                f"{_escape_markdown(value.get('purpose', ''))} |"
            )

    lines.extend(["", "## Scientific boundaries", ""])
    lines.extend(f"- {_escape_markdown(value)}" for value in packet["scientific_boundaries"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            packet["interpretation"]["statement"],
            "",
            "Generalisation, affinity, and screening hit-rate claims are all "
            "`NOT_EVALUATED` by this packet.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, data: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def persist_frozen_study_protocol(protocol: dict[str, Any], output: Path) -> None:
    """Persist a self-hashed frozen protocol."""

    core = _protocol_core(protocol)
    _validate_protocol_core(core)
    if protocol.get("protocol_sha256") != sha256_bytes(canonical_json_bytes(core)):
        raise StudyEvidenceIntegrityError("study protocol hash mismatch")
    _atomic_write(
        output,
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n",
    )


def persist_academic_evidence(
    packet: dict[str, Any],
    output: Path,
    *,
    markdown_output: Path | None = None,
) -> None:
    """Persist JSON evidence and an optional human-readable companion report."""

    core = {key: value for key, value in packet.items() if key != "evidence_sha256"}
    if packet.get("evidence_sha256") != sha256_bytes(canonical_json_bytes(core)):
        raise StudyEvidenceIntegrityError("academic evidence packet hash mismatch")
    _atomic_write(
        output,
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n",
    )
    if markdown_output is not None:
        _atomic_write(
            markdown_output,
            render_academic_evidence_markdown(packet).encode("utf-8"),
        )
