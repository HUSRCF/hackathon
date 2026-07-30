"""Deterministic, artifact-cited workflow experience memory."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import ArtifactStore, sha256_bytes, sha256_file
from .manifest import RunState
from .models import ArtifactRef
from .validation import classify_evidence
from .workflow import ProtBindWorkflow, _validation_bundle

EXPERIENCE_SCHEMA_VERSION = "1.0"


def _terms(value: str) -> set[str]:
    normalized = value.lower()
    terms = set(re.findall(r"[a-z0-9_.-]+", normalized))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        terms.update(sequence)
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms


def _artifact_ids(value: dict[str, ArtifactRef]) -> list[str]:
    return sorted({reference.artifact_id for reference in value.values()})


def _find_version(value: Any, name: str) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() != name:
                continue
            if isinstance(item, str) and item.strip():
                return item.strip()
            if (
                isinstance(item, dict)
                and isinstance(item.get("version"), str)
                and item["version"].strip()
            ):
                return item["version"].strip()
        for item in value.values():
            found = _find_version(item, name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_version(item, name)
            if found is not None:
                return found
    return None


class ExperienceStore:
    """Artifact is exact; SQLite is a replaceable local retrieval projection."""

    def __init__(self, workspace: Path, workflow: ProtBindWorkflow | None = None) -> None:
        self.workspace = workspace.resolve()
        self.workflow = workflow or ProtBindWorkflow(self.workspace)
        self.artifacts = ArtifactStore(self.workspace)
        self.root = self.workspace / "experience"
        self.catalog = self.root / "catalog.sqlite"
        self._initialize()

    def _initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.catalog) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experience (
                    experience_id TEXT PRIMARY KEY,
                    artifact_json TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    receptor_identity TEXT NOT NULL,
                    evidence_grade TEXT NOT NULL,
                    search_text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS experience_case_idx
                    ON experience(case_id, mode, evidence_grade);
                """
            )

    def _toolchain(self, manifest: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        for key in ("support_vina_environment_lock", "support_validation_toolchain"):
            reference = manifest.artifacts.get(key)
            if reference is None:
                continue
            try:
                value = self.artifacts.read_json(reference)
            except (OSError, TypeError, ValueError):
                continue
            for name in ("vina", "meeko", "posebusters", "prolif", "spyrmsd", "openmm"):
                version = _find_version(value, name)
                if version is not None:
                    result[name] = version
        return dict(sorted(result.items()))

    def _selected_candidates(self, manifest: Any) -> list[str]:
        record = manifest.stage_records.get(RunState.SELECTED.value)
        if record is None or not record.outputs:
            return []
        value = self.artifacts.read_json(record.outputs[0])
        candidates = value.get("candidates") if isinstance(value, dict) else None
        if not isinstance(candidates, list):
            return []
        return [
            str(item.get("candidate_id", item.get("molecule_id")))
            for item in candidates
            if isinstance(item, dict) and (item.get("candidate_id") or item.get("molecule_id"))
        ]

    def _evidence(self, manifest: Any) -> tuple[str, list[dict[str, str]]]:
        record = manifest.stage_records.get(RunState.VALIDATED.value)
        if record is None or not record.outputs:
            return "UNKNOWN", []
        value = self.artifacts.read_json(record.outputs[0])
        candidates = value.get("candidates") if isinstance(value, dict) else None
        if not isinstance(candidates, list):
            return "UNKNOWN", []
        values: list[dict[str, str]] = []
        priority = {
            "REDOCKING_RECOVERED": 0,
            "REFERENCE_SUPPORTED": 0,
            "METHOD_CONSENSUS": 1,
            "CONSENSUS_SUPPORTED": 1,
            "HYPOTHESIS_ONLY": 2,
            "REJECTED": 3,
        }
        for item in candidates:
            if not isinstance(item, dict):
                continue
            try:
                grade = classify_evidence(
                    _validation_bundle(item.get("bundle")),
                    has_reference_pose=bool(item.get("has_reference_pose", False)),
                ).value
            except (TypeError, ValueError):
                continue
            values.append(
                {
                    "candidate_id": str(
                        item.get("candidate_id", item.get("molecule_id", "unknown"))
                    ),
                    "molecule_id": str(item.get("molecule_id", "unknown")),
                    "evidence_grade": grade,
                }
            )
        values.sort(
            key=lambda item: (
                priority.get(item["evidence_grade"], 99),
                item["molecule_id"],
                item["candidate_id"],
            )
        )
        return (values[0]["evidence_grade"] if values else "UNKNOWN"), values

    def write(self, run_id: str, preference: str | None = None) -> dict[str, Any]:
        manifest = self.workflow.manifests.load(run_id)
        case = self.workflow.audit_manifest(manifest)
        if manifest.last_completed_stage is not RunState.REPORTED:
            raise ValueError("experience memory requires an audited REPORTED run")
        receptor = (
            case.target.structure
            or manifest.artifacts.get("support_esmfold_structure")
            or manifest.artifacts.get("support_receptor_structure")
        )
        receptor_identity = receptor.artifact_id if receptor is not None else "unknown"
        evidence_grade, candidate_evidence = self._evidence(manifest)
        cited = {
            **manifest.input_artifacts,
            **manifest.artifacts,
        }
        for record in manifest.stage_records.values():
            for index, reference in enumerate(record.outputs):
                cited[f"stage_{record.stage.value}_{index}"] = reference
        manifest_path = self.workflow.manifests.path_for(run_id)
        record = {
            "schema_version": EXPERIENCE_SCHEMA_VERSION,
            "kind": "protbind.experience-record",
            "case_id": case.case_id,
            "run_id": run_id,
            "stage": RunState.REPORTED.value,
            "mode": case.mode.value,
            "receptor_identity": receptor_identity,
            "ligand_identity": (
                sha256_bytes(case.ligand.smiles.encode("utf-8"))
                if case.ligand is not None and case.ligand.smiles
                else (
                    case.ligand.structure.artifact_id
                    if case.ligand is not None and case.ligand.structure is not None
                    else "unknown"
                )
            ),
            "selected_candidates": self._selected_candidates(manifest),
            "candidate_evidence": candidate_evidence,
            "failure_codes": sorted({failure.code for failure in manifest.failures}),
            "toolchain": self._toolchain(manifest),
            "artifact_ids": _artifact_ids(cited),
            "evidence_grade": evidence_grade,
            "preference": preference.strip() if preference and preference.strip() else None,
            "source_manifest_sha256": sha256_file(manifest_path),
            "semantics": (
                "Experience hint only. It cannot alter docking boxes, seeds, thresholds, "
                "stage gates, or scientific conclusions."
            ),
        }
        artifact = self.artifacts.put_json(
            record,
            producer="protbind.experience",
            producer_version=__version__,
            source=f"run:{run_id}",
        )
        search_text = " ".join(
            [
                case.case_id,
                run_id,
                case.mode.value,
                receptor_identity,
                str(record["ligand_identity"]),
                evidence_grade,
                " ".join(record["selected_candidates"]),
                " ".join(record["failure_codes"]),
                " ".join(f"{key} {value}" for key, value in record["toolchain"].items()),
                preference or "",
            ]
        )
        experience_id = artifact.sha256
        with sqlite3.connect(self.catalog) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experience (
                    experience_id, artifact_json, case_id, run_id, mode,
                    receptor_identity, evidence_grade, search_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience_id,
                    json.dumps(artifact.to_dict(), sort_keys=True),
                    case.case_id,
                    run_id,
                    case.mode.value,
                    receptor_identity,
                    evidence_grade,
                    search_text,
                ),
            )
        return {
            "written": True,
            "experience_id": experience_id,
            "artifact": artifact.to_dict(),
            "case_id": case.case_id,
            "run_id": run_id,
            "evidence_grade": evidence_grade,
            "scientific_state_changed": False,
            "semantics": record["semantics"],
        }

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("experience query cannot be empty")
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        query_terms = _terms(query)
        with sqlite3.connect(self.catalog) as connection:
            rows = connection.execute(
                "SELECT artifact_json, case_id, run_id, mode, receptor_identity, "
                "evidence_grade, search_text FROM experience"
            ).fetchall()
        hits: list[dict[str, Any]] = []
        for (
            artifact_json,
            case_id,
            run_id,
            mode,
            receptor_identity,
            evidence_grade,
            search_text,
        ) in rows:
            terms = _terms(search_text)
            overlap = len(query_terms & terms)
            if overlap == 0:
                continue
            reference = ArtifactRef.from_dict(json.loads(artifact_json))
            self.artifacts.resolve(reference)
            hits.append(
                {
                    "score": overlap / max(len(query_terms), 1),
                    "case_id": case_id,
                    "run_id": run_id,
                    "mode": mode,
                    "receptor_identity": receptor_identity,
                    "evidence_grade": evidence_grade,
                    "artifact": {
                        **reference.to_dict(),
                        "artifact_id": reference.artifact_id,
                    },
                }
            )
        hits.sort(
            key=lambda item: (
                -item["score"],
                item["case_id"],
                item["run_id"],
            )
        )
        return {
            "query": query,
            "hits": hits[:top_k],
            "semantics": (
                "Experience hints only; callers must re-run current scientific gates and "
                "must not copy boxes, seeds, thresholds, or conclusions automatically."
            ),
        }
