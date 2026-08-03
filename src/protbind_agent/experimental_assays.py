"""Immutable experimental-assay ingestion and deterministic curve fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import ArtifactStore, canonical_json_bytes, sha256_file
from .models import ArtifactRef

ASSAY_SCHEMA_VERSION = "1.0"
ASSAY_TYPES = (
    "coip-western",
    "cetsa",
    "spr",
    "bli",
    "enzyme-activity",
    "cellular-response",
)
FIT_MODELS = ("four-parameter-logistic", "one-site-binding")
_REQUIRED_COLUMNS = (
    "experiment_id",
    "assay_type",
    "target_id",
    "candidate_id",
    "batch_id",
    "lab_id",
    "condition_id",
    "replicate",
    "concentration",
    "concentration_unit",
    "response",
    "response_unit",
    "control_type",
)


@dataclass(frozen=True, slots=True)
class AssayImportPreview:
    plan_id: str
    source_sha256: str
    experiment_id: str
    assay_type: str
    target_id: str
    candidate_id: str
    row_count: int
    batch_count: int
    lab_count: int
    control_counts: dict[str, int]
    concentration_unit: str
    response_unit: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSAY_SCHEMA_VERSION,
            "kind": "experimental-assay-import-preview",
            "plan_id": self.plan_id,
            "source_sha256": self.source_sha256,
            "experiment_id": self.experiment_id,
            "assay_type": self.assay_type,
            "target_id": self.target_id,
            "candidate_id": self.candidate_id,
            "row_count": self.row_count,
            "batch_count": self.batch_count,
            "lab_count": self.lab_count,
            "control_counts": dict(self.control_counts),
            "concentration_unit": self.concentration_unit,
            "response_unit": self.response_unit,
            "writes_performed": False,
            "raw_measurements_returned": False,
            "next_action": "commit the same source with this exact plan_id after approval",
        }


class ExperimentalAssayStore:
    """Append-only experiment catalog backed by immutable content artifacts."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.artifacts = ArtifactStore(self.workspace)
        self.database = self.workspace / "experiments" / "catalog.sqlite"

    def preview_import(self, source: Path) -> dict[str, Any]:
        rows = _read_assay_rows(source)
        return _build_preview(source, rows).to_dict()

    def commit_import(
        self,
        source: Path,
        *,
        plan_id: str,
        data_access_confirmed: bool,
    ) -> dict[str, Any]:
        if data_access_confirmed is not True:
            raise PermissionError("experiment import requires fresh private-data approval")
        rows = _read_assay_rows(source)
        preview = _build_preview(source, rows)
        if preview.plan_id != plan_id:
            raise PermissionError("experiment import plan is stale or does not match the source")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._initialize(connection)
            existing = connection.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ?",
                (preview.experiment_id,),
            ).fetchone()
            if existing is not None:
                raise FileExistsError(
                    "experiment already exists; use a future approved supersede operation"
                )
        raw_artifact = self.artifacts.import_file(
            source,
            media_type=(
                "text/tab-separated-values"
                if source.suffix.lower() in {".tsv", ".tab"}
                else "text/csv"
            ),
            producer="protbind.experiment-import",
            producer_version=__version__,
            source=f"local-private:{source.name}",
        )
        canonical = {
            "schema_version": ASSAY_SCHEMA_VERSION,
            "kind": "experimental-assay-series",
            "experiment_id": preview.experiment_id,
            "assay_type": preview.assay_type,
            "target_id": preview.target_id,
            "candidate_id": preview.candidate_id,
            "concentration_unit": preview.concentration_unit,
            "response_unit": preview.response_unit,
            "source_sha256": preview.source_sha256,
            "rows": rows,
        }
        canonical_artifact = self.artifacts.put_json(
            canonical,
            producer="protbind.experiment-normalizer",
            producer_version=__version__,
            source=raw_artifact.artifact_id,
        )
        imported_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            self._initialize(connection)
            existing = connection.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ?",
                (preview.experiment_id,),
            ).fetchone()
            if existing is not None:
                raise FileExistsError(
                    "experiment already exists; use a future approved supersede operation"
                )
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_id, revision, assay_type, target_id, candidate_id,
                    source_sha256, raw_artifact_json, canonical_artifact_json,
                    imported_at, state
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
                """,
                (
                    preview.experiment_id,
                    preview.assay_type,
                    preview.target_id,
                    preview.candidate_id,
                    preview.source_sha256,
                    _artifact_json(raw_artifact),
                    _artifact_json(canonical_artifact),
                    imported_at,
                ),
            )
            connection.executemany(
                """
                INSERT INTO measurements (
                    experiment_id, revision, row_index, batch_id, lab_id,
                    condition_id, replicate, concentration, concentration_unit,
                    response, response_unit, control_type
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        preview.experiment_id,
                        index,
                        row["batch_id"],
                        row["lab_id"],
                        row["condition_id"],
                        row["replicate"],
                        row["concentration"],
                        row["concentration_unit"],
                        row["response"],
                        row["response_unit"],
                        row["control_type"],
                    )
                    for index, row in enumerate(rows)
                ],
            )
            connection.commit()
        receipt = {
            **preview.to_dict(),
            "kind": "experimental-assay-import-receipt",
            "status": "IMPORTED",
            "revision": 1,
            "writes_performed": True,
            "raw_artifact": raw_artifact.to_dict(),
            "canonical_artifact": canonical_artifact.to_dict(),
            "imported_at": imported_at,
            "mutation_policy": (
                "append-only; replacement and deletion are unavailable; future changes "
                "must use separately approved supersede or tombstone operations"
            ),
        }
        receipt_artifact = self.artifacts.put_json(
            receipt,
            producer="protbind.experiment-import-receipt",
            producer_version=__version__,
            source=canonical_artifact.artifact_id,
        )
        return {
            "status": "IMPORTED",
            "experiment_id": preview.experiment_id,
            "revision": 1,
            "row_count": preview.row_count,
            "receipt_artifact": receipt_artifact.to_dict(),
            "canonical_artifact": canonical_artifact.to_dict(),
            "raw_artifact": raw_artifact.to_dict(),
        }

    def list_experiments(self, *, limit: int = 100) -> dict[str, Any]:
        if limit < 1 or limit > 1000:
            raise ValueError("experiment list limit must be between 1 and 1000")
        if not self.database.is_file():
            return {"experiments": [], "count": 0, "raw_measurements_returned": False}
        with self._connect() as connection:
            self._initialize(connection)
            rows = connection.execute(
                """
                SELECT experiment_id, revision, assay_type, target_id, candidate_id,
                       source_sha256, canonical_artifact_json, imported_at, state
                FROM experiments
                ORDER BY imported_at DESC, experiment_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return {
            "experiments": [dict(row) for row in rows],
            "count": len(rows),
            "raw_measurements_returned": False,
        }

    def fit_curve(
        self,
        *,
        experiment_id: str,
        model: str,
        data_access_confirmed: bool,
    ) -> dict[str, Any]:
        if data_access_confirmed is not True:
            raise PermissionError("curve fitting requires fresh private-data approval")
        if model not in FIT_MODELS:
            raise ValueError(f"unsupported deterministic fit model: {model}")
        with self._connect() as connection:
            self._initialize(connection)
            experiment = connection.execute(
                """
                SELECT * FROM experiments
                WHERE experiment_id = ? AND state = 'ACTIVE'
                ORDER BY revision DESC LIMIT 1
                """,
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                raise KeyError("active experiment does not exist")
            measurements = connection.execute(
                """
                SELECT concentration, response, control_type, row_index
                FROM measurements
                WHERE experiment_id = ? AND revision = ?
                ORDER BY row_index
                """,
                (experiment_id, experiment["revision"]),
            ).fetchall()
        selected = [
            row
            for row in measurements
            if str(row["control_type"]).lower() in {"none", "sample", "treatment"}
        ]
        if len(selected) < 5:
            raise ValueError("curve fitting requires at least five non-control measurements")
        x = [float(row["concentration"]) for row in selected]
        y = [float(row["response"]) for row in selected]
        if model == "four-parameter-logistic":
            fit = _fit_four_parameter_logistic(x, y)
        else:
            fit = _fit_one_site_binding(x, y)
        fit_payload = {
            "schema_version": ASSAY_SCHEMA_VERSION,
            "kind": "experimental-assay-curve-fit",
            "status": "FITTED",
            "experiment_id": experiment_id,
            "experiment_revision": experiment["revision"],
            "source_sha256": experiment["source_sha256"],
            "model": model,
            "point_count": len(selected),
            "fit": fit,
            "scientific_semantics": (
                "Deterministic numerical fit to imported measurements. Model fit does not "
                "by itself establish direct binding, cellular engagement, or mechanism."
            ),
        }
        fit_artifact = self.artifacts.put_json(
            fit_payload,
            producer="protbind.experiment-fit",
            producer_version=__version__,
            source=(
                "sha256:"
                + str(json.loads(experiment["canonical_artifact_json"])["sha256"])
            ),
        )
        with self._connect() as connection:
            self._initialize(connection)
            connection.execute(
                """
                INSERT INTO fits (
                    experiment_id, experiment_revision, model, fit_artifact_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    experiment["revision"],
                    model,
                    _artifact_json(fit_artifact),
                    datetime.now(UTC).isoformat(),
                ),
            )
            connection.commit()
        return {
            "status": "FITTED",
            "experiment_id": experiment_id,
            "model": model,
            "point_count": len(selected),
            "fit_artifact": fit_artifact.to_dict(),
            "parameters": fit["parameters"],
            "r_squared": fit["r_squared"],
        }

    def _connect(self) -> sqlite3.Connection:
        if not self.database.parent.exists():
            self.database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                assay_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                raw_artifact_json TEXT NOT NULL,
                canonical_artifact_json TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('ACTIVE', 'SUPERSEDED', 'TOMBSTONED')),
                PRIMARY KEY (experiment_id, revision)
            );
            CREATE TABLE IF NOT EXISTS measurements (
                experiment_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                lab_id TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                replicate INTEGER NOT NULL,
                concentration REAL NOT NULL,
                concentration_unit TEXT NOT NULL,
                response REAL NOT NULL,
                response_unit TEXT NOT NULL,
                control_type TEXT NOT NULL,
                PRIMARY KEY (experiment_id, revision, row_index),
                FOREIGN KEY (experiment_id, revision)
                    REFERENCES experiments(experiment_id, revision)
            );
            CREATE TABLE IF NOT EXISTS fits (
                fit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                experiment_revision INTEGER NOT NULL,
                model TEXT NOT NULL,
                fit_artifact_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id, experiment_revision)
                    REFERENCES experiments(experiment_id, revision)
            );
            """
        )


def _read_assay_rows(source: Path) -> list[dict[str, Any]]:
    if not source.is_file():
        raise FileNotFoundError("experimental assay source does not exist")
    suffix = source.suffix.lower()
    if suffix not in {".csv", ".tsv", ".tab"}:
        raise ValueError("experimental assay input must be CSV or TSV")
    delimiter = "," if suffix == ".csv" else "\t"
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or not set(_REQUIRED_COLUMNS).issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "experimental assay input is missing one or more schema-1.0 columns"
            )
        raw_rows = list(reader)
    if not raw_rows or len(raw_rows) > 1_000_000:
        raise ValueError("experimental assay input must contain 1-1000000 rows")
    rows = []
    for row_index, raw in enumerate(raw_rows):
        text = {
            name: str(raw.get(name, "")).strip() for name in _REQUIRED_COLUMNS
        }
        if not all(text.values()):
            raise ValueError(f"experimental assay row {row_index + 1} has empty values")
        if text["assay_type"] not in ASSAY_TYPES:
            raise ValueError(f"unsupported assay type: {text['assay_type']}")
        try:
            replicate = int(text["replicate"])
            concentration = float(text["concentration"])
            response = float(text["response"])
        except ValueError as exc:
            raise ValueError("replicate, concentration, and response must be numeric") from exc
        if replicate < 1:
            raise ValueError("experimental replicate numbers must be positive")
        if not math.isfinite(concentration) or concentration < 0:
            raise ValueError("experimental concentrations must be finite and non-negative")
        if not math.isfinite(response):
            raise ValueError("experimental responses must be finite")
        rows.append(
            {
                **text,
                "replicate": replicate,
                "concentration": concentration,
                "response": response,
            }
        )
    for field in (
        "experiment_id",
        "assay_type",
        "target_id",
        "candidate_id",
        "concentration_unit",
        "response_unit",
    ):
        if len({str(row[field]) for row in rows}) != 1:
            raise ValueError(f"one import must use exactly one {field}")
    return rows


def _build_preview(source: Path, rows: Sequence[dict[str, Any]]) -> AssayImportPreview:
    source_sha256 = sha256_file(source)
    controls: dict[str, int] = {}
    for row in rows:
        control = str(row["control_type"]).lower()
        controls[control] = controls.get(control, 0) + 1
    identity = {
        "schema_version": ASSAY_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "canonical_rows_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
        "experiment_id": rows[0]["experiment_id"],
        "assay_type": rows[0]["assay_type"],
    }
    return AssayImportPreview(
        plan_id=hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
        source_sha256=source_sha256,
        experiment_id=str(rows[0]["experiment_id"]),
        assay_type=str(rows[0]["assay_type"]),
        target_id=str(rows[0]["target_id"]),
        candidate_id=str(rows[0]["candidate_id"]),
        row_count=len(rows),
        batch_count=len({str(row["batch_id"]) for row in rows}),
        lab_count=len({str(row["lab_id"]) for row in rows}),
        control_counts=controls,
        concentration_unit=str(rows[0]["concentration_unit"]),
        response_unit=str(rows[0]["response_unit"]),
    )


def _artifact_json(artifact: ArtifactRef) -> str:
    return canonical_json_bytes(artifact.to_dict()).decode()


def _fit_four_parameter_logistic(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> dict[str, Any]:
    if any(value <= 0 for value in x_values):
        raise ValueError("four-parameter logistic concentrations must be positive")
    if len(set(x_values)) < 4:
        raise ValueError("four-parameter logistic fitting requires four concentrations")
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError as exc:
        raise RuntimeError("four-parameter logistic fitting requires NumPy and SciPy") from exc

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    log_x = np.log10(x)

    def model(log_concentration, bottom, top, log_ec50, hill):
        return bottom + (top - bottom) / (
            1.0 + np.power(10.0, (log_ec50 - log_concentration) * hill)
        )

    initial = [float(y.min()), float(y.max()), float(np.median(log_x)), 1.0]
    parameters, covariance = curve_fit(
        model,
        log_x,
        y,
        p0=initial,
        bounds=(
            [-math.inf, -math.inf, float(log_x.min() - 6), -20.0],
            [math.inf, math.inf, float(log_x.max() + 6), 20.0],
        ),
        maxfev=100_000,
    )
    predicted = model(log_x, *parameters)
    return _fit_summary(
        y,
        predicted,
        covariance,
        {
            "bottom": float(parameters[0]),
            "top": float(parameters[1]),
            "log10_ec50": float(parameters[2]),
            "ec50": float(10 ** parameters[2]),
            "hill_slope": float(parameters[3]),
        },
    )


def _fit_one_site_binding(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> dict[str, Any]:
    if len(set(x_values)) < 3:
        raise ValueError("one-site binding fitting requires three concentrations")
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError as exc:
        raise RuntimeError("one-site binding fitting requires NumPy and SciPy") from exc

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    def model(concentration, baseline, bmax, kd):
        return baseline + bmax * concentration / (kd + concentration)

    positive = x[x > 0]
    initial_kd = float(np.median(positive)) if positive.size else 1.0
    parameters, covariance = curve_fit(
        model,
        x,
        y,
        p0=[float(y.min()), float(y.max() - y.min()), initial_kd],
        bounds=([-math.inf, -math.inf, 1e-15], [math.inf, math.inf, math.inf]),
        maxfev=100_000,
    )
    predicted = model(x, *parameters)
    return _fit_summary(
        y,
        predicted,
        covariance,
        {
            "baseline": float(parameters[0]),
            "bmax": float(parameters[1]),
            "kd_in_input_units": float(parameters[2]),
        },
    )


def _fit_summary(y, predicted, covariance, parameters: dict[str, float]) -> dict[str, Any]:
    import numpy as np

    residual = y - predicted
    ss_residual = float(np.sum(residual**2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else None
    diagonal = np.diag(covariance)
    standard_errors = [
        float(math.sqrt(value)) if math.isfinite(value) and value >= 0 else None
        for value in diagonal
    ]
    return {
        "parameters": parameters,
        "parameter_standard_errors": standard_errors,
        "r_squared": r_squared,
        "sum_squared_residuals": ss_residual,
        "converged": True,
        "confidence_intervals": "NOT_EVALUATED",
        "model_selection": "operator-declared; no best-looking-model search",
    }
