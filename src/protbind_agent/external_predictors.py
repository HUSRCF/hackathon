"""Contracts for optional P2Rank and DrutAI integrations.

P2Rank produces prospective site hypotheses. DrutAI remains annotation-only
and disabled until its environment, model provenance, license, and bake-off
gates are all satisfied.
"""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import canonical_json_bytes, sha256_file
from .privacy import redact_text

P2RANK_BUNDLE_SCHEMA_VERSION = "1.0"
P2RANK_RECOMMENDED_VERSION = "2.5"


def p2rank_command(
    receptor: Path,
    output_dir: Path,
    *,
    executable: str = "prank",
    profile: str = "default",
) -> tuple[str, ...]:
    """Build the fixed local P2Rank invocation without shell interpolation."""

    if profile not in {"default", "alphafold"}:
        raise ValueError("P2Rank profile must be default or alphafold")
    resolved = shutil.which(executable)
    if resolved is None:
        candidate = Path(executable)
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError("P2Rank 'prank' executable is unavailable")
        resolved = str(candidate.resolve())
    if not receptor.is_file() or receptor.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise ValueError("P2Rank receptor must be an existing PDB or mmCIF file")
    command = [resolved, "predict", "-f", str(receptor.resolve()), "-o", str(output_dir.resolve())]
    if profile == "alphafold":
        command.extend(("-c", "alphafold"))
    return tuple(command)


def run_p2rank(
    receptor: Path,
    output_dir: Path,
    *,
    executable: str = "prank",
    profile: str = "default",
    timeout_seconds: float = 1800.0,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run local P2Rank and parse its predictions into a bounded hypothesis bundle."""

    if timeout_seconds <= 0 or timeout_seconds > 86_400:
        raise ValueError("P2Rank timeout must be in (0, 86400] seconds")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = p2rank_command(
        receptor,
        output_dir,
        executable=executable,
        profile=profile,
    )
    version = _p2rank_version(command[0])
    if not _is_pinned_p2rank_version(version):
        raise RuntimeError(
            "P2Rank version gate failed: production adapter requires pinned 2.5"
        )
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C", "LC_ALL": "C"},
    )
    if completed.returncode != 0:
        diagnostic = redact_text(completed.stderr or completed.stdout)
        diagnostic = " ".join(diagnostic.split())[-1000:]
        raise RuntimeError(
            f"P2Rank exited with status {completed.returncode}: {diagnostic}"
        )
    predictions = sorted(output_dir.glob("*_predictions.csv"))
    if len(predictions) != 1:
        raise RuntimeError("P2Rank did not produce exactly one predictions CSV")
    return parse_p2rank_predictions(
        predictions[0],
        receptor_sha256=sha256_file(receptor),
        p2rank_version=version,
        profile=profile,
        top_k=top_k,
    )


def parse_p2rank_predictions(
    path: Path,
    *,
    receptor_sha256: str,
    p2rank_version: str,
    profile: str = "default",
    top_k: int = 3,
) -> dict[str, Any]:
    """Parse official P2Rank CSV fields without treating them as observed sites."""

    if top_k < 1 or top_k > 100:
        raise ValueError("P2Rank top_k must be between 1 and 100")
    if (
        not isinstance(p2rank_version, str)
        or not p2rank_version.strip()
        or len(p2rank_version) > 200
        or not _is_pinned_p2rank_version(p2rank_version)
    ):
        raise ValueError("P2Rank predictions must declare the pinned 2.5 version")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError("P2Rank predictions CSV has no header")
        normalized_rows = [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in reader
        ]
    pockets: list[dict[str, Any]] = []
    for row in normalized_rows:
        try:
            rank = int(row["rank"])
            score = float(row["score"])
            probability = float(row["probability"])
            center = [
                float(row["center_x"]),
                float(row["center_y"]),
                float(row["center_z"]),
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("P2Rank CSV is missing required numeric fields") from exc
        if (
            rank < 1
            or not all(math.isfinite(value) for value in (score, probability, *center))
            or probability < 0
            or probability > 1
        ):
            raise ValueError("P2Rank CSV contains invalid rank, score, probability, or center")
        pockets.append(
            {
                "rank": rank,
                "score": score,
                "probability": probability,
                "center": center,
                "residue_ids": _split_ids(row.get("residue_ids", "")),
                "surface_atom_ids": _split_ids(
                    row.get("surf_atom_ids", row.get("surface_atom_ids", ""))
                ),
            }
        )
    pockets.sort(key=lambda pocket: (pocket["rank"], -pocket["score"]))
    selected = pockets[:top_k]
    if not selected:
        raise ValueError("P2Rank predictions CSV contains no pockets")
    return {
        "schema_version": P2RANK_BUNDLE_SCHEMA_VERSION,
        "kind": "p2rank-site-hypotheses",
        "adapter_version": __version__,
        "p2rank_version": p2rank_version,
        "receptor_sha256": receptor_sha256,
        "predictions_sha256": sha256_file(path),
        "profile": profile,
        "requested_top_k": top_k,
        "pockets": selected,
        "biological_site_validity_inferred": False,
        "docking_box_validated": False,
        "semantics": (
            "Ranked P2Rank site hypotheses only. Each center requires receptor-frame, "
            "box-overlap, and downstream docking validation; probability is model output."
        ),
    }


def write_p2rank_bundle(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"P2Rank bundle already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def drutai_admission_report() -> dict[str, Any]:
    """Expose the current fail-closed DrutAI integration decision."""

    return {
        "status": "BLOCKED_PENDING_BAKEOFF",
        "execution_mode": "separate-python-3.11-worker",
        "scientific_role_if_admitted": "annotation-only; never hard-filter or binding evidence",
        "known_upstream_constraints": {
            "python": ">=3.11,<3.12",
            "tensorflow": "2.14",
            "numpy": "1.24.3",
            "rdkit": "<2025",
        },
        "blocking_gates": [
            "resolve repository GPL-3.0 versus source-header MIT license conflict",
            "pin model weights and publish/record SHA-256",
            "record training set, split leakage controls, domain, and calibration",
            "run public protein-SMILES positive/negative and decoy bake-off",
            "show no regression when used only as an annotation",
        ],
        "model_or_score_may_be_used": False,
    }


def _split_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split() if item.strip()]


def _is_pinned_p2rank_version(value: str) -> bool:
    return re.search(r"(?:^|\D)2\.5(?:\D|$)", value) is not None


def _p2rank_version(executable: str) -> str:
    completed = subprocess.run(
        (executable, "-version"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C", "LC_ALL": "C"},
    )
    value = " ".join((completed.stdout or completed.stderr).split())
    if completed.returncode != 0 or not value:
        raise RuntimeError("P2Rank version probe failed")
    return value[:200]
