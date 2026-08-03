# SPDX-License-Identifier: MIT
"""Optional, license-isolated DrutAI acquisition and annotation adapter.

ProtBind never ships DrutAI weights or imports upstream implementation code.  This
module downloads one hash-pinned ONNX file only after an explicit GPL acknowledgement,
then invokes a separately installed ``drutai.predict`` executable through a plain TSV
contract.  Results remain annotation-only and can never hard-filter candidates.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import ArtifactStore, canonical_json_bytes, sha256_file
from .privacy import redact_text, require_network_approval
from .public_data import CurlTransport

DRUTAI_SCHEMA_VERSION = "1.0"
DRUTAI_SOURCE_COMMIT = "5ee6ba7037466609edc06329782dee9298f20f2b"
DRUTAI_SOURCE_REPOSITORY = "https://github.com/HUSRCF/drutai_snap"
DRUTAI_UPSTREAM_REPOSITORY = "https://github.com/2003100127/drutai"
DRUTAI_DOWNLOAD_HOST = "raw.githubusercontent.com"
DRUTAI_LICENSE = "GPL-3.0-only"
DRUTAI_LICENSE_ACKNOWLEDGEMENT = "GPL-3.0-only"
DRUTAI_PUBLICATION_PARITY = "VERIFIED_BY_PROJECT_MAINTAINER"
_TARGET_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_CANONICAL_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class DrutAIModelSpec:
    name: str
    filename: str
    size_bytes: int
    git_blob_sha1: str

    def __post_init__(self) -> None:
        if _SHA1.fullmatch(self.git_blob_sha1) is None:
            raise ValueError("DrutAI Git blob commitment must be a lowercase SHA-1")

    @property
    def source_url(self) -> str:
        return (
            "https://raw.githubusercontent.com/HUSRCF/drutai_snap/"
            f"{DRUTAI_SOURCE_COMMIT}/{self.filename}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "git_blob_sha1": self.git_blob_sha1,
            "source_url": self.source_url,
        }


DRUTAI_MODELS: Mapping[str, DrutAIModelSpec] = {
    spec.name: spec
    for spec in (
        DrutAIModelSpec("cnn", "cnn.onnx", 2_850_474, "59eb35f473c3d635a61178bc2b35fb6b3edbbdd0"),
        DrutAIModelSpec(
            "convmixer64",
            "convmixer64.onnx",
            70_187,
            "33ec49d3d973a9f5e042cd4f6254d542a803002d",
        ),
        DrutAIModelSpec(
            "dsconv",
            "dsconv.onnx",
            296_943,
            "be6ec6a44f6359f3611771eedd7929782191e910",
        ),
        DrutAIModelSpec(
            "lstmcnn",
            "lstmcnn.onnx",
            3_605_174,
            "0442f3df6505cb47e0ddb32fa2f04a0499d44b79",
        ),
        DrutAIModelSpec(
            "mobilenetv2",
            "mobilenetv2.onnx",
            2_495_349,
            "9413983172d33bab2e51696727e32a10ee55fa78",
        ),
        DrutAIModelSpec(
            "resnet_prea18_tf2",
            "resnet_prea18_tf2.onnx",
            841_690,
            "95c5b0b5e8b9d507de36d2c6fd91b09f3c8d8e47",
        ),
        DrutAIModelSpec(
            "scaresnet",
            "scaresnet.onnx",
            925_754,
            "a08ebc68b6264df87c7b4958e6acf3134c049f89",
        ),
    )
}


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _model_spec(model: str) -> DrutAIModelSpec:
    try:
        return DRUTAI_MODELS[model]
    except KeyError as exc:
        raise ValueError(f"unsupported DrutAI model: {model}") from exc


def _snap_instance_name(executable: str) -> str | None:
    """Return the snap instance for a canonical /snap/bin application alias."""

    path = Path(executable)
    if path.parent != Path("/snap/bin"):
        return None
    instance = path.name.split(".", maxsplit=1)[0]
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", instance) is None:
        return None
    return instance


def _snap_value(output: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return None


def _audit_snap_network_isolation(
    *,
    instance: str,
    runner: Runner,
) -> dict[str, Any]:
    snap = shutil.which("snap")
    if snap is None:
        raise RuntimeError("snap metadata command is required for Snap isolation audit")
    common = {
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "text": True,
        "timeout": 30.0,
        "check": False,
    }
    info = runner([snap, "info", "--verbose", instance], **common)
    if info.returncode != 0:
        raise RuntimeError("unable to audit DrutAI Snap confinement")
    confinement = _snap_value(info.stdout, "confinement")
    devmode = _snap_value(info.stdout, "devmode")
    trymode = _snap_value(info.stdout, "trymode")
    enabled = _snap_value(info.stdout, "enabled")
    broken = _snap_value(info.stdout, "broken")
    if (
        confinement != "strict"
        or devmode != "false"
        or trymode != "false"
        or enabled != "true"
        or broken != "false"
    ):
        raise RuntimeError(
            "DrutAI Snap must be enabled, unbroken, strict-confined, and outside "
            "devmode/trymode"
        )

    connections = runner([snap, "connections", instance], **common)
    if connections.returncode != 0:
        raise RuntimeError("unable to audit DrutAI Snap interface connections")
    connected_network_interfaces: list[str] = []
    for line in connections.stdout.splitlines():
        fields = line.split()
        if (
            len(fields) >= 3
            and fields[0].startswith("network")
            and fields[2] != "-"
        ):
            connected_network_interfaces.append(fields[0])
    if connected_network_interfaces:
        raise RuntimeError(
            "DrutAI Snap has a connected network interface; private annotation "
            "fails closed"
        )
    return {
        "mode": "snap-strict-confinement",
        "verified": True,
        "snap_instance": instance,
        "confinement": confinement,
        "devmode": False,
        "trymode": False,
        "connected_network_interfaces": [],
        "snap_info_sha256": hashlib.sha256(info.stdout.encode()).hexdigest(),
        "snap_connections_sha256": hashlib.sha256(
            connections.stdout.encode()
        ).hexdigest(),
    }


def _atomic_write(path: Path, data: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to replace existing DrutAI artifact: {path.name}")
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_model_bytes(data: bytes, spec: DrutAIModelSpec) -> dict[str, Any]:
    if len(data) != spec.size_bytes:
        raise ValueError("DrutAI model size differs from the pinned Git object")
    blob_sha1 = _git_blob_sha1(data)
    if blob_sha1 != spec.git_blob_sha1:
        raise ValueError("DrutAI model Git blob commitment mismatch")
    if not data.startswith(b"\x08"):
        raise ValueError("DrutAI model does not resemble an ONNX protobuf")
    return {
        "size_bytes": len(data),
        "git_blob_sha1": blob_sha1,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class DrutAIManager:
    """Manage optional third-party weights and a separate executable adapter."""

    def __init__(
        self,
        workspace: Path,
        *,
        transport: CurlTransport | None = None,
        runner: Runner = subprocess.run,
        control_runner: Runner = subprocess.run,
        executable: str = "drutai.predict",
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.artifacts = ArtifactStore(self.workspace)
        self.transport = transport
        self.runner = runner
        self.control_runner = control_runner
        self.executable = executable
        self.now = now or (lambda: datetime.now(UTC))

    def _model_directory(self) -> Path:
        return self.workspace / "third-party" / "drutai" / DRUTAI_SOURCE_COMMIT

    def _model_path(self, spec: DrutAIModelSpec) -> Path:
        return self._model_directory() / spec.filename

    def _sidecar_path(self, spec: DrutAIModelSpec) -> Path:
        return self._model_directory() / f"{spec.filename}.acquisition.json"

    def status(self) -> dict[str, Any]:
        models: list[dict[str, Any]] = []
        for spec in DRUTAI_MODELS.values():
            path = self._model_path(spec)
            status: dict[str, Any] = {
                **spec.to_dict(),
                "acquired": False,
                "valid": False,
                "observed_sha256": None,
            }
            if path.is_file():
                try:
                    validation = _validate_model_bytes(path.read_bytes(), spec)
                except ValueError as exc:
                    status["validation_error"] = str(exc)
                else:
                    status.update(
                        acquired=True,
                        valid=True,
                        observed_sha256=validation["sha256"],
                    )
            models.append(status)
        return {
            "schema_version": DRUTAI_SCHEMA_VERSION,
            "enabled_by_default": False,
            "execution_mode": "separate-executable-tsv-contract",
            "source_repository": DRUTAI_SOURCE_REPOSITORY,
            "source_commit": DRUTAI_SOURCE_COMMIT,
            "upstream_repository": DRUTAI_UPSTREAM_REPOSITORY,
            "license_policy": DRUTAI_LICENSE,
            "weights_distributed_by_protbind": False,
            "upstream_code_copied_into_protbind": False,
            "publication_parity": DRUTAI_PUBLICATION_PARITY,
            "scientific_role": "optional annotation-only sequence-SMILES DTI concordance",
            "hard_filter_allowed": False,
            "models": models,
        }

    def acquire_model(
        self,
        *,
        model: str,
        approved_domain: str,
        license_acknowledgement: str,
        replace: bool = False,
    ) -> dict[str, Any]:
        spec = _model_spec(model)
        if license_acknowledgement != DRUTAI_LICENSE_ACKNOWLEDGEMENT:
            raise PermissionError(
                "DrutAI acquisition requires exact GPL-3.0-only acknowledgement"
            )
        require_network_approval(spec.source_url, (approved_domain,))
        path = self._model_path(spec)
        sidecar = self._sidecar_path(spec)
        if path.is_file() and not replace:
            validation = _validate_model_bytes(path.read_bytes(), spec)
            return {
                "status": "PRESENT",
                "model": spec.name,
                "observed_sha256": validation["sha256"],
                "network_request_performed": False,
                "license_acknowledged": True,
                "scientific_role": "annotation-only",
                "hard_filter_allowed": False,
            }
        if sidecar.exists() and not replace:
            raise FileExistsError(
                "DrutAI acquisition sidecar exists without a reusable model; "
                "explicit replacement is required"
            )
        transport = self.transport or CurlTransport()
        result = transport.request(
            spec.source_url,
            accept="application/octet-stream",
            max_bytes=spec.size_bytes,
        )
        validation = _validate_model_bytes(result.data, spec)
        model_artifact = self.artifacts.put_bytes(
            result.data,
            media_type="application/onnx",
            producer="protbind.drutai-model-acquisition",
            producer_version=__version__,
            source=spec.source_url,
            license=DRUTAI_LICENSE,
        )
        receipt = {
            "schema_version": DRUTAI_SCHEMA_VERSION,
            "kind": "third-party-model-acquisition",
            "status": "ACQUIRED",
            "model": spec.name,
            "source_repository": DRUTAI_SOURCE_REPOSITORY,
            "source_commit": DRUTAI_SOURCE_COMMIT,
            "source_url": spec.source_url,
            "upstream_repository": DRUTAI_UPSTREAM_REPOSITORY,
            "expected_git_blob_sha1": spec.git_blob_sha1,
            "observed_git_blob_sha1": validation["git_blob_sha1"],
            "observed_sha256": validation["sha256"],
            "size_bytes": validation["size_bytes"],
            "license": DRUTAI_LICENSE,
            "license_acknowledged": True,
            "license_policy": (
                "Conservatively treat converted upstream weights as GPL-3.0-only; "
                "ProtBind stores no copy in its source distribution."
            ),
            "distributed_by_protbind": False,
            "upstream_code_copied_into_protbind": False,
            "publication_parity": DRUTAI_PUBLICATION_PARITY,
            "model_artifact": model_artifact.to_dict(),
            "retrieval": result.receipt_dict(),
            "acquired_at": self.now().isoformat(),
            "scientific_role": "annotation-only",
            "hard_filter_allowed": False,
        }
        receipt_artifact = self.artifacts.put_json(
            receipt,
            producer="protbind.drutai-model-acquisition-receipt",
            producer_version=__version__,
            source=spec.source_url,
            license=DRUTAI_LICENSE,
        )
        _atomic_write(path, result.data, replace=replace)
        _atomic_write(
            sidecar,
            canonical_json_bytes(receipt) + b"\n",
            replace=replace,
        )
        return {
            **receipt,
            "receipt_artifact": receipt_artifact.to_dict(),
            "network_request_performed": True,
        }

    def annotate(
        self,
        *,
        input_tsv: Path,
        fasta_directory: Path,
        model: str,
        data_access_confirmed: bool,
        threads: int | None = None,
        batch_size: int = 2000,
        abstention_margin: float = 0.05,
        timeout_seconds: float = 3600.0,
        isolate_network: bool = True,
    ) -> dict[str, Any]:
        if data_access_confirmed is not True:
            raise PermissionError("DrutAI annotation requires fresh private-data approval")
        if threads is not None and (threads < 1 or threads > 64):
            raise ValueError("DrutAI threads must be between 1 and 64")
        if batch_size < 1 or batch_size > 100_000:
            raise ValueError("DrutAI batch size must be between 1 and 100000")
        if (
            not isinstance(abstention_margin, int | float)
            or isinstance(abstention_margin, bool)
            or not math.isfinite(abstention_margin)
            or abstention_margin < 0
            or abstention_margin >= 0.5
        ):
            raise ValueError("DrutAI abstention margin must be in [0, 0.5)")
        if timeout_seconds <= 0 or timeout_seconds > 86_400:
            raise ValueError("DrutAI timeout must be in (0, 86400]")
        spec = _model_spec(model)
        model_path = self._model_path(spec)
        if not model_path.is_file():
            raise FileNotFoundError(
                "DrutAI model is not acquired; run the separately approved acquisition first"
            )
        model_validation = _validate_model_bytes(model_path.read_bytes(), spec)
        rows, input_commitment = _validate_input(input_tsv, fasta_directory)
        executable = shutil.which(self.executable)
        if executable is None:
            candidate = Path(self.executable)
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise RuntimeError("optional drutai.predict executable is unavailable")
            executable = str(candidate.resolve())
        started = time.monotonic()
        temp_root = self.workspace / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="drutai-run-", dir=temp_root) as temporary:
            exchange = Path(temporary)
            output = exchange / "predictions.tsv"
            home = exchange / "home"
            home.mkdir()
            command = [
                executable,
                "-m",
                spec.name,
                "-i",
                str(input_tsv.resolve()),
                "-t",
                str(fasta_directory.resolve()),
                "-o",
                str(output.resolve()),
                "--models-dir",
                str(model_path.parent.resolve()),
                "--no-cache",
                "--batch-size",
                str(batch_size),
                "--silence",
            ]
            if threads is not None:
                command.extend(("--threads", str(threads)))
            isolation_receipt: dict[str, Any] = {
                "mode": "disabled",
                "verified": False,
            }
            if isolate_network:
                snap_instance = _snap_instance_name(executable)
                if snap_instance is not None:
                    isolation_receipt = _audit_snap_network_isolation(
                        instance=snap_instance,
                        runner=self.control_runner,
                    )
                else:
                    bwrap = shutil.which("bwrap")
                    if bwrap is None:
                        raise RuntimeError(
                            "bubblewrap is required for OS-isolated private DrutAI execution"
                        )
                    command = [
                        bwrap,
                        "--die-with-parent",
                        "--unshare-net",
                        "--ro-bind",
                        "/",
                        "/",
                        "--dev-bind",
                        "/dev",
                        "/dev",
                        "--proc",
                        "/proc",
                        "--bind",
                        str(exchange.resolve()),
                        str(exchange.resolve()),
                        "--chdir",
                        str(Path.cwd()),
                        "--",
                        *command,
                    ]
                    isolation_receipt = {
                        "mode": "bubblewrap-unshare-net",
                        "verified": True,
                    }
            environment = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C",
                "LC_ALL": "C",
                "HOME": str(home.resolve()),
                "TMPDIR": str(exchange.resolve()),
                "PYTHONNOUSERSITE": "1",
                "PROTBIND_NETWORK_POLICY": "deny",
            }
            try:
                completed = self.runner(
                    command,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("DrutAI worker exceeded its declared timeout") from exc
            if completed.returncode != 0:
                diagnostic = redact_text(completed.stderr or completed.stdout)
                raise RuntimeError(
                    "DrutAI worker failed: " + " ".join(diagnostic.split())[-1000:]
                )
            if not output.is_file():
                raise RuntimeError("DrutAI worker returned success without predictions")
            output_rows = _validate_output(output, rows)
            raw_output = self.artifacts.import_file(
                output,
                media_type="text/tab-separated-values",
                producer="drutai.predict",
                producer_version=DRUTAI_SOURCE_COMMIT,
                source=DRUTAI_SOURCE_REPOSITORY,
                license=DRUTAI_LICENSE,
            )
        annotations = []
        for row, prediction in zip(rows, output_rows, strict=True):
            probability = prediction["prob_inter"]
            if abs(probability - 0.5) <= abstention_margin:
                concordance = "ABSTAIN"
            elif probability > 0.5:
                concordance = "SUPPORTIVE"
            else:
                concordance = "DISCORDANT"
            annotations.append(
                {
                    "candidate_id": row["sm"],
                    "target_id": row["target"],
                    "prob_inter": probability,
                    "model_direction": prediction["pred_type"],
                    "concordance": concordance,
                    "decision_eligible": False,
                }
            )
        bundle = {
            "schema_version": DRUTAI_SCHEMA_VERSION,
            "kind": "drutai-sequence-smiles-annotation",
            "status": "COMPLETED",
            "model": spec.name,
            "source_repository": DRUTAI_SOURCE_REPOSITORY,
            "source_commit": DRUTAI_SOURCE_COMMIT,
            "model_sha256": model_validation["sha256"],
            "model_git_blob_sha1": model_validation["git_blob_sha1"],
            "license": DRUTAI_LICENSE,
            "publication_parity": DRUTAI_PUBLICATION_PARITY,
            "input": input_commitment,
            "record_count": len(rows),
            "abstention_margin": float(abstention_margin),
            "annotations": annotations,
            "raw_output_artifact": raw_output.to_dict(),
            "duration_seconds": round(time.monotonic() - started, 6),
            "execution": {
                "external_process": True,
                "plain_tsv_contract": True,
                "network_isolated": isolation_receipt["verified"],
                "network_isolation": isolation_receipt,
                "worker_cache_disabled": True,
            },
            "scientific_role": (
                "Annotation-only sequence-SMILES DTI concordance. Scores are not "
                "binding, affinity, activity, pose, calibration, or experimental evidence."
            ),
            "hard_filter_allowed": False,
            "evidence_grade_upgrade_allowed": False,
        }
        bundle_artifact = self.artifacts.put_json(
            bundle,
            producer="protbind.drutai-adapter",
            producer_version=__version__,
            source=DRUTAI_SOURCE_REPOSITORY,
            license=DRUTAI_LICENSE,
        )
        return {
            "status": "COMPLETED",
            "model": spec.name,
            "record_count": len(rows),
            "concordance_counts": {
                name: sum(item["concordance"] == name for item in annotations)
                for name in ("SUPPORTIVE", "DISCORDANT", "ABSTAIN")
            },
            "bundle_artifact": bundle_artifact.to_dict(),
            "raw_output_artifact": raw_output.to_dict(),
            "scientific_role": "annotation-only",
            "hard_filter_allowed": False,
        }


def _validate_input(
    input_tsv: Path,
    fasta_directory: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not input_tsv.is_file():
        raise FileNotFoundError("DrutAI input TSV does not exist")
    if not fasta_directory.is_dir():
        raise FileNotFoundError("DrutAI FASTA directory does not exist")
    with input_tsv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sm", "target", "smile"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("DrutAI TSV requires sm, target, and smile columns")
        rows = [
            {name: str(row.get(name, "")).strip() for name in ("sm", "target", "smile")}
            for row in reader
        ]
    if not rows or len(rows) > 250_000:
        raise ValueError("DrutAI TSV must contain between 1 and 250000 rows")
    molecule_smiles: dict[str, str] = {}
    targets: set[str] = set()
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for fail-closed DrutAI input validation") from exc
    for row in rows:
        if not all(row.values()):
            raise ValueError("DrutAI TSV contains an empty required value")
        if _TARGET_ID.fullmatch(row["target"]) is None:
            raise ValueError("DrutAI target ID is unsafe for FASTA file resolution")
        if any(character in row["sm"] for character in "\r\n\t"):
            raise ValueError("DrutAI candidate ID contains a control delimiter")
        previous = molecule_smiles.setdefault(row["sm"], row["smile"])
        if previous != row["smile"]:
            raise ValueError("one DrutAI candidate ID maps to conflicting SMILES")
        if Chem.MolFromSmiles(row["smile"]) is None:
            raise ValueError("DrutAI TSV contains an invalid SMILES; zero fallback is forbidden")
        targets.add(row["target"])
    fasta_commitments = []
    for target in sorted(targets):
        fasta = fasta_directory / f"{target}.fasta"
        sequence = _read_single_fasta(fasta)
        fasta_commitments.append(
            {
                "target_id": target,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "sequence_length": len(sequence),
                "file_sha256": sha256_file(fasta),
            }
        )
    return rows, {
        "tsv_sha256": sha256_file(input_tsv),
        "fasta_commitments": fasta_commitments,
        "raw_sequence_in_receipt": False,
        "raw_smiles_in_receipt": False,
    }


def _read_single_fasta(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required DrutAI FASTA is missing: {path.name}")
    text = path.read_text(encoding="utf-8-sig")
    records: list[str] = []
    current: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                records.append("".join(current))
            current = []
        elif current is None:
            raise ValueError("DrutAI FASTA sequence appears before its header")
        else:
            current.append(line.upper())
    if current is not None:
        records.append("".join(current))
    if len(records) != 1 or not records[0]:
        raise ValueError("DrutAI requires exactly one non-empty sequence per FASTA")
    sequence = records[0]
    if len(sequence) > 10_000 or _CANONICAL_SEQUENCE.fullmatch(sequence) is None:
        raise ValueError("DrutAI FASTA must contain 1-10000 canonical amino acids")
    return sequence


def _validate_output(
    path: Path,
    expected_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sm", "target", "smile", "prob_inter", "pred_type"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("DrutAI output is missing required prediction columns")
        rows = list(reader)
    if len(rows) != len(expected_rows):
        raise ValueError("DrutAI output row count differs from the validated input")
    parsed = []
    for expected, row in zip(expected_rows, rows, strict=True):
        for name in ("sm", "target", "smile"):
            if str(row.get(name, "")).strip() != expected[name]:
                raise ValueError("DrutAI output identity or ordering differs from input")
        try:
            probability = float(row["prob_inter"])
        except (TypeError, ValueError) as exc:
            raise ValueError("DrutAI output probability is not numeric") from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("DrutAI output probability must be finite and in [0, 1]")
        expected_type = "Interaction" if probability > 0.5 else "Non-interaction"
        if row["pred_type"] != expected_type:
            raise ValueError("DrutAI output class conflicts with its declared threshold")
        parsed.append({"prob_inter": probability, "pred_type": expected_type})
    return parsed


def validate_drutai_receipt_sha256(value: str) -> str:
    """Validate a receipt commitment supplied by an external audit surface."""

    if _SHA256.fullmatch(value) is None:
        raise ValueError("DrutAI receipt commitment must be a lowercase SHA-256")
    return value
