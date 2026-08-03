"""Optional, local-only MMseqs2 homology search and clustering receipts.

MMseqs2 is deliberately kept outside the ProtBind scientific main path.  It can
be used to search a private protein library or to generate a hash-bound sequence
cluster assignment for a future split/leakage audit.  It never produces docking
scores, binding-affinity claims, or automatic workflow state changes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from .privacy import redact_text

MMSEQS_SCHEMA_VERSION = "1.0"
MMSEQS_CLUSTER_KIND = "PROTBIND_MMSEQS_CLUSTER_RECEIPT"
MMSEQS_SEARCH_KIND = "PROTBIND_MMSEQS_SEARCH_RECEIPT"


@dataclass(frozen=True, slots=True)
class MMseqsConfig:
    """Pinned parameters shared by local search and clustering."""

    min_seq_id: float = 0.3
    coverage: float = 0.8
    cov_mode: int = 0
    sensitivity: float = 7.5
    threads: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < float(self.min_seq_id) <= 1.0:
            raise ValueError("MMseqs min_seq_id must be in (0, 1]")
        if not 0.0 < float(self.coverage) <= 1.0:
            raise ValueError("MMseqs coverage must be in (0, 1]")
        if not isinstance(self.cov_mode, int) or not 0 <= self.cov_mode <= 5:
            raise ValueError("MMseqs cov_mode must be an integer in [0, 5]")
        if not 0.0 < float(self.sensitivity) <= 20.0:
            raise ValueError("MMseqs sensitivity must be in (0, 20]")
        if not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError("MMseqs threads must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_seq_id": float(self.min_seq_id),
            "coverage": float(self.coverage),
            "cov_mode": self.cov_mode,
            "sensitivity": float(self.sensitivity),
            "threads": self.threads,
        }


def _resolve_executable(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        candidate = Path(executable)
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError("MMseqs2 executable is unavailable")
        resolved = str(candidate.resolve())
    return str(Path(resolved).resolve())


def _require_fasta(path: Path, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"MMseqs {name} FASTA does not exist")
    if resolved.stat().st_size == 0:
        raise ValueError(f"MMseqs {name} FASTA is empty")
    if resolved.suffix.lower() not in {".fa", ".faa", ".fasta", ".fas"}:
        raise ValueError(f"MMseqs {name} input must use a FASTA suffix")
    return resolved


def _version(executable: str) -> str:
    completed = subprocess.run(
        (executable, "version"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C", "LC_ALL": "C"},
    )
    value = " ".join((completed.stdout or completed.stderr).split())
    if completed.returncode != 0 or not value:
        raise RuntimeError("MMseqs2 version probe failed")
    return value[:200]


def _config_args(config: MMseqsConfig) -> tuple[str, ...]:
    return (
        "--min-seq-id",
        str(config.min_seq_id),
        "-c",
        str(config.coverage),
        "--cov-mode",
        str(config.cov_mode),
        "-s",
        str(config.sensitivity),
        "--threads",
        str(config.threads),
    )


def mmseqs_cluster_command(
    input_fasta: Path,
    output_prefix: Path,
    temporary_dir: Path,
    *,
    executable: str = "mmseqs",
    config: MMseqsConfig | None = None,
) -> tuple[str, ...]:
    config = config or MMseqsConfig()
    return (
        _resolve_executable(executable),
        "easy-cluster",
        str(_require_fasta(input_fasta, "cluster input")),
        str(output_prefix.resolve()),
        str(temporary_dir.resolve()),
        *_config_args(config),
        "--cluster-mode",
        "0",
    )


def mmseqs_search_command(
    query_fasta: Path,
    target_fasta: Path,
    output_path: Path,
    temporary_dir: Path,
    *,
    executable: str = "mmseqs",
    config: MMseqsConfig | None = None,
) -> tuple[str, ...]:
    config = config or MMseqsConfig()
    return (
        _resolve_executable(executable),
        "easy-search",
        str(_require_fasta(query_fasta, "query")),
        str(_require_fasta(target_fasta, "target")),
        str(output_path.resolve()),
        str(temporary_dir.resolve()),
        *_config_args(config),
        "--format-output",
        "query,target,fident,alnlen,evalue,bits",
    )


def _run(command: tuple[str, ...], *, timeout_seconds: float) -> None:
    if timeout_seconds <= 0 or timeout_seconds > 86_400:
        raise ValueError("MMseqs timeout must be in (0, 86400]")
    environment = {
        key: os.environ[key]
        for key in ("PATH", "LD_LIBRARY_PATH", "MMSEQS_IGNORE_INDEX")
        if os.environ.get(key)
    }
    environment.update({"LANG": "C", "LC_ALL": "C"})
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("MMseqs2 execution timed out") from exc
    if completed.returncode != 0:
        diagnostic = " ".join(redact_text(completed.stderr or completed.stdout).split())
        raise RuntimeError(
            "MMseqs2 execution failed: " + (diagnostic[-500:] if diagnostic else "non-zero exit")
        )


def _line_stats(path: Path) -> tuple[int, int]:
    line_count = 0
    first_column_values: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            line_count += 1
            first_column_values.add(line.split("\t", 1)[0])
    return line_count, len(first_column_values)


def _receipt(
    *,
    kind: str,
    operation: str,
    executable: str,
    version: str,
    config: MMseqsConfig,
    inputs: dict[str, dict[str, Any]],
    output: Path,
    output_kind: str,
    row_count: int,
    unique_first_column_count: int,
) -> dict[str, Any]:
    core = {
        "schema_version": MMSEQS_SCHEMA_VERSION,
        "kind": kind,
        "operation": operation,
        "adapter_version": __version__,
        "implementation": {
            "executable": Path(executable).name,
            "version": version,
            "module": "protbind_agent.mmseqs",
            "module_sha256": sha256_file(Path(__file__)),
        },
        "parameters": config.to_dict(),
        "inputs": inputs,
        "output": {
            "kind": output_kind,
            "filename": output.name,
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            "row_count": row_count,
            "unique_first_column_count": unique_first_column_count,
        },
        "scientific_scope": (
            "protein sequence homology/cluster support only; no docking score, "
            "binding-affinity claim, or automatic ProtBind state mutation"
        ),
        "privacy": {
            "raw_sequences_in_receipt": False,
            "raw_sequence_ids_in_receipt": False,
            "absolute_input_paths_in_receipt": False,
            "examples": "file names and SHA-256 commitments only",
        },
    }
    return {**core, "receipt_sha256": sha256_bytes(canonical_json_bytes(core))}


def run_mmseqs_cluster(
    input_fasta: Path,
    assignments_output: Path,
    *,
    executable: str = "mmseqs",
    config: MMseqsConfig | None = None,
    timeout_seconds: float = 3600.0,
    replace: bool = False,
) -> dict[str, Any]:
    config = config or MMseqsConfig()
    if assignments_output.exists() and not replace:
        raise FileExistsError("MMseqs cluster assignment output already exists")
    input_path = _require_fasta(input_fasta, "cluster input")
    resolved = _resolve_executable(executable)
    version = _version(resolved)
    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".protbind-mmseqs-cluster-", dir=assignments_output.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        prefix = temporary_dir / "cluster"
        command = mmseqs_cluster_command(
            input_path,
            prefix,
            temporary_dir / "tmp",
            executable=resolved,
            config=config,
        )
        (temporary_dir / "tmp").mkdir()
        _run(command, timeout_seconds=timeout_seconds)
        produced = prefix.with_name(prefix.name + "_cluster.tsv")
        if not produced.is_file():
            raise RuntimeError("MMseqs2 easy-cluster produced no cluster assignment TSV")
        temporary_output = assignments_output.with_name(f".{assignments_output.name}.tmp")
        shutil.copyfile(produced, temporary_output)
        os.replace(temporary_output, assignments_output)
    row_count, unique_count = _line_stats(assignments_output)
    return _receipt(
        kind=MMSEQS_CLUSTER_KIND,
        operation="easy-cluster",
        executable=resolved,
        version=version,
        config=config,
        inputs={
            "protein_fasta": {
                "filename": input_path.name,
                "sha256": sha256_file(input_path),
                "size_bytes": input_path.stat().st_size,
            }
        },
        output=assignments_output,
        output_kind="mmseqs-cluster-assignment-tsv",
        row_count=row_count,
        unique_first_column_count=unique_count,
    )


def run_mmseqs_search(
    query_fasta: Path,
    target_fasta: Path,
    output_path: Path,
    *,
    executable: str = "mmseqs",
    config: MMseqsConfig | None = None,
    timeout_seconds: float = 3600.0,
    replace: bool = False,
) -> dict[str, Any]:
    config = config or MMseqsConfig()
    if output_path.exists() and not replace:
        raise FileExistsError("MMseqs search output already exists")
    query_path = _require_fasta(query_fasta, "query")
    target_path = _require_fasta(target_fasta, "target")
    resolved = _resolve_executable(executable)
    version = _version(resolved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".protbind-mmseqs-search-", dir=output_path.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        command = mmseqs_search_command(
            query_path,
            target_path,
            output_path,
            temporary_dir / "tmp",
            executable=resolved,
            config=config,
        )
        (temporary_dir / "tmp").mkdir()
        _run(command, timeout_seconds=timeout_seconds)
    if not output_path.is_file():
        raise RuntimeError("MMseqs2 easy-search produced no result file")
    row_count, unique_count = _line_stats(output_path)
    return _receipt(
        kind=MMSEQS_SEARCH_KIND,
        operation="easy-search",
        executable=resolved,
        version=version,
        config=config,
        inputs={
            "query_fasta": {
                "filename": query_path.name,
                "sha256": sha256_file(query_path),
                "size_bytes": query_path.stat().st_size,
            },
            "target_fasta": {
                "filename": target_path.name,
                "sha256": sha256_file(target_path),
                "size_bytes": target_path.stat().st_size,
            },
        },
        output=output_path,
        output_kind="mmseqs-search-tsv",
        row_count=row_count,
        unique_first_column_count=unique_count,
    )


def persist_mmseqs_receipt(receipt: dict[str, Any], output: Path, *, replace: bool = False) -> None:
    if output.exists() and not replace:
        raise FileExistsError("MMseqs receipt already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n"
    )
    os.replace(temporary, output)
