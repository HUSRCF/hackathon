from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import protbind_agent.mmseqs as mmseqs
from protbind_agent.mmseqs import MMseqsConfig, mmseqs_cluster_command, mmseqs_search_command


def _fasta(path: Path, name: str = "query") -> Path:
    path.write_text(f">{name}\nACDEFGHIKLMNPQRSTVWY\n", encoding="utf-8")
    return path


def test_mmseqs_config_and_commands_are_parameter_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mmseqs.shutil, "which", lambda name: "/opt/mmseqs/bin/mmseqs")
    fasta = _fasta(tmp_path / "proteins.fasta")
    config = MMseqsConfig(
        min_seq_id=0.3,
        coverage=0.8,
        cov_mode=0,
        sensitivity=7.5,
        threads=4,
    )

    cluster = mmseqs_cluster_command(
        fasta,
        tmp_path / "cluster",
        tmp_path / "tmp",
        config=config,
    )
    search = mmseqs_search_command(
        fasta,
        fasta,
        tmp_path / "hits.tsv",
        tmp_path / "tmp",
        config=config,
    )

    assert cluster[:3] == ("/opt/mmseqs/bin/mmseqs", "easy-cluster", str(fasta.resolve()))
    assert "--min-seq-id" in cluster and "0.3" in cluster
    assert "--cluster-mode" in cluster and cluster[-1] == "0"
    assert search[:2] == ("/opt/mmseqs/bin/mmseqs", "easy-search")
    assert "--format-output" in search
    assert "query,target,fident,alnlen,evalue,bits" in search


def test_mmseqs_cluster_receipt_has_no_raw_sequence_or_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = _fasta(tmp_path / "private-proteins.fasta", name="private-target")
    assignments = tmp_path / "assignments.tsv"

    monkeypatch.setattr(mmseqs.shutil, "which", lambda name: "/opt/mmseqs/bin/mmseqs")
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command)
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, stdout="MMseqs2 Version: 15", stderr="")
        prefix = Path(command[3])
        prefix.with_name(prefix.name + "_cluster.tsv").write_text(
            "cluster-A\tprivate-target\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(mmseqs.subprocess, "run", fake_run)
    receipt = mmseqs.run_mmseqs_cluster(fasta, assignments)

    encoded = json.dumps(receipt, ensure_ascii=False)
    assert "ACDEFGHIKLMNPQRSTVWY" not in encoded
    assert "private-target" not in encoded
    assert receipt["kind"] == "PROTBIND_MMSEQS_CLUSTER_RECEIPT"
    assert receipt["output"]["row_count"] == 1
    assert receipt["output"]["unique_first_column_count"] == 1
    assert calls[0][1] == "version"
    assert calls[1][1] == "easy-cluster"


def test_mmseqs_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="min_seq_id"):
        MMseqsConfig(min_seq_id=0.0)
    with pytest.raises(ValueError, match="coverage"):
        MMseqsConfig(coverage=1.1)
    with pytest.raises(ValueError, match="cov_mode"):
        MMseqsConfig(cov_mode=6)
