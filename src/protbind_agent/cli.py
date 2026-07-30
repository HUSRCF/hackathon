"""ProtBind command-line interface."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from radeon_agent.agent import AgentLimitError
from radeon_agent.backends import BackendError

from . import __version__
from .agent_benchmark import (
    AgentBenchmarkConfig,
    run_agent_benchmark,
    save_agent_benchmark,
)
from .agent_runtime import create_runtime
from .artifacts import ArtifactStore, sha256_file
from .benchmark import benchmark_cpu, benchmark_hip, save_benchmark
from .capabilities import doctor_report
from .caseio import ingest_case
from .chemistry import ChemistryCapabilityError, load_chemical_library
from .dossier import (
    build_run_dossier,
    dossier_content,
    persist_run_dossier,
)
from .external_predictors import (
    parse_p2rank_predictions,
    run_p2rank,
    write_p2rank_bundle,
)
from .knowledge import (
    KnowledgeCapabilityError,
    SeekDBKnowledgeStore,
    extract_document_bytes,
    freeze_embedding_model_manifest,
    import_document,
    inspect_embedding_model,
    sync_library_rag,
)
from .library import (
    ImportState,
    LibraryManager,
    load_library_config,
    save_library_config,
)
from .manifest import RunManifest, RunState
from .models import RCSBCoordinatePolicy, ResearchMode
from .pose_view import build_pose_scene_summary
from .posebusters_holdout import (
    DEFAULT_NAMESPACE as POSEBUSTERS_HOLDOUT_NAMESPACE,
)
from .posebusters_holdout import (
    freeze_posebusters_holdout,
    write_holdout_manifest,
)
from .privacy import require_network_approval
from .public_data import (
    PUBLIC_DATA_SOURCES,
    PublicDataFetcher,
    materialize_public_fetch,
    required_domain,
    validate_public_output,
)
from .redock_benchmark import RedockBenchmarkConfig, run_redock_benchmark
from .redock_holdout_batch import (
    RedockHoldoutBatchConfig,
    run_frozen_redock_holdout,
)
from .redock_regression import (
    RegressionDesign,
    build_redock_regression,
    persist_redock_regression,
)
from .structure import StructureCapabilityError
from .tripharm import (
    TriPharmConfig,
    build_index,
    build_jsonl_index,
    index_identity,
)
from .web import create_app
from .web_assets import THREEDMOL_HOST, install_3dmol_asset
from .worker_protocol import WorkerProvenance
from .workflow import PipelineConfig, ProtBindWorkflow, WorkerConfig

_STAGE_CHOICES = tuple(stage.value.lower() for stage in RunState if stage not in {
    RunState.CREATED,
    RunState.COFOLDED,
    RunState.DEGRADED,
    RunState.FAILED,
})


def _workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("artifacts/protbind"),
        help="Private content-addressed run workspace.",
    )


def _library_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".protbind/library.json"),
        help=(
            "Private library configuration containing the separately selected protein "
            "and ligand roots."
        ),
    )


def _data_access_confirmation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-data-access",
        action="store_true",
        required=True,
        help=(
            "Confirm this invocation may inspect or mutate the selected private data. "
            "Agent clients additionally require an interactive permission decision."
        ),
    )


def _worker_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    workers_value = value.get("workers", {})
    if not isinstance(workers_value, dict):
        raise ValueError("worker config [workers] must be a TOML table")
    aliases = {
        "select": RunState.SELECTED,
        "selected": RunState.SELECTED,
        "quick_vina": RunState.SELECTED,
        "cofold": RunState.COFOLDED,
        "cofolded": RunState.COFOLDED,
        "dock": RunState.DOCKED,
        "docked": RunState.DOCKED,
        "validate": RunState.VALIDATED,
        "validated": RunState.VALIDATED,
    }
    workers: dict[RunState, WorkerConfig] = {}
    for name, item in workers_value.items():
        if name not in aliases or not isinstance(item, dict):
            raise ValueError(f"invalid worker section: workers.{name}")
        if aliases[name] in workers:
            raise ValueError(
                f"multiple worker sections configure {aliases[name].value}: workers.{name}"
            )
        argv = item.get("argv")
        if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
            raise ValueError(f"workers.{name}.argv must be an array of strings")
        provenance_value = item.get("provenance", {})
        if not isinstance(provenance_value, dict):
            raise ValueError(f"workers.{name}.provenance must be a table")
        provenance = WorkerProvenance(
            model_revision=str(provenance_value["model_revision"]),
            weight_sha256=str(provenance_value["weight_sha256"]),
            code_sha256=str(provenance_value["code_sha256"]),
        )
        parameters = item.get("parameters", {})
        environment = item.get("environment", {})
        if not isinstance(parameters, dict) or not isinstance(environment, dict):
            raise ValueError(f"workers.{name} parameters/environment must be tables")
        isolate_network = item.get("isolate_network", True)
        if not isinstance(isolate_network, bool):
            raise ValueError(f"workers.{name}.isolate_network must be boolean")
        workers[aliases[name]] = WorkerConfig(
            engine=str(item["engine"]),
            argv=tuple(argv),
            provenance=provenance,
            parameters=parameters,
            timeout_seconds=float(item.get("timeout_seconds", 3600.0)),
            environment={str(key): str(entry) for key, entry in environment.items()},
            isolate_network=isolate_network,
        )
    screening = value.get("screening", {})
    if not isinstance(screening, dict):
        raise ValueError("[screening] must be a TOML table")
    hip_executable_value = screening.get("hip_executable")
    if hip_executable_value is not None and not isinstance(
        hip_executable_value, str
    ):
        raise ValueError("screening.hip_executable must be a path string")
    return PipelineConfig(
        screen_top_k=int(screening.get("top_k", 512)),
        rrf_k=int(screening.get("rrf_k", 60)),
        screen_backend=str(screening.get("backend", "auto")),
        hip_executable=(
            Path(hip_executable_value) if hip_executable_value else None
        ),
        parity_top_k=int(screening.get("parity_top_k", 512)),
        hip_timeout_seconds=int(screening.get("hip_timeout_seconds", 600)),
        workers=workers,
    )


def _state(value: str) -> RunState:
    return RunState(value.upper())


def _manifest_summary(
    manifest: RunManifest,
    artifacts: ArtifactStore | None = None,
) -> dict[str, Any]:
    resolution_summary: dict[str, Any] = {
        "decision": manifest.provenance.get("target_structure_resolution")
    }
    resolution_ref = manifest.input_artifacts.get("target_resolution")
    if artifacts is not None and resolution_ref is not None:
        resolution = artifacts.read_json(resolution_ref)
        if isinstance(resolution, dict):
            for name in (
                "folding_required",
                "reason",
                "pdb_id",
                "selected_chain_ids",
                "coordinate_file_policy",
                "assembly_id",
            ):
                if name in resolution:
                    resolution_summary[name] = resolution[name]
    return {
        "schema_version": manifest.schema_version,
        "run_id": manifest.run_id,
        "case_id": manifest.case_id,
        "state": manifest.state.value,
        "last_completed_stage": manifest.last_completed_stage.value,
        "optional_cofold": {
            "status": manifest.cofold_status.value,
            "failure": (
                manifest.cofold_failure.to_dict()
                if manifest.cofold_failure is not None
                else None
            ),
            "output_artifact_ids": (
                [artifact.artifact_id for artifact in manifest.cofold_record.outputs]
                if manifest.cofold_record is not None
                else []
            ),
        },
        "target_structure_resolution": resolution_summary,
        "failures": [failure.to_dict() for failure in manifest.failures],
        "input_artifacts": {
            name: artifact.artifact_id
            for name, artifact in sorted(manifest.input_artifacts.items())
        },
        "artifacts": {
            name: artifact.artifact_id for name, artifact in sorted(manifest.artifacts.items())
        },
    }


def _manifest_exit(manifest: RunManifest) -> int:
    if manifest.state is RunState.FAILED:
        return 2
    if manifest.state is RunState.DEGRADED:
        return 3
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protbind",
        description="Local private protein--ligand research workflow for AMD Radeon.",
    )
    parser.add_argument("--version", action="version", version=f"protbind {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Probe scientific runtimes and Radeon evidence.")

    agent = commands.add_parser(
        "agent",
        help="Run the standalone, confirmation-gated local ProtBind Agent.",
    )
    agent.add_argument(
        "prompt",
        nargs="?",
        help="One research request. If omitted, read it interactively from stdin.",
    )
    agent.add_argument("--backend", choices=("hipfire",), default="hipfire")
    agent.add_argument("--model", default="qwen3.5:9b")
    agent.add_argument("--base-url", default="http://127.0.0.1:11435/v1")
    agent.add_argument("--project-root", type=Path, default=Path("."))
    agent.add_argument("--library-config", type=Path)
    agent.add_argument("--knowledge-model", type=Path)
    agent.add_argument("--worker-config", type=Path)
    agent.add_argument("--max-steps", type=int, default=16)
    agent.add_argument("--json", action="store_true")
    _workspace_argument(agent)

    agent_benchmark = commands.add_parser(
        "agent-benchmark",
        help="Run the hash-bound HipFire tool-call workload on local Radeon.",
    )
    agent_benchmark.add_argument("--run-id", required=True)
    agent_benchmark.add_argument("--knowledge-query", required=True)
    agent_benchmark.add_argument("--preference", required=True)
    agent_benchmark.add_argument("--knowledge-model", type=Path, required=True)
    agent_benchmark.add_argument("--library-config", type=Path)
    agent_benchmark.add_argument("--worker-config", type=Path)
    agent_benchmark.add_argument("--project-root", type=Path, default=Path("."))
    agent_benchmark.add_argument(
        "--workload",
        type=Path,
        default=Path("benchmarks/suites/protbind_agent_toolcall.json"),
    )
    agent_benchmark.add_argument("--base-url", default="http://127.0.0.1:11435/v1")
    agent_benchmark.add_argument("--model", default="qwen3.5:9b")
    agent_benchmark.add_argument("--label", required=True)
    agent_benchmark.add_argument("--model-revision", required=True)
    agent_benchmark.add_argument("--model-sha256", required=True)
    agent_benchmark.add_argument(
        "--model-weights",
        type=Path,
        required=True,
        help=(
            "Exact local model file or directory; measured hash must equal "
            "--model-sha256."
        ),
    )
    agent_benchmark.add_argument("--quantization", required=True)
    agent_benchmark.add_argument("--hipfire-revision", required=True)
    agent_benchmark.add_argument(
        "--hipfire-source-root",
        type=Path,
        required=True,
        help="Clean local HipFire source checkout matching --hipfire-revision.",
    )
    agent_benchmark.add_argument(
        "--hipfire-daemon",
        type=Path,
        required=True,
        help="Exact running HipFire daemon binary; its process and SHA-256 are verified.",
    )
    agent_benchmark.add_argument("--hipfire-visible-device", type=int, required=True)
    agent_benchmark.add_argument(
        "--hipfire-speculation",
        choices=("off", "auto", "ngram", "dflash", "mtp"),
        required=True,
    )
    agent_benchmark.add_argument(
        "--hipfire-jinja-mode",
        choices=("default-on", "explicit-on", "explicit-off"),
        required=True,
    )
    agent_benchmark.add_argument("--code-revision", required=True)
    agent_benchmark.add_argument("--repetitions", type=int, default=3)
    agent_benchmark.add_argument("--warmup-runs", type=int, default=1)
    agent_benchmark.add_argument("--output", type=Path, required=True)
    agent_benchmark.add_argument(
        "--confirm-benchmark-data",
        action="store_true",
        required=True,
        help=(
            "Confirm this run may read the named local case/knowledge and write its "
            "idempotent experience record."
        ),
    )
    _workspace_argument(agent_benchmark)

    index = commands.add_parser("index", help="Build or inspect TriPharm indexes.")
    index_commands = index.add_subparsers(dest="index_command", required=True)
    index_build = index_commands.add_parser("build", help="Build a deterministic index.")
    index_build.add_argument("--input", type=Path, required=True)
    index_build.add_argument("--output", type=Path, required=True)
    index_build.add_argument("--bin-width", type=float, default=0.5)
    index_build.add_argument("--tolerance", type=float, default=1.0)
    index_build.add_argument("--max-conformers", type=int, default=4)
    index_build.add_argument("--seed", type=int, default=20260721)
    index_build.add_argument(
        "--force", action="store_true", help="Explicitly replace an existing index atomically."
    )
    index_inspect = index_commands.add_parser("inspect", help="Show index hashes and counts.")
    index_inspect.add_argument("index", type=Path)

    assets = commands.add_parser("assets", help="Install pinned offline Web UI assets.")
    assets_commands = assets.add_subparsers(dest="assets_command", required=True)
    assets_3dmol = assets_commands.add_parser(
        "install-3dmol",
        help="Install the hash-pinned 3Dmol.js build and its license.",
    )
    assets_3dmol.add_argument(
        "--approve-network",
        action="append",
        default=[],
        metavar="EXACT_DOMAIN",
        help=f"Approve exact download domain; required value is {THREEDMOL_HOST}.",
    )
    assets_3dmol.add_argument(
        "--from-file",
        type=Path,
        help="Use a previously reviewed 3Dmol-min.js instead of network access.",
    )
    assets_3dmol.add_argument(
        "--license-file",
        type=Path,
        help="Use the matching reviewed 3Dmol.js LICENSE with --from-file.",
    )
    _workspace_argument(assets_3dmol)

    data = commands.add_parser(
        "data",
        help="Acquire identifier-bound public protein or small-molecule files.",
    )
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_fetch = data_commands.add_parser(
        "fetch",
        help="Fetch one whitelisted public record with bounded curl and parse validation.",
    )
    data_fetch.add_argument("--source", required=True, choices=PUBLIC_DATA_SOURCES)
    data_fetch.add_argument("--identifier", required=True)
    data_fetch.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Project-relative output file; a provenance sidecar is written beside it.",
    )
    data_fetch.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Root that bounds the materialized output path.",
    )
    data_fetch.add_argument(
        "--approve-network",
        action="append",
        required=True,
        metavar="EXACT_DOMAIN",
        help="Approve the exact registry domain; arbitrary URLs remain unsupported.",
    )
    data_fetch.add_argument(
        "--skip-propka",
        action="store_true",
        help="Skip optional local PROPKA audit for downloaded protein structures.",
    )
    data_fetch.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing materialized file with different bytes.",
    )
    _workspace_argument(data_fetch)

    library = commands.add_parser(
        "library",
        help="Manage reusable private protein and ligand libraries with explicit consent.",
    )
    library_commands = library.add_subparsers(dest="library_command", required=True)
    library_init = library_commands.add_parser(
        "init",
        help="Create independent content-addressed protein and ligand roots.",
    )
    library_init.add_argument("--protein-root", type=Path, required=True)
    library_init.add_argument("--ligand-root", type=Path, required=True)
    library_init.add_argument(
        "--replace-config",
        action="store_true",
        help="Replace only the private config file; existing library objects are preserved.",
    )
    _library_config_argument(library_init)
    _data_access_confirmation(library_init)

    library_status = library_commands.add_parser(
        "status",
        help="Show path-redacted library counts and incoming queue status.",
    )
    library_status.add_argument(
        "--show-paths",
        action="store_true",
        help="Explicitly disclose configured absolute roots to this CLI terminal.",
    )
    _library_config_argument(library_status)
    _data_access_confirmation(library_status)

    library_scan = library_commands.add_parser(
        "scan",
        help="Hash a local selection and freeze a non-mutating import plan.",
    )
    library_scan.add_argument("--kind", choices=("protein", "ligand"), required=True)
    library_scan.add_argument("--source", type=Path, required=True)
    library_scan.add_argument("--recursive", action="store_true")
    library_scan.add_argument("--max-files", type=int, default=10_000)
    library_scan.add_argument(
        "--max-file-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    _library_config_argument(library_scan)
    _data_access_confirmation(library_scan)

    library_import = library_commands.add_parser(
        "import",
        help="Apply a frozen hash-bound plan using copy (default) or confirmed move.",
    )
    library_import.add_argument("--kind", choices=("protein", "ligand"), required=True)
    library_import.add_argument("--plan-id", required=True)
    library_import.add_argument("--mode", choices=("copy", "move"), default="copy")
    library_import.add_argument(
        "--confirm-move",
        help="For move mode, repeat the exact plan ID after reviewing the scan.",
    )
    _library_config_argument(library_import)
    _data_access_confirmation(library_import)

    library_list = library_commands.add_parser(
        "list",
        help="List path-redacted catalog entries.",
    )
    library_list.add_argument("--kind", choices=("protein", "ligand"), required=True)
    library_list.add_argument(
        "--state",
        choices=tuple(state.value for state in ImportState),
    )
    library_list.add_argument("--limit", type=int, default=100)
    _library_config_argument(library_list)
    _data_access_confirmation(library_list)

    library_show = library_commands.add_parser(
        "show",
        help="Show QC, identity state, and artifact references for one entry.",
    )
    library_show.add_argument("--kind", choices=("protein", "ligand"), required=True)
    library_show.add_argument("entry_id")
    _library_config_argument(library_show)
    _data_access_confirmation(library_show)

    library_verify = library_commands.add_parser(
        "verify-uniprot",
        help=(
            "Fetch one explicitly approved accession FASTA and compare it locally; "
            "the private sequence is never uploaded."
        ),
    )
    library_verify.add_argument("entry_id")
    library_verify.add_argument("--accession", required=True)
    library_verify.add_argument(
        "--approve-network",
        action="append",
        required=True,
        metavar="EXACT_DOMAIN",
        help="Must explicitly include rest.uniprot.org.",
    )
    _library_config_argument(library_verify)
    _workspace_argument(library_verify)
    _data_access_confirmation(library_verify)

    library_rag_sync = library_commands.add_parser(
        "rag-sync",
        help="Build a path/sequence/coordinate-free seekdb projection of a library.",
    )
    library_rag_sync.add_argument(
        "--kind", choices=("protein", "ligand"), default="protein"
    )
    library_rag_sync.add_argument(
        "--embedding-model",
        "--bge-model",
        dest="embedding_model",
        type=Path,
        required=True,
    )
    library_rag_sync.add_argument("--include-quarantined", action="store_true")
    _library_config_argument(library_rag_sync)
    _workspace_argument(library_rag_sync)
    _data_access_confirmation(library_rag_sync)

    library_rag_search = library_commands.add_parser(
        "rag-search",
        help="Retrieve candidate catalog entries from the derived library projection.",
    )
    library_rag_search.add_argument("question")
    library_rag_search.add_argument(
        "--kind", choices=("protein", "ligand"), default="protein"
    )
    library_rag_search.add_argument(
        "--embedding-model",
        "--bge-model",
        dest="embedding_model",
        type=Path,
        required=True,
    )
    library_rag_search.add_argument("--top-k", type=int, default=5)
    _library_config_argument(library_rag_search)
    _workspace_argument(library_rag_search)
    _data_access_confirmation(library_rag_search)

    site = commands.add_parser(
        "site",
        help="Run or parse prospective binding-site predictors without claiming truth.",
    )
    site_commands = site.add_subparsers(dest="site_command", required=True)
    site_p2rank_run = site_commands.add_parser(
        "p2rank-run",
        help="Run local P2Rank and emit a bounded site-hypothesis bundle.",
    )
    site_p2rank_run.add_argument("--receptor", type=Path, required=True)
    site_p2rank_run.add_argument("--output-dir", type=Path, required=True)
    site_p2rank_run.add_argument("--bundle", type=Path, required=True)
    site_p2rank_run.add_argument("--executable", default="prank")
    site_p2rank_run.add_argument(
        "--profile", choices=("default", "alphafold"), default="default"
    )
    site_p2rank_run.add_argument("--timeout", type=float, default=1800.0)
    site_p2rank_run.add_argument("--top-k", type=int, default=3)
    site_p2rank_run.add_argument("--replace", action="store_true")

    site_p2rank_parse = site_commands.add_parser(
        "p2rank-parse",
        help="Parse reviewed P2Rank predictions into the same hypothesis schema.",
    )
    site_p2rank_parse.add_argument("--predictions", type=Path, required=True)
    site_p2rank_parse.add_argument("--receptor", type=Path, required=True)
    site_p2rank_parse.add_argument("--bundle", type=Path, required=True)
    site_p2rank_parse.add_argument(
        "--p2rank-version",
        required=True,
        help="Exact version string recorded with the reviewed predictions CSV.",
    )
    site_p2rank_parse.add_argument(
        "--profile", choices=("default", "alphafold"), default="default"
    )
    site_p2rank_parse.add_argument("--top-k", type=int, default=3)
    site_p2rank_parse.add_argument("--replace", action="store_true")

    case = commands.add_parser("case", help="Run, resume, or inspect research cases.")
    case_commands = case.add_subparsers(dest="case_command", required=True)
    case_run = case_commands.add_parser("run", help="Create and execute a case.")
    case_run.add_argument("--case", type=Path, required=True)
    case_run.add_argument("--index", type=Path, required=True)
    case_run.add_argument("--run-id")
    case_run.add_argument("--mode", choices=tuple(mode.value for mode in ResearchMode))
    rcsb_identity = case_run.add_mutually_exclusive_group()
    rcsb_identity.add_argument(
        "--rcsb-pdb-id",
        help="Try this explicit PDB ID before protein folding.",
    )
    rcsb_identity.add_argument(
        "--rcsb-uniprot-accession",
        help="Discover RCSB polymer entities by this UniProt accession.",
    )
    case_run.add_argument(
        "--rcsb-chain",
        action="append",
        default=[],
        help="Explicit coordinate/auth chain ID; repeat for a two-chain target.",
    )
    case_run.add_argument(
        "--rcsb-assembly-id",
        metavar="POSITIVE_ID",
        help=(
            "For an explicit PDB ID, request this biological-assembly mmCIF. "
            "Without this flag ProtBind explicitly uses the deposited asymmetric unit."
        ),
    )
    case_run.add_argument(
        "--approve-network",
        action="append",
        default=[],
        metavar="EXACT_DOMAIN",
        help=(
            "Approve one exact RCSB domain for this invocation; repeat as needed. "
            "RCSB import uses files.rcsb.org and discovery uses search.rcsb.org."
        ),
    )
    case_run.add_argument(
        "--approve-sequence-upload",
        action="store_true",
        help="Separately allow target sequence upload to RCSB Search API.",
    )
    case_run.add_argument("--stop-after", choices=_STAGE_CHOICES, default="reported")
    case_run.add_argument("--worker-config", type=Path)
    case_run.add_argument(
        "--vina-environment-lock",
        type=Path,
        help="Freeze the exact local Vina/Meeko environment lock before SELECTED.",
    )
    _workspace_argument(case_run)
    case_resume = case_commands.add_parser("resume", help="Resume a hash-verified run.")
    case_resume.add_argument("run_id")
    case_resume.add_argument("--stop-after", choices=_STAGE_CHOICES, default="reported")
    case_resume.add_argument("--worker-config", type=Path)
    case_resume.add_argument(
        "--vina-environment-lock",
        type=Path,
        help="Attach the exact Vina/Meeko environment lock before resuming SELECTED.",
    )
    _workspace_argument(case_resume)
    case_show = case_commands.add_parser("show", help="Show a run manifest summary.")
    case_show.add_argument("run_id")
    _workspace_argument(case_show)
    case_report = case_commands.add_parser("report", help="Print a local report artifact.")
    case_report.add_argument("run_id")
    case_report.add_argument(
        "--format",
        choices=("markdown", "html", "degraded"),
        default="markdown",
    )
    _workspace_argument(case_report)
    case_dossier = case_commands.add_parser(
        "dossier",
        help="Build a detailed current-stage completion/control/artifact dossier.",
    )
    case_dossier.add_argument("run_id")
    case_dossier.add_argument(
        "--format",
        choices=("json", "markdown", "html"),
        default="markdown",
    )
    _workspace_argument(case_dossier)
    case_poses = case_commands.add_parser(
        "poses",
        help="Print coordinate-free docked-pose scene and geometry QA metadata.",
    )
    case_poses.add_argument("run_id")
    _workspace_argument(case_poses)
    case_attach = case_commands.add_parser(
        "attach", help="Attach a hash-bound support input before its consuming stage."
    )
    case_attach.add_argument("run_id")
    case_attach.add_argument("--name", required=True)
    case_attach.add_argument("--file", type=Path, required=True)
    case_attach.add_argument("--media-type")
    case_attach.add_argument("--replace", action="store_true")
    _workspace_argument(case_attach)
    case_gate = case_commands.add_parser(
        "gate",
        help="Deep-audit a run and issue a content-addressed one-stage continuation gate.",
    )
    case_gate.add_argument("run_id")
    case_gate.add_argument("--worker-config", type=Path)
    _workspace_argument(case_gate)
    case_advance = case_commands.add_parser(
        "advance",
        help="Execute and postflight exactly one stage using a fresh gate token.",
    )
    case_advance.add_argument("run_id")
    case_advance.add_argument("--continuation-token", required=True)
    case_advance.add_argument("--worker-config", type=Path)
    _workspace_argument(case_advance)

    knowledge = commands.add_parser("knowledge", help="Manage seekdb-backed evidence.")
    knowledge_commands = knowledge.add_subparsers(dest="knowledge_command", required=True)
    knowledge_inspect = knowledge_commands.add_parser(
        "inspect",
        help="Inspect local PDF/Markdown extraction and OCR readiness without indexing.",
    )
    knowledge_inspect.add_argument("document", type=Path)
    knowledge_inspect.add_argument(
        "--pdf-backend", choices=("auto", "pymupdf", "pdftotext"), default="auto"
    )
    knowledge_inspect.add_argument(
        "--ocr", choices=("off", "auto", "required"), default="off"
    )
    knowledge_inspect.add_argument("--ocr-language", default="eng")
    _data_access_confirmation(knowledge_inspect)
    knowledge_import = knowledge_commands.add_parser("import", help="Import local PDF/Markdown.")
    knowledge_import.add_argument("document", type=Path)
    knowledge_import.add_argument(
        "--embedding-model",
        "--bge-model",
        dest="embedding_model",
        type=Path,
        required=True,
    )
    knowledge_import.add_argument(
        "--pdf-backend", choices=("auto", "pymupdf", "pdftotext"), default="auto"
    )
    knowledge_import.add_argument(
        "--ocr", choices=("off", "auto", "required"), default="off"
    )
    knowledge_import.add_argument("--ocr-language", default="eng")
    knowledge_import.add_argument("--license")
    _workspace_argument(knowledge_import)
    _data_access_confirmation(knowledge_import)
    knowledge_model = knowledge_commands.add_parser(
        "model-doctor",
        help="Check a hash-pinned BGE-M3 or Qwen3-Embedding-0.6B local model.",
    )
    knowledge_model.add_argument("--embedding-model", type=Path)
    knowledge_freeze = knowledge_commands.add_parser(
        "model-freeze",
        help="Generate a local model file-hash manifest without loading or downloading it.",
    )
    knowledge_freeze.add_argument("--embedding-model", type=Path, required=True)
    knowledge_freeze.add_argument(
        "--model-name",
        choices=("BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B"),
        required=True,
    )
    knowledge_freeze.add_argument("--model-revision", required=True)
    knowledge_freeze.add_argument("--replace", action="store_true")
    knowledge_fetch = knowledge_commands.add_parser(
        "fetch", help="Fetch one explicitly approved HTTPS resource into artifacts."
    )
    knowledge_fetch.add_argument("url")
    knowledge_fetch.add_argument("--approve-network", action="append", required=True)
    knowledge_fetch.add_argument("--media-type", default="application/octet-stream")
    knowledge_fetch.add_argument("--license")
    knowledge_fetch.add_argument("--max-bytes", type=int, default=100 * 1024 * 1024)
    _workspace_argument(knowledge_fetch)

    ask = commands.add_parser("ask", help="Retrieve cited local seekdb evidence.")
    ask.add_argument("question")
    ask.add_argument(
        "--embedding-model",
        "--bge-model",
        dest="embedding_model",
        type=Path,
        required=True,
    )
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument(
        "--scope", choices=("evidence", "protein-library", "ligand-library")
    )
    _workspace_argument(ask)
    _data_access_confirmation(ask)

    serve = commands.add_parser("serve", help="Serve the private six-page web UI.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    _workspace_argument(serve)

    mcp = commands.add_parser(
        "mcp",
        help="Expose the closed-loop workflow through a restricted local MCP server.",
    )
    mcp_commands = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_serve = mcp_commands.add_parser(
        "serve",
        help="Serve stage-gated ProtBind tools over local stdio.",
    )
    mcp_serve.add_argument(
        "--transport",
        choices=("stdio",),
        default="stdio",
        help="Only local stdio is supported; network transports are intentionally disabled.",
    )
    mcp_serve.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Root that bounds all case and support input paths.",
    )
    mcp_serve.add_argument(
        "--library-config",
        type=Path,
        help=(
            "Optional private library config. Agent tools remain unavailable until this "
            "operator-selected file is supplied."
        ),
    )
    mcp_serve.add_argument("--worker-config", type=Path)
    mcp_serve.add_argument(
        "--knowledge-model",
        type=Path,
        help="Optional operator-selected, hash-pinned local embedding model.",
    )
    _workspace_argument(mcp_serve)

    benchmark = commands.add_parser(
        "benchmark", help="Benchmark TriPharm or run a scientific redocking calibration."
    )
    benchmark.add_argument(
        "benchmark_command",
        nargs="?",
        choices=(
            "redock",
            "redock-holdout",
            "redock-holdout-run",
            "redock-regression",
        ),
        help=(
            "Use 'redock' for one sealed-reference Meeko/Vina calibration or "
            "'redock-holdout' to freeze a result-blind PoseBusters ten-case holdout or "
            "'redock-holdout-run' to execute/resume that exact holdout or "
            "'redock-regression' to evaluate a hash-bound pilot/frozen manifest."
        ),
    )
    # These remain optional at parse time so the historical flat command and the
    # new redock subcommand can coexist.  _run enforces the mode-specific inputs.
    benchmark.add_argument("--index", type=Path)
    benchmark.add_argument("--query", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--backend", choices=("cpu", "hip"), default="cpu")
    benchmark.add_argument(
        "--hip-executable",
        type=Path,
        help="Production tripharm_hip_query executable; required for --backend hip.",
    )
    benchmark.add_argument("--repetitions", type=int, default=5)
    benchmark.add_argument("--warmup-runs", type=int, default=1)
    benchmark.add_argument("--top-k", type=int, default=512)
    benchmark.add_argument("--receptor", type=Path)
    benchmark.add_argument("--native-ligand", type=Path)
    benchmark.add_argument(
        "--receptor-source",
        help="Public provenance identifier (HTTPS URL or dataset:/pdb:/rcsb: label).",
    )
    benchmark.add_argument(
        "--native-ligand-source",
        help="Public provenance identifier (HTTPS URL or dataset:/pdb:/rcsb: label).",
    )
    benchmark.add_argument(
        "--input-license", help="License or data-policy identifier for both inputs."
    )
    benchmark.add_argument(
        "--conservative-receptor-repair",
        action="store_true",
        help=(
            "Repair only missing standard-residue heavy atoms outside the protected "
            "pocket; never rebuild loops or add hydrogens."
        ),
    )
    benchmark.add_argument(
        "--repair-protected-radius",
        type=float,
        default=6.0,
        metavar="ANGSTROM",
        help="Native-ligand atom radius inside which missing heavy atoms are never repaired.",
    )
    benchmark.add_argument(
        "--restrained-sidechain-optimization",
        action="store_true",
        help=(
            "After conservative repair, use CPU OpenMM to move only newly added "
            "side-chain atoms while fixing all original heavy atoms; every candidate "
            "must still pass geometry, chirality, RDKit, and Meeko gates."
        ),
    )
    benchmark.add_argument(
        "--sidechain-optimization-iterations",
        type=int,
        nargs="+",
        default=(250, 1000, 5000),
        metavar="N",
        help=(
            "Strictly increasing OpenMM iteration limits tried only when an earlier "
            "geometry or RDKit/Meeko receptor-chemistry gate fails."
        ),
    )
    benchmark.add_argument(
        "--protocol-revision",
        help=(
            "Lowercase frozen protocol identifier; required for a holdout run when "
            "--conservative-receptor-repair is enabled."
        ),
    )
    benchmark.add_argument("--seed", type=int, default=20260721)
    benchmark.add_argument("--padding", type=float, default=5.0)
    benchmark.add_argument("--exhaustiveness", type=int, default=32)
    benchmark.add_argument("--num-modes", type=int, default=9)
    benchmark.add_argument("--energy-range", type=float, default=3.0)
    benchmark.add_argument("--cpu", type=int, default=1)
    benchmark.add_argument(
        "--calibration-target-id",
        help=(
            "Emit a target-bound known-site receipt for a later both-mode screen; "
            "the reference ligand remains calibration-only."
        ),
    )
    benchmark.add_argument(
        "--calibration-required-rank",
        choices=("top1", "top5"),
        default="top1",
    )
    benchmark.add_argument(
        "--calibration-rmsd-threshold",
        type=float,
        default=2.0,
        metavar="ANGSTROM",
    )
    benchmark.add_argument("--timeout", type=float, default=1800.0)
    benchmark.add_argument("--vina-bin")
    benchmark.add_argument("--mk-prepare-receptor")
    benchmark.add_argument("--mk-prepare-ligand")
    benchmark.add_argument("--mk-export")
    benchmark.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root against which regression manifest paths are resolved.",
    )
    benchmark.add_argument(
        "--manifest",
        type=Path,
        help="Repo-relative hash-bound redock regression manifest.",
    )
    benchmark.add_argument(
        "--prolif-addhs-artifacts",
        type=Path,
        help=(
            "Optional independent artifact directory authorizing receipted RDKit AddHs "
            "for ProLIF; omit to require existing explicit ligand hydrogens."
        ),
    )
    benchmark.add_argument(
        "--archive",
        type=Path,
        help="Official checksum-pinned PoseBusters paper-data ZIP for redock-holdout.",
    )
    benchmark.add_argument(
        "--candidate-list",
        type=Path,
        help="Commit-pinned PoseBench 308-case PDB_CCD identifier list.",
    )
    benchmark.add_argument(
        "--holdout-artifacts",
        type=Path,
        help=(
            "Content-addressed store for selected public receptor/ligand inputs; "
            "required again when running the frozen holdout."
        ),
    )
    benchmark.add_argument(
        "--holdout",
        type=Path,
        help="Frozen schema-1.1 holdout manifest for redock-holdout-run.",
    )
    benchmark.add_argument(
        "--max-parallel-cases",
        type=int,
        default=2,
        choices=(1, 2),
        help=(
            "Maximum simultaneous single-CPU Vina cases; capped at two so the host and "
            "Radeon workflow retain headroom."
        ),
    )
    benchmark.add_argument(
        "--holdout-namespace",
        default=POSEBUSTERS_HOLDOUT_NAMESPACE,
        help="Frozen namespace used only for result-blind SHA-256 case ordering.",
    )
    benchmark.add_argument(
        "--holdout-pocket-radius",
        type=float,
        default=6.0,
        metavar="ANGSTROM",
        help="Ligand-atom radius used only for predeclared pocket input-QC exclusions.",
    )
    benchmark.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing holdout manifest output.",
    )

    return parser


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):  # noqa: ANN001
        return None


def _fetch(args: argparse.Namespace) -> dict[str, Any]:
    approved = tuple(args.approve_network)
    domain = require_network_approval(args.url, approved)
    parsed = urlsplit(args.url)
    if parsed.username or parsed.password:
        raise ValueError("credentials in fetch URLs are forbidden")
    if args.max_bytes < 1:
        raise ValueError("max-bytes must be >= 1")
    request = urllib.request.Request(
        args.url,
        headers={"User-Agent": f"ProtBind/{__version__} explicit-knowledge-import"},
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=60) as response:
            announced = response.headers.get("Content-Length")
            if announced and int(announced) > args.max_bytes:
                raise ValueError("remote resource exceeds max-bytes before download")
            data = response.read(args.max_bytes + 1)
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise PermissionError(
                "redirects are not followed automatically; approve the final target URL explicitly"
            ) from exc
        raise
    if len(data) > args.max_bytes:
        raise ValueError("remote resource exceeds max-bytes")
    source_without_query = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    artifact = ArtifactStore(args.workspace).put_bytes(
        data,
        media_type=args.media_type,
        producer="protbind.knowledge-fetch",
        producer_version=__version__,
        source=source_without_query,
        license=args.license,
    )
    return {
        "artifact_id": artifact.artifact_id,
        "size_bytes": artifact.size_bytes,
        "approved_domain": domain,
        "source": source_without_query,
    }


def _run(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        print(json.dumps(doctor_report(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "agent":
        if args.backend != "hipfire":
            raise ValueError("the built-in competition Agent currently supports HipFire only")
        prompt = args.prompt
        if prompt is None:
            if sys.stdin.isatty():
                prompt = input("ProtBind research request: ")
            else:
                prompt = sys.stdin.readline()
        if not prompt or not prompt.strip():
            raise ValueError("Agent prompt cannot be empty")
        result = create_runtime(
            workspace=args.workspace,
            project_root=args.project_root,
            model=args.model,
            base_url=args.base_url,
            library_config=args.library_config,
            knowledge_model=args.knowledge_model,
            pipeline_config=_worker_config(args.worker_config),
            max_steps=args.max_steps,
        ).run(prompt)
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        else:
            print(result.answer)
            if result.tool_timeline:
                print(
                    json.dumps(
                        {"tool_timeline": result.tool_timeline},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    file=sys.stderr,
                )
        return 0

    if args.command == "agent-benchmark":
        if not args.confirm_benchmark_data:
            raise PermissionError("Agent benchmark requires explicit local-data confirmation")
        result = run_agent_benchmark(
            workspace=args.workspace,
            project_root=args.project_root,
            workload_path=args.workload,
            run_id=args.run_id,
            knowledge_query=args.knowledge_query,
            preference=args.preference,
            knowledge_model=args.knowledge_model,
            model_weights=args.model_weights,
            hipfire_source_root=args.hipfire_source_root,
            hipfire_daemon=args.hipfire_daemon,
            library_config=args.library_config,
            pipeline_config=_worker_config(args.worker_config),
            config=AgentBenchmarkConfig(
                label=args.label,
                model=args.model,
                model_revision=args.model_revision,
                model_sha256=args.model_sha256,
                quantization=args.quantization,
                hipfire_revision=args.hipfire_revision,
                hipfire_visible_device=args.hipfire_visible_device,
                hipfire_speculation=args.hipfire_speculation,
                hipfire_jinja_mode=args.hipfire_jinja_mode,
                code_revision=args.code_revision,
                repetitions=args.repetitions,
                warmup_runs=args.warmup_runs,
            ),
            base_url=args.base_url,
        )
        save_agent_benchmark(result, args.output)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"saved: {args.output}", file=sys.stderr)
        return 0 if result["evidence_eligible"] else 3

    if args.command == "index":
        if args.index_command == "inspect":
            print(json.dumps(index_identity(args.index), ensure_ascii=False, indent=2))
            return 0
        config = TriPharmConfig(
            bin_width_angstrom=args.bin_width,
            tolerance_angstrom=args.tolerance,
            max_conformers=args.max_conformers,
        )
        existed = args.output.exists()
        if args.input.suffix.lower() == ".jsonl":
            stats = build_jsonl_index(
                args.input, args.output, config=config, overwrite=args.force
            )
        else:
            stats = build_index(
                load_chemical_library(
                    args.input,
                    seed=args.seed,
                    max_conformers=args.max_conformers,
                ),
                args.output,
                config=config,
                input_sha256=sha256_file(args.input),
                chemistry_verified=True,
                overwrite=args.force,
            )
        print(
            json.dumps(
                {**asdict(stats), "index_sha256": sha256_file(args.output), "replaced": existed},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "assets":
        if args.assets_command != "install-3dmol":
            raise AssertionError("unhandled assets command")
        print(
            json.dumps(
                install_3dmol_asset(
                    args.workspace,
                    approved_domains=tuple(dict.fromkeys(args.approve_network)),
                    javascript_file=args.from_file,
                    license_file=args.license_file,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    if args.command == "data":
        if args.data_command != "fetch":
            raise AssertionError("unhandled data command")
        validate_public_output(args.source, args.project_root, args.output)
        print(
            "Public data network preflight: "
            + json.dumps(
                {
                    "source": args.source,
                    "public_identifier": args.identifier,
                    "required_exact_domain": required_domain(args.source),
                    "approved_exact_domains": list(args.approve_network),
                    "private_sequence_uploaded": False,
                    "transport": "bounded direct curl; no proxy or redirect",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        fetcher = PublicDataFetcher(args.workspace)
        result = fetcher.fetch(
            source=args.source,
            identifier=args.identifier,
            approved_domains=tuple(dict.fromkeys(args.approve_network)),
            run_propka=not args.skip_propka,
        )
        materialized = materialize_public_fetch(
            result,
            fetcher.artifacts,
            project_root=args.project_root,
            output=args.output,
            replace=args.replace,
        )
        print(json.dumps(materialized, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    if args.command == "library":
        if not args.confirm_data_access:
            raise PermissionError("library commands require --confirm-data-access")
        if args.library_command == "init":
            config = save_library_config(
                args.config,
                protein_root=args.protein_root,
                ligand_root=args.ligand_root,
                replace=args.replace_config,
            )
            result = LibraryManager(config).status()
            result["config_file"] = args.config.name
        else:
            config = load_library_config(args.config)
            manager = LibraryManager(config)
            if args.library_command == "status":
                result = manager.status()
                if args.show_paths:
                    result["operator_disclosed_paths"] = {
                        "protein": str(config.protein_root),
                        "ligand": str(config.ligand_root),
                    }
            elif args.library_command == "scan":
                plan = manager.scan(
                    args.kind,
                    args.source,
                    recursive=args.recursive,
                    max_files=args.max_files,
                    max_file_bytes=args.max_file_bytes,
                )
                result = {
                    "schema_version": plan["schema_version"],
                    "plan_id": plan["plan_id"],
                    "kind": plan["kind"],
                    "library_root_id": plan["library_root_id"],
                    "file_count": len(plan["files"]),
                    "skipped": plan["skipped"],
                    "semantics": plan["semantics"],
                    "source_path_disclosed": False,
                    "next_command": (
                        f"protbind library import --kind {args.kind} "
                        f"--plan-id {plan['plan_id']} --confirm-data-access"
                    ),
                }
            elif args.library_command == "import":
                result = manager.apply_saved(
                    args.kind,
                    args.plan_id,
                    mode=args.mode,
                    confirm_move=args.confirm_move,
                )
            elif args.library_command == "list":
                result = manager.list_entries(
                    args.kind,
                    state=args.state,
                    limit=args.limit,
                )
            elif args.library_command == "show":
                result = manager.show_entry(args.kind, args.entry_id)
            elif args.library_command == "rag-sync":
                result = sync_library_rag(
                    args.workspace,
                    manager,
                    args.embedding_model,
                    kind=args.kind,
                    include_quarantined=args.include_quarantined,
                )
            elif args.library_command == "rag-search":
                hits = SeekDBKnowledgeStore(
                    args.workspace, args.embedding_model
                ).search(
                    args.question,
                    top_k=args.top_k,
                    scope=f"{args.kind}-library",
                )
                result = {
                    "question": args.question,
                    "scope": f"{args.kind}-library",
                    "answer_mode": (
                        "retrieval-only; verify selected entries against catalog.sqlite"
                    ),
                    "evidence": hits,
                }
            elif args.library_command == "verify-uniprot":
                fetcher = PublicDataFetcher(args.workspace)
                fetch = fetcher.fetch(
                    source="uniprot-fasta",
                    identifier=args.accession,
                    approved_domains=tuple(dict.fromkeys(args.approve_network)),
                    run_propka=False,
                )
                result = manager.verify_uniprot_bytes(
                    args.entry_id,
                    args.accession,
                    fetcher.artifacts.read_bytes(fetch.artifact),
                    source_artifact=fetch.artifact,
                )
                result["network_receipt"] = fetch.receipt.to_dict()
            else:
                raise AssertionError("unhandled library command")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    if args.command == "site":
        if args.site_command == "p2rank-run":
            result = run_p2rank(
                args.receptor,
                args.output_dir,
                executable=args.executable,
                profile=args.profile,
                timeout_seconds=args.timeout,
                top_k=args.top_k,
            )
        elif args.site_command == "p2rank-parse":
            result = parse_p2rank_predictions(
                args.predictions,
                receptor_sha256=sha256_file(args.receptor),
                p2rank_version=args.p2rank_version,
                profile=args.profile,
                top_k=args.top_k,
            )
        else:
            raise AssertionError("unhandled site command")
        write_p2rank_bundle(args.bundle, result, replace=args.replace)
        print(
            json.dumps(
                {
                    **result,
                    "bundle_file": args.bundle.name,
                    "bundle_sha256": sha256_file(args.bundle),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    if args.command == "case":
        workflow = ProtBindWorkflow(
            args.workspace,
            config=_worker_config(getattr(args, "worker_config", None)),
        )
        if args.case_command in {"gate", "advance"}:
            from .control import StageGateController

            controller = StageGateController(workflow)
            if args.case_command == "gate":
                result = controller.inspect(args.run_id)
            else:
                result = controller.advance(
                    args.run_id,
                    args.continuation_token,
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.case_command == "run":
            case = ingest_case(args.case, workflow.artifacts)
            if args.mode:
                case = replace(case, mode=ResearchMode(args.mode))
            target = case.target
            if args.rcsb_pdb_id:
                target = replace(
                    target,
                    pdb_id=args.rcsb_pdb_id,
                    uniprot_accession=None,
                    rcsb_coordinate_policy=(
                        RCSBCoordinatePolicy.DEPOSITED_ASYMMETRIC_UNIT
                    ),
                    rcsb_assembly_id=None,
                )
            elif args.rcsb_uniprot_accession:
                target = replace(
                    target,
                    pdb_id=None,
                    uniprot_accession=args.rcsb_uniprot_accession,
                    rcsb_chain_ids=(),
                    rcsb_coordinate_policy=(
                        RCSBCoordinatePolicy.DEPOSITED_ASYMMETRIC_UNIT
                    ),
                    rcsb_assembly_id=None,
                )
            if args.rcsb_assembly_id is not None:
                target = replace(
                    target,
                    rcsb_coordinate_policy=RCSBCoordinatePolicy.BIOLOGICAL_ASSEMBLY,
                    rcsb_assembly_id=args.rcsb_assembly_id,
                )
            if args.rcsb_chain:
                target = replace(target, rcsb_chain_ids=tuple(args.rcsb_chain))
            approved_domains = tuple(dict.fromkeys(args.approve_network))
            case = replace(
                case,
                target=target,
                privacy=replace(
                    case.privacy,
                    network_allowed=bool(approved_domains),
                    approved_domains=approved_domains,
                    sequence_upload_allowed=args.approve_sequence_upload,
                ),
            )
            if approved_domains:
                if target.pdb_id is not None:
                    sent_data = "PDB identifier only; no protein sequence"
                elif target.uniprot_accession is not None:
                    sent_data = "UniProt accession only; no protein sequence"
                elif args.approve_sequence_upload:
                    sent_data = "target protein sequence (separately approved)"
                else:
                    sent_data = "no RCSB discovery payload"
                print(
                    "RCSB network preflight: "
                    + json.dumps(
                        {
                            "approved_exact_domains": list(approved_domains),
                            "eligible_data": sent_data,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
            manifest = workflow.create(case, args.index, run_id=args.run_id)
            if args.vina_environment_lock is not None:
                workflow.attach_support(
                    manifest,
                    "vina_environment_lock",
                    args.vina_environment_lock,
                    media_type="application/json",
                )
            manifest = workflow.run(manifest, stop_after=_state(args.stop_after))
        elif args.case_command == "resume":
            manifest = workflow.manifests.load(args.run_id)
            if args.vina_environment_lock is not None:
                workflow.attach_support(
                    manifest,
                    "vina_environment_lock",
                    args.vina_environment_lock,
                    media_type="application/json",
                )
            manifest = workflow.run(manifest, stop_after=_state(args.stop_after))
        elif args.case_command == "show":
            manifest = workflow.manifests.load(args.run_id)
        elif args.case_command == "report":
            manifest = workflow.manifests.load(args.run_id)
            key = {
                "markdown": "report_markdown",
                "html": "report_html",
                "degraded": "degraded_report",
            }[args.format]
            artifact = manifest.artifacts.get(key)
            if artifact is None:
                raise ValueError(f"run has no {args.format} report artifact")
            report = workflow.artifacts.read_bytes(artifact)
            sys.stdout.buffer.write(report)
            if not report.endswith(b"\n"):
                print()
            return 0
        elif args.case_command == "dossier":
            from .control import StageGateController

            manifest = workflow.manifests.load(args.run_id)
            workflow.audit_manifest(manifest)
            pose_summary = build_pose_scene_summary(manifest, workflow.artifacts)
            controller = StageGateController(workflow)
            dossier = build_run_dossier(
                manifest,
                workflow.artifacts,
                control_history=controller.store.read(args.run_id),
                pose_summary=pose_summary,
            )
            references = persist_run_dossier(dossier, workflow.artifacts)
            content = dossier_content(dossier, args.format)
            if args.format == "json":
                value = json.loads(content)
                value["dossier_artifacts"] = {
                    name: reference.to_dict()
                    for name, reference in references.items()
                }
                content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            sys.stdout.write(content)
            if not content.endswith("\n"):
                print()
            return 0
        elif args.case_command == "poses":
            manifest = workflow.manifests.load(args.run_id)
            workflow.audit_manifest(manifest)
            print(
                json.dumps(
                    build_pose_scene_summary(manifest, workflow.artifacts),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0
        elif args.case_command == "attach":
            manifest = workflow.manifests.load(args.run_id)
            media_type = (
                args.media_type
                or mimetypes.guess_type(args.file.name)[0]
                or "application/octet-stream"
            )
            artifact = workflow.attach_support(
                manifest,
                args.name,
                args.file,
                media_type=media_type,
                replace=args.replace,
            )
            print(
                json.dumps(
                    {
                        "run_id": manifest.run_id,
                        "support_name": f"support_{args.name}",
                        "artifact_id": artifact.artifact_id,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        else:
            raise AssertionError("unhandled case command")
        print(
            json.dumps(
                _manifest_summary(manifest, workflow.artifacts),
                ensure_ascii=False,
                indent=2,
            )
        )
        return _manifest_exit(manifest)

    if args.command == "knowledge":
        if args.knowledge_command == "inspect":
            extraction = extract_document_bytes(
                args.document.read_bytes(),
                suffix=args.document.suffix,
                pdf_backend=args.pdf_backend,
                ocr=args.ocr,
                ocr_language=args.ocr_language,
            )
            print(
                json.dumps(
                    {
                        **extraction.receipt,
                        "source_name": args.document.name,
                        "text_returned": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.knowledge_command == "model-doctor":
            print(
                json.dumps(
                    inspect_embedding_model(args.embedding_model),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.knowledge_command == "model-freeze":
            print(
                json.dumps(
                    freeze_embedding_model_manifest(
                        args.embedding_model,
                        model_name=args.model_name,
                        model_revision=args.model_revision,
                        replace=args.replace,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.knowledge_command == "import":
            artifact, chunks, receipt = import_document(
                args.workspace,
                args.document,
                args.embedding_model,
                license=args.license,
                pdf_backend=args.pdf_backend,
                ocr=args.ocr,
                ocr_language=args.ocr_language,
            )
            print(
                json.dumps(
                    {
                        "artifact_id": artifact.artifact_id,
                        "chunks_indexed": chunks,
                        "extraction_receipt_artifact_id": receipt.artifact_id,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.knowledge_command == "fetch":
            print(json.dumps(_fetch(args), ensure_ascii=False, indent=2))
            return 0
        raise AssertionError("unhandled knowledge command")

    if args.command == "ask":
        hits = SeekDBKnowledgeStore(args.workspace, args.embedding_model).search(
            args.question, top_k=args.top_k, scope=args.scope
        )
        print(
            json.dumps(
                {
                    "question": args.question,
                    "answer_mode": "retrieval-only; no unsupported synthesis",
                    "evidence": hits,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if hits else 4

    if args.command == "serve":
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("ProtBind web UI may only bind to a loopback address")
        try:
            import uvicorn
        except ImportError as exc:
            raise ValueError("uvicorn is required for 'protbind serve'") from exc
        uvicorn.run(create_app(args.workspace), host=args.host, port=args.port)
        return 0

    if args.command == "mcp":
        if args.mcp_command != "serve":
            raise AssertionError("unhandled MCP command")
        from .mcp_server import serve_mcp

        serve_mcp(
            workspace=args.workspace,
            project_root=args.project_root,
            config=_worker_config(args.worker_config),
            library_config=args.library_config,
            knowledge_model=args.knowledge_model,
            transport=args.transport,
        )
        return 0

    if args.command == "benchmark":
        if args.benchmark_command == "redock-holdout":
            if args.archive is None or args.candidate_list is None:
                raise ValueError(
                    "benchmark redock-holdout requires --archive and --candidate-list"
                )
            artifact_root = args.holdout_artifacts
            if artifact_root is None:
                artifact_root = args.output.parent / f"{args.output.stem}-artifacts"
            frozen = freeze_posebusters_holdout(
                args.archive,
                args.candidate_list,
                ArtifactStore(artifact_root),
                namespace=args.holdout_namespace,
                pocket_radius_angstrom=args.holdout_pocket_radius,
            )
            write_holdout_manifest(
                args.output,
                frozen.manifest,
                overwrite=args.force,
            )
            print(
                json.dumps(
                    {
                        **frozen.summary(),
                        "manifest_file_sha256": sha256_file(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.benchmark_command == "redock-holdout-run":
            if args.holdout is None or args.holdout_artifacts is None:
                raise ValueError(
                    "benchmark redock-holdout-run requires --holdout and "
                    "--holdout-artifacts"
                )
            batch = run_frozen_redock_holdout(
                args.repo_root,
                args.holdout,
                ArtifactStore(args.holdout_artifacts),
                args.output,
                config=RedockHoldoutBatchConfig(
                    redock=RedockBenchmarkConfig(
                        seed=args.seed,
                        padding_angstrom=args.padding,
                        exhaustiveness=args.exhaustiveness,
                        num_modes=args.num_modes,
                        energy_range=args.energy_range,
                        cpu=args.cpu,
                        timeout_seconds=args.timeout,
                        vina=args.vina_bin,
                        mk_prepare_receptor=args.mk_prepare_receptor,
                        mk_prepare_ligand=args.mk_prepare_ligand,
                        mk_export=args.mk_export,
                        conservative_receptor_repair=(
                            args.conservative_receptor_repair
                        ),
                        repair_protected_radius_angstrom=(
                            args.repair_protected_radius
                        ),
                        restrained_sidechain_optimization=(
                            args.restrained_sidechain_optimization
                        ),
                        sidechain_optimization_iteration_limits=tuple(
                            args.sidechain_optimization_iterations
                        ),
                    ),
                    max_parallel_cases=args.max_parallel_cases,
                    protocol_revision=args.protocol_revision,
                ),
            )
            exit_code = 0 if batch["failed_count"] == 0 else 3
            print(
                json.dumps(
                    {
                        "schema_version": batch["schema_version"],
                        "kind": batch["kind"],
                        "process_exit_code": exit_code,
                        "batch_result_sha256": batch["batch_result_sha256"],
                        "target_case_count": batch["target_case_count"],
                        "terminal_count": batch["terminal_count"],
                        "completed_count": batch["completed_count"],
                        "failed_count": batch["failed_count"],
                        "top1_recovered_count": batch["top1_recovered_count"],
                        "top5_recovered_count": batch["top5_recovered_count"],
                        "regression_manifest": batch["regression_manifest"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return exit_code
        if args.benchmark_command == "redock-regression":
            if args.manifest is None:
                raise ValueError("benchmark redock-regression requires --manifest")
            prolif_artifacts = (
                ArtifactStore(args.prolif_addhs_artifacts)
                if args.prolif_addhs_artifacts is not None
                else None
            )
            result = build_redock_regression(
                args.repo_root,
                args.manifest,
                prolif_artifact_store=prolif_artifacts,
            )
            persist_redock_regression(result, args.output)
            frozen = result["evaluation_design"] == RegressionDesign.FROZEN_HOLDOUT.value
            if frozen:
                gate_status = "PASS" if result["gate_complete"] else "INCOMPLETE"
                exit_code = 0 if result["gate_complete"] else 3
            else:
                gate_status = "PILOT_NOT_ELIGIBLE"
                exit_code = 0
            print(
                json.dumps(
                    {
                        "schema_version": result["schema_version"],
                        "analysis": result["analysis"],
                        "evaluation_design": result["evaluation_design"],
                        "gate_status": gate_status,
                        "gate_complete": result["gate_complete"],
                        "process_exit_code": exit_code,
                        "regression_sha256": result["regression_sha256"],
                        "prolif_ligand_preparation_mode": result["config"][
                            "prolif_ligand_preparation_mode"
                        ],
                        "denominators": result["denominators"],
                        "pose_recovery_rates": result["pose_recovery_rates"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return exit_code
        if args.benchmark_command == "redock":
            if args.receptor is None or args.native_ligand is None:
                raise ValueError(
                    "benchmark redock requires --receptor PDB and --native-ligand SDF"
                )
            result = run_redock_benchmark(
                args.receptor,
                args.native_ligand,
                args.output,
                config=RedockBenchmarkConfig(
                    seed=args.seed,
                    padding_angstrom=args.padding,
                    exhaustiveness=args.exhaustiveness,
                    num_modes=args.num_modes,
                    energy_range=args.energy_range,
                    cpu=args.cpu,
                    timeout_seconds=args.timeout,
                    vina=args.vina_bin,
                    mk_prepare_receptor=args.mk_prepare_receptor,
                    mk_prepare_ligand=args.mk_prepare_ligand,
                    mk_export=args.mk_export,
                    receptor_source=args.receptor_source,
                    native_ligand_source=args.native_ligand_source,
                    input_license=args.input_license,
                    conservative_receptor_repair=(
                        args.conservative_receptor_repair
                    ),
                    repair_protected_radius_angstrom=args.repair_protected_radius,
                    restrained_sidechain_optimization=(
                        args.restrained_sidechain_optimization
                    ),
                    sidechain_optimization_iteration_limits=tuple(
                        args.sidechain_optimization_iterations
                    ),
                    calibration_target_id=args.calibration_target_id,
                    calibration_required_rank=args.calibration_required_rank,
                    calibration_rmsd_threshold_angstrom=args.calibration_rmsd_threshold,
                ),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["status"] != "COMPLETED":
                return 2
            if args.calibration_target_id is not None and result[
                "screening_calibration"
            ]["decision"]["status"] != "PASS":
                return 3
            return 0
        if args.index is None or args.query is None:
            raise ValueError(
                "flat benchmark mode requires --index, --query, and --output"
            )
        if args.backend == "hip":
            if args.hip_executable is None:
                raise ValueError("--hip-executable is required for --backend hip")
            result = benchmark_hip(
                args.index,
                args.query,
                args.hip_executable,
                repetitions=args.repetitions,
                warmup_runs=args.warmup_runs,
                top_k=args.top_k,
            )
        else:
            result = benchmark_cpu(
                args.index,
                args.query,
                repetitions=args.repetitions,
                warmup_runs=args.warmup_runs,
                top_k=args.top_k,
            )
        save_benchmark(result, args.output)
        print(json.dumps(result["duration_seconds"], ensure_ascii=False, indent=2))
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (
        ChemistryCapabilityError,
        BackendError,
        AgentLimitError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        KnowledgeCapabilityError,
        OSError,
        PermissionError,
        RuntimeError,
        StructureCapabilityError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
