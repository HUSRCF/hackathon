#!/usr/bin/env python3
"""Strict offline adapter for the official OpenFold3 0.4.3 ROCm CLI."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protbind_agent.artifacts import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from protbind_agent.models import ArtifactRef  # noqa: E402
from protbind_agent.openfold_contract import (  # noqa: E402
    OFFICIAL_CHECKPOINT_SIZES,
    OFFICIAL_RUNTIME_FILE_COUNT,
    OFFICIAL_RUNTIME_SHA256,
    OPENFOLD_BUNDLE_PRODUCER,
    OPENFOLD_ENGINE,
    OPENFOLD_QUERY_MANIFEST_PRODUCER,
    OPENFOLD_REVISION,
    OPENFOLD_RUN_METADATA_PRODUCER,
    OPENFOLD_RUNNER_PRODUCER,
    OPENFOLD_RUNTIME_ENGINE,
    OPENFOLD_SCM_NODE,
    OPENFOLD_VERSION,
)
from protbind_agent.privacy import redact_text  # noqa: E402
from protbind_agent.structure import (  # noqa: E402
    StructureCapabilityError,
    inspect_predicted_complex,
)
from protbind_agent.worker_protocol import (  # noqa: E402
    WorkerRequest,
    WorkerResponse,
)
from protbind_agent.worker_sdk import WorkerFailure, serve_worker  # noqa: E402

ENGINE = OPENFOLD_ENGINE
_SAFE_QUERY = re.compile(r"^[A-Za-z0-9_.-]+$")


def protbind_runtime_sha256() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src" / "protbind_agent"
    sources = [
        (
            str(path.relative_to(repository_root)),
            sha256_file(path),
        )
        for path in sorted(source_root.rglob("*.py"))
    ]
    if not sources:
        raise RuntimeError("ProtBind runtime source manifest is empty")
    return sha256_bytes(canonical_json_bytes(sources))


def composite_code_sha256(
    environment_lock_sha256: str, package_source_sha256: str
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "adapter_sha256": sha256_file(Path(__file__)),
                "protbind_runtime_sha256": protbind_runtime_sha256(),
                "environment_lock_sha256": environment_lock_sha256,
                "openfold_package_source_sha256": package_source_sha256,
                "openfold_revision": OPENFOLD_REVISION,
            }
        )
    )


def _runtime_attestation() -> dict[str, Any]:
    try:
        distribution = importlib_metadata.distribution("openfold3")
        scm_text = distribution.read_text("scm_version.json")
    except importlib_metadata.PackageNotFoundError as exc:
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "the pinned OpenFold3 distribution metadata is unavailable",
            recoverable=False,
        ) from exc
    try:
        scm = json.loads(scm_text) if scm_text is not None else None
    except json.JSONDecodeError as exc:
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "the installed OpenFold3 SCM metadata is malformed",
            recoverable=False,
        ) from exc
    expected_scm = {
        "tag": OPENFOLD_VERSION,
        "distance": 0,
        "node": OPENFOLD_SCM_NODE,
        "dirty": False,
    }
    if distribution.version != OPENFOLD_VERSION:
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "installed OpenFold3 version differs from the pinned runtime",
            recoverable=False,
        )
    entry_points = {
        entry.name: entry.value for entry in distribution.entry_points
    }
    if entry_points.get("run_openfold") != "openfold3.run_openfold:cli":
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "installed OpenFold3 run_openfold entry point is invalid",
            recoverable=False,
        )
    package_spec = importlib_util.find_spec("openfold3")
    module_spec = importlib_util.find_spec("openfold3.run_openfold")
    package_locations = (
        tuple(package_spec.submodule_search_locations or ())
        if package_spec is not None
        else ()
    )
    if (
        len(package_locations) != 1
        or module_spec is None
        or module_spec.origin is None
    ):
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "the executable OpenFold module is shadowed by another installation",
            recoverable=False,
        )
    package_root = Path(package_locations[0]).resolve()
    installed_module = (package_root / "run_openfold.py").resolve()
    if Path(module_spec.origin).resolve() != installed_module:
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "the executable OpenFold module is outside the attested package root",
            recoverable=False,
        )
    source_hashes: list[tuple[str, str]] = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = f"openfold3/{path.relative_to(package_root).as_posix()}"
        if (
            not (
                relative.endswith(".py")
                or "/core/data/resources/" in relative
                or relative.endswith("model_setting_presets.yml")
            )
        ):
            continue
        source_hashes.append((relative, sha256_file(path)))
    if not source_hashes or not any(
        name == "openfold3/run_openfold.py" for name, _ in source_hashes
    ):
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "installed OpenFold3 source files cannot be attested",
            recoverable=False,
        )
    package_source_sha256 = sha256_bytes(
        canonical_json_bytes(sorted(source_hashes))
    )
    official_release = (
        len(source_hashes) == OFFICIAL_RUNTIME_FILE_COUNT
        and package_source_sha256 == OFFICIAL_RUNTIME_SHA256
    )
    if not official_release and os.environ.get("PROTBIND_TEST_RUNTIME") != "1":
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "installed OpenFold3 files do not match the official 0.4.3 allowlist",
            recoverable=False,
        )
    # Upstream editable installs do not currently place a custom
    # ``scm_version.json`` inside dist-info.  The exact 317-file allowlist hash
    # is stronger evidence for that real installation than an invented metadata
    # file.  If SCM metadata is present it must still match exactly; if it is
    # absent, only the official source allowlist may supply the revision proof.
    if scm is None:
        if not official_release:
            raise WorkerFailure(
                "MODEL_RUNTIME_INVALID",
                "OpenFold3 SCM metadata is absent and source is not the official allowlist",
                recoverable=False,
            )
        scm_attestation = "official-source-allowlist"
        verified_scm = expected_scm
    elif not isinstance(scm, dict) or any(
        scm.get(name) != expected for name, expected in expected_scm.items()
    ):
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "installed OpenFold3 SCM revision differs from the pinned runtime",
            recoverable=False,
        )
    else:
        scm_attestation = "distribution-metadata+official-source-allowlist"
        verified_scm = scm
    try:
        import torch
        import triton

        torch_version = str(torch.__version__)
        hip_version = str(torch.version.hip) if torch.version.hip is not None else None
        triton_module_version = str(triton.__version__)
        triton_distribution = None
        triton_version = None
        for distribution_name in ("triton-rocm", "pytorch-triton-rocm", "triton"):
            try:
                triton_version = importlib_metadata.version(distribution_name)
                triton_distribution = distribution_name
                break
            except importlib_metadata.PackageNotFoundError:
                continue
        if triton_distribution is None or triton_version is None:
            raise importlib_metadata.PackageNotFoundError("ROCm Triton distribution")
    except (ImportError, importlib_metadata.PackageNotFoundError) as exc:
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "OpenFold3 requires an attested PyTorch/ROCm/Triton runtime",
            recoverable=False,
        ) from exc
    if hip_version is None and os.environ.get("PROTBIND_TEST_RUNTIME") != "1":
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "OpenFold3 production worker requires a ROCm-enabled PyTorch build",
            recoverable=False,
        )
    return {
        "distribution": "openfold3",
        "version": distribution.version,
        "scm_tag": verified_scm["tag"],
        "scm_distance": verified_scm["distance"],
        "scm_node": verified_scm["node"],
        "scm_dirty": verified_scm["dirty"],
        "scm_attestation": scm_attestation,
        "entry_point": entry_points["run_openfold"],
        "package_source_sha256": package_source_sha256,
        "runtime_file_count": len(source_hashes),
        "official_release": official_release,
        "torch_version": torch_version,
        "torch_hip_version": hip_version,
        "triton_distribution": triton_distribution,
        "triton_version": triton_version,
        "triton_module_version": triton_module_version,
    }


def _reference(value: Any, name: str) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", f"{name} is not an ArtifactRef", recoverable=False
        ) from exc


def _json(store: Any, reference: ArtifactRef, name: str) -> dict[str, Any]:
    value = store.read_json(reference)
    if not isinstance(value, dict):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", f"{name} must be a JSON object", recoverable=False
        )
    return value


def _contains_url(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower().startswith(("http://", "https://"))
    if isinstance(value, dict):
        return any(_contains_url(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_url(item) for item in value)
    return False


def _finite_sanitized(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkerFailure(
                "OUTPUT_INVALID", "OpenFold output contains NaN/Inf", recoverable=False
            )
        return value
    if isinstance(value, dict):
        return {str(key): _finite_sanitized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_sanitized(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _ranking_score(value: Any) -> float | None:
    scores: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "sample_ranking_score" and isinstance(child, int | float) and (
                    not isinstance(child, bool) and math.isfinite(float(child))
                ):
                    scores.append(float(child))
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return max(scores) if scores else None


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nested(value: dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate_effective_json(
    path: Path,
    value: dict[str, Any],
    *,
    query: dict[str, Any],
    seed: int,
    templates: bool,
    low_mem: bool,
    triton: bool,
    samples: int,
) -> None:
    name = path.name
    if name.endswith("_confidences_aggregated.json"):
        required = ("sample_ranking_score", "avg_plddt", "gpde", "ptm", "iptm")
        if any(not _finite_number(value.get(key)) for key in required) or not isinstance(
            value.get("has_clash"), bool
        ):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "OpenFold aggregate confidence is missing official finite metrics",
                recoverable=False,
            )
    elif name.endswith("_confidences.json"):
        if any(not isinstance(value.get(key), list) for key in ("plddt", "pde", "pae")):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "OpenFold full confidence is missing plddt/pde/pae arrays",
                recoverable=False,
            )
    elif name == "timing.json":
        runtime = value.get("runtime_s")
        if not _finite_number(runtime) or float(runtime) < 0:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "OpenFold timing lacks a finite nonnegative runtime_s",
                recoverable=False,
            )
    elif name == "experiment_config.json":
        settings = value.get("experiment_settings")
        output = value.get("output_writer_settings")
        if (
            not isinstance(settings, dict)
            or settings.get("seeds") != [seed]
            or settings.get("use_msa_server") is not False
            or settings.get("use_templates") is not templates
            or not isinstance(output, dict)
            or output.get("structure_format") != "cif"
            or output.get("write_full_confidence_scores") is not True
        ):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "effective OpenFold experiment config violates the requested offline settings",
                recoverable=False,
            )
    elif name == "model_config.json":
        memory_path = ("settings", "memory", "eval")
        expected = {
            (*memory_path, "use_triton_triangle_kernels"): triton,
            (*memory_path, "use_deepspeed_evo_attention"): False,
            (*memory_path, "use_cueq_triangle_kernels"): False,
            (
                "architecture",
                "shared",
                "diffusion",
                "no_full_rollout_samples",
            ): samples,
        }
        if low_mem:
            expected.update(
                {
                    ("settings", "clear_cache_between_steps"): True,
                    (*memory_path, "per_sample_token_cutoff"): 0,
                    (*memory_path, "per_sample_atom_cutoff"): 0,
                    (*memory_path, "offload_inference", "template_module"): True,
                    (*memory_path, "offload_inference", "msa_module"): True,
                    (*memory_path, "offload_inference", "confidence_heads"): True,
                    (*memory_path, "offload_inference", "token_cutoff"): 0,
                }
            )
        else:
            expected.update(
                {
                    ("settings", "clear_cache_between_steps"): False,
                    (*memory_path, "per_sample_token_cutoff"): 750,
                    (*memory_path, "per_sample_atom_cutoff"): 10000,
                    (*memory_path, "offload_inference", "template_module"): False,
                    (*memory_path, "offload_inference", "msa_module"): False,
                    (*memory_path, "offload_inference", "confidence_heads"): True,
                    (*memory_path, "offload_inference", "token_cutoff"): 2800,
                }
            )
        if any(
            _nested(value, *path) != expected_value
            for path, expected_value in expected.items()
        ):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "effective OpenFold model config does not prove the requested memory/kernel mode",
                recoverable=False,
            )
    elif name == "inference_query_set.json":
        effective_queries = value.get("queries")
        if (
            value.get("seeds") != [seed]
            or not isinstance(effective_queries, dict)
            or set(effective_queries) != set(query["queries"])
        ):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "effective OpenFold query set differs from the requested seeds/queries",
                recoverable=False,
            )
        for query_id, requested in query["queries"].items():
            effective = effective_queries[query_id]
            if not isinstance(effective, dict) or any(
                effective.get(flag) is not False
                for flag in ("use_msas", "use_paired_msas", "use_main_msas")
            ):
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "effective OpenFold query unexpectedly enables MSA inputs",
                    recoverable=False,
                )
            requested_chains = requested["chains"]
            effective_chains = effective.get("chains")
            if not isinstance(effective_chains, list) or len(effective_chains) != len(
                requested_chains
            ):
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "effective OpenFold chain set differs from the request",
                    recoverable=False,
                )
            for expected_chain, actual_chain in zip(
                requested_chains, effective_chains, strict=True
            ):
                if not isinstance(actual_chain, dict):
                    raise WorkerFailure(
                        "OUTPUT_INVALID",
                        "effective OpenFold chain is not an object",
                        recoverable=False,
                    )
                expected_ids = expected_chain["chain_ids"]
                actual_ids = actual_chain.get("chain_ids")
                expected_ids = (
                    [expected_ids] if isinstance(expected_ids, str) else expected_ids
                )
                actual_ids = [actual_ids] if isinstance(actual_ids, str) else actual_ids
                expected_type = str(expected_chain.get("molecule_type", "")).lower()
                actual_type = str(actual_chain.get("molecule_type", "")).lower()
                if (
                    actual_ids != expected_ids
                    or actual_type != expected_type
                    or actual_chain.get("sequence") != expected_chain.get("sequence")
                    or actual_chain.get("smiles") != expected_chain.get("smiles")
                ):
                    raise WorkerFailure(
                        "OUTPUT_INVALID",
                        "effective OpenFold chain identity differs from the request",
                        recoverable=False,
                    )
                if "template_cif_paths" in expected_chain and not actual_chain.get(
                    "template_cif_paths"
                ):
                    raise WorkerFailure(
                        "OUTPUT_INVALID",
                        "effective OpenFold query dropped a requested direct-CIF template",
                        recoverable=False,
                    )


def _runner_yaml(seed: int, *, low_mem: bool, triton: bool, templates: bool) -> str:
    presets = "\n    - predict" + ("\n    - low_mem" if low_mem else "")
    return (
        "experiment_settings:\n"
        "  mode: predict\n"
        f"  seeds: [{seed}]\n"
        "  use_msa_server: false\n"
        f"  use_templates: {str(templates).lower()}\n"
        "  skip_existing: false\n\n"
        "model_update:\n"
        f"  presets:{presets}\n"
        "  custom:\n"
        "    settings:\n"
        "      memory:\n"
        "        eval:\n"
        f"          use_triton_triangle_kernels: {str(triton).lower()}\n"
        "          use_deepspeed_evo_attention: false\n"
        "          use_cueq_triangle_kernels: false\n\n"
        "output_writer_settings:\n"
        "  structure_format: cif\n"
        "  write_full_confidence_scores: true\n\n"
        "pl_trainer_args:\n"
        "  devices: 1\n"
        "  precision: 32-true\n"
    )


def _query_payload(
    batch: dict[str, Any], store: Any, directory: Path, *, seed: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], bool]:
    chains = batch.get("protein_chains")
    candidates = batch.get("cofold_candidates")
    if not isinstance(chains, list) or not isinstance(candidates, list) or not candidates:
        raise WorkerFailure(
            "INPUT_NOT_PREPARED",
            "OpenFold batch requires protein_chains and cofold_candidates",
            recoverable=False,
        )
    queries: dict[str, Any] = {}
    mapping: dict[str, dict[str, Any]] = {}
    templates_enabled = False
    for rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise WorkerFailure(
                "INPUT_NOT_PREPARED", "cofold candidate is not an object", recoverable=False
            )
        candidate_id = str(candidate.get("candidate_id", ""))
        digest = sha256_bytes(candidate_id.encode())[:8]
        query_id = f"pb_{rank:04d}_{digest}"
        if not _SAFE_QUERY.fullmatch(query_id):
            raise AssertionError("internally generated query ID is unsafe")
        query_chains: list[dict[str, Any]] = []
        for chain in chains:
            if not isinstance(chain, dict):
                raise WorkerFailure(
                    "INPUT_NOT_PREPARED", "protein chain is not an object", recoverable=False
                )
            query_chain: dict[str, Any] = {
                "molecule_type": "protein",
                "chain_ids": str(chain["chain_id"]),
                "sequence": str(chain["sequence"]).removesuffix("*"),
            }
            if chain.get("template_cif") is not None:
                template = _reference(chain["template_cif"], "template_cif")
                source = store.resolve(template)
                destination = directory / (
                    f"template-{query_id}-{chain['chain_id']}-{template.sha256[:12]}.cif"
                )
                destination.symlink_to(source)
                query_chain["template_cif_paths"] = [str(destination)]
                if chain.get("template_chain_id") is not None:
                    query_chain["template_cif_chain_ids"] = [
                        str(chain["template_chain_id"])
                    ]
                templates_enabled = True
            query_chains.append(query_chain)
        smiles = candidate.get("canonical_isomeric_smiles")
        if not isinstance(smiles, str) or not smiles:
            raise WorkerFailure(
                "UNSUPPORTED_CHEMISTRY",
                "cofold candidate has no canonical isomeric SMILES",
                recoverable=False,
            )
        query_chains.append(
            {
                "molecule_type": "ligand",
                "chain_ids": "Z",
                "smiles": smiles,
            }
        )
        queries[query_id] = {
            "use_msas": False,
            "use_paired_msas": False,
            "use_main_msas": False,
            "chains": query_chains,
        }
        mapping[query_id] = candidate
    return {"seeds": [seed], "queries": queries}, mapping, templates_enabled


def _handler(request: WorkerRequest, store: Any) -> WorkerResponse:
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise WorkerFailure(
            "OFFLINE_POLICY_VIOLATION",
            "HSA_OVERRIDE_GFX_VERSION is forbidden",
            recoverable=False,
        )
    if request.provenance.model_revision != OPENFOLD_REVISION:
        raise WorkerFailure(
            "PROVENANCE_MISMATCH",
            "OpenFold3 adapter revision is not the pinned release",
            recoverable=False,
        )
    if _contains_url(request.parameters):
        raise WorkerFailure(
            "OFFLINE_POLICY_VIOLATION",
            "URLs are forbidden in offline OpenFold parameters",
            recoverable=False,
        )
    if request.parameters.get("use_msa_server", False) is not False:
        raise WorkerFailure(
            "OFFLINE_POLICY_VIOLATION", "MSA server use is forbidden", recoverable=False
        )
    allowed_parameters = {
        "num_diffusion_samples",
        "command_timeout_seconds",
        "low_mem",
        "use_triton_triangle_kernels",
        "use_msa_server",
        "checkpoint_name",
        "minimum_free_vram_gib",
    }
    unknown_parameters = set(request.parameters) - allowed_parameters
    if unknown_parameters:
        raise WorkerFailure(
            "OPENFOLD_INPUT_REJECTED",
            "unsupported OpenFold worker parameters: "
            + ", ".join(sorted(unknown_parameters)),
            recoverable=False,
        )
    visible_device = os.environ.get("HIP_VISIBLE_DEVICES", "")
    conflicting_masks = sorted(
        name
        for name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
        if os.environ.get(name)
    )
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", visible_device):
        raise WorkerFailure(
            "RESOURCE_POLICY_VIOLATION",
            "OpenFold3 must be pinned to one canonical HIP_VISIBLE_DEVICES index",
            recoverable=False,
        )
    if conflicting_masks:
        raise WorkerFailure(
            "RESOURCE_POLICY_VIOLATION",
            "OpenFold3 GPU assignment must use HIP_VISIBLE_DEVICES only",
            recoverable=False,
        )
    envelope = _json(store, request.input, "stage envelope")
    if envelope.get("kind") != "protbind.stage-input" or envelope.get("stage") != (
        "COFOLDED"
    ):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", "worker requires a COFOLDED stage envelope", recoverable=False
        )
    support = envelope.get("supporting_artifacts")
    if not isinstance(support, dict):
        raise WorkerFailure(
            "INPUT_NOT_PREPARED", "stage envelope has no support inputs", recoverable=False
        )
    batch_ref = _reference(support.get("support_openfold_batch"), "OpenFold batch")
    checkpoint = _reference(
        support.get("support_openfold_checkpoint"), "OpenFold checkpoint"
    )
    lock = _reference(
        support.get("support_openfold_environment_lock"), "OpenFold environment lock"
    )
    if checkpoint.sha256 != request.provenance.weight_sha256:
        raise WorkerFailure(
            "PROVENANCE_MISMATCH", "checkpoint SHA-256 differs from provenance", recoverable=False
        )
    checkpoint_path = store.resolve(checkpoint)
    store.resolve(lock)
    batch = _json(store, batch_ref, "OpenFold batch")
    samples = request.parameters.get("num_diffusion_samples", 1)
    timeout = request.parameters.get("command_timeout_seconds", 7200.0)
    low_mem = request.parameters.get("low_mem", True)
    triton = request.parameters.get("use_triton_triangle_kernels", True)
    checkpoint_name = request.parameters.get(
        "checkpoint_name", "openfold3-p2-155k"
    )
    minimum_free_vram_gib = request.parameters.get("minimum_free_vram_gib", 28.0)
    if (
        not isinstance(samples, int)
        or isinstance(samples, bool)
        or not 1 <= samples <= 20
        or not isinstance(timeout, int | float)
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
        or low_mem is not True
        or triton is not True
        or not isinstance(checkpoint_name, str)
        or checkpoint_name not in OFFICIAL_CHECKPOINT_SIZES
        or not isinstance(minimum_free_vram_gib, int | float)
        or isinstance(minimum_free_vram_gib, bool)
        or not math.isfinite(float(minimum_free_vram_gib))
        or float(minimum_free_vram_gib) < 24.0
    ):
        raise WorkerFailure(
            "OPENFOLD_INPUT_REJECTED", "invalid OpenFold worker parameters", recoverable=False
        )
    if os.environ.get("PROTBIND_TEST_RUNTIME") != "1" and samples != 1:
        raise WorkerFailure(
            "RESOURCE_POLICY_VIOLATION",
            "production OpenFold3 requires exactly one diffusion sample",
            recoverable=False,
        )
    runtime_attestation = _runtime_attestation()
    runtime_engine = (
        OPENFOLD_RUNTIME_ENGINE
        if runtime_attestation["official_release"]
        else "test-fixture-openfold3"
    )
    free_vram_bytes: int | None = None
    total_vram_bytes: int | None = None
    if runtime_attestation["official_release"]:
        import torch

        if checkpoint.size_bytes != OFFICIAL_CHECKPOINT_SIZES[checkpoint_name]:
            raise WorkerFailure(
                "PROVENANCE_MISMATCH",
                "checkpoint byte size differs from the declared official checkpoint",
                recoverable=False,
            )
        if not torch.cuda.is_available():
            raise WorkerFailure(
                "MODEL_UNAVAILABLE",
                "OpenFold3 production worker cannot access its reserved ROCm GPU",
                recoverable=True,
            )
        free_vram_bytes, total_vram_bytes = (
            int(value) for value in torch.cuda.mem_get_info(0)
        )
        required_free_bytes = int(float(minimum_free_vram_gib) * 1024**3)
        if free_vram_bytes < required_free_bytes:
            raise WorkerFailure(
                "GPU_CAPACITY_UNAVAILABLE",
                "reserved OpenFold3 GPU does not have the configured free-VRAM budget",
                recoverable=True,
            )
    if (
        composite_code_sha256(
            lock.sha256, str(runtime_attestation["package_source_sha256"])
        )
        != request.provenance.code_sha256
    ):
        raise WorkerFailure(
            "PROVENANCE_MISMATCH",
            "adapter/environment/OpenFold-package composite SHA-256 differs from provenance",
            recoverable=False,
        )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="openfold3-", dir=os.environ.get("TMPDIR")) as tmp:
        directory = Path(tmp)
        output_dir = directory / "output"
        output_dir.mkdir()
        checkpoint_link = directory / "checkpoint.pt"
        checkpoint_link.symlink_to(checkpoint_path)
        query, mapping, templates = _query_payload(
            batch, store, directory, seed=request.seed
        )
        query_path = directory / "query.json"
        query_path.write_bytes(canonical_json_bytes(query))
        runner_path = directory / "runner.yaml"
        runner_text = _runner_yaml(
            request.seed, low_mem=low_mem, triton=triton, templates=templates
        )
        runner_path.write_text(runner_text, encoding="utf-8")
        command = (
            sys.executable,
            "-m",
            "openfold3.run_openfold",
            "predict",
            "--query-json",
            str(query_path),
            "--inference-ckpt-path",
            str(checkpoint_link),
            "--use-msa-server=False",
            f"--use-templates={str(templates).lower()}",
            "--output-dir",
            str(output_dir),
            "--num-diffusion-samples",
            str(samples),
            "--runner-yaml",
            str(runner_path),
        )
        command_started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=float(timeout),
                env=dict(os.environ),
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkerFailure(
                "TIMEOUT", "OpenFold3 inference timed out", recoverable=True
            ) from exc
        command_elapsed = time.perf_counter() - command_started
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout or "").lower()
            if "out of memory" in diagnostic or "hip out of memory" in diagnostic:
                raise WorkerFailure(
                    "OUT_OF_MEMORY", "OpenFold3 exhausted device memory", recoverable=True
                )
            raise WorkerFailure(
                "OPENFOLD_INPUT_REJECTED",
                f"run_openfold failed with status {completed.returncode}",
                recoverable=True,
            )
        if any(path.is_symlink() for path in output_dir.rglob("*")):
            raise WorkerFailure(
                "OUTPUT_INVALID", "OpenFold output cannot contain symlinks", recoverable=False
            )
        expected_models: dict[str, list[Path]] = {}
        expected_confidences: dict[str, list[Path]] = {}
        expected_full_confidences: list[Path] = []
        expected_timings: list[Path] = []
        for query_id in sorted(mapping):
            seed_dir = output_dir / query_id / f"seed_{request.seed}"
            expected_models[query_id] = []
            expected_confidences[query_id] = []
            for sample in range(1, samples + 1):
                prefix = f"{query_id}_seed_{request.seed}_sample_{sample}"
                expected_models[query_id].append(seed_dir / f"{prefix}_model.cif")
                expected_confidences[query_id].append(
                    seed_dir / f"{prefix}_confidences_aggregated.json"
                )
                expected_full_confidences.append(
                    seed_dir / f"{prefix}_confidences.json"
                )
            expected_timings.append(seed_dir / "timing.json")
        models = [path for paths in expected_models.values() for path in paths]
        confidences = [
            path for paths in expected_confidences.values() for path in paths
        ]
        root_configs = [
            output_dir / "experiment_config.json",
            output_dir / "model_config.json",
            output_dir / "inference_query_set.json",
        ]
        expected_paths = (
            models
            + confidences
            + expected_full_confidences
            + expected_timings
            + root_configs
        )
        if any(not path.is_file() for path in expected_paths):
            raise WorkerFailure(
                "OUTPUT_INCOMPLETE",
                "OpenFold output does not match the requested query/seed/sample layout",
                recoverable=False,
            )
        if set(output_dir.rglob("*_model.cif")) != set(models) or set(
            output_dir.rglob("*_confidences_aggregated.json")
        ) != set(confidences):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "OpenFold emitted unexpected model or aggregate-confidence outputs",
                recoverable=False,
            )
        raw_references: dict[Path, ArtifactRef] = {}
        confidence_values: dict[Path, tuple[ArtifactRef, float]] = {}
        for path in sorted(
            set(confidences)
            | set(expected_full_confidences)
            | set(expected_timings)
            | set(root_configs)
        ):
            try:
                parsed = _finite_sanitized(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                raise WorkerFailure(
                    "OUTPUT_INVALID", "OpenFold emitted invalid JSON", recoverable=False
                ) from exc
            if not isinstance(parsed, dict):
                raise WorkerFailure(
                    "OUTPUT_INVALID", "OpenFold JSON output must be an object", recoverable=False
                )
            _validate_effective_json(
                path,
                parsed,
                query=query,
                seed=request.seed,
                templates=templates,
                low_mem=low_mem,
                triton=triton,
                samples=samples,
            )
            reference = store.put_json(
                parsed,
                producer=f"{runtime_engine}.sanitized-output",
                producer_version=OPENFOLD_REVISION,
                source=f"local-output:{path.name}",
            )
            raw_references[path] = reference
            if path.name.endswith("_confidences_aggregated.json"):
                score = _ranking_score(parsed)
                if score is None:
                    raise WorkerFailure(
                        "OUTPUT_INCOMPLETE",
                        "OpenFold confidence lacks sample_ranking_score",
                        recoverable=False,
                    )
                confidence_values[path] = (reference, score)
        for path in models:
            data = path.read_bytes()
            if len(data) < 10 or b"data_" not in data[:1024]:
                raise WorkerFailure(
                    "OUTPUT_INVALID", "OpenFold emitted an empty/invalid CIF", recoverable=False
                )
            query_id = path.parent.parent.name
            if query_id not in mapping:
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "OpenFold model path is not bound to a requested query",
                    recoverable=False,
                )
            try:
                inspection = inspect_predicted_complex(
                    path,
                    expected_sequences=tuple(
                        str(chain["sequence"]).removesuffix("*")
                        for chain in batch["protein_chains"]
                    ),
                )
            except (StructureCapabilityError, KeyError, TypeError, ValueError) as exc:
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "OpenFold emitted an invalid protein-ligand mmCIF complex",
                    recoverable=False,
                ) from exc
            expected_elements = mapping[query_id].get("heavy_element_counts")
            if (
                not isinstance(expected_elements, dict)
                or inspection.ligand_heavy_element_counts != expected_elements
            ):
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "OpenFold ligand elements differ from the requested microstate",
                    recoverable=False,
                )
            raw_references[path] = store.put_bytes(
                data,
                media_type="chemical/x-mmcif",
                producer=runtime_engine,
                producer_version=OPENFOLD_REVISION,
                source=f"local-output:{path.name}",
            )
        candidate_outputs: list[dict[str, Any]] = []
        for query_id, candidate in sorted(mapping.items()):
            sample_records: list[dict[str, Any]] = []
            for confidence_path in expected_confidences[query_id]:
                prefix = confidence_path.name.removesuffix(
                    "_confidences_aggregated.json"
                )
                model_path = confidence_path.with_name(f"{prefix}_model.cif")
                if model_path not in raw_references:
                    continue
                confidence_ref, score = confidence_values[confidence_path]
                sample_records.append(
                    {
                        "structure": raw_references[model_path].to_dict(),
                        "confidence": confidence_ref.to_dict(),
                        "sample_ranking_score": score,
                    }
                )
            if len(sample_records) != samples:
                raise WorkerFailure(
                    "OUTPUT_INCOMPLETE",
                    f"OpenFold model/confidence pairing is incomplete for {query_id}",
                    recoverable=False,
                )
            sample_records.sort(
                key=lambda item: (
                    -float(item["sample_ranking_score"]),
                    str(item["structure"]["sha256"]),
                )
            )
            best = sample_records[0]
            candidate_outputs.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "molecule_id": candidate["molecule_id"],
                    "microstate_id": candidate["microstate_id"],
                    "engine": runtime_engine,
                    "seed": request.seed,
                    "structure": best["structure"],
                    "confidence_name": "sample_ranking_score",
                    "confidence_value": best["sample_ranking_score"],
                    "confidence_semantics": (
                        "OpenFold3 model confidence; not binding affinity"
                    ),
                    "samples": sample_records,
                }
            )
        query_manifest = store.put_json(
            {
                "schema_version": "1.0",
                "query_ids": sorted(mapping),
                "candidate_ids": [mapping[key]["candidate_id"] for key in sorted(mapping)],
                "msa_server": False,
                "templates": templates,
                "stage_envelope": request.input.to_dict(),
                "input_batch": batch_ref.to_dict(),
                "checkpoint": checkpoint.to_dict(),
                "environment_lock": lock.to_dict(),
                "provenance": request.provenance.to_dict(),
                "runtime_attestation": runtime_attestation,
                "effective_config_artifacts": {
                    path.stem: raw_references[path].to_dict() for path in root_configs
                },
                "raw_outputs": [
                    reference.to_dict()
                    for _, reference in sorted(
                        raw_references.items(),
                        key=lambda item: str(item[0].relative_to(output_dir)),
                    )
                ],
            },
            producer=OPENFOLD_QUERY_MANIFEST_PRODUCER,
            producer_version=OPENFOLD_REVISION,
        )
        runner_artifact = store.put_bytes(
            runner_text.encode(),
            media_type="application/yaml",
            producer=OPENFOLD_RUNNER_PRODUCER,
            producer_version=OPENFOLD_REVISION,
        )
        metadata = store.put_json(
            {
                "schema_version": "1.0",
                "openfold_revision": OPENFOLD_REVISION,
                "stage_envelope": request.input.to_dict(),
                "input_batch": batch_ref.to_dict(),
                "checkpoint": checkpoint.to_dict(),
                "environment_lock": lock.to_dict(),
                "provenance": request.provenance.to_dict(),
                "checkpoint_name": checkpoint_name,
                "seed": request.seed,
                "num_diffusion_samples": samples,
                "low_mem": low_mem,
                "rocm_triton": triton,
                "msa_server": False,
                "templates": templates,
                "precision": "32-true",
                "runtime_attestation": runtime_attestation,
                "resource_policy": {
                    "hip_visible_device": visible_device,
                    "trainer_devices": 1,
                    "concurrent_openfold_jobs": 1,
                    "minimum_free_vram_gib": float(minimum_free_vram_gib),
                    "free_vram_bytes_before_run": free_vram_bytes,
                    "total_vram_bytes": total_vram_bytes,
                },
            },
            producer=OPENFOLD_RUN_METADATA_PRODUCER,
            producer_version=OPENFOLD_REVISION,
        )
        bundle = store.put_json(
            {
                "schema_version": "1.0",
                "kind": "protbind.cofold-bundle",
                "score_semantics": "model confidence only; not binding affinity",
                "candidates": candidate_outputs,
                "query_manifest": query_manifest.to_dict(),
                "runner": runner_artifact.to_dict(),
                "run_metadata": metadata.to_dict(),
            },
            producer=OPENFOLD_BUNDLE_PRODUCER,
            producer_version=OPENFOLD_REVISION,
        )
        ordered_raw = tuple(
            reference
            for _, reference in sorted(
                raw_references.items(), key=lambda item: str(item[0].relative_to(output_dir))
            )
        ) + (query_manifest, runner_artifact, metadata)
    return WorkerResponse(
        job_id=request.job_id,
        engine=request.engine,
        outputs=(bundle, *ordered_raw),
        provenance=request.provenance,
        timings_seconds={
            "openfold_command": command_elapsed,
            "worker_total": time.perf_counter() - started,
        },
        peak_vram_bytes=None,
        warnings=(
            "peak VRAM is unavailable from the child CLI and was not fabricated",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(serve_worker(ENGINE, _handler))
