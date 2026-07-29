"""Deterministic, resumable ProtBind case orchestration."""

from __future__ import annotations

import html
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from radeon_agent.hardware import probe_hardware

from . import __version__
from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes, sha256_file
from .chemistry import (
    ChemistryCapabilityError,
    heavy_element_counts,
    inspect_reference_ligand,
    smiles_pharmacophore,
)
from .cofold_batch import validate_cofold_batch
from .fusion import reciprocal_rank_fusion
from .interaction_fingerprint import interaction_fingerprint_metrics
from .manifest import (
    CofoldStatus,
    ManifestStore,
    RunManifest,
    RunState,
    StageRecord,
    stage_cache_key,
)
from .models import (
    ArtifactRef,
    ResearchCase,
    ResearchMode,
    SiteProvenanceKind,
    ValidationBundle,
)
from .openfold_contract import (
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
from .privacy import redact_text
from .selection import (
    QUICK_VINA_PURPOSE,
    build_quick_vina_input,
    build_selection_preparation,
    finalize_selection_bundle,
    known_site_calibration_summary,
    validate_quick_vina_batch,
)
from .structure import (
    StructureCapabilityError,
    inspect_declared_connections,
    inspect_predicted_complex,
    inspect_structure,
    pocket_pharmacophore,
)
from .structure_resolver import ResolutionDecision, StructureResolver
from .tripharm import (
    FeaturePoint,
    TriPharmHit,
    index_identity,
    query_index,
    read_index_metadata,
)
from .validation import classify_evidence
from .validation_input import (
    build_validation_input_batch,
    build_validation_toolchain,
)
from .worker_protocol import (
    JsonSubprocessWorker,
    WorkerExecutionError,
    WorkerProvenance,
    WorkerRequest,
)

_NORMAL_STAGES = (
    RunState.INPUT_VALIDATED,
    RunState.RECEPTOR_READY,
    RunState.INDEXED,
    RunState.SCREENED,
    RunState.SELECTED,
    RunState.DOCKED,
    RunState.VALIDATED,
    RunState.REPORTED,
)

_WORKER_BUNDLE_KIND = {
    RunState.COFOLDED: "protbind.cofold-bundle",
    RunState.DOCKED: "protbind.docking-bundle",
    RunState.VALIDATED: "protbind.validation-bundle",
}

_REPORT_CONFIG = {
    "report_schema": "1.2",
    "top_candidates": 5,
    "reference_rmsd_threshold_angstrom": 2.0,
    "ifp_consensus_threshold": 0.5,
    "exclude_rejected_from_top_evidence": True,
}

_INPUT_VALIDATION_CONFIG = {
    "schema": "1.4",
    "limits": "chains<=2,residues<=700,ligand_heavy<=100",
    "sequence_structure_identity": "exact-standard-residue-sequence",
    "required_backbone_atoms": ["N", "CA", "C"],
    "coordinate_qc": (
        "finite-no-unresolved-altloc-and-no-declared-covalent-crosslink"
    ),
    "structure_resolution_precedence": [
        "user_supplied",
        "local_exact_sequence_cache",
        "explicitly_approved_rcsb",
        "folding_required",
    ],
}

_RECEPTOR_READY_CONFIG = {
    "schema": "2.0",
    "precedence": [
        "case_target_structure",
        "verified_esmfold_structure",
        "explicit_support_receptor_structure",
    ],
    "require_content_addressed_coordinates": True,
}

_SELECTION_CONFIG = {
    "schema": "2.5",
    "maximum_candidates": 16,
    "source_contracts": [
        "protbind.selection-bundle",
        "protbind.cofold-input-batch",
    ],
    "ranking": "quick-vina-score,molecule-id,microstate-id",
}

_QUICK_VINA_ENGINE = "vina-quick"
_QUICK_SELECTION_KEYS = {
    "preparation": "selection_preparation",
    "input": "worker_input_selected",
    "batch": "selection_quick_vina_bundle",
    "receipt": "selection_quick_vina_receipt",
}


class PipelineStageError(RuntimeError):
    def __init__(self, code: str, message: str, *, recoverable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    engine: str
    argv: tuple[str, ...]
    provenance: WorkerProvenance
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 3600.0
    environment: dict[str, str] = field(default_factory=dict)
    isolate_network: bool = True
    allow_unisolated_test_fixture: bool = False

    def __post_init__(self) -> None:
        if not self.engine.strip():
            raise ValueError("worker engine cannot be empty")
        if not self.argv or any(not argument for argument in self.argv):
            raise ValueError("worker argv cannot be empty")
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("worker timeout must be positive")
        forbidden = [
            name
            for name in self.environment
            if name == "HSA_OVERRIDE_GFX_VERSION"
            or any(
                fragment in name.upper()
                for fragment in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
            )
        ]
        if forbidden:
            raise ValueError(
                "worker environment contains forbidden secret/spoofing variables: "
                + ", ".join(sorted(forbidden))
            )
        if "PROTBIND_TEST_RUNTIME" in self.environment:
            raise ValueError("PROTBIND_TEST_RUNTIME is reserved for protocol tests")
        visible_device = self.environment.get("HIP_VISIBLE_DEVICES")
        if visible_device is not None:
            if (
                not visible_device.isascii()
                or not visible_device.isdecimal()
                or (len(visible_device) > 1 and visible_device.startswith("0"))
            ):
                raise ValueError(
                    "GPU workers must reserve exactly one numeric HIP_VISIBLE_DEVICES "
                    "index in canonical form"
                )
            conflicting_masks = sorted(
                set(self.environment)
                & {
                    "CUDA_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL",
                    "ROCR_VISIBLE_DEVICES",
                }
            )
            if conflicting_masks:
                raise ValueError(
                    "GPU assignment must use HIP_VISIBLE_DEVICES only; "
                    "remove conflicting masks: " + ", ".join(conflicting_masks)
                )
        if self.engine == OPENFOLD_ENGINE and visible_device is None:
            raise ValueError(
                "OpenFold3 must reserve exactly one numeric HIP_VISIBLE_DEVICES index"
            )
        self.validate_launch_profile()
        if self.allow_unisolated_test_fixture and (
            self.isolate_network or self.provenance.model_revision != "fixture-only"
        ):
            raise ValueError(
                "unisolated fixture bypass is restricted to fixture-only provenance"
            )

    @property
    def identity_hash(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "engine": self.engine,
                    "argv": self.argv,
                    "provenance": self.provenance.to_dict(),
                    "parameters": self.parameters,
                    "timeout_seconds": self.timeout_seconds,
                    # Environment values may contain internal paths, so only bind their
                    # canonical digest into the public configuration identity.
                    "environment_sha256": sha256_bytes(
                        canonical_json_bytes(dict(sorted(self.environment.items())))
                    ),
                    "isolate_network": self.isolate_network,
                    "allow_unisolated_test_fixture": self.allow_unisolated_test_fixture,
                }
            )
        )

    def validate_launch_profile(self) -> None:
        """Recheck mutable parameter mappings immediately before worker launch."""

        if self.engine == _QUICK_VINA_ENGINE and not self.allow_unisolated_test_fixture:
            gpu_masks = sorted(
                set(self.environment)
                & {
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL",
                }
            )
            if gpu_masks:
                raise ValueError(
                    "production vina-quick is CPU-only and forbids GPU masks: "
                    + ", ".join(gpu_masks)
                )
            cpu = self.parameters.get("cpu", 1)
            exhaustiveness = self.parameters.get("exhaustiveness", 8)
            num_modes = self.parameters.get("num_modes", 1)
            if cpu != 1 or isinstance(cpu, bool):
                raise ValueError("production vina-quick requires cpu=1")
            if (
                not isinstance(exhaustiveness, int)
                or isinstance(exhaustiveness, bool)
                or not 1 <= exhaustiveness <= 16
            ):
                raise ValueError("production vina-quick exhaustiveness must be in [1, 16]")
            if (
                not isinstance(num_modes, int)
                or isinstance(num_modes, bool)
                or not 1 <= num_modes <= 3
            ):
                raise ValueError("production vina-quick num_modes must be in [1, 3]")
            if self.parameters.get("scoring", "vina") != "vina":
                raise ValueError("production vina-quick requires scoring='vina'")
        if self.engine != OPENFOLD_ENGINE or self.allow_unisolated_test_fixture:
            return
        low_mem = self.parameters.get("low_mem", True)
        triton = self.parameters.get("use_triton_triangle_kernels", True)
        msa_server = self.parameters.get("use_msa_server", False)
        checkpoint_name = self.parameters.get(
            "checkpoint_name", "openfold3-p2-155k"
        )
        samples = self.parameters.get("num_diffusion_samples", 1)
        minimum_free_vram_gib = self.parameters.get(
            "minimum_free_vram_gib", 28.0
        )
        if low_mem is not True or triton is not True or msa_server is not False:
            raise ValueError(
                "production OpenFold3 requires low_mem=true, ROCm Triton=true, "
                "and use_msa_server=false before launch"
            )
        if checkpoint_name not in OFFICIAL_CHECKPOINT_SIZES:
            raise ValueError("production OpenFold3 checkpoint_name is not allowed")
        if (
            not isinstance(samples, int)
            or isinstance(samples, bool)
            or samples != 1
            or not isinstance(minimum_free_vram_gib, int | float)
            or isinstance(minimum_free_vram_gib, bool)
            or not math.isfinite(float(minimum_free_vram_gib))
            or float(minimum_free_vram_gib) < 24.0
        ):
            raise ValueError(
                "production OpenFold3 requires one diffusion sample and valid "
                "resource parameters before launch"
            )


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    screen_top_k: int = 512
    rrf_k: int = 60
    workers: dict[RunState, WorkerConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.screen_top_k < 1:
            raise ValueError("screen_top_k must be >= 1")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        invalid = set(self.workers) - {
            RunState.SELECTED,
            RunState.COFOLDED,
            RunState.DOCKED,
            RunState.VALIDATED,
        }
        if invalid:
            names = sorted(item.value for item in invalid)
            raise ValueError(f"workers cannot implement stages: {names}")
        cofold_worker = self.workers.get(RunState.COFOLDED)
        if (
            cofold_worker is not None
            and not cofold_worker.allow_unisolated_test_fixture
            and cofold_worker.engine != OPENFOLD_ENGINE
        ):
            raise ValueError(
                "production COFOLDED worker engine must be the pinned openfold3 adapter"
            )
        selection_worker = self.workers.get(RunState.SELECTED)
        if (
            selection_worker is not None
            and not selection_worker.allow_unisolated_test_fixture
            and selection_worker.engine != _QUICK_VINA_ENGINE
        ):
            raise ValueError(
                "production SELECTED worker engine must be the pinned vina-quick adapter"
            )


def _artifact_input_hash(*artifacts: ArtifactRef, extra: Any = None) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "artifacts": [artifact.sha256 for artifact in artifacts],
                "extra": extra,
            }
        )
    )


def _package_hash() -> str:
    source_root = Path(__file__).parents[1]
    digest_parts = []
    for package_name in ("protbind_agent", "radeon_agent"):
        package_root = source_root / package_name
        for path in sorted(package_root.rglob("*.py")):
            digest_parts.append(
                (str(path.relative_to(source_root)), sha256_file(path))
            )
    return sha256_bytes(canonical_json_bytes(digest_parts))


def _stable_hardware_identity(hardware: dict[str, Any]) -> str:
    fields = (
        "host_fingerprint",
        "platform",
        "python_version",
        "rocm_version",
        "device_architectures",
        "architectures",
        "competition_roles",
        "hsa_override_active",
    )
    return sha256_bytes(
        canonical_json_bytes({name: hardware.get(name) for name in fields})
    )


def _worker_input_key(stage: RunState) -> str:
    return f"worker_input_{stage.value.lower()}"


def _worker_input_payload(
    manifest: RunManifest, stage: RunState
) -> dict[str, Any]:
    """Build a path-free, content-addressed dependency envelope for a stage worker."""

    previous_stage = {
        RunState.COFOLDED: RunState.SELECTED,
        RunState.DOCKED: RunState.SELECTED,
        RunState.VALIDATED: RunState.DOCKED,
    }[stage]
    previous_outputs = manifest.stage_records[previous_stage.value].outputs
    receipt: ArtifactRef | None = None
    scientific_outputs = previous_outputs
    if previous_outputs and previous_outputs[-1].producer == "protbind.worker-receipt":
        scientific_outputs = previous_outputs[:-1]
        receipt = previous_outputs[-1]
    # Workers receive only the support classes required by their exact stage.
    # This is both a stable causal projection (late validation material cannot
    # change an earlier envelope) and a least-disclosure boundary: an arbitrary
    # support name can never smuggle a native/reference artifact into docking.
    visible_support = {
        RunState.COFOLDED: {
            "support_openfold_batch",
            "support_openfold_checkpoint",
            "support_openfold_environment_lock",
        },
        RunState.DOCKED: {
            "support_selection_batch",
            "support_openfold_batch",
            "support_vina_environment_lock",
        },
        RunState.VALIDATED: {
            "support_validation_batch",
            "support_validation_toolchain",
            "support_reference_pose",
        },
    }[stage]
    supporting = {
        name: artifact.to_dict()
        for name, artifact in sorted(manifest.artifacts.items())
        if name in visible_support
    }
    if stage is not RunState.COFOLDED and manifest.cofold_record is not None:
        supporting["cofold_evidence_bundle"] = (
            manifest.cofold_record.outputs[0].to_dict()
        )
    return {
        "schema_version": "2.0",
        "kind": "protbind.stage-input",
        "stage": stage.value,
        # Full case and input artifacts may contain a reference ligand or raw
        # holo coordinates.  Downstream scientific workers do not need them;
        # keep only the non-sensitive identifier and hash-bound stage outputs.
        "case_id": manifest.case_id,
        "input_artifacts": {},
        "supporting_artifacts": supporting,
        "previous": {
            "stage": previous_stage.value,
            "scientific_outputs": [
                artifact.to_dict() for artifact in scientific_outputs
            ],
            "receipt": receipt.to_dict() if receipt is not None else None,
        },
    }


@contextmanager
def _gpu_lease(workspace: Path, device: str | None):
    """Serialize same-user ProtBind workers assigned to one physical GPU."""

    if device is None:
        yield
        return
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - ROCm production is Linux-only
        raise PipelineStageError(
            "GPU_LEASE_UNAVAILABLE",
            "GPU worker leases require Linux fcntl support",
            recoverable=False,
        ) from exc
    del workspace  # Leases intentionally span every ProtBind workspace on this host.
    lease_directory = _host_lease_directory()
    lease_path = lease_directory / f"gpu-{device}.lock"
    with lease_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineStageError(
                "GPU_BUSY",
                f"GPU {device} is leased by another ProtBind worker",
                recoverable=True,
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _openfold_job_lease(workspace: Path):
    """Enforce one OpenFold job across same-user ProtBind workspaces."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - ROCm production is Linux-only
        raise PipelineStageError(
            "GPU_LEASE_UNAVAILABLE",
            "OpenFold worker leases require Linux fcntl support",
            recoverable=False,
        ) from exc
    del workspace
    lease_directory = _host_lease_directory()
    lease_path = lease_directory / "openfold3-job.lock"
    with lease_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineStageError(
                "OPENFOLD_BUSY",
                "another same-user OpenFold3 job is already running on this host",
                recoverable=True,
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _host_lease_directory() -> Path:
    """Return a private, deterministic host-level lease directory for this UID."""

    uid = os.getuid()
    lease_directory = Path("/tmp") / f"protbind-gpu-leases-{uid}"
    try:
        lease_directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        stat = lease_directory.lstat()
    except OSError as exc:
        raise PipelineStageError(
            "GPU_LEASE_UNAVAILABLE",
            "cannot create the private host-level GPU lease directory",
            recoverable=False,
        ) from exc
    if (
        not lease_directory.is_dir()
        or stat.st_uid != uid
        or stat.st_mode & 0o077
    ):
        raise PipelineStageError(
            "GPU_LEASE_UNAVAILABLE",
            "host-level GPU lease directory has unsafe ownership or permissions",
            recoverable=False,
        )
    return lease_directory


@contextmanager
def _worker_resource_lease(workspace: Path, engine: str, device: str | None):
    """Acquire global engine capacity before the assigned physical-GPU lease."""

    if engine == OPENFOLD_ENGINE:
        with _openfold_job_lease(workspace), _gpu_lease(workspace, device):
            yield
        return
    with _gpu_lease(workspace, device):
        yield


def _parse_feature_artifact(
    store: ArtifactStore, artifact: ArtifactRef
) -> tuple[FeaturePoint, ...]:
    value = store.read_json(artifact)
    if isinstance(value, dict):
        value = value.get("features")
    if not isinstance(value, list):
        raise PipelineStageError(
            "INVALID_PHARMACOPHORE",
            f"artifact {artifact.artifact_id} is not a feature array",
            recoverable=False,
        )
    try:
        features = tuple(FeaturePoint.from_dict(item) for item in value)
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineStageError(
            "INVALID_PHARMACOPHORE",
            f"artifact {artifact.artifact_id} contains invalid features: {exc}",
            recoverable=False,
        ) from exc
    return features


def _feature_artifact(
    store: ArtifactStore,
    features: tuple[FeaturePoint, ...],
    *,
    producer: str,
    heuristic: bool,
) -> ArtifactRef:
    return store.put_json(
        {
            "schema_version": "1.0",
            "heuristic": heuristic,
            "features": [feature.to_dict() for feature in features],
        },
        producer=producer,
        producer_version=__version__,
    )


def _serialize_match(hit: TriPharmHit) -> dict[str, Any]:
    return {
        "molecule_id": hit.molecule_id,
        "conformer_id": hit.conformer_id,
        "query_coverage": hit.query_coverage,
        "median_normalized_distance_error": hit.median_normalized_distance_error,
        "geometric_match_score": hit.geometric_match_score,
        "original_smiles": hit.original_smiles,
        "standardized_smiles": hit.standardized_smiles,
        "matches": [
            {
                "query_feature_indices": list(match.query_feature_indices),
                "candidate_feature_indices": list(match.candidate_feature_indices),
                "normalized_distance_error": match.normalized_distance_error,
                "overlay": [list(row) for row in match.overlay],
            }
            for match in hit.matches
        ],
    }


def _validation_bundle(value: Any) -> ValidationBundle:
    if not isinstance(value, dict):
        raise ValueError("validation candidate bundle must be a JSON object")
    payload = dict(value)
    payload["unsupported_reasons"] = tuple(payload.get("unsupported_reasons", ()))
    evidence = payload.get("evidence", ())
    if not isinstance(evidence, list | tuple):
        raise ValueError("validation evidence must be an artifact reference array")
    payload["evidence"] = tuple(ArtifactRef.from_dict(reference) for reference in evidence)
    return ValidationBundle(**payload)


def _artifact_reference(value: Any, name: str) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid artifact reference") from exc


def _require_exact_reference(
    value: Any, expected: ArtifactRef, name: str
) -> ArtifactRef:
    reference = _artifact_reference(value, name)
    if reference != expected:
        raise ValueError(f"{name} does not name the exact frozen artifact")
    return reference


def _require_returned_reference(
    value: Any,
    returned: set[ArtifactRef],
    name: str,
) -> ArtifactRef:
    reference = _artifact_reference(value, name)
    if reference not in returned:
        raise ValueError(f"{name} must be returned with its bundle")
    return reference


def _finite_vector(value: Any, name: str, *, positive: bool = False) -> tuple[float, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            not isinstance(item, int | float)
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    result = tuple(float(item) for item in value)
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _require_same_vector(actual: Any, expected: Any, name: str) -> None:
    left = _finite_vector(actual, name, positive="size" in name.lower())
    right = _finite_vector(expected, f"frozen {name}", positive="size" in name.lower())
    if left != right:
        raise ValueError(f"{name} differs from the frozen quick-docking box")


def _has_fixture_label(value: Any) -> bool:
    return isinstance(value, str) and "fixture" in value.lower()


def _tool_evidence(
    store: ArtifactStore,
    reference: ArtifactRef,
    *,
    molecule_id: str,
    candidate_id: str,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any] | None, bool | None]:
    value = store.read_json(reference)
    if not isinstance(value, dict):
        raise ValueError("tool evidence must be a JSON object")
    if value.get("schema_version") != "1.0" or value.get("kind") != (
        "protbind.tool-evidence"
    ):
        raise ValueError("tool evidence must satisfy protbind.tool-evidence v1.0")
    tool = value.get("tool")
    allowed = {"posebusters", "spyrmsd", "prolif", "vina", "openfold3", "openmm"}
    if not isinstance(tool, str) or tool not in allowed:
        raise ValueError("tool evidence names an unsupported validator")
    if str(value.get("molecule_id")) != molecule_id:
        raise ValueError("tool evidence molecule_id does not match its candidate")
    if str(value.get("candidate_id")) != candidate_id:
        raise ValueError("tool evidence candidate_id does not match its pose candidate")
    if tool not in reference.producer.lower().replace("-", "").replace("_", ""):
        raise ValueError("tool evidence producer does not identify the claimed tool")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("tool evidence requires a metrics object")
    for metric_value in metrics.values():
        if isinstance(metric_value, float) and not math.isfinite(metric_value):
            raise ValueError("tool evidence metrics must be finite")
    inputs = value.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("tool evidence inputs must be an object")
    runtime = value.get("runtime")
    if runtime is not None and not isinstance(runtime, dict):
        raise ValueError("tool evidence runtime must be an object")
    fixture = value.get("test_fixture")
    if fixture is not None and not isinstance(fixture, bool):
        raise ValueError("tool evidence test_fixture label must be boolean")
    return tool, metrics, inputs, runtime, fixture


def _require_evidence_metric(
    evidence: dict[str, dict[str, Any]],
    tool: str,
    metric: str,
    expected: bool | float | None,
) -> None:
    if expected is None:
        return
    if tool not in evidence or metric not in evidence[tool]:
        raise ValueError(f"{tool} evidence is missing metric {metric}")
    actual = evidence[tool][metric]
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or actual is not expected:
            raise ValueError(f"{tool} metric {metric} does not match the bundle")
        return
    if (
        not isinstance(actual, int | float)
        or isinstance(actual, bool)
        or not math.isfinite(float(actual))
        or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{tool} metric {metric} does not match the bundle")


def _evidence_input(
    evidence_inputs: dict[str, dict[str, Any]], tool: str, name: str
) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(evidence_inputs[tool][name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{tool} evidence is missing artifact input {name}") from exc


class ProtBindWorkflow:
    def __init__(self, workspace: Path, *, config: PipelineConfig | None = None) -> None:
        self.workspace = workspace.resolve()
        self.artifacts = ArtifactStore(self.workspace)
        self.manifests = ManifestStore(self.workspace)
        self.structure_resolver = StructureResolver(
            self.workspace, artifacts=self.artifacts
        )
        self.config = config or PipelineConfig()
        self._hardware: dict[str, Any] | None = None

    def _hardware_evidence(self) -> dict[str, Any]:
        if self._hardware is None:
            self._hardware = probe_hardware().to_dict()
        return self._hardware

    def audit_manifest(self, manifest: RunManifest) -> ResearchCase:
        """Revalidate every completed binding without advancing the workflow.

        Interactive controllers use this before and after one-stage execution.
        Keeping the audit in the scientific workflow prevents an MCP/UI adapter
        from inventing a weaker definition of stage acceptance.
        """

        self._audit_artifacts(manifest)
        case = self.load_case(manifest)
        self._verify_case_artifacts(case)
        self._audit_configuration(manifest, case)
        return case

    def create(
        self,
        case: ResearchCase,
        index_path: Path,
        *,
        run_id: str | None = None,
    ) -> RunManifest:
        # Fail every local gate before an approved resolver is allowed to make a
        # network request.  In particular, a private sequence must never be sent
        # merely to discover later that the local index/artifacts are invalid or
        # that architecture spoofing makes the run inadmissible.
        hardware = self._hardware_evidence()
        if hardware.get("hsa_override_active"):
            raise ValueError(
                "HSA_OVERRIDE_GFX_VERSION is active; ProtBind competition runs forbid "
                "architecture spoofing"
            )
        if not index_path.is_file():
            raise FileNotFoundError(f"library index does not exist: {index_path.name}")
        self._verify_case_artifacts(case)
        if run_id is not None and self.manifests.path_for(run_id).exists():
            raise FileExistsError(
                f"run already exists: {run_id}; use 'protbind case resume'"
            )

        # Freeze and fully parse the local screening index before any approved
        # sequence can reach a discovery service.  The resolver must never be
        # used to discover that an unrelated local scientific input was bad.
        index_artifact = self.artifacts.import_file(
            index_path,
            media_type="application/vnd.sqlite3",
            producer="protbind.tripharm.index",
            producer_version=__version__,
        )
        frozen_index_path = self.artifacts.resolve(index_artifact)
        index_config, _ = read_index_metadata(frozen_index_path)
        index_identity(frozen_index_path)
        self._preflight_local_case_inputs(case, index_config.max_query_points)

        resolution = self.structure_resolver.resolve(case.target, case.privacy)
        if resolution.structure is not None and case.target.structure != resolution.structure:
            case = replace(
                case,
                target=replace(case.target, structure=resolution.structure),
            )
        self._preflight_resolved_queries(case, index_config.max_query_points)
        case_artifact = self.artifacts.put_json(
            case.to_dict(), producer="protbind.case", producer_version=__version__
        )
        resolved_run_id = run_id or f"{case.case_id}-{case_artifact.sha256[:12]}"
        manifest_path = self.manifests.path_for(resolved_run_id)
        if manifest_path.exists():
            raise FileExistsError(
                f"run already exists: {resolved_run_id}; use 'protbind case resume'"
            )
        input_artifacts = {
            "library_index": index_artifact,
            "target_resolution": resolution.receipt,
        }
        if resolution.raw_source is not None:
            input_artifacts["target_raw_source"] = resolution.raw_source
        manifest = RunManifest(
            run_id=resolved_run_id,
            case_id=case.case_id,
            case_artifact=case_artifact,
            input_artifacts=input_artifacts,
            provenance={
                "protbind_version": __version__,
                "code_sha256": _package_hash(),
                "hardware_sha256": _stable_hardware_identity(hardware),
                "hsa_override_active": str(hardware.get("hsa_override_active", False)).lower(),
                "target_structure_resolution": resolution.decision.value,
            },
        )
        self.manifests.save(manifest)
        return manifest

    def _verify_case_artifacts(self, case: ResearchCase) -> None:
        references = []
        if case.target.structure is not None:
            references.append(case.target.structure)
        if case.ligand is not None:
            references.extend(
                item
                for item in (case.ligand.structure, case.ligand.pharmacophore)
                if item is not None
            )
        if case.pocket is not None:
            references.extend(
                item
                for item in (
                    case.pocket.pharmacophore,
                    case.pocket.site_evidence,
                    case.pocket.known_site_calibration_receipt,
                )
                if item is not None
            )
        for reference in references:
            self.artifacts.resolve(reference)

    @staticmethod
    def _require_query_feature_count(
        features: tuple[FeaturePoint, ...],
        max_query_points: int,
        branch: str,
    ) -> None:
        if not 3 <= len(features) <= max_query_points:
            raise PipelineStageError(
                "INVALID_PHARMACOPHORE",
                f"{branch} pharmacophore requires 3..{max_query_points} feature points",
                recoverable=False,
            )

    def _preflight_ligand_chemistry(self, case: ResearchCase) -> Any | None:
        if case.ligand is None or (
            case.ligand.smiles is None and case.ligand.structure is None
        ):
            return None
        inspection = inspect_reference_ligand(
            smiles=case.ligand.smiles,
            structure_path=(
                self.artifacts.resolve(case.ligand.structure)
                if case.ligand.structure is not None
                else None
            ),
            structure_media_type=(
                case.ligand.structure.media_type
                if case.ligand.structure is not None
                else None
            ),
        )
        if inspection.metal_elements:
            raise PipelineStageError(
                "UNSUPPORTED_METAL_LIGAND",
                "v1 rejects metal-containing ligands: "
                + ", ".join(inspection.metal_elements),
                recoverable=False,
            )
        if inspection.heavy_atom_count > 100:
            raise PipelineStageError(
                "UNSUPPORTED_LIGAND_SIZE",
                "v1 supports ligands with at most 100 heavy atoms",
                recoverable=False,
            )
        if inspection.unassigned_stereocenters:
            raise PipelineStageError(
                "AMBIGUOUS_STEREOCHEMISTRY",
                "reference ligand contains unassigned tetrahedral stereocenters",
                recoverable=False,
            )
        if case.ligand.heavy_atom_count is not None and (
            case.ligand.heavy_atom_count != inspection.heavy_atom_count
        ):
            raise PipelineStageError(
                "LIGAND_IDENTITY_MISMATCH",
                "declared and computed ligand heavy-atom counts differ",
                recoverable=False,
            )
        return inspection

    def _preflight_local_case_inputs(
        self, case: ResearchCase, max_query_points: int
    ) -> None:
        self._preflight_ligand_chemistry(case)
        if case.mode in {ResearchMode.BOTH, ResearchMode.LIGAND_ONLY}:
            if case.ligand is None:
                raise ValueError("ligand mode has no ligand hypothesis")
            if case.ligand.pharmacophore is not None:
                features = _parse_feature_artifact(
                    self.artifacts, case.ligand.pharmacophore
                )
                self._require_query_feature_count(
                    features, max_query_points, "ligand"
                )
            elif case.ligand.smiles:
                features = smiles_pharmacophore(case.ligand.smiles, seed=case.seed)
                self._require_query_feature_count(
                    features, max_query_points, "ligand"
                )
            else:
                raise PipelineStageError(
                    "LIGAND_QUERY_UNAVAILABLE",
                    "ligand screening requires reference SMILES or a pharmacophore artifact",
                    recoverable=False,
                )
        if case.mode in {ResearchMode.BOTH, ResearchMode.POCKET_ONLY}:
            if case.pocket is None:
                raise ValueError("pocket mode has no pocket hypothesis")
            if case.pocket.pharmacophore is not None:
                features = _parse_feature_artifact(
                    self.artifacts, case.pocket.pharmacophore
                )
                self._require_query_feature_count(
                    features, max_query_points, "pocket"
                )
            elif case.target.structure is not None:
                features = pocket_pharmacophore(
                    self.artifacts.resolve(case.target.structure),
                    residues=case.pocket.residues,
                    center=case.pocket.center,
                    box_size=case.pocket.box_size,
                )
                self._require_query_feature_count(
                    features, max_query_points, "pocket"
                )

    def _preflight_resolved_queries(
        self, case: ResearchCase, max_query_points: int
    ) -> None:
        if (
            case.mode in {ResearchMode.BOTH, ResearchMode.POCKET_ONLY}
            and case.pocket is not None
            and case.pocket.pharmacophore is None
            and case.target.structure is None
        ):
            # A folding-required case may attach a verified ESMFold receptor before
            # SCREENED.  All already-available local ligand/pocket inputs were
            # checked above; do not make manifest creation impossible here.
            return
        queries, _ = self._queries(case)
        for branch, features in queries.items():
            self._require_query_feature_count(features, max_query_points, branch)

    def load_case(self, manifest: RunManifest) -> ResearchCase:
        value = self.artifacts.read_json(manifest.case_artifact)
        if not isinstance(value, dict):
            raise ValueError("case artifact is not a JSON object")
        case = ResearchCase.from_dict(value)
        if case.case_id != manifest.case_id:
            raise ValueError("manifest case_id does not match its case artifact")
        return case

    @staticmethod
    def _allowed_receptors(
        manifest: RunManifest, case: ResearchCase
    ) -> tuple[ArtifactRef, ...]:
        candidates = [case.target.structure]
        candidates.extend(
            manifest.artifacts.get(name)
            for name in (
                "receptor_ready_structure",
                "support_receptor_structure",
                "support_esmfold_structure",
            )
        )
        unique: dict[str, ArtifactRef] = {}
        for candidate in candidates:
            if candidate is not None:
                unique.setdefault(candidate.sha256, candidate)
        return tuple(unique.values())

    def attach_support(
        self,
        manifest: RunManifest,
        name: str,
        path: Path,
        *,
        media_type: str,
        replace: bool = False,
    ) -> ArtifactRef:
        """Freeze an explicit local support artifact before its consuming stage."""

        if not name or len(name) > 64 or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in name
        ):
            raise ValueError("support name must be a safe 1-64 character identifier")
        if not media_type or "/" not in media_type:
            raise ValueError("support media_type must be a MIME type")
        if manifest.state is RunState.FAILED:
            raise ValueError("cannot attach support to a failed run")
        if manifest.is_read_only:
            raise ValueError(
                "manifest schema 1.0 is read-only; create a new schema-2 run"
            )
        reference_support = name == "reference_pose"
        receptor_support = name in {
            "esmfold_receipt",
            "receptor_structure",
        }
        selection_support = name == "selection_batch"
        cofold_support = name in {
            "openfold_batch",
            "openfold_checkpoint",
            "openfold_environment_lock",
        }
        validation_support = name.startswith("validation_")
        if reference_support:
            if RunState.DOCKED.value not in manifest.stage_records:
                raise ValueError(
                    "reference_pose is VALIDATION_ONLY and may be attached only after "
                    "DOCKED completes"
                )
            if RunState.VALIDATED.value in manifest.stage_records:
                raise ValueError(
                    "reference_pose is frozen once VALIDATED completes; create a new run"
                )
        if receptor_support and RunState.RECEPTOR_READY.value in manifest.stage_records:
            raise ValueError(
                "receptor inputs are frozen once RECEPTOR_READY completes; create a new run"
            )
        if selection_support and RunState.SELECTED.value in manifest.stage_records:
            raise ValueError(
                "selection inputs are frozen once SELECTED completes; create a new run"
            )
        if cofold_support and (
            RunState.DOCKED.value in manifest.stage_records
            or manifest.cofold_status in {CofoldStatus.RUNNING, CofoldStatus.COMPLETED}
        ):
            raise ValueError(
                "optional cofold inputs are frozen once cofold runs or DOCKED completes; "
                "create a new run"
            )
        if validation_support and RunState.VALIDATED.value in manifest.stage_records:
            raise ValueError(
                "validation inputs are frozen once VALIDATED completes; create a new run"
            )
        if not (
            reference_support
            or receptor_support
            or selection_support
            or cofold_support
            or validation_support
        ) and (
            RunState.SELECTED.value in manifest.stage_records
        ):
            raise ValueError(
                "support inputs are frozen once SELECTED completes; create a new run"
            )
        case = self.audit_manifest(manifest)
        if name == "esmfold_structure":
            raise ValueError(
                "ESMFold structures must be attached through a verified "
                "esmfold_receipt, not as an unprovenanced generic file"
            )
        reference = self.artifacts.import_file(
            path,
            media_type=media_type,
            producer="protbind.support-import",
            producer_version=__version__,
        )
        key = f"support_{name}"
        existing = manifest.artifacts.get(key)
        if existing is not None and existing != reference and not replace:
            raise FileExistsError(
                f"support {name!r} already exists; pass replace=True explicitly"
            )
        if name == "esmfold_receipt":
            structure, result_metadata, source_metadata = (
                self._validate_esmfold_receipt(reference, case)
            )
            for support_name, support_reference in (
                ("support_esmfold_structure", structure),
                ("support_esmfold_result_metadata", result_metadata),
            ):
                support_existing = manifest.artifacts.get(support_name)
                if (
                    support_existing is not None
                    and support_existing != support_reference
                    and not replace
                ):
                    raise FileExistsError(
                        f"support {support_name.removeprefix('support_')!r} already "
                        "exists; pass replace=True explicitly"
                    )
            self.structure_resolver.register(
                structure,
                case.target.sequences,
                origin="local_esmfold_v1",
                source_metadata=source_metadata,
            )
            manifest.artifacts["support_esmfold_structure"] = structure
            manifest.artifacts["support_esmfold_result_metadata"] = result_metadata
        manifest.artifacts[key] = reference
        self.manifests.save(manifest)
        return reference

    def _validate_esmfold_receipt(
        self,
        receipt_reference: ArtifactRef,
        case: ResearchCase,
    ) -> tuple[ArtifactRef, ArtifactRef, dict[str, object]]:
        value = self.artifacts.read_json(receipt_reference)
        if not isinstance(value, dict) or value.get("schema_version") != "1.0" or (
            value.get("kind") != "protbind.esmfold-v1-smoke-receipt"
            or value.get("engine") != "esmfold_v1"
            or value.get("success") is not True
            or value.get("error") is not None
        ):
            raise ValueError("ESMFold receipt is not a successful v1 offline smoke result")
        outputs = value.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 2:
            raise ValueError("ESMFold receipt must bind one structure and one result metadata")
        structure = _artifact_reference(outputs[0], "ESMFold structure")
        result_metadata = _artifact_reference(outputs[1], "ESMFold result metadata")
        self.artifacts.resolve(structure)
        self.artifacts.resolve(result_metadata)
        if (
            structure.producer != "fair-esm.esmfold_v1"
            or structure.media_type != "chemical/x-pdb"
            or result_metadata.producer != "protbind.esmfold-v1-result"
        ):
            raise ValueError("ESMFold receipt output producers are not the pinned adapters")
        input_reference = _artifact_reference(value.get("input"), "ESMFold input")
        if input_reference.producer != "protbind.esmfold-v1.smoke-input" or (
            structure.source != input_reference.artifact_id
            or result_metadata.source != structure.artifact_id
        ):
            raise ValueError("ESMFold receipt input/output artifact lineage is invalid")
        input_value = self.artifacts.read_json(input_reference)
        expected_sequences = tuple(
            sequence.removesuffix("*") for sequence in case.target.sequences
        )
        if not isinstance(input_value, dict) or tuple(input_value.get("sequences", ())) != (
            expected_sequences
        ):
            raise ValueError("ESMFold receipt input sequences differ from the research case")
        expected_sequence_hashes = [
            sha256_bytes(sequence.encode("ascii")) for sequence in expected_sequences
        ]
        if value.get("sequence_identity_sha256") != expected_sequence_hashes:
            raise ValueError("ESMFold receipt sequence identity hashes are inconsistent")
        provenance_value = value.get("provenance")
        if not isinstance(provenance_value, dict):
            raise ValueError("ESMFold receipt is missing worker provenance")
        provenance = WorkerProvenance.from_dict(provenance_value)
        if provenance.model_revision != "esmfold_3B_v1":
            raise ValueError("ESMFold receipt names an unsupported model revision")
        metadata = self.artifacts.read_json(result_metadata)
        if not isinstance(metadata, dict) or metadata.get("schema_version") != "1.0" or (
            metadata.get("kind") != "protbind.esmfold-v1-result"
        ):
            raise ValueError("ESMFold result metadata contract is invalid")
        _require_exact_reference(metadata.get("input"), input_reference, "ESMFold input")
        _require_exact_reference(metadata.get("structure"), structure, "ESMFold structure")
        if (
            metadata.get("seed") != case.seed
            or metadata.get("model_revision") != provenance.model_revision
            or metadata.get("weight_sha256") != provenance.weight_sha256
            or metadata.get("code_sha256") != provenance.code_sha256
        ):
            raise ValueError("ESMFold result metadata differs from its receipt/provenance")
        runtime = metadata.get("runtime")
        if not isinstance(runtime, dict) or any(
            not isinstance(runtime.get(name), str)
            or len(runtime[name]) != 64
            or any(character not in "0123456789abcdef" for character in runtime[name])
            for name in ("environment_lock_sha256", "runtime_source_sha256")
        ):
            raise ValueError("ESMFold runtime attestation is incomplete")
        if (
            runtime.get("fair_esm_version") != structure.producer_version
            or result_metadata.producer_version != structure.producer_version
        ):
            raise ValueError("ESMFold runtime/output versions are inconsistent")
        output_qc = metadata.get("output_qc")
        if not isinstance(output_qc, dict) or (
            output_qc.get("sequence_identity_sha256") != expected_sequence_hashes
            or output_qc.get("coordinate_finite") is not True
            or output_qc.get("backbone_complete") is not True
            or output_qc.get("alternate_locations") is not False
        ):
            raise ValueError("ESMFold output QC receipt is incomplete")
        hardware_sha256 = value.get("hardware_sha256")
        if not isinstance(hardware_sha256, str) or len(hardware_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in hardware_sha256
        ):
            raise ValueError("ESMFold receipt hardware identity is invalid")
        return (
            structure,
            result_metadata,
            {
                "receipt_artifact": receipt_reference.to_dict(),
                "result_metadata_artifact": result_metadata.to_dict(),
                "input_artifact": input_reference.to_dict(),
                "seed": case.seed,
                "model_revision": provenance.model_revision,
                "weight_sha256": provenance.weight_sha256,
                "code_sha256": provenance.code_sha256,
                "hardware_sha256": hardware_sha256,
                "environment_lock_sha256": runtime["environment_lock_sha256"],
                "runtime_source_sha256": runtime["runtime_source_sha256"],
            },
        )

    def _audit_artifacts(self, manifest: RunManifest) -> None:
        self.artifacts.resolve(manifest.case_artifact)
        for artifact in manifest.input_artifacts.values():
            self.artifacts.resolve(artifact)
        for artifact in manifest.artifacts.values():
            self.artifacts.resolve(artifact)
        for record in manifest.stage_records.values():
            if record.cache_key != stage_cache_key(
                record.stage, record.input_hash, record.config_hash
            ):
                raise ValueError(f"stage cache key is corrupt: {record.stage.value}")
            for artifact in record.outputs:
                self.artifacts.resolve(artifact)
        if manifest.cofold_record is not None:
            record = manifest.cofold_record
            if record.cache_key != stage_cache_key(
                record.stage, record.input_hash, record.config_hash
            ):
                raise ValueError("cofold evidence cache key is corrupt")
            for artifact in record.outputs:
                self.artifacts.resolve(artifact)

    def _audit_configuration(
        self, manifest: RunManifest, case: ResearchCase
    ) -> None:
        """Reject changed inputs/config instead of silently reusing a completed stage."""

        if manifest.provenance.get("code_sha256") != _package_hash():
            raise ValueError(
                "run code hash differs from the current ProtBind package; create a new run"
            )
        hardware = self._hardware_evidence()
        if hardware.get("hsa_override_active"):
            raise ValueError("HSA_OVERRIDE_GFX_VERSION is active and forbidden")
        if manifest.provenance.get("hardware_sha256") != _stable_hardware_identity(
            hardware
        ):
            raise ValueError(
                "run hardware identity differs from the current host/runtime; create a new run"
            )

        records = manifest.stage_records
        input_record = records.get(RunState.INPUT_VALIDATED.value)
        if input_record is not None:
            expected_input = _artifact_input_hash(
                manifest.case_artifact,
                manifest.input_artifacts["target_resolution"],
                *(
                    (manifest.input_artifacts["target_raw_source"],)
                    if "target_raw_source" in manifest.input_artifacts
                    else ()
                ),
            )
            expected_config = sha256_bytes(canonical_json_bytes(_INPUT_VALIDATION_CONFIG))
            self._require_record_identity(input_record, expected_input, expected_config)
        receptor_record = records.get(RunState.RECEPTOR_READY.value)
        if receptor_record is not None and input_record is not None:
            receptor_value = self.artifacts.read_json(receptor_record.outputs[0])
            if not isinstance(receptor_value, dict):
                raise ValueError("RECEPTOR_READY output is not a JSON object")
            receptor = _artifact_reference(
                receptor_value.get("receptor"), "RECEPTOR_READY receptor"
            )
            self.artifacts.resolve(receptor)
            self._require_record_identity(
                receptor_record,
                _artifact_input_hash(
                    input_record.outputs[0],
                    receptor,
                    manifest.input_artifacts["target_resolution"],
                ),
                sha256_bytes(canonical_json_bytes(_RECEPTOR_READY_CONFIG)),
            )
        indexed_record = records.get(RunState.INDEXED.value)
        index_artifact = manifest.input_artifacts.get("library_index")
        if (
            indexed_record is not None
            and receptor_record is not None
            and index_artifact is not None
        ):
            index_config, _ = read_index_metadata(self.artifacts.resolve(index_artifact))
            expected_input = _artifact_input_hash(
                receptor_record.outputs[0], index_artifact
            )
            self._require_record_identity(
                indexed_record, expected_input, index_config.config_hash
            )
        screened_record = records.get(RunState.SCREENED.value)
        if screened_record is not None and index_artifact is not None:
            screen = self.artifacts.read_json(screened_record.outputs[0])
            query_values = screen.get("query_artifacts", {}) if isinstance(screen, dict) else {}
            if not isinstance(query_values, dict) or not query_values:
                raise ValueError("screen artifact has no query artifact identities")
            query_artifacts = {
                name: ArtifactRef.from_dict(value)
                for name, value in sorted(query_values.items())
            }
            for artifact in query_artifacts.values():
                self.artifacts.resolve(artifact)
            expected_input = _artifact_input_hash(
                index_artifact,
                *query_artifacts.values(),
                extra=case.mode.value,
            )
            expected_config = sha256_bytes(
                canonical_json_bytes(
                    {
                        "screen_top_k": self.config.screen_top_k,
                        "rrf_k": self.config.rrf_k,
                        "mode": case.mode.value,
                        "query_sha256": {
                            name: artifact.sha256
                            for name, artifact in query_artifacts.items()
                        },
                    }
                )
            )
            self._require_record_identity(
                screened_record, expected_input, expected_config
            )
        selected_record = records.get(RunState.SELECTED.value)
        if selected_record is not None and screened_record is not None:
            selection_value = self.artifacts.read_json(selected_record.outputs[0])
            automatic = (
                isinstance(selection_value, dict)
                and selection_value.get("automatic_orchestration") is not None
            )
            if automatic:
                preparation = manifest.artifacts.get(
                    _QUICK_SELECTION_KEYS["preparation"]
                )
                quick_input = manifest.artifacts.get(_QUICK_SELECTION_KEYS["input"])
                batch = manifest.artifacts.get(_QUICK_SELECTION_KEYS["batch"])
                receipt = manifest.artifacts.get(_QUICK_SELECTION_KEYS["receipt"])
                worker_config = self.config.workers.get(RunState.SELECTED)
                if any(
                    item is None
                    for item in (
                        preparation,
                        quick_input,
                        batch,
                        receipt,
                        worker_config,
                    )
                ):
                    raise ValueError(
                        "completed automatic SELECTED stage lacks its worker inputs/config"
                    )
                assert preparation is not None
                assert quick_input is not None
                assert batch is not None
                assert receipt is not None
                assert worker_config is not None
                if (
                    len(selected_record.outputs) != 2
                    or selected_record.outputs[1] != receipt
                    or manifest.artifacts.get("selection_bundle")
                    != selected_record.outputs[0]
                    or manifest.artifacts.get("support_selection_batch")
                    != selected_record.outputs[0]
                ):
                    raise ValueError(
                        "automatic SELECTED output/receipt manifest bindings differ"
                    )
                outputs = self._quick_selection_receipt_outputs(
                    receipt,
                    expected_job_id=(
                        f"{manifest.run_id}-selected-quick-vina"
                    ),
                    expected_case_id=case.case_id,
                    quick_input=quick_input,
                    batch=batch,
                    worker_config=worker_config,
                )
                evaluations = validate_quick_vina_batch(
                    self.artifacts,
                    preparation,
                    quick_input,
                    outputs,
                    case_id=case.case_id,
                    seed=case.seed,
                )
                rebuilt = finalize_selection_bundle(
                    self.artifacts,
                    preparation,
                    evaluations,
                    quick_vina_input=quick_input,
                    quick_vina_batch=batch,
                    worker_receipt=receipt,
                )
                if rebuilt != selected_record.outputs[0]:
                    raise ValueError(
                        "automatic SELECTED output is not reproducible from quick evidence"
                    )
                receptor = _artifact_reference(
                    selection_value.get("receptor"), "automatic selection receptor"
                )
                _, candidates = self._normalize_selection_batch(
                    manifest, case, selected_record.outputs[0]
                )
                if selection_value.get("candidates") != candidates:
                    raise ValueError(
                        "automatic SELECTED candidates differ after normalization"
                    )
                self._require_record_identity(
                    selected_record,
                    self._automatic_selection_input_hash(
                        screened_record.outputs[0],
                        preparation,
                        quick_input,
                        batch,
                        receptor,
                    ),
                    self._automatic_selection_config_hash(
                        worker_config, preparation, quick_input, batch
                    ),
                )
            else:
                source = manifest.artifacts.get("support_selection_batch") or (
                    manifest.artifacts.get("support_openfold_batch")
                )
                if source is None:
                    raise ValueError(
                        "completed SELECTED stage has no frozen selection batch"
                    )
                receptor, candidates = self._normalize_selection_batch(
                    manifest, case, source
                )
                if not isinstance(selection_value, dict) or selection_value.get(
                    "candidates"
                ) != candidates:
                    raise ValueError(
                        "SELECTED output differs from its frozen selection batch"
                    )
                self._require_record_identity(
                    selected_record,
                    _artifact_input_hash(
                        screened_record.outputs[0], source, receptor
                    ),
                    sha256_bytes(
                        canonical_json_bytes(
                            {
                                **_SELECTION_CONFIG,
                                "execution_mode": "manual-support",
                                "source_sha256": source.sha256,
                            }
                        )
                    ),
                )
        if manifest.cofold_record is not None:
            batch = manifest.artifacts.get("support_openfold_batch")
            if batch is None:
                raise ValueError("completed cofold evidence has no frozen input batch")
            validate_cofold_batch(
                self.artifacts,
                batch,
                case=case,
                screening_artifact=records[RunState.SCREENED.value].outputs[0],
                library_index=manifest.input_artifacts["library_index"],
                allowed_receptors=self._allowed_receptors(manifest, case),
                verify_chemistry=read_index_metadata(
                    self.artifacts.resolve(
                        manifest.input_artifacts["library_index"]
                    )
                )[1].chemistry_verified,
            )
            envelope = manifest.artifacts.get(_worker_input_key(RunState.COFOLDED))
            if envelope is None or self.artifacts.read_json(envelope) != (
                _worker_input_payload(manifest, RunState.COFOLDED)
            ):
                raise ValueError("cofold worker input envelope does not match selection")
            configured = self.config.workers.get(RunState.COFOLDED)
            expected_config = (
                configured.identity_hash
                if configured is not None
                else manifest.cofold_record.config_hash
            )
            self._require_record_identity(
                manifest.cofold_record,
                _artifact_input_hash(envelope),
                expected_config,
            )
        for stage in (RunState.DOCKED, RunState.VALIDATED):
            record = records.get(stage.value)
            if record is None:
                continue
            envelope = manifest.artifacts.get(_worker_input_key(stage))
            if envelope is None:
                raise ValueError(
                    f"completed stage {stage.value} has no worker input envelope"
                )
            expected_payload = _worker_input_payload(manifest, stage)
            if self.artifacts.read_json(envelope) != expected_payload:
                raise ValueError(
                    f"worker input envelope does not match dependencies: {stage.value}"
                )
            expected_input = _artifact_input_hash(envelope)
            configured = self.config.workers.get(stage)
            expected_config = (
                configured.identity_hash if configured is not None else record.config_hash
            )
            self._require_record_identity(record, expected_input, expected_config)
        report_record = records.get(RunState.REPORTED.value)
        if report_record is not None:
            screen_output = records[RunState.SCREENED.value].outputs[0]
            validation_output = records[RunState.VALIDATED.value].outputs[0]
            self._require_record_identity(
                report_record,
                _artifact_input_hash(
                    screen_output,
                    validation_output,
                    manifest.input_artifacts["target_resolution"],
                ),
                sha256_bytes(canonical_json_bytes(_REPORT_CONFIG)),
            )

    @staticmethod
    def _require_record_identity(
        record: StageRecord, expected_input: str, expected_config: str
    ) -> None:
        if record.input_hash != expected_input or record.config_hash != expected_config:
            raise ValueError(
                f"completed stage {record.stage.value} does not match current input/config; "
                "create a new run instead of reusing stale artifacts"
            )

    def run(
        self,
        manifest: RunManifest,
        *,
        stop_after: RunState = RunState.REPORTED,
    ) -> RunManifest:
        if stop_after not in _NORMAL_STAGES:
            raise ValueError("stop_after must be a normal workflow stage")
        if manifest.is_read_only:
            raise ValueError(
                "manifest schema 1.0 is read-only; create a new schema-2 run instead "
                "of resuming it"
            )
        if manifest.state is RunState.FAILED:
            raise ValueError("failed run cannot be resumed")
        if manifest.state is RunState.DEGRADED:
            manifest.prepare_resume()
        case = self.audit_manifest(manifest)
        target_position = _NORMAL_STAGES.index(stop_after)
        while manifest.next_stage is not None:
            stage = manifest.next_stage
            if stage is None or _NORMAL_STAGES.index(stage) > target_position:
                break
            try:
                self._execute_stage(manifest, case, stage)
            except PipelineStageError as exc:
                if exc.recoverable:
                    self._degrade(manifest, stage, exc.code, str(exc))
                else:
                    manifest.fail(stage=stage, code=exc.code, message=redact_text(str(exc)))
                self.manifests.save(manifest)
                return manifest
            except (ChemistryCapabilityError, StructureCapabilityError) as exc:
                self._degrade(manifest, stage, "CAPABILITY_UNAVAILABLE", str(exc))
                self.manifests.save(manifest)
                return manifest
            except (OSError, ValueError) as exc:
                manifest.fail(
                    stage=stage,
                    code=type(exc).__name__.upper(),
                    message=redact_text(str(exc)),
                )
                self.manifests.save(manifest)
                return manifest
            self.manifests.save(manifest)
            if stage is stop_after:
                break
        return manifest

    def _degrade(
        self, manifest: RunManifest, stage: RunState, code: str, message: str
    ) -> None:
        safe_message = redact_text(message)
        manifest.degrade(stage=stage, code=code, message=safe_message)
        report = (
            "# ProtBind degraded run\n\n"
            f"- Run: `{manifest.run_id}`\n"
            f"- Last completed stage: `{manifest.last_completed_stage.value}`\n"
            f"- Blocked stage: `{stage.value}`\n"
            f"- Reason code: `{code}`\n"
            f"- Detail: {safe_message}\n\n"
            "No missing scientific result was imputed. Resume after installing or "
            "configuring the stated capability.\n"
        )
        manifest.artifacts["degraded_report"] = self.artifacts.put_bytes(
            report.encode("utf-8"),
            media_type="text/markdown",
            producer="protbind.report.degraded",
            producer_version=__version__,
        )

    def _execute_stage(
        self, manifest: RunManifest, case: ResearchCase, stage: RunState
    ) -> None:
        if stage is RunState.INPUT_VALIDATED:
            self._validate_input(manifest, case)
        elif stage is RunState.RECEPTOR_READY:
            self._prepare_receptor(manifest, case)
        elif stage is RunState.INDEXED:
            self._validate_index(manifest)
        elif stage is RunState.SCREENED:
            self._screen(manifest, case)
        elif stage is RunState.SELECTED:
            self._select(manifest, case)
        elif stage is RunState.DOCKED:
            self._run_optional_cofold(manifest, case)
            self._run_worker_stage(manifest, case, stage)
        elif stage is RunState.VALIDATED:
            self._run_worker_stage(manifest, case, stage)
        elif stage is RunState.REPORTED:
            self._report(manifest, case)
        else:
            raise AssertionError(f"unhandled stage: {stage.value}")

    def _run_optional_cofold(
        self, manifest: RunManifest, case: ResearchCase
    ) -> None:
        """Run configured cofold evidence without making it a docking dependency."""

        if manifest.cofold_status is CofoldStatus.COMPLETED:
            return
        if RunState.COFOLDED not in self.config.workers:
            return
        if manifest.artifacts.get("support_openfold_batch") is None:
            manifest.mark_cofold_unavailable(
                code="INPUT_NOT_PREPARED",
                message=(
                    "optional OpenFold3 evidence was configured without a frozen "
                    "support_openfold_batch"
                ),
            )
            return
        manifest.begin_cofold()
        self.manifests.save(manifest)
        try:
            self._run_worker_stage(manifest, case, RunState.COFOLDED)
        except (
            ChemistryCapabilityError,
            StructureCapabilityError,
            PipelineStageError,
            OSError,
            ValueError,
        ) as exc:
            code = exc.code if isinstance(exc, PipelineStageError) else (
                type(exc).__name__.upper()
            )
            manifest.mark_cofold_failed(
                code=code,
                message=redact_text(str(exc)),
            )
            self.manifests.save(manifest)

    def _validate_resolution_receipt(
        self, manifest: RunManifest, case: ResearchCase
    ) -> tuple[dict[str, Any], ArtifactRef | None]:
        resolution_ref = manifest.input_artifacts["target_resolution"]
        if (
            resolution_ref.producer != "protbind.structure-resolver"
            or resolution_ref.producer_version != __version__
            or resolution_ref.media_type != "application/json"
        ):
            raise PipelineStageError(
                "STRUCTURE_RESOLUTION_INVALID",
                "target structure resolution artifact has an invalid producer/version/type",
                recoverable=False,
            )
        resolution = self.artifacts.read_json(resolution_ref)
        if not isinstance(resolution, dict) or resolution.get("schema_version") != (
            "1.0"
        ) or resolution.get("kind") != "protbind.structure-resolution":
            raise PipelineStageError(
                "STRUCTURE_RESOLUTION_INVALID",
                "target structure resolution receipt is missing or invalid",
                recoverable=False,
            )
        try:
            decision = ResolutionDecision(str(resolution.get("decision")))
        except ValueError as exc:
            raise PipelineStageError(
                "STRUCTURE_RESOLUTION_INVALID",
                "target structure resolution decision is unknown",
                recoverable=False,
            ) from exc
        folding_required = decision is ResolutionDecision.FOLDING_REQUIRED
        if resolution.get("folding_required") is not folding_required or (
            manifest.provenance.get("target_structure_resolution") != decision.value
        ):
            raise PipelineStageError(
                "STRUCTURE_RESOLUTION_INVALID",
                "resolution decision/folding flag differs from manifest provenance",
                recoverable=False,
            )
        expected_sequences = tuple(
            sequence.removesuffix("*") for sequence in case.target.sequences
        )
        expected_identity = {
            "chain_count": len(expected_sequences),
            "lengths": [len(sequence) for sequence in expected_sequences],
            "sha256": [
                sha256_bytes(sequence.encode("ascii"))
                for sequence in expected_sequences
            ],
            "ordering": "target_chain_order",
        }
        if resolution.get("sequence_identity") != expected_identity:
            raise PipelineStageError(
                "STRUCTURE_RESOLUTION_INVALID",
                "resolution receipt sequence identity differs from the case",
                recoverable=False,
            )
        network_requests = resolution.get("network_requests")
        if not isinstance(network_requests, list) or (
            decision
            in {ResolutionDecision.USER_SUPPLIED, ResolutionDecision.LOCAL_EXACT_CACHE}
            and network_requests
        ):
            raise PipelineStageError(
                "STRUCTURE_RESOLUTION_INVALID",
                "resolution receipt has inconsistent network request evidence",
                recoverable=False,
            )

        selected_value = resolution.get("selected_receptor_artifact")
        if folding_required:
            if case.target.structure is not None or selected_value is not None:
                raise PipelineStageError(
                    "STRUCTURE_RESOLUTION_INVALID",
                    "folding-required receipt cannot name a selected receptor",
                    recoverable=False,
                )
        else:
            if case.target.structure is None:
                raise PipelineStageError(
                    "STRUCTURE_RESOLUTION_INVALID",
                    "resolved structure receipt has no receptor in the frozen case",
                    recoverable=False,
                )
            try:
                selected = ArtifactRef.from_dict(selected_value)
                self.artifacts.resolve(selected)
            except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                raise PipelineStageError(
                    "STRUCTURE_RESOLUTION_INVALID",
                    "selected receptor artifact in the receipt is invalid",
                    recoverable=False,
                ) from exc
            if selected != case.target.structure:
                raise PipelineStageError(
                    "STRUCTURE_RESOLUTION_INVALID",
                    "selected receptor differs from the frozen research case",
                    recoverable=False,
                )
            if decision is ResolutionDecision.RCSB_IMPORTED and (
                selected.producer != "protbind.rcsb-import"
                or selected.license != "CC0-1.0"
            ):
                raise PipelineStageError(
                    "STRUCTURE_RESOLUTION_INVALID",
                    "RCSB-selected receptor lacks the importer/CC0 provenance",
                    recoverable=False,
                )
            selected_connection = inspect_declared_connections(
                self.artifacts.resolve(selected)
            )
            selected_receipt = resolution.get(
                "selected_connection_check", resolution.get("connection_check")
            )
            if selected_connection.to_dict() != selected_receipt or (
                selected_connection.covalent_detected
            ):
                raise PipelineStageError(
                    "STRUCTURE_SOURCE_PROVENANCE_INVALID",
                    "selected receptor connection evidence is invalid",
                    recoverable=False,
                )

        receipt_raw = resolution.get("raw_source_artifact")
        manifest_raw = manifest.input_artifacts.get("target_raw_source")
        if receipt_raw is None:
            if manifest_raw is not None or decision is ResolutionDecision.RCSB_IMPORTED:
                raise PipelineStageError(
                    "STRUCTURE_SOURCE_PROVENANCE_INVALID",
                    "manifest/receipt is missing or adds an unexpected raw target source",
                    recoverable=False,
                )
        elif not isinstance(receipt_raw, dict):
            raise PipelineStageError(
                "STRUCTURE_SOURCE_PROVENANCE_INVALID",
                "structure resolution raw source is not an ArtifactRef",
                recoverable=False,
            )
        else:
            try:
                receipt_raw_ref = ArtifactRef.from_dict(receipt_raw)
                self.artifacts.resolve(receipt_raw_ref)
            except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                raise PipelineStageError(
                    "STRUCTURE_SOURCE_PROVENANCE_INVALID",
                    "structure resolution raw source artifact is missing or invalid",
                    recoverable=False,
                ) from exc
            if manifest_raw != receipt_raw_ref or (
                receipt_raw_ref.producer != "protbind.rcsb-download"
                or receipt_raw_ref.license != "CC0-1.0"
            ):
                raise PipelineStageError(
                    "STRUCTURE_SOURCE_PROVENANCE_INVALID",
                    "raw target source differs from the receipt or lacks RCSB provenance",
                    recoverable=False,
                )
            raw_connection = inspect_declared_connections(
                self.artifacts.resolve(receipt_raw_ref)
            )
            if raw_connection.to_dict() != resolution.get("connection_check") or (
                raw_connection.covalent_detected
            ):
                raise PipelineStageError(
                    "STRUCTURE_SOURCE_PROVENANCE_INVALID",
                    "raw source connection evidence is invalid",
                    recoverable=False,
                )
        return resolution, manifest_raw

    def _validate_input(self, manifest: RunManifest, case: ResearchCase) -> None:
        started = time.perf_counter()
        resolution_ref = manifest.input_artifacts["target_resolution"]
        resolution, manifest_raw = self._validate_resolution_receipt(manifest, case)
        structure_summary: dict[str, Any] | None = None
        ligand_summary: dict[str, Any] | None = None
        if case.target.structure is not None:
            inspection = inspect_structure(self.artifacts.resolve(case.target.structure))
            structure_summary = asdict(inspection)
            case_sequences = tuple(
                sequence.removesuffix("*") for sequence in case.target.sequences
            )
            if case_sequences and inspection.sequences != case_sequences:
                raise PipelineStageError(
                    "TARGET_IDENTITY_MISMATCH",
                    "supplied protein sequences do not exactly match the standard-residue "
                    "sequences extracted from the target structure",
                    recoverable=False,
                )
            if inspection.missing_backbone_residues:
                raise PipelineStageError(
                    "STRUCTURE_REQUIRES_PREPARATION",
                    "target structure has standard residues missing N/CA/C backbone atoms: "
                    + ", ".join(inspection.missing_backbone_residues[:12]),
                    recoverable=True,
                )
            if inspection.alternate_location_atoms:
                raise PipelineStageError(
                    "STRUCTURE_REQUIRES_PREPARATION",
                    "target structure contains unresolved alternate-location atoms",
                    recoverable=True,
                )
            if inspection.metal_elements:
                raise PipelineStageError(
                    "UNSUPPORTED_METAL_CENTER",
                    "v1 does not run the ordinary non-covalent validation chain for "
                    f"metal-containing targets ({', '.join(inspection.metal_elements)})",
                    recoverable=False,
                )
        ligand_inspection = self._preflight_ligand_chemistry(case)
        if ligand_inspection is not None:
            ligand_summary = asdict(ligand_inspection)
        output = self.artifacts.put_json(
            {
                "schema_version": "1.0",
                "case_id": case.case_id,
                "mode": case.mode.value,
                "chain_count": len(case.target.sequences) or (
                    structure_summary["chain_count"] if structure_summary else None
                ),
                "residue_count": sum(len(item.replace("*", "")) for item in case.target.sequences)
                or (structure_summary["residue_count"] if structure_summary else None),
                "structure_summary": structure_summary,
                "structure_resolution": {
                    "decision": resolution.get("decision"),
                    "folding_required": resolution.get("folding_required"),
                    "receipt_artifact_id": resolution_ref.artifact_id,
                    "resolved_structure_artifact_id": (
                        case.target.structure.artifact_id
                        if case.target.structure is not None
                        else None
                    ),
                    "raw_source_artifact_id": (
                        manifest_raw.artifact_id if manifest_raw is not None else None
                    ),
                    "coordinate_file_policy": resolution.get(
                        "coordinate_file_policy"
                    ),
                    "assembly_id": resolution.get("assembly_id"),
                    "connection_check": resolution.get("connection_check"),
                    "selected_connection_check": resolution.get(
                        "selected_connection_check"
                    ),
                },
                "ligand_chemistry_summary": ligand_summary,
                "ligand_molecular_gate": (
                    "computed"
                    if ligand_summary is not None
                    else "not_applicable_to_pharmacophore_only_hypothesis"
                ),
                "privacy": {
                    "network_allowed": case.privacy.network_allowed,
                    "sequence_upload_allowed": case.privacy.sequence_upload_allowed,
                },
            },
            producer="protbind.input-validator",
            producer_version=__version__,
        )
        manifest.complete_stage(
            StageRecord.create(
                RunState.INPUT_VALIDATED,
                input_hash=_artifact_input_hash(
                    manifest.case_artifact,
                    resolution_ref,
                    *((manifest_raw,) if manifest_raw is not None else ()),
                ),
                config_hash=sha256_bytes(canonical_json_bytes(_INPUT_VALIDATION_CONFIG)),
                outputs=(output,),
                duration_seconds=time.perf_counter() - started,
            )
        )

    def _prepare_receptor(self, manifest: RunManifest, case: ResearchCase) -> None:
        """Freeze the exact receptor coordinates consumed by downstream stages."""

        started = time.perf_counter()
        receptor = case.target.structure
        source = "case_target_structure"
        if receptor is None:
            receptor = manifest.artifacts.get("support_esmfold_structure")
            source = "verified_esmfold_structure"
        if receptor is None:
            receptor = manifest.artifacts.get("support_receptor_structure")
            source = "explicit_support_receptor_structure"
        if receptor is None:
            raise PipelineStageError(
                "RECEPTOR_UNAVAILABLE",
                "no experimental/cached receptor or verified ESMFold-v1 receptor is "
                "available; attach one before continuing",
                recoverable=True,
            )
        self.artifacts.resolve(receptor)
        resolution = manifest.input_artifacts["target_resolution"]
        output = self.artifacts.put_json(
            {
                "schema_version": "2.0",
                "kind": "protbind.receptor-ready-bundle",
                "case_id": case.case_id,
                "receptor": receptor.to_dict(),
                "source": source,
                "resolution_receipt": resolution.to_dict(),
                "semantics": (
                    "content-addressed receptor coordinates; ligand pose prediction has "
                    "not occurred"
                ),
            },
            producer="protbind.receptor-ready",
            producer_version=__version__,
        )
        manifest.artifacts["receptor_ready_structure"] = receptor
        manifest.complete_stage(
            StageRecord.create(
                RunState.RECEPTOR_READY,
                input_hash=_artifact_input_hash(
                    manifest.stage_records[RunState.INPUT_VALIDATED.value].outputs[0],
                    receptor,
                    resolution,
                ),
                config_hash=sha256_bytes(
                    canonical_json_bytes(_RECEPTOR_READY_CONFIG)
                ),
                outputs=(output,),
                duration_seconds=time.perf_counter() - started,
            )
        )

    def _validate_index(self, manifest: RunManifest) -> None:
        started = time.perf_counter()
        index_artifact = manifest.input_artifacts["library_index"]
        index_path = self.artifacts.resolve(index_artifact)
        config, _ = read_index_metadata(index_path)
        identity = index_identity(index_path)
        output = self.artifacts.put_json(
            identity,
            producer="protbind.index-validator",
            producer_version=__version__,
        )
        manifest.complete_stage(
            StageRecord.create(
                RunState.INDEXED,
                input_hash=_artifact_input_hash(
                    manifest.stage_records[RunState.RECEPTOR_READY.value].outputs[0],
                    index_artifact,
                ),
                config_hash=config.config_hash,
                outputs=(index_artifact, output),
                duration_seconds=time.perf_counter() - started,
            )
        )

    @staticmethod
    def _selection_candidate_id(molecule_id: str, microstate_id: str) -> str:
        digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "2.0",
                    "molecule_id": molecule_id,
                    "microstate_id": microstate_id,
                }
            )
        )
        return f"selected-{digest[:20]}"

    def _normalize_selection_batch(
        self,
        manifest: RunManifest,
        case: ResearchCase,
        source_reference: ArtifactRef,
    ) -> tuple[ArtifactRef, list[dict[str, Any]]]:
        """Validate a selection receipt and return its receptor and top candidates."""

        value = self.artifacts.read_json(source_reference)
        if not isinstance(value, dict):
            raise ValueError("selection input must be a JSON object")
        screen = manifest.stage_records[RunState.SCREENED.value].outputs[0]
        library = manifest.input_artifacts["library_index"]
        receptor_ready = self.artifacts.read_json(
            manifest.stage_records[RunState.RECEPTOR_READY.value].outputs[0]
        )
        if not isinstance(receptor_ready, dict):
            raise ValueError("RECEPTOR_READY output is not a JSON object")
        receptor = _artifact_reference(
            receptor_ready.get("receptor"), "RECEPTOR_READY receptor"
        )
        self.artifacts.resolve(receptor)

        if value.get("schema_version") == "1.0" and value.get("kind") == (
            "protbind.cofold-input-batch"
        ):
            validated = validate_cofold_batch(
                self.artifacts,
                source_reference,
                case=case,
                screening_artifact=screen,
                library_index=library,
                allowed_receptors=(receptor,),
                verify_chemistry=read_index_metadata(
                    self.artifacts.resolve(library)
                )[1].chemistry_verified,
            )
            _require_exact_reference(
                validated.get("receptor"), receptor, "selection receptor"
            )
            microstates = {
                (str(item["molecule_id"]), str(item["microstate_id"])): item
                for item in validated["microstates"]
            }
            evaluated = sorted(
                validated["quick_vina"]["evaluated"],
                key=lambda item: (
                    float(item["score"]),
                    str(item["molecule_id"]),
                    str(item["microstate_id"]),
                ),
            )
            best: dict[str, dict[str, Any]] = {}
            for item in evaluated:
                best.setdefault(str(item["molecule_id"]), item)
            candidates: list[dict[str, Any]] = []
            for molecule_id in validated["quick_vina"]["retained_molecule_ids"]:
                quick = best[str(molecule_id)]
                microstate_id = str(quick["microstate_id"])
                microstate = microstates[(str(molecule_id), microstate_id)]
                candidates.append(
                    {
                        "candidate_id": self._selection_candidate_id(
                            str(molecule_id), microstate_id
                        ),
                        "molecule_id": str(molecule_id),
                        "microstate_id": microstate_id,
                        "canonical_isomeric_smiles": microstate[
                            "canonical_isomeric_smiles"
                        ],
                        "heavy_element_counts": microstate["heavy_element_counts"],
                        "receptor": receptor.to_dict(),
                        "quick_vina_score": float(quick["score"]),
                        "quick_vina_score_semantics": quick["score_semantics"],
                        "quick_vina_pose": quick["pose"],
                        "quick_vina_evidence": quick["evidence"],
                        "box_center": list(quick["box_center"]),
                        "box_size": list(quick["box_size"]),
                    }
                )
            return receptor, candidates

        if value.get("schema_version") not in {"1.0", "2.0"} or value.get(
            "kind"
        ) != "protbind.selection-bundle":
            raise ValueError(
                "selection input must be protbind.selection-bundle or the legacy "
                "validated cofold-input-batch"
            )
        _require_exact_reference(
            value.get("screening_artifact"), screen, "selection screening_artifact"
        )
        _require_exact_reference(
            value.get("library_index"), library, "selection library_index"
        )
        _require_exact_reference(value.get("receptor"), receptor, "selection receptor")
        raw_candidates = value.get("candidates", value.get("selected_candidates"))
        if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 16:
            raise ValueError("selection bundle must contain 1..16 candidates")
        candidates = []
        candidate_ids: set[str] = set()
        identities: set[tuple[str, str]] = set()
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                raise ValueError("selection candidate must be an object")
            candidate = dict(raw)
            molecule_id = candidate.get("molecule_id")
            microstate_id = candidate.get("microstate_id")
            candidate_id = candidate.get("candidate_id")
            if (
                not isinstance(molecule_id, str)
                or not molecule_id.strip()
                or not isinstance(microstate_id, str)
                or not microstate_id.strip()
                or not isinstance(candidate_id, str)
                or not candidate_id.strip()
            ):
                raise ValueError(
                    "selection candidate requires candidate_id, molecule_id, and "
                    "microstate_id"
                )
            if candidate_id in candidate_ids or (molecule_id, microstate_id) in identities:
                raise ValueError("selection candidate identities must be unique")
            candidate_ids.add(candidate_id)
            identities.add((molecule_id, microstate_id))
            smiles = candidate.get("canonical_isomeric_smiles")
            if not isinstance(smiles, str) or not smiles.strip():
                raise ValueError("selection candidate requires canonical isomeric SMILES")
            _finite_vector(candidate.get("box_center"), "selection box_center")
            _finite_vector(
                candidate.get("box_size"), "selection box_size", positive=True
            )
            candidate_receptor = candidate.get("receptor")
            if candidate_receptor is None:
                candidate["receptor"] = receptor.to_dict()
            else:
                _require_exact_reference(
                    candidate_receptor, receptor, "selection candidate receptor"
                )
            candidates.append(candidate)
        return receptor, candidates

    def _selection_receptor_and_chains(
        self,
        manifest: RunManifest,
        case: ResearchCase,
        worker_config: WorkerConfig,
    ) -> tuple[ArtifactRef, tuple[tuple[str, str], ...]]:
        receptor_ready = self.artifacts.read_json(
            manifest.stage_records[RunState.RECEPTOR_READY.value].outputs[0]
        )
        if not isinstance(receptor_ready, dict):
            raise PipelineStageError(
                "INPUT_NOT_PREPARED",
                "RECEPTOR_READY output is not a JSON object",
                recoverable=False,
            )
        receptor = _artifact_reference(
            receptor_ready.get("receptor"), "RECEPTOR_READY receptor"
        )
        receptor_path = self.artifacts.resolve(receptor)
        try:
            inspection = inspect_structure(receptor_path)
            chains = tuple(
                zip(inspection.chain_ids, inspection.sequences, strict=True)
            )
        except (OSError, ValueError, StructureCapabilityError) as exc:
            if worker_config.allow_unisolated_test_fixture and case.target.sequences:
                chains = tuple(
                    (chr(ord("A") + index), sequence.rstrip("*"))
                    for index, sequence in enumerate(case.target.sequences)
                )
            else:
                raise PipelineStageError(
                    "INPUT_NOT_PREPARED",
                    f"cannot derive exact receptor chains for selection: {exc}",
                    recoverable=False,
                ) from exc
        expected_sequences = tuple(
            sequence.rstrip("*") for sequence in case.target.sequences
        )
        if expected_sequences and tuple(sequence for _, sequence in chains) != (
            expected_sequences
        ):
            raise PipelineStageError(
                "INPUT_NOT_PREPARED",
                "selection receptor sequences differ from the frozen target",
                recoverable=False,
            )
        return receptor, chains

    @staticmethod
    def _explicit_selection_box(
        case: ResearchCase,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        pocket = case.pocket
        if pocket is None or pocket.center is None or pocket.box_size is None:
            raise PipelineStageError(
                "SITE_DISCOVERY_UNAVAILABLE",
                "automatic quick Vina requires an explicit pocket center and box_size; "
                "fpocket/P2Rank site discovery is not configured, and no whole-protein "
                "box was guessed",
                recoverable=True,
            )
        return pocket.center, pocket.box_size

    def _case_known_site_calibration(
        self,
        case: ResearchCase,
        receptor: ArtifactRef,
        candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        pocket = case.pocket
        if pocket is None or pocket.known_site_calibration_receipt is None:
            return None
        assert pocket.center is not None and pocket.box_size is not None
        assert pocket.site_provenance_kind is not None
        summary = known_site_calibration_summary(
            self.artifacts,
            calibration=pocket.known_site_calibration_receipt,
            target_id=pocket.known_site_calibration_target_id,
            receptor=receptor,
            source_kind=pocket.site_provenance_kind,
            center=[float(item) for item in pocket.center],
            size=[float(item) for item in pocket.box_size],
        )
        for candidate in candidates or []:
            _require_same_vector(
                candidate.get("box_center"),
                list(pocket.center),
                "selection box_center",
            )
            _require_same_vector(
                candidate.get("box_size"),
                list(pocket.box_size),
                "selection box_size",
            )
        return summary

    def _quick_selection_receipt_outputs(
        self,
        receipt: ArtifactRef,
        *,
        expected_job_id: str,
        expected_case_id: str,
        quick_input: ArtifactRef,
        batch: ArtifactRef,
        worker_config: WorkerConfig,
    ) -> tuple[ArtifactRef, ...]:
        value = self.artifacts.read_json(receipt)
        quick_value = self.artifacts.read_json(quick_input)
        outputs_value = value.get("output_artifacts") if isinstance(value, dict) else None
        timings = value.get("timings_seconds") if isinstance(value, dict) else None
        warnings = value.get("warnings") if isinstance(value, dict) else None
        elapsed = value.get("end_to_end_seconds") if isinstance(value, dict) else None
        expected_isolation = (
            "bwrap-unshare-net"
            if worker_config.isolate_network
            else "fixture-application-policy-only"
        )
        if (
            receipt.media_type != "application/json"
            or receipt.producer != "protbind.worker-receipt"
            or receipt.producer_version != __version__
            or receipt.source != quick_input.artifact_id
            or not isinstance(quick_value, dict)
            or quick_value.get("case_id") != expected_case_id
            or not isinstance(value, dict)
            or value.get("schema_version") != "1.0"
            or value.get("kind") != "protbind.quick-vina-worker-receipt"
            or value.get("job_id") != expected_job_id
            or value.get("case_id") != expected_case_id
            or value.get("engine") != worker_config.engine
            or value.get("worker_identity_hash") != worker_config.identity_hash
            or value.get("provenance") != worker_config.provenance.to_dict()
            or _artifact_reference(value.get("input"), "quick receipt input")
            != quick_input
            or value.get("output_contract")
            != "protbind.quick-vina-evaluation-batch"
            or value.get("network_isolation") != expected_isolation
            or value.get("gpu_lease_device") is not None
            or value.get("peak_vram_bytes") is not None
            or value.get("scientific_scope") != QUICK_VINA_PURPOSE
            or not isinstance(timings, dict)
            or any(
                not isinstance(name, str)
                or not isinstance(seconds, int | float)
                or isinstance(seconds, bool)
                or not math.isfinite(float(seconds))
                or float(seconds) < 0
                for name, seconds in timings.items()
            )
            or not isinstance(elapsed, int | float)
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
            or not isinstance(warnings, list)
            or any(not isinstance(item, str) for item in warnings)
            or not isinstance(outputs_value, list)
            or not outputs_value
        ):
            raise ValueError("cached quick Vina receipt differs from current configuration")
        outputs = tuple(
            _artifact_reference(item, "quick receipt output") for item in outputs_value
        )
        if outputs[0] != batch or len(set(outputs)) != len(outputs):
            raise ValueError("cached quick Vina output list is ambiguous or has another batch")
        for output in outputs:
            self.artifacts.resolve(output)
        return outputs

    @staticmethod
    def _automatic_selection_input_hash(
        screen: ArtifactRef,
        preparation: ArtifactRef,
        quick_input: ArtifactRef,
        batch: ArtifactRef,
        receptor: ArtifactRef,
    ) -> str:
        return _artifact_input_hash(
            screen, preparation, quick_input, batch, receptor
        )

    @staticmethod
    def _automatic_selection_config_hash(
        worker_config: WorkerConfig,
        preparation: ArtifactRef,
        quick_input: ArtifactRef,
        batch: ArtifactRef,
    ) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    **_SELECTION_CONFIG,
                    "execution_mode": "automatic-quick-vina",
                    "worker_identity_hash": worker_config.identity_hash,
                    "preparation_sha256": preparation.sha256,
                    "quick_input_sha256": quick_input.sha256,
                    "quick_batch_sha256": batch.sha256,
                }
            )
        )

    def _select_manual(
        self,
        manifest: RunManifest,
        case: ResearchCase,
        source: ArtifactRef,
        *,
        started: float,
    ) -> None:
        try:
            receptor, candidates = self._normalize_selection_batch(
                manifest, case, source
            )
            calibration = self._case_known_site_calibration(
                case, receptor, candidates
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise PipelineStageError(
                "INPUT_NOT_PREPARED",
                f"invalid selection batch: {exc}",
                recoverable=True,
            ) from exc
        screen = manifest.stage_records[RunState.SCREENED.value].outputs[0]
        output_payload = {
                "schema_version": "2.0",
                "kind": "protbind.selection-bundle",
                "case_id": case.case_id,
                "screening_artifact": screen.to_dict(),
                "library_index": manifest.input_artifacts["library_index"].to_dict(),
                "receptor": receptor.to_dict(),
                "source_batch": source.to_dict(),
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        if calibration is not None:
            assert case.pocket is not None
            assert case.pocket.known_site_calibration_receipt is not None
            for candidate in candidates:
                candidate["known_site_calibration"] = calibration
            output_payload["known_site_calibration_receipt"] = (
                case.pocket.known_site_calibration_receipt.to_dict()
            )
            output_payload["known_site_calibration"] = calibration
        output = self.artifacts.put_json(
            output_payload,
            producer="protbind.selection",
            producer_version=__version__,
        )
        manifest.artifacts["selection_bundle"] = output
        selection_config = {
            **_SELECTION_CONFIG,
            "execution_mode": "manual-support",
            "source_sha256": source.sha256,
        }
        selection_inputs = [screen, source, receptor]
        if case.pocket is not None and (
            case.pocket.known_site_calibration_receipt is not None
        ):
            selection_inputs.append(case.pocket.known_site_calibration_receipt)
        manifest.complete_stage(
            StageRecord.create(
                RunState.SELECTED,
                input_hash=_artifact_input_hash(*selection_inputs),
                config_hash=sha256_bytes(canonical_json_bytes(selection_config)),
                outputs=(output,),
                duration_seconds=time.perf_counter() - started,
            )
        )

    def _select_automatic_quick_vina(
        self,
        manifest: RunManifest,
        case: ResearchCase,
        *,
        started: float,
    ) -> None:
        worker_config = self.config.workers.get(RunState.SELECTED)
        if worker_config is None:
            raise PipelineStageError(
                "INPUT_NOT_PREPARED",
                "SELECTED has neither a frozen manual selection batch nor a configured "
                "vina-quick worker",
                recoverable=True,
            )
        try:
            worker_config.validate_launch_profile()
        except ValueError as exc:
            raise PipelineStageError(
                "RESOURCE_POLICY_VIOLATION", str(exc), recoverable=False
            ) from exc
        if not worker_config.isolate_network and not (
            worker_config.allow_unisolated_test_fixture
        ):
            raise PipelineStageError(
                "OFFLINE_POLICY_VIOLATION",
                "scientific quick-Vina workers require OS-level network isolation",
                recoverable=False,
            )
        _, index_stats = read_index_metadata(
            self.artifacts.resolve(manifest.input_artifacts["library_index"])
        )
        if not index_stats.chemistry_verified and not (
            worker_config.allow_unisolated_test_fixture
        ):
            raise PipelineStageError(
                "UNVERIFIED_CHEMISTRY_INDEX",
                "precomputed feature JSONL lacks RDKit standardization/identity proof; "
                "it cannot feed scientific quick docking",
                recoverable=False,
            )
        environment_lock = manifest.artifacts.get(
            "support_vina_environment_lock"
        )
        if environment_lock is None:
            raise PipelineStageError(
                "INPUT_NOT_PREPARED",
                "automatic quick Vina requires a frozen support_vina_environment_lock",
                recoverable=True,
            )
        self.artifacts.resolve(environment_lock)
        center, size = self._explicit_selection_box(case)
        assert case.pocket is not None
        site_kind = case.pocket.site_provenance_kind or (
            SiteProvenanceKind.USER_RESIDUES
            if case.pocket.residues
            else SiteProvenanceKind.USER_CENTER
        )
        receptor, protein_chains = self._selection_receptor_and_chains(
            manifest, case, worker_config
        )
        screen = manifest.stage_records[RunState.SCREENED.value].outputs[0]
        try:
            preparation = build_selection_preparation(
                self.artifacts,
                screening=screen,
                library_index=manifest.input_artifacts["library_index"],
                receptor=receptor,
                protein_chains=protein_chains,
                box_center=center,
                box_size=size,
                box_source=site_kind,
                site_derivation_evidence=case.pocket.site_evidence,
                known_site_calibration_receipt=(
                    case.pocket.known_site_calibration_receipt
                ),
                known_site_calibration_target_id=(
                    case.pocket.known_site_calibration_target_id
                ),
            )
            quick_input = build_quick_vina_input(
                self.artifacts,
                preparation,
                environment_lock,
                case_id=case.case_id,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise PipelineStageError(
                "INPUT_NOT_PREPARED",
                f"cannot prepare automatic quick Vina selection: {exc}",
                recoverable=False,
            ) from exc
        manifest.artifacts[_QUICK_SELECTION_KEYS["preparation"]] = preparation
        manifest.artifacts[_QUICK_SELECTION_KEYS["input"]] = quick_input
        self.manifests.save(manifest)

        cached_batch = manifest.artifacts.get(_QUICK_SELECTION_KEYS["batch"])
        cached_receipt = manifest.artifacts.get(_QUICK_SELECTION_KEYS["receipt"])
        if (cached_batch is None) != (cached_receipt is None):
            raise PipelineStageError(
                "WORKER_OUTPUT_INVALID",
                "automatic selection cache has only one of quick batch/receipt",
                recoverable=False,
            )
        warnings: tuple[str, ...]
        if cached_batch is not None and cached_receipt is not None:
            try:
                outputs = self._quick_selection_receipt_outputs(
                    cached_receipt,
                    expected_job_id=(
                        f"{manifest.run_id}-selected-quick-vina"
                    ),
                    expected_case_id=case.case_id,
                    quick_input=quick_input,
                    batch=cached_batch,
                    worker_config=worker_config,
                )
                receipt_value = self.artifacts.read_json(cached_receipt)
                warnings_value = receipt_value.get("warnings", [])
                if not isinstance(warnings_value, list):
                    raise ValueError("quick Vina receipt warnings are not an array")
                warnings = tuple(str(item) for item in warnings_value)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise PipelineStageError(
                    "WORKER_OUTPUT_INVALID",
                    f"cached automatic selection is invalid: {exc}",
                    recoverable=False,
                ) from exc
            batch = cached_batch
            receipt = cached_receipt
        else:
            request = WorkerRequest(
                job_id=f"{manifest.run_id}-selected-quick-vina",
                engine=worker_config.engine,
                input=quick_input,
                parameters=worker_config.parameters,
                seed=case.seed,
                provenance=worker_config.provenance,
            )
            worker = JsonSubprocessWorker(
                worker_config.argv,
                timeout_seconds=worker_config.timeout_seconds,
                environment=worker_config.environment,
                artifact_root=self.workspace,
                isolate_network=worker_config.isolate_network,
            )
            try:
                with _worker_resource_lease(
                    self.workspace, worker_config.engine, None
                ):
                    response, elapsed = worker.run(request)
            except WorkerExecutionError as exc:
                raise PipelineStageError(
                    "WORKER_CRASH", str(exc), recoverable=True
                ) from exc
            if response.error is not None:
                raise PipelineStageError(
                    response.error.code,
                    response.error.message,
                    recoverable=response.error.recoverable,
                )
            outputs = response.outputs
            try:
                validate_quick_vina_batch(
                    self.artifacts,
                    preparation,
                    quick_input,
                    outputs,
                    case_id=case.case_id,
                    seed=case.seed,
                )
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise PipelineStageError(
                    "WORKER_OUTPUT_INVALID",
                    f"quick Vina worker output is invalid: {exc}",
                    recoverable=False,
                ) from exc
            batch = outputs[0]
            warnings = response.warnings
            receipt = self.artifacts.put_json(
                {
                    "schema_version": "1.0",
                    "kind": "protbind.quick-vina-worker-receipt",
                    "job_id": response.job_id,
                    "case_id": case.case_id,
                    "engine": response.engine,
                    "input": quick_input.to_dict(),
                    "output_artifacts": [item.to_dict() for item in outputs],
                    "worker_identity_hash": worker_config.identity_hash,
                    "timings_seconds": response.timings_seconds,
                    "end_to_end_seconds": elapsed,
                    "peak_vram_bytes": response.peak_vram_bytes,
                    "warnings": list(response.warnings),
                    "provenance": response.provenance.to_dict()
                    if response.provenance is not None
                    else None,
                    "output_contract": "protbind.quick-vina-evaluation-batch",
                    "network_isolation": (
                        "bwrap-unshare-net"
                        if worker_config.isolate_network
                        else "fixture-application-policy-only"
                    ),
                    "gpu_lease_device": None,
                    "scientific_scope": "selection-pruning-only",
                },
                producer="protbind.worker-receipt",
                producer_version=__version__,
                source=quick_input.artifact_id,
            )
            manifest.artifacts[_QUICK_SELECTION_KEYS["batch"]] = batch
            manifest.artifacts[_QUICK_SELECTION_KEYS["receipt"]] = receipt
            self.manifests.save(manifest)

        try:
            evaluations = validate_quick_vina_batch(
                self.artifacts,
                preparation,
                quick_input,
                outputs,
                case_id=case.case_id,
                seed=case.seed,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise PipelineStageError(
                "WORKER_OUTPUT_INVALID",
                f"quick Vina worker output is invalid: {exc}",
                recoverable=False,
            ) from exc
        if not any(item.get("status") == "completed" for item in evaluations):
            raise PipelineStageError(
                "NO_SELECTABLE_CANDIDATES",
                "every planned quick Vina request failed explicitly; no candidate was selected",
                recoverable=True,
            )
        try:
            output = finalize_selection_bundle(
                self.artifacts,
                preparation,
                evaluations,
                quick_vina_input=quick_input,
                quick_vina_batch=batch,
                worker_receipt=receipt,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise PipelineStageError(
                "WORKER_OUTPUT_INVALID",
                f"cannot finalize automatic selection: {exc}",
                recoverable=False,
            ) from exc
        manifest.artifacts["selection_bundle"] = output
        manifest.artifacts["support_selection_batch"] = output
        manifest.complete_stage(
            StageRecord.create(
                RunState.SELECTED,
                input_hash=self._automatic_selection_input_hash(
                    screen, preparation, quick_input, batch, receptor
                ),
                config_hash=self._automatic_selection_config_hash(
                    worker_config, preparation, quick_input, batch
                ),
                outputs=(output, receipt),
                duration_seconds=time.perf_counter() - started,
                warnings=warnings,
            )
        )

    def _select(self, manifest: RunManifest, case: ResearchCase) -> None:
        started = time.perf_counter()
        source = manifest.artifacts.get("support_selection_batch")
        if source is None:
            source = manifest.artifacts.get("support_openfold_batch")
        if source is not None:
            self._select_manual(manifest, case, source, started=started)
            return
        self._select_automatic_quick_vina(manifest, case, started=started)

    def _queries(
        self,
        case: ResearchCase,
        *,
        receptor_structure: ArtifactRef | None = None,
    ) -> tuple[dict[str, tuple[FeaturePoint, ...]], dict[str, ArtifactRef]]:
        branches: dict[str, tuple[FeaturePoint, ...]] = {}
        artifacts: dict[str, ArtifactRef] = {}
        if case.mode in {ResearchMode.BOTH, ResearchMode.LIGAND_ONLY}:
            if case.ligand is None:
                raise AssertionError("validated ligand mode has no ligand")
            if case.ligand.pharmacophore is not None:
                ligand_ref = case.ligand.pharmacophore
                ligand_features = _parse_feature_artifact(self.artifacts, ligand_ref)
            elif case.ligand.smiles:
                ligand_features = smiles_pharmacophore(case.ligand.smiles, seed=case.seed)
                ligand_ref = _feature_artifact(
                    self.artifacts,
                    ligand_features,
                    producer="protbind.rdkit.reference-pharmacophore",
                    heuristic=False,
                )
            else:
                raise PipelineStageError(
                    "LIGAND_QUERY_UNAVAILABLE",
                    "ligand screening requires reference SMILES or a pharmacophore artifact",
                    recoverable=True,
                )
            branches["ligand"] = ligand_features
            artifacts["ligand"] = ligand_ref
        if case.mode in {ResearchMode.BOTH, ResearchMode.POCKET_ONLY}:
            if case.pocket is None:
                raise AssertionError("validated pocket mode has no pocket")
            if case.pocket.pharmacophore is not None:
                pocket_ref = case.pocket.pharmacophore
                pocket_features = _parse_feature_artifact(self.artifacts, pocket_ref)
            elif (receptor_structure or case.target.structure) is not None:
                receptor = receptor_structure or case.target.structure
                assert receptor is not None
                pocket_features = pocket_pharmacophore(
                    self.artifacts.resolve(receptor),
                    residues=case.pocket.residues,
                    center=case.pocket.center,
                    box_size=case.pocket.box_size,
                )
                pocket_ref = _feature_artifact(
                    self.artifacts,
                    pocket_features,
                    producer="protbind.gemmi.pocket-pharmacophore-heuristic",
                    heuristic=True,
                )
            else:
                raise PipelineStageError(
                    "POCKET_QUERY_UNAVAILABLE",
                    "pocket screening requires a pocket pharmacophore or target structure",
                    recoverable=True,
                )
            branches["pocket"] = pocket_features
            artifacts["pocket"] = pocket_ref
        return branches, artifacts

    def _screen(self, manifest: RunManifest, case: ResearchCase) -> None:
        started = time.perf_counter()
        index_artifact = manifest.input_artifacts["library_index"]
        index_path = self.artifacts.resolve(index_artifact)
        receptor = case.target.structure
        if receptor is None:
            receptor = manifest.artifacts.get("support_esmfold_structure") or (
                manifest.artifacts.get("support_receptor_structure")
            )
        queries, query_artifacts = self._queries(
            case, receptor_structure=receptor
        )
        branch_hits = {
            name: query_index(
                index_path,
                features,
                top_k=self.config.screen_top_k,
            )
            for name, features in sorted(queries.items())
        }
        if case.mode is ResearchMode.BOTH:
            fused = reciprocal_rank_fusion(
                branch_hits,
                rrf_k=self.config.rrf_k,
                top_k=self.config.screen_top_k,
            )
            ranking = [
                {
                    "molecule_id": hit.molecule_id,
                    "rank": hit.rank,
                    "rrf_score": hit.rrf_score,
                    "branch_ranks": hit.branch_ranks,
                    "branch_scores": hit.branch_scores,
                    "branches": {
                        name: _serialize_match(branch_hit)
                        for name, branch_hit in sorted(hit.branch_hits.items())
                    },
                }
                for hit in fused
            ]
            ranking_method = "equal-weight reciprocal-rank fusion"
        else:
            name = next(iter(branch_hits))
            ranking = [
                {
                    **_serialize_match(hit),
                    "rank": rank,
                    "branch_ranks": {name: rank},
                    "branch_scores": {name: hit.geometric_match_score},
                }
                for rank, hit in enumerate(branch_hits[name], start=1)
            ]
            ranking_method = "TriPharm deterministic geometric ordering"
        payload = {
            "schema_version": "1.0",
            "case_id": case.case_id,
            "mode": case.mode.value,
            "score_semantics": "geometric pharmacophore match; not binding affinity",
            "ranking_method": ranking_method,
            "query_artifacts": {
                name: artifact.to_dict() for name, artifact in sorted(query_artifacts.items())
            },
            "branch_rankings": {
                name: [hit.molecule_id for hit in hits]
                for name, hits in sorted(branch_hits.items())
            },
            "hits": ranking,
            "funnel": {
                "indexed": index_identity(index_path)["molecule_count"],
                "tripharm_retained": len(ranking),
                "planned_next": "scaffold diversity top 128, microstates, quick Vina",
            },
        }
        output = self.artifacts.put_json(
            payload,
            producer="protbind.tripharm.cpu",
            producer_version=__version__,
        )
        screen_config_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "screen_top_k": self.config.screen_top_k,
                    "rrf_k": self.config.rrf_k,
                    "mode": case.mode.value,
                    "query_sha256": {
                        name: artifact.sha256 for name, artifact in sorted(query_artifacts.items())
                    },
                }
            )
        )
        for name, artifact in query_artifacts.items():
            manifest.artifacts[f"query_{name}"] = artifact
        manifest.complete_stage(
            StageRecord.create(
                RunState.SCREENED,
                input_hash=_artifact_input_hash(
                    index_artifact,
                    *(artifact for _, artifact in sorted(query_artifacts.items())),
                    extra=case.mode.value,
                ),
                config_hash=screen_config_hash,
                outputs=(output,),
                duration_seconds=time.perf_counter() - started,
            )
        )

    def _ensure_validation_support(
        self,
        manifest: RunManifest,
        worker_config: WorkerConfig,
    ) -> WorkerProvenance:
        """Build missing validation inputs and bind a generated runtime exactly."""

        if worker_config.allow_unisolated_test_fixture:
            return worker_config.provenance
        docking_record = manifest.stage_records.get(RunState.DOCKED.value)
        if docking_record is None:
            raise PipelineStageError(
                "MISSING_DEPENDENCY",
                "validation input generation requires a completed DOCKED bundle",
                recoverable=False,
            )
        docking_bundle = docking_record.outputs[0]
        batch = manifest.artifacts.get("support_validation_batch")
        if batch is None:
            reference = manifest.artifacts.get("support_reference_pose")
            try:
                batch = build_validation_input_batch(
                    self.artifacts,
                    docking_bundle,
                    reference_pose=reference,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise PipelineStageError(
                    "VALIDATION_INPUT_INVALID",
                    f"cannot derive validation inputs from the exact DOCKED bundle: {exc}",
                    recoverable=False,
                ) from exc
            manifest.artifacts["support_validation_batch"] = batch

        toolchain = manifest.artifacts.get("support_validation_toolchain")
        if toolchain is not None:
            return worker_config.provenance
        try:
            binding = build_validation_toolchain(
                self.artifacts,
                repository_root=Path(__file__).resolve().parents[2],
            )
        except (OSError, TypeError, ValueError) as exc:
            raise PipelineStageError(
                "CAPABILITY_UNAVAILABLE",
                f"cannot attest the local validation toolchain: {exc}",
                recoverable=True,
            ) from exc
        if worker_config.provenance != binding.provenance:
            raise PipelineStageError(
                "VALIDATION_RUNTIME_MISMATCH",
                "configured validation worker provenance differs from the locally "
                "attested validation toolchain",
                recoverable=False,
            )
        manifest.artifacts["support_validation_toolchain"] = binding.artifact
        return binding.provenance

    def _run_worker_stage(
        self, manifest: RunManifest, case: ResearchCase, stage: RunState
    ) -> None:
        worker_config = self.config.workers.get(stage)
        if worker_config is None:
            required = {
                RunState.COFOLDED: "OpenFold3 plus diversity/quick-docking selection worker",
                RunState.DOCKED: "Meeko and AutoDock Vina evidence worker",
                RunState.VALIDATED: "PoseBusters/ProLIF/sPyRMSD/OpenMM validation worker",
            }[stage]
            raise PipelineStageError(
                "CAPABILITY_UNAVAILABLE",
                f"{required} is not configured; no scientific output was fabricated",
                recoverable=True,
            )
        try:
            worker_config.validate_launch_profile()
        except ValueError as exc:
            raise PipelineStageError(
                "RESOURCE_POLICY_VIOLATION",
                str(exc),
                recoverable=False,
            ) from exc
        if not worker_config.isolate_network and not (
            worker_config.allow_unisolated_test_fixture
        ):
            raise PipelineStageError(
                "OFFLINE_POLICY_VIOLATION",
                "scientific workers require OS-level network isolation",
                recoverable=False,
            )
        if stage is RunState.COFOLDED:
            if not worker_config.allow_unisolated_test_fixture and (
                worker_config.engine != OPENFOLD_ENGINE
            ):
                raise PipelineStageError(
                    "RESOURCE_POLICY_VIOLATION",
                    "production COFOLDED requires the pinned openfold3 engine",
                    recoverable=False,
                )
            _, index_stats = read_index_metadata(
                self.artifacts.resolve(manifest.input_artifacts["library_index"])
            )
            if not index_stats.chemistry_verified and not (
                worker_config.allow_unisolated_test_fixture
            ):
                raise PipelineStageError(
                    "UNVERIFIED_CHEMISTRY_INDEX",
                    "precomputed feature JSONL has no RDKit standardization/identity proof; "
                    "it may be used for screening fixtures but not scientific cofolding",
                    recoverable=False,
                )
        previous_stage = {
            RunState.COFOLDED: RunState.SELECTED,
            RunState.DOCKED: RunState.SELECTED,
            RunState.VALIDATED: RunState.DOCKED,
        }[stage]
        if previous_stage.value not in manifest.stage_records:
            raise PipelineStageError(
                "MISSING_DEPENDENCY",
                f"worker stage {stage.value} has no completed {previous_stage.value} input",
                recoverable=False,
            )
        request_provenance = worker_config.provenance
        if stage is RunState.VALIDATED:
            request_provenance = self._ensure_validation_support(
                manifest, worker_config
            )
        if stage is RunState.COFOLDED:
            batch = manifest.artifacts.get("support_openfold_batch")
            if batch is None:
                raise PipelineStageError(
                    "INPUT_NOT_PREPARED",
                    "COFOLDED requires support_openfold_batch with diversity, microstate, "
                    "quick-Vina, top-16, and top-8 evidence",
                    recoverable=True,
                )
            try:
                validate_cofold_batch(
                    self.artifacts,
                    batch,
                    case=case,
                    screening_artifact=manifest.stage_records[
                        RunState.SCREENED.value
                    ].outputs[0],
                    library_index=manifest.input_artifacts["library_index"],
                    allowed_receptors=self._allowed_receptors(manifest, case),
                    verify_chemistry=index_stats.chemistry_verified,
                )
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise PipelineStageError(
                    "INPUT_NOT_PREPARED",
                    f"invalid support_openfold_batch: {exc}",
                    recoverable=True,
                ) from exc
        input_artifact = self.artifacts.put_json(
            _worker_input_payload(manifest, stage),
            producer="protbind.worker-stage-input",
            producer_version=__version__,
        )
        manifest.artifacts[_worker_input_key(stage)] = input_artifact
        request = WorkerRequest(
            job_id=f"{manifest.run_id}-{stage.value.lower()}",
            engine=worker_config.engine,
            input=input_artifact,
            parameters=worker_config.parameters,
            seed=case.seed,
            provenance=request_provenance,
        )
        worker = JsonSubprocessWorker(
            worker_config.argv,
            timeout_seconds=worker_config.timeout_seconds,
            environment=worker_config.environment,
            artifact_root=self.workspace,
            isolate_network=worker_config.isolate_network,
        )
        lease_device = worker_config.environment.get("HIP_VISIBLE_DEVICES")
        if stage is RunState.VALIDATED:
            validation_batch_ref = manifest.artifacts.get("support_validation_batch")
            if validation_batch_ref is not None:
                validation_batch = self.artifacts.read_json(validation_batch_ref)
                prepared_candidates = (
                    validation_batch.get("candidates")
                    if isinstance(validation_batch, dict)
                    else None
                )
                hip_requested = isinstance(prepared_candidates, list) and any(
                    isinstance(candidate, dict)
                    and isinstance(candidate.get("openmm"), dict)
                    and candidate["openmm"].get("platform") == "HIP"
                    for candidate in prepared_candidates
                )
                if hip_requested and lease_device is None:
                    raise PipelineStageError(
                        "RESOURCE_POLICY_VIOLATION",
                        "OpenMM HIP validation requires one leased HIP_VISIBLE_DEVICES index",
                        recoverable=False,
                    )
        if lease_device is not None:
            detected_devices = self._hardware_evidence().get(
                "device_architectures", ()
            )
            if detected_devices and int(lease_device) >= len(detected_devices):
                raise PipelineStageError(
                    "RESOURCE_POLICY_VIOLATION",
                    f"worker GPU index {lease_device} is outside the "
                    f"{len(detected_devices)} detected devices",
                    recoverable=False,
                )
        try:
            with _worker_resource_lease(
                self.workspace, worker_config.engine, lease_device
            ):
                response, elapsed = worker.run(request)
        except WorkerExecutionError as exc:
            raise PipelineStageError(
                "WORKER_CRASH", str(exc), recoverable=True
            ) from exc
        if response.error is not None:
            raise PipelineStageError(
                response.error.code,
                response.error.message,
                recoverable=response.error.recoverable,
            )
        for output in response.outputs:
            self.artifacts.resolve(output)
        self._validate_worker_outputs(manifest, stage, response.outputs)
        receipt = self.artifacts.put_json(
            {
                "schema_version": "1.0",
                "job_id": response.job_id,
                "engine": response.engine,
                "input_artifact_id": input_artifact.artifact_id,
                "output_artifact_ids": [item.artifact_id for item in response.outputs],
                "timings_seconds": response.timings_seconds,
                "end_to_end_seconds": elapsed,
                "peak_vram_bytes": response.peak_vram_bytes,
                "warnings": list(response.warnings),
                "provenance": response.provenance.to_dict()
                if response.provenance is not None
                else None,
                "output_contract": _WORKER_BUNDLE_KIND[stage],
                "network_isolation": (
                    "bwrap-unshare-net"
                    if worker_config.isolate_network
                    else "application-offline-policy-only"
                ),
                "gpu_lease_device": lease_device,
                "openfold_global_lease": worker_config.engine == OPENFOLD_ENGINE,
            },
            producer="protbind.worker-receipt",
            producer_version=__version__,
        )
        record = StageRecord.create(
            stage,
            input_hash=_artifact_input_hash(input_artifact),
            config_hash=worker_config.identity_hash,
            outputs=response.outputs + (receipt,),
            duration_seconds=elapsed,
            warnings=response.warnings,
        )
        if stage is RunState.COFOLDED:
            manifest.complete_cofold(record)
            manifest.artifacts["cofold_evidence_bundle"] = response.outputs[0]
        else:
            manifest.complete_stage(record)

    @staticmethod
    def _worker_is_fixture_only(
        config: PipelineConfig, stage: RunState
    ) -> bool:
        worker = config.workers.get(stage)
        return bool(worker is not None and worker.allow_unisolated_test_fixture)

    @staticmethod
    def _candidate_identity(value: Any, name: str) -> str:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")
        candidate_id = value.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"{name} requires a candidate_id")
        return candidate_id

    def _require_fixture_stage_coverage(
        self,
        stage: RunState,
        value: dict[str, Any],
        upstream_candidates: list[Any],
    ) -> None:
        """Keep fixture workers simple while forbidding silent candidate loss."""

        upstream = [
            self._candidate_identity(item, "upstream candidate")
            for item in upstream_candidates
        ]
        if len(upstream) != len(set(upstream)):
            raise ValueError("upstream candidate_id values must be unique")
        candidates = value["candidates"]
        if stage is RunState.DOCKED:
            successes = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError("docking candidate must be an object")
                parent = candidate.get("parent_candidate_id")
                if not isinstance(parent, str) or not parent:
                    raise ValueError("docking candidate requires parent_candidate_id")
                successes.append(parent)
            failures_value = value.get("failures", [])
            if not isinstance(failures_value, list):
                raise ValueError("docking failures must be an array")
            failures = [
                self._candidate_identity(item, "docking failure")
                for item in failures_value
            ]
            covered = [*successes, *failures]
        else:
            covered = [
                self._candidate_identity(item, "validation candidate")
                for item in candidates
            ]
        if len(covered) != len(set(covered)) or set(covered) != set(upstream):
            raise ValueError(
                f"{stage.value} successes/failures must exactly cover upstream candidates"
            )

    def _validate_cofolded_production_contract(
        self,
        manifest: RunManifest,
        value: dict[str, Any],
        outputs: tuple[ArtifactRef, ...],
        upstream_candidates: list[Any],
    ) -> None:
        """Bind a production COFOLDED response to the pinned OpenFold runtime."""

        worker = self.config.workers.get(RunState.COFOLDED)
        if worker is None or worker.engine != OPENFOLD_ENGINE:
            raise ValueError("production COFOLDED requires the pinned openfold3 engine")
        primary = outputs[0]
        if (
            primary.producer != OPENFOLD_BUNDLE_PRODUCER
            or primary.producer_version != OPENFOLD_REVISION
            or _has_fixture_label(primary.producer)
        ):
            raise ValueError("production COFOLDED bundle producer is not official OpenFold3")
        if value.get("score_semantics") != (
            "model confidence only; not binding affinity"
        ):
            raise ValueError("OpenFold3 bundle has invalid score semantics")

        returned_values = outputs[1:]
        if len({item.sha256 for item in returned_values}) != len(returned_values):
            raise ValueError("COFOLDED returned artifacts must be content-unique")
        returned = set(returned_values)
        required_specs = {
            "query_manifest": (
                OPENFOLD_QUERY_MANIFEST_PRODUCER,
                "application/json",
            ),
            "runner": (OPENFOLD_RUNNER_PRODUCER, "application/yaml"),
            "run_metadata": (
                OPENFOLD_RUN_METADATA_PRODUCER,
                "application/json",
            ),
        }
        required: dict[str, ArtifactRef] = {}
        for name, (producer, media_type) in required_specs.items():
            reference = _require_returned_reference(
                value.get(name), returned, f"OpenFold3 {name}"
            )
            self.artifacts.resolve(reference)
            if (
                reference.producer != producer
                or reference.producer_version != OPENFOLD_REVISION
                or reference.media_type != media_type
            ):
                raise ValueError(f"OpenFold3 {name} producer/version/type is invalid")
            required[name] = reference

        query_manifest = self.artifacts.read_json(required["query_manifest"])
        metadata = self.artifacts.read_json(required["run_metadata"])
        if not isinstance(query_manifest, dict) or query_manifest.get(
            "schema_version"
        ) != "1.0":
            raise ValueError("OpenFold3 query manifest must satisfy schema v1.0")
        if not isinstance(metadata, dict) or metadata.get("schema_version") != "1.0":
            raise ValueError("OpenFold3 run metadata must satisfy schema v1.0")

        envelope = manifest.artifacts.get(_worker_input_key(RunState.COFOLDED))
        batch = manifest.artifacts.get("support_openfold_batch")
        checkpoint = manifest.artifacts.get("support_openfold_checkpoint")
        environment_lock = manifest.artifacts.get("support_openfold_environment_lock")
        if None in {envelope, batch, checkpoint, environment_lock}:
            raise ValueError("production COFOLDED is missing frozen support artifacts")
        assert envelope is not None
        assert batch is not None
        assert checkpoint is not None
        assert environment_lock is not None
        expected_bindings = {
            "stage_envelope": envelope,
            "input_batch": batch,
            "checkpoint": checkpoint,
            "environment_lock": environment_lock,
        }
        for name, expected in expected_bindings.items():
            _require_exact_reference(
                query_manifest.get(name), expected, f"query manifest {name}"
            )
            _require_exact_reference(metadata.get(name), expected, f"run metadata {name}")
        expected_provenance = worker.provenance.to_dict()
        if query_manifest.get("provenance") != expected_provenance or metadata.get(
            "provenance"
        ) != expected_provenance:
            raise ValueError("OpenFold3 outputs differ from configured provenance")
        if (
            worker.provenance.model_revision != OPENFOLD_REVISION
            or worker.provenance.weight_sha256 != checkpoint.sha256
        ):
            raise ValueError("OpenFold3 WorkerConfig provenance is not pinned to its checkpoint")

        checkpoint_name = worker.parameters.get(
            "checkpoint_name", "openfold3-p2-155k"
        )
        if checkpoint_name not in OFFICIAL_CHECKPOINT_SIZES or checkpoint.size_bytes != (
            OFFICIAL_CHECKPOINT_SIZES[checkpoint_name]
        ):
            raise ValueError("OpenFold3 checkpoint name/official byte size is inconsistent")
        samples = worker.parameters.get("num_diffusion_samples", 1)
        low_mem = worker.parameters.get("low_mem", True)
        triton = worker.parameters.get("use_triton_triangle_kernels", True)
        msa_server = worker.parameters.get("use_msa_server", False)
        minimum_free_vram_gib = worker.parameters.get("minimum_free_vram_gib", 28.0)
        if low_mem is not True or triton is not True or msa_server is not False:
            raise ValueError(
                "production OpenFold3 requires low_mem, ROCm Triton, and no MSA server"
            )
        if (
            not isinstance(samples, int)
            or isinstance(samples, bool)
            or samples != 1
            or not isinstance(minimum_free_vram_gib, int | float)
            or isinstance(minimum_free_vram_gib, bool)
            or not math.isfinite(float(minimum_free_vram_gib))
            or float(minimum_free_vram_gib) < 24.0
        ):
            raise ValueError("production OpenFold3 resource parameters are invalid")
        expected_metadata = {
            "openfold_revision": OPENFOLD_REVISION,
            "checkpoint_name": checkpoint_name,
            "seed": self.load_case(manifest).seed,
            "num_diffusion_samples": samples,
            "low_mem": True,
            "rocm_triton": True,
            "msa_server": False,
            "precision": "32-true",
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(f"OpenFold3 run metadata changed {name}")
        if not isinstance(metadata.get("templates"), bool) or query_manifest.get(
            "templates"
        ) != metadata["templates"]:
            raise ValueError("OpenFold3 template mode differs across output metadata")

        runtime = metadata.get("runtime_attestation")
        if runtime != query_manifest.get("runtime_attestation") or not isinstance(
            runtime, dict
        ):
            raise ValueError("OpenFold3 runtime attestation is missing or inconsistent")
        expected_runtime = {
            "distribution": "openfold3",
            "version": OPENFOLD_VERSION,
            "scm_tag": OPENFOLD_VERSION,
            "scm_distance": 0,
            "scm_node": OPENFOLD_SCM_NODE,
            "scm_dirty": False,
            "entry_point": "openfold3.run_openfold:cli",
            "package_source_sha256": OFFICIAL_RUNTIME_SHA256,
            "runtime_file_count": OFFICIAL_RUNTIME_FILE_COUNT,
            "official_release": True,
        }
        if any(runtime.get(name) != expected for name, expected in expected_runtime.items()):
            raise ValueError("OpenFold3 runtime is not the pinned official release")
        if not isinstance(runtime.get("torch_hip_version"), str) or not runtime.get(
            "torch_hip_version"
        ) or runtime.get("triton_distribution") not in {
            "triton-rocm",
            "pytorch-triton-rocm",
        }:
            raise ValueError("OpenFold3 runtime lacks attested ROCm PyTorch/Triton")

        resource = metadata.get("resource_policy")
        visible_device = worker.environment.get("HIP_VISIBLE_DEVICES")
        if not isinstance(resource, dict) or any(
            resource.get(name) != expected
            for name, expected in {
                "hip_visible_device": visible_device,
                "trainer_devices": 1,
                "concurrent_openfold_jobs": 1,
                "minimum_free_vram_gib": float(minimum_free_vram_gib),
            }.items()
        ):
            raise ValueError("OpenFold3 run metadata violates the resource policy")
        free_vram = resource.get("free_vram_bytes_before_run")
        total_vram = resource.get("total_vram_bytes")
        if (
            not isinstance(free_vram, int)
            or isinstance(free_vram, bool)
            or not isinstance(total_vram, int)
            or isinstance(total_vram, bool)
            or free_vram < int(float(minimum_free_vram_gib) * 1024**3)
            or free_vram > total_vram
        ):
            raise ValueError("OpenFold3 run metadata has invalid VRAM admission evidence")

        raw_values = query_manifest.get("raw_outputs")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError("OpenFold3 query manifest has no raw output inventory")
        raw_outputs = {
            _artifact_reference(item, "OpenFold3 raw output") for item in raw_values
        }
        if len(raw_outputs) != len(raw_values):
            raise ValueError("OpenFold3 raw output inventory contains duplicates")
        for reference in raw_outputs:
            self.artifacts.resolve(reference)
            expected_type = (
                "chemical/x-mmcif"
                if reference.producer == OPENFOLD_RUNTIME_ENGINE
                else "application/json"
            )
            if (
                reference.producer
                not in {
                    OPENFOLD_RUNTIME_ENGINE,
                    f"{OPENFOLD_RUNTIME_ENGINE}.sanitized-output",
                }
                or reference.producer_version != OPENFOLD_REVISION
                or reference.media_type != expected_type
                or not isinstance(reference.source, str)
                or not reference.source.startswith("local-output:")
            ):
                raise ValueError("OpenFold3 raw output has invalid provenance/type")
        effective = query_manifest.get("effective_config_artifacts")
        if not isinstance(effective, dict) or set(effective) != {
            "experiment_config",
            "inference_query_set",
            "model_config",
        }:
            raise ValueError("OpenFold3 effective config artifact inventory is incomplete")
        for name, raw_reference in effective.items():
            reference = _artifact_reference(raw_reference, f"OpenFold3 {name}")
            if reference not in raw_outputs:
                raise ValueError("OpenFold3 effective config is absent from raw outputs")

        expected_returned = raw_outputs | set(required.values())
        if returned != expected_returned:
            raise ValueError("COFOLDED response contains missing or unreferenced artifacts")
        runner_text = self.artifacts.read_bytes(required["runner"]).decode("utf-8")
        required_runner_fragments = (
            "use_msa_server: false",
            "use_triton_triangle_kernels: true",
            "precision: 32-true",
            "devices: 1",
            "- low_mem",
        )
        if "://" in runner_text or any(
            fragment not in runner_text for fragment in required_runner_fragments
        ):
            raise ValueError("OpenFold3 runner does not prove the offline low-memory profile")

        upstream_ids = {
            self._candidate_identity(item, "COFOLDED input candidate")
            for item in upstream_candidates
        }
        output_ids = {
            self._candidate_identity(item, "COFOLDED output candidate")
            for item in value["candidates"]
        }
        if len(output_ids) != len(value["candidates"]) or output_ids != upstream_ids:
            raise ValueError("COFOLDED must exactly cover its frozen top-8 candidates")
        query_ids = query_manifest.get("query_ids")
        manifest_candidate_ids = query_manifest.get("candidate_ids")
        if (
            not isinstance(query_ids, list)
            or len(query_ids) != len(set(query_ids))
            or len(query_ids) != len(output_ids)
            or not isinstance(manifest_candidate_ids, list)
            or len(manifest_candidate_ids) != len(set(manifest_candidate_ids))
            or set(manifest_candidate_ids) != output_ids
        ):
            raise ValueError("OpenFold3 query manifest candidate coverage is inconsistent")
        for candidate in value["candidates"]:
            if not isinstance(candidate, dict) or candidate.get(
                "engine"
            ) != OPENFOLD_RUNTIME_ENGINE:
                raise ValueError("production COFOLDED candidate is not official OpenFold3")
            sample_values = candidate.get("samples")
            if not isinstance(sample_values, list) or len(sample_values) != samples:
                raise ValueError("OpenFold3 candidate sample count differs from configuration")

    def _validate_docked_production_contract(
        self,
        manifest: RunManifest,
        value: dict[str, Any],
        outputs: tuple[ArtifactRef, ...],
        upstream_candidates: list[Any],
    ) -> None:
        worker = self.config.workers.get(RunState.DOCKED)
        if worker is None:
            raise ValueError("production DOCKED validation requires its WorkerConfig")
        if outputs[0].producer != "attested-local-autodock-vina.bundle" or value.get(
            "test_fixture", False
        ) is not False:
            raise ValueError("production DOCKED output names the wrong engine/fixture label")
        returned_values = outputs[1:]
        if len({item.sha256 for item in returned_values}) != len(returned_values):
            raise ValueError("DOCKED returned artifacts must be unique")
        returned = set(returned_values)
        selection_mode = RunState.SELECTED.value in manifest.stage_records
        upstream_stage = RunState.SELECTED if selection_mode else RunState.COFOLDED
        upstream_bundle = manifest.stage_records[upstream_stage.value].outputs[0]
        upstream_field = (
            "upstream_selection_bundle"
            if selection_mode
            else "upstream_cofold_bundle"
        )
        _require_exact_reference(
            value.get(upstream_field),
            upstream_bundle,
            f"DOCKED {upstream_field}",
        )
        score_semantics = value.get("score_semantics")
        if not isinstance(score_semantics, str) or (
            "not an experimental binding free energy" not in score_semantics.lower()
        ):
            raise ValueError("DOCKED bundle must preserve the Vina score disclaimer")

        upstream_by_candidate: dict[str, dict[str, Any]] = {}
        upstream_order: list[str] = []
        for item in upstream_candidates:
            candidate_id = self._candidate_identity(
                item, f"{upstream_stage.value} candidate"
            )
            if candidate_id in upstream_by_candidate:
                raise ValueError(
                    f"{upstream_stage.value} candidate_id values must be unique"
                )
            upstream_by_candidate[candidate_id] = item
            upstream_order.append(candidate_id)
        if value.get("upstream_candidate_ids") != upstream_order:
            raise ValueError(
                "DOCKED upstream_candidate_ids must preserve the complete upstream order"
            )

        if selection_mode:
            batch = self.artifacts.read_json(upstream_bundle)
        else:
            support_batch = manifest.artifacts.get("support_openfold_batch")
            if support_batch is None:
                raise ValueError("DOCKED has no frozen support_openfold_batch")
            batch = self.artifacts.read_json(support_batch)
        if not isinstance(batch, dict):
            raise ValueError("DOCKED upstream selection input must be a JSON object")
        receptor = _artifact_reference(batch.get("receptor"), "frozen receptor")
        self.artifacts.resolve(receptor)
        _require_exact_reference(value.get("receptor"), receptor, "DOCKED receptor")
        prepared_receptor = _require_returned_reference(
            value.get("prepared_receptor"), returned, "prepared receptor"
        )
        self.artifacts.resolve(prepared_receptor)
        receptor_preparation_outputs: set[ArtifactRef] = set()
        if selection_mode:
            receptor_preparation_input = _require_returned_reference(
                value.get("receptor_preparation_input"),
                returned,
                "receptor preparation input",
            )
            receptor_preparation_receipt = _require_returned_reference(
                value.get("receptor_preparation_receipt"),
                returned,
                "receptor preparation receipt",
            )
            self.artifacts.resolve(receptor_preparation_input)
            self.artifacts.resolve(receptor_preparation_receipt)
            receptor_preparation_outputs.update(
                {receptor_preparation_input, receptor_preparation_receipt}
            )

        quick_vina = batch.get("quick_vina")
        evaluated = (
            batch.get("candidates")
            if selection_mode
            else quick_vina.get("evaluated")
            if isinstance(quick_vina, dict)
            else None
        )
        if not isinstance(evaluated, list):
            raise ValueError("upstream selection has no frozen quick-Vina entries")
        frozen_boxes: dict[tuple[str, str], dict[str, Any]] = {}
        for item in evaluated:
            if not isinstance(item, dict):
                raise ValueError("frozen quick-Vina entry must be an object")
            key = (str(item.get("molecule_id")), str(item.get("microstate_id")))
            if key in frozen_boxes:
                raise ValueError("frozen quick-Vina identities must be unique")
            _finite_vector(item.get("box_center"), "frozen box_center")
            _finite_vector(item.get("box_size"), "frozen box_size", positive=True)
            frozen_boxes[key] = item

        candidates = value["candidates"]
        failures = value.get("failures")
        if not isinstance(failures, list):
            raise ValueError("production DOCKED output requires a failures array")
        for name, actual in (
            ("candidate_count", len(candidates)),
            ("failure_count", len(failures)),
        ):
            reported = value.get(name)
            if not isinstance(reported, int) or isinstance(reported, bool) or reported != actual:
                raise ValueError(f"DOCKED {name} does not match the bundle")

        expected_returned = {prepared_receptor, *receptor_preparation_outputs}
        covered: list[str] = []
        successful_parents: set[str] = set()
        expected_execution = {
            "cpu_threads": worker.parameters.get("cpu", 1),
            "exhaustiveness": worker.parameters.get("exhaustiveness", 32),
            "num_modes": worker.parameters.get("num_modes", 9),
            "energy_range": float(worker.parameters.get("energy_range", 3.0)),
        }
        envelope = manifest.artifacts.get(_worker_input_key(RunState.DOCKED))
        if envelope is None:
            raise ValueError("DOCKED has no frozen worker input envelope")
        runtime_assets_sha256 = worker.provenance.weight_sha256
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("DOCKED candidate must be an object")
            if candidate.get("engine") != "attested-local-autodock-vina":
                raise ValueError("production docking requires attested-local-autodock-vina")
            parent_id = candidate.get("parent_candidate_id")
            if not isinstance(parent_id, str) or parent_id not in upstream_by_candidate:
                raise ValueError("DOCKED candidate has an unknown parent_candidate_id")
            if parent_id in successful_parents:
                raise ValueError("DOCKED contains duplicate successful parents")
            successful_parents.add(parent_id)
            covered.append(parent_id)
            if candidate.get("candidate_id") != f"vina-{parent_id}":
                raise ValueError("DOCKED candidate_id is not derived from its frozen parent")
            upstream_item = upstream_by_candidate[parent_id]
            for name in ("molecule_id", "microstate_id"):
                if candidate.get(name) != upstream_item.get(name):
                    raise ValueError(f"DOCKED candidate has a mismatched {name}")
            expected_structure: ArtifactRef | None = None
            if not selection_mode:
                expected_structure = _artifact_reference(
                    upstream_item.get("structure"), "COFOLDED structure"
                )
                _require_exact_reference(
                    candidate.get("cofold_structure"),
                    expected_structure,
                    "DOCKED cofold_structure",
                )
            elif "cofold_structure" in candidate:
                raise ValueError(
                    "Vina-only DOCKED candidates cannot claim a required cofold structure"
                )
            _require_exact_reference(candidate.get("receptor"), receptor, "candidate receptor")
            _require_exact_reference(
                candidate.get("prepared_receptor"),
                prepared_receptor,
                "candidate prepared_receptor",
            )
            key = (str(candidate.get("molecule_id")), str(candidate.get("microstate_id")))
            if key not in frozen_boxes:
                raise ValueError("DOCKED candidate has no frozen quick-Vina box")
            _require_same_vector(
                candidate.get("box_center"), frozen_boxes[key].get("box_center"), "box_center"
            )
            _require_same_vector(
                candidate.get("box_size"), frozen_boxes[key].get("box_size"), "box_size"
            )
            pose = _require_returned_reference(candidate.get("pose"), returned, "Vina pose")
            prepared_ligand = _require_returned_reference(
                candidate.get("prepared_ligand"), returned, "prepared ligand"
            )
            all_modes = _require_returned_reference(
                candidate.get("all_modes"), returned, "all Vina modes"
            )
            evidence_reference = _require_returned_reference(
                candidate.get("evidence"), returned, "Vina evidence"
            )
            selection_pose_outputs: set[ArtifactRef] = set()
            if selection_mode:
                pose_sdf = _require_returned_reference(
                    candidate.get("pose_sdf"), returned, "docked pose SDF"
                )
                if pose_sdf != pose:
                    raise ValueError("canonical DOCKED pose must be the reconstructed SDF")
                pose_pdbqt = _require_returned_reference(
                    candidate.get("pose_pdbqt"), returned, "docked pose PDBQT"
                )
                all_modes_sdf = _require_returned_reference(
                    candidate.get("all_modes_sdf"), returned, "all Vina modes SDF"
                )
                all_modes_pdbqt = _require_returned_reference(
                    candidate.get("all_modes_pdbqt"),
                    returned,
                    "all Vina modes PDBQT",
                )
                if all_modes_pdbqt != all_modes:
                    raise ValueError("legacy all_modes must alias all_modes_pdbqt")
                extraction = _require_returned_reference(
                    candidate.get("pose_extraction_receipt"),
                    returned,
                    "pose extraction receipt",
                )
                selection_pose_outputs.update(
                    {pose_sdf, pose_pdbqt, all_modes_sdf, all_modes_pdbqt, extraction}
                )
            for reference in (
                pose,
                prepared_ligand,
                all_modes,
                evidence_reference,
                *selection_pose_outputs,
            ):
                self.artifacts.resolve(reference)
            evidence = self.artifacts.read_json(evidence_reference)
            if not isinstance(evidence, dict) or evidence.get("schema_version") != "1.0" or (
                evidence.get("kind") != "protbind.tool-evidence"
                or evidence.get("tool") != "vina"
            ):
                raise ValueError("Vina evidence does not satisfy the tool-evidence contract")
            candidate_id = self._candidate_identity(candidate, "DOCKED candidate")
            if (
                evidence.get("candidate_id") != candidate_id
                or evidence.get("parent_candidate_id") != parent_id
                or evidence.get("molecule_id") != candidate.get("molecule_id")
                or evidence.get("microstate_id") != candidate.get("microstate_id")
                or evidence.get("seed") != self.load_case(manifest).seed
            ):
                raise ValueError("Vina evidence does not preserve candidate lineage/seed")
            if _has_fixture_label(evidence_reference.producer):
                raise ValueError("production DOCKED evidence carries a fixture label")
            metrics = evidence.get("metrics")
            inputs = evidence.get("inputs")
            if not isinstance(metrics, dict) or not isinstance(inputs, dict):
                raise ValueError("Vina evidence requires metrics and inputs objects")
            score = candidate.get("vina_score")
            if candidate.get("vina_score_semantics") != score_semantics:
                raise ValueError("Vina candidate semantics differ from its bundle")
            if metrics.get("score") != score or metrics.get("score_semantics") != candidate.get(
                "vina_score_semantics"
            ):
                raise ValueError("Vina evidence score/semantics differ from the candidate")
            _require_same_vector(
                metrics.get("box_center"), candidate.get("box_center"), "box_center"
            )
            _require_same_vector(metrics.get("box_size"), candidate.get("box_size"), "box_size")
            for name, expected in expected_execution.items():
                if metrics.get("cpu" if name == "cpu_threads" else name) != expected:
                    raise ValueError(f"Vina evidence execution setting {name} changed")
            _require_exact_reference(inputs.get("stage_envelope"), envelope, "evidence envelope")
            if selection_mode:
                _require_exact_reference(
                    inputs.get("selection_bundle"),
                    upstream_bundle,
                    "evidence selection bundle",
                )
                if "cofold_bundle" in inputs or "cofold_structure" in inputs:
                    raise ValueError(
                        "Vina-only evidence cannot use cofold artifacts as dependencies"
                    )
            else:
                _require_exact_reference(
                    inputs.get("cofold_bundle"),
                    upstream_bundle,
                    "evidence cofold bundle",
                )
                assert expected_structure is not None
                _require_exact_reference(
                    inputs.get("cofold_structure"),
                    expected_structure,
                    "evidence cofold structure",
                )
            _require_exact_reference(inputs.get("receptor"), receptor, "evidence receptor")
            _require_exact_reference(
                inputs.get("prepared_receptor"), prepared_receptor, "evidence prepared receptor"
            )
            if selection_mode:
                _require_exact_reference(
                    inputs.get("receptor_preparation_input"),
                    receptor_preparation_input,
                    "evidence receptor preparation input",
                )
                _require_exact_reference(
                    inputs.get("receptor_preparation_receipt"),
                    receptor_preparation_receipt,
                    "evidence receptor preparation receipt",
                )
            _require_exact_reference(
                inputs.get("prepared_ligand"), prepared_ligand, "evidence prepared ligand"
            )
            _require_exact_reference(inputs.get("pose"), pose, "evidence Vina pose")
            _require_exact_reference(inputs.get("all_modes"), all_modes, "evidence all modes")
            ligand_sdf = _require_returned_reference(
                inputs.get("ligand_sdf"), returned, "evidence ligand SDF"
            )
            self.artifacts.resolve(ligand_sdf)
            if evidence.get("runtime_assets_sha256") != runtime_assets_sha256:
                raise ValueError("Vina evidence runtime assets differ from worker provenance")
            timings = evidence.get("timings_seconds")
            if not isinstance(timings, dict) or any(
                not isinstance(item, int | float)
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                or float(item) < 0
                for item in timings.values()
            ):
                raise ValueError("Vina evidence timings must be finite and non-negative")
            expected_returned.update(
                {pose, prepared_ligand, all_modes, evidence_reference, ligand_sdf}
            )
            expected_returned.update(selection_pose_outputs)

        failed_ids: set[str] = set()
        for failure in failures:
            candidate_id = self._candidate_identity(failure, "DOCKED failure")
            if failure.get("engine") != "attested-local-autodock-vina":
                raise ValueError("production docking failure names an unattested engine")
            if candidate_id not in upstream_by_candidate or candidate_id in failed_ids:
                raise ValueError("DOCKED failure has an unknown or duplicate candidate_id")
            failed_ids.add(candidate_id)
            covered.append(candidate_id)
            upstream_item = upstream_by_candidate[candidate_id]
            if failure.get("parent_candidate_id") != candidate_id:
                raise ValueError("DOCKED failure does not preserve its parent candidate")
            if failure.get("stage") != "ligand_preparation_or_vina":
                raise ValueError("DOCKED failure has an unknown processing stage")
            for name in ("molecule_id", "microstate_id"):
                if failure.get(name) != upstream_item.get(name):
                    raise ValueError(f"DOCKED failure has a mismatched {name}")
            if selection_mode:
                if "cofold_structure" in failure:
                    raise ValueError(
                        "Vina-only docking failure cannot claim a cofold structure"
                    )
            else:
                _require_exact_reference(
                    failure.get("cofold_structure"),
                    _artifact_reference(
                        upstream_item.get("structure"), "COFOLDED structure"
                    ),
                    "failure cofold_structure",
                )
            _require_exact_reference(failure.get("receptor"), receptor, "failure receptor")
            if failure.get("seed") != self.load_case(manifest).seed:
                raise ValueError("DOCKED failure seed differs from the case")
            key = (str(failure.get("molecule_id")), str(failure.get("microstate_id")))
            if key not in frozen_boxes:
                raise ValueError("DOCKED failure has no frozen quick-Vina box")
            _require_same_vector(
                failure.get("box_center"), frozen_boxes[key].get("box_center"), "box_center"
            )
            _require_same_vector(
                failure.get("box_size"), frozen_boxes[key].get("box_size"), "box_size"
            )
            error = failure.get("error")
            if not isinstance(error, dict) or not isinstance(error.get("code"), str) or (
                not isinstance(error.get("message"), str)
                or not isinstance(error.get("recoverable"), bool)
            ):
                raise ValueError("DOCKED failure requires an explicit structured error")
            forbidden_failure_outputs = {
                "pose",
                "evidence",
                "vina_score",
                "score",
                "prepared_ligand",
                "all_modes",
            }
            present_forbidden = sorted(forbidden_failure_outputs & set(failure))
            if present_forbidden:
                raise ValueError(
                    "failed docking candidates cannot carry pose/evidence/score outputs: "
                    + ", ".join(present_forbidden)
                )
        if len(covered) != len(set(covered)) or set(covered) != set(upstream_order):
            raise ValueError(
                "DOCKED successes and failures must exactly partition selected candidates"
            )

        metadata_reference = _require_returned_reference(
            value.get("run_metadata"), returned, "Vina run metadata"
        )
        self.artifacts.resolve(metadata_reference)
        metadata = self.artifacts.read_json(metadata_reference)
        if not isinstance(metadata, dict) or metadata.get("schema_version") != "1.0":
            raise ValueError("Vina run metadata must satisfy schema v1.0")
        lock = manifest.artifacts.get("support_vina_environment_lock")
        if lock is None:
            raise ValueError("production DOCKED has no frozen Vina environment lock")
        _require_exact_reference(metadata.get("environment_lock"), lock, "Vina environment lock")
        toolchain = metadata.get("toolchain")
        execution = metadata.get("execution")
        if not isinstance(toolchain, dict) or not isinstance(execution, dict):
            raise ValueError("Vina run metadata requires toolchain and execution objects")
        if toolchain.get("runtime_assets_sha256") != runtime_assets_sha256:
            raise ValueError("Vina run metadata runtime assets differ from provenance")
        if toolchain.get("official_runtime") is not False or toolchain.get(
            "trust_level"
        ) != "hash-attested-local-without-reviewed-upstream-allowlist":
            raise ValueError("production Vina runtime attestation has unknown trust semantics")
        expected_metadata = {
            "device": "cpu",
            "cpu_threads": expected_execution["cpu_threads"],
            "seed": self.load_case(manifest).seed,
            "scoring": "vina",
            "exhaustiveness": expected_execution["exhaustiveness"],
            "num_modes": expected_execution["num_modes"],
            "energy_range": expected_execution["energy_range"],
            "input_candidate_count": len(upstream_order),
            "successful_candidate_count": len(candidates),
            "failed_candidate_count": len(failures),
        }
        for name, expected in expected_metadata.items():
            if execution.get(name) != expected:
                raise ValueError(f"Vina run metadata execution setting {name} changed")
        vina_runtime = toolchain.get("vina")
        if not isinstance(vina_runtime, dict) or not isinstance(
            vina_runtime.get("version"), str
        ):
            raise ValueError("Vina runtime attestation has no exact version")
        if metadata_reference.producer_version != vina_runtime["version"] or (
            outputs[0].producer_version != vina_runtime["version"]
        ):
            raise ValueError("Vina bundle/run metadata producer_version differs from runtime")
        for candidate in candidates:
            evidence_reference = _artifact_reference(candidate.get("evidence"), "Vina evidence")
            if evidence_reference.producer_version != vina_runtime["version"]:
                raise ValueError("Vina evidence producer_version differs from runtime")
        expected_returned.add(metadata_reference)
        if returned != expected_returned:
            raise ValueError("DOCKED response contains missing or unreferenced artifacts")

    def _validate_validation_production_contract(
        self,
        manifest: RunManifest,
        value: dict[str, Any],
        outputs: tuple[ArtifactRef, ...],
        upstream_candidates: list[Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        worker = self.config.workers.get(RunState.VALIDATED)
        if worker is None:
            raise ValueError("production VALIDATED validation requires its WorkerConfig")
        if outputs[0].producer != worker.engine or value.get("test_fixture") is not False:
            raise ValueError("production VALIDATED output carries a fixture label")
        returned_values = outputs[1:]
        if len({item.sha256 for item in returned_values}) != len(returned_values):
            raise ValueError("VALIDATED returned evidence artifacts must be unique")
        returned = set(returned_values)
        upstream_by_candidate: dict[str, dict[str, Any]] = {}
        for item in upstream_candidates:
            candidate_id = self._candidate_identity(item, "DOCKED candidate")
            if candidate_id in upstream_by_candidate:
                raise ValueError("DOCKED candidate_id values must be unique")
            upstream_by_candidate[candidate_id] = item
        output_ids = [
            self._candidate_identity(item, "validation candidate")
            for item in value["candidates"]
        ]
        if len(output_ids) != len(set(output_ids)) or set(output_ids) != set(
            upstream_by_candidate
        ):
            raise ValueError("VALIDATED must exactly cover all successful DOCKED candidates")

        batch_reference = manifest.artifacts.get("support_validation_batch")
        toolchain_reference = manifest.artifacts.get("support_validation_toolchain")
        if batch_reference is None or toolchain_reference is None:
            raise ValueError("production VALIDATED requires frozen batch and toolchain support")
        batch = self.artifacts.read_json(batch_reference)
        toolchain = self.artifacts.read_json(toolchain_reference)
        if not isinstance(batch, dict) or batch.get("schema_version") not in {
            "1.0",
            "2.0",
        } or (
            batch.get("kind") != "protbind.validation-input-batch"
        ):
            raise ValueError("support_validation_batch has an invalid contract")
        docking_bundle = manifest.stage_records[RunState.DOCKED.value].outputs[0]
        _require_exact_reference(
            batch.get("docking_bundle"), docking_bundle, "validation batch docking_bundle"
        )
        if not isinstance(toolchain, dict) or toolchain.get("schema_version") != "1.0" or (
            toolchain.get("kind") != "protbind.validation-toolchain-manifest"
        ):
            raise ValueError("support_validation_toolchain has an invalid contract")
        if toolchain.get("test_fixture", False) is not False:
            raise ValueError("production validation toolchain carries a fixture label")
        configured_pb = toolchain.get(
            "posebusters_configs", toolchain.get("posebusters_config")
        )
        if isinstance(configured_pb, str):
            configured_pb = [configured_pb]
        if (
            not isinstance(configured_pb, list)
            or not configured_pb
            or any(item not in {"dock", "redock"} for item in configured_pb)
        ):
            raise ValueError(
                "production validation toolchain must pin PoseBusters dock/redock mode"
            )
        _require_exact_reference(
            value.get("toolchain"), toolchain_reference, "validation bundle toolchain"
        )
        pins = toolchain.get("tools")
        if not isinstance(pins, dict) or "posebusters" not in pins:
            raise ValueError("validation toolchain must pin PoseBusters")
        allowed_validation_tools = {"posebusters", "spyrmsd", "prolif", "openmm"}
        if set(pins) - allowed_validation_tools:
            raise ValueError("validation toolchain names an unsupported tool")
        normalized_pins: dict[str, dict[str, Any]] = {}
        for tool, pin in pins.items():
            if not isinstance(tool, str) or not isinstance(pin, dict):
                raise ValueError("validation toolchain pins must be objects")
            version = pin.get("version")
            source_sha = pin.get("package_source_sha256")
            if not isinstance(version, str) or not version.strip() or not isinstance(
                source_sha, str
            ) or len(source_sha) != 64 or any(
                character not in "0123456789abcdef" for character in source_sha
            ):
                raise ValueError(f"validation toolchain has an invalid {tool} pin")
            normalized_pins[tool] = pin

        prepared_values = batch.get("candidates")
        if not isinstance(prepared_values, list):
            raise ValueError("validation batch candidates must be an array")
        prepared_by_candidate: dict[str, dict[str, Any]] = {}
        for prepared in prepared_values:
            candidate_id = self._candidate_identity(prepared, "prepared validation candidate")
            if candidate_id in prepared_by_candidate:
                raise ValueError("validation batch candidate_id values must be unique")
            if candidate_id not in upstream_by_candidate:
                raise ValueError("validation batch contains a candidate absent from DOCKED")
            upstream = upstream_by_candidate[candidate_id]
            for name in ("molecule_id", "microstate_id"):
                if prepared.get(name) != upstream.get(name):
                    raise ValueError(f"validation batch has a mismatched {name}")
            _require_exact_reference(
                prepared.get("docked_pose"),
                _artifact_reference(upstream.get("pose"), "DOCKED pose"),
                "prepared docked_pose",
            )
            upstream_cofold = upstream.get("cofold_structure")
            if upstream_cofold is not None:
                _require_exact_reference(
                    prepared.get("cofold_pose"),
                    _artifact_reference(upstream_cofold, "DOCKED cofold pose"),
                    "prepared cofold_pose",
                )
            elif prepared.get("cofold_pose") is not None:
                raise ValueError(
                    "Vina-only validation input cannot invent a cofold pose"
                )
            for tool_name in ("posebusters", "spyrmsd", "prolif", "openmm"):
                prepared_tool = prepared.get(tool_name)
                if prepared_tool is None:
                    continue
                if not isinstance(prepared_tool, dict):
                    raise ValueError(f"prepared {tool_name} inputs must be an object")
                for input_name, raw_reference in prepared_tool.items():
                    if tool_name == "openmm" and input_name == "platform":
                        if raw_reference not in {"CPU", "HIP"}:
                            raise ValueError("prepared OpenMM platform must be CPU or HIP")
                        continue
                    reference = _artifact_reference(
                        raw_reference, f"prepared {tool_name} {input_name}"
                    )
                    self.artifacts.resolve(reference)
            prepared_by_candidate[candidate_id] = prepared
        if set(prepared_by_candidate) != set(upstream_by_candidate):
            raise ValueError("validation batch must exactly cover successful docking candidates")

        expected_evidence: set[ArtifactRef] = set()
        for candidate in value["candidates"]:
            if not isinstance(candidate, dict):
                raise ValueError("validation candidate must be an object")
            if candidate.get("engine") != worker.engine or _has_fixture_label(
                candidate.get("engine")
            ):
                raise ValueError("production validation candidate names the wrong engine")
            raw_bundle = candidate.get("bundle")
            if not isinstance(raw_bundle, dict) or raw_bundle.get(
                "preparation_attested"
            ) is not False:
                raise ValueError(
                    "production validation must cap self-certified preparation evidence"
                )
            bundle = _validation_bundle(raw_bundle)
            expected_evidence.update(bundle.evidence)
        if returned != expected_evidence:
            raise ValueError("VALIDATED response contains missing or unreferenced evidence")
        return prepared_by_candidate, normalized_pins

    def _bind_prepared_validation_inputs(
        self,
        prepared: dict[str, Any],
        evidence_inputs: dict[str, dict[str, Any]],
        evidence_metrics: dict[str, dict[str, Any]],
        *,
        docked_pose: ArtifactRef,
        cofold_pose: ArtifactRef | None,
        has_reference_pose: bool,
        reference_pose: ArtifactRef | None,
    ) -> None:
        """Bind every evidence artifact to the frozen per-tool preparation batch."""

        for tool in evidence_inputs:
            if tool not in {"posebusters", "spyrmsd", "prolif", "openmm"}:
                # Fixture-era Vina/OpenFold boolean evidence is forbidden by the
                # production toolchain gate before this helper is reached.
                raise ValueError(f"production validation emitted unsupported {tool} evidence")
            prepared_tool = prepared.get(tool)
            if not isinstance(prepared_tool, dict):
                raise ValueError(f"{tool} evidence has no frozen prepared inputs")
            for name, raw_reference in prepared_tool.items():
                if tool == "openmm" and name == "platform":
                    platform = evidence_metrics[tool].get(
                        "platform", evidence_metrics[tool].get("requested_platform")
                    )
                    if platform is not None and platform != raw_reference:
                        raise ValueError("OpenMM evidence used a different platform")
                    continue
                prepared_reference = _artifact_reference(
                    raw_reference, f"prepared {tool} {name}"
                )
                evidence_reference = _evidence_input(evidence_inputs, tool, name)
                self.artifacts.resolve(evidence_reference)
                if evidence_reference != prepared_reference:
                    raise ValueError(
                        f"{tool} evidence input {name} differs from its frozen preparation"
                    )

        if "posebusters" not in evidence_inputs:
            raise ValueError("production validation requires PoseBusters evidence")
        if "spyrmsd" in evidence_inputs:
            if not has_reference_pose or reference_pose is None:
                raise ValueError("sPyRMSD evidence requires an authorized reference pose")
            prepared_reference_pose = _artifact_reference(
                prepared.get("reference_pose"), "prepared reference_pose"
            )
            if prepared_reference_pose != reference_pose:
                raise ValueError("sPyRMSD reference differs from the frozen batch")
            rmsd_reference = _evidence_input(
                evidence_inputs, "spyrmsd", "reference_pose"
            )
            rmsd_prediction = _evidence_input(
                evidence_inputs, "spyrmsd", "predicted_pose"
            )
            if rmsd_reference != reference_pose or rmsd_prediction != docked_pose:
                raise ValueError("sPyRMSD evidence is bound to the wrong poses")
        if "prolif" in evidence_inputs:
            prolif_inputs = evidence_inputs["prolif"]
            if "pose_a" in prolif_inputs or "pose_b" in prolif_inputs:
                pose_a = _evidence_input(evidence_inputs, "prolif", "pose_a")
                pose_b = _evidence_input(evidence_inputs, "prolif", "pose_b")
                if cofold_pose is not None:
                    if {pose_a, pose_b} != {docked_pose, cofold_pose}:
                        raise ValueError("ProLIF evidence is bound to the wrong pose pair")
                elif docked_pose not in {pose_a, pose_b}:
                    raise ValueError("ProLIF evidence is not bound to the docked pose")
            else:
                prolif_pose = _evidence_input(evidence_inputs, "prolif", "pose")
                if prolif_pose != docked_pose:
                    raise ValueError("ProLIF evidence is not bound to the docked pose")
        if "openmm" in evidence_inputs:
            openmm_pose = _evidence_input(evidence_inputs, "openmm", "pose")
            if openmm_pose != docked_pose:
                raise ValueError("OpenMM evidence is bound to the wrong docked pose")

    def _validate_worker_outputs(
        self,
        manifest: RunManifest,
        stage: RunState,
        outputs: tuple[ArtifactRef, ...],
    ) -> None:
        if not outputs:
            raise PipelineStageError(
                "OUTPUT_INVALID",
                f"{stage.value} worker returned no scientific output",
                recoverable=False,
            )
        primary = outputs[0]
        if primary.media_type != "application/json":
            raise PipelineStageError(
                "OUTPUT_INVALID",
                f"{stage.value} primary output must be an application/json bundle",
                recoverable=False,
            )
        try:
            value = self.artifacts.read_json(primary)
        except (OSError, ValueError) as exc:
            raise PipelineStageError(
                "OUTPUT_INVALID",
                f"{stage.value} primary output is not valid JSON: {exc}",
                recoverable=False,
            ) from exc
        if not isinstance(value, dict):
            raise PipelineStageError(
                "OUTPUT_INVALID",
                f"{stage.value} primary output must be a JSON object",
                recoverable=False,
            )
        allowed_bundle_schemas = (
            {"1.0", "2.0"} if stage is RunState.DOCKED else {"1.0"}
        )
        if value.get("schema_version") not in allowed_bundle_schemas or value.get(
            "kind"
        ) != _WORKER_BUNDLE_KIND[stage]:
            raise PipelineStageError(
                "OUTPUT_INVALID",
                f"{stage.value} output does not satisfy "
                f"{_WORKER_BUNDLE_KIND[stage]} {sorted(allowed_bundle_schemas)}",
                recoverable=False,
            )
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise PipelineStageError(
                "OUTPUT_INVALID",
                f"{stage.value} output candidates must be an array",
                recoverable=False,
            )
        if stage is RunState.COFOLDED:
            batch_reference = manifest.artifacts.get("support_openfold_batch")
            if batch_reference is None:
                raise PipelineStageError(
                    "OUTPUT_INVALID",
                    "COFOLDED output has no frozen top-8 input batch",
                    recoverable=False,
                )
            upstream = validate_cofold_batch(
                self.artifacts,
                batch_reference,
                case=self.load_case(manifest),
                screening_artifact=manifest.stage_records[
                    RunState.SCREENED.value
                ].outputs[0],
                library_index=manifest.input_artifacts["library_index"],
                allowed_receptors=self._allowed_receptors(
                    manifest, self.load_case(manifest)
                ),
                verify_chemistry=read_index_metadata(
                    self.artifacts.resolve(manifest.input_artifacts["library_index"])
                )[1].chemistry_verified,
            )
            upstream_candidates = upstream.get("cofold_candidates")
        else:
            previous_stage = RunState.DOCKED
            if stage is RunState.DOCKED:
                previous_stage = (
                    RunState.SELECTED
                    if RunState.SELECTED.value in manifest.stage_records
                    else RunState.COFOLDED
                )
            upstream = self.artifacts.read_json(
                manifest.stage_records[previous_stage.value].outputs[0]
            )
            upstream_candidates = (
                upstream.get("candidates") if isinstance(upstream, dict) else None
            )
        if not isinstance(upstream_candidates, list):
            raise PipelineStageError(
                "OUTPUT_INVALID",
                f"{stage.value} upstream dependency has no candidate array",
                recoverable=False,
            )
        upstream_ids = {
            str(candidate["molecule_id"])
            for candidate in upstream_candidates
            if isinstance(candidate, dict) and "molecule_id" in candidate
        }
        upstream_by_id = {
            str(candidate["molecule_id"]): candidate
            for candidate in upstream_candidates
            if isinstance(candidate, dict) and "molecule_id" in candidate
        }
        returned_artifacts = set(outputs[1:])
        case_seed = self.load_case(manifest).seed
        candidate_ids: set[str] = set()
        fixture_only = self._worker_is_fixture_only(self.config, stage)
        validation_prepared: dict[str, dict[str, Any]] = {}
        validation_pins: dict[str, dict[str, Any]] = {}
        try:
            if fixture_only:
                self._require_fixture_stage_coverage(
                    stage, value, upstream_candidates
                )
            elif stage is RunState.COFOLDED:
                self._validate_cofolded_production_contract(
                    manifest, value, outputs, upstream_candidates
                )
            elif stage is RunState.DOCKED:
                self._validate_docked_production_contract(
                    manifest, value, outputs, upstream_candidates
                )
            elif stage is RunState.VALIDATED:
                validation_prepared, validation_pins = (
                    self._validate_validation_production_contract(
                        manifest, value, outputs, upstream_candidates
                    )
                )
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError("worker candidate must be a JSON object")
                molecule_id = candidate.get("molecule_id")
                if not isinstance(molecule_id, str) or not molecule_id.strip():
                    raise ValueError("worker candidate requires a molecule_id")
                if molecule_id not in upstream_ids:
                    raise ValueError(
                        f"worker candidate was not retained by the previous stage: {molecule_id}"
                    )
                if molecule_id in candidate_ids:
                    raise ValueError(f"duplicate worker candidate: {molecule_id}")
                candidate_ids.add(molecule_id)
                candidate_id = candidate.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id.strip():
                    raise ValueError("worker candidate requires a candidate_id")
                engine = candidate.get("engine")
                if not isinstance(engine, str) or not engine.strip():
                    raise ValueError("worker candidate requires an engine")
                seed = candidate.get("seed")
                if not isinstance(seed, int) or isinstance(seed, bool) or seed != case_seed:
                    raise ValueError("worker candidate seed does not match the research case")
                if stage is RunState.COFOLDED:
                    expected = upstream_by_id[molecule_id]
                    if candidate_id != expected.get("candidate_id") or (
                        candidate.get("microstate_id") != expected.get("microstate_id")
                    ):
                        raise ValueError(
                            "cofold output does not match its top-8 candidate/microstate"
                        )
                    structure = ArtifactRef.from_dict(candidate["structure"])
                    self.artifacts.resolve(structure)
                    if structure not in returned_artifacts:
                        raise ValueError("cofold structure must be returned with its bundle")
                    if not fixture_only and engine != OPENFOLD_RUNTIME_ENGINE:
                        raise ValueError(
                            "production COFOLDED candidate is not official OpenFold3"
                        )
                    if engine == OPENFOLD_RUNTIME_ENGINE:
                        if structure.media_type not in {
                            "chemical/x-mmcif",
                            "chemical/mmcif",
                        }:
                            raise ValueError("official OpenFold3 structures must be mmCIF")
                        expected_sequences = tuple(
                            str(chain["sequence"]).removesuffix("*")
                            for chain in upstream["protein_chains"]
                        )
                        expected_elements = heavy_element_counts(
                            str(expected["canonical_isomeric_smiles"])
                        )
                        sample_values = candidate.get("samples")
                        if not isinstance(sample_values, list) or not sample_values:
                            raise ValueError(
                                "official OpenFold3 output requires its complete sample set"
                            )
                        sample_structures: list[ArtifactRef] = []
                        for sample in sample_values:
                            if not isinstance(sample, dict):
                                raise ValueError("OpenFold3 sample must be an object")
                            sample_structure = ArtifactRef.from_dict(sample["structure"])
                            sample_confidence = ArtifactRef.from_dict(sample["confidence"])
                            self.artifacts.resolve(sample_structure)
                            self.artifacts.resolve(sample_confidence)
                            if sample_structure not in returned_artifacts or (
                                sample_confidence not in returned_artifacts
                            ):
                                raise ValueError(
                                    "OpenFold3 samples/confidences must be returned artifacts"
                                )
                            sample_structures.append(sample_structure)
                        if structure.sha256 not in {
                            item.sha256 for item in sample_structures
                        }:
                            raise ValueError(
                                "selected OpenFold3 structure is absent from its sample set"
                            )
                        for sample_structure in sample_structures:
                            inspection = inspect_predicted_complex(
                                self.artifacts.resolve(sample_structure),
                                expected_sequences=expected_sequences,
                            )
                            if inspection.ligand_heavy_element_counts != expected_elements:
                                raise ValueError(
                                    "OpenFold3 ligand elements differ from the selected microstate"
                                )
                    confidence_name = candidate.get("confidence_name")
                    confidence = candidate.get("confidence_value")
                    confidence_semantics = candidate.get("confidence_semantics")
                    if not isinstance(confidence_name, str) or not confidence_name.strip():
                        raise ValueError("cofold confidence requires a metric name")
                    if (
                        not isinstance(confidence, int | float)
                        or isinstance(confidence, bool)
                        or not math.isfinite(float(confidence))
                    ):
                        raise ValueError("cofold confidence must be finite")
                    if not isinstance(confidence_semantics, str) or (
                        "not binding affinity" not in confidence_semantics.lower()
                    ):
                        raise ValueError(
                            "cofold confidence semantics must disclaim binding affinity"
                        )
                elif stage is RunState.DOCKED:
                    expected = upstream_by_id[molecule_id]
                    if candidate.get("parent_candidate_id") != expected.get(
                        "candidate_id"
                    ) or candidate.get("microstate_id") != expected.get("microstate_id"):
                        raise ValueError(
                            "docked pose does not preserve selected candidate lineage"
                        )
                    if "structure" in expected:
                        cofold_structure = ArtifactRef.from_dict(
                            candidate["cofold_structure"]
                        )
                        self.artifacts.resolve(cofold_structure)
                        expected_cofold = ArtifactRef.from_dict(expected["structure"])
                        if cofold_structure != expected_cofold:
                            raise ValueError("docked pose names the wrong cofold structure")
                    elif "cofold_structure" in candidate:
                        raise ValueError(
                            "Vina-only docking output cannot require a cofold structure"
                        )
                    pose = ArtifactRef.from_dict(candidate["pose"])
                    self.artifacts.resolve(pose)
                    if pose not in returned_artifacts:
                        raise ValueError("docked pose must be returned with its bundle")
                    vina_score = candidate.get("vina_score")
                    if (
                        not isinstance(vina_score, int | float)
                        or isinstance(vina_score, bool)
                        or not math.isfinite(float(vina_score))
                    ):
                        raise ValueError("Vina score must be finite")
                    score_semantics = candidate.get("vina_score_semantics")
                    if not isinstance(score_semantics, str) or (
                        "not an experimental binding free energy"
                        not in score_semantics.lower()
                    ):
                        raise ValueError(
                            "Vina score semantics must disclaim experimental binding free energy"
                        )
                    center = candidate.get("box_center")
                    size = candidate.get("box_size")
                    if (
                        not isinstance(center, list)
                        or not isinstance(size, list)
                        or len(center) != 3
                        or len(size) != 3
                        or any(
                            not isinstance(item, int | float)
                            or isinstance(item, bool)
                            or not math.isfinite(float(item))
                            for item in (*center, *size)
                        )
                        or any(float(item) <= 0 for item in size)
                    ):
                        raise ValueError("docking box must contain finite center/positive size")
                else:
                    expected = upstream_by_id[molecule_id]
                    if candidate_id != expected.get("candidate_id") or (
                        candidate.get("microstate_id") != expected.get("microstate_id")
                    ):
                        raise ValueError(
                            "validation bundle does not preserve docked candidate lineage"
                        )
                    docked_pose = ArtifactRef.from_dict(candidate["docked_pose"])
                    self.artifacts.resolve(docked_pose)
                    expected_docked = ArtifactRef.from_dict(expected["pose"])
                    cofold_pose: ArtifactRef | None = None
                    if expected.get("cofold_structure") is not None:
                        cofold_pose = ArtifactRef.from_dict(candidate["cofold_pose"])
                        self.artifacts.resolve(cofold_pose)
                        expected_cofold = ArtifactRef.from_dict(
                            expected["cofold_structure"]
                        )
                        if cofold_pose != expected_cofold:
                            raise ValueError(
                                "validation bundle is bound to the wrong cofold pose"
                            )
                    elif candidate.get("cofold_pose") is not None:
                        raise ValueError(
                            "Vina-only validation cannot invent a cofold pose"
                        )
                    if docked_pose != expected_docked:
                        raise ValueError("validation bundle is bound to the wrong docked pose")
                    has_reference_pose = candidate.get("has_reference_pose", False)
                    if not isinstance(has_reference_pose, bool):
                        raise ValueError("has_reference_pose must be boolean")
                    bundle = _validation_bundle(candidate.get("bundle"))
                    if bundle.posebusters_valid is None:
                        raise ValueError(
                            "validation requires an explicit PoseBusters result"
                        )
                    if not bundle.evidence:
                        raise ValueError(
                            "validation metrics require returned per-tool evidence artifacts"
                        )
                    tool_metrics: dict[str, dict[str, Any]] = {}
                    tool_inputs: dict[str, dict[str, Any]] = {}
                    for evidence_reference in bundle.evidence:
                        self.artifacts.resolve(evidence_reference)
                        if evidence_reference not in returned_artifacts:
                            raise ValueError(
                                "validation evidence must be returned with its bundle"
                            )
                        tool, metrics, inputs, runtime, evidence_fixture = _tool_evidence(
                            self.artifacts,
                            evidence_reference,
                            molecule_id=molecule_id,
                            candidate_id=candidate_id,
                        )
                        if tool in tool_metrics:
                            raise ValueError(f"duplicate {tool} evidence")
                        tool_metrics[tool] = metrics
                        tool_inputs[tool] = inputs
                        if not fixture_only:
                            if tool not in {
                                "posebusters",
                                "spyrmsd",
                                "prolif",
                                "openmm",
                            }:
                                raise ValueError(
                                    f"production emitted unsupported validation tool: {tool}"
                                )
                            if evidence_fixture is not False or _has_fixture_label(
                                evidence_reference.producer
                            ):
                                raise ValueError(
                                    "production validation evidence carries a fixture label"
                                )
                            pin = validation_pins.get(tool)
                            if pin is None:
                                raise ValueError(
                                    f"{tool} evidence is absent from the frozen toolchain"
                                )
                            if runtime is None or runtime.get("version") != pin["version"] or (
                                runtime.get("package_source_sha256")
                                != pin["package_source_sha256"]
                            ):
                                raise ValueError(
                                    f"{tool} evidence runtime differs from the frozen toolchain"
                                )
                            if evidence_reference.producer_version != pin["version"]:
                                raise ValueError(
                                    f"{tool} evidence producer_version differs from runtime"
                                )
                    _require_evidence_metric(
                        tool_metrics,
                        "posebusters",
                        "valid",
                        bundle.posebusters_valid,
                    )
                    _require_evidence_metric(
                        tool_metrics,
                        "spyrmsd",
                        "symmetry_rmsd_angstrom",
                        bundle.symmetry_rmsd_angstrom,
                    )
                    _require_evidence_metric(
                        tool_metrics,
                        "prolif",
                        "ifp_similarity",
                        bundle.ifp_similarity,
                    )
                    prolif_metrics = tool_metrics.get("prolif")
                    if prolif_metrics is not None:
                        docked_labels = prolif_metrics.get("docked_labels")
                        comparison_labels = prolif_metrics.get("comparison_labels")
                        comparison_name = prolif_metrics.get("comparison")
                        if not isinstance(docked_labels, list) or any(
                            not isinstance(label, str) or not label
                            for label in docked_labels
                        ):
                            raise ValueError("ProLIF evidence has invalid docked labels")
                        if comparison_labels is not None and (
                            not isinstance(comparison_labels, list)
                            or any(
                                not isinstance(label, str) or not label
                                for label in comparison_labels
                            )
                        ):
                            raise ValueError("ProLIF evidence has invalid comparison labels")
                        recomputed_ifp = interaction_fingerprint_metrics(
                            docked_labels,
                            comparison_labels,
                            comparison_name=comparison_name,
                        )
                        for metric_name in (
                            "ifp_similarity",
                            "reference_interaction_recovery",
                            "predicted_interaction_precision",
                        ):
                            if prolif_metrics.get(metric_name) != recomputed_ifp[metric_name]:
                                raise ValueError(
                                    f"ProLIF metric {metric_name} differs from its labels"
                                )
                        if prolif_metrics.get("counts") != recomputed_ifp["counts"]:
                            raise ValueError("ProLIF counts differ from its labels")
                        expected_bundle_ifp = {
                            "ifp_similarity": recomputed_ifp["ifp_similarity"],
                            "ifp_reference_recovery": recomputed_ifp[
                                "reference_interaction_recovery"
                            ],
                            "ifp_predicted_precision": recomputed_ifp[
                                "predicted_interaction_precision"
                            ],
                            "ifp_docked_label_count": recomputed_ifp["counts"][
                                "docked"
                            ],
                            "ifp_comparison_label_count": recomputed_ifp["counts"][
                                "comparison"
                            ],
                            "ifp_intersection_count": recomputed_ifp["counts"][
                                "intersection"
                            ],
                            "ifp_union_count": recomputed_ifp["counts"]["union"],
                        }
                        observed_bundle_ifp = {
                            name: getattr(bundle, name) for name in expected_bundle_ifp
                        }
                        if observed_bundle_ifp != expected_bundle_ifp:
                            raise ValueError(
                                "ValidationBundle IFP summary differs from ProLIF evidence"
                            )
                    elif any(
                        getattr(bundle, name) is not None
                        for name in (
                            "ifp_similarity",
                            "ifp_reference_recovery",
                            "ifp_predicted_precision",
                            "ifp_docked_label_count",
                            "ifp_comparison_label_count",
                            "ifp_intersection_count",
                            "ifp_union_count",
                        )
                    ):
                        raise ValueError("ValidationBundle IFP metrics lack ProLIF evidence")
                    if fixture_only:
                        _require_evidence_metric(
                            tool_metrics,
                            "vina",
                            "pose_valid",
                            bundle.vina_pose_valid,
                        )
                        _require_evidence_metric(
                            tool_metrics,
                            "openfold3",
                            "pose_valid",
                            bundle.cofold_pose_valid,
                        )
                    else:
                        _require_evidence_metric(
                            tool_metrics,
                            "posebusters",
                            "docked_valid",
                            bundle.vina_pose_valid,
                        )
                        if cofold_pose is not None:
                            _require_evidence_metric(
                                tool_metrics,
                                "posebusters",
                                "cofold_valid",
                                bundle.cofold_pose_valid,
                            )
                        elif bundle.cofold_pose_valid is not None:
                            raise ValueError(
                                "Vina-only validation cannot claim cofold validity"
                            )
                        posebusters_metrics = tool_metrics.get("posebusters", {})
                        # The primary PB gate belongs to the docked Vina pose.
                        # Optional cofold validity is independent evidence and
                        # must not veto an otherwise valid docking result.
                        expected_pb_valid = (
                            posebusters_metrics.get("docked_valid") is True
                        )
                        if posebusters_metrics.get("config") not in {
                            "dock",
                            "redock",
                        } or posebusters_metrics.get("valid") is not expected_pb_valid:
                            raise ValueError(
                                "PoseBusters validity/configuration is internally inconsistent"
                            )
                    _require_evidence_metric(
                        tool_metrics,
                        "openmm",
                        "parameterized",
                        bundle.openmm_parameterized,
                    )
                    _require_evidence_metric(
                        tool_metrics,
                        "openmm",
                        "stable",
                        bundle.openmm_stable,
                    )
                    posebusters_docked = _evidence_input(
                        tool_inputs, "posebusters", "docked_pose"
                    )
                    self.artifacts.resolve(posebusters_docked)
                    if posebusters_docked != docked_pose:
                        raise ValueError(
                            "PoseBusters evidence is not bound to the docked pose"
                        )
                    if cofold_pose is not None:
                        posebusters_cofold = _evidence_input(
                            tool_inputs, "posebusters", "cofold_pose"
                        )
                        self.artifacts.resolve(posebusters_cofold)
                        if posebusters_cofold != cofold_pose:
                            raise ValueError(
                                "PoseBusters evidence is bound to the wrong cofold pose"
                            )
                    if fixture_only and bundle.vina_pose_valid is not None:
                        vina_pose = _evidence_input(tool_inputs, "vina", "pose")
                        self.artifacts.resolve(vina_pose)
                        if vina_pose != docked_pose:
                            raise ValueError("Vina evidence is bound to the wrong pose")
                    if fixture_only and bundle.cofold_pose_valid is not None:
                        if cofold_pose is None:
                            raise ValueError(
                                "Vina-only fixture cannot claim OpenFold3 validity"
                            )
                        openfold_pose = _evidence_input(
                            tool_inputs, "openfold3", "pose"
                        )
                        self.artifacts.resolve(openfold_pose)
                        if openfold_pose != cofold_pose:
                            raise ValueError(
                                "OpenFold3 evidence is bound to the wrong pose"
                            )
                    if "prolif" in tool_inputs:
                        prolif_inputs = tool_inputs["prolif"]
                        if "pose_a" in prolif_inputs or "pose_b" in prolif_inputs:
                            ifp_left = _evidence_input(
                                tool_inputs, "prolif", "pose_a"
                            )
                            ifp_right = _evidence_input(
                                tool_inputs, "prolif", "pose_b"
                            )
                            self.artifacts.resolve(ifp_left)
                            self.artifacts.resolve(ifp_right)
                            if cofold_pose is not None and {
                                ifp_left,
                                ifp_right,
                            } != {docked_pose, cofold_pose}:
                                raise ValueError(
                                    "ProLIF consensus is not bound to docked/cofold poses"
                                )
                            if cofold_pose is None and docked_pose not in {
                                ifp_left,
                                ifp_right,
                            }:
                                raise ValueError(
                                    "ProLIF evidence is not bound to the docked pose"
                                )
                        else:
                            ifp_pose = _evidence_input(
                                tool_inputs, "prolif", "pose"
                            )
                            self.artifacts.resolve(ifp_pose)
                            if ifp_pose != docked_pose:
                                raise ValueError(
                                    "ProLIF evidence is not bound to the docked pose"
                                )
                    if bundle.openmm_parameterized is not None or (
                        bundle.openmm_stable is not None
                    ):
                        openmm_pose = _evidence_input(tool_inputs, "openmm", "pose")
                        self.artifacts.resolve(openmm_pose)
                        if openmm_pose != docked_pose:
                            raise ValueError("OpenMM evidence is bound to the wrong pose")
                    if not fixture_only:
                        prepared = validation_prepared[candidate_id]
                        self._bind_prepared_validation_inputs(
                            prepared,
                            tool_inputs,
                            tool_metrics,
                            docked_pose=docked_pose,
                            cofold_pose=cofold_pose,
                            has_reference_pose=has_reference_pose,
                            reference_pose=(
                                _artifact_reference(
                                    candidate.get("reference_pose"), "reference pose"
                                )
                                if has_reference_pose
                                else None
                            ),
                        )
                    if has_reference_pose:
                        reference_pose = ArtifactRef.from_dict(
                            candidate["reference_pose"]
                        )
                        self.artifacts.resolve(reference_pose)
                        support_reference = manifest.artifacts.get(
                            "support_reference_pose"
                        )
                        allowed_reference_hashes = (
                            {support_reference.sha256}
                            if support_reference is not None
                            else set()
                        )
                        if reference_pose.sha256 not in allowed_reference_hashes:
                            raise ValueError(
                                "reference pose is not bound to the case/support inputs"
                            )
                        if bundle.symmetry_rmsd_angstrom is not None:
                            spyrmsd_inputs = tool_inputs.get("spyrmsd", {})
                            rmsd_reference = ArtifactRef.from_dict(
                                spyrmsd_inputs["reference_pose"]
                            )
                            rmsd_prediction = ArtifactRef.from_dict(
                                spyrmsd_inputs["predicted_pose"]
                            )
                            dock_pose = ArtifactRef.from_dict(
                                upstream_by_id[molecule_id]["pose"]
                            )
                            if rmsd_reference.sha256 != reference_pose.sha256 or (
                                rmsd_prediction.sha256 != dock_pose.sha256
                            ):
                                raise ValueError(
                                    "sPyRMSD evidence is not bound to reference and docked poses"
                                )
                    elif bundle.symmetry_rmsd_angstrom is not None:
                        raise ValueError(
                            "symmetry RMSD cannot be claimed without a reference pose"
                        )
                    decision_reason = candidate.get("decision_reason")
                    if not isinstance(decision_reason, str) or not decision_reason.strip():
                        raise ValueError("validation candidate requires a decision_reason")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise PipelineStageError(
                "OUTPUT_INVALID",
                f"{stage.value} candidate bundle is invalid: {exc}",
                recoverable=False,
            ) from exc

    def _report(self, manifest: RunManifest, case: ResearchCase) -> None:
        started = time.perf_counter()
        screen_artifact = manifest.stage_records[RunState.SCREENED.value].outputs[0]
        validation_artifact = manifest.stage_records[RunState.VALIDATED.value].outputs[0]
        resolution_artifact = manifest.input_artifacts["target_resolution"]
        resolution = self.artifacts.read_json(resolution_artifact)
        if not isinstance(resolution, dict) or not isinstance(
            resolution.get("decision"), str
        ):
            raise ValueError("target structure resolution receipt is invalid")
        screen = self.artifacts.read_json(screen_artifact)
        screen_hits = screen.get("hits") if isinstance(screen, dict) else None
        if not isinstance(screen_hits, list):
            raise ValueError("screening artifact has no hits array")
        screen_rank = {
            str(hit["molecule_id"]): int(hit.get("rank", position))
            for position, hit in enumerate(screen_hits, start=1)
            if isinstance(hit, dict) and "molecule_id" in hit
        }
        validation = self.artifacts.read_json(validation_artifact)
        validation_candidates = (
            validation.get("candidates") if isinstance(validation, dict) else None
        )
        if not isinstance(validation_candidates, list):
            raise ValueError("validation artifact has no candidates array")
        evaluated: list[
            tuple[
                int,
                str,
                str,
                tuple[ArtifactRef, ...],
                ValidationBundle,
                str,
            ]
        ] = []
        for item in validation_candidates:
            if not isinstance(item, dict):
                raise ValueError("validation candidate must be a JSON object")
            molecule_id = str(item["molecule_id"])
            if molecule_id not in screen_rank:
                raise ValueError("validation candidate is absent from screening results")
            bundle = _validation_bundle(item.get("bundle"))
            has_reference_pose = item.get("has_reference_pose", False)
            if not isinstance(has_reference_pose, bool):
                raise ValueError("has_reference_pose must be boolean")
            for evidence in bundle.evidence:
                self.artifacts.resolve(evidence)
            grade = classify_evidence(
                bundle,
                has_reference_pose=has_reference_pose,
            ).value
            decision_reason = item.get("decision_reason")
            if not isinstance(decision_reason, str) or not decision_reason.strip():
                raise ValueError("validation candidate requires a decision reason")
            evaluated.append(
                (
                    screen_rank[molecule_id],
                    molecule_id,
                    grade,
                    bundle.evidence,
                    bundle,
                    redact_text(decision_reason.strip()),
                )
            )
        evaluated.sort(key=lambda item: (item[0], item[1]))
        grade_priority = {
            "REDOCKING_RECOVERED": 0,
            "REFERENCE_SUPPORTED": 0,
            "METHOD_CONSENSUS": 1,
            "CONSENSUS_SUPPORTED": 1,
            "HYPOTHESIS_ONLY": 2,
        }
        accepted = sorted(
            (item for item in evaluated if item[2] != "REJECTED"),
            key=lambda item: (grade_priority[item[2]], item[0], item[1]),
        )[: _REPORT_CONFIG["top_candidates"]]
        rejected = [item for item in evaluated if item[2] == "REJECTED"]
        evaluated_ids = {item[1] for item in evaluated}
        unevaluated = [
            molecule_id for molecule_id in screen_rank if molecule_id not in evaluated_ids
        ]
        lines = [
            "# ProtBind local private research report",
            "",
            f"- Run: `{manifest.run_id}`",
            f"- Case: `{case.case_id}`",
            f"- Mode: `{case.mode.value}`",
            f"- Initial receptor resolution: `{resolution['decision']}` — evidence: "
            f"`{resolution_artifact.artifact_id}`",
            f"- Screening evidence: `{screen_artifact.artifact_id}`",
            f"- Validation evidence: `{validation_artifact.artifact_id}`",
            "",
            "## Top evidence candidates",
            "",
        ]
        if not accepted:
            lines.append("No validated, non-rejected candidate is available.")
        for rank, molecule_id, grade, evidence, bundle, reason in accepted:
            citations = evidence or (validation_artifact,)
            metrics = self._validation_metric_text(bundle)
            lines.append(
                f"- #{rank} `{molecule_id}` — `{grade}` — evidence: "
                + ", ".join(f"`{item.artifact_id}`" for item in citations)
                + f" — {metrics} — reason: {reason}"
            )
        if rejected:
            lines.extend(["", "## Rejected candidates", ""])
            for rank, molecule_id, grade, evidence, bundle, reason in rejected:
                citations = evidence or (validation_artifact,)
                metrics = self._validation_metric_text(bundle)
                lines.append(
                    f"- #{rank} `{molecule_id}` — `{grade}` — evidence: "
                    + ", ".join(f"`{item.artifact_id}`" for item in citations)
                    + f" — {metrics} — reason: {reason}"
                )
        if unevaluated:
            lines.extend(
                [
                    "",
                    "## Not independently validated",
                    "",
                    f"{len(unevaluated)} screened candidate(s) have no validation bundle and "
                    "are not assigned an evidence grade.",
                ]
            )
        lines.extend(
            [
                "",
                "## Interpretation limits",
                "",
                "- TriPharm values are geometric pharmacophore matches, not binding scores.",
                "- AutoDock Vina values, when present, are pose-ranking tool scores and not "
                "experimental binding free energies.",
                "- A cofolded pose is a model prediction, not a demonstrated binding fact.",
                "- Every supported result must resolve to the artifact IDs listed above.",
                "",
            ]
        )
        markdown = "\n".join(lines)
        markdown_artifact = self.artifacts.put_bytes(
            markdown.encode("utf-8"),
            media_type="text/markdown",
            producer="protbind.evidence-report",
            producer_version=__version__,
        )
        html_artifact = self.artifacts.put_bytes(
            (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>ProtBind report</title></head><body><pre>"
                + html.escape(markdown)
                + "</pre></body></html>"
            ).encode("utf-8"),
            media_type="text/html",
            producer="protbind.evidence-report",
            producer_version=__version__,
        )
        manifest.artifacts["report_markdown"] = markdown_artifact
        manifest.artifacts["report_html"] = html_artifact
        manifest.complete_stage(
            StageRecord.create(
                RunState.REPORTED,
                input_hash=_artifact_input_hash(
                    screen_artifact,
                    validation_artifact,
                    resolution_artifact,
                ),
                config_hash=sha256_bytes(canonical_json_bytes(_REPORT_CONFIG)),
                outputs=(markdown_artifact, html_artifact),
                duration_seconds=time.perf_counter() - started,
            )
        )

    @staticmethod
    def _validation_metric_text(bundle: ValidationBundle) -> str:
        values = [f"PB-valid={bundle.posebusters_valid}"]
        if bundle.symmetry_rmsd_angstrom is not None:
            values.append(f"symmetry-RMSD={bundle.symmetry_rmsd_angstrom:.3f} Å")
        if bundle.ifp_similarity is not None:
            values.append(f"IFP-Jaccard={bundle.ifp_similarity:.3f}")
        if bundle.ifp_comparison_label_count is not None:
            intersection = bundle.ifp_intersection_count
            comparison = bundle.ifp_comparison_label_count
            docked = bundle.ifp_docked_label_count
            recovery = (
                f"{bundle.ifp_reference_recovery:.3f}"
                if bundle.ifp_reference_recovery is not None
                else "N/A"
            )
            precision = (
                f"{bundle.ifp_predicted_precision:.3f}"
                if bundle.ifp_predicted_precision is not None
                else "N/A"
            )
            values.append(
                f"IFP-reference-recovery={intersection}/{comparison} ({recovery})"
            )
            values.append(
                f"IFP-predicted-precision={intersection}/{docked} ({precision})"
            )
        if bundle.openmm_parameterized is not None:
            values.append(f"OpenMM-parameterized={bundle.openmm_parameterized}")
        if bundle.openmm_stable is not None:
            values.append(f"OpenMM-stable={bundle.openmm_stable}")
        if bundle.unsupported_reasons:
            values.append(
                "unsupported="
                + "; ".join(redact_text(item) for item in bundle.unsupported_reasons)
            )
        return ", ".join(values)
