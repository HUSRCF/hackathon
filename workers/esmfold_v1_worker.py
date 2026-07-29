#!/usr/bin/env python3
"""Official fair-esm ESMFold v1 adapter with local-only weights and OOM retries."""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
from argparse import Namespace
from collections import defaultdict
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

# The model environment remains isolated, while the versioned protocol code comes
# from this repository. No existing project-specific ESMFold wrapper is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protbind_agent.artifacts import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from protbind_agent.esmfold_compat import (  # noqa: E402
    COMPATIBILITY_ID,
    install_fair_esm_py312_compat,
)
from protbind_agent.worker_protocol import WorkerRequest, WorkerResponse  # noqa: E402
from protbind_agent.worker_sdk import WorkerFailure, serve_worker  # noqa: E402

ENGINE = "esmfold_v1"
MODEL_REVISION = "esmfold_3B_v1"

# fair-esm serializes this checkpoint's OmegaConf model configuration alongside
# an ordinary tensor state_dict.  PyTorch 2.6+ deliberately refuses those
# classes unless they are explicitly allowlisted.  Keep the restricted
# weights-only unpickler enabled and reject any checkpoint that references a
# global outside this small, reviewed set.
EXPECTED_CHECKPOINT_GLOBALS = frozenset(
    {
        "builtins.dict",
        "argparse.Namespace",
        "collections.defaultdict",
        "omegaconf.base.ContainerMetadata",
        "omegaconf.base.Metadata",
        "omegaconf.dictconfig.DictConfig",
        "omegaconf.nodes.AnyNode",
        "typing.Any",
    }
)
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
FAIR_ESM_VERSION = "2.0.0"
LEGACY_OPENFOLD_VERSION = "2.2.0"
LEGACY_OPENFOLD_REVISION = "e938c184a291bf053af3b14c1e3e8bb29aee57e2"
LEGACY_OPENFOLD_SOURCE_SHA256 = (
    "75e2d37fbc3cdeab557eda055c03b8dbcd5694940d55ddb99e7132afabf80498"
)
LEGACY_OPENFOLD_SOURCE_FILE_COUNT = 79
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _protbind_runtime_sha256() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    sources = [
        (str(path.relative_to(repository_root)), sha256_file(path))
        for path in sorted((repository_root / "src" / "protbind_agent").rglob("*.py"))
    ]
    if not sources:
        raise RuntimeError("ProtBind runtime source manifest is empty")
    return sha256_bytes(canonical_json_bytes(sources))


def _runtime_file_manifest(
    torch: Any,
) -> tuple[str, int, dict[str, str], dict[str, int]]:
    """Hash the model code plus the PyTorch files that load/execute ESMFold.

    The environment lock binds the complete solved environment.  This manifest
    additionally prevents a shadowed or locally modified fair-esm/OmegaConf
    source tree, Python executable, Torch loader, or core ROCm library from
    retaining the same adapter identity.
    """

    entries: list[tuple[str, str]] = []
    components: dict[str, list[tuple[str, str]]] = {}
    for module_name in ("esm", "omegaconf", "openfold"):
        module = __import__(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"{module_name} package root is unavailable")
        root = Path(module_file).resolve().parent
        package_entries = [
            (f"{module_name}/{path.relative_to(root).as_posix()}", sha256_file(path))
            for path in sorted(root.rglob("*.py"))
            if path.is_file() and "__pycache__" not in path.parts
        ]
        if not package_entries:
            raise RuntimeError(f"{module_name} package source manifest is empty")
        components[module_name] = package_entries
        entries.extend(package_entries)

    native_spec = importlib_util.find_spec("attn_core_inplace_cuda")
    native_origin = native_spec.origin if native_spec is not None else None
    if not isinstance(native_origin, str) or not Path(native_origin).is_file():
        raise RuntimeError("legacy OpenFold attention extension is unavailable")
    native_entry = (
        "openfold_native/attn_core_inplace_cuda",
        sha256_file(Path(native_origin)),
    )
    components["openfold_native"] = [native_entry]
    entries.append(native_entry)

    torch_root = Path(torch.__file__).resolve().parent
    required_torch_files = (
        torch_root / "__init__.py",
        torch_root / "serialization.py",
        torch_root / "hub.py",
        torch_root / "cuda" / "__init__.py",
    )
    native_candidates = (
        tuple(sorted(torch_root.glob("_C*.so")))
        + tuple(sorted((torch_root / "lib").glob("libtorch_python.so")))
        + tuple(sorted((torch_root / "lib").glob("libtorch_hip.so")))
        + tuple(sorted((torch_root / "lib").glob("libc10_hip.so")))
    )
    if any(not path.is_file() for path in required_torch_files) or not native_candidates:
        raise RuntimeError("PyTorch loader/native runtime files are unavailable")
    torch_entries = [
        (f"torch/{path.relative_to(torch_root).as_posix()}", sha256_file(path))
        for path in (*required_torch_files, *native_candidates)
    ]
    components["torch"] = torch_entries
    entries.extend(torch_entries)
    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        raise RuntimeError("Python executable cannot be attested")
    python_entry = ("python/executable", sha256_file(executable))
    components["python"] = [python_entry]
    entries.append(python_entry)
    entries.sort()
    component_hashes = {
        name: sha256_bytes(canonical_json_bytes(sorted(values)))
        for name, values in sorted(components.items())
    }
    component_file_counts = {
        name: len(values) for name, values in sorted(components.items())
    }
    return (
        sha256_bytes(canonical_json_bytes(entries)),
        len(entries),
        component_hashes,
        component_file_counts,
    )


def runtime_attestation(environment_lock_path: Path, torch: Any) -> dict[str, Any]:
    if not environment_lock_path.is_file():
        raise RuntimeError("the frozen ESMFold environment lock is unavailable")
    fair_esm_version = importlib_metadata.version("fair-esm")
    if fair_esm_version != FAIR_ESM_VERSION:
        raise RuntimeError("installed fair-esm version differs from the pinned runtime")
    source_sha256, file_count, component_hashes, component_file_counts = (
        _runtime_file_manifest(torch)
    )
    if importlib_metadata.version("openfold") != LEGACY_OPENFOLD_VERSION or (
        component_hashes.get("openfold") != LEGACY_OPENFOLD_SOURCE_SHA256
        or component_file_counts.get("openfold") != LEGACY_OPENFOLD_SOURCE_FILE_COUNT
    ):
        raise RuntimeError("legacy OpenFold source differs from the pinned runtime")
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    if hip_version is None:
        raise RuntimeError("the ESMFold runtime is not a ROCm PyTorch build")
    return {
        "fair_esm_version": fair_esm_version,
        "torch_version": str(torch.__version__),
        "torch_hip_version": str(hip_version),
        "omegaconf_version": importlib_metadata.version("omegaconf"),
        "legacy_openfold_version": LEGACY_OPENFOLD_VERSION,
        "legacy_openfold_revision": LEGACY_OPENFOLD_REVISION,
        "legacy_openfold_source_sha256": LEGACY_OPENFOLD_SOURCE_SHA256,
        "fair_esm_compatibility_id": COMPATIBILITY_ID,
        "environment_lock_sha256": sha256_file(environment_lock_path),
        "runtime_source_sha256": source_sha256,
        "runtime_file_count": file_count,
        "runtime_component_sha256": component_hashes,
        "runtime_component_file_counts": component_file_counts,
        "attestation_scope": (
            "environment lock plus fair-esm/OmegaConf/legacy OpenFold Python sources, "
            "the OpenFold attention extension, Python executable, and PyTorch "
            "loader/core ROCm runtime files"
        ),
    }


def composite_code_sha256(
    environment_lock_path: Path,
    torch: Any,
    attestation: dict[str, Any] | None = None,
) -> str:
    attestation = attestation or runtime_attestation(environment_lock_path, torch)
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "adapter_sha256": sha256_file(Path(__file__)),
                "protbind_runtime_sha256": _protbind_runtime_sha256(),
                **attestation,
            }
        )
    )


def checkpoint_set_sha256(
    model_path: Path, esm2_model_path: Path, esm2_regression_path: Path
) -> str:
    entries = []
    for role, path in (
        ("esmfold", model_path),
        ("esm2_backbone", esm2_model_path),
        ("esm2_contact_regression", esm2_regression_path),
    ):
        entries.append(
            {
                "role": role,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "model_revision": MODEL_REVISION,
                "artifacts": entries,
            }
        )
    )


def _safe_checkpoint_globals(torch, paths: tuple[Path, ...]) -> list[object]:  # noqa: ANN001
    try:
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.dictconfig import DictConfig
        from omegaconf.nodes import AnyNode
    except ImportError as exc:
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE",
            "the pinned fair-esm environment is missing OmegaConf checkpoint support",
            recoverable=True,
        ) from exc

    serialization = getattr(torch, "serialization", None)
    inspect_globals = getattr(serialization, "get_unsafe_globals_in_checkpoint", None)
    safe_globals = getattr(serialization, "safe_globals", None)
    if not callable(inspect_globals) or not callable(safe_globals):
        raise WorkerFailure(
            "UNSAFE_CHECKPOINT_RUNTIME",
            "this PyTorch build cannot inspect and restrict checkpoint globals",
            recoverable=True,
        )
    referenced = frozenset(
        str(item)
        for path in paths
        for item in inspect_globals(path)
    )
    unexpected = sorted(referenced - EXPECTED_CHECKPOINT_GLOBALS)
    if unexpected:
        raise WorkerFailure(
            "UNTRUSTED_CHECKPOINT_GLOBALS",
            "the pinned checkpoint references globals outside the ESMFold allowlist: "
            + ", ".join(unexpected),
            recoverable=False,
        )
    return [
        dict,
        Namespace,
        defaultdict,
        Any,
        ContainerMetadata,
        Metadata,
        DictConfig,
        AnyNode,
    ]


def _input_sequences(request: WorkerRequest, store) -> tuple[str, ...]:  # noqa: ANN001
    value = store.read_json(request.input)
    if not isinstance(value, dict) or not isinstance(value.get("sequences"), list):
        raise WorkerFailure(
            "INVALID_INPUT",
            "ESMFold input must be a JSON object with a sequences array",
            recoverable=False,
        )
    sequences = tuple(str(item).strip().upper() for item in value["sequences"])
    if not 1 <= len(sequences) <= 2:
        raise WorkerFailure(
            "UNSUPPORTED_TARGET",
            "ESMFold v1 worker accepts one or two protein chains",
            recoverable=False,
        )
    if any(
        not sequence
        or any(residue not in CANONICAL_AMINO_ACIDS for residue in sequence)
        for sequence in sequences
    ):
        raise WorkerFailure(
            "INVALID_INPUT",
            "protein sequences must contain only the 20 canonical amino acids",
            recoverable=False,
        )
    if sum(len(sequence) for sequence in sequences) > 700:
        raise WorkerFailure(
            "UNSUPPORTED_TARGET",
            "v1 total sequence length exceeds 700 residues",
            recoverable=False,
        )
    return sequences


def validate_predicted_pdb(pdb: str, sequences: tuple[str, ...]) -> dict[str, Any]:
    """Fail closed on malformed, non-finite, or identity-changing ESMFold PDB."""

    expected_chain_ids = tuple(chr(ord("A") + index) for index in range(len(sequences)))
    residues: dict[str, list[tuple[tuple[str, str], str, set[str]]]] = {
        chain_id: [] for chain_id in expected_chain_ids
    }
    residue_indexes: dict[tuple[str, str, str], int] = {}
    saw_atom = False
    saw_model = False
    finished_first_model = False
    for line in pdb.splitlines():
        record = line[:6].strip().upper()
        if record == "MODEL":
            if saw_model or finished_first_model:
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "ESMFold output must contain exactly one coordinate model",
                    recoverable=False,
                )
            saw_model = True
            continue
        if record == "ENDMDL":
            finished_first_model = True
            continue
        if finished_first_model:
            if record in {"ATOM", "HETATM", "MODEL"}:
                raise WorkerFailure(
                    "OUTPUT_INVALID",
                    "ESMFold output contains coordinates after the first model",
                    recoverable=False,
                )
            continue
        if record == "HETATM":
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "ESMFold receptor output contains an unexpected non-protein atom",
                recoverable=False,
            )
        if record != "ATOM":
            continue
        if len(line) < 54:
            raise WorkerFailure(
                "OUTPUT_INVALID", "ESMFold emitted a truncated ATOM record", recoverable=False
            )
        saw_atom = True
        atom_name = line[12:16].strip().upper()
        altloc = line[16:17]
        residue_name = line[17:20].strip().upper()
        chain_id = line[21:22].strip()
        residue_id = (line[22:26].strip(), line[26:27].strip())
        if altloc not in {"", " "}:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "ESMFold output contains unresolved alternate locations",
                recoverable=False,
            )
        if chain_id not in residues or residue_name not in _THREE_TO_ONE:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "ESMFold output contains unexpected chain or residue identities",
                recoverable=False,
            )
        try:
            coordinates = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        except ValueError as exc:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "ESMFold output contains an unreadable coordinate",
                recoverable=False,
            ) from exc
        if any(not math.isfinite(value) for value in coordinates):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "ESMFold output contains a non-finite coordinate",
                recoverable=False,
            )
        key = (chain_id, *residue_id)
        index = residue_indexes.get(key)
        if index is None:
            index = len(residues[chain_id])
            residue_indexes[key] = index
            residues[chain_id].append((residue_id, residue_name, set()))
        _, existing_name, atom_names = residues[chain_id][index]
        if existing_name != residue_name or atom_name in atom_names:
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "ESMFold output has inconsistent residue or duplicate atom identities",
                recoverable=False,
            )
        atom_names.add(atom_name)
    if not saw_atom:
        raise WorkerFailure(
            "OUTPUT_INVALID", "ESMFold output contains no protein atoms", recoverable=False
        )
    actual_sequences: list[str] = []
    for chain_id in expected_chain_ids:
        chain_residues = residues[chain_id]
        if not chain_residues or any(
            not {"N", "CA", "C"}.issubset(atom_names)
            for _, _, atom_names in chain_residues
        ):
            raise WorkerFailure(
                "OUTPUT_INVALID",
                "ESMFold output is missing a chain or N/CA/C backbone atoms",
                recoverable=False,
            )
        actual_sequences.append(
            "".join(_THREE_TO_ONE[residue_name] for _, residue_name, _ in chain_residues)
        )
    if tuple(actual_sequences) != sequences:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "ESMFold output protein sequences differ from the request",
            recoverable=False,
        )
    return {
        "chain_ids": list(expected_chain_ids),
        "chain_lengths": [len(sequence) for sequence in sequences],
        "residue_count": sum(len(sequence) for sequence in sequences),
        "coordinate_finite": True,
        "backbone_complete": True,
        "alternate_locations": False,
        "sequence_identity_sha256": [
            sha256_bytes(sequence.encode("ascii")) for sequence in sequences
        ],
    }


def _handler(request: WorkerRequest, store) -> WorkerResponse:  # noqa: ANN001
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise WorkerFailure(
            "ARCHITECTURE_SPOOFING",
            "HSA_OVERRIDE_GFX_VERSION is forbidden",
            recoverable=False,
        )
    model_path_value = request.parameters.get("model_path")
    allowed_parameters = {
        "model_path",
        "esm2_model_path",
        "esm2_regression_path",
        "environment_lock_path",
        "chunk_sizes",
        "minimum_free_vram_gib",
    }
    unknown_parameters = set(request.parameters) - allowed_parameters
    if unknown_parameters:
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "unsupported ESMFold parameters: " + ", ".join(sorted(unknown_parameters)),
            recoverable=False,
        )
    if request.provenance.model_revision != MODEL_REVISION:
        raise WorkerFailure(
            "MODEL_REVISION_MISMATCH",
            f"ESMFold v1 requires model revision {MODEL_REVISION}",
            recoverable=False,
        )
    if not isinstance(model_path_value, str):
        raise WorkerFailure(
            "MODEL_UNAVAILABLE",
            "a local model_path is required; hub download is disabled",
            recoverable=True,
        )
    esm2_model_path_value = request.parameters.get("esm2_model_path")
    esm2_regression_path_value = request.parameters.get("esm2_regression_path")
    environment_lock_path_value = request.parameters.get("environment_lock_path")
    if not isinstance(esm2_model_path_value, str) or not isinstance(
        esm2_regression_path_value, str
    ) or not isinstance(environment_lock_path_value, str):
        raise WorkerFailure(
            "MODEL_UNAVAILABLE",
            "local ESM2 checkpoints and a frozen environment lock are required",
            recoverable=True,
        )
    model_path = Path(model_path_value)
    esm2_model_path = Path(esm2_model_path_value)
    esm2_regression_path = Path(esm2_regression_path_value)
    environment_lock_path = Path(environment_lock_path_value)
    if not all(
        path.is_file()
        for path in (
            model_path,
            esm2_model_path,
            esm2_regression_path,
            environment_lock_path,
        )
    ):
        raise WorkerFailure(
            "MODEL_UNAVAILABLE",
            "one or more configured local ESMFold checkpoint files do not exist",
            recoverable=True,
        )
    if (
        model_path.suffix != ".pt"
        or esm2_model_path.name != "esm2_t36_3B_UR50D.pt"
        or esm2_regression_path.name
        != "esm2_t36_3B_UR50D-contact-regression.pt"
        or esm2_model_path.parent.resolve()
        != esm2_regression_path.parent.resolve()
        or esm2_model_path.parent.name != "checkpoints"
    ):
        raise WorkerFailure(
            "MODEL_UNAVAILABLE",
            "fair-esm checkpoint filenames/layout do not match the pinned offline model set",
            recoverable=False,
        )
    if (
        checkpoint_set_sha256(model_path, esm2_model_path, esm2_regression_path)
        != request.provenance.weight_sha256
    ):
        raise WorkerFailure(
            "WEIGHT_HASH_MISMATCH",
            "the composite ESMFold/ESM2 checkpoint identity does not match provenance",
            recoverable=False,
        )
    sequences = _input_sequences(request, store)
    visible_device = os.environ.get("HIP_VISIBLE_DEVICES", "")
    conflicting_masks = {
        name
        for name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
        if os.environ.get(name)
    }
    if (
        not visible_device.isascii()
        or not visible_device.isdecimal()
        or (len(visible_device) > 1 and visible_device.startswith("0"))
        or conflicting_masks
    ):
        raise WorkerFailure(
            "RESOURCE_POLICY_VIOLATION",
            "ESMFold requires one canonical HIP_VISIBLE_DEVICES index and no mask aliases",
            recoverable=False,
        )
    minimum_free_vram_gib = request.parameters.get("minimum_free_vram_gib", 12.0)
    if (
        not isinstance(minimum_free_vram_gib, int | float)
        or isinstance(minimum_free_vram_gib, bool)
        or not math.isfinite(float(minimum_free_vram_gib))
        or not 8 <= float(minimum_free_vram_gib) <= 48
    ):
        raise WorkerFailure(
            "INVALID_PARAMETERS",
            "minimum_free_vram_gib must be finite and in [8, 48]",
            recoverable=False,
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    compatibility_applied = False
    try:
        compatibility_applied = install_fair_esm_py312_compat()
        import torch
        from esm.esmfold.v1.pretrained import _load_model
    except ImportError as exc:
        raise WorkerFailure(
            "CAPABILITY_UNAVAILABLE",
            "official fair-esm ESMFold dependencies are not installed",
            recoverable=True,
        ) from exc
    except RuntimeError as exc:
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            "the fair-esm Python compatibility gate failed",
            recoverable=False,
        ) from exc
    if not torch.cuda.is_available():
        raise WorkerFailure(
            "ROCM_UNAVAILABLE",
            "PyTorch cannot access a ROCm device",
            recoverable=True,
        )
    try:
        runtime = runtime_attestation(environment_lock_path, torch)
    except (ImportError, importlib_metadata.PackageNotFoundError, OSError, RuntimeError) as exc:
        raise WorkerFailure(
            "MODEL_RUNTIME_INVALID",
            f"the pinned ESMFold runtime could not be attested: {type(exc).__name__}",
            recoverable=False,
        ) from exc
    computed_code_sha256 = composite_code_sha256(
        environment_lock_path, torch, runtime
    )
    if computed_code_sha256 != request.provenance.code_sha256:
        raise WorkerFailure(
            "CODE_HASH_MISMATCH",
            "ESMFold adapter/runtime code does not match request provenance; "
            f"computed identity is sha256:{computed_code_sha256}; "
            f"adapter=sha256:{sha256_file(Path(__file__))}; "
            f"protbind=sha256:{_protbind_runtime_sha256()}; "
            f"runtime=sha256:{runtime['runtime_source_sha256']}; "
            f"components={runtime['runtime_component_sha256']}; "
            f"lock=sha256:{runtime['environment_lock_sha256']}",
            recoverable=False,
        )
    free_vram, total_vram = torch.cuda.mem_get_info()
    required_free_vram = int(float(minimum_free_vram_gib) * 1024**3)
    if free_vram < required_free_vram:
        raise WorkerFailure(
            "GPU_BUSY",
            "the selected Radeon device does not meet the ESMFold free-VRAM admission gate",
            recoverable=True,
        )
    torch.manual_seed(request.seed)
    torch.cuda.manual_seed_all(request.seed)
    torch.cuda.reset_peak_memory_stats()
    allowed_globals = _safe_checkpoint_globals(
        torch, (model_path, esm2_model_path, esm2_regression_path)
    )
    # ESMFold constructs its frozen ESM2 backbone through fair-esm's hub helper.
    # Point that helper at the verified local files; the enclosing worker has no
    # network namespace and the expected filenames above prevent cache aliasing.
    torch.hub.set_dir(str(esm2_model_path.parent.parent.resolve()))
    load_started = time.perf_counter()
    # fair-esm 2.0.0's official local path enters _load_model when its argument
    # ends in .pt; this avoids torch.hub and all network access.
    try:
        with torch.serialization.safe_globals(allowed_globals):
            model = _load_model(str(model_path)).eval().cuda()
    except torch.OutOfMemoryError as exc:
        raise WorkerFailure(
            "OUT_OF_MEMORY",
            "ESMFold v1 could not place the model on the selected Radeon device",
            recoverable=True,
        ) from exc
    except pickle.UnpicklingError as exc:
        raise WorkerFailure(
            "UNTRUSTED_CHECKPOINT_FORMAT",
            "the ESMFold checkpoint was rejected by PyTorch's restricted unpickler",
            recoverable=False,
        ) from exc
    load_seconds = time.perf_counter() - load_started
    sequence = ":".join(sequences)
    chunk_sizes = request.parameters.get("chunk_sizes", [128, 64, 32])
    if not isinstance(chunk_sizes, list) or not chunk_sizes:
        raise WorkerFailure(
            "INVALID_PARAMETERS", "chunk_sizes must be a non-empty array", recoverable=False
        )
    warnings: list[str] = []
    if compatibility_applied:
        warnings.append(
            "applied exact-hash fair-esm Python dataclass compatibility shim"
        )
    inference_seconds = 0.0
    pdb: str | None = None
    for chunk_size_value in chunk_sizes:
        chunk_size = int(chunk_size_value)
        if chunk_size < 1:
            raise WorkerFailure(
                "INVALID_PARAMETERS", "chunk sizes must be positive", recoverable=False
            )
        model.set_chunk_size(chunk_size)
        try:
            started = time.perf_counter()
            with torch.no_grad():
                pdb = model.infer_pdb(sequence)
            inference_seconds = time.perf_counter() - started
            break
        except torch.OutOfMemoryError:
            warnings.append(f"OOM at chunk_size={chunk_size}; retrying with a smaller chunk")
            torch.cuda.empty_cache()
    if pdb is None:
        raise WorkerFailure(
            "OUT_OF_MEMORY",
            "ESMFold v1 exhausted the configured chunk-size retries",
            recoverable=True,
        )
    output_qc = validate_predicted_pdb(pdb, sequences)
    output = store.put_bytes(
        pdb.encode(),
        media_type="chemical/x-pdb",
        producer="fair-esm.esmfold_v1",
        producer_version=runtime["fair_esm_version"],
        source=request.input.artifact_id,
    )
    metadata = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.esmfold-v1-result",
            "input": request.input.to_dict(),
            "structure": output.to_dict(),
            "seed": request.seed,
            "model_revision": MODEL_REVISION,
            "weight_sha256": request.provenance.weight_sha256,
            "code_sha256": request.provenance.code_sha256,
            "runtime": runtime,
            "resource_policy": {
                "hip_visible_device": visible_device,
                "logical_device": 0,
                "minimum_free_vram_gib": float(minimum_free_vram_gib),
                "free_vram_bytes_at_admission": int(free_vram),
                "total_vram_bytes": int(total_vram),
            },
            "output_qc": output_qc,
        },
        producer="protbind.esmfold-v1-result",
        producer_version=runtime["fair_esm_version"],
        source=output.artifact_id,
    )
    return WorkerResponse(
        job_id=request.job_id,
        engine=request.engine,
        outputs=(output, metadata),
        provenance=request.provenance,
        timings_seconds={
            "model_load": load_seconds,
            "inference": inference_seconds,
        },
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
        warnings=tuple(warnings),
    )


if __name__ == "__main__":
    raise SystemExit(serve_worker(ENGINE, _handler))
