"""Build lineage-bound validation inputs from a completed docking bundle."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes, sha256_file
from .models import ArtifactRef
from .worker_protocol import WorkerProvenance

_LIGAND_MEDIA = {
    "chemical/x-mdl-sdfile",
    "chemical/x-sdf",
    "chemical/x-mdl-molfile",
    "chemical/x-mol2",
}
_RECEPTOR_MEDIA = {"chemical/x-pdb"}
_TOOL_MODULES = {
    "posebusters": ("posebusters", "posebusters"),
    "spyrmsd": ("spyrmsd", "spyrmsd"),
    "prolif": ("prolif", "prolif"),
    "openmm": ("openmm", "openmm"),
}


@dataclass(frozen=True, slots=True)
class ValidationToolchainBinding:
    artifact: ArtifactRef
    provenance: WorkerProvenance


def _reference(value: Any, name: str) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an ArtifactRef") from exc


def build_validation_input_batch(
    store: ArtifactStore,
    docking_bundle: ArtifactRef,
    *,
    reference_pose: ArtifactRef | None = None,
) -> ArtifactRef:
    """Derive all validation inputs that do not require chemical guessing.

    The canonical docked pose must already be an SDF exported and attested by
    the Vina worker.  A native reference is copied only into the validation
    batch; the caller is responsible for attaching it after DOCKED completes.
    """

    value = store.read_json(docking_bundle)
    if not isinstance(value, dict) or value.get("kind") != "protbind.docking-bundle":
        raise ValueError("docking_bundle is not a ProtBind docking bundle")
    receptor = _reference(
        value.get("receptor_preparation_input", value.get("receptor")),
        "docking receptor",
    )
    if receptor.media_type not in _RECEPTOR_MEDIA:
        raise ValueError("validation requires the normalized receptor as PDB")
    store.resolve(receptor)
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
        raise ValueError("validation requires one to sixteen successful docked candidates")
    if reference_pose is not None:
        if len(candidates) != 1:
            raise ValueError(
                "a single validation reference is valid only for a one-candidate "
                "redocking control; prospective multi-candidate runs require no RMSD "
                "reference or an explicit per-candidate reference mapping"
            )
        if reference_pose.media_type not in _LIGAND_MEDIA:
            raise ValueError("reference pose must preserve ligand chemistry in SDF/MOL2")
        store.resolve(reference_pose)

    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("docking candidate must be an object")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("docking candidate IDs must be unique non-empty strings")
        seen.add(candidate_id)
        pose = _reference(candidate.get("pose_sdf", candidate.get("pose")), "docked pose")
        if pose.media_type not in _LIGAND_MEDIA:
            raise ValueError("canonical docked pose must be an SDF/MOL2 artifact")
        store.resolve(pose)
        entry: dict[str, Any] = {
            "candidate_id": candidate_id,
            "molecule_id": candidate.get("molecule_id"),
            "microstate_id": candidate.get("microstate_id"),
            "docked_pose": pose.to_dict(),
            "posebusters": {
                "docked_ligand": pose.to_dict(),
                "docked_receptor": receptor.to_dict(),
            },
            "prolif": {
                "docked_ligand": pose.to_dict(),
                "docked_receptor": receptor.to_dict(),
            },
        }
        cofold_value = candidate.get("cofold_structure")
        if cofold_value is not None:
            cofold = _reference(cofold_value, "optional cofold structure")
            store.resolve(cofold)
            entry["cofold_pose"] = cofold.to_dict()
        if reference_pose is not None:
            entry["reference_pose"] = reference_pose.to_dict()
            entry["posebusters"]["reference_ligand"] = reference_pose.to_dict()
            entry["spyrmsd"] = {
                "reference_ligand": reference_pose.to_dict(),
                "predicted_ligand": pose.to_dict(),
            }
            entry["prolif"].update(
                {
                    "reference_ligand": reference_pose.to_dict(),
                    "reference_receptor": receptor.to_dict(),
                }
            )
        prepared.append(entry)

    return store.put_json(
        {
            "schema_version": "2.0",
            "kind": "protbind.validation-input-batch",
            "docking_bundle": docking_bundle.to_dict(),
            "reference_scope": (
                "VALIDATION_ONLY" if reference_pose is not None else "NOT_PROVIDED"
            ),
            "candidates": prepared,
        },
        producer="protbind.validation-input-builder",
        producer_version="2.0",
        source=docking_bundle.artifact_id,
    )


def _package_tree_sha256(root: Path) -> str:
    entries = [
        (path.relative_to(root).as_posix(), sha256_file(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    if not entries:
        raise ValueError(f"validation package has no attestable files: {root.name}")
    return sha256_bytes(canonical_json_bytes(entries))


def _installed_tool_pin(name: str) -> dict[str, str] | None:
    distribution_name, module_name = _TOOL_MODULES[name]
    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    spec = importlib.util.find_spec(module_name)
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if len(locations) != 1:
        raise ValueError(f"installed {name} module root is unavailable or ambiguous")
    package_root = Path(locations[0]).resolve()
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not package_root.is_relative_to(distribution_root):
        raise ValueError(f"installed {name} module is shadowed")
    return {
        "version": distribution.version,
        "package_source_sha256": _package_tree_sha256(package_root),
    }


def validation_worker_code_sha256(repository_root: Path) -> str:
    worker = repository_root / "workers" / "validation_worker.py"
    sources = [
        (str(path.relative_to(repository_root)), sha256_file(path))
        for path in sorted((repository_root / "src" / "protbind_agent").rglob("*.py"))
    ]
    runtime_sha = sha256_bytes(canonical_json_bytes(sources))
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "adapter_sha256": sha256_file(worker),
                "protbind_runtime_sha256": runtime_sha,
            }
        )
    )


def build_validation_toolchain(
    store: ArtifactStore,
    *,
    repository_root: Path,
    include_optional: bool = True,
) -> ValidationToolchainBinding:
    """Pin the exact installed validation packages used by the worker."""

    tools: dict[str, dict[str, str]] = {}
    for name in _TOOL_MODULES:
        pin = _installed_tool_pin(name)
        if pin is None:
            if name == "posebusters":
                raise ValueError("PoseBusters is required for validation")
            continue
        if name == "posebusters" or include_optional:
            tools[name] = pin
    artifact = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.validation-toolchain-manifest",
            "test_fixture": False,
            "posebusters_configs": ["dock", "redock"],
            "tools": tools,
            "assets": [],
        },
        producer="protbind.validation-toolchain-builder",
        producer_version="2.0",
    )
    return ValidationToolchainBinding(
        artifact=artifact,
        provenance=WorkerProvenance(
            model_revision=f"validation-toolchain:{artifact.sha256}",
            weight_sha256=artifact.sha256,
            code_sha256=validation_worker_code_sha256(repository_root),
        ),
    )
