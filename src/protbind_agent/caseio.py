"""Trusted CLI-boundary ingestion of local case files into the artifact store."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .models import ArtifactRef, ResearchCase, SiteProvenanceKind
from .selection import build_site_derivation_evidence


def _media_type(path: Path, *, pharmacophore: bool = False) -> str:
    if pharmacophore:
        return "application/json"
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return "chemical/x-pdb"
    if suffix in {".cif", ".mmcif"}:
        return "chemical/x-mmcif"
    if suffix in {".sdf", ".sd"}:
        return "chemical/x-mdl-sdfile"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _import_file_field(
    section: dict[str, Any],
    key: str,
    store: ArtifactStore,
    *,
    pharmacophore: bool = False,
    input_root: Path | None = None,
) -> None:
    file_key = f"{key}_file"
    value = section.pop(file_key, None)
    if value is None:
        return
    if section.get(key) is not None:
        raise ValueError(f"case cannot specify both {key!r} and {file_key!r}")
    path = Path(str(value))
    if input_root is not None and not path.is_absolute():
        path = input_root / path
    if not path.is_file():
        raise FileNotFoundError(f"case input file not found: {path.name}")
    section[key] = store.import_file(
        path,
        media_type=_media_type(path, pharmacophore=pharmacophore),
        producer="protbind.case-import",
        producer_version="0.1.0",
    ).to_dict()


def ingest_case(
    path: Path,
    store: ArtifactStore,
    *,
    input_root: Path | None = None,
) -> ResearchCase:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("case file must contain one JSON object")
    payload = dict(value)
    target = dict(payload.get("target", {}))
    _import_file_field(target, "structure", store, input_root=input_root)
    payload["target"] = target
    if payload.get("ligand") is not None:
        ligand = dict(payload["ligand"])
        _import_file_field(ligand, "structure", store, input_root=input_root)
        _import_file_field(
            ligand,
            "pharmacophore",
            store,
            pharmacophore=True,
            input_root=input_root,
        )
        payload["ligand"] = ligand
    if payload.get("pocket") is not None:
        pocket = dict(payload["pocket"])
        _import_file_field(
            pocket,
            "pharmacophore",
            store,
            pharmacophore=True,
            input_root=input_root,
        )
        if "site_evidence_file" in pocket:
            raise ValueError(
                "prebuilt site_evidence_file JSON is not trusted; use typed "
                "site_derivation_source_files"
            )
        source_files = pocket.pop("site_derivation_source_files", None)
        derivation_method = pocket.pop("site_derivation_method", None)
        derivation_license = pocket.pop("site_derivation_license", None)
        if source_files is not None:
            if pocket.get("site_evidence") is not None:
                raise ValueError(
                    "case cannot combine site_evidence and site_derivation_source_files"
                )
            if (
                not isinstance(source_files, list)
                or not source_files
                or any(not isinstance(item, str) or not item for item in source_files)
            ):
                raise ValueError("site_derivation_source_files must be a non-empty path list")
            target_structure = target.get("structure")
            if target_structure is None:
                raise ValueError(
                    "typed site derivation currently requires a local target structure"
                )
            if pocket.get("center") is None or pocket.get("box_size") is None:
                raise ValueError(
                    "typed site derivation currently requires explicit center and box_size"
                )
            if pocket.get("site_provenance_kind") is None:
                raise ValueError(
                    "typed site derivation requires an explicit site_provenance_kind"
                )
            sources: list[ArtifactRef] = []
            for raw_path in source_files:
                source_path = Path(raw_path)
                if input_root is not None and not source_path.is_absolute():
                    source_path = input_root / source_path
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"site derivation source file not found: {source_path.name}"
                    )
                sources.append(
                    store.import_file(
                        source_path,
                        media_type=_media_type(source_path),
                        producer="protbind.case-site-source-import",
                        producer_version="0.1.0",
                        license=(
                            str(derivation_license)
                            if derivation_license is not None
                            else None
                        ),
                    )
                )
            evidence = build_site_derivation_evidence(
                store,
                receptor=ArtifactRef.from_dict(target_structure),
                source_kind=SiteProvenanceKind(pocket["site_provenance_kind"]),
                center=pocket["center"],
                size=pocket["box_size"],
                derivation_method=str(derivation_method or ""),
                source_artifacts=tuple(sources),
                license=(
                    str(derivation_license)
                    if derivation_license is not None
                    else None
                ),
            )
            pocket["site_evidence"] = evidence.to_dict()
        elif derivation_method is not None or derivation_license is not None:
            raise ValueError(
                "site_derivation_method/license requires site_derivation_source_files"
            )
        payload["pocket"] = pocket
    return ResearchCase.from_dict(payload)
