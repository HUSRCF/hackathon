"""Identifier-only public protein and ligand acquisition through bounded curl."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import __version__
from .artifacts import (
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
)
from .models import ArtifactRef
from .privacy import redact_text, require_network_approval

PUBLIC_DATA_SCHEMA_VERSION = "1.0"
PUBLIC_DATA_SOURCES = (
    "rcsb-mmcif",
    "alphafold-mmcif",
    "uniprot-fasta",
    "rcsb-ccd-sdf",
    "pubchem-cid-sdf-3d",
)

_PDB_ID = re.compile(r"(?:[0-9][A-Z0-9]{3}|PDB_[A-Z0-9]{8})")
_UNIPROT_ACCESSION = re.compile(r"[A-Z0-9][A-Z0-9-]{5,15}")
_CCD_ID = re.compile(r"[A-Z0-9]{1,5}")
_PUBCHEM_CID = re.compile(r"[1-9][0-9]{0,11}")
_FASTA_SEQUENCE = re.compile(r"[A-Z*]+")
_CANONICAL_PROTEIN_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+\*?")
_METAL_ATOMIC_NUMBERS = frozenset(
    {3, 4, 11, 12, 13}
    | set(range(19, 33))
    | set(range(37, 52))
    | set(range(55, 85))
    | set(range(87, 119))
)
_STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


@dataclass(frozen=True, slots=True)
class PublicSourceSpec:
    source: str
    kind: str
    hostname: str
    media_type: str
    accept: str
    license: str | None
    max_bytes: int
    suffix: str


_SOURCE_SPECS = {
    "rcsb-mmcif": PublicSourceSpec(
        source="rcsb-mmcif",
        kind="protein_structure_candidate",
        hostname="files.rcsb.org",
        media_type="chemical/x-mmcif",
        accept="chemical/x-mmcif,text/plain;q=0.9",
        license="CC0-1.0",
        max_bytes=64 * 1024 * 1024,
        suffix=".cif",
    ),
    "alphafold-mmcif": PublicSourceSpec(
        source="alphafold-mmcif",
        kind="predicted_protein_structure_candidate",
        hostname="alphafold.ebi.ac.uk",
        media_type="chemical/x-mmcif",
        accept="chemical/x-mmcif,text/plain;q=0.9",
        license="CC-BY-4.0",
        max_bytes=64 * 1024 * 1024,
        suffix=".cif",
    ),
    "uniprot-fasta": PublicSourceSpec(
        source="uniprot-fasta",
        kind="protein_sequence",
        hostname="rest.uniprot.org",
        media_type="text/x-fasta",
        accept="text/plain; format=fasta",
        license="CC-BY-4.0",
        max_bytes=2 * 1024 * 1024,
        suffix=".fasta",
    ),
    "rcsb-ccd-sdf": PublicSourceSpec(
        source="rcsb-ccd-sdf",
        kind="small_molecule_ideal_coordinates",
        hostname="files.rcsb.org",
        media_type="chemical/x-mdl-sdfile",
        accept="chemical/x-mdl-sdfile,text/plain;q=0.9",
        license="CC0-1.0",
        max_bytes=8 * 1024 * 1024,
        suffix=".sdf",
    ),
    "pubchem-cid-sdf-3d": PublicSourceSpec(
        source="pubchem-cid-sdf-3d",
        kind="small_molecule_computed_3d_record",
        hostname="pubchem.ncbi.nlm.nih.gov",
        media_type="chemical/x-mdl-sdfile",
        accept="chemical/x-mdl-sdfile",
        # PubChem aggregates contributor records with source-specific terms.
        license=None,
        max_bytes=8 * 1024 * 1024,
        suffix=".sdf",
    ),
}


@dataclass(frozen=True, slots=True)
class CurlResult:
    data: bytes
    url: str
    http_status: int
    content_type: str | None
    elapsed_seconds: float
    size_download: int
    curl_version: str

    def receipt_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "http_status": self.http_status,
            "content_type": self.content_type,
            "elapsed_seconds": self.elapsed_seconds,
            "size_download": self.size_download,
            "curl_version": self.curl_version,
            "redirect_followed": False,
            "ambient_proxy_allowed": False,
        }


Runner = Callable[..., subprocess.CompletedProcess[str]]


class CurlTransport:
    """Direct HTTPS curl transport with no config, proxy, credentials, or redirects."""

    def __init__(
        self,
        *,
        curl_binary: str | None = None,
        runner: Runner = subprocess.run,
    ) -> None:
        binary = curl_binary or shutil.which("curl")
        if not binary:
            raise RuntimeError("curl is required for public data acquisition")
        self.curl_binary = binary
        self.runner = runner
        self._version: str | None = None

    @staticmethod
    def _environment() -> dict[str, str]:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C",
            "LC_ALL": "C",
        }
        for name in ("SSL_CERT_FILE", "CURL_CA_BUNDLE"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
        return environment

    def version(self) -> str:
        if self._version is not None:
            return self._version
        completed = self.runner(
            [self.curl_binary, "-q", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=self._environment(),
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            raise RuntimeError("curl version probe failed")
        first_line = completed.stdout.splitlines()[0].strip()
        if not first_line.startswith("curl "):
            raise RuntimeError("unexpected curl version response")
        self._version = first_line
        return first_line

    def request(
        self,
        url: str,
        *,
        accept: str,
        max_bytes: int,
    ) -> CurlResult:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("public data downloads require a constructed HTTPS URL")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("credentials and URL fragments are forbidden")
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        curl_version = self.version()
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="protbind-public-fetch-") as temporary:
            output = Path(temporary) / "response.bin"
            command = [
                self.curl_binary,
                "-q",
                "--fail",
                "--silent",
                "--show-error",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--tlsv1.2",
                "--noproxy",
                "*",
                "--connect-timeout",
                "15",
                "--max-time",
                "120",
                "--retry",
                "2",
                "--retry-connrefused",
                "--retry-delay",
                "1",
                "--retry-max-time",
                "60",
                "--max-filesize",
                str(max_bytes),
                "--header",
                f"Accept: {accept}",
                "--header",
                f"User-Agent: ProtBind/{__version__} bounded-public-fetch",
                "--output",
                str(output),
                "--write-out",
                "%{json}",
                url,
            ]
            completed = self.runner(
                command,
                capture_output=True,
                text=True,
                timeout=135,
                check=False,
                env=self._environment(),
            )
            elapsed = time.monotonic() - started
            if completed.returncode != 0:
                detail = redact_text(completed.stderr.strip())[:500]
                raise OSError(
                    f"curl public fetch failed with exit {completed.returncode}: "
                    f"{detail or 'no diagnostic'}"
                )
            try:
                transfer = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError("curl did not return parseable transfer metadata") from exc
            if not isinstance(transfer, dict):
                raise RuntimeError("curl transfer metadata must be an object")
            status = int(transfer.get("http_code", 0))
            effective_url = str(transfer.get("url_effective", ""))
            if status != 200:
                raise OSError(f"public data source returned HTTP {status}")
            if effective_url != url:
                raise PermissionError("curl effective URL differs from the approved URL")
            if not output.is_file():
                raise OSError("curl completed without a response file")
            data = output.read_bytes()
        if len(data) > max_bytes:
            raise ValueError("public data response exceeds its source-specific size limit")
        announced_size = int(float(transfer.get("size_download", len(data))))
        if announced_size != len(data):
            raise ValueError("curl transfer size does not match downloaded bytes")
        content_type_value = transfer.get("content_type")
        content_type = (
            str(content_type_value).split(";", maxsplit=1)[0].strip().lower()
            if content_type_value
            else None
        )
        return CurlResult(
            data=data,
            url=url,
            http_status=status,
            content_type=content_type,
            elapsed_seconds=round(elapsed, 6),
            size_download=len(data),
            curl_version=curl_version,
        )


@dataclass(frozen=True, slots=True)
class PropkaAudit:
    summary: Mapping[str, Any]
    output: bytes | None = None
    version: str | None = None


class PropkaAuditor:
    """Optional local PROPKA pKa/protonation audit over a temporary PDB view."""

    def __init__(
        self,
        *,
        python_executable: str = sys.executable,
        runner: Runner = subprocess.run,
        timeout_seconds: int = 120,
    ) -> None:
        self.python_executable = python_executable
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self._version: str | None = None

    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE", "1"),
        }

    def version(self) -> str | None:
        if self._version is not None:
            return self._version
        try:
            completed = self.runner(
                [self.python_executable, "-m", "propka", "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                env=self._environment(),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        version = (completed.stdout or completed.stderr).strip().splitlines()
        self._version = version[0][:120] if version else "PROPKA version unknown"
        return self._version

    def audit(self, mmcif_data: bytes) -> PropkaAudit:
        version = self.version()
        if version is None:
            return PropkaAudit(
                summary={
                    "status": "UNAVAILABLE",
                    "scientific_gate": False,
                    "reason": "PROPKA module is not available in the active local runtime.",
                }
            )
        try:
            import gemmi

            document = gemmi.cif.read_string(mmcif_data.decode("utf-8"))
            structure = gemmi.make_structure_from_block(document.sole_block())
            pdb_text = structure.make_minimal_pdb()
        except (ImportError, UnicodeDecodeError, ValueError, RuntimeError) as exc:
            return PropkaAudit(
                summary={
                    "status": "SKIPPED",
                    "scientific_gate": False,
                    "reason": redact_text(
                        f"temporary PDB conversion failed: {type(exc).__name__}: {exc}"
                    ),
                },
                version=version,
            )
        with tempfile.TemporaryDirectory(prefix="protbind-propka-audit-") as temporary:
            directory = Path(temporary)
            pdb_path = directory / "input.pdb"
            pdb_path.write_text(pdb_text, encoding="utf-8")
            try:
                completed = self.runner(
                    [
                        self.python_executable,
                        "-m",
                        "propka",
                        "--quiet",
                        str(pdb_path),
                    ],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                    env=self._environment(),
                )
            except subprocess.TimeoutExpired:
                return PropkaAudit(
                    summary={
                        "status": "FAILED",
                        "scientific_gate": False,
                        "reason": f"PROPKA exceeded {self.timeout_seconds} seconds.",
                    },
                    version=version,
                )
            except OSError as exc:
                return PropkaAudit(
                    summary={
                        "status": "FAILED",
                        "scientific_gate": False,
                        "reason": redact_text(f"PROPKA launch failed: {exc}"),
                    },
                    version=version,
                )
            diagnostics = [
                redact_text(line)[:500]
                for line in (completed.stdout + "\n" + completed.stderr).splitlines()
                if any(
                    marker in line.upper()
                    for marker in ("WARNING", "ERROR", "MISSING", "UNKNOWN", "IGNOR")
                )
            ][:50]
            output_path = directory / "input.pka"
            if completed.returncode != 0 or not output_path.is_file():
                return PropkaAudit(
                    summary={
                        "status": "FAILED",
                        "scientific_gate": False,
                        "returncode": completed.returncode,
                        "diagnostics": diagnostics,
                        "reason": (
                            "PROPKA did not produce a .pka report; inspect deterministic "
                            "Gemmi completeness checks and diagnostics."
                        ),
                    },
                    version=version,
                )
            output = output_path.read_bytes()
        return PropkaAudit(
            summary={
                "status": "SUCCEEDED",
                "scientific_gate": False,
                "returncode": 0,
                "diagnostics": diagnostics,
                "output_size_bytes": len(output),
                "output_sha256": sha256_bytes(output),
                "semantics": (
                    "PROPKA pKa/protonation feasibility and diagnostics only; success "
                    "does not prove structural completeness or docking readiness."
                ),
            },
            output=output,
            version=version,
        )


def _normalized_identifier(source: str, identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("public identifier must be a non-empty string")
    value = identifier.strip().upper()
    pattern = {
        "rcsb-mmcif": _PDB_ID,
        "alphafold-mmcif": _UNIPROT_ACCESSION,
        "uniprot-fasta": _UNIPROT_ACCESSION,
        "rcsb-ccd-sdf": _CCD_ID,
        "pubchem-cid-sdf-3d": _PUBCHEM_CID,
    }.get(source)
    if pattern is None:
        raise ValueError(f"unsupported public data source: {source}")
    if pattern.fullmatch(value) is None:
        raise ValueError(f"identifier is invalid for source {source}")
    return value


def suggested_filename(source: str, identifier: str) -> str:
    value = _normalized_identifier(source, identifier)
    names = {
        "rcsb-mmcif": f"rcsb-{value.lower()}.cif",
        "alphafold-mmcif": f"afdb-{value}.cif",
        "uniprot-fasta": f"uniprot-{value}.fasta",
        "rcsb-ccd-sdf": f"rcsb-ccd-{value}-ideal.sdf",
        "pubchem-cid-sdf-3d": f"pubchem-cid-{value}-3d.sdf",
    }
    return names[source]


def required_domain(source: str) -> str:
    try:
        return _SOURCE_SPECS[source].hostname
    except KeyError as exc:
        raise ValueError(f"unsupported public data source: {source}") from exc


def _direct_url(source: str, identifier: str) -> str:
    if source == "rcsb-mmcif":
        return f"https://files.rcsb.org/download/{identifier}.cif"
    if source == "uniprot-fasta":
        return f"https://rest.uniprot.org/uniprotkb/{identifier}.fasta"
    if source == "rcsb-ccd-sdf":
        return f"https://files.rcsb.org/ligands/download/{identifier}_ideal.sdf"
    if source == "pubchem-cid-sdf-3d":
        return (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{identifier}/SDF?record_type=3d"
        )
    raise ValueError(f"source {source} requires metadata discovery")


def _validate_mmcif(data: bytes) -> dict[str, Any]:
    try:
        import gemmi
    except ImportError as exc:
        raise RuntimeError("Gemmi is required to validate downloaded mmCIF") from exc
    try:
        text = data.decode("utf-8")
        document = gemmi.cif.read_string(text)
        block = document.sole_block()
        structure = gemmi.make_structure_from_block(block)
    except (UnicodeDecodeError, ValueError, RuntimeError) as exc:
        raise ValueError("downloaded protein structure is not valid mmCIF") from exc
    atom_count = 0
    chain_ids: set[str] = set()
    model_count = 0
    alternate_location_atom_count = 0
    zero_occupancy_atom_count = 0
    missing_backbone: list[str] = []
    for model in structure:
        model_count += 1
        for chain in model:
            chain_ids.add(str(chain.name))
            for residue in chain:
                atom_names: set[str] = set()
                for atom in residue:
                    atom_count += 1
                    atom_names.add(str(atom.name).strip())
                    altloc = str(atom.altloc).strip()
                    if altloc not in {"", ".", "?", "\x00"}:
                        alternate_location_atom_count += 1
                    if float(atom.occ) <= 0:
                        zero_occupancy_atom_count += 1
                if str(residue.name).upper() in _STANDARD_AMINO_ACIDS:
                    absent = sorted(_BACKBONE_ATOMS - atom_names)
                    if absent:
                        missing_backbone.append(
                            f"{chain.name}:{residue.name}{residue.seqid.num}:"
                            + ",".join(absent)
                        )
    if atom_count == 0:
        raise ValueError("downloaded mmCIF contains no atoms")
    try:
        declared_unobserved_residues = len(
            block.find_mmcif_category("_pdbx_unobs_or_zero_occ_residues.")
        )
        declared_unobserved_atoms = len(
            block.find_mmcif_category("_pdbx_unobs_or_zero_occ_atoms.")
        )
    except (AttributeError, RuntimeError):
        declared_unobserved_residues = 0
        declared_unobserved_atoms = 0
    return {
        "parser": f"Gemmi {gemmi.__version__}",
        "parse_valid": True,
        "model_count": model_count,
        "chain_count": len(chain_ids),
        "atom_count": atom_count,
        "entry_id": str(block.find_value("_entry.id") or ""),
        "completeness": {
            "declared_unobserved_residue_count": declared_unobserved_residues,
            "declared_unobserved_atom_count": declared_unobserved_atoms,
            "standard_residue_missing_backbone_or_carbonyl_count": len(
                missing_backbone
            ),
            "standard_residue_missing_backbone_or_carbonyl_examples": missing_backbone[
                :50
            ],
            "alternate_location_atom_count": alternate_location_atom_count,
            "zero_occupancy_atom_count": zero_occupancy_atom_count,
            "no_detected_completeness_findings": (
                declared_unobserved_residues == 0
                and declared_unobserved_atoms == 0
                and not missing_backbone
                and alternate_location_atom_count == 0
                and zero_occupancy_atom_count == 0
            ),
            "scientific_gate": False,
        },
        "scientific_gate": False,
        "semantics": (
            "Parsed public structure candidate only; target identity, chain selection, "
            "metal/covalent checks, and receptor QC remain mandatory."
        ),
    }


def _validate_fasta(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("downloaded FASTA is not UTF-8 text") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError("downloaded protein sequence is not FASTA")
    if sum(line.startswith(">") for line in lines) != 1:
        raise ValueError("single-identifier fetch must return exactly one FASTA record")
    sequence = "".join(line for line in lines[1:] if not line.startswith(">")).upper()
    if not sequence or _FASTA_SEQUENCE.fullmatch(sequence) is None:
        raise ValueError("downloaded FASTA contains invalid protein sequence symbols")
    canonical = _CANONICAL_PROTEIN_SEQUENCE.fullmatch(sequence) is not None
    return {
        "parser": "ProtBind strict single-record FASTA",
        "parse_valid": True,
        "sequence_length": len(sequence.removesuffix("*")),
        "canonical_20_amino_acids": canonical,
        "v1_length_supported": len(sequence.removesuffix("*")) <= 700,
        "v1_case_compatible": canonical and len(sequence.removesuffix("*")) <= 700,
        "header": lines[0][1:256],
        "scientific_gate": False,
        "semantics": (
            "Sequence acquisition only; accession identity and downstream structure "
            "quality are separate gates."
        ),
    }


def _validate_sdf(data: bytes, *, require_3d: bool) -> dict[str, Any]:
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise RuntimeError("RDKit is required to validate downloaded SDF") from exc
    supplier = Chem.ForwardSDMolSupplier(
        BytesIO(data),
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    molecules = [molecule for molecule in supplier if molecule is not None]
    if len(molecules) != 1:
        raise ValueError("single-identifier SDF fetch must contain one valid molecule")
    molecule = molecules[0]
    if molecule.GetNumConformers() != 1:
        raise ValueError("downloaded SDF must contain one coordinate conformer")
    conformer = molecule.GetConformer()
    is_3d = bool(conformer.Is3D())
    if require_3d and not is_3d:
        raise ValueError("downloaded SDF does not declare a 3D conformer")
    heavy_atoms = sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms())
    metals = sorted(
        {
            atom.GetSymbol()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() in _METAL_ATOMIC_NUMBERS
        }
    )
    fragments = len(Chem.GetMolFrags(molecule))
    unassigned_stereocenters = sum(
        label == "?"
        for _index, label in Chem.FindMolChiralCenters(
            molecule,
            includeUnassigned=True,
            useLegacyImplementation=False,
        )
    )
    unsupported_reasons: list[str] = []
    if metals:
        unsupported_reasons.append("metal-containing ligand")
    if heavy_atoms > 100:
        unsupported_reasons.append("more than 100 heavy atoms")
    if fragments != 1:
        unsupported_reasons.append("multi-fragment record")
    if not is_3d:
        unsupported_reasons.append("no declared 3D conformer")
    return {
        "parser": f"RDKit {rdBase.rdkitVersion}",
        "parse_valid": True,
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": heavy_atoms,
        "fragment_count": fragments,
        "declared_3d": is_3d,
        "formal_charge": sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
        "metal_elements": metals,
        "unassigned_stereocenters": unassigned_stereocenters,
        "v1_ordinary_ligand_candidate": not unsupported_reasons,
        "unsupported_reasons": unsupported_reasons,
        "scientific_gate": False,
        "semantics": (
            "Raw public small-molecule record only; ProtBind standardization, microstate, "
            "stereochemistry, covalency, and parameterization gates still apply."
        ),
    }


def _validate_download(spec: PublicSourceSpec, data: bytes) -> dict[str, Any]:
    if spec.media_type == "chemical/x-mmcif":
        return _validate_mmcif(data)
    if spec.media_type == "text/x-fasta":
        return _validate_fasta(data)
    if spec.media_type == "chemical/x-mdl-sdfile":
        return _validate_sdf(data, require_3d=True)
    raise AssertionError(f"no validator for {spec.media_type}")


def _alphafold_model_url(
    identifier: str,
    metadata: bytes,
) -> tuple[str, dict[str, Any]]:
    try:
        value = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise ValueError("AlphaFold DB metadata response is not JSON") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("AlphaFold DB has no prediction for this accession")
    candidates = [
        item
        for item in value
        if isinstance(item, dict)
        and str(item.get("uniprotAccession", "")).upper() == identifier
        and item.get("isComplex") is not True
    ]
    if not candidates:
        raise ValueError("AlphaFold DB metadata has no matching monomer accession")
    candidate = max(candidates, key=lambda item: int(item.get("latestVersion", 0)))
    url = str(candidate.get("cifUrl", ""))
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "alphafold.ebi.ac.uk"
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(".cif")
    ):
        raise PermissionError("AlphaFold DB returned an unexpected model URL")
    expected_prefix = f"/files/AF-{identifier}-F"
    if not parsed.path.startswith(expected_prefix):
        raise ValueError("AlphaFold DB model URL is not bound to the requested accession")
    safe_metadata = {
        key: candidate.get(key)
        for key in (
            "entryId",
            "modelEntityId",
            "toolUsed",
            "providerId",
            "modelCreatedDate",
            "latestVersion",
            "sequenceChecksum",
            "sequenceStart",
            "sequenceEnd",
            "globalMetricValue",
            "uniprotAccession",
            "uniprotId",
            "isUniProtReviewed",
        )
    }
    safe_metadata["cifUrl"] = url
    return url, safe_metadata


@dataclass(frozen=True, slots=True)
class PublicFetchResult:
    source: str
    identifier: str
    filename: str
    artifact: ArtifactRef
    receipt: ArtifactRef
    metadata_artifact: ArtifactRef | None
    propka_artifact: ArtifactRef | None
    validation: Mapping[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "identifier": self.identifier,
            "suggested_filename": self.filename,
            "artifact": self.artifact.to_dict(),
            "receipt": self.receipt.to_dict(),
            "metadata_artifact": (
                self.metadata_artifact.to_dict()
                if self.metadata_artifact is not None
                else None
            ),
            "propka_artifact": (
                self.propka_artifact.to_dict()
                if self.propka_artifact is not None
                else None
            ),
            "validation": dict(self.validation),
            "warnings": list(self.warnings),
        }


class PublicDataFetcher:
    def __init__(
        self,
        workspace: Path,
        *,
        transport: CurlTransport | None = None,
        propka: PropkaAuditor | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.artifacts = ArtifactStore(workspace)
        self.transport = transport or CurlTransport()
        self.propka = propka or PropkaAuditor()
        self.now = now or (lambda: datetime.now(UTC))

    def fetch(
        self,
        *,
        source: str,
        identifier: str,
        approved_domains: Sequence[str],
        run_propka: bool = True,
    ) -> PublicFetchResult:
        normalized = _normalized_identifier(source, identifier)
        spec = _SOURCE_SPECS[source]
        approved = tuple(approved_domains)
        retrievals: list[CurlResult] = []
        metadata_bytes: bytes | None = None
        safe_remote_metadata: dict[str, Any] | None = None
        if source == "alphafold-mmcif":
            metadata_url = (
                f"https://alphafold.ebi.ac.uk/api/prediction/{normalized}"
            )
            require_network_approval(metadata_url, approved)
            metadata_result = self.transport.request(
                metadata_url,
                accept="application/json",
                max_bytes=2 * 1024 * 1024,
            )
            retrievals.append(metadata_result)
            metadata_bytes = metadata_result.data
            model_url, safe_remote_metadata = _alphafold_model_url(
                normalized,
                metadata_bytes,
            )
            require_network_approval(model_url, approved)
            data_result = self.transport.request(
                model_url,
                accept=spec.accept,
                max_bytes=spec.max_bytes,
            )
        else:
            url = _direct_url(source, normalized)
            require_network_approval(url, approved)
            data_result = self.transport.request(
                url,
                accept=spec.accept,
                max_bytes=spec.max_bytes,
            )
        retrievals.append(data_result)
        validation = _validate_download(spec, data_result.data)
        propka_audit: PropkaAudit | None = None
        propka_artifact: ArtifactRef | None = None
        if spec.media_type == "chemical/x-mmcif" and run_propka:
            propka_audit = self.propka.audit(data_result.data)
            if propka_audit.output is not None:
                propka_artifact = self.artifacts.put_bytes(
                    propka_audit.output,
                    media_type="text/plain",
                    producer="protbind.propka-audit",
                    producer_version=propka_audit.version or "unknown",
                    source=data_result.url,
                    license=spec.license,
                )
            validation = {
                **validation,
                "propka": {
                    **dict(propka_audit.summary),
                    "artifact": (
                        propka_artifact.to_dict()
                        if propka_artifact is not None
                        else None
                    ),
                    "tool_version": propka_audit.version,
                },
            }
        elif spec.media_type == "chemical/x-mmcif":
            validation = {
                **validation,
                "propka": {
                    "status": "SKIPPED",
                    "scientific_gate": False,
                    "reason": "disabled by explicit acquisition option",
                },
            }
        metadata_artifact = (
            self.artifacts.put_bytes(
                metadata_bytes,
                media_type="application/json",
                producer="protbind.public-data-fetch",
                producer_version=__version__,
                source=retrievals[0].url,
                license=spec.license,
            )
            if metadata_bytes is not None
            else None
        )
        artifact = self.artifacts.put_bytes(
            data_result.data,
            media_type=spec.media_type,
            producer="protbind.public-data-fetch",
            producer_version=__version__,
            source=data_result.url,
            license=spec.license,
        )
        warnings = []
        if source == "pubchem-cid-sdf-3d":
            warnings.append(
                "PubChem aggregates contributor records; no uniform license is asserted. "
                "Review source-specific provenance before redistribution."
            )
        request_identity = {
            "schema_version": PUBLIC_DATA_SCHEMA_VERSION,
            "source": source,
            "identifier": normalized,
            "hostname": spec.hostname,
            "media_type": spec.media_type,
        }
        receipt_value = {
            "schema_version": PUBLIC_DATA_SCHEMA_VERSION,
            "kind": "protbind.public-data-fetch",
            "request": request_identity,
            "request_sha256": sha256_bytes(canonical_json_bytes(request_identity)),
            "privacy": {
                "sent_data": "public identifier only",
                "private_sequence_uploaded": False,
                "credentials_used": False,
                "approved_exact_domains": sorted(set(approved)),
            },
            "transport": {
                "implementation": "curl",
                "policy": (
                    "HTTPS only; ambient proxy disabled; no redirects; no arbitrary URL; "
                    "source-specific size and timeout limits"
                ),
                "retrievals": [item.receipt_dict() for item in retrievals],
            },
            "result": {
                "artifact": artifact.to_dict(),
                "metadata_artifact": (
                    metadata_artifact.to_dict()
                    if metadata_artifact is not None
                    else None
                ),
                "propka_artifact": (
                    propka_artifact.to_dict()
                    if propka_artifact is not None
                    else None
                ),
                "validation": validation,
                "remote_metadata": safe_remote_metadata,
            },
            "retrieved_at": self.now().isoformat(),
            "license": spec.license,
            "warnings": warnings,
            "scientific_semantics": (
                "Acquisition and parse validation only. This receipt does not establish "
                "target identity, receptor suitability, ligand support, or binding."
            ),
        }
        receipt = self.artifacts.put_json(
            receipt_value,
            producer="protbind.public-data-receipt",
            producer_version=__version__,
            source=data_result.url,
            license=spec.license,
        )
        return PublicFetchResult(
            source=source,
            identifier=normalized,
            filename=suggested_filename(source, normalized),
            artifact=artifact,
            receipt=receipt,
            metadata_artifact=metadata_artifact,
            propka_artifact=propka_artifact,
            validation=validation,
            warnings=tuple(warnings),
        )


def validate_public_output(
    source: str,
    project_root: Path,
    relative_output: Path,
) -> Path:
    """Resolve and validate a source-bound output before any network request."""

    try:
        expected_suffix = _SOURCE_SPECS[source].suffix
    except KeyError as exc:
        raise ValueError(f"unsupported public data source: {source}") from exc
    root = project_root.resolve()
    if relative_output.is_absolute() or not relative_output.parts:
        raise ValueError("output must be a non-empty project-relative path")
    path = (root / relative_output).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("output escapes the configured project root")
    if path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"output for {source} must use the {expected_suffix} suffix"
        )
    return path


def materialize_public_fetch(
    result: PublicFetchResult,
    store: ArtifactStore,
    *,
    project_root: Path,
    output: Path,
    replace: bool = False,
) -> dict[str, Any]:
    """Atomically materialize verified bytes plus a coordinate-free provenance sidecar."""

    destination = validate_public_output(result.source, project_root, output)
    data = store.read_bytes(result.artifact)
    sidecar = destination.with_name(destination.name + ".protbind.json")
    if destination.exists() and destination.read_bytes() != data and not replace:
        raise FileExistsError("output exists with different bytes; pass replace=True")
    sidecar_value = {
        "schema_version": PUBLIC_DATA_SCHEMA_VERSION,
        "kind": "protbind.public-data-materialization",
        "source": result.source,
        "identifier": result.identifier,
        "artifact": result.artifact.to_dict(),
        "receipt": result.receipt.to_dict(),
        "metadata_artifact": (
            result.metadata_artifact.to_dict()
            if result.metadata_artifact is not None
            else None
        ),
        "propka_artifact": (
            result.propka_artifact.to_dict()
            if result.propka_artifact is not None
            else None
        ),
        "output_filename": destination.name,
        "validation": dict(result.validation),
        "warnings": list(result.warnings),
    }
    sidecar_data = (
        json.dumps(
            sidecar_value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    if sidecar.exists() and sidecar.read_bytes() != sidecar_data and not replace:
        raise FileExistsError(
            "provenance sidecar exists with different content; pass replace=True"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, data)
    _atomic_write(sidecar, sidecar_data)
    return {
        "output": str(output),
        "sidecar": str(output.with_name(output.name + ".protbind.json")),
        **result.to_dict(),
    }


def _atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
