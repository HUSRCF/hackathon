"""Local PDF extraction and seekdb-backed, citation-preserving retrieval."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, canonical_json_bytes, sha256_file
from .library import LibraryManager
from .models import ArtifactRef


class KnowledgeCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    section: str | None
    page: int | None


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    chunks: tuple[DocumentChunk, ...]
    receipt: dict[str, Any]


_EMBEDDING_MODELS = {
    "BAAI/bge-m3": {
        "dimension": 1024,
        "backend": "FlagEmbedding",
        "minimum_transformers": None,
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "dimension": 1024,
        "backend": "sentence-transformers",
        "minimum_transformers": "4.51.0",
    },
}


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", value)[:3])


def inspect_embedding_model(model_path: Path | None) -> dict[str, Any]:
    """Report whether a pinned local embedding model can be admitted offline."""

    runtime: dict[str, str | None] = {}
    for package in ("transformers", "sentence-transformers", "FlagEmbedding"):
        try:
            runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime[package] = None
    if model_path is None:
        return {
            "status": "NOT_CONFIGURED",
            "supported_models": sorted(_EMBEDDING_MODELS),
            "runtime": runtime,
            "network": "disabled",
            "gpu_policy": "CPU by default; scientific GPU capacity is not reserved",
        }
    try:
        identity = _validate_model_manifest(model_path)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "status": "REJECTED",
            "reason": str(exc),
            "runtime": runtime,
            "network": "disabled",
        }
    specification = _EMBEDDING_MODELS[identity["model_name"]]
    minimum = specification["minimum_transformers"]
    if minimum is not None:
        installed = runtime["transformers"]
        if installed is None or _version_tuple(installed) < _version_tuple(minimum):
            return {
                **identity,
                "status": "BLOCKED_RUNTIME_COMPATIBILITY",
                "reason": (
                    f"{identity['model_name']} requires transformers>={minimum}; "
                    f"the current environment has {installed or 'none'}"
                ),
                "runtime": runtime,
                "network": "disabled",
            }
    return {
        **identity,
        "status": "ADMITTED",
        "dimension": specification["dimension"],
        "backend": specification["backend"],
        "runtime": runtime,
        "network": "disabled",
        "gpu_policy": "CPU by default; opt-in acceleration requires a separate benchmark",
    }


def _validate_model_manifest(model_path: Path) -> dict[str, str]:
    if not model_path.is_dir() or model_path.is_symlink():
        raise FileNotFoundError(
            f"local embedding model directory not found: {model_path.name}"
        )
    manifest_path = model_path / "protbind-model-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "local embedding model requires protbind-model-manifest.json with file hashes"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("embedding model manifest is invalid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "1.0"
        or manifest.get("model_name") not in _EMBEDDING_MODELS
        or not isinstance(manifest.get("model_revision"), str)
        or not manifest["model_revision"].strip()
        or not isinstance(manifest.get("files"), dict)
        or not manifest["files"]
    ):
        raise ValueError("embedding model manifest has an invalid identity schema")
    for relative_name, expected_sha256 in manifest["files"].items():
        relative = Path(str(relative_name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("embedding manifest file paths must stay inside the model")
        candidate = model_path / relative
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or not isinstance(expected_sha256, str)
            or sha256_file(candidate) != expected_sha256
        ):
            raise ValueError(f"embedding model file hash mismatch: {relative.name}")
    return {
        "model_name": str(manifest["model_name"]),
        "model_revision": str(manifest["model_revision"]),
        "manifest_sha256": sha256_file(manifest_path),
    }


def freeze_embedding_model_manifest(
    model_path: Path,
    *,
    model_name: str,
    model_revision: str,
    replace: bool = False,
    max_files: int = 10_000,
) -> dict[str, Any]:
    """Hash a reviewed local model directory without downloading or loading it."""

    if model_name not in _EMBEDDING_MODELS:
        raise ValueError("unsupported embedding model identity")
    if not model_revision.strip() or len(model_revision) > 200:
        raise ValueError("model_revision must be a non-empty value of at most 200 characters")
    root = model_path.resolve()
    if not root.is_dir() or model_path.is_symlink():
        raise FileNotFoundError("embedding model directory is missing or is a symbolic link")
    destination = root / "protbind-model-manifest.json"
    if destination.exists() and not replace:
        raise FileExistsError(
            "embedding model manifest already exists; use --replace only after review"
        )
    candidates = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path != destination
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if len(candidates) > max_files:
        raise ValueError("embedding model directory exceeds max_files")
    if not candidates:
        raise ValueError("embedding model directory contains no weight/config files")
    files: dict[str, str] = {}
    total_bytes = 0
    for path in candidates:
        relative = path.relative_to(root)
        parent_symlink = any(
            parent.is_symlink() for parent in path.parents if parent != root
        )
        if path.is_symlink() or parent_symlink:
            raise ValueError(f"embedding model contains a symbolic link: {relative.name}")
        files[relative.as_posix()] = sha256_file(path)
        total_bytes += path.stat().st_size
    manifest = {
        "schema_version": "1.0",
        "model_name": model_name,
        "model_revision": model_revision.strip(),
        "files": files,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(manifest) + b"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    return {
        "model_name": model_name,
        "model_revision": model_revision.strip(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "manifest_sha256": sha256_file(destination),
        "network": "disabled",
        "model_loaded": False,
        "path_disclosed": False,
    }


class LocalBGEM3EmbeddingFunction:
    """pyseekdb-compatible dense BGE-M3 function that never downloads weights."""

    def __init__(self, model_path: Path) -> None:
        identity = _validate_model_manifest(model_path)
        if identity["model_name"] != "BAAI/bge-m3":
            raise ValueError("LocalBGEM3EmbeddingFunction requires BAAI/bge-m3")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise KnowledgeCapabilityError(
                "FlagEmbedding is required for the local BGE-M3 knowledge backend"
            ) from exc
        self.model_path = model_path
        self.model_revision = identity["model_revision"]
        self.manifest_sha256 = identity["manifest_sha256"]
        self.model = BGEM3FlagModel(str(model_path), use_fp16=True)

    def __call__(self, input: str | list[str]) -> list[list[float]]:
        values = [input] if isinstance(input, str) else input
        encoded = self.model.encode(values, return_dense=True)
        return [[float(value) for value in row] for row in encoded["dense_vecs"]]

    @property
    def dimension(self) -> int:
        return 1024

    @staticmethod
    def name() -> str:
        return "protbind_local_bge_m3"

    def get_config(self) -> dict[str, str]:
        return {
            "model": "BAAI/bge-m3",
            "model_revision": self.model_revision,
            "weight_manifest_sha256": self.manifest_sha256,
            "network": "disabled",
        }


class LocalQwen3EmbeddingFunction:
    """Offline Qwen3-Embedding-0.6B adapter, CPU-first to preserve Radeon capacity."""

    def __init__(self, model_path: Path) -> None:
        admission = inspect_embedding_model(model_path)
        if admission["status"] != "ADMITTED":
            raise KnowledgeCapabilityError(str(admission.get("reason", admission["status"])))
        if admission["model_name"] != "Qwen/Qwen3-Embedding-0.6B":
            raise ValueError("Qwen adapter requires Qwen/Qwen3-Embedding-0.6B")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise KnowledgeCapabilityError(
                "sentence-transformers is required for local Qwen3 embeddings"
            ) from exc
        self.model_path = model_path
        self.model_revision = str(admission["model_revision"])
        self.manifest_sha256 = str(admission["manifest_sha256"])
        self.model = SentenceTransformer(
            str(model_path),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
        )

    def __call__(self, input: str | list[str]) -> list[list[float]]:
        values = [input] if isinstance(input, str) else input
        encoded = self.model.encode(values, normalize_embeddings=True)
        return [[float(value) for value in row] for row in encoded]

    @property
    def dimension(self) -> int:
        return 1024

    @staticmethod
    def name() -> str:
        return "protbind_local_qwen3_embedding_0_6b"

    def get_config(self) -> dict[str, str]:
        return {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "model_revision": self.model_revision,
            "weight_manifest_sha256": self.manifest_sha256,
            "network": "disabled",
            "device": "cpu",
        }


def _embedding_function(model_path: Path) -> Any:
    admission = inspect_embedding_model(model_path)
    if admission["status"] != "ADMITTED":
        raise KnowledgeCapabilityError(str(admission.get("reason", admission["status"])))
    if admission["model_name"] == "BAAI/bge-m3":
        return LocalBGEM3EmbeddingFunction(model_path)
    return LocalQwen3EmbeddingFunction(model_path)


def _markdown_chunks(text: str, *, max_chars: int = 1800) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(buffer).strip()
        while content:
            piece = content[:max_chars]
            if len(content) > max_chars:
                split = piece.rfind("\n")
                if split > max_chars // 2:
                    piece = piece[:split]
            digest = hashlib.sha256(
                f"{section or ''}\0{len(chunks)}\0{piece}".encode()
            ).hexdigest()[:20]
            chunks.append(
                DocumentChunk(
                    chunk_id=digest,
                    text=piece.strip(),
                    section=section,
                    page=None,
                )
            )
            content = content[len(piece) :].lstrip()
        buffer = []

    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            flush()
            section = heading.group(1)
        else:
            buffer.append(line)
            if sum(len(item) + 1 for item in buffer) >= max_chars:
                flush()
    flush()
    return chunks


def _command_version(binary: str, *arguments: str) -> str | None:
    executable = shutil.which(binary)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "available; version probe failed"
    value = (result.stdout or result.stderr).splitlines()
    return value[0][:160] if value else "available; version unknown"


def _poppler_pages(data: bytes, *, expected_pages: int) -> list[str] | None:
    executable = shutil.which("pdftotext")
    if executable is None:
        return None
    with tempfile.TemporaryDirectory(prefix="protbind-pdf-") as temporary:
        source = Path(temporary) / "input.pdf"
        source.write_bytes(data)
        try:
            result = subprocess.run(
                [executable, "-layout", "-enc", "UTF-8", str(source), "-"],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise KnowledgeCapabilityError(f"pdftotext extraction failed: {exc}") from exc
    pages = result.stdout.decode("utf-8", errors="replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    pages.extend("" for _ in range(max(0, expected_pages - len(pages))))
    return pages[:expected_pages]


def _ocr_page(page: Any, *, language: str, timeout_seconds: int) -> str:
    executable = shutil.which("tesseract")
    if executable is None:
        raise KnowledgeCapabilityError(
            "OCR was requested but Tesseract is not installed; no OCR fallback was fabricated"
        )
    with tempfile.TemporaryDirectory(prefix="protbind-ocr-") as temporary:
        image = Path(temporary) / "page.png"
        page.get_pixmap(dpi=200, alpha=False).save(image)
        try:
            result = subprocess.run(
                [executable, str(image), "stdout", "-l", language, "--psm", "3"],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise KnowledgeCapabilityError(f"Tesseract OCR failed: {exc}") from exc
    return result.stdout.strip()


def extract_document_bytes(
    data: bytes,
    *,
    suffix: str,
    pdf_backend: str = "auto",
    ocr: str = "off",
    ocr_language: str = "eng",
    max_chars: int = 1800,
    max_pdf_bytes: int = 100 * 1024 * 1024,
    max_pdf_pages: int = 500,
    max_ocr_pages: int = 50,
    scan_text_threshold: int = 40,
) -> DocumentExtraction:
    if pdf_backend not in {"auto", "pymupdf", "pdftotext"}:
        raise ValueError("pdf_backend must be auto, pymupdf, or pdftotext")
    if ocr not in {"off", "auto", "required"}:
        raise ValueError("ocr must be off, auto, or required")
    if not re.fullmatch(r"[A-Za-z0-9_+-]{1,32}", ocr_language):
        raise ValueError("ocr_language contains unsupported characters")
    if suffix.lower() != ".pdf":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "knowledge import supports UTF-8 text/Markdown or PDF"
            ) from exc
        chunks = _markdown_chunks(text, max_chars=max_chars)
        return DocumentExtraction(
            chunks=tuple(chunks),
            receipt={
                "schema_version": "1.0",
                "media_type": "text/markdown",
                "backend": "utf-8",
                "page_count": 0,
                "chunk_count": len(chunks),
                "citation_policy": "section when present",
                "warnings": [],
            },
        )
    if len(data) > max_pdf_bytes:
        raise ValueError("PDF exceeds the configured extraction byte limit")
    try:
        import pymupdf
    except ImportError as exc:
        raise KnowledgeCapabilityError(
            "PyMuPDF is required to import PDF evidence with page citations"
        ) from exc
    document = pymupdf.open(stream=data, filetype="pdf")
    try:
        if document.page_count > max_pdf_pages:
            raise ValueError("PDF exceeds the configured page limit")
        native_pages = [page.get_text("text").strip() for page in document]
        poppler_pages = (
            _poppler_pages(data, expected_pages=document.page_count)
            if pdf_backend in {"auto", "pdftotext"}
            else None
        )
        if pdf_backend == "pdftotext" and poppler_pages is None:
            raise KnowledgeCapabilityError(
                "pdftotext backend was requested but Poppler is unavailable"
            )
        selected_pages: list[str] = []
        selected_backends: list[str] = []
        for index, native in enumerate(native_pages):
            poppler = poppler_pages[index].strip() if poppler_pages is not None else ""
            if pdf_backend == "pymupdf" or len(native) >= len(poppler):
                selected_pages.append(native)
                selected_backends.append("pymupdf")
            else:
                selected_pages.append(poppler)
                selected_backends.append("pdftotext")
        scan_pages = [
            index + 1
            for index, text in enumerate(selected_pages)
            if len(re.sub(r"\s+", "", text)) < scan_text_threshold
        ]
        ocr_pages: list[int] = []
        warnings: list[str] = []
        tesseract_available = shutil.which("tesseract") is not None
        if scan_pages and ocr == "required" and not tesseract_available:
            raise KnowledgeCapabilityError(
                "scan-like pages were detected and required OCR is unavailable"
            )
        if scan_pages and ocr in {"auto", "required"} and tesseract_available:
            if len(scan_pages) > max_ocr_pages:
                raise ValueError("scan-like page count exceeds the configured OCR limit")
            for page_number in scan_pages:
                text = _ocr_page(
                    document[page_number - 1],
                    language=ocr_language,
                    timeout_seconds=90,
                )
                if text:
                    selected_pages[page_number - 1] = text
                    selected_backends[page_number - 1] = "tesseract"
                    ocr_pages.append(page_number)
        elif scan_pages and ocr == "auto":
            warnings.append(
                "scan-like pages detected; Tesseract unavailable, so they remain unindexed"
            )
        elif scan_pages and ocr == "off":
            warnings.append("scan-like pages detected; OCR policy is off")

        chunks: list[DocumentChunk] = []
        for page_index, text in enumerate(selected_pages, start=1):
            for offset in range(0, len(text), max_chars):
                piece = text[offset : offset + max_chars].strip()
                if not piece:
                    continue
                digest = hashlib.sha256(
                    f"{page_index}\0{offset}\0{piece}".encode()
                ).hexdigest()[:20]
                chunks.append(
                    DocumentChunk(
                        chunk_id=digest,
                        text=piece,
                        section=None,
                        page=page_index,
                    )
                )
    finally:
        document.close()
    unresolved = sorted(set(scan_pages) - set(ocr_pages))
    receipt = {
        "schema_version": "1.0",
        "media_type": "application/pdf",
        "backend_policy": pdf_backend,
        "page_count": len(selected_pages),
        "chunk_count": len(chunks),
        "page_backends": {
            backend: selected_backends.count(backend)
            for backend in sorted(set(selected_backends))
        },
        "scan_like_pages": scan_pages,
        "ocr": {
            "policy": ocr,
            "language": ocr_language,
            "available": tesseract_available,
            "processed_pages": ocr_pages,
            "unresolved_pages": unresolved,
        },
        "tools": {
            "pymupdf": getattr(pymupdf, "__version__", "unknown"),
            "pdftotext": _command_version("pdftotext", "-v"),
            "tesseract": _command_version("tesseract", "--version"),
        },
        "citation_policy": "one-based PDF page number",
        "warnings": warnings,
    }
    return DocumentExtraction(chunks=tuple(chunks), receipt=receipt)


def document_chunks(path: Path) -> list[DocumentChunk]:
    return document_chunks_bytes(path.read_bytes(), suffix=path.suffix)


def document_chunks_bytes(data: bytes, *, suffix: str) -> list[DocumentChunk]:
    return list(extract_document_bytes(data, suffix=suffix).chunks)


class SeekDBKnowledgeStore:
    def __init__(self, root: Path, model_path: Path) -> None:
        try:
            import pyseekdb
        except ImportError as exc:
            raise KnowledgeCapabilityError(
                "pyseekdb is required; no fallback database may replace the authoritative store"
            ) from exc
        embedding = _embedding_function(model_path)
        self.client = pyseekdb.Client(path=str(root / "seekdb"), database="protbind")
        self.collection = self.client.get_or_create_collection(
            f"evidence_chunks_{embedding.manifest_sha256[:16]}",
            embedding_function=embedding,
        )

    def import_chunks(
        self,
        artifact: ArtifactRef,
        chunks: list[DocumentChunk],
        *,
        source_name: str,
        scope: str = "evidence",
        extra_metadata: dict[str, Any] | None = None,
        replace_scope: bool = False,
    ) -> int:
        if replace_scope:
            self.collection.delete(where={"scope": {"$eq": scope}})
        if not chunks and not replace_scope:
            raise ValueError("document contains no indexable text")
        if not chunks:
            self.collection.refresh_index()
            return 0
        ids = [f"{artifact.sha256}:{chunk.chunk_id}" for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {
                **(extra_metadata or {}),
                "artifact_id": artifact.artifact_id,
                "source_name": source_name,
                "section": chunk.section or "",
                "page": chunk.page or 0,
                "scope": scope,
            }
            metadatas.append(metadata)
        self.collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )
        self.collection.refresh_index()
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("knowledge query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        keyword_match = re.search(r"[\w\u4e00-\u9fff]{3,}", query)
        keyword = keyword_match.group(0) if keyword_match else query.strip()
        where = {"scope": {"$eq": scope}} if scope is not None else None
        lexical: dict[str, Any] = {
            "where_document": {"$contains": keyword},
            "n_results": top_k * 2,
        }
        vector: dict[str, Any] = {"query_texts": [query], "n_results": top_k * 2}
        if where is not None:
            lexical["where"] = where
            vector["where"] = where
        result = self.collection.hybrid_search(
            query=lexical,
            knn=vector,
            rank={"rrf": {}},
            n_results=top_k,
        )
        ids = result.get("ids", [[]])
        documents = result.get("documents", [[]])
        metadatas = result.get("metadatas", [[]])
        id_values = ids[0] if ids and isinstance(ids[0], list) else ids
        document_values = (
            documents[0] if documents and isinstance(documents[0], list) else documents
        )
        metadata_values = (
            metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
        )
        return [
            {"id": identifier, "text": document, "metadata": metadata}
            for identifier, document, metadata in zip(
                id_values, document_values, metadata_values, strict=False
            )
        ]


def import_document(
    workspace: Path,
    document_path: Path,
    model_path: Path,
    *,
    license: str | None = None,
    pdf_backend: str = "auto",
    ocr: str = "off",
    ocr_language: str = "eng",
) -> tuple[ArtifactRef, int, ArtifactRef]:
    artifacts = ArtifactStore(workspace)
    media_type = "application/pdf" if document_path.suffix.lower() == ".pdf" else "text/markdown"
    artifact = artifacts.import_file(
        document_path,
        media_type=media_type,
        producer="protbind.knowledge-import",
        producer_version="0.1.0",
        license=license,
    )
    # Parse the immutable imported bytes, not the source path, so an input file
    # changed after import cannot desynchronize artifact identity and chunks.
    extraction = extract_document_bytes(
        artifacts.read_bytes(artifact),
        suffix=document_path.suffix,
        pdf_backend=pdf_backend,
        ocr=ocr,
        ocr_language=ocr_language,
    )
    receipt = artifacts.put_json(
        {
            **extraction.receipt,
            "source_artifact_id": artifact.artifact_id,
            "source_name": document_path.name,
        },
        producer="protbind.document-extraction",
        producer_version="0.1.0",
        source=artifact.artifact_id,
    )
    count = SeekDBKnowledgeStore(workspace, model_path).import_chunks(
        artifact, list(extraction.chunks), source_name=document_path.name
    )
    return artifact, count, receipt


def sync_library_rag(
    workspace: Path,
    manager: LibraryManager,
    model_path: Path,
    *,
    kind: str = "protein",
    include_quarantined: bool = False,
) -> dict[str, Any]:
    """Rebuild a seekdb projection from the exact private-library catalog."""

    projection = manager.rag_projection(
        kind,
        include_quarantined=include_quarantined,
    )
    artifacts = ArtifactStore(workspace)
    snapshot = artifacts.put_json(
        projection,
        producer="protbind.library-rag-projection",
        producer_version="0.1.0",
        source=f"{kind}-library-catalog-projection",
    )
    chunks: list[DocumentChunk] = []
    for record in projection["records"]:
        facts = [
            f"{kind.title()} library entry {record['entry_id']}.",
            f"Catalog state {record['state']}.",
            f"Verification state {record['verification_state']}.",
        ]
        for name, value in record.items():
            if name in {"entry_id", "kind", "state", "verification_state"}:
                continue
            if value is not None and value != []:
                facts.append(f"{name.replace('_', ' ')}: {value}.")
        text = " ".join(facts)
        chunks.append(
            DocumentChunk(
                chunk_id=hashlib.sha256(
                    f"{snapshot.sha256}\0{record['entry_id']}\0{text}".encode()
                ).hexdigest()[:20],
                text=text,
                section=str(record["entry_id"]),
                page=None,
            )
        )
    count = SeekDBKnowledgeStore(workspace, model_path).import_chunks(
        snapshot,
        chunks,
        source_name=f"{kind}-library-catalog-projection",
        scope=f"{kind}-library",
        extra_metadata={
            "library_kind": kind,
            "authoritative_source": "catalog.sqlite",
        },
        replace_scope=True,
    )
    return {
        "kind": kind,
        "scope": f"{kind}-library",
        "snapshot_artifact": snapshot.to_dict(),
        "records_projected": projection["record_count"],
        "chunks_indexed": count,
        "authoritative_source": "catalog.sqlite",
        "retrieval_semantics": (
            "Discovery aid only; re-read the catalog entry and normal QC gates before use."
        ),
        "private_values_disclosed": False,
    }
