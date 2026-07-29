"""seekdb-backed local evidence retrieval with BGE-M3 capability gates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, sha256_file
from .models import ArtifactRef


class KnowledgeCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    section: str | None
    page: int | None


class LocalBGEM3EmbeddingFunction:
    """pyseekdb-compatible dense BGE-M3 function that never downloads weights."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"local BGE-M3 model directory not found: {model_path.name}"
            )
        manifest_path = model_path / "protbind-model-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "local BGE-M3 requires protbind-model-manifest.json with file hashes"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("BGE-M3 model manifest is invalid JSON") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != "1.0"
            or manifest.get("model_name") != "BAAI/bge-m3"
            or not isinstance(manifest.get("model_revision"), str)
            or not manifest["model_revision"].strip()
            or not isinstance(manifest.get("files"), dict)
            or not manifest["files"]
        ):
            raise ValueError("BGE-M3 model manifest has an invalid identity schema")
        for relative_name, expected_sha256 in manifest["files"].items():
            relative = Path(str(relative_name))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("BGE-M3 manifest file paths must stay inside the model")
            candidate = model_path / relative
            if (
                not candidate.is_file()
                or not isinstance(expected_sha256, str)
                or sha256_file(candidate) != expected_sha256
            ):
                raise ValueError(
                    f"BGE-M3 model file hash mismatch: {relative.name}"
                )
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise KnowledgeCapabilityError(
                "FlagEmbedding is required for the local BGE-M3 knowledge backend"
            ) from exc
        self.model_path = model_path
        self.model_revision = manifest["model_revision"]
        self.manifest_sha256 = sha256_file(manifest_path)
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


def _pdf_chunks(data: bytes, *, max_chars: int = 1800) -> list[DocumentChunk]:
    try:
        import pymupdf
    except ImportError as exc:
        raise KnowledgeCapabilityError(
            "PyMuPDF is required to import PDF evidence with page citations"
        ) from exc
    chunks: list[DocumentChunk] = []
    document = pymupdf.open(stream=data, filetype="pdf")
    try:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
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
    return chunks


def document_chunks(path: Path) -> list[DocumentChunk]:
    return document_chunks_bytes(path.read_bytes(), suffix=path.suffix)


def document_chunks_bytes(data: bytes, *, suffix: str) -> list[DocumentChunk]:
    if suffix.lower() == ".pdf":
        return _pdf_chunks(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("knowledge import supports UTF-8 text/Markdown or PDF") from exc
    return _markdown_chunks(text)


class SeekDBKnowledgeStore:
    def __init__(self, root: Path, model_path: Path) -> None:
        try:
            import pyseekdb
        except ImportError as exc:
            raise KnowledgeCapabilityError(
                "pyseekdb is required; no fallback database may replace the authoritative store"
            ) from exc
        embedding = LocalBGEM3EmbeddingFunction(model_path)
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
    ) -> int:
        if not chunks:
            raise ValueError("document contains no indexable text")
        ids = [f"{artifact.sha256}:{chunk.chunk_id}" for chunk in chunks]
        metadatas = [
            {
                "artifact_id": artifact.artifact_id,
                "source_name": source_name,
                "section": chunk.section or "",
                "page": chunk.page or 0,
            }
            for chunk in chunks
        ]
        self.collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )
        self.collection.refresh_index()
        return len(chunks)

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("knowledge query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        keyword_match = re.search(r"[\w\u4e00-\u9fff]{3,}", query)
        keyword = keyword_match.group(0) if keyword_match else query.strip()
        result = self.collection.hybrid_search(
            query={"where_document": {"$contains": keyword}, "n_results": top_k * 2},
            knn={"query_texts": [query], "n_results": top_k * 2},
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
) -> tuple[ArtifactRef, int]:
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
    chunks = document_chunks_bytes(
        artifacts.read_bytes(artifact), suffix=document_path.suffix
    )
    count = SeekDBKnowledgeStore(workspace, model_path).import_chunks(
        artifact, chunks, source_name=document_path.name
    )
    return artifact, count
