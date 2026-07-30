from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from protbind_agent.knowledge import (
    DocumentChunk,
    KnowledgeCapabilityError,
    SeekDBKnowledgeStore,
    extract_document_bytes,
    freeze_embedding_model_manifest,
    inspect_embedding_model,
)
from protbind_agent.library import LibraryManager, load_library_config, save_library_config


def test_markdown_extraction_preserves_section_citations() -> None:
    extraction = extract_document_bytes(
        b"# Methods\nLocal docking protocol.\n",
        suffix=".md",
    )

    assert extraction.receipt["backend"] == "utf-8"
    assert extraction.chunks[0].section == "Methods"
    assert extraction.chunks[0].page is None


def test_blank_pdf_is_explicitly_reported_as_unresolved_without_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    original_which = __import__("shutil").which

    def which(name: str) -> str | None:
        if name == "tesseract":
            return None
        return original_which(name)

    monkeypatch.setattr("protbind_agent.knowledge.shutil.which", which)
    extraction = extract_document_bytes(
        data,
        suffix=".pdf",
        pdf_backend="pymupdf",
        ocr="auto",
    )

    assert extraction.receipt["scan_like_pages"] == [1]
    assert extraction.receipt["ocr"]["unresolved_pages"] == [1]
    assert extraction.receipt["warnings"]
    assert extraction.chunks == ()


def test_required_ocr_fails_closed_when_tesseract_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    monkeypatch.setattr("protbind_agent.knowledge.shutil.which", lambda _name: None)

    with pytest.raises(KnowledgeCapabilityError, match="required OCR is unavailable"):
        extract_document_bytes(
            data,
            suffix=".pdf",
            pdf_backend="pymupdf",
            ocr="required",
        )


def test_qwen_model_is_hash_pinned_and_runtime_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "qwen"
    model.mkdir()
    weight = model / "weights.bin"
    weight.write_bytes(b"pinned")
    manifest = {
        "schema_version": "1.0",
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": "reviewed-revision",
        "files": {"weights.bin": hashlib.sha256(b"pinned").hexdigest()},
    }
    (model / "protbind-model-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    versions = {
        "transformers": "4.48.1",
        "sentence-transformers": "5.6.0",
        "FlagEmbedding": "1.3.5",
    }
    monkeypatch.setattr(
        "protbind_agent.knowledge.importlib.metadata.version",
        lambda name: versions[name],
    )

    result = inspect_embedding_model(model)

    assert result["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert result["status"] == "BLOCKED_RUNTIME_COMPATIBILITY"
    assert "transformers>=4.51.0" in result["reason"]


def test_model_freeze_hashes_reviewed_files_without_loading_model(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")

    result = freeze_embedding_model_manifest(
        model,
        model_name="BAAI/bge-m3",
        model_revision="pinned-revision",
    )
    manifest = json.loads(
        (model / "protbind-model-manifest.json").read_text(encoding="utf-8")
    )

    assert result["file_count"] == 2
    assert result["model_loaded"] is False
    assert set(manifest["files"]) == {"config.json", "model.safetensors"}
    assert manifest["model_revision"] == "pinned-revision"
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_embedding_model_manifest(
            model,
            model_name="BAAI/bge-m3",
            model_revision="pinned-revision",
        )


def test_library_rag_projection_excludes_private_values_and_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "library.json"
    save_library_config(
        config_path,
        protein_root=tmp_path / "proteins",
        ligand_root=tmp_path / "ligands",
    )
    manager = LibraryManager(load_library_config(config_path))
    private_sequence = "ACDEFGHIK"
    source = tmp_path / "secret-name.fasta"
    source.write_text(f">private\n{private_sequence}\n", encoding="utf-8")
    manager.apply(manager.scan("protein", source))

    projection = manager.rag_projection("protein")
    serialized = json.dumps(projection)

    assert projection["record_count"] == 1
    assert projection["records"][0]["sequence_lengths"] == [9]
    assert private_sequence not in serialized
    assert "secret-name" not in serialized
    assert str(tmp_path) not in serialized
    assert projection["private_values_disclosed"] is False


def test_seekdb_scope_filter_is_applied_to_both_hybrid_branches() -> None:
    class FakeCollection:
        request: dict[str, object]

        def hybrid_search(self, **kwargs):  # noqa: ANN003, ANN202
            self.request = kwargs
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

    store = SeekDBKnowledgeStore.__new__(SeekDBKnowledgeStore)
    store.collection = FakeCollection()

    assert store.search("kinase structure", scope="protein-library") == []
    expected = {"scope": {"$eq": "protein-library"}}
    assert store.collection.request["query"]["where"] == expected
    assert store.collection.request["knn"]["where"] == expected


def test_library_projection_replaces_its_scope_before_upsert() -> None:
    class FakeCollection:
        deleted: dict[str, object] | None = None
        upserted: dict[str, object] | None = None

        def delete(self, **kwargs):  # noqa: ANN003, ANN202
            self.deleted = kwargs

        def upsert(self, **kwargs):  # noqa: ANN003, ANN202
            self.upserted = kwargs

        def refresh_index(self) -> None:
            return None

    store = SeekDBKnowledgeStore.__new__(SeekDBKnowledgeStore)
    store.collection = FakeCollection()

    count = store.import_chunks(
        type(
            "Artifact",
            (),
            {"sha256": "a" * 64, "artifact_id": f"sha256:{'a' * 64}"},
        )(),
        [
            DocumentChunk(
                chunk_id="entry",
                text="Protein library entry protein-a.",
                section="protein-a",
                page=None,
            )
        ],
        source_name="protein-library-catalog-projection",
        scope="protein-library",
        replace_scope=True,
    )

    assert count == 1
    assert store.collection.deleted == {
        "where": {"scope": {"$eq": "protein-library"}}
    }
    assert store.collection.upserted is not None
