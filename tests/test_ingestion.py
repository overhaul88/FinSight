"""Tests for the ingestion layer."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.chunker import ChunkingPipeline, RecursiveChunker, SectionAwareChunker
from src.ingestion.loader import DocumentLoader


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_rbi_guideline.txt"


def test_document_loader_loads_text_fixture(tmp_path):
    target = tmp_path / "RBI_sample_guideline.txt"
    target.write_text(FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    documents = DocumentLoader(tmp_path).load_all()

    assert len(documents) == 1
    document = documents[0]
    assert document.metadata["filename"] == "RBI_sample_guideline.txt"
    assert document.metadata["doc_type"] == "RBI_Circular"
    assert document.pages[0]["page_number"] == 1
    assert "CUSTOMER IDENTIFICATION" in document.content


def test_recursive_chunker_creates_chunks(tmp_path):
    target = tmp_path / "policy.txt"
    target.write_text(FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    document = DocumentLoader(tmp_path).load_all()[0]

    chunker = RecursiveChunker(chunk_size=220, chunk_overlap=30)
    chunks = chunker.chunk(document)

    assert chunks
    assert all(chunk.metadata["chunk_strategy"] == "recursive" for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_section_aware_chunker_preserves_section_titles(tmp_path):
    target = tmp_path / "RBI_master_direction.txt"
    target.write_text(FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    document = DocumentLoader(tmp_path).load_all()[0]

    chunker = SectionAwareChunker(chunk_size=260, chunk_overlap=40)
    chunks = chunker.chunk(document)

    assert chunks
    assert all("section_title" in chunk.metadata for chunk in chunks)
    assert any(chunk.metadata["section_title"].startswith("3.") for chunk in chunks)


def test_chunking_pipeline_prefers_section_aware_for_rbi_docs(tmp_path):
    target = tmp_path / "RBI_kyc_note.txt"
    target.write_text(FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    document = DocumentLoader(tmp_path).load_all()[0]

    pipeline = ChunkingPipeline()
    chunks = pipeline.chunk_documents([document])

    assert chunks
    assert all(chunk.metadata["chunk_strategy"].startswith("section_aware") for chunk in chunks)
