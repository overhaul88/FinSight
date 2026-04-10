"""Chunking strategies for FinSight."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, Iterable, List, Sequence

from src.config import settings
from src.ingestion.loader import RawDocument


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    cleaned = text.strip()
    if not cleaned:
        return []

    step = max(chunk_size - max(overlap, 0), 1)
    chunks = []
    for start in range(0, len(cleaned), step):
        piece = cleaned[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(cleaned):
            break
    return chunks


def _split_large_unit(unit: str, chunk_size: int, overlap: int) -> List[str]:
    separators = ["\n\n", "\n", ". ", " "]
    current_units = [unit.strip()]

    for separator in separators:
        next_units: List[str] = []
        changed = False
        for current in current_units:
            if len(current) <= chunk_size:
                next_units.append(current)
                continue

            parts = [part.strip() for part in current.split(separator) if part.strip()]
            if len(parts) <= 1:
                next_units.append(current)
                continue

            changed = True
            suffix = "." if separator == ". " else ""
            next_units.extend([f"{part}{suffix}".strip() for part in parts if part.strip()])

        current_units = next_units
        if changed:
            break

    result: List[str] = []
    for current in current_units:
        if len(current) <= chunk_size:
            result.append(current)
        else:
            result.extend(_hard_split(current, chunk_size, overlap))
    return result


def _pack_units(units: Sequence[str], chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    current = ""

    for unit in units:
        clean_unit = unit.strip()
        if not clean_unit:
            continue

        candidate = clean_unit if not current else f"{current}\n\n{clean_unit}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            overlap_text = current[-overlap:].strip() if overlap > 0 else ""
            current = clean_unit if not overlap_text else f"{overlap_text}\n\n{clean_unit}"
        else:
            chunks.extend(_hard_split(clean_unit, chunk_size, overlap))
            current = ""

    if current.strip():
        chunks.append(current.strip())

    normalized: List[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            normalized.append(chunk)
        else:
            normalized.extend(_hard_split(chunk, chunk_size, overlap))

    return normalized


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text.encode("utf-8")).hexdigest()[:12]


class RecursiveChunker:
    """General-purpose fallback chunker."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, document: RawDocument) -> List[Chunk]:
        units = [part.strip() for part in document.content.split("\n\n") if part.strip()]
        split_units: List[str] = []
        for unit in units:
            if len(unit) <= self.chunk_size:
                split_units.append(unit)
            else:
                split_units.extend(_split_large_unit(unit, self.chunk_size, self.chunk_overlap))

        raw_chunks = _pack_units(split_units, self.chunk_size, self.chunk_overlap)
        return [
            Chunk(
                text=text,
                metadata={
                    **document.metadata,
                    "chunk_index": index,
                    "chunk_count": len(raw_chunks),
                    "chunk_strategy": "recursive",
                },
            )
            for index, text in enumerate(raw_chunks)
            if len(text.strip()) >= 50
        ]


class SectionAwareChunker:
    """Chunker that preserves detected section boundaries."""

    SECTION_PATTERNS = [
        r"^(CHAPTER|SECTION|PART)\s+[IVXLCDM\d]+",
        r"^\d+\.\s+[A-Z]",
        r"^\d+\.\d+\s+",
        r"^[A-Z]{2,}\s*:\s*",
        r"^(Circular No|RBI\/|SEBI\/)",
        r"^Schedule\s+[IVXLCDM\d]+",
        r"^Annex(ure)?\s+[A-Z\d]+",
    ]

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.pattern = re.compile("|".join(self.SECTION_PATTERNS), re.MULTILINE)

    def chunk(self, document: RawDocument) -> List[Chunk]:
        sections = self._extract_sections(document.content)
        chunks: List[Chunk] = []
        index = 0

        for title, section_text in sections:
            if len(section_text) <= self.chunk_size:
                if len(section_text.strip()) >= 50:
                    chunks.append(
                        Chunk(
                            text=section_text.strip(),
                            metadata={
                                **document.metadata,
                                "section_title": title,
                                "chunk_index": index,
                                "chunk_strategy": "section_aware",
                            },
                        )
                    )
                    index += 1
                continue

            for text in _pack_units(
                _split_large_unit(section_text, self.chunk_size, self.chunk_overlap),
                self.chunk_size,
                self.chunk_overlap,
            ):
                if len(text.strip()) < 50:
                    continue
                chunks.append(
                    Chunk(
                        text=text.strip(),
                        metadata={
                            **document.metadata,
                            "section_title": title,
                            "chunk_index": index,
                            "chunk_strategy": "section_aware_recursive",
                        },
                    )
                )
                index += 1

        return chunks

    def _extract_sections(self, text: str) -> List[tuple[str, str]]:
        lines = text.splitlines()
        sections: List[tuple[str, str]] = []
        current_title = "Introduction"
        current_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if self.pattern.match(stripped):
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append((current_title, section_text))
                current_title = stripped
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((current_title, section_text))

        return sections or [("Full Document", text)]


class ChunkingPipeline:
    """Select the chunking strategy based on document structure."""

    def __init__(self) -> None:
        self.recursive = RecursiveChunker()
        self.section_aware = SectionAwareChunker()

    def chunk_documents(self, documents: Iterable[RawDocument]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for document in documents:
            strategy = self._select_strategy(document)
            chunks = (
                self.section_aware.chunk(document)
                if strategy == "section_aware"
                else self.recursive.chunk(document)
            )
            all_chunks.extend(chunks)
        return all_chunks

    def _select_strategy(self, document: RawDocument) -> str:
        doc_type = document.metadata.get("doc_type", "")
        if doc_type in {"RBI_Circular", "SEBI_Guideline"}:
            return "section_aware"

        numbered_lines = len(re.findall(r"^\d+\.\s+", document.content, flags=re.MULTILINE))
        return "section_aware" if numbered_lines > 5 else "recursive"

