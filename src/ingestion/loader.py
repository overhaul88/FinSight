"""Document loading for FinSight."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.config import settings


logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """Represents a source document before chunking."""

    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[Dict[str, Any]] = field(default_factory=list)


class DocumentLoader:
    """Load PDF and plain-text documents from disk."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}

    def __init__(self, data_dir: str | Path = settings.raw_data_dir, min_chars: int = 50):
        self.data_dir = Path(data_dir)
        self.min_chars = min_chars

    def load_all(self) -> List[RawDocument]:
        documents: List[RawDocument] = []
        for path in self._iter_supported_files():
            try:
                document = self._load_single(path)
            except Exception as exc:
                logger.exception("Failed to load %s: %s", path.name, exc)
                continue

            if len(document.content.strip()) < self.min_chars:
                logger.warning("Skipping near-empty document: %s", path.name)
                continue

            documents.append(document)

        return documents

    def _iter_supported_files(self) -> List[Path]:
        if not self.data_dir.exists():
            return []

        files = [
            path
            for path in self.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        return sorted(files)

    def _load_single(self, path: Path) -> RawDocument:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf_document(path)
        return self._load_text_document(path)

    def _load_text_document(self, path: Path) -> RawDocument:
        content = path.read_text(encoding="utf-8")
        metadata = {
            "title": path.stem,
            "author": "",
            "source": str(path),
            "filename": path.name,
            "total_pages": 1,
            "doc_type": self._classify_document(path.name),
        }
        pages = [
            {
                "page_number": 1,
                "text": content,
                "char_count": len(content),
            }
        ]
        return RawDocument(source=str(path), content=content, metadata=metadata, pages=pages)

    def _load_pdf_document(self, path: Path) -> RawDocument:
        fitz_document = self._try_load_with_fitz(path)
        if fitz_document is not None:
            return fitz_document

        pdfplumber_document = self._try_load_with_pdfplumber(path)
        if pdfplumber_document is not None:
            return pdfplumber_document

        raise RuntimeError(
            "No PDF backend available. Install PyMuPDF or pdfplumber to load PDF files."
        )

    def _try_load_with_fitz(self, path: Path) -> RawDocument | None:
        try:
            import fitz  # type: ignore
        except ImportError:
            return None

        pages: List[Dict[str, Any]] = []
        text_parts: List[str] = []

        with fitz.open(str(path)) as doc:
            metadata = {
                "title": (doc.metadata or {}).get("title") or path.stem,
                "author": (doc.metadata or {}).get("author", ""),
                "source": str(path),
                "filename": path.name,
                "total_pages": len(doc),
                "doc_type": self._classify_document(path.name),
            }

            for index, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if not text.strip():
                    continue
                pages.append(
                    {
                        "page_number": index,
                        "text": text,
                        "char_count": len(text),
                    }
                )
                text_parts.append(f"[Page {index}]\n{text.strip()}")

        content = "\n\n".join(text_parts)
        return RawDocument(source=str(path), content=content, metadata=metadata, pages=pages)

    def _try_load_with_pdfplumber(self, path: Path) -> RawDocument | None:
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            return None

        pages: List[Dict[str, Any]] = []
        text_parts: List[str] = []

        with pdfplumber.open(str(path)) as pdf:
            metadata = {
                "title": path.stem,
                "author": "",
                "source": str(path),
                "filename": path.name,
                "total_pages": len(pdf.pages),
                "doc_type": self._classify_document(path.name),
            }

            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                pages.append(
                    {
                        "page_number": index,
                        "text": text,
                        "char_count": len(text),
                    }
                )
                text_parts.append(f"[Page {index}]\n{text.strip()}")

        content = "\n\n".join(text_parts)
        return RawDocument(source=str(path), content=content, metadata=metadata, pages=pages)

    def _classify_document(self, filename: str) -> str:
        filename_lower = filename.lower()
        if "rbi" in filename_lower:
            return "RBI_Circular"
        if "sebi" in filename_lower:
            return "SEBI_Guideline"
        if "loan" in filename_lower or "credit" in filename_lower:
            return "Loan_Policy"
        if "kyc" in filename_lower:
            return "KYC_Guideline"
        return "Financial_Document"

