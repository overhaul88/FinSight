"""Pydantic schemas for the FinSight API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for question answering."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Natural language question about financial regulations",
    )
    top_k: Optional[int] = Field(default=3, ge=1, le=10)
    streaming: Optional[bool] = Field(default=False)


class SourceDocument(BaseModel):
    """Retrieved source summary."""

    text_preview: str
    source_file: str
    section: str = ""
    relevance_score: float
    doc_type: str = ""


class QueryResponse(BaseModel):
    """Standard non-streaming API response."""

    answer: str
    sources: List[SourceDocument]
    query: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    vector_store_loaded: bool
    model_loaded: bool
    total_indexed_chunks: int
    load_error: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestRequest(BaseModel):
    """Request to trigger ingestion in future API extensions."""

    data_dir: str = Field(default="data/raw")
    vector_store: str = Field(default="faiss")
