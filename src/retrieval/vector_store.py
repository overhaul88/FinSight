"""Vector-store abstractions for retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.config import settings
from src.ingestion.embedder import EmbeddingModel, FAISSVectorStore


class BaseVectorStore(ABC):
    """Abstract query interface for vector-backed retrieval."""

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the top-k documents for a query."""


class FAISSRetriever(BaseVectorStore):
    """FAISS-backed retriever that loads a persisted local index."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        store: FAISSVectorStore | None = None,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.store = store or FAISSVectorStore(embedding_dim=self.embedding_model.embedding_dim)
        if store is None:
            self.store.load()

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.embed_query(query)
        return self.store.search(query_embedding, top_k=top_k or settings.top_k_retrieval)


def get_vector_store(store_type: str = "faiss") -> BaseVectorStore:
    """Factory for supported vector-store backends."""

    if store_type == "faiss":
        return FAISSRetriever()
    raise ValueError(f"Unknown vector store type: {store_type}")

