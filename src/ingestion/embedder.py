"""Embedding and local vector-store utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.config import settings
from src.ingestion.chunker import Chunk


class EmbeddingModel:
    """Lazy sentence-transformers wrapper for BGE embeddings."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self.embedding_dim = 384

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for embedding generation."
                ) from exc

            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self.embedding_dim = self._model.get_embedding_dimension()
            else:
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_documents(self, texts: List[str], batch_size: int = 32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_query(self, query: str):
        return self.model.encode(
            [f"{self.QUERY_PREFIX}{query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]


class FAISSVectorStore:
    """Local FAISS index with parallel JSON metadata storage."""

    def __init__(self, index_path: str | None = None, embedding_dim: int = 384):
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.embedding_dim = embedding_dim
        self.metadata_store: List[Dict[str, Any]] = []
        self.index = None
        self.index_path.mkdir(parents=True, exist_ok=True)

    def build_index(self, embeddings, chunks: List[Chunk]) -> None:
        try:
            import faiss  # type: ignore
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("faiss-cpu and numpy are required to build the local index.") from exc

        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks length mismatch")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch")

        base_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIDMap(base_index)
        ids = np.arange(len(embeddings), dtype=np.int64)
        self.index.add_with_ids(embeddings.astype("float32"), ids)
        self.metadata_store = [
            {"text": chunk.text, "metadata": chunk.metadata, "chunk_id": chunk.chunk_id}
            for chunk in chunks
        ]

    def save(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required to save the local index.") from exc

        if self.index is None:
            raise RuntimeError("Index has not been built.")

        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with (self.index_path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata_store, handle, indent=2, ensure_ascii=False)

    def load(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required to load the local index.") from exc

        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.json"
        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Missing FAISS artifacts in {self.index_path}")

        self.index = faiss.read_index(str(index_file))
        with metadata_file.open(encoding="utf-8") as handle:
            self.metadata_store = json.load(handle)

    def search(self, query_embedding, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to query the local index.") from exc

        if self.index is None:
            raise RuntimeError("Index has not been loaded.")

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, ids = self.index.search(query, top_k)
        results: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], ids[0]):
            if index == -1:
                continue
            result = dict(self.metadata_store[index])
            result["score"] = float(score)
            results.append(result)
        return results

    @property
    def total_vectors(self) -> int:
        if self.index is None:
            return 0
        return int(self.index.ntotal)
