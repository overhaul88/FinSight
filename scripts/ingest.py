"""CLI entry point for the FinSight ingestion pipeline."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.ingestion.chunker import ChunkingPipeline
from src.ingestion.embedder import EmbeddingModel, FAISSVectorStore
from src.ingestion.loader import DocumentLoader


@contextmanager
def _maybe_mlflow_run(run_name: str):
    try:
        import mlflow  # type: ignore
    except ImportError:
        yield None
        return

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def _save_chunks(chunks, output_dir: str) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    with (destination / "chunks.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_ingestion(data_dir: str, vector_store_type: str = "faiss") -> dict:
    started_at = time.time()

    with _maybe_mlflow_run("ingestion"):
        loader = DocumentLoader(data_dir=data_dir)
        documents = loader.load_all()
        if not documents:
            raise RuntimeError(f"No supported documents found in {data_dir}")

        chunker = ChunkingPipeline()
        chunks = chunker.chunk_documents(documents)
        if not chunks:
            raise RuntimeError("No chunks were created from the input corpus.")

        _save_chunks(chunks, settings.processed_data_dir)

        if vector_store_type != "faiss":
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")

        embedding_model = EmbeddingModel()
        embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])
        store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
        store.build_index(embeddings, chunks)
        store.save()

        return {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "vectors_indexed": store.total_vectors,
            "duration_seconds": round(time.time() - started_at, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FinSight ingestion pipeline.")
    parser.add_argument("--data-dir", default=settings.raw_data_dir, help="Input document directory")
    parser.add_argument("--vector-store", default="faiss", choices=["faiss"])
    args = parser.parse_args()

    metrics = run_ingestion(args.data_dir, args.vector_store)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
