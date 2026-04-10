"""End-to-end smoke test for local FinSight retrieval and generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.ingestion.chunker import ChunkingPipeline
from src.ingestion.embedder import EmbeddingModel, FAISSVectorStore
from src.ingestion.loader import DocumentLoader
from src.llm.model import detect_runtime_capabilities, load_local_llm, recommended_local_model_profile
from src.retrieval.chain import FinSightChain
from src.retrieval.retriever import ProductionRetriever
from src.retrieval.vector_store import FAISSRetriever


@dataclass
class PassthroughReranker:
    """Keep similarity-search ordering for a fast smoke run."""

    def rerank(self, query, candidates, top_k=None):
        ranked = []
        for candidate in candidates:
            item = dict(candidate)
            item["rerank_score"] = float(item.get("score", 0.0))
            ranked.append(item)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[: top_k or len(ranked)]


def _prepare_corpus(data_dir: Path | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if data_dir is not None:
        return data_dir, None

    temp_dir = tempfile.TemporaryDirectory(prefix="finsight-smoke-corpus-")
    target_dir = Path(temp_dir.name)
    fixture = ROOT / "tests" / "fixtures" / "sample_rbi_guideline.txt"
    shutil.copy2(fixture, target_dir / "RBI_sample_guideline.txt")
    return target_dir, temp_dir


def run_smoke(question: str, data_dir: Path | None = None) -> dict:
    started_at = time.time()
    corpus_dir, temp_corpus = _prepare_corpus(data_dir)
    runtime = detect_runtime_capabilities()
    recommended_profile = recommended_local_model_profile(runtime)

    with tempfile.TemporaryDirectory(prefix="finsight-smoke-index-") as index_dir:
        loader = DocumentLoader(corpus_dir)
        documents = loader.load_all()
        if not documents:
            raise RuntimeError(f"No supported documents found in {corpus_dir}")

        chunker = ChunkingPipeline()
        chunks = chunker.chunk_documents(documents)
        if not chunks:
            raise RuntimeError("No chunks were created from the smoke corpus.")

        embedding_model = EmbeddingModel(settings.embedding_model)
        embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])

        store = FAISSVectorStore(index_path=index_dir, embedding_dim=embeddings.shape[1])
        store.build_index(embeddings, chunks)
        store.save()

        vector_store = FAISSRetriever(embedding_model=embedding_model, store=store)
        llm = load_local_llm(
            base_model=settings.llm_model,
            use_4bit=bool(recommended_profile.get("use_4bit", True)),
            fallback_to_dummy=False,
        )
        retriever = ProductionRetriever(
            vector_store=vector_store,
            llm=llm,
            reranker=PassthroughReranker(),
        )
        chain = FinSightChain(retriever, llm)
        result = chain.invoke(question)

    if temp_corpus is not None:
        temp_corpus.cleanup()

    return {
        "runtime": {
            "cuda_available": runtime.cuda_available,
            "device_name": runtime.device_name,
            "total_memory_gb": runtime.total_memory_gb,
            "cuda_version": runtime.cuda_version,
        },
        "recommended_profile": recommended_profile,
        "corpus_dir": str(corpus_dir),
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "question": question,
        "answer_preview": result["answer"][:500],
        "sources": result["sources"],
        "duration_seconds": round(time.time() - started_at, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an end-to-end FinSight smoke test with the local models.")
    parser.add_argument(
        "--question",
        default="What must entities obtain before account activation?",
        help="Question to ask against the smoke corpus.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional data directory. If omitted, the bundled RBI fixture corpus is used.",
    )
    args = parser.parse_args()

    payload = run_smoke(
        question=args.question,
        data_dir=Path(args.data_dir).resolve() if args.data_dir else None,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
