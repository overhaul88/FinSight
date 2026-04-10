"""CLI runner for FinSight evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.evaluation.mlflow_tracker import FinSightTracker
from src.evaluation.ragas_eval import build_eval_dataset, load_eval_dataset, run_ragas_evaluation
from src.llm.model import load_mistral_with_adapter
from src.retrieval.chain import FinSightChain
from src.retrieval.retriever import ProductionRetriever
from src.retrieval.vector_store import get_vector_store


def main() -> None:
    seed = load_eval_dataset()
    questions = [item["question"] for item in seed]
    ground_truths = [item["ground_truth"] for item in seed]

    vector_store = get_vector_store("faiss")
    llm = load_mistral_with_adapter()
    retriever = ProductionRetriever(vector_store, llm=llm)
    chain = FinSightChain(retriever, llm)

    eval_dataset = build_eval_dataset(questions, chain, ground_truths)
    scores = run_ragas_evaluation(eval_dataset, dry_run=True)

    tracker = FinSightTracker()
    with tracker.ragas_run(run_name="ragas-baseline-eval"):
        tracker.log_ragas_scores(
            scores,
            params={
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model,
                "top_k_retrieval": settings.top_k_retrieval,
                "top_k_rerank": settings.top_k_rerank,
                "chunk_size": settings.chunk_size,
            },
        )

    for metric, score in scores.items():
        print(f"{metric}: {score}")


if __name__ == "__main__":
    main()

