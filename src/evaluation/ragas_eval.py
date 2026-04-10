"""Evaluation helpers for FinSight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_eval_dataset(qa_pairs_path: str = "data/eval/qa_pairs.json") -> List[Dict[str, Any]]:
    """Load evaluation seed data from JSON."""

    with Path(qa_pairs_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload)


def build_eval_dataset(
    questions: Sequence[str],
    rag_chain: Any,
    ground_truths: Sequence[str],
) -> List[Dict[str, Any]]:
    """Run the chain over evaluation questions and collect contexts and answers."""

    dataset: List[Dict[str, Any]] = []
    for question, ground_truth in zip(questions, ground_truths):
        result = rag_chain.invoke(question)
        dataset.append(
            {
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": [source.get("text", "") or source.get("text_preview", "") for source in result.get("sources", [])],
                "ground_truth": ground_truth,
            }
        )
    return dataset


def run_ragas_evaluation(
    eval_dataset: Iterable[Dict[str, Any]],
    dry_run: bool = True,
    metrics: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Run dry-mode heuristics locally or delegate to Ragas when available."""

    dataset = list(eval_dataset)
    if not dataset:
        return {}

    if not dry_run:
        try:
            from datasets import Dataset  # type: ignore
            from ragas import evaluate  # type: ignore
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Ragas dependencies are not installed.") from exc

        ragas_dataset = Dataset.from_list(dataset)
        selected_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        if metrics:
            metric_lookup = {metric.name: metric for metric in selected_metrics}
            selected_metrics = [metric_lookup[name] for name in metrics if name in metric_lookup]
        results = evaluate(ragas_dataset, metrics=selected_metrics)
        return results.to_pandas().mean().to_dict()

    total = len(dataset)
    answered = sum(1 for row in dataset if row.get("answer", "").strip())
    with_context = sum(1 for row in dataset if row.get("contexts"))
    grounded = sum(
        1
        for row in dataset
        if row.get("ground_truth", "").strip().lower()[:20] in row.get("answer", "").strip().lower()
    )

    return {
        "answer_non_empty_rate": round(answered / total, 4),
        "context_non_empty_rate": round(with_context / total, 4),
        "ground_truth_prefix_match_rate": round(grounded / total, 4),
    }

