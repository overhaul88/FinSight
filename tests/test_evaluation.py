"""Tests for the evaluation scaffold."""

from __future__ import annotations

from src.evaluation.mlflow_tracker import FinSightTracker
from src.evaluation.ragas_eval import build_eval_dataset, load_eval_dataset, run_ragas_evaluation


class FakeChain:
    def invoke(self, question: str):
        return {
            "answer": f"{question} answer",
            "sources": [
                {
                    "text": "retrieved context",
                    "source": "doc.txt",
                    "section": "Section 1",
                    "rerank_score": 0.9,
                }
            ],
        }


def test_load_eval_dataset_reads_seed_file():
    dataset = load_eval_dataset()
    assert len(dataset) >= 3
    assert "question" in dataset[0]


def test_build_eval_dataset_collects_answers_and_contexts():
    questions = ["Question 1", "Question 2"]
    ground_truths = ["Truth 1", "Truth 2"]
    dataset = build_eval_dataset(questions, FakeChain(), ground_truths)

    assert len(dataset) == 2
    assert dataset[0]["answer"] == "Question 1 answer"
    assert dataset[0]["contexts"] == ["retrieved context"]


def test_run_ragas_evaluation_supports_dry_mode():
    dataset = [
        {
            "question": "Question 1",
            "answer": "Truth 1 answer",
            "contexts": ["context"],
            "ground_truth": "Truth 1",
        },
        {
            "question": "Question 2",
            "answer": "",
            "contexts": [],
            "ground_truth": "Truth 2",
        },
    ]
    scores = run_ragas_evaluation(dataset, dry_run=True)

    assert set(scores) == {
        "answer_non_empty_rate",
        "context_non_empty_rate",
        "ground_truth_prefix_match_rate",
    }
    assert scores["answer_non_empty_rate"] == 0.5


def test_tracker_is_noop_without_mlflow():
    tracker = FinSightTracker()
    with tracker.ragas_run("dry-run") as run:
        if tracker._mlflow is None:
            assert run is None
        else:
            assert run is not None
