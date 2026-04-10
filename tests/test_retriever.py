"""Tests for the retrieval baseline."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.retrieval.retriever import CrossEncoderReranker, ProductionRetriever
from src.retrieval.vector_store import BaseVectorStore, get_vector_store


class FakeVectorStore(BaseVectorStore):
    def __init__(self, mapping: Dict[str, List[Dict[str, Any]]]) -> None:
        self.mapping = mapping

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return list(self.mapping.get(query, []))[:top_k]


class FakeReranker:
    def __init__(self, score_map: Dict[str, float], should_fail: bool = False) -> None:
        self.score_map = score_map
        self.should_fail = should_fail

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int | None = None):
        if self.should_fail:
            raise RuntimeError("reranker unavailable")

        reranked = []
        for candidate in candidates:
            item = dict(candidate)
            item["rerank_score"] = self.score_map.get(candidate["chunk_id"], 0.0)
            reranked.append(item)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[: top_k or len(reranked)]


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        return type("Response", (), {"content": "alt query one\nalt query two"})()


def test_lexical_expansion_creates_alternatives():
    retriever = ProductionRetriever(vector_store=FakeVectorStore({}), reranker=FakeReranker({}))
    expansions = retriever._lexical_expansion("What penalty applies for KYC non-compliance?", 3)

    assert expansions
    assert any("fine" in expansion.lower() or "know your customer" in expansion.lower() for expansion in expansions)


def test_lexical_expansion_adds_domain_specific_ltv_queries():
    retriever = ProductionRetriever(vector_store=FakeVectorStore({}), reranker=FakeReranker({}))

    expansions = retriever._lexical_expansion(
        "What is the loan-to-value ratio requirement for loans against gold collateral, and cite the source?",
        3,
    )

    assert any("maximum ltv ratio" in expansion.lower() for expansion in expansions)


def test_retrieve_deduplicates_across_query_variants():
    mapping = {
        "What penalty applies for KYC non-compliance?": [
            {"chunk_id": "a", "text": "Penalty under section 1", "score": 0.81},
            {"chunk_id": "b", "text": "General compliance note", "score": 0.60},
        ],
        "What fine applies for KYC non-compliance?": [
            {"chunk_id": "a", "text": "Penalty under section 1", "score": 0.79},
            {"chunk_id": "c", "text": "Specific sanctions clause", "score": 0.75},
        ],
    }
    retriever = ProductionRetriever(
        vector_store=FakeVectorStore(mapping),
        reranker=FakeReranker({"c": 0.95, "a": 0.85, "b": 0.30}),
        query_expander=lambda _query, _limit: ["What fine applies for KYC non-compliance?"],
    )

    results = retriever.retrieve("What penalty applies for KYC non-compliance?", rerank_top_k=2)

    assert [item["chunk_id"] for item in results] == ["c", "a"]
    assert results[1]["matched_queries"] == [
        "What penalty applies for KYC non-compliance?",
        "What fine applies for KYC non-compliance?",
    ]


def test_retrieve_falls_back_to_similarity_scores_when_reranker_fails():
    mapping = {
        "gold loan ltv": [
            {"chunk_id": "a", "text": "Seventy five percent", "score": 0.52},
            {"chunk_id": "b", "text": "Loan to value ratio", "score": 0.91},
        ]
    }
    retriever = ProductionRetriever(
        vector_store=FakeVectorStore(mapping),
        reranker=FakeReranker({}, should_fail=True),
    )

    results = retriever.retrieve("gold loan ltv", rerank_top_k=2)

    assert [item["chunk_id"] for item in results] == ["b", "a"]


def test_retrieve_promotes_requirement_chunk_when_query_asks_for_threshold():
    query = "What is the loan-to-value ratio requirement for loans against gold collateral?"
    mapping = {
        query: [
            {
                "chunk_id": "definition",
                "text": "Loan to Value (LTV) ratio means the ratio of the outstanding loan amount to the value of the pledged eligible collateral.",
                "score": 0.91,
                "metadata": {"section_title": "6. Definitions"},
            },
        ],
        "maximum LTV ratio for loans against gold collateral": [
            {
                "chunk_id": "requirement",
                "text": (
                    "19. The maximum LTV ratio shall not exceed the table below: "
                    "up to Rs. 2.5 lakh 85 per cent; above Rs. 2.5 lakh and up to Rs. 5 lakh "
                    "80 per cent; above Rs. 5 lakh 75 per cent."
                ),
                "score": 0.62,
                "metadata": {"section_title": "19. Maximum LTV ratio"},
            },
        ],
    }
    retriever = ProductionRetriever(
        vector_store=FakeVectorStore(mapping),
        reranker=FakeReranker({"definition": 4.7, "requirement": 0.7}),
        query_expander=lambda _query, _limit: ["maximum LTV ratio for loans against gold collateral"],
    )

    results = retriever.retrieve(query, rerank_top_k=2)

    assert [item["chunk_id"] for item in results] == ["requirement", "definition"]
    assert results[0]["query_bonus"] > results[1]["query_bonus"]


def test_retrieve_does_not_use_llm_query_expansion_by_default():
    llm = CountingLLM()
    mapping = {
        "What penalty applies for KYC non-compliance?": [
            {"chunk_id": "a", "text": "Penalty under section 1", "score": 0.81},
        ]
    }
    retriever = ProductionRetriever(
        vector_store=FakeVectorStore(mapping),
        llm=llm,
        reranker=FakeReranker({"a": 0.81}),
    )

    results = retriever.retrieve("What penalty applies for KYC non-compliance?", rerank_top_k=1)

    assert [item["chunk_id"] for item in results] == ["a"]
    assert llm.calls == 0


def test_cross_encoder_reranker_returns_full_list_when_top_k_is_none():
    reranker = CrossEncoderReranker(
        scorer=lambda _query, _candidates: [0.2, 0.1],
    )
    candidates = [
        {"chunk_id": "a", "text": "First candidate"},
        {"chunk_id": "b", "text": "Second candidate"},
    ]

    results = reranker.rerank("ltv ratio requirement", candidates, top_k=None)

    assert [item["chunk_id"] for item in results] == ["a", "b"]


def test_get_vector_store_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_vector_store("pinecone")
