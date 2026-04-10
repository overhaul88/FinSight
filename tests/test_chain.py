"""Tests for the chain and model fallback layer."""

from __future__ import annotations

import pytest

from src.llm.model import DevelopmentFallbackLLM
from src.retrieval.chain import FinSightChain, format_context


class FakeRetriever:
    def retrieve(self, query: str):
        return [
            {
                "text": "The maximum LTV ratio is 75 percent.",
                "metadata": {
                    "filename": "RBI_gold_loan.pdf",
                    "section_title": "3. CUSTOMER IDENTIFICATION",
                    "doc_type": "RBI_Circular",
                },
                "rerank_score": 0.91,
            }
        ]


def test_format_context_includes_source_metadata():
    context = format_context(FakeRetriever().retrieve("gold loan"))

    assert "Source File: RBI_gold_loan.pdf" in context
    assert "Document Type: RBI_Circular" in context
    assert "Section: 3. CUSTOMER IDENTIFICATION" in context
    assert "RBI_gold_loan.pdf" in context
    assert "3. CUSTOMER IDENTIFICATION" in context
    assert "75 percent" in context


def test_build_prompt_contains_answer_contract():
    chain = FinSightChain(FakeRetriever(), DevelopmentFallbackLLM(reason="prompt test"))
    prompt = chain.build_prompt(
        "What is the gold loan LTV ratio?",
        format_context(FakeRetriever().retrieve("gold loan")),
    )

    assert "Return exactly this format:" in prompt
    assert "Answer:" in prompt
    assert "Citations:" in prompt
    assert "Return only the Answer block and the Citations block." in prompt


def test_chain_returns_answer_and_sources():
    chain = FinSightChain(FakeRetriever(), DevelopmentFallbackLLM(reason="test fallback"))
    result = chain.invoke("What is the gold loan LTV ratio?")

    assert result["query"] == "What is the gold loan LTV ratio?"
    assert result["sources"][0]["source"] == "RBI_gold_loan.pdf"
    assert result["sources"][0]["doc_type"] == "RBI_Circular"
    assert "Development fallback response" in result["answer"]


@pytest.mark.asyncio
async def test_chain_streams_tokens_from_fallback_model():
    chain = FinSightChain(FakeRetriever(), DevelopmentFallbackLLM(reason="stream test"))
    chunks = []

    async for token in chain.astream("What is the gold loan LTV ratio?"):
        chunks.append(token)

    assert chunks
    assert "Development" in "".join(chunks)
