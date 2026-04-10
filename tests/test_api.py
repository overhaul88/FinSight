"""Integration tests for the FastAPI surface."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from src.serving.api import create_app


class FakeChain:
    def invoke(self, question: str, retrieval_top_k=None, rerank_top_k=None):
        return {
            "answer": f"Answer for: {question}",
            "sources": [
                {
                    "text": "The maximum LTV ratio is 75 percent.",
                    "source": "RBI_gold_loan.pdf",
                    "section": "3. CUSTOMER IDENTIFICATION",
                    "rerank_score": 0.91,
                    "doc_type": "RBI_Circular",
                }
            ],
            "query": question,
        }

    async def astream(self, question: str, retrieval_top_k=None, rerank_top_k=None):
        for token in ["Answer ", "stream"]:
            yield token


def build_app(ready: bool):
    app = create_app()
    runtime = app.state.runtime
    runtime.is_ready = ready
    runtime.chain = FakeChain() if ready else None
    runtime.llm = object() if ready else None
    runtime.vector_store = object() if ready else None
    runtime.total_chunks = 12 if ready else 0
    return app


@pytest.mark.asyncio
async def test_health_endpoint_returns_200():
    app = build_app(ready=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["total_indexed_chunks"] == 12
    assert payload["load_error"] == ""


@pytest.mark.asyncio
async def test_root_endpoint():
    app = build_app(ready=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "FinSight API"


@pytest.mark.asyncio
async def test_query_returns_503_when_not_ready():
    app = build_app(ready=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"question": "What is the gold loan LTV ratio?"})

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_health_endpoint_reports_loader_error():
    app = build_app(ready=False)
    runtime = app.state.runtime
    runtime.load_error = "Missing FAISS artifacts"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["load_error"] == "Missing FAISS artifacts"


@pytest.mark.asyncio
async def test_query_returns_loader_error_when_startup_failed():
    app = build_app(ready=False)
    runtime = app.state.runtime
    runtime.load_error = "Missing FAISS artifacts"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"question": "What is the gold loan LTV ratio?"})

    assert response.status_code == 503
    assert "Missing FAISS artifacts" in response.json()["detail"]


@pytest.mark.asyncio
async def test_query_validates_short_question():
    app = build_app(ready=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"question": "Hi"})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_returns_answer_and_sources():
    app = build_app(ready=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"question": "What is the gold loan LTV ratio?", "top_k": 2},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Answer for: What is the gold loan LTV ratio?"
    assert payload["sources"][0]["source_file"] == "RBI_gold_loan.pdf"


@pytest.mark.asyncio
async def test_stream_endpoint_returns_sse_payload():
    app = build_app(ready=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query/stream",
            json={"question": "What is the gold loan LTV ratio?"},
        )

    assert response.status_code == 200
    body = response.text
    events = [line.removeprefix("data: ").strip() for line in body.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(event) for event in events]
    assert any(payload["type"] == "token" for payload in payloads)
    assert payloads[-1]["type"] == "done"
