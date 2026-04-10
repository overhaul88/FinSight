"""FastAPI application for FinSight."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.llm.model import load_mistral_with_adapter
from src.retrieval.chain import FinSightChain
from src.retrieval.retriever import ProductionRetriever
from src.retrieval.vector_store import get_vector_store
from src.serving.schemas import HealthResponse, QueryRequest, QueryResponse, SourceDocument


logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    """Shared application state."""

    vector_store: Any = None
    retriever: Any = None
    chain: Any = None
    llm: Any = None
    is_ready: bool = False
    total_chunks: int = 0
    load_error: str = ""


async def _default_component_loader(state: RuntimeState) -> None:
    vector_store = get_vector_store("faiss")
    llm = load_mistral_with_adapter()
    retriever = ProductionRetriever(vector_store, llm=llm)
    chain = FinSightChain(retriever, llm)

    state.vector_store = vector_store
    state.llm = llm
    state.retriever = retriever
    state.chain = chain
    store = getattr(vector_store, "store", None)
    state.total_chunks = getattr(store, "total_vectors", 0)


async def _run_component_loader(
    loader: Callable[[RuntimeState], Awaitable[None]],
    state: RuntimeState,
) -> None:
    try:
        await loader(state)
    except Exception as exc:
        state.vector_store = None
        state.llm = None
        state.retriever = None
        state.chain = None
        state.total_chunks = 0
        state.is_ready = False
        state.load_error = str(exc)
        logger.exception("FinSight startup failed: %s", exc)
        return

    state.is_ready = True
    state.load_error = ""


def _check_ready(state: RuntimeState) -> None:
    if not state.is_ready:
        if state.load_error:
            raise HTTPException(
                status_code=503,
                detail=f"Service failed to initialize: {state.load_error}",
            )
        raise HTTPException(
            status_code=503,
            detail="Service is starting up. Model loading in progress.",
        )


def _format_sources(raw_sources: list[dict[str, Any]]) -> list[SourceDocument]:
    return [
        SourceDocument(
            text_preview=source.get("text", source.get("text_preview", ""))[:200] + "...",
            source_file=source.get("source", source.get("source_file", "")),
            section=source.get("section", ""),
            relevance_score=round(float(source.get("rerank_score", source.get("relevance_score", 0.0))), 4),
            doc_type=source.get("doc_type", ""),
        )
        for source in raw_sources
    ]


def create_app(
    component_loader: Callable[[RuntimeState], Awaitable[None]] | None = None,
    background_load: bool = True,
) -> FastAPI:
    """Application factory with injectable startup loading."""

    loader = component_loader or _default_component_loader
    runtime = RuntimeState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if background_load:
            asyncio.create_task(_run_component_loader(loader, runtime))
        else:
            await _run_component_loader(loader, runtime)
        yield

    app = FastAPI(
        title="FinSight API",
        description="Production RAG system for Indian financial regulatory documents",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["System"])
    async def root():
        return {
            "service": "FinSight API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        if runtime.is_ready:
            status = "healthy"
        elif runtime.load_error:
            status = "error"
        else:
            status = "loading"
        return HealthResponse(
            status=status,
            vector_store_loaded=runtime.vector_store is not None,
            model_loaded=runtime.llm is not None,
            total_indexed_chunks=runtime.total_chunks,
            load_error=runtime.load_error,
        )

    @app.post("/query", response_model=QueryResponse, tags=["RAG"])
    async def query(request: QueryRequest):
        _check_ready(runtime)
        started_at = time.time()
        result = runtime.chain.invoke(request.question, rerank_top_k=request.top_k)
        latency_ms = round((time.time() - started_at) * 1000, 2)
        return QueryResponse(
            answer=result["answer"],
            sources=_format_sources(result.get("sources", [])),
            query=request.question,
            latency_ms=latency_ms,
        )

    @app.post("/query/stream", tags=["RAG"])
    async def query_stream(request: QueryRequest):
        _check_ready(runtime)

        async def event_generator() -> AsyncIterator[str]:
            try:
                async for token in runtime.chain.astream(
                    request.question,
                    rerank_top_k=request.top_k,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
