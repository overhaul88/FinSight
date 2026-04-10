"""RAG chain assembly for FinSight."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List

from src.retrieval.retriever import ProductionRetriever


SYSTEM_PROMPT = """You are FinSight, an AI assistant specializing in Indian financial regulations.
You answer questions based exclusively on the provided regulatory context.

Strict rules:
1. Use only the supplied context.
2. If the context is insufficient, say exactly: "Insufficient evidence in retrieved context."
3. If the context explicitly states a definition, ratio, percentage, threshold, date, or procedural requirement, state that exact point directly in the first sentence.
4. Do not hedge when the retrieved text is explicit.
5. Cite the supporting documents in a separate citations block.
6. Keep the answer short and compliance-oriented.

Return exactly this format:
Answer: <2-4 sentences. Use the exact rule or definition when available.>
Citations:
- [Document N] <source file> | <section>
- [Document N] <source file> | <section>

Context Documents:
{context}
"""

USER_TEMPLATE = """Question: {question}

Return only the Answer block and the Citations block."""


def format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a prompt-ready context block."""

    context_parts = []
    for index, document in enumerate(retrieved_docs, start=1):
        metadata = document.get("metadata", {})
        source = metadata.get("filename", "Unknown")
        section = metadata.get("section_title", "")
        doc_type = metadata.get("doc_type", "")
        score = document.get("rerank_score", document.get("score", 0.0))

        header_lines = [
            f"[Document {index}]",
            f"Source File: {source}",
        ]
        if doc_type:
            header_lines.append(f"Document Type: {doc_type}")
        if section:
            header_lines.append(f"Section: {section}")
        header_lines.append(f"Relevance: {score:.3f}")
        header_lines.append("Content:")

        header = "\n".join(header_lines)
        context_parts.append(f"{header}\n{document.get('text', '')}")

    return "\n\n---\n\n".join(context_parts)


class FinSightChain:
    """Compose retrieval, prompt rendering, and model invocation."""

    def __init__(self, retriever: ProductionRetriever, llm: Any):
        self.retriever = retriever
        self.llm = llm

    def build_prompt(self, question: str, context: str) -> str:
        return f"{SYSTEM_PROMPT.format(context=context)}\n{USER_TEMPLATE.format(question=question)}"

    def invoke(
        self,
        query: str,
        retrieval_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> Dict[str, Any]:
        documents = self._retrieve_documents(query, retrieval_top_k, rerank_top_k)
        context = format_context(documents)
        prompt = self.build_prompt(query, context)
        response = self.llm.invoke(prompt)
        answer = self._coerce_text(response)

        return {
            "answer": answer,
            "sources": [
                {
                    "text": document.get("text", "")[:200] + "...",
                    "source": document.get("metadata", {}).get("filename", ""),
                    "section": document.get("metadata", {}).get("section_title", ""),
                    "rerank_score": document.get("rerank_score", document.get("score", 0.0)),
                    "doc_type": document.get("metadata", {}).get("doc_type", ""),
                }
                for document in documents
            ],
            "query": query,
            "context": context,
        }

    async def astream(
        self,
        query: str,
        retrieval_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> AsyncIterator[str]:
        documents = self._retrieve_documents(query, retrieval_top_k, rerank_top_k)
        context = format_context(documents)
        prompt = self.build_prompt(query, context)

        if hasattr(self.llm, "astream"):
            async for chunk in self.llm.astream(prompt):
                yield self._coerce_text(chunk)
            return

        yield self._coerce_text(self.llm.invoke(prompt))

    @staticmethod
    def _coerce_text(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        content = getattr(payload, "content", None)
        if content is not None:
            return str(content)
        return str(payload)

    def _retrieve_documents(
        self,
        query: str,
        retrieval_top_k: int | None,
        rerank_top_k: int | None,
    ) -> List[Dict[str, Any]]:
        try:
            return self.retriever.retrieve(
                query,
                retrieval_top_k=retrieval_top_k,
                rerank_top_k=rerank_top_k,
            )
        except TypeError:
            return self.retriever.retrieve(query)
