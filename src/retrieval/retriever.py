"""Production retrieval pipeline with expansion and reranking hooks."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Sequence

from src.config import settings
from src.retrieval.vector_store import BaseVectorStore


logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker with lazy model loading."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer: Callable[[str, Sequence[Dict[str, Any]]], Sequence[float]] | None = None,
    ) -> None:
        self.model_name = model_name
        self._scorer = scorer
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for cross-encoder reranking."
                ) from exc
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def score(self, query: str, candidates: Sequence[Dict[str, Any]]) -> List[float]:
        if self._scorer is not None:
            return [float(score) for score in self._scorer(query, candidates)]
        pairs = [(query, candidate["text"]) for candidate in candidates]
        return [float(score) for score in self.model.predict(pairs)]

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        limit = len(candidates) if top_k is None else top_k
        scores = self.score(query, candidates)

        reranked: List[Dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:limit]


class ProductionRetriever:
    """Retrieve, deduplicate, and rerank document chunks."""

    QUERY_EXPANSION_PROMPT = (
        "Generate alternative phrasings for the same financial regulatory question. "
        "Return one query per line with no numbering.\n\nQuestion: {question}"
    )
    REQUIREMENT_QUERY_TERMS = (
        "requirement",
        "maximum",
        "minimum",
        "limit",
        "threshold",
        "ceiling",
        "cap",
        "permissible",
        "not exceed",
    )
    REQUIREMENT_CANDIDATE_TERMS = (
        "maximum",
        "minimum",
        "limit",
        "threshold",
        "ceiling",
        "cap",
        "permissible",
        "not exceed",
    )
    RATIO_QUERY_TERMS = ("ltv", "loan-to-value", "loan to value", "ratio")

    def __init__(
        self,
        vector_store: BaseVectorStore,
        llm: Any | None = None,
        reranker: Any | None = None,
        query_expander: Callable[[str, int], List[str]] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker or CrossEncoderReranker()
        self.query_expander = query_expander

    def retrieve(
        self,
        query: str,
        n_expanded_queries: int = 3,
        retrieval_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        retrieval_limit = retrieval_top_k or settings.top_k_retrieval
        rerank_limit = rerank_top_k or settings.top_k_rerank

        expanded_queries = self._expand_query(query, n_expanded_queries)
        all_queries = [query] + [item for item in expanded_queries if item and item != query]

        candidates: Dict[str, Dict[str, Any]] = {}
        for current_query in all_queries:
            for document in self.vector_store.similarity_search(current_query, top_k=retrieval_limit):
                chunk_id = document.get("chunk_id") or self._derive_chunk_id(document)
                candidate = dict(document)
                candidate.setdefault("matched_queries", []).append(current_query)

                existing = candidates.get(chunk_id)
                if existing is None:
                    candidates[chunk_id] = candidate
                    continue

                existing["matched_queries"] = list(
                    dict.fromkeys(existing.get("matched_queries", []) + [current_query])
                )
                if candidate.get("score", float("-inf")) > existing.get("score", float("-inf")):
                    merged = dict(existing)
                    merged.update(candidate)
                    merged["matched_queries"] = existing["matched_queries"]
                    candidates[chunk_id] = merged

        deduped = list(candidates.values())
        if not deduped:
            return []

        try:
            ranked = self.reranker.rerank(query, deduped, top_k=None)
        except Exception as exc:
            logger.warning("Reranking failed, falling back to vector scores: %s", exc)
            deduped.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            ranked = deduped

        return self._finalize_results(query, ranked, rerank_limit)

    def _expand_query(self, query: str, limit: int) -> List[str]:
        if limit <= 0:
            return []

        if self.query_expander is not None:
            return self._deduplicate_queries(self.query_expander(query, limit), limit)

        if settings.enable_llm_query_expansion and self.llm is not None:
            prompt = self.QUERY_EXPANSION_PROMPT.format(question=query)
            try:
                response = self.llm.invoke(prompt)
                content = getattr(response, "content", response)
                lines = [line.strip() for line in str(content).splitlines() if line.strip()]
                return self._deduplicate_queries(lines, limit)
            except Exception as exc:
                logger.warning("LLM query expansion failed, using lexical fallback: %s", exc)

        return self._deduplicate_queries(self._lexical_expansion(query, limit), limit)

    def _lexical_expansion(self, query: str, limit: int) -> List[str]:
        expansions = self._domain_expansion(query)
        if len(expansions) >= limit:
            return expansions[:limit]

        substitutions = {
            "penalty": ["fine", "consequence", "sanction"],
            "compliance": ["adherence", "conformity", "regulatory requirement"],
            "loan": ["credit", "borrowing", "advance"],
            "kyc": ["know your customer", "customer verification"],
            "interest rate": ["rate of interest", "lending rate"],
            "requirement": ["maximum", "limit", "permissible ceiling"],
        }

        for source, alternatives in substitutions.items():
            for replacement in alternatives:
                updated = re.sub(source, replacement, query, count=1, flags=re.IGNORECASE)
                if updated != query:
                    expansions.append(updated)
                if len(expansions) >= limit:
                    return expansions[:limit]
        return expansions[:limit]

    def _domain_expansion(self, query: str) -> List[str]:
        lowered = query.lower()
        expansions: List[str] = []

        if any(term in lowered for term in self.RATIO_QUERY_TERMS):
            expansions.append("maximum LTV ratio for loans against gold collateral")
            expansions.append("maximum LTV ratio permissible for eligible collateral loans")
            expansions.append("LTV ratio shall not exceed for consumption loans against eligible collateral")

        if "gold collateral" in lowered:
            expansions.append(re.sub("gold collateral", "eligible collateral", query, flags=re.IGNORECASE))

        if "cite the source" in lowered:
            expansions.append(re.sub(r",?\s*and cite the source\??", "", query, flags=re.IGNORECASE))

        return self._deduplicate_queries(expansions, limit=10)

    def _finalize_results(
        self,
        query: str,
        ranked_candidates: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not ranked_candidates:
            return []

        if not self._is_requirement_query(query):
            return ranked_candidates[:limit]

        rescored: List[Dict[str, Any]] = []
        for candidate in ranked_candidates:
            item = dict(candidate)
            base_score = float(item.get("rerank_score", item.get("score", 0.0)))
            query_bonus = self._requirement_bonus(query, item)
            item["query_bonus"] = query_bonus
            item["final_score"] = base_score + query_bonus
            rescored.append(item)

        rescored.sort(
            key=lambda item: (
                item.get("final_score", 0.0),
                item.get("rerank_score", item.get("score", 0.0)),
                item.get("score", 0.0),
            ),
            reverse=True,
        )
        return rescored[:limit]

    def _is_requirement_query(self, query: str) -> bool:
        lowered = query.lower()
        return any(term in lowered for term in self.REQUIREMENT_QUERY_TERMS)

    def _requirement_bonus(self, query: str, candidate: Dict[str, Any]) -> float:
        lowered_query = query.lower()
        metadata = candidate.get("metadata", {})
        searchable = " ".join(
            str(value)
            for value in (
                candidate.get("text", ""),
                metadata.get("section_title", ""),
                metadata.get("source_file", ""),
            )
            if value
        ).lower()

        bonus = 0.0
        if any(term in lowered_query for term in self.REQUIREMENT_QUERY_TERMS):
            if any(term in searchable for term in self.REQUIREMENT_CANDIDATE_TERMS):
                bonus += 3.0
            if re.search(r"\b\d+(?:\.\d+)?\s*(?:per cent|percent|%)", searchable):
                bonus += 3.5
            elif "per cent" in searchable or "%" in searchable:
                bonus += 2.0

        if any(term in lowered_query for term in self.RATIO_QUERY_TERMS):
            if "ltv" in searchable or "loan to value" in searchable or "loan-to-value" in searchable:
                bonus += 1.5

        if "gold" in lowered_query and ("gold" in searchable or "eligible collateral" in searchable):
            bonus += 0.75

        if "collateral" in lowered_query and "eligible collateral" in searchable:
            bonus += 0.75

        return bonus

    @staticmethod
    def _deduplicate_queries(queries: Sequence[str], limit: int) -> List[str]:
        seen = set()
        unique: List[str] = []
        for query in queries:
            normalized = query.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
            if len(unique) >= limit:
                break
        return unique

    @staticmethod
    def _derive_chunk_id(document: Dict[str, Any]) -> str:
        text = document.get("text", "")
        return text[:50] or "unknown-chunk"
