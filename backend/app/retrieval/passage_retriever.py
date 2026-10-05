"""Passage Retriever: Multi-Document Candidate Harvesting, Structural Reranking, and Strict Constraint Validation."""

import asyncio
from typing import Optional, List, Dict
from app.schemas.retrieval import (
    PassageRetrievalRequest,
    PassageRetrievalResponse,
    EvidencePassage,
    Document,
    SearchCandidate,
    Passage,
)
from app.retrieval.document_retriever import DocumentRetriever
from app.retrieval.passage_builder import PassageBuilder
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.bm25_scorer import BM25RelevanceScorer
from app.retrieval.tfidf_scorer import TfidfRelevanceScorer
from app.retrieval.reranker import MultiFactorReranker
from app.retrieval.evidence_validator import EvidenceValidator


class PassageRetriever:
    """Coordinates universal question-understanding, multi-document candidate harvesting, multi-factor reranking, and constraint validation."""

    def __init__(self, document_retriever: Optional[DocumentRetriever] = None):
        self.doc_retriever = document_retriever or DocumentRetriever()
        self.passage_builder = PassageBuilder()
        self.query_analyzer = QueryAnalyzer()

    async def retrieve_passages(
        self, request: PassageRetrievalRequest
    ) -> PassageRetrievalResponse:
        """Execute question-aware retrieval pipeline with multi-document harvesting and strict constraint validation."""
        question = request.question.strip()
        
        # 1. Structured Query Intent Analysis
        intent = self.query_analyzer.analyze(question if question else "unknown")

        if not question:
            return PassageRetrievalResponse(
                question="",
                document_title="",
                documents_searched=[],
                algorithm_used=request.algorithm,
                structured_intent=intent,
                status="ERROR",
                is_evidence_sufficient=False,
                validation_reason="Question cannot be empty.",
                total_passages_evaluated=0,
                threshold_used=request.min_score,
                candidate_documents_considered=[],
                evidence=[],
                all_ranked_candidates=[],
                error_message="Question cannot be empty.",
            )

        # 2. Multi-Document Candidate Discovery
        target_doc_name = request.document.strip() if request.document else intent.subject_entity
        candidates_considered: List[SearchCandidate] = []
        documents_to_fetch: List[str] = []

        if request.document and request.document.strip():
            documents_to_fetch = [request.document.strip()]
        else:
            # Search Wikipedia candidates using question and subject entity
            search_query = f"{intent.subject_entity} {intent.relation}".strip() if intent.relation != "general" else question
            search_resp = await self.doc_retriever.search_candidates(search_query, limit=5)
            candidates_considered = search_resp.candidates

            # Take top 2-3 most promising candidate documents for cross-document harvesting
            if candidates_considered:
                documents_to_fetch = [c.title for c in candidates_considered[:3]]
            else:
                documents_to_fetch = [intent.subject_entity]

        # 3. Harvest Passages Across Candidate Documents
        all_harvested_passages: List[Passage] = []
        documents_searched_titles: List[str] = []

        for doc_title in documents_to_fetch:
            try:
                doc_resp = await self.doc_retriever.retrieve_document(
                    query="",
                    explicit_title=doc_title,
                )
                if doc_resp.status == "success" and doc_resp.document:
                    doc: Document = doc_resp.document
                    doc_passages = self.passage_builder.build_passages(doc)
                    all_harvested_passages.extend(doc_passages)
                    documents_searched_titles.append(doc.title)
            except Exception:
                continue

        total_passages = len(all_harvested_passages)
        primary_doc_title = documents_searched_titles[0] if documents_searched_titles else target_doc_name

        if total_passages == 0:
            return PassageRetrievalResponse(
                question=question,
                document_title=primary_doc_title,
                documents_searched=documents_searched_titles,
                algorithm_used=request.algorithm,
                structured_intent=intent,
                status="DOCUMENT_NOT_FOUND",
                is_evidence_sufficient=False,
                validation_reason=f"Target Wikipedia articles could not be retrieved for '{target_doc_name}'",
                total_passages_evaluated=0,
                threshold_used=request.min_score,
                candidate_documents_considered=candidates_considered,
                evidence=[],
                all_ranked_candidates=[],
                error_message=f"No content passages retrieved for '{target_doc_name}'",
            )

        # 4. First-Stage Candidate Retrieval (BM25 or TF-IDF)
        algorithm_choice = request.algorithm.lower().strip()
        if algorithm_choice == "tfidf":
            tfidf_scorer = TfidfRelevanceScorer(all_harvested_passages)
            candidate_items = tfidf_scorer.score_query(question)
        else:
            algorithm_choice = "bm25"
            bm25_scorer = BM25RelevanceScorer(all_harvested_passages)
            candidate_items = bm25_scorer.score_query(intent.expanded_semantic_cues)

        # 5. Second-Stage Structural Relevance Scoring & Multi-Factor Constraint Reranking
        reranked_results = MultiFactorReranker.rerank(candidate_items, intent)

        # 6. Format Candidate Evidence Passages
        all_candidates: List[EvidencePassage] = []
        for rank, (passage, score, matched_terms, unmatched_terms, breakdown, val_details) in enumerate(
            reranked_results[: request.top_k], start=1
        ):
            is_above = score >= request.min_score
            evidence_item = EvidencePassage(
                rank=rank,
                passage_id=passage.passage_id,
                document_title=passage.document_title,
                section_title=passage.section_title,
                section_level=passage.section_level,
                paragraph_id=passage.paragraph_id,
                text=passage.text,
                score=score,
                is_above_threshold=is_above,
                matched_terms=matched_terms,
                unmatched_terms=unmatched_terms,
                score_breakdown=breakdown,
                validation_details=val_details,
            )
            all_candidates.append(evidence_item)

        # 7. Third-Stage Strict Evidence Validation Gate
        validation_result = EvidenceValidator.validate(
            intent=intent,
            candidate_passages=all_candidates,
            min_score=request.min_score,
        )

        return PassageRetrievalResponse(
            question=question,
            document_title=primary_doc_title,
            documents_searched=documents_searched_titles,
            algorithm_used=algorithm_choice,
            structured_intent=intent,
            status=validation_result.status,
            is_evidence_sufficient=validation_result.is_sufficient,
            validation_reason=validation_result.reason,
            total_passages_evaluated=total_passages,
            threshold_used=request.min_score,
            candidate_documents_considered=candidates_considered,
            evidence=validation_result.validated_evidence,
            all_ranked_candidates=all_candidates,
        )
