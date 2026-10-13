"""Stage 7 Main Orchestrator: Wires Stage 6 reasoning, DB caching, hybrid retrieval, and LLM generation."""

import logging
import re
import time
from typing import Dict, List, Optional, Set
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.connection import get_session_factory
from app.db.models import NexiaChunk, NexiaDocument
from app.retrieval.evidence_reasoner import EvidenceReasoner
from app.schemas.answer import (
    AnswerRequest,
    AnswerResponse,
    RetrievalDiagnostics,
    SourceReference,
    TimingBreakdown,
)
from app.schemas.evidence_package import EvidenceReasoningRequest
from app.services.document_cache import DocumentCacheService
from app.services.hybrid_retrieval import HybridRetrievalService, RetrievedChunk
from app.services.llm_adapter import (
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    OllamaLLMAdapter,
)
from app.services.prompts import PromptContext

logger = logging.getLogger("nexia.orchestrator")


from app.retrieval.passage_retriever import PassageRetriever
from app.schemas.retrieval import PassageRetrievalRequest


class AnswerOrchestrator:
    """End-to-end question answering engine integrating all Nexia stages."""

    def __init__(
        self,
        evidence_reasoner: Optional[EvidenceReasoner] = None,
        document_cache: Optional[DocumentCacheService] = None,
        hybrid_retrieval: Optional[HybridRetrievalService] = None,
        llm_adapter: Optional[OllamaLLMAdapter] = None,
        passage_retriever: Optional[PassageRetriever] = None,
    ):
        self.evidence_reasoner = evidence_reasoner or EvidenceReasoner()
        self.document_cache = document_cache or DocumentCacheService()
        self.hybrid_retrieval = hybrid_retrieval or HybridRetrievalService()
        self.llm_adapter = llm_adapter or OllamaLLMAdapter()
        self.passage_retriever = passage_retriever or PassageRetriever()

    async def answer(self, request: AnswerRequest) -> AnswerResponse:
        """Process a user question through the complete pipeline to generate a grounded answer."""
        t_req_start = time.perf_counter()
        question = request.question.strip()
        logger.info(">>> [Stage 7] Starting pipeline for question: '%s'", question)

        diagnostics = RetrievalDiagnostics(
            embedding_model=settings.EMBEDDING_MODEL,
            llm_model=settings.LLM_MODEL,
        )

        # 1. Execute Stage 6 Evidence Reasoning
        logger.info("[Stage 7: 1/4] Running Stage 6 Evidence Reasoning...")
        t_s6_start = time.perf_counter()
        reasoning_req = EvidenceReasoningRequest(
            question=question,
            subject_entity=request.subject_entity,
            object_entity=request.object_entity,
            predicate_filter=request.predicate_filter,
            top_k_per_document=request.top_k_per_document,
            min_confidence=request.min_confidence,
        )

        try:
            reasoning_res = await self.evidence_reasoner.reason(reasoning_req)
            evidence_package = reasoning_res.evidence_package
        except Exception as exc:
            logger.error("Stage 6 Evidence reasoning failed: %s", exc)
            return AnswerResponse(
                question=question,
                answer_text=f"Failed to execute evidence reasoning pipeline: {exc}",
                answer_status="RETRIEVAL_FAILED",
                source_references=[],
                conflicts=[],
                sufficiency_rating="INSUFFICIENT",
                retrieval_diagnostics=diagnostics,
                error_message=str(exc),
            )
        stage6_ms = round((time.perf_counter() - t_s6_start) * 1000, 2)
        logger.info("[Stage 7: 1/4] Stage 6 Reasoning finished in %.2f ms", stage6_ms)

        # 2. Identify all relevant source documents from evidence clusters & candidates
        target_docs: Set[str] = set()
        for cluster in evidence_package.evidence_clusters:
            for item in cluster.items:
                doc_title = getattr(item, "source_document", None) or getattr(item, "document_title", None)
                if doc_title:
                    target_docs.add(doc_title)

        # If question intent has entities, add them as well if needed
        if evidence_package.question_intent:
            for ent in evidence_package.question_intent.entities:
                target_docs.add(ent.text)
                if getattr(ent, "normalized_text", None):
                    target_docs.add(ent.normalized_text)

            rel = evidence_package.question_intent.relation
            if rel:
                q_low = question.lower()
                office = (
                    "rulers" if "ruler" in q_low
                    else "prime ministers" if "prime minister" in q_low
                    else "presidents" if "president" in q_low
                    else "monarchs" if "monarch" in q_low
                    else "kings" if "king" in q_low
                    else "emperors" if "emperor" in q_low
                    else "chancellors" if "chancellor" in q_low
                    else "leaders"
                )
                if rel.type == "FATHER_OF_NATION_OF":
                    target_docs.add("Father of the Nation")
                    for e in evidence_package.question_intent.entities:
                        target_docs.add(f"Father of the Nation ({e.normalized_text})")
                        target_docs.add(e.normalized_text)
                elif rel.type in ("OFFICE_HOLDERS_OF", "ORDINAL_OFFICE_HOLDER"):
                    terms = getattr(evidence_package.question_intent, "retrieval_terms", []) or []
                    for e in evidence_package.question_intent.entities:
                        e_clean = e.normalized_text
                        target_docs.add(e_clean)
                        if office != "leaders":
                            target_docs.add(f"List of {office} of {e_clean}")
                            target_docs.add(f"{office} of {e_clean}")
                    for t in terms[:4]:
                        if t and len(t.split()) <= 6:
                            target_docs.add(t)
                elif rel.type == "FOUNDED_BY":
                    for e in evidence_package.question_intent.entities:
                        e_clean = e.normalized_text
                        target_docs.add(e_clean)
                        if "dynasty" in e_clean.lower():
                            target_docs.add(re.sub(r"\bdynasty\b", "Empire", e_clean, flags=re.IGNORECASE).strip())
                            target_docs.add(re.sub(r"\bdynasty\b", "Kingdom", e_clean, flags=re.IGNORECASE).strip())

        STOPWORD_DOCS = {
            "list", "what", "who", "where", "when", "why", "how", "which",
            "name", "names", "tell", "show", "give", "explain", "describe",
            "find", "enumerate", "details", "information", "the", "a", "an", "is", "are"
        }
        target_docs = {d for d in target_docs if d and d.strip().lower() not in STOPWORD_DOCS}

        diagnostics.documents_checked = len(target_docs)

        # 3. Synchronize document cache & database ingestion
        logger.info("[Stage 7: 2/4] Checking PostgreSQL document cache for %d documents: %s...", len(target_docs), list(target_docs))
        t_cache_start = time.perf_counter()
        page_ids: List[int] = []
        session_factory = get_session_factory()

        async with session_factory() as session:
            for doc_title in target_docs:
                try:
                    doc, status = await self.document_cache.get_or_fetch_document(
                        title=doc_title,
                        force_refresh=request.force_refresh,
                        session=session,
                    )
                    logger.info("  --> Cache status for '%s': %s (page_id=%s)", doc_title, status, getattr(doc, 'page_id', 'unknown'))
                    if status == "HIT":
                        diagnostics.cache_hits += 1
                    else:
                        diagnostics.cache_misses += 1
                        diagnostics.documents_fetched += 1

                    if doc.page_id not in page_ids:
                        page_ids.append(doc.page_id)
                except Exception as e:
                    logger.warning("Cache fetch failed for document '%s': %s", doc_title, e)

            # If no document was found from entities, search Wikipedia using the question itself
            if not page_ids:
                try:
                    logger.info("Attempting question fallback document search for: '%s'", question)
                    doc, status = await self.document_cache.get_or_fetch_document(
                        title=question,
                        force_refresh=request.force_refresh,
                        session=session,
                    )
                    logger.info("  --> Fallback resolved to document '%s' (page_id=%s, status=%s)", doc.title, doc.page_id, status)
                    if status == "HIT":
                        diagnostics.cache_hits += 1
                    else:
                        diagnostics.cache_misses += 1
                        diagnostics.documents_fetched += 1
                    if doc.page_id not in page_ids:
                        page_ids.append(doc.page_id)
                except Exception as e:
                    logger.warning("Fallback question document fetch failed for '%s': %s", question, e)

            # Record total chunks currently in DB
            chunks_count_res = await session.execute(select(func.count()).select_from(NexiaChunk))
            diagnostics.chunks_in_db = chunks_count_res.scalar() or 0
        cache_total_ms = round((time.perf_counter() - t_cache_start) * 1000, 2)

        # 4. Perform Hybrid Vector + Keyword Retrieval
        logger.info("[Stage 7: 3/4] Running Hybrid Retrieval (pgvector + FTS) for '%s'...", question)
        t_ret_start = time.perf_counter()
        retrieved_chunks: List[RetrievedChunk] = []

        if not page_ids:
            logger.warning("No relevant Wikipedia document found for question '%s'. Skipping retrieval to prevent cross-document contamination.", question)
        else:
            async with session_factory() as session:
                try:
                    retrieved_chunks = await self.hybrid_retrieval.retrieve(
                        question=question,
                        page_ids=page_ids,
                        top_k=settings.HYBRID_TOP_K,
                        session=session,
                    )
                except Exception as e:
                    logger.error("Hybrid retrieval failed: %s", e)
        retrieval_ms = round((time.perf_counter() - t_ret_start) * 1000, 2)

        diagnostics.vector_candidates = sum(
            1 for c in retrieved_chunks if c.vector_rank is not None
        )
        diagnostics.keyword_candidates = sum(
            1 for c in retrieved_chunks if c.keyword_rank is not None
        )
        logger.info("[Stage 7: 3/4] Hybrid retrieval found %d candidates (vector=%d, keyword=%d) in %.2f ms",
                    len(retrieved_chunks), diagnostics.vector_candidates, diagnostics.keyword_candidates, retrieval_ms)

        # 4b. Harvest Stage 3 BM25 multi-factor reranked evidence passages
        stage3_passages: List[RetrievedChunk] = []
        if page_ids:
            try:
                async with session_factory() as session:
                    doc_titles_res = await session.execute(
                        select(NexiaDocument.title).where(NexiaDocument.page_id.in_(page_ids))
                    )
                    doc_titles = [r[0] for r in doc_titles_res.fetchall()]

                for d_title in doc_titles:
                    p_res = await self.passage_retriever.retrieve_passages(
                        PassageRetrievalRequest(
                            question=question,
                            document=d_title,
                            top_k=3,
                        )
                    )
                    for idx, ep in enumerate(p_res.evidence):
                        if ep.score >= 0.5:
                            stage3_passages.append(
                                RetrievedChunk(
                                    chunk_id=-idx - 1,
                                    document_title=ep.document_title,
                                    source_url=f"https://en.wikipedia.org/wiki/{ep.document_title.replace(' ', '_')}",
                                    section_title=ep.section_title,
                                    passage_text=ep.text,
                                    rrf_score=ep.score,
                                    retrieval_method="bm25",
                                )
                            )
            except Exception as e:
                logger.warning("Stage 3 BM25 passage harvesting failed: %s", e)

        # 5. Evidence Selection for LLM
        t_prompt_start = time.perf_counter()

        # Combine Stage 3 BM25 passages with PostgreSQL hybrid retrieval chunks
        merged_chunks: List[RetrievedChunk] = []
        seen_texts: Set[str] = set()

        for sc in stage3_passages:
            snippet_key = sc.passage_text[:60].strip().lower()
            if snippet_key not in seen_texts:
                seen_texts.add(snippet_key)
                merged_chunks.append(sc)

        for rc in retrieved_chunks:
            snippet_key = rc.passage_text[:60].strip().lower()
            if snippet_key not in seen_texts:
                seen_texts.add(snippet_key)
                merged_chunks.append(rc)

        selected_chunks = merged_chunks[: request.max_evidence_passages]
        diagnostics.passages_sent_to_llm = len(selected_chunks)

        # Build SourceReference models
        source_refs: List[SourceReference] = []
        passages_for_prompt: List[Dict] = []

        for c in selected_chunks:
            source_refs.append(
                SourceReference(
                    chunk_id=c.chunk_id,
                    document_title=c.document_title,
                    source_url=c.source_url,
                    section_title=c.section_title,
                    passage_text=c.passage_text,
                    rrf_score=c.rrf_score,
                    retrieval_method=c.retrieval_method,
                )
            )
            # Bound passage text sent to LLM to prevent prompt token bloat on CPU
            words = c.passage_text.strip().split()
            bounded_text = " ".join(words[:120]) if len(words) > 120 else c.passage_text.strip()
            if len(words) > 120:
                bounded_text += "..."

            passages_for_prompt.append(
                {
                    "document_title": c.document_title,
                    "section_title": c.section_title,
                    "source_url": c.source_url,
                    "passage_text": bounded_text,
                }
            )

        # Extract structured conflict descriptions
        conflicts_text = [
            f"Conflict between {getattr(c, 'entity_1', 'e1')} and {getattr(c, 'entity_2', 'e2')}: {getattr(c, 'discrepancy', str(c))}"
            for c in evidence_package.conflicts
        ]

        sufficiency_status = (
            evidence_package.sufficiency_assessment.status
            if evidence_package.sufficiency_assessment
            else "UNKNOWN"
        )
        if selected_chunks and sufficiency_status in ("INSUFFICIENT", "UNKNOWN"):
            sufficiency_status = "SUFFICIENT"

        # Intent fields
        q_type = (
            evidence_package.question_intent.question_type
            if evidence_package.question_intent
            else "FACTOID"
        )
        q_entities = (
            [e.text for e in evidence_package.question_intent.entities]
            if evidence_package.question_intent
            else []
        )
        raw_rel = (
            evidence_package.question_intent.relation
            if evidence_package.question_intent
            else None
        )
        if raw_rel is not None:
            q_relation = getattr(raw_rel, "type", str(raw_rel))
        else:
            q_relation = None

        # 6. Build PromptContext and Invoke LLM Adapter
        prompt_ctx = PromptContext(
            question=question,
            question_type=q_type,
            entities=q_entities,
            relation=q_relation,
            passages=passages_for_prompt,
            conflicts=conflicts_text,
            sufficiency_rating=sufficiency_status,
        )
        prompt_construction_ms = round((time.perf_counter() - t_prompt_start) * 1000, 2)

        logger.info("[Stage 7: 4/4] Sending %d bounded passages to Ollama (%s)...", len(passages_for_prompt), settings.LLM_MODEL)
        t_llm_start = time.perf_counter()
        llm_load_ms = None
        llm_pe_ms = None
        llm_eval_ms = None
        llm_tps = None
        prompt_tokens = None
        resp_tokens = None

        try:
            llm_result = await self.llm_adapter.generate_answer(prompt_ctx)
            answer_text = llm_result.answer_text
            # Clean any leading meta-phrasing or robotic formulas from LLM output
            answer_text = re.sub(
                r"^(?:According to the provided (?:document|documents|text|passages?)|Based on the provided (?:document|documents|text|passages?)|The provided text (?:states|defines|describes|indicates|explains)|The passage states that)\s*[:,]?\s*",
                "",
                answer_text,
                flags=re.IGNORECASE,
            ).strip()
            if answer_text and answer_text[0].islower():
                answer_text = answer_text[0].upper() + answer_text[1:]
            status = llm_result.status
            llm_load_ms = llm_result.load_duration_ms
            llm_pe_ms = llm_result.prompt_eval_duration_ms
            llm_eval_ms = llm_result.eval_duration_ms
            llm_tps = llm_result.tokens_per_second
            prompt_tokens = llm_result.prompt_tokens
            resp_tokens = llm_result.response_tokens
            error_msg = None
            logger.info("[Stage 7: 4/4] Ollama answer generated in %.2f ms (tokens: %s prompt, %s eval)",
                        llm_result.generation_time_ms, prompt_tokens, resp_tokens)
        except (LLMUnavailableError, LLMTimeoutError, LLMOutputError) as exc:
            logger.error("LLM Generation failed: %s", exc)
            answer_text = f"LLM answer generation failed: {exc}"
            status = "LLM_FAILED"
            error_msg = str(exc)

        llm_duration_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
        total_req_ms = round((time.perf_counter() - t_req_start) * 1000, 2)

        # Assemble Fine-Grained Timing Breakdown
        diagnostics.timings = TimingBreakdown(
            total_request_ms=total_req_ms,
            stage6_reasoning_ms=stage6_ms,
            db_cache_lookup_ms=cache_total_ms,
            wikipedia_network_ms=0.0,
            chunking_ms=0.0,
            embedding_generation_ms=0.0,
            embedding_batch_count=0,
            db_write_ms=0.0,
            vector_retrieval_ms=0.0,
            keyword_retrieval_ms=0.0,
            rrf_fusion_ms=0.0,
            hybrid_retrieval_total_ms=retrieval_ms,
            evidence_selection_prompt_ms=prompt_construction_ms,
            llm_request_ms=llm_duration_ms,
            llm_load_duration_ms=llm_load_ms,
            llm_prompt_eval_duration_ms=llm_pe_ms,
            llm_eval_duration_ms=llm_eval_ms,
            llm_tokens_per_second=llm_tps,
            llm_prompt_tokens=prompt_tokens,
            llm_response_tokens=resp_tokens,
            llm_call_count=1,
            response_serialization_ms=0.0,
        )

        logger.info("=== [Stage 7] Finished in %.2fs (status: %s, LLM: %.2fs) ===",
                    total_req_ms / 1000, status, llm_duration_ms / 1000)


        # 7. Assemble final AnswerResponse
        return AnswerResponse(
            question=question,
            answer_text=answer_text,
            answer_status=status,
            source_references=source_refs,
            conflicts=conflicts_text,
            sufficiency_rating=sufficiency_status,
            retrieval_diagnostics=diagnostics,
            stage6_evidence_package=evidence_package,
            error_message=error_msg,
        )
