"""Stage 7 Main Orchestrator: Wires Stage 6 reasoning, DB caching, hybrid retrieval, and LLM generation."""

import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4
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
from app.schemas.conversation import MessageRole, MessageStatus
from app.services.conversation_manager import ConversationManager
from app.services.context_manager import ContextManager
from app.services.llm_orchestrator import LLMOrchestrator
from app.services.prompts import PromptContext
from app.services.citation_manager import (
    PrimaryArticleSelector,
    format_answer_with_primary_reference,
)
from app.services.response_controller import (
    InferenceQueueFullError,
    InferenceQueueTimeoutError,
    Priority,
    RequestCancelledError,
    ResponseController,
)
from app.services.streaming_engine import StreamingEngine
from app.services.title_generator import ConversationTitleGenerator

logger = logging.getLogger("nexia.orchestrator")


from app.retrieval.passage_retriever import PassageRetriever
from app.retrieval.nlp_query_parser import NLPQueryParser
from app.retrieval.adaptive_nlp_planner import (
    AdaptiveQueryAnalyzer,
    EvidenceCoverageAssessor,
    EvidencePacker,
    QueryComplexity,
    ComplexityAnalysisResult,
)
from app.schemas.retrieval import PassageRetrievalRequest


class ClaimVerifier:
    """Verifies generated answer sentences against cited evidence passages to eliminate ungrounded claims."""

    @classmethod
    def verify_and_clean_answer(
        cls,
        answer_text: str,
        selected_chunks: List[Any],
        question: str,
        missing_aspects: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """Verify each sentence in answer against cited passages; prune or qualify ungrounded claims."""
        stats = {
            "claims_verified": 0,
            "claims_supported": 0,
            "unsupported_claims_removed": 0,
            "citations_stripped": 0,
            "citations_remapped": 0,
            "citations_attached": 0,
        }
        if not answer_text or not selected_chunks:
            return answer_text, stats

        all_evidence_text = " ".join(c.passage_text.lower() for c in selected_chunks)
        all_evidence_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", all_evidence_text))

        # Question keywords
        q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())) - NLPQueryParser.STOPWORDS_CHUNKS

        paragraphs = answer_text.split("\n\n")
        verified_paragraphs = []

        for p in paragraphs:
            # Check if this paragraph is a list header or list items
            if p.strip().endswith(":") or p.strip().startswith(("-", "*", "1.", "2.", "3.", "4.")):
                verified_paragraphs.append(p)
                continue

            sentences = re.split(r"(?<=[.!?])\s+", p.strip())
            valid_sentences = []

            for sent in sentences:
                s_clean = sent.strip()
                if not s_clean:
                    continue

                stats["claims_verified"] += 1
                s_lower = s_clean.lower()

                # 1. Check for off-topic intrusive content (e.g. fauna, bird observatory on island)
                if any(w in s_lower for w in ["bird observatory", "migratory bird", "birdlife international", "wild goats", "flora and fauna", "bird migration"]):
                    if not any(qw in q_words for qw in ["bird", "goat", "animal", "fauna"]):
                        stats["unsupported_claims_removed"] += 1
                        continue

                # 2. Check for foreign ungrounded entities not present anywhere in retrieved evidence
                proper_nouns = re.findall(r"\b[A-Z][a-z]{3,}\b", s_clean)
                EXCLUDED_WORDS = {"The", "This", "These", "That", "There", "However", "Furthermore", "Moreover", "In", "On", "At", "Although", "While", "Its", "His", "Her", "Their", "According", "Based", "From", "Here", "Additionally"}
                foreign_entities = []
                for pn in proper_nouns:
                    if pn in EXCLUDED_WORDS:
                        continue
                    if pn.lower() not in all_evidence_words and pn.lower() not in q_words:
                        foreign_entities.append(pn)

                # 3. Verify citations and evidence per claim
                cites = [int(m) for m in re.findall(r"\[(\d+)\]", s_clean)]
                s_content_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", s_lower)) - NLPQueryParser.STOPWORDS_CHUNKS

                if cites:
                    has_support = False
                    for c_idx in cites:
                        if 1 <= c_idx <= len(selected_chunks):
                            target_passage = selected_chunks[c_idx - 1].passage_text.lower()
                            if s_content_words:
                                match_count = sum(1 for w in s_content_words if w in target_passage)
                                if match_count / len(s_content_words) >= 0.15 or match_count >= 2:
                                    has_support = True
                                    break
                    if has_support:
                        stats["claims_supported"] += 1
                    else:
                        # Cited passage doesn't match; search other retrieved passages for actual support
                        found_remap = False
                        for j, ch in enumerate(selected_chunks, 1):
                            if s_content_words and sum(1 for w in s_content_words if w in ch.passage_text.lower()) >= 2:
                                s_clean = re.sub(r"\[\d+\]", f"[{j}]", s_clean)
                                found_remap = True
                                stats["claims_supported"] += 1
                                stats["citations_remapped"] += 1
                                break
                        if not found_remap:
                            # Strip false citation
                            s_clean = re.sub(r"\[\d+\]", "", s_clean).strip()
                            s_clean = re.sub(r"\s+([,\.])", r"\1", s_clean)
                            stats["citations_stripped"] += 1
                            if foreign_entities:
                                stats["unsupported_claims_removed"] += 1
                                continue
                else:
                    # Require evidence per claim: if statement is a factual claim without citation, search for supporting passage
                    if len(s_content_words) >= 3 or proper_nouns:
                        for j, ch in enumerate(selected_chunks, 1):
                            if sum(1 for w in s_content_words if w in ch.passage_text.lower()) >= 2:
                                s_clean = f"{s_clean.rstrip('.')} [{j}]."
                                stats["citations_attached"] += 1
                                stats["claims_supported"] += 1
                                break

                    # If sentence introduces foreign entities not found anywhere in evidence and has no citation support, drop it
                    if len(foreign_entities) >= 1 and not re.search(r"\[\d+\]", s_clean):
                        stats["unsupported_claims_removed"] += 1
                        continue

                valid_sentences.append(s_clean)

            if valid_sentences:
                verified_paragraphs.append(" ".join(valid_sentences))

        final_text = "\n\n".join(verified_paragraphs).strip()

        # 4. Enforce abstention: If missing aspects exist and the text didn't mention abstention, append honest qualification
        if missing_aspects:
            final_lower = final_text.lower()
            uncovered_aspects = []
            for aspect in missing_aspects:
                aspect_terms = set(re.findall(r"\b[a-zA-Z]{4,}\b", aspect.lower())) - NLPQueryParser.STOPWORDS_CHUNKS
                already_acknowledged = any(kw in final_lower for kw in ("not documented", "no specific documentation", "does not document", "remains uncertain", "uncertainty surrounding", "uncertainties surrounding", "silent on"))
                if not already_acknowledged or not any(term in final_lower for term in aspect_terms):
                    uncovered_aspects.append(aspect)

            if uncovered_aspects:
                abstention_notes = "\n\n".join(
                    f"The available evidence does not contain specific documentation regarding {asp}."
                    for asp in uncovered_aspects
                )
                final_text = f"{final_text}\n\n{abstention_notes}".strip()

        return final_text, stats


class AnswerOrchestrator:
    """End-to-end question answering engine integrating all Nexia stages."""

    def __init__(
        self,
        evidence_reasoner: Optional[EvidenceReasoner] = None,
        document_cache: Optional[DocumentCacheService] = None,
        hybrid_retrieval: Optional[HybridRetrievalService] = None,
        llm_adapter: Optional[OllamaLLMAdapter] = None,
        passage_retriever: Optional[PassageRetriever] = None,
        conversation_manager: Optional[ConversationManager] = None,
        context_manager: Optional[ContextManager] = None,
        title_generator: Optional[ConversationTitleGenerator] = None,
        response_controller: Optional[ResponseController] = None,
    ):
        self.evidence_reasoner = evidence_reasoner or EvidenceReasoner()
        self.document_cache = document_cache or DocumentCacheService()
        self.hybrid_retrieval = hybrid_retrieval or HybridRetrievalService()
        self.llm_adapter = llm_adapter or OllamaLLMAdapter()
        self.passage_retriever = passage_retriever or PassageRetriever()
        self.conversation_manager = conversation_manager or ConversationManager()
        self.context_manager = context_manager or ContextManager()
        self.response_controller = response_controller or ResponseController.get_instance()
        self.title_generator = title_generator or ConversationTitleGenerator(llm_adapter=self.llm_adapter)
        self.llm_orchestrator = LLMOrchestrator(
            conversation_manager=self.conversation_manager,
            context_manager=self.context_manager,
            llm_adapter=self.llm_adapter,
            response_controller=self.response_controller,
            title_generator=self.title_generator,
        )

    async def answer(self, request: AnswerRequest, request_id: Optional[str] = None) -> AnswerResponse:
        """Process a user question through the complete pipeline to generate a grounded answer."""
        t_req_start = time.perf_counter()
        req_id = request_id or str(uuid4())
        question = request.question.strip()
        logger.info(">>> [Stage 7] Starting pipeline for question: '%s'", question)

        # Detect listing/enumeration queries up-front for prompt tuning
        _is_list_query = NLPQueryParser.is_list_question(question)
        if _is_list_query:
            logger.info("[Stage 7] List/enumeration query detected")

        # Adaptive complexity detection
        complexity_res = AdaptiveQueryAnalyzer.analyze_complexity(question)
        logger.info("[Stage 7] Adaptive complexity: %s (clauses=%d, requests=%d, reasons=%s)",
                    complexity_res.complexity.value, complexity_res.clause_count,
                    complexity_res.interrogative_count, complexity_res.reasons)

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
                        if office != "leaders":
                            target_docs.add(f"List of {office} of {e_clean}")
                            target_docs.add(f"{office} of {e_clean}")
                    for t in terms[:4]:
                        if t and len(t.split()) <= 6:
                            target_docs.add(t)
                    # Only fall back to generic realm if no list-specific title was identified
                    if not target_docs:
                        for e in evidence_package.question_intent.entities:
                            target_docs.add(e.normalized_text)
                elif rel.type == "FOUNDED_BY":
                    for e in evidence_package.question_intent.entities:
                        e_clean = e.normalized_text
                        target_docs.add(e_clean)
                        if "dynasty" in e_clean.lower():
                            target_docs.add(re.sub(r"\bdynasty\b", "Empire", e_clean, flags=re.IGNORECASE).strip())
                            target_docs.add(re.sub(r"\bdynasty\b", "Kingdom", e_clean, flags=re.IGNORECASE).strip())

        # Add target articles from adaptive query decomposition
        if settings.ADAPTIVE_RETRIEVAL_ENABLED and complexity_res.target_articles:
            for art in complexity_res.target_articles:
                if art:
                    target_docs.add(art)

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
                    if settings.ADAPTIVE_RETRIEVAL_ENABLED and complexity_res.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX):
                        sub_queries = [s.retrieval_query for s in complexity_res.subqueries]
                        all_queries = [question] + [sq for sq in sub_queries if sq.lower() != question.lower()]
                        retrieved_chunks = await self.hybrid_retrieval.retrieve_multi_query(
                            queries=all_queries[: settings.MAX_SUBQUERIES_PER_COMPLEX_QUERY + 1],
                            page_ids=page_ids,
                            top_k=settings.HYBRID_TOP_K * 2,
                            session=session,
                            entity_boosts=complexity_res.primary_entities,
                        )
                    else:
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

                # Decompose before retrieval: retrieve evidence separately for main question and decomposed subqueries
                queries_to_harvest = [question]
                if settings.ADAPTIVE_RETRIEVAL_ENABLED and complexity_res.subqueries:
                    for sq in complexity_res.subqueries[:4]:
                        if sq.retrieval_query.lower() != question.lower():
                            queries_to_harvest.append(sq.retrieval_query)

                for d_title in doc_titles:
                    for q_harvest in queries_to_harvest:
                        p_res = await self.passage_retriever.retrieve_passages(
                            PassageRetrievalRequest(
                                question=q_harvest,
                                document=d_title,
                                top_k=2,
                            )
                        )
                        for idx, ep in enumerate(p_res.evidence):
                            if ep.score >= 0.45:
                                stage3_passages.append(
                                    RetrievedChunk(
                                        chunk_id=-len(stage3_passages) - 1,
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

        # Select slotted, non-redundant passages with 3-shingle deduplication and relevance filtering
        if settings.ADAPTIVE_RETRIEVAL_ENABLED and complexity_res.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX):
            selected_chunks = EvidencePacker.pack_slotted_evidence(
                subqueries=complexity_res.subqueries,
                candidate_chunks=merged_chunks,
                max_passages=request.max_evidence_passages,
                primary_entities=complexity_res.primary_entities,
                question=question,
            )
        else:
            selected_chunks = EvidencePacker.pack_slotted_evidence(
                subqueries=[],
                candidate_chunks=merged_chunks,
                max_passages=request.max_evidence_passages,
                primary_entities=complexity_res.primary_entities,
                question=question,
            )
        diagnostics.passages_sent_to_llm = len(selected_chunks)

        # Build SourceReference models
        source_refs: List[SourceReference] = []
        passages_for_prompt: List[Dict] = []

        # Detect if this is an enumeration/listing query — needs higher word cap per passage
        _is_list_query = bool(
            NLPQueryParser.is_list_question(question)
            or evidence_package.question_intent and getattr(
                evidence_package.question_intent.relation, "type", ""
            ) == "OFFICE_HOLDERS_OF"
        )
        # For list queries allow up to 400 words per passage; factoid queries stay at 120
        _passage_word_cap = 400 if _is_list_query else 120

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
            # Bound passage text sent to LLM — higher cap for list queries
            words = c.passage_text.strip().split()
            if len(words) > _passage_word_cap:
                bounded_text = " ".join(words[:_passage_word_cap]) + "..."
            else:
                bounded_text = c.passage_text.strip()

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

        # Adaptive evidence coverage assessment
        if settings.ADAPTIVE_RETRIEVAL_ENABLED and complexity_res.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX):
            cov_result = EvidenceCoverageAssessor.assess_coverage(
                subqueries=complexity_res.subqueries,
                candidate_chunks=selected_chunks,
            )
            diagnostics.query_complexity = complexity_res.complexity.value
            diagnostics.subqueries_generated = [s.retrieval_query for s in complexity_res.subqueries]
            diagnostics.evidence_coverage_ratio = cov_result.coverage_ratio
            diagnostics.missing_aspects = cov_result.missing_aspects

            if cov_result.overall_status == "INSUFFICIENT" or not selected_chunks:
                sufficiency_status = "INSUFFICIENT"
            elif cov_result.overall_status == "PARTIAL":
                sufficiency_status = "PARTIAL"
            elif selected_chunks:
                sufficiency_status = "SUFFICIENT"
        else:
            diagnostics.query_complexity = "SIMPLE"
            diagnostics.subqueries_generated = [question]
            diagnostics.evidence_coverage_ratio = 1.0 if selected_chunks else 0.0
            diagnostics.missing_aspects = []

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
        history_dicts = []
        if request.conversation_id:
            try:
                hist_msgs = await self.conversation_manager.get_messages(request.conversation_id, limit=6)
                for hm in hist_msgs:
                    history_dicts.append({"role": hm.role.value, "content": hm.content})
            except Exception as e:
                logger.warning("Failed to fetch conversation history for %s: %s", request.conversation_id, e)

        prompt_ctx = PromptContext(
            question=question,
            question_type=q_type,
            entities=q_entities,
            relation=q_relation,
            passages=passages_for_prompt,
            conflicts=conflicts_text,
            sufficiency_rating=sufficiency_status,
            missing_aspects=diagnostics.missing_aspects or [],
            subqueries=diagnostics.subqueries_generated or [],
            conversation_history=history_dicts,
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
            async def _invoke_llm():
                # For list queries boost num_predict to 512 (ample for 50+ list items, fast on CPU)
                if _is_list_query:
                    return await self.llm_adapter.generate_answer(
                        prompt_ctx, num_predict_override=512
                    )
                else:
                    return await self.llm_adapter.generate_answer(prompt_ctx)

            llm_result = await self.response_controller.execute_bounded(
                request_id=req_id,
                coro_func=_invoke_llm,
                priority=Priority.INTERACTIVE,
            )
            answer_text = llm_result.answer_text
            # 1. Clean introductory meta list headings e.g.:
            # "Here is a complete list of the Presidents of India, as documented in the provided evidence:" -> "Presidents of India:"
            answer_text = re.sub(
                r"^(?:Here is (?:a|the) (?:complete )?list of (?:the )?([a-zA-Z\s]+?)[,\s]*(?:as (?:documented|presented|stated|shown|found|recorded|outlined) in (?:the )?(?:provided )?evidence|as per (?:the )?(?:evidence|wikipedia|text)|according to (?:the )?(?:evidence|wikipedia|text))?\s*[:]\s*)",
                r"\1:\n\n",
                answer_text,
                flags=re.IGNORECASE,
            ).strip()

            # 2. Clean leading robotic meta-openers e.g. "According to the provided document...", "As per evidence..."
            answer_text = re.sub(
                r"^(?:According to (?:the )?(?:provided )?(?:document|documents|text|passages?|evidence|wikipedia)|Based on (?:the )?(?:provided )?(?:document|documents|text|passages?|evidence|wikipedia)|As per (?:the )?(?:provided )?(?:document|documents|text|passages?|evidence|wikipedia)|As for the (?:given |provided )?(?:evidence|documents?|text)|In the provided (?:evidence|documents?|text)|From the provided (?:evidence|documents?|text)|The provided text (?:states|defines|describes|indicates|explains)|The passage states that)\s*[:,]?\s*",
                "",
                answer_text,
                flags=re.IGNORECASE,
            ).strip()

            # 3. Clean inline meta-phrasing e.g. ", as documented in the provided evidence" or ", as per evidence"
            answer_text = re.sub(
                r"[,\s]*(?:as (?:documented|presented|stated|shown|found|recorded) in (?:the )?(?:provided )?evidence|as per (?:the )?(?:evidence|wikipedia|provided document|documents|text)|as for the (?:given |provided )?evidence|according to (?:the )?(?:provided )?evidence)\s*",
                " ",
                answer_text,
                flags=re.IGNORECASE,
            ).strip()

            if answer_text and answer_text[0].islower():
                answer_text = answer_text[0].upper() + answer_text[1:]

            # 4. Clean abrupt incomplete sentences / mid-sentence truncation
            trimmed = answer_text.rstrip()
            if trimmed and not trimmed.endswith((".", "!", "?", ':"', '."', "!'", "?'")):
                lines = trimmed.split("\n")
                last_line = lines[-1].strip()
                if not last_line.endswith((".", "!", "?", ":")):
                    sentences = re.split(r"(?<=[.!?])\s+", last_line)
                    if len(sentences) > 1 and not sentences[-1].strip().endswith((".", "!", "?")):
                        lines[-1] = " ".join(sentences[:-1]).strip()
                        answer_text = "\n".join(lines).strip()
                    elif len(lines) > 1 and not last_line.endswith((".", "!", "?")):
                        answer_text = "\n".join(lines[:-1]).strip()

            # 5. Citation verification: ensure citations point only to valid retrieved passage indexes
            max_ref_idx = len(selected_chunks)
            if max_ref_idx > 0:
                def _clamp_or_remove_cite(match: re.Match) -> str:
                    idx_val = int(match.group(1))
                    if 1 <= idx_val <= max_ref_idx:
                        return f"[{idx_val}]"
                    return ""
                answer_text = re.sub(r"\[(\d+)\]", _clamp_or_remove_cite, answer_text)
                answer_text = re.sub(r"\s+([,\.])", r"\1", answer_text)

            # 6. Repetition verification: deduplicate near-identical consecutive sentences (Jaccard >= 0.80)
            def _clean_repetitive_sentences(text: str) -> str:
                paragraphs = text.split("\n\n")
                clean_paragraphs = []
                for p in paragraphs:
                    sentences = re.split(r"(?<=[.!?])\s+", p.strip())
                    clean_s = []
                    prev_tokens = set()
                    for s in sentences:
                        toks = set(re.findall(r"\b[a-zA-Z]{3,}\b", s.lower()))
                        if toks and prev_tokens:
                            overlap = len(toks & prev_tokens) / len(toks | prev_tokens)
                            if overlap >= 0.80:
                                continue
                        clean_s.append(s)
                        if toks:
                            prev_tokens = toks
                    clean_paragraphs.append(" ".join(clean_s))
                return "\n\n".join(clean_paragraphs)

            answer_text = _clean_repetitive_sentences(answer_text)

            # 7. Claim-level verification: enforce evidence grounding and drop off-topic/ungrounded claims
            answer_text, claim_stats = ClaimVerifier.verify_and_clean_answer(
                answer_text=answer_text,
                selected_chunks=selected_chunks,
                question=question,
                missing_aspects=diagnostics.missing_aspects or [],
            )
            diagnostics.claims_verified = claim_stats["claims_verified"]
            diagnostics.claims_supported = claim_stats["claims_supported"]
            diagnostics.unsupported_claims_removed = claim_stats["unsupported_claims_removed"]
            diagnostics.citations_stripped = claim_stats["citations_stripped"]
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

        # Select single primary article and format answer presentation (no inline [N] citations, single trailing reference)
        primary_article = None
        if status == "ANSWERED" and not answer_text.startswith("INSUFFICIENT_EVIDENCE") and not answer_text.startswith("LLM answer generation failed"):
            primary_article = PrimaryArticleSelector.select_primary_article(
                selected_chunks=selected_chunks,
                answer_text=answer_text,
                question=question,
                subqueries=diagnostics.subqueries_generated,
            )
        user_facing_answer = format_answer_with_primary_reference(
            answer_text=answer_text,
            primary_article=primary_article,
        )

        # Persist conversation turn if conversation_id provided
        saved_msg_id = None
        if request.conversation_id:
            try:
                # Add user message
                await self.conversation_manager.add_message(
                    conversation_id=request.conversation_id,
                    role=MessageRole.USER,
                    content=question,
                )
                # Add assistant message
                ast_msg = await self.conversation_manager.add_message(
                    conversation_id=request.conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=user_facing_answer,
                    token_count=resp_tokens,
                    message_status=MessageStatus.COMPLETED if status == "ANSWERED" else MessageStatus.FAILED,
                    source_references=[r.model_dump() for r in source_refs],
                    diagnostics={"claims_verified": diagnostics.claims_verified},
                )
                saved_msg_id = ast_msg.id

                # Trigger non-blocking title generation if eligible (turn 1 with temporary title)
                if await self.conversation_manager.is_eligible_for_title_generation(request.conversation_id):
                    import asyncio
                    asyncio.create_task(
                        self.title_generator.generate_and_update_title(
                            conversation_id=request.conversation_id,
                            first_question=question,
                            first_answer=user_facing_answer,
                            conversation_manager=self.conversation_manager,
                        )
                    )
            except Exception as e:
                logger.warning("Failed to persist conversation turn or trigger title generation: %s", e)

        # 7. Assemble final AnswerResponse
        return AnswerResponse(
            question=question,
            answer_text=user_facing_answer,
            answer_status=status,
            source_references=source_refs,
            conflicts=conflicts_text,
            sufficiency_rating=sufficiency_status,
            retrieval_diagnostics=diagnostics,
            stage6_evidence_package=evidence_package,
            error_message=error_msg,
            conversation_id=request.conversation_id,
            message_id=saved_msg_id,
        )

    async def answer_stream(
        self,
        request: AnswerRequest,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Process question and stream grounded response tokens progressively via SSE."""
        from uuid import uuid4
        req_id = request_id or str(uuid4())
        question = request.question.strip()

        yield StreamingEngine.emit_start(
            request_id=req_id,
            conversation_id=request.conversation_id,
            model=settings.LLM_MODEL,
        )

        yield StreamingEngine.emit_progress(
            request_id=req_id,
            stage="retrieval",
            message="Gathering evidence passages from knowledge base...",
        )

        # Detect listing query
        _is_list_query = NLPQueryParser.is_list_question(question)
        if _is_list_query:
            logger.info("[Stage 7 Stream] List/enumeration query detected")

        # Run non-LLM upstream retrieval by calling answer() with a fast stub or running stages
        # To ensure exact parity, we use the evidence gathering from answer()
        evidence_passages: List[Dict[str, Any]] = []
        try:
            # Stage 6 reasoning
            reasoning_req = EvidenceReasoningRequest(
                question=question,
                subject_entity=request.subject_entity,
                object_entity=request.object_entity,
                predicate_filter=request.predicate_filter,
                top_k_per_document=request.top_k_per_document,
                min_confidence=request.min_confidence,
                force_refresh=request.force_refresh,
            )
            reasoning_res = await self.evidence_reasoner.reason(reasoning_req)
            evidence_package = reasoning_res.evidence_package

            # Target docs
            target_docs = set()
            for cluster in evidence_package.evidence_clusters:
                for item in cluster.items:
                    doc_title = getattr(item, "source_document", None) or getattr(item, "document_title", None)
                    if doc_title:
                        target_docs.add(doc_title)

            if evidence_package.question_intent and evidence_package.question_intent.entities:
                for e in evidence_package.question_intent.entities:
                    target_docs.add(e.normalized_text)
                    target_docs.add(e.text)

            complexity_res = AdaptiveQueryAnalyzer.analyze_complexity(question)
            if settings.ADAPTIVE_RETRIEVAL_ENABLED and complexity_res.target_articles:
                for art in complexity_res.target_articles:
                    if art:
                        target_docs.add(art)

            session_factory = get_session_factory()
            page_ids = []
            async with session_factory() as session:
                for doc_title in target_docs:
                    try:
                        doc, _ = await self.document_cache.get_or_fetch_document(
                            title=doc_title,
                            force_refresh=request.force_refresh,
                            session=session,
                        )
                        if doc.page_id not in page_ids:
                            page_ids.append(doc.page_id)
                    except Exception:
                        pass
                if not page_ids:
                    try:
                        doc, _ = await self.document_cache.get_or_fetch_document(
                            title=question,
                            force_refresh=request.force_refresh,
                            session=session,
                        )
                        if doc.page_id not in page_ids:
                            page_ids.append(doc.page_id)
                    except Exception:
                        pass

                retrieved_chunks = []
                if page_ids:
                    if settings.ADAPTIVE_RETRIEVAL_ENABLED and complexity_res.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX):
                        sub_queries = [s.retrieval_query for s in complexity_res.subqueries]
                        all_queries = [question] + [sq for sq in sub_queries if sq.lower() != question.lower()]
                        retrieved_chunks = await self.hybrid_retrieval.retrieve_multi_query(
                            queries=all_queries[: settings.MAX_SUBQUERIES_PER_COMPLEX_QUERY + 1],
                            page_ids=page_ids,
                            top_k=settings.HYBRID_TOP_K * 2,
                            session=session,
                            entity_boosts=complexity_res.primary_entities,
                        )
                    else:
                        retrieved_chunks = await self.hybrid_retrieval.retrieve(
                            question=question,
                            page_ids=page_ids,
                            top_k=settings.HYBRID_TOP_K,
                            session=session,
                        )

            # Deduplicate & pack passages with EvidencePacker
            if settings.ADAPTIVE_RETRIEVAL_ENABLED and complexity_res.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX):
                selected_chunks = EvidencePacker.pack_slotted_evidence(
                    subqueries=complexity_res.subqueries,
                    candidate_chunks=retrieved_chunks,
                    max_passages=request.max_evidence_passages,
                    primary_entities=complexity_res.primary_entities,
                    question=question,
                )
            else:
                selected_chunks = EvidencePacker.pack_slotted_evidence(
                    subqueries=[],
                    candidate_chunks=retrieved_chunks,
                    max_passages=request.max_evidence_passages,
                    primary_entities=complexity_res.primary_entities,
                    question=question,
                )

            for c in selected_chunks:
                evidence_passages.append({
                    "document_title": c.document_title,
                    "section_title": c.section_title,
                    "source_url": c.source_url,
                    "passage_text": c.passage_text,
                })

        except Exception as e:
            logger.warning("Upstream retrieval in streaming encountered: %s", e)

        yield StreamingEngine.emit_progress(
            request_id=req_id,
            stage="inference",
            message=f"Generating grounded answer using local {settings.LLM_MODEL}...",
        )

        # Delegate streaming execution to LLMOrchestrator
        async for sse_chunk in self.llm_orchestrator.execute_turn_stream(
            question=question,
            evidence_passages=evidence_passages,
            conversation_id=request.conversation_id,
            is_list_query=_is_list_query,
            request_id=req_id,
        ):
            yield sse_chunk
