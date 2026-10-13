"""Stage 6 Core Engine: Advanced Evidence Reasoning, Clustering, Conflict Detection & LLM Evidence Package Assembly."""

import re
from typing import List, Optional, Dict, Set, Tuple
from app.schemas.retrieval import Stage4StructuredQuery, EntityItem
from app.schemas.relationships import (
    RelationshipDiscoveryRequest,
    RelationshipDiscoveryResponse,
    DiscoveredRelationshipEdge,
    EvidenceProvenanceItem,
)
from app.schemas.evidence_package import (
    EvidenceItem,
    EvidenceCluster,
    EvidenceConflict,
    SufficiencyAssessment,
    AnswerCandidate,
    EvidencePackage,
    EvidenceReasoningRequest,
    EvidenceReasoningResponse,
)
from app.retrieval.relationship_engine import RelationshipEngine


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "who", "whom", "whose", "where", "when", "how", "why", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "to", "at", "in", "for", "on", "by", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "in", "out", "of",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}


class EvidenceReasoner:
    """Stage 6 Engine: Deterministically dedupes, scores, clusters, and audits evidence."""

    def __init__(self, relationship_engine: Optional[RelationshipEngine] = None):
        self.relationship_engine = relationship_engine or RelationshipEngine()

    async def reason(self, request: EvidenceReasoningRequest) -> EvidenceReasoningResponse:
        """Execute the full Stage 6 pipeline: run Stage 5 then perform evidence reasoning."""
        question = request.question.strip()
        if not question:
            empty_package = EvidencePackage(
                question="",
                question_intent=None,
                evidence_clusters=[],
                conflicts=[],
                sufficiency_assessment=SufficiencyAssessment(
                    status="INSUFFICIENT",
                    confidence=0.0,
                    missing_aspects=["Question text is empty"],
                    reasoning="No question provided."
                ),
                answer_candidates=[],
                total_evidence_items=0,
                total_clusters=0,
                total_conflicts=0,
                deduplication_summary="No items to evaluate.",
                stage6_status="ERROR",
            )
            return EvidenceReasoningResponse(
                evidence_package=empty_package,
                status="ERROR",
                validation_summary="Question cannot be empty.",
                error_message="Question cannot be empty.",
            )

        # 1. Invoke Stage 5 Relationship Discovery
        stage5_req = RelationshipDiscoveryRequest(
            question=question,
            subject_entity=request.subject_entity,
            object_entity=request.object_entity,
            predicate_filter=request.predicate_filter,
            top_k_per_document=request.top_k_per_document,
            min_confidence=request.min_confidence,
        )
        stage5_resp = await self.relationship_engine.discover_relationships(stage5_req)

        # 2. Perform Stage 6 reasoning on the Stage 5 output
        package = self.reason_from_stage5(
            stage5_response=stage5_resp,
            question=question,
            dedup_similarity_threshold=request.dedup_similarity_threshold,
        )

        overall_status = "COMPLETED"
        if package.sufficiency_assessment.status == "INSUFFICIENT":
            overall_status = "NO_EVIDENCE"
        elif package.sufficiency_assessment.status == "PARTIAL":
            overall_status = "PARTIALLY_SUPPORTED"

        return EvidenceReasoningResponse(
            evidence_package=package,
            status=overall_status,
            validation_summary=package.sufficiency_assessment.reasoning,
        )

    def reason_from_stage5(
        self,
        stage5_response: RelationshipDiscoveryResponse,
        question: str,
        dedup_similarity_threshold: float = 0.75,
    ) -> EvidencePackage:
        """Perform deterministic reasoning steps over Stage 5 output."""
        structured_query = stage5_response.structured_query

        # Step 1: Extract raw evidence items
        raw_items = self._extract_raw_items(stage5_response)

        # Step 2: Deduplication using Jaccard token overlap
        deduped_items, dedup_summary = self._deduplicate(raw_items, threshold=dedup_similarity_threshold)

        # Step 3: Re-score relevance against question intent
        rescored_items = self._rescore_relevance(deduped_items, structured_query, question)

        # Step 4: Classify answer-support role
        classified_items = self._classify_support(rescored_items, structured_query, question, stage5_response)

        # Step 5: Cluster evidence by document & section
        clusters = self._cluster_evidence(classified_items)

        # Step 6: Conflict detection (temporal/factual discrepancies)
        conflicts = self._detect_conflicts(classified_items)

        # Step 7: Assess sufficiency
        sufficiency = self._assess_sufficiency(classified_items, conflicts, structured_query, stage5_response.status)

        # Step 8: Surface grounded answer candidates
        candidates = self._surface_answer_candidates(classified_items, structured_query, stage5_response)

        stage6_status = "COMPLETED" if sufficiency.status in ("SUFFICIENT", "PARTIAL") else "NO_EVIDENCE"

        return EvidencePackage(
            question=question,
            question_intent=structured_query,
            evidence_clusters=clusters,
            conflicts=conflicts,
            sufficiency_assessment=sufficiency,
            answer_candidates=candidates,
            total_evidence_items=len(classified_items),
            total_clusters=len(clusters),
            total_conflicts=len(conflicts),
            deduplication_summary=dedup_summary,
            stage6_status=stage6_status,
        )

    def _extract_raw_items(self, stage5_response: RelationshipDiscoveryResponse) -> List[EvidenceItem]:
        """Extract individual evidence items from Stage 5 provenance chains."""
        items: List[EvidenceItem] = []
        seen_keys: Set[str] = set()
        item_counter = 1

        all_edges = list(stage5_response.all_relationships)
        if stage5_response.primary_relationship and stage5_response.primary_relationship not in all_edges:
            all_edges.insert(0, stage5_response.primary_relationship)

        # Check evidence graph as well
        if stage5_response.evidence_graph and stage5_response.evidence_graph.relationships:
            for g_edge in stage5_response.evidence_graph.relationships:
                if g_edge not in all_edges:
                    all_edges.append(g_edge)

        for edge in all_edges:
            for prov in edge.evidence_chain:
                # Key by document + exact sentence to prevent identical duplicate entries from ingestion
                k = f"{prov.document_title}::{prov.exact_sentence.strip()}"
                if k in seen_keys:
                    continue
                seen_keys.add(k)

                text = prov.exact_sentence.strip() if prov.exact_sentence else prov.passage_text.strip()
                item = EvidenceItem(
                    item_id=f"item_{item_counter}",
                    text=text,
                    source_document=prov.document_title,
                    source_section=prov.section_title or "General",
                    relevance_score=prov.support_score,
                    support_class="SUPPORTING",
                    is_duplicate=False,
                    provenance=prov,
                )
        # If no relationship edges were formed (single-entity, biographical, or factoid queries),
        # extract directly from candidate passages so Stage 6 clusters and evaluates them.
        if not items and stage5_response.evidence_graph and getattr(stage5_response.evidence_graph, "candidate_passages", None):
            for prov in stage5_response.evidence_graph.candidate_passages:
                k = f"{prov.document_title}::{prov.exact_sentence.strip()}"
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                text = prov.exact_sentence.strip() if prov.exact_sentence else prov.passage_text.strip()
                item = EvidenceItem(
                    item_id=f"item_{item_counter}",
                    text=text,
                    source_document=prov.document_title,
                    source_section=prov.section_title or "General",
                    relevance_score=prov.support_score,
                    support_class="SUPPORTING",
                    is_duplicate=False,
                    provenance=prov,
                )
                items.append(item)
                item_counter += 1

        return items

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Extract set of lowercase alphanumeric tokens excluding single characters."""
        words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
        return set(words)

    def _deduplicate(self, items: List[EvidenceItem], threshold: float = 0.75) -> Tuple[List[EvidenceItem], str]:
        """Detect near-duplicate evidence items using Jaccard word-set similarity."""
        if not items:
            return [], "No evidence items to evaluate."

        token_sets = [self._tokenize(it.text) for it in items]
        duplicate_count = 0

        for i in range(len(items)):
            if items[i].is_duplicate:
                continue
            set_i = token_sets[i]
            if not set_i:
                continue

            for j in range(i + 1, len(items)):
                if items[j].is_duplicate:
                    continue
                set_j = token_sets[j]
                if not set_j:
                    continue

                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard >= threshold:
                    items[j].is_duplicate = True
                    items[j].support_class = "IRRELEVANT"
                    duplicate_count += 1

        total = len(items)
        canonical = total - duplicate_count
        summary = (
            f"Evaluated {total} evidence items; detected and marked {duplicate_count} near-duplicates "
            f"(threshold >= {threshold:.2f}). {canonical} canonical items retained."
        )
        return items, summary

    def _rescore_relevance(
        self,
        items: List[EvidenceItem],
        structured_query: Optional[Stage4StructuredQuery],
        question: str,
    ) -> List[EvidenceItem]:
        """Re-score relevance using question tokens, entity matches, and relation cues."""
        q_tokens = self._tokenize(question) - STOP_WORDS

        entity_tokens: Set[str] = set()
        subject_tokens: Set[str] = set()
        object_tokens: Set[str] = set()
        relation_tokens: Set[str] = set()

        if structured_query:
            if structured_query.entities:
                for ent in structured_query.entities:
                    tokens = (self._tokenize(ent.text) | self._tokenize(ent.normalized_text)) - STOP_WORDS
                    entity_tokens |= tokens
                    role_upper = getattr(ent, "role", "SUBJECT").upper()
                    if "SUBJECT" in role_upper:
                        subject_tokens |= tokens
                    else:
                        object_tokens |= tokens
            if structured_query.relation:
                relation_tokens |= self._tokenize(structured_query.relation.type) - STOP_WORDS
                for rt in getattr(structured_query.relation, "relation_terms", []):
                    relation_tokens |= self._tokenize(rt) - STOP_WORDS
            if structured_query.retrieval_terms:
                for rt in structured_query.retrieval_terms:
                    relation_tokens |= self._tokenize(rt) - STOP_WORDS

        for it in items:
            it_tokens = self._tokenize(it.text)
            if not it_tokens:
                it.relevance_score = 0.0
                continue

            # Question overlap (0.0 to 1.0)
            q_match = len(it_tokens & q_tokens) / max(len(q_tokens), 1)

            # Entity coverage
            subj_match = 1.0 if subject_tokens and (subject_tokens & it_tokens) else 0.0
            obj_match = 1.0 if object_tokens and (object_tokens & it_tokens) else 0.0
            any_ent_match = 1.0 if entity_tokens and (entity_tokens & it_tokens) else 0.0

            entity_score = 0.2
            if subj_match and obj_match:
                entity_score = 1.0
            elif subj_match or obj_match:
                entity_score = 0.70
            elif any_ent_match:
                entity_score = 0.60

            # Relation cue coverage
            rel_score = 1.0 if relation_tokens and (relation_tokens & it_tokens) else 0.3

            base_score = it.provenance.support_score if it.provenance else 0.5

            # Weighted combination: 35% base, 30% entity, 20% question, 15% relation
            rescored = (0.35 * base_score) + (0.30 * entity_score) + (0.20 * q_match) + (0.15 * rel_score)

            # Penalize duplicates
            if it.is_duplicate:
                rescored *= 0.3

            it.relevance_score = round(min(1.0, max(0.0, rescored)), 4)

        return items

    def _classify_support(
        self,
        items: List[EvidenceItem],
        structured_query: Optional[Stage4StructuredQuery],
        question: str,
        stage5_response: RelationshipDiscoveryResponse,
    ) -> List[EvidenceItem]:
        """Classify each evidence item as DIRECT_ANSWER, SUPPORTING, CONTEXTUAL, or IRRELEVANT."""
        direct_cues = {
            "successor", "succeeded", "succeeding", "predecessor", "founded", "founder",
            "co-founder", "established", "alumni", "doctoral", "doctorate", "phd", "dissertation",
            "professor", "appointed", "worked with", "associated with", "ruler", "king", "reign",
            "enrolled", "studied", "graduated", "faculty"
        }

        # Collect candidate names from discovered relationships
        discovered_entities: Set[str] = set()
        if stage5_response.primary_relationship:
            discovered_entities.add(stage5_response.primary_relationship.object.text.lower())
            if stage5_response.primary_relationship.object.normalized_text:
                discovered_entities.add(stage5_response.primary_relationship.object.normalized_text.lower())
        for r in stage5_response.all_relationships:
            discovered_entities.add(r.object.text.lower())
            if r.object.normalized_text:
                discovered_entities.add(r.object.normalized_text.lower())

        for it in items:
            if it.is_duplicate:
                it.support_class = "IRRELEVANT"
                continue

            text_lower = it.text.lower()
            it_tokens = self._tokenize(it.text)

            has_direct_cue = bool(it_tokens & direct_cues)
            has_candidate_entity = any(de and de in text_lower for de in discovered_entities)

            if it.relevance_score >= 0.55 and (has_direct_cue or has_candidate_entity):
                it.support_class = "DIRECT_ANSWER"
            elif it.relevance_score >= 0.40:
                it.support_class = "SUPPORTING"
            elif it.relevance_score >= 0.20:
                it.support_class = "CONTEXTUAL"
            else:
                it.support_class = "IRRELEVANT"

        return items

    def _cluster_evidence(self, items: List[EvidenceItem]) -> List[EvidenceCluster]:
        """Group evidence items into thematic clusters based on (source_document, source_section)."""
        grouped: Dict[Tuple[str, str], List[EvidenceItem]] = {}
        for it in items:
            key = (it.source_document, it.source_section)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(it)

        clusters: List[EvidenceCluster] = []
        cluster_idx = 1
        for (doc, sec), group_items in grouped.items():
            topic = f"{doc} - {sec}" if sec and sec != "General" else doc
            non_dupes = [it for it in group_items if not it.is_duplicate]
            avg_rel = sum(it.relevance_score for it in non_dupes) / max(len(non_dupes), 1) if non_dupes else 0.0

            clusters.append(
                EvidenceCluster(
                    cluster_id=f"cluster_{cluster_idx}",
                    cluster_topic=topic,
                    items=group_items,
                    cluster_relevance=round(avg_rel, 4),
                )
            )
            cluster_idx += 1

        clusters.sort(key=lambda c: c.cluster_relevance, reverse=True)
        return clusters

    def _detect_conflicts(self, items: List[EvidenceItem]) -> List[EvidenceConflict]:
        """Detect temporal or factual discrepancies across canonical items."""
        conflicts: List[EvidenceConflict] = []
        active_items = [it for it in items if not it.is_duplicate]

        # Extract event years associated with common verbs
        year_pattern = re.compile(r"\b(1\d{3}|20\d{2})\b")
        item_years: Dict[str, Set[str]] = {}
        for it in active_items:
            years = set(year_pattern.findall(it.text))
            if years:
                item_years[it.item_id] = years

        # Compare items that discuss succession or foundation dates with diverging years
        conflict_counter = 1
        seen_pairs: Set[Tuple[str, str]] = set()

        for i, it_a in enumerate(active_items):
            years_a = item_years.get(it_a.item_id, set())
            if not years_a:
                continue
            for j in range(i + 1, len(active_items)):
                it_b = active_items[j]
                years_b = item_years.get(it_b.item_id, set())
                if not years_b:
                    continue

                # Check if both items mention conflicting succession years for the same subject
                if (
                    it_a.source_document == it_b.source_document
                    and it_a.support_class == "DIRECT_ANSWER"
                    and it_b.support_class == "DIRECT_ANSWER"
                ):
                    diff_years = years_a.symmetric_difference(years_b)
                    if diff_years and not (years_a & years_b):
                        pair = (it_a.item_id, it_b.item_id)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            conflicts.append(
                                EvidenceConflict(
                                    conflict_id=f"conflict_{conflict_counter}",
                                    description=(
                                        f"Conflicting dates detected between item {it_a.item_id} (years: {sorted(years_a)}) "
                                        f"and item {it_b.item_id} (years: {sorted(years_b)}) in {it_a.source_document}."
                                    ),
                                    conflicting_items=[it_a.item_id, it_b.item_id],
                                    severity="MEDIUM",
                                )
                            )
                            conflict_counter += 1

        return conflicts

    def _assess_sufficiency(
        self,
        items: List[EvidenceItem],
        conflicts: List[EvidenceConflict],
        structured_query: Optional[Stage4StructuredQuery],
        stage5_status: str,
    ) -> SufficiencyAssessment:
        """Deterministically assess whether evidence is SUFFICIENT, PARTIAL, or INSUFFICIENT."""
        active_items = [it for it in items if not it.is_duplicate]
        direct_count = sum(1 for it in active_items if it.support_class == "DIRECT_ANSWER")
        supporting_count = sum(1 for it in active_items if it.support_class == "SUPPORTING")

        has_high_conflict = any(c.severity == "HIGH" for c in conflicts)

        if not active_items or stage5_status in ("DOCUMENT_NOT_FOUND", "ERROR"):
            return SufficiencyAssessment(
                status="INSUFFICIENT",
                confidence=0.10,
                missing_aspects=["No relevant relationship or document evidence was retrieved."],
                reasoning="Retrieved zero relevant evidence items supporting the question intent.",
            )

        if direct_count >= 1 and not has_high_conflict:
            conf = min(0.95, 0.75 + (0.05 * min(direct_count, 4)))
            return SufficiencyAssessment(
                status="SUFFICIENT",
                confidence=round(conf, 2),
                missing_aspects=[],
                reasoning=(
                    f"Direct answer evidence verified across {direct_count} passage(s) "
                    f"with {supporting_count} supporting item(s) and no unresolved high conflicts."
                ),
            )

        if supporting_count >= 1 or direct_count >= 1:
            return SufficiencyAssessment(
                status="PARTIAL",
                confidence=0.60,
                missing_aspects=["Definitive direct answer statement or cross-source confirmation."],
                reasoning=(
                    f"Found {supporting_count} supporting item(s), but direct answer evidence remains "
                    "inconclusive or only partially corroborates the question."
                ),
            )

        return SufficiencyAssessment(
            status="INSUFFICIENT",
            confidence=0.25,
            missing_aspects=["Target entity relationships not grounded in evidence."],
            reasoning="Evidence collected contains only contextual or irrelevant background statements.",
        )

    def _surface_answer_candidates(
        self,
        items: List[EvidenceItem],
        structured_query: Optional[Stage4StructuredQuery],
        stage5_response: RelationshipDiscoveryResponse,
    ) -> List[AnswerCandidate]:
        """Extract grounded candidate answers from Stage 5 edges and Stage 6 direct items."""
        candidates: List[AnswerCandidate] = []
        seen_texts: Set[str] = set()

        # Primary relationship candidate
        if stage5_response.primary_relationship:
            edge = stage5_response.primary_relationship
            target_entity = edge.object.text.strip()
            if target_entity and target_entity.lower() not in seen_texts:
                seen_texts.add(target_entity.lower())
                supp_ids = [it.item_id for it in items if not it.is_duplicate and it.support_class in ("DIRECT_ANSWER", "SUPPORTING")]
                candidates.append(
                    AnswerCandidate(
                        candidate_text=target_entity,
                        support_count=max(1, len(supp_ids)),
                        supporting_items=supp_ids,
                        confidence=round(edge.confidence_score, 2),
                    )
                )

        # Secondary relationships candidates
        for edge in stage5_response.all_relationships:
            target_entity = edge.object.text.strip()
            if target_entity and target_entity.lower() not in seen_texts:
                seen_texts.add(target_entity.lower())
                supp_ids = [
                    it.item_id for it in items
                    if not it.is_duplicate and (target_entity.lower() in it.text.lower() or it.support_class in ("DIRECT_ANSWER", "SUPPORTING"))
                ]
                candidates.append(
                    AnswerCandidate(
                        candidate_text=target_entity,
                        support_count=max(1, len(supp_ids)),
                        supporting_items=supp_ids,
                        confidence=round(edge.confidence_score * 0.9, 2),
                    )
                )

        # Direct answer items fallback candidate if no edges surfaced
        if not candidates:
            for it in items:
                if not it.is_duplicate and it.support_class == "DIRECT_ANSWER":
                    text_snip = it.text[:120].strip()
                    if text_snip.lower() not in seen_texts:
                        seen_texts.add(text_snip.lower())
                        candidates.append(
                            AnswerCandidate(
                                candidate_text=text_snip,
                                support_count=1,
                                supporting_items=[it.item_id],
                                confidence=round(it.relevance_score, 2),
                            )
                        )
                        break

        return candidates
