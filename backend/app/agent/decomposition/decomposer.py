"""
QueryDecomposer — Assesses complexity, decomposes queries into InformationNeeds,
and performs adaptive goal-driven need management.
"""
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from app.agent.decomposition.models import (
    InformationNeed, NeedPriority, NeedStatus, QueryComplexity,
)
from app.agent.research.context import ResearchContext
from app.core.query_understanding import IntentCategory, QueryUnderstanding
from app.knowledge.workspace import KnowledgeWorkspace

logger = logging.getLogger("nexia.agent.decomposition")


def _norm(text: str) -> str:
    """Normalize text for matching."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


class QueryDecomposer:
    """
    Component responsible for query complexity analysis, initial information need
    decomposition, dependency tracking, evidence reuse, and adaptive need creation.
    """

    def determine_complexity(
        self,
        qu: QueryUnderstanding,
        resolved_query: str,
    ) -> QueryComplexity:
        """
        Assess whether a query is SIMPLE (single fact/definition) or COMPLEX
        (multi-entity, multi-hop, causal, or comparative).
        """
        lower = resolved_query.lower()

        # Comparisons are always complex
        if qu.intent == IntentCategory.COMPARISON:
            return QueryComplexity.COMPLEX
        if any(w in lower for w in ["compare", "difference between", "similarities between", " vs ", " versus "]):
            return QueryComplexity.COMPLEX

        # Causal and multi-part explanatory questions
        if any(w in lower for w in [
            "why was", "why did", "why is", "how did", "how does", "influence on",
            "relationship between", "lead to", "impact on", "connected to"
        ]):
            return QueryComplexity.COMPLEX

        # Multi-entity conjunctions ("Einstein and Newton", "photosynthesis and cellular respiration")
        if " and " in lower and len(resolved_query.split()) > 4:
            return QueryComplexity.COMPLEX

        return QueryComplexity.SIMPLE

    def decompose(
        self,
        qu: QueryUnderstanding,
        resolved_query: str,
        workspace: Optional[KnowledgeWorkspace] = None,
    ) -> List[InformationNeed]:
        """
        Decompose a resolved query into a prioritized list of structured InformationNeeds.
        Applies evidence reuse from KnowledgeWorkspace where available.
        """
        complexity = self.determine_complexity(qu, resolved_query)
        needs: List[InformationNeed] = []

        if complexity == QueryComplexity.SIMPLE:
            # 1. Simple Single-Need Path
            primary_entity = qu.entity or resolved_query
            need = InformationNeed(
                question=resolved_query,
                purpose=f"Direct factual retrieval for {primary_entity}",
                priority=NeedPriority.HIGH,
                required_entities=[primary_entity] if qu.entity else [],
                depth=1,
            )
            # Check Evidence Reuse
            if workspace and primary_entity:
                facts = workspace.find_facts(subject=primary_entity)
                if facts:
                    need.status = NeedStatus.SATISFIED
                    need.supporting_evidence = [f.evidence_refs[0].page_title for f in facts if f.evidence_refs]
            needs.append(need)
            return needs

        # 2. Comparison Decomposition Path
        if qu.intent == IntentCategory.COMPARISON or any(w in resolved_query.lower() for w in ["compare", "difference", " vs "]):
            entities = self._extract_comparison_entities(resolved_query, qu)
            if len(entities) >= 2:
                e1, e2 = entities[0], entities[1]
                need1 = InformationNeed(
                    question=f"Key contributions and background of {e1}",
                    purpose=f"Profile primary subject {e1}",
                    priority=NeedPriority.HIGH,
                    required_entities=[e1],
                    depth=1,
                )
                need2 = InformationNeed(
                    question=f"Key contributions and background of {e2}",
                    purpose=f"Profile comparison subject {e2}",
                    priority=NeedPriority.HIGH,
                    required_entities=[e2],
                    depth=1,
                )
                need3 = InformationNeed(
                    question=f"Comparative dimensions, similarities, and differences between {e1} and {e2}",
                    purpose=f"Synthesize comparative analysis between {e1} and {e2}",
                    priority=NeedPriority.MEDIUM,
                    required_entities=[e1, e2],
                    dependencies=[need1.id, need2.id],
                    depth=1,
                )

                # Check evidence reuse
                if workspace:
                    for n, ent in [(need1, e1), (need2, e2)]:
                        facts = workspace.find_facts(subject=ent)
                        if facts:
                            n.status = NeedStatus.SATISFIED
                            n.supporting_evidence = [f.evidence_refs[0].page_title for f in facts if f.evidence_refs]

                needs.extend([need1, need2, need3])
                return needs

        # 3. Causal / Explanatory Decomposition Path
        primary_entity = qu.entity or self._extract_primary_concept(resolved_query)
        need_foundational = InformationNeed(
            question=f"Core background and contributions of {primary_entity}",
            purpose=f"Establish foundational factual basis for {primary_entity}",
            priority=NeedPriority.HIGH,
            required_entities=[primary_entity],
            depth=1,
        )
        need_explanatory = InformationNeed(
            question=resolved_query,
            purpose=f"Explanatory analysis and causal relationships for {primary_entity}",
            priority=NeedPriority.HIGH,
            required_entities=[primary_entity],
            dependencies=[need_foundational.id],
            depth=1,
        )

        # Check evidence reuse
        if workspace and primary_entity:
            facts = workspace.find_facts(subject=primary_entity)
            if facts:
                need_foundational.status = NeedStatus.SATISFIED
                need_foundational.supporting_evidence = [f.evidence_refs[0].page_title for f in facts if f.evidence_refs]

        needs.extend([need_foundational, need_explanatory])
        return needs

    def evaluate_and_adapt(
        self,
        needs: List[InformationNeed],
        research_ctx: ResearchContext,
        workspace: KnowledgeWorkspace,
        max_depth: int = 2,
    ) -> List[InformationNeed]:
        """
        Evaluate fulfillment status of information needs against accumulated evidence
        and structured knowledge. Generates new justified sub-needs if critical gaps appear.
        """
        existing_page_titles = {_norm(e.page_title) for e in research_ctx.evidence}
        existing_content = " ".join(e.content.lower() for e in research_ctx.evidence)

        for need in needs:
            if need.status == NeedStatus.SATISFIED:
                continue

            # Check dependencies
            if need.dependencies:
                dep_needs = [n for n in needs if n.id in need.dependencies]
                if any(d.status not in (NeedStatus.SATISFIED, NeedStatus.PARTIALLY_SATISFIED) for d in dep_needs):
                    need.status = NeedStatus.BLOCKED
                    continue

            # Check if required entities are covered
            if need.required_entities:
                covered_entities = []
                for ent in need.required_entities:
                    norm_ent = _norm(ent)
                    if (
                        any(norm_ent in title or title in norm_ent for title in existing_page_titles)
                        or (workspace and workspace.find_entities(ent))
                    ):
                        covered_entities.append(ent)

                if len(covered_entities) == len(need.required_entities):
                    # Check for contradiction / conflict preventing false satisfaction
                    has_contradiction = False
                    conflict_details = []
                    if workspace and (getattr(workspace, "contradictions", None) or getattr(workspace, "conflicts", None)):
                        for c in getattr(workspace, "contradictions", []):
                            if any(_norm(ent) in _norm(c.topic) for ent in need.required_entities):
                                has_contradiction = True
                                conflict_details.append(c.details)
                                break
                        for cf in getattr(workspace, "conflicts", []):
                            if getattr(cf, "status", None) == "UNRESOLVED":
                                if any(_norm(ent) in _norm(cf.topic) for ent in need.required_entities):
                                    has_contradiction = True
                                    conflict_details.append(cf.topic)
                                    break

                    if has_contradiction:
                        need.status = NeedStatus.PARTIALLY_SATISFIED
                        need.metadata["contradiction_detected"] = True
                        need.metadata["conflict_detected"] = True
                        need.metadata["conflict_details"] = conflict_details
                    else:
                        need.status = NeedStatus.SATISFIED
                        need.supporting_evidence = list(existing_page_titles)
                elif covered_entities:
                    need.status = NeedStatus.PARTIALLY_SATISFIED
                else:
                    need.status = NeedStatus.IN_PROGRESS
            else:
                # Need without specific entities (e.g. general factual query)
                has_conflicts = bool(workspace and getattr(workspace, "has_unresolved_conflicts", lambda: False)())
                if research_ctx.evidence and not has_conflicts:
                    need.status = NeedStatus.SATISFIED
                    need.supporting_evidence = list(existing_page_titles)
                elif research_ctx.evidence:
                    need.status = NeedStatus.PARTIALLY_SATISFIED

        # ── Adaptive Sub-Need Creation (Bounded) ─────────────────────────────
        # If a high-priority need is satisfied but reveals a missing secondary concept in the query
        if len(needs) < 4:
            for need in list(needs):
                if need.status == NeedStatus.SATISFIED and need.depth < max_depth:
                    # Look for unaddressed concepts in the original request
                    missing_concept = self._find_uncovered_query_concept(
                        research_ctx.resolved_request, existing_page_titles, needs
                    )
                    if missing_concept:
                        norm_mc = _norm(missing_concept)
                        # Check duplicate prevention
                        if not any(_norm(n.question) == norm_mc or _norm(n.purpose) == norm_mc for n in needs):
                            logger.info(
                                "[DECOMPOSER] Adaptively created sub-need for missing concept: %r",
                                missing_concept
                            )
                            new_need = InformationNeed(
                                question=f"Details on {missing_concept}",
                                purpose=f"Supplementary knowledge on {missing_concept} required by user query",
                                priority=NeedPriority.MEDIUM,
                                required_entities=[missing_concept],
                                dependencies=[need.id],
                                depth=need.depth + 1,
                            )
                            needs.append(new_need)
                            break

        return needs

    def _extract_comparison_entities(
        self, text: str, qu: QueryUnderstanding
    ) -> List[str]:
        """Extract comparison entities with regex."""
        patterns = [
            r"(?i)compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\s*\?|\s*$)",
            r"(?i)differences?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*\?|\s*$)",
            r"(?i)similarities\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*\?|\s*$)",
            r"(?i)(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s*\?|\s*$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                e1 = m.group(1).strip().rstrip("?.!,;:")
                e2 = m.group(2).strip().rstrip("?.!,;:")
                # Strip leading/trailing filler
                e1 = re.sub(r"^(?:the|a|an)\s+", "", e1, flags=re.IGNORECASE).strip()
                e2 = re.sub(r"^(?:the|a|an)\s+", "", e2, flags=re.IGNORECASE).strip()
                if len(e1) >= 2 and len(e2) >= 2:
                    return [e1, e2]
        return [qu.entity] if qu.entity else []

    def _extract_primary_concept(self, text: str) -> str:
        """Extract main subject concept from causal/explanatory query."""
        m = re.search(r"(?i)(?:why was|why did|why is|how did|importance of|influence of)\s+(.+?)(?:'s|\s+important|\s+influence|\s+matter|\s*\?|\s*$)", text)
        if m:
            concept = m.group(1).strip().rstrip("?.!,;:")
            return re.sub(r"^(?:the|a|an)\s+", "", concept, flags=re.IGNORECASE).strip()
        words = [w for w in text.split() if w.lower() not in {"what", "why", "how", "is", "was", "the", "a", "an", "important", "to"}]
        return " ".join(words[:3]) if words else text

    def _find_uncovered_query_concept(
        self,
        query: str,
        covered_pages: Set[str],
        current_needs: List[InformationNeed],
    ) -> Optional[str]:
        """Find key named entities or topics in query not yet covered in evidence or needs."""
        # Filter out common leading instruction / question verbs
        skip_words = {
            "what", "why", "how", "who", "when", "where", "which",
            "compare", "between", "explain", "describe", "detail", "outline",
            "summarize", "tell", "show", "give", "provide", "discuss"
        }
        current_need_entities = {
            _norm(e) for n in current_needs for e in n.required_entities
        }
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        for c in candidates:
            # Strip leading skip words (e.g., 'Explain Newton' -> 'Newton')
            words = c.split()
            while words and words[0].lower() in skip_words:
                words.pop(0)
            if not words:
                continue
            cleaned_c = " ".join(words)
            norm_c = _norm(cleaned_c)
            if norm_c in skip_words or len(norm_c) <= 2:
                continue

            # Check if this concept is already covered by a page title or existing need entity
            already_covered = (
                any(norm_c in page or page in norm_c for page in covered_pages)
                or any(norm_c in ent or ent in norm_c for ent in current_need_entities)
            )
            if not already_covered:
                return cleaned_c
        return None


query_decomposer = QueryDecomposer()
