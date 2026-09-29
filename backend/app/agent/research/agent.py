"""
WikipediaResearchAgent — the agentic Wikipedia research loop for Stage 3.

The agent owns:
    - Research planning (deterministic, based on QU + coreference results)
    - Action selection and scheduling
    - Wikipedia action execution (via Stage 2 wikipedia_retriever)
    - Evidence accumulation and workspace management
    - Completion evaluation (deterministic heuristic)
    - Duplicate action prevention
    - Action budget enforcement
    - Follow-up action generation for multi-hop questions
    - Evidence synthesis (building combined evidence for Stage 2 grounded_generator)
    - Multi-source attribution

Gemma is NOT the agent. Gemma is a language-model dependency used ONLY for
final answer generation (not for planning, action selection, or completion decisions).

The agent uses the existing Stage 2 wikipedia_retriever singleton without modification.
"""
import logging
import re
import time
import unicodedata
import uuid
from typing import Dict, List, Optional, Tuple

from app.agent.research.action import (
    ResearchAction, ResearchActionStatus, ResearchActionType,
)
from app.agent.research.context import ResearchContext
from app.agent.research.evidence import CompletionStatus, EvidenceItem
from app.agent.research.finding import Finding
from app.agent.research.plan import PlanStatus, ResearchPlan
from app.agent.decomposition import (
    InformationNeed, NeedPriority, NeedStatus, QueryComplexity,
    query_decomposer, research_strategy,
)
from app.knowledge import KnowledgeWorkspace, knowledge_extractor
from app.core.answer_validator import ValidationResult
from app.core.query_understanding import QueryUnderstanding, IntentCategory
from app.core.context_manager import ContextResolution
from app.core.source_attribution import SourceMetadata, source_attributor
from app.services.wikipedia_retriever import (
    RetrievalResult, RetrievalStatus, wikipedia_retriever,
)
from app.services.wikipedia_navigator import (
    NavigationHistory, NavigationStep, WikipediaArticle, WikipediaNavigator, wikipedia_navigator,
)

from app.optimization.concurrency import bounded_executor
from app.optimization.telemetry import telemetry_tracker

logger = logging.getLogger("nexia.agent.research")

# ── Configuration ─────────────────────────────────────────────────────────────

# Hard upper bound on the total number of Wikipedia actions per request.
# Prevents runaway loops. Simple questions will exit early before hitting this.
MAX_ACTIONS: int = 4

# Minimum total evidence characters before an answer can be generated.
MIN_EVIDENCE_CHARS: int = 150

# Characters of evidence expected per planned entity (for sufficiency check).
CHARS_PER_ENTITY: int = 300

# Minimum evidence characters for a follow-up action to be worthwhile.
FOLLOWUP_EVIDENCE_THRESHOLD: int = 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _new_action_id() -> str:
    return str(uuid.uuid4())[:8]


def _norm(text: Optional[str]) -> str:
    """Lowercase, strip diacritics, collapse whitespace — for duplicate detection."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


def _clean_entity_phrase(text: str) -> str:
    """
    Strip filler words from an entity phrase extracted by pattern matching.
    E.g. "Albert Einstein's theories" -> "Albert Einstein"
    Carefully avoids stripping letter sequences that are part of the entity name
    (e.g. "Newton" must not lose its "on" suffix).
    """
    # Remove trailing standalone filler WORDS only (preceded by whitespace).
    # Use \b to avoid stripping suffixes that happen to match filler words.
    text = re.sub(
        r"\s+(?:and|or|but|with|of|in|by|for|from|the)\s*$",
        "", text, flags=re.IGNORECASE,
    ).strip()
    # Remove leading article
    text = re.sub(
        r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE
    ).strip()
    # Strip trailing punctuation
    text = text.rstrip("?.!,;:")
    return text.strip()


# ── WikipediaResearchAgent ────────────────────────────────────────────────────

class WikipediaResearchAgent:
    """
    Executes the Stage 3 Wikipedia research loop.

    Per-request lifecycle:
        1. plan()   — creates a structured ResearchPlan from QU + resolution
        2. run()    — executes the plan in a controlled loop
            a. pick next pending action
            b. check for duplicates → skip if duplicate
            c. execute via wikipedia_retriever
            d. accumulate evidence
            e. evaluate completion
            f. generate follow-up action if MORE_INFORMATION_NEEDED
        3. After loop: synthesize evidence for generation

    All decisions are deterministic.
    Gemma is NOT called during research — only during final answer generation.
    """

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        *,
        original_query: str,
        resolved_query: str,
        resolved_entity: Optional[str],
        qu: QueryUnderstanding,
        conversation_id: str,
        request_id: str,
        existing_context: Optional[ResearchContext] = None,
    ) -> ResearchContext:
        """
        Execute the full research loop for one user request.

        Returns a ResearchContext with all accumulated evidence.
        The caller (WikipediaKnowledgeCapability) uses the evidence
        for grounded generation via the Stage 2 pipeline.
        """
        t_start = time.monotonic()

        # ── 1. Initialize or resume research context ──────────────────────────
        if existing_context is not None:
            context = existing_context
            needs = context.information_needs or query_decomposer.decompose(qu, resolved_query)
            context.information_needs = needs
            if not context.plan:
                plan = self._create_plan(qu, resolved_query, resolved_entity)
                context.plan = plan
                context.pending_actions = list(plan.actions)
                plan.status = PlanStatus.IN_PROGRESS
            else:
                plan = context.plan
        else:
            context = ResearchContext(
                request_id=request_id,
                conversation_id=conversation_id,
                original_request=original_query,
                resolved_request=resolved_query,
                objectives=[resolved_query],
                plan=None,
                completed_actions=[],
                pending_actions=[],
                evidence=[],
                findings=[],
                sources=[],
                unresolved_questions=[],
                completion_status=CompletionStatus.MORE_INFORMATION_NEEDED,
                navigation_history=NavigationHistory(max_depth=2),
            )

            # ── 1.5. Decompose Request into InformationNeeds ──────────────────────
            needs = query_decomposer.decompose(qu, resolved_query)
            context.information_needs = needs

            # ── 2. Create research plan ───────────────────────────────────────────
            plan = self._create_plan(qu, resolved_query, resolved_entity)
            context.plan = plan
            context.pending_actions = list(plan.actions)
            plan.status = PlanStatus.IN_PROGRESS

        logger.info(
            "[RESEARCH] run_id=%s actions_planned=%d needs=%d objective=%r resumed=%s",
            request_id[:8], len(plan.actions), len(needs), resolved_query[:80], existing_context is not None,
        )

        # ── 3. Bounded Parallel Initial Execution (Stage 13) ──────────────────
        if len(context.pending_actions) > 1 and all(
            a.action_type == ResearchActionType.WIKIPEDIA_RETRIEVE for a in context.pending_actions
        ):
            independent_actions = list(context.pending_actions)
            context.pending_actions.clear()

            async def _run_single_action(act: ResearchAction):
                if self._is_duplicate(act, context):
                    act.status = ResearchActionStatus.SKIPPED
                    return act
                act.status = ResearchActionStatus.RUNNING
                t_act = time.monotonic()
                await self._execute_action(act, context)
                act.duration_ms = int((time.monotonic() - t_act) * 1000)
                return act

            task_factories = [
                (lambda a=action: _run_single_action(a))
                for action in independent_actions
            ]
            completed = await bounded_executor.run_parallel(task_factories)
            for res in completed:
                if isinstance(res, ResearchAction):
                    context.completed_actions.append(res)

            # Evaluate initial parallel outcomes
            temp_ws = KnowledgeWorkspace(run_id=request_id)
            for ev in context.evidence:
                knowledge_extractor.extract_from_evidence(ev, temp_ws)
            query_decomposer.evaluate_and_adapt(context.information_needs, context, temp_ws)
            self._evaluate_completion(context, qu)

        # ── 3b. Execute sequential loop for remaining / follow-up needs ──────
        while (context.pending_actions or any(n.status != NeedStatus.SATISFIED for n in context.information_needs)) and self._should_continue(context):
            if not context.pending_actions:
                # Plan next action from unresolved information needs
                next_action = research_strategy.plan_next_action(context.information_needs, context)
                if next_action:
                    context.pending_actions.append(next_action)
                else:
                    break

            action = context.pending_actions.pop(0)

            # Duplicate check
            if self._is_duplicate(action, context):
                action.status = ResearchActionStatus.SKIPPED
                context.completed_actions.append(action)
                logger.debug(
                    "[RESEARCH] Skipped duplicate action: %r", action.query
                )
                continue

            # Execute
            action.status = ResearchActionStatus.RUNNING
            t_action = time.monotonic()
            await self._execute_action(action, context)
            action.duration_ms = int((time.monotonic() - t_action) * 1000)
            context.completed_actions.append(action)

            # Evaluate evidence completeness and update structured workspace
            temp_ws = KnowledgeWorkspace(run_id=request_id)
            for ev in context.evidence:
                knowledge_extractor.extract_from_evidence(ev, temp_ws)
            query_decomposer.evaluate_and_adapt(context.information_needs, context, temp_ws)

            self._evaluate_completion(context, qu)

            # Check if all high priority needs are satisfied -> mark SUFFICIENT
            high_priority = [n for n in context.information_needs if n.priority == NeedPriority.HIGH]
            if high_priority and all(n.status == NeedStatus.SATISFIED for n in high_priority):
                context.completion_status = CompletionStatus.SUFFICIENT
                if not context.pending_actions:
                    break

            logger.debug(
                "[RESEARCH] action=%s status=%s page=%r ev_count=%d "
                "completion=%s duration_ms=%d",
                action.action_id,
                action.status.value,
                action.result.page_title if action.result else None,
                len(context.evidence),
                context.completion_status.value,
                action.duration_ms,
            )

            # Generate follow-up if more info needed and budget allows
            if (
                context.completion_status == CompletionStatus.MORE_INFORMATION_NEEDED
                and len(context.completed_actions) < MAX_ACTIONS
                and not context.pending_actions
            ):
                followup = self._generate_followup_action(context, qu)
                if followup:
                    context.pending_actions.append(followup)
                    logger.debug(
                        "[RESEARCH] follow-up planned: %r", followup.query
                    )

        # ── 4. Finalize ───────────────────────────────────────────────────────
        plan.status = PlanStatus.COMPLETE

        # Invariant check: If any action fetched SUCCESS with content but context.evidence is empty,
        # flag an INTERNAL_ERROR rather than misclassifying as INSUFFICIENT_KNOWLEDGE.
        had_successful_fetch = any(
            bool(
                getattr(a, "result", None)
                and getattr(a.result, "status", None) == RetrievalStatus.SUCCESS
                and getattr(a.result, "content", None)
            )
            for a in context.completed_actions
        )

        if had_successful_fetch and len(context.evidence) == 0:
            logger.error(
                "[RESEARCH] Invariant violation: Wikipedia returned SUCCESS with non-empty extract, "
                "but context.evidence is 0! Setting status=INTERNAL_ERROR."
            )
            context.completion_status = CompletionStatus.INTERNAL_ERROR
        elif context.completion_status == CompletionStatus.MORE_INFORMATION_NEEDED:
            if context.evidence:
                context.completion_status = CompletionStatus.SUFFICIENT
            else:
                context.completion_status = CompletionStatus.INSUFFICIENT_KNOWLEDGE

        total_ms = int((time.monotonic() - t_start) * 1000)
        context.metadata["total_ms"] = total_ms
        context.metadata["actions_executed"] = len([
            a for a in context.completed_actions
            if a.status not in (ResearchActionStatus.SKIPPED,)
        ])

        logger.info(
            "[RESEARCH] run_id=%s complete: status=%s evidence=%d "
            "actions=%d total_ms=%d pages=%s",
            request_id[:8],
            context.completion_status.value,
            len(context.evidence),
            len(context.completed_actions),
            total_ms,
            [e.page_title for e in context.evidence],
        )

        return context

    # ── Planning ──────────────────────────────────────────────────────────────

    def _create_plan(
        self,
        qu: QueryUnderstanding,
        resolved_query: str,
        resolved_entity: Optional[str],
    ) -> ResearchPlan:
        """
        Create a structured research plan.

        Planning is fully deterministic — no LLM call.
        The plan is derived from the QU result and context resolution output.

        Rules:
          - Simple/entity questions → 1 action (primary entity/topic)
          - Comparison questions   → 1 action per identified entity (up to MAX_ACTIONS)
          - All others             → 1 initial action; follow-ups added dynamically
        """
        actions: List[ResearchAction] = []
        primary_entity = resolved_entity or qu.entity

        # Primary action — always created
        actions.append(ResearchAction(
            action_id=_new_action_id(),
            action_type=ResearchActionType.WIKIPEDIA_RETRIEVE,
            objective="Retrieve Wikipedia evidence for the main subject",
            query=resolved_query,
            entity=primary_entity,
        ))

        # Comparison: detect additional entities and add parallel actions
        if qu.intent == IntentCategory.COMPARISON:
            additional = self._extract_comparison_targets(qu, resolved_query)
            seen = {_norm(primary_entity)} if primary_entity else set()
            for entity_name, entity_query in additional:
                if _norm(entity_name) not in seen:
                    seen.add(_norm(entity_name))
                    actions.append(ResearchAction(
                        action_id=_new_action_id(),
                        action_type=ResearchActionType.WIKIPEDIA_RETRIEVE,
                        objective=f"Retrieve Wikipedia evidence for {entity_name}",
                        query=entity_query,
                        entity=entity_name,
                    ))
                    if len(actions) >= MAX_ACTIONS:
                        break

        is_multi = len(actions) > 1
        return ResearchPlan(
            objective=resolved_query,
            actions=actions,
            completion_criteria=(
                "Sufficient Wikipedia evidence retrieved for all subjects"
                if is_multi else
                "Sufficient Wikipedia evidence retrieved to answer the question"
            ),
            metadata={
                "intent": qu.intent,
                "entity": primary_entity,
                "multi_entity": is_multi,
                "initial_action_count": len(actions),
            },
        )

    def _extract_comparison_targets(
        self, qu: QueryUnderstanding, resolved_query: str
    ) -> List[Tuple[str, str]]:
        """
        Extract secondary entity names from comparison queries.
        Returns list of (entity_name, search_query) pairs.

        Uses pattern matching — no LLM call.
        """
        text = resolved_query.strip()

        # Ordered by specificity
        patterns = [
            # "compare X and Y" / "compare X with Y" / "compare X to Y"
            r"(?i)compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\s*\?|\s*$)",
            # "difference(s) between X and Y"
            r"(?i)differences?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*\?|\s*$)",
            # "similarities between X and Y"
            r"(?i)similarities\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*\?|\s*$)",
            # "X vs Y" / "X vs. Y"
            r"(?i)(.+?)\s+vs\.?\s+(.+?)(?:\s*\?|\s*$)",
            # "X versus Y"
            r"(?i)(.+?)\s+versus\s+(.+?)(?:\s*\?|\s*$)",
            # "X and Y" in knowledge context (fallback — only if other patterns fail)
            r"(?i)(?:who|what)\s+(?:was|is|were|are)\s+(.+?)\s+and\s+(.+?)(?:\s*\?|\s*$)",
        ]

        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                e1 = _clean_entity_phrase(m.group(1))
                e2 = _clean_entity_phrase(m.group(2))
                if e1 and e2 and len(e1) >= 2 and len(e2) >= 2:
                    return [(e1, e1), (e2, e2)]

        return []

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute_repair_action(
        self, action: ResearchAction, context: ResearchContext
    ) -> bool:
        """
        Execute a targeted repair action planned by the ReflectionEngine.
        Accumulates new evidence, appends to completed_actions, and updates timing.
        Returns True if successful and new evidence was added.
        """
        action.status = ResearchActionStatus.RUNNING
        t_action = time.monotonic()
        await self._execute_action(action, context)
        action.duration_ms = int((time.monotonic() - t_action) * 1000)
        context.completed_actions.append(action)
        return action.status == ResearchActionStatus.COMPLETED

    async def _execute_action(
        self, action: ResearchAction, context: ResearchContext
    ) -> None:
        """Execute one Wikipedia research or navigation action and store result in context."""
        try:
            if action.action_type == ResearchActionType.WIKIPEDIA_RETRIEVE_SECTION:
                result: RetrievalResult = await wikipedia_navigator.retrieve_section(
                    page_title=action.entity or action.query,
                    section_title=action.section_title or action.query,
                )
            elif action.action_type == ResearchActionType.WIKIPEDIA_FOLLOW_LINK:
                target = action.target_link or action.query
                result = await wikipedia_retriever.retrieve(
                    query=target,
                    entity=target,
                )
            elif action.action_type == ResearchActionType.WIKIPEDIA_RESOLVE_REDIRECT:
                struct = await wikipedia_navigator.retrieve_article_structure(action.query)
                result = await wikipedia_retriever.retrieve(
                    query=struct.page_title,
                    entity=struct.page_title,
                )
            elif action.action_type == ResearchActionType.WIKIPEDIA_RESOLVE_DISAMBIGUATION:
                struct = await wikipedia_navigator.retrieve_article_structure(action.query)
                selected, clarify = wikipedia_navigator.resolve_disambiguation(
                    title=action.query,
                    candidates=struct.disambiguation_options,
                    context_text=context.resolved_request,
                )
                if selected:
                    result = await wikipedia_retriever.retrieve(query=selected, entity=selected)
                else:
                    action.status = ResearchActionStatus.FAILED
                    action.error = f"Disambiguation for '{action.query}' could not be resolved from context."
                    return
            else:
                result = await wikipedia_retriever.retrieve(
                    query=action.query,
                    entity=action.entity,
                )
            action.result = result

            if result.status == RetrievalStatus.SUCCESS and result.content:
                # Record navigation step
                if context.navigation_history:
                    context.navigation_history.record_step(
                        NavigationStep(
                            action_type=action.action_type.value,
                            source_page=action.source_page or action.entity,
                            target_page=result.page_title or action.query,
                            section=action.section_title,
                            depth=action.navigation_depth,
                        )
                    )

                # Deduplicate evidence: don't store the exact same page & section twice
                existing_page_sections = {
                    f"{_norm(e.page_title)}::{_norm(e.metadata.get('section_title', ''))}"
                    for e in context.evidence
                }
                curr_key = f"{_norm(result.page_title or '')}::{_norm(result.metadata.get('section_title', ''))}"

                if curr_key not in existing_page_sections:
                    context.evidence.append(EvidenceItem(
                        action_id=action.action_id,
                        page_title=result.page_title or "",
                        page_url=result.page_url or "",
                        content=result.content,
                        source=result.source,
                        metadata={
                            "query": action.query,
                            "strategy": result.metadata.get("strategy", "unknown"),
                            "section_title": result.metadata.get("section_title"),
                        },
                    ))
                    action.status = ResearchActionStatus.COMPLETED
                else:
                    action.status = ResearchActionStatus.SKIPPED
                    logger.debug(
                        "[RESEARCH] Skipped duplicate page/section: %r", result.page_title
                    )
            else:
                action.status = ResearchActionStatus.FAILED
                action.error = f"No content found (status={result.status})"

        except Exception as exc:
            action.status = ResearchActionStatus.FAILED
            action.error = str(exc)
            logger.error(
                "[RESEARCH] action=%s unhandled exception: %s", action.action_id, exc, exc_info=True
            )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_completion(
        self, context: ResearchContext, qu: QueryUnderstanding
    ) -> None:
        """
        Determine whether the current evidence is sufficient.

        All decisions are deterministic — no LLM call.

        Sufficient criteria (any of):
          1. All initially planned entities have Wikipedia evidence AND
             total chars >= MIN_EVIDENCE_CHARS
          2. Total chars >= CHARS_PER_ENTITY * (number of planned entities)
          3. Action budget exhausted (use whatever evidence was gathered)

        Insufficient criteria:
          - No evidence at all → INSUFFICIENT_KNOWLEDGE
          - Evidence too sparse AND budget remains → MORE_INFORMATION_NEEDED
        """
        evidence_count = len(context.evidence)
        total_chars = sum(len(e.content) for e in context.evidence)
        actions_used = len(context.completed_actions)

        # No evidence — cannot proceed
        if evidence_count == 0:
            if actions_used >= MAX_ACTIONS:
                context.completion_status = CompletionStatus.INSUFFICIENT_KNOWLEDGE
            else:
                context.completion_status = CompletionStatus.MORE_INFORMATION_NEEDED
            return

        # Count distinct planned entities (excluding None)
        planned_entities = [
            _norm(a.entity) for a in (context.plan.actions if context.plan else [])
            if a.entity
        ]
        n_planned = max(1, len(planned_entities))

        # Check entity coverage: is each planned entity represented in evidence?
        covered_pages = {_norm(e.page_title) for e in context.evidence}
        covered_count = sum(
            1 for ent in planned_entities
            if any(ent in page or page in ent for page in covered_pages)
        )
        all_covered = (covered_count >= len(planned_entities)) if planned_entities else True

        # Sufficient?
        by_coverage = all_covered and total_chars >= MIN_EVIDENCE_CHARS
        by_volume = total_chars >= CHARS_PER_ENTITY * n_planned
        budget_exhausted = actions_used >= MAX_ACTIONS

        if by_coverage or by_volume or budget_exhausted:
            context.completion_status = CompletionStatus.SUFFICIENT
        else:
            context.completion_status = CompletionStatus.MORE_INFORMATION_NEEDED

    def _should_continue(self, context: ResearchContext) -> bool:
        """Return True if the research loop should continue."""
        if len(context.completed_actions) >= MAX_ACTIONS:
            return False
        if context.completion_status == CompletionStatus.SUFFICIENT:
            return False
        if context.completion_status == CompletionStatus.INSUFFICIENT_KNOWLEDGE:
            return False
        if context.completion_status == CompletionStatus.FAILED:
            return False
        if context.completion_status == CompletionStatus.CANCELLED:
            return False
        return True

    # ── Duplicate prevention ──────────────────────────────────────────────────

    def _is_duplicate(
        self, action: ResearchAction, context: ResearchContext
    ) -> bool:
        """
        Return True if this action is essentially the same as one already executed,
        or forms a navigation cycle, or exceeds depth bounds.
        """
        if context.navigation_history:
            if action.navigation_depth > context.navigation_history.max_depth:
                return True
            target_cand = action.target_link or action.entity or action.query
            if action.action_type in (ResearchActionType.WIKIPEDIA_FOLLOW_LINK, ResearchActionType.WIKIPEDIA_RETRIEVE_PAGE):
                if context.navigation_history.is_cycle(target_cand):
                    return True

        norm_query = _norm(action.query)
        norm_entity = _norm(action.entity or "")
        norm_sec = _norm(action.section_title or "")
        norm_link = _norm(action.target_link or "")

        for completed in context.completed_actions:
            if completed.action_type == action.action_type:
                if norm_sec and _norm(completed.section_title or "") == norm_sec and _norm(completed.entity or completed.query) == _norm(action.entity or action.query):
                    return True
                if norm_link and _norm(completed.target_link or "") == norm_link:
                    return True
                if not norm_sec and not norm_link:
                    if _norm(completed.query) == norm_query or (norm_entity and _norm(completed.entity or "") == norm_entity):
                        return True
        return False

    # ── Follow-up action generation ───────────────────────────────────────────

    def _generate_followup_action(
        self, context: ResearchContext, qu: QueryUnderstanding
    ) -> Optional[ResearchAction]:
        """
        Generate one follow-up Wikipedia action when evidence is insufficient.

        Used for multi-hop questions where the first retrieval doesn't fully
        cover the query's informational needs.

        Strategy (deterministic, no LLM):
          1. Look for named concepts in the query that aren't in retrieved evidence
          2. Check for "X's Y theory" / "theory of Y" patterns
          3. Only create a follow-up if the concept appears undercovered

        Returns None if no productive follow-up can be identified.
        """
        evidence_text = " ".join(e.content for e in context.evidence).lower()
        retrieved_pages = {_norm(e.page_title) for e in context.evidence}
        original = context.original_request
        resolved = context.resolved_request

        # ── Pattern 1: "X's [concept]" — try to retrieve [concept] separately
        poss_match = re.search(
            r"(?i)(?:\w+)'s\s+((?:theory|law|principle|work|discovery|equation|"
            r"contribution|model|concept|idea)\s*(?:of\s+[\w\s]+)?)",
            original,
        )
        if poss_match:
            concept = poss_match.group(1).strip()
            concept_norm = _norm(concept)
            if (
                concept_norm not in evidence_text[:3000]
                and concept_norm not in retrieved_pages
                and not self._is_duplicate(
                    ResearchAction(
                        action_id="", action_type=ResearchActionType.WIKIPEDIA_SEARCH,
                        objective="", query=concept,
                    ),
                    context,
                )
            ):
                return ResearchAction(
                    action_id=_new_action_id(),
                    action_type=ResearchActionType.WIKIPEDIA_SEARCH,
                    objective=f"Find additional information about: {concept}",
                    query=concept,
                    entity=concept,
                )

        # ── Pattern 2: "theory of Y" / "concept of Y" / "law of Y"
        topic_match = re.search(
            r"(?i)(?:theory|concept|law|principle|work|discovery)\s+of\s+"
            r"([\w][\w\s]{1,40}?)(?:\s*\?|$|\s+(?:important|significant))",
            resolved,
        )
        if topic_match:
            topic = _clean_entity_phrase(topic_match.group(1))
            topic_norm = _norm(topic)
            if (
                topic
                and topic_norm not in evidence_text[:3000]
                and topic_norm not in retrieved_pages
            ):
                return ResearchAction(
                    action_id=_new_action_id(),
                    action_type=ResearchActionType.WIKIPEDIA_RETRIEVE,
                    objective=f"Retrieve information about: {topic}",
                    query=topic,
                    entity=topic,
                )

        return None

    # ── Synthesis helpers ─────────────────────────────────────────────────────

    def build_combined_retrieval(self, context: ResearchContext) -> RetrievalResult:
        """
        Build a single RetrievalResult combining all accumulated evidence items.

        The combined result is passed to the Stage 2 GroundedGenerator, which
        requires a single RetrievalResult. This avoids modifying any Stage 2 modules.

        Single evidence: returned as-is (no wrapping overhead).
        Multiple evidence: concatenated with section headers for each article.
        """
        if not context.evidence:
            return RetrievalResult(
                status=RetrievalStatus.FAILED,
                query=context.resolved_request,
                metadata={"reason": "no_evidence"},
            )

        if len(context.evidence) == 1:
            ev = context.evidence[0]
            return RetrievalResult(
                status=RetrievalStatus.SUCCESS,
                query=context.resolved_request,
                page_title=ev.page_title,
                page_url=ev.page_url,
                content=ev.content,
                source=ev.source,
                metadata={"multi_source": False, "action_id": ev.action_id},
            )

        # Multiple evidence items — build a clearly sectioned content block
        sections = []
        page_titles = []
        page_urls = []

        for i, ev in enumerate(context.evidence, 1):
            sections.append(f"[Article {i}: {ev.page_title}]\n{ev.content}")
            page_titles.append(ev.page_title)
            if ev.page_url:
                page_urls.append(ev.page_url)

        combined_content = "\n\n".join(sections)
        combined_title = " | ".join(page_titles)
        primary_url = page_urls[0] if page_urls else ""

        return RetrievalResult(
            status=RetrievalStatus.SUCCESS,
            query=context.resolved_request,
            page_title=combined_title,
            page_url=primary_url,
            content=combined_content,
            source="Wikipedia",
            metadata={
                "multi_source": True,
                "source_count": len(context.evidence),
                "pages": page_titles,
                "urls": page_urls,
            },
        )

    def build_multi_sources(
        self,
        context: ResearchContext,
        val_res: Optional[ValidationResult],
    ) -> List[SourceMetadata]:
        """
        Build SourceMetadata for each successfully retrieved Wikipedia page.

        Calls source_attributor.build_sources() for each EvidenceItem so that
        Stage 2 source attribution logic is respected for each article independently.
        Deduplicates by URL to avoid repeated chips for the same page.
        """
        all_sources: List[SourceMetadata] = []
        seen_urls: set = set()

        for ev in context.evidence:
            single_result = RetrievalResult(
                status=RetrievalStatus.SUCCESS,
                query=context.resolved_request,
                page_title=ev.page_title,
                page_url=ev.page_url,
                content=ev.content,
                source=ev.source,
            )
            for src in source_attributor.build_sources(single_result, val_res):
                if src.url not in seen_urls:
                    seen_urls.add(src.url)
                    all_sources.append(src)

        return all_sources

    def augment_query_for_intent(
        self, query: str, qu: QueryUnderstanding, evidence_count: int
    ) -> str:
        """
        Add intent-specific formatting guidance to the query string
        passed to the grounded prompt when the question type warrants it.

        Comparison questions deserve structured answers, not one sentence.
        Other questions remain as-is to keep responses concise.
        """
        if (
            qu.intent == IntentCategory.COMPARISON
            and evidence_count > 1
            and not any(
                kw in query.lower()
                for kw in ["provide", "structured", "compare", "format"]
            )
        ):
            return (
                f"{query}\n\n"
                "Provide a structured comparison: cover key contributions, "
                "similarities, and differences for each person."
            )
        return query


# ── Global singleton ──────────────────────────────────────────────────────────

wikipedia_research_agent = WikipediaResearchAgent()
