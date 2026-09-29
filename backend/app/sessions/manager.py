"""
ResearchSessionManager for Stage 10 Long-Horizon Task Execution & Research Session Continuity.

Provides the application-level interface for creating, loading, persisting, pausing,
resuming, completing, cancelling, archiving, and deleting long-horizon research sessions.
Handles snapshotting and restoring of live ResearchContext and KnowledgeWorkspace state.
"""
from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.agent.decomposition.models import InformationNeed, NeedPriority, NeedStatus
from app.agent.research.action import ResearchAction, ResearchActionStatus, ResearchActionType
from app.agent.research.context import ResearchContext
from app.agent.research.evidence import CompletionStatus, EvidenceItem
from app.agent.research.finding import Finding
from app.agent.research.plan import PlanStatus, ResearchPlan
from app.core.source_attribution import SourceMetadata
from app.knowledge.models import (
    Claim, Conflict, Contradiction, Entity, EntityType, EvidenceReference, Fact, FactStatus,
    GroundingStatus, KnowledgeQuality, Relationship,
)
from app.knowledge.workspace import KnowledgeWorkspace
from app.services.wikipedia_navigator import NavigationHistory, NavigationStep
from app.sessions.models import ResearchSession, SessionStatus
from app.sessions.store import ResearchSessionStore, StaleSessionError, research_session_store

logger = logging.getLogger("nexia.sessions.manager")


class ResearchSessionManager:
    """
    Dedicated manager for task-continuity research sessions.
    
    Maintains strict separation between:
      1. Conversation State
      2. Persistent User Memory
      3. Research Sessions
      4. Transient Runtime State
    """

    def __init__(self, store: Optional[ResearchSessionStore] = None):
        self.store = store or research_session_store

    def create_session(
        self,
        objective: str,
        original_request: str,
        resolved_request: Optional[str] = None,
        user_id: str = "default_user",
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        information_needs: Optional[List[Dict[str, Any]]] = None,
        research_plan: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchSession:
        """Create a new persistent research session."""
        resolved = resolved_request or original_request
        session_title = title or objective[:60]
        session = ResearchSession(
            user_id=user_id,
            conversation_id=conversation_id,
            title=session_title,
            objective=objective,
            original_request=original_request,
            resolved_request=resolved,
            information_needs=information_needs or [],
            research_plan=research_plan,
            status=SessionStatus.IN_PROGRESS,
            metadata=metadata or {},
        )
        saved = self.store.save(session)
        logger.info("[SESSION MGR] Created session id=%s title=%r", saved.session_id, saved.title)
        return saved

    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Fetch session by ID."""
        return self.store.get(session_id)

    def save_session(self, session: ResearchSession) -> ResearchSession:
        """Persist session state."""
        return self.store.save(session)

    def pause_session(
        self,
        session_id: str,
        research_ctx: Optional[ResearchContext] = None,
        workspace: Optional[KnowledgeWorkspace] = None,
    ) -> Optional[ResearchSession]:
        """Pause an active research session and snapshot current state."""
        session = self.get_session(session_id)
        if not session:
            return None

        if research_ctx:
            self.snapshot_research_context(session, research_ctx, workspace=workspace)

        session.status = SessionStatus.PAUSED
        session.updated_at = time.time()
        saved = self.store.save(session)
        logger.info("[SESSION MGR] Paused session id=%s", session_id)
        return saved

    def resume_session(self, session_id: str) -> Optional[ResearchSession]:
        """Resume a paused or in-progress session."""
        session = self.get_session(session_id)
        if not session:
            return None

        session.status = SessionStatus.IN_PROGRESS
        session.updated_at = time.time()
        saved = self.store.save(session)
        logger.info("[SESSION MGR] Resumed session id=%s", session_id)
        return saved

    def complete_session(
        self,
        session_id: str,
        research_ctx: Optional[ResearchContext] = None,
        workspace: Optional[KnowledgeWorkspace] = None,
        claims: Optional[List[Any]] = None,
        sources: Optional[List[Any]] = None,
    ) -> Optional[ResearchSession]:
        """Mark session as completed and persist final snapshot."""
        session = self.get_session(session_id)
        if not session:
            return None

        if research_ctx:
            self.snapshot_research_context(
                session, research_ctx, workspace=workspace, claims=claims, sources=sources
            )

        session.status = SessionStatus.COMPLETED
        session.updated_at = time.time()
        saved = self.store.save(session)
        logger.info("[SESSION MGR] Completed session id=%s", session_id)
        return saved

    def cancel_session(self, session_id: str) -> Optional[ResearchSession]:
        """Cancel an active session."""
        session = self.get_session(session_id)
        if not session:
            return None

        session.status = SessionStatus.CANCELLED
        session.updated_at = time.time()
        saved = self.store.save(session)
        logger.info("[SESSION MGR] Cancelled session id=%s", session_id)
        return saved

    def archive_session(self, session_id: str) -> Optional[ResearchSession]:
        """Archive a completed session."""
        session = self.get_session(session_id)
        if not session:
            return None

        session.status = SessionStatus.ARCHIVED
        session.updated_at = time.time()
        saved = self.store.save(session)
        logger.info("[SESSION MGR] Archived session id=%s", session_id)
        return saved

    def delete_session(self, session_id: str) -> bool:
        """Permanently delete a single session."""
        deleted = self.store.delete(session_id)
        logger.info("[SESSION MGR] Deleted session id=%s (success=%s)", session_id, deleted)
        return deleted

    def list_active_sessions(self, user_id: str = "default_user") -> List[ResearchSession]:
        """List active sessions (IN_PROGRESS, PAUSED, DRAFT) for a user."""
        all_sessions = self.store.list(user_id=user_id)
        return [
            s for s in all_sessions
            if s.status in (SessionStatus.IN_PROGRESS, SessionStatus.PAUSED, SessionStatus.DRAFT)
        ]

    def list_all_sessions(
        self, user_id: str = "default_user", status: Optional[SessionStatus] = None
    ) -> List[ResearchSession]:
        """List all sessions for a user."""
        return self.store.list(user_id=user_id, status=status)

    def find_matching_session(
        self, query: str, user_id: str = "default_user"
    ) -> Tuple[Optional[ResearchSession], List[ResearchSession], str]:
        """
        Identify the matching active session for a resume/action request.
        
        Returns:
            (matched_session, candidate_list, match_type)
            
        Match types:
            "EXACT_ID"          - Matched exact session ID
            "TOPIC_MATCH"       - Unambiguously matched search keywords
            "DEFAULT_SINGLE"    - Generic query and exactly 1 active session exists
            "AMBIGUOUS"         - Multiple sessions match or multiple active sessions exist on generic query
            "NO_MATCH"          - No active session matches the query
            "NO_ACTIVE_SESSIONS"- No active sessions exist at all
        """
        active_sessions = self.list_active_sessions(user_id=user_id)
        if not active_sessions:
            return None, [], "NO_ACTIVE_SESSIONS"

        cleaned_q = query.strip().lower()

        # 1. Check for explicit session_id match
        for s in active_sessions:
            if s.session_id.lower() == cleaned_q or s.session_id.lower() in cleaned_q.split():
                return s, [s], "EXACT_ID"

        # 2. Extract potential topic keywords
        stop_words = {
            "continue", "resume", "my", "research", "session", "task", "about",
            "the", "on", "please", "with", "work", "investigation", "study",
        }
        tokens = [w for w in re.findall(r"\w+", cleaned_q) if w not in stop_words and len(w) > 2]

        if not tokens:
            # Generic query like "continue my research"
            if len(active_sessions) == 1:
                return active_sessions[0], active_sessions, "DEFAULT_SINGLE"
            else:
                return None, active_sessions, "AMBIGUOUS"

        # Match tokens against session title, objective, request
        matches: List[ResearchSession] = []
        for s in active_sessions:
            text_to_search = f"{s.title} {s.objective} {s.original_request} {s.resolved_request}".lower()
            if any(token in text_to_search for token in tokens):
                matches.append(s)

        if len(matches) == 1:
            return matches[0], matches, "TOPIC_MATCH"
        elif len(matches) > 1:
            return None, matches, "AMBIGUOUS"
        else:
            return None, [], "NO_MATCH"

    def snapshot_research_context(
        self,
        session: ResearchSession,
        research_ctx: ResearchContext,
        workspace: Optional[KnowledgeWorkspace] = None,
        claims: Optional[List[Any]] = None,
        sources: Optional[List[Any]] = None,
    ) -> ResearchSession:
        """
        Snapshot live ResearchContext and KnowledgeWorkspace into the persistent ResearchSession.
        """
        # 1. Information Needs
        if hasattr(research_ctx, "information_needs") and research_ctx.information_needs:
            session.information_needs = [
                n.model_dump() if hasattr(n, "model_dump") else (n.to_dict() if hasattr(n, "to_dict") else dict(n))
                for n in research_ctx.information_needs
            ]

        # 2. Plan
        if research_ctx.plan:
            session.research_plan = research_ctx.plan.to_dict()

        # 3. Actions
        session.completed_actions = [a.to_dict() for a in research_ctx.completed_actions]
        session.pending_actions = [a.to_dict() for a in research_ctx.pending_actions]

        # 4. Evidence (preserves provenance without dumping massive duplicate text)
        evidence_dicts: List[Dict[str, Any]] = []
        for ev in research_ctx.evidence:
            ev_dict = {
                "action_id": ev.action_id,
                "page_title": ev.page_title,
                "page_url": ev.page_url,
                "content": ev.content,
                "source": ev.source,
                "metadata": ev.metadata,
            }
            evidence_dicts.append(ev_dict)
        session.evidence = evidence_dicts

        # 5. Navigation History
        if hasattr(research_ctx, "navigation_history") and research_ctx.navigation_history:
            session.navigation_history = research_ctx.navigation_history.to_summary_dict()

        # 6. Structured Knowledge Workspace
        if workspace:
            session.knowledge_workspace_state = {
                "run_id": workspace.run_id,
                "entities": [e.model_dump() for e in workspace.entities.values()],
                "facts": [f.model_dump() for f in workspace.facts],
                "relationships": [r.model_dump() for r in workspace.relationships],
                "claims": [c.model_dump() for c in workspace.claims],
                "contradictions": [ct.model_dump() for ct in workspace.contradictions],
                "conflicts": [cf.model_dump() for cf in workspace.conflicts],
                "evidence_references": [er.model_dump() for er in workspace.evidence_references],
                "quality": workspace.quality.model_dump() if workspace.quality else None,
            }

        # 7. Claims and Sources
        if claims:
            session.claims = [
                c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in claims
            ]
        elif workspace and workspace.claims:
            session.claims = [c.model_dump() for c in workspace.claims]

        if sources:
            session.sources = [
                s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in sources
            ]

        # 8. Unresolved items
        unresolved: List[str] = []
        if hasattr(research_ctx, "information_needs"):
            for n in research_ctx.information_needs:
                status = getattr(n, "status", None)
                if status and status in (NeedStatus.PENDING, NeedStatus.IN_PROGRESS, NeedStatus.PARTIALLY_SATISFIED, NeedStatus.UNRESOLVED):
                    q = getattr(n, "question", str(n))
                    unresolved.append(q)
        if research_ctx.unresolved_questions:
            unresolved.extend(research_ctx.unresolved_questions)
        session.unresolved_items = list(dict.fromkeys(unresolved))

        return session

    def restore_research_context(
        self, session: ResearchSession
    ) -> Tuple[ResearchContext, Optional[KnowledgeWorkspace]]:
        """
        Reconstruct live ResearchContext and KnowledgeWorkspace from a persisted ResearchSession.
        """
        # 1. Restore Actions
        completed_actions: List[ResearchAction] = []
        for act_dict in session.completed_actions:
            act = ResearchAction(
                action_id=act_dict.get("action_id", "act_0"),
                action_type=ResearchActionType(act_dict.get("action_type", ResearchActionType.WIKIPEDIA_RETRIEVE.value)),
                objective=act_dict.get("objective", ""),
                query=act_dict.get("query", ""),
                entity=act_dict.get("entity"),
                section_title=act_dict.get("section_title"),
                source_page=act_dict.get("source_page"),
                target_link=act_dict.get("target_link"),
                navigation_depth=act_dict.get("navigation_depth", 0),
                status=ResearchActionStatus(act_dict.get("status", ResearchActionStatus.COMPLETED.value)),
                duration_ms=act_dict.get("duration_ms", 0),
                error=act_dict.get("error"),
            )
            completed_actions.append(act)

        pending_actions: List[ResearchAction] = []
        for act_dict in session.pending_actions:
            act = ResearchAction(
                action_id=act_dict.get("action_id", "act_p"),
                action_type=ResearchActionType(act_dict.get("action_type", ResearchActionType.WIKIPEDIA_RETRIEVE.value)),
                objective=act_dict.get("objective", ""),
                query=act_dict.get("query", ""),
                entity=act_dict.get("entity"),
                section_title=act_dict.get("section_title"),
                source_page=act_dict.get("source_page"),
                target_link=act_dict.get("target_link"),
                navigation_depth=act_dict.get("navigation_depth", 0),
                status=ResearchActionStatus(act_dict.get("status", ResearchActionStatus.PENDING.value)),
            )
            pending_actions.append(act)

        # 2. Restore Evidence
        evidence_items: List[EvidenceItem] = []
        for ev_dict in session.evidence:
            ev = EvidenceItem(
                action_id=ev_dict.get("action_id", "act_0"),
                page_title=ev_dict.get("page_title", ""),
                page_url=ev_dict.get("page_url", ""),
                content=ev_dict.get("content", ""),
                source=ev_dict.get("source", "Wikipedia"),
                metadata=ev_dict.get("metadata", {}),
            )
            evidence_items.append(ev)

        # 3. Restore Information Needs
        info_needs: List[InformationNeed] = []
        for n_dict in session.information_needs:
            try:
                info_needs.append(InformationNeed.model_validate(n_dict))
            except Exception as exc:
                logger.warning("[SESSION MGR] Failed parsing InformationNeed: %s", exc)

        # 4. Restore Navigation History
        nav_history = NavigationHistory()
        if session.navigation_history:
            nav_data = session.navigation_history
            nav_history.max_depth = nav_data.get("max_depth", 2)
            for page in nav_data.get("visited_pages", []):
                nav_history.visited_pages.add(page.lower())
            for step_dict in nav_data.get("steps", []):
                step = NavigationStep(
                    action_type=step_dict.get("action_type", "retrieve"),
                    source_page=step_dict.get("source_page"),
                    target_page=step_dict.get("target_page"),
                    section=step_dict.get("section"),
                    status=step_dict.get("status", "success"),
                    depth=step_dict.get("depth", 0),
                    metadata=step_dict.get("metadata", {}),
                )
                nav_history.steps.append(step)

        # 5. Restore Plan
        plan_obj: Optional[ResearchPlan] = None
        if session.research_plan:
            p_data = session.research_plan
            plan_obj = ResearchPlan(
                objective=p_data.get("objective", session.objective),
                actions=completed_actions + pending_actions,
                completion_criteria=p_data.get("completion_criteria", "Satisfy information needs"),
                status=PlanStatus(p_data.get("status", PlanStatus.IN_PROGRESS.value)),
                metadata=p_data.get("metadata", {}),
            )

        # 6. Construct ResearchContext
        sources_list: List[SourceMetadata] = []
        for s_dict in session.sources:
            try:
                sources_list.append(SourceMetadata.model_validate(s_dict))
            except Exception:
                pass

        completion_status = CompletionStatus.SUFFICIENT if session.status == SessionStatus.COMPLETED else CompletionStatus.MORE_INFORMATION_NEEDED

        research_ctx = ResearchContext(
            request_id=session.session_id,
            conversation_id=session.conversation_id or "default_conv",
            original_request=session.original_request,
            resolved_request=session.resolved_request,
            objectives=[session.objective],
            plan=plan_obj,
            completed_actions=completed_actions,
            pending_actions=pending_actions,
            evidence=evidence_items,
            findings=[],
            sources=sources_list,
            unresolved_questions=session.unresolved_items,
            completion_status=completion_status,
            information_needs=info_needs,
            navigation_history=nav_history,
            metadata={"resumed": True, "session_id": session.session_id, "version": session.version},
        )

        # 7. Restore KnowledgeWorkspace
        workspace: Optional[KnowledgeWorkspace] = None
        if session.knowledge_workspace_state:
            kw_data = session.knowledge_workspace_state
            workspace = KnowledgeWorkspace(run_id=session.session_id)
            for ent_dict in kw_data.get("entities", []):
                ent = Entity.model_validate(ent_dict)
                workspace.entities[ent.canonical_name.lower()] = ent
            for f_dict in kw_data.get("facts", []):
                workspace.facts.append(Fact.model_validate(f_dict))
            for r_dict in kw_data.get("relationships", []):
                workspace.relationships.append(Relationship.model_validate(r_dict))
            for c_dict in kw_data.get("claims", []):
                workspace.claims.append(Claim.model_validate(c_dict))
            for ct_dict in kw_data.get("contradictions", []):
                workspace.contradictions.append(Contradiction.model_validate(ct_dict))
            for cf_dict in kw_data.get("conflicts", []):
                workspace.conflicts.append(Conflict.model_validate(cf_dict))
            for er_dict in kw_data.get("evidence_references", []):
                workspace.evidence_references.append(EvidenceReference.model_validate(er_dict))
            if kw_data.get("quality"):
                workspace.quality = KnowledgeQuality.model_validate(kw_data["quality"])

        return research_ctx, workspace

    def evaluate_evidence_freshness(
        self,
        query_or_topic: str,
        evidence_timestamp: Optional[datetime] = None,
        max_age_days: int = 30,
    ) -> bool:
        """
        Determine if evidence remains sufficiently fresh or if a time-sensitive query
        requires fresh Wikipedia retrieval.
        """
        time_sensitive_keywords = {
            "current", "currently", "now", "present", "today", "recent", "recently",
            "leader", "president", "ceo", "prime minister", "status", "latest",
            "holding office", "incumbent",
        }
        query_words = set(re.findall(r"\b\w+\b", query_or_topic.lower()))
        is_time_sensitive = bool(query_words & time_sensitive_keywords)

        if not is_time_sensitive:
            return True

        if not evidence_timestamp:
            return False

        age = datetime.now(timezone.utc) - evidence_timestamp
        return age.days <= max_age_days


research_session_manager = ResearchSessionManager()
