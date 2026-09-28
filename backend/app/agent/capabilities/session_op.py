"""
ResearchSessionCapability — Handles explicit user research session operations (list, pause, cancel, archive, delete, status, save).
"""
import logging
import time
from typing import List, Optional

from app.agent.capability import Capability, CapabilityRequest, CapabilityResult, CapabilityStatus
from app.core.conversation_manager import conversation_manager
from app.core.source_attribution import SourceMetadata
from app.sessions.commands import SessionCommand, SessionCommandType, session_command_parser
from app.sessions.manager import research_session_manager
from app.sessions.models import SessionStatus

logger = logging.getLogger("nexia.agent.session_op")

CAPABILITY_ID = "session_operation"


class ResearchSessionCapability(Capability):
    """
    Handles explicit user session directives (LIST, PAUSE, CANCEL, ARCHIVE, DELETE, STATUS, SAVE).
    Does NOT execute external search or heavy generation.
    """

    @property
    def id(self) -> str:
        return CAPABILITY_ID

    @property
    def description(self) -> str:
        return "Handles persistent research session lifecycle operations."

    async def execute(self, request: CapabilityRequest) -> CapabilityResult:
        ctx = request.context
        cid = ctx.conversation_id
        user_id = ctx.metadata.get("user_id", "default_user")
        t_start = time.monotonic()

        session_cmd: Optional[SessionCommand] = request.input.get("session_cmd")
        if not session_cmd:
            session_cmd = session_command_parser.parse(ctx.original_query)

        # Record user message in conversation history
        conversation_manager.add_message(cid, "user", ctx.original_query)

        response_text = ""
        metadata = {
            "capability": CAPABILITY_ID,
            "conversation_id": cid,
            "user_id": user_id,
        }

        if not session_cmd:
            response_text = "I wasn't able to understand the research session instruction."

        elif session_cmd.command_type == SessionCommandType.LIST:
            active = research_session_manager.list_active_sessions(user_id=user_id)
            if active:
                lines = [f"- [{s.session_id}] **{s.title}** (Status: `{s.status.value}`, Needs: {len(s.information_needs)}, Evidence: {len(s.evidence)})" for s in active]
                response_text = f"You have {len(active)} active research session(s):\n" + "\n".join(lines)
            else:
                response_text = "You have no active research sessions."
            metadata["active_count"] = len(active)

        elif session_cmd.command_type == SessionCommandType.STATUS:
            matched, candidates, match_type = research_session_manager.find_matching_session(
                session_cmd.target_identifier or "", user_id=user_id
            )
            if matched:
                response_text = (
                    f"**Research Session [{matched.session_id}]**: {matched.title}\n"
                    f"- Status: `{matched.status.value}`\n"
                    f"- Objective: {matched.objective}\n"
                    f"- Information Needs: {len(matched.information_needs)}\n"
                    f"- Accumulated Evidence: {len(matched.evidence)} articles\n"
                    f"- Claims: {len(matched.claims)}\n"
                    f"- Unresolved items: {len(matched.unresolved_items)}"
                )
                metadata["session_id"] = matched.session_id
                metadata["status"] = matched.status.value
            elif match_type == "AMBIGUOUS":
                cand_str = ", ".join(f"'{c.title}' (ID: {c.session_id})" for c in candidates)
                response_text = f"Multiple active research sessions found: {cand_str}. Please specify which session you would like to inspect."
            else:
                response_text = "No matching research session found."

        elif session_cmd.command_type == SessionCommandType.PAUSE:
            matched, candidates, match_type = research_session_manager.find_matching_session(
                session_cmd.target_identifier or "", user_id=user_id
            )
            if matched:
                paused = research_session_manager.pause_session(matched.session_id)
                response_text = f"Paused research session: **{paused.title}** (ID: `{paused.session_id}`). You can resume it anytime."
                metadata["session_id"] = matched.session_id
                metadata["action"] = "paused"
            elif match_type == "AMBIGUOUS":
                cand_str = ", ".join(f"'{c.title}' (ID: {c.session_id})" for c in candidates)
                response_text = f"Multiple active research sessions found: {cand_str}. Please specify which session to pause."
            else:
                response_text = "No active research session to pause."

        elif session_cmd.command_type == SessionCommandType.SAVE:
            matched, candidates, match_type = research_session_manager.find_matching_session(
                session_cmd.target_identifier or "", user_id=user_id
            )
            if matched:
                research_session_manager.save_session(matched)
                response_text = f"Saved research session state for: **{matched.title}** (ID: `{matched.session_id}`)."
                metadata["session_id"] = matched.session_id
                metadata["action"] = "saved"
            elif match_type == "AMBIGUOUS":
                cand_str = ", ".join(f"'{c.title}' (ID: {c.session_id})" for c in candidates)
                response_text = f"Multiple active research sessions found: {cand_str}. Please specify which session to save."
            else:
                response_text = "No active research session to save."

        elif session_cmd.command_type == SessionCommandType.CANCEL:
            matched, candidates, match_type = research_session_manager.find_matching_session(
                session_cmd.target_identifier or "", user_id=user_id
            )
            if matched:
                cancelled = research_session_manager.cancel_session(matched.session_id)
                response_text = f"Cancelled research session: **{cancelled.title}** (ID: `{cancelled.session_id}`)."
                metadata["session_id"] = matched.session_id
                metadata["action"] = "cancelled"
            elif match_type == "AMBIGUOUS":
                cand_str = ", ".join(f"'{c.title}' (ID: {c.session_id})" for c in candidates)
                response_text = f"Multiple active research sessions found: {cand_str}. Please specify which session to cancel."
            else:
                response_text = "No active research session to cancel."

        elif session_cmd.command_type == SessionCommandType.ARCHIVE:
            matched, candidates, match_type = research_session_manager.find_matching_session(
                session_cmd.target_identifier or "", user_id=user_id
            )
            if matched:
                archived = research_session_manager.archive_session(matched.session_id)
                response_text = f"Archived research session: **{archived.title}** (ID: `{archived.session_id}`)."
                metadata["session_id"] = matched.session_id
                metadata["action"] = "archived"
            elif match_type == "AMBIGUOUS":
                cand_str = ", ".join(f"'{c.title}' (ID: {c.session_id})" for c in candidates)
                response_text = f"Multiple active research sessions found: {cand_str}. Please specify which session to archive."
            else:
                response_text = "No matching research session found to archive."

        elif session_cmd.command_type == SessionCommandType.DELETE:
            matched, candidates, match_type = research_session_manager.find_matching_session(
                session_cmd.target_identifier or "", user_id=user_id
            )
            if matched:
                sid = matched.session_id
                title = matched.title
                research_session_manager.delete_session(sid)
                response_text = f"Permanently deleted research session: **{title}** (ID: `{sid}`)."
                metadata["session_id"] = sid
                metadata["action"] = "deleted"
            elif match_type == "AMBIGUOUS":
                cand_str = ", ".join(f"'{c.title}' (ID: {c.session_id})" for c in candidates)
                response_text = f"Multiple active research sessions found: {cand_str}. Please specify which session to delete."
            else:
                response_text = "No matching research session found to delete."

        async def stream_gen():
            conversation_manager.add_message(cid, "assistant", response_text)
            total_ms = int((time.monotonic() - t_start) * 1000)
            logger.debug(
                "[SESSION CAP] cid=%s user_id=%s cmd=%s total_ms=%d",
                cid[:8], user_id, session_cmd.command_type if session_cmd else "NONE", total_ms
            )
            yield response_text

        return CapabilityResult(
            status=CapabilityStatus.SUCCESS,
            stream=stream_gen(),
            sources=[],
            metadata=metadata,
        )


session_operation_capability = ResearchSessionCapability()
