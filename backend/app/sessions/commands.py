"""
Session Command Parser for Stage 10 user-controlled research session operations.

Recognizes explicit task lifecycle directives deterministically:
- SAVE
- PAUSE
- RESUME
- LIST
- CANCEL
- ARCHIVE
- DELETE
- STATUS
"""
from enum import Enum
import re
from typing import Optional
from pydantic import BaseModel


class SessionCommandType(str, Enum):
    SAVE = "SAVE"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    LIST = "LIST"
    CANCEL = "CANCEL"
    ARCHIVE = "ARCHIVE"
    DELETE = "DELETE"
    STATUS = "STATUS"


class SessionCommand(BaseModel):
    """Parsed structured representation of an explicit research session command."""
    command_type: SessionCommandType
    raw_query: str
    target_identifier: Optional[str] = None


class SessionCommandParser:
    """
    Deterministic rule-based parser for research session lifecycle commands.
    """

    # ── SAVE patterns ──────────────────────────────────────────────────────────
    _SAVE_PATTERNS = [
        r"^(?:please\s+)?(?:save|persist)\s+(?:this\s+|my\s+|the\s+)?(?:research|research\s+session|task|session)(?:\s+(?:as|titled)\s+(?P<target>.+))?$",
    ]

    # ── PAUSE patterns ─────────────────────────────────────────────────────────
    _PAUSE_PATTERNS = [
        r"^(?:please\s+)?(?:pause|suspend|freeze)\s+(?:this\s+|my\s+|the\s+)?(?:research|research\s+session|task|session)(?:\s+(?:on|about|for)?\s*(?P<target>.+))?$",
    ]

    # ── RESUME patterns ────────────────────────────────────────────────────────
    _RESUME_PATTERNS = [
        r"^(?:please\s+)?(?:continue|resume|restart)\s+(?:my\s+|the\s+|this\s+)?(?:research|research\s+session|task|session|investigation)?(?:\s+(?:on|about|for|with|regarding)?\s*(?P<target>.+))?$",
        r"^(?:continue|resume)\s+(?P<target>.+?)\s+(?:research|session|task)$",
    ]

    # ── LIST patterns ──────────────────────────────────────────────────────────
    _LIST_PATTERNS = [
        r"^(?:please\s+)?(?:show|list|display|get)\s+(?:all\s+|my\s+|the\s+)?(?:active\s+|saved\s+|ongoing\s+)?(?:research|research\s+sessions|sessions|tasks)$",
        r"^(?:what\s+(?:are\s+my\s+active|active)\s+(?:research\s+sessions|research|tasks))\??$",
    ]

    # ── CANCEL patterns ────────────────────────────────────────────────────────
    _CANCEL_PATTERNS = [
        r"^(?:please\s+)?(?:cancel|abort|stop)\s+(?:this\s+|my\s+|the\s+)?(?:research|research\s+session|task|session)(?:\s+(?:on|about|for)?\s*(?P<target>.+))?$",
    ]

    # ── ARCHIVE patterns ───────────────────────────────────────────────────────
    _ARCHIVE_PATTERNS = [
        r"^(?:please\s+)?(?:archive)\s+(?:this\s+|my\s+|the\s+)?(?:research|research\s+session|task|session)(?:\s+(?:on|about|for)?\s*(?P<target>.+))?$",
    ]

    # ── DELETE patterns ────────────────────────────────────────────────────────
    _DELETE_PATTERNS = [
        r"^(?:please\s+)?(?:delete|remove|erase)\s+(?:this\s+|my\s+|the\s+)?(?:research|research\s+session|task|session)(?:\s+(?:on|about|for|id)?\s*(?P<target>.+))?$",
    ]

    # ── STATUS patterns ────────────────────────────────────────────────────────
    _STATUS_PATTERNS = [
        r"^(?:what\s+is\s+the\s+status\s+of\s+(?:my\s+|this\s+|the\s+)?(?:research|task|session))(?:\s+(?:on|about|for)?\s*(?P<target>.+))?\??$",
        r"^(?:research|task|session)\s+status(?:\s+(?:for|on)?\s*(?P<target>.+))?$",
    ]

    def parse(self, query: str) -> Optional[SessionCommand]:
        """
        Parse user query to check if it matches an explicit research session command.
        Returns a SessionCommand if matched, otherwise None.
        """
        cleaned = query.strip()
        cleaned_no_punct = cleaned.rstrip(".!? ")

        # 1. LIST
        for pattern in self._LIST_PATTERNS:
            match = re.match(pattern, cleaned, re.IGNORECASE)
            if match:
                return SessionCommand(
                    command_type=SessionCommandType.LIST,
                    raw_query=query,
                    target_identifier=None,
                )

        # 2. STATUS
        for pattern in self._STATUS_PATTERNS:
            match = re.match(pattern, cleaned, re.IGNORECASE)
            if match:
                target = match.groupdict().get("target")
                return SessionCommand(
                    command_type=SessionCommandType.STATUS,
                    raw_query=query,
                    target_identifier=target.strip("\"' ") if target else None,
                )

        # 3. SAVE
        for pattern in self._SAVE_PATTERNS:
            match = re.match(pattern, cleaned_no_punct, re.IGNORECASE)
            if match:
                target = match.groupdict().get("target")
                return SessionCommand(
                    command_type=SessionCommandType.SAVE,
                    raw_query=query,
                    target_identifier=target.strip("\"' ") if target else None,
                )

        # 4. PAUSE
        for pattern in self._PAUSE_PATTERNS:
            match = re.match(pattern, cleaned_no_punct, re.IGNORECASE)
            if match:
                target = match.groupdict().get("target")
                return SessionCommand(
                    command_type=SessionCommandType.PAUSE,
                    raw_query=query,
                    target_identifier=target.strip("\"' ") if target else None,
                )

        # 5. CANCEL
        for pattern in self._CANCEL_PATTERNS:
            match = re.match(pattern, cleaned_no_punct, re.IGNORECASE)
            if match:
                target = match.groupdict().get("target")
                return SessionCommand(
                    command_type=SessionCommandType.CANCEL,
                    raw_query=query,
                    target_identifier=target.strip("\"' ") if target else None,
                )

        # 6. ARCHIVE
        for pattern in self._ARCHIVE_PATTERNS:
            match = re.match(pattern, cleaned_no_punct, re.IGNORECASE)
            if match:
                target = match.groupdict().get("target")
                return SessionCommand(
                    command_type=SessionCommandType.ARCHIVE,
                    raw_query=query,
                    target_identifier=target.strip("\"' ") if target else None,
                )

        # 7. DELETE
        for pattern in self._DELETE_PATTERNS:
            match = re.match(pattern, cleaned_no_punct, re.IGNORECASE)
            if match:
                target = match.groupdict().get("target")
                return SessionCommand(
                    command_type=SessionCommandType.DELETE,
                    raw_query=query,
                    target_identifier=target.strip("\"' ") if target else None,
                )

        # 8. RESUME
        for pattern in self._RESUME_PATTERNS:
            match = re.match(pattern, cleaned_no_punct, re.IGNORECASE)
            if match:
                target = match.groupdict().get("target")
                return SessionCommand(
                    command_type=SessionCommandType.RESUME,
                    raw_query=query,
                    target_identifier=target.strip("\"' ") if target else None,
                )

        return None


session_command_parser = SessionCommandParser()
