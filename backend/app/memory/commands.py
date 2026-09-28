"""
Memory Command Parser for user-controlled memory operations.

Recognizes explicit REMEMBER, FORGET, and RECALL directives deterministically.
"""
from enum import Enum
import re
from typing import Optional
from pydantic import BaseModel
from app.memory.models import MemoryType
from app.memory.policy import memory_policy


class MemoryCommandType(str, Enum):
    REMEMBER = "REMEMBER"
    FORGET = "FORGET"
    RECALL = "RECALL"


class MemoryCommand(BaseModel):
    """Parsed structured representation of an explicit memory command."""
    command_type: MemoryCommandType
    raw_query: str
    target_content: str
    target_type: Optional[MemoryType] = None


class MemoryCommandParser:
    """
    Deterministic regex and rule-based parser for explicit memory commands.
    """

    # ── REMEMBER patterns ──────────────────────────────────────────────────────
    _REMEMBER_PATTERNS = [
        r"^(?:please\s+)?remember\s+(?:that\s+|:\s*|\s*)(?P<content>.+)$",
        r"^(?:please\s+)?keep\s+in\s+mind\s+(?:that\s+|:\s*|\s*)(?P<content>.+)$",
        r"^(?:please\s+)?note\s+(?:that\s+|:\s*|\s*)(?P<content>.+)$",
        r"^(?:save|store)\s+(?:in\s+memory|to\s+memory)(?:\s*:\s*|\s+that\s+|\s+)(?P<content>.+)$",
    ]

    # ── FORGET patterns ────────────────────────────────────────────────────────
    _FORGET_PATTERNS = [
        r"^(?:please\s+)?forget\s+(?:what\s+you\s+remember\s+about\s+|everything\s+about\s+|about\s+|that\s+|the\s+fact\s+that\s+|:\s*|\s*)(?P<content>.+)$",
        r"^(?:please\s+)?(?:delete|remove|clear)\s+(?:memory\s+about\s+|memory\s+of\s+|the\s+memory\s+that\s+|all\s+memories\s+about\s+|memories\s+about\s+|memory\s*:?\s*)(?P<content>.+)$",
        r"^(?:please\s+)?don['’]t\s+remember\s+(?:that\s+|about\s+)?(?P<content>.+)$",
    ]

    # ── RECALL patterns ────────────────────────────────────────────────────────
    _RECALL_PATTERNS = [
        r"^(?:what\s+do\s+you\s+remember\s+(?:about|regarding)\s+(?P<content>.+))\??$",
        r"^(?:what\s+do\s+you\s+know\s+about\s+my\s+(?P<content>preferences|profile|project|projects|settings))\??$",
        r"^(?:what\s+are\s+my\s+(?P<content>preferences|profile\s+facts|saved\s+memories|memories))\??$",
        r"^(?:what\s+memories\s+do\s+you\s+have(?:\s+about\s+(?P<content>.+))?)\??$",
        r"^(?:recall\s+(?:what\s+you\s+know\s+about\s+|what\s+you\s+remember\s+about\s+|about\s+|everything\s+about\s+|)(?P<content>.+))\??$",
        r"^(?:list\s+(?:my\s+)?(?:memories|saved\s+memories|preferences)(?:\s+about\s+(?P<content>.+))?)\??$",
        r"^(?:what\s+did\s+i\s+tell\s+you\s+about\s+(?P<content>.+))\??$",
    ]

    def parse(self, query: str) -> Optional[MemoryCommand]:
        """
        Parse user query to check if it matches an explicit memory command.
        Returns a MemoryCommand if matched, otherwise None.
        """
        cleaned = query.strip()
        cleaned_no_punct = cleaned.rstrip(".!? ")

        # 1. Test REMEMBER
        for pattern in self._REMEMBER_PATTERNS:
            match = re.match(pattern, cleaned_no_punct, re.IGNORECASE)
            if match:
                content = match.group("content").strip()
                # Clean up quotes if present
                content = content.strip("\"'")
                if content:
                    mem_type = memory_policy.extract_memory_type(content)
                    return MemoryCommand(
                        command_type=MemoryCommandType.REMEMBER,
                        raw_query=query,
                        target_content=content,
                        target_type=mem_type,
                    )

        # 2. Test FORGET
        for pattern in self._FORGET_PATTERNS:
            match = re.match(pattern, cleaned_no_punct, re.IGNORECASE)
            if match:
                content = match.group("content").strip()
                content = content.strip("\"'")
                if content:
                    return MemoryCommand(
                        command_type=MemoryCommandType.FORGET,
                        raw_query=query,
                        target_content=content,
                        target_type=None,
                    )

        # 3. Test RECALL
        for pattern in self._RECALL_PATTERNS:
            match = re.match(pattern, cleaned, re.IGNORECASE)
            if match:
                content = match.group("content")
                target_content = content.strip("\"' ") if content else "all"
                return MemoryCommand(
                    command_type=MemoryCommandType.RECALL,
                    raw_query=query,
                    target_content=target_content,
                    target_type=None,
                )

        return None


memory_command_parser = MemoryCommandParser()
