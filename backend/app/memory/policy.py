"""
Memory Policy for Nexia Stage 4.

Determines:
  1. Selectivity: whether a candidate statement is appropriate for long-term retention vs transient.
  2. Classification: mapping statements to structured MemoryType categories.
  3. Supersession / Conflict detection: identifying older active memories replaced by newer statements.
"""
import re
from typing import List, Optional, Set
from app.memory.models import Memory, MemoryType


# Regex patterns matching transient, ephemeral, or momentary user states
_TRANSIENT_PATTERNS = [
    r"\b(?:today|right now|at the moment|currently today|this morning|this afternoon|this evening|tonight)\b",
    r"\b(?:i(?:'m| am)\s+(?:tired|sleepy|hungry|thirsty|sick|bored|busy|free|rushing|leaving|going out|back))\b",
    r"\b(?:just\s+(?:woke up|ate|finished lunch|drinking coffee))\b",
    r"\b(?:weather is|it(?:'s| is)\s+(?:hot|cold|raining|sunny|cloudy))\b",
    r"\b(?:feeling\s+(?:good|bad|sad|happy|tired|down))\b",
]

# Question indicators (questions should never become permanent memory)
_QUESTION_STARTS = (
    "what", "who", "where", "when", "why", "how", "can you", "could you",
    "will you", "tell me", "is there", "are there", "do you", "does", "did"
)


class MemoryPolicy:
    """
    Policy engine governing memory lifecycle, storage eligibility, and supersession.
    """

    def should_store(self, text: str) -> bool:
        """
        Evaluate whether candidate text should be stored as long-term persistent memory.
        Rejects transient states, questions, and ephemeral conversational noise.
        """
        cleaned = text.strip().lower()
        if not cleaned:
            return False

        # 1. Questions must not be stored as memory
        if cleaned.endswith("?") or any(cleaned.startswith(q) for q in _QUESTION_STARTS):
            return False

        # 2. Check transient patterns
        for pattern in _TRANSIENT_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return False

        # 3. Text must have meaningful semantic content
        words = [w for w in re.sub(r"[^\w\s]", " ", cleaned).split() if len(w) > 2]
        if len(words) < 2:
            return False

        return True

    def extract_memory_type(self, text: str) -> MemoryType:
        """
        Classify text into a structured MemoryType.
        """
        lower = text.lower()

        # Interaction preferences
        if any(w in lower for w in [
            "concise", "detailed", "brief", "bullet points", "tone",
            "formatting", "style", "length", "answers", "responses", "explain simply"
        ]) and any(w in lower for w in ["prefer", "like", "want", "always", "give me", "provide"]):
            return MemoryType.INTERACTION_PREFERENCE

        # Project context
        if any(w in lower for w in [
            "project", "building", "nexia", "codebase", "repository", "app", "application",
            "stack", "framework", "backend", "frontend", "developing", "architecture"
        ]):
            return MemoryType.PROJECT_CONTEXT

        # Profile facts
        if any(w in lower for w in [
            "my name is", "i am a", "i work as", "my role", "software engineer",
            "developer", "student", "live in", "from", "location", "years old"
        ]):
            return MemoryType.PROFILE_FACT

        # General preference
        if any(w in lower for w in ["prefer", "favorite", "favourite", "like", "dislike", "hate"]):
            return MemoryType.PREFERENCE

        return MemoryType.LEARNED_CONTEXT

    def find_superseded(
        self,
        new_content: str,
        new_type: MemoryType,
        active_memories: List[Memory],
    ) -> List[Memory]:
        """
        Detect existing active memories that are superseded or contradicted by the new memory.
        """
        if not active_memories:
            return []

        superseded: List[Memory] = []
        new_lower = new_content.lower()

        # Semantic conflict domains
        domains = {
            "answer_length": {"concise", "detailed", "brief", "long", "short", "elaborate", "depth"},
            "programming_language": {"python", "java", "javascript", "typescript", "c++", "rust", "go", "ruby", "c#", "kotlin"},
            "project_name": {"building", "project called", "project named", "my project"},
            "user_role": {"software engineer", "developer", "student", "architect", "manager", "designer"},
        }

        # Identify which domain the new memory touches
        matched_domains = set()
        for domain_name, keywords in domains.items():
            if any(k in new_lower for k in keywords):
                matched_domains.add(domain_name)

        for mem in active_memories:
            # Same type or matching domain
            mem_lower = mem.content.lower()

            # 1. Check if same semantic conflict domain
            for domain_name in matched_domains:
                domain_kws = domains[domain_name]
                if any(k in mem_lower for k in domain_kws):
                    superseded.append(mem)
                    break
            else:
                # 2. Check high similarity or direct topic overlap within same memory type
                if mem.type == new_type:
                    # Token overlap ratio
                    mem_words = set(re.findall(r"\w+", mem_lower))
                    new_words = set(re.findall(r"\w+", new_lower))
                    overlap = mem_words.intersection(new_words)
                    if len(overlap) >= 3 and len(overlap) / max(len(mem_words), 1) > 0.5:
                        superseded.append(mem)

        return superseded


memory_policy = MemoryPolicy()
