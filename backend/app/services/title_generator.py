"""Dedicated Conversation Title Generator using local Gemma 3 4B via Ollama.

Generates concise, descriptive conversation titles (<= 5 words) using both
the user's first question and the assistant's finalized first answer.
"""

import asyncio
import logging
import re
from typing import Optional, Set
from uuid import UUID, uuid4

from app.core.config import settings
from app.services.citation_manager import strip_inline_citations
from app.services.conversation_manager import ConversationManager
from app.services.llm_adapter import OllamaLLMAdapter

logger = logging.getLogger("nexia.title_generator")

TITLE_SYSTEM_PROMPT = (
    "You are an expert conversation titling assistant. "
    "Generate a concise, descriptive title for a conversation based on the user's first question and first assistant response. "
    "Rules:\n"
    "- Maximum 5 words.\n"
    "- Output ONLY the plain text title.\n"
    "- Do NOT include quotation marks, markdown, punctuation at the end, or prefixes like 'Title:'.\n"
    "- Capture the primary subject and intent accurately."
)

TITLE_PROMPT_TEMPLATE = """User question: {question}
Assistant response summary: {answer_summary}

Title:"""

# Common conversational stop words for keyword fallback
QUESTION_STOPWORDS: Set[str] = {
    "what", "is", "the", "who", "was", "were", "where", "when", "why", "how",
    "did", "does", "do", "can", "could", "would", "should", "tell", "me", "about",
    "explain", "describe", "give", "a", "an", "in", "of", "to", "for", "on", "and",
    "or", "by", "from", "at", "with", "which", "are", "been", "being", "have", "has",
    "had", "i", "you", "we", "they", "it", "its", "please", "summary", "brief",
}

# Trailing prepositions and conjunctions that should not end a truncated title
TRAILING_TRIM_WORDS: Set[str] = {
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "into",
    "through", "over", "before", "after", "and", "or", "but", "the", "a", "an",
}


class ConversationTitleGenerator:
    """Generates concise, meaningful titles for new conversations using local Gemma 3 4B."""

    def __init__(self, llm_adapter: Optional[OllamaLLMAdapter] = None):
        self.llm_adapter = llm_adapter or OllamaLLMAdapter()

    @classmethod
    def clean_and_validate_title(cls, raw_title: str, question: str) -> str:
        """Sanitize, format, and validate raw LLM title output.
        
        Guarantees:
        - <= 5 words
        - No quotes, prefixes, markdown, or trailing punctuation
        - Proper title casing while preserving uppercase acronyms
        - Safe keyword fallback if output is empty or invalid
        """
        if not raw_title or not raw_title.strip():
            return cls.fallback_title_from_question(question)

        text = raw_title.strip()

        # 1. Strip preambles and common prefixes like 'Title:', 'Here is a title:', etc.
        text = re.sub(
            r"^(here\s+(is|are)\s+[^:]*:|title\s*:|topic\s*:|subject\s*:|conversation\s*:|chat\s*:|heading\s*:)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # 2. Strip markdown bold/italics and backticks
        text = text.replace("**", "").replace("*", "").replace("__", "").replace("_", "").replace("`", "")

        # 3. Strip surrounding quotation marks (single, double, curly)
        text = text.strip("\"'“”‘’«»`")

        # 4. Remove leading/trailing punctuation and whitespace
        text = text.strip(" \t\n\r.!?,:;~-—#")

        # Check for refusal / failure patterns
        lower_text = text.lower()
        if any(marker in lower_text for marker in ("i cannot", "i am sorry", "as an ai", "insufficient evidence", "no title")):
            return cls.fallback_title_from_question(question)

        # Split into words
        words = text.split()
        if not words:
            return cls.fallback_title_from_question(question)

        # 5. Enforce 5-word maximum
        if len(words) > 5:
            words = words[:5]
            # Trim trailing dangling prepositions/conjunctions
            while len(words) > 2 and words[-1].lower() in TRAILING_TRIM_WORDS:
                words.pop()

        # 6. Apply title casing while preserving acronyms/Roman numerals (e.g. NASA, DNA, AI, II, III)
        formatted_words = []
        for w in words:
            # Strip internal punctuation from ends of token if any
            clean_w = w.strip(".,;:!?()[]{}")
            if not clean_w:
                continue
            if clean_w.isupper() and len(clean_w) > 1:
                formatted_words.append(clean_w)
            elif "-" in clean_w:
                # Handle hyphenated words (e.g. Indo-Greek)
                parts = [p.capitalize() if not (p.isupper() and len(p) > 1) else p for p in clean_w.split("-")]
                formatted_words.append("-".join(parts))
            else:
                formatted_words.append(clean_w.capitalize())

        if not formatted_words:
            return cls.fallback_title_from_question(question)

        final_title = " ".join(formatted_words).strip(" \t.,;:!?-—")
        return final_title if final_title else cls.fallback_title_from_question(question)

    @classmethod
    def fallback_title_from_question(cls, question: str) -> str:
        """Extract a meaningful title (<= 5 words) from question keywords."""
        if not question or not question.strip():
            return "General Query"

        # Clean question of special characters
        cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", question)
        tokens = cleaned.split()

        # Filter out stopwords
        meaningful = [t for t in tokens if t.lower() not in QUESTION_STOPWORDS]

        target_tokens = meaningful if meaningful else tokens
        chosen = target_tokens[:5]

        # Trim dangling prepositions/conjunctions if more than 2 words
        while len(chosen) > 2 and chosen[-1].lower() in TRAILING_TRIM_WORDS:
            chosen.pop()

        formatted = []
        for w in chosen:
            if w.isupper() and len(w) > 1:
                formatted.append(w)
            elif "-" in w:
                parts = [p.capitalize() if not (p.isupper() and len(p) > 1) else p for p in w.split("-")]
                formatted.append("-".join(parts))
            else:
                formatted.append(w.capitalize())

        return " ".join(formatted).strip() or "General Query"

    async def generate_title(self, question: str, answer: str) -> str:
        """Generate a concise title using local Gemma 3 4B with keyword fallback."""
        clean_q = question.strip()

        # Prepare concise answer summary
        clean_ans = strip_inline_citations(answer or "")
        # Remove markdown hyperlinks [Title](url) -> Title
        clean_ans = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_ans).strip()
        sentences = re.split(r"(?<=[.!?])\s+", clean_ans)
        summary = " ".join(sentences[:2])[:250].strip()

        prompt = TITLE_PROMPT_TEMPLATE.format(
            question=clean_q,
            answer_summary=summary if summary else "None provided",
        )

        try:
            from app.services.response_controller import Priority, ResponseController
            resp_ctrl = ResponseController.get_instance()
            title_req_id = f"title_{uuid4()}"

            async def _run():
                return await self.llm_adapter.generate_text(
                    prompt=prompt,
                    system_prompt=TITLE_SYSTEM_PROMPT,
                    num_predict=24,
                    temperature=0.3,
                    timeout=12.0,
                )

            raw_title = await resp_ctrl.execute_bounded(
                request_id=title_req_id,
                coro_func=_run,
                priority=Priority.BACKGROUND,
                timeout=15.0,
            )
            title = self.clean_and_validate_title(raw_title, clean_q)
            logger.info("Generated title for question '%s': '%s'", clean_q[:40], title)
            return title
        except Exception as e:
            logger.warning("LLM title generation failed, using keyword fallback: %s", e)
            return self.fallback_title_from_question(clean_q)

    async def generate_and_update_title(
        self,
        conversation_id: UUID,
        first_question: str,
        first_answer: str,
        conversation_manager: ConversationManager,
    ) -> Optional[str]:
        """Atomically generate and persist a title for eligible conversations."""
        try:
            eligible = await conversation_manager.is_eligible_for_title_generation(conversation_id)
            if not eligible:
                logger.debug(
                    "Conversation %s is not eligible for automatic title generation.",
                    conversation_id,
                )
                return None

            title = await self.generate_title(first_question, first_answer)
            if not title:
                return None

            updated = await conversation_manager.update_title_atomic(conversation_id, title)
            if updated:
                logger.info(
                    "Updated conversation %s title to '%s'",
                    conversation_id,
                    title,
                )
                return title
            else:
                logger.debug(
                    "Conversation %s title was already updated or renamed concurrently.",
                    conversation_id,
                )
                return None

        except Exception as e:
            logger.warning("Failed to generate and update title for %s: %s", conversation_id, e)
            return None

