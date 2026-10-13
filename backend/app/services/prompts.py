"""Prompt formatting and system templates for LLM answer generation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class PromptContext(BaseModel):
    """Context packaged for prompt assembly."""

    question: str
    question_type: Optional[str] = "FACTOID"
    entities: List[str] = []
    relation: Optional[str] = None
    passages: List[Dict[str, Any]] = []
    conflicts: List[str] = []
    sufficiency_rating: str = "SUFFICIENT"


SYSTEM_PROMPT = """You are Nexia, an authoritative, knowledgeable conversational assistant.
Answer the user's question directly, fluently, and conversationally in complete sentences.

CRITICAL TONE RULES:
1. NEVER say "According to the provided document", "Based on the provided passages", "The provided text defines...", "The passage states", "As mentioned in the source", or any similar meta-language referencing the prompt or text. Speak directly as an expert stating facts.
2. Ground all facts in the evidence and attach citations like [1], [2] at the end of relevant sentences.
3. Do NOT fabricate information not present in the evidence.
4. Only if the evidence passages completely lack information to answer the question, begin your reply with:
INSUFFICIENT_EVIDENCE: <concise reason>"""


def build_answer_prompt(ctx: PromptContext) -> str:
    """Format the complete prompt sent to the LLM."""
    passage_blocks: List[str] = []
    for idx, p in enumerate(ctx.passages, start=1):
        doc_title = p.get("document_title", "Unknown Source")
        section = p.get("section_title", "General")
        url = p.get("source_url", "")
        text = p.get("passage_text", "").strip()
        passage_blocks.append(
            f"[{idx}] Source: {doc_title} | Section: {section}\n"
            f"URL: {url}\n"
            f"Text: {text}"
        )

    passages_str = "\n\n".join(passage_blocks) if passage_blocks else "No passages retrieved."

    conflicts_str = (
        "\n".join(f"- {c}" for c in ctx.conflicts)
        if ctx.conflicts
        else "None identified."
    )

    entities_str = ", ".join(ctx.entities) if ctx.entities else "None detected"

    prompt_body = f"""USER QUESTION: {ctx.question}

QUESTION INTENT:
- Question Type: {ctx.question_type or 'FACTOID'}
- Entities: {entities_str}
- Relation: {ctx.relation or 'DIRECT_RELATION'}

EVIDENCE PASSAGES ({len(ctx.passages)} passages):
{passages_str}

KNOWN CONFLICTS OR UNCERTAINTIES:
{conflicts_str}

SUFFICIENCY RATING (from Nexia deterministic analysis): {ctx.sufficiency_rating}

INSTRUCTIONS:
Provide a clear, authoritative, conversational answer in complete sentences.
DO NOT use phrases such as "According to the provided document", "Based on the provided text", or "The passage states". State the answer directly and cite sources with [1], [2].
Only if the passages do not contain the answer, reply with:
INSUFFICIENT_EVIDENCE: <brief reason why the evidence is insufficient>
"""
    return prompt_body
