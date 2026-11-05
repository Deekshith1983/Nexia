"""Prompt formatting and system templates for LLM answer generation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import re


class PromptContext(BaseModel):
    """Context packaged for prompt assembly."""

    question: str
    question_type: Optional[str] = "FACTOID"
    entities: List[str] = []
    relation: Optional[str] = None
    passages: List[Dict[str, Any]] = []
    conflicts: List[str] = []
    sufficiency_rating: str = "SUFFICIENT"
    missing_aspects: List[str] = []
    subqueries: List[str] = []
    conversation_history: List[Dict[str, str]] = []  # List of {"role": "user"|"assistant", "content": "..."}
    conversation_summary: Optional[str] = None
    memories: List[str] = []


SYSTEM_PROMPT = """You are Nexia, an authoritative, knowledgeable conversational assistant.
Answer the user's question directly, fluently, and completely.

CRITICAL TONE RULES:
1. NEVER use meta-language. NEVER say "According to the provided document", "Based on the provided passages", "as documented in the provided evidence", "as per evidence", "as per Wikipedia", "as for the given evidence", "as presented in the evidence", "in the provided evidence", "The provided text defines...", "The passage states", or ANY mention of evidence, passages, documents, or Wikipedia. Speak directly as an authoritative expert stating facts.
2. Ground all facts in the evidence and attach citations like [1], [2] at the end of relevant sentences.
3. Do NOT fabricate information not present in the evidence.
4. LISTING RULE: If the user asks you to list, enumerate, or name items, output the COMPLETE list — do not stop early, do not summarize, do not say "and others". Every item from the evidence must appear. Begin directly with a heading (e.g. "Presidents of India:") or start the numbered list immediately. Do NOT write introductory filler like "Here is a list of... as documented in the provided evidence".
5. COMPLETENESS RULE: Never end a sentence mid-way. Always complete every sentence you start.
6. ANTI-REPETITION: Do NOT repeat the same fact, sentence structure, or general background multiple times. State each distinct point clearly once.
7. STRICT EVIDENCE BOUNDARY & ANTI-ELABORATION: State ONLY facts explicitly contained in the provided evidence. NEVER extrapolate, hypothesize, or introduce unmentioned external connections (such as unmentioned civilizations, Egyptian or Roman engineering, or architectural styles) not found in the text. If any requested aspect is not documented in the evidence, state clearly that it is not documented rather than inventing a bridge.
8. Only if the evidence passages completely lack information to answer the question, begin your reply with:
INSUFFICIENT_EVIDENCE: <concise reason>"""


from app.retrieval.nlp_query_parser import NLPQueryParser


def _is_list_question(question: str) -> bool:
    """Detect if the question is asking for an enumeration or listing."""
    return NLPQueryParser.is_list_question(question)


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

    # Inject a list-specific instruction when the question is an enumeration
    list_instruction = ""
    if _is_list_question(ctx.question):
        list_instruction = (
            "\nSPECIAL INSTRUCTION: The user is asking for a COMPLETE LIST. "
            "Output every single item found in the evidence as a numbered list. "
            "Do NOT say 'and others', 'etc.', or truncate the list in any way. "
            "Every item must appear. Complete every sentence fully before stopping.\n"
        )

    # Inject multi-part instruction for complex queries with multiple aspects
    multi_part_instruction = ""
    if any(m in ctx.question.lower() for m in [", and ", ";", " and who ", " and what ", " and how ", " and when ", " and where ", "including", "such as", "specifically"]):
        multi_part_instruction = (
            "\nSPECIAL INSTRUCTION: The user has asked a multi-part question with several distinct requirements. "
            "Address EVERY requested aspect directly and thoroughly using ONLY the provided evidence passages:\n"
            "- Do not repeat background information or fill space with generic descriptions.\n"
            "- Address each requested aspect (e.g. roles, specific goods/commodities, mechanics, evidence sources) with specific named items and facts.\n"
            "- STRICT EVIDENCE BOUNDARY: Do NOT extrapolate, speculate, or introduce unmentioned outside historical/engineering claims (such as unmentioned Egyptian or Roman engineering or architectural orders) not contained in the text.\n"
            "- ABSTENTION: If the evidence passages do not contain information for any specific requested subquestion, state clearly that the available evidence does not document that aspect. Do NOT invent a plausible connection.\n"
            "- Attach inline citations like [1], [2] at the end of each factual statement.\n"
            "- Complete all sentences cleanly.\n"
        )

    # Inject missing evidence & abstention instructions if any aspect lacks evidence
    missing_notice = ""
    if ctx.missing_aspects:
        missing_fmt = "\n".join(f"- {a}" for a in ctx.missing_aspects)
        missing_notice = (
            f"\nMISSING EVIDENCE & MANDATORY ABSTENTION:\n"
            f"The provided evidence passages DO NOT contain documentation for the following requested aspect(s):\n"
            f"{missing_fmt}\n"
            f"You MUST explicitly state in your answer that the available evidence does not document these aspects (or that they remain historically uncertain). "
            f"Under NO circumstances should you fabricate, speculate, or draw upon outside knowledge to invent an explanation for them.\n"
        )

    # Inject multi-turn conversation context when present
    history_str = ""
    if ctx.conversation_summary:
        history_str += f"\nPREVIOUS CONVERSATION SUMMARY:\n{ctx.conversation_summary}\n"
    if ctx.conversation_history:
        history_lines = []
        for turn in ctx.conversation_history:
            role_label = "User" if turn.get("role") == "user" else "Assistant"
            history_lines.append(f"{role_label}: {turn.get('content', '')}")
        history_str += "\nRECENT CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n"

    # Inject relevant conversational memories when present
    memory_str = ""
    if ctx.memories:
        memory_str = "\nRELEVANT CONVERSATION MEMORIES (FOR CONTEXT RESOLUTION ONLY):\n" + "\n".join(f"- {m}" for m in ctx.memories) + "\n"

    prompt_body = f"""{history_str}{memory_str}USER QUESTION: {ctx.question}

QUESTION INTENT:
- Question Type: {ctx.question_type or 'FACTOID'}
- Entities: {entities_str}
- Relation: {ctx.relation or 'DIRECT_RELATION'}

EVIDENCE PASSAGES ({len(ctx.passages)} passages):
{passages_str}

KNOWN CONFLICTS OR UNCERTAINTIES:
{conflicts_str}

SUFFICIENCY RATING (from Nexia deterministic analysis): {ctx.sufficiency_rating}
{list_instruction}
{multi_part_instruction}
{missing_notice}
INSTRUCTIONS:
Provide a clear, authoritative, complete answer.
1. CITATION REQUIREMENT: Every factual claim must be followed immediately by a citation [k] referencing the specific passage that supports it.
2. DO NOT use meta-language like "According to the provided document", "Based on the provided text", "as documented in the provided evidence", "as per evidence", "as per Wikipedia", or "The passage states". State the answer directly.
3. If the question asks for a list: output a numbered list with EVERY item — do not summarize or truncate. Start directly with the list or a clean heading (e.g. "Presidents of India:").
4. If the question is factual: answer in complete sentences.
5. ABSTENTION: If evidence for any requested sub-aspect is not in the passages, state clearly: "The available evidence does not document [aspect]."
6. Only if the passages completely lack information to answer the question, reply with:
INSUFFICIENT_EVIDENCE: <brief reason why the evidence is insufficient>
"""
    return prompt_body
