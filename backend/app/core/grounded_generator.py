from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.core.conversation_manager import Message
from app.services.wikipedia_retriever import RetrievalResult, RetrievalStatus


class GenerationInput(BaseModel):
    """Structured container holding all context required for grounded response generation."""
    original_query: str
    resolved_query: str
    conversation_context: List[Message] = Field(default_factory=list)
    retrieved_evidence: Optional[RetrievalResult] = None
    structured_knowledge: Optional[str] = None
    relevant_memories: Optional[List[str]] = None
    answer_contract: Optional[Any] = None
    personalization_guidance: Optional[str] = None
    is_knowledge_query: bool = True


def _format_answer_contract(contract: Optional[Any]) -> str:
    if not contract:
        return "No special answer structure was detected."

    answer_type = getattr(contract, "answer_type", "FACT")
    answer_type_val = getattr(answer_type, "value", str(answer_type))
    req_rel = getattr(contract, "requested_relation", None) or "none"
    target_ent = getattr(contract, "target_entity", None)
    concept_a = getattr(contract, "concept_a", None)
    concept_b = getattr(contract, "concept_b", None)
    is_neg = getattr(contract, "negative_or_exclusion", False)

    lines = [
        f"Answer type: {answer_type_val}",
        f"Requested relation: {req_rel}",
    ]
    if target_ent:
        lines.append(f"Target entity: {target_ent}")
    if concept_a:
        lines.append(f"Concept A: {concept_a}")
    if concept_b:
        lines.append(f"Concept B: {concept_b}")
    if is_neg:
        lines.append("Note: This is a negative or exclusion question.")

    return "\n".join(lines)


GROUNDED_SYSTEM_PROMPT = """You are Nexia, an accurate AI assistant that answers questions using Wikipedia evidence.

Evidence ({page_title}):
{content}

{structured_knowledge}Answer Contract:
{answer_contract}

Instructions:
- Answer using ONLY facts directly supported by the evidence above.
- Satisfy the exact relation and format requested (e.g. direct list, factual explanation, dates).
- Dates in parentheses indicate regnal periods or lifespan.
- For lists: Output a clean, organized list without meta-phrases like "The evidence mentions...".
- For distinction/why queries: Clearly explain both Concept A and Concept B from the evidence.
- If the evidence does not contain the answer, reply: "I couldn't find that in the Wikipedia article."
{personalization_guidance}
Question: {original_query}
Answer:"""

CONVERSATIONAL_SYSTEM_PROMPT = """You are Nexia, a helpful and concise conversational assistant.

Respond naturally and directly to the user's message. Follow these rules:
- Be brief and friendly — match the user's conversational tone.
- Do not list your capabilities unless the user explicitly asks what you can do.
- Do not describe yourself as an AI assistant or explain what you are, unless asked.
- Do not start with unnecessary preamble like "Of course!" or "Certainly!".
- Answer the user's actual message directly, using known user context/preferences if relevant.
{personalization_guidance}
{memories}{context}Assistant:"""

INSUFFICIENT_EVIDENCE_RESPONSE = "I couldn't find that in the Wikipedia article."


class GroundedGenerator:
    """Component responsible for assembling grounded prompts and handling evidence validation."""

    def has_sufficient_evidence(self, gen_input: GenerationInput) -> bool:
        """Check if retrieved Wikipedia evidence is valid and contains content."""
        evidence = gen_input.retrieved_evidence
        if not evidence:
            return False
        if evidence.status != RetrievalStatus.SUCCESS:
            return False
        if not evidence.content or not evidence.content.strip():
            return False
        return True

    def build_prompt(self, gen_input: GenerationInput) -> str:
        """Construct the prompt sent to the Language Model Interface."""
        pers_guidance = gen_input.personalization_guidance or ""
        if gen_input.is_knowledge_query and self.has_sufficient_evidence(gen_input):
            ev = gen_input.retrieved_evidence
            sk_block = f"{gen_input.structured_knowledge}\n" if gen_input.structured_knowledge else ""
            contract_block = _format_answer_contract(gen_input.answer_contract)
            return GROUNDED_SYSTEM_PROMPT.format(
                page_title=ev.page_title or "Wikipedia Article",
                content=ev.content,
                structured_knowledge=sk_block,
                answer_contract=contract_block,
                personalization_guidance=pers_guidance,
                original_query=gen_input.original_query,
                query=gen_input.resolved_query,
            )

        # Conversational path: build focused system prompt with recent context and relevant memories
        mem_block = ""
        if gen_input.relevant_memories:
            mem_lines = "\n".join(f"- {m}" for m in gen_input.relevant_memories)
            mem_block = f"Known User Context / Preferences:\n{mem_lines}\n\n"

        context_parts = []
        # Include last 6 messages for context (avoid bloating the prompt)
        recent = gen_input.conversation_context[-6:] if gen_input.conversation_context else []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg.content}")
        # Append the current user message
        context_parts.append(f"User: {gen_input.original_query}")
        context_parts.append("")  # blank line before "Assistant:"

        context_block = "\n".join(context_parts) + "\n" if context_parts else ""
        return CONVERSATIONAL_SYSTEM_PROMPT.format(
            personalization_guidance=pers_guidance,
            memories=mem_block,
            context=context_block,
        )


grounded_generator = GroundedGenerator()
