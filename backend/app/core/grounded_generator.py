from typing import List, Optional
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
    personalization_guidance: Optional[str] = None
    is_knowledge_query: bool = True


GROUNDED_SYSTEM_PROMPT = """You are Nexia, an AI assistant that answers questions using Wikipedia evidence.

Wikipedia Evidence ({page_title}):
{content}

{structured_knowledge}How to read Wikipedia evidence:
- Dates in parentheses immediately after a name are birth and death dates. Example: "Marie Curie (7 November 1867 – 4 July 1934)" means she was born on 7 November 1867 and died on 4 July 1934.
- Use all information present in the evidence to answer the question, including information in parentheses.
- When the evidence contains multiple articles (indicated by [Article 1: ...] headers), use all of them.
- When structured facts/relationships are provided, use them to support connected reasoning.
- When evidence contains conflicting dates or contradictory accounts across sources, explicitly state the differing accounts cautiously (e.g., "Available Wikipedia evidence contains conflicting accounts...") rather than arbitrarily picking one side.

Rules:
- Answer using ONLY facts present in the evidence above.
- Match the depth to the question: concise for simple facts, thorough for comparisons or complex topics.
- If the evidence genuinely does not contain the answer, say: "I couldn't find that in the Wikipedia article."
- Do not invent artificial confidence numbers or output internal system codes.
- Do not add commentary, caveats, or explanations beyond what the user asked for.
{personalization_guidance}
Question: {query}
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
            return GROUNDED_SYSTEM_PROMPT.format(
                page_title=ev.page_title or "Wikipedia Article",
                content=ev.content,
                structured_knowledge=sk_block,
                personalization_guidance=pers_guidance,
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
