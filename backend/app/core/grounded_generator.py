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
    is_knowledge_query: bool = True

GROUNDED_SYSTEM_PROMPT = """You are Nexia, a conversational AI agent grounded strictly in Wikipedia factual evidence.

Factual Evidence:
Source: {source} ({page_title})
URL: {page_url}
Evidence Content:
{content}

Instructions:
1. Answer the user's question directly and concisely using ONLY the factual evidence provided above.
2. Do not invent, assume, or fabricate facts that are not explicitly supported by the evidence.
3. If the provided evidence does not contain sufficient information to answer the question, clearly state: "I couldn't find sufficient information about that in Wikipedia."
4. Do not pretend to have accessed information that was not retrieved.
5. Format your response cleanly using Markdown.

Question: {query}
Assistant:"""

INSUFFICIENT_EVIDENCE_RESPONSE = "I couldn't find sufficient information about that in Wikipedia."

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
        if gen_input.is_knowledge_query and self.has_sufficient_evidence(gen_input):
            ev = gen_input.retrieved_evidence
            return GROUNDED_SYSTEM_PROMPT.format(
                source=ev.source,
                page_title=ev.page_title or "Wikipedia Article",
                page_url=ev.page_url or "https://en.wikipedia.org",
                content=ev.content,
                query=gen_input.resolved_query,
            )

        # Fallback to standard conversational context prompt for non-knowledge / casual queries
        prompt_parts = []
        for msg in gen_input.conversation_context:
            role_label = "User" if msg.role == "user" else "Assistant"
            prompt_parts.append(f"{role_label}: {msg.content}")
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

grounded_generator = GroundedGenerator()
