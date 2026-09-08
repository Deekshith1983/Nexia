import json
import re
from typing import Dict, Optional
from pydantic import BaseModel, Field, ValidationError
from app.core.llm_interface import BaseLanguageModel

class IntentCategory:
    FACTUAL_LOOKUP = "factual_lookup"
    ENTITY_INFORMATION = "entity_information"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    SUMMARIZATION = "summarization"
    GENERAL_QUESTION = "general_question"

class QueryUnderstanding(BaseModel):
    """Structured representation of a parsed user query."""
    original_query: str
    intent: str = IntentCategory.GENERAL_QUESTION
    entity: Optional[str] = None
    information_needed: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)

ANALYSIS_PROMPT_TEMPLATE = """You are the Query Understanding module for Nexia Core.
Analyze the user query and extract its structural attributes.

Categories of intent:
- factual_lookup (asking for a specific fact, date, capital, or attribute)
- entity_information (asking who/what an entity is)
- explanation (asking how or why something works)
- comparison (asking for differences or comparisons between things)
- definition (asking for the definition of a term or concept)
- summarization (asking to summarize a topic)
- general_question (general chat or uncategorized query)

User Query: "{query}"

Respond strictly with a valid JSON object in this format (no conversational filler, markdown formatting allowed):
{{
  "intent": "<category>",
  "entity": "<explicitly mentioned entity or concept, or null>",
  "information_needed": "<specific detail requested, or null>"
}}"""

class QueryAnalyzer:
    """Component responsible for converting natural language queries into structured representations."""

    async def analyze(self, query: str, llm: Optional[BaseLanguageModel] = None) -> QueryUnderstanding:
        cleaned_query = query.strip()
        fallback = QueryUnderstanding(
            original_query=cleaned_query,
            intent=IntentCategory.GENERAL_QUESTION,
            entity=None,
            information_needed=None,
        )

        if not cleaned_query or not llm:
            return fallback

        try:
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(query=cleaned_query)
            raw_response = await llm.generate_response(prompt)
            
            # Extract JSON block if enclosed in markdown code fences
            json_str = raw_response.strip()
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Find first '{' and last '}'
                start = json_str.find("{")
                end = json_str.rfind("}")
                if start != -1 and end != -1:
                    json_str = json_str[start : end + 1]

            parsed_data = json.loads(json_str)
            
            intent = str(parsed_data.get("intent", IntentCategory.GENERAL_QUESTION)).strip().lower()
            entity = parsed_data.get("entity")
            if entity and str(entity).strip().lower() in ["null", "none", "n/a", ""]:
                entity = None
            elif entity:
                entity = str(entity).strip()

            info_needed = parsed_data.get("information_needed")
            if info_needed and str(info_needed).strip().lower() in ["null", "none", "n/a", ""]:
                info_needed = None
            elif info_needed:
                info_needed = str(info_needed).strip()

            return QueryUnderstanding(
                original_query=cleaned_query,
                intent=intent if hasattr(IntentCategory, intent.upper()) else IntentCategory.GENERAL_QUESTION,
                entity=entity,
                information_needed=info_needed,
            )

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            print(f"[INFO] Query understanding parsing failed ({e}). Utilizing safe fallback.")
            return fallback

query_analyzer = QueryAnalyzer()
