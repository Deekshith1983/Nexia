"""Schemas for Nexia Advanced Query Constraints, Multi-Document Retrieval, and Evidence Validation."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- Stage 1 & 2: Candidates & Structured Documents ---

class SearchCandidate(BaseModel):
    """A search result candidate from Wikipedia."""
    title: str
    page_id: int
    snippet: str = ""
    url: str
    word_count: Optional[int] = None
    selection_score: Optional[float] = Field(default=None, description="Heuristic candidate relevance score")


class SearchResponse(BaseModel):
    """Response schema for Wikipedia search candidates."""
    query: str
    total_hits: int
    candidates: List[SearchCandidate]


class Paragraph(BaseModel):
    """Individually addressable, traceable paragraph within an article section."""
    id: str = Field(..., description="Unique addressable ID: doc_{page_id}_sec_{sec_idx}_p_{p_idx}")
    document_title: str = Field(..., description="Parent document title for source traceability")
    section_title: str = Field(..., description="Section title where this paragraph belongs")
    index: int = Field(..., description="0-based index of this paragraph within the section")
    text: str = Field(..., description="Complete, unmodified text content of the paragraph")
    word_count: int
    char_count: int


class Section(BaseModel):
    """Structured section of an article preserving heading hierarchy."""
    index: int = Field(..., description="0-based sequence index in document order")
    title: str = Field(..., description="Heading title of the section")
    level: int = Field(default=2, description="Heading level: 2 for ==, 3 for ===, 4 for ====")
    parent_section: Optional[str] = Field(default=None, description="Title of parent section if subsection")
    paragraphs: List[Paragraph] = Field(default_factory=list)
    subsections: List[str] = Field(default_factory=list, description="Titles of nested child subsections")


class DocumentMetadata(BaseModel):
    """Metadata describing the retrieved article."""
    page_id: int
    url: str
    language: str = "en"
    total_sections: int
    total_paragraphs: int
    total_words: int
    categories: List[str] = Field(default_factory=list)


class Document(BaseModel):
    """Structured internal representation of a full Wikipedia article."""
    title: str
    page_id: int
    url: str
    introduction: str = Field(..., description="Lead section before the first heading")
    sections: List[Section] = Field(default_factory=list)
    metadata: DocumentMetadata


class DocumentRetrievalResponse(BaseModel):
    """Response schema for full document retrieval inspection."""
    query: str
    status: str = Field(..., description="'success', 'not_found', or 'error'")
    selected_candidate: Optional[SearchCandidate] = None
    document: Optional[Document] = None
    other_candidates: List[SearchCandidate] = Field(default_factory=list)
    error_message: Optional[str] = None


# --- Structured Query Representation & Constraints ---

class GeographicConstraint(BaseModel):
    """Geographic constraint entity extracted from query."""
    country: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None


class QueryConstraints(BaseModel):
    """Strict required vs optional constraints that must be satisfied for valid evidence."""
    required_entities: List[str] = Field(default_factory=list, description="Entities that MUST be satisfied")
    optional_terms: List[str] = Field(default_factory=list, description="Supporting terms that assist ranking")
    excluded_contexts: List[str] = Field(default_factory=list, description="Incompatible contexts that trigger penalties")
    geographic: Optional[GeographicConstraint] = None
    temporal: Optional[str] = None       # 'current', 'first', 'third', 'historical'
    superlative: Optional[str] = None    # 'nearest', 'largest', 'oldest', 'longest', 'smallest'
    superlative_property: Optional[str] = None # 'distance', 'size', 'age'
    superlative_direction: Optional[str] = None # 'MIN', 'MAX'
    expected_answer_type: str = "PERSON" # PERSON, LOCATION, DATE_TIME, NUMERIC, DEFINITION, PROCESS, ENTITY


class StructuredQueryRepresentation(BaseModel):
    """Universal Structured Query Intent Schema."""
    raw_question: str
    question_type: str                  # PERSON_IDENTIFICATION, ATTRIBUTE_OR_TITLE_LOOKUP, LOCATION, DATE_TIME, DEFINITION, RELATIONSHIP_SUPERLATIVE, PROCESS_MECHANISM, CAUSE_REASON, NUMERIC_QUANTITY, COMPARISON, GENERAL
    subject_entity: str                 # Primary subject: "India", "Milky Way", "Hoysala Kingdom", "Albert Einstein"
    secondary_entities: List[str] = Field(default_factory=list) # e.g. ["galaxy", "country"]
    relation: str                       # "Father of the Nation", "nearest_distance", "prime_minister", "birthplace", "established"
    constraints: QueryConstraints
    retrieval_terms: List[str]          # Cleaned tokens for candidate search
    expanded_semantic_cues: List[str]   # Relational cues & domain synonyms for reranking


# --- Evidence Passages & Multi-Factor Reranking ---

class Passage(BaseModel):
    """An independently retrievable passage preserving strict source traceability."""
    passage_id: str = Field(..., description="Unique deterministic passage ID")
    document_title: str = Field(..., description="Source document title")
    document_page_id: int = Field(default=0, description="Source document page ID")
    section_title: str = Field(..., description="Section title")
    section_level: int = Field(default=2, description="Heading level of the section")
    paragraph_id: str = Field(..., description="Original paragraph ID")
    text: str = Field(..., description="Complete text of the passage")
    word_count: int


class ScoreBreakdown(BaseModel):
    """Detailed multi-factor relevance and constraint score decomposition."""
    bm25_score: float
    entity_match_score: float
    relation_match_score: float
    question_type_score: float
    context_consistency_score: float
    final_score: float


class EvidenceValidationDetails(BaseModel):
    """Detailed validation gate evaluation for individual evidence passages."""
    entity_match: bool
    relation_match: bool
    context_match: bool
    answer_type_match: bool
    required_entities_present: bool
    geographic_consistency: bool


class EvidencePassage(BaseModel):
    """A retrieved passage scored, ranked, and diagnosed against a structured question intent."""
    rank: int = Field(..., description="Rank position (1-based)")
    passage_id: str
    document_title: str
    section_title: str
    section_level: int
    paragraph_id: str
    text: str
    score: float = Field(..., description="Multi-factor relevance score (0.0 to 1.0)")
    is_above_threshold: bool = Field(..., description="Whether score meets minimum relevance threshold")
    matched_terms: List[str] = Field(default_factory=list)
    unmatched_terms: List[str] = Field(default_factory=list)
    score_breakdown: Optional[ScoreBreakdown] = None
    validation_details: Optional[EvidenceValidationDetails] = None


class PassageRetrievalRequest(BaseModel):
    """Request schema for question-aware passage retrieval."""
    question: str = Field(..., description="User question to answer")
    document: Optional[str] = Field(
        default=None,
        description="Target Wikipedia article title. If omitted, multi-document candidate search is used."
    )
    algorithm: str = Field(
        default="bm25",
        description="Candidate retrieval scoring algorithm: 'bm25' (recommended) or 'tfidf'"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top passages to return")
    min_score: float = Field(default=0.15, ge=0.0, le=1.0, description="Minimum relevance threshold")


class PassageRetrievalResponse(BaseModel):
    """Response schema for question-aware passage retrieval."""
    question: str
    document_title: str
    documents_searched: List[str] = Field(
        default_factory=list,
        description="All candidate Wikipedia articles retrieved and evaluated"
    )
    algorithm_used: str = Field(default="bm25", description="Candidate retrieval algorithm used")
    structured_intent: StructuredQueryRepresentation
    status: str = Field(
        ...,
        description="'VALIDATED_SUFFICIENT_EVIDENCE', 'PARTIALLY_SUPPORTED', 'INSUFFICIENT_EVIDENCE', 'DOCUMENT_NOT_FOUND', or 'ERROR'"
    )
    is_evidence_sufficient: bool = Field(..., description="Whether retrieved evidence satisfies question intent")
    validation_reason: str = Field(..., description="Detailed explanation of the evidence validation outcome")
    total_passages_evaluated: int
    threshold_used: float
    candidate_documents_considered: List[SearchCandidate] = Field(
        default_factory=list,
        description="All candidate documents discovered during document search"
    )
    evidence: List[EvidencePassage] = Field(
        default_factory=list,
        description="Ranked top-k passages meeting relevance & validation criteria"
    )
    all_ranked_candidates: List[EvidencePassage] = Field(
        default_factory=list,
        description="All candidate passages for diagnostic inspection"
    )
    error_message: Optional[str] = None
