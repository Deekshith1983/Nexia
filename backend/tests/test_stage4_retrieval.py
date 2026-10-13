"""Stage 4 Unit and Integration Tests for Relationship-Aware Retrieval and Multi-Entity Extraction."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.entity_extractor import EntityExtractor
from app.schemas.retrieval import Stage4StructuredQuery

client = TestClient(app)


# --- 1. Unit Tests: Stage 4 Question Interpretation & Entity Normalization ---

def test_successor_extraction_with_honorific():
    """Verify 'successor of Shri Krishnadevaraya' extracts entity without pollution and strips honorific."""
    query: Stage4StructuredQuery = EntityExtractor.analyze_stage4_query("successor of Shri Krishnadevaraya")
    assert query.question_type == "RELATIONSHIP"
    assert query.relation is not None
    assert query.relation.type == "SUCCESSOR_OF"
    assert query.relation.direction == "OUTGOING"
    assert len(query.entities) == 1
    assert query.entities[0].text == "Shri Krishnadevaraya"
    assert query.entities[0].normalized_text == "Krishnadevaraya"
    assert query.entities[0].type == "PERSON"
    assert "Krishnadevaraya" in query.retrieval_terms
    assert "successor" in query.retrieval_terms


def test_alternative_successor_wording():
    """Verify 'Who succeeded Krishnadevaraya?' produces identical normalized relationship intent."""
    query: Stage4StructuredQuery = EntityExtractor.analyze_stage4_query("Who succeeded Krishnadevaraya?")
    assert query.question_type == "RELATIONSHIP"
    assert query.relation is not None
    assert query.relation.type == "SUCCESSOR_OF"
    assert query.entities[0].normalized_text == "Krishnadevaraya"


def test_predecessor_extraction():
    """Verify 'Who was the predecessor of Krishnadevaraya?' produces PREDECESSOR_OF."""
    query: Stage4StructuredQuery = EntityExtractor.analyze_stage4_query("Who was the predecessor of Krishnadevaraya?")
    assert query.question_type == "RELATIONSHIP"
    assert query.relation is not None
    assert query.relation.type == "PREDECESSOR_OF"
    assert query.relation.direction == "INCOMING"
    assert query.entities[0].normalized_text == "Krishnadevaraya"


def test_founder_extraction():
    """Verify 'Who founded Apple?' extracts Apple and FOUNDED_BY."""
    query: Stage4StructuredQuery = EntityExtractor.analyze_stage4_query("Who founded Apple?")
    assert query.question_type == "RELATIONSHIP"
    assert query.relation is not None
    assert query.relation.type == "FOUNDED_BY"
    assert query.entities[0].normalized_text == "Apple"


def test_multi_entity_relationship_extraction():
    """Verify 'What is the relationship between Albert Einstein and the University of Zurich?' extracts both entities."""
    query: Stage4StructuredQuery = EntityExtractor.analyze_stage4_query(
        "What is the relationship between Albert Einstein and the University of Zurich?"
    )
    assert query.question_type == "MULTI_ENTITY_RELATION"
    assert len(query.entities) == 2
    assert query.entities[0].normalized_text == "Albert Einstein"
    assert query.entities[1].normalized_text == "University of Zurich"
    assert query.entities[0].type == "PERSON"
    assert query.entities[1].type == "ORGANIZATION"


def test_multi_entity_comparison_extraction():
    """Verify 'What is the difference between India and Pakistan?' extracts both countries and comparison intent."""
    query: Stage4StructuredQuery = EntityExtractor.analyze_stage4_query(
        "What is the difference between India and Pakistan?"
    )
    assert query.question_type == "MULTI_ENTITY_COMPARISON"
    assert len(query.entities) == 2
    assert query.entities[0].normalized_text == "India"
    assert query.entities[1].normalized_text == "Pakistan"
    assert query.entities[0].type == "GPE_COUNTRY"
    assert query.entities[1].type == "GPE_COUNTRY"


def test_non_relationship_query_passthrough():
    """Verify non-relationship query like 'nearest galaxy to Milky Way' preserves single document intent."""
    query: Stage4StructuredQuery = EntityExtractor.analyze_stage4_query("nearest galaxy to Milky Way")
    assert query.question_type in ["SINGLE_DOCUMENT", "MULTI_ENTITY_RELATION"]
    assert len(query.entities) >= 1


# --- 2. Live Integration Tests: Stage 4 Multi-Document Endpoints ---

def test_api_successor_krishnadevaraya():
    """Verify Stage 4 retrieval for 'successor of Shri Krishnadevaraya' returns Achyuta Deva Raya as object."""
    response = client.get(
        "/api/v1/retrieval/multi-document?question=successor%20of%20Shri%20Krishnadevaraya&top_k_per_document=3&min_score=0.15"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "RELATIONSHIP_SUPPORTED"
    assert data["is_relationship_proven"] is True
    assert len(data["entities_identified"]) >= 1
    assert data["entities_identified"][0]["normalized_text"] == "Krishnadevaraya"
    assert len(data["relationships_discovered"]) >= 1
    assert "achyuta" in data["relationships_discovered"][0]["object"].lower()
    assert data["diagnostics"]["failure_stage"] == "NONE"


def test_api_multi_doc_einstein_zurich():
    """Verify Stage 4 multi-document retrieval for Einstein and University of Zurich."""
    response = client.get(
        "/api/v1/retrieval/multi-document?question=What%20is%20the%20relationship%20between%20Albert%20Einstein%20and%20the%20University%20of%20Zurich%3F&top_k_per_document=3&min_score=0.15"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["RELATIONSHIP_SUPPORTED", "PARTIALLY_SUPPORTED"]
    assert len(data["documents"]) >= 2
    assert data["total_passages_evaluated"] > 0


def test_api_multi_doc_india_pakistan():
    """Verify Stage 4 multi-document retrieval for India and Pakistan."""
    response = client.get(
        "/api/v1/retrieval/multi-document?question=What%20is%20the%20difference%20between%20India%20and%20Pakistan%3F&top_k_per_document=3&min_score=0.15"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["documents"]) >= 2
    assert data["total_passages_evaluated"] > 0


# --- 3. Backward Compatibility Regression Tests for Stages 1, 2, 3 ---

def test_stage1_search_regression():
    """Verify Stage 1 search endpoint is untouched and functional."""
    response = client.get("/api/v1/retrieval/search?query=Albert%20Einstein&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data["candidates"]) > 0
    assert data["candidates"][0]["title"] == "Albert Einstein"


def test_stage2_document_regression():
    """Verify Stage 2 document retrieval is untouched and functional."""
    response = client.get("/api/v1/retrieval/document?title=Albert%20Einstein")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["document"]["title"] == "Albert Einstein"
    assert len(data["document"]["sections"]) > 0


def test_stage3_passage_retrieval_regression():
    """Verify Stage 3 passage retrieval remains fully operational for prior queries."""
    response = client.get(
        "/api/v1/retrieval/passages?question=first%20prime%20minister%20of%20india&top_k=3&min_score=0.15"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["VALIDATED_SUFFICIENT_EVIDENCE", "PARTIALLY_SUPPORTED"]
    assert len(data["evidence"]) > 0
