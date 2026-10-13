"""Stage 5 Unit and Integration Tests for Relationship Discovery & Evidence Linking."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.predicate_registry import PredicateRegistry
from app.schemas.relationships import RelationshipDiscoveryResponse

client = TestClient(app)


# --- 1. Unit Tests: Predicate Registry & Inverses ---

def test_predicate_registry_lookup():
    """Verify predicate registry finds canonical definitions and inverses."""
    studied_def = PredicateRegistry.get_predicate_def("STUDIED_AT")
    assert studied_def is not None
    assert studied_def.category == "ACADEMIC_INSTITUTIONAL"
    assert studied_def.inverse_predicate == "ALMA_MATER_OF"

    successor_def = PredicateRegistry.get_predicate_def("SUCCESSOR_OF")
    assert successor_def is not None
    assert successor_def.inverse_predicate == "PREDECESSOR_OF"

    founded_def = PredicateRegistry.get_predicate_def("FOUNDED_BY")
    assert founded_def is not None
    assert founded_def.inverse_predicate == "FOUNDER_OF"


# --- 2. Live Integration Tests: Stage 5 Discovery Endpoint ---

def test_stage5_einstein_zurich_evidence_linking():
    """Verify Stage 5 discovers and links EARNED_DOCTORATE_FROM / STUDIED_AT with full provenance."""
    response = client.get(
        "/api/v1/retrieval/relationships?question=What%20is%20the%20relationship%20between%20Albert%20Einstein%20and%20the%20University%20of%20Zurich%3F"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["PROVEN_RELATIONSHIP", "MULTI_RELATIONSHIP_DISCOVERED"]
    assert len(data["all_relationships"]) >= 1

    primary = data["primary_relationship"]
    assert primary is not None
    assert "Einstein" in primary["subject"]["normalized_text"]
    assert "Zurich" in primary["object"]["normalized_text"]
    assert primary["status"] in ["PROVEN", "PARTIALLY_SUPPORTED"]
    assert len(primary["evidence_chain"]) >= 1
    # Check exact provenance fields exist
    top_ev = primary["evidence_chain"][0]
    assert "document_title" in top_ev
    assert "paragraph_id" in top_ev
    assert "exact_sentence" in top_ev
    assert len(top_ev["exact_sentence"]) > 5


def test_stage5_krishnadevaraya_successor():
    """Verify Stage 5 links SUCCESSOR_OF with proven evidence and provenance."""
    response = client.get(
        "/api/v1/retrieval/relationships?question=Who%20was%20the%20successor%20of%20Shri%20Krishnadevaraya%3F"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PROVEN_RELATIONSHIP"
    primary = data["primary_relationship"]
    assert primary is not None
    assert "Krishnadevaraya" in primary["subject"]["normalized_text"]
    assert "Achyuta" in primary["object"]["normalized_text"]
    assert primary["predicate"] == "SUCCESSOR_OF"
    assert primary["status"] == "PROVEN"
    assert len(primary["evidence_chain"]) >= 1


def test_stage5_apple_founder():
    """Verify Stage 5 links FOUNDED_BY for Apple."""
    response = client.get(
        "/api/v1/retrieval/relationships?question=Who%20founded%20Apple%3F"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["PROVEN_RELATIONSHIP", "MULTI_RELATIONSHIP_DISCOVERED"]
    assert len(data["all_relationships"]) >= 1
    # Verify founder edge exists linking Apple with Steve Jobs / Steve Wozniak
    founder_match = any(
        "Apple" in r["subject"]["normalized_text"] and
        any(name in r["object"]["normalized_text"] for name in ["Steve", "Jobs", "Wozniak", "Ronald", "Wayne"])
        for r in data["all_relationships"]
    )
    assert founder_match is True





def test_stage5_evidence_graph_structure():
    """Verify the in-memory evidence graph contains entities and edges."""
    response = client.get(
        "/api/v1/retrieval/relationships?question=What%20is%20the%20relationship%20between%20Albert%20Einstein%20and%20the%20University%20of%20Zurich%3F"
    )
    assert response.status_code == 200
    data = response.json()
    graph = data["evidence_graph"]
    assert len(graph["entities"]) >= 2
    assert len(graph["relationships"]) >= 1
    assert graph["total_documents_consulted"] >= 1
    assert graph["total_passages_evaluated"] > 0
    assert len(graph["graph_provenance_summary"]) > 0
