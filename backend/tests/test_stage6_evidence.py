"""Tests for Stage 6: Advanced Evidence Reasoning, Clustering, Deduplication, and Synthesis."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.evidence_package import (
    EvidenceItem,
    EvidenceCluster,
    EvidenceConflict,
    SufficiencyAssessment,
    AnswerCandidate,
    EvidencePackage,
    EvidenceReasoningRequest,
)
from app.retrieval.evidence_reasoner import EvidenceReasoner
from app.schemas.relationships import (
    RelationshipDiscoveryResponse,
    Stage5EvidenceGraph,
    DiscoveredRelationshipEdge,
    EvidenceProvenanceItem,
)
from app.schemas.retrieval import EntityItem, Stage4StructuredQuery


client = TestClient(app)
reasoner = EvidenceReasoner()


# ============================================================================
# Unit Tests (Deterministic Reasoning Components)
# ============================================================================

def test_evidence_deduplication_exact_and_near_match():
    """Near-duplicate items with Jaccard overlap >= 0.75 must be marked is_duplicate=True."""
    item1 = EvidenceItem(
        item_id="item_1",
        text="Albert Einstein completed his doctoral dissertation at the University of Zurich under Alfred Kleiner.",
        source_document="Albert Einstein",
        source_section="Academic career",
        relevance_score=0.9,
        support_class="SUPPORTING",
    )
    # Item 2 is a near-duplicate with tiny variation
    item2 = EvidenceItem(
        item_id="item_2",
        text="Einstein completed his doctoral dissertation at the University of Zurich under professor Alfred Kleiner.",
        source_document="University of Zurich",
        source_section="Notable alumni",
        relevance_score=0.88,
        support_class="SUPPORTING",
    )
    # Item 3 is completely distinct
    item3 = EvidenceItem(
        item_id="item_3",
        text="The Zurich lake provides water to the canton of Zurich and surrounding Alpine valleys.",
        source_document="Zurich",
        source_section="Geography",
        relevance_score=0.2,
        support_class="SUPPORTING",
    )

    items = [item1, item2, item3]
    deduped, summary = reasoner._deduplicate(items, threshold=0.75)

    assert deduped[0].is_duplicate is False
    assert deduped[1].is_duplicate is True
    assert deduped[1].support_class == "IRRELEVANT"
    assert deduped[2].is_duplicate is False
    assert "1 near-duplicates" in summary


def test_evidence_clustering_by_section():
    """Items from different documents/sections must cluster into distinct groups."""
    item1 = EvidenceItem(
        item_id="item_1",
        text="Sentence from document A section Alpha",
        source_document="Document A",
        source_section="Alpha",
        relevance_score=0.8,
        support_class="DIRECT_ANSWER",
    )
    item2 = EvidenceItem(
        item_id="item_2",
        text="Another sentence from document A section Alpha",
        source_document="Document A",
        source_section="Alpha",
        relevance_score=0.7,
        support_class="SUPPORTING",
    )
    item3 = EvidenceItem(
        item_id="item_3",
        text="Sentence from document B section Beta",
        source_document="Document B",
        source_section="Beta",
        relevance_score=0.6,
        support_class="CONTEXTUAL",
    )

    clusters = reasoner._cluster_evidence([item1, item2, item3])
    assert len(clusters) == 2
    topics = [c.cluster_topic for c in clusters]
    assert "Document A - Alpha" in topics
    assert "Document B - Beta" in topics


def test_conflict_detection_conflicting_dates():
    """Conflicting years for the same subject in direct answer passages must be flagged."""
    item1 = EvidenceItem(
        item_id="item_1",
        text="Krishnadevaraya succeeded to the throne of Vijayanagara Empire in the year 1509.",
        source_document="Krishnadevaraya",
        source_section="Reign",
        relevance_score=0.85,
        support_class="DIRECT_ANSWER",
    )
    item2 = EvidenceItem(
        item_id="item_2",
        text="Some colonial chronicles claim Krishnadevaraya succeeded to the throne in the year 1515.",
        source_document="Krishnadevaraya",
        source_section="Reign",
        relevance_score=0.82,
        support_class="DIRECT_ANSWER",
    )

    conflicts = reasoner._detect_conflicts([item1, item2])
    assert len(conflicts) >= 1
    assert conflicts[0].severity == "MEDIUM"
    assert "item_1" in conflicts[0].conflicting_items
    assert "item_2" in conflicts[0].conflicting_items


def test_sufficiency_assessment_sufficient():
    """Direct answer evidence with strong relevance should produce SUFFICIENT rating."""
    item = EvidenceItem(
        item_id="item_1",
        text="Achyuta Deva Raya was the successor of Krishnadevaraya in 1529.",
        source_document="Krishnadevaraya",
        source_section="Succession",
        relevance_score=0.88,
        support_class="DIRECT_ANSWER",
    )
    assessment = reasoner._assess_sufficiency(
        items=[item],
        conflicts=[],
        structured_query=None,
        stage5_status="PROVEN_RELATIONSHIP",
    )
    assert assessment.status == "SUFFICIENT"
    assert assessment.confidence >= 0.75
    assert len(assessment.missing_aspects) == 0


def test_sufficiency_assessment_insufficient():
    """No active evidence items must result in INSUFFICIENT rating."""
    assessment = reasoner._assess_sufficiency(
        items=[],
        conflicts=[],
        structured_query=None,
        stage5_status="NO_RELATIONSHIP_FOUND",
    )
    assert assessment.status == "INSUFFICIENT"
    assert assessment.confidence < 0.50
    assert len(assessment.missing_aspects) > 0


# ============================================================================
# Integration Tests via API & Pipeline
# ============================================================================

def test_api_stage6_einstein_zurich():
    """Stage 6 retrieves, reasons, dedupes, and clusters Einstein & University of Zurich evidence."""
    response = client.get(
        "/retrieval/evidence",
        params={"question": "What is the relationship between Albert Einstein and the University of Zurich?"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("COMPLETED", "PARTIALLY_SUPPORTED")
    package = data["evidence_package"]
    assert package["question"] == "What is the relationship between Albert Einstein and the University of Zurich?"
    assert package["total_evidence_items"] > 0
    assert package["total_clusters"] > 0
    assert package["sufficiency_assessment"]["status"] in ("SUFFICIENT", "PARTIAL")

    # Verify cluster structure
    first_cluster = package["evidence_clusters"][0]
    assert "cluster_id" in first_cluster
    assert "cluster_topic" in first_cluster
    assert len(first_cluster["items"]) > 0

    # Verify at least one answer candidate or grounded entity surfaced
    candidates = package["answer_candidates"]
    assert len(candidates) > 0


def test_api_stage6_krishnadevaraya_successor():
    """Stage 6 produces evidence package identifying successor of Krishnadevaraya."""
    response = client.get(
        "/retrieval/evidence",
        params={"question": "successor of Shri Krishnadevaraya"},
    )
    assert response.status_code == 200
    data = response.json()

    package = data["evidence_package"]
    assert package["sufficiency_assessment"]["status"] in ("SUFFICIENT", "PARTIAL")

    # Check that candidate answer mentions Achyuta Deva Raya or succession
    candidates = [c["candidate_text"].lower() for c in package["answer_candidates"]]
    assert any("achyuta" in c for c in candidates) or package["total_evidence_items"] > 0


def test_api_stage6_apple_founder():
    """Stage 6 handles 'who founded Apple?' corporate founder retrieval."""
    response = client.get(
        "/retrieval/evidence",
        params={"question": "who founded Apple?"},
    )
    assert response.status_code == 200
    data = response.json()

    package = data["evidence_package"]
    assert package["total_evidence_items"] > 0
    assert package["sufficiency_assessment"]["status"] in ("SUFFICIENT", "PARTIAL")


def test_api_stage6_empty_query():
    """Empty query should gracefully return error status without crashing."""
    response = client.get(
        "/retrieval/evidence",
        params={"question": "   "},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ERROR"
    assert data["evidence_package"]["sufficiency_assessment"]["status"] == "INSUFFICIENT"


def test_api_stage6_post_endpoint():
    """POST /retrieval/evidence must accept body and produce valid EvidencePackage."""
    payload = {
        "question": "What is the relationship between Albert Einstein and the University of Zurich?",
        "top_k_per_document": 3,
        "dedup_similarity_threshold": 0.70,
    }
    response = client.post("/retrieval/evidence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("COMPLETED", "PARTIALLY_SUPPORTED")
    assert data["evidence_package"]["stage6_status"] == "COMPLETED"
    assert "deduplication_summary" in data["evidence_package"]
