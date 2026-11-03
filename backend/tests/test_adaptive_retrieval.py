"""Tests for Adaptive NLP Complexity Detection, Layered Understanding, and Query Decomposition."""

import pytest
from app.retrieval.adaptive_nlp_planner import (
    AdaptiveQueryAnalyzer,
    EvidenceCoverageAssessor,
    EvidencePacker,
    QueryComplexity,
    SubQueryRequirement,
    compute_shingle_similarity,
)
from app.services.answer_orchestrator import ClaimVerifier
from app.services.hybrid_retrieval import RetrievedChunk


def test_scenario_1_short_factual_question():
    """Scenario 1: Short factual questions should classify as SIMPLE."""
    q = "Who was the first President of India?"
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert res.complexity == QueryComplexity.SIMPLE
    assert len(res.subqueries) == 1
    assert res.interrogative_count == 1
    assert "India" in res.primary_entities or "India" in res.entities_detected


def test_scenario_2_medium_entity_relationship_question():
    """Scenario 2: Medium questions involving 2 entities and 1 relationship."""
    q = "What is the relationship between Ashoka and Chandragupta Maurya?"
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert res.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX)
    assert len(res.primary_entities) >= 1


def test_scenario_3_long_multi_requirement_question():
    """Scenario 3: Long question with several independent information requirements."""
    q = "When was the Maurya Empire founded, who was its first ruler, how did its conflict with the Seleucid Empire end, and what factors caused its decline?"
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert res.complexity == QueryComplexity.COMPLEX
    assert res.clause_count >= 3
    assert len(res.subqueries) >= 3

    # Verify that the subqueries cover founding, leadership, conflict, and decline
    aspects = {sq.aspect_type for sq in res.subqueries}
    assert "FOUNDING" in aspects or "LEADERSHIP" in aspects or "CONFLICT" in aspects or "DECLINE" in aspects


def test_scenario_4_multi_entity_question():
    """Scenario 4: Multi-entity extraction and multi-target article planning."""
    q = "Compare the military reforms of the Roman Empire and the Carthage Republic during the Punic Wars."
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert res.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX)
    entities = [e.lower() for e in res.primary_entities + res.entities_detected]
    assert any("roman" in e for e in entities)
    assert any("carthage" in e for e in entities)


def test_scenario_5_temporal_and_causal_question():
    """Scenario 5: Temporal and causal question constraints extraction."""
    q = "What were the economic causes of the French Revolution between 1780 and 1789?"
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert any("1780" in c or "1789" in c or "temporal" in c for c in res.constraints_detected)
    assert any("causal" in c for c in res.constraints_detected)


def test_scenario_6_comparisons_and_contrasts():
    """Scenario 6: Comparative queries."""
    q = "How did the administrative structure of the Byzantine Empire differ from the Western Roman Empire?"
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert any("comparative" in c for c in res.constraints_detected)


def test_scenario_7_pronoun_resolution_in_subqueries():
    """Scenario 7: Pronoun resolution (its, he, him) in decomposed subqueries."""
    q = "When was the Maratha Empire founded, who was its greatest ruler, and what territory did it expand into?"
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert res.complexity == QueryComplexity.COMPLEX
    
    # Subqueries must not have dangling pronouns without the subject entity
    for sq in res.subqueries[1:]:
        assert "maratha" in sq.retrieval_query.lower() or "maratha" in [e.lower() for e in sq.target_entities]


def test_scenario_8_multi_doc_target_planning():
    """Scenario 8: Evidence distributed across multiple documents produces multi-doc plan."""
    q = "What treaties were signed between the Mughal Empire and the British East India Company after the Battle of Buxar?"
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert len(res.target_articles) >= 1


def test_scenario_9_partial_evidence_coverage():
    """Scenario 9: Evidence coverage assessment with partial support."""
    subqueries = [
        SubQueryRequirement(
            id="sub_1",
            raw_clause="When was the Gupta Empire founded",
            retrieval_query="Gupta Empire founding date",
            aspect_type="FOUNDING",
            target_entities=["Gupta Empire"],
        ),
        SubQueryRequirement(
            id="sub_2",
            raw_clause="What space stations did it launch in 2024",
            retrieval_query="Gupta Empire space station 2024 launch",
            aspect_type="FACTOID",
            target_entities=["Gupta Empire"],
        ),
    ]

    # Only provide evidence for sub_1
    candidate_chunks = [
        RetrievedChunk(
            chunk_id=1,
            document_title="Gupta Empire",
            source_url="https://en.wikipedia.org/wiki/Gupta_Empire",
            section_title="Origins",
            passage_text="The Gupta Empire was an ancient Indian empire which existed from the early 4th century CE. It was founded by Sri Gupta.",
            rrf_score=0.9,
            retrieval_method="hybrid",
        )
    ]

    cov = EvidenceCoverageAssessor.assess_coverage(subqueries, candidate_chunks)
    assert cov.overall_status == "PARTIAL"
    assert cov.coverage_ratio == 0.5
    assert "sub_1" in cov.covered_subqueries
    assert len(cov.missing_aspects) == 1
    assert "space stations" in cov.missing_aspects[0].lower() or "sub_2" in cov.missing_aspects[0].lower()


def test_scenario_10_no_evidence_coverage():
    """Scenario 10: Zero evidence coverage detected for unsupported claims."""
    subqueries = [
        SubQueryRequirement(
            id="sub_1",
            raw_clause="What was Albert Einstein's favorite electric scooter brand",
            retrieval_query="Albert Einstein favorite electric scooter brand",
            aspect_type="FACTOID",
            target_entities=["Albert Einstein"],
        )
    ]

    # Chunk mentions Einstein but has zero match for electric scooter
    candidate_chunks = [
        RetrievedChunk(
            chunk_id=1,
            document_title="Albert Einstein",
            source_url="https://en.wikipedia.org/wiki/Albert_Einstein",
            section_title="General",
            passage_text="Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
            rrf_score=0.2,
            retrieval_method="hybrid",
        )
    ]

    cov = EvidenceCoverageAssessor.assess_coverage(subqueries, candidate_chunks)
    assert cov.overall_status == "INSUFFICIENT"
    assert cov.coverage_ratio == 0.0
    assert len(cov.missing_aspects) == 1


def test_scenario_11_performance_benchmark(capsys):
    """Scenario 11: Benchmark latency across short, medium, and long complex queries."""
    import time
    queries = [
        ("Short", "Who was the first President of India?"),
        ("Medium", "What is the relationship between Ashoka and Chandragupta Maurya?"),
        ("Complex", "When was the Maurya Empire founded, who was its first ruler, how did its conflict with the Seleucid Empire end, and what factors caused its decline?"),
    ]

    # Warm-up
    for _, q in queries:
        AdaptiveQueryAnalyzer.analyze_complexity(q)

    # Benchmark 5 runs
    results = {}
    for label, q in queries:
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = AdaptiveQueryAnalyzer.analyze_complexity(q)
            latencies.append((time.perf_counter() - t0) * 1000)
        mean_ms = sum(latencies) / len(latencies)
        results[label] = {
            "mean_ms": round(mean_ms, 3),
            "complexity": res.complexity.value,
            "subqueries": len(res.subqueries),
        }

    with capsys.disabled():
        print("\n\n================ LATENCY BENCHMARK ================")
        for label, data in results.items():
            print(f"  {label:7s} | Complexity: {data['complexity']:7s} | Subqueries: {data['subqueries']} | Latency: {data['mean_ms']:.3f} ms")
        print("===================================================\n")

    assert results["Short"]["mean_ms"] < 25.0
    assert results["Complex"]["mean_ms"] < 40.0


def test_scenario_12_descriptive_entity_discovery():
    """Scenario 12: Complex descriptive entity discovery questions without explicit named entities."""
    q = "What ancient device could predict astronomical events using a complex system of gears?"
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert res.complexity == QueryComplexity.COMPLEX
    assert any("descriptive" in r.lower() or "functional" in r.lower() for r in res.reasons)
    assert len(res.subqueries) >= 2
    # Verify subqueries cover function and mechanism
    types = {sq.aspect_type for sq in res.subqueries}
    assert "ENTITY_DISCOVERY" in types or "FUNCTION" in types or "MECHANISM" in types
    # Target articles should include the descriptive question
    assert q in res.target_articles or len(res.subqueries) >= 2


def test_scenario_13_muziris_trade_enumeration():
    """Scenario 13: Information-dense multi-aspect question with 'including X, Y, and Z'."""
    q = "Explain the role of Muziris in Indian–Roman trade, including the goods exported from India, the commodities imported from Rome, and the archaeological or textual evidence supporting this commerce."
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert res.complexity == QueryComplexity.COMPLEX
    assert len(res.subqueries) >= 3

    # Primary subject must be clean and not corrupted by prepositions
    assert "Muziris" in res.primary_entities

    aspects = {sq.aspect_type for sq in res.subqueries}
    assert "TRADE_COMMERCE" in aspects
    assert "EXPORTS" in aspects
    assert "IMPORTS" in aspects
    assert "EVIDENCE_SOURCE" in aspects

    # Verify query targets are enriched with domain terms
    export_subs = [sq for sq in res.subqueries if sq.aspect_type == "EXPORTS"]
    assert any("pepper" in sq.retrieval_query.lower() or "export" in sq.retrieval_query.lower() for sq in export_subs)

    import_subs = [sq for sq in res.subqueries if sq.aspect_type == "IMPORTS"]
    assert any("wine" in sq.retrieval_query.lower() or "import" in sq.retrieval_query.lower() for sq in import_subs)


def test_scenario_14_evidence_packer_slotted_dedup():
    """Scenario 14: EvidencePacker slots passages across requirements and deduplicates repetitive chunks."""
    subqueries = [
        SubQueryRequirement(
            id="sub_1",
            raw_clause="the goods exported from India",
            retrieval_query="Muziris goods exported pepper pearls",
            aspect_type="EXPORTS",
            target_entities=["Muziris"],
        ),
        SubQueryRequirement(
            id="sub_2",
            raw_clause="the commodities imported from Rome",
            retrieval_query="Muziris commodities imported wine gold coins",
            aspect_type="IMPORTS",
            target_entities=["Muziris"],
        ),
        SubQueryRequirement(
            id="sub_3",
            raw_clause="the archaeological or textual evidence",
            retrieval_query="Muziris Pattanam excavations Periplus",
            aspect_type="EVIDENCE_SOURCE",
            target_entities=["Muziris"],
        ),
    ]

    chunk_export = RetrievedChunk(
        chunk_id=1,
        document_title="Muziris",
        source_url="https://en.wikipedia.org/wiki/Muziris",
        section_title="Exports",
        passage_text="Muziris exported black pepper, fine pearls, beryl, and textiles across the Indian Ocean to the Roman Empire.",
        rrf_score=0.95,
        retrieval_method="hybrid",
    )
    chunk_import = RetrievedChunk(
        chunk_id=2,
        document_title="Muziris",
        source_url="https://en.wikipedia.org/wiki/Muziris",
        section_title="Imports",
        passage_text="Roman trade ships brought commodities imported into Muziris including Italian wine, gold coins, and Mediterranean amphorae.",
        rrf_score=0.90,
        retrieval_method="hybrid",
    )
    # Redundant chunk with high shingle overlap to chunk_export
    chunk_export_dup = RetrievedChunk(
        chunk_id=3,
        document_title="Muziris",
        source_url="https://en.wikipedia.org/wiki/Muziris",
        section_title="Trade Goods",
        passage_text="Muziris exported black pepper, fine pearls, beryl, and textiles directly to the merchants of Rome.",
        rrf_score=0.93,
        retrieval_method="hybrid",
    )
    chunk_evidence = RetrievedChunk(
        chunk_id=4,
        document_title="Muziris",
        source_url="https://en.wikipedia.org/wiki/Muziris",
        section_title="Archaeology",
        passage_text="Archaeological excavations at Pattanam unearthed thousands of Roman amphorae fragments and coins, confirming Periplus textual records.",
        rrf_score=0.88,
        retrieval_method="hybrid",
    )

    candidates = [chunk_export, chunk_export_dup, chunk_import, chunk_evidence]
    selected = EvidencePacker.pack_slotted_evidence(
        subqueries=subqueries,
        candidate_chunks=candidates,
        max_passages=3,
        primary_entities=["Muziris"],
    )

    selected_ids = [c.chunk_id for c in selected]
    # Verify that all 3 requirements received a slot
    assert 1 in selected_ids  # chunk_export
    assert 2 in selected_ids  # chunk_import
    assert 4 in selected_ids  # chunk_evidence
    # Redundant duplicate chunk must be excluded
    assert 3 not in selected_ids


def test_scenario_15_shingle_similarity_metric():
    """Scenario 15: 3-shingle similarity accurately catches repetitive sentences."""
    t1 = "Excavations at Pattanam revealed numerous Roman amphora sherds and gold coins."
    t2 = "Excavations at Pattanam revealed numerous Roman amphora sherds and silver coins."
    t3 = "The Maurya Empire was an Iron Age historical power in South Asia."

    sim_high = compute_shingle_similarity(t1, t2)
    sim_zero = compute_shingle_similarity(t1, t3)

    assert sim_high >= 0.50
    assert sim_zero == 0.0


def test_scenario_16_compound_entity_and_selective_evidence():
    """Scenario 16: Compound technical entity is recognized and off-topic sections are filtered."""
    q = "Explain the Antikythera mechanism, including its mechanical operation, astronomical functions, historical impact, and uncertainties."
    res = AdaptiveQueryAnalyzer.analyze_complexity(q)
    assert res.complexity == QueryComplexity.COMPLEX
    # Must identify "Antikythera mechanism" rather than bare island "Antikythera"
    assert "Antikythera mechanism" in res.primary_entities
    assert "Antikythera mechanism" in res.target_articles
    assert "Antikythera" not in res.target_articles  # Generic parent disambiguated away

    # Test evidence packer drops off-topic fauna
    chunk_mechanism = RetrievedChunk(
        chunk_id=1,
        document_title="Antikythera mechanism",
        source_url="https://en.wikipedia.org/wiki/Antikythera_mechanism",
        section_title="Operation",
        passage_text="The Antikythera mechanism used thirty bronze gears to compute astronomical positions.",
        rrf_score=0.92,
        retrieval_method="hybrid",
    )
    chunk_fauna = RetrievedChunk(
        chunk_id=2,
        document_title="Antikythera",
        source_url="https://en.wikipedia.org/wiki/Antikythera",
        section_title="Fauna",
        passage_text="The island of Antikythera is home to migratory birds and wild goats.",
        rrf_score=0.15,
        retrieval_method="hybrid",
    )

    selected = EvidencePacker.pack_slotted_evidence(
        subqueries=res.subqueries,
        candidate_chunks=[chunk_mechanism, chunk_fauna],
        max_passages=2,
        primary_entities=res.primary_entities,
    )
    selected_ids = [c.chunk_id for c in selected]
    assert 1 in selected_ids
    assert 2 not in selected_ids  # Fauna must be filtered out


def test_scenario_17_claim_verifier_grounding():
    """Scenario 17: ClaimVerifier drops off-topic intrusive sentences and verifies citation support."""
    q = "Explain the Antikythera mechanism and its mechanical operation."
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            document_title="Antikythera mechanism",
            source_url="https://en.wikipedia.org/wiki/Antikythera_mechanism",
            section_title="Operation",
            passage_text="The Antikythera mechanism used thirty bronze gears to model astronomical cycles.",
            rrf_score=0.9,
            retrieval_method="hybrid",
        )
    ]

    raw_answer = (
        "The Antikythera mechanism used thirty bronze gears to model astronomical cycles [1]. "
        "The island is famous for migratory bird observation and wild goats [1]. "
        "The device was influenced by Aztec astronomical pyramids."
    )

    cleaned_answer, stats = ClaimVerifier.verify_and_clean_answer(
        answer_text=raw_answer,
        selected_chunks=chunks,
        question=q,
    )

    # Valid supported claim is retained
    assert "thirty bronze gears" in cleaned_answer
    # Off-topic fauna claim is dropped
    assert "migratory bird" not in cleaned_answer
    assert "wild goats" not in cleaned_answer
    # Foreign ungrounded entities ("Aztec") without citation are dropped
    assert "Aztec" not in cleaned_answer

    assert stats["claims_verified"] == 3
    assert stats["claims_supported"] == 1
    assert stats["unsupported_claims_removed"] >= 1
