"""
ReasoningEngine — Multi-Hop Reasoning, Entity Comparison & Contradiction Detection.
"""
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.query_understanding import IntentCategory, QueryUnderstanding
from app.knowledge.models import Contradiction, Entity, Fact, Relationship
from app.knowledge.workspace import KnowledgeWorkspace

logger = logging.getLogger("nexia.knowledge.reasoning")


def _norm(text: str) -> str:
    """Normalize string for lookup."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


class ReasoningEngine:
    """
    Reasoning engine operating over a KnowledgeWorkspace.
    Traverses multi-hop relationship graphs, structures multi-entity comparisons,
    and isolates contradictions without hallucinating unsupported edges.
    """

    def multi_hop_traverse(
        self,
        workspace: KnowledgeWorkspace,
        start_entity: str,
        target_concept_or_entity: str,
        max_depth: int = 3,
    ) -> List[List[Relationship]]:
        """
        Find directed multi-hop paths connecting start_entity to target_concept_or_entity.
        Traverses only verified relationships in the workspace.
        """
        norm_start = _norm(start_entity)
        norm_target = _norm(target_concept_or_entity)

        if not norm_start or not norm_target:
            return []

        # BFS for paths
        queue: List[Tuple[str, List[Relationship]]] = [(norm_start, [])]
        visited_nodes: Set[str] = {norm_start}
        valid_paths: List[List[Relationship]] = []

        while queue:
            current_node, current_path = queue.pop(0)

            # Check if current node satisfies target condition
            if current_path and (norm_target in current_node or current_node in norm_target):
                valid_paths.append(current_path)
                continue

            if len(current_path) >= max_depth:
                continue

            # Outgoing edges
            for rel in workspace.relationships:
                if _norm(rel.source_entity) == current_node:
                    next_node = _norm(rel.target_entity)
                    if next_node not in visited_nodes:
                        visited_nodes.add(next_node)
                        queue.append((next_node, current_path + [rel]))

        return valid_paths

    def compare_entities(
        self,
        workspace: KnowledgeWorkspace,
        entity_a: str,
        entity_b: str,
    ) -> Dict[str, Any]:
        """
        Extract structured comparison profile for two entities.
        """
        facts_a = workspace.find_facts(subject=entity_a)
        facts_b = workspace.find_facts(subject=entity_b)
        rels_a = workspace.find_relationships(source=entity_a)
        rels_b = workspace.find_relationships(source=entity_b)

        return {
            "entity_a": {
                "name": entity_a,
                "facts": [f"{f.predicate}: {f.object_value}" for f in facts_a],
                "relationships": [f"{r.relation} -> {r.target_entity}" for r in rels_a],
            },
            "entity_b": {
                "name": entity_b,
                "facts": [f"{f.predicate}: {f.object_value}" for f in facts_b],
                "relationships": [f"{r.relation} -> {r.target_entity}" for r in rels_b],
            },
        }

    def build_reasoned_knowledge_summary(
        self,
        workspace: KnowledgeWorkspace,
        query: str,
        qu: QueryUnderstanding,
    ) -> str:
        """
        Construct structured knowledge context block to include in the grounded prompt.
        Presents verified facts, multi-hop relationship chains, and contradictions.
        """
        sections = []

        # 1. Contradictions (if any)
        if workspace.contradictions:
            contra_lines = []
            for c in workspace.contradictions:
                contra_lines.append(f"- Note: Conflicting evidence found on '{c.topic}': '{c.claim_a}' vs '{c.claim_b}'")
            sections.append("Conflicting Evidence Detected:\n" + "\n".join(contra_lines))

        # 2. Key Facts
        if workspace.facts:
            fact_lines = []
            for f in workspace.facts[:8]:  # Keep prompt concise
                fact_lines.append(f"- {f.subject} ({f.predicate}): {f.object_value}")
            sections.append("Verified Facts:\n" + "\n".join(fact_lines))

        # 3. Relationships & Multi-Hop Chains
        if workspace.relationships:
            rel_lines = []
            for r in workspace.relationships[:8]:
                rel_lines.append(f"- {r.source_entity} --[{r.relation}]--> {r.target_entity}")
            sections.append("Verified Relationships:\n" + "\n".join(rel_lines))

        # 4. Multi-hop chains for connective queries ("why", "how", "connection", "influence")
        q_lower = query.lower()
        if any(w in q_lower for w in ["why", "how", "connected", "connection", "influence", "relation", "relate", "lead to"]):
            # Try finding multi-hop paths from primary entity
            primary_ent = qu.entity or (list(workspace.entities.values())[0].canonical_name if workspace.entities else None)
            if primary_ent:
                for word in re.findall(r"\b[A-Za-z]{4,}\b", query):
                    if _norm(word) != _norm(primary_ent):
                        paths = self.multi_hop_traverse(workspace, primary_ent, word, max_depth=3)
                        if paths:
                            chain_lines = []
                            for p in paths[:2]:
                                chain_str = " -> ".join([p[0].source_entity] + [f"[{step.relation}]-> {step.target_entity}" for step in p])
                                chain_lines.append(f"- {chain_str}")
                            if chain_lines:
                                sections.append("Multi-Hop Relationship Chains:\n" + "\n".join(chain_lines))
                                break

        if not sections:
            return ""

        return "Structured Knowledge (Extracted from Wikipedia Evidence):\n" + "\n\n".join(sections) + "\n\n"


reasoning_engine = ReasoningEngine()
