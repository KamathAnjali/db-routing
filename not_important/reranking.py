"""Reranking code based on DBRouting_paper1.pdf (Routing End User Queries to Enterprise Databases).

Implements the paper's re-ranking stage:
1) Build table adjacency list (join graph) per DB (LLM-produced in paper).
2) Map query phrases to schema entities (LLM-produced in paper).
3) Compute coverage + connectivity; total_score = coverage * connectivity.
4) Use semantic_score for tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple
import math


# ---- Data structures ----

# A schema entity is represented as (table, column, optional description)
SchemaEntity = Tuple[str, str, str | None]

# Phrase -> list of candidate schema entities. Empty list means N/A.
PhraseMapping = Dict[str, List[SchemaEntity]]

# Table adjacency list (join graph). Each table maps to its neighboring tables.
Adjacency = Dict[str, Set[str]]


# ---- Core scoring ----

def coverage_score(phrase_mapping: PhraseMapping, n: float = 2.0) -> float:
    """Compute phrase coverage score.

    The paper describes an exponential penalty for N/A phrase mappings.
    We interpret it as: coverage = exp(-n * x)
    where x = (# N/A phrases) / (total phrases).
    """
    total_phrases = len(phrase_mapping)
    if total_phrases == 0:
        return 0.0

    na_phrases = sum(1 for candidates in phrase_mapping.values() if not candidates)
    x = na_phrases / total_phrases
    return math.exp(-n * x)


def _connected_component(adj: Adjacency, start: str) -> Set[str]:
    """BFS over the full table graph."""
    stack = [start]
    seen = {start}
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def connectivity_score(adj: Adjacency, phrase_mapping: PhraseMapping) -> int:
    """Return 1 if mapped entities can form a connected subgraph, else 0.

    Each phrase may map to multiple candidate schema entities (columns). We
    treat each candidate as belonging to a table. Connectivity holds if there
    exists a connected component in the schema graph that contains at least
    one candidate table from every phrase.
    """
    if not phrase_mapping:
        return 0

    # Build candidate table sets per phrase (ignore N/A phrases here).
    phrase_tables: List[Set[str]] = []
    for candidates in phrase_mapping.values():
        if not candidates:
            # N/A mapping: connectivity fails only if you want strict coverage.
            # The paper uses coverage for N/A; connectivity checks mapped entities.
            continue
        phrase_tables.append({t for (t, _c, _d) in candidates})

    if not phrase_tables:
        return 0

    candidate_tables = set().union(*phrase_tables)
    all_nodes = set(adj.keys()) | candidate_tables
    seen: Set[str] = set()

    for node in all_nodes:
        if node in seen:
            continue
        component = _connected_component(adj, node)
        seen |= component
        # Check if this component intersects every phrase's candidate set.
        if all(component & group for group in phrase_tables):
            return 1

    return 0


def semantic_score(
    phrase_mapping: PhraseMapping,
    embed_phrase: Callable[[str], Sequence[float]],
    embed_entity: Callable[[SchemaEntity], Sequence[float]],
) -> float:
    """Average cosine similarity for all mapped phrase-entity pairs.

    The paper: For each mapped query phrase, compute cosine similarity between
    the phrase embedding and its matched column name + info; semantic score is
    the average of these similarities.
    """
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    sims: List[float] = []
    for phrase, candidates in phrase_mapping.items():
        if not candidates:
            continue
        p_emb = embed_phrase(phrase)
        for ent in candidates:
            sims.append(cosine(p_emb, embed_entity(ent)))

    return sum(sims) / len(sims) if sims else 0.0


def total_score(
    adj: Adjacency,
    phrase_mapping: PhraseMapping,
    n: float = 2.0,
) -> float:
    """Total score = coverage * connectivity."""
    return coverage_score(phrase_mapping, n=n) * connectivity_score(adj, phrase_mapping)


# ---- Reranking ----

def rerank_dbs(
    db_ids: Iterable[str],
    adj_by_db: Dict[str, Adjacency],
    mapping_by_db: Dict[str, PhraseMapping],
    embed_phrase: Callable[[str], Sequence[float]] | None = None,
    embed_entity: Callable[[SchemaEntity], Sequence[float]] | None = None,
    n: float = 2.0,
) -> List[Dict[str, float | str]]:
    """Return DBs sorted by total_score, tie-broken by semantic_score."""
    results: List[Tuple[str, float, float]] = []

    for db_id in db_ids:
        adj = adj_by_db[db_id]
        mapping = mapping_by_db[db_id]
        tscore = total_score(adj, mapping, n=n)
        if embed_phrase and embed_entity and tscore > 0:
            sscore = semantic_score(mapping, embed_phrase, embed_entity)
        else:
            sscore = 0.0
        results.append((db_id, tscore, sscore))

    results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [
        {"db_id": db_id, "total_score": tscore, "semantic_score": sscore}
        for db_id, tscore, sscore in results
    ]


# ---- Example wiring (placeholders) ----

def example_usage():
    # Adjacency list per DB (join graph). In the paper this is produced by an LLM prompt.
    adj_by_db = {
        "activity": {
            "Activity": {"Participates_in", "Faculty_Participates_in"},
            "Participates_in": {"Activity", "Student"},
            "Faculty_Participates_in": {"Activity", "Faculty"},
            "Student": {"Participates_in"},
            "Faculty": {"Faculty_Participates_in"},
        }
    }

    # Phrase-to-entity mapping per DB (LLM prompt in paper).
    mapping_by_db = {
        "activity": {
            "John": [("Student", "student_name", None), ("Faculty", "faculty_name", None)],
            "do": [("Activity", "activity_name", None)],
        }
    }

    # Minimal dummy embeddings
    def embed_phrase(p: str) -> Sequence[float]:
        return [1.0, 0.0] if p.lower() == "john" else [0.0, 1.0]

    def embed_entity(e: SchemaEntity) -> Sequence[float]:
        table, col, _ = e
        return [1.0, 0.0] if "student" in col else [0.0, 1.0]

    ranked = rerank_dbs(
        db_ids=["activity"],
        adj_by_db=adj_by_db,
        mapping_by_db=mapping_by_db,
        embed_phrase=embed_phrase,
        embed_entity=embed_entity,
        n=2.0,
    )
    print(ranked)


if __name__ == "__main__":
    example_usage()
