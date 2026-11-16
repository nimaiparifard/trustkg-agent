"""Build the ground-truth world and the (noisy) agent-visible KG and text corpus."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .graph_store import InMemoryGraph
from .schema import (CHAIN, RELATIONS, TYPE_PREFIX, Document, Fact, object_type,
                     render_fact)

Truth = Dict[Tuple[str, str], str]   # (subject, relation) -> object


def make_names(etype: str, start: int, n: int) -> List[str]:
    return [f"{TYPE_PREFIX[etype]}_{i:04d}" for i in range(start, start + n)]


def build_world(counts: Dict[str, int], rng: random.Random):
    """Return (entities_by_type, truth). Every relation is functional."""
    ents = {t: make_names(t, 0, n) for t, n in counts.items()}
    truth: Truth = {}
    for rel in CHAIN:
        st, ot = RELATIONS[rel][0], RELATIONS[rel][1]
        for s in ents[st]:
            truth[(s, rel)] = rng.choice(ents[ot])
    return ents, truth


def wrong_object(truth_obj: str, rel: str, ents_by_type: Dict[str, List[str]], rng: random.Random) -> str:
    pool = ents_by_type[object_type(rel)]
    while True:
        o = rng.choice(pool)
        if o != truth_obj or len(pool) == 1:
            return o


def build_kg(truth: Truth, ents_by_type: Dict[str, List[str]], rng: random.Random,
             missing_rate: float, conflict_rate: float) -> InMemoryGraph:
    """Agent-visible KG at stage 0: curated facts (conf 0.9) minus omissions,
    plus conflicting low-confidence facts from a noisy source."""
    g = InMemoryGraph()
    for t, names in ents_by_type.items():
        for n in names:
            g.add_entity(n, t)
    facts: List[Fact] = []
    for (s, r), o in truth.items():
        if rng.random() >= missing_rate:
            facts.append(Fact(s, r, o, source="curated", confidence=0.9))
        if rng.random() < conflict_rate:
            facts.append(Fact(s, r, wrong_object(o, r, ents_by_type, rng), source="noisy_web",
                              confidence=round(rng.uniform(0.3, 0.7), 3)))
    rng.shuffle(facts)  # insertion order must not leak which fact is true
    for f in facts:
        g.add_fact(f)
    return g


def build_corpus(truth: Truth, ents_by_type: Dict[str, List[str]], rng: random.Random,
                 missing_rate: float, conflict_rate: float) -> List[Document]:
    docs: List[Document] = []
    for (s, r), o in truth.items():
        if rng.random() >= missing_rate:
            docs.append(Document(render_fact(s, r, o), "web"))
        if rng.random() < conflict_rate:
            docs.append(Document(render_fact(s, r, wrong_object(o, r, ents_by_type, rng)), "web"))
    rng.shuffle(docs)
    return docs
