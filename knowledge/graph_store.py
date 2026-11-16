"""Knowledge-graph storage backends sharing one small interface.

InMemoryGraph is the default (used for all reported experiments, no server
needed). Neo4jGraph in neo4j_store.py implements the same interface.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Dict, Iterable, List, Optional

from .schema import Fact


class InMemoryGraph:
    def __init__(self) -> None:
        self._facts: List[Fact] = []
        self._by_key: Dict[tuple, List[int]] = defaultdict(list)
        self.entity_type: Dict[str, str] = {}

    # ---- entities -------------------------------------------------------
    def add_entity(self, name: str, etype: str) -> None:
        self.entity_type[name] = etype

    def has_entity(self, name: str) -> bool:
        return name in self.entity_type

    def entities(self, etype: Optional[str] = None) -> List[str]:
        return [e for e, t in self.entity_type.items() if etype is None or t == etype]

    # ---- facts ----------------------------------------------------------
    def add_fact(self, fact: Fact) -> int:
        fact = replace(fact, fid=len(self._facts))
        self._facts.append(fact)
        self._by_key[(fact.subject, fact.relation)].append(fact.fid)
        return fact.fid

    def get(self, subject: str, relation: str, include_invalid: bool = False) -> List[Fact]:
        out = [self._facts[i] for i in self._by_key.get((subject, relation), [])]
        return out if include_invalid else [f for f in out if f.valid]

    def deprecate(self, fid: int) -> None:
        self._facts[fid].valid = False

    def mark_verified(self, fid: int, stage: int, confidence: Optional[float] = None) -> None:
        f = self._facts[fid]
        f.last_verified = stage
        if confidence is not None:
            f.confidence = max(f.confidence, confidence)

    def facts(self, valid_only: bool = True) -> Iterable[Fact]:
        return (f for f in self._facts if f.valid or not valid_only)

    def num_facts(self, valid_only: bool = True) -> int:
        return sum(1 for _ in self.facts(valid_only))

    def copy(self) -> "InMemoryGraph":
        g = InMemoryGraph()
        g.entity_type = dict(self.entity_type)
        for f in self._facts:
            g.add_fact(replace(f))
        return g
