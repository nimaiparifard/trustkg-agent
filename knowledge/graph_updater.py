"""Knowledge updates: add / modify / deprecate facts with versioning and validity time."""
from __future__ import annotations

from typing import Optional, Tuple

from .schema import RELATIONS, Fact, object_type, subject_type


class GraphUpdater:
    def __init__(self, graph) -> None:
        self.graph = graph

    def _ensure_entities(self, s: str, r: str, o: str) -> None:
        if not self.graph.has_entity(s):
            self.graph.add_entity(s, subject_type(r))
        if not self.graph.has_entity(o):
            self.graph.add_entity(o, object_type(r))

    def upsert(self, s: str, r: str, o: str, now: int, source: str = "feedback",
               confidence: float = 0.95) -> Tuple[str, Optional[int]]:
        """Reconcile a trusted observation (s, r, o) with the graph.

        Returns (status, age) where status is
          'confirmed' - an existing valid fact already states o (re-verified),
          'changed'   - the graph stated something else (old facts deprecated),
          'new'       - the graph had nothing for (s, r).
        `age` = stages since the replaced/confirmed fact was last verified.
        """
        self._ensure_entities(s, r, o)
        existing = self.graph.get(s, r)
        match = [f for f in existing if f.object == o]
        others = [f for f in existing if f.object != o]
        if match:
            ref = max(match, key=lambda f: f.last_verified)
            age = now - ref.last_verified
            self.graph.mark_verified(ref.fid, now, confidence)
            for f in others:                       # contradicted by trusted observation
                self.graph.deprecate(f.fid)
            return "confirmed", age
        if existing:
            age = now - max(f.last_verified for f in existing)
            for f in existing:
                self.graph.deprecate(f.fid)
            self.graph.add_fact(Fact(s, r, o, source, confidence, now, now))
            return "changed", age
        self.graph.add_fact(Fact(s, r, o, source, confidence, now, now))
        return "new", None
