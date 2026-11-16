"""Adaptive memory: episodic + semantic memory and knowledge updates over time."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from knowledge.graph_updater import GraphUpdater


@dataclass
class Episode:
    stage: int
    question: str
    relations: Tuple[str, ...]
    answer: Optional[str]
    abstained: bool
    failure_events: int
    confidence: float
    correct: Optional[bool] = None   # filled in when feedback arrives


class EpisodicMemory:
    """What happened on past tasks (plans, outcomes, failures)."""

    def __init__(self) -> None:
        self.episodes: List[Episode] = []

    def add(self, ep: Episode) -> Episode:
        self.episodes.append(ep)
        return ep

    def success_rate(self, relations: Sequence[str], min_n: int = 3) -> Optional[float]:
        """Empirical success of past episodes with the same relation path (None if too few)."""
        done = [e for e in self.episodes if e.relations == tuple(relations) and e.correct is not None]
        return sum(e.correct for e in done) / len(done) if len(done) >= min_n else None


class SemanticMemory:
    """Distilled knowledge: how quickly each relation's facts go stale.

    Per-relation Gamma-prior estimate of the per-stage change probability
    from confirmed/changed observations (exposure-based).
    """

    def __init__(self, prior_hazard: float = 0.1, prior_strength: float = 5.0) -> None:
        self.a0 = prior_hazard * prior_strength
        self.b0 = prior_strength
        self.changes: Dict[str, float] = defaultdict(float)
        self.exposure: Dict[str, float] = defaultdict(float)
        self.n_corrections = 0

    def observe(self, relation: str, changed: bool, age: float) -> None:
        age = max(float(age), 1.0)
        self.exposure[relation] += age
        if changed:
            self.changes[relation] += 1.0
            self.n_corrections += 1

    def hazard(self, relation: str) -> float:
        return (self.changes[relation] + self.a0) / (self.exposure[relation] + self.b0)


class AdaptiveMemory:
    def __init__(self, updater: GraphUpdater) -> None:
        self.updater = updater
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.last_episode: Optional[Episode] = None
        self.learned = {"confirmed": 0, "changed": 0, "new": 0}

    def hazard(self, relation: str) -> float:
        return self.semantic.hazard(relation)

    def record_episode(self, ep: Episode) -> None:
        self.last_episode = self.episodic.add(ep)

    def ingest_feedback(self, chain: Sequence[Tuple[str, str, str]], now: int,
                        correct: Optional[bool] = None) -> None:
        """Learn from a trusted observation: update the KG and the staleness model."""
        for s, r, o in chain:
            status, age = self.updater.upsert(s, r, o, now)
            self.learned[status] += 1
            if status == "confirmed" and age:
                self.semantic.observe(r, False, age)
            elif status == "changed":
                self.semantic.observe(r, True, age or 1)
        if self.last_episode is not None and correct is not None:
            self.last_episode.correct = correct
