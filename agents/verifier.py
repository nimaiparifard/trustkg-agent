"""Trust & verification: evidence scoring, conflict handling, claim checking, abstention."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from knowledge.schema import Evidence

SUPPORTED, CONTRADICTED, UNSUPPORTED = "supported", "contradicted", "unsupported"


@dataclass
class Candidate:
    obj: str
    confidence: float            # step confidence (after conflict penalty)
    evidence: List[Evidence]
    score: float = 0.0           # support score before conflict penalty


@dataclass
class ClaimVerdict:
    claim: Tuple[str, str, str]
    status: str


class Verifier:
    def __init__(self, hazard_of: Callable[[str], float], freshness: bool = True,
                 conflict_resolution: bool = True, threshold: float = 0.5) -> None:
        self.hazard_of, self.freshness = hazard_of, freshness
        self.conflict_resolution, self.threshold = conflict_resolution, threshold

    # --- evidence -> confidence -------------------------------------------
    def effective_confidence(self, ev: Evidence, now: int) -> float:
        """Source confidence discounted by the chance the fact changed since last verified."""
        if not self.freshness or ev.kind != "kg":
            return ev.confidence
        age = max(now - ev.last_verified, 0)
        return ev.confidence * (1.0 - self.hazard_of(ev.relation)) ** age

    def rank_candidates(self, evidence: Sequence[Evidence], now: int) -> List[Candidate]:
        """Group evidence by answer; combine support with noisy-OR; penalise conflict."""
        if not evidence:
            return []
        if not self.conflict_resolution:
            # naive resolution: most recent assertion wins, no conflict penalty
            ev = max(evidence, key=lambda e: e.timestamp)   # ties: first retrieved
            c = self.effective_confidence(ev, now)
            return [Candidate(ev.object, c, [ev], c)]
        groups: Dict[str, List[Evidence]] = defaultdict(list)
        for e in evidence:
            groups[e.object].append(e)
        scores = {}
        for o, evs in groups.items():
            miss = 1.0
            for e in evs:
                miss *= 1.0 - self.effective_confidence(e, now)
            scores[o] = 1.0 - miss
        out = []
        for o, sc in scores.items():
            rival = 1.0
            for o2, sc2 in scores.items():
                if o2 != o:
                    rival *= 1.0 - sc2
            out.append(Candidate(o, sc * rival, groups[o], sc))
        return sorted(out, key=lambda c: -c.confidence)

    # --- claim-level verification -----------------------------------------
    def verify_claims(self, claims: Sequence[Tuple[str, str, str]],
                      evidence: Sequence[Evidence], now: int) -> List[ClaimVerdict]:
        """Check every atomic claim (hop) against the retrieved evidence."""
        verdicts = []
        for s, r, o in claims:
            rel = [e for e in evidence if e.subject == s and e.relation == r]
            mine = sum(self.effective_confidence(e, now) for e in rel if e.object == o and e.kind != "llm")
            other = max([self.effective_confidence(e, now) for e in rel if e.object != o] or [0.0])
            if mine <= 0:
                verdicts.append(ClaimVerdict((s, r, o), UNSUPPORTED))
            elif other > mine:
                verdicts.append(ClaimVerdict((s, r, o), CONTRADICTED))
            else:
                verdicts.append(ClaimVerdict((s, r, o), SUPPORTED))
        return verdicts

    def accept(self, confidence: float, verdicts: Sequence[ClaimVerdict]) -> bool:
        return confidence >= self.threshold and all(v.status == SUPPORTED for v in verdicts)
