"""Task understanding & planning: question -> Task DAG with per-step uncertainty."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from knowledge.schema import RELATIONS, phrase


@dataclass
class PlanStep:
    idx: int
    relation: str
    depends_on: Optional[int]       # index of the step whose output is this step's input
    uncertainty: float = 0.0        # estimated P(the fact used here is outdated)


@dataclass
class TaskPlan:
    start: str
    steps: List[PlanStep]
    uncertainty: float = 0.0
    prior_success: Optional[float] = None   # from episodic memory (same relation path)

    @property
    def relations(self) -> List[str]:
        return [s.relation for s in self.steps]

    def edges(self) -> List[Tuple[int, int]]:
        return [(s.depends_on, s.idx) for s in self.steps if s.depends_on is not None]

    def topological_order(self) -> List[PlanStep]:
        return list(self.steps)     # the DAG of an n-hop question is a chain

    def suffix(self, idx: int) -> List[PlanStep]:
        """Steps that must be re-planned if step `idx` is replaced (its descendants)."""
        return [s for s in self.steps if s.idx >= idx]


class Planner:
    def parse(self, question: str) -> Tuple[str, List[str]]:
        """'What is the A of the B of X?' -> (X, [rel_B, rel_A]) in execution order."""
        s = question.strip().rstrip("?")
        if s.startswith("What is "):
            s = s[len("What is "):]
        outer: List[str] = []
        progressed = True
        while progressed:
            progressed = False
            for rel in RELATIONS:
                p = phrase(rel) + " "
                if s.startswith(p):
                    outer.append(rel)
                    s = s[len(p):]
                    progressed = True
                    break
        return s, list(reversed(outer))

    def plan(self, question: str, hazard_of: Callable[[str], float],
             prior_success: Optional[float] = None) -> TaskPlan:
        start, rels = self.parse(question)
        steps = [PlanStep(i, r, i - 1 if i else None, hazard_of(r)) for i, r in enumerate(rels)]
        keep = 1.0
        for st in steps:
            keep *= 1.0 - st.uncertainty
        return TaskPlan(start, steps, 1.0 - keep, prior_success)


class LLMPlanner(Planner):
    """Question understanding by a real LLM (natural-language question -> relation chain).

    An invalid / unparsable plan yields an empty chain; the executor then
    abstains (we deliberately do NOT fall back to the rule-based parser).
    """

    def __init__(self, llm) -> None:
        self.llm = llm

    def parse(self, question: str) -> Tuple[str, List[str]]:
        out = self.llm.plan(question)
        return out if out else ("", [])
