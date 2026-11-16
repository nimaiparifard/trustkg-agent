"""Dynamic-KG environment for studying reliability and adaptation.

A ground-truth world (~1000 entities) evolves over stages: relations change
(job changes, HQ moves, bloc changes), new entities appear, and new tasks are
issued. The agent only sees a stage-0 snapshot (a noisy KG and a noisy text
corpus) and must cope with incomplete, contradictory and outdated knowledge.
After some tasks the environment returns feedback (the true fact chain).
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

from knowledge.graph_builder import Truth, build_corpus, build_kg, build_world, make_names, wrong_object
from knowledge.schema import CHAIN, RELATIONS, object_type, phrase, subject_type

DEFAULT_HAZARDS = {"works_for": 0.20, "headquartered_in": 0.10, "located_in": 0.0, "member_of": 0.05}


@dataclass
class EnvConfig:
    counts: Dict[str, int] = field(default_factory=lambda: {
        "person": 600, "company": 200, "city": 120, "country": 60, "bloc": 20})
    n_stages: int = 5                      # stages 0..n_stages (stage 0 = no drift yet)
    tasks_per_stage: int = 60
    hazards: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HAZARDS))
    drift_scale: float = 1.0
    new_persons_per_stage: int = 20
    focus_persons: int = 150               # persons that are repeatedly queried
    kg_missing: float = 0.08               # incomplete
    kg_conflict: float = 0.06              # contradictory
    text_missing: float = 0.15
    text_conflict: float = 0.06
    feedback_rate: float = 0.3

    @staticmethod
    def from_dict(d: dict) -> "EnvConfig":
        base = EnvConfig()
        for k, v in d.items():
            setattr(base, k, v)
        return base


CONDITIONS = {
    # each condition isolates one KG defect; "mixed" combines all of them
    "incomplete":    dict(kg_missing=0.20, kg_conflict=0.0, drift_scale=0.0),
    "contradictory": dict(kg_missing=0.0, kg_conflict=0.20, drift_scale=0.0),
    "stale":         dict(kg_missing=0.0, kg_conflict=0.0, drift_scale=2.0),
    "mixed":         dict(),
}


def condition_config(base: EnvConfig, name: str) -> EnvConfig:
    cfg = replace(base, hazards=dict(base.hazards))
    for k, v in CONDITIONS[name].items():
        setattr(cfg, k, v)
    return cfg


@dataclass
class Task:
    tid: str
    stage: int
    text: str
    start: str
    rels: List[str]
    gold: str
    chain: List[Tuple[str, str, str]]     # true (s, r, o) facts at this stage
    category: str                         # 'stable' | 'changed' | 'novel'
    feedback: bool                        # whether feedback is returned after the task
    nl_text: str = ""                      # natural-language paraphrase (used with a real LLM planner)

    @property
    def hops(self) -> int:
        return len(self.rels)


def question_text(start: str, rels: List[str]) -> str:
    s = start
    for r in rels:
        s = f"{phrase(r)} {s}"
    return f"What is {s}?"


_NL = {
    "works_for": ["the company that employs {x}", "the organisation {x} works for"],
    "headquartered_in": ["the city where {x} has its headquarters", "the city in which {x} is headquartered"],
    "located_in": ["the country in which {x} is located", "the country that {x} lies in"],
    "member_of": ["the bloc that {x} belongs to", "the trade bloc {x} is a member of"],
}
_NL_FRAME = ["Name {np}.", "Which is {np}?", "Can you tell me {np}?"]


def nl_question(tid: str, start: str, rels: List[str]) -> str:
    """Deterministic natural-language paraphrase (hash-seeded; does not touch the env RNG)."""
    np_ = start
    for r in rels:
        np_ = _NL[r][zlib.crc32(f"{tid}|{r}".encode()) % 2].format(x=np_)
    return _NL_FRAME[zlib.crc32(tid.encode()) % 3].format(np=np_)


class DynamicKGEnvironment:
    def __init__(self, cfg: EnvConfig, seed: int) -> None:
        self.cfg, self.seed = cfg, seed
        rng = random.Random(seed)
        ents0, truth0 = build_world(cfg.counts, rng)
        self.entities0 = {t: list(v) for t, v in ents0.items()}
        self.entities = {t: list(v) for t, v in ents0.items()}   # grows over time
        self.truths: List[Truth] = [truth0]
        self.new_persons: Dict[int, List[str]] = {0: []}
        n_added = 0
        for stage in range(1, cfg.n_stages + 1):
            cur = dict(self.truths[-1])
            for (s, r), o in list(cur.items()):                   # drift
                if rng.random() < cfg.hazards[r] * cfg.drift_scale:
                    cur[(s, r)] = wrong_object(o, r, self.entities, rng)
            new = make_names("person", cfg.counts["person"] + n_added, cfg.new_persons_per_stage)
            n_added += cfg.new_persons_per_stage
            self.entities["person"] += new
            for p in new:
                cur[(p, "works_for")] = rng.choice(self.entities["company"])
            self.new_persons[stage] = new
            self.truths.append(cur)
        self.tasks: List[List[Task]] = [self._make_tasks(st, rng) for st in range(cfg.n_stages + 1)]
        # agent-visible stage-0 knowledge (noisy)
        self.kg0 = build_kg(truth0, self.entities0, random.Random(seed * 7 + 1), cfg.kg_missing, cfg.kg_conflict)
        self.corpus0 = build_corpus(truth0, self.entities0, random.Random(seed * 7 + 2),
                                    cfg.text_missing, cfg.text_conflict)

    # ------------------------------------------------------------------ tasks
    def _follow(self, truth: Truth, start: str, rels: List[str]):
        cur, chain = start, []
        for r in rels:
            o = truth[(cur, r)]
            chain.append((cur, r, o))
            cur = o
        return cur, chain

    def _make_tasks(self, stage: int, rng: random.Random) -> List[Task]:
        cfg, truth, truth0 = self.cfg, self.truths[stage], self.truths[0]
        persons = self.entities0["person"][: cfg.focus_persons] + \
            [p for st in range(1, stage + 1) for p in self.new_persons[st]]
        new_set = {p for st in range(1, stage + 1) for p in self.new_persons[st]}
        tasks = []
        for i in range(cfg.tasks_per_stage):
            u = rng.random()
            if u < 0.6:
                start, first, hops = rng.choice(persons), 0, rng.choice([2, 3, 4])
            elif u < 0.85:
                start, first, hops = rng.choice(self.entities0["company"]), 1, rng.choice([2, 3])
            else:
                start, first, hops = rng.choice(self.entities0["city"]), 2, 2
            rels = CHAIN[first:first + hops]
            gold, chain = self._follow(truth, start, rels)
            changed = any(truth0.get((s, r)) != o for s, r, o in chain)
            cat = "novel" if start in new_set else ("changed" if changed else "stable")
            tasks.append(Task(f"s{stage}t{i}", stage, question_text(start, rels), start, rels, gold, chain,
                              cat, rng.random() < cfg.feedback_rate,
                              nl_question(f"s{stage}t{i}", start, rels)))
        return tasks

    # --------------------------------------------------------------- accessors
    def truth(self, stage: int) -> Truth:
        return self.truths[stage]

    def feedback(self, task: Task) -> List[Tuple[str, str, str]]:
        """Trusted observation returned to the agent after a task."""
        return list(task.chain) if task.feedback else []

    def drift_stats(self) -> List[float]:
        """Fraction of stage-0 facts that differ from the truth at each stage."""
        t0 = self.truths[0]
        return [sum(1 for k, v in t0.items() if tr.get(k) != v) / len(t0) for tr in self.truths]
