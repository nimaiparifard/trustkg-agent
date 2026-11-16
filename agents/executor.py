"""Agent execution: ReAct-style loop over the plan DAG with tool use.

Tools: kg_lookup (graph retrieval), text_search (vector retrieval), llm (parametric).
Variants differ only through AgentConfig flags so ablations are exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from knowledge.graph_retriever import HybridRetriever
from knowledge.schema import Evidence

from .memory import AdaptiveMemory, Episode
from .planner import Planner, TaskPlan
from .verifier import Candidate, Verifier


@dataclass
class AgentConfig:
    name: str
    mode: str                       # 'direct' | 'react' | 'rag' | 'kg'
    verify: bool = False            # evidence-based verification + abstention logic
    conflict_resolution: bool = True
    freshness: bool = True
    recovery: bool = True           # text cross-check when KG is incomplete/contradictory
    abstain: bool = True
    memory: bool = False
    fallback_llm: bool = False      # guess from parametric knowledge when retrieval fails
    threshold: float = 0.15         # tuned on dev seeds (evaluation/tune.py), not test seeds
    beam: int = 2                   # candidates explored per step (backtracking)


@dataclass
class AgentResult:
    answer: Optional[str] = None
    abstained: bool = False
    confidence: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)
    tool_calls: int = 0
    llm_calls: int = 0
    failure_events: int = 0
    recovery_attempted: bool = False
    plan: Optional[TaskPlan] = None
    trace: List[str] = field(default_factory=list)

    def fully_supported(self, hops: int) -> bool:
        return len(self.evidence) >= hops and all(e.kind != "llm" for e in self.evidence)


class Executor:
    def __init__(self, cfg: AgentConfig, llm, retriever: HybridRetriever, verifier: Verifier,
                 planner: Planner, memory: Optional[AdaptiveMemory] = None) -> None:
        self.cfg, self.llm, self.retriever = cfg, llm, retriever
        self.verifier, self.planner, self.memory = verifier, planner, memory

    # ------------------------------------------------------------------ tools
    def _hazard(self, rel: str) -> float:
        return self.memory.hazard(rel) if self.memory else 0.1

    def _kg(self, s: str, r: str, res: AgentResult) -> List[Evidence]:
        res.tool_calls += 1
        res.trace.append(f"ACTION kg_lookup({s}, {r})")
        return self.retriever.kg(s, r)

    def _text(self, s: str, r: str, res: AgentResult) -> List[Evidence]:
        res.tool_calls += 1
        res.trace.append(f"ACTION text_search({s}, {r})")
        return self.retriever.text(s, r)

    def _llm_hop(self, s: str, r: str, res: AgentResult) -> Evidence:
        res.llm_calls += 1
        res.trace.append(f"ACTION llm({s}, {r})")
        return Evidence(s, r, self.llm.answer_hop(s, r), "llm", 0.5)

    # -------------------------------------------------------------------- run
    def run(self, question: str, now: int) -> AgentResult:
        cfg = self.cfg
        start, rels = self.planner.parse(question)
        prior = self.memory.episodic.success_rate(rels) if self.memory else None
        plan = self.planner.plan(question, self._hazard, prior)
        res = AgentResult(plan=plan)
        res.trace.append(f"PLAN {start} -> {' > '.join(rels)}  (uncertainty={plan.uncertainty:.2f})")
        if not plan.steps:                                   # planner could not understand the question
            res.trace.append("PLAN FAILED -> abstain")
            res.abstained = True
            return res
        if cfg.mode == "direct":
            res.llm_calls += 1
            res.answer = self.llm.answer_direct(start, rels, question)
            res.abstained = res.answer is None               # real LLM may decline (UNKNOWN)
            return res
        if cfg.mode == "react":
            cur = start
            for st in plan.topological_order():
                ev = self._llm_hop(cur, st.relation, res)
                if ev.object is None:                        # LLM declined
                    res.abstained = True
                    return res
                res.evidence.append(ev)
                cur = ev.object
            res.answer = cur
            return res
        if cfg.verify:
            return self._run_verified(plan, now, res)
        return self._run_naive(plan, now, res)

    # ------------------------------------------------- baselines w/o verification
    def _run_naive(self, plan: TaskPlan, now: int, res: AgentResult) -> AgentResult:
        cur = plan.start
        for st in plan.topological_order():
            ev = (self._text if self.cfg.mode == "rag" else self._kg)(cur, st.relation, res)
            if len({e.object for e in ev}) != 1:
                res.failure_events += 1
            if not ev:
                if not self.cfg.fallback_llm:
                    res.abstained = True
                    return res
                chosen = self._llm_hop(cur, st.relation, res)
                if chosen.object is None:
                    res.abstained = True
                    return res
            else:
                cand = self.verifier.rank_candidates(ev, now)[0]   # naive: most recent / first
                chosen = cand.evidence[0]
            res.evidence.append(chosen)
            cur = chosen.object
        res.answer = cur
        return res

    # --------------------------------------------- evidence-grounded verified path
    def _step_candidates(self, s: str, r: str, now: int, res: AgentResult, cache: Dict):
        if (s, r) in cache:
            return cache[(s, r)]
        ev = self._kg(s, r, res)
        if not ev or len({e.object for e in ev}) > 1:
            res.failure_events += 1                       # incomplete or contradictory
            if self.cfg.recovery:
                res.recovery_attempted = True
                res.trace.append("RECOVER cross-check with text evidence")
                ev = ev + self._text(s, r, res)
        cache[(s, r)] = (self.verifier.rank_candidates(ev, now), ev)
        return cache[(s, r)]

    def _run_verified(self, plan: TaskPlan, now: int, res: AgentResult) -> AgentResult:
        rels, cfg = plan.relations, self.cfg
        cache: Dict = {}
        best: List = [None]   # (conf, answer, evidence, claims, all_evidence)

        def expand(i, subject, chain_ev, all_ev, conf, claims):
            if i == len(rels):
                if best[0] is None or conf > best[0][0]:
                    best[0] = (conf, subject, chain_ev, claims, all_ev)
                return
            cands, ev_all = self._step_candidates(subject, rels[i], now, res, cache)
            for cand in cands[: cfg.beam]:
                c = conf * cand.confidence
                if c < 0.1:
                    continue
                expand(i + 1, cand.obj, chain_ev + cand.evidence[:1], all_ev + ev_all, c,
                       claims + [(subject, rels[i], cand.obj)])

        expand(0, plan.start, [], [], 1.0, [])
        if best[0] is None:
            res.trace.append("NO EVIDENCE CHAIN")
            if cfg.abstain:
                res.abstained = True
            else:
                res.llm_calls += 1
                res.answer = self.llm.answer_direct(plan.start, rels, None)
                res.abstained = res.answer is None
            return res
        conf, answer, chain_ev, claims, all_ev = best[0]
        verdicts = self.verifier.verify_claims(claims, all_ev, now)
        ok = self.verifier.accept(conf, verdicts)
        res.confidence, res.evidence = conf, chain_ev
        res.trace.append(f"VERIFY conf={conf:.2f} claims={[v.status for v in verdicts]} accept={ok}")
        if ok or not cfg.abstain:
            res.answer = answer
        else:
            res.abstained = True
        return res


class Agent:
    """Executor + memory bookkeeping + feedback ingestion."""

    def __init__(self, executor: Executor) -> None:
        self.executor = executor
        self.memory = executor.memory
        self.name = executor.cfg.name

    def solve(self, question: str, now: int) -> AgentResult:
        res = self.executor.run(question, now)
        if self.memory:
            _, rels = self.executor.planner.parse(question)
            self.memory.record_episode(Episode(now, question, tuple(rels), res.answer, res.abstained,
                                               res.failure_events, res.confidence))
        return res

    def feedback(self, chain, now: int, correct: Optional[bool]) -> None:
        if self.memory and chain:
            self.memory.ingest_feedback(chain, now, correct)
