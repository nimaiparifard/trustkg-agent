"""Metrics computed from per-task records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class TaskRecord:
    stage: int
    category: str            # stable | changed | novel
    hops: int
    answered: bool
    correct: bool            # answered and equal to gold
    supported: bool          # answered and every hop backed by retrieved evidence
    evidence_total: int      # cited kg/text evidence items
    evidence_true: int       # ... that are true in the world at that stage
    tool_calls: int
    llm_calls: int
    failure_events: int
    recovery_attempted: bool
    plan_correct: bool = True   # planner recovered the gold (start, relations)


def _rate(num: float, den: float) -> Optional[float]:
    return None if den == 0 else num / den


def compute_metrics(rs: Sequence[TaskRecord], n_stages: int) -> Dict[str, Optional[float]]:
    n = len(rs)
    answered = [r for r in rs if r.answered]
    wrong = [r for r in answered if not r.correct]
    post = [r for r in rs if r.stage >= 1]
    fail = [r for r in rs if r.failure_events > 0]
    ev_tot = sum(r.evidence_total for r in answered)

    def success(sub):  # task success = answered correctly
        return _rate(sum(r.correct for r in sub), len(sub))

    stage_success = [success([r for r in rs if r.stage == s]) for s in range(n_stages + 1)]
    stage_halluc = []
    for s in range(n_stages + 1):
        sub = [r for r in rs if r.stage == s]
        stage_halluc.append(_rate(sum(r.answered and not r.correct for r in sub), len(sub)))
    return {
        "task_success": success(rs),
        "reasoning_accuracy": _rate(sum(r.correct for r in answered), len(answered)),   # selective accuracy
        "coverage": _rate(len(answered), n),
        "abstain_rate": _rate(n - len(answered), n),
        "hallucination_rate": _rate(len(wrong), n),                                   # confident wrong answers
        "unsupported_rate": _rate(sum(r.answered and not r.supported for r in rs), n),
        "evidence_precision": _rate(sum(r.evidence_true for r in answered), ev_tot) if ev_tot else None,
        "recovery_rate": _rate(sum(r.correct for r in fail), len(fail)) if fail else None,
        "failure_task_rate": _rate(len(fail), n),
        # of tasks hitting incomplete/contradictory KG evidence: fraction NOT answered wrongly
        "failure_containment": _rate(sum(not (r.answered and not r.correct) for r in fail), len(fail)) if fail else None,
        "plan_accuracy": float(np.mean([r.plan_correct for r in rs])),
        "tool_calls": float(np.mean([r.tool_calls for r in rs])),
        "llm_calls": float(np.mean([r.llm_calls for r in rs])),
        # adaptation under drift
        "post_shift_success": success(post),
        "post_shift_hallucination": _rate(sum(r.answered and not r.correct for r in post), len(post)),
        "final_stage_success": stage_success[-1],
        "retention": success([r for r in post if r.category == "stable"]),
        "plasticity": success([r for r in post if r.category == "changed"]),
        "novel_success": success([r for r in post if r.category == "novel"]),
        "stage_success": stage_success,
        "stage_hallucination": stage_halluc,
    }


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0):
    """Mean paired difference (a-b) with 95% bootstrap CI over seeds, and P(diff<=0)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), \
        float((boots <= 0).mean())
