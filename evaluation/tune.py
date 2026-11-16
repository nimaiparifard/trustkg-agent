"""Tune the abstention threshold on held-out *dev seeds* (never the test seeds).

Objective: selective-classification utility = P(correct) - wrong_cost * P(confidently wrong),
averaged over all KG-defect conditions, for the verification-only agent.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np

from agents.systems import SYSTEMS
from environments.dynamic_kg import EnvConfig

from .benchmarks import make_env
from .experiments import run_agent


def tune_threshold(base: EnvConfig, conditions: List[str], dev_seeds: List[int], llm_cfg: dict, wrong_cost: float = 2.0,
                   grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)) -> Dict:
    rows = []
    for th in grid:
        cfg = replace(SYSTEMS["trustkg_verify"], threshold=th)
        per_cond = {}
        for cond in conditions:
            ms = [run_agent(cfg, make_env(base, cond, s), s, llm_cfg) for s in dev_seeds]
            per_cond[cond] = dict(success=float(np.mean([m["task_success"] for m in ms])),
                                  halluc=float(np.mean([m["hallucination_rate"] for m in ms])))
        util = float(np.mean([v["success"] - wrong_cost * v["halluc"] for v in per_cond.values()]))
        rows.append(dict(threshold=th, utility=util, per_condition=per_cond))
    best = max(rows, key=lambda r: r["utility"])
    return dict(best=best["threshold"], rows=rows)
