"""Benchmark scenarios = environment configurations (defects x drift) x seeds."""
from __future__ import annotations

import json
from typing import Dict, List

from environments.dynamic_kg import CONDITIONS, DynamicKGEnvironment, EnvConfig, condition_config


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_env(base: EnvConfig, condition: str, seed: int) -> DynamicKGEnvironment:
    """One benchmark instance: a fixed world trajectory + task stream for (condition, seed)."""
    return DynamicKGEnvironment(condition_config(base, condition), seed)


def describe(env: DynamicKGEnvironment) -> Dict:
    tasks = [t for st in env.tasks for t in st]
    cats: Dict[str, int] = {}
    for t in tasks:
        cats[t.category] = cats.get(t.category, 0) + 1
    return {"n_tasks": len(tasks), "categories": cats, "kg_facts": env.kg0.num_facts(),
            "corpus_docs": len(env.corpus0), "drift_from_stage0": env.drift_stats()}


__all__ = ["CONDITIONS", "load_config", "make_env", "describe"]
