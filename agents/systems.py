"""System registry (baselines, full system, ablations) and agent factory."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

from environments.dynamic_kg import DynamicKGEnvironment
from knowledge.graph_retriever import GraphRetriever, HybridRetriever, VectorRetriever
from knowledge.graph_updater import GraphUpdater

from .executor import Agent, AgentConfig, Executor
from .llm import OllamaLLM, SimulatedLLM
from .memory import AdaptiveMemory
from .planner import LLMPlanner, Planner
from .verifier import Verifier

_FULL = AgentConfig("trustkg_full", "kg", verify=True, memory=True)

# Main comparison (Baseline, Experiments 1-5)
SYSTEMS: Dict[str, AgentConfig] = {
    "llm_only": AgentConfig("llm_only", "direct"),
    "react": AgentConfig("react", "react"),
    "react_vector_rag": AgentConfig("react_vector_rag", "rag", fallback_llm=True),
    "react_kg": AgentConfig("react_kg", "kg", fallback_llm=True),
    "trustkg_verify": AgentConfig("trustkg_verify", "kg", verify=True),
    "trustkg_full": _FULL,
}

# Ablations of the full architecture (one component removed at a time)
ABLATIONS: Dict[str, AgentConfig] = {
    "full": _FULL,
    "-memory": replace(_FULL, name="-memory", memory=False),
    "-conflict_resolution": replace(_FULL, name="-conflict_resolution", conflict_resolution=False),
    "-freshness": replace(_FULL, name="-freshness", freshness=False),
    "-recovery": replace(_FULL, name="-recovery", recovery=False),
    "-backtracking": replace(_FULL, name="-backtracking", beam=1),
    "-abstention": replace(_FULL, name="-abstention", abstain=False),
}


def build_agent(cfg: AgentConfig, env: DynamicKGEnvironment, seed: int = 0,
                p_know: float = 0.55, shortcut_fail: float = 0.15, llm=None, graph=None) -> Agent:
    """`llm` given (e.g. OllamaLLM) -> real-LLM mode: the LLM plans (natural-language questions) and
    reads retrieved text; otherwise the SimulatedLLM + rule-based planner/reader are used."""
    graph = graph if graph is not None else env.kg0.copy()   # each system gets its own mutable KG
    if getattr(env, "_vector", None) is None:
        env._vector = VectorRetriever(env.corpus0)   # read-only; shared across systems
    real = llm is not None
    retriever = HybridRetriever(GraphRetriever(graph), env._vector, reader=llm.read if real else None)
    llm = llm or SimulatedLLM(env.truths[0], env.entities0, p_know, shortcut_fail, seed)
    planner = LLMPlanner(llm) if real else Planner()
    memory = AdaptiveMemory(GraphUpdater(graph)) if cfg.memory else None
    if cfg.verify:
        hazard = memory.hazard if memory else (lambda r: 0.1)
        verifier = Verifier(hazard, cfg.freshness, cfg.conflict_resolution, cfg.threshold)
    else:  # baselines: naive resolution (most recent assertion), no freshness handling
        verifier = Verifier(lambda r: 0.0, freshness=False, conflict_resolution=False)
    return Agent(Executor(cfg, llm, retriever, verifier, planner, memory))
