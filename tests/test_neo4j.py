"""Integration test: the Neo4j backend must reproduce the in-memory agent behaviour.
Skipped automatically when no Neo4j server is reachable (docker compose up -d)."""
import pytest

from agents.systems import SYSTEMS
from environments.dynamic_kg import DynamicKGEnvironment, EnvConfig
from evaluation.experiments import run_agent

SMALL = EnvConfig(counts={"person": 80, "company": 25, "city": 15, "country": 8, "bloc": 4},
                  n_stages=3, tasks_per_stage=25, focus_persons=30, new_persons_per_stage=4)


@pytest.fixture(scope="module")
def neo():
    try:
        from knowledge.neo4j_store import Neo4jGraph
        g = Neo4jGraph("bolt://localhost:7687", "neo4j", "trustkg-pass")
        g._run("RETURN 1")
    except Exception as e:  # no server / driver
        pytest.skip(f"Neo4j not available: {type(e).__name__}")
    yield g
    g.clear()
    g.close()


@pytest.mark.parametrize("name", ["react_kg", "trustkg_verify", "trustkg_full"])
def test_neo4j_matches_in_memory(neo, name):
    env = DynamicKGEnvironment(SMALL, 1)
    mem = run_agent(SYSTEMS[name], env, 1, {})
    neo.clear()
    neo._next = 0
    neo.load_from(env.kg0)
    assert neo.num_facts() == env.kg0.num_facts()
    on_neo = run_agent(SYSTEMS[name], env, 1, {}, graph=neo)
    for k in ["task_success", "hallucination_rate", "abstain_rate", "evidence_precision", "stage_success"]:
        assert on_neo[k] == mem[k], k
