from agents.executor import AgentConfig
from agents.planner import Planner
from agents.systems import ABLATIONS, SYSTEMS, build_agent
from agents.verifier import Verifier
from environments.dynamic_kg import DynamicKGEnvironment, EnvConfig, question_text
from evaluation.experiments import run_agent
from knowledge.schema import Evidence

SMALL = EnvConfig(counts={"person": 60, "company": 20, "city": 12, "country": 6, "bloc": 3},
                  n_stages=2, tasks_per_stage=15, focus_persons=20, new_persons_per_stage=3)


def test_planner_parse_roundtrip():
    q = question_text("Person_0007", ["works_for", "headquartered_in", "located_in"])
    assert Planner().parse(q) == ("Person_0007", ["works_for", "headquartered_in", "located_in"])
    plan = Planner().plan(q, lambda r: 0.1)
    assert plan.edges() == [(0, 1), (1, 2)] and 0 < plan.uncertainty < 1


def test_environment_is_deterministic_and_drifts():
    a, b = DynamicKGEnvironment(SMALL, 3), DynamicKGEnvironment(SMALL, 3)
    assert [t.text for t in a.tasks[1]] == [t.text for t in b.tasks[1]]
    assert a.drift_stats()[0] == 0 and a.drift_stats()[-1] > 0


def test_verifier_conflict_lowers_confidence_and_picks_higher_support():
    v = Verifier(lambda r: 0.0, freshness=False)
    good = Evidence("A", "works_for", "X", "kg", 0.9)
    bad = Evidence("A", "works_for", "Y", "kg", 0.5)
    c = v.rank_candidates([bad, good], now=0)
    assert c[0].obj == "X" and c[0].confidence < 0.9 * 0.99      # penalised by the rival
    assert Verifier(lambda r: 0.0, freshness=False).rank_candidates([good], 0)[0].confidence == 0.9


def test_freshness_decays_confidence_with_age():
    v = Verifier(lambda r: 0.2, freshness=True)
    e = Evidence("A", "works_for", "X", "kg", 0.9, last_verified=0)
    assert v.effective_confidence(e, 3) < v.effective_confidence(e, 1) < 0.9


def test_all_systems_run_and_metrics_are_sane():
    env = DynamicKGEnvironment(SMALL, 0)
    for cfg in list(SYSTEMS.values()) + list(ABLATIONS.values()):
        m = run_agent(cfg, env, 0, {})
        assert 0 <= m["task_success"] <= 1 and 0 <= m["hallucination_rate"] <= 1
        assert abs(m["task_success"] + m["hallucination_rate"] + m["abstain_rate"] - 1) < 1e-9


def test_abstains_when_no_evidence_and_never_uses_llm():
    env = DynamicKGEnvironment(SMALL, 0)
    agent = build_agent(SYSTEMS["trustkg_verify"], env, 0)
    res = agent.solve(question_text("Person_9999", ["works_for", "headquartered_in"]), 0)   # unknown entity
    assert res.abstained and res.answer is None and res.llm_calls == 0


def test_memory_learns_from_feedback_and_updates_kg():
    env = DynamicKGEnvironment(SMALL, 0)
    agent = build_agent(SYSTEMS["trustkg_full"], env, 0)
    q = question_text("Person_9999", ["works_for"])
    assert agent.solve(q, 1).abstained
    agent.feedback([("Person_9999", "works_for", "Company_0001")], 1, False)
    res = agent.solve(q, 1)
    assert res.answer == "Company_0001" and not res.abstained
    assert agent.memory.learned["new"] == 1


def test_systems_do_not_share_mutable_state():
    env = DynamicKGEnvironment(SMALL, 0)
    a1 = build_agent(SYSTEMS["trustkg_full"], env, 0)
    a1.feedback([("Person_9999", "works_for", "Company_0001")], 1, None)
    a2 = build_agent(SYSTEMS["trustkg_full"], env, 0)
    assert a2.solve(question_text("Person_9999", ["works_for"]), 1).abstained
