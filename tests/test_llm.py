"""OllamaLLM post-processing and LLM-driven planner/reader, with the HTTP call stubbed."""
from agents.executor import AgentConfig
from agents.llm import OllamaLLM
from agents.systems import SYSTEMS, build_agent
from environments.dynamic_kg import DynamicKGEnvironment, EnvConfig, nl_question

SMALL = EnvConfig(counts={"person": 60, "company": 20, "city": 12, "country": 6, "bloc": 3},
                  n_stages=1, tasks_per_stage=10, focus_persons=20, new_persons_per_stage=3)


class Stub(OllamaLLM):
    def __init__(self, reply):
        super().__init__(cache_path=None)
        self.reply = reply

    def _generate(self, prompt, num_predict=40):
        return self.reply(prompt) if callable(self.reply) else self.reply


def test_plan_snaps_relation_names_and_checks_types():
    q = "Which bloc does the country of the city where the firm of Person_0001 sits belong to? Person_0001"
    ok = Stub("Person_0001 | works_for, headquarters_in, located_in, member_of").plan(q)   # misspelled relation
    assert ok == ("Person_0001", ["works_for", "headquartered_in", "located_in", "member_of"])
    assert Stub("Company_0001 | works_for, headquartered_in").plan("Company_0001?") is None   # type-inconsistent
    assert Stub("START | works_for").plan("Person_0001?") is None                              # no entity
    assert Stub("Person_0009 | works_for").plan("about Person_0001") is None                   # entity not in question
    assert Stub("no idea").plan("Person_0001?") is None


def test_reader_must_be_grounded_in_passages():
    chunks = ["Person_0001 works for Company_0002."]
    assert Stub("Company_0002").read("Person_0001", "works_for", chunks) == "Company_0002"
    assert Stub("Company_0009").read("Person_0001", "works_for", chunks) is None     # hallucinated
    assert Stub("NONE").read("Person_0001", "works_for", chunks) is None
    assert Stub("Company_0002").read("Person_0001", "works_for", []) is None


def test_llm_that_declines_means_abstention():
    env = DynamicKGEnvironment(SMALL, 0)
    t = env.tasks[0][0]
    stub = Stub("UNKNOWN")
    res = build_agent(SYSTEMS["llm_only"], env, 0, llm=stub).solve("What is x of Person_0001?", 0)
    assert res.abstained and res.answer is None


def test_failed_plan_abstains_and_good_plan_answers_from_kg():
    env = DynamicKGEnvironment(SMALL, 0)
    t = env.tasks[0][0]
    bad = build_agent(SYSTEMS["trustkg_verify"], env, 0, llm=Stub("garbage")).solve(t.nl_text, 0)
    assert bad.abstained and "PLAN FAILED -> abstain" in bad.trace
    plan_reply = f"{t.start} | {', '.join(t.rels)}"
    good = build_agent(SYSTEMS["trustkg_verify"], env, 0, llm=Stub(plan_reply)).solve(t.nl_text, 0)
    assert good.plan.relations == t.rels and good.llm_calls == 0


def test_nl_questions_are_deterministic_and_keep_entity():
    env = DynamicKGEnvironment(SMALL, 0)
    t = env.tasks[0][0]
    assert t.nl_text == nl_question(t.tid, t.start, t.rels) and t.start in t.nl_text and t.nl_text != t.text
