import random

from knowledge.graph_builder import build_corpus, build_kg, build_world
from knowledge.graph_retriever import GraphRetriever, VectorRetriever
from knowledge.graph_store import InMemoryGraph
from knowledge.graph_updater import GraphUpdater
from knowledge.schema import Fact, parse_sentence, render_fact

COUNTS = {"person": 30, "company": 10, "city": 6, "country": 4, "bloc": 2}


def test_sentence_roundtrip():
    for r, (s, o) in {"works_for": ("Person_0001", "Company_0002"), "member_of": ("Country_0001", "Bloc_0000")}.items():
        assert parse_sentence(render_fact(s, r, o)) == (s, r, o)


def test_graph_store_deprecate_and_copy():
    g = InMemoryGraph()
    fid = g.add_fact(Fact("a", "works_for", "b"))
    g2 = g.copy()
    g.deprecate(fid)
    assert g.get("a", "works_for") == [] and len(g2.get("a", "works_for")) == 1
    assert len(g.get("a", "works_for", include_invalid=True)) == 1


def test_updater_statuses():
    g = InMemoryGraph()
    u = GraphUpdater(g)
    assert u.upsert("Person_1", "works_for", "Company_1", 1)[0] == "new"
    assert u.upsert("Person_1", "works_for", "Company_1", 3) == ("confirmed", 2)
    status, age = u.upsert("Person_1", "works_for", "Company_2", 5)
    assert status == "changed" and age == 2
    assert [f.object for f in g.get("Person_1", "works_for")] == ["Company_2"]


def test_kg_builder_noise_rates_and_multihop():
    rng = random.Random(0)
    ents, truth = build_world(COUNTS, rng)
    g = build_kg(truth, ents, random.Random(1), missing_rate=0.0, conflict_rate=0.0)
    assert g.num_facts() == len(truth)
    path = GraphRetriever(g).multi_hop("Person_0000", ["works_for", "headquartered_in"])
    assert len(path) == 1 and len(path[0]) == 2
    noisy = build_kg(truth, ents, random.Random(1), missing_rate=0.5, conflict_rate=0.5)
    assert noisy.num_facts() != len(truth)


def test_vector_retriever_finds_subject_sentence():
    rng = random.Random(0)
    ents, truth = build_world(COUNTS, rng)
    docs = build_corpus(truth, ents, random.Random(2), 0.0, 0.0)
    vr = VectorRetriever(docs)
    ev = vr.lookup("Person_0003", "works_for")
    assert ev and ev[0].object == truth[("Person_0003", "works_for")]
