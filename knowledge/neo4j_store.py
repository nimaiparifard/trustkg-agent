"""Neo4j backend implementing the same interface as InMemoryGraph.

Start a server with `docker compose up -d` (see docker-compose.yml) and use
`Neo4jGraph("bolt://localhost:7687", "neo4j", "trustkg-pass")`. The main
experiments use InMemoryGraph (no server needed); tests/test_neo4j.py checks
that this backend gives the same agent behaviour (skipped without a server).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .schema import Fact


class Neo4jGraph:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        from neo4j import GraphDatabase  # imported lazily: optional dependency

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database
        self.entity_type: Dict[str, str] = {}
        self._run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        rows = self._run("MATCH ()-[r:FACT]->() RETURN coalesce(max(r.fid), -1) AS m")
        self._next = int(rows[0]["m"]) + 1       # monotone integer fact ids (same numbering as InMemoryGraph)

    def _run(self, query: str, **params):
        with self._driver.session(database=self._db) as s:
            return list(s.run(query, **params))

    def close(self) -> None:
        self._driver.close()

    def clear(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")

    def load_from(self, g) -> None:
        """Bulk-load an InMemoryGraph (entities + facts, preserving fact order/ids)."""
        ents = [{"n": n, "t": t} for n, t in g.entity_type.items()]
        self._run("UNWIND $rows AS x MERGE (e:Entity {name:x.n}) SET e.type=x.t", rows=ents)
        self.entity_type.update(g.entity_type)
        rows = []
        for f in g.facts(valid_only=False):
            rows.append({"s": f.subject, "o": f.object, "r": f.relation, "src": f.source, "c": f.confidence,
                         "ts": f.timestamp, "lv": f.last_verified, "v": f.valid, "fid": self._next})
            self._next += 1
        for i in range(0, len(rows), 500):
            self._run("UNWIND $rows AS x MATCH (a:Entity {name:x.s}), (b:Entity {name:x.o}) "
                      "CREATE (a)-[:FACT {relation:x.r, source:x.src, confidence:x.c, timestamp:x.ts, "
                      "last_verified:x.lv, valid:x.v, fid:x.fid}]->(b)", rows=rows[i:i + 500])

    # ---- entities -------------------------------------------------------
    def add_entity(self, name: str, etype: str) -> None:
        self._run("MERGE (e:Entity {name:$n}) SET e.type=$t", n=name, t=etype)
        self.entity_type[name] = etype

    def has_entity(self, name: str) -> bool:
        return bool(self._run("MATCH (e:Entity {name:$n}) RETURN e LIMIT 1", n=name))

    def entities(self, etype: Optional[str] = None) -> List[str]:
        q = "MATCH (e:Entity) WHERE $t IS NULL OR e.type=$t RETURN e.name AS n"
        return [r["n"] for r in self._run(q, t=etype)]

    # ---- facts ----------------------------------------------------------
    def add_fact(self, fact: Fact) -> int:
        q = (
            "MERGE (a:Entity {name:$s}) MERGE (b:Entity {name:$o}) "
            "CREATE (a)-[r:FACT {relation:$r, source:$src, confidence:$c, timestamp:$ts, "
            "last_verified:$lv, valid:true, fid:$fid}]->(b)"
        )
        fid, self._next = self._next, self._next + 1
        self._run(q, s=fact.subject, o=fact.object, r=fact.relation, src=fact.source,
                  c=fact.confidence, ts=fact.timestamp, lv=fact.last_verified, fid=fid)
        return fid

    def get(self, subject: str, relation: str, include_invalid: bool = False) -> List[Fact]:
        q = (
            "MATCH (a:Entity {name:$s})-[r:FACT {relation:$r}]->(b:Entity) "
            "WHERE $inv OR r.valid RETURN a.name AS s, b.name AS o, r ORDER BY r.fid"
        )
        out = []
        for row in self._run(q, s=subject, r=relation, inv=include_invalid):
            r = row["r"]
            out.append(Fact(row["s"], relation, row["o"], r["source"], r["confidence"],
                            r["timestamp"], r["last_verified"], r["valid"], r["fid"]))
        return out

    def deprecate(self, fid) -> None:
        self._run("MATCH ()-[r:FACT {fid:$f}]->() SET r.valid=false", f=fid)

    def mark_verified(self, fid, stage: int, confidence: Optional[float] = None) -> None:
        self._run("MATCH ()-[r:FACT {fid:$f}]->() SET r.last_verified=$s, "
                  "r.confidence=CASE WHEN $c IS NULL OR r.confidence>$c THEN r.confidence ELSE $c END",
                  f=fid, s=stage, c=confidence)

    def facts(self, valid_only: bool = True) -> Iterable[Fact]:
        q = ("MATCH (a)-[r:FACT]->(b) WHERE NOT $v OR r.valid "
             "RETURN a.name AS s, r.relation AS rel, b.name AS o, r ORDER BY r.fid")
        for row in self._run(q, v=valid_only):
            r = row["r"]
            yield Fact(row["s"], row["rel"], row["o"], r["source"], r["confidence"],
                       r["timestamp"], r["last_verified"], r["valid"], r["fid"])

    def num_facts(self, valid_only: bool = True) -> int:
        return sum(1 for _ in self.facts(valid_only))
