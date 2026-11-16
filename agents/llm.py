"""LLM backends: parametric-knowledge interface used by the agents.

SimulatedLLM models the properties that matter for this study with explicit,
controllable parameters (so effects are attributable to the architecture):
  * parametric knowledge is frozen at the training cutoff (= stage 0 world),
  * it knows only a fraction `p_know` of facts and *hallucinates* a
    type-correct but arbitrary entity for the rest (never abstains),
  * one-shot multi-hop answering skips hops with prob `shortcut_fail`.

OllamaLLM (below) is the real local-model backend (Qwen/Llama via Ollama).
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import random
import re
import urllib.request
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

from knowledge.schema import RELATIONS, TYPE_PREFIX, object_type, phrase, subject_type


class SimulatedLLM:
    def __init__(self, truth0: Dict, entities_by_type: Dict[str, List[str]], p_know: float = 0.55,
                 shortcut_fail: float = 0.15, seed: int = 0) -> None:
        self.truth0, self.ents = truth0, entities_by_type
        self.p_know, self.shortcut_fail, self.seed = p_know, shortcut_fail, seed

    def _rng(self, *key) -> random.Random:   # deterministic across processes
        return random.Random(zlib.crc32(f"{self.seed}|{key}".encode()))

    def knows(self, s: str, r: str) -> bool:
        return (s, r) in self.truth0 and self._rng("know", s, r).random() < self.p_know

    def answer_hop(self, s: str, r: str) -> str:
        if self.knows(s, r):
            return self.truth0[(s, r)]
        return self._rng("halluc", s, r).choice(self.ents[object_type(r)])

    def answer_direct(self, start: str, rels: Sequence[str], question: Optional[str] = None) -> str:
        cur = start
        for i, r in enumerate(rels):
            if i >= 1 and self._rng("short", start, tuple(rels), i).random() < self.shortcut_fail:
                cur = self._rng("shortguess", start, tuple(rels), i).choice(self.ents[object_type(r)])
            else:
                cur = self.answer_hop(cur, r)
        return cur


ENTITY_RX = re.compile(r"(?:Person|Company|City|Country|Bloc)_\d{4}")
NL = "\n"


class OllamaLLM:
    """Real local LLM through an Ollama server (default: qwen2.5:7b-instruct).

    Deterministic (temperature 0, fixed seed) and cached on disk, so reruns are
    reproducible and free. Besides parametric answers it can act as the
    *planner* (question -> relation chain) and *reader* (evidence chunks ->
    answer). The entities of the synthetic world are invented, so the model's
    parametric knowledge of them is ~nil; the interesting uses are planning
    and reading.

    Methods return None when the model gives no valid entity (= abstention).
    """

    def __init__(self, model: str = "qwen2.5:7b-instruct", host: str = "http://localhost:11434",
                 cache_path: Optional[str] = "results/ollama_cache.jsonl", timeout: int = 300) -> None:
        self.model, self.host, self.timeout = model, host, timeout
        self.cache_path, self.cache = cache_path, {}
        self.calls = self.cache_hits = 0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    self.cache[d["k"]] = d["v"]

    # ---- low level --------------------------------------------------------
    def _generate(self, prompt: str, num_predict: int = 40) -> str:
        key = hashlib.sha1(f"{self.model}|{num_predict}|{prompt}".encode()).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        body = {"model": self.model, "prompt": prompt, "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0, "seed": 0, "num_predict": num_predict}}
        req = urllib.request.Request(f"{self.host}/api/generate", method="POST", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())["response"].strip()
        self.calls += 1
        self.cache[key] = out
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "v": out}) + NL)
        return out

    @staticmethod
    def _entity(text: str, etype: str) -> Optional[str]:
        """First entity of the expected type in the reply; None if the model gave none."""
        prefix = TYPE_PREFIX[etype] + "_"
        for m in ENTITY_RX.findall(text):
            if m.startswith(prefix):
                return m
        return None

    # ---- parametric knowledge (baselines) ---------------------------------
    def answer_hop(self, s: str, r: str) -> Optional[str]:
        out = self._generate("Answer with the entity name only, or UNKNOWN if you do not know. "
                             f"What is {phrase(r)} {s}?")
        return self._entity(out, object_type(r))

    def answer_direct(self, start: str, rels: Sequence[str], question: Optional[str] = None) -> Optional[str]:
        q = question
        if q is None:
            q = start
            for r in rels:
                q = f"{phrase(r)} {q}"
            q = f"What is {q}?"
        out = self._generate(f"Answer with the entity name only, or UNKNOWN if you do not know.{NL}{q}")
        return self._entity(out, object_type(rels[-1])) if rels else None

    # ---- planning ----------------------------------------------------------
    def plan(self, question: str) -> Optional[Tuple[str, List[str]]]:
        rel_doc = NL.join(f"- {r}: {RELATIONS[r][0]} -> {RELATIONS[r][1]}" for r in RELATIONS)
        out = self._generate(
            "Decompose the question into the chain of relations that leads from the entity named in the "
            f"question to the answer. List the relations in the order they are applied.{NL}"
            f"Allowed relations (each one starts from the entity type on its left):{NL}{rel_doc}{NL}{NL}"
            f"The first relation must start from the type of the named entity. Reply on one line as: <entity name from the question> | <relation>, <relation>, ...{NL}{NL}"
            f"Question: Which city is home to the headquarters of Company_9001?{NL}"
            f"Reply: Company_9001 | headquartered_in{NL}{NL}"
            f"Question: Tell me which trade bloc the country of City_9002 belongs to.{NL}"
            f"Reply: City_9002 | located_in, member_of{NL}{NL}"
            f"Question: {question}{NL}Reply:", num_predict=40)
        if "|" not in out:
            return None
        left, right = out.split("|", 1)
        start = ENTITY_RX.search(left)
        rels = [x.strip(" .`'\"") for x in right.strip().splitlines()[0].split(",")]
        if not start or start.group(0) not in question or not rels:
            return None
        # schema-grounded post-processing: snap relation names to the schema, require a type-consistent chain
        fixed = []
        for x in rels:
            m = x if x in RELATIONS else (difflib.get_close_matches(x, list(RELATIONS), n=1, cutoff=0.6) or [None])[0]
            if m is None:
                return None
            fixed.append(m)
        cur_type = {v: k for k, v in TYPE_PREFIX.items()}.get(start.group(0).split("_")[0])
        for r in fixed:
            if subject_type(r) != cur_type:
                return None
            cur_type = object_type(r)
        return start.group(0), fixed

    # ---- reading -----------------------------------------------------------
    def read(self, subject: str, relation: str, chunks: Sequence[str]) -> Optional[str]:
        """Answer (subject, relation, ?) from retrieved text only; None if the text does not say."""
        if not chunks:
            return None
        ctx = NL.join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
        out = self._generate("Use ONLY the passages. Answer with the entity name only, or NONE if they do not "
                             f"state it.{NL}Passages:{NL}{ctx}{NL}Question: What is {phrase(relation)} {subject}?",
                             num_predict=16)
        ans = self._entity(out, object_type(relation))
        return ans if ans and any(ans in c for c in chunks) else None   # must be grounded in a passage
