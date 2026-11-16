"""Retrieval: KG lookup / multi-hop path retrieval and TF-IDF vector search over text."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import RELATIONS, Document, Evidence, parse_sentence, phrase

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class GraphRetriever:
    """Structured retrieval over a graph store."""

    def __init__(self, graph) -> None:
        self.graph = graph

    def lookup(self, subject: str, relation: str) -> List[Evidence]:
        """One-hop retrieval; returns every valid fact with provenance."""
        return [Evidence(f.subject, f.relation, f.object, "kg", f.confidence, f.timestamp,
                         f.last_verified, f.source, f.fid)
                for f in self.graph.get(subject, relation)]

    def multi_hop(self, start: str, relations: Sequence[str], beam: int = 3) -> List[List[Evidence]]:
        """All KG paths following `relations` from `start` (beam-limited per hop)."""
        paths: List[List[Evidence]] = [[]]
        frontier = [(start, [])]
        for rel in relations:
            nxt = []
            for node, path in frontier:
                cands = sorted(self.lookup(node, rel), key=lambda e: -e.confidence)[:beam]
                for e in cands:
                    nxt.append((e.object, path + [e]))
            frontier = nxt
        return [p for _, p in frontier]


class VectorRetriever:
    """TF-IDF cosine search over text chunks (the 'vector RAG' baseline)."""

    def __init__(self, docs: List[Document]) -> None:
        self.docs = docs
        vocab: Dict[str, int] = {}
        toks = [tokenize(d.text) for d in docs]
        for t in toks:
            for w in t:
                vocab.setdefault(w, len(vocab))
        self.vocab = vocab
        n, v = len(docs), max(len(vocab), 1)
        tf = np.zeros((n, v), dtype=np.float32)
        for i, t in enumerate(toks):
            for w in t:
                tf[i, vocab[w]] += 1.0
        df = (tf > 0).sum(0)
        self.idf = np.log((1 + n) / (1 + df)).astype(np.float32) + 1.0
        m = tf * self.idf
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        self.matrix = m / np.maximum(norm, 1e-9)

    def _query_vec(self, query: str) -> np.ndarray:
        q = np.zeros(self.matrix.shape[1], dtype=np.float32)
        for w in tokenize(query):
            j = self.vocab.get(w)
            if j is not None:
                q[j] += 1.0
        q *= self.idf
        return q / max(float(np.linalg.norm(q)), 1e-9)

    def search(self, query: str, k: int = 8) -> List[Tuple[float, Document]]:
        if not len(self.docs):
            return []
        scores = self.matrix @ self._query_vec(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(float(scores[i]), self.docs[i]) for i in order if scores[i] > 0]

    def lookup(self, subject: str, relation: str, k: int = 8, text_conf: float = 0.6,
               reader=None) -> List[Evidence]:
        """Retrieve chunks, then 'read' them: keep sentences stating (subject, relation, ?).

        Default reader is rule-based; `reader(subject, relation, chunk_texts) -> object|None`
        lets an LLM read the top chunks instead.
        """
        hits = self.search(f"{subject} {phrase(relation)}".replace("the ", ""), k)
        if reader is not None:
            chunks = [d.text for _, d in hits[:5]]
            obj = reader(subject, relation, chunks)
            return [Evidence(subject, relation, obj, "text", text_conf, 0, 0, "web", -1)] if obj else []
        out: List[Evidence] = []
        for score, doc in hits:
            parsed = parse_sentence(doc.text)
            if parsed and parsed[0] == subject and parsed[1] == relation:
                out.append(Evidence(subject, relation, parsed[2], "text", text_conf, 0, 0,
                                    doc.source, -1))
        return out


class HybridRetriever:
    """GraphRAG-style hybrid: structured KG evidence first, text as complementary evidence."""

    def __init__(self, graph_retriever: GraphRetriever, vector_retriever: Optional[VectorRetriever],
                 reader=None):
        self.graph = graph_retriever
        self.vector = vector_retriever
        self.reader = reader

    def kg(self, subject: str, relation: str) -> List[Evidence]:
        return self.graph.lookup(subject, relation)

    def text(self, subject: str, relation: str) -> List[Evidence]:
        return self.vector.lookup(subject, relation, reader=self.reader) if self.vector else []
