"""Shared schema: relations, entity types, facts, evidence, text templates."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# relation -> (subject type, object type, question phrase, sentence template)
RELATIONS: Dict[str, Tuple[str, str, str, str]] = {
    "works_for": ("person", "company", "the employer of", "{s} works for {o}."),
    "headquartered_in": ("company", "city", "the headquarters city of", "{s} is headquartered in {o}."),
    "located_in": ("city", "country", "the country of", "{s} is located in {o}."),
    "member_of": ("country", "bloc", "the bloc of", "{s} is a member of {o}."),
}
# canonical multi-hop chain order
CHAIN = ["works_for", "headquartered_in", "located_in", "member_of"]

TYPE_PREFIX = {"person": "Person", "company": "Company", "city": "City",
               "country": "Country", "bloc": "Bloc"}


def subject_type(rel: str) -> str:
    return RELATIONS[rel][0]


def object_type(rel: str) -> str:
    return RELATIONS[rel][1]


def phrase(rel: str) -> str:
    return RELATIONS[rel][2]


def render_fact(s: str, r: str, o: str) -> str:
    return RELATIONS[r][3].format(s=s, o=o)


_PARSERS = {
    r: re.compile("^" + re.escape(t[3]).replace(r"\{s\}", r"(\S+)").replace(r"\{o\}", r"(\S+?)") + "$")
    for r, t in RELATIONS.items()
}


def parse_sentence(text: str) -> Optional[Tuple[str, str, str]]:
    """Inverse of render_fact (the 'reader' used on retrieved text chunks)."""
    for r, rx in _PARSERS.items():
        m = rx.match(text.strip())
        if m:
            return m.group(1), r, m.group(2)
    return None


@dataclass
class Fact:
    subject: str
    relation: str
    object: str
    source: str = "curated"
    confidence: float = 0.9
    timestamp: int = 0        # stage at which the fact was asserted
    last_verified: int = 0    # stage at which it was last confirmed
    valid: bool = True        # False once deprecated
    fid: int = -1

    @property
    def triple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


@dataclass
class Evidence:
    """A piece of support for one reasoning hop (carries provenance)."""
    subject: str
    relation: str
    object: str
    kind: str                 # 'kg' | 'text' | 'llm'
    confidence: float
    timestamp: int = 0
    last_verified: int = 0
    source: str = ""
    fid: int = -1

    @property
    def triple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


@dataclass
class Document:
    text: str
    source: str = "web"
