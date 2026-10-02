"""Approved-source corpus: sources and the passages cut from them.

Every passage keeps its source, its location inside the source (surah and
ayah, or book and chapter) and, for hadith, the grading stated by the source.
A source also carries a one-line introduction, because the source card has
to tell the reader what the source is, not only its name.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KINDS = frozenset({"quran", "hadith", "fiqh", "aqeedah", "seerah", "other"})


class CorpusError(ValueError):
    pass


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    about: str  # one line: what this source is
    url: str = ""


@dataclass(frozen=True)
class Passage:
    id: str
    source_id: str
    location: str
    text: str
    kind: str = "other"
    grade: str = ""  # hadith grading as stated by the source, if any
    keywords: str = ""  # search-only terms (e.g. topic names); never shown as the quote


@dataclass
class Corpus:
    sources: dict[str, Source]
    passages: list[Passage]
    synthetic: bool = False
    _by_id: dict[str, Passage] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {p.id: p for p in self.passages}

    def passage(self, passage_id: str) -> Passage | None:
        return self._by_id.get(passage_id)

    def source_of(self, passage: Passage) -> Source:
        return self.sources[passage.source_id]


def _require(obj: dict, keys: tuple[str, ...], where: str) -> None:
    for key in keys:
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CorpusError(f"{where}: missing or empty '{key}'")


def parse_corpus(data: dict) -> Corpus:
    if not isinstance(data, dict):
        raise CorpusError("corpus must be a JSON object")
    sources: dict[str, Source] = {}
    for i, raw in enumerate(data.get("sources", [])):
        _require(raw, ("id", "name", "about"), f"source #{i}")
        if raw["id"] in sources:
            raise CorpusError(f"duplicate source id '{raw['id']}'")
        sources[raw["id"]] = Source(raw["id"], raw["name"], raw["about"], raw.get("url", ""))

    passages: list[Passage] = []
    seen: set[str] = set()
    for i, raw in enumerate(data.get("passages", [])):
        _require(raw, ("id", "source_id", "location", "text"), f"passage #{i}")
        if raw["id"] in seen:
            raise CorpusError(f"duplicate passage id '{raw['id']}'")
        if raw["source_id"] not in sources:
            raise CorpusError(f"passage '{raw['id']}' refers to unknown source '{raw['source_id']}'")
        kind = raw.get("kind", "other")
        if kind not in KINDS:
            raise CorpusError(f"passage '{raw['id']}' has unknown kind '{kind}'")
        seen.add(raw["id"])
        passages.append(Passage(raw["id"], raw["source_id"], raw["location"], raw["text"],
                                kind, raw.get("grade", ""), raw.get("keywords", "")))
    if not passages:
        raise CorpusError("corpus has no passages")
    return Corpus(sources, passages, bool(data.get("synthetic", False)))


def load_corpus(path: str | Path) -> Corpus:
    with open(path, encoding="utf-8") as fh:
        return parse_corpus(json.load(fh))
