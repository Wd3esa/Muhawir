"""Say plainly when the reasons-of-revelation source has no entry for the ayah or surah asked about.

Used only after the normal pipeline found no answer to a question about a
reason of revelation. The message states a fact about the source ("this
source records no reason for it"); it never claims that no reason exists.
The ayah or surah is named only when it is identified with confidence:
  - a surah, when its name follows the word "سورة" or "نزول" in the question;
  - an ayah, when one Quran passage alone contains all the remaining words.
An entry in the book often covers several ayahs but is filed under the first
one, so an ayah also counts as covered when any entry of the book quotes all
the words of the question; this keeps the message from ever being wrong.
Otherwise the general abstain message is kept.
"""
from __future__ import annotations

import re

from .normalize import normalize, tokenize

KIND = "asbab"
_ASKS = re.compile(r"\b(سبب|اسباب) (ال)?نزول\b|\bلماذا نزلت\b|\bفيم نزلت\b"
                   r"|\breasons? (for|of|behind) (the )?revelation\b|\boccasion of revelation\b")
# words that only say "reason of revelation of the ayah/surah", removed before finding the ayah
_FRAME = {"سبب", "اسباب", "نزول", "النزول", "ايه", "الايه", "سوره", "السوره", "لماذا", "فيم", "نزلت", "قوله", "تعالي"}
_SPAN = re.compile(r"(?:الآية|الآيات) (\d+)(?:–(\d+))?")
MIN_WORDS = 2


def asks_for_reason(question: str) -> bool:
    return bool(_ASKS.search(normalize(question)))


class AsbabIndex:
    def __init__(self, corpus, retriever) -> None:
        self.corpus, self.retriever = corpus, retriever
        self.covered: set[tuple[int, int]] = set()
        self.surahs_covered: set[int] = set()
        self.source_name = ""
        rows = self._asbab_rows(corpus)
        self.entry_words = [set(tokens.split()) for _pid, _sid, _loc, tokens in rows]
        for pid, source_id, location, _tokens in rows:
            surah = int(pid.split(":")[1])
            self.surahs_covered.add(surah)
            self.source_name = self.source_name or corpus.sources[source_id].name
            m = _SPAN.search(location)
            if m:
                first, last = int(m.group(1)), int(m.group(2) or m.group(1))
                self.covered.update((surah, a) for a in range(first, last + 1))
        self.surah_names: dict[str, tuple[int, str]] = {}
        if rows:
            for s in range(1, 115):
                p = corpus.passage(f"q:{s}:1")
                if p:
                    name = p.location.split("،")[0].strip()  # "سورة الفاتحة"
                    bare = normalize(name).removeprefix("سوره ").strip()
                    self.surah_names[bare] = (s, name)

    @staticmethod
    def _asbab_rows(corpus) -> list[tuple[str, str, str, str]]:
        con = getattr(corpus, "con", None)
        if con is not None:
            return con.execute("SELECT id, source_id, location, tokens FROM passages WHERE kind = ?",
                               (KIND,)).fetchall()
        return [(p.id, p.source_id, p.location, " ".join(tokenize(f"{p.text} {p.keywords}")))
                for p in corpus.passages if p.kind == KIND]

    @property
    def active(self) -> bool:
        return bool(self.covered or self.surahs_covered)

    def _surah(self, q: str) -> tuple[int, str] | None:
        for bare, found in self.surah_names.items():
            if re.search(rf"\b(سوره|نزول) {re.escape(bare)}\b", q):
                return found
        return None

    def _ayah(self, q: str) -> tuple[int, int, str] | None:
        words = [w for w in tokenize(q) if w not in _FRAME]
        if len(words) < MIN_WORDS:
            return None
        if any(set(words) <= entry for entry in self.entry_words):  # the book quotes it somewhere
            return None
        quran = [h for h in self.retriever.search(" ".join(words), k=60) if h.passage.kind == "quran"]
        full = [h for h in quran if h.coverage == 1.0]
        if len(full) != 1:  # none, or several ayahs equally likely: do not guess
            return None
        _, surah, ayah = full[0].passage.id.split(":")
        return int(surah), int(ayah), full[0].passage.location

    def missing(self, question: str) -> dict | None:
        """{"what": "ayah"|"surah", "where": location, "source": name} when the source has no entry."""
        if not self.active or not asks_for_reason(question):
            return None
        q = normalize(question)
        surah = self._surah(q)
        if surah:
            if surah[0] not in self.surahs_covered:
                return {"what": "surah", "where": surah[1], "source": self.source_name}
            return None
        ayah = self._ayah(q)
        if ayah and (ayah[0], ayah[1]) not in self.covered:
            return {"what": "ayah", "where": ayah[2], "source": self.source_name}
        return None
