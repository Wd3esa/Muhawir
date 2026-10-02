"""Question in, checked answer with source cards out.

Order: fixed rules -> retrieval -> sufficiency check -> generation ->
verifier -> response. The generator never sees a question that the rules
stopped, and never answers when retrieval found nothing sufficient.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import classify
from .corpus import Corpus
from .generate import Generator
from .messages import LANGS, STYLES, TEXT
from .retrieve import Retriever, is_sufficient
from .verify import verify

MODEL_CANDIDATES = 8      # passages offered to the model, which judges relevance itself
MODEL_MIN_COVERAGE = 0.34  # loose filter: the model, not keyword overlap, decides

MAX_QUESTION_CHARS = 500

ANSWERED, ABSTAINED, REFERRED, DECLINED, INVALID = (
    "answered", "abstained", "referred", "declined", "invalid")


@dataclass
class Response:
    status: str
    message: str
    claims: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    synthetic: bool = False
    note: str = ""
    views: list[dict] = field(default_factory=list)  # scholars' views as named in the sources

    def to_dict(self) -> dict:
        return asdict(self)


class Muhawir:
    def __init__(self, corpus: Corpus, generator: Generator, retriever=None) -> None:
        self.corpus = corpus
        self.retriever = retriever or Retriever(corpus)
        self.generator = generator

    def _cards(self, passage_ids: list[str]) -> list[dict]:
        cards = []
        for pid in dict.fromkeys(passage_ids):
            p = self.corpus.passage(pid)
            s = self.corpus.source_of(p)
            cards.append({"passage_id": p.id, "quote": p.text, "location": p.location,
                          "kind": p.kind, "grade": p.grade, "topics": p.keywords, "source_name": s.name,
                          "source_about": s.about, "source_url": s.url})
        return cards

    def ask(self, question: str, style: str = "youth", lang: str = "ar") -> Response:
        lang = lang if lang in LANGS else "ar"
        style = style if style in STYLES else "youth"
        t = TEXT[lang]
        synthetic = self.corpus.synthetic
        question = (question or "").strip()
        if not question:
            return Response(INVALID, t["empty"], synthetic=synthetic)
        if len(question) > MAX_QUESTION_CHARS:
            return Response(INVALID, t["too_long"], synthetic=synthetic)

        gate = classify.check(question)
        if gate.kind in (classify.JUDGING_PEOPLE, classify.OVERRIDE):
            return Response(DECLINED, t[gate.kind], synthetic=synthetic)

        personal = gate.kind == classify.PERSONAL_CASE
        if self.generator.strict_retrieval:
            hits = self.retriever.search(question)
            passages = [h.passage for h in hits if is_sufficient([h])]
        else:
            best: dict[str, object] = {}
            expand = getattr(self.generator, "expand", None)
            for query in [question] + (expand(question) if expand else []):
                for h in self.retriever.search(query, k=MODEL_CANDIDATES):
                    if h.coverage >= MODEL_MIN_COVERAGE and (
                            h.passage.id not in best or h.score > best[h.passage.id].score):
                        best[h.passage.id] = h
            ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
            passages = [h.passage for h in ranked[:MODEL_CANDIDATES]]
        if not passages:
            if personal:
                return Response(REFERRED, t["personal_case"], synthetic=synthetic)
            return Response(ABSTAINED, t["abstain"], synthetic=synthetic)

        allowed = {p.id for p in passages}
        kept, _rejected = verify(
            self.generator.generate(question, passages, style, lang, personal=personal),
            self.corpus, allowed)
        if not kept:
            status = REFERRED if gate.kind == classify.PERSONAL_CASE else ABSTAINED
            return Response(status, t["personal_case" if status == REFERRED else "abstain"],
                            synthetic=synthetic)

        answer = [c for c in kept if not c.school]
        if not answer:  # views alone, without a sourced answer, are not shown
            status = REFERRED if personal else ABSTAINED
            return Response(status, t["personal_case" if personal else "abstain"], synthetic=synthetic)
        claims = [{"text": c.text, "passage_ids": list(c.passage_ids)} for c in answer]
        views = [{"school": c.school, "text": c.text, "passage_ids": list(c.passage_ids)}
                 for c in kept if c.school]
        cards = self._cards([pid for c in answer + [v for v in kept if v.school] for pid in c.passage_ids])
        note = t["translation_pending"] if lang == "en" and self.generator.name == "extractive" else ""
        if gate.kind == classify.PERSONAL_CASE:
            message = t["personal_case"] + "\n" + t["personal_case_info"]
            return Response(REFERRED, message, claims, cards, synthetic, note, views)
        return Response(ANSWERED, t["intro"][style], claims, cards, synthetic, note, views)
