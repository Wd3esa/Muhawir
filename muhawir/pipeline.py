"""Question in, checked answer with source cards out.

Order: fixed rules -> retrieval -> sufficiency check -> generation ->
verifier -> response. The generator never sees a question that the rules
stopped, and never answers when retrieval found nothing sufficient.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field

from . import classify
from .asbab import AsbabIndex
from .corpus import Corpus, Passage
from .generate import Generator
from .messages import LANGS, STYLES, TEXT
from .normalize import STOPWORDS, normalize
from .retrieve import Retriever, is_sufficient
from .verify import Rejected, verify

MODEL_CANDIDATES = int(os.environ.get("MUHAWIR_PASSAGES") or 8)  # passages offered to the model; fewer = faster on slow machines
MODEL_MIN_COVERAGE = 0.34  # loose filter: the model, not keyword overlap, decides
MAX_HADITH = 3  # live hadith results offered to the model, in addition to the passages above

MAX_QUESTION_CHARS = 500

ANSWERED, ABSTAINED, REFERRED, DECLINED, INVALID, CHAT = (
    "answered", "abstained", "referred", "declined", "invalid", "chat")
DEBUG = os.environ.get("MUHAWIR_DEBUG") == "1"  # adds the reason for not answering to each response
log = logging.getLogger("muhawir")
MAX_HISTORY_TURNS = 6
MAX_TURN_CHARS = 600


@dataclass
class Response:
    status: str
    message: str
    claims: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    synthetic: bool = False
    note: str = ""
    views: list[dict] = field(default_factory=list)  # scholars' views as named in the sources
    understood: str = ""  # the follow-up question as rewritten for search, when it differs
    why: str = ""  # with MUHAWIR_DEBUG=1: why there is no answer (never contains the question)

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_ids(text: str, ids: set[str]) -> str:
    """Remove passage ids a model wrote into the answer text; the source cards already show them."""
    for pid in sorted(ids, key=len, reverse=True):
        text = re.sub(rf"\s*[\[(]?(?<![\w:]){re.escape(pid)}(?![\w:])[\])]?", "", text)
    return re.sub(r"\s+([.،,؛])", r"\1", text).strip()


class _WithLive:
    """The corpus plus passages fetched live for one question (e.g. dorar hadith results)."""

    def __init__(self, corpus, live: list[Passage], sources: dict) -> None:
        self.base, self.live, self.extra_sources = corpus, {p.id: p for p in live}, sources

    def passage(self, pid: str):
        return self.live.get(pid) or self.base.passage(pid)

    def source_of(self, p):
        return self.extra_sources.get(p.source_id) or self.base.source_of(p)


def _search_words(text: str) -> str:
    """The question without filler words, kept in their written form for an outside search."""
    return " ".join(w for w in text.split() if normalize(w) and normalize(w) not in STOPWORDS)


class Muhawir:
    def __init__(self, corpus: Corpus, generator: Generator, retriever=None, hadith_search=None,
                 hadith_source=None) -> None:
        self.corpus = corpus
        self.retriever = retriever or Retriever(corpus)
        self.generator = generator
        # live hadith search (dorar.net); never used with synthetic test data or in extractive mode
        self.hadith_search = hadith_search if not corpus.synthetic else None
        self.hadith_source = hadith_source
        self.asbab = AsbabIndex(corpus, self.retriever)

    def _abstain(self, question: str, t: dict, synthetic: bool) -> Response:
        """General abstain, or a precise one when the reasons-of-revelation source has no entry."""
        missing = self.asbab.missing(question)
        if missing:
            return Response(ABSTAINED, t["no_reason"][missing["what"]].format(**missing), synthetic=synthetic)
        return Response(ABSTAINED, t["abstain"], synthetic=synthetic)

    @staticmethod
    def _why(res: Response, reason: str, reply_start: str = "") -> Response:
        log.warning("not answered: %s", reason[:500])  # the reply itself is never logged
        if DEBUG:
            res.why = reason[:500] + (f" | reply began: {reply_start}" if reply_start else "")
        return res

    def _hadith(self, queries: list[str]) -> list[Passage]:
        found: dict[str, Passage] = {}
        for query in queries[:2]:
            for p in self.hadith_search(_search_words(query)):
                found.setdefault(p.id, p)
            if len(found) >= MAX_HADITH:
                break
        return list(found.values())[:MAX_HADITH]

    def _cards(self, passage_ids: list[str], corpus=None) -> list[dict]:
        corpus = corpus or self.corpus
        cards = []
        for pid in dict.fromkeys(passage_ids):
            p = corpus.passage(pid)
            s = corpus.source_of(p)
            cards.append({"passage_id": p.id, "quote": p.text, "location": p.location,
                          "kind": p.kind, "grade": p.grade, "topics": p.keywords, "source_name": s.name,
                          "source_about": s.about, "source_url": s.url})
        return cards

    def ask(self, question: str, style: str = "youth", lang: str = "ar",
            history: list[dict] | None = None) -> Response:
        """Answer one message. `history` (earlier turns) is used only to understand a follow-up;
        the answer itself still comes from retrieved passages alone."""
        question = (question or "").strip()
        lang_ok = lang if lang in LANGS else "ar"
        if question and len(question) <= MAX_QUESTION_CHARS and classify.is_small_talk(question):
            reply = "thanks" if classify.is_thanks(question) else "small_talk"
            return Response(CHAT, TEXT[lang_ok][reply], synthetic=self.corpus.synthetic)
        turns = [{"role": t.get("role"), "text": str(t.get("text", ""))[:MAX_TURN_CHARS]}
                 for t in (history or []) if isinstance(t, dict) and t.get("role") in ("user", "assistant")]
        turns = turns[-MAX_HISTORY_TURNS:]
        if question and len(question) <= MAX_QUESTION_CHARS and classify.check(question).kind is None:
            missing = self.asbab.missing(question)  # needs no model: same answer in every style, at once
            if missing:
                return Response(ABSTAINED, TEXT[lang_ok]["no_reason"][missing["what"]].format(**missing),
                                synthetic=self.corpus.synthetic)
        understand = getattr(self.generator, "understand", None)
        understood, queries = "", None
        if question and understand and len(question) <= MAX_QUESTION_CHARS:
            gate = classify.check(question)  # the user's own words are checked before any rewording
            if gate.kind not in (classify.JUDGING_PEOPLE, classify.OVERRIDE):
                u = understand(question, turns)
                if u is not None:
                    if not u["question"]:  # no question in the message (e.g. only an insult): no judgement, an invitation
                        return Response(CHAT, TEXT[lang_ok]["no_question"], synthetic=self.corpus.synthetic)
                    queries = u["queries"]
                    if normalize(u["question"]) != normalize(question):
                        understood = u["question"]
        res = self._ask(understood or question, style, lang, original=question, queries=queries)
        if understood and res.status not in (INVALID,):
            res.understood = understood
        return res

    def _ask(self, question: str, style: str, lang: str, original: str = "",
             queries: list[str] | None = None) -> Response:
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
        if original and original != question:  # a rewritten follow-up: the user's own words decide too
            own = classify.check(original)
            if own.kind:
                gate = own
        if gate.kind in (classify.JUDGING_PEOPLE, classify.OVERRIDE):
            return Response(DECLINED, t[gate.kind], synthetic=synthetic)

        personal = gate.kind == classify.PERSONAL_CASE
        if not personal:  # decided before the model, so every style gets the same answer
            missing = self.asbab.missing(question)
            if missing:
                return Response(ABSTAINED, t["no_reason"][missing["what"]].format(**missing),
                                synthetic=synthetic)
        if self.generator.strict_retrieval:
            hits = self.retriever.search(question)
            passages = [h.passage for h in hits if is_sufficient([h])]
        else:
            best: dict[str, object] = {}
            expand = getattr(self.generator, "expand", None)
            extra = queries if queries is not None else (expand(question) if expand else [])
            queries = [question] + extra
            for query in queries:
                for h in self.retriever.search(query, k=MODEL_CANDIDATES):
                    if h.coverage >= MODEL_MIN_COVERAGE and (
                            h.passage.id not in best or h.score > best[h.passage.id].score):
                        best[h.passage.id] = h
            ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
            passages = [h.passage for h in ranked[:MODEL_CANDIDATES]]
            if self.hadith_search:
                live = self._hadith(queries)
                passages += live
        if not passages:
            if personal:
                return Response(REFERRED, t["personal_case"], synthetic=synthetic)
            return self._why(self._abstain(question, t, synthetic), "search found no passage")

        allowed = {p.id for p in passages}
        live = [p for p in passages if self.corpus.passage(p.id) is None]
        corpus = _WithLive(self.corpus, live, {self.hadith_source.id: self.hadith_source}) if live else self.corpus
        if hasattr(self.generator, "last_note"):
            self.generator.last_note = self.generator.last_raw = ""
        draft = self.generator.generate(question, passages, style, lang, personal=personal)
        kept, rejected = verify(draft, corpus, allowed)
        check = getattr(self.generator, "check_support", None)
        if kept and check:  # second reading against the cited passages; fail closed if it cannot run
            flags = check(kept, {p.id: p for p in passages})
            if flags is None:
                rejected += [Rejected(c, "support check could not run") for c in kept]
                kept = []
            else:
                rejected += [Rejected(c, "not supported by the cited passage") for c, ok in zip(kept, flags) if not ok]
                kept = [c for c, ok in zip(kept, flags) if ok]
        offered = f"{len(passages)} passages offered"
        if not kept:
            reason = "; ".join(r.reason for r in rejected) or getattr(self.generator, "last_note", "") or "no claims"
            raw = getattr(self.generator, "last_raw", "")
            if personal:
                return self._why(Response(REFERRED, t["personal_case"], synthetic=synthetic), f"{offered}; {reason}")
            return self._why(self._abstain(question, t, synthetic), f"{offered}; {reason}", raw)
        if rejected:
            log.warning("some claims dropped: %s", "; ".join(r.reason for r in rejected)[:500])

        answer = [c for c in kept if not c.school]
        if not answer:  # views alone, without a sourced answer, are not shown
            if personal:
                return Response(REFERRED, t["personal_case"], synthetic=synthetic)
            return self._why(self._abstain(question, t, synthetic), f"{offered}; only scholars' views, no sourced answer")
        claims = [{"text": _strip_ids(c.text, allowed), "passage_ids": list(c.passage_ids)} for c in answer]
        views = [{"school": c.school, "text": _strip_ids(c.text, allowed), "passage_ids": list(c.passage_ids)}
                 for c in kept if c.school]
        cards = self._cards([pid for c in answer + [v for v in kept if v.school] for pid in c.passage_ids], corpus)
        note = t["translation_pending"] if lang == "en" and self.generator.name == "extractive" else ""
        if gate.kind == classify.PERSONAL_CASE:
            message = t["personal_case"] + "\n" + t["personal_case_info"]
            return Response(REFERRED, message, claims, cards, synthetic, note, views)
        return Response(ANSWERED, "", claims, cards, synthetic, note, views)
