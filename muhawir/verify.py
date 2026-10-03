"""Verifier: every claim must cite retrieved passages, and every quotation
must appear verbatim in one of the cited passages.

A generator (model or extractive) returns a draft as a list of claims. Claims
that fail are dropped; if nothing survives, the pipeline abstains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Corpus
from .normalize import normalize

# quotations of the sources: Arabic text inside «» or ﴿﴾ (English "..." marks a word, not a quotation)
_QUOTES = re.compile(r"«([^»]+)»|﴿([^﴾]+)﴾")
_ARABIC = re.compile(r"[؀-ۿ]")


@dataclass(frozen=True)
class Claim:
    text: str
    passage_ids: tuple[str, ...]
    school: str = ""  # set for a scholar's or school's view; must be named in the cited passage
    section: str = ""  # heading of the part of the answer this sentence belongs to (layout only)
    label: str = ""  # a short bold word that leads the sentence, e.g. «المقدار» (layout only)


@dataclass(frozen=True)
class Rejected:
    claim: Claim
    reason: str


def quotes_in(text: str) -> list[str]:
    return [q for m in _QUOTES.finditer(text) for q in [next(g for g in m.groups() if g)] if _ARABIC.search(q)]


def verify(claims: list[Claim], corpus: Corpus,
           allowed_ids: set[str]) -> tuple[list[Claim], list[Rejected]]:
    kept: list[Claim] = []
    rejected: list[Rejected] = []
    for claim in claims:
        if not claim.passage_ids:
            rejected.append(Rejected(claim, "no citation"))
            continue
        unknown = [pid for pid in claim.passage_ids if pid not in allowed_ids]
        if unknown:
            rejected.append(Rejected(claim, f"cites passages that were not retrieved: {unknown}"))
            continue
        # compared without diacritics or punctuation: «الكوثر» matches الْكَوْثَرَ; the card shows the exact text
        cited = [normalize(corpus.passage(pid).text) for pid in claim.passage_ids]
        bad = [q for q in quotes_in(claim.text)
               if not normalize(q) or not any(normalize(q) in text for text in cited)]
        if bad:
            rejected.append(Rejected(claim, f"quotation not found verbatim: {bad}"))
            continue
        if claim.school and not any(normalize(claim.school) in normalize(corpus.passage(pid).text)
                                    for pid in claim.passage_ids):
            rejected.append(Rejected(claim, f"'{claim.school}' is not named in the cited passage"))
            continue
        kept.append(claim)
    return kept, rejected
