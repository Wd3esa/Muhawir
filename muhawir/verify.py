"""Verifier: every claim must cite retrieved passages, and every quotation
must appear verbatim in one of the cited passages.

A generator (model or extractive) returns a draft as a list of claims. Claims
that fail are dropped; if nothing survives, the pipeline abstains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Corpus
from .normalize import collapse_spaces, normalize

_QUOTES = re.compile(r"«([^»]+)»|\"([^\"]+)\"|“([^”]+)”|﴿([^﴾]+)﴾")


@dataclass(frozen=True)
class Claim:
    text: str
    passage_ids: tuple[str, ...]
    school: str = ""  # set for a scholar's or school's view; must be named in the cited passage


@dataclass(frozen=True)
class Rejected:
    claim: Claim
    reason: str


def quotes_in(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in _QUOTES.finditer(text)]


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
        cited = [collapse_spaces(corpus.passage(pid).text) for pid in claim.passage_ids]
        bad = [q for q in quotes_in(claim.text)
               if not any(collapse_spaces(q) in text for text in cited)]
        if bad:
            rejected.append(Rejected(claim, f"quotation not found verbatim: {bad}"))
            continue
        if claim.school and not any(normalize(claim.school) in normalize(corpus.passage(pid).text)
                                    for pid in claim.passage_ids):
            rejected.append(Rejected(claim, f"'{claim.school}' is not named in the cited passage"))
            continue
        kept.append(claim)
    return kept, rejected
