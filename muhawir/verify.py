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
# a sentence without a source may explain, but may not quote or attribute anything to Allah, the Prophet ﷺ
# or a scholar: verses, hadith and sayings come only from the sources
_SACRED = re.compile(
    r"[«»﴿﴾]|ﷺ|صلى الله عليه وسلم|عليه الصلاة والسلام|رسول الله|\bالنبي\b|\bقال تعالى|\bقوله تعالى|يقول الله|قال الله|"
    r"\bفي الحديث|\bحديث\b|\bروى\b|\bرواه\b|\bيُروى|\bقال (?:ابن|الإمام|الشيخ|مالك|الشافعي|أحمد|أبو حنيفة)|"
    r"\bthe Prophet\b|\bAllah says\b|\bhadith\b|\bnarrated\b", re.IGNORECASE)


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


# a school may be named by its founder or its followers: «الحنفية» is named when the passage says «أبو حنيفة»
_SCHOOLS = [("حنيفه", "حنفيه", "احناف"), ("مالك", "مالكيه"), ("شافعي", "شافعيه"),
            ("احمد", "حنبل", "حنابله")]


def school_names(school: str) -> list[str]:
    """Normalized names that count as naming this school in a passage (the school as written, plus its
    founder or followers for the four schools)."""
    own = normalize(school)
    names = [own] if own else []
    for group in _SCHOOLS:
        if any(g in own for g in group):
            names += list(group)
    return names


def quotes_in(text: str) -> list[str]:
    return [q for m in _QUOTES.finditer(text) for q in [next(g for g in m.groups() if g)] if _ARABIC.search(q)]


def verify(claims: list[Claim], corpus: Corpus,
           allowed_ids: set[str]) -> tuple[list[Claim], list[Rejected]]:
    kept: list[Claim] = []
    rejected: list[Rejected] = []
    for claim in claims:
        if not claim.passage_ids:
            if claim.school or _SACRED.search(claim.text):
                rejected.append(Rejected(claim, "quotes or attributes to Allah, the Prophet or a scholar without a source"))
            else:
                kept.append(claim)  # Muhawir's own explanation: shown without a source number
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
        if claim.school and not any(name in normalize(corpus.passage(pid).text)
                                    for name in school_names(claim.school) for pid in claim.passage_ids):
            rejected.append(Rejected(claim, f"'{claim.school}' is not named in the cited passage"))
            continue
        kept.append(claim)
    return kept, rejected
