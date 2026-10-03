"""Fixed rules that stop a question before retrieval or generation.

Only the cases the evaluation list marks as level D (personal fatwa or case),
judging specific people, and attempts to override the rules are detected
here. Levels A to C are not guessed by rules: telling them apart needs the
approved material itself, which is not in the repository yet.

Note: these rules were written with the draft evaluation list visible, so
their result on that list is not independent evidence of accuracy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import normalize

PERSONAL_CASE = "personal_case"      # level D: general info only + referral
JUDGING_PEOPLE = "judging_people"    # out of scope: decline politely
OVERRIDE = "override_attempt"        # keep the rules, explain, refer
SMALL_TALK = "small_talk"            # greeting or thanks only: short fixed reply, no search

_SMALL_TALK = re.compile(
    r"^(?:(?:و ?)?(?:السلام عليكم(?: ورحمه الله(?: وبركاته)?)?|عليكم السلام(?: ورحمه الله(?: وبركاته)?)?|"
    r"مرحبا|اهلا(?: وسهلا)?|صباح الخير|مساء الخير|شكرا(?: جزيلا)?|جزاك الله خيرا|بارك الله فيك|"
    r"hello|hi|hey|salam|assalamu alaikum|thank you|thanks|thank you very much)\s*)+$")


def is_small_talk(question: str) -> bool:
    return bool(_SMALL_TALK.match(normalize(question)))

_PERSONAL = [re.compile(p) for p in (
    r"\bهل يجوز لي\b", r"\bيلزمني\b", r"\bتلزمني\b", r"\bهل علي\b", r"\bفي حالتي\b",
    r"\bحلفت\b", r"\bطلقت\b", r"\bزواجي\b", r"\bزوجي\b", r"\bزوجتي\b",
    r"\bam i allowed\b", r"\bis it permissible for me\b", r"\bcan i\b", r"\bin my case\b",
    r"\bmy (husband|wife|marriage|divorce)\b",
)]

_JUDGING = [re.compile(p) for p in (
    r"\bهل (?!ال)\S+( \S+){0,3} (كافر|مرتد|منافق|مبتدع)\b",
    r"\bis (?!it\b)\S+( \S+){0,3} (a )?(kafir|disbeliever|apostate|infidel)\b",
)]

_OVERRIDE = [re.compile(p) for p in (
    r"\bتجاهل\b.*\b(ال)?تعليمات", r"\bانس\b.*\b(ال)?تعليمات", r"\bبما تراه انت\b", r"\bرايك الشخصي\b",
    r"\bignore\b.*\binstructions\b", r"\bdisregard\b.*\b(rules|instructions)\b",
    r"\byour own (opinion|fatwa)\b",
)]


@dataclass(frozen=True)
class Gate:
    kind: str | None   # one of the constants above, or None when the question may proceed
    rule: str = ""     # the pattern that matched, for logs and tests


def check(question: str) -> Gate:
    text = normalize(question)
    for kind, patterns in ((OVERRIDE, _OVERRIDE), (JUDGING_PEOPLE, _JUDGING),
                           (PERSONAL_CASE, _PERSONAL)):
        for pattern in patterns:
            if pattern.search(text):
                return Gate(kind, pattern.pattern)
    return Gate(None)
