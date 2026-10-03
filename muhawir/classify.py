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
CRISIS = "crisis"                    # thoughts of suicide or self-harm: a fixed caring reply, no model
JUDGING_PEOPLE = "judging_people"    # out of scope: decline politely
OVERRIDE = "override_attempt"        # keep the rules, explain, refer
OUT_OF_SCOPE = "out_of_scope"        # contemporary financial rulings: refer to fatwa bodies (scope of version 1)
SMALL_TALK = "small_talk"            # greeting or thanks only: short fixed reply, no search

_GREETING = (r"السلام عليكم(?: ورحمه الله(?: وبركاته)?)?|عليكم السلام(?: ورحمه الله(?: وبركاته)?)?|"
             r"مرحبا|اهلا(?: وسهلا)?|صباح الخير|مساء الخير|hello|hi|hey|salam|assalamu alaikum")
_THANKS = (r"شكرا(?: جزيلا)?|جزاك(?:م)? الله(?: كل)? خيرا?|بارك الله فيك(?:م)?|فتح الله عليك(?:م)?|"
           r"احسن الله اليك(?:م)?|thank you(?: very much)?|thanks|jazakallah(?: khair)?")


def _only(phrases: str) -> re.Pattern:
    return re.compile(rf"^(?:(?:و ?)?(?:{phrases})\s*)+$")


_SMALL_TALK, _THANKS_ONLY = _only(f"{_GREETING}|{_THANKS}"), _only(_THANKS)


def is_small_talk(question: str) -> bool:
    return bool(_SMALL_TALK.match(normalize(question)))


def is_thanks(question: str) -> bool:
    return bool(_THANKS_ONLY.match(normalize(question)))

_PERSONAL = [re.compile(p) for p in (
    r"\bهل يجوز لي\b", r"\bيلزمني\b", r"\bتلزمني\b", r"\bهل علي\b", r"\bفي حالتي\b",
    r"\bحلفت\b", r"\bطلقت\b", r"\bزواجي\b", r"\bزوجي\b", r"\bزوجتي\b",
    r"\bam i allowed\b", r"\bis it permissible for me\b", r"\bin my case\b",
    r"\bmy (husband|wife|marriage|divorce)\b",
)]

# matched on normalized text (أ/إ → ا, ؤ → و, ة → ه, ى → ي, no diacritics)
_CRISIS = [re.compile(p) for p in (
    # the person speaking about themselves; a question about the topic («ما حكم الانتحار») is answered normally
    r"\bسانتحر", r"\bساقتل نفسي\b", r"\bاريد ان انتحر", r"\bابي انتحر", r"\bابغي انتحر", r"\bودي انتحر",
    r"\bافكر (?:في |ب)(?:ال)?انتحار", r"\bافكر ان انتحر", r"\bافكر في قتل نفسي\b", r"\bاقتل نفسي\b", r"\bاقتل حالي\b", r"\bانهي حياتي\b", r"\bانهاء حياتي\b",
    r"\bاوذي نفسي\b", r"\bايذاء نفسي\b", r"\bاذي نفسي\b",
    r"\bلا اريد ان اعيش\b", r"\bما ابي اعيش\b", r"\bما بدي عيش\b", r"\bمش عايز اعيش\b",
    r"\bاريد ان اموت\b", r"\bابغي اموت\b", r"\bابي اموت\b", r"\bنفسي اموت\b", r"\bتمنيت الموت\b",
    r"\bkill myself\b", r"\bend my life\b", r"\bwant to die\b", r"\bhurt myself\b",
    r"\bi (?:want|am going|m going|will|plan) to (?:commit suicide|end it all)\b", r"\bthinking (?:about|of) (?:suicide|killing myself)\b",
    r"\bdon ?t want to live\b",
)]

_JUDGING = [re.compile(p) for p in (
    # a named person ("هل فلان كافر"); not a category ("هل تارك الصلاة كافر") or a figure the sources name
    r"\bهل (?!ال|تارك|من\b|ما\b|كل\b|ابليس\b|فرعون\b|قارون\b|هامان\b)\S+( \S+){0,3} (كافر|مرتد|منافق|مبتدع)\b",
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


# Rulings on contemporary financial products need ijtihad by fatwa bodies and fiqh academies;
# they are outside version 1 (EVALUATION.md, Q25 and Q26). Only questions asking for a ruling are stopped.
_OUT_OF_SCOPE = [re.compile(p) for p in (
    r"\b(حكم|حلال|حرام|يجوز|جائز)\b.*\b(البنوك?|بنكي[هة]?|المصارف|فوائد|عملات رقمي[هة]|العملات الرقمي[هة]|الرقمي[هة]|بيتكوين|البيتكوين|كريبتو|التداول|تداول|الفوركس|فوركس|الاسهم|اسهم|التامين|تامين)\b",
    r"\b(البنوك?|بنكي[هة]?|المصارف|فوائد|عملات رقمي[هة]|العملات الرقمي[هة]|بيتكوين|البيتكوين|كريبتو|التداول|تداول|الفوركس|فوركس|الاسهم|التامين)\b.*\b(حلال|حرام|يجوز|جائز)\b",
    r"\b(is|are)\b.*\b(bank interest|crypto|bitcoin|forex|stock trading|insurance)\b.*\b(halal|haram|permissible|allowed)\b",
)]


_SUICIDE_TOPIC = re.compile(r"انتحار|انتحر|\bsuicid")


def mentions_suicide(question: str) -> bool:
    """The topic of suicide, e.g. a question about its ruling: answered, with a short caring line."""
    return bool(_SUICIDE_TOPIC.search(normalize(question)))


def check(question: str) -> Gate:
    text = normalize(question)
    for kind, patterns in ((CRISIS, _CRISIS), (OVERRIDE, _OVERRIDE), (JUDGING_PEOPLE, _JUDGING),
                           (PERSONAL_CASE, _PERSONAL), (OUT_OF_SCOPE, _OUT_OF_SCOPE)):
        for pattern in patterns:
            if pattern.search(text):
                return Gate(kind, pattern.pattern)
    return Gate(None)
