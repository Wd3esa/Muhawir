"""Live hadith search through the public API of the Hadith Encyclopedia at dorar.net.

dorar.net offers this service so that website owners can show its search
results on their own sites (https://dorar.net/article/389). Muhawir uses it
that way only: results are fetched when a question is asked, shown with their
source and the grading as dorar reports it, credited to dorar, and never stored.

Every grading is kept and shown, as the user decided; the model is told to
state the grading and never to present a weak hadith as established.
If dorar cannot be reached, Muhawir answers from its other sources.
"""
from __future__ import annotations

import hashlib
import html
import logging
import re
import time

from .corpus import Passage, Source

API = "https://dorar.net/dorar_api.json"
SOURCE = Source(
    id="dorar-hadith",
    name="الموسوعة الحديثية – الدرر السنية",
    about="نتيجة بحث في الموسوعة الحديثية بموقع الدرر السنية (dorar.net)، تُجلب عند السؤال ولا تُخزَّن. "
          "الحكم على الحديث منقول كما أورده الموقع عن المحدث المذكور.",
    url="https://dorar.net/hadith",
)
KIND = "hadith"
TIMEOUT = 8.0
# identifies Muhawir honestly; it does not pretend to be a browser
HEADERS = {"User-Agent": "Muhawir/0.1 (educational Islamic Q&A; https://github.com/Wd3esa/muhawir)",
           "Accept": "application/json"}
REFUSED_PAUSE = 3600  # after dorar refuses (HTTP 403), stop asking for an hour instead of on every question
_refused_until = 0.0
log = logging.getLogger("muhawir")

_TAG = re.compile(r"<[^>]+>")
_NUMBER = re.compile(r"^\s*\d+\s*-\s*")
_FIELDS = {"الراوي": "rawi", "المحدث": "mohdith", "المصدر": "book",
           "الصفحة أو الرقم": "number", "خلاصة حكم المحدث": "grade"}
_FIELD = re.compile(r'<span class="info-subtitle">\s*([^<:]+?)\s*:\s*</span>(.*?)(?=<span class="info-subtitle">|</div>)', re.S)


def _plain(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", fragment))).strip()


def parse(result_html: str) -> list[dict]:
    """The hadiths in one API result: text, rawi, mohdith, book, number, grade (wording unchanged)."""
    out = []
    for block in result_html.split("--------------"):
        m = re.search(r'<div class="hadith"[^>]*>(.*?)</div>', block, re.S)
        if not m:
            continue
        text = _NUMBER.sub("", _plain(m.group(1))).rstrip(" .").strip()
        info = {key: "" for key in _FIELDS.values()}
        for label, value in _FIELD.findall(block):
            key = _FIELDS.get(label.strip())
            if key:
                info[key] = _plain(value)
        if text and info["book"] and info["grade"]:  # no hadith without a source and a grading
            out.append({"text": text, **info})
    return out


def to_passage(h: dict) -> Passage:
    digest = hashlib.sha1(f"{h['text']}|{h['book']}|{h['number']}|{h['mohdith']}".encode()).hexdigest()[:10]
    parts = [f"الراوي: {h['rawi']}" if h["rawi"] and h["rawi"] != "-" else "",
             f"المحدث: {h['mohdith']}" if h["mohdith"] else "",
             f"المصدر: {h['book']}" + (f"، {h['number']}" if h["number"] else "")]
    return Passage(id=f"d:{digest}", source_id=SOURCE.id, location=" — ".join(p for p in parts if p),
                   text=h["text"], kind=KIND, grade=h["grade"])


def search(query: str, limit: int = 3, timeout: float = TIMEOUT) -> list[Passage]:
    """Up to `limit` hadiths for `query`; [] when dorar cannot be reached or finds nothing."""
    import httpx

    global _refused_until
    query = (query or "").strip()
    if not query or time.time() < _refused_until:
        return []
    try:
        r = httpx.get(API, params={"skey": query}, headers=HEADERS, timeout=timeout, follow_redirects=True)
        if r.status_code == 403:
            _refused_until = time.time() + REFUSED_PAUSE
            log.warning("dorar.net refused the hadith search (HTTP 403); not asking again for an hour. "
                        "Its permission is needed for server-side use.")
            return []
        r.raise_for_status()
        found = parse(r.json().get("ahadith", {}).get("result", ""))
    except Exception as exc:  # network, format change: answer from the other sources
        log.warning("dorar hadith search failed: %s", f"{type(exc).__name__}: {exc}"[:300])
        return []
    return [to_passage(h) for h in found[:limit]]
