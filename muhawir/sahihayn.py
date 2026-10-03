"""Import Sahih al-Bukhari and Sahih Muslim (Arabic) from the open hadith-api data.

Source: https://github.com/fawazahmed0/hadith-api (released into the public domain,
"The Unlicense"), files editions/ara-bukhari.min.json and editions/ara-muslim.min.json.
The organizers' guide names «الصحيحان» first among hadith sources.

Numbering: each hadith keeps its standard number ("arabicnumber"): al-Bukhari as in
Fath al-Bari, Muslim as in Muhammad Fuad Abd al-Baqi's edition (further chains of the
same hadith, numbered 8.01, 8.02 in the data, are shown under their main number). Entries without a
standard number (mostly Muslim's introduction) are left out, so every card can be
looked up in a printed copy. The wording is unchanged; only invisible direction marks
(U+200E, U+200F) are removed and spaces collapsed. Very long hadiths are split at
sentence ends into parts of about MAX_PART characters, never inside a word.

No grading is added: both books are accepted as authentic, and the card names the book
and number instead of inventing a grade.
"""
from __future__ import annotations

import re

BOOKS = {
    "bukhari": {"file": "ara-bukhari.min.json", "prefix": "b", "source_id": "sahih-bukhari",
                "name": "صحيح البخاري",
                "about": "«الجامع المسند الصحيح» للإمام محمد بن إسماعيل البخاري (ت 256هـ)، بالترقيم المتداول "
                         "(ترقيم فتح الباري). النص العربي من بيانات hadith-api المفتوحة "
                         "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
    "muslim": {"file": "ara-muslim.min.json", "prefix": "m", "source_id": "sahih-muslim",
               "name": "صحيح مسلم",
               "about": "«المسند الصحيح» للإمام مسلم بن الحجاج (ت 261هـ)، بترقيم محمد فؤاد عبد الباقي. "
                        "النص العربي من بيانات hadith-api المفتوحة "
                        "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
}
DOWNLOAD = ("https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{file}",
            "https://raw.githubusercontent.com/fawazahmed0/hadith-api/1/editions/{file}")
MAX_PART = 1500
MIN_HADITH = {"bukhari": 7000, "muslim": 5000}  # a truncated download is refused

_MARKS = re.compile("[‎‏]")
_SPACES = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.؟!])\s+|(?<=\.\")\s+")


class ImportError_(ValueError):
    pass


def clean(text: str) -> str:
    return _SPACES.sub(" ", _MARKS.sub("", text)).strip()


def split_long(text: str, limit: int = MAX_PART) -> list[str]:
    """Whole sentences joined up to about `limit` characters; a hadith shorter than that stays whole."""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + 1 + len(sentence) > limit:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        parts.append(current)
    return parts


def build_book(key: str, dump: dict) -> tuple[dict, list[dict]]:
    book = BOOKS[key]
    passages, seen = [], set()
    for h in dump.get("hadiths", []):
        number, text = h.get("arabicnumber"), clean(h.get("text") or "")
        if number in (None, "") or not text:
            continue
        number = str(number)
        if number in seen:
            raise ImportError_(f"{key}: hadith number {number} appears twice")
        seen.add(number)
        parts = split_long(text)
        for i, part in enumerate(parts, 1):
            # Muslim's sub-numbers (8.01, 8.02: further chains of hadith 8) are cited by the main number
            shown = number.split(".")[0]
            where = f"{book['name']}، رقم {shown}" + (f" (الجزء {i} من {len(parts)})" if len(parts) > 1 else "")
            passages.append({"id": f"{book['prefix']}:{number}" + (f":{i}" if len(parts) > 1 else ""),
                             "source_id": book["source_id"], "location": where, "kind": "hadith",
                             "text": part})
    if len(seen) < MIN_HADITH[key]:
        raise ImportError_(f"{key}: only {len(seen)} numbered hadiths; the file looks incomplete")
    # no "open source" link: the data repository is credited in `about`; each card links to dorar to check the hadith
    source = {"id": book["source_id"], "name": book["name"], "about": book["about"], "url": ""}
    return source, passages


def add_to_corpus(corpus: dict, dumps: dict[str, dict]) -> dict:
    """Add the books in `dumps` ({"bukhari": ..., "muslim": ...}) to a corpus built by quranpedia.build_corpus."""
    for key, dump in dumps.items():
        source, passages = build_book(key, dump)
        corpus["sources"].append(source)
        corpus["passages"].extend(passages)
        corpus.setdefault("_provenance", {})[f"{key}_entries"] = len({p["id"].rsplit(":", 1)[0] if p["id"].count(":") > 1
                                                                       else p["id"] for p in passages})
    return corpus
