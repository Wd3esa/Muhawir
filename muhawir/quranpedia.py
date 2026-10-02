"""Import the Quran text from a Quranpedia.net data dump into a Muhawir corpus.

Input: the Hafs mushaf file from https://quranpedia.net/dumps
(mushafs-1.json.gz: "Hafs from Asim, matching the King Fahd Complex print").
The ayah text is kept exactly as in the dump; only leading byte-order marks
(U+FEFF) present in the dump are removed. The dump version is recorded so the
source card can state it, as the Quranpedia licence asks for republication.

Optional: the verse-topics dump (topics.json.gz). Topic names are attached as
search-only keywords so a question can reach the ayahs that concern it; they
are never shown as the quoted text.

Optional: a tafsir book dump (tafsir-book-<id>.json.gz). Its text is cut into
paragraph-sized passages; HTML markup is removed, the wording is not changed.

Optional: a reasons-of-revelation book dump (asbab-book-<id>.json.gz). It has
the same layout as a tafsir dump and is imported the same way, as kind "asbab".

Usage: python -m muhawir.quranpedia mushafs-1.json.gz [--topics topics.json.gz]
       [--tafsir tafsir-book-4.json.gz] [--asbab asbab-book-460.json.gz] -o data/quran_corpus.json
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import re
from pathlib import Path

EXPECTED_SURAHS = 114
EXPECTED_AYAHS = 6236
HAFS_MUSHAF_ID = 1
SOURCE_ID = "quranpedia-hafs"


class ImportError_(ValueError):
    pass


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def topic_keywords(topics_dump: dict | None) -> dict[str, str]:
    """Map "surah:ayah" to its topic names (topic and parent), joined with "؛ "."""
    if not topics_dump:
        return {}
    out: dict[str, str] = {}
    for row in topics_dump.get("data", []):
        names: list[str] = []
        for topic in row.get("topics", []):
            parent = (topic.get("parent") or {}).get("name")
            for name in (parent, topic.get("name")):
                if name and name not in names:
                    names.append(name)
        if names:
            out[f"{int(row['surah'])}:{int(row['ayah'])}"] = "؛ ".join(names)
    return out


_BREAK = re.compile(r"<br\s*/?>|</p>|</div>|</h3>|</tr>", re.I)
_TAG = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"[ \t\r\f\v]+")
MAX_CHUNK = 1200


def clean_html(text: str) -> list[str]:
    """Paragraphs of plain text: tags removed, entities decoded, wording unchanged."""
    text = _TAG.sub("", _BREAK.sub("\n", text))
    paragraphs = [_BLANKS.sub(" ", html.unescape(p)).strip() for p in text.split("\n")]
    return [p for p in paragraphs if p]


def chunk(paragraphs: list[str], limit: int = MAX_CHUNK) -> list[str]:
    """Join whole paragraphs up to about `limit` characters; never cut a paragraph."""
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + 1 + len(para) > limit:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


ASBAB_NOTE = "يجمع روايات أسباب النزول الواردة في الكتب التسعة، ومعها دراسة المؤلف لها"


def build_tafsir(tafsir_dump: dict, surah_names: dict[int, str],
                 keywords: dict[str, str], kind: str = "tafsir", note: str = "") -> tuple[dict, list[dict]]:
    """Source record and passages for one Quranpedia book dump keyed by ayah (tafsir or asbab)."""
    book, version = tafsir_dump.get("book", {}), tafsir_dump.get("license", {}).get("version", "")
    if not book.get("id") or not version:
        raise ImportError_(f"{kind} dump has no book id or licence version")
    source_id = f"quranpedia-{kind}-{book['id']}"
    prefix = "t" if kind == "tafsir" else "a"
    author = (book.get("author") or {}).get("ar_name", "")
    edition = "، ".join(x for x in (book.get("nasher"), book.get("edition"),
                                   f"تحقيق {book['mohaqeq']}" if book.get("mohaqeq") else "") if x)
    source = {"id": source_id, "name": f"{book.get('short_name') or book.get('name')}",
              "about": f"«{book.get('name')}»، تأليف {author}" + (f" ({edition})" if edition else "")
                       + (f"، {note}" if note else "")
                       + f"، من بيانات الموسوعة القرآنية quranpedia.net (نسخة {version}).",
              "url": "https://quranpedia.net"}

    groups: dict[str, dict] = {}  # one tafsir text may cover several ayahs
    for row in tafsir_dump.get("ayahs", []):
        surah, ayah = int(row["surah"]), int(row["ayah"])
        for part in row.get("content", []):
            text = part.get("text", "")
            if not text.strip():
                continue
            g = groups.setdefault(text, {"surah": surah, "ayahs": [], "part": part.get("part"),
                                         "page": part.get("page")})
            g["ayahs"].append(ayah)

    passages = []
    for n, (text, g) in enumerate(groups.items(), 1):
        first, last = min(g["ayahs"]), max(g["ayahs"])
        span = f"الآية {first}" if first == last else f"الآيات {first}–{last}"
        where = f"{surah_names.get(g['surah'], g['surah'])}، {span}"
        if g["part"] and g["page"]:
            where += f" (ج{g['part']}، ص{g['page']})"
        kw = "؛ ".join(dict.fromkeys(k for a in sorted(set(g["ayahs"]))
                                     for k in keywords.get(f"{g['surah']}:{a}", "").split("؛ ") if k))
        for i, piece in enumerate(chunk(clean_html(text)), 1):
            passages.append({"id": f"{prefix}{book['id']}:{g['surah']}:{first}:{n}:{i}",
                             "source_id": source_id, "location": where, "kind": kind,
                             "text": piece, "keywords": kw})
    return source, passages


def build_corpus(dump: dict, expected_surahs: int = EXPECTED_SURAHS,
                 expected_ayahs: int = EXPECTED_AYAHS, topics_dump: dict | None = None,
                 tafsir_dump: dict | None = None, asbab_dump: dict | None = None) -> dict:
    data, licence = dump.get("data", {}), dump.get("license", {})
    version = licence.get("version", "")
    if data.get("id") != HAFS_MUSHAF_ID:
        raise ImportError_(f"expected the Hafs mushaf (id {HAFS_MUSHAF_ID}), got id {data.get('id')}")
    if not version:
        raise ImportError_("dump has no licence version")
    surahs = data.get("surahs", [])
    if len(surahs) != expected_surahs:
        raise ImportError_(f"expected {expected_surahs} surahs, got {len(surahs)}")

    keywords = topic_keywords(topics_dump)
    passages = []
    for surah in surahs:
        number, name = int(surah["id"]), surah["name"]
        for ayah in surah["ayahs"]:
            text = ayah["text"].lstrip("\ufeff").strip()
            if not text or "\ufeff" in text:
                raise ImportError_(f"unexpected text in {number}:{ayah['number']}")
            passages.append({
                "id": f"q:{number}:{ayah['number']}",
                "source_id": SOURCE_ID,
                "location": f"{name}، الآية {ayah['number']}",
                "kind": "quran",
                "text": text,
                "keywords": keywords.get(f"{number}:{ayah['number']}", ""),
            })
    if len(passages) != expected_ayahs:  # checked before tafsir passages are added
        raise ImportError_(f"expected {expected_ayahs} ayahs, got {len(passages)}")

    sources = [{
        "id": SOURCE_ID,
        "name": "القرآن الكريم (رواية حفص عن عاصم)",
        "about": f"نص المصحف موافقًا لطبعة مجمع الملك فهد، من بيانات الموسوعة القرآنية "
                 f"quranpedia.net (نسخة {version}).",
        "url": "https://quranpedia.net",
    }]
    names = {int(s["id"]): s["name"] for s in surahs}
    tafsir_version = asbab_version = ""
    if tafsir_dump:
        tafsir_source, tafsir_passages = build_tafsir(tafsir_dump, names, keywords)
        sources.append(tafsir_source)
        passages.extend(tafsir_passages)
        tafsir_version = tafsir_dump["license"]["version"]
    if asbab_dump:
        asbab_source, asbab_passages = build_tafsir(asbab_dump, names, keywords,
                                                    kind="asbab", note=ASBAB_NOTE)
        sources.append(asbab_source)
        passages.extend(asbab_passages)
        asbab_version = asbab_dump["license"]["version"]

    return {
        "synthetic": False,
        "_provenance": {"source": "https://quranpedia.net/dumps", "mushaf": data.get("name"),
                        "description": data.get("description"), "version": version,
                        "topics_version": (topics_dump or {}).get("license", {}).get("version", ""),
                        "tafsir_version": tafsir_version, "asbab_version": asbab_version},
        "sources": sources,
        "passages": passages,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path)
    parser.add_argument("--topics", type=Path, help="topics.json.gz from quranpedia.net/dumps")
    parser.add_argument("--tafsir", type=Path, help="tafsir-book-<id>.json.gz from quranpedia.net/dumps")
    parser.add_argument("--asbab", type=Path, help="asbab-book-<id>.json.gz from quranpedia.net/dumps")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/quran_corpus.json"))
    args = parser.parse_args(argv)
    with _open(args.dump) as fh:
        dump = json.load(fh)
    topics = None
    if args.topics:
        with _open(args.topics) as fh:
            topics = json.load(fh)
    tafsir = None
    if args.tafsir:
        with _open(args.tafsir) as fh:
            tafsir = json.load(fh)
    asbab = None
    if args.asbab:
        with _open(args.asbab) as fh:
            asbab = json.load(fh)
    corpus = build_corpus(dump, topics_dump=topics, tafsir_dump=tafsir, asbab_dump=asbab)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    print(f"{len(corpus['passages'])} passages written to {args.output} "
          f"(Quranpedia version {corpus['_provenance']['version']})")


if __name__ == "__main__":
    main()
