"""Import the Quran text from a Quranpedia.net data dump into a Muhawir corpus.

Input: the Hafs mushaf file from https://quranpedia.net/dumps
(mushafs-1.json.gz: "Hafs from Asim, matching the King Fahd Complex print").
The ayah text is kept exactly as in the dump; only leading byte-order marks
(U+FEFF) present in the dump are removed. The dump version is recorded so the
source card can state it, as the Quranpedia licence asks for republication.

Usage: python -m muhawir.quranpedia mushafs-1.json.gz -o data/quran_corpus.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

EXPECTED_SURAHS = 114
EXPECTED_AYAHS = 6236
HAFS_MUSHAF_ID = 1
SOURCE_ID = "quranpedia-hafs"


class ImportError_(ValueError):
    pass


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def build_corpus(dump: dict, expected_surahs: int = EXPECTED_SURAHS,
                 expected_ayahs: int = EXPECTED_AYAHS) -> dict:
    data, licence = dump.get("data", {}), dump.get("license", {})
    version = licence.get("version", "")
    if data.get("id") != HAFS_MUSHAF_ID:
        raise ImportError_(f"expected the Hafs mushaf (id {HAFS_MUSHAF_ID}), got id {data.get('id')}")
    if not version:
        raise ImportError_("dump has no licence version")
    surahs = data.get("surahs", [])
    if len(surahs) != expected_surahs:
        raise ImportError_(f"expected {expected_surahs} surahs, got {len(surahs)}")

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
            })
    if len(passages) != expected_ayahs:
        raise ImportError_(f"expected {expected_ayahs} ayahs, got {len(passages)}")

    return {
        "synthetic": False,
        "_provenance": {"source": "https://quranpedia.net/dumps", "mushaf": data.get("name"),
                        "description": data.get("description"), "version": version},
        "sources": [{
            "id": SOURCE_ID,
            "name": "القرآن الكريم (رواية حفص عن عاصم)",
            "about": f"نص المصحف موافقًا لطبعة مجمع الملك فهد، من بيانات الموسوعة القرآنية "
                     f"quranpedia.net (نسخة {version}).",
            "url": "https://quranpedia.net",
        }],
        "passages": passages,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("data/quran_corpus.json"))
    args = parser.parse_args(argv)
    with _open(args.dump) as fh:
        corpus = build_corpus(json.load(fh))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    print(f"{len(corpus['passages'])} ayahs written to {args.output} "
          f"(Quranpedia version {corpus['_provenance']['version']})")


if __name__ == "__main__":
    main()
