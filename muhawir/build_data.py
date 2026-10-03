"""Build the Muhawir database from Quranpedia data dumps.

Downloads (or reads from a folder) four files from https://quranpedia.net/dumps:
  mushafs-all.zip          Quran text; the Hafs file (mushafs-1.json.gz) is used
  topics.json.gz           verse topics, used as search-only keywords
  tafsir-book-4.json.gz    al-Tabari, "Jami al-Bayan"
  asbab-book-460.json.gz   al-Muzaini, "al-Muharrar fi Asbab Nuzul al-Quran" (reasons of revelation)
plus Sahih al-Bukhari and Sahih Muslim in Arabic from github.com/fawazahmed0/hadith-api
(ara-bukhari.min.json, ara-muslim.min.json; public domain), and Ibn Rushd's
«بداية المجتهد» from the OpenITI corpus (comparative fiqh; CC BY-NC-SA 4.0),
and writes data/muhawir.db. Run at deploy time so the copy is always current,
as the Quranpedia licence asks; the database is never committed.

Usage:
  python -m muhawir.build_data --download            # hosting / first run
  python -m muhawir.build_data --from-dir data/dumps  # files already downloaded
  python -m muhawir.build_data --from-dir data/dumps --get-hadith  # add the two Sahih books and «بداية المجتهد» to an existing folder
"""
from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from pathlib import Path

from . import bidaya, sahihayn
from .quranpedia import build_corpus
from .store import build_db

BASE = "https://quranpedia.net/dumps/"
MUSHAFS_ZIP = "mushafs-all.zip"
MUSHAF_FILE = "mushafs-1.json.gz"
TOPICS_FILE = "topics.json.gz"
TAFSIR_FILE = "tafsir-book-4.json.gz"
ASBAB_FILE = "asbab-book-460.json.gz"


HADITH_FILES = [book["file"] for book in sahihayn.BOOKS.values()]


def _fetch(urls: list[str], target: Path) -> None:
    import httpx

    last = None
    for url in urls:  # the hadith data has a mirror, as its author advises
        try:
            with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
                r.raise_for_status()
                with open(target, "wb") as fh:
                    for block in r.iter_bytes():
                        fh.write(block)
            print(f"downloaded {target.name} ({target.stat().st_size // 1024} KB)")
            return
        except Exception as exc:
            last = exc
    raise RuntimeError(f"could not download {target.name}: {last}")


def download_hadith(folder: Path) -> None:
    """The open-data books: the two Sahih books and «بداية المجتهد»."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in HADITH_FILES:
        _fetch([u.format(file=name) for u in sahihayn.DOWNLOAD], folder / name)
    _fetch(list(bidaya.DOWNLOAD), folder / bidaya.FILE)


def download(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in (MUSHAFS_ZIP, TOPICS_FILE, TAFSIR_FILE, ASBAB_FILE):
        _fetch([BASE + name], folder / name)
    download_hadith(folder)


def _load_gz(data: bytes) -> dict:
    return json.loads(gzip.decompress(data).decode("utf-8"))


def read_mushaf(folder: Path) -> dict:
    direct = folder / MUSHAF_FILE
    if direct.exists():
        return _load_gz(direct.read_bytes())
    with zipfile.ZipFile(folder / MUSHAFS_ZIP) as z:
        name = next(n for n in z.namelist() if n.endswith(MUSHAF_FILE))
        return _load_gz(z.read(name))


def build(folder: Path, out: Path) -> int:
    mushaf = read_mushaf(folder)
    topics = _load_gz((folder / TOPICS_FILE).read_bytes())
    tafsir = _load_gz((folder / TAFSIR_FILE).read_bytes())
    asbab = _load_gz((folder / ASBAB_FILE).read_bytes())
    corpus = build_corpus(mushaf, topics_dump=topics, tafsir_dump=tafsir, asbab_dump=asbab)
    del mushaf, topics, tafsir, asbab
    books = {key: json.loads((folder / book["file"]).read_text(encoding="utf-8"))
             for key, book in sahihayn.BOOKS.items() if (folder / book["file"]).exists()}
    if books:
        sahihayn.add_to_corpus(corpus, books)
    else:
        print("note: the two Sahih books are not in this folder; add them with --get-hadith")
    del books
    if (folder / bidaya.FILE).exists():
        bidaya.add_to_corpus(corpus, (folder / bidaya.FILE).read_text(encoding="utf-8"))
    else:
        print("note: «بداية المجتهد» is not in this folder; add it with --get-hadith")
    out.parent.mkdir(parents=True, exist_ok=True)
    count = build_db(corpus, out)
    print(f"{count} passages written to {out} "
          f"(Quranpedia mushaf {corpus['_provenance']['version']}, "
          f"tafsir {corpus['_provenance']['tafsir_version']}, "
          f"asbab {corpus['_provenance']['asbab_version']}, "
          f"bukhari {corpus['_provenance'].get('bukhari_entries', 0)}, "
          f"muslim {corpus['_provenance'].get('muslim_entries', 0)}, "
          f"bidaya {'yes' if corpus['_provenance'].get('bidaya_commit') else 'no'})")
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build data/muhawir.db from Quranpedia dumps")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true", help="download the dumps into data/dumps")
    group.add_argument("--from-dir", type=Path, help="folder that already holds the dumps")
    parser.add_argument("--get-hadith", action="store_true",
                        help="download only the two Sahih books and «بداية المجتهد» into the folder, then build")
    parser.add_argument("--out", type=Path, default=Path("data/muhawir.db"))
    args = parser.parse_args(argv)
    folder = args.from_dir or Path("data/dumps")
    if args.download:
        download(folder)
    elif args.get_hadith:
        download_hadith(folder)
    build(folder, args.out)


if __name__ == "__main__":
    main()
