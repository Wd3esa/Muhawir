"""Build the Muhawir database from Quranpedia data dumps.

Downloads (or reads from a folder) four files from https://quranpedia.net/dumps:
  mushafs-all.zip          Quran text; the Hafs file (mushafs-1.json.gz) is used
  topics.json.gz           verse topics, used as search-only keywords
  tafsir-book-4.json.gz    al-Tabari, "Jami al-Bayan"
  asbab-book-460.json.gz   al-Muzaini, "al-Muharrar fi Asbab Nuzul al-Quran" (reasons of revelation)
and writes data/muhawir.db. Run at deploy time so the copy is always current,
as the Quranpedia licence asks; the database is never committed.

Usage:
  python -m muhawir.build_data --download            # hosting / first run
  python -m muhawir.build_data --from-dir data/dumps  # files already downloaded
"""
from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from pathlib import Path

from .quranpedia import build_corpus
from .store import build_db

BASE = "https://quranpedia.net/dumps/"
MUSHAFS_ZIP = "mushafs-all.zip"
MUSHAF_FILE = "mushafs-1.json.gz"
TOPICS_FILE = "topics.json.gz"
TAFSIR_FILE = "tafsir-book-4.json.gz"
ASBAB_FILE = "asbab-book-460.json.gz"


def download(folder: Path) -> None:
    import httpx

    folder.mkdir(parents=True, exist_ok=True)
    for name in (MUSHAFS_ZIP, TOPICS_FILE, TAFSIR_FILE, ASBAB_FILE):
        target = folder / name
        with httpx.stream("GET", BASE + name, timeout=300, follow_redirects=True) as r:
            r.raise_for_status()
            with open(target, "wb") as fh:
                for block in r.iter_bytes():
                    fh.write(block)
        print(f"downloaded {name} ({target.stat().st_size // 1024} KB)")


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
    out.parent.mkdir(parents=True, exist_ok=True)
    count = build_db(corpus, out)
    print(f"{count} passages written to {out} "
          f"(Quranpedia mushaf {corpus['_provenance']['version']}, "
          f"tafsir {corpus['_provenance']['tafsir_version']}, "
          f"asbab {corpus['_provenance']['asbab_version']})")
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build data/muhawir.db from Quranpedia dumps")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true", help="download the dumps into data/dumps")
    group.add_argument("--from-dir", type=Path, help="folder that already holds the dumps")
    parser.add_argument("--out", type=Path, default=Path("data/muhawir.db"))
    args = parser.parse_args(argv)
    folder = args.from_dir or Path("data/dumps")
    if args.download:
        download(folder)
    build(folder, args.out)


if __name__ == "__main__":
    main()
