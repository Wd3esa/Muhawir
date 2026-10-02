"""Importer tests. The fixture is a tiny dump in Quranpedia's format; the real
dump is checked too when it is present locally (it is not in the repository)."""
import gzip
import json
import os
from pathlib import Path

import pytest

from muhawir.corpus import parse_corpus
from muhawir.quranpedia import ImportError_, build_corpus, main


def _dump(mushaf_id=1, version="2026-10-02"):
    return {"license": {"version": version}, "data": {
        "id": mushaf_id, "name": "مصحف حفص", "description": "وصف",
        "surahs": [{"id": 1, "name": "سورة أ", "ayahs": [
            {"number": 1, "text": "\ufeff\ufeffنَصٌّ أَوَّلُ"},
            {"number": 2, "text": "\ufeffنَصٌّ ثَانٍ ۚ"}]}]}}


def test_builds_valid_corpus_and_keeps_text_exact():
    corpus = build_corpus(_dump(), expected_surahs=1, expected_ayahs=2)
    parsed = parse_corpus(corpus)
    assert not parsed.synthetic
    first = parsed.passage("q:1:1")
    assert first.text == "نَصٌّ أَوَّلُ" and first.kind == "quran"
    assert parsed.passage("q:1:2").text == "نَصٌّ ثَانٍ ۚ"
    assert first.location == "سورة أ، الآية 1"
    assert "2026-10-02" in parsed.sources["quranpedia-hafs"].about


def test_rejects_other_mushaf():
    with pytest.raises(ImportError_):
        build_corpus(_dump(mushaf_id=2), expected_surahs=1, expected_ayahs=2)


def test_rejects_missing_version():
    with pytest.raises(ImportError_):
        build_corpus(_dump(version=""), expected_surahs=1, expected_ayahs=2)


def test_rejects_wrong_counts():
    with pytest.raises(ImportError_):
        build_corpus(_dump())  # full Quran counts expected by default


def test_cli_refuses_incomplete_dump(tmp_path):
    src = tmp_path / "m.json.gz"
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        json.dump(_dump(), fh)
    with pytest.raises(ImportError_):
        main([str(src), "-o", str(tmp_path / "out.json")])


REAL_DUMP = os.environ.get("QURANPEDIA_MUSHAF")


@pytest.mark.skipif(not REAL_DUMP or not Path(REAL_DUMP).exists(),
                    reason="set QURANPEDIA_MUSHAF to the real mushafs-1.json.gz to run")
def test_real_dump(tmp_path):
    out = tmp_path / "quran.json"
    main([REAL_DUMP, "-o", str(out)])
    corpus = parse_corpus(json.loads(out.read_text(encoding="utf-8")))
    assert len(corpus.passages) == 6236
    with gzip.open(REAL_DUMP, "rt", encoding="utf-8") as fh:
        dump = json.load(fh)["data"]
    for surah in dump["surahs"]:  # every ayah equals the dump text minus leading BOMs
        for ayah in surah["ayahs"]:
            text = corpus.passage(f"q:{surah['id']}:{ayah['number']}").text
            assert text == ayah["text"].lstrip("\ufeff").strip()
    assert all("\ufeff" not in p.text for p in corpus.passages)


def test_topics_become_search_only_keywords():
    topics = {"data": [{"surah": 1, "ayah": 2, "topics": [
        {"name": "موضوع فرعي", "parent": {"name": "موضوع أصل"}},
        {"name": "موضوع أصل", "parent": None}]}]}
    corpus = parse_corpus(build_corpus(_dump(), expected_surahs=1, expected_ayahs=2,
                                       topics_dump=topics))
    second = corpus.passage("q:1:2")
    assert second.keywords == "موضوع أصل؛ موضوع فرعي"
    assert second.text == "نَصٌّ ثَانٍ ۚ"  # the quote itself is unchanged
    assert corpus.passage("q:1:1").keywords == ""
