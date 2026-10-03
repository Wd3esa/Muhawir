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


def test_tafsir_passages_strip_markup_and_keep_wording():
    text = '{١ - ٢} <span class="x">﴿نَصٌّ﴾</span> أي: شرح أول.<br />\rشرح &amp; ثانٍ.'
    tafsir = {"license": {"version": "2026-08-10"},
              "book": {"id": 3, "name": "كتاب", "short_name": "مختصر", "author": {"ar_name": "مؤلف"}},
              "ayahs": [{"surah": 1, "ayah": 1, "content": [{"text": text, "part": 1, "page": 5}]},
                        {"surah": 1, "ayah": 2, "content": [{"text": text, "part": 1, "page": 5}]}]}
    corpus = parse_corpus(build_corpus(_dump(), expected_surahs=1, expected_ayahs=2,
                                       tafsir_dump=tafsir))
    tafsir_passages = [p for p in corpus.passages if p.kind == "tafsir"]
    assert len(tafsir_passages) == 1  # one text covering two ayahs is stored once
    p = tafsir_passages[0]
    assert p.text == "{١ - ٢} ﴿نَصٌّ﴾ أي: شرح أول.\nشرح & ثانٍ."
    assert p.location == "سورة أ، الآيات 1–2 (ج1، ص5)"
    assert "2026-08-10" in corpus.sources["quranpedia-tafsir-3"].about


def test_asbab_book_is_imported_as_its_own_kind():
    book = {"license": {"version": "2026-08-10"},
            "book": {"id": 460, "name": "كتاب الأسباب", "short_name": "الأسباب", "author": {"ar_name": "مؤلف"}},
            "ayahs": [{"surah": 1, "ayah": 2, "content": [
                {"text": "<strong>* سَبَبُ النُّزُولِ:</strong><br />\rأخرج البخاري قال: فنزلت.", "part": "2", "page": 9}]}]}
    tafsir = {"license": {"version": "2026-08-10"},
              "book": {"id": 4, "name": "تفسير", "author": {"ar_name": "مفسر"}},
              "ayahs": [{"surah": 1, "ayah": 2, "content": [{"text": "شرح.", "part": 1, "page": 1}]}]}
    corpus = parse_corpus(build_corpus(_dump(), expected_surahs=1, expected_ayahs=2,
                                       tafsir_dump=tafsir, asbab_dump=book))
    asbab = [p for p in corpus.passages if p.kind == "asbab"]
    assert len(asbab) == 1 and asbab[0].id.startswith("a460:1:2:")
    assert asbab[0].text == "* سَبَبُ النُّزُولِ:\nأخرج البخاري قال: فنزلت."
    about = corpus.sources["quranpedia-asbab-460"].about
    assert "الكتب التسعة" in about and "دراسة المؤلف" in about and "2026-08-10" in about
    assert [p.kind for p in corpus.passages].count("tafsir") == 1  # tafsir ids unchanged
    assert any(p.id.startswith("t4:") for p in corpus.passages)


def test_editor_footnotes_are_left_out_of_the_tafsir_and_the_author_text_is_kept():
    page = ('قال أبو جعفر: كلام المؤلف (١).<br />وكلام آخر للمؤلف.<br />'
            '<div class="foot-notes">(١) في المطبوعة: "كذا".<br />بيت شعر في الحاشية</div>')
    tafsir = {"license": {"version": "2026-08-10"},
              "book": {"id": 4, "name": "تفسير", "author": {"ar_name": "مفسر"}},
              "ayahs": [{"surah": 1, "ayah": 1, "content": [{"text": page, "part": 1, "page": 1}]}]}
    asbab = {"license": {"version": "2026-08-10"},
             "book": {"id": 460, "name": "الأسباب", "author": {"ar_name": "مؤلف"}},
             "ayahs": [{"surah": 1, "ayah": 1, "content": [{"text": page, "part": 1, "page": 1}]}]}
    corpus = parse_corpus(build_corpus(_dump(), expected_surahs=1, expected_ayahs=2,
                                       tafsir_dump=tafsir, asbab_dump=asbab))
    [t] = [p for p in corpus.passages if p.kind == "tafsir"]
    assert t.text == "قال أبو جعفر: كلام المؤلف (١).\nوكلام آخر للمؤلف."  # the author's words, unchanged
    assert "حواشي محقق الطبعة" in corpus.sources["quranpedia-tafsir-4"].about
    [a] = [p for p in corpus.passages if p.kind == "asbab"]
    assert "في المطبوعة" in a.text  # only the tafsir edition's notes are left out
