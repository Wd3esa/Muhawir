"""SQLite store tests: same behaviour as the in-memory corpus, on synthetic data."""
import gzip
import json
import zipfile
from pathlib import Path

import pytest

from muhawir import build_data, server
from muhawir.corpus import CorpusError
from muhawir.generate import ExtractiveGenerator
from muhawir.pipeline import ABSTAINED, ANSWERED, Muhawir
from muhawir.retrieve import Retriever
from muhawir.corpus import load_corpus
from muhawir.store import SqliteCorpus, SqliteRetriever, build_db

SYNTHETIC = Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json"


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "m.db"
    build_db(json.loads(SYNTHETIC.read_text(encoding="utf-8")), path)
    return path


def test_build_rejects_invalid_corpus(tmp_path):
    with pytest.raises(CorpusError):
        build_db({"sources": [], "passages": []}, tmp_path / "bad.db")
    assert not (tmp_path / "bad.db").exists()


def test_store_matches_in_memory_search(db):
    corpus = SqliteCorpus(db)
    assert corpus.synthetic is True and len(corpus.passages) == 5
    memory = Retriever(load_corpus(SYNTHETIC))
    for q in ("متى يظهر القمر بدرا؟", "ماذا تحتاج النخلة في الصيف؟", "الطيور في الخريف"):
        assert [h.passage.id for h in SqliteRetriever(corpus).search(q)] == \
               [h.passage.id for h in memory.search(q)]


def test_store_passages_are_identical(db):
    corpus, memory = SqliteCorpus(db), load_corpus(SYNTHETIC)
    for p in memory.passages:
        assert corpus.passage(p.id) == p
    assert corpus.passage("missing") is None


def test_pipeline_on_store(db):
    corpus = SqliteCorpus(db)
    engine = Muhawir(corpus, ExtractiveGenerator(), SqliteRetriever(corpus))
    res = engine.ask("ماذا تحتاج النخلة في الصيف؟")
    assert res.status == ANSWERED and res.sources[0]["source_about"]
    assert engine.ask("ما عاصمة اليابان الاقتصادية؟").status == ABSTAINED


def test_quotes_in_query_do_not_break_search(db):
    corpus = SqliteCorpus(db)
    assert SqliteRetriever(corpus).search('النخلة "OR" * ( ) -') is not None


def test_server_prefers_database(db, monkeypatch):
    monkeypatch.setenv("MUHAWIR_DB", str(db))
    monkeypatch.delenv("MUHAWIR_CORPUS", raising=False)
    engine = server.build()
    assert isinstance(engine.corpus, SqliteCorpus)


def test_read_mushaf_from_zip(tmp_path):
    payload = gzip.compress(json.dumps({"data": {"id": 1}}).encode())
    with zipfile.ZipFile(tmp_path / build_data.MUSHAFS_ZIP, "w") as z:
        z.writestr("mushafs/mushafs-1.json.gz", payload)
        z.writestr("mushafs/mushafs-11.json.gz", gzip.compress(b'{"data": {"id": 11}}'))
    assert build_data.read_mushaf(tmp_path)["data"]["id"] == 1


def test_real_sources_refuse_extractive_mode(tmp_path, monkeypatch):
    data = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    data["synthetic"] = False
    build_db(data, tmp_path / "real.db")
    monkeypatch.setenv("MUHAWIR_DB", str(tmp_path / "real.db"))
    monkeypatch.delenv("MUHAWIR_CORPUS", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    with pytest.raises(RuntimeError):
        server.build()
