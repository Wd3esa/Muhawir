"""Disk-backed corpus and search (SQLite FTS5).

Same interface as the in-memory Corpus and Retriever, so the pipeline does not
change, but passages stay on disk: memory stays low and start-up is instant.
Text is tokenized with muhawir.normalize before indexing and querying, so the
Arabic matching rules are identical to the in-memory search.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .corpus import Passage, Source, parse_corpus
from .normalize import tokenize
from .retrieve import Hit

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (id TEXT PRIMARY KEY, name TEXT NOT NULL, about TEXT NOT NULL, url TEXT NOT NULL);
CREATE TABLE passages (
    rid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, source_id TEXT NOT NULL REFERENCES sources(id),
    location TEXT NOT NULL, kind TEXT NOT NULL, grade TEXT NOT NULL, text TEXT NOT NULL,
    keywords TEXT NOT NULL, tokens TEXT NOT NULL);
CREATE VIRTUAL TABLE passages_fts USING fts5(tokens, content='passages', content_rowid='rid',
                                             tokenize='unicode61 remove_diacritics 0');
"""


def build_db(corpus_data: dict, path: str | Path) -> int:
    """Validate a corpus (same rules as the JSON corpus) and write it to a new SQLite file."""
    corpus = parse_corpus(corpus_data)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO meta VALUES (?, ?)", [
            ("synthetic", json.dumps(corpus.synthetic)),
            ("provenance", json.dumps(corpus_data.get("_provenance", {}), ensure_ascii=False))])
        con.executemany("INSERT INTO sources VALUES (?, ?, ?, ?)",
                        [(s.id, s.name, s.about, s.url) for s in corpus.sources.values()])
        con.executemany(
            "INSERT INTO passages (id, source_id, location, kind, grade, text, keywords, tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(p.id, p.source_id, p.location, p.kind, p.grade, p.text, p.keywords,
              " ".join(tokenize(f"{p.text} {p.keywords}"))) for p in corpus.passages])
        con.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
        con.commit()
    finally:
        con.close()
    tmp.replace(path)  # atomic: a half-built database is never served
    return len(corpus.passages)


class SqliteCorpus:
    """Read-only corpus backed by a database written by build_db."""

    def __init__(self, path: str | Path) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        meta = dict(self.con.execute("SELECT key, value FROM meta"))
        self.synthetic = json.loads(meta.get("synthetic", "false"))
        self.provenance = json.loads(meta.get("provenance", "{}"))
        self.sources = {r[0]: Source(*r) for r in self.con.execute("SELECT id, name, about, url FROM sources")}
        self.count = self.con.execute("SELECT count(*) FROM passages").fetchone()[0]

    @property
    def passages(self) -> range:  # only its length is used outside this module
        return range(self.count)

    @staticmethod
    def _row(r) -> Passage:
        return Passage(id=r[0], source_id=r[1], location=r[2], text=r[5], kind=r[3], grade=r[4],
                       keywords=r[6])

    def passage(self, passage_id: str) -> Passage | None:
        r = self.con.execute("SELECT id, source_id, location, kind, grade, text, keywords "
                             "FROM passages WHERE id = ?", (passage_id,)).fetchone()
        return self._row(r) if r else None

    def source_of(self, passage: Passage) -> Source:
        return self.sources[passage.source_id]


class SqliteRetriever:
    """BM25 search through FTS5, returning the same Hit objects as Retriever."""

    def __init__(self, corpus: SqliteCorpus) -> None:
        self.corpus = corpus

    def search(self, question: str, k: int = 3) -> list[Hit]:
        terms = list(dict.fromkeys(tokenize(question)))
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        rows = self.corpus.con.execute(
            "SELECT p.id, p.source_id, p.location, p.kind, p.grade, p.text, p.keywords, p.tokens, "
            "bm25(passages_fts) FROM passages_fts JOIN passages p ON p.rid = passages_fts.rowid "
            "WHERE passages_fts MATCH ? ORDER BY bm25(passages_fts) LIMIT ?", (match, k)).fetchall()
        hits = []
        for r in rows:
            doc_terms = set(r[7].split())
            coverage = sum(1 for t in terms if t in doc_terms) / len(terms)
            hits.append(Hit(SqliteCorpus._row(r), -r[8], coverage))  # FTS5 bm25: lower is better
        return hits
