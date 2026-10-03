"""HTTP API and the web page.

The server does not log question text and stores nothing about the user.
Run: uvicorn muhawir.server:app
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import dorar
from .corpus import load_corpus
from .generate import get_generator
from .pipeline import MAX_QUESTION_CHARS, Muhawir
from .store import SqliteCorpus, SqliteRetriever

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT.parent / "data" / "synthetic_corpus.json"


DEFAULT_DB = ROOT.parent / "data" / "muhawir.db"


def build() -> Muhawir:
    """Use the SQLite database when it exists (built by muhawir.build_data), else a JSON corpus.

    Real religious sources are only served in model mode: keyword matching alone
    returns passages that share words with a question without answering it.
    """
    db = Path(os.environ.get("MUHAWIR_DB") or DEFAULT_DB)
    generator = get_generator()
    # live hadith search from dorar.net: off unless MUHAWIR_DORAR=1 (dorar refused server requests;
    # the two Sahih books in the database are the hadith source until dorar gives permission)
    hadith = dorar.search if generator.name != "extractive" and os.environ.get("MUHAWIR_DORAR", "0") == "1" else None
    if db.exists() and not os.environ.get("MUHAWIR_CORPUS"):
        corpus = SqliteCorpus(db)
        engine = Muhawir(corpus, generator, SqliteRetriever(corpus), hadith_search=hadith,
                         hadith_source=dorar.SOURCE)
    else:
        corpus = load_corpus(os.environ.get("MUHAWIR_CORPUS") or DEFAULT_CORPUS)
        engine = Muhawir(corpus, generator)
    if not corpus.synthetic and generator.name == "extractive":
        raise RuntimeError("real sources need LLM_PROVIDER=model; extractive mode is for test data only")
    return engine


app = FastAPI(title="Muhawir", docs_url=None, redoc_url=None)
engine = build()


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(max_length=4000)


class Ask(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS * 2)
    style: str = "youth"
    lang: str = "ar"
    history: list[Turn] = Field(default_factory=list, max_length=20)  # kept by the browser, not stored here


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    return engine.ask(body.question, body.style, body.lang,
                      [t.model_dump() for t in body.history]).to_dict()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "generator": engine.generator.name, "synthetic": engine.corpus.synthetic,
            "passages": len(engine.corpus.passages)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
