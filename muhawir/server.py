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

from .corpus import load_corpus
from .generate import get_generator
from .pipeline import MAX_QUESTION_CHARS, Muhawir

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT.parent / "data" / "synthetic_corpus.json"


def build() -> Muhawir:
    corpus = load_corpus(os.environ.get("MUHAWIR_CORPUS") or DEFAULT_CORPUS)
    return Muhawir(corpus, get_generator())


app = FastAPI(title="Muhawir", docs_url=None, redoc_url=None)
engine = build()


class Ask(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS * 2)
    style: str = "youth"
    lang: str = "ar"


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    return engine.ask(body.question, body.style, body.lang).to_dict()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "generator": engine.generator.name, "synthetic": engine.corpus.synthetic,
            "passages": len(engine.corpus.passages)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
