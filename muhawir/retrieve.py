"""Keyword retrieval (BM25) with an explicit sufficiency check.

If no passage covers the question well enough, the pipeline abstains instead
of letting a model answer from its general knowledge. The thresholds below
are starting values; they are to be calibrated on the evaluation list.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .corpus import Corpus, Passage
from .normalize import tokenize

MIN_SCORE = 1.0      # BM25 score of the best passage
MIN_COVERAGE = 0.5   # share of distinct question terms found in the best passage


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    coverage: float


class Retriever:
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.docs = [tokenize(f"{p.text} {p.keywords}") for p in corpus.passages]
        self.tfs = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, question: str, k: int = 3) -> list[Hit]:
        terms = list(dict.fromkeys(tokenize(question)))
        if not terms:
            return []
        hits = []
        for passage, tf, doc in zip(self.corpus.passages, self.tfs, self.docs):
            score, matched = 0.0, 0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                matched += 1
                denom = f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
                score += self.idf[t] * f * (self.k1 + 1) / denom
            if score > 0:
                hits.append(Hit(passage, score, matched / len(terms)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def is_sufficient(hits: list[Hit], min_score: float = MIN_SCORE,
                  min_coverage: float = MIN_COVERAGE) -> bool:
    return bool(hits) and hits[0].score >= min_score and hits[0].coverage >= min_coverage
