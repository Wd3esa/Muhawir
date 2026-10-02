"""Answer generators.

Until the model provider is chosen (decision D1), the only generator is the
extractive one: it quotes the retrieved passages word for word and adds
nothing of its own. A model-based generator must return the same structure
(claims tagged with passage ids) so the verifier can check it.
"""
from __future__ import annotations

import os
from typing import Protocol

from .corpus import Passage
from .verify import Claim


class Generator(Protocol):
    name: str

    def generate(self, question: str, passages: list[Passage],
                 style: str, lang: str) -> list[Claim]: ...


class ExtractiveGenerator:
    """Quotes each passage verbatim, one claim per passage."""

    name = "extractive"

    def generate(self, question: str, passages: list[Passage],
                 style: str, lang: str) -> list[Claim]:
        return [Claim(f"«{p.text}»", (p.id,)) for p in passages]


def get_generator() -> Generator:
    provider = os.environ.get("LLM_PROVIDER", "extractive").strip().lower()
    if provider in ("", "extractive"):
        return ExtractiveGenerator()
    raise NotImplementedError(
        f"LLM_PROVIDER={provider!r} is not available yet: the model provider is decision D1")
