"""Precise abstain when the reasons-of-revelation source has no entry. Synthetic data."""
import json

from muhawir.corpus import parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.pipeline import ABSTAINED, Muhawir

DATA = {"synthetic": True,
        "sources": [{"id": "q", "name": "نص تجريبي", "about": "مصطنع"},
                    {"id": "a", "name": "كتاب أسباب تجريبي", "about": "مصطنع"}],
        "passages": [
            {"id": "q:1:1", "source_id": "q", "location": "سورة الشمس، الآية 1", "kind": "quran", "text": "نور ساطع في السماء"},
            {"id": "q:2:1", "source_id": "q", "location": "سورة القمر، الآية 1", "kind": "quran", "text": "قمر منير في الليل"},
            {"id": "q:2:2", "source_id": "q", "location": "سورة القمر، الآية 2", "kind": "quran", "text": "نجم بعيد يلمع وحده"},
            {"id": "q:2:3", "source_id": "q", "location": "سورة القمر، الآية 3", "kind": "quran", "text": "سحاب ثقيل يحمل المطر"},
            {"id": "a1:2:1:1:1", "source_id": "a", "location": "سورة القمر، الآية 1", "kind": "asbab",
             "text": "قال تعالى: قمر منير في الليل، نجم بعيد يلمع وحده. سبب النزول: رواية تجريبية."}]}


def engine():
    abstain = lambda system, user, schema=None: json.dumps(  # noqa: E731
        {"queries": []} if "queries" in json.dumps(schema or {}) else {"abstain": True, "claims": []})
    return Muhawir(parse_corpus(DATA), ModelGenerator([("m", abstain)]))


def test_surah_without_entry_is_named():
    res = engine().ask("سبب نزول سورة الشمس؟")
    assert res.status == ABSTAINED
    assert res.message.startswith("لم يُذكر لسورة الشمس سبب نزول") and "كتاب أسباب تجريبي" in res.message


def test_ayah_without_entry_is_named():
    res = engine().ask("سبب نزول آية سحاب ثقيل يحمل المطر")
    assert res.status == ABSTAINED and "(سورة القمر، الآية 3)" in res.message


def test_ayah_quoted_inside_an_entry_filed_under_another_ayah_is_not_called_missing():
    res = engine().ask("سبب نزول نجم بعيد يلمع وحده")  # ayah 2, filed under ayah 1 in the book
    assert res.status == ABSTAINED and res.message.startswith("لم أجد")


def test_surah_with_entries_keeps_general_message():
    assert engine().ask("سبب نزول سورة القمر").message.startswith("لم أجد")


def test_other_questions_keep_general_message():
    assert engine().ask("ما معنى سحاب ثقيل يحمل المطر").message.startswith("لم أجد")
