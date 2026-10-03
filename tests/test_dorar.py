"""Live hadith search from dorar.net: parsing and use in answers. Synthetic HTML, fake search, no network."""
import json

import pytest

from muhawir import dorar
from muhawir.corpus import parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.pipeline import ABSTAINED, ANSWERED, Muhawir

# same layout as the dorar.net API result; the texts are made up
SAMPLE = """<div class="hadith" style="text-align:justify;">1 -   <span class="search-keys">نص</span> حديث تجريبي أول  .</div>
<div class="hadith-info">
    <span class="info-subtitle">الراوي:</span> راو أول</span>
    <span class="info-subtitle">المحدث:</span> محدث أول
    <span class="info-subtitle">المصدر:</span>  كتاب أول
    <span class="info-subtitle">الصفحة أو الرقم:</span>  12/3
    <span class="info-subtitle">خلاصة حكم المحدث:</span>  <span >صحيح</span>
</div>
--------------
<br/>
<div class="hadith" style="text-align:justify;">2 -  نص حديث تجريبي ثان .</div>
<div class="hadith-info">
    <span class="info-subtitle">الراوي:</span> -</span>
    <span class="info-subtitle">المحدث:</span> محدث ثان
    <span class="info-subtitle">المصدر:</span>  كتاب ثان
    <span class="info-subtitle">الصفحة أو الرقم:</span>  7
    <span class="info-subtitle">خلاصة حكم المحدث:</span>  <span >إسناده ضعيف</span>
</div>
--------------
<br/>
<a href="https://dorar.net/hadith/search?q=x">المزيد</a>"""


def test_parse_keeps_text_source_and_grade():
    found = dorar.parse(SAMPLE)
    assert [h["text"] for h in found] == ["نص حديث تجريبي أول", "نص حديث تجريبي ثان"]
    assert found[0] == {"text": "نص حديث تجريبي أول", "rawi": "راو أول", "mohdith": "محدث أول",
                        "book": "كتاب أول", "number": "12/3", "grade": "صحيح"}
    assert found[1]["grade"] == "إسناده ضعيف"


def test_hadith_without_source_or_grade_is_dropped():
    assert dorar.parse('<div class="hadith">1 - نص</div><div class="hadith-info"></div>') == []


def test_passage_carries_grade_and_location():
    p = dorar.to_passage(dorar.parse(SAMPLE)[1])
    assert p.kind == "hadith" and p.grade == "إسناده ضعيف" and p.id.startswith("d:")
    assert p.location == "المحدث: محدث ثان — المصدر: كتاب ثان، 7"  # "-" narrator left out


def test_search_failure_returns_nothing(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "get", boom)
    assert dorar.search("نص") == []


REAL = {"synthetic": False, "sources": [{"id": "s", "name": "مصدر", "about": "وصف"}],
        "passages": [{"id": "s:1", "source_id": "s", "location": "م1", "kind": "tafsir", "text": "شرح عن النية والعمل"}]}
LIVE = [dorar.to_passage(h) for h in dorar.parse(SAMPLE)]


def engine(reply_for, search=lambda q: LIVE, synthetic=False):
    def call(system, user, schema=None):
        if "queries" in json.dumps(schema or {}):
            return '{"queries": []}'
        return json.dumps(reply_for(user), ensure_ascii=False)
    data = dict(REAL, synthetic=synthetic)
    return Muhawir(parse_corpus(data), ModelGenerator([("m", call)]), hadith_search=search,
                   hadith_source=dorar.SOURCE)


def test_answer_can_cite_a_live_hadith_and_card_shows_grade_and_dorar():
    hid = LIVE[0].id
    res = engine(lambda user: {"abstain": False, "claims": [
        {"text": "ورد في حديث صححه المحدث أن العمل بالنية.", "passage_ids": [hid]}]}).ask("النية والعمل")
    assert res.status == ANSWERED
    card = res.sources[0]
    assert card["grade"] == "صحيح" and card["kind"] == "hadith"
    assert card["source_name"] == dorar.SOURCE.name and card["source_url"] == "https://dorar.net/hadith"


def test_prompt_shows_the_grade_to_the_model():
    seen = []
    engine(lambda user: seen.append(user) or {"abstain": True, "claims": []}).ask("النية والعمل")
    assert "حكم المحدث: إسناده ضعيف" in seen[0] and "حكم المحدث" in __import__("muhawir.generate", fromlist=["x"]).SYSTEM_PROMPT


def test_misquoted_hadith_is_rejected():
    hid = LIVE[0].id
    res = engine(lambda user: {"abstain": False, "claims": [
        {"text": "قال: «نص مختلق لا وجود له»", "passage_ids": [hid]}]}).ask("النية والعمل")
    assert res.status == ABSTAINED


def test_dorar_down_still_answers_from_other_sources():
    res = engine(lambda user: {"abstain": False, "claims": [{"text": "شرح.", "passage_ids": ["s:1"]}]},
                 search=lambda q: []).ask("النية والعمل")
    assert res.status == ANSWERED


def test_no_live_search_with_synthetic_test_data():
    calls = []
    engine(lambda user: {"abstain": True, "claims": []}, search=lambda q: calls.append(q) or LIVE,
           synthetic=True).ask("النية والعمل")
    assert calls == []
