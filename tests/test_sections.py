"""Chapter and section openings of «بداية المجتهد» are offered first; pasted quotations are rejected."""
import json
from pathlib import Path

import pytest

from muhawir.corpus import Passage, load_corpus, parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.pipeline import ANSWERED, Muhawir
from muhawir.sections import SectionIndex

B = "bidayat-al-mujtahid"


def _corpus():
    def p(pid, kw, text):
        return {"id": pid, "source_id": B, "location": kw, "kind": "fiqh", "text": text, "keywords": kw}
    return parse_corpus({"synthetic": True, "sources": [{"id": B, "name": "بداية المجتهد", "about": "كتاب", "url": ""}],
        "passages": [
            p("f:1", "كتاب الزكاة؛ الجملة الأولى في معرفة من تجب عليه الزكاة", "كتاب الزكاة ينحصر في خمس جمل."),
            p("f:2", "كتاب الزكاة؛ الجملة الأولى في معرفة من تجب عليه الزكاة", "واختلفوا في مال اليتيم."),
            p("f:3", "كتاب الزكاة؛ الجملة الثانية في معرفة ما تجب فيه الزكاة من الأموال", "الذهب والفضة والإبل والبقر والغنم."),
            p("f:4", "كتاب القسمة؛ الباب الأول في أنواع القسمة", "أنواع القسمة ثلاثة."),
            p("f:5", "كتاب الصلاة؛ الجملة الأولى في معرفة وجوب الصلاة", "الصلاة واجبة."),
        ]})


def test_matching_chapter_opening_and_section_opening_come_first():
    ix = SectionIndex(_corpus())
    assert ix.match(["ما معنى الزكاة وما أنواعها؟", "أنواع الأموال التي تجب فيها الزكاة"]) == ["f:1", "f:3"]


def test_common_words_alone_do_not_pull_in_another_chapter():
    ix = SectionIndex(_corpus())
    assert "f:4" not in ix.match(["ما أنواع الزكاة؟"])  # «أنواع» alone does not bring «أنواع القسمة»
    assert ix.match(["من خلق الله؟"]) == []


SYN = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")


@pytest.mark.real_check
def test_pasted_quotation_is_rejected_and_rewritten_in_own_words():
    prompts = []
    answers = iter([
        {"plan": "", "abstain": False, "as_list": False, "views": [], "claims": [
            {"text": "حيث ذُكر: «تحتاج النخلة إلى ماء كثير في الصيف»", "passage_ids": ["test-a:1"]}]},
        {"plan": "", "abstain": False, "as_list": False, "views": [], "claims": [
            {"text": "النخلة تشرب ماء كثيرا في الصيف.", "passage_ids": ["test-a:1"]}]}])

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "supported" in keys:
            return json.dumps({"supported": [True] * user.count("الجملة ")})
        if '"question"' in keys or "queries" in keys:
            return '{"queries": []}'
        prompts.append(user)
        return json.dumps(next(answers), ensure_ascii=False)

    res = Muhawir(SYN, ModelGenerator([("m", call)])).ask("ماذا تحتاج النخلة في الصيف؟")
    assert res.status == ANSWERED and [c["text"] for c in res.claims] == ["النخلة تشرب ماء كثيرا في الصيف."]
    assert len(prompts) == 2 and "مراجعة لجواب سابق" in prompts[1]
