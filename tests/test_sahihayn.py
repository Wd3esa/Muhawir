"""Import of Sahih al-Bukhari and Sahih Muslim. Made-up texts in the same layout as hadith-api."""
import pytest

from muhawir import sahihayn
from muhawir.corpus import parse_corpus


def dump(entries, filler=0):
    hadiths = [{"hadithnumber": i + 1, "arabicnumber": n, "text": t} for i, (n, t) in enumerate(entries)]
    hadiths += [{"hadithnumber": 90000 + i, "arabicnumber": 90000 + i, "text": f"نص {i}"} for i in range(filler)]
    return {"metadata": {"name": "x"}, "hadiths": hadiths}


def test_keeps_standard_number_and_wording(monkeypatch):
    monkeypatch.setitem(sahihayn.MIN_HADITH, "bukhari", 1)
    _, ps = sahihayn.build_book("bukhari", dump([(1, "حَدَّثَنَا فلان‏ قال : \"نص تجريبي\"")]))
    assert ps[0]["id"] == "b:1" and ps[0]["location"] == "صحيح البخاري، رقم 1"
    assert ps[0]["text"] == "حَدَّثَنَا فلان قال : \"نص تجريبي\"" and ps[0]["kind"] == "hadith"


def test_entries_without_a_standard_number_are_left_out(monkeypatch):
    monkeypatch.setitem(sahihayn.MIN_HADITH, "muslim", 1)
    _, ps = sahihayn.build_book("muslim", dump([(None, "مقدمة"), ("", "أخرى"), (5, "نص")]))
    assert [p["id"] for p in ps] == ["m:5"]


def test_muslim_sub_number_is_shown_under_its_main_number(monkeypatch):
    monkeypatch.setitem(sahihayn.MIN_HADITH, "muslim", 1)
    _, ps = sahihayn.build_book("muslim", dump([("8.01", "طريق أول"), ("8.02", "طريق ثان")]))
    assert [p["id"] for p in ps] == ["m:8.01", "m:8.02"]
    assert {p["location"] for p in ps} == {"صحيح مسلم، رقم 8"}


def test_long_hadith_is_split_at_sentence_ends_only(monkeypatch):
    monkeypatch.setitem(sahihayn.MIN_HADITH, "bukhari", 1)
    sentence = "كلمة " * 100 + "انتهت."
    _, ps = sahihayn.build_book("bukhari", dump([(3, " ".join([sentence] * 6))]))
    assert len(ps) > 1 and ps[0]["id"] == "b:3:1" and "(الجزء 1 من" in ps[0]["location"]
    assert all(p["text"].endswith("انتهت.") for p in ps)


def test_duplicate_numbers_and_short_files_are_refused(monkeypatch):
    monkeypatch.setitem(sahihayn.MIN_HADITH, "bukhari", 1)
    with pytest.raises(sahihayn.ImportError_):
        sahihayn.build_book("bukhari", dump([(1, "أ"), (1, "ب")]))
    monkeypatch.setitem(sahihayn.MIN_HADITH, "bukhari", 7000)
    with pytest.raises(sahihayn.ImportError_):
        sahihayn.build_book("bukhari", dump([(1, "أ")]))


def test_added_books_form_a_valid_corpus(monkeypatch):
    monkeypatch.setitem(sahihayn.MIN_HADITH, "bukhari", 1)
    corpus = {"synthetic": False, "sources": [], "passages": []}
    sahihayn.add_to_corpus(corpus, {"bukhari": dump([(1, "نص أول"), (2, "نص ثان")])})
    parsed = parse_corpus(corpus)
    assert parsed.sources["sahih-bukhari"].name == "صحيح البخاري" and len(parsed.passages) == 2
    assert corpus["_provenance"]["bukhari_entries"] == 2
