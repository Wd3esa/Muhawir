"""«بداية المجتهد» from OpenITI mARkdown: markers removed, wording and pages kept."""
import pytest

from muhawir import bidaya
from muhawir.corpus import parse_corpus

SAMPLE = """######OpenITI#
#META# 020.BookTITLE	:: بداية المجتهد ونهاية المقتصد
#META#Header#End#

# PageV01P009
### | [كتاب الصلاة] 
### | [الباب الأول في القراءة]
# المسألة الرابعة اختلفوا في قراءة بسم الله الرحمن الرحيم، فمنع ذلك مالك
~~في المكتوبة. وقال الشافعي: يقرؤها ms0127 جهرا.
# PageV01P131
# وسبب الخلاف آيل إلى شيئين. PageV01P132 ms0128
### | [كتاب الزكاة]
# والزكاة ركن.
# PageV01P200
"""


def test_markers_are_removed_and_wording_kept():
    _, ps = bidaya.build(SAMPLE, min_passages=1)
    assert ps[0]["text"] == ("المسألة الرابعة اختلفوا في قراءة بسم الله الرحمن الرحيم، فمنع ذلك مالك "
                             "في المكتوبة. وقال الشافعي: يقرؤها جهرا.\nوسبب الخلاف آيل إلى شيئين.")
    assert all("ms0" not in p["text"] and "PageV" not in p["text"] and "~~" not in p["text"] for p in ps)


def test_page_marker_ends_its_page_and_location_names_book_chapter_and_pages():
    _, ps = bidaya.build(SAMPLE, min_passages=1)
    assert ps[0]["location"] == "كتاب الصلاة، الباب الأول في القراءة (ج1، ص131–132)"
    assert ps[0]["keywords"] == "كتاب الصلاة؛ الباب الأول في القراءة"
    assert ps[1]["location"] == "كتاب الزكاة (ج1، ص200)"  # a new heading starts a new passage
    assert ps[0]["id"] == "f:1" and ps[0]["kind"] == "fiqh"


def test_passages_join_into_a_valid_corpus_with_attribution():
    source, ps = bidaya.build(SAMPLE, min_passages=1)
    parsed = parse_corpus({"synthetic": False, "sources": [source], "passages": ps})
    about = parsed.sources[bidaya.SOURCE_ID].about
    assert "ابن رشد" in about and "OpenITI" in about and "CC BY-NC-SA 4.0" in about and "رأيه" in about


def test_truncated_or_wrong_file_is_refused():
    with pytest.raises(bidaya.ImportError_):
        bidaya.build(SAMPLE)  # far fewer passages than the real book
    with pytest.raises(bidaya.ImportError_):
        bidaya.build(SAMPLE.replace("بداية المجتهد", "كتاب آخر"), min_passages=1)
