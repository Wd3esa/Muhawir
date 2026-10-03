"""Model-mode tests with fake model calls. No network, no API key."""
import json
from pathlib import Path

import pytest

from muhawir import generate
from muhawir.corpus import load_corpus
from muhawir.generate import ModelGenerator, build_user_prompt, get_generator, parse_draft
from muhawir.pipeline import ABSTAINED, ANSWERED, REFERRED, UNAVAILABLE, Muhawir

CORPUS = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")
QUESTION = "ماذا تحتاج النخلة في الصيف؟"


def fake(reply):
    calls = []

    def call(system, user, schema=None):
        calls.append((system, user))
        if "queries" in json.dumps(schema or {}):
            return '{"queries": []}'
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
    call.calls = calls
    return call


def engine(*replies):
    return Muhawir(CORPUS, ModelGenerator([(f"m{i}", fake(r)) for i, r in enumerate(replies)]))


def test_parse_draft():
    assert parse_draft('{"abstain": true, "claims": []}') == []
    assert parse_draft("not json") == []
    assert parse_draft('{"abstain": false, "claims": [{"text": 1, "passage_ids": []}]}') == []
    claims = parse_draft('{"abstain": false, "claims": [{"text": "ت", "passage_ids": ["a"]}]}')
    assert claims[0].text == "ت" and claims[0].passage_ids == ("a",)


def test_answer_cites_only_retrieved_passages():
    res = engine({"abstain": False, "claims": [
        {"text": "تحتاج النخلة إلى ماء كثير في الصيف.", "passage_ids": ["test-a:1"]}]}).ask(QUESTION)
    assert res.status == ANSWERED
    assert [c["passage_id"] for c in res.sources] == ["test-a:1"]


def test_invented_quote_is_rejected_and_answer_abstains():
    res = engine({"abstain": False, "claims": [
        {"text": "قال: ﴿النخلة تحتاج إلى الثلج﴾", "passage_ids": ["test-a:1"]}]}).ask(QUESTION)
    assert res.status == ABSTAINED


def test_citation_outside_retrieved_set_is_rejected():
    res = engine({"abstain": False, "claims": [
        {"text": "الجمل يصبر على العطش.", "passage_ids": ["test-b:1"]}]}).ask(QUESTION)
    assert res.status == ABSTAINED


def test_model_abstain_is_respected():
    assert engine({"abstain": True, "claims": []}).ask(QUESTION).status == ABSTAINED


def test_fallback_used_when_primary_fails():
    gen = ModelGenerator([("claude", fake(RuntimeError("down"))),
                          ("gemini", fake({"abstain": False, "claims": [
                              {"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]}))])
    res = Muhawir(CORPUS, gen).ask(QUESTION)
    assert res.status == ANSWERED and gen.last_used == "gemini"


def test_all_models_failing_says_unavailable_not_abstain():
    res = engine(RuntimeError("a"), RuntimeError("b")).ask(QUESTION)
    assert res.status == UNAVAILABLE and res.claims == [] and "غير متاحة مؤقتًا" in res.message


def test_personal_case_with_model_is_referred_and_prompt_says_so():
    call = fake({"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]})
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask("هل يلزمني أن أسقي النخلة في الصيف؟")
    assert res.status == REFERRED
    assert any("حالة شخصية" in user for _system, user in call.calls)


def test_prompt_marks_question_as_data_and_lists_passage_ids():
    prompt = build_user_prompt("سؤال", [CORPUS.passage("test-a:1")], "kids", "en", False)
    assert "<<<سؤال>>>" in prompt and "[test-a:1]" in prompt and "الإنجليزية" in prompt
    assert "تجاهل" in generate.SYSTEM_PROMPT  # rule against instructions inside data


def test_model_mode_needs_a_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "model")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_generator()


def test_anthropic_request_shape(monkeypatch):
    sent = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": '{"abstain": true, "claims": []}'}]}

    def post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, body=json)
        return Resp()

    import httpx
    monkeypatch.setattr(httpx, "post", post)
    out = generate.anthropic_call("k", "claude-sonnet-5-5")("sys", "user", generate.SCHEMA)
    assert out == '{"abstain": true, "claims": []}'
    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert sent["headers"]["x-api-key"] == "k" and sent["headers"]["anthropic-version"]
    assert sent["body"]["model"] == "claude-sonnet-5-5"
    assert sent["body"]["output_config"]["format"]["type"] == "json_schema"


def test_expanded_queries_reach_passages_the_question_wording_misses():
    def call(system, user, schema=None):
        if "queries" in json.dumps(schema or {}):
            return '{"queries": ["النخلة الصيف ماء"]}'
        ids = ["test-a:1"] if "[test-a:1]" in user else []
        return json.dumps({"abstain": not ids, "claims": [{"text": "ماء كثير.", "passage_ids": ids}] if ids else []})
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask("ما حاجة شجرة البلح؟")
    assert res.status == ANSWERED and res.sources[0]["passage_id"] == "test-a:1"


def test_expansion_failure_falls_back_to_question_only():
    def call(system, user, schema=None):
        if "queries" in json.dumps(schema or {}):
            raise RuntimeError("down")
        return json.dumps({"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]})
    assert Muhawir(CORPUS, ModelGenerator([("m", call)])).ask(QUESTION).status == ANSWERED


# --- scholars' views panel and the "new to Islam" style -------------------

from muhawir.corpus import parse_corpus  # noqa: E402

VIEWS_CORPUS = parse_corpus({"synthetic": True, "sources": [{"id": "s", "name": "مصدر تجريبي", "about": "مصطنع"}],
    "passages": [
        {"id": "v1", "source_id": "s", "location": "ب1",
         "text": "اختلفوا في حكم السقي: فقال العالم سين: السقي واجب. وقال العالم صاد: السقي مستحب."},
        {"id": "v2", "source_id": "s", "location": "ب2", "text": "حديث عن البحر"},
        {"id": "v3", "source_id": "s", "location": "ب3", "text": "حديث عن الجبال"},
        {"id": "v4", "source_id": "s", "location": "ب4", "text": "حديث عن المطر"}]})


def views_engine(reply):
    return Muhawir(VIEWS_CORPUS, ModelGenerator([("m", fake(reply))]))


ANSWER = {"text": "في المسألة أكثر من قول، فاسأل مختصًا.", "passage_ids": ["v1"]}


def test_named_views_are_shown_without_preference():
    res = views_engine({"abstain": False, "claims": [ANSWER], "views": [
        {"school": "العالم سين", "text": "واجب", "passage_ids": ["v1"]},
        {"school": "العالم صاد", "text": "مستحب", "passage_ids": ["v1"]}]}).ask("حكم السقي")
    assert res.status == ANSWERED
    assert [v["school"] for v in res.views] == ["العالم سين", "العالم صاد"]
    assert res.claims == [{"text": ANSWER["text"], "passage_ids": ["v1"]}]


def test_view_of_a_school_not_named_in_the_passage_is_dropped():
    res = views_engine({"abstain": False, "claims": [ANSWER], "views": [
        {"school": "الحنابلة", "text": "واجب", "passage_ids": ["v1"]}]}).ask("حكم السقي")
    assert res.status == ANSWERED and res.views == []


def test_views_without_a_sourced_answer_abstain():
    res = views_engine({"abstain": False, "claims": [], "views": [
        {"school": "العالم سين", "text": "واجب", "passage_ids": ["v1"]}]}).ask("حكم السقي")
    assert res.status == ABSTAINED and res.views == []


def test_newcomer_style_reaches_the_prompt():
    call = fake({"abstain": True, "claims": [], "views": []})
    Muhawir(CORPUS, ModelGenerator([("m", call)])).ask(QUESTION, style="newcomer")
    assert any("جديد على الإسلام" in user for _s, user in call.calls)


def test_parse_draft_reads_views():
    claims = parse_draft(json.dumps({"abstain": False, "claims": [ANSWER], "views": [
        {"school": "س", "text": "ق", "passage_ids": ["v1"]}]}))
    assert [c.school for c in claims] == ["", "س"]


def test_failed_call_is_logged_without_the_question(caplog):
    caplog.set_level("WARNING", logger="muhawir")
    engine(RuntimeError("model not found")).ask(QUESTION)
    assert "model not found" in caplog.text and QUESTION not in caplog.text


# --- retry on busy errors, open-model connector, tolerant JSON ---------------

class _HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = type("R", (), {"status_code": status, "text": "busy"})()


def test_busy_error_is_retried_once():
    replies = [_HTTPError(503), '{"queries": []}']

    def call(system, user, schema):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    assert generate.with_retry(call, "s", "u", {}, sleep=lambda s: None) == '{"queries": []}'


def test_other_errors_are_not_retried():
    calls = []

    def call(system, user, schema):
        calls.append(1)
        raise _HTTPError(404)
    with pytest.raises(_HTTPError):
        generate.with_retry(call, "s", "u", {}, sleep=lambda s: None)
    assert len(calls) == 1


def test_reply_with_think_block_and_fence_is_parsed():
    raw = '<think>reasoning</think>\n```json\n{"abstain": false, "claims": [{"text": "ت", "passage_ids": ["a"]}], "views": []}\n```'
    assert parse_draft(raw)[0].text == "ت"


def test_open_model_request_shape(monkeypatch):
    sent = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"queries": ["أ"]}'}}]}

    def post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, body=json)
        return Resp()

    import httpx
    monkeypatch.setattr(httpx, "post", post)
    out = generate.openai_compatible_call("http://localhost:11434/v1/", "qwen")("s", "u", {})
    assert out == '{"queries": ["أ"]}'
    assert sent["url"] == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in sent["headers"]
    assert sent["body"]["model"] == "qwen" and sent["body"]["messages"][0]["role"] == "system"


def test_open_model_selected_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "model")
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "qwen")
    assert get_generator().name == "open-model"


def test_passage_ids_written_in_the_answer_are_hidden():
    res = engine({"abstain": False, "claims": [
        {"text": "تحتاج النخلة إلى ماء كثير [test-a:1].", "passage_ids": ["test-a:1"]}]}).ask(QUESTION)
    assert res.claims[0]["text"] == "تحتاج النخلة إلى ماء كثير."


def test_open_model_json_slips_do_not_discard_good_claims():
    good = {"text": "ماء كثير.", "passage_ids": ["[test-a:1]"]}
    assert parse_draft(json.dumps({"claims": [good]}))[0].passage_ids == ("test-a:1",)   # no "abstain"
    assert parse_draft(json.dumps({"abstain": "false", "claims": [good]}))                # string
    assert parse_draft(json.dumps({"abstain": False, "claims": [{"text": 1}, good]}))     # one bad item
    assert parse_draft(json.dumps({"abstain": False, "claims": [dict(good, passage_ids="test-a:1")]}))
    assert parse_draft(json.dumps({"abstain": True, "claims": [good]})) == []
    assert parse_draft(json.dumps({"abstain": "true", "claims": [good]})) == []


def test_reason_for_not_answering_is_logged_without_the_question(caplog, monkeypatch):
    from muhawir import pipeline
    monkeypatch.setattr(pipeline, "DEBUG", True)
    caplog.set_level("WARNING", logger="muhawir")
    res = engine({"abstain": False, "claims": [{"text": "قال: «نص مختلق»", "passage_ids": ["test-a:1"]}]}).ask(QUESTION)
    assert res.status == ABSTAINED and "quotation not found" in res.why
    assert "not answered" in caplog.text and QUESTION not in caplog.text


def test_json_with_text_around_it_is_read():
    raw = 'Here is the answer:\n{"abstain": false, "claims": [{"text": "ت", "passage_ids": ["a"]}]}\nI hope this helps.'
    assert parse_draft(raw)[0].text == "ت"
    assert parse_draft('```json\n{"abstain": false, "claims": [{"text": "ت", "passage_ids": ["a"]}]}') [0].text == "ت"
    assert parse_draft("لا أعرف") == []


# --- second reading of each sentence against its passage --------------------

def _checker(answer, verdict):
    seen = []

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "supported" in keys:
            seen.append(user)
            if isinstance(verdict, Exception):
                raise verdict
            return json.dumps({"supported": verdict})
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps(answer, ensure_ascii=False)
    return Muhawir(CORPUS, ModelGenerator([("m", call)])), seen


TWO = {"abstain": False, "claims": [{"text": "تحتاج النخلة إلى ماء كثير.", "passage_ids": ["test-a:1"]},
                                    {"text": "لا تحتاج النخلة إلى الماء.", "passage_ids": ["test-a:1"]}]}


@pytest.mark.real_check
def test_sentence_not_supported_by_its_passage_is_dropped():
    m, seen = _checker(TWO, [True, False])
    res = m.ask(QUESTION)
    assert res.status == ANSWERED and [c["text"] for c in res.claims] == ["تحتاج النخلة إلى ماء كثير."]
    assert "[test-a:1]" in seen[0] and QUESTION not in seen[0]  # the check sees claims and passages only


@pytest.mark.real_check
def test_failed_support_check_fails_closed():
    m, _ = _checker(TWO, RuntimeError("down"))
    res = m.ask(QUESTION)
    assert res.status == UNAVAILABLE and res.claims == []  # nothing unchecked is shown


@pytest.mark.real_check
def test_wrong_number_of_verdicts_fails_closed():
    m, _ = _checker(TWO, [True])
    res = m.ask(QUESTION)
    assert res.status == UNAVAILABLE and res.claims == []


def test_prose_answer_with_ids_becomes_claims():
    raw = "الروح من أمر الله [test-a:1]. وهذه جملة بلا مصدر. وجملة ثالثة [b:1، q:2:3]."
    claims = parse_draft(raw)
    assert [(c.text, c.passage_ids) for c in claims] == [
        ("الروح من أمر الله.", ("test-a:1",)), ("وجملة ثالثة.", ("b:1", "q:2:3"))]


def test_attribution_and_exact_meaning_rules_are_in_the_instructions():
    assert "يُنسب إلى قائله" in generate.SYSTEM_PROMPT and "لا تقلب نفيًا إلى إثبات" in generate.SYSTEM_PROMPT


@pytest.mark.real_check
def test_answer_broken_by_the_check_is_rewritten_once_from_the_feedback():
    first = {"abstain": False, "claims": [
        {"text": "النخلة لا تحتاج إلى الماء.", "passage_ids": ["test-a:1"]},
        {"text": "وهي تحتاجه أكثر في الصيف.", "passage_ids": ["test-a:1"]}]}
    second = {"abstain": False, "claims": [
        {"text": "تحتاج النخلة إلى ماء كثير.", "passage_ids": ["test-a:1"]},
        {"text": "ويزداد ذلك في الصيف.", "passage_ids": ["test-a:1"]}]}
    prompts, verdicts = [], iter([[False, True], [True, True]])

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "supported" in keys:
            return json.dumps({"supported": next(verdicts)})
        if "queries" in keys:
            return '{"queries": []}'
        prompts.append(user)
        return json.dumps(first if len(prompts) == 1 else second, ensure_ascii=False)

    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask(QUESTION)
    assert [c["text"] for c in res.claims] == ["تحتاج النخلة إلى ماء كثير.", "ويزداد ذلك في الصيف."]
    assert len(prompts) == 2 and "النخلة لا تحتاج إلى الماء." in prompts[1].split("مراجعة لجواب سابق")[1]


def test_second_reading_accepts_plain_explanations_and_rejects_additions():
    assert "إن شككت" not in generate.CHECK_PROMPT
    assert "شرحًا له بلغة سهلة" in generate.CHECK_PROMPT and "أضافت معلومة شرعية ليست في المقاطع" in generate.CHECK_PROMPT


def test_never_fabricate_or_distort_otherwise_free_to_explain():
    assert "تأليف آية أو حديث أو قول أو مصدر أو نسبة" in generate.SYSTEM_PROMPT
    assert "تحريف نص آية أو حديث أو معناه" in generate.SYSTEM_PROMPT and "وما سوى ذلك فأنت حر فيه" in generate.SYSTEM_PROMPT
    assert "الشرح اللغوي العام الذي لا يضيف معلومة شرعية" in generate.CHECK_PROMPT
    assert "حديثًا، أو قولًا لعالم" in generate.CHECK_PROMPT  # added religious content is still rejected


def test_quotes_are_compared_without_diacritics_and_english_quotes_are_not_quotations():
    from muhawir.verify import Claim, quotes_in, verify
    assert quotes_in('Tawhid means "oneness"') == []
    corpus = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")
    pid = corpus.passages[0].id
    word = corpus.passages[0].text.split()[0]
    kept, _ = verify([Claim(f"وردت كلمة «{word}» في المقطع.", (pid,))], corpus, {pid})
    assert len(kept) == 1


def test_per_request_state_is_not_shared_between_threads():
    import threading
    gen = ModelGenerator([("m", lambda s, u, schema=None: "{}")])
    gen.last_as_list = True
    seen = []
    t = threading.Thread(target=lambda: seen.append(gen.last_as_list))
    t.start(); t.join()
    assert seen == [False] and gen.last_as_list is True


def test_sections_labels_and_follow_up_are_read_and_follow_up_must_be_a_short_question():
    from muhawir.generate import follow_up
    raw = json.dumps({"plan": "", "abstain": False, "as_list": False, "views": [],
                      "claims": [{"section": "زكاة الفطر", "label": "وقتها", "text": "آخر رمضان.", "passage_ids": ["a:1"]}],
                      "follow_up": "هل تريد أن تعرف لمن تُعطى؟"}, ensure_ascii=False)
    c = parse_draft(raw)[0]
    assert (c.section, c.label, c.text) == ("زكاة الفطر", "وقتها", "آخر رمضان.")
    assert follow_up(raw) == "هل تريد أن تعرف لمن تُعطى؟"
    assert follow_up(raw.replace("تُعطى؟", "تُعطى.")) == ""  # not a question: not shown
    assert follow_up(raw.replace("هل تريد", "[q:1] هل تريد")) == ""


def test_a_verse_in_brackets_is_not_copying_but_must_match_its_passage():
    from muhawir.pipeline import _COPIED
    from muhawir.verify import Claim, verify
    assert not _COPIED.search("يُخرج حقها يوم الحصاد: ﴿وآتوا حقه يوم حصاده يوم كذا وكذا﴾")
    pid = CORPUS.passages[0].id
    kept, rejected = verify([Claim("قال تعالى: ﴿كلام ليس في المقطع أبدًا﴾", (pid,))], CORPUS, {pid})
    assert kept == [] and rejected  # an invented verse is rejected


def test_an_answer_of_explanation_alone_is_not_shown():
    raw = {"plan": "", "abstain": False, "as_list": False, "views": [], "follow_up": "",
           "claims": [{"section": "", "label": "", "text": "شرح عام من فهم مُحاور.", "passage_ids": []}]}
    m = Muhawir(CORPUS, ModelGenerator([("m", lambda s, u, schema=None:
                                         '{"queries": []}' if "queries" in json.dumps(schema or {}) else json.dumps(raw, ensure_ascii=False))]))
    assert m.ask(QUESTION).status != ANSWERED  # every answer must rest on the sources


def test_own_explanation_is_shown_next_to_sourced_sentences_without_a_number():
    raw = {"plan": "", "abstain": False, "as_list": False, "views": [], "follow_up": "",
           "claims": [{"section": "", "label": "", "text": "تحتاج النخلة إلى ماء كثير.", "passage_ids": ["test-a:1"]},
                      {"section": "", "label": "", "text": "وهذا لأن الصيف حار.", "passage_ids": []}]}
    m = Muhawir(CORPUS, ModelGenerator([("m", lambda s, u, schema=None:
                                         '{"queries": []}' if "queries" in json.dumps(schema or {}) else json.dumps(raw, ensure_ascii=False))]))
    res = m.ask(QUESTION)
    assert res.status == ANSWERED and [c["passage_ids"] for c in res.claims] == [["test-a:1"], []]
