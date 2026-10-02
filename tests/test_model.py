"""Model-mode tests with fake model calls. No network, no API key."""
import json
from pathlib import Path

import pytest

from muhawir import generate
from muhawir.corpus import load_corpus
from muhawir.generate import ModelGenerator, build_user_prompt, get_generator, parse_draft
from muhawir.pipeline import ABSTAINED, ANSWERED, REFERRED, Muhawir

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


def test_all_models_failing_means_abstain():
    assert engine(RuntimeError("a"), RuntimeError("b")).ask(QUESTION).status == ABSTAINED


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
