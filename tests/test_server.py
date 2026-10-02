from fastapi.testclient import TestClient

from muhawir.server import app

client = TestClient(app)


def test_health():
    body = client.get("/api/health").json()
    assert body["ok"] and body["generator"] == "extractive" and body["synthetic"] is True


def test_ask_answers_with_sources():
    body = client.post("/api/ask", json={"question": "ماذا تحتاج النخلة في الصيف؟"}).json()
    assert body["status"] == "answered" and body["sources"]


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200 and "مُحاور" in r.text
