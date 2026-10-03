"""Ask a running Muhawir a set of dialogue questions and print what it replied.

The questions cover objections often raised in debates about Islam, written in
our own words. Their choice of topics follows the chapters of «حوار مع صديقي
الملحد» by Mustafa Mahmoud (Dar al-Awda), which also inspired Muhawir's calm
way of answering objections; no text from the book is used. Some questions are
deliberately worded with mockery, to check that Muhawir answers the question
calmly and never judges the person.

Usage (with the server running):  python scripts/try_questions.py [--url http://localhost:8000] [--style youth]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

QUESTIONS = [
    ("objection", "إذا كان لكل شيء خالق، فمن خلق الله؟"),
    ("objection", "إذا كان الله قد قدّر أفعالي، فلماذا يحاسبني عليها؟"),
    ("objection", "لماذا خلق الله الشر والمرض والزلازل إذا كان رحيمًا؟"),
    ("objection", "ما ذنب من لم يسمع بالإسلام ولم يصله القرآن؟"),
    ("objection", "لماذا يكون عذاب النار دائمًا على ذنوب في عمر قصير؟"),
    ("objection", "أليس الدين مخدّرًا يُسكت به الفقراء عن حقوقهم؟"),
    ("objection", "لماذا ترث المرأة نصف ما يرثه الرجل؟"),
    ("question", "ما هي الروح؟"),
    ("objection", "أليس الضمير مجرد عادة يصنعها المجتمع؟"),
    ("objection", "أليس الطواف حول الكعبة وتقبيل الحجر الأسود تعظيمًا للحجارة؟"),
    ("objection", "ما الدليل على أن القرآن ليس من تأليف محمد؟"),
    ("question", "ما معنى الحروف المقطعة مثل الم وكهيعص في أوائل السور؟"),
    ("question", "ماذا يقول القرآن عمّن تراوده الشكوك؟"),
    ("question", "ما معنى لا إله إلا الله؟"),
    ("question", "ما هي الشفاعة؟"),
    ("objection", "لماذا لا يُرينا الله معجزة اليوم حتى نؤمن؟"),
    ("objection", "لماذا تتركون متع الدنيا من أجل آخرة لم يرها أحد؟"),
    ("mockery", "دينكم مليء بالخرافات، قولوا لي إذن من خلق ربكم؟"),
    ("mockery", "يا أصحاب الخرافات، لماذا خلق إلهكم الشر إذا كان رحيمًا كما تزعمون؟"),
    ("mockery-only", "كلامكم سخيف ولا يصدقه عاقل"),
    ("mockery-only", "هههه أنتم تصدقون أي شيء"),
    ("follow-up", "وما الدليل على ذلك من القرآن؟"),
    ("english", "Why would a merciful God allow suffering?"),
]


def ask(url: str, question: str, style: str, lang: str, history: list[dict]) -> dict:
    body = json.dumps({"question": question, "style": style, "lang": lang, "history": history}).encode()
    req = urllib.request.Request(f"{url}/api/ask", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--style", default="youth")
    args = parser.parse_args()
    history: list[dict] = []
    for kind, question in QUESTIONS:
        lang = "en" if kind == "english" else "ar"
        started = time.time()
        res = ask(args.url, question, args.style, lang, history if kind == "follow-up" else [])
        print(f"\n=== [{kind}] {question}")
        print(f"status: {res['status']}   ({time.time() - started:.0f}s)")
        if res.get("understood"):
            print(f"understood as: {res['understood']}")
        if res.get("message"):
            print(res["message"])
        for c in res.get("claims", []):
            print(f"- {c['text']}  {c['passage_ids']}")
        for s in res.get("sources", []):
            grade = f"  [{s['grade']}]" if s.get("grade") else ""
            print(f"  source: {s['source_name']} · {s['location']}{grade}")
        history = [{"role": "user", "text": res.get("understood") or question},
                   {"role": "assistant", "text": " ".join(c["text"] for c in res.get("claims", []))}]


if __name__ == "__main__":
    main()
