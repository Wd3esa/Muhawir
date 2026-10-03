"""Find the opening passage of the matching chapter and section of a structured book.

Keyword search ranks passages by shared words, so the passage that defines a topic or lists
its kinds can lose to passages that merely repeat a common word (e.g. «أنواع» matching a
chapter on division of property). Ibn Rushd's «بداية المجتهد» is organised in chapters
(«كتاب الزكاة») and sections («الجملة الثانية في معرفة ما تجب فيه الزكاة من الأموال»), and
the first passage of each usually holds the overview, the definition or the list. When the
question or a search phrase matches a chapter or section name, its first passage is offered
to the model first.
"""
from __future__ import annotations

from .normalize import tokenize

SOURCE_ID = "bidayat-al-mujtahid"
# words found in every heading, which say nothing about its topic
_HEADING_WORDS = frozenset(tokenize(
    "كتاب الجملة الباب الفصل القسم المسألة معرفة الأول الأولى الثاني الثانية الثالث الثالثة الرابع "
    "الرابعة الخامس الخامسة السادس السادسة السابع السابعة الثامن الثامنة فيه فيها وهو وهي هذه"))
MAX_SECTIONS = 2
MAX_CHAPTERS = 2
MIN_SHARED = 2


def _topic(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _HEADING_WORDS}


class SectionIndex:
    """First passage of each chapter and section, from the passages' heading keywords."""

    def __init__(self, corpus) -> None:
        self.chapters: dict[str, str] = {}   # chapter name -> first passage id
        self.sections: list[tuple[str, str, set[str]]] = []  # (chapter, first passage id, topic words)
        seen: set[str] = set()
        for pid, keywords in self._rows(corpus):
            parts = [p.strip() for p in (keywords or "").split("؛") if p.strip()]
            if not parts:
                continue
            chapter = parts[0] if parts[0].startswith("كتاب") else ""
            if chapter and chapter not in self.chapters:
                self.chapters[chapter] = pid
            key = "؛".join(parts)
            if key in seen:
                continue
            seen.add(key)
            section = parts[-1]
            if section != chapter:
                self.sections.append((chapter, pid, _topic(section)))

    @staticmethod
    def _rows(corpus):
        con = getattr(corpus, "con", None)
        if con is not None:  # SQLite corpus: in book order
            return con.execute("SELECT id, keywords FROM passages WHERE source_id = ? ORDER BY rid",
                               (SOURCE_ID,)).fetchall()
        return [(p.id, p.keywords) for p in getattr(corpus, "passages", []) if p.source_id == SOURCE_ID]

    def match(self, texts: list[str]) -> list[str]:
        """Passage ids to offer first: the openings of the matching chapters (the general one first,
        e.g. «كتاب الزكاة» before «كتاب زكاة الفطر»), then the best matching sections; when no section
        of the general chapter matches, its second section's opening, which usually follows the overview."""
        words = [_topic(t) for t in texts if t]
        if not words or not self.chapters and not self.sections:
            return []
        every = set().union(*words)
        chapters = sorted((c for c in self.chapters if _topic(c) and _topic(c) <= every),
                          key=lambda c: len(_topic(c)))[:MAX_CHAPTERS]
        out = [self.chapters[c] for c in chapters]
        scored = []
        for ch, pid, topic in self.sections:
            if chapters and ch not in chapters or not topic:
                continue
            shared = max(len(topic & w) for w in words)
            if shared >= MIN_SHARED and shared / len(topic) >= 0.5:
                scored.append((shared, shared / len(topic), pid, ch))
        scored.sort(reverse=True)
        for _, _, pid, _ in scored[:MAX_SECTIONS]:
            if pid not in out:
                out.append(pid)
        if chapters and not any(ch == chapters[0] for *_, ch in scored[:MAX_SECTIONS]):
            following = [pid for ch, pid, _ in self.sections if ch == chapters[0] and pid not in out]
            out += following[:1]
        return out
