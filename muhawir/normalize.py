"""Arabic text normalization and tokenization for retrieval.

Normalization is used for matching only. Quotations shown to the user are
always taken verbatim from the stored passage, never from normalized text.
"""
import re

# Harakat, tanwin, shadda, sukun, superscript alef and Quranic annotation marks.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")

_CHAR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
})

# Function words that carry no topic. Kept short on purpose.
STOPWORDS = frozenset("""
في من على عن الى الي ما ماذا هل هو هي هم انا انت نحن هذا هذه ذلك تلك التي الذي الذين
او ام ثم لا لم لن قد كل بعض مع عند بين كيف لماذا متى اين كم اي ان كان يكون
the a an of to in on is are was were what why how who which do does did i you it and or
""".split())

_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def normalize(text: str) -> str:
    """Lowercase, strip diacritics/tatweel/punctuation, unify letter variants."""
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_CHAR_MAP).lower()
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _strip_prefix(token: str) -> str:
    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            return token[len(prefix):]
    return token


def tokenize(text: str) -> list[str]:
    """Normalized content tokens with the definite article removed."""
    tokens = []
    for raw in normalize(text).split():
        if raw in STOPWORDS:
            continue
        token = _strip_prefix(raw)
        if len(token) >= 2 and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def collapse_spaces(text: str) -> str:
    """Whitespace-only normalization, used for verbatim quote checks."""
    return _SPACES.sub(" ", text).strip()
