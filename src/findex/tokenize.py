import re
import unicodedata
from collections.abc import Iterator

_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})

# letters or digits, no underscore
_WORD = r"[^\W_]+"
TOKEN_RE = re.compile(rf"{_WORD}(?:['-]{_WORD})*")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().translate(_APOSTROPHES)


def tokenize(text: str) -> Iterator[str]:
    for match in TOKEN_RE.finditer(normalize(text)):
        yield match.group()
