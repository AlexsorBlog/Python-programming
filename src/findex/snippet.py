import json
import logging
import re
from pathlib import Path
from typing import Any

from findex.index import DocMeta
from findex.tokenize import normalize

log = logging.getLogger(__name__)
WIDTH = 80


def document_text(meta: DocMeta) -> str:
    path = Path(meta.path)
    if path.suffix == ".jsonl":
        with path.open("rb") as f:
            f.seek(meta.offset)
            record: dict[str, Any] = json.loads(
                f.readline().decode("utf-8", errors="replace")
            )
        title = str(record.get("title", ""))
        text = str(record.get("text", ""))
        return f"{title}\n{text}" if title else text
    return path.read_text(encoding="utf-8", errors="replace")


def _word(term: str) -> re.Pattern[str]:
    # a whole token only: "disaster" must not light up inside "disasters"
    return re.compile(rf"(?<![^\W_]){re.escape(term)}(?![^\W_])")


def make_snippet(
    text: str,
    terms: list[str],
    width: int = WIDTH,
    marker: tuple[str, str] = ("[", "]"),
) -> str:
    low = normalize(text)
    patterns = {term: _word(term) for term in dict.fromkeys(terms)}
    best: int | None = None
    found: list[str] = []
    for term, pattern in patterns.items():
        match = pattern.search(low)
        if match:
            found.append(term)
            if best is None or match.start() < best:
                best = match.start()
    if best is None:
        return " ".join(text.split())[: width * 2] + "..."

    start = max(0, best - width)
    end = min(len(text), best + width)
    window = " ".join(text[start:end].split())

    low_window = normalize(window)
    spans: list[tuple[int, int]] = []
    for term in sorted(found, key=len, reverse=True):
        for match in patterns[term].finditer(low_window):
            a, b = match.span()
            if all(b <= x or a >= y for x, y in spans):
                spans.append((a, b))

    out: list[str] = []
    last = 0
    for a, b in sorted(spans):
        out.append(window[last:a])
        out.append(marker[0] + window[a:b] + marker[1])
        last = b
    out.append(window[last:])

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{''.join(out)}{suffix}"
