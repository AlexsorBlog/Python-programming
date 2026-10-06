import json
import logging
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


def make_snippet(
    text: str,
    terms: list[str],
    width: int = WIDTH,
    marker: tuple[str, str] = ("[", "]"),
) -> str:
    low = normalize(text)
    best: int | None = None
    found: list[str] = []
    for term in terms:
        at = low.find(term)
        if at != -1:
            found.append(term)
            if best is None or at < best:
                best = at
    if best is None:
        return " ".join(text.split())[: width * 2] + "..."

    start = max(0, best - width)
    end = min(len(text), best + width)
    window = " ".join(text[start:end].split())

    low_window = normalize(window)
    for term in sorted(set(found), key=len, reverse=True):
        at = 0
        while (at := low_window.find(term, at)) != -1:
            window = (
                window[:at]
                + marker[0]
                + window[at : at + len(term)]
                + marker[1]
                + window[at + len(term) :]
            )
            low_window = normalize(window)
            at += len(term) + len(marker[0]) + len(marker[1])
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{window}{suffix}"
