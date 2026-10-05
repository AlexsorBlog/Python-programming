import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str
    offset: int = 0


def _iter_jsonl(path: Path) -> Iterator[Document]:
    with path.open("rb") as f:
        lineno = 0
        while True:
            # readline instead of "for line in f", so tell() gives the real offset
            offset = f.tell()
            raw = f.readline()
            if not raw:
                return
            lineno += 1
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                log.warning("пропускаю %s:%d (%s)", path, lineno, e)
                continue
            doc_id = str(record.get("id", f"{path.name}:{lineno}"))
            title = record.get("title", "")
            text = record.get("text", "")
            yield Document(doc_id, path, f"{title}\n{text}" if title else text, offset)


def _iter_txt(path: Path) -> Iterator[Document]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        log.warning("погане кодування в %s (%s)", path, e)
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("не можу прочитати %s (%s)", path, e)
        return
    yield Document(str(path), path, text)


def iter_documents(root: Path) -> Iterator[Document]:
    root = Path(root)
    paths = [root] if root.is_file() else root.rglob("*")
    for path in paths:
        if path.suffix == ".jsonl":
            yield from _iter_jsonl(path)
        elif path.suffix == ".txt":
            yield from _iter_txt(path)
