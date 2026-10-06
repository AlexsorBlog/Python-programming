import json
import logging
import pickle
from array import array
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from findex.decorators import timed
from findex.index import (
    ArrayEntry,
    DocId,
    DocMeta,
    Index,
    PlainPosting,
    Posting,
    PostingLike,
    PostingsEntry,
    ReprKind,
)

log = logging.getLogger(__name__)


@timed
def save(index: Index, path: Path) -> None:
    path = Path(path)
    if path.suffix == ".json":
        _save_json(index, path)
    else:
        # pickle is fast but unsafe: pickle.load() on a file from someone else
        # runs any code the file wants, so I only load my own index files
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("індекс збережено в %s", path)


@timed
def load(path: Path) -> Index:
    path = Path(path)
    if path.suffix == ".json":
        return _load_json(path)
    with path.open("rb") as f:
        # pickle.load returns Any by design; this file is one I wrote myself
        return cast(Index, pickle.load(f))


@contextmanager
def open_index(path: Path) -> Generator[Index]:
    """Everything after yield runs even if the with-block raises."""
    index = load(path)
    try:
        yield index
    finally:
        index.close()


def _save_json(index: Index, path: Path) -> None:
    postings: dict[str, Any] = {}
    for term, entry in index.postings.items():
        if isinstance(entry, tuple):
            postings[term] = [list(entry[0]), list(entry[1])]
        else:
            postings[term] = [[p.doc_id, p.tf, list(p.positions)] for p in entry]

    payload: dict[str, Any] = {
        "repr_kind": index.repr_kind,
        "has_positions": index.has_positions,
        "postings": postings,
        "doc_lengths": index.doc_lengths,
        "doc_meta": {
            str(k): [v.doc_id, v.path, v.title, v.offset]
            for k, v in index.doc_meta.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _load_json(path: Path) -> Index:
    with path.open(encoding="utf-8") as f:
        # json.load gives Any: the shape is checked by the round-trip tests
        payload = cast(dict[str, Any], json.load(f))

    repr_kind = cast(ReprKind, payload["repr_kind"])
    raw_postings = cast(dict[str, Any], payload["postings"])
    postings: dict[str, PostingsEntry] = {}

    if repr_kind == "array":
        for term, (ids, tfs) in raw_postings.items():
            entry: ArrayEntry = (array("I", ids), array("I", tfs))
            postings[term] = entry
    else:
        cls = Posting if repr_kind == "slots" else PlainPosting
        for term, records in raw_postings.items():
            objects: list[PostingLike] = [
                cls(doc_id, tf, tuple(poss)) for doc_id, tf, poss in records
            ]
            postings[term] = objects

    doc_lengths: dict[DocId, int] = {
        int(k): int(v) for k, v in cast(dict[str, Any], payload["doc_lengths"]).items()
    }
    doc_meta: dict[DocId, DocMeta] = {
        int(k): DocMeta(v[0], v[1], v[2], v[3])
        for k, v in cast(dict[str, Any], payload["doc_meta"]).items()
    }
    return Index(
        postings,
        doc_lengths,
        doc_meta,
        repr_kind,
        bool(payload["has_positions"]),
    )
