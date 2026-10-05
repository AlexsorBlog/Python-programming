import json
import pickle
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from findex.decorators import timed
from findex.index import DocMeta, Index, PlainPosting, Posting


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


@timed
def load(path: Path) -> Index:
    path = Path(path)
    if path.suffix == ".json":
        return _load_json(path)
    with path.open("rb") as f:
        return pickle.load(f)


@contextmanager
def open_index(path: Path) -> Iterator[Index]:
    """Everything after yield runs even if the with-block raises."""
    index = load(path)
    try:
        yield index
    finally:
        index.close()


def _save_json(index: Index, path: Path) -> None:
    if index.repr_kind == "array":
        postings = {t: [list(e[0]), list(e[1])] for t, e in index.postings.items()}
    else:
        postings = {
            t: [[p.doc_id, p.tf, list(p.positions)] for p in e]
            for t, e in index.postings.items()
        }
    payload = {
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
        payload = json.load(f)

    repr_kind = payload["repr_kind"]
    if repr_kind == "array":
        postings = {
            t: (array("I", ids), array("I", tfs))
            for t, (ids, tfs) in payload["postings"].items()
        }
    else:
        cls = Posting if repr_kind == "slots" else PlainPosting
        postings = {
            t: [cls(doc_id, tf, tuple(poss)) for doc_id, tf, poss in e]
            for t, e in payload["postings"].items()
        }
    return Index(
        postings=postings,
        doc_lengths={int(k): v for k, v in payload["doc_lengths"].items()},
        doc_meta={int(k): DocMeta(*v) for k, v in payload["doc_meta"].items()},
        repr_kind=repr_kind,
        has_positions=payload["has_positions"],
    )
