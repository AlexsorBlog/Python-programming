"""Postings as flat NumPy arrays (CSR layout) and vectorized BM25 / TF-IDF.

    term_id = vocab["kyiv"]
    rows    = slice(offsets[term_id], offsets[term_id + 1])
    doc_ids[rows], tfs[rows]  ->  the postings of "kyiv", sorted by doc_id

On disk it is a folder of .npy files, so np.load(mmap_mode="r") opens it
without reading millions of Python objects back, the way pickle has to.
"""

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from findex.index import DocId, DocMeta, Index

type IntArray = npt.NDArray[np.int32]
type FloatArray = npt.NDArray[np.float64]
type ScorerKind = Literal["bm25", "tfidf"]

K1 = 1.5
B = 0.75


class DocMetaView(Mapping[DocId, DocMeta]):
    """doc_id -> DocMeta, built on access instead of 240k objects up front."""

    def __init__(
        self,
        titles: Sequence[str],
        paths: Sequence[str],
        path_idx: IntArray,
        offsets: npt.NDArray[np.int64],
    ) -> None:
        self._titles = titles
        self._paths = paths
        self._path_idx = path_idx
        self._offsets = offsets

    def __getitem__(self, doc_id: DocId) -> DocMeta:
        if not 0 <= doc_id < len(self._titles):
            raise KeyError(doc_id)
        return DocMeta(
            doc_id,
            self._paths[int(self._path_idx[doc_id])],
            self._titles[doc_id],
            int(self._offsets[doc_id]),
        )

    def __iter__(self) -> Iterator[DocId]:
        return iter(range(len(self._titles)))

    def __len__(self) -> int:
        return len(self._titles)


# eq=False keeps the identity hash, which lru_cache in search.py needs
@dataclass(eq=False)
class NumpyIndex:
    vocab: dict[str, int]
    offsets: npt.NDArray[np.int64]
    doc_ids_flat: IntArray
    tfs: IntArray
    doc_lengths_arr: IntArray
    doc_meta: DocMetaView
    has_positions: bool = False
    repr_kind: str = "numpy"

    # --- the same small API as Index, so the lab 3 parser works on it -------

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, term: object) -> bool:
        return term in self.vocab

    def __repr__(self) -> str:
        return (
            f"NumpyIndex(terms={len(self):_}, docs={self.num_docs:_}, "
            f"postings={len(self.doc_ids_flat):_})"
        )

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths_arr)

    @property
    def avg_doc_length(self) -> float:
        # sum as a Python int, then divide: the exact same float as Index gives
        if not self.num_docs:
            return 0.0
        return int(self.doc_lengths_arr.sum(dtype=np.int64)) / self.num_docs

    def doc_length(self, doc_id: DocId) -> int:
        return int(self.doc_lengths_arr[doc_id])

    def rows(self, term: str) -> tuple[IntArray, IntArray]:
        term_id = self.vocab.get(term)
        if term_id is None:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty
        start, end = int(self.offsets[term_id]), int(self.offsets[term_id + 1])
        return self.doc_ids_flat[start:end], self.tfs[start:end]

    def df(self, term: str) -> int:
        return len(self.rows(term)[0])

    def doc_ids(self, term: str) -> list[DocId]:
        ids: list[int] = self.rows(term)[0].tolist()
        return ids

    def positions(self, term: str, doc_id: DocId) -> tuple[int, ...]:
        return ()

    def close(self) -> None:
        pass


def from_index(index: Index) -> NumpyIndex:
    """Convert a lab 2-3 Index (any representation) to flat arrays."""
    terms = list(index.postings)
    lengths = np.fromiter(
        (index.df(t) for t in terms), dtype=np.int64, count=len(terms)
    )
    offsets = np.zeros(len(terms) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    doc_ids = np.empty(total, dtype=np.int32)
    tfs = np.empty(total, dtype=np.int32)

    for term_id, term in enumerate(terms):
        start, end = int(offsets[term_id]), int(offsets[term_id + 1])
        entry = index.postings[term]
        if isinstance(entry, tuple):
            # array('I') pairs: copy the raw buffer, no Python loop at all
            doc_ids[start:end] = np.frombuffer(entry[0], dtype=np.uint32)
            tfs[start:end] = np.frombuffer(entry[1], dtype=np.uint32)
        else:
            doc_ids[start:end] = [p.doc_id for p in entry]
            tfs[start:end] = [p.tf for p in entry]

    n_docs = index.num_docs
    doc_lengths = np.zeros(n_docs, dtype=np.int32)
    titles: list[str] = [""] * n_docs
    path_names: dict[str, int] = {}
    path_idx = np.zeros(n_docs, dtype=np.int32)
    doc_offsets = np.zeros(n_docs, dtype=np.int64)
    for doc_id, meta in index.doc_meta.items():
        doc_lengths[doc_id] = index.doc_length(doc_id)
        titles[doc_id] = meta.title
        path_idx[doc_id] = path_names.setdefault(meta.path, len(path_names))
        doc_offsets[doc_id] = meta.offset

    return NumpyIndex(
        vocab={t: i for i, t in enumerate(terms)},
        offsets=offsets,
        doc_ids_flat=doc_ids,
        tfs=tfs,
        doc_lengths_arr=doc_lengths,
        doc_meta=DocMetaView(titles, list(path_names), path_idx, doc_offsets),
    )


# --- saving and loading -----------------------------------------------------

_ARRAYS = ("offsets", "doc_ids", "tfs", "doc_lengths", "path_idx", "doc_offsets")


def save_numpy(index: NumpyIndex, folder: Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = index.doc_meta
    arrays = {
        "offsets": index.offsets,
        "doc_ids": index.doc_ids_flat,
        "tfs": index.tfs,
        "doc_lengths": index.doc_lengths_arr,
        "path_idx": meta._path_idx,  # pyright: ignore[reportPrivateUsage]  # own class
        "doc_offsets": meta._offsets,  # pyright: ignore[reportPrivateUsage]  # own class
    }
    for name, array in arrays.items():
        np.save(folder / f"{name}.npy", array)
    terms = sorted(index.vocab, key=index.vocab.__getitem__)
    (folder / "vocab.json").write_text(json.dumps(terms, ensure_ascii=False), "utf-8")
    (folder / "docs.json").write_text(
        json.dumps(
            {
                "titles": list(meta._titles),  # pyright: ignore[reportPrivateUsage]
                "paths": list(meta._paths),  # pyright: ignore[reportPrivateUsage]
            },
            ensure_ascii=False,
        ),
        "utf-8",
    )


def load_numpy(folder: Path, mmap: bool = True) -> NumpyIndex:
    folder = Path(folder)
    mode: Literal["r"] | None = "r" if mmap else None
    a = {name: np.load(folder / f"{name}.npy", mmap_mode=mode) for name in _ARRAYS}
    terms: list[str] = json.loads((folder / "vocab.json").read_text("utf-8"))
    docs = json.loads((folder / "docs.json").read_text("utf-8"))
    return NumpyIndex(
        vocab={t: i for i, t in enumerate(terms)},
        offsets=a["offsets"],
        doc_ids_flat=a["doc_ids"],
        tfs=a["tfs"],
        doc_lengths_arr=a["doc_lengths"],
        doc_meta=DocMetaView(
            docs["titles"], docs["paths"], a["path_idx"], a["doc_offsets"]
        ),
    )


def is_numpy_index(path: Path) -> bool:
    return Path(path).is_dir() and (Path(path) / "offsets.npy").exists()


# --- vectorized scoring -----------------------------------------------------


def score_numpy(
    index: NumpyIndex,
    terms: Sequence[str],
    scorer: ScorerKind = "bm25",
    k1: float = K1,
    b: float = B,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Scores for every document plus a mask of who got any contribution.

    The expressions are written in the same order as BM25.score / TfIdf.score
    in scoring.py, so the floats come out bit for bit the same.
    """
    n = index.num_docs
    scores = np.zeros(n, dtype=np.float64)
    touched = np.zeros(n, dtype=np.bool_)
    avgdl = index.avg_doc_length

    for term in terms:
        ids, tf_int = index.rows(term)
        df = len(ids)
        if not df:
            continue
        tf = tf_int.astype(np.float64)
        if scorer == "bm25":
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            dl = index.doc_lengths_arr[ids].astype(np.float64)
            norm = 1 - b + b * dl / avgdl
            weight = idf * tf * (k1 + 1) / (tf + k1 * norm)
        else:
            idf = math.log(n / df)
            weight = (1 + np.log(tf)) * idf
        # doc_ids are unique inside one term's postings, so plain += is safe
        scores[ids] += weight
        touched[ids] = True
    return scores, touched


def top_k_numpy(
    scores: FloatArray, candidates: IntArray, k: int
) -> list[tuple[DocId, float]]:
    """Best k by score; ties broken by smaller doc_id, same rule as top_k()."""
    if not len(candidates):
        return []
    cand_scores = scores[candidates]
    k = min(k, len(candidates))
    picked = np.argpartition(-cand_scores, k - 1)[:k]
    # argpartition is O(n), but it may cut a group of equal scores at the edge;
    # take every candidate that ties with the k-th score, then sort exactly
    threshold = cand_scores[picked].min()
    pool = np.nonzero(cand_scores >= threshold)[0]
    order = np.lexsort((candidates[pool], -cand_scores[pool]))[:k]
    chosen = pool[order]
    return [(int(candidates[i]), float(cand_scores[i])) for i in chosen]
