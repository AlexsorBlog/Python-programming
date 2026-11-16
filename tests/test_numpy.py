import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.index import Index, build_index
from findex.numpy_index import (
    from_index,
    is_numpy_index,
    load_numpy,
    save_numpy,
    score_numpy,
    top_k_numpy,
)
from findex.scoring import BM25, TfIdf
from findex.search import ranked_search, ranked_search_numpy

QUERIES = [
    "city",
    "ukraine",
    "city ukraine",
    "kyiv OR paris",
    "city NOT france",
    "the OR is OR a",
    "nothinghere",
]


def lab3(index: Index, q: str, scorer: str, k: int = 10) -> list[tuple[int, float]]:
    s = BM25() if scorer == "bm25" else TfIdf()
    return [(r.doc_id, r.score) for r in ranked_search(index, q, s, k=k)]


def lab8(index, q: str, scorer: str, k: int = 10) -> list[tuple[int, float]]:
    return [(r.doc_id, r.score) for r in ranked_search_numpy(index, q, scorer, k=k)]


def test_postings_survive_conversion(index: Index):
    nidx = from_index(index)
    for term in index:
        assert nidx.doc_ids(term) == index.doc_ids(term)
        ids, tfs = nidx.rows(term)
        assert ids.dtype == np.int32 and tfs.dtype == np.int32
        assert tfs.tolist() == [index.term_frequencies(term)[d] for d in ids.tolist()]
    assert nidx.num_docs == index.num_docs
    assert nidx.avg_doc_length == index.avg_doc_length


def test_array_repr_converts_the_same(corpus: Path):
    slots = from_index(build_index(corpus))
    arrays = from_index(build_index(corpus, repr_kind="array"))
    for term in slots.vocab:
        assert slots.doc_ids(term) == arrays.doc_ids(term)


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("query", QUERIES)
def test_order_and_scores_match_lab3(index: Index, query: str, scorer: str):
    # exact equality on floats on purpose: same formula, same order of operations
    assert lab8(from_index(index), query, scorer) == lab3(index, query, scorer)


def test_save_load_round_trip(index: Index, tmp_path: Path):
    nidx = from_index(index)
    folder = tmp_path / "np_index"
    save_numpy(nidx, folder)
    assert is_numpy_index(folder)
    back = load_numpy(folder, mmap=True)
    assert isinstance(back.doc_ids_flat, np.memmap)
    assert back.doc_meta[1] == index.doc_meta[1]
    assert lab8(back, "city ukraine", "bm25") == lab3(index, "city ukraine", "bm25")


def test_top_k_breaks_ties_by_doc_id():
    scores = np.array([1.0, 5.0, 5.0, 5.0, 2.0])
    candidates = np.arange(5, dtype=np.int32)
    # four equal 5.0 would be cut arbitrarily by argpartition alone
    assert top_k_numpy(scores, candidates, 2) == [(1, 5.0), (2, 5.0)]
    assert top_k_numpy(scores, candidates, 10)[-1] == (0, 1.0)


def test_top_k_empty():
    assert top_k_numpy(np.zeros(3), np.empty(0, dtype=np.int32), 5) == []


def test_unknown_term_scores_nothing(index: Index):
    scores, touched = score_numpy(from_index(index), ["nothinghere"])
    assert not touched.any()
    assert not scores.any()


words = st.sampled_from(["kyiv", "lviv", "city", "river", "the", "a", "war", "sea"])
docs = st.lists(
    st.lists(words, min_size=1, max_size=12).map(" ".join), min_size=1, max_size=15
)


@given(texts=docs, q=st.lists(words, min_size=1, max_size=3).map(" OR ".join))
@settings(max_examples=40, deadline=None)
def test_property_numpy_matches_lab3(texts: list[str], q: str, tmp_path_factory):
    path = Path(tmp_path_factory.mktemp("c")) / "d.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": i, "text": t}) for i, t in enumerate(texts)),
        encoding="utf-8",
    )
    index = build_index(path)
    nidx = from_index(index)
    for scorer in ("bm25", "tfidf"):
        assert lab8(nidx, q, scorer, k=5) == lab3(index, q, scorer, k=5)
