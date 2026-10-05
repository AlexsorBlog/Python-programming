import json
from pathlib import Path

import pytest

from findex.index import DocMeta, Posting, build_index
from findex.search import merge_and, merge_not, merge_or, parse_query, search
from findex.store import load, save

DOCS = [
    {"id": 0, "title": "Kyiv", "text": "Kyiv is the capital of Ukraine"},
    {"id": 1, "title": "Lviv", "text": "Lviv is a city in Ukraine"},
    {"id": 2, "title": "Paris", "text": "Paris is a city in France city"},
]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    f = tmp_path / "docs.jsonl"
    f.write_text("\n".join(json.dumps(d) for d in DOCS), encoding="utf-8")
    return f


def test_postings_sorted_and_tf(corpus: Path):
    index = build_index(corpus)
    assert index.doc_ids("ukraine") == [0, 1]
    assert index.doc_ids("city") == [1, 2]
    assert index.term_frequencies("city") == {1: 1, 2: 2}
    assert index.doc_lengths[0] == 7


def test_frozen_slotted_hashable():
    p = Posting(1, 2)
    assert {p, Posting(1, 2)} == {p}
    assert {DocMeta(1, "a", "b")}
    with pytest.raises(AttributeError):
        p.extra = 1
    with pytest.raises(AttributeError):
        p.tf = 5


def test_merges():
    a, b = [1, 3, 5, 7], [3, 4, 5]
    assert merge_and(a, b) == [3, 5]
    assert merge_or(a, b) == [1, 3, 4, 5, 7]
    assert merge_not(a, b) == [1, 7]


def test_parse_query():
    assert parse_query("Kyiv OR Lviv NOT river") == [
        ("AND", "kyiv"),
        ("OR", "lviv"),
        ("NOT", "river"),
    ]


def test_both_engines_agree(corpus: Path):
    index = build_index(corpus)
    for query in ("is city", "kyiv OR paris", "city NOT france", "ukraine"):
        assert search(index, query, "merge") == search(index, query, "set")


def test_boolean_results(corpus: Path):
    index = build_index(corpus)
    assert search(index, "city ukraine") == [1]
    assert search(index, "kyiv OR paris") == [0, 2]
    assert search(index, "city NOT france") == [1]
    assert search(index, "nothinghere") == []


def test_positions(corpus: Path):
    index = build_index(corpus, positions=True)
    assert index.has_positions
    # "Paris\nParis is a city in France city" - the title counts as a token too
    assert index.postings["city"][1].positions == (4, 7)


def test_array_repr(corpus: Path):
    index = build_index(corpus, repr_kind="array")
    assert index.doc_ids("city") == [1, 2]
    assert index.term_frequencies("city") == {1: 1, 2: 2}
    assert search(index, "city ukraine") == [1]


@pytest.mark.parametrize("name", ["index.pkl", "index.json"])
@pytest.mark.parametrize("repr_kind", ["slots", "plain", "array"])
def test_save_load_roundtrip(corpus: Path, tmp_path: Path, name: str, repr_kind: str):
    index = build_index(corpus, repr_kind=repr_kind)
    path = tmp_path / name
    save(index, path)
    back = load(path)
    assert back.repr_kind == repr_kind
    assert back.doc_lengths == index.doc_lengths
    assert back.doc_meta == index.doc_meta
    assert back.doc_ids("ukraine") == [0, 1]
    assert search(back, "city ukraine") == [1]
