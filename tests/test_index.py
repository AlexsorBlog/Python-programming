from pathlib import Path

import pytest

from findex.index import DocMeta, Index, Posting, build_index
from findex.merge import merge_and, merge_not, merge_or
from findex.search import boolean_search, parse_flat
from findex.store import load, save


def test_postings_sorted_and_tf(index: Index):
    assert index.doc_ids("ukraine") == [0, 1]
    assert index.doc_ids("city") == [1, 2]
    assert index.term_frequencies("city") == {1: 1, 2: 2}
    assert index.doc_length(0) == 7


def test_frozen_slotted_hashable():
    p = Posting(1, 2)
    assert {p, Posting(1, 2)} == {p}
    assert {DocMeta(1, "a", "b")}
    with pytest.raises(AttributeError):
        p.extra = 1  # type: ignore[attr-defined]  # slots forbid new attributes
    with pytest.raises(AttributeError):
        p.tf = 5  # type: ignore[misc]  # frozen forbids assignment


def test_merges():
    a, b = [1, 3, 5, 7], [3, 4, 5]
    assert merge_and(a, b) == [3, 5]
    assert merge_or(a, b) == [1, 3, 4, 5, 7]
    assert merge_not(a, b) == [1, 7]


def test_parse_flat():
    assert parse_flat("Kyiv OR Lviv NOT river") == [
        ("AND", "kyiv"),
        ("OR", "lviv"),
        ("NOT", "river"),
    ]


def test_both_engines_agree(index: Index):
    for query in ("is city", "kyiv OR paris", "city NOT france", "ukraine"):
        assert boolean_search(index, query, "merge") == boolean_search(
            index, query, "set"
        )


def test_boolean_results(index: Index):
    assert boolean_search(index, "city ukraine") == [1]
    assert boolean_search(index, "kyiv OR paris") == [0, 2]
    assert boolean_search(index, "city NOT france") == [1]
    assert boolean_search(index, "nothinghere") == []


def test_positions(index: Index):
    assert index.has_positions
    entry = index["city"]
    assert not isinstance(entry, tuple)
    # "Paris\nParis is a city in France city" - the title counts as a token too
    assert entry[1].positions == (4, 7)


def test_array_repr(corpus: Path):
    built = build_index(corpus, repr_kind="array")
    assert built.doc_ids("city") == [1, 2]
    assert built.term_frequencies("city") == {1: 1, 2: 2}
    assert boolean_search(built, "city ukraine") == [1]


def test_array_repr_rejects_positions(corpus: Path):
    with pytest.raises(ValueError, match="array"):
        build_index(corpus, positions=True, repr_kind="array")


@pytest.mark.parametrize("name", ["index.pkl", "index.json"])
@pytest.mark.parametrize("repr_kind", ["slots", "plain", "array"])
def test_save_load_roundtrip(corpus: Path, tmp_path: Path, name: str, repr_kind: str):
    built = build_index(corpus, repr_kind=repr_kind)  # type: ignore[arg-type]  # parametrized literal
    path = tmp_path / name
    save(built, path)
    back = load(path)
    assert back.repr_kind == repr_kind
    assert back.doc_lengths == built.doc_lengths
    assert back.doc_meta == built.doc_meta
    assert back.doc_ids("ukraine") == [0, 1]
    assert boolean_search(back, "city ukraine") == [1]
